// Offscreen MP4 capture for machines without a display.
// Call puf_record_request() before the first puf_render. Each capture reads
// the raylib framebuffer and pipes RGBA frames to ffmpeg. If DISPLAY is unset,
// an Xvfb is started for the GL context and torn down when the file closes.
#pragma once

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

typedef struct PufRecord {
    int active;
    int opened;
    int done;
    int frames;
    int limit;
    int fps;
    pid_t ffmpeg;
    pid_t xvfb;
    int pipefd;
    char path[512];
} PufRecord;

static PufRecord* puf_record_state(void) {
    static PufRecord rec;
    return &rec;
}

static inline int puf_record_enabled(void) {
    return puf_record_state()->active;
}

static inline int puf_record_done(void) {
    return puf_record_state()->done;
}

static void puf_record_request(const char* path, int frames, int fps) {
    PufRecord* rec = puf_record_state();
    memset(rec, 0, sizeof(*rec));
    rec->pipefd = -1;
    if (!path || !path[0]) {
        return;
    }
    snprintf(rec->path, sizeof(rec->path), "%s", path);
    rec->limit = frames > 0 ? frames : 91;
    rec->fps = fps > 0 ? fps : 10;
    rec->active = 1;
    char* slash = strrchr(rec->path, '/');
    if (slash && slash != rec->path) {
        *slash = 0;
        mkdir(rec->path, 0755);
        *slash = '/';
    }
}

// A hidden GLFW window still needs an X server. Start one when the machine
// has no DISPLAY, and use llvmpipe so the context does not need a GPU screen.
static void puf_record_ensure_display(void) {
    PufRecord* rec = puf_record_state();
    const char* display = getenv("DISPLAY");
    if (display && display[0]) {
        return;
    }
    if (rec->xvfb > 0) {
        return;
    }
    int number = 99;
    for (int n = 80; n < 140; n++) {
        char sock[64];
        snprintf(sock, sizeof(sock), "/tmp/.X11-unix/X%d", n);
        if (access(sock, F_OK) != 0) {
            number = n;
            break;
        }
    }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork Xvfb");
        exit(1);
    }
    if (pid == 0) {
        char disp[16];
        snprintf(disp, sizeof(disp), ":%d", number);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        execlp("Xvfb", "Xvfb", disp, "-screen", "0", "1280x704x24",
            "+extension", "GLX", (char*)NULL);
        _exit(127);
    }
    char disp[16];
    snprintf(disp, sizeof(disp), ":%d", number);
    setenv("DISPLAY", disp, 1);
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    rec->xvfb = pid;
    char sock[64];
    snprintf(sock, sizeof(sock), "/tmp/.X11-unix/X%d", number);
    for (int i = 0; i < 100; i++) {
        if (access(sock, F_OK) == 0) {
            return;
        }
        usleep(20000);
    }
    fprintf(stderr, "Xvfb did not start on %s\n", disp);
    exit(1);
}

static int puf_record_open(int width, int height) {
    PufRecord* rec = puf_record_state();
    int fds[2];
    if (pipe(fds) < 0) {
        perror("pipe");
        return 0;
    }
    pid_t pid = fork();
    if (pid < 0) {
        perror("fork ffmpeg");
        return 0;
    }
    if (pid == 0) {
        close(fds[1]);
        dup2(fds[0], STDIN_FILENO);
        close(fds[0]);
        char size[32];
        char fps[16];
        snprintf(size, sizeof(size), "%dx%d", width, height);
        snprintf(fps, sizeof(fps), "%d", rec->fps);
        execlp("ffmpeg", "ffmpeg", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", size, "-r", fps, "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "veryfast", "-crf", "23",
            "-movflags", "+faststart",
            "-loglevel", "error",
            rec->path, (char*)NULL);
        _exit(127);
    }
    close(fds[0]);
    rec->ffmpeg = pid;
    rec->pipefd = fds[1];
    rec->opened = 1;
    signal(SIGPIPE, SIG_IGN);
    printf("recording %s (%d frames at %d fps)\n", rec->path, rec->limit, rec->fps);
    fflush(stdout);
    return 1;
}

static int puf_record_write_all(int fd, const unsigned char* data, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t wrote = write(fd, data + off, n - off);
        if (wrote < 0) {
            if (errno == EINTR) {
                continue;
            }
            return 0;
        }
        off += (size_t)wrote;
    }
    return 1;
}

static void puf_record_finish(void) {
    PufRecord* rec = puf_record_state();
    if (!rec->active && !rec->opened) {
        return;
    }
    if (rec->pipefd >= 0) {
        close(rec->pipefd);
        rec->pipefd = -1;
    }
    if (rec->ffmpeg > 0) {
        int status = 0;
        waitpid(rec->ffmpeg, &status, 0);
        rec->ffmpeg = 0;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
    if (rec->xvfb > 0) {
        kill(rec->xvfb, SIGTERM);
        waitpid(rec->xvfb, NULL, 0);
        rec->xvfb = 0;
    }
    if (rec->frames > 0) {
        printf("wrote %s (%d frames)\n", rec->path, rec->frames);
    } else {
        fprintf(stderr, "record failed: no frames written to %s\n", rec->path);
    }
    fflush(stdout);
    rec->done = 1;
    rec->active = 0;
    rec->opened = 0;
}

// Read the framebuffer just drawn. Closes the mp4 after `limit` frames.
static void puf_record_capture(int width, int height) {
    PufRecord* rec = puf_record_state();
    if (!rec->active || rec->done) {
        return;
    }
    if (!rec->opened && !puf_record_open(width, height)) {
        rec->done = 1;
        puf_record_finish();
        return;
    }
    rlDrawRenderBatchActive();
    unsigned char* pixels = rlReadScreenPixels(width, height);
    int ok = pixels && puf_record_write_all(
        rec->pipefd, pixels, (size_t)width * (size_t)height * 4);
    if (pixels) {
        RL_FREE(pixels);
    }
    if (!ok) {
        fprintf(stderr, "record: failed to write a frame\n");
        rec->done = 1;
        puf_record_finish();
        return;
    }
    rec->frames++;
    if (rec->frames >= rec->limit) {
        rec->done = 1;
        puf_record_finish();
    }
}
