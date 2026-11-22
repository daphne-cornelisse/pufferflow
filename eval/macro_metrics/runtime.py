"""Thin runtime bindings for full-scene checkpoint rollouts."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

I24_POLICY_HIDDEN = 128
I24_POLICY_LAYERS = 1


class DriveRuntime:
    def __init__(self, library_path: Path):
        self.lib = ctypes.CDLL(str(library_path))
        handle = ctypes.c_void_p
        float_p = ctypes.POINTER(ctypes.c_float)

        def bind(name, restype, args):
            function = getattr(self.lib, name)
            function.restype = restype
            function.argtypes = args

        bind("pf_create_scenario", handle, [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p])
        bind("pf_close", None, [handle])
        bind("pf_reset", None, [handle, ctypes.c_uint])
        bind("pf_integrate", None, [handle, ctypes.c_int])
        bind("pf_recompute", None, [handle])
        bind("pf_spawn_due", None, [handle])
        bind("pf_set_timestep", None, [handle, ctypes.c_int])
        bind("pf_n", ctypes.c_int, [handle])
        bind("pf_obs_size", ctypes.c_int, [handle])
        bind("pf_log_n", ctypes.c_int, [handle])
        bind("pf_log_traj", ctypes.c_int, [handle])
        bind("pf_obs", float_p, [handle])
        bind("pf_set_action", None, [handle, ctypes.c_int, ctypes.c_float, ctypes.c_float])
        bind("pf_set_actions", None, [handle, float_p])
        bind("pf_agent", None, [handle, ctypes.c_int, float_p])
        bind("pf_agents", None, [handle, float_p])
        bind("pf_censor_invalid", None, [handle, ctypes.c_int])
        bind("pf_set_pose", None, [handle, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float])
        bind("pf_invalid", ctypes.c_float, [])
        bind("pf_mean", None, [handle, float_p, float_p])
        bind("pf_allow_all_logs", None, [handle])

    def create_scenario(self, data_dir: Path, map_id: int, json_path: Path):
        handle = self.lib.pf_create_scenario(
            str(data_dir).encode(), int(map_id), str(json_path).encode()
        )
        if not handle:
            raise RuntimeError("pf_create_scenario failed")
        return handle

    def observations(self, handle) -> np.ndarray:
        agents = int(self.lib.pf_n(handle))
        width = int(self.lib.pf_obs_size(handle))
        return np.ctypeslib.as_array(self.lib.pf_obs(handle), shape=(agents, width)).copy()

    def states(self, handle) -> np.ndarray:
        agents = int(self.lib.pf_n(handle))
        values = np.empty((agents, 11), dtype=np.float32)
        self.lib.pf_agents(
            handle, values.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return values

    def set_actions(self, handle, actions: np.ndarray) -> None:
        actions = np.ascontiguousarray(actions, dtype=np.float32)
        expected = (int(self.lib.pf_n(handle)), 2)
        if actions.shape != expected:
            raise ValueError(f"Actions have shape {actions.shape}, expected {expected}")
        self.lib.pf_set_actions(
            handle, actions.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

    def mean_xy(self, handle) -> tuple[float, float]:
        x = ctypes.c_float()
        y = ctypes.c_float()
        self.lib.pf_mean(handle, ctypes.byref(x), ctypes.byref(y))
        return float(x.value), float(y.value)


class PolicySession:
    """Persistent sampled PufferNet process, including per-slot MinGRU carry."""

    def __init__(
        self,
        executable: Path,
        checkpoint: Path,
        batch: int,
        observation_size: int,
        hidden: int = I24_POLICY_HIDDEN,
        layers: int = I24_POLICY_LAYERS,
        seed: int = 1,
    ):
        built = int(subprocess.check_output([str(executable), "--obs"], text=True))
        if built != observation_size:
            raise RuntimeError(f"Policy observation size {built} != runtime {observation_size}")
        self.batch = int(batch)
        self.observation_size = int(observation_size)
        self.process = subprocess.Popen(
            [
                str(executable), str(checkpoint), str(hidden), str(layers),
                str(batch), str(int(seed)),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def act(self, observations: np.ndarray, terminals: np.ndarray) -> np.ndarray:
        observations = np.ascontiguousarray(observations, dtype=np.float32)
        terminals = np.ascontiguousarray(terminals, dtype=np.float32)
        if observations.shape != (self.batch, self.observation_size):
            raise ValueError("Policy observations do not match the fixed session batch")
        if terminals.shape != (self.batch,):
            raise ValueError("Policy terminals do not match the fixed session batch")
        request = b"".join(
            (
                np.int32(1).tobytes(),
                np.int32(self.batch).tobytes(),
                observations.tobytes(),
                terminals.tobytes(),
            )
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(request)
        self.process.stdin.flush()
        size = self.batch * 2 * np.dtype(np.float32).itemsize
        response = self.process.stdout.read(size)
        if len(response) != size:
            error = self.process.stderr.read() if self.process.stderr else b""
            raise RuntimeError(f"Policy process closed early: {error.decode(errors='replace')}")
        return np.frombuffer(response, dtype=np.float32).reshape(self.batch, 2).copy()

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(np.int32(0).tobytes())
            self.process.stdin.flush()
        except BrokenPipeError:
            pass
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
