"""Behavioral clone on one Waymo map, then ten closed-loop rollouts.

The policy sees teleport observations from every moving vehicle and is
rolled out on the self-driving car. Other vehicles replay the log.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_bind as drv

DATA = drv.ROOT / "data" / "womd"
MAP_ID = 0
JSONS = sorted((DATA / "json").glob("*.json"))
OUT = drv.ROOT / "analysis" / "offline"
CKPT = drv.ROOT / "checkpoints" / "bc_womd_map000.npz"
PLOT = OUT / "bc_map000_rollouts.png"
METRICS = OUT / "bc_map000_rollouts.json"
TRAIN_PLOT = OUT / "bc_map000_train.png"
DISP_PLOT = OUT / "bc_map000_displacement.png"
HORIZON_STEPS = 100
GOAL_RADIUS_M = 5.0
DATASET_TELEPORT = OUT / "bc_map000_teleport.npz"
DATASET_OPEN = OUT / "bc_map000_nonteleport.npz"


def apply_figure_style():
    sns.set_style("ticks")


def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)


class Policy(torch.nn.Module):
    def __init__(self, obs_n, n_accel, n_steer, hidden=512, layers=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        din = obs_n
        self.hidden = torch.nn.ModuleList()
        for _ in range(layers):
            layer = torch.nn.Linear(din, hidden)
            torch.nn.init.kaiming_normal_(layer.weight, a=0.01, nonlinearity="leaky_relu")
            torch.nn.init.zeros_(layer.bias)
            self.hidden.append(layer)
            din = hidden
        self.head_a = torch.nn.Linear(hidden, n_accel)
        self.head_s = torch.nn.Linear(hidden, n_steer)
        torch.nn.init.normal_(self.head_a.weight, std=0.01)
        torch.nn.init.normal_(self.head_s.weight, std=0.01)
        torch.nn.init.zeros_(self.head_a.bias)
        torch.nn.init.zeros_(self.head_s.bias)
        self.mu = None
        self.sig = None

    def forward(self, x):
        h = x
        for layer in self.hidden:
            h = torch.nn.functional.leaky_relu(layer(h), negative_slope=0.01)
        return self.head_a(h), self.head_s(h)

    def _batch(self, obs):
        values = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.head_a.weight.device)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        return values

    def predict(self, obs):
        self.eval()
        with torch.no_grad():
            la, ls = self.forward(self._batch(obs))
        return int(la.argmax()), int(ls.argmax())

    def action_entropy(self, obs):
        self.eval()
        with torch.no_grad():
            la, ls = self.forward(self._batch(obs))
            pa = torch.softmax(la, dim=1)
            ps = torch.softmax(ls, dim=1)
            ent_a = -(pa * torch.log(pa.clamp_min(1e-12))).sum()
            ent_s = -(ps * torch.log(ps.clamp_min(1e-12))).sum()
        return float(ent_a + ent_s)

    def act_batch(self, obs):
        self.eval()
        with torch.no_grad():
            la, ls = self.forward(self._batch(obs))
            pa = torch.softmax(la, dim=1)
            ps = torch.softmax(ls, dim=1)
            ent = (
                -(pa * torch.log(pa.clamp_min(1e-12))).sum(dim=1)
                - (ps * torch.log(ps.clamp_min(1e-12))).sum(dim=1)
            )
            accel = la.argmax(dim=1).detach().cpu().numpy()
            steer = ls.argmax(dim=1).detach().cpu().numpy()
        return accel, steer, ent.detach().cpu().numpy()


def pairs(obs, acts, valid):
    mask = valid.astype(bool)
    x = obs[mask].astype(np.float32)
    y = acts[mask].astype(np.int64)
    return x, y[:, 0], y[:, 1]


def conflicts(x, ya, ys):
    seen = {}
    bad = 0
    for i in range(len(x)):
        key = x[i].tobytes()
        lab = (int(ya[i]), int(ys[i]))
        prev = seen.get(key)
        if prev is None:
            seen[key] = lab
        elif prev != lab:
            bad += 1
    return bad, len(x)


def empty_history():
    keys = (
        "epoch", "phase", "acc", "acc_a", "acc_s",
        "logp", "logp_a", "logp_s", "ent", "ent_a", "ent_s",
    )
    return {k: [] for k in keys}


def record_stats(hist, epoch, phase, stats):
    hist["epoch"].append(int(epoch))
    hist["phase"].append(phase)
    for key, value in stats.items():
        hist[key].append(float(value))


def _device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    print(f"Device: {device} {torch.cuda.get_device_name(0)}", flush=True)
    return device


def _loader(obs, actions, batch_size):
    return DataLoader(
        TensorDataset(obs, actions),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )


def _epoch_loss(pol, loader, opt, device):
    pol.train()
    total = 0.0
    count = 0
    for batch_obs, batch_actions in loader:
        batch_obs = batch_obs.to(device, non_blocking=True)
        batch_actions = batch_actions.to(device, non_blocking=True)
        la, ls = pol(batch_obs)
        loss = -(
            Categorical(logits=la).log_prob(batch_actions[:, 0]).mean()
            + Categorical(logits=ls).log_prob(batch_actions[:, 1]).mean()
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.detach()) * batch_obs.shape[0]
        count += batch_obs.shape[0]
    return total / max(count, 1)


def _full_stats(pol, obs, actions, device):
    pol.eval()
    n = obs.shape[0]
    both = acc_a = acc_s = 0
    logp_a = logp_s = ent_a = ent_s = 0.0
    with torch.no_grad():
        for start in range(0, n, 8192):
            xb = obs[start:start + 8192].to(device, non_blocking=True)
            yb = actions[start:start + 8192].to(device, non_blocking=True)
            la, ls = pol(xb)
            da = Categorical(logits=la)
            ds = Categorical(logits=ls)
            pred_a = la.argmax(dim=1)
            pred_s = ls.argmax(dim=1)
            good_a = pred_a == yb[:, 0]
            good_s = pred_s == yb[:, 1]
            both += int((good_a & good_s).sum())
            acc_a += int(good_a.sum())
            acc_s += int(good_s.sum())
            logp_a += float(da.log_prob(yb[:, 0]).sum())
            logp_s += float(ds.log_prob(yb[:, 1]).sum())
            ent_a += float(da.entropy().sum())
            ent_s += float(ds.entropy().sum())
    return {
        "acc": both / n,
        "acc_a": acc_a / n,
        "acc_s": acc_s / n,
        "logp": (logp_a + logp_s) / n,
        "logp_a": logp_a / n,
        "logp_s": logp_s / n,
        "ent": (ent_a + ent_s) / n,
        "ent_a": ent_a / n,
        "ent_s": ent_s / n,
    }


def _mistakes(pol, obs, actions, device):
    pol.eval()
    wrong = []
    with torch.no_grad():
        for start in range(0, obs.shape[0], 8192):
            xb = obs[start:start + 8192].to(device, non_blocking=True)
            yb = actions[start:start + 8192].to(device, non_blocking=True)
            la, ls = pol(xb)
            bad = (la.argmax(dim=1) != yb[:, 0]) | (ls.argmax(dim=1) != yb[:, 1])
            if int(bad.sum()):
                wrong.append(torch.arange(start, start + xb.shape[0])[bad.cpu()])
    if not wrong:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(wrong)


def _set_lr(opt, lr):
    for group in opt.param_groups:
        group["lr"] = lr


# Joint accuracy above this ends the main loop and the hard-example phase.
STOP_JOINT = 0.99


def train(x, ya, ys, n_accel, n_steer, batch_size=2048):
    device = _device()
    pol = Policy(x.shape[1], n_accel, n_steer).to(device)
    opt = Adam(pol.parameters(), lr=1e-3)
    obs = torch.from_numpy(np.ascontiguousarray(x))
    actions = torch.stack((
        torch.from_numpy(np.ascontiguousarray(ya)),
        torch.from_numpy(np.ascontiguousarray(ys)),
    ), dim=1).long()
    loader = _loader(obs, actions, batch_size)
    hist = empty_history()
    lr = 1e-3
    best = 0.0
    stall = 0
    acc = acc_a = acc_s = 0.0
    epoch = 0
    stats = None
    for epoch in range(1, 2001):
        loss = _epoch_loss(pol, loader, opt, device)
        stats = _full_stats(pol, obs, actions, device)
        record_stats(hist, epoch, "train", stats)
        acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
        if acc > best + 1e-4:
            best = acc
            stall = 0
        else:
            stall += 1
        if stall == 60:
            lr *= 0.5
            _set_lr(opt, lr)
            stall = 0
            print(f"lr {lr:.2e}", flush=True)
        if epoch % 25 == 0 or acc > STOP_JOINT:
            print(f"epoch {epoch:4d} loss {loss:.4f} joint {acc:.3f} "
                f"logp {stats['logp']:.3f} ent {stats['ent']:.3f}", flush=True)
        if acc > STOP_JOINT or lr < 1e-6:
            break
    if acc <= STOP_JOINT:
        wrong = _mistakes(pol, obs, actions, device)
        print(f"hard examples {len(wrong)}", flush=True)
        lr = 1e-3
        _set_lr(opt, lr)
        for extra in range(1, 401):
            epoch += 1
            wrong = _mistakes(pol, obs, actions, device)
            if len(wrong) == 0:
                stats = _full_stats(pol, obs, actions, device)
                record_stats(hist, epoch, "hard", stats)
                acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
                break
            idx = torch.cat((torch.arange(obs.shape[0]), wrong.repeat(12)))
            loss = _epoch_loss(pol, _loader(obs[idx], actions[idx], batch_size), opt, device)
            stats = _full_stats(pol, obs, actions, device)
            record_stats(hist, epoch, "hard", stats)
            acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
            if extra % 20 == 0 or acc > STOP_JOINT:
                print(f"hard {extra:3d} loss {loss:.4f} joint {acc:.3f} "
                    f"logp {stats['logp']:.3f} ent {stats['ent']:.3f} "
                    f"left {len(wrong)}", flush=True)
            if acc > STOP_JOINT:
                break
            if extra % 80 == 0:
                lr *= 0.5
                _set_lr(opt, lr)
    pol.cpu()
    pol.eval()
    return pol, acc, acc_a, acc_s, epoch, hist


def save_policy(pol, acc, n_accel, n_steer, path=None):
    path = Path(path) if path else CKPT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "Wa": pol.head_a.weight.detach().cpu().numpy().T.astype(np.float32),
        "ba": pol.head_a.bias.detach().cpu().numpy().astype(np.float32),
        "Ws": pol.head_s.weight.detach().cpu().numpy().T.astype(np.float32),
        "bs": pol.head_s.bias.detach().cpu().numpy().astype(np.float32),
        "accuracy": np.float32(acc),
        "n_accel": np.int32(n_accel),
        "n_steer": np.int32(n_steer),
        "mu": pol.mu,
        "sig": pol.sig,
    }
    for i, layer in enumerate(pol.hidden):
        payload[f"W{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        payload[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez(path, **payload)
    print(f"wrote {path}")


def rollout(lib, h, pol, controlled, steps=None):
    if steps is None:
        steps = drv.TRAJ - 1
    lib.pf_reset(h, 0)
    xs = []
    ys = []
    entropy = []
    for _ in range(steps):
        st = drv.agent_state(lib, h, controlled)
        xs.append(float(st[0]))
        ys.append(float(st[1]))
        obs = drv.observations(lib, h)[controlled]
        obs = (obs - pol.mu) / pol.sig
        entropy.append(pol.action_entropy(obs))
        a, s = pol.predict(obs)
        lib.pf_set_action(h, controlled, float(a), float(s))
        lib.pf_replay(h, controlled)
    st = drv.agent_state(lib, h, controlled)
    xs.append(float(st[0]))
    ys.append(float(st[1]))
    out = drv.outcome(lib, h, controlled)
    out["x"] = xs
    out["y"] = ys
    out["entropy"] = entropy
    out["score"] = int(out["reached"] and not out["collided_before"])
    return out


def _copy_linear(layer, weight_in_out, bias):
    with torch.no_grad():
        layer.weight.copy_(torch.from_numpy(np.ascontiguousarray(weight_in_out.T)))
        layer.bias.copy_(torch.from_numpy(np.ascontiguousarray(bias)))


def load_policy(path):
    z = np.load(path)
    obs_n = int(z["W0"].shape[0])
    n_layers = sum(1 for k in z.files if k.startswith("W") and k[1:].isdigit())
    pol = Policy(obs_n, int(z["n_accel"]), int(z["n_steer"]), layers=n_layers)
    for i, layer in enumerate(pol.hidden):
        _copy_linear(layer, z[f"W{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32))
    _copy_linear(pol.head_a, z["Wa"].astype(np.float32), z["ba"].astype(np.float32))
    _copy_linear(pol.head_s, z["Ws"].astype(np.float32), z["bs"].astype(np.float32))
    pol.mu = z["mu"].astype(np.float32)
    pol.sig = z["sig"].astype(np.float32)
    pol.eval()
    return pol, float(z["accuracy"])


def controllable_agents(lib, h):
    return [i for i in range(lib.pf_log_n(h)) if not lib.pf_base_unfit(h, i)]


def log_track(lib, h, agent):
    xs = []
    ys = []
    for t in range(drv.TRAJ):
        if not lib.pf_log_valid(h, agent, t):
            continue
        x, y = drv.log_xy(lib, h, agent, t)
        xs.append(x)
        ys.append(y)
    return xs, ys


def _draw_scene(ax, roads, human, rollouts, goal, zoom, legend=True):
    apply_figure_style()
    first_road = True
    for _kind, rx, ry in roads:
        ax.plot(rx, ry, color="#b0b0b0", lw=0.6, zorder=0,
            label="road" if first_road else None)
        first_road = False
    ax.plot(human[0], human[1], color="black", lw=1.0, zorder=2, label="log")
    ax.scatter(human[0], human[1], c="black", s=10 if zoom else 6, zorder=3)
    colors = sns.color_palette(n_colors=max(len(rollouts), 1))
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    point = 28 if zoom else 14
    for i, r in enumerate(rollouts):
        ax.scatter(
            r["x"], r["y"], c=[colors[i]], s=point, marker=markers[i % len(markers)],
            alpha=0.35, zorder=4, linewidths=0.3, edgecolors=colors[i],
            label=f"rollout {i}" if legend else None,
        )
    ax.scatter([human[0][0]], [human[1][0]], c="black", s=36, zorder=5,
        label="start" if legend else None)
    ax.scatter([goal[0]], [goal[1]], marker="*", c="black", s=120, zorder=5,
        label="goal" if legend else None)
    if zoom:
        xs = list(human[0]) + [p for r in rollouts for p in r["x"]]
        ys = list(human[1]) + [p for r in rollouts for p in r["y"]]
        pad = 8.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    style_ax(ax)


def rollout_record(rollouts):
    rows = []
    for r in rollouts:
        rows.append({k: r[k] for k in (
            "return", "ade", "goal_dist", "collided", "offroad",
            "reached", "collided_before", "score", "x", "y",
        )})
    return rows


def plot(lib, h, human, rollouts, acc, controlled, out_path=None):
    out_path = Path(out_path) if out_path else PLOT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    road = drv.roads(lib, h)
    st = drv.agent_state(lib, h, controlled)
    goal = (float(st[5]), float(st[6]))
    ades = [r["ade"] for r in rollouts]
    scores = [r["score"] for r in rollouts]
    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.4), constrained_layout=True)
    _draw_scene(axes[0], road, human, rollouts, goal, zoom=False)
    _draw_scene(axes[1], road, human, rollouts, goal, zoom=True)
    axes[0].set_title("full map")
    axes[1].set_title("path")
    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        f"map {MAP_ID:03d} agent {controlled}  "
        f"train acc {acc:.3f}  mean ADE {np.mean(ades):.2f} m  "
        f"score {sum(scores)}/10"
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return goal


def plot_all(lib, h, records, acc):
    PLOT.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    n = len(records)
    cols = 2 if n > 1 else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.6 * cols, 5.8 * rows), squeeze=False,
        constrained_layout=True,
    )
    road = drv.roads(lib, h)
    for k, rec in enumerate(records):
        ax = axes[k // cols][k % cols]
        _draw_scene(ax, road, rec["human"], rec["rollouts"], rec["goal"],
            zoom=True, legend=False)
        ades = [r["ade"] for r in rec["rollouts"]]
        scores = sum(r["score"] for r in rec["rollouts"])
        ax.set_title(
            f"agent {rec['agent']}  ADE {np.mean(ades):.2f} m  score {scores}/10"
        )
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(f"map {MAP_ID:03d}  train acc {acc:.3f}  controllable agents")
    fig.savefig(PLOT, dpi=140)
    plt.close(fig)


def squared_displacement(human, rollout):
    """Per-timestep (dx^2 + dy^2) against the log. One point per logged frame."""
    n = min(len(human[0]), len(rollout["x"]), drv.TRAJ)
    dx = np.asarray(rollout["x"][:n], dtype=np.float64) - np.asarray(human[0][:n], dtype=np.float64)
    dy = np.asarray(rollout["y"][:n], dtype=np.float64) - np.asarray(human[1][:n], dtype=np.float64)
    return dx * dx + dy * dy


def reached_end(human, rollout):
    ex = float(human[0][-1])
    ey = float(human[1][-1])
    n = min(len(rollout["x"]), HORIZON_STEPS + 1)
    x = np.asarray(rollout["x"][:n], dtype=np.float64)
    y = np.asarray(rollout["y"][:n], dtype=np.float64)
    return bool(np.hypot(x - ex, y - ey).min() <= GOAL_RADIUS_M)


def failed_episode(rollout):
    return bool(rollout["collided"] or rollout["offroad"])


def plot_displacement(records, path=None):
    path = Path(path) if path else DISP_PLOT
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    colors = sns.color_palette(n_colors=max(len(records), 1))
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.2), constrained_layout=True)
    ax_d, ax_h, ax_c, ax_f = axes
    completion = []
    failure = []
    labels = []
    for rec, color in zip(records, colors):
        agent = rec["agent"]
        labels.append(str(agent))
        first = True
        for roll in rec["rollouts"]:
            err = squared_displacement(rec["human"], roll)
            ent = np.asarray(roll["entropy"], dtype=np.float64)
            ax_d.scatter(
                np.arange(len(err)), err, c=[color], s=14, alpha=0.35, linewidths=0,
                label=f"agent {agent}" if first else None,
            )
            ax_h.scatter(
                np.arange(len(ent)), ent, c=[color], s=14, alpha=0.35, linewidths=0,
            )
            first = False
        completion.append(float(np.mean([reached_end(rec["human"], r) for r in rec["rollouts"]])))
        failure.append(float(np.mean([failed_episode(r) for r in rec["rollouts"]])))
    ax_d.set_xlabel("timestep")
    ax_d.set_ylabel("squared displacement (m²)")
    ax_d.set_title("displacement")
    style_ax(ax_d)
    ax_d.legend(loc="upper left", fontsize=8)
    ax_h.set_xlabel("step")
    ax_h.set_ylabel("entropy (nats)")
    ax_h.set_title("action entropy")
    style_ax(ax_h)
    idx = np.arange(len(records))
    for axr, title, values in (
        (ax_c, "completion", completion),
        (ax_f, "failure", failure),
    ):
        axr.bar(idx, values, color=colors)
        axr.set_xticks(idx, labels)
        axr.set_ylim(0.0, 1.18)
        axr.set_xlabel("agent")
        axr.set_title(title)
        style_ax(axr)
        for i, value in enumerate(values):
            axr.text(i, value + 0.03, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax_c.set_ylabel("rate")
    fig.suptitle(
        f"map {MAP_ID:03d}  completion is the logged end within {GOAL_RADIUS_M:.0f} m "
        f"by step {HORIZON_STEPS}"
    )
    fig.savefig(path, dpi=140)
    plt.close(fig)
    summary = {
        "map_id": MAP_ID,
        "horizon_steps": HORIZON_STEPS,
        "goal_radius_m": GOAL_RADIUS_M,
        "agents": [
            {
                "agent": rec["agent"],
                "completion_rate": c,
                "failure_rate": f,
                "n": len(rec["rollouts"]),
            }
            for rec, c, f in zip(records, completion, failure)
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {path}")
    return summary


def evaluate_displacement(lib, h, pol):
    records = []
    for agent in controllable_agents(lib, h):
        human = log_track(lib, h, agent)
        rollouts = [rollout(lib, h, pol, agent, steps=HORIZON_STEPS) for _ in range(10)]
        comp = float(np.mean([reached_end(human, r) for r in rollouts]))
        fail = float(np.mean([failed_episode(r) for r in rollouts]))
        print(f"agent {agent} completion {comp:.2f} failure {fail:.2f}")
        records.append({
            "agent": agent,
            "human": human,
            "rollouts": rollouts,
        })
    return plot_displacement(records)


def evaluate_agents(lib, h, pol, acc):
    agents = controllable_agents(lib, h)
    print(f"controllable agents {agents}")
    records = []
    for agent in agents:
        human = log_track(lib, h, agent)
        rollouts = [rollout(lib, h, pol, agent) for _ in range(10)]
        out_path = PLOT.with_name(f"bc_map000_agent{agent}_rollouts.png")
        goal = plot(lib, h, human, rollouts, acc, agent, out_path)
        ades = [r["ade"] for r in rollouts]
        scores = [r["score"] for r in rollouts]
        print(f"agent {agent} ADE {np.mean(ades):.3f} score {sum(scores)}/10  wrote {out_path}")
        records.append({
            "agent": agent,
            "human": human,
            "rollouts": rollouts,
            "goal": goal,
            "rows": rollout_record(rollouts),
        })
    plot_all(lib, h, records, acc)
    print(f"wrote {PLOT}")
    evaluate_displacement(lib, h, pol)
    return records


def plot_training(hist, path=None, title="BC training on map 000"):
    path = Path(path) if path else TRAIN_PLOT
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    epoch = np.asarray(hist["epoch"])
    phases = [str(p) for p in hist["phase"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.0), sharex=True, constrained_layout=True)
    series = (
        (axes[0], ("acc", "acc_a", "acc_s"), ("joint", "accel", "steer"), "accuracy"),
        (axes[1], ("logp", "logp_a", "logp_s"), ("joint", "accel", "steer"), "log prob (nats)"),
        (axes[2], ("ent", "ent_a", "ent_s"), ("joint", "accel", "steer"), "entropy (nats)"),
    )
    colors = sns.color_palette(n_colors=3)
    for ax, keys, labels, ylabel in series:
        for key, label, color in zip(keys, labels, colors):
            ax.plot(epoch, hist[key], label=label, lw=1.4, color=color)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        style_ax(ax)
        ax.legend(loc="best", fontsize=8)
    if "hard" in phases:
        cut = float(epoch[phases.index("hard")])
        for ax in axes:
            ax.axvline(cut, color="0.45", lw=0.8, ls="--", zorder=0)
        top = axes[0].get_ylim()[1]
        axes[0].text(cut, top, " hard examples",
            va="top", ha="left", fontsize=8, color="0.35")
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    np.savez(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in hist.items()})
    print(f"wrote {path}")


def save_dataset(obs, acts, valid, ade, teleport, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        obs=np.asarray(obs),
        acts=np.asarray(acts),
        valid=np.asarray(valid),
        ade=np.asarray(ade),
        teleport=np.int32(teleport),
        map_id=np.int32(MAP_ID),
    )
    print(f"wrote {path}")


N_MAPS = 100
CKPT_100 = drv.ROOT / "checkpoints" / "bc_womd_100.npz"
TRAIN_PLOT_100 = OUT / "bc_100_train.png"
EVAL_PLOT_100 = OUT / "bc_100_eval.png"
EVAL_JSON_100 = OUT / "bc_100_eval.json"
DATASET_100 = OUT / "bc_100_teleport.npz"


def collect_pairs(lib, map_ids, teleport=1):
    xs, yas, yss = [], [], []
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            obs, acts, valid, _ade = drv.collect(lib, h, teleport)
            x, ya, ys = pairs(obs, acts, valid)
            xs.append(x)
            yas.append(ya)
            yss.append(ys)
            print(f"map {map_id} pairs {len(x)}", flush=True)
        finally:
            lib.pf_close(h)
    return np.concatenate(xs), np.concatenate(yas), np.concatenate(yss)


def plot_scenario_eval(steps_d, err, steps_h, ent, comp_mean, comp_std,
        fail_mean, fail_std, disp_mean, disp_std, pair_disp, pair_ent,
        n_maps, path=None, title=None, n_accel=21, n_steer=31):
    path = Path(path) if path else EVAL_PLOT_100
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    colors = sns.color_palette(n_colors=3)
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.3), constrained_layout=True)
    ax_d, ax_h, ax_b, ax_c = axes
    ax_h.sharex(ax_d)
    dot = {"s": 8, "alpha": 0.06, "linewidths": 0, "rasterized": True}
    ax_d.scatter(steps_d, err, color=colors[0], **dot)
    ax_d.set_xlabel("timestep")
    ax_d.set_ylabel("displacement (m)")
    ax_d.set_title("displacement")
    style_ax(ax_d)
    ax_h.scatter(steps_h, ent, color=colors[0], **dot)
    uniform = float(np.log(n_accel) + np.log(n_steer))
    ax_h.axhline(
        uniform, color="black", ls=(0, (6, 4)), lw=1.3, zorder=3,
        label=f"uniform {uniform:.2f}",
    )
    ax_h.set_ylim(0.0, uniform * 1.12)
    ax_h.legend(loc="upper right", fontsize=8, frameon=False)
    ax_h.set_xlabel("step")
    ax_h.set_ylabel("entropy (nats)")
    ax_h.set_title("action entropy")
    style_ax(ax_h)
    rate_means = [comp_mean, fail_mean]
    rate_stds = [comp_std, fail_std]
    ax_b.bar(
        [0, 1], rate_means, yerr=rate_stds, color=colors[:2], capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0}, width=0.7,
    )
    ax_m = ax_b.twinx()
    ax_m.bar(
        [2], [disp_mean], yerr=[disp_std], color=colors[2], capsize=4,
        error_kw={"elinewidth": 1.0, "capthick": 1.0}, width=0.7,
    )
    ax_b.set_xticks([0, 1, 2], ["completion", "failure", "displacement"])
    ax_b.set_ylabel("rate")
    ax_m.set_ylabel("displacement (m)")
    ax_b.set_title("mean ± std across maps")
    rate_top = max(m + s for m, s in zip(rate_means, rate_stds))
    ax_b.set_ylim(0.0, max(1.15, rate_top + 0.28))
    ax_m.set_ylim(0.0, max(disp_mean + disp_std, 1e-6) * 1.35)
    ax_b.set_axisbelow(True)
    ax_b.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax_b, top=True, right=True)
    sns.despine(ax=ax_m, top=True, left=True, right=False)
    for i, (mean, std) in enumerate(zip(rate_means, rate_stds)):
        ax_b.text(i, mean + std + 0.03, f"{mean:.2f} ± {std:.2f}",
            ha="center", va="bottom", fontsize=8)
    ax_m.text(2, disp_mean + disp_std + 0.04 * ax_m.get_ylim()[1],
        f"{disp_mean:.2f} ± {disp_std:.2f}",
        ha="center", va="bottom", fontsize=8)
    sns.regplot(
        x=pair_disp, y=pair_ent, ax=ax_c, ci=95,
        scatter_kws={"s": 8, "alpha": 0.06, "linewidths": 0, "rasterized": True,
            "color": colors[0]},
        line_kws={"color": colors[1], "lw": 1.5},
    )
    if len(pair_disp) > 1:
        corr = float(np.corrcoef(pair_disp, pair_ent)[0, 1])
        ax_c.text(0.97, 0.96, f"r = {corr:.2f}", transform=ax_c.transAxes,
            va="top", ha="right", fontsize=8, color="0.2")
    ax_c.set_xlabel("displacement (m)")
    ax_c.set_ylabel("entropy (nats)")
    ax_c.set_title("displacement vs entropy")
    style_ax(ax_c)
    fig.suptitle(title or (
        f"{n_maps} maps  completion is the logged end within {GOAL_RADIUS_M:.0f} m "
        f"by step {HORIZON_STEPS}"
    ))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def evaluate_maps(lib, pol, map_ids):
    disp_steps, disp_err = [], []
    ent_steps, ent_vals = [], []
    per_map = []
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            agents = controllable_agents(lib, h)
            completed = []
            failed = []
            for agent in agents:
                human = log_track(lib, h, agent)
                rollouts = [rollout(lib, h, pol, agent, steps=HORIZON_STEPS) for _ in range(10)]
                for roll in rollouts:
                    err = squared_displacement(human, roll)
                    disp_steps.append(np.arange(len(err), dtype=np.int32))
                    disp_err.append(err.astype(np.float32))
                    ent = np.asarray(roll["entropy"], dtype=np.float32)
                    ent_steps.append(np.arange(len(ent), dtype=np.int32))
                    ent_vals.append(ent)
                    completed.append(reached_end(human, roll))
                    failed.append(failed_episode(roll))
            comp = float(np.mean(completed)) if completed else float("nan")
            fail = float(np.mean(failed)) if failed else float("nan")
            per_map.append({
                "map_id": int(map_id),
                "n_agents": len(agents),
                "n_rollouts": len(completed),
                "completion_rate": comp,
                "failure_rate": fail,
            })
            print(f"eval map {map_id} agents {len(agents)} "
                f"completion {comp:.2f} failure {fail:.2f}", flush=True)
        finally:
            lib.pf_close(h)
    return (
        np.concatenate(disp_steps), np.concatenate(disp_err),
        np.concatenate(ent_steps), np.concatenate(ent_vals),
        per_map,
    )


def _park_unfit(lib, h):
    invalid = float(lib.pf_invalid())
    for i in range(lib.pf_log_n(h)):
        if lib.pf_base_unfit(h, i):
            lib.pf_set_agent(h, i, invalid, invalid, 0, 0, 0, 1, 1, 0, 0)


def _on_map(state):
    return float(state[10]) > 0.5


def closed_loop_rollout(lib, h, pol, agents, ends, steps=None):
    """Every controllable agent is driven by the policy. Nobody is log-replayed."""
    if steps is None:
        steps = HORIZON_STEPS
    lib.pf_reset(h, 0)
    _park_unfit(lib, h)
    tracks = {
        a: {
            "x": [], "y": [], "disp_t": [], "disp": [], "disp_ent": [],
            "ent_t": [], "ent": [],
        }
        for a in agents
    }
    collided = {a: False for a in agents}
    offroad = {a: False for a in agents}
    for t in range(steps):
        lib.pf_set_timestep(h, t)
        lib.pf_spawn_due(h)
        lib.pf_recompute(h)
        obs = drv.observations(lib, h)
        live = []
        states = {}
        for agent in agents:
            state = drv.agent_state(lib, h, agent)
            if not _on_map(state):
                continue
            states[agent] = state
            live.append(agent)
            tracks[agent]["x"].append(float(state[0]))
            tracks[agent]["y"].append(float(state[1]))
        ent_of = {}
        if live:
            batch = (obs[np.asarray(live)] - pol.mu) / pol.sig
            accels, steers, ent = pol.act_batch(batch)
            for agent, accel, steer, entropy in zip(live, accels, steers, ent):
                lib.pf_set_action(h, agent, float(accel), float(steer))
                ent_of[agent] = float(entropy)
                tracks[agent]["ent_t"].append(t)
                tracks[agent]["ent"].append(float(entropy))
        for agent in live:
            if t < drv.TRAJ and lib.pf_log_valid(h, agent, t):
                lx, ly = drv.log_xy(lib, h, agent, t)
                state = states[agent]
                dist = float(np.hypot(float(state[0]) - lx, float(state[1]) - ly))
                tracks[agent]["disp_t"].append(t)
                tracks[agent]["disp"].append(dist)
                tracks[agent]["disp_ent"].append(ent_of[agent])
        lib.pf_integrate(h, 1)
        for agent in live:
            state = drv.agent_state(lib, h, agent)
            kind = int(state[9])
            if kind == 1:
                collided[agent] = True
            elif kind == 2:
                offroad[agent] = True
    for agent in agents:
        state = drv.agent_state(lib, h, agent)
        if _on_map(state):
            tracks[agent]["x"].append(float(state[0]))
            tracks[agent]["y"].append(float(state[1]))
        xs = np.asarray(tracks[agent]["x"], dtype=np.float64)
        ys = np.asarray(tracks[agent]["y"], dtype=np.float64)
        ex, ey = ends[agent]
        if len(xs) == 0:
            reached = False
        else:
            reached = bool(np.hypot(xs - ex, ys - ey).min() <= GOAL_RADIUS_M)
        tracks[agent]["completed"] = reached
        tracks[agent]["failed"] = bool(collided[agent] or offroad[agent])
    return tracks


def evaluate_closed_loop(lib, pol, map_ids, rollouts=10, keep_maps=()):
    disp_steps, disp_err = [], []
    ent_steps, ent_vals = [], []
    pair_disp, pair_ent = [], []
    per_map = []
    scenes = {}
    keep = set(int(m) for m in keep_maps)
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            agents = controllable_agents(lib, h)
            ends = {}
            logs = {}
            for agent in agents:
                xs, ys = log_track(lib, h, agent)
                ends[agent] = (xs[-1], ys[-1])
                logs[agent] = (xs, ys)
            completed = []
            failed = []
            kept = []
            agent_disp = []
            for _ in range(rollouts):
                tracks = closed_loop_rollout(lib, h, pol, agents, ends)
                one = []
                for agent in agents:
                    rec = tracks[agent]
                    if rec["disp"]:
                        disp_steps.append(np.asarray(rec["disp_t"], dtype=np.int32))
                        disp_err.append(np.asarray(rec["disp"], dtype=np.float32))
                        pair_disp.append(np.asarray(rec["disp"], dtype=np.float32))
                        pair_ent.append(np.asarray(rec["disp_ent"], dtype=np.float32))
                        agent_disp.append(float(np.mean(rec["disp"])))
                    if rec["ent"]:
                        ent_steps.append(np.asarray(rec["ent_t"], dtype=np.int32))
                        ent_vals.append(np.asarray(rec["ent"], dtype=np.float32))
                    completed.append(rec["completed"])
                    failed.append(rec["failed"])
                    one.append({"agent": agent, "x": rec["x"], "y": rec["y"]})
                if int(map_id) in keep:
                    kept.append(one)
            comp = float(np.mean(completed)) if completed else float("nan")
            fail = float(np.mean(failed)) if failed else float("nan")
            disp_mean_map = float(np.mean(agent_disp)) if agent_disp else float("nan")
            per_map.append({
                "map_id": int(map_id),
                "n_agents": len(agents),
                "n_rollouts": len(completed),
                "completion_rate": comp,
                "failure_rate": fail,
                "displacement_mean": disp_mean_map,
            })
            if int(map_id) in keep:
                scenes[int(map_id)] = {
                    "roads": drv.roads(lib, h),
                    "logs": logs,
                    "ends": ends,
                    "rollouts": kept,
                }
            print(f"eval map {map_id} agents {len(agents)} "
                f"rollouts {rollouts} completion {comp:.2f} failure {fail:.2f}",
                flush=True)
        finally:
            lib.pf_close(h)
    empty = np.zeros(0, dtype=np.float32)
    steps_d = np.concatenate(disp_steps) if disp_steps else empty.astype(np.int32)
    err = np.concatenate(disp_err) if disp_err else empty
    steps_h = np.concatenate(ent_steps) if ent_steps else empty.astype(np.int32)
    ent = np.concatenate(ent_vals) if ent_vals else empty
    paired_d = np.concatenate(pair_disp) if pair_disp else empty
    paired_e = np.concatenate(pair_ent) if pair_ent else empty
    return steps_d, err, steps_h, ent, per_map, scenes, paired_d, paired_e


def _draw_closed_scene(ax, roads, logs, ends, rollouts, zoom):
    apply_figure_style()
    first_road = True
    for _kind, rx, ry in roads:
        ax.plot(rx, ry, color="#b0b0b0", lw=0.6, zorder=0,
            label="road" if first_road else None)
        first_road = False
    agents = sorted(logs)
    colors = sns.color_palette(n_colors=max(len(agents), 1))
    color_of = {agent: colors[i] for i, agent in enumerate(agents)}
    first_log = True
    xs, ys = [], []
    for agent in agents:
        lx, ly = logs[agent]
        ax.plot(lx, ly, color="black", lw=0.8, zorder=2,
            label="log" if first_log else None)
        first_log = False
        ax.scatter([ends[agent][0]], [ends[agent][1]], marker="*",
            c="black", s=70, zorder=5)
        xs.extend(lx)
        ys.extend(ly)
    for run in rollouts:
        for rec in run:
            agent = rec["agent"]
            pts = [
                (x, y) for x, y in zip(rec["x"], rec["y"])
                if abs(x) < 5000 and abs(y) < 5000
            ]
            if not pts:
                continue
            px, py = zip(*pts)
            ax.scatter(
                px, py, color=color_of[agent], s=16 if zoom else 8,
                alpha=0.35, linewidths=0, zorder=4,
            )
            xs.extend(px)
            ys.extend(py)
    if zoom and xs:
        pad = 8.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    style_ax(ax)


def plot_closed_maps(scenes, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_ids = sorted(scenes)
    apply_figure_style()
    fig, axes = plt.subplots(
        len(map_ids), 2, figsize=(12.6, 4.6 * len(map_ids)),
        squeeze=False, constrained_layout=True,
    )
    for row, map_id in enumerate(map_ids):
        scene = scenes[map_id]
        n_agents = len(scene["logs"])
        n_runs = len(scene["rollouts"])
        for col, zoom in enumerate((False, True)):
            ax = axes[row][col]
            _draw_closed_scene(
                ax, scene["roads"], scene["logs"], scene["ends"],
                scene["rollouts"], zoom,
            )
            ax.set_title(
                f"map {map_id}  {'path' if zoom else 'full map'}  "
                f"{n_agents} agents  {n_runs} rollouts"
            )
        axes[row][1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Closed loop, policy controls every controllable agent")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")


def _run_paths(n_maps):
    tag = f"bc_{int(n_maps)}"
    return {
        "dataset": OUT / f"{tag}_teleport.npz",
        "train_plot": OUT / f"{tag}_train.png",
        "eval_plot": OUT / f"{tag}_eval.png",
        "eval_json": OUT / f"{tag}_eval.json",
        "ckpt": drv.ROOT / "checkpoints" / f"bc_womd_{int(n_maps)}.npz",
    }


def train_and_eval_maps(n_maps=N_MAPS):
    paths = _run_paths(n_maps)
    map_ids = list(range(n_maps))
    lib = drv.load()
    x, ya, ys = collect_pairs(lib, map_ids, teleport=1)
    mu = x.mean(axis=0)
    sig = x.std(axis=0)
    sig = np.where(sig < 1e-6, 1.0, sig).astype(np.float32)
    paths["dataset"].parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        paths["dataset"], x=x, ya=ya, ys=ys, mu=mu.astype(np.float32), sig=sig,
        n_maps=np.int32(n_maps),
    )
    print(f"wrote {paths['dataset']} samples {len(x)}", flush=True)
    x = ((x - mu) / sig).astype(np.float32)
    n_accel = lib.pf_n_accel()
    n_steer = lib.pf_n_steer()
    pol, acc, acc_a, acc_s, epoch, hist = train(x, ya, ys, n_accel, n_steer)
    plot_training(hist, paths["train_plot"], title=f"BC training on {n_maps} maps")
    pol.mu = mu.astype(np.float32)
    pol.sig = sig
    save_policy(pol, acc, n_accel, n_steer, paths["ckpt"])
    print(f"checkpoint accuracy joint {acc:.4f} accel {acc_a:.4f} steer {acc_s:.4f} "
        f"epochs {epoch}", flush=True)
    steps_d, err, steps_h, ent, per_map, _scenes, pair_d, pair_e = evaluate_closed_loop(
        lib, pol, map_ids,
    )
    used = [row for row in per_map if row["n_agents"] > 0]
    comp = np.asarray([row["completion_rate"] for row in used], dtype=np.float64)
    fail = np.asarray([row["failure_rate"] for row in used], dtype=np.float64)
    disp = np.asarray([row["displacement_mean"] for row in used], dtype=np.float64)
    comp_mean, comp_std = float(comp.mean()), float(comp.std())
    fail_mean, fail_std = float(fail.mean()), float(fail.std())
    disp_mean, disp_std = float(disp.mean()), float(disp.std())
    plot_scenario_eval(
        steps_d, err, steps_h, ent,
        comp_mean, comp_std, fail_mean, fail_std, disp_mean, disp_std,
        pair_d, pair_e, len(used), paths["eval_plot"],
        n_accel=n_accel, n_steer=n_steer,
        title=(
            f"{len(used)} maps, all controllable agents closed loop. "
            f"completion is the logged end within {GOAL_RADIUS_M:.0f} m "
            f"by step {HORIZON_STEPS}"
        ),
    )
    summary = {
        "n_maps": n_maps,
        "n_maps_with_agents": len(used),
        "samples": int(len(ya)),
        "epochs": int(epoch),
        "accuracy": acc,
        "accel_accuracy": acc_a,
        "steer_accuracy": acc_s,
        "completion_mean": comp_mean,
        "completion_std": comp_std,
        "failure_mean": fail_mean,
        "failure_std": fail_std,
        "displacement_mean": disp_mean,
        "displacement_std": disp_std,
        "maps": per_map,
    }
    paths["eval_json"].write_text(json.dumps(summary, indent=2))
    print(f"wrote {paths['eval_json']}", flush=True)
    print(f"completion {comp_mean:.3f} ± {comp_std:.3f}  "
        f"failure {fail_mean:.3f} ± {fail_std:.3f}", flush=True)
    print("DONE", flush=True)


def eval_held_out(train_maps=100, test_maps=100):
    """Closed-loop eval of the train_maps checkpoint on the next test_maps scenarios."""
    paths = _run_paths(train_maps)
    lib = drv.load()
    pol, acc = load_policy(paths["ckpt"])
    z = np.load(paths["ckpt"])
    n_accel, n_steer = int(z["n_accel"]), int(z["n_steer"])
    map_ids = list(range(train_maps, train_maps + test_maps))
    steps_d, err, steps_h, ent, per_map, _scenes, pair_d, pair_e = evaluate_closed_loop(
        lib, pol, map_ids, rollouts=10,
    )
    used = [row for row in per_map if row["n_agents"] > 0]
    comp = np.asarray([row["completion_rate"] for row in used], dtype=np.float64)
    fail = np.asarray([row["failure_rate"] for row in used], dtype=np.float64)
    disp = np.asarray([row["displacement_mean"] for row in used], dtype=np.float64)
    comp_mean, comp_std = float(comp.mean()), float(comp.std())
    fail_mean, fail_std = float(fail.mean()), float(fail.std())
    disp_mean, disp_std = float(disp.mean()), float(disp.std())
    out_plot = OUT / f"bc_{train_maps}_eval_test.png"
    out_json = OUT / f"bc_{train_maps}_eval_test.json"
    plot_scenario_eval(
        steps_d, err, steps_h, ent,
        comp_mean, comp_std, fail_mean, fail_std, disp_mean, disp_std,
        pair_d, pair_e, len(used), out_plot,
        n_accel=n_accel, n_steer=n_steer,
        title=(
            f"{len(used)} held-out maps ({map_ids[0]}–{map_ids[-1]}), "
            f"all controllable agents closed loop. "
            f"completion is the logged end within {GOAL_RADIUS_M:.0f} m "
            f"by step {HORIZON_STEPS}"
        ),
    )
    summary = {
        "split": "held_out",
        "train_maps": train_maps,
        "test_map_first": map_ids[0],
        "test_map_last": map_ids[-1],
        "n_maps": len(map_ids),
        "n_maps_with_agents": len(used),
        "rollouts_per_map": 10,
        "checkpoint": str(paths["ckpt"]),
        "train_accuracy": acc,
        "completion_mean": comp_mean,
        "completion_std": comp_std,
        "failure_mean": fail_mean,
        "failure_std": fail_std,
        "displacement_mean": disp_mean,
        "displacement_std": disp_std,
        "maps": per_map,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)
    print(f"completion {comp_mean:.3f} ± {comp_std:.3f}  "
        f"failure {fail_mean:.3f} ± {fail_std:.3f}  "
        f"displacement {disp_mean:.3f} ± {disp_std:.3f}", flush=True)
    print("DONE", flush=True)


def _solid_legend(ax, **kwargs):
    legend = ax.legend(**kwargs)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return legend


def _map_rates(per_map):
    used = [row for row in per_map if row["n_agents"] > 0]
    comp = np.asarray([row["completion_rate"] for row in used], dtype=np.float64)
    fail = np.asarray([row["failure_rate"] for row in used], dtype=np.float64)
    disp = np.asarray([row["displacement_mean"] for row in used], dtype=np.float64)
    return {
        "n": len(used),
        "comp_mean": float(comp.mean()), "comp_std": float(comp.std()),
        "fail_mean": float(fail.mean()), "fail_std": float(fail.std()),
        "disp_mean": float(disp.mean()), "disp_std": float(disp.std()),
    }


def plot_train_test(train, test, path, n_accel=21, n_steer=31, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    blue, orange = sns.color_palette("muted", n_colors=2)
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.3), constrained_layout=True)
    ax_d, ax_h, ax_b, ax_c = axes
    ax_h.sharex(ax_d)
    dot = {"s": 8, "alpha": 0.2, "linewidths": 0, "rasterized": True}
    ax_d.scatter(train["steps_d"], train["err"], color=blue, label="train", **dot)
    ax_d.scatter(test["steps_d"], test["err"], color=orange, label="test", **dot)
    ax_d.set_xlabel("timestep")
    ax_d.set_ylabel("displacement (m)")
    ax_d.set_title("displacement")
    _solid_legend(ax_d, loc="upper left", fontsize=8, frameon=False, markerscale=2)
    style_ax(ax_d)
    ax_h.scatter(train["steps_h"], train["ent"], color=blue, label="train", **dot)
    ax_h.scatter(test["steps_h"], test["ent"], color=orange, label="test", **dot)
    uniform = float(np.log(n_accel) + np.log(n_steer))
    ax_h.axhline(
        uniform, color="black", ls=(0, (6, 4)), lw=1.3, zorder=3,
        label=f"uniform {uniform:.2f}",
    )
    ax_h.set_ylim(0.0, uniform * 1.12)
    _solid_legend(ax_h, loc="upper right", fontsize=8, frameon=False, markerscale=2)
    ax_h.set_xlabel("step")
    ax_h.set_ylabel("entropy (nats)")
    ax_h.set_title("action entropy")
    style_ax(ax_h)
    width = 0.34
    rate_x = np.asarray([0.0, 1.0])
    for shift, split, color in ((-width / 2, train, blue), (width / 2, test, orange)):
        ax_b.bar(
            rate_x + shift,
            [split["comp_mean"], split["fail_mean"]],
            yerr=[split["comp_std"], split["fail_std"]],
            color=color, width=width, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=split["name"],
        )
    ax_m = ax_b.twinx()
    for shift, split, color in ((-width / 2, train, blue), (width / 2, test, orange)):
        ax_m.bar(
            [2.0 + shift], [split["disp_mean"]], yerr=[split["disp_std"]],
            color=color, width=width, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )
    ax_b.set_xticks([0, 1, 2], ["completion", "failure", "displacement"])
    ax_b.set_ylabel("rate")
    ax_m.set_ylabel("displacement (m)")
    ax_b.set_title("mean ± std across maps")
    rate_top = max(
        split[key] + split[std]
        for split in (train, test)
        for key, std in (("comp_mean", "comp_std"), ("fail_mean", "fail_std"))
    )
    disp_top = max(split["disp_mean"] + split["disp_std"] for split in (train, test))
    ax_b.set_ylim(0.0, max(1.25, rate_top + 0.32))
    ax_m.set_ylim(0.0, max(disp_top, 1e-6) * 1.45)
    ax_b.set_axisbelow(True)
    ax_b.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax_b, top=True, right=True)
    sns.despine(ax=ax_m, top=True, left=True, right=False)
    ax_b.legend(loc="upper right", fontsize=8, frameon=False)
    for shift, split in ((-width / 2, train), (width / 2, test)):
        for i, key, std in ((0, "comp_mean", "comp_std"), (1, "fail_mean", "fail_std")):
            ax_b.text(
                i + shift, split[key] + split[std] + 0.02,
                f"{split[key]:.2f} ± {split[std]:.2f}",
                ha="center", va="bottom", fontsize=6.5,
            )
        ax_m.text(
            2.0 + shift,
            split["disp_mean"] + split["disp_std"] + 0.03 * ax_m.get_ylim()[1],
            f"{split['disp_mean']:.2f} ± {split['disp_std']:.2f}",
            ha="center", va="bottom", fontsize=6.5,
        )
    for split, color in ((train, blue), (test, orange)):
        sns.regplot(
            x=split["pair_d"], y=split["pair_e"], ax=ax_c, ci=95,
            scatter_kws={"s": 8, "alpha": 0.2, "linewidths": 0, "rasterized": True,
                "color": color},
            line_kws={"color": color, "lw": 1.6},
        )
    y_text = 0.96
    for split, color in ((train, blue), (test, orange)):
        if len(split["pair_d"]) > 1:
            corr = float(np.corrcoef(split["pair_d"], split["pair_e"])[0, 1])
            ax_c.text(
                0.97, y_text, f"{split['name']} r = {corr:.2f}",
                transform=ax_c.transAxes, va="top", ha="right", fontsize=8, color=color,
            )
            y_text -= 0.08
    ax_c.set_xlabel("displacement (m)")
    ax_c.set_ylabel("entropy (nats)")
    ax_c.set_title("displacement vs entropy")
    style_ax(ax_c)
    fig.suptitle(title or "Train and held-out closed-loop eval")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path}")


def eval_train_test(train_maps=100, test_maps=100):
    paths = _run_paths(train_maps)
    lib = drv.load()
    pol, acc = load_policy(paths["ckpt"])
    z = np.load(paths["ckpt"])
    n_accel, n_steer = int(z["n_accel"]), int(z["n_steer"])
    train_ids = list(range(train_maps))
    test_ids = list(range(train_maps, train_maps + test_maps))
    packs = {}
    for name, map_ids in (("train", train_ids), ("test", test_ids)):
        steps_d, err, steps_h, ent, per_map, _scenes, pair_d, pair_e = evaluate_closed_loop(
            lib, pol, map_ids, rollouts=10,
        )
        rates = _map_rates(per_map)
        packs[name] = {
            "name": name,
            "steps_d": steps_d, "err": err, "steps_h": steps_h, "ent": ent,
            "pair_d": pair_d, "pair_e": pair_e,
            "per_map": per_map, **rates,
        }
    out_plot = OUT / f"bc_{train_maps}_eval_train_test.png"
    out_json = OUT / f"bc_{train_maps}_eval_train_test.json"
    plot_train_test(
        packs["train"], packs["test"], out_plot, n_accel=n_accel, n_steer=n_steer,
        title=(
            f"Train maps 0–{train_maps - 1} and held-out maps "
            f"{test_ids[0]}–{test_ids[-1]}. "
            f"completion is the logged end within {GOAL_RADIUS_M:.0f} m "
            f"by step {HORIZON_STEPS}"
        ),
    )
    summary = {
        "checkpoint": str(paths["ckpt"]),
        "train_accuracy": acc,
        "rollouts_per_map": 10,
        "train_map_first": train_ids[0],
        "train_map_last": train_ids[-1],
        "test_map_first": test_ids[0],
        "test_map_last": test_ids[-1],
        "train": {k: packs["train"][k] for k in (
            "n", "comp_mean", "comp_std", "fail_mean", "fail_std",
            "disp_mean", "disp_std")},
        "test": {k: packs["test"][k] for k in (
            "n", "comp_mean", "comp_std", "fail_mean", "fail_std",
            "disp_mean", "disp_std")},
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)
    print("DONE", flush=True)


CARLA_DIR = drv.ROOT / "data" / "mixed"
CARLA_TOWNS = ((1, "Town01"), (2, "Town02"))
CARLA_AGENTS = 32
CARLA_GOAL_MIN = 19.5
CARLA_GOAL_MAX = 20.5
CARLA_HORIZON = 200
CARLA_ROLLOUTS = 10
# Above the horizon so puf_step does not reset the episode at step 200.
CARLA_EPISODE = 100000


def _mean_std(flags):
    per = np.asarray(flags, dtype=np.float64).mean(axis=1)
    return float(per.mean()), float(per.std())


def carla_rollout(lib, h, pol, steps):
    """Greedy BC on a drive.h sim. Observations and both action bins are the env's."""
    n = lib.pf_n(h)
    assert lib.pf_obs_size(h) == int(np.asarray(pol.mu).shape[0])
    completed = np.zeros(n, dtype=bool)
    collided = np.zeros(n, dtype=bool)
    offroad = np.zeros(n, dtype=bool)
    hit = np.zeros((steps, n), dtype=bool)
    pose = drv.states(lib, h)
    assert np.all(pose[:, 10] > 0.5)
    goal_dist = np.hypot(pose[:, 0] - pose[:, 5], pose[:, 1] - pose[:, 6])
    coast_a = float(lib.pf_coast_accel())
    coast_s = float(lib.pf_coast_steer())
    for t in range(steps):
        obs = drv.observations(lib, h)
        live = np.flatnonzero(pose[:, 10] > 0.5)
        acc = np.full(n, coast_a, dtype=np.float64)
        steer = np.full(n, coast_s, dtype=np.float64)
        if len(live):
            batch = (obs[live] - pol.mu) / pol.sig
            pa, ps, _ent = pol.act_batch(batch)
            acc[live] = pa
            steer[live] = ps
        for i in range(n):
            lib.pf_set_action(h, i, float(acc[i]), float(steer[i]))
        old_xy = pose[:, 0:2].copy()
        old_goal = pose[:, 5:7].copy()
        lib.pf_step(h)
        pose = drv.states(lib, h)
        kind = pose[:, 9].astype(np.int32)
        step_col = kind == 1
        collided |= step_col
        offroad |= kind == 2
        hit[t] = step_col
        moved = np.hypot(pose[:, 5] - old_goal[:, 0], pose[:, 6] - old_goal[:, 1]) > 1.0
        dist = np.hypot(pose[:, 0] - old_goal[:, 0], pose[:, 1] - old_goal[:, 1])
        jumped = np.hypot(pose[:, 0] - old_xy[:, 0], pose[:, 1] - old_xy[:, 1]) > 12.0
        completed |= moved | (dist < GOAL_RADIUS_M) | jumped
    return completed, collided, offroad, hit, goal_dist


def _womd_split(name):
    path = OUT / f"bc_100_eval.json" if name == "train" else OUT / f"bc_100_eval_test.json"
    summary = json.loads(path.read_text())
    return {
        "name": f"WOMD {name}",
        "completion_mean": float(summary["completion_mean"]),
        "completion_std": float(summary["completion_std"]),
        "failure_mean": float(summary["failure_mean"]),
        "failure_std": float(summary["failure_std"]),
    }


def plot_carla(towns, womd, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    colors = sns.color_palette("muted", n_colors=4)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), constrained_layout=True)
    ax_r, ax_c, ax_w = axes

    width = 0.36
    metrics = ("completion", "offroad", "collision")
    labels = ("completion", "off-road", "collision")
    x = np.arange(len(metrics))
    for shift, town, color in ((-width / 2, towns[0], colors[0]), (width / 2, towns[1], colors[1])):
        means = [town[f"{key}_mean"] for key in metrics]
        stds = [town[f"{key}_std"] for key in metrics]
        ax_r.bar(
            x + shift, means, yerr=stds, color=color, width=width, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0}, label=town["name"],
        )
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax_r.text(
                i + shift, mean + std + 0.02, f"{mean:.2f} ± {std:.2f}",
                ha="center", va="bottom", fontsize=6.5,
            )
    ax_r.set_xticks(x, labels)
    ax_r.set_ylabel("rate")
    ax_r.set_ylim(0.0, 1.45)
    ax_r.set_title("mean ± std across 10 rollouts")
    ax_r.legend(loc="upper right", fontsize=8, frameon=False)
    ax_r.set_axisbelow(True)
    ax_r.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax_r)

    for town, color, dodge in (
        (towns[0], colors[0], -0.16),
        (towns[1], colors[1], 0.16),
    ):
        counts = np.asarray(town["collision_counts"], dtype=np.int32)
        marked = False
        for rollout, row in enumerate(counts):
            for step, count in enumerate(row):
                if count <= 0:
                    continue
                ys = rollout + 1 + dodge + np.linspace(-0.12, 0.12, int(count))
                ax_c.scatter(
                    np.full(int(count), step), ys, s=9, alpha=0.55, linewidths=0,
                    color=color, label=town["name"] if not marked else None,
                )
                marked = True
    ax_c.set_xlim(-2, CARLA_HORIZON + 1)
    ax_c.set_ylim(0.4, CARLA_ROLLOUTS + 0.6)
    ax_c.set_yticks(np.arange(1, CARLA_ROLLOUTS + 1))
    ax_c.set_xlabel("step")
    ax_c.set_ylabel("rollout")
    ax_c.set_title("collisions over the horizon")
    _solid_legend(ax_c, loc="upper right", fontsize=8, frameon=True, markerscale=2)
    style_ax(ax_c)

    contrast = list(towns) + list(womd)
    width_w = 0.18
    shifts = (np.arange(len(contrast)) - (len(contrast) - 1) / 2.0) * width_w
    xw = np.arange(2)
    for shift, split, color in zip(shifts, contrast, colors):
        means = [split["completion_mean"], split["failure_mean"]]
        stds = [split["completion_std"], split["failure_std"]]
        ax_w.bar(
            xw + shift, means, yerr=stds, color=color, width=width_w, capsize=2,
            error_kw={"elinewidth": 1.0, "capthick": 1.0}, label=split["name"],
        )
    ax_w.set_xticks(xw, ["completion", "failure"])
    ax_w.set_ylabel("rate")
    ax_w.set_ylim(0.0, 1.45)
    ax_w.set_title("against WOMD train and test")
    ax_w.set_xlabel("failure is collision or off-road")
    ax_w.legend(loc="upper right", fontsize=7, frameon=False)
    ax_w.set_axisbelow(True)
    ax_w.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax_w)

    fig.suptitle(
        "BC policy in the drive.h env on CARLA Town01 and Town02\n"
        f"{CARLA_AGENTS} agents, goals {CARLA_GOAL_MIN:.1f}–{CARLA_GOAL_MAX:.1f} m, "
        f"{CARLA_HORIZON} steps, {CARLA_ROLLOUTS} rollouts. "
        f"Completion is the {GOAL_RADIUS_M:.0f} m goal disk."
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path}")


def eval_carla(rollouts=CARLA_ROLLOUTS, steps=CARLA_HORIZON):
    """Closed-loop BC in CARLA Town01 and Town02 through drive.h."""
    paths = _run_paths(100)
    lib = drv.load()
    pol, acc = load_policy(paths["ckpt"])
    towns = []
    for map_id, name in CARLA_TOWNS:
        h = drv.create_sim(
            lib, CARLA_DIR, map_id, CARLA_AGENTS, episode_length=CARLA_EPISODE,
            goal_min=CARLA_GOAL_MIN, goal_max=CARLA_GOAL_MAX, physics="through",
        )
        try:
            done = []
            collided = []
            offroad = []
            counts = []
            goal_dists = []
            for r in range(rollouts):
                lib.pf_reset(h, 1000 * map_id + r)
                comp, col, off, hit, gdist = carla_rollout(lib, h, pol, steps)
                assert np.all((gdist >= CARLA_GOAL_MIN - 1e-3) & (gdist <= CARLA_GOAL_MAX + 1e-3))
                done.append(comp)
                collided.append(col)
                offroad.append(off)
                counts.append(hit.sum(axis=1))
                goal_dists.append(gdist)
                print(
                    f"{name} rollout {r} completion {comp.mean():.2f} "
                    f"offroad {off.mean():.2f} collision {col.mean():.2f} "
                    f"goal {gdist.mean():.2f} m",
                    flush=True,
                )
            done = np.asarray(done)
            collided = np.asarray(collided)
            offroad = np.asarray(offroad)
            failure = collided | offroad
            comp_mean, comp_std = _mean_std(done)
            off_mean, off_std = _mean_std(offroad)
            col_mean, col_std = _mean_std(collided)
            fail_mean, fail_std = _mean_std(failure)
            towns.append({
                "name": name,
                "map_id": map_id,
                "completion_mean": comp_mean,
                "completion_std": comp_std,
                "offroad_mean": off_mean,
                "offroad_std": off_std,
                "collision_mean": col_mean,
                "collision_std": col_std,
                "failure_mean": fail_mean,
                "failure_std": fail_std,
                "collision_counts": np.asarray(counts, dtype=np.int32).tolist(),
                "goal_dist_mean": float(np.mean(goal_dists)),
                "goal_dist_min": float(np.min(goal_dists)),
                "goal_dist_max": float(np.max(goal_dists)),
            })
            print(
                f"{name} completion {comp_mean:.3f} ± {comp_std:.3f}  "
                f"offroad {off_mean:.3f} ± {off_std:.3f}  "
                f"collision {col_mean:.3f} ± {col_std:.3f}",
                flush=True,
            )
        finally:
            lib.pf_close(h)
    womd = [_womd_split("train"), _womd_split("test")]
    out_plot = OUT / "bc_100_eval_carla.png"
    out_json = OUT / "bc_100_eval_carla.json"
    plot_carla(towns, womd, out_plot)
    summary = {
        "checkpoint": str(paths["ckpt"]),
        "train_accuracy": acc,
        "env": "drive.h",
        "data_dir": str(CARLA_DIR),
        "num_agents": CARLA_AGENTS,
        "goal_min_m": CARLA_GOAL_MIN,
        "goal_max_m": CARLA_GOAL_MAX,
        "horizon_steps": steps,
        "rollouts": rollouts,
        "physics": "through",
        "completion": "reached the 5 m goal disk at least once",
        "offroad": "off-road at least once",
        "collision": "vehicle collision at least once",
        "failure": "vehicle collision or off-road at least once",
        "carla_std": "across rollouts",
        "womd_std": "across maps",
        "towns": [{k: v for k, v in town.items() if k != "collision_counts"} for town in towns],
        "collision_counts": {town["name"]: town["collision_counts"] for town in towns},
        "womd": womd,
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_json}", flush=True)
    print("DONE", flush=True)


def main():
    lib = drv.load()
    h = drv.create_scenario(lib, DATA, MAP_ID, JSONS[MAP_ID])
    try:
        obs, acts, valid, _ade = drv.collect(lib, h, 1)
        save_dataset(obs, acts, valid, _ade, 1, DATASET_TELEPORT)
        obs_nt, acts_nt, valid_nt, ade_nt = drv.collect(lib, h, 0)
        save_dataset(obs_nt, acts_nt, valid_nt, ade_nt, 0, DATASET_OPEN)
        n_accel = lib.pf_n_accel()
        n_steer = lib.pf_n_steer()
        sdc = lib.pf_sdc(h)
        controlled = sdc
        if lib.pf_base_unfit(h, sdc):
            best = -1
            best_n = -1
            for i in range(lib.pf_log_n(h)):
                n = int(valid[:, i].sum())
                if n > best_n:
                    best_n = n
                    best = i
            controlled = best
            print(f"sdc {sdc} is unfit, controlling agent {controlled}")
        x_all, ya_all, ys_all = pairs(obs, acts, valid)
        bad, total = conflicts(x_all, ya_all, ys_all)
        print(f"samples {total} conflicting labels {bad} controlled {controlled}")
        x, ya, ys = x_all, ya_all, ys_all
        if bad > 0 and (total - bad) / total < 0.99:
            mask = valid[:, controlled].astype(bool)
            x = obs[:, controlled][mask].astype(np.float32)
            y = acts[:, controlled][mask].astype(np.int64)
            ya, ys = y[:, 0], y[:, 1]
            print(f"training on agent {controlled} only, {len(x)} samples")
        mu = x.mean(axis=0)
        sig = x.std(axis=0)
        sig = np.where(sig < 1e-6, 1.0, sig).astype(np.float32)
        x = ((x - mu) / sig).astype(np.float32)
        pol, acc, acc_a, acc_s, epoch, hist = train(x, ya, ys, n_accel, n_steer)
        plot_training(hist)
        pol.mu = mu.astype(np.float32)
        pol.sig = sig
        print(f"checkpoint accuracy joint {acc:.4f} accel {acc_a:.4f} steer {acc_s:.4f}")
        if acc < 0.99:
            raise SystemExit(f"accuracy {acc:.4f} stayed below 0.99")
        save_policy(pol, acc, n_accel, n_steer)
        records = evaluate_agents(lib, h, pol, acc)
        sdc_rec = next(r for r in records if r["agent"] == sdc) if any(
            r["agent"] == sdc for r in records) else records[0]
        METRICS.parent.mkdir(parents=True, exist_ok=True)
        METRICS.write_text(json.dumps({
            "map_id": MAP_ID,
            "json": JSONS[MAP_ID].name,
            "controlled": [r["agent"] for r in records],
            "sdc": sdc,
            "samples": int(len(x)),
            "epochs": int(epoch),
            "accuracy": acc,
            "accel_accuracy": acc_a,
            "steer_accuracy": acc_s,
            "n_accel": n_accel,
            "n_steer": n_steer,
            "agents": [{
                "agent": r["agent"],
                "human_x": r["human"][0],
                "human_y": r["human"][1],
                "rollouts": r["rows"],
            } for r in records],
            "rollouts": sdc_rec["rows"],
            "human_x": sdc_rec["human"][0],
            "human_y": sdc_rec["human"][1],
        }, indent=2))
        print(f"wrote {CKPT}")
        print(f"wrote {METRICS}")
    finally:
        lib.pf_close(h)


if __name__ == "__main__":
    main()
