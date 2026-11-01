"""Behavioral cloning on Waymo logs, then closed-loop rollouts."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
TRAIN_PLOT = OUT / "bc_map000_train.png"
HORIZON_STEPS = 100
GOAL_RADIUS_M = 5.0

def apply_figure_style():
    sns.set_style("ticks")

def style_ax(ax):
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax)

# Width is a multiple of 64. The loader is linear layers and leaky ReLU.
POLICY_HIDDEN = 704
POLICY_LAYERS = 3

class Policy(torch.nn.Module):
    def __init__(self, obs_n, n_accel, n_steer, hidden=POLICY_HIDDEN,
            layers=POLICY_LAYERS, seed=0):
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

    def trunk(self, x):
        h = x
        for layer in self.hidden:
            h = torch.nn.functional.leaky_relu(layer(h), negative_slope=0.01)
        return h

    def forward(self, x):
        h = self.trunk(x)
        return self.head_a(h), self.head_s(h)

    def _batch(self, obs):
        values = torch.as_tensor(
            np.asarray(obs), dtype=torch.float32,
            device=self.head_a.weight.device,
        )
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
        "epoch", "phase", "step", "loss", "acc", "acc_a", "acc_s",
        "logp", "logp_a", "logp_s", "ent", "ent_a", "ent_s",
        "bin_a", "bin_s",
        "acc_a1", "acc_s1", "acc1",
        "acc_a5", "acc_s5", "acc5", "acc_a10", "acc_s10", "acc10",
    )
    return {k: [] for k in keys}

def record_stats(hist, epoch, phase, stats, loss, step):
    hist["epoch"].append(int(epoch))
    hist["phase"].append(phase)
    hist["loss"].append(float(loss))
    hist["step"].append(int(step))
    for key, value in stats.items():
        hist[key].append(float(value))

def _metric_row(epoch, phase, step, loss, lr, stats):
    row = {
        "epoch": int(epoch),
        "phase": phase,
        "step": int(step),
        "loss": float(loss),
        "lr": float(lr),
    }
    for key, value in stats.items():
        row[key] = float(value)
    return row

def _write_metric(fh, row):
    if fh is None:
        return
    fh.write(json.dumps(row) + "\n")
    fh.flush()

def _device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device("cuda")
    print(f"Device: {device} {torch.cuda.get_device_name(0)}", flush=True)
    return device

class _IndexedPairs(torch.utils.data.Dataset):
    # Hard-example passes repeat hundreds of thousands of rows. Indexing the
    # big observation tensor would copy tens of gigabytes.
    def __init__(self, obs, actions, index):
        self.obs = obs
        self.actions = actions
        self.index = index

    def __len__(self):
        return int(self.index.shape[0])

    def __getitem__(self, i):
        j = int(self.index[i])
        return self.obs[j], self.actions[j]

def _loader(obs, actions, batch_size, index=None):
    if index is None:
        data = TensorDataset(obs, actions)
    else:
        data = _IndexedPairs(obs, actions, index)
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
    )

def _epoch_loss(pol, loader, opt, device):
    pol.train()
    total = 0.0
    count = 0
    steps = 0
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
        steps += 1
    return total / max(count, 1), steps

def _full_stats(pol, obs, actions, device):
    pol.eval()
    n = obs.shape[0]
    both = acc_a = acc_s = 0
    logp_a = logp_s = ent_a = ent_s = 0.0
    bin_a = bin_s = 0.0
    a1 = s1 = 0
    a5 = s5 = 0
    a10 = s10 = 0
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
            err_a = (pred_a - yb[:, 0]).abs()
            err_s = (pred_s - yb[:, 1]).abs()
            bin_a += float(err_a.sum())
            bin_s += float(err_s.sum())
            a1 += int((err_a <= 1).sum())
            s1 += int((err_s <= 1).sum())
            near_a5 = err_a <= 5
            near_s5 = err_s <= 5
            near_a10 = err_a <= 10
            near_s10 = err_s <= 10
            a5 += int(near_a5.sum())
            s5 += int(near_s5.sum())
            a10 += int(near_a10.sum())
            s10 += int(near_s10.sum())
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
        "bin_a": bin_a / n,
        "bin_s": bin_s / n,
        "acc_a1": a1 / n,
        "acc_s1": s1 / n,
        "acc1": (a1 + s1) / (2 * n),
        "acc_a5": a5 / n,
        "acc_s5": s5 / n,
        "acc5": (a5 + s5) / (2 * n),
        "acc_a10": a10 / n,
        "acc_s10": s10 / n,
        "acc10": (a10 + s10) / (2 * n),
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

STOP_JOINT = 0.99
# Mean of accel-within-10 and steer-within-10. train_and_eval_maps stops
# here and writes the checkpoint.
STOP_WITHIN10 = 0.95
MAX_EPOCHS = 5000

def bin_centers(lo, hi, n):
    n = int(n)
    if n % 2 == 0:
        raise ValueError(f"bin count {n} is even, so no center bin is exactly 0")
    values = np.linspace(float(lo), float(hi), n).astype(np.float32)
    values[n // 2] = np.float32(0.0)
    return values

def train(x, ya, ys, n_accel, n_steer, batch_size=2048, ckpt_path=None, mu=None, sig=None,
        metric_path=None, accel_min=None, accel_max=None, steer_min=None, steer_max=None,
        resume_ckpt=None, start_epoch=0, start_step=0, start_lr=None, resume_loss=None,
        metric_append=False, enter_hard=False, max_epochs=None,
        stop_within10=None, hidden=POLICY_HIDDEN, layers=POLICY_LAYERS):
    device = _device()
    pol = Policy(x.shape[1], n_accel, n_steer, hidden=hidden, layers=layers).to(device)
    nparam = sum(p.numel() for p in pol.parameters())
    print(f"policy hidden {hidden} x {layers} params {nparam}", flush=True)
    if mu is not None:
        pol.mu = np.asarray(mu, dtype=np.float32)
        pol.sig = np.asarray(sig, dtype=np.float32)
    if accel_min is not None:
        pol.accel_min = float(accel_min)
        pol.accel_max = float(accel_max)
        pol.steer_min = float(steer_min)
        pol.steer_max = float(steer_max)
        pol.accel_values = bin_centers(pol.accel_min, pol.accel_max, n_accel)
        pol.steer_values = bin_centers(pol.steer_min, pol.steer_max, n_steer)
    opt = Adam(pol.parameters(), lr=1e-3 if start_lr is None else float(start_lr))
    obs = torch.from_numpy(np.ascontiguousarray(x))
    actions = torch.stack((
        torch.from_numpy(np.ascontiguousarray(ya)),
        torch.from_numpy(np.ascontiguousarray(ys)),
    ), dim=1).long()
    loader = _loader(obs, actions, batch_size)
    hist = empty_history()
    lr = 1e-3 if start_lr is None else float(start_lr)
    best = 0.0
    stall = 0
    acc = acc_a = acc_s = 0.0
    epoch = int(start_epoch)
    step = int(start_step)
    loss = float("nan") if resume_loss is None else float(resume_loss)
    stats = None
    best_acc = -1.0
    best_acc_a = best_acc_s = 0.0
    best_score = -1.0
    best_epoch = 0
    best_state = None
    metric_fh = None
    if metric_path is not None:
        metric_path = Path(metric_path)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_fh = open(metric_path, "a" if metric_append else "w")
    if resume_ckpt is not None:
        loaded, saved_acc = load_policy(resume_ckpt)
        pol.load_state_dict(loaded.state_dict())
        best_acc = float(saved_acc)
        best = float(saved_acc)
        best_epoch = int(start_epoch)
        best_state = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
        print(
            f"resume weights joint {saved_acc:.4f} from {resume_ckpt} "
            f"at epoch {start_epoch} step {start_step} lr {lr:.3e}",
            flush=True,
        )

    def _score():
        if stop_within10 is not None:
            return stats["acc10"]
        return acc

    def _keep():
        nonlocal best_acc, best_acc_a, best_acc_s, best_epoch, best_state, best_score
        score = _score()
        if score <= best_score:
            return
        best_score = score
        best_acc = acc
        best_acc_a = acc_a
        best_acc_s = acc_s
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in pol.state_dict().items()}
        if ckpt_path is not None and pol.mu is not None:
            save_policy(pol, acc, n_accel, n_steer, ckpt_path)
            if stop_within10 is not None:
                print(
                    f"best within10 {stats['acc10']:.4f} joint {acc:.4f} "
                    f"epoch {epoch}",
                    flush=True,
                )
            else:
                print(f"best joint {acc:.4f} epoch {epoch}", flush=True)

    def _log(phase, left=None):
        record_stats(hist, epoch, phase, stats, loss, step)
        _write_metric(metric_fh, _metric_row(epoch, phase, step, loss, lr, stats))
        tail = ""
        if left is not None:
            tail = f" left {left}"
        print(
            f"epoch {epoch:4d} step {step:8d} loss {loss:.4f} "
            f"joint {acc:.3f} accel {acc_a:.3f} steer {acc_s:.3f} "
            f"a1 {stats['acc_a1']:.3f} s1 {stats['acc_s1']:.3f} m1 {stats['acc1']:.3f} "
            f"a5 {stats['acc_a5']:.3f} s5 {stats['acc_s5']:.3f} m5 {stats['acc5']:.3f} "
            f"a10 {stats['acc_a10']:.3f} s10 {stats['acc_s10']:.3f} m10 {stats['acc10']:.3f} "
            f"logp {stats['logp']:.3f} ent {stats['ent']:.3f} "
            f"bin {stats['bin_a']:.2f}/{stats['bin_s']:.2f}{tail}",
            flush=True,
        )

    limit = MAX_EPOCHS if max_epochs is None else int(max_epochs)
    if limit < 1:
        raise ValueError("max_epochs must be at least 1")
    try:
        if resume_ckpt is not None:
            stats = _full_stats(pol, obs, actions, device)
            acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
            best_score = _score()
            best = best_score
            _log("resume")
        if not enter_hard:
            for epoch in range(int(start_epoch) + 1, limit + 1):
                loss, nstep = _epoch_loss(pol, loader, opt, device)
                step += nstep
                stats = _full_stats(pol, obs, actions, device)
                acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
                _log("train")
                _keep()
                watched = _score()
                if watched > best + 1e-4:
                    best = watched
                    stall = 0
                else:
                    stall += 1
                if stall == 60:
                    lr *= 0.5
                    _set_lr(opt, lr)
                    stall = 0
                    print(f"lr {lr:.2e}", flush=True)
                if stop_within10 is not None:
                    if watched >= float(stop_within10):
                        break
                elif acc > STOP_JOINT or lr < 1e-6:
                    break
        if (
            stop_within10 is None
            and max_epochs is None
            and acc <= STOP_JOINT
            and epoch < MAX_EPOCHS
        ):
            wrong = _mistakes(pol, obs, actions, device)
            print(f"hard examples {len(wrong)}", flush=True)
            # 1e-3 dropped a 0.564 policy to 0.386 and it did not recover.
            lr = 1e-4
            _set_lr(opt, lr)
            print(f"hard lr reset {lr:.2e}", flush=True)
            extra = 0
            while epoch < MAX_EPOCHS:
                extra += 1
                epoch += 1
                wrong = _mistakes(pol, obs, actions, device)
                if len(wrong) == 0:
                    stats = _full_stats(pol, obs, actions, device)
                    acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
                    _log("hard", left=0)
                    _keep()
                    break
                idx = torch.cat((torch.arange(obs.shape[0]), wrong.repeat(12)))
                loss, nstep = _epoch_loss(
                    pol, _loader(obs, actions, batch_size, index=idx), opt, device,
                )
                step += nstep
                stats = _full_stats(pol, obs, actions, device)
                acc, acc_a, acc_s = stats["acc"], stats["acc_a"], stats["acc_s"]
                _log("hard", left=len(wrong))
                _keep()
                if acc > STOP_JOINT:
                    break
                if extra % 80 == 0 and lr > 1e-5:
                    lr *= 0.5
                    _set_lr(opt, lr)
                    print(f"hard lr {lr:.2e}", flush=True)
        if best_state is not None and stats is not None and _score() < best_score:
            pol.load_state_dict(best_state)
            if stop_within10 is not None:
                print(
                    f"restored best within10 {best_score:.4f} from epoch {best_epoch} "
                    f"(last within10 {_score():.4f})",
                    flush=True,
                )
            else:
                print(
                    f"restored best joint {best_acc:.4f} from epoch {best_epoch} "
                    f"(last joint {acc:.4f})",
                    flush=True,
                )
            acc, acc_a, acc_s = best_acc, best_acc_a, best_acc_s
    finally:
        if metric_fh is not None:
            metric_fh.close()
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
    if hasattr(pol, "accel_min"):
        payload["accel_min"] = np.float32(pol.accel_min)
        payload["accel_max"] = np.float32(pol.accel_max)
        payload["steer_min"] = np.float32(pol.steer_min)
        payload["steer_max"] = np.float32(pol.steer_max)
    for i, layer in enumerate(pol.hidden):
        payload[f"W{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float32)
        payload[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    tmp = path.with_name(f".{path.stem}.partial.npz")
    np.savez(tmp, **payload)
    os.replace(tmp, path)
    print(f"wrote {path}")

def rollout(lib, h, pol, controlled, steps=None):
    if steps is None:
        steps = drv.TRAJ - 1
    _require_policy_bins(lib, pol)
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
    hidden = int(z["W0"].shape[1])
    n_layers = sum(1 for k in z.files if k.startswith("W") and k[1:].isdigit())
    pol = Policy(
        obs_n, int(z["n_accel"]), int(z["n_steer"]),
        hidden=hidden, layers=n_layers,
    )
    for i, layer in enumerate(pol.hidden):
        _copy_linear(layer, z[f"W{i}"].astype(np.float32), z[f"b{i}"].astype(np.float32))
    _copy_linear(pol.head_a, z["Wa"].astype(np.float32), z["ba"].astype(np.float32))
    _copy_linear(pol.head_s, z["Ws"].astype(np.float32), z["bs"].astype(np.float32))
    pol.mu = z["mu"].astype(np.float32)
    pol.sig = z["sig"].astype(np.float32)
    if "accel_min" in z.files:
        pol.accel_min = float(z["accel_min"])
        pol.accel_max = float(z["accel_max"])
        pol.steer_min = float(z["steer_min"])
        pol.steer_max = float(z["steer_max"])
        n_accel = int(z["n_accel"])
        n_steer = int(z["n_steer"])
        if n_accel % 2 == 0 or n_steer % 2 == 0:
            pol.accel_values = np.linspace(
                pol.accel_min, pol.accel_max, n_accel,
            ).astype(np.float32)
            pol.steer_values = np.linspace(
                pol.steer_min, pol.steer_max, n_steer,
            ).astype(np.float32)
        else:
            pol.accel_values = bin_centers(pol.accel_min, pol.accel_max, n_accel)
            pol.steer_values = bin_centers(pol.steer_min, pol.steer_max, n_steer)
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

def squared_displacement(human, rollout):
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

def plot_training(hist, path=None, title="BC training on map 000"):
    path = Path(path) if path else TRAIN_PLOT
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    epoch = np.asarray(hist["epoch"])
    phases = [str(p) for p in hist["phase"]]
    fig, axes = plt.subplots(
        2, 2, figsize=(10.4, 7.2), sharex=True, constrained_layout=True,
    )
    axes = axes.ravel()
    series = (
        (axes[0], ("acc", "acc_a", "acc_s"), ("joint", "accel", "steer"), "accuracy"),
        (axes[1], ("logp", "logp_a", "logp_s"),
            ("joint", "accel", "steer"), "log prob (nats)"),
        (axes[2], ("ent", "ent_a", "ent_s"),
            ("joint", "accel", "steer"), "entropy (nats)"),
        (axes[3], ("acc10", "acc_a10", "acc_s10"),
            ("mean", "accel", "steer"), "within 10 bins"),
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
    fig.savefig(path, dpi=100)
    plt.close(fig)
    np.savez(path.with_suffix(".npz"), **{k: np.asarray(v) for k, v in hist.items()})
    print(f"wrote {path}")

N_MAPS = 100
CKPT_100 = drv.ROOT / "checkpoints" / "bc_womd_100.npz"
TRAIN_PLOT_100 = OUT / "bc_100_train.png"

def collect_pairs(lib, map_ids, teleport=1, grid=None):
    xs, yas, yss = [], [], []
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            if grid is not None:
                lib.pf_relabel_grid(
                    h, grid[0], grid[1], grid[2], grid[3], grid[4], grid[5],
                )
            obs, acts, valid, _ade = drv.collect(lib, h, teleport)
            x, ya, ys = pairs(obs, acts, valid)
            if grid is not None and len(ya):
                if int(ya.min()) < 0 or int(ya.max()) >= grid[2]:
                    raise RuntimeError(f"accel bin outside 0..{grid[2] - 1}")
                if int(ys.min()) < 0 or int(ys.max()) >= grid[5]:
                    raise RuntimeError(f"steer bin outside 0..{grid[5] - 1}")
            xs.append(x)
            yas.append(ya)
            yss.append(ys)
            print(f"map {map_id} pairs {len(x)}", flush=True)
        finally:
            lib.pf_close(h)
    return np.concatenate(xs), np.concatenate(yas), np.concatenate(yss)

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
    _require_policy_bins(lib, pol)
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
                # Bin indices. move_dynamics reads the same table RL trains on.
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

def evaluate_closed_loop(lib, pol, map_ids, rollouts=10):
    disp_steps, disp_err = [], []
    ent_steps, ent_vals = [], []
    pair_disp, pair_ent = [], []
    per_map = []
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            agents = controllable_agents(lib, h)
            ends = {}
            for agent in agents:
                xs, ys = log_track(lib, h, agent)
                ends[agent] = (xs[-1], ys[-1])
            completed = []
            failed = []
            scored = []
            agent_disp = []
            for _ in range(rollouts):
                tracks = closed_loop_rollout(lib, h, pol, agents, ends)
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
                    scored.append(bool(rec["completed"]) and not rec["failed"])
            comp = float(np.mean(completed)) if completed else float("nan")
            fail = float(np.mean(failed)) if failed else float("nan")
            score = float(np.mean(scored)) if scored else float("nan")
            disp_mean_map = float(np.mean(agent_disp)) if agent_disp else float("nan")
            per_map.append({
                "map_id": int(map_id),
                "n_agents": len(agents),
                "n_rollouts": len(completed),
                "completion_rate": comp,
                "failure_rate": fail,
                "score_rate": score,
                "displacement_mean": disp_mean_map,
            })
            print(
                f"eval map {map_id} agents {len(agents)} "
                f"rollouts {rollouts} completion {comp:.2f} failure {fail:.2f}",
                flush=True,
            )
        finally:
            lib.pf_close(h)
    empty = np.zeros(0, dtype=np.float32)
    steps_d = np.concatenate(disp_steps) if disp_steps else empty.astype(np.int32)
    err = np.concatenate(disp_err) if disp_err else empty
    steps_h = np.concatenate(ent_steps) if ent_steps else empty.astype(np.int32)
    ent = np.concatenate(ent_vals) if ent_vals else empty
    paired_d = np.concatenate(pair_disp) if pair_disp else empty
    paired_e = np.concatenate(pair_ent) if pair_ent else empty
    return steps_d, err, steps_h, ent, per_map, paired_d, paired_e

def plot_action_histogram(ya, ys, path, n_samples, accel_min=-8.0, accel_max=8.0,
        n_accel=257, steer_min=-1.5, steer_max=1.5, n_steer=257, title=None):
    # Endpoint bars include commands clipped onto the bound.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    accel = np.linspace(accel_min, accel_max, n_accel)[np.asarray(ya, dtype=np.int64)]
    steer = np.linspace(steer_min, steer_max, n_steer)[np.asarray(ys, dtype=np.int64)]
    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.3), constrained_layout=True)
    blue = sns.color_palette()[0]
    panels = (
        (axes[0], accel, accel_min, accel_max, n_accel, "acceleration (m/s²)"),
        (axes[1], steer, steer_min, steer_max, n_steer, "steering (rad)"),
    )
    for ax, values, lo, hi, n_bins, xlabel in panels:
        step = (hi - lo) / (n_bins - 1)
        edges = np.linspace(lo - step / 2.0, hi + step / 2.0, n_bins + 1)
        ax.hist(values, bins=edges, density=True, color=blue, alpha=0.9, zorder=2)
        p1, p99 = np.percentile(values, [1, 99])
        ax.axvline(lo, color="#c0392b", lw=1.4, zorder=3)
        ax.axvline(hi, color="#c0392b", lw=1.4, zorder=3)
        ax.axvline(p1, color="#e0b000", lw=1.6, ls=(0, (4, 2)), zorder=4)
        ax.axvline(p99, color="#e0b000", lw=1.6, ls=(0, (4, 2)), zorder=4)
        span = hi - lo
        ax.set_xlim(lo - 0.06 * span, hi + 0.06 * span)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ends = float(np.mean((values <= lo + step * 0.5) | (values >= hi - step * 0.5)))
        ax.set_title(f"1st {p1:.2f}    99th {p99:.2f}    endpoint bins {ends:.1%}")
        style_ax(ax)
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=blue, lw=6, label="empirical density"),
            Line2D([0], [0], color="#e0b000", lw=1.6, ls=(0, (4, 2)),
                label="1st and 99th percentile"),
            Line2D([0], [0], color="#c0392b", lw=1.2, label="action bounds"),
        ],
        fontsize=8, loc="upper right", frameon=False,
    )
    if title is None:
        title = (
            f"Kinematic bicycle labels, {n_samples:,} pairs    "
            f"accel [{accel_min:g}, {accel_max:g}] × {n_accel}    "
            f"steer [{steer_min:g}, {steer_max:g}] × {n_steer}"
        )
    fig.suptitle(title)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")

def _run_paths(n_maps):
    tag = f"bc_{int(n_maps)}"
    return {
        "dataset": OUT / f"{tag}_teleport.npz",
        "train_plot": OUT / f"{tag}_train.png",
        "ckpt": drv.ROOT / "checkpoints" / f"bc_womd_{int(n_maps)}.npz",
    }

def rl_action_space():
    text = (drv.ROOT / "ocean" / "drive" / "constants.h").read_text()

    def macro(name):
        match = re.search(
            rf"#define\s+{name}\s+([+-]?(?:\d+\.\d*|\.\d+|\d+)f?)",
            text,
        )
        if match is None:
            raise RuntimeError(f"constants.h has no {name}")
        return match.group(1).rstrip("f")

    sizes = re.search(r"#define\s+ACT_SIZES\s+\{(\d+)\s*,\s*(\d+)\}", text)
    if sizes is None:
        raise RuntimeError("constants.h has no ACT_SIZES")
    n_accel = int(macro("NUM_ACCEL_BINS"))
    n_steer = int(macro("NUM_STEER_BINS"))
    if int(sizes.group(1)) != n_accel or int(sizes.group(2)) != n_steer:
        raise RuntimeError("ACT_SIZES does not match NUM_ACCEL_BINS and NUM_STEER_BINS")
    if n_accel % 2 == 0 or n_steer % 2 == 0:
        raise RuntimeError(
            "NUM_ACCEL_BINS and NUM_STEER_BINS must be odd so the center bin is 0"
        )
    return (
        float(macro("ACCEL_MIN")), float(macro("ACCEL_MAX")), n_accel,
        float(macro("STEER_MIN")), float(macro("STEER_MAX")), n_steer,
    )

def _same_grid(grid, header):
    for got, exp in zip(grid, header):
        if abs(float(got) - float(exp)) > 1e-5:
            return False
    return True

def _require_compiled_grid(lib, grid):
    amin, amax, na, smin, smax, ns = grid
    na = int(na)
    ns = int(ns)
    got_a = int(lib.pf_n_accel())
    got_s = int(lib.pf_n_steer())
    if got_a != na or got_s != ns:
        raise SystemExit(
            f"libdrive_offline.so is {got_a}x{got_s}, constants.h is {na}x{ns}"
        )
    if int(lib.pf_coast_accel()) != na // 2 or int(lib.pf_coast_steer()) != ns // 2:
        raise SystemExit("compiled coast bin is not the center")
    if abs(float(lib.pf_accel_value(na // 2))) > 1e-6:
        raise SystemExit("compiled accel center is not 0")
    if abs(float(lib.pf_steer_value(ns // 2))) > 1e-6:
        raise SystemExit("compiled steer center is not 0")
    if abs(float(lib.pf_accel_value(0)) - float(amin)) > 1e-5:
        raise SystemExit("compiled accel min does not match constants.h")
    if abs(float(lib.pf_accel_value(na - 1)) - float(amax)) > 1e-5:
        raise SystemExit("compiled accel max does not match constants.h")
    if abs(float(lib.pf_steer_value(0)) - float(smin)) > 1e-5:
        raise SystemExit("compiled steer min does not match constants.h")
    if abs(float(lib.pf_steer_value(ns - 1)) - float(smax)) > 1e-5:
        raise SystemExit("compiled steer max does not match constants.h")
    if int(lib.pf_accel_bin(0.0)) != na // 2 or int(lib.pf_steer_bin(0.0)) != ns // 2:
        raise SystemExit("quantize(0) is not the center bin")

def _require_policy_bins(lib, pol):
    na = int(pol.head_a.out_features)
    ns = int(pol.head_s.out_features)
    got_a = int(lib.pf_n_accel())
    got_s = int(lib.pf_n_steer())
    if got_a != na or got_s != ns:
        raise SystemExit(
            f"policy heads are {na}x{ns} and libdrive_offline.so is {got_a}x{got_s}"
        )

def _backup_foreign_ckpt(path, n_accel, n_steer):
    if not path.is_file():
        return
    with np.load(path) as stored:
        old_accel = int(stored["n_accel"]) if "n_accel" in stored.files else -1
        old_steer = int(stored["n_steer"]) if "n_steer" in stored.files else -1
    if old_accel == int(n_accel) and old_steer == int(n_steer):
        return
    dest = path.with_name(f"{path.stem}_a{old_accel}s{old_steer}_partial.npz")
    if dest.exists():
        print(f"kept {dest}", flush=True)
        return
    shutil.copy2(path, dest)
    print(f"copied {path.name} to {dest.name}", flush=True)

def train_and_eval_maps(n_maps=N_MAPS, accel_min=None, accel_max=None,
        n_accel=None, steer_min=None, steer_max=None, n_steer=None,
        resume=False, enter_hard=False, stop_within10=STOP_WITHIN10):
    paths = _run_paths(n_maps)
    header = rl_action_space()
    if n_accel is None and accel_min is None:
        grid = header
    else:
        if None in (accel_min, accel_max, steer_min, steer_max, n_accel, n_steer):
            raise ValueError("accel and steer bounds and bin counts are set together")
        grid = (
            float(accel_min), float(accel_max), int(n_accel),
            float(steer_min), float(steer_max), int(n_steer),
        )
        if not _same_grid(grid, header):
            raise SystemExit(
                "BC action grid "
                f"accel [{grid[0]:g}, {grid[1]:g}] x {grid[2]}  "
                f"steer [{grid[3]:g}, {grid[4]:g}] x {grid[5]} "
                "does not match ocean/drive/constants.h "
                f"accel [{header[0]:g}, {header[1]:g}] x {header[2]}  "
                f"steer [{header[3]:g}, {header[4]:g}] x {header[5]}. "
                "RL steps that table in move_dynamics."
            )
    accel_min, accel_max, n_accel, steer_min, steer_max, n_steer = grid
    n_accel = int(n_accel)
    n_steer = int(n_steer)
    paths["dataset"] = OUT / f"bc_{int(n_maps)}_a{n_accel}s{n_steer}_teleport.npz"
    paths["metric"] = (
        drv.ROOT / "logs" / "drive" / f"bc_{int(n_maps)}_a{n_accel}s{n_steer}.jsonl"
    )
    _backup_foreign_ckpt(paths["ckpt"], n_accel, n_steer)
    print(
        f"label grid accel [{grid[0]:g}, {grid[1]:g}] x {n_accel}  "
        f"steer [{grid[3]:g}, {grid[4]:g}] x {n_steer}",
        flush=True,
    )
    if stop_within10 is not None:
        print(
            f"stop and store the checkpoint at within-10 mean {float(stop_within10):.2f}",
            flush=True,
        )
    map_ids = list(range(n_maps))
    lib = drv.load()
    _require_compiled_grid(lib, grid)
    if paths["dataset"].is_file():
        stored = np.load(paths["dataset"])
        inverse = ""
        if "inverse" in stored.files:
            inverse = stored["inverse"].item()
            if isinstance(inverse, bytes):
                inverse = inverse.decode()
        same = (
            "n_accel" in stored.files and int(stored["n_accel"]) == n_accel
            and int(stored["n_steer"]) == n_steer
            and abs(float(stored["accel_min"]) - grid[0]) < 1e-6
            and abs(float(stored["accel_max"]) - grid[1]) < 1e-6
            and abs(float(stored["steer_min"]) - grid[3]) < 1e-6
            and abs(float(stored["steer_max"]) - grid[4]) < 1e-6
            and str(inverse) == "kinematic"
        )
        if not same:
            raise SystemExit(
                f"{paths['dataset']} is not the kinematic {n_accel}x{n_steer} grid"
            )
        x = stored["x"]
        ya = stored["ya"]
        ys = stored["ys"]
        mu = np.asarray(stored["mu"], dtype=np.float32)
        sig = np.asarray(stored["sig"], dtype=np.float32)
        print(f"loaded {paths['dataset']} samples {len(x)}", flush=True)
    else:
        x, ya, ys = collect_pairs(lib, map_ids, teleport=1, grid=grid)
        mu = x.mean(axis=0)
        sig = x.std(axis=0)
        sig = np.where(sig < 1e-6, 1.0, sig).astype(np.float32)
        mu = np.asarray(mu, dtype=np.float32)
        np.savez(
            paths["dataset"],
            x=x, ya=ya, ys=ys, mu=mu, sig=sig,
            n_maps=np.int32(n_maps),
            n_accel=np.int32(n_accel),
            n_steer=np.int32(n_steer),
            accel_min=np.float32(grid[0]),
            accel_max=np.float32(grid[1]),
            steer_min=np.float32(grid[3]),
            steer_max=np.float32(grid[4]),
            inverse=np.array("kinematic"),
        )
        print(f"wrote {paths['dataset']} samples {len(x)}", flush=True)
    x = ((x - mu) / sig).astype(np.float32)
    resume_epoch = 0
    resume_step = 0
    resume_lr = None
    resume_loss = None
    if resume:
        last = json.loads(Path(paths["metric"]).read_text().splitlines()[-1])
        resume_epoch = int(last["epoch"])
        resume_step = int(last["step"])
        resume_lr = float(last["lr"])
        resume_loss = float(last["loss"])
    pol, acc, acc_a, acc_s, epoch, hist = train(
        x, ya, ys, n_accel, n_steer,
        ckpt_path=paths["ckpt"], mu=mu, sig=sig,
        metric_path=paths.get("metric"),
        accel_min=grid[0],
        accel_max=grid[1],
        steer_min=grid[3],
        steer_max=grid[4],
        resume_ckpt=paths["ckpt"] if resume else None,
        start_epoch=resume_epoch,
        start_step=resume_step,
        start_lr=resume_lr,
        resume_loss=resume_loss,
        metric_append=bool(resume),
        enter_hard=bool(enter_hard),
        stop_within10=stop_within10,
    )
    plot_training(hist, paths["train_plot"], title=f"BC training on {n_maps} maps")
    pol.mu = mu.astype(np.float32)
    pol.sig = sig
    save_policy(pol, acc, n_accel, n_steer, paths["ckpt"])
    print(f"checkpoint accuracy joint {acc:.4f} accel {acc_a:.4f} steer {acc_s:.4f} "
        f"epochs {epoch}", flush=True)
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
    scored = [row["score_rate"] for row in used if "score_rate" in row]
    out = {
        "n": len(used),
        "comp_mean": float(comp.mean()), "comp_std": float(comp.std()),
        "fail_mean": float(fail.mean()), "fail_std": float(fail.std()),
        "disp_mean": float(disp.mean()), "disp_std": float(disp.std()),
    }
    if scored:
        score = np.asarray(scored, dtype=np.float64)
        out["score_mean"] = float(score.mean())
        out["score_std"] = float(score.std())
    return out

def plot_train_test(train, test, path, n_accel=257, n_steer=257, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    splits = [train] if test is None else [train, test]
    colors = sns.color_palette("muted", n_colors=max(2, len(splits)))[:len(splits)]
    width = 0.5 if len(splits) == 1 else 0.34
    shifts = [0.0] if len(splits) == 1 else [-width / 2, width / 2]
    fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.3), constrained_layout=True)
    ax_d, ax_h, ax_b, ax_c = axes
    ax_h.sharex(ax_d)
    dot = {"s": 8, "alpha": 0.2, "linewidths": 0, "rasterized": True}
    for split, color in zip(splits, colors):
        ax_d.scatter(
            split["steps_d"], split["err"], color=color, label=split["name"], **dot,
        )
    ax_d.set_xlabel("timestep")
    ax_d.set_ylabel("displacement (m)")
    ax_d.set_title("displacement")
    _solid_legend(ax_d, loc="upper left", fontsize=8, frameon=False, markerscale=2)
    style_ax(ax_d)
    for split, color in zip(splits, colors):
        ax_h.scatter(
            split["steps_h"], split["ent"], color=color, label=split["name"], **dot,
        )
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
    rate_x = np.asarray([0.0, 1.0])
    for shift, split, color in zip(shifts, splits, colors):
        ax_b.bar(
            rate_x + shift,
            [split["comp_mean"], split["fail_mean"]],
            yerr=[split["comp_std"], split["fail_std"]],
            color=color, width=width, capsize=3,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            label=split["name"],
        )
    ax_m = ax_b.twinx()
    for shift, split, color in zip(shifts, splits, colors):
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
        for split in splits
        for key, std in (("comp_mean", "comp_std"), ("fail_mean", "fail_std"))
    )
    disp_top = max(split["disp_mean"] + split["disp_std"] for split in splits)
    ax_b.set_ylim(0.0, max(1.25, rate_top + 0.32))
    ax_m.set_ylim(0.0, max(disp_top, 1e-6) * 1.45)
    ax_b.set_axisbelow(True)
    ax_b.grid(True, alpha=0.3, axis="y")
    sns.despine(ax=ax_b, top=True, right=True)
    sns.despine(ax=ax_m, top=True, left=True, right=False)
    ax_b.legend(loc="upper right", fontsize=8, frameon=False)
    for shift, split in zip(shifts, splits):
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
    for split, color in zip(splits, colors):
        sns.regplot(
            x=split["pair_d"], y=split["pair_e"], ax=ax_c, ci=95,
            scatter_kws={"s": 8, "alpha": 0.2, "linewidths": 0, "rasterized": True,
                "color": color},
            line_kws={"color": color, "lw": 1.6},
        )
    y_text = 0.96
    for split, color in zip(splits, colors):
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
        steps_d, err, steps_h, ent, per_map, pair_d, pair_e = evaluate_closed_loop(
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

def eval_saved(n_maps=1000, rollouts=1):
    # Every Waymo map is in the training set, so this is train-only.
    # Score is the logged end within 5 m and no collision or off-road.
    paths = _run_paths(n_maps)
    lib = drv.load()
    pol, acc = load_policy(paths["ckpt"])
    z = np.load(paths["ckpt"])
    n_accel, n_steer = int(z["n_accel"]), int(z["n_steer"])
    map_ids = list(range(n_maps))
    steps_d, err, steps_h, ent, per_map, pair_d, pair_e = evaluate_closed_loop(
        lib, pol, map_ids, rollouts=rollouts,
    )
    rates = _map_rates(per_map)
    pack = {
        "name": "train",
        "steps_d": steps_d, "err": err, "steps_h": steps_h, "ent": ent,
        "pair_d": pair_d, "pair_e": pair_e,
        "per_map": per_map, **rates,
    }
    out_plot = OUT / f"bc_{n_maps}_eval.png"
    out_json = OUT / f"bc_{n_maps}_eval.json"
    plot_train_test(
        pack, None, out_plot, n_accel=n_accel, n_steer=n_steer,
        title=(
            f"Train maps 0–{n_maps - 1}. "
            f"score {pack['score_mean']:.2f} ± {pack['score_std']:.2f}. "
            f"Completion is the logged end within {GOAL_RADIUS_M:.0f} m "
            f"by step {HORIZON_STEPS}"
        ),
    )
    summary = {
        "checkpoint": str(paths["ckpt"]),
        "train_joint_accuracy": acc,
        "maps": [map_ids[0], map_ids[-1]],
        "rollouts_per_map": int(rollouts),
        "greedy": True,
        "score_mean": pack["score_mean"],
        "score_std": pack["score_std"],
        "completion_mean": pack["comp_mean"],
        "completion_std": pack["comp_std"],
        "failure_mean": pack["fail_mean"],
        "failure_std": pack["fail_std"],
        "displacement_mean": pack["disp_mean"],
        "displacement_std": pack["disp_std"],
        "n_maps_scored": pack["n"],
        "score_definition": (
            "reached the logged end within 5 m by step 100 "
            "with no collision or off-road"
        ),
    }
    out_json.write_text(json.dumps(summary, indent=2))
    print(
        f"score {pack['score_mean']:.3f} ± {pack['score_std']:.3f}  "
        f"completion {pack['comp_mean']:.3f} ± {pack['comp_std']:.3f}  "
        f"displacement {pack['disp_mean']:.3f} ± {pack['disp_std']:.3f}",
        flush=True,
    )
    print(f"wrote {out_json}", flush=True)
    print("DONE", flush=True)
    return summary

CARLA_DIR = drv.ROOT / "data" / "mixed"

def plot_action_compare(expert_a, expert_s, anchor_a, anchor_s, path, grid):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    amin, amax, na, smin, smax, ns = grid
    na = int(na)
    ns = int(ns)
    apply_figure_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4), constrained_layout=True)
    colors = sns.color_palette(n_colors=2)
    panels = (
        (axes[0], expert_a, anchor_a, amin, amax, na, "acceleration (m/s²)"),
        (axes[1], expert_s, anchor_s, smin, smax, ns, "steering (rad)"),
    )
    for ax, expert, anchor, lo, hi, n_bins, xlabel in panels:
        centers = bin_centers(lo, hi, n_bins)
        step = float(centers[1] - centers[0])
        edges = np.linspace(float(lo) - step / 2.0, float(hi) + step / 2.0, n_bins + 1)
        expert_v = centers[np.asarray(expert, dtype=np.int64)]
        anchor_v = centers[np.asarray(anchor, dtype=np.int64)]
        ax.hist(
            expert_v, bins=edges, density=True, color=colors[0], alpha=0.55,
            label=f"expert  {len(expert_v):,}", zorder=2,
        )
        ax.hist(
            anchor_v, bins=edges, density=True, histtype="step",
            color=colors[1], lw=1.3,
            label=f"anchor  {len(anchor_v):,}", zorder=3,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
        ax.set_xlim(float(lo) - 0.04 * (float(hi) - float(lo)),
            float(hi) + 0.04 * (float(hi) - float(lo)))
        style_ax(ax)
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle(
        "Train labels versus closed-loop anchor actions, "
        f"{int(na)}×{int(ns)} bins"
    )
    fig.savefig(path, dpi=100)
    plt.close(fig)
    print(f"wrote {path}")

def rollout_predicted(lib, pol, map_ids, steps=None):
    if steps is None:
        steps = HORIZON_STEPS
    _require_policy_bins(lib, pol)
    accel = []
    steer = []
    maps = []
    agents = []
    times = []
    for map_id in map_ids:
        h = drv.create_scenario(lib, DATA, int(map_id), JSONS[int(map_id)])
        try:
            controlled = controllable_agents(lib, h)
            lib.pf_reset(h, 0)
            _park_unfit(lib, h)
            before = len(accel)
            for t in range(int(steps)):
                lib.pf_set_timestep(h, t)
                lib.pf_spawn_due(h)
                lib.pf_recompute(h)
                obs = drv.observations(lib, h)
                live = []
                for agent in controlled:
                    state = drv.agent_state(lib, h, agent)
                    if _on_map(state):
                        live.append(agent)
                if live:
                    batch = (obs[np.asarray(live)] - pol.mu) / pol.sig
                    pred_a, pred_s, _ent = pol.act_batch(batch)
                    for agent, a_bin, s_bin in zip(live, pred_a, pred_s):
                        lib.pf_set_action(h, agent, float(a_bin), float(s_bin))
                        accel.append(int(a_bin))
                        steer.append(int(s_bin))
                        maps.append(int(map_id))
                        agents.append(int(agent))
                        times.append(int(t))
                lib.pf_integrate(h, 1)
            print(
                f"map {map_id} rollout actions {len(accel) - before}",
                flush=True,
            )
        finally:
            lib.pf_close(h)
    if not accel:
        raise RuntimeError("closed-loop rollout recorded no actions")
    return (
        np.asarray(accel, dtype=np.int16),
        np.asarray(steer, dtype=np.int16),
        np.asarray(maps, dtype=np.int16),
        np.asarray(agents, dtype=np.int16),
        np.asarray(times, dtype=np.int16),
    )

def _hist_from_jsonl(path):
    hist = empty_history()
    skip = {"epoch", "phase", "step", "loss", "lr"}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("phase") == "resume":
            continue
        stats = {key: row[key] for key in row if key not in skip}
        record_stats(hist, row["epoch"], row["phase"], stats, row["loss"], row["step"])
    return hist


def train_brief(n_maps=10, epochs=40, out_dir=None, resume=False, stop_within10=None):
    """BC run on the shared action table. Writes only under out_dir."""
    out = Path(out_dir) if out_dir is not None else (
        drv.ROOT / "analysis" / "offline" / "keep"
    )
    out.mkdir(parents=True, exist_ok=True)
    grid = rl_action_space()
    amin, amax, n_accel, smin, smax, n_steer = grid
    n_accel = int(n_accel)
    n_steer = int(n_steer)
    lib = drv.load()
    _require_compiled_grid(lib, grid)
    map_ids = list(range(int(n_maps)))
    dataset = out / f"bc_{int(n_maps)}_dataset.npz"
    if dataset.is_file():
        stored = np.load(dataset)
        inverse = stored["inverse"].item() if "inverse" in stored.files else ""
        if isinstance(inverse, bytes):
            inverse = inverse.decode()
        same = (
            int(stored["n_accel"]) == n_accel
            and int(stored["n_steer"]) == n_steer
            and abs(float(stored["accel_min"]) - amin) < 1e-6
            and abs(float(stored["accel_max"]) - amax) < 1e-6
            and abs(float(stored["steer_min"]) - smin) < 1e-6
            and abs(float(stored["steer_max"]) - smax) < 1e-6
            and str(inverse) == "kinematic"
            and int(stored["n_maps"]) == int(n_maps)
        )
        if not same:
            raise SystemExit(f"{dataset} is not the current {n_maps}-map grid")
        x = stored["x"]
        ya = stored["ya"]
        ys = stored["ys"]
        mu = np.asarray(stored["mu"], dtype=np.float32)
        sig = np.asarray(stored["sig"], dtype=np.float32)
        print(f"loaded {dataset} samples {len(x)}", flush=True)
    else:
        x, ya, ys = collect_pairs(lib, map_ids, teleport=1, grid=grid)
        mu = x.mean(axis=0).astype(np.float32)
        sig = x.std(axis=0).astype(np.float32)
        sig = np.where(sig < 1e-6, 1.0, sig).astype(np.float32)
        np.savez(
            dataset,
            x=x, ya=ya, ys=ys, mu=mu, sig=sig,
            n_maps=np.int32(n_maps),
            n_accel=np.int32(n_accel),
            n_steer=np.int32(n_steer),
            accel_min=np.float32(amin),
            accel_max=np.float32(amax),
            steer_min=np.float32(smin),
            steer_max=np.float32(smax),
            inverse=np.array("kinematic"),
        )
        print(f"wrote {dataset} samples {len(x)}", flush=True)
    x_n = ((x - mu) / sig).astype(np.float32)
    ckpt = out / f"bc_{int(n_maps)}.npz"
    metric = out / f"bc_{int(n_maps)}.jsonl"
    start_epoch = 0
    start_step = 0
    start_lr = None
    resume_loss = None
    resume_ckpt = None
    if resume:
        lines = [ln for ln in metric.read_text().splitlines() if ln.strip()]
        last = json.loads(lines[-1])
        start_epoch = int(last["epoch"])
        start_step = int(last["step"])
        start_lr = float(last["lr"])
        resume_loss = float(last["loss"])
        resume_ckpt = ckpt
    pol, acc, acc_a, acc_s, epoch, _hist = train(
        x_n, ya, ys, n_accel, n_steer,
        ckpt_path=ckpt, mu=mu, sig=sig,
        metric_path=metric,
        accel_min=amin, accel_max=amax,
        steer_min=smin, steer_max=smax,
        max_epochs=None if epochs is None else int(epochs),
        resume_ckpt=resume_ckpt,
        start_epoch=start_epoch,
        start_step=start_step,
        start_lr=start_lr,
        resume_loss=resume_loss,
        metric_append=bool(resume),
        stop_within10=stop_within10,
    )
    hist = _hist_from_jsonl(metric)
    plot_training(
        hist, out / f"bc_{int(n_maps)}_train.png",
        title=f"BC on {int(n_maps)} scenes",
    )
    pol.mu = mu.astype(np.float32)
    pol.sig = sig
    save_policy(pol, acc, n_accel, n_steer, ckpt)
    pred_a, pred_s, pred_map, pred_agent, pred_t = rollout_predicted(lib, pol, map_ids)
    actions_path = out / f"bc_{int(n_maps)}_rollout_actions.npz"
    np.savez(
        actions_path,
        accel=pred_a, steer=pred_s, map_id=pred_map, agent=pred_agent, t=pred_t,
        n_accel=np.int32(n_accel), n_steer=np.int32(n_steer),
        accel_min=np.float32(amin), accel_max=np.float32(amax),
        steer_min=np.float32(smin), steer_max=np.float32(smax),
    )
    print(f"wrote {actions_path} actions {len(pred_a)}", flush=True)
    plot_action_compare(
        ya, ys, pred_a, pred_s,
        out / f"bc_{int(n_maps)}_action_dist.png",
        grid,
    )
    plot_action_histogram(
        pred_a, pred_s,
        out / f"bc_{int(n_maps)}_anchor_hist.png",
        len(pred_a),
        accel_min=amin, accel_max=amax, n_accel=n_accel,
        steer_min=smin, steer_max=smax, n_steer=n_steer,
        title=(
            f"Closed-loop anchor, {len(pred_a):,} actions    "
            f"accel [{amin:g}, {amax:g}] × {n_accel}    "
            f"steer [{smin:g}, {smax:g}] × {n_steer}"
        ),
    )
    last = json.loads([ln for ln in metric.read_text().splitlines() if ln.strip()][-1])
    print(
        f"brief joint {acc:.4f} accel {acc_a:.4f} steer {acc_s:.4f} "
        f"within10 {float(last['acc10']):.4f} "
        f"epochs {epoch} samples {len(ya)}",
        flush=True,
    )
    return pol, hist

def main():
    lib = drv.load()
    grid = rl_action_space()
    _require_compiled_grid(lib, grid)
    h = drv.create_scenario(lib, DATA, MAP_ID, JSONS[MAP_ID])
    try:
        lib.pf_relabel_grid(
            h, grid[0], grid[1], grid[2], grid[3], grid[4], grid[5],
        )
        obs, acts, valid, _ade = drv.collect(lib, h, 1)
        n_accel = int(grid[2])
        n_steer = int(grid[5])
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
        pol, acc, acc_a, acc_s, _epoch, _hist = train(
            x, ya, ys, n_accel, n_steer,
            accel_min=grid[0], accel_max=grid[1],
            steer_min=grid[3], steer_max=grid[4],
        )
        pol.mu = mu.astype(np.float32)
        pol.sig = sig
        print(f"checkpoint accuracy joint {acc:.4f} accel {acc_a:.4f} steer {acc_s:.4f}")
        if acc < 0.99:
            raise SystemExit(f"accuracy {acc:.4f} stayed below 0.99")
        save_policy(pol, acc, n_accel, n_steer)
        print(f"wrote {CKPT}")
    finally:
        lib.pf_close(h)

if __name__ == "__main__":
    main()
