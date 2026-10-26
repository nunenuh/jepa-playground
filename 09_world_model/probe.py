"""What does a JEPA world model's latent know? Velocity check + impossible rollouts.

A. Velocity check (Velocity Blind Spot): probes (linear and small MLP) from
   LeWM latents to positions (agent, block, block angle) and velocities
   (agent, block), from one frame's latent vs two latents one action block
   (0.5 s) apart. Probe data: 10k random simulator states, each rendered twice,
   moved by velocities drawn from the expert data (the 21 validation episodes
   are too few: probes fit on 15 of them don't transfer to the other 6).
B. Impossible rollouts: the predictor sees 3 real frames + actions and predicts
   the next latent; surprise = squared error to what actually came next. Real
   next frames vs impossible ones: block teleported (re-rendered in the
   simulator), frozen (no change), reversed (the previous frame).

Inference only, runs on the laptop GPU (Docker image from this folder):
    docker run --rm --gpus all -v jepa-wm-data:/data -v "$PWD/09_world_model:/work" \
        -v "$PWD/09_world_model/assets:/out" --entrypoint python jepa-wm /work/probe.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import h5py
import hdf5plugin  # noqa: F401  (pixels are blosc-compressed)
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MODELS = ["quentinll/lewm-pusht", "ours_epoch3", "ours_epoch1", "random_init"]
FS, HIST = 5, 3                                   # frames per action block, predictor history
N_PROBE, N_PROBE_TEST, DT = 10_000, 2_000, FS / 10     # synthetic probe states; 10 Hz control
TELEPORT_PX, TELEPORT_DEG = 40, 30
DATA_REPO, DATA_FILE = "jepa-world-models/h-jepa", "pusht_expert_val.h5"
MEAN = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)


def load_data():
    home = Path(os.environ.get("STABLEWM_HOME", "~/.stable-wm")).expanduser()
    path = hf_hub_download(DATA_REPO, DATA_FILE, repo_type="dataset", local_dir=home / "datasets")
    with h5py.File(path) as f:
        return {k: f[k][:] for k in ["pixels", "state", "action", "episode_idx", "step_idx"]}


def load_model(name):
    if name == "random_init":                     # same architecture, untrained: the probe's floor
        model = swm.wm.utils.load_pretrained("quentinll/lewm-pusht")
        torch.manual_seed(0)
        for m in model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
    else:
        model = swm.wm.utils.load_pretrained(name)
    return model.to("cuda").eval().requires_grad_(False)


@torch.no_grad()
def encode(model, pixels, bs=256):
    out = []
    for i in range(0, len(pixels), bs):
        x = torch.from_numpy(pixels[i:i + bs]).permute(0, 3, 1, 2).float() / 255
        x = ((x - MEAN) / STD).cuda()
        out.append(model.encode({"pixels": x[:, None]})["emb"][:, 0].cpu())
    return torch.cat(out).numpy()


def action_blocks(d):
    """Normalised action block leaving each frame (FS actions concatenated), NaN if it runs past the episode."""
    a = d["action"]
    a = (a - np.nanmean(a, 0)) / np.nanstd(a, 0)
    blocks = np.full((len(a), FS * 2), np.nan, dtype=np.float32)
    ep = d["episode_idx"]
    for i in range(len(a) - FS + 1):
        if ep[i] == ep[i + FS - 1]:
            blocks[i] = a[i:i + FS].reshape(-1)
    return blocks


def probe_states(d, rng):
    """Random positions/angles; agent and block velocities resampled from the expert episodes."""
    s, ep = d["state"], d["episode_idx"]
    same = np.r_[False, ep[1:] == ep[:-1]]
    vb = ((s[1:, 2:4] - s[:-1, 2:4]) * 10)[same[1:]]                          # px/s at 10 Hz
    n = N_PROBE
    st = np.c_[rng.uniform(60, 450, (n, 2)), rng.uniform(110, 400, (n, 2)), rng.uniform(0, 2 * np.pi, n)]
    va, vb = s[rng.integers(len(s), size=n), 5:7], vb[rng.integers(len(vb), size=n)]
    st2 = st.copy()
    st2[:, 0:2] = np.clip(st[:, 0:2] + va * DT, 15, 495)
    st2[:, 2:4] = st[:, 2:4] + vb * DT
    va = (st2[:, 0:2] - st[:, 0:2]) / DT                                      # after clipping at the walls
    ys = {"agent position": st[:, 0:2], "block position": st[:, 2:4],
          "block angle": np.c_[np.sin(st[:, 4]), np.cos(st[:, 4])], "agent velocity": va, "block velocity": vb}
    return st, st2, ys


def render(states):
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    env.reset(seed=0)
    frames = []
    for s in states:
        env._set_state(np.asarray(s[:5], dtype=np.float64))   # 5 values: no velocity, so nothing drifts
        frames.append(env.render())
    return np.stack(frames)


def probe(feats, y):
    tr, te = slice(N_PROBE_TEST, None), slice(0, N_PROBE_TEST)
    sx, sy = StandardScaler().fit(feats[tr]), StandardScaler().fit(y[tr])
    x_tr, x_te, y_tr = sx.transform(feats[tr]), sx.transform(feats[te]), sy.transform(y[tr])
    lin = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(x_tr, y_tr)
    mlp = MLPRegressor(hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0).fit(x_tr, y_tr)
    return {kind: float(r2_score(y[te], sy.inverse_transform(m.predict(x_te)))) for kind, m in
            [("linear", lin), ("mlp", mlp)]}


def clips(d):
    """Indices t with frames t-2FS, t-FS, t and t+FS inside one episode, every FS steps."""
    ep, step = d["episode_idx"], d["step_idx"]
    n = len(ep)
    idx = [t for t in range(2 * FS, n - FS, FS)
           if ep[t - 2 * FS] == ep[t + FS] and step[t] - step[t - 2 * FS] == 2 * FS]
    return np.array(idx)


def render_frames(d, ts):
    """Re-render the real next state and a block-teleported copy, so both go through the same renderer."""
    env = gym.make("swm/PushT-v1", resolution=224).unwrapped
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    real, tele = [], []
    for t in ts:
        s = d["state"][t + FS].astype(np.float64).copy()
        env._set_state(s)
        real.append(env.render())
        ang = rng.uniform(0, 2 * np.pi)
        s[2:4] = np.clip(s[2:4] + TELEPORT_PX * np.array([np.cos(ang), np.sin(ang)]), 60, 450)
        s[4] += np.deg2rad(TELEPORT_DEG) * rng.choice([-1, 1])
        env._set_state(s)
        tele.append(env.render())
    return np.stack(real), np.stack(tele)


@torch.no_grad()
def surprise(model, emb, blocks, ts, next_emb):
    hist = np.stack([emb[ts - 2 * FS], emb[ts - FS], emb[ts]], 1)
    act = np.stack([blocks[ts - 2 * FS], blocks[ts - FS], blocks[ts]], 1)
    act_emb = model.action_encoder(torch.from_numpy(act).cuda())
    pred = model.predict(torch.from_numpy(hist).cuda(), act_emb)[:, -1].cpu().numpy()
    return {k: ((pred - v) ** 2).mean(1) for k, v in next_emb.items()}


def main():
    out = Path(os.environ.get("OUT", "/out"))
    d = load_data()
    blocks = action_blocks(d)
    st1, st2, ys = probe_states(d, np.random.default_rng(0))
    probe_px1, probe_px2 = render(st1), render(st2)
    ts = clips(d)
    ts = ts[~np.isnan(blocks[ts - 2 * FS]).any(1) & ~np.isnan(blocks[ts]).any(1)]
    rendered_real, rendered_tele = render_frames(d, ts)
    render_gap = float(np.abs(rendered_real.astype(float) - d["pixels"][ts + FS]).mean())
    speed = np.linalg.norm(d["state"][ts, 5:7], axis=1)
    print(f"{len(ts)} clips, renderer vs dataset pixels: mean abs diff {render_gap:.1f}/255", flush=True)

    results = {"clips": len(ts), "probe_states": N_PROBE, "probe_test": N_PROBE_TEST, "render_gap": render_gap,
               "teleport": {"px": TELEPORT_PX, "deg": TELEPORT_DEG}, "models": {}}
    for name in MODELS:
        model = load_model(name)
        emb = encode(model, d["pixels"])
        z1, z2 = encode(model, probe_px1), encode(model, probe_px2)
        r = {"probe_r2": {}, "surprise_auc": {}}
        for k, y in ys.items():
            r["probe_r2"][k] = {"one frame": probe(z1, y), "two frames": probe(np.c_[z1, z2], y)}

        nxt = {"real": emb[ts + FS], "frozen": emb[ts], "reversed": emb[ts - FS],
               "real (re-rendered)": encode(model, rendered_real), "teleported": encode(model, rendered_tele)}
        s = surprise(model, emb, blocks, ts, nxt)
        auc = lambda a, b, m=slice(None): float(roc_auc_score(np.r_[np.zeros(len(a[m])), np.ones(len(b[m]))],
                                                              np.r_[a[m], b[m]]))
        slow, fast = speed < np.median(speed), speed >= np.median(speed)
        r["surprise_auc"] = {"frozen": auc(s["real"], s["frozen"]),
                             "frozen, slow agent": auc(s["real"], s["frozen"], slow),
                             "frozen, fast agent": auc(s["real"], s["frozen"], fast),
                             "reversed": auc(s["real"], s["reversed"]),
                             "teleported block": auc(s["real (re-rendered)"], s["teleported"])}
        r["mean_surprise"] = {k: float(v.mean()) for k, v in s.items()}
        results["models"][name] = r
        print(name, json.dumps(r, indent=1), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    (out / "results_probe.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out / "probe_examples.npz", real=rendered_real[:8], tele=rendered_tele[:8],
                        probe1=probe_px1[:8], probe2=probe_px2[:8])
    st = os.stat(out)
    for p in [out / "results_probe.json", out / "probe_examples.npz"]:
        os.chown(p, st.st_uid, st.st_gid)


if __name__ == "__main__":
    main()
