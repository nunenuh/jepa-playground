# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 06 · V-JEPA 2 on real video: zero-shot anomaly detection
#
# Experiment 03 trained a tiny world model on a bouncing ball and used its
# *surprise* to catch impossible motion. Here we do the same with a real,
# pretrained world model: Meta's **V-JEPA 2** (ViT-L, 300M parameters,
# trained on over a million hours of internet video), on real CCTV footage.
#
# - Give V-JEPA 2 the **past 12 frames**, let its predictor guess the latents
#   of the **next 4 frames**, compare with the real ones.
# - Surprise per image patch = a heatmap of *where* something unexpected is.
# - **Zero-shot**: V-JEPA 2 has never seen this camera. No training at all.
#
# **Data: UCSD Ped2.** A fixed camera over a pedestrian walkway. Training
# videos contain only people walking; test videos also contain bikes,
# skateboards, a cart and a wheelchair. Frame-level labels say when.
#
# ![architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/06_vjepa2_video/assets/architecture-light.svg)

# %%
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
SIZE, PATCH, TUBELET = 256, 16, 2
T_CTX, T_PRED = 12, 4                  # 12 past frames -> predict the next 4
T = T_CTX + T_PRED
GRID = SIZE // PATCH                   # 16 x 16 patches per tubelet
STRIDE = 2                             # slide the window 2 frames at a time

# Ped2 mirror on the Hugging Face Hub, pinned. Original: UCSD Anomaly Detection
# Dataset (http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm).
PED2_URL = "https://huggingface.co/datasets/123zdm/ped2_datasets/resolve/19dcbe727b56f7dbf53a544e0b5a2ccbaaae14b6/ped2.tar.gz"
DATA = Path("data")
if not (DATA / "ped2").exists():
    DATA.mkdir(exist_ok=True)
    urllib.request.urlretrieve(PED2_URL, DATA / "ped2.tar.gz")
    with tarfile.open(DATA / "ped2.tar.gz") as tar:
        tar.extractall(DATA, filter="data")


def load_videos(split):
    root = DATA / "ped2" / split / "frames"
    return {d.name: np.stack([np.array(Image.open(f).convert("RGB")) for f in sorted(d.glob("*.jpg"))])
            for d in sorted(root.iterdir())}


train_videos, test_videos = load_videos("training"), load_videos("testing")
gt = sio.loadmat(DATA / "ped2" / "ped2.mat")["gt"][0]
labels = {}
for (name, frames), (start, end) in zip(test_videos.items(), [g.ravel() for g in gt]):
    y = np.zeros(len(frames), dtype=bool)
    y[start - 1:end] = True                                  # 1-based, inclusive
    labels[name] = y
print(f"{len(train_videos)} training videos, {len(test_videos)} test videos, "
      f"{sum(len(v) for v in test_videos.values())} test frames, "
      f"{sum(y.sum() for y in labels.values())} anomalous")

# %% [markdown]
# ## V-JEPA 2
#
# Frames are resized to 256 x 256 (the whole frame, no crop: anomalies can be
# at the edges) and normalised like ImageNet. The encoder turns every
# 2-frame x 16 x 16-pixel tubelet into a 1024-D latent.

# %%
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)


def to_pixels(clip):
    """(T, H, W, 3) uint8 -> (1, T, 3, 256, 256) normalised half tensor."""
    x = torch.from_numpy(clip).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    return ((x.unsqueeze(0) - MEAN) / STD).half()


N_CTX = (T_CTX // TUBELET) * GRID * GRID
N_ALL = (T // TUBELET) * GRID * GRID


@torch.no_grad()
def surprise_map(clip):
    """Predict the last T_PRED frames from the first T_CTX. Returns (map 16x16, target latents)."""
    px = to_pixels(clip)
    ctx = model(pixel_values_videos=px[:, :T_CTX], skip_predictor=True).last_hidden_state       # past only
    target = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state[:, N_CTX:]     # real future
    pred = model.predictor(encoder_hidden_states=ctx,
                           context_mask=[torch.arange(N_CTX, device=DEVICE).unsqueeze(0)],
                           target_mask=[torch.arange(N_CTX, N_ALL, device=DEVICE).unsqueeze(0)]).last_hidden_state
    target = F.layer_norm(target.float(), (target.size(-1),))                                   # as in V-JEPA training
    err = (pred.float() - target).abs().mean(-1)                                                 # L1 per token
    return err.view(T_PRED // TUBELET, GRID, GRID).mean(0), target.view(T_PRED // TUBELET, GRID, GRID, -1)


# %% [markdown]
# ## Baseline: "have I seen this patch before?"
#
# Like the seen-before baseline in 04/05, but for video: the same V-JEPA 2
# encoder, no predictor. A test patch's score is its distance to the closest
# latent at the **same position** in the normal training videos.

# %%
def unit(target):
    """(2, 16, 16, 1024) latents -> (2, 256, 1024) unit vectors."""
    return F.normalize(target.flatten(1, 2), dim=-1).half()


t0 = time.time()
bank, train_surprise = [], []
for frames in train_videos.values():
    for s in range(0, len(frames) - T + 1, 8):
        m, target = surprise_map(frames[s:s + T])
        bank.append(unit(target))
        train_surprise.append(m)
bank = torch.cat(bank)                                        # (n_tubelets, 256 positions, 1024)
train_surprise = torch.stack(train_surprise)                  # (n_windows, 16, 16)
MU, SD = train_surprise.mean(0), train_surprise.std(0) + 1e-6  # how surprising each spot usually is
print(f"memory bank: {bank.shape[0]} tubelets from training videos in {time.time() - t0:.0f}s")


@torch.no_grad()
def window_maps(clip):
    """All scores for one window, each 16 x 16: raw surprise, calibrated surprise, seen-before distance."""
    surprise, target = surprise_map(clip)
    calibrated = (surprise - MU) / SD                         # surprise relative to this spot's normal level
    d = torch.stack([1 - torch.einsum("pd,npd->np", zt, bank).max(0).values.float() for zt in unit(target)])
    return surprise, calibrated, d.mean(0).view(GRID, GRID)


METHODS = ["V-JEPA 2 surprise, raw", "V-JEPA 2 surprise, calibrated", "seen-before baseline"]


# %% [markdown]
# ## Score every test frame
#
# Slide a 16-frame window over each test video (2 frames at a time). The
# window's surprise belongs to the 4 frames that were predicted. A frame's
# map is the average over the windows that predicted it; frames at the start
# and end of a video that were never predicted copy their nearest neighbour.
#
# A frame's score is the **mean of its 5 most surprising patches**: anomalies
# (a bike, a cart) cover only a few patches, so averaging the whole frame
# would wash them out.

# %%
def frame_maps(frames, map_fn=None, n_maps=None):
    """-> one (n_frames, 16, 16) map per output of map_fn (default: the 3 METHODS)."""
    map_fn, n_maps = map_fn or window_maps, n_maps or len(METHODS)
    maps = torch.zeros(n_maps, len(frames), GRID, GRID, device=DEVICE)
    count = torch.zeros(len(frames), device=DEVICE)
    for s in range(0, len(frames) - T + 1, STRIDE):
        for k, m in enumerate(map_fn(frames[s:s + T])):
            maps[k, s + T_CTX:s + T] += m
        count[s + T_CTX:s + T] += 1
    seen = torch.nonzero(count).flatten()
    maps = maps / count.clamp(min=1)[None, :, None, None]
    first, last = int(seen[0]), int(seen[-1])
    maps[:, :first], maps[:, last + 1:] = maps[:, first:first + 1], maps[:, last:last + 1]   # edges copy neighbours
    return [m.cpu() for m in maps]


def topk_score(maps, k=5):
    return maps.flatten(1).topk(k, dim=1).values.mean(1).numpy()


t0 = time.time()
maps = {m: {} for m in METHODS}
for name, frames in test_videos.items():
    for method, m in zip(METHODS, frame_maps(frames)):
        maps[method][name] = m
print(f"scored {len(test_videos)} test videos in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Frame-level AUC
#
# The standard Ped2 metric: AUC over all test frames, anomalous vs normal.
# Two versions:
# - **raw**: one threshold for every video.
# - **per-video normalised**: each video's scores min-max scaled to [0, 1]
#   first, as most Ped2 papers do (it removes per-video offsets).

# %%
def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-12)


y_all = np.concatenate([labels[n] for n in test_videos])
results, scores = {}, {}
for method, per_video in maps.items():
    s = {n: topk_score(m) for n, m in per_video.items()}
    scores[method] = s
    results[method] = {
        "AUC raw": auc(y_all, np.concatenate([s[n] for n in test_videos])),
        "AUC per-video normalised": auc(y_all, np.concatenate([minmax(s[n]) for n in test_videos])),
        "AUC raw, whole-frame mean": auc(y_all, np.concatenate([per_video[n].flatten(1).mean(1).numpy() for n in test_videos])),
    }
for method, r in results.items():
    print(f"{method:22s} " + "  ".join(f"{k} {v:.3f}" for k, v in r.items()))

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)

name = "04"                                                   # a test video with a clear anomaly
frames, y = test_videos[name], labels[name]
fig, ax = plt.subplots(figsize=(10, 3.2))
for method, color in [("V-JEPA 2 surprise, calibrated", "#1baf7a"), ("seen-before baseline", "#898781")]:
    ax.plot(minmax(scores[method][name]), color=color, label=method)
ax.fill_between(range(len(y)), 0, 1, where=y, color="#e34948", alpha=.12, label="anomaly (ground truth)")
ax.set_xlabel("frame"); ax.set_ylabel("score (scaled to 0–1)"); ax.set_xlim(0, len(y) - 1); ax.legend(fontsize=8)
ax.set_title(f"Test video {name}: score over time")
plt.tight_layout(); plt.savefig(OUT / "timeline.png", dpi=110); plt.show()

# %%
picks = [int(np.argmin(np.where(y, np.inf, scores["V-JEPA 2 surprise, calibrated"][name]))),
         int(np.argmax(np.where(y, scores["V-JEPA 2 surprise, calibrated"][name], -np.inf)))]
vmax = float(maps["V-JEPA 2 surprise, calibrated"][name][picks].max())
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, i, title in zip(axes, picks, ["normal frame", "anomalous frame"]):
    heat = F.interpolate(maps["V-JEPA 2 surprise, calibrated"][name][i][None, None], size=frames.shape[1:3], mode="bilinear")[0, 0]
    ax.imshow(frames[i]); ax.imshow(heat, cmap="inferno", alpha=.5, vmin=0, vmax=vmax)
    ax.set_title(f"{title} (frame {i}): where V-JEPA 2 is surprised", fontsize=9); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "heatmap.png", dpi=110); plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
methods = list(results)
keys = ["AUC raw", "AUC per-video normalised"]
for k, (key, color) in enumerate(zip(keys, ["#9fc5f0", "#2a78d6"])):
    vals = [results[m][key] for m in methods]
    bars = ax.bar([x + (k - 0.5) * 0.38 for x in range(len(methods))], vals, 0.38, color=color, label=key)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(methods)), methods); ax.set_ylim(0.4, 1.05); ax.set_ylabel("frame-level AUC")
ax.set_title("UCSD Ped2, zero-shot (no training on Ped2)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "auc.png", dpi=110); plt.show()

# %%
fig, axes = plt.subplots(2, 4, figsize=(10, 4.4))
clip_ex = test_videos["04"][60:76]
for idx, (f_idx, role) in enumerate([(0, "past t-11"), (4, "past t-7"), (8, "past t-3"), (11, "past t"),
                                     (12, "future t+1 (predict)"), (13, "future t+2 (predict)"), (14, "future t+3 (predict)"), (15, "future t+4 (predict)")]):
    ax = axes[idx // 4, idx % 4]
    ax.imshow(clip_ex[f_idx])
    color = "#2a78d6" if f_idx < 12 else "#e0622f"
    for spine in ax.spines.values():
        spine.set_edgecolor(color); spine.set_linewidth(2.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{role} (frame {60+f_idx})", fontsize=8.5, color=color, fontweight="bold")
fig.suptitle("What V-JEPA 2 sees on UCSD Ped2 (video 04): 12 context frames (blue) → predict 4 future frames (orange)", fontsize=9.5)
plt.tight_layout(); plt.savefig(OUT / "task.png", dpi=110); plt.show()

# %%
raw_ex = maps["V-JEPA 2 surprise, raw"]["04"][picks[1]]
cal_ex = maps["V-JEPA 2 surprise, calibrated"]["04"][picks[1]]
sb_ex  = maps["seen-before baseline"]["04"][picks[1]]
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1))
for ax, m_img, title in [
    (axes[0], MU.detach().cpu(), "Normal mean surprise μ\n(trees/edges always noisy)"),
    (axes[1], raw_ex, f"Raw surprise (frame {picks[1]})\nAUC 0.474 (drowned by background)"),
    (axes[2], cal_ex, f"Calibrated (s − μ)/σ (frame {picks[1]})\nAUC 0.683 (spots the cyclist)"),
    (axes[3], sb_ex,  f"Seen-before baseline (frame {picks[1]})\nAUC 0.832 (nearest normal patch)"),
]:
    ax.imshow(frames[picks[1]])
    heat = F.interpolate(m_img.float()[None, None], size=frames.shape[1:3], mode="bilinear")[0, 0]
    ax.imshow(heat, cmap="inferno", alpha=0.55)
    ax.set_title(title, fontsize=8.5); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=110); plt.show()

# %% [markdown]
# ## Part 2: adapt the predictor to this camera
#
# Zero-shot, V-JEPA 2's predictor doesn't know that *on this camera* the
# trees always sway and people always walk the same way. So we fine-tune
# **only the predictor** (22M parameters; the 300M encoder stays frozen) to
# predict the future of Ped2's **normal training videos**, exactly the 03
# recipe on a real world model. 12 training videos are used for fitting,
# 4 are held out for early stopping and for calibrating the surprise.

# %%
import copy

HOLDOUT = list(train_videos)[-4:]
FIT = [n for n in train_videos if n not in HOLDOUT]


@torch.no_grad()
def encode_window(clip):
    """Context latents (past 12 frames only) and layer-normed target latents (next 4)."""
    px = to_pixels(clip)
    ctx = model(pixel_values_videos=px[:, :T_CTX], skip_predictor=True).last_hidden_state[0]
    tgt = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state[0, N_CTX:]
    return ctx, F.layer_norm(tgt.float(), (tgt.size(-1),)).half()


def cache(names, stride):
    pairs = [encode_window(train_videos[n][s:s + T]) for n in names for s in range(0, len(train_videos[n]) - T + 1, stride)]
    return torch.stack([c for c, _ in pairs]), torch.stack([t for _, t in pairs])


def token_l1(predictor, ctx, tgt):
    """(B, 512) L1 error per predicted token."""
    B = ctx.size(0)
    cm = [torch.arange(N_CTX, device=DEVICE).unsqueeze(0).expand(B, -1)]
    tm = [torch.arange(N_CTX, N_ALL, device=DEVICE).unsqueeze(0).expand(B, -1)]
    with torch.autocast("cuda", dtype=torch.float16):
        pred = predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
    return (pred.float() - tgt.float()).abs().mean(-1)


t0 = time.time()
CTX_FIT, TGT_FIT = cache(FIT, 4)
CTX_HOLD, TGT_HOLD = cache(HOLDOUT, 8)
print(f"cached {len(CTX_FIT)} fit / {len(CTX_HOLD)} held-out windows in {time.time() - t0:.0f}s")


@torch.no_grad()
def holdout_loss(predictor):
    return torch.cat([token_l1(predictor, c, t).mean(1) for c, t in zip(CTX_HOLD.split(8), TGT_HOLD.split(8))]).mean().item()


adapted = copy.deepcopy(model.predictor).float()
opt = torch.optim.AdamW(adapted.parameters(), lr=1e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler()
EPOCHS = 6
adapt_history = [(0, None, holdout_loss(model.predictor))]
best, best_state = adapt_history[0][2], copy.deepcopy(adapted.state_dict())
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    adapted.train()
    losses = []
    for b in torch.randperm(len(CTX_FIT), device=DEVICE).split(8):
        loss = token_l1(adapted, CTX_FIT[b], TGT_FIT[b]).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    adapted.eval()
    hold = holdout_loss(adapted)
    adapt_history.append((epoch, sum(losses) / len(losses), hold))
    if hold < best:
        best, best_state = hold, copy.deepcopy(adapted.state_dict())
    print(f"epoch {epoch}  fit L1 {adapt_history[-1][1]:.4f}  held-out L1 {hold:.4f}  ({time.time() - t0:.0f}s)")
adapted.load_state_dict(best_state)
adapted.eval()
print(f"held-out L1: zero-shot {adapt_history[0][2]:.4f} -> adapted {best:.4f}")

# %%
@torch.no_grad()
def adapted_map(clip):
    ctx, tgt = encode_window(clip)
    return token_l1(adapted, ctx[None], tgt[None])[0].view(T_PRED // TUBELET, GRID, GRID).mean(0)


del opt, scaler, CTX_FIT, TGT_FIT                             # free training memory before scoring
torch.cuda.empty_cache()
with torch.no_grad():
    hold_maps = torch.stack([token_l1(adapted, c[None], t[None])[0].view(T_PRED // TUBELET, GRID, GRID).mean(0)
                             for c, t in zip(CTX_HOLD, TGT_HOLD)])
MU_A, SD_A = hold_maps.mean(0), hold_maps.std(0) + 1e-6     # calibrated on held-out normal video only

t0 = time.time()
maps["adapted surprise, raw"], maps["adapted surprise, calibrated"] = {}, {}
for name, frames in test_videos.items():
    raw = frame_maps(frames, lambda c: [adapted_map(c)], 1)[0]
    maps["adapted surprise, raw"][name] = raw
    maps["adapted surprise, calibrated"][name] = (raw - MU_A.cpu()) / SD_A.cpu()
print(f"rescored test videos with the adapted predictor in {time.time() - t0:.0f}s")


def rank01(x):
    return np.argsort(np.argsort(x)) / (len(x) - 1)


frame_score = {
    "top5": lambda m: topk_score(m),
    "mean": lambda m: m.flatten(1).mean(1).numpy(),
}
all_scores = {(method, kind): np.concatenate([fn(maps[method][n]) for n in test_videos])
              for method in maps for kind, fn in frame_score.items()}
part2 = {f"{method} ({kind})": auc(y_all, s) for (method, kind), s in all_scores.items()}
best_surprise = max((k for k in all_scores if k[0].startswith("adapted")), key=lambda k: part2[f"{k[0]} ({k[1]})"])
combined = rank01(all_scores[best_surprise]) + rank01(all_scores[("seen-before baseline", "mean")])
part2[f"combined: {best_surprise[0]} ({best_surprise[1]}) + seen-before (mean)"] = auc(y_all, combined)
for k, v in sorted(part2.items(), key=lambda kv: -kv[1]):
    print(f"{v:.3f}  {k}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
ep, fit_l, hold_l = zip(*adapt_history)
axes[0].plot(ep[1:], fit_l[1:], marker="o", color="#2a78d6", label="fit videos")
axes[0].plot(ep, hold_l, marker="o", color="#1baf7a", label="held-out normal videos")
axes[0].set_xlabel("epoch (0 = zero-shot)"); axes[0].set_ylabel("prediction L1"); axes[0].grid(alpha=.3)
axes[0].set_title("Adapting the predictor to Ped2"); axes[0].legend(fontsize=8)
items = sorted(part2.items(), key=lambda kv: kv[1])
axes[1].barh([k for k, _ in items], [v for _, v in items], color="#1baf7a")
for i, (_, v) in enumerate(items):
    axes[1].text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=7)
axes[1].set_xlim(0.4, 1.02); axes[1].tick_params(axis="y", labelsize=7); axes[1].set_xlabel("frame-level AUC (raw)")
axes[1].set_title("Every score, zero-shot vs adapted")
plt.tight_layout(); plt.savefig(OUT / "adapted.png", dpi=110); plt.show()

# %%
import json

(OUT / "results_part2.json").write_text(json.dumps({
    "adapt_history": adapt_history, "holdout_videos": HOLDOUT, "epochs": EPOCHS,
    "auc_raw": part2, "best_adapted_score": list(best_surprise)}, indent=1))
