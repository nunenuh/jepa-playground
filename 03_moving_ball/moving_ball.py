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
# # 03 · Moving-ball world model
#
# Experiments 01–02 hid part of an **image**. Now we hide the **future**:
# given 4 frames of a bouncing ball, predict the *embedding* of the next frame.
# That is a tiny world model, the idea behind V-JEPA.
#
# The payoff is **anomaly detection for free**. The model is trained only on
# normal physics. When something impossible happens (the ball teleports, or
# turns around mid-air), its prediction is wrong and the error spikes.
# Prediction error = *surprise* = anomaly score. No anomaly labels needed.
#
# This is the same idea as experiment 04: swap "ball frames" for "agent tool
# calls", train on normal behaviour, and flag the step the model didn't expect.
#
# No dataset to download: the frames are generated in code.

# %%
import copy
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIZE, RADIUS = 32, 2.5   # 32×32 frames, ball radius in pixels
CTX = 4                  # frames the predictor sees
DIM = 128                # embedding size
STEPS = 3000             # training steps (each = a fresh batch of simulated clips)
BATCH = 256
SIGREG_WEIGHT = 0.002    # anti-collapse (02 showed why it's needed)
torch.manual_seed(0)


# %% [markdown]
# ## The world: one ball, constant speed, bounces off walls
#
# `simulate` returns ball positions; `render` turns them into frames.
# Anomalies happen at frame `at`:
# - **teleport**: the ball jumps to a random place.
# - **reverse**: the ball turns around in mid-air, with no wall to bounce off.

# %%
LO, HI = RADIUS, SIZE - 1 - RADIUS
COORDS = torch.arange(SIZE, device=DEVICE).float()


def simulate(n, T, anomaly=None, at=None):
    """Ball positions, shape (n, T, 2)."""
    pos = torch.rand(n, 2, device=DEVICE) * (HI - LO) + LO
    angle = torch.rand(n, device=DEVICE) * 2 * math.pi
    speed = torch.rand(n, device=DEVICE) * 1.5 + 1.0              # 1–2.5 px per frame
    vel = torch.stack([angle.cos(), angle.sin()], 1) * speed[:, None]
    frames = []
    for t in range(T):
        if anomaly == "teleport" and t == at:
            pos = torch.rand(n, 2, device=DEVICE) * (HI - LO) + LO
        frames.append(pos)
        if anomaly == "reverse" and t == at - 1:
            vel = -vel
        pos = pos + vel
        low, high = pos < LO, pos > HI                             # bounce
        pos = torch.where(low, 2 * LO - pos, torch.where(high, 2 * HI - pos, pos))
        vel = torch.where(low | high, -vel, vel)
    return torch.stack(frames, 1)


def render(pos):
    """(..., 2) positions -> (..., 1, SIZE, SIZE) frames with a soft ball."""
    x, y = pos[..., 0, None, None], pos[..., 1, None, None]
    d2 = (COORDS[None, :] - x) ** 2 + (COORDS[:, None] - y) ** 2
    return torch.exp(-d2 / (2 * (RADIUS / 1.5) ** 2)).unsqueeze(-3)


# %% [markdown]
# ## Model
#
# - **Encoder**: a small CNN, one frame -> one 128-D embedding. The same
#   encoder embeds the context frames and the target frame (LeJEPA style:
#   no EMA copy, no stop-gradient).
# - **Predictor**: an MLP, 4 frame embeddings -> embedding of the next frame.
# - **SIGReg** (from 02) keeps embeddings spread out like a Gaussian, so the
#   encoder can't collapse. The 01 trick (EMA target) collapsed here: with a
#   single dot on a black frame, it's too easy to make every frame look alike.

# %%
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.GELU(),     # 16×16
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),    # 8×8
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),   # 4×4
            nn.Flatten(), nn.Linear(128 * 16, DIM), nn.LayerNorm(DIM))

    def forward(self, frames):                         # (B, T, 1, S, S) -> (B, T, DIM)
        B, T = frames.shape[:2]
        return self.net(frames.flatten(0, 1)).view(B, T, DIM)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CTX * DIM, 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(),
                                 nn.Linear(512, DIM))

    def forward(self, ctx):                            # (B, CTX, DIM) -> (B, DIM)
        return self.net(ctx.flatten(1))


T_GRID = torch.linspace(-5, 5, 17, device=DEVICE)


def sigreg(z, n_slices=256):
    """How far random 1-D projections of z (N, D) are from N(0, 1). See 02."""
    dirs = F.normalize(torch.randn(z.size(1), n_slices, device=z.device), dim=0)
    xt = (z @ dirs).unsqueeze(-1) * T_GRID
    gauss = torch.exp(-T_GRID ** 2 / 2)
    err = (xt.cos().mean(0) - gauss) ** 2 + xt.sin().mean(0) ** 2
    return torch.trapezoid(err * gauss, T_GRID, dim=-1).mean() * z.size(0)


encoder, predictor = Encoder().to(DEVICE), Predictor().to(DEVICE)
opt = torch.optim.AdamW([*encoder.parameters(), *predictor.parameters()],
                        lr=1e-3, weight_decay=0.05)


# %% [markdown]
# ## Surprise score
#
# For every frame after the first 4: predict its embedding from the 4 frames
# before it, and measure how far the real embedding is from the prediction.

# %%
@torch.no_grad()
def surprise(frames, enc, pred):
    """frames (B, T, 1, S, S) -> prediction error per frame, (B, T - CTX)."""
    z = enc(frames)
    return torch.stack([(pred(z[:, t - CTX:t]) - z[:, t]).pow(2).mean(-1)
                        for t in range(CTX, frames.size(1))], 1)


def auc(normal, anomalous):
    """Chance a random anomalous clip scores higher than a random normal one."""
    return (anomalous[:, None] > normal[None, :]).float().mean().item()


T_TEST, AT = 12, 8                                     # 12-frame clips, anomaly at frame 8
torch.manual_seed(1)
test = {kind: render(simulate(1000, T_TEST, kind, AT)) for kind in [None, "teleport", "reverse"]}


def detection_report(enc, pred):
    s = {k: surprise(v, enc, pred)[:, AT - CTX] for k, v in test.items()}
    return {k: auc(s[None], s[k]) for k in ["teleport", "reverse"]}


untrained = (copy.deepcopy(encoder), copy.deepcopy(predictor))
auc_before = detection_report(*untrained)
print("anomaly AUC before training:", {k: f"{v:.2f}" for k, v in auc_before.items()})

# %% Train
history = []
t0 = time.time()
torch.manual_seed(0)
for step in range(1, STEPS + 1):
    frames = render(simulate(BATCH, CTX + 1))          # fresh clips every step
    z = encoder(frames)                                 # (B, CTX + 1, DIM)
    pred_loss = F.mse_loss(predictor(z[:, :CTX]), z[:, CTX])
    loss = pred_loss + SIGREG_WEIGHT * sigreg(z.flatten(0, 1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    if step % 100 == 0:
        spread = z.detach().flatten(0, 1).std(0).mean().item()
        history.append((step, pred_loss.item(), spread))
    if step % 500 == 0:
        print(f"step {step:5d}  prediction loss {pred_loss.item():.4f}  spread {spread:.3f}  ({time.time() - t0:.0f}s)")

auc_after = detection_report(encoder, predictor)
print("anomaly AUC after training: ", {k: f"{v:.2f}" for k, v in auc_after.items()})


# %% [markdown]
# ## Did the encoder learn where the ball is?
#
# A linear regression from one frame's embedding to the ball's (x, y).
# R² = 1 means position can be read off perfectly.

# %%
@torch.no_grad()
def position_r2(encoder):
    torch.manual_seed(2)
    pos = simulate(4000, 1).squeeze(1)
    z = encoder(render(pos.unsqueeze(1))).squeeze(1)
    X = torch.cat([z, torch.ones(len(z), 1, device=DEVICE)], 1)
    W = torch.linalg.lstsq(X[:3000].cpu(), pos[:3000].cpu()).solution.to(DEVICE)
    err = (X[3000:] @ W - pos[3000:]).pow(2).sum()
    return 1 - (err / (pos[3000:] - pos[3000:].mean(0)).pow(2).sum()).item()


print(f"position R²  before: {position_r2(untrained[0]):.3f}   after: {position_r2(encoder):.3f}")

# %% [markdown]
# ## Pictures
#
# 1. One clip of each kind, with the surprise under every frame.
# 2. Surprise at the anomaly frame across 1000 clips of each kind.
# 3. Training curves.

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
COLORS = {None: "#2a78d6", "teleport": "#e34948", "reverse": "#eda100"}
NAMES = {None: "normal", "teleport": "teleport", "reverse": "reverse"}

idx = 0
fig, axes = plt.subplots(4, T_TEST, figsize=(T_TEST * 1.1, 5.6),
                         gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
for row, (kind, clip) in enumerate(test.items()):
    for t in range(T_TEST):
        ax = axes[row, t]
        ax.imshow(clip[idx, t, 0].cpu(), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if t == AT and kind:
            for s in ax.spines.values():
                s.set_color(COLORS[kind]); s.set_linewidth(2.5)
        if t == 0:
            ax.set_ylabel(NAMES[kind], fontsize=9)
        if row == 0:
            ax.set_title(f"t={t}", fontsize=8)
for ax in axes[3]:
    ax.remove()
ax = fig.add_subplot(4, 1, 4)
for kind, clip in test.items():
    s = surprise(clip[idx:idx + 1], encoder, predictor)[0].cpu()
    ax.plot(range(CTX, T_TEST), s, marker="o", color=COLORS[kind], label=NAMES[kind])
ax.axvline(AT, color="#898781", ls="--", lw=1)
ax.set_xlim(-0.5, T_TEST - 0.5); ax.set_xticks(range(T_TEST))
ax.set_xlabel("frame"); ax.set_ylabel("surprise"); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "clips.png", dpi=110); plt.show()

# %%
scores = {k: surprise(v, encoder, predictor)[:, AT - CTX].cpu() for k, v in test.items()}
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = torch.logspace(math.log10(scores[None].min()), math.log10(max(s.max() for s in scores.values())), 50)
for kind, s in scores.items():
    label = NAMES[kind] + (f"  (AUC {auc_after[kind]:.2f})" if kind else "")
    ax.hist(s, bins=bins, alpha=.6, color=COLORS[kind], label=label)
ax.set_xscale("log"); ax.set_xlabel(f"surprise at frame {AT}"); ax.set_ylabel("clips")
ax.legend(fontsize=8); ax.set_title("Normal physics vs anomalies (1000 clips each)")
plt.tight_layout(); plt.savefig(OUT / "surprise.png", dpi=110); plt.show()

# %%
steps, losses, spreads = zip(*history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(steps, losses, color="#2a78d6"); axes[0].set_title("Prediction loss")
axes[1].plot(steps, spreads, color="#1baf7a"); axes[1].set_title("Embedding spread (0 = collapsed)")
for ax in axes:
    ax.set_xlabel("step"); ax.set_ylim(bottom=0); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=110); plt.show()
