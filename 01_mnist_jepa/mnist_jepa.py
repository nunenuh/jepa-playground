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
# # MNIST JEPA — the "hello world" of JEPA
#
# Idea: cut each digit into 16 patches (4x4 grid of 7x7 pixels).
# Hide some patches. From the visible patches, predict the *embedding* of the
# hidden patches — not their pixels.
#
#   context encoder  : sees only visible patches      -> context embeddings
#   target encoder   : EMA copy, sees the full image  -> target embeddings (no gradient)
#   predictor        : context embeddings + "where"   -> predicted target embeddings
#   loss             : distance(predicted, target) on the hidden patches only
#
# No labels are used for training. At the end we check what it learned with a
# linear probe: freeze the encoder, train only one linear layer on the labels.
# Run as a script, or open in VS Code / Jupyter (the `# %%` lines are cells).

# %% [markdown]
# ## Architecture: one training step
#
# ![MNIST JEPA architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/01_mnist_jepa/assets/architecture-light.svg)
#
# Blue boxes are trained by the loss. The dashed target encoder is never trained
# directly: after each step it moves 0.4% of the way towards the context encoder (EMA).

# %%
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PATCH = 7                 # 28 / 7 = 4 patches per side
N_PATCHES = 16            # 4 x 4
N_TARGET = 6              # how many patches to hide per image
DIM = 128                 # embedding size
EPOCHS = 10
EMA = 0.996               # target encoder = slow moving average of context encoder

# %% Data
tf = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1024)


def patchify(x):
    """(B, 1, 28, 28) -> (B, 16, 49): one row per 7x7 patch."""
    x = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)    # B,1,4,4,7,7
    return x.reshape(x.size(0), N_PATCHES, PATCH * PATCH)


def gather(tokens, idx):
    """Pick tokens at positions idx. tokens: (B, N, D), idx: (B, K)."""
    return tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))


# %% Model
def transformer(depth):
    layer = nn.TransformerEncoderLayer(DIM, nhead=4, dim_feedforward=DIM * 2,
                                       batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(PATCH * PATCH, DIM)
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, DIM) * 0.02)
        self.blocks = transformer(depth=4)
        self.norm = nn.LayerNorm(DIM)

    def forward(self, patches, idx=None):
        x = self.embed(patches) + self.pos
        if idx is not None:                  # keep only the visible patches
            x = gather(x, idx)
        return self.norm(self.blocks(x))


class Predictor(nn.Module):
    """Given context embeddings and the *positions* of hidden patches,
    predict the hidden patches' embeddings."""
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, DIM) * 0.02)
        self.blocks = transformer(depth=2)
        self.out = nn.Linear(DIM, DIM)

    def forward(self, ctx, ctx_idx, tgt_idx):
        pos = self.pos.expand(ctx.size(0), -1, -1)
        ctx = ctx + gather(pos, ctx_idx)
        queries = self.mask_token + gather(pos, tgt_idx)   # "what is at position k?"
        x = self.blocks(torch.cat([ctx, queries], dim=1))
        return self.out(x[:, ctx.size(1):])


context_enc = Encoder().to(DEVICE)
target_enc = copy.deepcopy(context_enc).requires_grad_(False)
predictor = Predictor().to(DEVICE)
opt = torch.optim.AdamW([*context_enc.parameters(), *predictor.parameters()], lr=1e-3, weight_decay=0.05)


# %% Linear probe: how much digit information is in the frozen embeddings?
def linear_probe(encoder, epochs=5):
    encoder.eval()

    def features(dl):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in dl:
                feats.append(encoder(patchify(x.to(DEVICE))).mean(1))   # average over patches
                labels.append(y.to(DEVICE))
        return torch.cat(feats), torch.cat(labels)

    xtr, ytr = features(train_dl)
    xte, yte = features(test_dl)
    probe = nn.Linear(DIM, 10).to(DEVICE)
    popt = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        for i in torch.randperm(len(xtr), device=DEVICE).split(512):
            popt.zero_grad()
            F.cross_entropy(probe(xtr[i]), ytr[i]).backward()
            popt.step()
    encoder.train()
    return (probe(xte).argmax(1) == yte).float().mean().item()


random_enc = copy.deepcopy(target_enc)       # keep the untrained encoder for comparison
probe_before = linear_probe(random_enc)
print(f"probe accuracy BEFORE training (random encoder): {probe_before:.1%}")

# %% Train
history = []
for epoch in range(EPOCHS):
    for x, _ in train_dl:                     # labels are ignored
        patches = patchify(x.to(DEVICE))
        B = patches.size(0)

        # random split per image: N_TARGET hidden patches, the rest visible
        perm = torch.rand(B, N_PATCHES, device=DEVICE).argsort(1)
        tgt_idx, ctx_idx = perm[:, :N_TARGET], perm[:, N_TARGET:]

        with torch.no_grad():                 # targets: full image through the EMA encoder
            target = gather(target_enc(patches), tgt_idx)

        pred = predictor(context_enc(patches, ctx_idx), ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        with torch.no_grad():                 # EMA update: target slowly follows context
            for pt, pc in zip(target_enc.parameters(), context_enc.parameters()):
                pt.mul_(EMA).add_(pc, alpha=1 - EMA)

    # collapse check: if every image gets the same embedding, std -> 0
    spread = target.reshape(-1, DIM).std(0).mean().item()
    history.append((loss.item(), spread))
    print(f"epoch {epoch + 1:2d}  loss {loss.item():.4f}  embedding spread {spread:.3f}")

# %% Result
probe_after = linear_probe(target_enc)
print(f"probe accuracy AFTER training (JEPA encoder):   {probe_after:.1%}")

# %% [markdown]
# ## Visualize what it learned
# Three figures, saved to `assets/` next to this file:
# 1. **The task** — hidden patches, and how close each predicted embedding is to the real one.
# 2. **Training curves** — loss goes *up* (targets get richer); spread is the collapse check.
# 3. **t-SNE** — embeddings before vs after training. After, digits form groups.

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
HIDDEN = "#eb6834"
context_enc.eval(); predictor.eval(); target_enc.eval()

# %% 1. The task: one test image per digit, 6 patches hidden
fig, axes = plt.subplots(2, 10, figsize=(12, 3))
gen = torch.Generator().manual_seed(3)
for d in range(10):
    x, _ = next((test_ds[i] for i in range(len(test_ds)) if test_ds.targets[i] == d))
    patches = patchify(x[None].to(DEVICE))
    perm = torch.rand(1, N_PATCHES, generator=gen).argsort(1).to(DEVICE)
    tgt_idx, ctx_idx = perm[:, :N_TARGET], perm[:, N_TARGET:]
    with torch.no_grad():
        target = gather(target_enc(patches), tgt_idx)
        pred = predictor(context_enc(patches, ctx_idx), ctx_idx, tgt_idx)
        sim = F.cosine_similarity(pred, target, dim=-1).mean().item()

    shown = x[0].clone()
    for p in tgt_idx[0].tolist():
        r, c = divmod(p, 4)
        shown[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = 0.25
    for row, img in enumerate([shown, x[0]]):
        ax = axes[row, d]
        ax.imshow(img, cmap="gray_r", vmin=0, vmax=1)
        for p in tgt_idx[0].tolist():
            r, c = divmod(p, 4)
            ax.add_patch(plt.Rectangle((c * PATCH - .5, r * PATCH - .5), PATCH, PATCH, fill=False, ec=HIDDEN, lw=1.2))
        ax.set_xticks([]); ax.set_yticks([])
    axes[1, d].set_xlabel(f"{sim:.0%}", fontsize=9)
axes[0, 0].set_ylabel("shown"); axes[1, 0].set_ylabel("full")
fig.suptitle("Hidden patches (orange). % = similarity of predicted vs real embedding")
plt.tight_layout(); plt.savefig(OUT / "task.png", dpi=110); plt.show()

# %% 2. Training curves (two charts, never one chart with two y-axes)
loss_hist, spread_hist = zip(*history)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3))
epochs = range(1, len(history) + 1)
a1.plot(epochs, loss_hist, marker="o"); a1.set_title("Loss per epoch (going up is normal)")
a2.plot(epochs, spread_hist, marker="o"); a2.set_title("Embedding spread (near 0 = collapse)")
for a in (a1, a2):
    a.set_xlabel("epoch"); a.set_ylim(bottom=0); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=110); plt.show()

# %% 3. t-SNE of embeddings: random encoder vs JEPA encoder
x, y = next(iter(torch.utils.data.DataLoader(test_ds, batch_size=1500)))
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, enc, title in [(axes[0], random_enc, f"Before training: probe {probe_before:.1%}"),
                       (axes[1], target_enc, f"After JEPA training: probe {probe_after:.1%}")]:
    with torch.no_grad():
        emb = enc(patchify(x.to(DEVICE))).mean(1).cpu().numpy()
    xy = TSNE(2, random_state=0, init="pca").fit_transform(emb)
    for (px, py), label in zip(xy[:900], y[:900].tolist()):      # each point drawn as its digit
        ax.text(px, py, str(label), fontsize=7, ha="center", va="center", color="0.35")
    for d in range(10):                                           # big label at each digit's centre
        cx, cy = np.median(xy[y.numpy() == d], 0)
        ax.text(cx, cy, str(d), fontsize=16, weight="bold", color="#2a78d6", ha="center", va="center")
    ax.set_xlim(xy[:, 0].min() - 3, xy[:, 0].max() + 3); ax.set_ylim(xy[:, 1].min() - 3, xy[:, 1].max() + 3)
    ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.savefig(OUT / "tsne.png", dpi=110); plt.show()

# %% Raw numbers
import json

(OUT / "results.json").write_text(json.dumps({
    "probe_accuracy": {"random_encoder": probe_before, "jepa_encoder": probe_after},
    "per_epoch": [{"epoch": i + 1, "loss": l, "spread": s} for i, (l, s) in enumerate(history)],
    "config": {"epochs": EPOCHS, "dim": DIM, "n_target": N_TARGET, "ema": EMA, "batch": 256,
               "device": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"}}, indent=1))
