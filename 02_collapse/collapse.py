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
# # 02 · Collapse on purpose
#
# JEPA has a cheat: if the encoder maps **every image to the same embedding**,
# the predictor can output that constant and the loss is ~0. That is
# *collapse*, and the model learns nothing.
#
# Here we train the same MNIST JEPA from experiment 01 four ways and watch what
# stops (or doesn't stop) the cheat:
#
# | Run | Target comes from | Gradient into target? | Extra loss |
# |---|---|---|---|
# | `ema+stopgrad` | EMA copy of the encoder (01 setup) | no | – |
# | `no-stopgrad` | the encoder itself | **yes** | – |
# | `stopgrad-only` | the encoder itself, detached | no | – |
# | `sigreg` | the encoder itself | **yes** | SIGReg (LeJEPA) |
#
# Three numbers per run tell the story:
# - **spread**: std of embeddings across images. ~0 = everything is the same.
# - **effective rank**: how many of the 128 dimensions are really used. 1 = collapsed.
# - **linear probe accuracy**: how much digit information survived.
#
# ![the 4 runs](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/02_collapse/assets/variants-light.svg)

# %%
import copy
import math

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATCH, N_PATCHES, N_TARGET, DIM = 7, 16, 6, 128
EPOCHS = 5
EMA = 0.996
SIGREG_WEIGHT = 0.05

tf = transforms.ToTensor()
train_ds = datasets.MNIST("data", train=True, download=True, transform=tf)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tf)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1024)


# %% Same model as experiment 01
def patchify(x):
    x = x.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)
    return x.reshape(x.size(0), N_PATCHES, PATCH * PATCH)


def gather(tokens, idx):
    return tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1)))


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
        if idx is not None:
            x = gather(x, idx)
        return self.norm(self.blocks(x))


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, DIM))
        self.pos = nn.Parameter(torch.randn(1, N_PATCHES, DIM) * 0.02)
        self.blocks = transformer(depth=2)
        self.out = nn.Linear(DIM, DIM)

    def forward(self, ctx, ctx_idx, tgt_idx):
        pos = self.pos.expand(ctx.size(0), -1, -1)
        ctx = ctx + gather(pos, ctx_idx)
        queries = self.mask_token + gather(pos, tgt_idx)
        x = self.blocks(torch.cat([ctx, queries], dim=1))
        return self.out(x[:, ctx.size(1):])


# %% [markdown]
# ## SIGReg (from LeJEPA, simplified)
#
# Instead of tricks (EMA, stop-gradient), LeJEPA *requires* the embeddings to
# look like a standard Gaussian. A collapsed embedding (a single point) is as
# far from a Gaussian as you can get, so it is penalised directly.
#
# Checking "is this 128-D cloud Gaussian?" is hard, so SIGReg checks many random
# 1-D projections instead: project onto a random direction, then compare the
# projection's characteristic function with that of N(0, 1) (Epps–Pulley test).

# %%
T_GRID = torch.linspace(-5, 5, 17)


def sigreg(z, n_slices=256):
    """z: (N, D) embeddings. Returns how far random 1-D projections are from N(0,1)."""
    dirs = F.normalize(torch.randn(z.size(1), n_slices, device=z.device), dim=0)
    proj = z @ dirs                                        # (N, S)
    t = T_GRID.to(z.device)
    xt = proj.unsqueeze(-1) * t                            # (N, S, T)
    ecf_re, ecf_im = xt.cos().mean(0), xt.sin().mean(0)    # empirical characteristic function
    gauss = torch.exp(-t ** 2 / 2)                         # characteristic function of N(0,1)
    err = (ecf_re - gauss) ** 2 + ecf_im ** 2
    return torch.trapezoid(err * gauss, t, dim=-1).mean() * z.size(0)


# %% Collapse metrics + linear probe
@torch.no_grad()
def embed_test(encoder, n=2048):
    x, y = next(iter(torch.utils.data.DataLoader(test_ds, batch_size=n)))
    return encoder(patchify(x.to(DEVICE))).mean(1), y.to(DEVICE)


def effective_rank(z):
    """exp(entropy of normalised singular values): 1 = collapsed, DIM = all dims used."""
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum()
    return math.exp(-(p * torch.log(p + 1e-12)).sum().item())


def linear_probe(encoder, epochs=5):
    encoder.eval()

    def features(dl):
        feats, labels = [], []
        with torch.no_grad():
            for x, y in dl:
                feats.append(encoder(patchify(x.to(DEVICE))).mean(1))
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


# %% One training run
def train(mode):
    torch.manual_seed(0)
    context_enc, predictor = Encoder().to(DEVICE), Predictor().to(DEVICE)
    target_enc = copy.deepcopy(context_enc).requires_grad_(False)   # only used by ema+stopgrad
    opt = torch.optim.AdamW([*context_enc.parameters(), *predictor.parameters()],
                            lr=1e-3, weight_decay=0.05)
    hist = {"loss": [], "spread": [], "erank": []}

    for epoch in range(EPOCHS):
        for x, _ in train_dl:
            patches = patchify(x.to(DEVICE))
            perm = torch.rand(patches.size(0), N_PATCHES, device=DEVICE).argsort(1)
            tgt_idx, ctx_idx = perm[:, :N_TARGET], perm[:, N_TARGET:]

            if mode == "ema+stopgrad":
                with torch.no_grad():
                    full = target_enc(patches)
            elif mode == "stopgrad-only":
                with torch.no_grad():
                    full = context_enc(patches)                 # same weights, no gradient
            else:                                               # no-stopgrad, sigreg
                full = context_enc(patches)                     # gradient flows into the target too
            target = gather(full, tgt_idx)

            pred = predictor(context_enc(patches, ctx_idx), ctx_idx, tgt_idx)
            loss = F.smooth_l1_loss(pred, target)
            if mode == "sigreg":
                loss = loss + SIGREG_WEIGHT * sigreg(full.mean(1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            if mode == "ema+stopgrad":
                with torch.no_grad():
                    for pt, pc in zip(target_enc.parameters(), context_enc.parameters()):
                        pt.mul_(EMA).add_(pc, alpha=1 - EMA)

        enc = target_enc if mode == "ema+stopgrad" else context_enc
        z, _ = embed_test(enc)
        hist["loss"].append(loss.item())
        hist["spread"].append(z.std(0).mean().item())
        hist["erank"].append(effective_rank(z))
        print(f"[{mode:13s}] epoch {epoch + 1}  loss {loss.item():.4f}  "
              f"spread {hist['spread'][-1]:.3f}  eff. rank {hist['erank'][-1]:.1f}")

    hist["probe"] = linear_probe(enc)
    print(f"[{mode:13s}] probe accuracy {hist['probe']:.1%}\n")
    return hist


# %% Run all four
MODES = ["ema+stopgrad", "no-stopgrad", "stopgrad-only", "sigreg"]
torch.manual_seed(0)
random_probe = linear_probe(Encoder().to(DEVICE))
print(f"random encoder probe accuracy: {random_probe:.1%}\n")
results = {m: train(m) for m in MODES}

# %% Summary table
print(f"{'run':15s} {'final loss':>10s} {'spread':>8s} {'eff. rank':>10s} {'probe':>7s}")
for m, h in results.items():
    print(f"{m:15s} {h['loss'][-1]:10.4f} {h['spread'][-1]:8.3f} {h['erank'][-1]:10.1f} {h['probe']:7.1%}")
print(f"{'random encoder':15s} {'':>10s} {'':>8s} {'':>10s} {random_probe:7.1%}")

# %% Plots: one chart per metric (never two y-axes on one chart)
from pathlib import Path

OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
COLORS = {"ema+stopgrad": "#2a78d6", "no-stopgrad": "#e34948", "stopgrad-only": "#eda100", "sigreg": "#1baf7a"}
epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for key, ax, title in [("loss", axes[0], "Training loss (low ≠ good!)"),
                       ("spread", axes[1], "Embedding spread (0 = collapsed)"),
                       ("erank", axes[2], f"Effective rank (max {DIM})")]:
    for m, h in results.items():
        ax.plot(epochs, h[key], marker="o", color=COLORS[m], label=m)
    ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylim(bottom=0); ax.grid(alpha=.3)
axes[0].legend(fontsize=8)

ax = axes[3]
names = list(results) + ["random"]
accs = [results[m]["probe"] for m in results] + [random_probe]
bars = ax.barh(names, accs, color=[COLORS.get(n, "#898781") for n in names])
for b, a in zip(bars, accs):
    ax.text(a + 0.01, b.get_y() + b.get_height() / 2, f"{a:.1%}", va="center", fontsize=9)
ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_title("Linear probe accuracy")
plt.tight_layout(); plt.savefig(OUT / "collapse.png", dpi=110); plt.show()
