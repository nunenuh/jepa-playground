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


print(f"probe accuracy BEFORE training (random encoder): {linear_probe(target_enc):.1%}")

# %% Train
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
    print(f"epoch {epoch + 1:2d}  loss {loss.item():.4f}  embedding spread {spread:.3f}")

# %% Result
print(f"probe accuracy AFTER training (JEPA encoder):   {linear_probe(target_enc):.1%}")
