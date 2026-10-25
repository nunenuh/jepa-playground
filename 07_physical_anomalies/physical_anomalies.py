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
# # 07 · V-JEPA 2 and physically impossible video
#
# In 06, V-JEPA 2's surprise lost to a simple "have I seen this before?"
# check on UCSD Ped2. Our explanation: Ped2's anomalies (a bike, a cart) are
# **unusual objects** doing **normal physics**, so a good world model isn't
# surprised by them.
#
# Here we test the other half of that claim: anomalies that are **physically
# impossible** but where **every single frame looks normal**. We take real
# Ped2 walkway clips and change the last 4 frames of a 16-frame clip:
#
# | Anomaly | The last 4 frames |
# |---|---|
# | **teleport** | jump 24 frames (2.4 s) ahead: everyone is suddenly elsewhere |
# | **reverse** | frames 11, 10, 9, 8: time runs backwards |
# | **freeze** | frame 11, four times: everything stops dead |
# | **vanish** | the empty background: all people disappear at once |
#
# If the world-model story is right, **surprise should win here** and
# seen-before should be near chance.
#
# ![architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/07_physical_anomalies/assets/architecture-light.svg)

# %%
import copy
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
SIZE, PATCH, TUBELET = 256, 16, 2
T_CTX, T_PRED = 12, 4
T = T_CTX + T_PRED
GRID = SIZE // PATCH
JUMP = 24                                 # teleport distance in frames
KINDS = ["teleport", "reverse", "freeze", "vanish"]

PED2_URL = "https://huggingface.co/datasets/123zdm/ped2_datasets/resolve/19dcbe727b56f7dbf53a544e0b5a2ccbaaae14b6/ped2.tar.gz"
DATA = Path("data")
if not (DATA / "ped2").exists():
    DATA.mkdir(exist_ok=True)
    urllib.request.urlretrieve(PED2_URL, DATA / "ped2.tar.gz")
    with tarfile.open(DATA / "ped2.tar.gz") as tar:
        tar.extractall(DATA, filter="data")

videos = {d.name: np.stack([np.array(Image.open(f).convert("RGB")) for f in sorted(d.glob("*.jpg"))])
          for d in sorted((DATA / "ped2" / "training" / "frames").iterdir())}
names = list(videos)
REF, EVAL = names[:8], names[8:]          # 8 normal videos as reference, 8 to build test clips from
print(f"reference videos {REF}, evaluation videos {EVAL}")

# %% [markdown]
# ## Making the impossible clips
#
# Test clips come only from the 8 evaluation videos: one normal clip every
# 4 frames, plus one manipulated copy per anomaly type. The 8 reference
# videos give the "normal" statistics (calibration and the seen-before
# memory bank), so nothing is scored against itself.

# %%
backgrounds = {n: np.median(v, axis=0).astype(np.uint8) for n, v in videos.items()}   # empty walkway


def make_clip(name, s, kind):
    v = videos[name]
    clip = v[s:s + T].copy()
    if kind == "teleport":
        j = s + T_CTX + JUMP if s + T + JUMP <= len(v) else s + T_CTX - JUMP
        clip[T_CTX:] = v[j:j + T_PRED]
    elif kind == "reverse":
        clip[T_CTX:] = v[s + T_CTX - 1:s + T_CTX - 1 - T_PRED:-1]
    elif kind == "freeze":
        clip[T_CTX:] = v[s + T_CTX - 1]
    elif kind == "vanish":
        clip[T_CTX:] = backgrounds[name]
    return clip


starts = [(n, s) for n in EVAL for s in range(T_PRED, len(videos[n]) - T + 1, 4)]
print(f"{len(starts)} normal test clips, each with {len(KINDS)} impossible copies")

# %% [markdown]
# ## V-JEPA 2 (same setup as 06)

# %%
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)
N_CTX = (T_CTX // TUBELET) * GRID * GRID
N_ALL = (T // TUBELET) * GRID * GRID


def to_pixels(clip):
    x = torch.from_numpy(clip).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    return ((x.unsqueeze(0) - MEAN) / STD).half()


@torch.no_grad()
def encode(clip):
    """Context latents (past 12 frames only) and layer-normed latents of the last 4 frames."""
    px = to_pixels(clip)
    ctx = model(pixel_values_videos=px[:, :T_CTX], skip_predictor=True).last_hidden_state
    tgt = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state[:, N_CTX:]
    return ctx, F.layer_norm(tgt.float(), (tgt.size(-1),))


def token_l1(predictor, ctx, tgt):
    B = ctx.size(0)
    cm = [torch.arange(N_CTX, device=DEVICE).unsqueeze(0).expand(B, -1)]
    tm = [torch.arange(N_CTX, N_ALL, device=DEVICE).unsqueeze(0).expand(B, -1)]
    with torch.autocast("cuda", dtype=torch.float16):
        pred = predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
    return (pred.float() - tgt.float()).abs().mean(-1)                       # (B, 512)


def to_map(err):
    return err.view(T_PRED // TUBELET, GRID, GRID).mean(0)                    # 16 x 16


def unit(tgt):
    return F.normalize(tgt.view(T_PRED // TUBELET, GRID * GRID, -1), dim=-1).half()


# %% [markdown]
# ## Normal statistics from the reference videos
#
# - **Calibration**: how surprising each patch position usually is, on
#   normal reference clips (as in 06).
# - **Seen-before memory bank**: the latents of every reference tubelet, per
#   position (as in 06).

# %%
t0 = time.time()
ref_maps, bank, ref_cache = [], [], []
with torch.no_grad():
    for n in REF:
        for s in range(0, len(videos[n]) - T + 1, 4):
            ctx, tgt = encode(videos[n][s:s + T])
            ref_maps.append(to_map(token_l1(model.predictor, ctx, tgt)[0]))
            bank.append(unit(tgt))
            ref_cache.append((ctx[0], tgt[0].half()))
ref_maps, bank = torch.stack(ref_maps), torch.cat(bank)
MU, SD = ref_maps.mean(0), ref_maps.std(0) + 1e-6
print(f"{len(ref_maps)} reference clips in {time.time() - t0:.0f}s")


# %% [markdown]
# ## Score every test clip
#
# Per clip, on the last 4 frames:
# - **surprise**: V-JEPA 2 predictor error, whole-frame mean;
# - **surprise, calibrated top-5**: per-position z-score, mean of the 5 most
#   surprising patches (as in 06);
# - **seen-before**: distance to the closest reference latent at the same
#   position, whole-frame mean.
#
# Metric: AUC of each impossible type vs the normal clips.

# %%
@torch.no_grad()
def scores_for(clip, predictor):
    ctx, tgt = encode(clip)
    m = to_map(token_l1(predictor, ctx, tgt)[0])
    z = (m - MU) / SD
    d = torch.stack([1 - torch.einsum("pd,npd->np", u, bank).max(0).values.float() for u in unit(tgt)]).mean(0)
    return {"surprise (mean)": m.mean().item(), "surprise, calibrated (top-5)": z.flatten().topk(5).values.mean().item(),
            "seen-before (mean)": d.mean().item()}, m.cpu(), z.cpu()


def auc(neg, pos):
    neg, pos = np.asarray(neg), np.asarray(pos)
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def evaluate(predictor, tag):
    t0 = time.time()
    rows = []
    for n, s in starts:
        for kind in ["normal"] + KINDS:
            sc, _, _ = scores_for(make_clip(n, s, kind if kind != "normal" else None), predictor)
            rows.append({"kind": kind, **sc})
    out = {}
    for score in rows[0]:
        if score == "kind":
            continue
        normal = [r[score] for r in rows if r["kind"] == "normal"]
        out[score] = {k: auc(normal, [r[score] for r in rows if r["kind"] == k]) for k in KINDS}
    print(f"[{tag}] scored {len(rows)} clips in {time.time() - t0:.0f}s")
    for score, per in out.items():
        print(f"  {score:30s} " + "  ".join(f"{k} {v:.3f}" for k, v in per.items()))
    return out, rows


zero_shot, rows_zs = evaluate(model.predictor, "zero-shot")

# %% [markdown]
# ## Adapting the predictor (as in 06, Part 2)
#
# Fine-tune only the predictor on the reference videos' normal clips, then
# score again. In 06 this made prediction better but detection no better;
# here it should help, if surprise is the right signal for this kind of
# anomaly.

# %%
CTX_R = torch.stack([c for c, _ in ref_cache])
TGT_R = torch.stack([t for _, t in ref_cache])
hold = torch.arange(len(CTX_R)) % 5 == 0                                      # every 5th clip held out
adapted = copy.deepcopy(model.predictor).float()
opt = torch.optim.AdamW(adapted.parameters(), lr=1e-4, weight_decay=0.01)
scaler = torch.amp.GradScaler()


@torch.no_grad()
def hold_loss(pred):
    return torch.cat([token_l1(pred, c, t).mean(1) for c, t in zip(CTX_R[hold].split(8), TGT_R[hold].split(8))]).mean().item()


adapt_history = [(0, None, hold_loss(model.predictor))]
best, best_state = adapt_history[0][2], copy.deepcopy(adapted.state_dict())
fit_idx = torch.nonzero(~hold).flatten()
for epoch in range(1, 7):
    adapted.train()
    losses = []
    for b in fit_idx[torch.randperm(len(fit_idx))].split(4):
        loss = token_l1(adapted, CTX_R[b], TGT_R[b]).mean()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    adapted.eval()
    h = hold_loss(adapted)
    adapt_history.append((epoch, sum(losses) / len(losses), h))
    if h < best:
        best, best_state = h, copy.deepcopy(adapted.state_dict())
    print(f"epoch {epoch}  fit L1 {adapt_history[-1][1]:.4f}  held-out L1 {h:.4f}")
adapted.load_state_dict(best_state)
adapted.eval()
del opt, scaler
torch.cuda.empty_cache()

with torch.no_grad():                                                          # recalibrate on reference clips
    ref_maps_a = torch.stack([to_map(token_l1(adapted, c[None], t[None])[0]) for c, t in ref_cache])
MU, SD = ref_maps_a.mean(0), ref_maps_a.std(0) + 1e-6
adapted_res, rows_ad = evaluate(adapted, "adapted")

# %% [markdown]
# ## Check: seen-before without the clip's context
#
# Seen-before uses the encoder's latents of the last 4 frames, but the
# encoder computes them while looking at **all 16 frames**, so they already
# carry "the motion just broke". Here the encoder sees **only the last 4
# frames** (2 tubelets): appearance plus a 2-frame glimpse of motion.

# %%
@torch.no_grad()
def future_only(clip):
    z = model(pixel_values_videos=to_pixels(clip[T_CTX:]), skip_predictor=True).last_hidden_state
    return F.normalize(F.layer_norm(z.float(), (z.size(-1),)).view(T_PRED // TUBELET, GRID * GRID, -1), dim=-1).half()


bank_fo = torch.cat([future_only(videos[n][s:s + T]) for n in REF for s in range(0, len(videos[n]) - T + 1, 4)])
fo = {"normal": []} | {k: [] for k in KINDS}
for n, s in starts:
    for kind in fo:
        u = future_only(make_clip(n, s, kind if kind != "normal" else None))
        fo[kind].append(torch.stack([1 - torch.einsum("pd,npd->np", x, bank_fo).max(0).values.float() for x in u]).mean().item())
future_only_auc = {k: auc(fo["normal"], fo[k]) for k in KINDS}
print("seen-before, last 4 frames only  " + "  ".join(f"{k} {v:.3f}" for k, v in future_only_auc.items()))

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps({
    "reference_videos": REF, "evaluation_videos": EVAL, "n_normal_clips": len(starts), "jump_frames": JUMP,
    "zero_shot": zero_shot, "adapted": adapted_res, "adapt_history": adapt_history,
    "seen_before_future_only": future_only_auc}, indent=1))
(OUT / "clip_scores.json").write_text(json.dumps({"zero_shot": rows_zs, "adapted": rows_ad}))

# %%
n, s = starts[len(starts) // 3]
normal_scores = np.array([r["surprise (mean)"] for r in rows_ad if r["kind"] == "normal"])
fig, axes = plt.subplots(1 + len(KINDS), 4, figsize=(11, 2.2 * (1 + len(KINDS))))
for row, kind in enumerate(["normal"] + KINDS):
    clip = make_clip(n, s, kind if kind != "normal" else None)
    sc, m, z = scores_for(clip, adapted)
    pct = (normal_scores < sc["surprise (mean)"]).mean() * 100
    for col, t in enumerate([T_CTX - 2, T_CTX - 1, T_CTX]):
        axes[row, col].imshow(clip[t]); axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(f"frame {t}" + (" (predicted)" if t >= T_CTX else " (context)"), fontsize=8)
    heat = F.interpolate(z.clamp(min=0)[None, None], size=clip.shape[1:3], mode="bilinear")[0, 0]
    axes[row, 3].imshow(clip[T_CTX]); axes[row, 3].imshow(heat, cmap="inferno", alpha=.55, vmin=0, vmax=4)
    axes[row, 3].axis("off")
    axes[row, 3].set_title(f"surprise {sc['surprise (mean)']:.3f} (higher than {pct:.0f}% of normal clips)", fontsize=7)
    axes[row, 0].text(-10, clip.shape[1] / 2, kind, ha="right", va="center", fontsize=10)
fig.suptitle("Adapted V-JEPA 2: where it is surprised in the first predicted frame", fontsize=10)
plt.tight_layout(); plt.savefig(OUT / "clips.png", dpi=110); plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True, gridspec_kw={"width_ratios": [3, 3, 1.6]})
COLORS = {"surprise (mean)": "#1baf7a", "surprise, calibrated (top-5)": "#2a78d6", "seen-before (mean)": "#898781"}
for ax, (tag, res) in zip(axes, [("zero-shot", zero_shot), ("adapted predictor", adapted_res)]):
    for k, (score, per) in enumerate(res.items()):
        vals = [per[kind] for kind in KINDS]
        bars = ax.bar([x + (k - 1) * 0.27 for x in range(len(KINDS))], vals, 0.27, color=COLORS[score], label=score)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
    ax.axhline(0.5, color="#e34948", ls="--", lw=1)
    ax.set_xticks(range(len(KINDS)), KINDS); ax.set_ylim(0, 1.18); ax.set_title(tag)
axes[0].set_ylabel("AUC: impossible vs normal clips")
axes[0].legend(fontsize=7, loc="upper center", ncol=3)
vals = [future_only_auc[k] for k in KINDS]
bars = axes[2].bar(range(len(KINDS)), vals, 0.6, color="#c3c2b7")
for b, v in zip(bars, vals):
    axes[2].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6)
axes[2].axhline(0.5, color="#e34948", ls="--", lw=1)
axes[2].set_xticks(range(len(KINDS)), KINDS, fontsize=7); axes[2].set_title("seen-before,\nlast 4 frames only", fontsize=9)
plt.tight_layout(); plt.savefig(OUT / "auc.png", dpi=110); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), sharey=True)
KIND_COLORS = {"normal": "#2a78d6", "teleport": "#e34948", "reverse": "#eda100", "freeze": "#6250d6", "vanish": "#1baf7a"}
for ax, (tag, rows) in zip(axes, [("zero-shot", rows_zs), ("adapted predictor", rows_ad)]):
    allv = [r["surprise (mean)"] for r in rows]
    bins = np.linspace(min(allv), max(allv), 40)
    for kind, color in KIND_COLORS.items():
        ax.hist([r["surprise (mean)"] for r in rows if r["kind"] == kind], bins=bins, alpha=.55, color=color, label=kind)
    ax.set_xlabel("clip surprise (mean predictor error)"); ax.set_title(tag)
axes[0].set_ylabel("clips"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "scores.png", dpi=110); plt.show()
