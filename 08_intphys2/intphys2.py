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
# # 08 · V-JEPA 2 on IntPhys 2: real intuitive-physics tests
#
# In 07, V-JEPA 2's surprise caught physically impossible motion that we
# made ourselves by editing walkway video, once its predictor was adapted to
# the camera. Here we use a benchmark built by Meta to test exactly this:
# **[IntPhys 2](https://huggingface.co/datasets/facebook/IntPhys2)**.
#
# Every scene comes as matched pairs of videos, identical except that in one
# of them physics breaks:
#
# | Condition | What breaks |
# |---|---|
# | **permanence** | an object vanishes or appears from nothing |
# | **immutability** | an object changes shape or colour |
# | **continuity** | an object teleports instead of moving through space |
# | **solidity** | an object passes through another solid object |
#
# The paper reports that most models are near chance (50%), including
# V-JEPA 2 (ViT-huge: 57.5% overall), while humans get ~96%.
#
# Question: does our recipe from 06/07 (**adapting the predictor** to the
# domain, on *possible* videos only) move V-JEPA 2 off chance?
#
# ![architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/08_intphys2/assets/architecture-light.svg)

# %%
import copy
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
SIZE, PATCH, TUBELET = 256, 16, 2
T_CTX, T_PRED = 12, 4                    # 12 frames of context -> predict the next 4
T = T_CTX + T_PRED
GRID = SIZE // PATCH
FRAME_STEP = 10                           # 60 fps videos -> 6 fps, as in the official evaluation
STRIDE = 4                                # every predicted frame is covered by exactly one window
REVISION = "a077a2f94e25889016fc6e5983cf21e2ddc25fb2"   # pinned dataset version

root = Path(snapshot_download("facebook/IntPhys2", repo_type="dataset", revision=REVISION,
                              allow_patterns=["Main/*"], local_dir="data/IntPhys2"))
meta = pd.read_csv(root / "Main" / "metadata.csv")
meta["pair"] = meta["type"].str[0].astype(int)                       # 1 or 2: which matched pair
meta["possible"] = meta["type"].str.endswith("_Possible")
print(len(meta), "videos,", meta.SceneIndex.nunique(), "scenes")
print(meta.groupby("condition").size().to_dict())

# %% [markdown]
# ## Decoding
#
# Every 10th frame (6 fps), resized to 256 x 256, cached as `uint8` so each
# video is decoded only once.

# %%
CACHE = Path("data/frames")
CACHE.mkdir(parents=True, exist_ok=True)


def frames_of(name):
    f = CACHE / f"{name}.npy"
    if not f.exists():
        cap, out, i = cv2.VideoCapture(str(root / "Main" / "Videos" / f"{name}.mp4")), [], 0
        while True:
            if i % FRAME_STEP == 0:
                ok, img = cap.read()
                if not ok:
                    break
                out.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            elif not cap.grab():
                break
            i += 1
        np.save(f, np.stack(out))
    return np.load(f)


t0 = time.time()
lengths = {n: len(frames_of(n)) for n in meta.name}
print(f"decoded {len(lengths)} videos in {time.time() - t0:.0f}s; frames per video: "
      f"{min(lengths.values())}–{max(lengths.values())}")

# %% [markdown]
# ## V-JEPA 2 surprise (same setup as 06/07)

# %%
model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float16).to(DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)
N_CTX = (T_CTX // TUBELET) * GRID * GRID
N_ALL = (T // TUBELET) * GRID * GRID


def windows(frames):
    """All 16-frame windows of a video, (W, 16, 3, 256, 256) normalised half."""
    starts = range(0, len(frames) - T + 1, STRIDE)
    x = torch.from_numpy(np.stack([frames[s:s + T] for s in starts])).to(DEVICE)
    x = x.permute(0, 1, 4, 2, 3).float() / 255
    return ((x - MEAN) / STD).half(), list(starts)


@torch.no_grad()
def encode(px):
    """Context latents (first 12 frames only) and layer-normed latents of the last 4 frames."""
    ctx = model(pixel_values_videos=px[:, :T_CTX], skip_predictor=True).last_hidden_state
    tgt = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state[:, N_CTX:]
    return ctx, F.layer_norm(tgt.float(), (tgt.size(-1),)).half()


def surprise(predictor, ctx, tgt):
    """(B,) mean L1 error of the predicted latents of the last 4 frames."""
    B = ctx.size(0)
    cm = [torch.arange(N_CTX, device=DEVICE).unsqueeze(0).expand(B, -1)]
    tm = [torch.arange(N_CTX, N_ALL, device=DEVICE).unsqueeze(0).expand(B, -1)]
    with torch.autocast("cuda", dtype=torch.float16):
        pred = predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
    return (pred.float() - tgt.float()).abs().mean((1, 2))


# %% [markdown]
# ## Adapting the predictor without seeing the test scenes
#
# Scenes are split in two halves (alternating scenes). For each half
# we fine-tune a copy of the predictor on the **possible** videos of the
# **other** half only, then use it to score this half. So no scene is ever
# scored by a predictor that saw it, and no impossible video is ever used
# for training.

# %%
def adapt(train_names, epochs=4, per_video=3, seed=0):
    torch.manual_seed(seed)
    ctx_all, tgt_all = [], []
    for n in train_names:                                             # a few windows per possible video
        px, _ = windows(frames_of(n))
        pick = torch.linspace(0, len(px) - 1, per_video).round().long()
        c, t = encode(px[pick])
        ctx_all.append(c)                                               # kept on the GPU: Colab RAM is only 12 GB
        tgt_all.append(t)
    ctx_all, tgt_all = torch.cat(ctx_all), torch.cat(tgt_all)
    hold = torch.arange(len(ctx_all), device=DEVICE) % 10 == 0
    pred = copy.deepcopy(model.predictor).float()
    opt = torch.optim.AdamW(pred.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler()

    def hold_loss(p):
        with torch.no_grad():
            return torch.cat([surprise(p, c.to(DEVICE), t.to(DEVICE)) for c, t in
                              zip(ctx_all[hold].split(8), tgt_all[hold].split(8))]).mean().item()

    history = [(0, None, hold_loss(model.predictor))]
    best, best_state = history[0][2], copy.deepcopy(pred.state_dict())
    fit = torch.nonzero(~hold).flatten()
    for epoch in range(1, epochs + 1):
        pred.train()
        losses = []
        for b in fit[torch.randperm(len(fit))].split(4):
            loss = surprise(pred, ctx_all[b].to(DEVICE), tgt_all[b].to(DEVICE)).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        pred.eval()
        h = hold_loss(pred)
        history.append((epoch, float(np.mean(losses)), h))
        if h < best:
            best, best_state = h, copy.deepcopy(pred.state_dict())
        print(f"  epoch {epoch}  fit L1 {history[-1][1]:.4f}  held-out L1 {h:.4f}")
    pred.load_state_dict(best_state)
    del opt, scaler, ctx_all, tgt_all
    torch.cuda.empty_cache()
    return pred.eval(), history


meta["half"] = pd.factorize(meta.SceneIndex)[0] % 2                  # scene ids look like "12" or "154_0"
adapted, adapt_histories = {}, {}
t0 = time.time()
for half in (0, 1):
    train_names = meta[(meta.half != half) & meta.possible].name.tolist()
    print(f"half {half}: adapting on {len(train_names)} possible videos from the other half")
    adapted[half], adapt_histories[half] = adapt(train_names)
print(f"adapted both predictors in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Score every video
#
# For each video: the surprise of every window, zero-shot and with the
# predictor adapted on the other half. A video's score is its **peak**
# surprise over time, as in the official evaluation.

# %%
t0 = time.time()
curves = {}
for k, row in enumerate(meta.itertuples()):
    px, starts = windows(frames_of(row.name))
    zs, ad = [], []
    for chunk in px.split(7):
        ctx, tgt = encode(chunk)
        with torch.no_grad():
            zs.append(surprise(model.predictor, ctx, tgt))
            ad.append(surprise(adapted[row.half], ctx, tgt))
    curves[row.name] = {"zero-shot": torch.cat(zs).cpu().numpy(), "adapted": torch.cat(ad).cpu().numpy(),
                        "starts": starts}
    if k % 100 == 0:
        print(f"  {k}/{len(meta)} videos ({time.time() - t0:.0f}s)")
print(f"scored {len(meta)} videos in {time.time() - t0:.0f}s")

# %% [markdown]
# ## Accuracy
#
# **Pairwise accuracy**: for every matched pair, is the impossible video's
# peak surprise higher than the possible one's? 50% = chance.

# %%
for method in ["zero-shot", "adapted"]:
    meta[method] = [curves[n][method].max() for n in meta.name]

pairs = meta.pivot_table(index=["SceneIndex", "pair", "condition", "Difficulty", "Camera"],
                         columns="possible", values=["zero-shot", "adapted"]).reset_index()
for method in ["zero-shot", "adapted"]:
    pairs[f"{method} correct"] = pairs[(method, False)] > pairs[(method, True)]
pairs.columns = [" ".join(str(c) for c in col if c != "").strip() if isinstance(col, tuple) else col for col in pairs.columns]


def acc(df):
    return {m: float(df[f"{m} correct"].mean()) for m in ["zero-shot", "adapted"]} | {"pairs": int(len(df))}


results = {"overall": acc(pairs)}
for col in ["condition", "Difficulty", "Camera"]:
    for v, g in pairs.groupby(col):
        results[f"{col}={v}"] = acc(g)
for k, r in results.items():
    print(f"{k:24s} zero-shot {r['zero-shot']:.1%}   adapted {r['adapted']:.1%}   ({r['pairs']} pairs)")

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results.json").write_text(json.dumps({
    "dataset_revision": REVISION, "frame_step": FRAME_STEP, "window": [T_CTX, T_PRED], "stride": STRIDE,
    "accuracy": results, "adapt_history": {str(h): v for h, v in adapt_histories.items()},
    "paper_vjepa2_huge": {"Easy": 0.54, "Medium": 0.585, "Hard": 0.5938, "Overall": 0.5751}}, indent=1))
meta[["name", "SceneIndex", "pair", "type", "condition", "Difficulty", "Camera", "zero-shot", "adapted"]].to_csv(
    OUT / "video_scores.csv", index=False)

# %%
groups = ["overall"] + [k for k in results if k.startswith("condition=")] + [k for k in results if k.startswith("Difficulty=")]
fig, ax = plt.subplots(figsize=(11, 3.6))
for k, (method, color) in enumerate([("zero-shot", "#9fc5f0"), ("adapted", "#1baf7a")]):
    vals = [results[g][method] for g in groups]
    bars = ax.bar([x + (k - 0.5) * 0.38 for x in range(len(groups))], vals, 0.38, color=color, label=method)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=7)
ax.axhline(0.5, color="#e34948", ls="--", lw=1, label="chance")
ax.set_xticks(range(len(groups)), [g.split("=")[-1] + f"\n({results[g]['pairs']})" for g in groups], fontsize=8)
ax.set_ylim(0, 1.05); ax.set_ylabel("pairwise accuracy"); ax.legend(fontsize=8)
ax.set_title("IntPhys 2 (main set): is the impossible video more surprising than its possible twin?")
plt.tight_layout(); plt.savefig(OUT / "accuracy.png", dpi=110); plt.show()

# %%
examples = []
for cond in ["permanence", "immutability", "continuity", "solidity"]:
    g = pairs[(pairs.condition == cond)]
    hit = (g[g["adapted correct"]] if g["adapted correct"].any() else g).iloc[0]
    examples.append(meta[(meta.SceneIndex == hit.SceneIndex) & (meta.pair == hit.pair)])
fig, axes = plt.subplots(2, 4, figsize=(15, 5.4))
for col, ex in enumerate(examples):
    pos, imp = ex[ex.possible].iloc[0], ex[~ex.possible].iloc[0]
    ax = axes[0, col]
    for r, color, label in [(pos, "#2a78d6", "possible"), (imp, "#e34948", "impossible")]:
        c = curves[r["name"]]
        ax.plot([(s + T_CTX) / 6 for s in c["starts"]], c["adapted"], color=color, label=label)
    ax.set_title(f"{pos.condition} ({pos.Difficulty})", fontsize=9); ax.set_xlabel("seconds"); ax.grid(alpha=.3)
    if col == 0:
        ax.set_ylabel("surprise (adapted)"); ax.legend(fontsize=7)
    fi = frames_of(imp["name"])
    c = curves[imp["name"]]
    peak = c["starts"][int(np.argmax(c["adapted"]))] + T_CTX
    axes[1, col].imshow(np.concatenate([fi[peak - 2], fi[min(peak + 1, len(fi) - 1)]], axis=1))
    axes[1, col].set_title(f"impossible video around its surprise peak ({peak / 6:.1f} s)", fontsize=8)
    axes[1, col].axis("off")
plt.tight_layout(); plt.savefig(OUT / "examples.png", dpi=110); plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 3.2))
for half, hist in adapt_histories.items():
    ep, fit_l, hold_l = zip(*hist)
    ax.plot(ep, hold_l, marker="o", label=f"half {half}: held-out windows")
ax.set_xlabel("epoch (0 = zero-shot)"); ax.set_ylabel("prediction L1"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("Adapting the predictor on possible videos")
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=110); plt.show()
