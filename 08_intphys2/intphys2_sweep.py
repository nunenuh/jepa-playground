# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: A100
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 08 · Part 2: closing the gap to the IntPhys 2 paper
#
# Part 1 got V-JEPA 2 (ViT-L, one fixed setting) to **50.2%** pairwise
# accuracy; the IntPhys 2 paper reports **57.5%** for V-JEPA 2 ViT-huge, best
# of about a dozen settings. Here we follow the paper's protocol (zero-shot,
# the official intuitive-physics evaluation from
# [jepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics)):
#
# - 16-frame windows; the encoder sees the first *c* frames, the predictor
#   predicts the latents of the remaining 16 − *c* frames; L1 error per window
# - a sweep over frame rate × context length × scoring rule
# - the video's surprise is the **max** or the **mean** over its windows
#
# and run it for **ViT-huge and ViT-large**, to see how much of the gap is
# model size. We report the best setting (the paper's way, picked on the
# test pairs, so optimistic) and an honest held-out number (setting picked on
# one half of the scenes, scored on the other half).

# %%
import json
import os
import pickle
import time
from itertools import product
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoModel

DEVICE = "cuda"
MODELS = os.environ.get("MODELS", "vith,vitl").split(",")
SIZE, PATCH, TUBELET, T = 256, 16, 2, 16
GRID = SIZE // PATCH
FRAME_STEPS = [5, 10]                     # 60 fps videos -> 12 fps or 6 fps
CTX_LENS = [4, 8, 12]                     # frames of context; predict the other 16 - c
STRIDE = 4                                # window stride, in sampled frames
TOPK = 0.1                                # "top-k" score: mean error of the 10% worst-predicted patches
BATCH = 8
REVISION = "a077a2f94e25889016fc6e5983cf21e2ddc25fb2"
WORK = Path("data")

root = Path(snapshot_download("facebook/IntPhys2", repo_type="dataset", revision=REVISION,
                              allow_patterns=["Main/*"], local_dir=WORK / "IntPhys2"))
meta = pd.read_csv(root / "Main" / "metadata.csv")
meta["pair"] = meta["type"].str[0].astype(int)
meta["possible"] = meta["type"].str.endswith("_Possible")
meta["half"] = pd.factorize(meta.SceneIndex)[0] % 2
if os.environ.get("SMOKE"):                 # quick test on a few scenes
    meta = meta[meta.SceneIndex.isin(meta.SceneIndex.unique()[:int(os.environ["SMOKE"])])]
print(len(meta), "videos,", meta.SceneIndex.nunique(), "scenes")


# %% [markdown]
# ## Decoding (cached per frame rate)

# %%
def frames_of(name, step):
    cache = WORK / f"frames_{step}"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{name}.npy"
    if not f.exists():
        cap, out, i = cv2.VideoCapture(str(root / "Main" / "Videos" / f"{name}.mp4")), [], 0
        while True:
            if i % step == 0:
                ok, img = cap.read()
                if not ok:
                    break
                out.append(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA))
            elif not cap.grab():
                break
            i += 1
        np.save(f, np.stack(out))
    return np.load(f)


# %% [markdown]
# ## Surprise for every window, context length and scoring rule
#
# One full-clip encoding gives the targets (layer-normed, as in the official
# code); each context length needs its own context encoding and prediction.

# %%
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)
TOK = GRID * GRID                         # tokens per tubelet (2 frames)


@torch.no_grad()
def window_scores(model, frames):
    """{(ctx, 'mean'|'topk'): (W,) array} for all 16-frame windows of one video."""
    starts = range(0, len(frames) - T + 1, STRIDE)
    out = {(c, r): [] for c in CTX_LENS for r in ("mean", "topk")}
    for b in range(0, len(starts), BATCH):
        x = torch.from_numpy(np.stack([frames[s:s + T] for s in list(starts)[b:b + BATCH]])).to(DEVICE)
        px = ((x.permute(0, 1, 4, 2, 3).float() / 255 - MEAN) / STD).half()
        B = len(px)
        h = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state
        h = F.layer_norm(h.float(), (h.size(-1),))
        for c in CTX_LENS:
            n_ctx, n_all = c // TUBELET * TOK, T // TUBELET * TOK
            ctx = model(pixel_values_videos=px[:, :c], skip_predictor=True).last_hidden_state
            cm = [torch.arange(n_ctx, device=DEVICE).expand(B, -1)]
            tm = [torch.arange(n_ctx, n_all, device=DEVICE).expand(B, -1)]
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model.predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
            err = (pred.float() - h[:, n_ctx:]).abs().mean(-1)                   # (B, target tokens)
            out[(c, "mean")].append(err.mean(1).cpu())
            out[(c, "topk")].append(err.topk(int(TOPK * err.size(1)), dim=1).values.mean(1).cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def run(model_key):
    """Scores every video for every frame step; resumable (Colab can disconnect)."""
    path = WORK / f"curves_{model_key}.pkl"
    curves = pickle.loads(path.read_bytes()) if path.exists() else {}
    todo = [(s, n) for s in FRAME_STEPS for n in meta.name if (s, n) not in curves]
    if not todo:
        return curves
    model = AutoModel.from_pretrained(f"facebook/vjepa2-{model_key}-fpc64-256", dtype=torch.float16).to(DEVICE).eval()
    t0 = time.time()
    for k, (step, name) in enumerate(todo, 1):
        curves[(step, name)] = window_scores(model, frames_of(name, step))
        if k % 100 == 0 or k == len(todo):
            path.write_bytes(pickle.dumps(curves))
            eta = (time.time() - t0) / k * (len(todo) - k)
            print(f"  {model_key}: {k}/{len(todo)}  ({time.time() - t0:.0f}s, ~{eta / 60:.0f} min left)", flush=True)
    del model
    torch.cuda.empty_cache()
    return curves


all_curves = {}
for m in MODELS:
    print("scoring with", m)
    all_curves[m] = run(m)

# %% [markdown]
# ## Pairwise accuracy for every setting
#
# A setting = frame step × context length × patch scoring (whole-frame mean
# or top-10% patches) × video aggregation (max or mean over windows): 24.

# %%
SETTINGS = list(product(FRAME_STEPS, CTX_LENS, ("mean", "topk"), ("max", "mean")))
pair_keys = ["SceneIndex", "pair"]


def correct_table(curves):
    """DataFrame: one row per pair, one bool column per setting (impossible more surprising?)."""
    rows = []
    for (scene, pair), g in meta.groupby(pair_keys):
        pos, imp = g[g.possible].iloc[0], g[~g.possible].iloc[0]
        row = {"SceneIndex": scene, "pair": pair, "condition": pos.condition, "Difficulty": pos.Difficulty,
               "Camera": pos.Camera, "half": pos.half}
        for s in SETTINGS:
            step, c, region, agg = s
            f = np.max if agg == "max" else np.mean
            row[s] = f(curves[(step, imp["name"])][(c, region)]) > f(curves[(step, pos["name"])][(c, region)])
        rows.append(row)
    return pd.DataFrame(rows)


def name(s):
    return f"{60 // s[0]}fps ctx{s[1]} {s[2]} {s[3]}"


summary = {}
tables = {}
for m in MODELS:
    tab = tables[m] = correct_table(all_curves[m])
    accs = {s: tab[s].mean() for s in SETTINGS}
    best = max(accs, key=accs.get)
    honest = []                            # pick the setting on one half of the scenes, score the other
    for h in (0, 1):
        fit, test = tab[tab.half != h], tab[tab.half == h]
        pick = max(SETTINGS, key=lambda s: fit[s].mean())
        honest.append(test[pick])
    honest = pd.concat(honest)
    summary[m] = {
        "best_setting": name(best), "best": float(accs[best]),
        "best_by_difficulty": {d: float(g[best].mean()) for d, g in tab.groupby("Difficulty")},
        "best_by_condition": {d: float(g[best].mean()) for d, g in tab.groupby("condition")},
        "held_out": float(honest.mean()),
        "part1_setting": float(accs.get((10, 12, "mean", "max"), np.nan)),
        "all_settings": {name(s): float(a) for s, a in accs.items()},
        "pairs": int(len(tab)),
    }
    print(f"{m}: best {accs[best]:.1%} ({name(best)}), held-out {honest.mean():.1%}, "
          f"Part 1 setting {summary[m]['part1_setting']:.1%}")

PAPER = {"best": 0.5751, "held_out_set": 0.564, "Easy": 0.54, "Medium": 0.585, "Hard": 0.5938}

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results_sweep.json").write_text(json.dumps({
    "dataset_revision": REVISION, "frame_steps": FRAME_STEPS, "context_lengths": CTX_LENS, "stride": STRIDE,
    "topk_fraction": TOPK, "results": summary, "paper_vjepa2_huge": PAPER}, indent=1))

# %%
fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 3.8), squeeze=False)
cols = [(r, a) for r in ("mean", "topk") for a in ("max", "mean")]
rows = [(s, c) for s in FRAME_STEPS for c in CTX_LENS]
for ax, m in zip(axes[0], MODELS):
    grid = np.array([[tables[m][(s, c, r, a)].mean() for r, a in cols] for s, c in rows])
    im = ax.imshow(grid, cmap="RdBu_r", vmin=0.4, vmax=0.6)
    for i, j in np.ndindex(grid.shape):
        ax.text(j, i, f"{grid[i, j]:.1%}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), [f"{r} patches\n{a} over time" for r, a in cols], fontsize=7)
    ax.set_yticks(range(len(rows)), [f"{60 // s} fps, ctx {c}" for s, c in rows], fontsize=8)
    ax.set_title(f"V-JEPA 2 {m}: pairwise accuracy per setting", fontsize=9)
fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="accuracy (50% = chance)")
plt.savefig(OUT / "sweep.png", dpi=110, bbox_inches="tight"); plt.show()

# %%
bars = [("paper ViT-H\nbest", PAPER["best"], "#bbbbbb"), ("paper ViT-H\nheld-out set", PAPER["held_out_set"], "#dddddd")]
for m in MODELS:
    bars += [(f"ours {m}\nPart 1 setting", summary[m]["part1_setting"], "#9fc5f0"),
             (f"ours {m}\nbest", summary[m]["best"], "#1baf7a"),
             (f"ours {m}\nheld-out", summary[m]["held_out"], "#0b7a52")]
fig, ax = plt.subplots(figsize=(1.5 * len(bars) + 1, 3.6))
for i, (label, v, color) in enumerate(bars):
    ax.bar(i, v, color=color)
    ax.text(i, v + 0.005, f"{v:.1%}", ha="center", fontsize=8)
ax.axhline(0.5, color="#e34948", ls="--", lw=1, label="chance")
ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7)
ax.set_ylim(0.4, 0.65); ax.set_ylabel("pairwise accuracy"); ax.legend(fontsize=8)
ax.set_title("IntPhys 2 main set: ours vs the paper (V-JEPA 2, zero-shot)")
plt.tight_layout(); plt.savefig(OUT / "vs_paper.png", dpi=110); plt.show()
