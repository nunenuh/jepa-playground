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
# # 08 · Part 3: the official IntPhys 2 protocol
#
# Part 2 swept 16-frame windows (at most 2 s of context) and stayed below the
# paper. The IntPhys 2 authors' own evaluation
# ([`prediction_evals/evals/intphys2`](https://github.com/facebookresearch/IntPhys2/tree/main/prediction_evals),
# config `vjepa_2.yaml`) uses much longer memory:
#
# | | Part 2 | Official (this part) |
# |---|---|---|
# | window | 16 frames | **48 frames** (8 s at 6 fps) |
# | context | 4–12 frames | **12–42 frames**, predict the rest of the window |
# | stride | 4 | 2 |
# | start of video | not scored before the first full window | **max-context mode**: the first window is also scored with every shorter context (2, 4, ...), so an early break isn't missed |
#
# A video's surprise curve = [first-window losses for short contexts] +
# [losses of every sliding window]; its score is the max (or mean) of that
# curve. Pairwise accuracy as before.

# %%
import json
import os
import pickle
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

DEVICE = "cuda"
MODEL_KEY = os.environ.get("MODEL", "vith")
SIZE, PATCH, TUBELET = 256, 16, 2
TOK = (SIZE // PATCH) ** 2                # tokens per tubelet (2 frames)
FPC = 48                                  # frames per window
CTX_LENS = [12, 18, 24, 30, 36, 42]
STRIDE = 2
FRAME_STEP = 10                           # 60 fps -> 6 fps
BATCH = 4
REVISION = "a077a2f94e25889016fc6e5983cf21e2ddc25fb2"
WORK = Path("data")

root = Path(snapshot_download("facebook/IntPhys2", repo_type="dataset", revision=REVISION,
                              allow_patterns=["Main/*"], local_dir=WORK / "IntPhys2"))
meta = pd.read_csv(root / "Main" / "metadata.csv")
meta["pair"] = meta["type"].str[0].astype(int)
meta["possible"] = meta["type"].str.endswith("_Possible")
meta["half"] = pd.factorize(meta.SceneIndex)[0] % 2
if os.environ.get("SMOKE"):
    meta = meta[meta.SceneIndex.isin(meta.SceneIndex.unique()[:int(os.environ["SMOKE"])])]
print(len(meta), "videos,", meta.SceneIndex.nunique(), "scenes")


def frames_of(name):
    cache = WORK / f"frames_{FRAME_STEP}"
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"{name}.npy"
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


# %% [markdown]
# ## Surprise curves
#
# For a window and a context length *c*: encode the first *c* frames, predict
# the latents of frames *c*..48, compare with the layer-normed latents of the
# full window (mean L1 over tokens and channels), as in the official wrapper.

# %%
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 1, 3, 1, 1)


@torch.no_grad()
def losses(model, px, h, c):
    """(B,) mean L1 of predicting frames c..FPC from frames 0..c."""
    B, n_ctx, n_all = len(px), c // TUBELET * TOK, FPC // TUBELET * TOK
    ctx = model(pixel_values_videos=px[:, :c], skip_predictor=True).last_hidden_state
    cm = [torch.arange(n_ctx, device=DEVICE).expand(B, -1)]
    tm = [torch.arange(n_ctx, n_all, device=DEVICE).expand(B, -1)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = model.predictor(encoder_hidden_states=ctx, context_mask=cm, target_mask=tm).last_hidden_state
    return (pred.float() - h[:, n_ctx:]).abs().mean((1, 2))


@torch.no_grad()
def video_curves(model, frames):
    """{c: surprise curve} with the official max-context mode."""
    x = torch.from_numpy(frames).to(DEVICE).permute(0, 3, 1, 2).float() / 255
    x = ((x.unsqueeze(0) - MEAN) / STD).to(torch.bfloat16)[0]           # (T, 3, H, W)
    starts = list(range(0, len(x) - FPC + 1, STRIDE))
    first = x[:FPC].unsqueeze(0)
    h_first = model(pixel_values_videos=first, skip_predictor=True).last_hidden_state
    h_first = F.layer_norm(h_first.float(), (h_first.size(-1),))
    begin = {c: losses(model, first, h_first, c).item() for c in range(2, max(CTX_LENS) - 1, 2)}
    out = {c: [] for c in CTX_LENS}
    for b in range(0, len(starts), BATCH):
        px = torch.stack([x[s:s + FPC] for s in starts[b:b + BATCH]])
        h = model(pixel_values_videos=px, skip_predictor=True).last_hidden_state
        h = F.layer_norm(h.float(), (h.size(-1),))
        for c in CTX_LENS:
            out[c].append(losses(model, px, h, c).cpu())
    return {c: np.concatenate([[begin[k] for k in range(2, c, 2)], torch.cat(out[c]).numpy()]) for c in CTX_LENS}


path = WORK / f"official_{MODEL_KEY}.pkl"
curves = pickle.loads(path.read_bytes()) if path.exists() else {}
todo = [n for n in meta.name if n not in curves]
if todo:
    model = AutoModel.from_pretrained(f"facebook/vjepa2-{MODEL_KEY}-fpc64-256", dtype=torch.bfloat16).to(DEVICE).eval()
    t0 = time.time()
    for k, name in enumerate(todo, 1):
        curves[name] = video_curves(model, frames_of(name))
        if k % 50 == 0 or k == len(todo):
            path.write_bytes(pickle.dumps(curves))
            print(f"  {MODEL_KEY}: {k}/{len(todo)}  ({time.time() - t0:.0f}s, "
                  f"~{(time.time() - t0) / k * (len(todo) - k) / 60:.0f} min left)", flush=True)
    del model
    torch.cuda.empty_cache()

# %% [markdown]
# ## Pairwise accuracy per context length and aggregation

# %%
SETTINGS = [(c, agg) for c in CTX_LENS for agg in ("max", "mean")]
rows = []
for (scene, pair), g in meta.groupby(["SceneIndex", "pair"]):
    pos, imp = g[g.possible].iloc[0], g[~g.possible].iloc[0]
    row = {"condition": pos.condition, "Difficulty": pos.Difficulty, "half": pos.half}
    for c, agg in SETTINGS:
        f = np.max if agg == "max" else np.mean
        row[(c, agg)] = f(curves[imp["name"]][c]) > f(curves[pos["name"]][c])
    rows.append(row)
tab = pd.DataFrame(rows)
accs = {s: float(tab[s].mean()) for s in SETTINGS}
best = max(accs, key=accs.get)
held = []
for h in (0, 1):
    fit, test = tab[tab.half != h], tab[tab.half == h]
    held.append(test[max(SETTINGS, key=lambda s: fit[s].mean())])
held = pd.concat(held)
summary = {
    "model": MODEL_KEY, "best_setting": f"ctx{best[0]} {best[1]}", "best": accs[best],
    "best_by_difficulty": {d: float(g[best].mean()) for d, g in tab.groupby("Difficulty")},
    "best_by_condition": {d: float(g[best].mean()) for d, g in tab.groupby("condition")},
    "held_out": float(held.mean()), "all_settings": {f"ctx{c} {a}": v for (c, a), v in accs.items()},
    "pairs": int(len(tab)),
}
for s, a in accs.items():
    print(f"ctx {s[0]:2d} {s[1]:4s} {a:.1%}")
print(f"{MODEL_KEY}: best {accs[best]:.1%} (ctx {best[0]}, {best[1]}), held-out {held.mean():.1%}")

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
(OUT / f"results_official_{MODEL_KEY}.json").write_text(json.dumps({
    "dataset_revision": REVISION, "frames_per_clip": FPC, "context_lengths": CTX_LENS, "stride": STRIDE,
    "frame_step": FRAME_STEP, "max_context_mode": True, "results": summary,
    "paper_vjepa2_huge": {"best": 0.5751, "held_out_set": 0.564}}, indent=1))

fig, ax = plt.subplots(figsize=(7, 3.2))
for agg, color in [("max", "#1baf7a"), ("mean", "#2a78d6")]:
    ax.plot(CTX_LENS, [accs[(c, agg)] for c in CTX_LENS], marker="o", color=color, label=f"{agg} over time")
ax.axhline(0.5, color="#e34948", ls="--", lw=1, label="chance")
ax.axhline(0.5751, color="#888888", ls=":", lw=1, label="paper ViT-H (best)")
ax.set_xlabel("context frames (of a 48-frame window, 6 fps)"); ax.set_ylabel("pairwise accuracy")
ax.set_title(f"Official protocol, V-JEPA 2 {MODEL_KEY}"); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / f"official_{MODEL_KEY}.png", dpi=110); plt.show()
