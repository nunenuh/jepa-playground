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
# # 06 · A Mahalanobis baseline
#
# Part 1's seen-before baseline is a nearest-neighbour (k-NN) check: distance
# to the closest normal latent at the same position. The other classic
# label-free baseline is **Mahalanobis**: how far a latent is from the normal
# latents' Gaussian at its position (per-position mean, one shared
# covariance, shrunk towards the identity).
#
# To keep the setup identical, this runs `vjepa2_anomaly.py` up to the
# scoring step (data, model, the normal memory bank), then scores every test
# frame with the new detector next to seen-before.

# %%
import json
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path("06_vjepa2_video")
exec((HERE / "vjepa2_anomaly.py").read_text().split("# %% [markdown]\n# ## Score every test frame")[0])

# %%
B = bank.float()                                                  # (n, 256 positions, 1024), unit vectors
POS_MEAN = B.mean(0)                                              # (256, 1024)
centred = (B - POS_MEAN).flatten(0, 1)
cov = torch.cov(centred.T)
shrink = 0.1
INV = torch.linalg.inv((1 - shrink) * cov + shrink * cov.diagonal().mean() * torch.eye(cov.size(0), device=DEVICE))
del B, centred


@torch.no_grad()
def baseline_maps(clip):
    """seen-before (k-NN) and Mahalanobis maps for one window, each 16 x 16."""
    _, target = surprise_map(clip)
    u = unit(target)                                              # (2, 256, 1024)
    knn = torch.stack([1 - torch.einsum("pd,npd->np", zt, bank).max(0).values.float() for zt in u]).mean(0)
    d = u.float() - POS_MEAN
    maha = torch.einsum("tpd,de,tpe->tp", d, INV, d).mean(0)
    return knn.view(GRID, GRID), maha.view(GRID, GRID)


def auc(y, s):
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    pos = y.sum()
    return (ranks[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def frame_maps(frames):
    maps = torch.zeros(2, len(frames), GRID, GRID, device=DEVICE)
    count = torch.zeros(len(frames), device=DEVICE)
    for s in range(0, len(frames) - T + 1, STRIDE):
        for k, m in enumerate(baseline_maps(frames[s:s + T])):
            maps[k, s + T_CTX:s + T] += m
        count[s + T_CTX:s + T] += 1
    seen = torch.nonzero(count).flatten()
    maps = maps / count.clamp(min=1)[None, :, None, None]
    first, last = int(seen[0]), int(seen[-1])
    maps[:, :first], maps[:, last + 1:] = maps[:, first:first + 1], maps[:, last:last + 1]
    return [m.cpu() for m in maps]


names = ["seen-before baseline (k-NN)", "Mahalanobis"]
maps = {n: {} for n in names}
for video, frames in test_videos.items():
    for n, m in zip(names, frame_maps(frames)):
        maps[n][video] = m
y_all = np.concatenate([labels[v] for v in test_videos])
results = {}
for n in names:
    results[n] = {
        "AUC top-5 patches": auc(y_all, np.concatenate([maps[n][v].flatten(1).topk(5, dim=1).values.mean(1).numpy() for v in test_videos])),
        "AUC whole-frame mean": auc(y_all, np.concatenate([maps[n][v].flatten(1).mean(1).numpy() for v in test_videos])),
    }
    print(n, {k: round(float(v), 3) for k, v in results[n].items()})

OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results_baselines.json").write_text(json.dumps({"bank_tubelets": int(bank.shape[0]), "shrinkage": shrink,
                                                        "results": {k: {a: float(b) for a, b in v.items()} for k, v in results.items()}}, indent=1))
