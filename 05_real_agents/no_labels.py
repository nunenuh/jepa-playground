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
# # 05 · Q9: catching hijacks with zero labels
#
# The main 05 notebook needs labels: a probe trained on 50–1,000 labelled
# runs. A security team often has **only logs**, no labels. How far do
# label-free scores get on the same fair split (unseen tasks and unseen
# attacker goals)?
#
# Every score rates each **tool call**; a run's score is its most anomalous
# call. Fit only on **benign training runs** (normal behaviour) or on all
# unlabelled training runs, never on labels.
#
# | Score | Idea |
# |---|---|
# | seen-before (raw) | 1 − cosine to the closest benign call (05's baseline) |
# | k-NN (JEPA) | distance to the closest benign call in JEPA latent space |
# | Mahalanobis (raw / JEPA) | how far from the benign calls' Gaussian |
# | surprise (JEPA) | the predictor's error on the call, given the run so far |
# | **JEPA-SCORE** | the density a JEPA learns implicitly ([Balestriero et al., 2025](https://huggingface.co/papers/2510.05949)): log-density = Σ log singular values of the encoder's Jacobian; low = unusual |
#
# JEPA-SCORE assumes Gaussian embeddings, which is exactly what SIGReg
# trains for. Each JEPA score is computed for two encoders: pretrained on
# **all** unlabelled training runs (as in 05) and on **benign** runs only.

# %%
import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jacrev, vmap

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM, MAXSTEPS = 128, 15
MAXTOK = 1 + 2 * MAXSTEPS
STEPS, BATCH, SIGREG_WEIGHT = 3000, 128, 0.002          # as in 05
SEEDS = 3

DOJO_COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
DOJO = Path("data/agentdojo")
if not (DOJO / "runs").exists():
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse",
                    "https://github.com/ethz-spylab/agentdojo", str(DOJO)], check=True)
    subprocess.run(["git", "-C", str(DOJO), "sparse-checkout", "set", "runs"], check=True)
    subprocess.run(["git", "-C", str(DOJO), "checkout", "-q", DOJO_COMMIT], check=True)

# %% [markdown]
# ## Data, split and features: identical to 05

# %%
def load_runs(root, attacks=("none", "important_instructions")):
    rows = []
    for f in sorted(root.glob("runs/*/*/*/*/*.json")):
        model, suite, user_task, attack = f.parts[-5:-1]
        if attack not in attacks:
            continue
        d = json.loads(f.read_text())
        if d.get("error") or not d.get("messages"):
            continue
        steps = []
        for m in d["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    steps.append({"call": f"CALL {tc['function']}({json.dumps(tc.get('args', {}))})", "result": ""})
            elif m["role"] == "tool" and steps and not steps[-1]["result"]:
                c = m.get("content")
                steps[-1]["result"] = "RESULT " + (c if isinstance(c, str) else json.dumps(c))
        if not steps:
            continue
        task = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
        kind = "benign" if attack == "none" else ("hijacked" if d.get("security") else "resisted")
        rows.append(dict(model=model, suite=suite, user_task=user_task, injection_task=d.get("injection_task_id"),
                         kind=kind, task=task if isinstance(task, str) else json.dumps(task), steps=steps))
    return rows


def split_runs(rows, seed=0, val_frac=0.15):
    g = torch.Generator().manual_seed(seed)
    pick = lambda items, frac: set(items[i] for i in torch.randperm(len(items), generator=g)[:max(1, round(frac * len(items)))].tolist())
    out = {"train": [], "val": [], "test": []}
    for suite in sorted({r["suite"] for r in rows}):
        tasks = sorted({r["user_task"] for r in rows if r["suite"] == suite})
        goals = sorted({r["injection_task"] for r in rows if r["suite"] == suite and r["injection_task"]})
        test_tasks, test_goals = pick(tasks, 0.3), pick(goals, 0.3)
        val_tasks = pick([t for t in tasks if t not in test_tasks], val_frac)
        for r in (r for r in rows if r["suite"] == suite):
            t_test, g_test = r["user_task"] in test_tasks, r["injection_task"] in test_goals
            if r["injection_task"] and t_test != g_test:
                continue
            part = "test" if t_test else ("val" if r["user_task"] in val_tasks else "train")
            out[part].append(r)
    return out


runs = load_runs(DOJO)
splits = split_runs(runs)
for p, rs in splits.items():
    print(p, len(rs), Counter(r["kind"] for r in rs))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    subprocess.run(["pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer
text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)


def tokens(r):
    out = ["TASK: " + r["task"]]
    for s in r["steps"][:MAXSTEPS]:
        out += [s["call"], s["result"] or "RESULT"]
    return out


def featurize(rows):
    texts = [tokens(r)[:MAXTOK] for r in rows]
    unique = sorted({t for ts in texts for t in ts})
    vecs = text_encoder.encode(unique, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    index = {t: i for i, t in enumerate(unique)}
    X = torch.zeros(len(rows), MAXTOK, vecs.size(1), device=DEVICE)
    M = torch.zeros(len(rows), MAXTOK, dtype=torch.bool, device=DEVICE)
    for n, ts in enumerate(texts):
        X[n, :len(ts)] = vecs[[index[t] for t in ts]]
        M[n, :len(ts)] = True
    return X, M


feats = {p: featurize(rs) for p, rs in splits.items()}
CALL = torch.zeros(MAXTOK, dtype=torch.bool, device=DEVICE)
CALL[1::2] = True
MC = {p: feats[p][1][:, CALL] for p in splits}
KIND = {p: [r["kind"] for r in rs] for p, rs in splits.items()}
benign_train = torch.tensor([k == "benign" for k in KIND["train"]], device=DEVICE)

# %% [markdown]
# ## JEPA: same model as 05

# %%
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(384, 256), nn.GELU(), nn.Linear(256, DIM))

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, MAXTOK, DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(DIM, nhead=4, dim_feedforward=DIM * 2, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(DIM, DIM)

    def forward(self, z):
        T = z.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=z.device)
        return self.out(self.blocks(z + self.pos[:, :T], mask=causal, is_causal=True))


T_GRID = torch.linspace(-5, 5, 17, device=DEVICE)


def sigreg(z, n_slices=256):
    dirs = F.normalize(torch.randn(z.size(1), n_slices, device=z.device), dim=0)
    xt = (z @ dirs).unsqueeze(-1) * T_GRID
    gauss = torch.exp(-T_GRID ** 2 / 2)
    err = (xt.cos().mean(0) - gauss) ** 2 + xt.sin().mean(0) ** 2
    return torch.trapezoid(err * gauss, T_GRID, dim=-1).mean() * z.size(0)


def pretrain(X_all, M_all, seed):
    torch.manual_seed(seed)
    enc, pred = Encoder().to(DEVICE), Predictor().to(DEVICE)
    opt = torch.optim.AdamW([*enc.parameters(), *pred.parameters()], lr=1e-3, weight_decay=0.05)
    for _ in range(STEPS):
        i = torch.randint(len(X_all), (BATCH,))
        X, M = X_all[i], M_all[i]
        z = enc(X)
        err = (pred(z)[:, :-1] - z[:, 1:]).pow(2).mean(-1)
        loss = err[M[:, 1:]].mean() + SIGREG_WEIGHT * sigreg(z[M])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return enc.eval(), pred.eval()


# %% [markdown]
# ## Label-free scores
#
# Each returns a per-call anomaly (higher = more suspicious); a run's score
# is its maximum over calls.

# %%
def run_score(per_call, p):
    return per_call.masked_fill(~MC[p], float("-inf")).max(1).values


def calls(p):
    return feats[p][0][:, CALL]                                         # (N, MAXSTEPS, 384)


BENIGN_RAW = calls("train")[MC["train"] & benign_train[:, None]]


@torch.no_grad()
def nearest(q, bank, cosine):
    out = []
    for c in q.split(4096):
        out.append(1 - (c @ bank.T).max(-1).values if cosine else torch.cdist(c, bank).min(-1).values)
    return torch.cat(out)


def mahalanobis_fn(bank, eps=1e-3):
    mu = bank.mean(0)
    cov = torch.cov((bank - mu).T) + eps * torch.eye(bank.size(1), device=DEVICE)
    inv = torch.linalg.inv(cov)
    return lambda q: (((q - mu) @ inv) * (q - mu)).sum(-1)


def jepa_score(enc, q, chunk=1024):
    """Per-call log-density from JEPA-SCORE: sum of log singular values of d enc / d x."""
    jac = vmap(jacrev(enc.net))
    out = []
    for c in q.split(chunk):
        s = torch.linalg.svdvals(jac(c))                                # (n, DIM)
        out.append(torch.log(s.clamp_min(1e-12)).sum(-1).detach())
    return torch.cat(out)


def per_call(fn, p):
    """Apply a (n, 384) -> (n,) scorer to every call of split p."""
    q = calls(p)
    return fn(q.flatten(0, 1)).view(q.shape[:2])


def jepa_scores(enc, pred):
    """All JEPA-based per-call anomaly scores for every split."""
    with torch.no_grad():
        z_bank = enc(BENIGN_RAW)
    maha_z = mahalanobis_fn(z_bank)
    out = {}
    for p in ("val", "test"):
        X, M = feats[p]
        with torch.no_grad():
            z = enc(X)
            surprise = (pred(z)[:, :-1] - z[:, 1:]).pow(2).mean(-1)[:, CALL[1:]]
            zc = z[:, CALL]
            knn = nearest(zc.flatten(0, 1), z_bank, cosine=False).view(zc.shape[:2])
            maha = maha_z(zc.flatten(0, 1)).view(zc.shape[:2])
        dens = per_call(lambda q: jepa_score(enc, q), p)
        out[p] = {"k-NN (JEPA)": knn, "Mahalanobis (JEPA)": maha, "surprise (JEPA)": surprise,
                  "JEPA-SCORE": -dens}
    return out


def auc(neg, pos):
    return (pos[:, None] > neg[None, :]).float().mean().item()


def evaluate(s_val, s_test):
    kv, kt = KIND["val"], KIND["test"]
    thr = s_val[torch.tensor([k == "benign" for k in kv], device=DEVICE)].quantile(0.95)
    of = lambda k: s_test[torch.tensor([x == k for x in kt], device=DEVICE)]
    h = of("hijacked")
    return {"AUC vs benign": auc(of("benign"), h), "AUC vs resisted": auc(of("resisted"), h),
            "recall hijacked": (h > thr).float().mean().item(),
            "flag rate benign": (of("benign") > thr).float().mean().item(),
            "flag rate resisted": (of("resisted") > thr).float().mean().item()}


# %% [markdown]
# ## Run

# %%
results = {}
maha_raw = mahalanobis_fn(BENIGN_RAW)
for name, fn in [("seen-before (raw)", lambda q: nearest(q, BENIGN_RAW, cosine=True)), ("Mahalanobis (raw)", maha_raw)]:
    s = {p: run_score(per_call(fn, p), p) for p in ("val", "test")}
    results[name] = {"mean": evaluate(s["val"], s["test"]), "std": None}

t0 = time.time()
for pre_name, mask in [("all runs", torch.ones_like(benign_train)), ("benign runs", benign_train)]:
    per_seed = {}
    for seed in range(SEEDS):
        enc, pred = pretrain(feats["train"][0][mask], feats["train"][1][mask], seed)
        sc = jepa_scores(enc, pred)
        for name in sc["test"]:
            r = evaluate(run_score(sc["val"][name], "val"), run_score(sc["test"][name], "test"))
            per_seed.setdefault(name, []).append(r)
        print(f"  JEPA on {pre_name}, seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    for name, rs in per_seed.items():
        results[f"{name}, JEPA on {pre_name}"] = {
            "mean": {k: sum(r[k] for r in rs) / len(rs) for k in rs[0]},
            "std": {k: torch.tensor([r[k] for r in rs]).std().item() for k in rs[0]}}

print(f"{'':44s}{'AUC vs benign':>15s}{'AUC vs resisted':>17s}{'caught @5%':>12s}")
for name, r in results.items():
    m = r["mean"]
    print(f"{name:44s}{m['AUC vs benign']:15.3f}{m['AUC vs resisted']:17.3f}{m['recall hijacked']:12.1%}")

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results_no_labels.json").write_text(json.dumps({
    "dojo_commit": DOJO_COMMIT, "seeds": SEEDS, "results": results,
    "labelled_reference": {"probe, jepa, 50 labels (Q1)": 0.750, "probe + argument sources, 1,000 labels (Q5)": 0.93}},
    indent=1))

names = list(results)
fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(names) + 1.2), sharey=True)
for ax, key, title in [(axes[0], "AUC vs resisted", "AUC hijacked vs resisted (50% = chance)"),
                       (axes[1], "recall hijacked", "hijacks caught at 5% false alarms on benign")]:
    vals = [results[n]["mean"][key] for n in names]
    errs = [results[n]["std"][key] if results[n]["std"] else 0 for n in names]
    colors = ["#1baf7a" if "JEPA-SCORE" in n else "#2a78d6" if "JEPA" in n else "#9a9a9a" for n in names]
    ax.barh(range(len(names)), vals, xerr=errs, color=colors, capsize=2)
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}" if key.startswith("AUC") else f"{v:.0%}", va="center", fontsize=7)
    if key.startswith("AUC"):
        ax.axvline(0.5, color="#e34948", ls="--", lw=1)
        ax.axvline(0.750, color="#888", ls=":", lw=1)
        ax.text(0.752, len(names) - 0.6, "probe, 50 labels", fontsize=7, color="#666")
    ax.set_title(title, fontsize=9); ax.set_xlim(0, 1.05)
axes[0].set_yticks(range(len(names)), names, fontsize=8); axes[0].invert_yaxis()
plt.tight_layout(); plt.savefig(OUT / "no_labels.png", dpi=110); plt.show()
