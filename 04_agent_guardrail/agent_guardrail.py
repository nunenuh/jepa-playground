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
# # 04 · Agent tool-call guardrail
#
# Experiment 03 learned normal ball physics and flagged frames it didn't
# expect. Here the "frames" are **steps of an AI agent**: a tool call and what
# the tool returned. Can JEPA catch an agent that has been hijacked by an
# indirect prompt injection?
#
# | 03 (ball) | 04 (agent) |
# |---|---|
# | a frame | a tool call, or a tool result |
# | 4 past frames | the task + everything the agent did and saw so far |
# | normal physics | normal agent behaviour |
# | teleport / reverse | a prompt injection hijacking the agent |
#
# **Data:** [AgentDrift](https://github.com/Asif-0209/AgentDrift) (CC BY 4.0):
# 12,536 synthetic tool-call trajectories over email, banking, web, coding and
# medical agents. Every step is labelled `benign`, `injection_point`,
# `hijacked` or `failed_injection`. We use the *task-disjoint* split, so test
# trajectories come from task templates never seen in training.
#
# Two attempts:
# - **Part A, anomaly detection** (the 03 recipe): train only on normal
#   trajectories, flag surprising steps. No attack labels at all.
# - **Part B, pretraining + a small probe**: pretrain JEPA on all trajectories
#   *without labels*, then train a linear classifier with a few labelled ones.
#
# ![architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/04_agent_guardrail/assets/architecture-light.svg)

# %%
import json
import re
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 128
MAXTOK = 1 + 2 * 11      # task, then (call, result) for up to 11 steps
STEPS = 3000
BATCH = 128
SIGREG_WEIGHT = 0.002    # same as 03
torch.manual_seed(0)

COMMIT = "014a514fa998b4ac4519579fceb8a5884b379bda"   # pinned dataset version
URL = f"https://raw.githubusercontent.com/Asif-0209/AgentDrift/{COMMIT}/data_taskdisjoint/{{}}.jsonl"
DATA = Path("data/agentdrift")
DATA.mkdir(parents=True, exist_ok=True)
splits = {}
for part in ["train", "val", "test"]:
    f = DATA / f"{part}.jsonl"
    if not f.exists():
        urllib.request.urlretrieve(URL.format(part), f)
    splits[part] = [json.loads(line) for line in open(f)]
    print(part, len(splits[part]), "trajectories")

# %% [markdown]
# ## From a trajectory to a sequence of vectors
#
# A trajectory becomes a sequence of short texts: the task, then for every
# step a `CALL tool(args)` and a `RESULT ...`. A small frozen sentence encoder
# (MiniLM, 384-D) turns each text into a vector. JEPA works on those vectors.
#
# **Leakage fix.** The dataset README warns that the world objects (user,
# company, contacts) alone predict the attacked class with 93–98% accuracy in
# four domains, because of how the data was generated. So every world name
# and address is replaced by a placeholder (`<USER>`, `<CONTACT_EMAIL>`, ...).
# Addresses that are *not* part of the world (e.g. an attacker's) stay.

# %%
def anonymize(text, world):
    pairs = [(world.get("email", ""), "<USER_EMAIL>"), (world.get("user", ""), "<USER>"),
             (world.get("company", ""), "<COMPANY>")]
    for c in world.get("contacts", []):
        pairs += [(c.get("email", ""), "<CONTACT_EMAIL>"), (c.get("name", ""), "<CONTACT>")]
    for old, new in sorted(pairs, key=lambda p: -len(p[0])):            # longest first
        if old:
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    return text


def tokens(r):
    w = r["world"]
    out = [anonymize("TASK: " + r["task"], w)]
    for s in r["steps"]:
        out += [anonymize(f"CALL {s['tool']}({json.dumps(s['args'])})", w),
                anonymize(f"RESULT {s['obs']}", w)]
    return out[:MAXTOK]


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"], check=True)
    from sentence_transformers import SentenceTransformer

text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)


def featurize(rows):
    """-> X (N, MAXTOK, 384) frozen vectors, M (N, MAXTOK) True where a token exists."""
    texts = [tokens(r) for r in rows]
    flat = text_encoder.encode([t for ts in texts for t in ts], batch_size=512,
                               convert_to_tensor=True, normalize_embeddings=True).float()
    X = torch.zeros(len(rows), MAXTOK, flat.size(1), device=DEVICE)
    M = torch.zeros(len(rows), MAXTOK, dtype=torch.bool, device=DEVICE)
    i = 0
    for n, ts in enumerate(texts):
        X[n, :len(ts)], M[n, :len(ts)] = flat[i:i + len(ts)], True
        i += len(ts)
    return X, M


t0 = time.time()
feats = {part: featurize(rows) for part, rows in splits.items()}
print(f"embedded every call and result in {time.time() - t0:.0f}s")
CALL = torch.zeros(MAXTOK, dtype=torch.bool, device=DEVICE)
CALL[1::2] = True                                       # call tokens 1, 3, 5, ... = step 0, 1, 2, ...


# %% [markdown]
# ## Model (same recipe as 03)
#
# - **Encoder**: an MLP, frozen 384-D vector -> 128-D latent. It embeds both
#   the context and the target (LeJEPA style, no EMA).
# - **Predictor**: a small *causal* transformer. Output `k` sees tokens up to
#   `k` and predicts the latent of token `k + 1`.
# - **SIGReg** stops collapse.

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

    def forward(self, z):                                  # (B, T, DIM) -> (B, T, DIM)
        T = z.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=z.device)
        return self.out(self.blocks(z + self.pos[:, :T], mask=causal, is_causal=True))


T_GRID = torch.linspace(-5, 5, 17, device=DEVICE)


def sigreg(z, n_slices=256):
    """How far random 1-D projections of z (N, D) are from N(0, 1). See 02."""
    dirs = F.normalize(torch.randn(z.size(1), n_slices, device=z.device), dim=0)
    xt = (z @ dirs).unsqueeze(-1) * T_GRID
    gauss = torch.exp(-T_GRID ** 2 / 2)
    err = (xt.cos().mean(0) - gauss) ** 2 + xt.sin().mean(0) ** 2
    return torch.trapezoid(err * gauss, T_GRID, dim=-1).mean() * z.size(0)


def pretrain(X_all, M_all, seed=0):
    """JEPA on unlabelled trajectories: predict every next token's latent."""
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
# ## Evaluation (same for every detector)
#
# Every detector gives each step a score; a trajectory's score is its
# highest-scoring step.
# - **Detection AUC** against each kind of negative, as the dataset asks:
#   benign, hard negatives (suspicious-looking but legitimate) and failed
#   attacks (the injection was there, but the agent ignored it).
# - **Recall per hijack pattern** at a threshold that flags 5% of benign
#   *validation* trajectories.
# - **Localization**: is the top-scoring step the injection or a hijacked step?

# %%
def auc(neg, pos):
    return (pos[:, None] > neg[None, :]).float().mean().item()


def evaluate(step_scores):
    """step_scores(part) -> (N, 11) scores, column j = r['steps'][j], -inf where there is no step."""
    val = step_scores("val").max(1).values
    is_benign = torch.tensor([r["category"] == "benign" for r in splits["val"]], device=DEVICE)
    threshold = val[is_benign].quantile(0.95)

    s = step_scores("test")
    traj = s.max(1).values
    rows = splits["test"]
    of = lambda cond: traj[torch.tensor([cond(r) for r in rows], device=DEVICE)]
    attacked = of(lambda r: r["category"] == "attacked")
    out = {}
    for neg in ["benign", "hard_negative", "failed_attack"]:
        out[f"AUC vs {neg}"] = auc(of(lambda r: r["category"] == neg), attacked)
    for comp in ["full_hijack", "partial_hijack", "delayed_execution"]:
        out[f"recall {comp}"] = (of(lambda r: r.get("compliance") == comp) > threshold).float().mean().item()
    out["flag rate benign"] = (of(lambda r: r["category"] == "benign") > threshold).float().mean().item()
    hits = [r["steps"][s[n].argmax().item()]["label"] in {"injection_point", "hijacked"}
            for n, r in enumerate(rows) if r["category"] == "attacked"]
    out["localization"] = sum(hits) / len(hits)
    return out, s, threshold


def show(results):
    names = list(results)
    print(f"{'':24s}" + "".join(f"{n[:24]:>26s}" for n in names))
    for key in results[names[0]]:
        print(f"{key:24s}" + "".join(f"{results[n][key]:26.3f}" for n in names))


# %% [markdown]
# ## Part A: anomaly detection, no attack labels
#
# Train JEPA only on normal trajectories (benign + hard negatives). A call's
# surprise = how far its latent is from what the predictor expected.
#
# Baseline: "have I seen a call like this before?" (1 minus the cosine
# similarity to the closest call in the normal training trajectories).

# %%
normal = torch.tensor([r["category"] in {"benign", "hard_negative"} for r in splits["train"]], device=DEVICE)
X_norm, M_norm = feats["train"][0][normal], feats["train"][1][normal]
t0 = time.time()
enc_a, pred_a = pretrain(X_norm, M_norm)
print(f"trained on {len(X_norm)} normal trajectories in {time.time() - t0:.0f}s")


@torch.no_grad()
def call_surprise(part, enc=enc_a, pred=pred_a):
    X, M = feats[part]
    z = enc(X)
    s = (pred(z)[:, :-1] - z[:, 1:]).pow(2).mean(-1)[:, CALL[1:]]     # surprise of each call token
    return s.masked_fill(~M[:, CALL], float("-inf"))


NORMAL_CALLS = X_norm[:, CALL][M_norm[:, CALL]]


@torch.no_grad()
def call_novelty(part):
    X, M = feats[part]
    calls = X[:, CALL]
    best = torch.cat([(c @ NORMAL_CALLS.T).max(-1).values for c in calls.flatten(0, 1).split(4096)])
    return (1 - best.view(calls.shape[:2])).masked_fill(~M[:, CALL], float("-inf"))


part_a = {"JEPA surprise": evaluate(call_surprise)[0], "seen-before baseline": evaluate(call_novelty)[0]}
show(part_a)

# %% [markdown]
# ## Part B: JEPA pretraining + a small labelled probe
#
# 1. **Pretrain** JEPA on *all* training trajectories, labels never looked at
#    (realistic: you log every agent run, but almost none are labelled).
# 2. Describe each call by features, and train a **linear classifier** to
#    say "this call is hijacked", using only `n` labelled trajectories.
#
# | Features | What they are |
# |---|---|
# | `raw` | the frozen text vector of the call + of the task |
# | `jepa` | the call's latent, the latent JEPA *predicted* for it from context, and their difference |
# | `raw+jepa` | both |

# %%
t0 = time.time()
enc_b, pred_b = pretrain(*feats["train"])
print(f"pretrained on all {len(splits['train'])} trajectories (no labels) in {time.time() - t0:.0f}s")


@torch.no_grad()
def call_features(part, kind):
    X, M = feats[part]
    calls = X[:, CALL]
    raw = torch.cat([calls, X[:, :1].expand_as(calls)], -1)
    if kind == "raw":
        return raw
    z = enc_b(X)
    zc, pc = z[:, CALL], pred_b(z)[:, :-1][:, CALL[1:]]                 # actual vs predicted call latent
    jepa = torch.cat([zc, pc, zc - pc], -1)
    return jepa if kind == "jepa" else torch.cat([raw, jepa], -1)


KINDS = ["raw", "jepa", "raw+jepa"]
FEATS = {(p, k): call_features(p, k) for p in splits for k in KINDS}
HIJACKED = torch.tensor([[float(j < len(r["steps"]) and r["steps"][j]["label"] == "hijacked") for j in range(11)]
                         for r in splits["train"]], device=DEVICE)


def fit_probe(kind, n_labelled, seed, epochs=300):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(splits["train"]), generator=g)[:n_labelled].to(DEVICE)
    m = feats["train"][1][:, CALL][idx]
    Xf, yf = FEATS[("train", kind)][idx][m], HIJACKED[idx][m]
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-6
    lin = nn.Linear(Xf.size(1), 1).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    pos_weight = ((1 - yf).sum() / yf.sum().clamp(min=1)).clamp(max=50)
    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(lin((Xf - mu) / sd).squeeze(-1), yf, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def step_scores(part):
        s = lin((FEATS[(part, kind)] - mu) / sd).squeeze(-1)
        return s.masked_fill(~feats[part][1][:, CALL], float("-inf"))
    return step_scores


LABELLED = [25, 50, 100, 250, 1000, len(splits["train"])]
SEEDS = 3
part_b = {}
t0 = time.time()
for n in LABELLED:
    for kind in KINDS:
        runs = [evaluate(fit_probe(kind, n, s))[0] for s in range(SEEDS)]
        part_b[(n, kind)] = {k: sum(r[k] for r in runs) / SEEDS for k in runs[0]}
print(f"{len(LABELLED) * len(KINDS) * SEEDS} probes in {time.time() - t0:.0f}s\n")

print("AUC vs benign (mean of 3 label samples)")
print(f"{'labelled':>9s}" + "".join(f"{k:>11s}" for k in KINDS))
for n in LABELLED:
    print(f"{n:9d}" + "".join(f"{part_b[(n, k)]['AUC vs benign']:11.3f}" for k in KINDS))
print("\nall labels:")
show({k: part_b[(LABELLED[-1], k)] for k in KINDS})

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
KIND_COLORS = {"raw": "#898781", "jepa": "#2a78d6", "raw+jepa": "#1baf7a"}

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for kind in KINDS:
    for ax, key in zip(axes, ["AUC vs benign", "AUC vs hard_negative"]):
        ax.plot(LABELLED, [part_b[(n, kind)][key] for n in LABELLED], marker="o",
                color=KIND_COLORS[kind], label=f"probe on {kind}")
for ax, key in zip(axes, ["AUC vs benign", "AUC vs hard_negative"]):
    ax.axhline(part_a["seen-before baseline"][key], color="#e34948", ls="--", lw=1, label="no labels: seen-before")
    ax.axhline(part_a["JEPA surprise"][key], color="#eda100", ls="--", lw=1, label="no labels: JEPA surprise")
    ax.set_xscale("log"); ax.set_xlabel("labelled trajectories"); ax.set_title(f"Attacked {key}")
    ax.set_ylim(0.5, 1); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "labels.png", dpi=110); plt.show()

# %%
patterns = ["full_hijack", "partial_hijack", "delayed_execution"]
bars_for = {"JEPA surprise (no labels)": (part_a["JEPA surprise"], "#eda100"),
            "seen-before (no labels)": (part_a["seen-before baseline"], "#e34948"),
            "probe on raw": (part_b[(LABELLED[-1], "raw")], "#898781"),
            "probe on raw+jepa": (part_b[(LABELLED[-1], "raw+jepa")], "#1baf7a")}
fig, ax = plt.subplots(figsize=(8, 3.4))
width = 0.2
for k, (name, (res, color)) in enumerate(bars_for.items()):
    vals = [res[f"recall {p}"] for p in patterns]
    bars = ax.bar([x + (k - 1.5) * width for x in range(len(patterns))], vals, width, color=color, label=name)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center", fontsize=7)
ax.set_xticks(range(len(patterns)), patterns); ax.set_ylim(0, 1.1); ax.set_ylabel("recall")
ax.set_title("Attacks caught at ~5% false alarms on benign runs"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "recall.png", dpi=110); plt.show()

# %%
LABEL_COLORS = {"benign": "#2a78d6", "injection_point": "#eda100", "hijacked": "#e34948",
                "failed_injection": "#1baf7a"}
probe = fit_probe("raw+jepa", LABELLED[-1], 0)
_, test_scores, THRESHOLD = evaluate(probe)
rows = splits["test"]
examples = [next(n for n, r in enumerate(rows) if r["source_category"] == c)
            for c in ["benign", "attacked_full", "attacked_delayed", "failed_attack"]]
fig, axes = plt.subplots(1, len(examples), figsize=(4.2 * len(examples), 3.6), sharey=True)
for ax, n in zip(axes, examples):
    r = rows[n]
    s = test_scores[n][:len(r["steps"])].cpu()
    ax.bar(range(len(s)), s, color=[LABEL_COLORS[st["label"]] for st in r["steps"]])
    ax.axhline(THRESHOLD.item(), color="#898781", ls="--", lw=1)
    ax.set_xticks(range(len(s)), [st["tool"] for st in r["steps"]], rotation=60, ha="right", fontsize=7)
    ax.set_title(f"{r['source_category']} ({r['agent']})", fontsize=9)
axes[0].set_ylabel("hijack score (probe on raw+jepa)")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLORS.values()]
fig.legend(handles, list(LABEL_COLORS), loc="upper center", ncol=4, fontsize=8, frameon=False)
plt.tight_layout(rect=(0, 0, 1, 0.92)); plt.savefig(OUT / "steps.png", dpi=110); plt.show()
