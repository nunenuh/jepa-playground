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
# # 05 · Guardrail on real agents
#
# Experiment 04 worked on **synthetic** agent runs (AgentDrift). Here we test
# the same recipe on **real** ones: AgentDojo's published benchmark runs, where
# 29 real LLM agents (GPT-4o, Claude 3.x, Gemini, Llama, ...) were attacked
# with indirect prompt injections.
#
# Three questions:
# 1. Does **JEPA pretraining + a small probe** (04, Part B) work on real agents?
# 2. Does a guardrail trained on **synthetic** data work on **real** agents?
# 3. Which agents' hijacks are easiest and hardest to catch?
#
# **Data:** [AgentDojo](https://github.com/ethz-spylab/agentdojo) runs (MIT),
# pinned commit. We keep the `important_instructions` attack and the
# no-attack runs. Each run is one of:
# - **benign**: no attack;
# - **hijacked**: the injection was in a tool result and the agent obeyed;
# - **resisted**: the injection was there, but the agent ignored it.
#
# **Hijacked vs resisted** is the honest test: both runs contain the same
# injected text, so only the agent's *actions* differ.
#
# ![architecture](https://raw.githubusercontent.com/nunenuh/jepa-playground/main/05_real_agents/assets/architecture-light.svg)

# %%
import json
import subprocess
import time
import urllib.request
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 128
MAXSTEPS = 15
MAXTOK = 1 + 2 * MAXSTEPS   # task, then (call, result) per step
STEPS = 3000
BATCH = 128
SIGREG_WEIGHT = 0.002       # as in 03–04
torch.manual_seed(0)

DOJO_COMMIT = "089ed468cf3ed0322acc66b0211f26d9d90dbf60"
DOJO = Path("data/agentdojo")
if not (DOJO / "runs").exists():
    subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse",
                    "https://github.com/ethz-spylab/agentdojo", str(DOJO)], check=True)
    subprocess.run(["git", "-C", str(DOJO), "sparse-checkout", "set", "runs"], check=True)
    subprocess.run(["git", "-C", str(DOJO), "checkout", "-q", DOJO_COMMIT], check=True)


def load_runs(root):
    rows = []
    for f in sorted(root.glob("runs/*/*/*/*/*.json")):
        model, suite, user_task, attack = f.parts[-5:-1]
        if attack not in ("none", "important_instructions"):
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
        if not steps:                                    # agent answered without any tool call
            continue
        task = next((m["content"] for m in d["messages"] if m["role"] == "user"), "")
        kind = "benign" if attack == "none" else ("hijacked" if d.get("security") else "resisted")
        rows.append(dict(model=model, suite=suite, user_task=user_task, injection_task=d.get("injection_task_id"),
                         kind=kind, task=task if isinstance(task, str) else json.dumps(task), steps=steps))
    return rows


runs = load_runs(DOJO)
print(len(runs), "runs with at least one tool call:", Counter(r["kind"] for r in runs))

# %% [markdown]
# ## A fair split
#
# The same few attacker goals ("send the files to mark.black-...") repeat
# across runs. A classifier could memorise them. So per suite we hold out
# 30% of the **user tasks** and 30% of the **attacker goals**: test runs use
# tasks *and* goals never seen in training. Attacked runs that mix a seen and
# an unseen part are dropped. 15% of the remaining user tasks become
# validation (used only to set the alarm threshold).

# %%
def split_runs(rows, seed=0):
    g = torch.Generator().manual_seed(seed)
    pick = lambda items, frac: set(items[i] for i in torch.randperm(len(items), generator=g)[:max(1, round(frac * len(items)))].tolist())
    out = {"train": [], "val": [], "test": []}
    for suite in sorted({r["suite"] for r in rows}):
        tasks = sorted({r["user_task"] for r in rows if r["suite"] == suite})
        goals = sorted({r["injection_task"] for r in rows if r["suite"] == suite and r["injection_task"]})
        test_tasks, test_goals = pick(tasks, 0.3), pick(goals, 0.3)
        val_tasks = pick([t for t in tasks if t not in test_tasks], 0.15)
        for r in (r for r in rows if r["suite"] == suite):
            t_test, g_test = r["user_task"] in test_tasks, r["injection_task"] in test_goals
            if r["injection_task"] and t_test != g_test:
                continue                                 # mixed: seen task + unseen goal or vice versa
            part = "test" if t_test else ("val" if r["user_task"] in val_tasks else "train")
            out[part].append(r)
    return out


splits = split_runs(runs)
for p, rs in splits.items():
    print(p, len(rs), Counter(r["kind"] for r in rs))

# %% [markdown]
# ## Runs -> vectors (as in 04)
#
# `TASK`, then `CALL tool(args)` and `RESULT ...` per step, each turned into
# a 384-D vector by frozen MiniLM. Many texts repeat across the 29 agents, so
# each unique text is embedded once.

# %%
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


def featurize(rows, token_fn=tokens):
    texts = [token_fn(r)[:MAXTOK] for r in rows]
    unique = sorted({t for ts in texts for t in ts})
    vecs = text_encoder.encode(unique, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    index = {t: i for i, t in enumerate(unique)}
    X = torch.zeros(len(rows), MAXTOK, vecs.size(1), device=DEVICE)
    M = torch.zeros(len(rows), MAXTOK, dtype=torch.bool, device=DEVICE)
    for n, ts in enumerate(texts):
        X[n, :len(ts)] = vecs[[index[t] for t in ts]]
        M[n, :len(ts)] = True
    return X, M


t0 = time.time()
feats = {p: featurize(rs) for p, rs in splits.items()}
print(f"embedded in {time.time() - t0:.0f}s")
CALL = torch.zeros(MAXTOK, dtype=torch.bool, device=DEVICE)
CALL[1::2] = True
LABEL = {p: torch.tensor([float(r["kind"] == "hijacked") for r in rs], device=DEVICE) for p, rs in splits.items()}


# %% [markdown]
# ## JEPA (same model as 04)

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


def pretrain(X_all, M_all, seed=0):
    """JEPA on unlabelled runs. Returns encoder, predictor, and (step, loss, spread) history."""
    torch.manual_seed(seed)
    enc, pred = Encoder().to(DEVICE), Predictor().to(DEVICE)
    opt = torch.optim.AdamW([*enc.parameters(), *pred.parameters()], lr=1e-3, weight_decay=0.05)
    history = []
    for step in range(1, STEPS + 1):
        i = torch.randint(len(X_all), (BATCH,))
        X, M = X_all[i], M_all[i]
        z = enc(X)
        err = (pred(z)[:, :-1] - z[:, 1:]).pow(2).mean(-1)
        pred_loss = err[M[:, 1:]].mean()
        loss = pred_loss + SIGREG_WEIGHT * sigreg(z[M])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            history.append((step, pred_loss.item(), z[M].detach().std(0).mean().item()))
    return enc.eval(), pred.eval(), history


t0 = time.time()
enc, pred, history = pretrain(*feats["train"])
print(f"JEPA pretrained on {len(splits['train'])} real runs (no labels) in {time.time() - t0:.0f}s")


@torch.no_grad()
def call_features(X, M, enc=enc, pred=pred):
    """(N, MAXSTEPS, 768 raw + 384 jepa) features for every call."""
    calls = X[:, CALL]
    raw = torch.cat([calls, X[:, :1].expand_as(calls)], -1)
    z = enc(X)
    zc, pc = z[:, CALL], pred(z)[:, :-1][:, CALL[1:]]
    return torch.cat([raw, zc, pc, zc - pc], -1)


FEATS = {p: call_features(*feats[p]) for p in splits}
COLS = {"raw": slice(0, 768), "jepa": slice(768, None), "raw+jepa": slice(0, None)}

# %% [markdown]
# ## Probe: learns from whole-run labels
#
# AgentDojo says whether a run was hijacked, not *which* call. So the probe
# scores every call, a run's score is its **highest** call score, and the
# training loss is on that run score (multiple-instance learning).
#
# Positives: hijacked. Negatives: benign **and** resisted, so the probe must
# learn "the agent did something bad", not "there is injected text".

# %%
def fit_probe(F_train, M_calls, y, epochs=300, seed=0):
    torch.manual_seed(seed)
    mu, sd = F_train[M_calls].mean(0), F_train[M_calls].std(0) + 1e-6
    lin = nn.Linear(F_train.size(-1), 1).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    pos_weight = ((1 - y).sum() / y.sum().clamp(min=1)).clamp(max=50)
    for _ in range(epochs):
        s = lin((F_train - mu) / sd).squeeze(-1).masked_fill(~M_calls, -1e4).max(1).values
        loss = F.binary_cross_entropy_with_logits(s, y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def score(F_any, M_any):
        return lin((F_any - mu) / sd).squeeze(-1).masked_fill(~M_any, float("-inf")).max(1).values
    return score


def auc(neg, pos):
    return (pos[:, None] > neg[None, :]).float().mean().item()


def evaluate(score_val, score_test, rows_val=None, rows_test=None):
    rows_val, rows_test = rows_val or splits["val"], rows_test or splits["test"]
    benign_val = torch.tensor([r["kind"] == "benign" for r in rows_val], device=DEVICE)
    threshold = score_val[benign_val].quantile(0.95)
    of = lambda k: score_test[torch.tensor([r["kind"] == k for r in rows_test], device=DEVICE)]
    h = of("hijacked")
    return {"AUC vs benign": auc(of("benign"), h), "AUC vs resisted": auc(of("resisted"), h),
            "recall hijacked": (h > threshold).float().mean().item(),
            "flag rate benign": (of("benign") > threshold).float().mean().item(),
            "flag rate resisted": (of("resisted") > threshold).float().mean().item()}


def show(results):
    names = list(results)
    print(f"{'':20s}" + "".join(f"{n[:26]:>28s}" for n in names))
    for key in results[names[0]]:
        print(f"{key:20s}" + "".join(f"{results[n][key]:28.3f}" for n in names))


MC = {p: feats[p][1][:, CALL] for p in splits}
is_benign = torch.tensor([r["kind"] == "benign" for r in splits["train"]], device=DEVICE)
BENIGN_CALLS = feats["train"][0][:, CALL][MC["train"] & is_benign[:, None]]          # calls from benign train runs


@torch.no_grad()
def seen_before(p):
    calls = feats[p][0][:, CALL]
    best = torch.cat([(c @ BENIGN_CALLS.T).max(-1).values for c in calls.flatten(0, 1).split(4096)])
    return (1 - best.view(calls.shape[:2])).masked_fill(~MC[p], float("-inf")).max(1).values


baseline = evaluate(seen_before("val"), seen_before("test"))

# %% [markdown]
# ## Q1: does pretraining + probe work on real agents?

# %%
LABELLED = [50, 100, 250, 1000, len(splits["train"])]
SEEDS = 3
q1 = {}
for n in LABELLED:
    for kind, cols in COLS.items():
        runs_ = []
        for seed in range(SEEDS):
            idx = torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(seed))[:n].to(DEVICE)
            probe = fit_probe(FEATS["train"][idx][..., cols], MC["train"][idx], LABEL["train"][idx], seed=seed)
            runs_.append(evaluate(probe(FEATS["val"][..., cols], MC["val"]), probe(FEATS["test"][..., cols], MC["test"])))
        q1[(n, kind)] = {k: sum(r[k] for r in runs_) / SEEDS for k in runs_[0]}

print("AUC hijacked vs resisted (mean of 3 label samples)")
print(f"{'labelled':>9s}" + "".join(f"{k:>11s}" for k in COLS))
for n in LABELLED:
    print(f"{n:9d}" + "".join(f"{q1[(n, k)]['AUC vs resisted']:11.3f}" for k in COLS))
print("\nall labels vs the no-label baseline:")
show({"seen-before (no labels)": baseline, **{f"probe on {k}": q1[(LABELLED[-1], k)] for k in COLS}})

# %% [markdown]
# ## Q2: synthetic -> real
#
# Train the probe on AgentDrift (synthetic, 04's data, call-level labels) and
# test it on the real AgentDojo test runs, without any AgentDojo labels.

# %%
DRIFT_COMMIT = "014a514fa998b4ac4519579fceb8a5884b379bda"
DRIFT = Path("data/agentdrift")
DRIFT.mkdir(parents=True, exist_ok=True)
f = DRIFT / "train.jsonl"
if not f.exists():
    urllib.request.urlretrieve(
        f"https://raw.githubusercontent.com/Asif-0209/AgentDrift/{DRIFT_COMMIT}/data_taskdisjoint/train.jsonl", f)
drift = [json.loads(line) for line in open(f)]


def drift_tokens(r):
    out = ["TASK: " + r["task"]]
    for s in r["steps"][:MAXSTEPS]:
        out += [f"CALL {s['tool']}({json.dumps(s['args'])})", f"RESULT {s['obs']}"]
    return out


X_d, M_d = featurize(drift, drift_tokens)
F_d, MC_d = call_features(X_d, M_d), M_d[:, CALL]
y_d = torch.tensor([[float(j < len(r["steps"]) and r["steps"][j]["label"] == "hijacked") for j in range(MAXSTEPS)]
                    for r in drift], device=DEVICE)


def fit_call_probe(F_train, M_calls, y_calls, epochs=300):
    """Call-level labels (AgentDrift has them)."""
    Xf, yf = F_train[M_calls], y_calls[M_calls]
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
    def score(F_any, M_any):
        return lin((F_any - mu) / sd).squeeze(-1).masked_fill(~M_any, float("-inf")).max(1).values
    return score


q2 = {}
for kind, cols in COLS.items():
    probe = fit_call_probe(F_d[..., cols], MC_d, y_d)
    q2[f"synthetic -> real, {kind}"] = evaluate(probe(FEATS["val"][..., cols], MC["val"]),
                                                probe(FEATS["test"][..., cols], MC["test"]))
q2["real -> real, raw+jepa"] = q1[(LABELLED[-1], "raw+jepa")]
show(q2)

# %% [markdown]
# ## Q4: a probe that reads the whole run
#
# The linear probe looks at one call (+ the task) at a time, so it can learn
# "sending money is risky" but not "this recipient is not the one in the
# bill". This probe is a small **bidirectional transformer over the whole
# run**: every call can attend to the task, the earlier results and the other
# calls before it is scored. Same multiple-instance loss, same split.
#
# Input per token: the frozen MiniLM vector, plus (for `+jepa`) the token's
# JEPA latent and the latent JEPA *predicted* for it.

# %%
@torch.no_grad()
def token_features(X, M, kind):
    if kind == "raw":
        return X
    z = enc(X)
    p = torch.cat([torch.zeros_like(z[:, :1]), pred(z)[:, :-1]], 1)     # prediction made for each token
    return torch.cat([X, z, p], -1)


class RunProbe(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.inp = nn.Linear(d_in, DIM)
        self.pos = nn.Parameter(torch.randn(1, MAXTOK, DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(DIM, nhead=4, dim_feedforward=DIM * 2, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, 2)
        self.out = nn.Linear(DIM, 1)

    def forward(self, T, M):                                            # -> one logit per call
        h = self.blocks(self.inp(T) + self.pos, src_key_padding_mask=~M)
        return self.out(h).squeeze(-1)[:, CALL]


def fit_run_probe(kind, idx, epochs=30, seed=0):
    torch.manual_seed(seed)
    Tt, Mt, yt = token_features(*feats["train"], kind)[idx], feats["train"][1][idx], LABEL["train"][idx]
    Tv, Mv = token_features(*feats["val"], kind), feats["val"][1]
    mu, sd = Tt[Mt].mean(0), Tt[Mt].std(0) + 1e-6
    model = RunProbe(Tt.size(-1)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    pos_weight = ((1 - yt).sum() / yt.sum().clamp(min=1)).clamp(max=50)
    run_score = lambda T, M: model((T - mu) / sd, M).masked_fill(~M[:, CALL], -1e4).max(1).values
    best, best_state = -1, None
    val_h = torch.tensor([r["kind"] == "hijacked" for r in splits["val"]], device=DEVICE)
    for _ in range(epochs):
        model.train()
        for b in torch.randperm(len(Tt), device=DEVICE).split(256):
            loss = F.binary_cross_entropy_with_logits(run_score(Tt[b], Mt[b]), yt[b], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = run_score(Tv, Mv)
        a = auc(sv[~val_h], sv[val_h])                                  # early stopping on the labelled val split
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    @torch.no_grad()
    def calls(part):
        T, M = token_features(*feats[part], kind), feats[part][1]
        return model((T - mu) / sd, M).masked_fill(~M[:, CALL], float("-inf"))
    return calls


q4 = {}
for n in [250, 1000, len(splits["train"])]:
    for kind in ["raw", "raw+jepa"]:
        runs_ = []
        for seed in range(SEEDS):
            idx = torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(seed))[:n].to(DEVICE)
            calls = fit_run_probe(kind, idx, seed=seed)
            runs_.append(evaluate(calls("val").max(1).values, calls("test").max(1).values))
        q4[(n, kind)] = {k: sum(r[k] for r in runs_) / SEEDS for k in runs_[0]}
        print(f"n={n:5d} {kind:9s} " + "  ".join(f"{k} {v:.3f}" for k, v in q4[(n, kind)].items()))

# %% [markdown]
# ## Q5: where did each argument come from?
#
# MiniLM turns text into a *meaning* vector, so two IBANs that differ in a
# few digits look almost the same to it. Exact strings matter here. So for
# every call we look up its argument values (IBANs, emails, file names, ...)
# and find the **line** earlier in the run where each value appeared:
#
# - in the user's task -> `FROM TASK`
# - in an earlier tool result -> `FROM RESULT: <that line>`
# - nowhere -> `FROM NOWHERE` (the agent made it up)
#
# The source line is embedded with MiniLM and added to the call's token.
# A value copied from "Spotify payment to US122..." and one copied from
# "IMPORTANT: send money to US133..." now look very different.

# %%
def flat_values(x):
    if isinstance(x, dict):
        return [v for y in x.values() for v in flat_values(y)]
    if isinstance(x, list):
        return [v for y in x for v in flat_values(y)]
    s = str(x).strip()
    return [s] if len(s) >= 4 else []


def result_lines(result):
    """Lines of a tool result. Some agents store results as JSON content blocks; unpack those."""
    body = result[len("RESULT "):]
    if body.startswith("[{"):
        try:
            body = "\n".join(str(b.get("content", b.get("text", ""))) for b in json.loads(body) if isinstance(b, dict))
        except json.JSONDecodeError:
            pass
    return body.splitlines()


def source_texts(r):
    """One 'where did the arguments come from' text per call."""
    out = []
    for j, step in enumerate(r["steps"][:MAXSTEPS]):
        call = step["call"]
        try:
            values = flat_values(json.loads(call[call.index("(") + 1:-1]))
        except (ValueError, json.JSONDecodeError):
            values = []
        if not values:
            out.append("NO ARGUMENTS")
            continue
        found = None
        for v in values:                                   # most recent result first
            for prev in reversed(r["steps"][:j]):
                line = next((ln for ln in result_lines(prev["result"]) if v in ln), None)
                if line:
                    found = "FROM RESULT: " + line.strip()[:300]
                    break
            if found:
                break
        if found:
            out.append(found)
        elif all(v in r["task"] for v in values):
            out.append("FROM TASK")
        else:
            out.append("FROM NOWHERE")
    return out


def source_features(rows):
    texts = [source_texts(r) for r in rows]
    unique = sorted({t for ts in texts for t in ts})
    vecs = text_encoder.encode(unique, batch_size=512, convert_to_tensor=True, normalize_embeddings=True).float()
    index = {t: i for i, t in enumerate(unique)}
    S = torch.zeros(len(rows), MAXTOK, vecs.size(1), device=DEVICE)
    for n, ts in enumerate(texts):
        if ts:
            S[n, 1:1 + 2 * len(ts):2] = vecs[[index[t] for t in ts]]   # on the call tokens
    return S


t0 = time.time()
SRC = {p: source_features(rs) for p, rs in splits.items()}
print(f"argument sources in {time.time() - t0:.0f}s")
r = next(r for r in splits["test"] if r["kind"] == "hijacked" and "pay the bill, like last month" in r["task"])
for step, src in zip(r["steps"], source_texts(r)):
    print(f"  {step['call'][:60]:60s} <- {src[:90]}")

_token_features = token_features


@torch.no_grad()
def token_features(X, M, kind):
    base = _token_features(X, M, kind.replace("+src", ""))
    if not kind.endswith("+src"):
        return base
    part = next(p for p in splits if feats[p][0] is X)
    return torch.cat([base, SRC[part]], -1)


q5 = {}
for n in [250, 1000, len(splits["train"])]:
    for kind in ["raw+src", "raw+jepa+src"]:
        runs_ = []
        for seed in range(SEEDS):
            idx = torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(seed))[:n].to(DEVICE)
            calls = fit_run_probe(kind, idx, seed=seed)
            runs_.append(evaluate(calls("val").max(1).values, calls("test").max(1).values))
        q5[(n, kind)] = {k: sum(r[k] for r in runs_) / SEEDS for k in runs_[0]}
        print(f"n={n:5d} {kind:13s} " + "  ".join(f"{k} {v:.3f}" for k, v in q5[(n, kind)].items()))

# %% [markdown]
# ## Q3: which agents are easiest / hardest to catch?
#
# Best probe (raw+jepa, all labels), AUC of hijacked vs resisted per agent
# on the test split (agents with at least 10 hijacked test runs).

# %%
probe = fit_probe(FEATS["train"], MC["train"], LABEL["train"])
s_test = probe(FEATS["test"], MC["test"])
per_model = {}
for m in sorted({r["model"] for r in splits["test"]}):
    sel = lambda k: torch.tensor([r["model"] == m and r["kind"] == k for r in splits["test"]], device=DEVICE)
    if sel("hijacked").sum() >= 10 and sel("resisted").sum() >= 10:
        per_model[m] = (auc(s_test[sel("resisted")], s_test[sel("hijacked")]), int(sel("hijacked").sum()))
for m, (a, n) in sorted(per_model.items(), key=lambda kv: -kv[1][0]):
    print(f"{m:55s} AUC {a:.3f}  ({n} hijacked)")

# %% [markdown]
# ## Pictures

# %%
OUT = (Path(__file__).parent if "__file__" in globals() else Path(".")) / "assets"
OUT.mkdir(exist_ok=True)
KIND_COLORS = {"benign": "#2a78d6", "resisted": "#eda100", "hijacked": "#e34948"}

counts = Counter((r["model"], r["kind"]) for r in runs)
models = sorted({r["model"] for r in runs}, key=lambda m: -counts[(m, "hijacked")] / max(1, counts[(m, "hijacked")] + counts[(m, "resisted")]))
fig, ax = plt.subplots(figsize=(11, 6.5))
left = [0] * len(models)
for kind, color in KIND_COLORS.items():
    vals = [counts[(m, kind)] for m in models]
    ax.barh(models, vals, left=left, color=color, label=kind)
    left = [a + b for a, b in zip(left, vals)]
ax.invert_yaxis(); ax.set_xlabel("runs"); ax.tick_params(axis="y", labelsize=7); ax.legend(fontsize=8)
ax.set_title("AgentDojo runs per agent (sorted by how often the attack succeeded)")
plt.tight_layout(); plt.savefig(OUT / "data.png", dpi=110); plt.show()

# %%
PROBE_COLORS = {"raw": "#898781", "jepa": "#2a78d6", "raw+jepa": "#1baf7a"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
for ax, key in zip(axes, ["AUC vs resisted", "AUC vs benign"]):
    for kind in COLS:
        ax.plot(LABELLED, [q1[(n, kind)][key] for n in LABELLED], marker="o", color=PROBE_COLORS[kind], label=f"probe on {kind}")
    ax.axhline(baseline[key], color="#e34948", ls="--", lw=1, label="no labels: seen-before")
    ax.set_xscale("log"); ax.set_xlabel("labelled runs"); ax.set_title(f"Hijacked {key}"); ax.set_ylim(0.5, 1); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "labels.png", dpi=110); plt.show()

# %%
fig, ax = plt.subplots(figsize=(8, 3.4))
names = list(q2)
vals = [q2[n]["AUC vs resisted"] for n in names]
bars = ax.barh(names, vals, color=["#c3c2b7", "#9fc5f0", "#7fd8b8", "#1baf7a"])
for b, v in zip(bars, vals):
    ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
ax.axvline(0.5, color="#898781", ls="--", lw=1)
ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("AUC hijacked vs resisted (real test runs)")
ax.set_title("Trained on synthetic data, tested on real agents")
plt.tight_layout(); plt.savefig(OUT / "transfer.png", dpi=110); plt.show()

# %%
items = sorted(per_model.items(), key=lambda kv: kv[1][0])
fig, ax = plt.subplots(figsize=(8, 0.32 * len(items) + 1))
ax.barh([m for m, _ in items], [a for _, (a, _) in items], color="#1baf7a")
for i, (_, (a, n)) in enumerate(items):
    ax.text(a + 0.005, i, f"{a:.2f} ({n})", va="center", fontsize=7)
ax.set_xlim(0.5, 1.05); ax.tick_params(axis="y", labelsize=7)
ax.set_xlabel("AUC hijacked vs resisted (number of hijacked test runs)")
ax.set_title("How well hijacks are caught, per agent")
plt.tight_layout(); plt.savefig(OUT / "per_model.png", dpi=110); plt.show()

# %%
steps_h, losses, spreads = zip(*history)
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].plot(steps_h, losses, color="#2a78d6"); axes[0].set_title("JEPA prediction loss")
axes[1].plot(steps_h, spreads, color="#1baf7a"); axes[1].set_title("Embedding spread (0 = collapsed)")
for ax in axes:
    ax.set_xlabel("step"); ax.set_ylim(bottom=0); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT / "curves.png", dpi=110); plt.show()

# %% [markdown]
# ## One run, call by call
#
# The best probe's score for every call of a hijacked run, next to a resisted
# run of the same task and the same attack.

# %%
def call_scores(F_train, M_calls, y, epochs=300):
    """Same probe as fit_probe, but returns per-call scores."""
    torch.manual_seed(0)
    mu, sd = F_train[M_calls].mean(0), F_train[M_calls].std(0) + 1e-6
    lin = nn.Linear(F_train.size(-1), 1).to(DEVICE)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=1e-3)
    pos_weight = ((1 - y).sum() / y.sum()).clamp(max=50)
    for _ in range(epochs):
        s = lin((F_train - mu) / sd).squeeze(-1).masked_fill(~M_calls, -1e4).max(1).values
        loss = F.binary_cross_entropy_with_logits(s, y, pos_weight=pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lambda F_any: lin((F_any - mu) / sd).squeeze(-1).detach()


scorer = call_scores(FEATS["train"], MC["train"], LABEL["train"])
benign_val = torch.tensor([r["kind"] == "benign" for r in splits["val"]], device=DEVICE)
THRESHOLD = scorer(FEATS["val"]).masked_fill(~MC["val"], float("-inf")).max(1).values[benign_val].quantile(0.95).item()
test_calls = scorer(FEATS["test"])
T = splits["test"]
same = lambda q, r: (q["suite"], q["user_task"], q["injection_task"]) == (r["suite"], r["user_task"], r["injection_task"])
i = next(i for i, r in enumerate(T) if r["kind"] == "hijacked" and r["model"] == "gpt-4o-2024-05-13"
         and 3 <= len(r["steps"]) <= 8 and any(q["kind"] == "resisted" and same(q, r) and len(q["steps"]) <= 8 for q in T))
j = next(j for j, q in enumerate(T) if q["kind"] == "resisted" and same(q, T[i]) and len(q["steps"]) <= 8)
def plot_pair(scores, threshold, name, probe_label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, n in zip(axes, (i, j)):
        r = T[n]
        s = scores[n, :min(len(r["steps"]), MAXSTEPS)].cpu()
        ax.bar(range(len(s)), s, color=["#e34948" if v > threshold else "#2a78d6" for v in s])
        ax.axhline(threshold, color="#898781", ls="--", lw=1)
        ax.set_xticks(range(len(s)), [st["call"][5:45] for st in r["steps"][:len(s)]], rotation=20, ha="right", fontsize=7)
        ax.set_title(f"{r['kind']} ({r['model']})", fontsize=9)
    axes[0].set_ylabel(f"hijack score ({probe_label})")
    fig.suptitle(f"Same task, same attack: \"{T[i]['task'][:70]}\". Red = flagged", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig(OUT / name, dpi=110); plt.show()


plot_pair(test_calls, THRESHOLD, "example.png", "linear probe, raw+jepa")

# %% [markdown]
# ## Linear vs whole-run probe

# %%
Q4N = [250, 1000, len(splits["train"])]
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for probe_name, table, ls, kinds in [("linear", q1, ":", ["raw", "raw+jepa"]), ("whole-run", q4, "--", ["raw", "raw+jepa"]),
                                     ("whole-run", q5, "-", ["raw+src", "raw+jepa+src"])]:
    for kind in kinds:
        ax.plot(Q4N, [table[(n, kind)]["AUC vs resisted"] for n in Q4N], ls=ls, marker="o",
                color=PROBE_COLORS[kind.replace("+src", "")], label=f"{probe_name} probe on {kind}")
ax.set_xscale("log"); ax.set_xlabel("labelled runs"); ax.set_ylim(0.6, 1); ax.grid(alpha=.3)
ax.set_title("Hijacked AUC vs resisted: context, then argument sources"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT / "q4.png", dpi=110); plt.show()

ctx = fit_run_probe("raw+jepa", torch.arange(len(splits["train"]), device=DEVICE))
ctx_threshold = ctx("val").max(1).values[benign_val].quantile(0.95).item()
plot_pair(ctx("test"), ctx_threshold, "example_context.png", "whole-run probe, raw+jepa")

src_probe = fit_run_probe("raw+jepa+src", torch.randperm(len(splits["train"]), generator=torch.Generator().manual_seed(0))[:1000].to(DEVICE))
src_threshold = src_probe("val").max(1).values[benign_val].quantile(0.95).item()
plot_pair(src_probe("test"), src_threshold, "example_source.png", "whole-run probe + argument sources")
