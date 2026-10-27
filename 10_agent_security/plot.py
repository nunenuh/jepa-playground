import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("results.json"))
q1, q3 = r["q1"], r["q3"]
rows = [  # label, raw, raw+jepa, jepa  (AUC hijacked vs resisted)
    ("AgentDojo test (in-domain, 05)", [q1[f"probe, {k}"]["AgentDojo test"] for k in ("raw", "raw+jepa", "jepa")]),
    ("InjecAgent, all runs", [q1[f"probe, {k}"]["InjecAgent"] for k in ("raw", "raw+jepa", "jepa")]),
    ("InjecAgent, runs where the agent acted", [q3[f"probe, {k}, runs where the agent acted"] for k in ("raw", "raw+jepa", "jepa")]),
]
colors = {"raw": "#2f7ed8", "raw+jepa": "#7b4fd6", "jepa": "#e4572e"}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, metric, title in [(axes[0], "AUC vs resisted", "AUC, hijacked vs resisted"),
                          (axes[1], "recall hijacked", "hijacks caught at the AgentDojo alarm line")]:
    for i, (label, ms) in enumerate(rows):
        for j, k in enumerate(("raw", "raw+jepa", "jepa")):
            v = ms[j][metric]
            ax.barh(i + (j - 1) * 0.26, v, height=0.24, color=colors[k], label=k if i == 0 else None)
            ax.text(v + 0.01, i + (j - 1) * 0.26, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, 1.12)
t = q3["trivial: agent made its own call"]
axes[0].axvline(t["AUC vs resisted"], color="k", ls="--", lw=1, label="trivial rule: agent made a call")
axes[1].axvline(t["recall hijacked"], color="k", ls="--", lw=1)
axes[0].axvline(0.5, color="#9aa5b1", ls=":", lw=1, label="chance")
axes[0].set_yticks(range(len(rows)), [l for l, _ in rows])
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, loc="lower center", ncol=5)
fig.suptitle("Hijack detector trained on AgentDojo, tested on InjecAgent without retraining", fontsize=11)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig("transfer.png", dpi=110)

# score histogram: where InjecAgent runs land relative to the AgentDojo alarm line
h = r["hist"]
edges = h["edges"]
fig, ax = plt.subplots(figsize=(8, 3.6))
for k, c in [("benign", "#9aa5b1"), ("resisted", "#2f7ed8"), ("hijacked", "#e4572e")]:
    n = sum(h[k])
    ax.stairs([v / n for v in h[k]], edges, fill=True, alpha=0.55, color=c, label=f"{k} ({n} runs)")
ax.axvline(h["threshold"], color="k", ls="--", lw=1, label="alarm line (5% of AgentDojo benign)")
ax.set_xlabel("run score (probe on raw+jepa, trained on AgentDojo only)")
ax.set_ylabel("share of runs")
ax.set_title("InjecAgent runs: hijacks score higher, but mostly below the transferred alarm line", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("scores.png", dpi=110)

# one task, two behaviours
ex, thr = r["example"], r["example_threshold"]
fig, axes = plt.subplots(2, 1, figsize=(10, 4.6))
for ax, run in zip(axes, ex):
    labels = [c.replace("CALL ", "").replace("https://www.", "")[:58] + ("…" if len(c) > 63 else "") for c in run["calls"]]
    cols = ["#9aa5b1"] + ["#e4572e" if run["kind"] == "hijacked" else "#2f7ed8"] * (len(labels) - 1)
    ax.barh(range(len(labels)), run["scores"], color=cols)
    ax.axvline(thr, color="k", ls="--", lw=1)
    ax.set_yticks(range(len(labels)), [("user's call:  " if i == 0 else "agent's call: ") + l for i, l in enumerate(labels)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(-2, 3)
    inj = run["result"].split("'snippet': '")[-1][:58]
    ax.set_title(f"{run['kind']}: injected text \"{inj}...\"", fontsize=8.5, loc="left")
    for i, v in enumerate(run["scores"]):
        ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
axes[-1].set_xlabel("call score (dashed: alarm line from AgentDojo)")
fig.suptitle(f"Task: \"{ex[0]['task']}\"", fontsize=10)
fig.tight_layout()
fig.savefig("example.png", dpi=110)
