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
