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
# # 04 · More label-free baselines for Part A
#
# Part A compared JEPA surprise with a seen-before baseline. Here we add the
# other label-free scores used in [05 Q9](../05_real_agents/): Mahalanobis
# (raw text / JEPA latent), k-NN in JEPA latent space, and **JEPA-SCORE**
# (the density a SIGReg-trained JEPA learns implicitly: Σ log singular
# values of the encoder's Jacobian; low = unusual).
#
# To keep the setup identical, this runs `agent_guardrail.py` up to Part B
# (data, split, features, the Part A JEPA trained on normal trajectories),
# then scores every call with the new detectors.

# %%
import json
from pathlib import Path

HERE = Path(__file__).parent if "__file__" in globals() else Path("04_agent_guardrail")
exec((HERE / "agent_guardrail.py").read_text().split("# %% [markdown]\n# ## Part B")[0])

# %%
from torch.func import jacrev, vmap


def per_call(fn, part):
    X, M = feats[part]
    calls = X[:, CALL]
    return fn(calls.flatten(0, 1)).view(calls.shape[:2]).masked_fill(~M[:, CALL], float("-inf"))


def mahalanobis_fn(bank, eps=1e-3):
    mu = bank.mean(0)
    inv = torch.linalg.inv(torch.cov((bank - mu).T) + eps * torch.eye(bank.size(1), device=DEVICE))
    return lambda q: (((q - mu) @ inv) * (q - mu)).sum(-1)


@torch.no_grad()
def knn_fn(bank):
    return lambda q: torch.cat([torch.cdist(c, bank).min(-1).values for c in q.split(4096)])


def jepa_score_fn(enc):
    jac = vmap(jacrev(enc.net))
    def score(q):
        out = []
        for c in q.split(1024):
            s = torch.linalg.svdvals(jac(c))
            out.append(-torch.log(s.clamp_min(1e-12)).sum(-1).detach())   # minus log-density: high = unusual
        return torch.cat(out)
    return score


with torch.no_grad():
    Z_NORMAL = enc_a(NORMAL_CALLS)
maha_raw, maha_z, knn_z = mahalanobis_fn(NORMAL_CALLS), mahalanobis_fn(Z_NORMAL), knn_fn(Z_NORMAL)
with torch.no_grad():
    detectors = {
        "seen-before baseline": call_novelty,
        "JEPA surprise": call_surprise,
        "Mahalanobis (raw)": lambda p: per_call(maha_raw, p),
        "k-NN (JEPA)": lambda p: per_call(lambda q: knn_z(enc_a(q)), p),
        "Mahalanobis (JEPA)": lambda p: per_call(lambda q: maha_z(enc_a(q)), p),
    }
    results = {name: evaluate(fn)[0] for name, fn in detectors.items()}
results["JEPA-SCORE"] = evaluate(lambda p: per_call(jepa_score_fn(enc_a), p))[0]
show(results)

OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
(OUT / "results_baselines.json").write_text(json.dumps({"commit": COMMIT, "results": results}, indent=1))
