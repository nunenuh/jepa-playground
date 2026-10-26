# JEPA research landscape: the goal, the blockers, and security

A snapshot (October 2026) of where JEPA research stands, what it still can't
do, and who (if anyone) uses it for security. Searched: the web, arXiv,
Hugging Face papers/models/datasets and GitHub. Work that isn't indexed yet
may be missing.

## The holy grail

Yann LeCun's goal for JEPA: **a machine that learns how the world works by
watching, like babies and animals do, then plans actions to reach goals**:
common sense and planning instead of next-word prediction. The full design,
**H-JEPA**, is a *hierarchical* world model that predicts at several time
scales ("the next second" up to "the next hour") and plans with them inside
an objective-driven agent (perception, world model, memory, cost, actor).
**Nobody has built H-JEPA yet.**

## The blockers

| # | Problem | Status | Seen in this repo |
|---|---|---|---|
| 1 | **Collapse**: the model cheats by mapping everything to the same latent | Mostly solved: EMA / stop-grad, LeJEPA's SIGReg | [02](02_collapse/) |
| 2 | **Hierarchy**: abstract, multi-time-scale prediction | **Open**, the core missing piece | |
| 3 | **Long-horizon planning** | Short horizons only; V-JEPA 2-AC, LeWAM push this | |
| 4 | **Uncertainty**: many possible futures | Open; a deterministic predictor gives one "average" future; probabilistic versions are toy-scale | |
| 5 | **Fine detail and motion** (small objects, velocity) | Open: IntPhys 2 near chance (humans ~96%); single-frame latents carry no velocity | [07](07_physical_anomalies/), [08](08_intphys2/) |
| 6 | **Action data**: learning to act from passive video | Partly: V-JEPA 2 pretrains on video, fine-tunes on a little robot data | |
| 7 | **No likelihood**: how "normal" is a sample? | New: JEPA-SCORE reads a density out of any trained JEPA | |
| 8 | **Language and reasoning** | Early: LLM-JEPA, JEPA-Reasoner, DLLM-JEPA | |
| 9 | **Interpretability** of the latents | Early | |

What JEPA can't do yet, in one line: it is good at learning "what normal looks
like" without labels; it is weak at fine detail, generating anything,
uncertainty, and long-horizon reasoning.

Lessons from our own experiments that match the literature:

- **Surprise ≠ anomaly.** Unusual objects doing normal things aren't
  surprising ([06](06_vjepa2_video/): "seen before?" beat surprise).
- **Predictable-but-wrong futures fool it zero-shot.** Freeze and reverse were
  *less* surprising than normal video until the predictor was adapted
  ([07](07_physical_anomalies/)); the "Velocity Blind Spot" paper explains why.
- **Small, local breaks in varied scenes get washed out**
  ([08](08_intphys2/): near chance, as in the IntPhys 2 paper).

## JEPA for security: who does it?

**No peer-reviewed paper found that applies JEPA to security** (agents,
intrusion, malware, fraud, prompt injection). What exists:

### Papers (closest)

| Paper | What it does | Security link |
|---|---|---|
| [JEPA for autonomous vehicle safety & security](https://www.sciencedirect.com/science/article/abs/pii/S1474034626002909) (survey, 2026) | Lists intrusion detection, adversarial robustness and privacy as JEPA opportunities | Ideas only, no experiments |
| [JEPA-SCORE: Gaussian Embeddings](https://huggingface.co/papers/2510.05949) (Balestriero, Ballas, Rabbat, LeCun, 2025) | Any trained JEPA provably estimates data density, usable for outlier detection | Usable for anomaly detection; not tested on security |
| [T-SAR-JEPA](https://arxiv.org/abs/2606.05700) (2026) | Temporal anomaly detection in radar by latent prediction | Same recipe as ours, not security |
| [UI-JEPA](https://huggingface.co/papers/2409.04081) (Apple, 2024) | User intent from on-screen activity | Closest to monitoring agents |
| [Frozen in a Frame: Velocity Blind Spot](https://huggingface.co/papers/2610.04585) (2026) | JEPA world models can't see velocity from single frames; TI-JEPA fixes it | A failure mode an attacker could exploit |

### Code (GitHub)

| Repo | What | Notes |
|---|---|---|
| [nkeightly/zero-shot-jepa-anomaly-detection](https://github.com/nkeightly/zero-shot-jepa-anomaly-detection) | JEPA (TCN encoder, EMA) + latent diffusion scoring for **network intrusion** on NSL-KDD | Student course project; the only JEPA intrusion-detection code found |
| [nicolasmelo1/jepa-fraudinha](https://github.com/nicolasmelo1/jepa-fraudinha) | A JEPA model for **fraud detection** (Brazilian "Rinha de Backend" contest) | Hobby project |
| [itsanderz/jepa-score-eval](https://github.com/itsanderz/jepa-score-eval) | Independent test of JEPA-SCORE for out-of-distribution detection | JEPA-SCORE loses to Mahalanobis / k-NN on DINOv2 |
| [AIChidera/ijepa-layer-probe](https://github.com/AIChidera/ijepa-layer-probe) | I-JEPA layers for industrial visual anomaly detection (MVTec AD, VisA) | Not security, but strong anomaly results |
| [nunenuh/jepa-playground](https://github.com/nunenuh/jepa-playground) (this repo) | JEPA guardrail for AI agent tool calls ([04](04_agent_guardrail/), [05](05_real_agents/)) | |

### Hugging Face

No JEPA model or dataset for security, intrusion or anomaly detection found.
JEPA models on the Hub are general-purpose encoders (I-JEPA, V-JEPA 2,
domain JEPAs for audio, EEG, ECG, jets, satellites).

### The gap

JEPA for **agent security** (detecting hijacked or misbehaving AI agents
from what they do) is open. The pieces that would make it work line up with
open JEPA problems: uncertainty (4), fine-grained local breaks (5) and a
calibrated "how normal is this" score (7, JEPA-SCORE).

## Newest breakthroughs (candidates for next experiments)

| Paper | Why it matters |
|---|---|
| [LeWAM](https://arxiv.org/abs/2610.12818) (Oct 2026), [code](https://github.com/MinghaoFu/lewam), [data](https://huggingface.co/LeWAM) | One JEPA for world model + actions (flow matching); planning 28.6% → 89.7%. Tasks: Push-T, Cube, Two-Room, Scene, Puzzle, real robot |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482) (2026) | Next video JEPA from Meta |
| [LeJEPA](https://huggingface.co/papers/2511.08544) (2025) | Collapse prevention without heuristics (SIGReg) |
| [JEPA-Reasoner](https://huggingface.co/papers/2512.19171) (2025) | Latent reasoning separate from token generation |
