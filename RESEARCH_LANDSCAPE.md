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
The first working version appeared in October 2026:
[H-JEPA](https://huggingface.co/papers/2610.06805) (Zhang, Terver, Rabbat,
LeCun, Balestriero) trains a hierarchy of action-conditioned JEPAs, each
level predicting farther ahead, and plans top-down (Visual AntMaze success
18% → 73%). It is still simulation plus offline real-robot video, not the
full objective-driven agent.

## The blockers

| # | Problem | Status | Seen in this repo |
|---|---|---|---|
| 1 | **Collapse**: the model cheats by mapping everything to the same latent | Mostly solved: EMA / stop-grad, LeJEPA's SIGReg | [02](02_collapse/) |
| 2 | **Hierarchy**: abstract, multi-time-scale prediction | **Just started**: first H-JEPA (Oct 2026), simulation and offline robot video only | |
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
intrusion, malware, fraud, prompt injection). One preprint-stage project uses
a JEPA-style objective for LLM jailbreak defense (JEPA-Align, below). What
exists:

### Papers (closest)

| Paper | What it does | Security link |
|---|---|---|
| [JEPA for autonomous vehicle safety & security](https://www.sciencedirect.com/science/article/abs/pii/S1474034626002909) (survey, 2026) | Lists intrusion detection, adversarial robustness and privacy as JEPA opportunities | Ideas only, no experiments |
| [JEPA-SCORE: Gaussian Embeddings](https://huggingface.co/papers/2510.05949) (Balestriero, Ballas, Rabbat, LeCun, 2025) | Any trained JEPA provably estimates data density, usable for outlier detection | Usable for anomaly detection; not tested on security |
| [T-SAR-JEPA](https://arxiv.org/abs/2606.05700) (2026) | Temporal anomaly detection in radar by latent prediction | Same recipe as ours, not security |
| [UI-JEPA](https://huggingface.co/papers/2409.04081) (Apple, 2024) | User intent from on-screen activity | Closest to monitoring agents |
| *Predictive Representation Alignment Improves Generalization in LLM Safety* (JEPA-Align / PRA, Ozdincer, 2026), [results](https://huggingface.co/datasets/memo-ozdincer/jepa-qwen3-32b-pure-baselines-2026-05-25), [JepaData](https://huggingface.co/datasets/memo-ozdincer/jepadata) | JEPA-style objective: a predictor maps a jailbreak-rewritten prompt's representation to the clean prompt with the same intent; added to Circuit Breakers / CE-floor on Qwen3-32B, mean attack success 18.1% → 0.7% | **The only JEPA security work found**: LLM jailbreak defense (model internals, not agent behaviour) |
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

Only security-related JEPA artifacts: the JEPA-Align / PRA adapters and
results, and JepaData (269K adversarial trajectories), listed above. No JEPA
model for intrusion, malware or agent monitoring.
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

## The JEPA ecosystem outside papers (GitHub and Hugging Face)

Most-starred and most-downloaded JEPA work, October 2026.

### GitHub

| Repo | Stars | What |
|---|---|---|
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) | ~4.6k | LeWorldModel: stable end-to-end JEPA world model (the most-starred JEPA repo) |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) | ~4.2k | V-JEPA official code |
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) | ~3.5k | I-JEPA official code |
| [galilai-group/stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) | ~2.3k | Platform for reproducible world-model research and evaluation |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | ~0.8k | Meta's library of community JEPA examples |
| [Gen-Verse/JEPA-Anything](https://github.com/Gen-Verse/JEPA-Anything), [RSI-Anything](https://github.com/Gen-Verse/RSI-Anything) | ~0.4k / ~0.8k | One JEPA recipe across domains (vision, cells, control, weather...); self-improvement with world models |
| [ginwind/VLA-JEPA](https://github.com/ginwind/VLA-JEPA) | ~0.6k | Robot vision-language-action model with a JEPA world model (models on HF via LeRobot) |
| [facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms) | ~0.5k | What drives success in physical planning with JEPA world models |
| [MinghaoFu/lewam](https://github.com/MinghaoFu/lewam) | ~0.3k | LeWAM world + action model |
| [galilai-group/llm-jepa](https://github.com/galilai-group/llm-jepa) | ~0.3k | JEPA objective for LLMs |
| [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) | ~0.2k | Hierarchical JEPA world models |
| [facebookresearch/jepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics) | ~0.3k | Official intuitive-physics evaluation (used in [08](08_intphys2/) Part 2) |
| [keon/jepa](https://github.com/keon/jepa), [lucidrains/x-jepa](https://github.com/lucidrains/x-jepa) | ~0.3k / ~0.1k | Minimal and exploratory implementations, good for learning |
| [AbdelStark/awesome-jepa](https://github.com/AbdelStark/awesome-jepa), [AI-in-Transportation-Lab/awesome-jepa](https://github.com/AI-in-Transportation-Lab/awesome-jepa) | ~0.3k / ~0.2k | Curated lists |
| [NiluK/worldmodels101](https://github.com/NiluK/worldmodels101) | ~0.2k | Free interactive course on world models |

Domain JEPAs on GitHub: medicine (EchoJEPA, Brain-JEPA, Neuro-JEPA), driving
(Drive-JEPA, WA-JEPA), robots and drones (Chengling-PWM, JEPLO, SkyJEPA),
3D (Meta's Locate 3D), radar (SAR-JEPA), biology (terra), anomaly detection
(FoundAD, ICLR 2026).

### Hugging Face

| Kind | Examples |
|---|---|
| Official Meta models | `facebook/vjepa2-*` (ViT-L/H/g, used in [06](06_vjepa2_video/)–[08](08_intphys2/)), `facebook/ijepa_*` |
| World models / robotics | `jepa-world-models/h-jepa`, `lerobot/VLA-JEPA-*`, LeWM Push-T and Cube data (`quentinll/lewm-*`), `facebook/jepa-wms` data (gated), `LeWAM` |
| Signals and medicine | `braindecode/signal-jepa` (EEG, most-downloaded JEPA model), `NYUMedML/Neuro-JEPA`, `hamzashafiq/CardioState-Jepa`, CGM-JEPA |
| Audio and speech | `ltuncay/Audio-JEPA-base`, `labhamlet/speech-jepa-base` |
| Language and reasoning | Text-JEPA reasoning data, ARC-AGI-2 JEPA episodes, JEPA-CoT |
| Running on small devices | `jepacpp/*-GGUF`: [jepa.cpp](https://huggingface.co/jepacpp) runs V-JEPA 2, I-JEPA, LeJEPA, LeWM and H-JEPA in C/C++ (GGUF, like llama.cpp) |
| Security | JEPA-Align / PRA results and JepaData (above) |

Trend: JEPA has moved from image pretraining (2023) to **world models for
planning and robots** (2025–2026), and is spreading into medical signals and
language; tooling (stable-worldmodel, eb_jepa, jepa.cpp) is maturing.

## Scaling: what happens with big models and big GPUs

| Work | Scale | Finding |
|---|---|---|
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Meta, 2025) | ViT-L 300M → **ViT-g 1B**, 2M → 22M videos (1M+ hours) | Every ingredient helps but modestly: data ≈ +1 pt, model ≈ +1.5, longer training ≈ +0.8; the biggest jump came from higher resolution and longer clips (cumulative +4 pts) |
| [V-JEPA 2.1](https://huggingface.co/papers/2603.14482) (Meta, Mar 2026) | ViT-L → **ViT-G 2B**, + 142M images | Scaling helps on all tasks; dense loss + deep supervision unlock fine spatial features (+20 pts robot grasping vs V-JEPA 2-AC) |
| [RoboJEPA](https://huggingface.co/papers/2610.10515) (Meta, Oct 2026) | **8B predictor**, 12 robot embodiments | **First JEPA scaling law**: imagination error follows a power law in compute; planning improves predictably; largest JEPA predictor to date; zero-shot long-horizon planning on real robots |
| [LeVJEPA](https://huggingface.co/papers/2608.27395) (Aug 2026) | ViT-S/B/L | Same quality as V-JEPA 2 at **5.6–20.8× less compute** (SIGReg, no EMA, token dropping) |
| [SALT](https://huggingface.co/papers/2509.24317) (Apple, 2025) | video | Frozen teacher beats V-JEPA's accuracy-per-FLOP frontier |
| DINOv3 / Web-SSL (Meta, 2025, joint-embedding cousins) | **7B** on 1.7–2B images | Image SSL keeps improving to 7B and matches CLIP without text; very long training degraded dense features (needed a fix) |

Takeaways:

- **Understanding and planning scale**; RoboJEPA shows it's now *predictable*
  (power law), like LLMs.
- **Scale alone doesn't give intuitive physics**: IntPhys 2 stays near
  chance for larger models; our [08](08_intphys2/) Part 2 found ViT-L ≈ ViT-H.
- **Efficiency is the other race**: LeVJEPA/SALT get the same results with
  5–20× less compute.
- JEPA sizes (1–8B) are still far below LLMs (100B–1T+).
