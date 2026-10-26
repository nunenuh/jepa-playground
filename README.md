# jepa-playground

Learning JEPA (Joint-Embedding Predictive Architecture) from scratch, one small
experiment at a time — ending in a JEPA-based guardrail for AI agent tool calls.

**The core idea:** hide part of the input, and predict the *embedding* of the
hidden part — not its pixels or words. Predicting a summary forces the model to
learn meaning (shape, structure, what happens next) and ignore noise.

## Experiments

| # | Experiment | What it teaches | Status | Open |
|---|---|---|---|---|
| 01 | [MNIST JEPA](01_mnist_jepa/) | Latent prediction, context vs target encoder, EMA, linear probe | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/01_mnist_jepa/mnist_jepa.ipynb) |
| 02 | [Collapse on purpose](02_collapse/) | Why JEPA needs anti-collapse (EMA, stop-gradient, SIGReg) | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/02_collapse/collapse.ipynb) |
| 03 | [Moving-ball world model](03_moving_ball/) | Predicting the future in latent space; anomaly = surprise | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/03_moving_ball/moving_ball.ipynb) |
| 04 | [Agent tool-call guardrail](04_agent_guardrail/) | JEPA on agent trajectories to catch prompt injection hijacks | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/04_agent_guardrail/agent_guardrail.ipynb) |
| 05 | [Guardrail on real agents](05_real_agents/) | The 04 recipe on 29 real LLM agents (AgentDojo); synthetic → real transfer | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/05_real_agents/real_agents.ipynb) |
| 06 | [V-JEPA 2 on real video](06_vjepa2_video/) | Zero-shot CCTV anomaly detection with Meta's 300M V-JEPA 2 world model (UCSD Ped2) | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/06_vjepa2_video/vjepa2_anomaly.ipynb) |
| 07 | [Physically impossible video](07_physical_anomalies/) | V-JEPA 2 surprise on real clips edited to break physics (teleport, reverse, freeze, vanish) | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/07_physical_anomalies/physical_anomalies.ipynb) |
| 08 | [IntPhys 2](08_intphys2/) | V-JEPA 2 surprise on Meta's intuitive-physics benchmark (permanence, immutability, continuity, solidity) | ✅ | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/08_intphys2/intphys2.ipynb) |
| 09 | [World model that plans](09_world_model/) | LeWorldModel: a JEPA world model choosing actions by planning in latent space (Push-T simulator, Docker on a laptop GPU) | 🔄 | — |

What's next and why: [ROADMAP.md](ROADMAP.md).

## Results

| # | Before training | After training | Setup |
|---|---|---|---|
| 01 | 56.1% (random encoder) | 82.7% | 10 epochs, no labels, RTX A2000 (~2 min). [Details and figures](01_mnist_jepa/README.md) |
| 02 | 58.7% (random encoder) | EMA 61.4% · SIGReg 63.1% · no anti-collapse 17–23% | 5 epochs per run, Colab T4. [Details and figures](02_collapse/README.md) |
| 03 | anomaly AUC 0.49 | anomaly AUC 1.00 (teleport & reverse) | 3000 steps, Colab T4 (~1 min), no anomaly labels. [Details and figures](03_moving_ball/README.md) |
| 04 | hijack AUC 0.83 (probe on raw text) | **0.88** (probe on raw + JEPA features); JEPA surprise alone 0.59 | AgentDrift, unseen tasks, Colab T4 (~3 min). [Details and figures](04_agent_guardrail/README.md) |
| 05 | hijacked vs resisted AUC 0.61 (raw, 50 labels) | **0.75** (JEPA, 50 labels) · **0.94** with argument sources (~80% caught at 5% false alarms; AUC 0.90–0.99 on unseen attack types) · 0.77 trained on synthetic only | AgentDojo, unseen tasks + unseen attacker goals, Colab T4. [Details and figures](05_real_agents/README.md) |
| 06 | anomaly AUC 0.474 (raw top-5 surprise, drowned by foliage) | **0.683** (spatially calibrated surprise) · **0.832** top-5 / **0.925** whole-frame (seen-before latent bank) · adapting the predictor: prediction −24% error, detection no better | UCSD Ped2, zero-shot + predictor fine-tune (`facebook/vjepa2-vitl-fpc64-256`), Colab T4 (~16 min). [Details and figures](06_vjepa2_video/README.md) |
| 07 | zero-shot surprise AUC 0.19 (reverse) / 0.32 (freeze) | **0.96–1.00** on all 4 types with the adapted predictor | Ped2 clips edited to break physics, Colab T4 (~17 min). [Details and figures](07_physical_anomalies/README.md) |
| 08 | pairwise accuracy 50.2% (ViT-L, one setting: chance) | **54.5%** ViT-H with the official protocol (held-out 54.2%), within noise of the paper's 57.5%; ViT-L ≈ ViT-H; adapting: no gain | IntPhys 2 main set, 506 pairs, Colab T4 + A100. [Details and figures](08_intphys2/README.md) |
| 09 | random actions: 0% success | **74%** (official LeWM checkpoint + CEM planning, 50 Push-T episodes) | Inference only, RTX A2000 laptop in Docker (~3 min). [Details](09_world_model/README.md) |

## Run

```bash
# local
uv run --with torch --with torchvision --with matplotlib --with scikit-learn \
    python 01_mnist_jepa/mnist_jepa.py

# or click the Colab badge above (or use the Google Colab extension in VS Code)
```

`.py` files are the source; `.ipynb` notebooks are generated with
`jupytext --to ipynb <file>.py`.

Every experiment follows [EXPERIMENT_STANDARD.md](EXPERIMENT_STANDARD.md):
architecture diagram, results with a baseline, intuitive pictures, raw numbers.

## Reading

Where JEPA research stands, its open problems, and who uses it for security: [RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md).

1. Meta AI blog — I-JEPA
2. Assran et al., 2023 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA)
3. Balestriero & LeCun, 2025 — *LeJEPA*
