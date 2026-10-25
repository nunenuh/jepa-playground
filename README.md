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

## Results

| # | Before training | After training | Setup |
|---|---|---|---|
| 01 | 56.1% (random encoder) | 82.7% | 10 epochs, no labels, RTX A2000 (~2 min). [Details and figures](01_mnist_jepa/README.md) |
| 02 | 58.7% (random encoder) | EMA 61.4% · SIGReg 63.1% · no anti-collapse 17–23% | 5 epochs per run, Colab T4. [Details and figures](02_collapse/README.md) |
| 03 | anomaly AUC 0.49 | anomaly AUC 1.00 (teleport & reverse) | 3000 steps, Colab T4 (~1 min), no anomaly labels. [Details and figures](03_moving_ball/README.md) |
| 04 | hijack AUC 0.83 (probe on raw text) | **0.88** (probe on raw + JEPA features); JEPA surprise alone 0.59 | AgentDrift, unseen tasks, Colab T4 (~3 min). [Details and figures](04_agent_guardrail/README.md) |
| 05 | hijacked vs resisted AUC 0.61 (raw, 50 labels) | **0.75** (JEPA, 50 labels) · **0.93** with argument sources (1,000 labels, 71% caught) · 0.77 trained on synthetic only | AgentDojo, unseen tasks + unseen attacker goals, Colab T4. [Details and figures](05_real_agents/README.md) |

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

1. Meta AI blog — I-JEPA
2. Assran et al., 2023 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA)
3. Balestriero & LeCun, 2025 — *LeJEPA*
