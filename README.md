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
| 02 | Collapse on purpose | Why JEPA needs anti-collapse (EMA, stop-gradient, SIGReg) | ⏳ | |
| 03 | Moving-ball world model | Predicting the future in latent space; anomaly = surprise | ⏳ | |
| 04 | Agent tool-call guardrail | JEPA on agent trajectories to catch prompt injection / exfiltration | ⏳ | |

## Results

| # | Probe accuracy before | Probe accuracy after | Setup |
|---|---|---|---|
| 01 | 56.1% (random encoder) | 82.7% | 10 epochs, no labels, RTX A2000 (~2 min) |

## Run

```bash
# local
uv run --with torch --with torchvision python 01_mnist_jepa/mnist_jepa.py

# or click the Colab badge above (or use the Google Colab extension in VS Code)
```

`.py` files are the source; `.ipynb` notebooks are generated with
`jupytext --to ipynb <file>.py`.

## Reading

1. Meta AI blog — I-JEPA
2. Assran et al., 2023 — *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA)
3. Balestriero & LeCun, 2025 — *LeJEPA*
