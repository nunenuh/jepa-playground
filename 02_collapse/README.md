# 02 · Collapse on purpose

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/02_collapse/collapse.ipynb)

JEPA has a cheat: if the encoder maps **every image to the same embedding**,
the predictor can output that constant and the loss is ~0. That is
*collapse*, and the model learns nothing.

Here the same MNIST JEPA from [experiment 01](../01_mnist_jepa/) is trained
four ways to see what stops the cheat:

| Run | Target comes from | Gradient into target? | Extra loss |
|---|---|---|---|
| `ema+stopgrad` | EMA copy of the encoder (01 setup) | no | – |
| `no-stopgrad` | the encoder itself | **yes** | – |
| `stopgrad-only` | the encoder itself, detached | no | – |
| `sigreg` | the encoder itself | **yes** | SIGReg (LeJEPA) |

## Results

5 epochs per run, Colab T4, about 70 s per run.

![collapse](assets/collapse.png)

| Run | Final loss | Spread | Linear probe |
|---|---|---|---|
| `ema+stopgrad` | 0.027 | 0.388 | **61.4%** |
| `no-stopgrad` | 0.0005 | 0.005 | 22.6% |
| `stopgrad-only` | 0.008 | 0.027 | 17.1% |
| `sigreg` (weight 0.05) | 0.475 | 0.883 | 57.8% |
| random encoder, no training | – | – | 58.7% |

Raw numbers: [`assets/results.json`](assets/results.json).

## What it shows

- **Collapse is real, and the loss hides it.** Without EMA the loss drops to
  almost 0 while spread falls to 0.005: every image gets nearly the same
  embedding. Probe accuracy ends up *below an untrained network*. In JEPA a
  tiny loss is a warning sign, not success.
- **Stop-gradient alone is not enough.** Detaching the target still collapses.
  What saves the 01 setup is the *slowly moving* EMA target.
- **EMA + stop-gradient works**: the only trick-based run that beats random,
  and its spread keeps growing. (5 epochs here; 01 reaches 82.7% with 10.)
- **SIGReg needs the right weight.** It never collapses, but at weight 0.05 the
  "be Gaussian" term (~0.47) swamps the prediction loss (~0.01), so the model
  learns little. A small sweep:

  | `SIGREG_WEIGHT` | Spread | Eff. rank | Linear probe |
  |---|---|---|---|
  | 0.05 | 0.883 | 32.6 | 57.8% |
  | **0.005** | 0.365 | 11.5 | **63.1%** (beats EMA) |
  | 0.001 | 0.207 | 5.4 | 48.1% (too weak, dimensions shrink) |

  So SIGReg replaces both tricks (no EMA, no stop-gradient) with one honest
  loss term, but it has one knob to tune.

- **Effective rank can mislead.** The collapsed runs score ~105: what is left
  of their embeddings is tiny random noise, which spreads evenly over all
  dimensions. Trust spread and probe accuracy first.

## Experiments to try

| Change | What you'll see |
|---|---|
| `SIGREG_WEIGHT = 0.005` | SIGReg beats EMA (see sweep above) |
| `EPOCHS = 20` | Whether SIGReg keeps its lead over EMA |
| `EMA = 0.9` / `EMA = 0.9999` | Target moves too fast (collapse risk) / too slow (learns slowly) |

## Run

Click the Colab badge above, pick a GPU runtime, then Run all. Or locally:

```bash
uv run --with torch --with torchvision --with matplotlib \
    python 02_collapse/collapse.py
```
