# 09 · A JEPA world model that plans (LeWorldModel)

🔄 In progress. Part 1: run a trained JEPA world model in a simulator on a
laptop GPU. Part 2: train our own on a Colab A100, then plan with it on the
laptop: **78% success after 3.5 h of training, on par with the official
checkpoint (74%)**. Next: the velocity check and impossible rollouts (see
[ROADMAP](../ROADMAP.md)).

Experiments 01–08 only *watched*: they predicted what comes next. A world
model that **plans** also takes actions: given the current camera image and a
goal image, it imagines what each candidate action sequence would do, in
latent space, and executes the one that ends closest to the goal. No reward,
no reinforcement learning.

## How it plans

```
camera image ─► encoder ─► latent ─┐
                                   ├─► predictor(latent, actions) ─► imagined future latent
300 candidate action sequences ────┘                                   │
                                                                       ▼
goal image ─► encoder ─► goal latent ──────────► cost = distance(imagined, goal)
                                                                       │
CEM: keep the 30 cheapest, resample around them, 30 rounds ◄───────────┘
execute the first 5 steps of the best plan, look again, replan
```

Model: [LeWorldModel](https://github.com/lucas-maes/le-wm) (LeCun, Balestriero
et al., 2026), ~15M parameters (ViT-tiny encoder + transformer predictor),
trained end to end from pixels with SIGReg (the anti-collapse loss from
[02](../02_collapse/)). Official checkpoint
[`quentinll/lewm-pusht`](https://huggingface.co/quentinll/lewm-pusht).

## Part 1: planning in Push-T on a laptop

**Push-T**: push the grey T so it matches a goal pose, using the blue pointer.
Following LeWM's evaluation settings: start from a state of an expert
episode, the goal is the state **25 steps later**, the planner gets **50
steps**. 50 episodes, same starts for every policy.

| Policy | Success | Time (50 episodes) |
|---|---|---|
| Random actions | **0–2%** | 21–35 s |
| **LeWM + CEM planning** | **74%** | 189–201 s |

(Two runs; the planner samples, so success moves by a few points between
runs: random 0% then 2%, our epoch-2 model below 48% then 44%.)

RTX A2000 Laptop GPU (4 GB), ~15 MB of GPU memory. Raw numbers:
[`assets/results.json`](assets/results.json).

A successful episode (left: the agent; middle: what the expert did; right:
the goal), every 12th frame top to bottom:

![success](assets/example_success.png)

A failed one:

![fail](assets/example_fail.png)

Notes:

- The action and state normalisers are fit on the 21 Push-T validation
  episodes released with [H-JEPA](https://huggingface.co/datasets/jepa-world-models/h-jepa)
  (16 MB), not on LeWM's 13 GB training set; this is the main difference
  from the official evaluation, and may cost some success.
- The LeWM paper reports Push-T only as a figure (18% above PLDM), so there's
  no exact number to compare with yet.


## Part 2: training our own on Colab, planning on the laptop

Same model and recipe as the official checkpoint
([`train_colab.ipynb`](train_colab.ipynb), stable-worldmodel's
`scripts/train/lewm.py`): LeWM's Push-T expert data (18,685 episodes,
13 GB download, 46 GB unpacked), ViT-tiny encoder, SIGReg weight 0.09, batch
128. Only the budget differs: **3 short epochs, 13,791 steps** (the official
run is 100 epochs × 13,933 steps, ~100× more). Colab A100, ~3.5 h.

![training](assets/training.png)

| Model | Training steps | Push-T success (same 50 episodes) |
|---|---|---|
| Random actions | 0 | 2% |
| Ours, epoch 1 | 4,597 | 14% |
| Ours, epoch 2 | 9,194 | 44% |
| **Ours, epoch 3** | **13,791** | **78%** |
| Official LeWM | ~1.4M | 74% |

- **A JEPA world model learns to plan fast.** Success climbs 14% → 44% →
  78% while the held-out prediction error falls 0.080 → 0.022; with ~1% of
  the official budget it matches the official checkpoint (78% vs 74% is
  within the ±12-point noise of 50 episodes).
- **Better prediction → better planning, here.** Unlike 06 (where a better
  predictor didn't detect anomalies better), in Push-T planning lower
  prediction error turned straight into more successes: the planner *uses*
  the predictions, so their quality is what matters.
- **The bottleneck is data, not the GPU.** Training ran at 1.5 steps/s
  because each sample decompresses a block of frames from the HDF5 file; the
  GPU sat mostly idle. Loading the file into RAM first (`cat > /dev/null`)
  and 11 loader workers were enough to make it practical on Colab.
- **Caveat.** One training run (seed 3072), 50 evaluation episodes; the
  official checkpoint may still be stronger on harder settings (longer
  goals, other environments).

Raw numbers: [`assets/results.json`](assets/results.json) (planning),
[`assets/training.json`](assets/training.json) (training).

## Run it on your laptop (Docker)

Needs Docker with the NVIDIA runtime (`nvidia-container-toolkit`) and an
NVIDIA driver ≥ 560. Inference only; nothing is trained.

```bash
docker build -t jepa-wm 09_world_model            # ~15 GB image (PyTorch + simulators)
docker run --rm --gpus all \
    -v jepa-wm-data:/data \
    -v "$PWD/09_world_model/assets:/out" \
    jepa-wm --episodes 50                          # results.json + one mp4 per episode
```

- `--policies quentinll/lewm-pusht random`: Hugging Face repo ids, checkpoint
  folders under `$STABLEWM_HOME/checkpoints` (a `config.json` + one `.pt`,
  e.g. our Colab-trained model copied into the `jepa-wm-data` volume), or
  `random`.
- Simulators render off-screen (pygame / EGL), so no display is needed; the
  videos land in `09_world_model/assets/videos/`.
- Pins, so it keeps working: `stable-worldmodel` at a fixed commit, PyTorch
  2.7.1 built for CUDA 12.6 (works with driver 560), `transformers < 5`
  (the released checkpoints use the 4.x ViT parameter names).
