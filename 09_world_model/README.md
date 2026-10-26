# 09 · A JEPA world model that plans (LeWorldModel)

🔄 In progress. Part 1 (below): run a trained JEPA world model in a simulator
on a laptop GPU. Next: train our own on Colab, the velocity check, and
impossible rollouts (see [ROADMAP](../ROADMAP.md)).

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
| Random actions | **0%** | 35 s |
| **LeWM + CEM planning** | **74%** | 201 s |

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
  paths under `$STABLEWM_HOME/checkpoints`, or `random`.
- Simulators render off-screen (pygame / EGL), so no display is needed; the
  videos land in `09_world_model/assets/videos/`.
- Pins, so it keeps working: `stable-worldmodel` at a fixed commit, PyTorch
  2.7.1 built for CUDA 12.6 (works with driver 560), `transformers < 5`
  (the released checkpoints use the 4.x ViT parameter names).
