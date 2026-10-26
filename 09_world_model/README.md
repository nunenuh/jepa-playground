# 09 · A JEPA world model that plans (LeWorldModel)

🔄 In progress. Part 1: run a trained JEPA world model in a simulator on a
laptop GPU. Part 2: train our own on a Colab A100, then plan with it on the
laptop: **78% success after 3.5 h of training, on par with the official
checkpoint (74%)**. Part 3: what the latent knows: **positions yes,
velocity no (from one frame)**, yet the predictor catches frozen, reversed and
teleported motion (AUC 0.93–1.00). Part 4: **play with it live** in a window
([`play.sh`](play.sh)).

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

## Part 3: what does the latent know? Velocity and impossible rollouts

[`probe.py`](probe.py), laptop GPU, ~10 min. Two questions:

**A. Velocity check** ([Velocity Blind Spot](https://huggingface.co/papers/2610.04585)).
Probes from the latent to positions and velocities. Data: 10,000 random
simulator states, each rendered twice 0.5 s apart (one action block), moved
by velocities drawn from the expert episodes; 8,000 to fit, 2,000 to test.
(Probes fit on the 21 validation episodes overfit: 15 episodes are too few.)

**B. Impossible rollouts.** The predictor sees 3 real frames and the actions,
and predicts the next latent; surprise = squared error to what came next.
Real next frames vs impossible ones: **frozen** (nothing moved), **reversed**
(the previous frame), **teleported** block (re-rendered in the simulator,
moved and turned). 440 clips from the validation episodes.

![probe](assets/probe.png)

| R² (MLP probe; 1 = perfect, 0 = nothing) | Official | Ours ep3 | Ours ep1 | Untrained |
|---|---|---|---|---|
| Agent position, 1 frame | 0.98 | 0.98 | 0.72 | ~0 |
| Block position, 1 frame | 0.99 | 0.97 | 0.95 | 0.77 |
| Block angle, 1 frame | 0.97 | 0.81 | 0.66 | 0.04 |
| **Agent velocity, 1 frame** | **~0** | **~0** | **~0** | ~0 |
| **Block velocity, 1 frame** | **~0** | **~0** | **~0** | ~0 |
| Agent velocity, 2 frames | 0.75 | 0.71 | 0.07 | ~0 |
| Block velocity, 2 frames | 0.85 | 0.85 | 0.75 | ~0 |

| Surprise AUC (impossible vs real) | Official | Ours ep3 | Ours ep1 | Untrained |
|---|---|---|---|---|
| Teleported block | 1.00 | 1.00 | 1.00 | 0.47 |
| Reversed | 1.00 | 0.99 | 0.90 | 0.49 |
| Frozen | 0.99 | 0.93 | 0.70 | 0.50 |
| Frozen, slow agent only | 0.98 | 0.91 | 0.73 | 0.49 |

![teleport](assets/example_teleport.png)

- **One frame knows *where*, not *how fast*.** Positions and angle are read
  almost perfectly, velocity not at all (R² ≈ 0): the Velocity Blind Spot,
  confirmed. No surprise: one picture can't show motion.
- **The predictor fixes it with history.** Two latents give velocity back
  (R² 0.71–0.85), and LeWM's predictor sees 3 frames, so it catches frozen and
  reversed motion (AUC 0.93–1.00), even when the agent moves slowly. That's
  what V-JEPA 2 missed in [07](../07_physical_anomalies/). The TI-JEPA fix
  (a separate motion code) isn't needed for Push-T.
- **Big jumps are easy, small ones need training.** Even the 1-epoch model
  catches a teleport; "frozen" is what separates 1 epoch (0.70) from 3 (0.93)
  and the official model (0.99).
- **Our weak spot: the block angle** (0.81 vs 0.97). A plausible reason the
  official checkpoint will be more reliable on harder goals.
- **Check the control.** An untrained encoder already reads block position
  (0.77): the grey block is a big patch of pixels. The gain from training is
  in agent position, angle and velocity.

Raw numbers: [`assets/results_probe.json`](assets/results_probe.json). The
re-rendered frames match the dataset's pixels (mean difference 0.3/255).

## Part 4: play with it (live window)

[`live.py`](live.py) / [`play.sh`](play.sh): Push-T in a window on your
desktop. The JEPA plans and pushes; you can interfere and watch its surprise.

```bash
./09_world_model/play.sh                         # our model, starting at validation episode 2
./09_world_model/play.sh --model quentinll/lewm-pusht --episode 11
```

| Key / mouse | What happens |
|---|---|
| (nothing) | The JEPA follows an expert episode as a chain of subgoals, each 25 steps ahead (the goal image, top right) |
| drag the T | Teleport it: impossible, so surprise spikes (orange line on the graph); the JEPA replans |
| F | Freeze the world for one block: the pointer "moves" but nothing happens, surprise spikes |
| SPACE | You drive: the pointer follows your mouse; the JEPA still predicts and shows surprise |
| G, Q / E | Your own goal at the mouse, turned with Q / E (experimental, see below) |
| R / ESC | Next episode / quit |

![live](assets/live_screenshot.png)

(Screenshot from the test mode: the T was teleported at the orange line.)

What it shows, measured with `live.py --eval-episodes 21`:

- **The planner matches the official evaluation.** Our CEM, on LeWM's own
  task (start mid-episode, goal 25 steps ahead, 50 steps): 21/30 = 70% (eval:
  78%). Replanning after every 2 blocks instead of 5 drops it to 43%: the plan
  always aims to arrive at the last step, so replanning early keeps pushing
  the arrival back.
- **Long tasks are the wall.** Following whole expert episodes as subgoal
  chains, it reaches 28 of 109 subgoals (26%) and finishes 2 of 21 episodes
  (2 and 11). One miss makes the next subgoal farther, and failures cascade.
- **Your own goals mostly fail** (1 of 20 goals 40 px / 20° away, even with the
  goal latent averaged over 16 pointer positions so the planner aims for the
  T, not the pointer): from a random start the pointer first has to walk
  around the T, which is beyond a 25-step plan. This is the long-horizon
  blocker in [RESEARCH_LANDSCAPE](../RESEARCH_LANDSCAPE.md) (hierarchical
  planning, H-JEPA, is the proposed fix).
- **Surprise works live.** Normal steps sit around 0.01–0.03; a teleport jumps
  to ~0.3 (10–20×).

Needs an X11 desktop. `play.sh` lets the container's root user draw on your
display (`xhost +SI:localuser:root`) and takes it back on exit.

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
