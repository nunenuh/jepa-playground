# Roadmap

What's done, what's next, and why. Based on the gaps found in
[RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md) (October 2026). Every new
experiment follows [EXPERIMENT_STANDARD.md](EXPERIMENT_STANDARD.md).

Status: ✅ done · 🔄 in progress · ⏳ next · 💡 later

## Done

| # | Experiment | Covers |
|---|---|---|
| 01 | MNIST JEPA | Latent prediction, EMA, linear probe |
| 02 | Collapse | Blocker 1: collapse (EMA, stop-grad, SIGReg) |
| 03 | Moving ball | Latent world model, surprise as anomaly |
| 04 | Agent guardrail (synthetic) | JEPA on agent tool-call trajectories |
| 05 | Guardrail on real agents | AgentDojo tool-call traces of 29 real LLM agents, behaviour only |
| 06 | V-JEPA 2 on CCTV | Surprise vs "seen before" on real video |
| 07 | Physically impossible video | Blocker 5: predictable-but-wrong futures |
| 08 | IntPhys 2 | Blocker 5: intuitive physics, near chance |

## Plan

### 1. ✅ 08 Parts 2–3: match the IntPhys 2 paper

The paper reports V-JEPA 2 ViT-huge at 57.5% (best of ~12 settings); Part 1
got 50.2% with ViT-L and one setting.

- [x] Read the official evaluation protocol ([jepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics))
- [x] Sweep script: 2 frame rates × 3 context lengths × 2 patch scores × 2 time aggregations = 24 settings ([`intphys2_sweep.py`](08_intphys2/intphys2_sweep.py))
- [x] Smoke test on Colab A100
- [x] Full run: ViT-huge, then ViT-large (sweep: best 53.8% / 54.0%, held-out 50.4% / 53.4%)
- [x] Report best setting (paper's way) **and** held-out (setting picked on the other half of the scenes)
- [x] Part 3: the IntPhys 2 authors' own protocol (48-frame windows, 12–42 context frames, max-context mode): **54.5%**, held-out 54.2%, within noise of 57.5%
- [x] ViT-huge vs ViT-large: size doesn't explain the gap (ViT-L ≈ ViT-H)
- [x] README Parts 2–3, root README results row, merge to `main`

### 2. ✅ Quick wins: JEPA-SCORE and simple baselines

Blocker 7 (no "how normal is this" score) and honest baselines. An
independent test found JEPA-SCORE losing to Mahalanobis on images
([jepa-score-eval](https://github.com/itsanderz/jepa-score-eval)), so check it
honestly.

- [ ] JEPA-SCORE ([paper](https://huggingface.co/papers/2510.05949)): density from the encoder's Jacobian, no retraining
  - [x] in 05 (Q9): works but loses to k-NN (AUC 0.64–0.66 vs 0.76); no label-free score catches > 18% of hijacks at 5% false alarms
  - [ ] ~~in 06~~ skipped: needs a SIGReg-trained encoder; V-JEPA 2 isn't
- [x] Mahalanobis and k-NN baselines in 04 (seen-before still best, JEPA-SCORE at chance), 05 (Q9), 06 (Mahalanobis 0.909 top-5, beats seen-before 0.832)
- [x] Add rows to each README's results table (keep failures)

Effort: ~half a day. GPU: T4 is enough.

### 3. 🔄 09: a world model that plans (LeWorldModel on Push-T)

The biggest trend: JEPA world models that choose actions. The repo has no
actions or planning yet. LeWM is the most-starred JEPA repo, small, and has
data and checkpoints on Hugging Face.

- [x] Docker image to run JEPA world models in simulators on the laptop GPU ([`09_world_model/Dockerfile`](09_world_model/Dockerfile))
- [x] Official LeWM checkpoint planning in Push-T: **74%** vs random 0% (50 episodes)
- [x] Train our own LeWM on Push-T on Colab (3.5 h A100, 13,791 steps), plan with it on the laptop: **78%** vs official 74%
- [ ] Planning success vs a baseline (copy the demonstrations / random actions)
- [ ] **Velocity check** ([Velocity Blind Spot](https://huggingface.co/papers/2610.04585)): linear probe for position vs velocity on the latents; ties back to 07's freeze/reverse failure
- [ ] Optional: TI-JEPA fix (pose code + motion code), does planning improve?
- [ ] Surprise on impossible Push-T rollouts (teleporting block): links 03/07 to planning

Effort: ~1 day. GPU: A100, a few hours.

### 4. ⏳ 10: agent security beyond 05

05 already detects hijacks from AgentDojo tool-call traces, but needs labels
(AUC 0.75 with 50 labels, 0.94 with argument sources) and is tested on one
benchmark. Nobody else has published JEPA for agent security, so this is the
repo's unique contribution.

- [x] **No labels** (05 Q9): best label-free ranking 0.76 ≈ a 50-label probe, but alarms catch ≤ 18%
- [ ] **PRA-style paired views** ([JEPA-Align](https://huggingface.co/datasets/memo-ozdincer/jepa-qwen3-32b-pure-baselines-2026-05-25)): train so that an attacked run and a clean run of the same user task map close together; a run that drifts away is suspicious
- [ ] **New benchmark, no retraining**: train on AgentDojo, test on another agent-security benchmark (Agent Security Bench or InjecAgent); does it transfer?
- [ ] Newer threats: malicious MCP servers / tool descriptions, if a dataset exists
- [ ] Write-up for a blog post / portfolio

Effort: ~1–2 days. GPU: T4 or A100.

### 5. 💡 Later

| Idea | Why later |
|---|---|
| H-JEPA (hierarchical world model) | Heavy: 22–90 GB data, multi-level training |
| LeWAM (world + action model) | After 09; needs robot data and flow matching |
| V-JEPA 2.1 / ViT-giant on IntPhys 2 | Only if 08 Part 2 lands far below the paper |
| jepa.cpp: run a JEPA guardrail on a laptop / edge device | Deployment story for 10 |

## Housekeeping

- [ ] Each experiment's "What it shows" links the matching blocker in RESEARCH_LANDSCAPE
- [ ] Keep the root README tables (experiments, results) in sync with this file
- [ ] Update RESEARCH_LANDSCAPE when new papers land (last check: 2026-10-26)
