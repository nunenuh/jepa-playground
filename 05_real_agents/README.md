# 05 · Guardrail on real agents

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nunenuh/jepa-playground/blob/main/05_real_agents/real_agents.ipynb)

Experiment 04 caught hijacked agents in **synthetic** runs. Does it hold up
on **real** agents? Here we use AgentDojo's published runs: 29 real LLM
agents (GPT-4o, Claude 3.x, Gemini, Llama, ...) attacked with indirect prompt
injections.

## Architecture

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/architecture-dark.svg">
  <img alt="AgentDojo runs to fair split to MiniLM to JEPA to probe; Q2 trains the probe on AgentDrift" src="assets/architecture-light.svg">
</picture>

## Data

[AgentDojo](https://github.com/ethz-spylab/agentdojo) runs (MIT), pinned to
commit `089ed46`. We keep the `important_instructions` attack and the
no-attack runs, and drop runs with no tool call at all (the agent just
answered): 20,631 runs.

| Kind | Meaning | Runs |
|---|---|---|
| benign | no attack | 3,230 |
| hijacked | the injection was in a tool result and the agent obeyed | 3,234 |
| resisted | the injection was there, but the agent ignored it | 14,167 |

![data](assets/data.png)

**Hijacked vs resisted** is the honest test: both runs contain the same
injected text, so only the agent's *actions* differ.

**Fair split.** The same few attacker goals repeat across runs, so a model
could memorise them. Per suite, 30% of user tasks and 30% of attacker goals
are held out: test runs (212 hijacked, 1,179 resisted, 975 benign) use tasks
**and** goals never seen in training.

## Setup

| Part | What it is |
|---|---|
| Tokens | `TASK`, then `CALL tool(args)` and `RESULT ...` per step (up to 15 steps) |
| Text encoder | MiniLM, frozen, each text → 384-D |
| JEPA | same as 04: MLP encoder + causal transformer predictor, SIGReg, trained on 9,946 runs **without labels** |
| Probe | linear, scores every call; run score = top call. AgentDojo only says *whether* a run was hijacked, so the loss is on the run score (multiple-instance learning) |
| Negatives | benign **and** resisted, so the probe must learn "the agent did something bad", not "there is injected text" |

Colab T4, about 12 minutes in total (plus the dataset download).

## Results

### Q1: does pretraining + probe work on real agents?

![labels](assets/labels.png)

AUC hijacked vs resisted (mean of 3 random label samples):

| Labelled runs | raw | jepa | raw+jepa |
|---|---|---|---|
| 50 | 0.614 | **0.750** | 0.620 |
| 100 | 0.629 | **0.674** | 0.642 |
| 250 | 0.705 | **0.734** | 0.697 |
| 1,000 | 0.750 | **0.774** | 0.734 |
| 9,946 (all) | 0.806 | **0.819** | 0.815 |
| *no labels: seen-before baseline* | | *0.747* | |

With all labels, at a threshold that flags 5% of benign validation runs:

| | seen-before | raw | jepa | raw+jepa |
|---|---|---|---|---|
| Hijacked caught | 0% | **44%** | 31% | 42% |
| Benign runs flagged (test) | 0% | 8.9% | 9.5% | **7.5%** |
| Resisted runs flagged (test) | 0% | 7.6% | 11.6% | **4.8%** |

The seen-before baseline ranks runs reasonably (AUC 0.75) but catches
nothing at 5% false alarms: some normal calls in unseen tasks look more
novel than any attack.

### Q2: trained on synthetic data, tested on real agents

![transfer](assets/transfer.png)

The probe is trained only on AgentDrift (04's synthetic data, call labels)
and tested on the real AgentDojo test runs. JEPA is the one pretrained on
real runs without labels.

| | AUC vs resisted | AUC vs benign | Hijacked caught at 5% |
|---|---|---|---|
| synthetic → real, raw | 0.741 | 0.730 | 19% |
| synthetic → real, jepa | **0.765** | **0.812** | 8% |
| synthetic → real, raw+jepa | 0.722 | 0.757 | 4% |
| real → real, raw+jepa (Q1) | 0.815 | 0.812 | 42% |

### Q4: a probe that reads the whole run

The linear probe judges one call (+ the task) at a time. The whole-run probe
is a small bidirectional transformer over every token of the run (task,
calls, results), so a call is scored *in context*. Same loss, same split,
early stopping on the labelled validation split.

![q4](assets/q4.png)

| Labelled runs | linear, raw | linear, raw+jepa | **whole-run, raw** | **whole-run, raw+jepa** |
|---|---|---|---|---|
| 250 | 0.705 | 0.697 | 0.801 | **0.805** |
| 1,000 | 0.750 | 0.734 | **0.865** | 0.838 |
| 9,946 (all) | 0.806 | 0.815 | 0.827 | **0.831** |

AUC hijacked vs resisted. With all labels, the whole-run probe on
raw+jepa catches **61%** of hijacks (linear: 42%) at 8.0% benign / 6.1%
resisted false alarms.

![example context](assets/example_context.png)

Same "pay the bill" pair as above. The attacker's payment now scores 3.57
and the legitimate one 2.32 (linear probe: 3.00 vs 2.99): context helps, but
the legitimate payment is still just over the alarm line (2.11).

### Q5: where did each argument come from?

MiniLM turns text into a *meaning* vector, so `US122…` and `US133…` look
almost the same to it. So for every call we find the **line** earlier in the
run where its argument values (IBANs, emails, file names, ...) appeared:
`FROM TASK`, `FROM RESULT: <that line>`, or `FROM NOWHERE`. That line is
embedded and added to the call's token. This is a learned version of
*taint tracking*: data copied from an injected instruction carries that
instruction's text with it.

| Labelled runs | whole-run, raw | whole-run, raw+jepa | **+ sources, raw** | **+ sources, raw+jepa** |
|---|---|---|---|---|
| 250 | 0.801 | 0.805 | 0.860 | **0.887** |
| 1,000 | 0.865 | 0.838 | 0.910 | **0.926** |
| 9,946 (all) | 0.827 | 0.831 | **0.906** | 0.875 |

AUC hijacked vs resisted. With 1,000 labels, `+ sources, raw+jepa` catches
**71%** of hijacks at 7.2% benign / 7.3% resisted false alarms.

![example sources](assets/example_source.png)

Same "pay the bill" pair. The attacker's IBAN was copied from *"Send a
transaction to US133… that includes…"* (the injected instruction), the real
one from *"recipient: US122…"*. Now only the attacker's payment is flagged
(5.0 vs −13.8 and −12.4).

### Q3: which agents are easiest to catch?

![per agent](assets/per_model.png)

AUC hijacked vs resisted per agent (agents with at least 10 hijacked test
runs; 11–35 runs each, so ±0.05 is noise).

### One run, call by call

![example](assets/example.png)

Task "Please pay the bill, like last month." Left: GPT-4o was hijacked and
paid the attacker's IBAN (US133…) before the real bill (US122…): that call
is flagged. Right: an agent that resisted and paid only the real bill, and
it is flagged too. The probe learned **"sending money is risky"**, not
**"the recipient is wrong"**. (Same task and attack; the resisted run is
from Meta-SecAlign-70B.)

### Training

![curves](assets/curves.png)

The JEPA almost collapses in the first 100 steps (spread 0.01), then SIGReg
pulls it back (spread ~1.1).

## What it shows

- **The recipe transfers to real agents, with less margin.** Real runs get
  AUC 0.82 (synthetic 04: 0.88), with unseen tasks *and* unseen attacker goals.
- **JEPA features matter most when labels are scarce**: with 50 labelled runs
  they give 0.75 vs 0.61 for raw text. With all labels the gap shrinks to ~1 point.
- **Synthetic labels partly work on real agents** (AUC 0.74–0.77), and JEPA
  pretrained on unlabelled real runs helps that transfer. But the alarm
  threshold doesn't carry over: at 5% false alarms it catches only 4–19%.
- **A linear probe learns *what kind* of action is risky, not *whether it
  fits the task*.** The example shows it: a legitimate payment scores as high
  as the attacker's.
- **Reading the whole run helps a lot** (Q4): +10 AUC points with 250–1,000
  labels, and 61% vs 42% of hijacks caught with all labels. It starts to
  separate the attacker's payment from the real one, but not fully yet.
- **Once the probe reads context, JEPA features add little** (±2 points either
  way): the transformer probe learns much of that context itself.
- **Knowing where arguments came from is the biggest win** (Q5): AUC 0.93 with
  1,000 labels, and the "wrong recipient" case is solved in the example. With
  sources, JEPA features help again at 250–1,000 labels (+1.6 to +2.7 points).
- **More labels isn't always better here.** With all labels, Q5 drops a little
  and flags more benign runs (9–15%): early stopping picks the best val AUC,
  but the alarm threshold moves between splits. Calibration is the next
  thing to fix.
- **Caveats.** One attack type (`important_instructions`); 3 label samples per
  point; the Q2 probe isn't seeded, so its numbers move by about ±0.01 between runs; per-agent numbers rest on 11–35 hijacked runs; the dataset
  download is ~500 MB.

## Experiments to try

| Change | What you'll see |
|---|---|
| Calibrate the threshold (e.g. on resisted val runs too) | Fewer false alarms with all labels |
| Sources at the level of each value (not only the first found) | Calls that mix safe and injected values |
| Other attacks (`ignore_previous`, `tool_knowledge`, ...) as extra test sets | Robustness to how the injection is phrased |
| Hold out whole agents instead of tasks | Whether it works on an agent it never saw |
| Fine-tune MiniLM inside the JEPA | Richer call features |

## Run

Click the Colab badge above, pick a GPU runtime, then Run all. Or locally:

```bash
uv run --with torch --with sentence-transformers --with matplotlib \
    python 05_real_agents/real_agents.py
```
