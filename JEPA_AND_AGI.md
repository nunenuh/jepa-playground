# JEPA, the brain and AGI: notes from our discussions

Opinion and synthesis (October 2026), not results. The evidence behind the
claims is in the experiments (01–09) and in
[RESEARCH_LANDSCAPE.md](RESEARCH_LANDSCAPE.md). Parts of this are speculation
and are marked as such.

## LLM vs JEPA, in one analogy

- **An LLM is like DNA/a seed.** Training compresses trillions of words into
  a few billion numbers; generating "grows" them back into something similar
  but never identical. Like DNA it stores the *growth rules*, not a copy
  ("Language Modeling Is Compression", DeepMind 2023). A wrong "growth" is a
  hallucination.
- **A JEPA world model is like the brain's predictive machinery.** It guesses
  what comes next *at the level of meaning*, ignores unpredictable detail,
  and learns from surprise (predictive coding). Our 03/07 surprise signal is
  this.
- Both use the same artificial neurons; the difference is the training goal:
  reproduce exact detail (LLM, decoder-only) vs predict meaning (JEPA, no
  decoder at all).

## Where the world model lives in the brain

Not in one place; it's a team, and it lines up with LeCun's design:

| Brain | Predicts / does | AI counterpart |
|---|---|---|
| Neocortex (sensory areas) | incoming sensations, from surprise (predictive coding) | JEPA encoder + predictor |
| Prefrontal cortex | imagines "what if I do X?", plans | the planner ([09](09_world_model/)) |
| Hippocampus + entorhinal cortex | a map of the world; replays the past, imagines futures | memory, H-JEPA |
| Cerebellum | the body's physics; why you can't tickle yourself | action-conditioned predictor (LeWM) |
| Parietal cortex | space, body and object positions | position/velocity in the latent |
| Basal ganglia + dopamine | action choice, reward prediction error | reinforcement learning |

The brain uses both prediction (self-supervised) and reward (RL). Most
JEPA world models plan without RL; the likely future is a hybrid: a world
model for deliberate planning (System 2) and a fast policy refined with RL
(System 1).

## Is JEPA the recipe for AGI?

**Probably one important ingredient, not the full recipe.** LeCun himself
places JEPA as the *world model* inside a larger "objective-driven AI"
(perception, world model, planner, memory, cost/guardrails, actor).

| AGI needs | JEPA today |
|---|---|
| Learning about the world without labels | ✅ core strength |
| Predicting meaning, efficiently | ✅ |
| Planning with a world model | 🟡 toy tasks (Push-T 74%), short horizons |
| Physical common sense | 🟡 ~55% IntPhys 2 vs humans 96% ([08](08_intphys2/)) |
| Hierarchy and abstraction | 🟡 first H-JEPA, Oct 2026 |
| Language, reasoning, maths, code | ❌ LLMs far ahead |
| Long-term memory, continual learning | ❌ unsolved for everyone |
| Goals and motivation | ❌ hand-designed costs |
| Knowing when it's wrong | ❌ open |
| Proven scaling to general ability | ❌ shown for robots (RoboJEPA), not general |

Competing bets: scale LLMs + RL (OpenAI, Anthropic); JEPA world models
(LeCun / AMI Labs); mix everything, LLM + world model + RL (DeepMind);
spatial intelligence / generated 3D worlds (Fei-Fei Li / World Labs).

## What AI has now, and what's missing

**Have (2026):** expert-level language and knowledge, strong reasoning, maths
and code, strong vision, narrow superhuman skills (AlphaZero, AlphaFold),
brittle but working agents, early world models and planning, lab-level
robotics, self-supervised learning at scale.

**Missing:**

1. Continual learning without forgetting
2. Long-term, organised memory
3. Physical common sense
4. Long-horizon planning
5. Sample efficiency (learn from a few examples)
6. Reliability and knowing its limits
7. Its own goals and motivation, with good values
8. Causal understanding ("why", "what if")
9. One system general across all domains
10. Energy efficiency (brain ~20 W)
11. Alignment and safety at scale

The main gap isn't one algorithm; it's an architecture that combines the
pieces and keeps learning, the way the brain's regions work as one. JEPA
targets 3, 4 and 5; agent security targets 6 and 11.

## If JEPA delivers: what likely happens next (speculation)

"Delivers" = calibrated world models with child-level physical common sense,
hours-long planning, few-shot skill learning, cheap on-device inference,
paired with LLMs for language.

1. **Physical AI goes mainstream:** adaptable general-purpose robots,
   predictive autonomous vehicles and drones, AR assistants that understand
   what you see, private on-device models.
2. **Combined systems:** LLM plans in words, the world model checks "will this
   actually work?"; household robots; science via world models of cells,
   weather and materials; training in generated 3D worlds.
3. **Society-level:** physical labour automated the way knowledge work is
   starting to be; cheaper, more efficient AI; closer to a brain-like
   architecture, still missing goals, values and continual learning.

### New risks it creates

| Risk | Example |
|---|---|
| Physical agent hijacking | prompt-injecting a robot or car, with physical consequences |
| World-model poisoning | corrupted training video teaches wrong physics |
| Adversarial reality | a pattern in the world that makes the model mis-predict |
| Planning exploits | tricking the cost/goal so the agent "rationally" does harm |
| Monitoring at scale | millions of autonomous agents need behavioural anomaly detection |
| Privacy | always-on cameras with models that understand everything they see |

### Where this repo fits

Security for agents that understand and act: detecting hijacked agents from
behaviour ([05](05_real_agents/)), surprise when something breaks physics
([03](03_moving_ball/), [07](07_physical_anomalies/)), and knowing how
world-model agents plan, and so where they can be attacked
([09](09_world_model/)). Today there is no published JEPA security work and
no production guardrail; if JEPA succeeds, this becomes essential.
