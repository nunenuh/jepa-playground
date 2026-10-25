# Experiment standard

Every experiment folder (`NN_name/`) follows this checklist. 01–04 are the
reference examples.

## Files

| File | Rule |
|---|---|
| `NN_name/name.py` | The source. jupytext percent format (`# %%` cells), Colab GPU header (`accelerator: GPU`, `gpuType: T4`). |
| `NN_name/name.ipynb` | Generated: `jupytext --to ipynb NN_name/name.py`. Never edit by hand. No outputs committed. |
| `NN_name/README.md` | The write-up (sections below). |
| `NN_name/assets/` | Every picture the README uses, plus `results.json` with the raw numbers. |

## Pictures (minimum)

1. **Architecture diagram**: `architecture-light.svg` + `architecture-dark.svg`,
   same style as 01 (white / `#0d1117` background, blue = learned,
   dashed grey = frozen, orange = loss, violet = EMA or shared weights).
   Shown in the README with a `<picture>` tag (dark-mode aware) and in the
   notebook's intro cell via the raw GitHub URL.
2. **The task / data**: what the model sees (inputs, what is hidden or predicted).
3. **Training curves**: loss, and spread (collapse check).
4. **The key result**: the one chart that answers the experiment's question.
5. **An intuitive picture**: something you understand without reading numbers
   (e.g. t-SNE in 01, collapsed embeddings in 02, ball paths in 03, per-step
   scores in 04).

Chart rules: one metric per axis (no dual y-axes), same colours for the same
thing across charts, every figure saved with `dpi=110`.

## README sections

1. Title + **Open in Colab** badge.
2. One-paragraph idea in plain words.
3. **Architecture** diagram.
4. **Setup** table: parts, sizes, training budget, hardware, run time.
5. **Results**: a table with *before vs after* (or baseline vs ours), then
   every picture with 1–3 sentences on what to look at.
6. **What it shows**: short bullets, including what did *not* work.
7. **Experiments to try**: one-line changes and what you'll see.
8. **Run**: the Colab badge and the local `uv run` command.

## Honesty rules

- Real numbers from a real run; raw numbers in `assets/results.json`.
- Always a baseline (untrained model, random encoder, or a simple method).
- Failed attempts get reported, not hidden.
- Train on Colab (GPU), not on the laptop.

## Also update

- The roadmap and results tables in the main `README.md`.
