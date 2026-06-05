---
name: sa3-notebook-steward
description: Maintain the expanded SA3/SAME Colab notebook as the source of truth. Use when editing colab/sa3_same_native_experimental_modes.ipynb, validating notebook cells, preserving the cell-based research workflow, updating Mode cells, or checking notebook JSON and Colab execution assumptions.
---

# SA3 Notebook Steward

## Scope

Use this skill for work centered on:

- `colab/sa3_same_native_experimental_modes.ipynb`
- `colab/sa3_medium_l4_runbook.md`
- `scripts/validate_colab_notebook.py`
- notebook mode text, code cells, toggles, manifests, and Colab setup behavior

The expanded notebook is the research object. Keep notebook edits direct,
cell-based, and easy to inspect in diff.

## Workflow

1. Inspect the relevant notebook cells.
   - Read cell headings and surrounding markdown/code before editing.
   - Preserve existing mode ordering, toggles, manifests, and validation paths.
   - Keep reusable math in `latent_audio_primitives/` when it is shared or
     test-worthy.

2. Edit conservatively.
   - Use `apply_patch` for small JSON source updates.
   - For broad mechanical notebook changes, use a deterministic script only
     after inspecting the target cells, then review the diff carefully.
   - Keep explanatory markdown aligned with current notebook-first research
     language.

3. Validate.
   - `python -m json.tool colab/sa3_same_native_experimental_modes.ipynb >/tmp/sa3_notebook.json`
   - `uv run python scripts/validate_colab_notebook.py --skip-setup`
   - Run focused primitive tests when code paths imported by the notebook change.
   - Run `git diff --check`.

## Notebook Checks

Confirm mode-level work preserves:

- custom Colab audio player usage
- dataset and long-file controls
- per-mode cells and mode toggles
- Mode 2 flow prompt scorer integration through `latent_audio_primitives.flow_prompt`
- Underfit handoff language for LoRA comparisons
- final experiment manifest cell

## Output

Report:

- cells or sections changed
- validation commands and results
- any model-weight or Colab-only behavior left for GPU smoke testing
