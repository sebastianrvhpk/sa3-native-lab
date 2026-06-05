---
name: sa3-research-map-curator
description: Curate the current SA3 Native Lab research docs, maps, and backlog. Use when updating README.md, docs/research/current, docs/research/methods, repo structure maps, method notes, implementation status, or Underfit comparison guidance from the current notebook-first perspective.
---

# SA3 Research Map Curator

## Scope

Use this skill for documentation that explains where the repo is going:

- `README.md`
- `docs/research/README.md`
- `docs/research/current/repo-structure-map.md`
- `docs/research/current/notebook-research-map-and-next-methods.md`
- `docs/research/current/native-experimental-modes-math.md`
- `docs/research/methods/*.md`

## Style

Write from the current notebook-first perspective. Prefer positive statements
about the active research path:

- expanded Colab notebook as source of truth
- reusable SA3/SAME primitives with tests
- native flow scoring, latent memory, DSP, geometry, controls, residual probes
- Underfit artifacts as the LoRA comparison path
- validation through notebook execution, JSON checks, unit tests, and listening
  notes

## Workflow

1. Inspect code before changing claims.
   - Confirm files, tests, mode numbers, helper names, and scripts with `rg`.
   - Mark research status as confirmed, scaffold, hypothesis, or unknown when
     useful.

2. Update the right layer.
   - README: short project entry and navigation.
   - Research README: doc index and ownership.
   - Repo structure map: surfaces, artifact graph, Markdown inventory.
   - Notebook map: capability inventory, backlog, implementation status.
   - Math notes: equations, conventions, mode-specific implementation details.
   - Methods: reusable research concepts that support notebook cells.

3. Keep docs converged.
   - Match mode names and numbers to the notebook.
   - Match helper names to `latent_audio_primitives/`.
   - Keep Underfit language as handoff/comparison workflow.
   - Preserve concise, current-state language.

4. Validate.
   - Run local Markdown link checks after moving or deleting docs.
   - Run notebook validation when docs describe changed notebook behavior.
   - Run `git diff --check`.

## Output

Report:

- docs updated
- claims verified from code/notebook/tests
- validation performed
- open research questions that remain
