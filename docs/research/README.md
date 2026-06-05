# SA3 Native Lab Research Docs

Status: current documentation index for the notebook-first SA3 Native Lab
direction.

This repo keeps the expanded Colab notebook as the research instrument. The docs
are organized around that boundary:

- `current/`: current notebook map, current math notes, repo structure map.
- `methods/`: reusable method notes that support current or near-term notebook
  cells.

## Current

- [Repo structure map](current/repo-structure-map.md): top-level repo map,
  Markdown inventory, active/inactive boundaries, artifact graph, and doc
  ownership rules.
- [Notebook research map and next methods](current/notebook-research-map-and-next-methods.md):
  capability map, research survey, implemented backlog status, and next
  implementable ideas.
- [Native experimental modes math](current/native-experimental-modes-math.md):
  core SA3/SAME math, mode taxonomy, prompt-flow scoring, and implementation
  notes for the expanded Colab notebook.

## Method References

- [Seven better operators](methods/seven-better-operators.md): geometry,
  covariance transport, periodic operators, guidance, prompt inversion,
  residual discovery, and observability.
- [Neural latent DSP](methods/neural-latent-dsp.md): SAME latent-time dynamics,
  FFT/EQ/phase operators, PCA component gain, SA3 polish, and MIR audits.

## Related Docs

- [Colab L4 runbook](../../colab/sa3_medium_l4_runbook.md): historical and
  practical Colab notes for SA3 Medium on L4.
- [Codex skills](../codex_skills.md): repo-local agent workflow skills.
- [Stable Audio 3 guide docs](../guides/model-overview.md): upstream/reference
  model documentation.
- [Stable Audio 3 workflow docs](../workflows/inference.md): upstream/reference
  inference and autoencoder documentation.

Local LoRA work is not an active SA3 Native Lab backlog. The active LoRA path is
externalized to [dada-bots/underfit](https://github.com/dada-bots/underfit);
local LoRA training docs, scripts, and notebook cells have been removed.

Earlier broad research notes were distilled into the current roadmap and then
removed from the active tree. Git history preserves the originals.

## Documentation Rules

- Current implementation claims should point to `colab/`,
  `scripts/build_sa3_same_native_notebook.py`, `latent_audio_primitives/`, or
  `tests/`.
- New notebook methods should update
  `current/notebook-research-map-and-next-methods.md` and, when math changes,
  `current/native-experimental-modes-math.md`.
- Historical surveys should not be reintroduced unless they directly drive a
  current notebook cell.
- Do not add app, dashboard, API, or product-interface docs for this project
  direction.
