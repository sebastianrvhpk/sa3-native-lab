# SA3 Native Lab Research Docs

Status: current documentation index for the notebook-first SA3 Native Lab
direction.

This repo keeps the expanded Colab notebook as the research instrument. The docs
are organized around that scope:

- `current/`: current notebook map, current math notes, repo structure map.
- `methods/`: reusable method notes that support current or near-term notebook
  cells.

## Current

- [Repo structure map](current/repo-structure-map.md): top-level repo map,
  Markdown inventory, artifact graph, and doc ownership rules.
- [Notebook research map and next methods](current/notebook-research-map-and-next-methods.md):
  capability map, research survey, implemented backlog status, and next
  implementable ideas.
- [Native experimental modes math](current/native-experimental-modes-math.md):
  core SA3/SAME math, mode taxonomy, prompt-flow scoring, and implementation
  notes for the expanded Colab notebook.

## Method References

- [Native operators and measurement](methods/native-operators-and-measurement.md): geometry,
  covariance transport, periodic operators, guidance, prompt inversion,
  residual discovery, and observability.
- [Neural latent DSP](methods/neural-latent-dsp.md): SAME latent-time dynamics,
  FFT/EQ/phase operators, PCA component gain, SA3 polish, and MIR audits.

## Related Docs

- [Colab L4 runbook](../../colab/sa3_medium_l4_runbook.md): runtime setup,
  access, Flash Attention, smoke-test, and failure-mode notes for SA3 Medium on L4.
- [Codex skills](../codex_skills.md): repo-local agent workflow skills.
Stable Audio 3 runtime docs live in the upstream
[Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
repo.

LoRA work uses [dada-bots/underfit](https://github.com/dada-bots/underfit).
Exported Underfit audio, checkpoints, and run notes can be compared through the
notebook's descriptor, memory, player, and annotation cells.

## Documentation Rules

- Current implementation claims should point to `colab/`,
  `latent_audio_primitives/`, `scripts/`, or `tests/`.
- New notebook methods should update
  `current/notebook-research-map-and-next-methods.md` and, when math changes,
  `current/native-experimental-modes-math.md`.
- Add external survey material only when it directly supports a current notebook
  cell or method note.
