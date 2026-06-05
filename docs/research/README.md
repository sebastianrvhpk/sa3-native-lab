# SA3 Native Lab Research Docs

Status: current documentation index for the notebook-first SA3 Native Lab
direction.

This repo keeps the expanded Colab notebook as the research instrument. The docs
are organized as a lab record:

- `current/research-state.md`: what exists now.
- `current/methods-and-math.md`: how the notebook methods work.
- `current/primitive-map.md`: how the helper modules cluster into a coherent
  notebook primitive library.
- `current/frontier-architecture-transfer.md`: current multimodal/audio
  architecture scout and transfer map into notebook experiments.
- `current/source-context.md`: why the source literature/repos matter here.
- `current/experiment-ledger.md`: what was actually run and decided.
- `current/backlog.md`: what to try next.

## Current

- [Research state](current/research-state.md): current project direction, repo
  surfaces, notebook mode inventory, active helper modules, artifact graph,
  runtime assumptions, and current unknowns.
- [Methods and math](current/methods-and-math.md): SA3/SAME objects, frozen-model
  principle, flow scoring, mode equations, latent DSP, geometry, control
  observability, guidance, residual steering, and implementation safety notes.
- [Primitive map](current/primitive-map.md): module clusters, artifact flow,
  placement rules, structure debt, and promotion criteria for
  `latent_audio_primitives/`.
- [Frontier architecture transfer](current/frontier-architecture-transfer.md):
  source-checked multimodal/audio SOTA scout, architecture deltas, transfer
  matrix, and candidate SA3/SAME notebook experiments.
- [Source context](current/source-context.md): external papers/repos summarized
  as source, relevant idea, notebook impact, and status.
- [Experiment ledger](current/experiment-ledger.md): template and initial empty
  ledger for actual Colab runs, listening notes, descriptor deltas, and
  keep/revise/drop decisions.
- [Backlog](current/backlog.md): implemented backlog reference, immediate
  priorities, open questions, promote/drop criteria, and next implementation
  order.

## Related Docs

- [Colab L4 runbook](../../colab/sa3_medium_l4_runbook.md): runtime setup,
  access, Flash Attention, smoke-test, and failure-mode notes for SA3 Medium on L4.

## Repo-Local Research Skills

- `.codex/skills/sa3-latent-research-scientist`: frames notebook work as
  object, intervention, measurement, claim, and decision.
- `.codex/skills/multimodal-ai-research-scout`: surveys current multimodal AI
  sources and translates architecture changes into SA3/SAME notebook
  experiments.

Stable Audio 3 runtime docs live in the upstream
[Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3)
repo.

LoRA work uses [dada-bots/underfit](https://github.com/dada-bots/underfit).
Exported Underfit audio, checkpoints, and run notes can be compared through the
notebook's descriptor, memory, player, and annotation cells.

## Documentation Rules

- Current implementation claims should point to `colab/`,
  `latent_audio_primitives/`, or current research docs.
- New notebook methods should update `current/research-state.md`,
  `current/methods-and-math.md`, `current/primitive-map.md`, and
  `current/backlog.md` as appropriate.
- Current/SOTA architecture scouting should update
  `current/frontier-architecture-transfer.md` and cite checked primary sources.
- Real notebook runs should update `current/experiment-ledger.md`.
- Add external survey material only when it directly supports a current notebook
  cell or source-context entry.
