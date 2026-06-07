# SA3 Native Lab Research Docs

Status: current documentation index for the notebook-first SA3 Native Lab
direction.

This repo keeps the expanded Colab notebook as the research instrument. The docs
are organized as a lab record:

- `current/research-state.md`: what exists now.
- `current/architecture-ontology.md`: four-layer research map for SAME-only,
  SA3 flow/conditioning, SA3 internal trajectory, coupled SA3-over-SAME
  editing, plus the evidence utilities that review all layers.
- `current/methods-and-math.md`: how the notebook methods work.
- `current/primitive-map.md`: how the helper modules cluster into a coherent
  notebook primitive library.
- `current/capability-map.md`: bottom-up map of native objects, operations,
  evidence maturity, artifacts, and justified notebook workbenches.
- `current/frontier-architecture-transfer.md`: current multimodal/audio
  architecture scout and transfer map into notebook experiments.
- `current/run-protocol.md`: how notebook runs become evidence.
- `current/source-context.md`: why the source literature/repos matter here.
- `current/experiment-ledger.md`: what was actually run and decided.
- `current/backlog.md`: what to try next.

## Current

- [Research state](current/research-state.md): current project direction, repo
  surfaces, native object graph, workbenches, maturity board, active helper
  modules, artifact graph, runtime assumptions, and current unknowns.
- [Architecture ontology](current/architecture-ontology.md): canonical four
  research layers, evidence utilities, object transitions, placement rules,
  existing coverage, new research programs, and priority order.
- [Methods and math](current/methods-and-math.md): SA3/SAME objects, frozen-model
  principle, flow scoring, object-transition equations, latent DSP, geometry,
  control observability, guidance, residual probing/steering, and implementation
  safety notes.
- [Primitive map](current/primitive-map.md): research-altitude layers, artifact
  flow, placement rules, and promotion criteria for
  `latent_audio_primitives/`.
- [Capability map](current/capability-map.md): evidence-backed capability and
  maturity matrix, artifact flow, parameter inventory, and the workbench shape
  derived from actual code and notebook cells.
- [Frontier architecture transfer](current/frontier-architecture-transfer.md):
  source-checked multimodal/audio SOTA scout, architecture deltas, transfer
  matrix, and candidate SA3/SAME notebook experiments.
- [Run protocol](current/run-protocol.md): research frame, claim ladder, run
  spine, workbenches, evidence panels, minimum evidence packet, and decision
  rules.
- [Source context](current/source-context.md): external papers/repos summarized
  as source, relevant idea, notebook impact, and status.
- [Experiment ledger](current/experiment-ledger.md): template and initial empty
  ledger for actual Colab runs, listening notes, descriptor deltas, and
  keep/revise/drop decisions.
- [Backlog](current/backlog.md): missing-evidence gaps, next notebook runs,
  promote/drop criteria, and priority order.

## Related Docs

- [Colab L4 runbook](../../colab/sa3_medium_l4_runbook.md): runtime setup,
  access, Flash Attention, smoke-test, and failure-case notes for SA3 Medium on L4.

## Repo-Local Research Skills

- `.codex/skills/sa3-latent-research-scientist`: frames notebook work as
  object transition, operation, measurement, claim maturity, evidence packet,
  and decision.
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
  `current/architecture-ontology.md`, `current/methods-and-math.md`,
  `current/primitive-map.md`, `current/capability-map.md`, and
  `current/backlog.md` as appropriate.
- Current/SOTA architecture scouting should update
  `current/frontier-architecture-transfer.md` and cite checked primary sources.
- Notebook execution discipline should follow `current/run-protocol.md`.
- Real notebook runs should update `current/experiment-ledger.md`.
- Add external survey material only when it directly supports a current notebook
  cell or source-context entry.
