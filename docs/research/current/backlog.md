# SA3 Native Lab Backlog

Status: future work and open questions for the science-ontology notebook.

This document answers: what should be tried next, why it matters, what evidence
would promote or drop it, and which questions remain open.

Implemented items are summarized for context. Future work should be added here,
not mixed into source context or method math.

## Backlog Discipline

Every backlog item should state:

```text
Goal:
Evidence already available:
Ontology touchpoints:
Promote if:
Drop or revise if:
```

Do not add a method just because a paper exists. Add it only when it can become
a concrete notebook run over SA3/SAME objects.

Use [Run protocol](run-protocol.md) before promoting a backlog item into a
notebook implementation or a kept method.

## Implemented Ontology Reference

These items now have notebook cells or probes:

| Item | Notebook Experiment | Helper Support | Status |
|---|---|---|---|
| Flow attribution prompt microscope | `SA3_FLOW_CONDITIONING.flow_attribution` | `flow_prompt.py` attribution rows plus `procedures/flow_scoring.py` scoring | implemented |
| Loss-by-timestep flow panel | `SA3_FLOW_CONDITIONING.flow_timestep_panel` | `flow_prompt.py` loss rows/summaries plus `procedures/flow_scoring.py` scoring | implemented |
| SAME control lanes | `EVIDENCE_DECISION_PROTOCOL.control_lanes` | `control_lanes.py` | implemented |
| Dataset memory curriculum | `DATASET_MEMORY_COMPOSITION.curriculum` | `curriculum.py` | implemented |
| Latent OT style transfer bench | `SAME_REPRESENTATION.ot_style_transfer` | `geometry.py`, `style.py` | implemented |
| Continuation as bridge search | `DATASET_MEMORY_COMPOSITION.bridge_search` | `composition.py` | implemented |
| Residual feature atlas | `CAUSAL_STEERING.residual_feature_atlas` | `residual_features.py` | implemented |
| SA3 null-condition inversion probe | `SA3_FLOW_CONDITIONING.null_inversion` | notebook probe over SA3 conditioning tensors | scaffold |
| Guidance-gradient latent edit | `CAUSAL_STEERING.gradient_edit` | `guidance.py` | scaffold |
| Audio-to-audio posterior guidance | `CAUSAL_STEERING.audio_posterior` | `guidance.py`, source/reference summaries | scaffold |
| Cross-model baseline harness | `EXTERNAL_COMPARISON.cross_model` | notebook command harness plus descriptors/player | scaffold |

## Immediate Priorities

### 1. Probe-Bank Cache

Goal: save target latent, timesteps, noise seeds, and per-prompt rows so
`SA3_FLOW_CONDITIONING.flow_attribution` and `SA3_FLOW_CONDITIONING.flow_timestep_panel` reuse identical
probes across sessions.

Evidence already available: shared probe banks reduce prompt-score variance and
make A/B prompt comparisons fair.

Ontology touchpoints: `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.flow_attribution`, `SA3_FLOW_CONDITIONING.flow_timestep_panel`.

Promote if:

- repeated prompt scoring becomes reproducible across notebook sessions,
- cached rows simplify prompt attribution and loss-by-timestep panels,
- cache metadata includes target audio, convention, logSNR values, seeds, and model.

Drop or revise if:

- cache files become harder to inspect than rerunning probes,
- cache mismatch risks become high.

### 2. Flow-Plus-Descriptor Prompt Search

Goal: score prompts by both SA3 flow loss and decoded descriptor movement after
short audition generations.

Evidence already available: flow score measures teacher-forced vector-field
agreement; descriptors measure decoded audio movement.

Ontology touchpoints: `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.readable_prompt_search`, flow-microscope cells, descriptor/player cells.

Promote if:

- combined scoring predicts better listening results than flow alone,
- prompt candidates remain readable or auditable,
- descriptor weights are explicit.

Drop or revise if:

- descriptor terms reward artifacts,
- generation cost is too high for prompt search.

### 3. Control-Lane Retrieval

Goal: add lane similarity to memory retrieval so searches can combine latent
summary, descriptor targets, and time-varying lane shape.

Evidence already available: `EVIDENCE_DECISION_PROTOCOL.control_lanes` extracts and compares control lanes;
`DATASET_MEMORY_COMPOSITION.memory_index` provides latent memory search.

Ontology touchpoints: `DATASET_MEMORY_COMPOSITION.memory_index`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `DATASET_MEMORY_COMPOSITION.curriculum`, `DATASET_MEMORY_COMPOSITION.bridge_search`.

Promote if:

- lane retrieval surfaces more useful source/donor/bridge candidates,
- nearest rows align with listening judgement,
- lane similarity remains interpretable.

Drop or revise if:

- lanes are too noisy across clips,
- similarity duplicates simpler descriptor retrieval.

### 4. Geometry-Aware Donor Selector

Goal: rank donor candidates for graft, DSP, OT, and continuation by Mahalanobis
distance, lane similarity, descriptor fit, and transition cost.

Evidence already available: geometry, memory, bridge search, control lanes, and
descriptor reports already exist.

Ontology touchpoints: `SAME_REPRESENTATION.audio_graft`, `SAME_REPRESENTATION.neural_dsp`, `SAME_REPRESENTATION.geometry_audit`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `SAME_REPRESENTATION.ot_style_transfer`, `DATASET_MEMORY_COMPOSITION.bridge_search`.

Promote if:

- donor selection improves graft/style/bridge auditions,
- nearest-memory checks separate useful preservation from copying,
- geometry metrics explain audible differences.

Drop or revise if:

- donor ranking is no better than random or manual selection,
- geometry distance predicts nothing audible.

### 5. Residual Temporal Patching

Goal: extend `CAUSAL_STEERING.residual_feature_atlas` from layer-level residual feature atlas to layer x
denoising-step or layer x latent-time patch tests.

Evidence already available: residual vectors and feature bases exist; activation
steering sources support causal intervention tests.

Ontology touchpoints: `CAUSAL_STEERING.prompt_residual`, `CAUSAL_STEERING.audio_residual`, `CAUSAL_STEERING.residual_feature_atlas`.

Promote if:

- patching changes target descriptors/listening notes with limited side effects,
- layer/time maps are repeatable,
- residual atlas ranks match intervention outcomes.

Drop or revise if:

- hooks are too fragile,
- interventions only create artifacts,
- results are not repeatable across prompts/seeds.

### 6. Guidance Objective Mixer

Goal: let `CAUSAL_STEERING.gradient_edit` choose profile, boundary, period, lane, descriptor, and
preservation losses from a compact JSON recipe.

Evidence already available: generic gradient guidance exists; `SAME_REPRESENTATION.geometry_audit`,
`EVIDENCE_DECISION_PROTOCOL.control_lanes`, and `SAME_REPRESENTATION.neural_dsp` provide measurable loss candidates.

Ontology touchpoints: `SAME_REPRESENTATION.geometry_audit`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `CAUSAL_STEERING.gradient_edit`, `CAUSAL_STEERING.audio_posterior`.

Promote if:

- objective recipes are inspectable,
- gradients move intended measurements,
- audio results beat prompt/audio-to-audio baselines.

Drop or revise if:

- objective hacking dominates audio quality,
- gradients are unstable or too expensive.

### 7. Null-Condition Edit Audition

Goal: after `SA3_FLOW_CONDITIONING.null_inversion` optimizes null conditioning, generate
fixed prompt edits and compare source preservation against
`SA3_FLOW_CONDITIONING.soft_prompt_inversion` and plain audio-to-audio.

Evidence already available: null-text inversion motivates branch-specific
conditioning; SA3 has CFG-like conditional/null behavior.

Ontology touchpoints: `SA3_FLOW_CONDITIONING.soft_prompt_inversion`, `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.null_inversion`.

Promote if:

- source identity is preserved better than baselines,
- text edits remain effective,
- the learned/null conditioning is reusable across nearby prompts.

Drop or revise if:

- prompt edits collapse,
- preservation is not better than audio-to-audio init,
- conditioner internals are too unstable.

### 8. Novelty / Source-Preservation Panel

Goal: for every generated artifact, show nearest memory rows plus descriptor
deltas so "kept source identity" and "copied dataset item" are separated.

Evidence already available: memory index, descriptors, manifest, player, and
annotation cells exist.

Ontology touchpoints: all generation/editing experiments, especially
`SAME_REPRESENTATION.neighborhood_renoise`, `SAME_REPRESENTATION.audio_graft`, `SAME_REPRESENTATION.neural_dsp`,
`SAME_REPRESENTATION.ot_style_transfer`, `DATASET_MEMORY_COMPOSITION.bridge_search`, `CAUSAL_STEERING.gradient_edit`,
and `CAUSAL_STEERING.audio_posterior`.

Promote if:

- reviewers can quickly distinguish copying, preservation, and novelty,
- panel becomes useful across methods,
- results can be summarized in the experiment ledger.

Drop or revise if:

- nearest-memory metrics do not match listening judgement,
- panel is too bulky for normal notebook use.

### 9. Seed-Family Atlas

Goal: for a fixed method recipe, generate a seed grid and cluster outputs by
SAME summaries, descriptor deltas, and listening tags.

Evidence already available: memory clustering and descriptor reports exist.

Ontology touchpoints: `SAME_REPRESENTATION.neighborhood_renoise`, `SA3_FLOW_CONDITIONING.hard_prompt_search`, `CAUSAL_STEERING.prompt_residual`, `SAME_REPRESENTATION.ot_style_transfer`, `CAUSAL_STEERING.gradient_edit`, `EXTERNAL_COMPARISON.cross_model`.

Promote if:

- seed families reveal repeatable behavior,
- clusters help select robust recipes,
- failure cases become visible.

Drop or revise if:

- seed grids consume too much runtime,
- clusters are not interpretable.

### 10. Notebook Report Packager

Goal: export selected manifest rows, audio paths, descriptor tables, and
annotations into a static report.

Evidence already available: manifest/player/descriptors already exist.

Ontology touchpoints: manifest cell, player, descriptor tables, experiment ledger.

Promote if:

- reports make review easier,
- selected outputs can be shared without rerunning Colab,
- report content remains honest about scaffold versus confirmed status.

Drop or revise if:

- report generation stops serving notebook research,
- it distracts from notebook-first research.

## Frontier-Informed Additions

These items come from the source-checked
[Frontier architecture transfer](frontier-architecture-transfer.md) pass. They
should enter implementation only as concrete notebook runs.

### A. SAME Bottleneck Stress Test

Goal: identify what SAME preserves semantically, acoustically, structurally, and
what SA3 polish reconstructs from the prior.

Evidence already available: SAME is the local native latent object; latent blur,
latent DSP, geometry, direct decode, SA3 polish, descriptors, and flow scoring
already exist.

Ontology touchpoints: `SAME_REPRESENTATION.blur_bottleneck`, `SAME_REPRESENTATION.neural_dsp`, `SAME_REPRESENTATION.geometry_audit`, `SA3_FLOW_CONDITIONING.flow_attribution`, `SA3_FLOW_CONDITIONING.flow_timestep_panel`.

Promote if:

- perturbation families reveal stable preservation/failure patterns across clips,
- direct decode and SA3 polish differences are audible and measurable,
- the result clarifies which operators are controls versus microscopes.

Drop or revise if:

- results reduce to loudness, silence, clipping, or decoder artifacts,
- SA3 polish erases every measured difference.

### B. Multi-Turn Audio Edit Consistency

Goal: test whether SA3/SAME supports iterative audio editing without losing
source identity or accumulating drift.

Evidence already available: flow attribution, null-condition inversion scaffold,
soft prompts, source-preservation guidance scaffold, descriptors, and memory
nearest-neighbor checks exist.

Ontology touchpoints: `SA3_FLOW_CONDITIONING.soft_prompt_inversion`, `SA3_FLOW_CONDITIONING.hard_prompt_search`, `SA3_FLOW_CONDITIONING.flow_attribution`, `SA3_FLOW_CONDITIONING.flow_timestep_panel`, `SA3_FLOW_CONDITIONING.null_inversion`, `CAUSAL_STEERING.audio_posterior`.

Promote if:

- each edit obeys the new target while preserving declared source attributes,
- flow/descriptor/listening evidence agrees,
- source-preservation rows detect drift early.

Drop or revise if:

- each turn behaves like unrelated regeneration,
- source identity collapses or prompt edits stop working.

### C. Segment / Block Prompt Plan

Goal: borrow long-form music structure discipline by scoring prompts and
controls per segment, then assembling with continuation and bridge search.

Evidence already available: chunk windows, prompt family search, memory
curriculum, control lanes, continuation, bridge ranking, and manifests exist.

Ontology touchpoints: `DATASET_MEMORY_COMPOSITION.prompt_family`, `DATASET_MEMORY_COMPOSITION.continuation`, `DATASET_MEMORY_COMPOSITION.memory_index`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `DATASET_MEMORY_COMPOSITION.curriculum`, `DATASET_MEMORY_COMPOSITION.bridge_search`.

Promote if:

- segment plans improve continuity versus one global prompt,
- bridge/control-lane scores predict listening results,
- generated structures remain auditable in manifest rows.

Drop or revise if:

- segment boundaries dominate artifacts,
- the workflow stops being notebook research.

### D. External Temporal Embedding Lane

Goal: optionally compare SAME-native evidence against CLAP/T-CLAP/ImageBind-like
semantic or temporal embedding distances.

Evidence already available: memory index, descriptors, control lanes, and
cross-model baseline harness exist.

Ontology touchpoints: `DATASET_MEMORY_COMPOSITION.memory_index`, `EVIDENCE_DECISION_PROTOCOL.control_lanes`, `DATASET_MEMORY_COMPOSITION.curriculum`, `EXTERNAL_COMPARISON.cross_model`, player/ledger cells.

Promote if:

- external embeddings catch failures that SAME summaries miss,
- disagreement rows improve review decisions,
- dependencies fit Colab without destabilizing SA3.

Drop or revise if:

- embedding scores reward wrong semantics,
- setup cost outweighs evidence value,
- external embeddings obscure SAME-native claims.

### E. Step / Polish Tradeoff Audit

Goal: map runtime, step count, init noise, and SA3 polish settings against
flow loss, descriptors, source preservation, and listening quality.

Evidence already available: SA3 supports fast generation/editing; local
experiments already sweep renoise, latent edits, cyclic sampler probes, and
guidance.

Ontology touchpoints: `SAME_REPRESENTATION.neighborhood_renoise`, `SAME_REPRESENTATION.blur_bottleneck`, `SAME_REPRESENTATION.neural_dsp`, `CAUSAL_STEERING.cyclic_trajectory`, `CAUSAL_STEERING.gradient_edit`, `CAUSAL_STEERING.audio_posterior`.

Promote if:

- a small settings grid gives stable quality/runtime guidance,
- low-step outputs remain useful for notebook audition,
- failure cases are visible before long runs.

Drop or revise if:

- upstream sampler internals hide the relevant controls,
- quality varies too much by prompt/seed for a compact rule.

### F. Annotation-Weighted Recipe Selection

Goal: convert player annotations and ledger notes into recipe-level evidence
without pretending to train a preference model.

Evidence already available: custom player, annotation search, descriptor rows,
manifests, and experiment ledger exist.

Ontology touchpoints: player, manifest, experiment ledger,
`SAME_REPRESENTATION.neighborhood_renoise`, `SA3_FLOW_CONDITIONING.hard_prompt_search`,
`SAME_REPRESENTATION.ot_style_transfer`, `CAUSAL_STEERING.gradient_edit`, `EXTERNAL_COMPARISON.cross_model`.

Promote if:

- repeated annotations identify robust recipes and failure families,
- descriptor/listening disagreement becomes visible,
- recipe summaries improve future run selection.

Drop or revise if:

- notes are too sparse or inconsistent,
- scalar ratings hide useful qualitative judgement.

## Lower-Priority Ideas

- Better visual styling for notebook panels.
- Compact printable manifest summaries.
- Cross-model fixed task packs beyond `EXTERNAL_COMPARISON.cross_model`.
- Prompt-token replacement suggestions beyond leave-one-out attribution.
- Control-lane editor for hand-drawn or imported lanes.
- Layer/time residual patch heatmaps.
- Segment/block prompt planner after bridge-search evidence improves.
- External temporal embedding lane if Colab dependencies remain manageable.

## Open Questions

- Does native flow score predict generated-audio similarity, or only teacher-forced vector-field agreement?
- Which logSNR bands correspond to style, structure, transient detail, or prompt adherence in SA3?
- Can null-condition inversion preserve source identity while allowing text edits?
- Which SAME descriptor/control lanes are observable, predictable, and intervenable?
- Does covariance transport outperform mean/std style transfer audibly?
- Which residual layers carry stable music/audio controls?
- Can memory nearest-neighbor checks separate useful source preservation from memorization-like copying?
- Which dataset clusters produce prompt/control curricula that generalize?
- Can sampler-level guidance improve loopability without half-period collapse?
- Which latent DSP operations are microscopes only, and which become useful controls?
- Which SAME perturbations reveal semantic versus acoustic bottleneck behavior?
- Can multi-turn audio edits preserve source identity under repeated prompt changes?
- Do external semantic/temporal embeddings disagree productively with SAME-native memory?
- What is the cheapest SA3 step/polish setting that still supports reliable audition?

## Priority Order

Recommended next sequence:

```text
probe-bank cache
-> flow-plus-descriptor prompt search
-> novelty/source-preservation panel
-> SAME bottleneck stress test
-> multi-turn audio edit consistency
-> control-lane retrieval
-> geometry-aware donor selector
-> segment/block prompt plan
-> external temporal embedding lane
-> residual temporal patching
-> guidance objective mixer
-> null-condition edit audition
-> step/polish tradeoff audit
-> seed-family atlas
-> annotation-weighted recipe selection
-> report packager
```

Rationale:

- Start with measurement stability.
- Then improve prompt search.
- Then add preservation/copying checks.
- Then promote memory/control/geometry into selection tools.
- Only after that invest in sampler and residual interventions.
