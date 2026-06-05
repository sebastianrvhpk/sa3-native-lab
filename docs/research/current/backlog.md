# SA3 Native Lab Backlog

Status: future work and open questions for the research-program notebook.

This document answers: what should be tried next, why it matters, what evidence
would promote or drop it, and which questions remain open.

Implemented items are summarized for context. Future work should be added here,
not mixed into source context or method math.

## Backlog Discipline

Every backlog item should state:

```text
Goal:
Evidence already available:
Program touchpoints:
Promote if:
Drop or revise if:
```

Do not add a method just because a paper exists. Add it only when it can become
a concrete notebook run over SA3/SAME objects.

Use [Run protocol](run-protocol.md) before promoting a backlog item into a
notebook implementation or a kept method.

## Implemented Program Reference

These items now have notebook cells or probes:

| Item | Program Experiment | Helper Support | Status |
|---|---|---|---|
| Flow attribution prompt microscope | `CONDITION.flow_attribution` | `flow_prompt.py` attribution rows | implemented |
| Loss-by-timestep flow panel | `CONDITION.flow_timestep_panel` | `flow_prompt.py` loss rows and summaries | implemented |
| SAME control lanes | `MEMORY.control_lanes` | `control_lanes.py` | implemented |
| Dataset memory curriculum | `MEMORY.curriculum` | `curriculum.py` | implemented |
| Latent OT style transfer bench | `MEMORY.ot_style_transfer` | `geometry.py`, `style.py` | implemented |
| Continuation as bridge search | `MEMORY.bridge_search` | `composition.py` | implemented |
| Residual feature atlas | `RESIDUAL.feature_atlas` | `residual_features.py` | implemented |
| SA3 null-condition inversion probe | `CONDITION.null_inversion` | notebook probe over SA3 conditioning tensors | scaffold |
| Guidance-gradient latent edit | `GUIDANCE.gradient_edit` | `guidance.py` | scaffold |
| Audio-to-audio posterior guidance | `GUIDANCE.audio_posterior` | `guidance.py`, source/reference summaries | scaffold |
| Cross-model baseline harness | `BASELINE.cross_model` | notebook command harness plus descriptors/player | scaffold |

## Immediate Priorities

### 1. Probe-Bank Cache

Goal: save target latent, timesteps, noise seeds, and per-prompt rows so
`CONDITION.flow_attribution` and `CONDITION.flow_timestep_panel` reuse identical
probes across sessions.

Evidence already available: shared probe banks reduce prompt-score variance and
make A/B prompt comparisons fair.

Program touchpoints: `CONDITION.hard_prompt_search`, `CONDITION.flow_attribution`, `CONDITION.flow_timestep_panel`.

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

Program touchpoints: `CONDITION.hard_prompt_search`, `CONDITION.readable_prompt_search`, flow-microscope cells, descriptor/player cells.

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

Evidence already available: `MEMORY.control_lanes` extracts and compares control lanes;
`MEMORY.index_instrument` provides latent memory search.

Program touchpoints: `MEMORY.index_instrument`, `MEMORY.control_lanes`, `MEMORY.curriculum`, `MEMORY.bridge_search`.

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

Program touchpoints: `LATENT.audio_graft`, `LATENT.neural_dsp`, `MEMORY.geometry_audit`, `MEMORY.control_lanes`, `MEMORY.ot_style_transfer`, `MEMORY.bridge_search`.

Promote if:

- donor selection improves graft/style/bridge auditions,
- nearest-memory checks separate useful preservation from copying,
- geometry metrics explain audible differences.

Drop or revise if:

- donor ranking is no better than random or manual selection,
- geometry distance predicts nothing audible.

### 5. Residual Temporal Patching

Goal: extend `RESIDUAL.feature_atlas` from layer-level residual feature atlas to layer x
denoising-step or layer x latent-time patch tests.

Evidence already available: residual vectors and feature bases exist; activation
steering sources support causal intervention tests.

Program touchpoints: `RESIDUAL.prompt_steering`, `RESIDUAL.audio_steering`, `RESIDUAL.feature_atlas`.

Promote if:

- patching changes target descriptors/listening notes with limited side effects,
- layer/time maps are repeatable,
- residual atlas ranks match intervention outcomes.

Drop or revise if:

- hooks are too fragile,
- interventions only create artifacts,
- results are not repeatable across prompts/seeds.

### 6. Guidance Objective Mixer

Goal: let `GUIDANCE.gradient_edit` choose profile, boundary, period, lane, descriptor, and
preservation losses from a compact JSON recipe.

Evidence already available: generic gradient guidance exists; `MEMORY.geometry_audit`,
`MEMORY.control_lanes`, and `LATENT.neural_dsp` provide measurable loss candidates.

Program touchpoints: `MEMORY.geometry_audit`, `MEMORY.control_lanes`, `GUIDANCE.gradient_edit`, `GUIDANCE.audio_posterior`.

Promote if:

- objective recipes are inspectable,
- gradients move intended measurements,
- audio results beat prompt/audio-to-audio baselines.

Drop or revise if:

- objective hacking dominates audio quality,
- gradients are unstable or too expensive.

### 7. Null-Condition Edit Audition

Goal: after `CONDITION.null_inversion` optimizes null conditioning, generate
fixed prompt edits and compare source preservation against
`CONDITION.soft_prompt_inversion` and plain audio-to-audio.

Evidence already available: null-text inversion motivates branch-specific
conditioning; SA3 has CFG-like conditional/null behavior.

Program touchpoints: `CONDITION.soft_prompt_inversion`, `CONDITION.hard_prompt_search`, `CONDITION.null_inversion`.

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

Program touchpoints: all generation/editing experiments, especially
`LATENT.neighborhood_renoise`, `LATENT.audio_graft`, `LATENT.neural_dsp`,
`MEMORY.ot_style_transfer`, `MEMORY.bridge_search`, `GUIDANCE.gradient_edit`,
and `GUIDANCE.audio_posterior`.

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

Program touchpoints: `LATENT.neighborhood_renoise`, `CONDITION.hard_prompt_search`, `RESIDUAL.prompt_steering`, `MEMORY.ot_style_transfer`, `GUIDANCE.gradient_edit`, `BASELINE.cross_model`.

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

Program touchpoints: manifest cell, player, descriptor tables, experiment ledger.

Promote if:

- reports make review easier,
- selected outputs can be shared without rerunning Colab,
- report content remains honest about scaffold versus confirmed status.

Drop or revise if:

- report generation becomes a product/app direction,
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

Program touchpoints: `LATENT.blur_bottleneck`, `LATENT.neural_dsp`, `MEMORY.geometry_audit`, `CONDITION.flow_attribution`, `CONDITION.flow_timestep_panel`.

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

Program touchpoints: `CONDITION.soft_prompt_inversion`, `CONDITION.hard_prompt_search`, `CONDITION.flow_attribution`, `CONDITION.flow_timestep_panel`, `CONDITION.null_inversion`, `GUIDANCE.audio_posterior`.

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

Program touchpoints: `MEMORY.prompt_family`, `GUIDANCE.continuation_composition`, `MEMORY.index_instrument`, `MEMORY.control_lanes`, `MEMORY.curriculum`, `MEMORY.bridge_search`.

Promote if:

- segment plans improve continuity versus one global prompt,
- bridge/control-lane scores predict listening results,
- generated structures remain auditable in manifest rows.

Drop or revise if:

- segment boundaries dominate artifacts,
- the workflow becomes song-app scaffolding instead of notebook research.

### D. External Temporal Embedding Lane

Goal: optionally compare SAME-native evidence against CLAP/T-CLAP/ImageBind-like
semantic or temporal embedding distances.

Evidence already available: memory index, descriptors, control lanes, and
cross-model baseline harness exist.

Program touchpoints: `MEMORY.index_instrument`, `MEMORY.control_lanes`, `MEMORY.curriculum`, `BASELINE.cross_model`, player/ledger cells.

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

Program touchpoints: `LATENT.neighborhood_renoise`, `LATENT.blur_bottleneck`, `LATENT.neural_dsp`, `GUIDANCE.cyclic_trajectory`, `GUIDANCE.gradient_edit`, `GUIDANCE.audio_posterior`.

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

Program touchpoints: player, manifest, experiment ledger,
`LATENT.neighborhood_renoise`, `CONDITION.hard_prompt_search`,
`MEMORY.ot_style_transfer`, `GUIDANCE.gradient_edit`, `BASELINE.cross_model`.

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
- Cross-model fixed task packs beyond `BASELINE.cross_model`.
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
