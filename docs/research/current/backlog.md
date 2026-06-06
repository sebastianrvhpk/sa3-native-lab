# SA3 Native Lab Backlog

Status: future work organized by missing evidence, not by method inventory.

This document answers: what proof is missing next, which notebook runs should
create that proof, and what would promote, revise, or drop each line of work.

## Backlog Discipline

Every backlog item should state:

```text
Evidence gap:
Native transition:
Current support:
Next run:
Promote if:
Drop or revise if:
```

Do not add a method just because a paper exists. Add it only when it can become
a concrete notebook run over SA3/SAME objects and produce an evidence packet.

Use [Run protocol](run-protocol.md) before promoting a backlog item into a kept
method.

## Current Maturity Reference

| Maturity | Items | Evidence status |
|---|---|---|
| Microscope | flow sign diagnostic, flow attribution, loss-by-timestep, geometry audit, periodicity, residual feature atlas | implemented, needs repeated listening linkage |
| Selector | memory index, curriculum, bridge search, prompt search, tokenizer vocabulary, donor/source ranking ideas | implemented or scaffolded, needs proof that rankings improve auditions |
| Intervention candidate | neighborhood renoise, selective renoise, graft, blur/filter, neural latent DSP, style profile/direction, cyclic repair, soft prompt audition | implemented, needs source/baseline/method packets |
| High-risk intervention candidate | residual steering, cyclic trajectory, gradient guidance, posterior guidance, null-condition inversion | implemented/scaffolded, needs causal proof and artifact checks |
| Promoted method | none yet | no repeated ledger evidence yet |
| External comparison | Underfit handoff, cross-model harness, optional external embeddings | scaffolded/import-only |

## Gap 1. Reproducible Flow Probe Evidence

Evidence gap: prompt/condition scores can vary unless probe banks are shared and
cached.

Native transition: `target z0 -> flow probe bank -> prompt/condition loss rows`.

Current support: `flow_prompt.py`, `procedures/flow_scoring.py`, prompt search,
flow attribution, and loss-by-timestep cells.

Next run: implement or use a probe-bank cache that records target latent,
timesteps/logSNRs, noise seeds, velocity convention, model ID, and per-prompt
rows.

Promote if:

- repeated prompt scoring is reproducible across notebook sessions,
- attribution and loss-by-timestep panels reuse identical probes,
- cache metadata is easy to inspect.

Drop or revise if:

- cache files become harder to audit than rerunning probes,
- stale probe metadata creates mismatch risk.

## Gap 2. Flow Score Predictive Validity

Evidence gap: it is unknown whether teacher-forced flow agreement predicts
generated audio quality, source similarity, or prompt adherence.

Native transition: `target z0 -> flow score -> generated output -> evidence packet`.

Current support: hard/readable prompt search, soft prompt inversion/audition,
flow attribution, descriptors, player, and annotations.

Next run: compare top, middle, and poor flow-ranked prompts or conditions
against decoded/generated audio, descriptors, and listening notes.

Promote if:

- flow-improved candidates sound closer or more useful than baselines,
- logSNR bands explain audible differences,
- readable prompts remain auditable.

Drop or revise if:

- flow rankings only measure vector-field agreement,
- descriptor/listening evidence contradicts flow rankings.

## Gap 3. Source Preservation Versus Copying

Evidence gap: the notebook needs a routine panel that separates source identity,
novelty, and accidental memory copying.

Native transition: `method output -> nearest-memory rows + descriptor delta + listening note`.

Current support: `LatentMemoryIndex`, descriptors, manifests, player,
annotations, curriculum rows.

Next run: add nearest-memory rows and descriptor deltas to every generated
artifact in one representative edit chain.

Promote if:

- reviewers can distinguish copying, preservation, and novelty,
- nearest-memory rows match listening judgement often enough to be useful,
- the panel is compact enough for normal Colab use.

Drop or revise if:

- nearest-memory metrics do not match listening judgement,
- the panel overwhelms the notebook.

## Gap 4. Direct Decode Versus SA3 Polish

Evidence gap: SAME latent edits may survive direct decode but be erased or
rewritten by SA3 polish.

Native transition: `edited z0' -> direct decode / SA3 polish -> evidence packet`.

Current support: latent blur/filter, neural DSP, selective renoise, graft,
style profile/direction, cyclic repair, SA3 polish procedures.

Next run: for each edit family, generate source, direct decode, plain polish,
and method polish with descriptor deltas and listening notes.

Promote if:

- some edits survive both direct decode and polish,
- polish improves audio without erasing intended movement,
- failure families are clear.

Drop or revise if:

- polish erases every measured difference,
- direct decodes are dominated by artifacts.

## Gap 5. Control-Lane And Geometry Selection Value

Evidence gap: geometry, control lanes, and bridge costs are measurable, but
their selection value is not yet proven.

Native transition: `collection -> lane/geometry/memory ranking -> donor/bridge/source choice`.

Current support: geometry reports, control lanes, memory index, curriculum,
bridge ranking, composition helpers.

Next run: compare manual/random donor or bridge choices against lane/geometry
ranked choices.

Promote if:

- ranked choices improve auditions,
- metrics explain audible differences,
- heldout rows prevent overfitting to one dataset.

Drop or revise if:

- rankings are no better than random/manual choice,
- metrics duplicate simple descriptors without added value.

## Gap 6. Residual Causality

Evidence gap: residual activations are measurable, but causal audio movement is
not yet established.

Native transition: `prompt/audio examples -> residual direction -> alpha sweep -> evidence packet`.

Current support: residual hooks, prompt/audio vector extraction, alpha sweeps,
residual feature basis, player/descriptors.

Next run: run layer/alpha sweeps on one prompt-pair vector and one audio-derived
vector with baseline, descriptor deltas, and listening notes.

Promote if:

- alpha changes target qualities monotonically or predictably,
- effects repeat across seeds/prompts,
- artifacts remain bounded.

Drop or revise if:

- hooks are too fragile,
- steering only creates artifacts,
- layer rankings do not repeat.

## Gap 7. Guidance Objective Honesty

Evidence gap: gradient/posterior guidance can optimize measurements without
improving audio.

Native transition: `objective recipe -> latent/sampler update -> output evidence`.

Current support: `guidance.py`, geometry, control lanes, periodicity, descriptor
loss candidates, gradient edit and audio posterior scaffolds.

Next run: test a compact JSON-like objective recipe against a prompt/audio
baseline.

Promote if:

- objective movement aligns with listening,
- source preservation improves over baselines,
- recipes remain inspectable.

Drop or revise if:

- objective hacking dominates,
- gradients are unstable or expensive,
- audio quality falls while metrics improve.

## Gap 8. Null-Condition Editing

Evidence gap: null-condition inversion may preserve source identity while
keeping text edits editable, but this is unproven locally.

Native transition: `target z0 -> optimized null condition -> fixed prompt edit -> evidence packet`.

Current support: null-condition inversion scaffold, soft prompt inversion,
flow scoring, CFG source context.

Next run: compare null-condition edit, soft prompt edit, and plain audio-to-audio
edit on one target.

Promote if:

- source identity is preserved better than baselines,
- text edits remain effective,
- learned null conditioning transfers to nearby prompts.

Drop or revise if:

- prompt edits collapse,
- preservation does not beat audio-to-audio init,
- conditioner internals are too unstable.

## Gap 9. Seed And Recipe Repeatability

Evidence gap: many current operators may work only for one seed or clip.

Native transition: `fixed recipe -> seed grid -> clusters + listening tags`.

Current support: descriptor reports, memory clustering, annotation tags,
manifest fields.

Next run: generate a small seed grid for one flow prompt, one SAME edit, and one
residual/trajectory candidate.

Promote if:

- seed families reveal repeatable behavior,
- clusters help select robust recipes,
- failure cases become visible.

Drop or revise if:

- seed grids consume too much runtime,
- clusters are not interpretable.

## Gap 10. External Comparison Packets

Evidence gap: Underfit and other external outputs need the same evidence packet
shape as local frozen-SA3 runs.

Native transition: `external output -> descriptor/player/memory evidence -> comparison decision`.

Current support: Underfit handoff, cross-model command harness, source context,
player, descriptors, memory.

Next run: import one Underfit or external-model output set and compare against
fixed SA3 prompts through the same descriptor/player/ledger path.

Promote if:

- comparisons are repeatable without local training or model-management
  infrastructure,
- fixed task packets expose what frozen SA3/SAME cannot do,
- results remain honest about external dependencies.

Drop or revise if:

- model management creeps into this repo,
- comparison setup distracts from notebook research.

## Gap 11. Reportable Evidence Packets

Evidence gap: generated artifacts are reviewable inside Colab, but not yet
packaged as concise shareable run reports.

Native transition: `manifest rows + audio + descriptors + annotations -> static report`.

Current support: manifests, player, descriptors, annotations, ledger template.

Next run: export one completed evidence packet into a static Markdown/JSON
report without adding another tool surface.

Promote if:

- reports make review easier,
- selected outputs can be shared without rerunning Colab,
- report content preserves uncertainty and maturity labels.

Drop or revise if:

- report generation becomes more important than evidence review,
- it distracts from notebook-first research.

## Priority Order

1. Reproducible flow probe evidence.
2. Source preservation versus copying panel.
3. Direct decode versus SA3 polish audit.
4. Flow score predictive validity.
5. Control-lane and geometry selection value.
6. Residual causality.
7. Guidance objective honesty.
8. Null-condition editing.
9. Seed and recipe repeatability.
10. External comparison packets.
11. Reportable evidence packets.

The first three stabilize evidence. The next five test whether the strongest
methods deserve promotion. The last three make the lab easier to review and
compare without adding extra infrastructure.
