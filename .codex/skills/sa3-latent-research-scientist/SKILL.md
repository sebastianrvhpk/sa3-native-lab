---
name: sa3-latent-research-scientist
description: Use when formalizing, auditing, extending, or reviewing SA3/SAME latent-audio notebook research; designing evidence studies over SAME latents, SA3 flow fields, prompt conditions, residual activations, sampler trajectories, control lanes, memory, descriptors, or listening evidence; framing object transitions, operations, measurements, claim maturity, code altitude, active-source evidence, null controls, and exact-vs-observed trajectory claims; deciding whether a method is a microscope, selector, intervention candidate, promoted method, or noise.
---

# SA3 Latent Research Scientist

Use this skill to keep SA3 Native Lab scientifically coherent. The goal is not
to add methods quickly; it is to make native objects, transitions, operations,
measurements, claim maturity, and decisions explicit.

## First Move

Before editing code or docs, state the research frame:

```text
Research layer / evidence utility: SAME representation / SA3 flow-conditioning /
  SA3 internal trajectory / SA3-over-SAME coupled editing / evidence utility
Object: what native object is under study?
Transition: what maps into what, or what state is compared?
Operation: observe, select, intervene, render, compare, or decide?
Measurement: what evidence is collected?
Claim: what would success mean?
Decision: promote, revise, drop, or keep as microscope/selector?
Maturity: microscope / selector / intervention candidate / promoted method
Notebook study surface: which concrete cell/group owns the run?
Altitude: root primitive / adapter / procedure / evidence / docs-only
```

If the user asks to implement immediately, still write this frame first in one
compact paragraph, then proceed. Keep altitude as code placement, not as the
scientific claim itself.

Research layer is not the same as notebook study surface. There are four
model-native research layers; evidence utilities are cross-cutting review and
promotion infrastructure:

- SAME representation: SAME on its own, direct decode, geometry, memory,
  bottleneck stress, latent DSP, active-source control lanes, MIR/DSP lane
  banks.
- SA3 flow-conditioning: prompt conditions, flow states, logSNR/timestep probes,
  prompt inversion, condition counterfactuals.
- SA3 internal trajectory: residual activations, sampler states, guidance
  objectives, control-lane probes, exact sampler-step maps, observed hook-call
  windows, layer/time causality.
- SA3-over-SAME coupled editing: SAME edits entering SA3 polish, inpainting,
  continuation, audio-to-audio, rescue/erasure tests.
- Evidence utility: descriptors, player notes, disagreement rows, manifests,
  and ledger decisions that audit all four research layers.

## Repo Context To Inspect

Prefer local evidence before invention:

- `colab/sa3_native_science_lab.ipynb`
- `latent_audio_primitives/README.md`
- `docs/research/current/primitive-map.md`
- `docs/research/current/capability-map.md`
- `docs/research/current/research-state.md`
- `docs/research/current/architecture-ontology.md`
- `docs/research/current/methods-and-math.md`
- `docs/research/current/run-protocol.md`
- `docs/research/current/experiment-ledger.md`
- `docs/research/current/backlog.md`

Use `rg` for references and notebook imports before moving or deleting anything.

## Native Study Grammar

Frame methods as evidence studies over native-object transitions. Notebook
sections are execution surfaces, not scientific categories or claims.

- Runtime and model boundary: proves what upstream SA3/SAME access exists.
- Evidence packet setup: defines artifacts, rows, listening notes, and ledger
  decisions.
- Audio and SAME preparation: maps source audio into SAME `z0` and renderable
  audio outputs.
- SAME representation science: observes latent geometry, descriptors,
  periodicity, control lanes, direct decode, memory, and source-preservation
  signals.
- SA3 flow and conditioning science: scores prompt/state relations through shared flow
  probe banks, prompt semantic rows, timestep/logSNR panels, and
  convention-explicit losses.
- SA3 internal trajectory science: captures activations, directions, and
  sampler paths without pretending probes are controls.
- SA3-over-SAME coupled editing: edits SAME latents and tests direct decode
  against SA3 polish, continuation, inpainting, and audio-to-audio paths.
- SAME memory and composition: indexes, retrieves, bridges, or compares donors
  while guarding against copying.
- External comparison: evaluates Underfit or other exported audio
  artifacts through the local evidence packet, without importing training
  scaffolding or another semantic judge into the repo.
- Ledger and decision board: turns evidence packets into promote/revise/drop
  decisions.

## Current Hard Rules

- Prefer true source duration. If a notebook duration pads the source, export an
  active-source span and use active-window summaries, regions, and correlations.
- A visible lane is not a control. Control-lane rows must beat shuffled,
  reversed, or random nulls before they become selectors, and still need
  decoded/listening evidence before intervention language.
- Control-lane probe rows should separate token-blocked visibility from
  call-held-out evidence when multiple observed calls are available.
- Observed hook-call windows are not sampler timesteps. Timestep rows must carry
  `mapping_status`; only `exact_one_call_per_step` is clean attribution.
- Continuous lane/timestep probes need token-preserving activations; mean-pooled
  timestep vectors collapse the lane to one sample and are diagnostics only.
- Prediction curves and active/quiet direction previews are diagnostics. They
  do not become residual steering vectors until a separate bounded sweep proves
  audible movement.
- External semantic judges, including CLIP/CLAP-style embeddings or deep MIR
  models, stay docs-only unless reframed as native SAME/SA3 measurement
  questions. Start with deterministic audio/SAME lanes.
- Do not add app, service, dashboard, framework, or orchestration scaffolding.
  The repo is a Colab-first research instrument.

Use these operation roles:

- Observe: reveal structure without changing generation.
- Select: rank prompts, donors, seeds, clusters, channels, or recipes.
- Intervene: change a native object and test audio/descriptor movement.
- Render: decode, polish, or export audio for audition.
- Compare: place local outputs against baselines, memories, or external runs.
- Decide: promote, revise, drop, or keep as microscope/selector.

Claim maturity:

- Microscope: reveals structure but is not yet a control.
- Selector: chooses among candidates but does not itself create the change.
- Intervention candidate: produces a measurable change that needs repeatability
  and listening evidence.
- Promoted method: has math rationale, compact notebook use, measurements,
  listening notes, and repeatability.

## Code Altitude

Place code by what it owns:

- Root modules define or transform native objects: latent math, geometry,
  control lanes, control-lane probes, flow probe banks, prompt semantic rows,
  token search, descriptors, memory, looping, DSP.
- `adapters/` talks to external machinery: upstream SA3/SAME, tokenizer access,
  residual hook locations, model loading, encode/decode wrappers.
- `procedures/` runs executable research methods that call SA3/SAME, optimize
  conditions, capture hooks, generate sweeps, or polish edited latents.
- `evidence/` supports auditioning, annotation, display, disagreement rows, and
  review.
- Docs-only belongs in `docs/research/current/` until it has a compact
  notebook-facing primitive or cell.

Do not add an orchestration framework to make this tidier. Prefer explicit
functions, dataclasses, rows, and notebook imports from concrete modules.

## Evidence Standard

Do not call a method promoted until it has:

1. a clear mathematical or architectural rationale,
2. a compact notebook-facing API or cell,
3. active-source-aware descriptor/latent/lane evidence when audio duration is
   involved,
4. null/baseline rows when a probe or selector is claimed,
5. listening notes or a plan for listening notes,
6. promote/revise/drop criteria.

Keep a method as a microscope if it only explains model behavior. Keep it as a
selector if it ranks candidates without proving an independent intervention.

Label claims as:

- `confirmed`: present in local code/docs/notebook or directly observed.
- `repo-inferred`: implied by local code but not executed.
- `source-inferred`: supported by an external source that was actually checked.
- `hypothesis`: plausible but unverified.
- `unknown`: cannot be determined yet.

## Output Shape

For audits or proposals, return:

```text
Research frame
Object transition map
Notebook study surface and maturity placement
Experiment cards
Evidence packet plan
Failure modes
Promote/revise/drop criteria
Notebook/docs/code impact
```

For implementation, use this loop:

1. Inspect current imports and nearby modules.
2. Decide notebook study surface, claim maturity, and code altitude before moving or adding
   code.
3. Preserve math and signatures unless the change is explicitly a redesign.
4. Update notebook study cells and imports when public paths move.
5. Update the relevant docs after code changes:

- Method math and conventions: `docs/research/current/methods-and-math.md`
- Current state/module role: `docs/research/current/research-state.md` or
  `docs/research/current/primitive-map.md`
- Future work or promotion criteria: `docs/research/current/backlog.md`
- Real runs/listening decisions: `docs/research/current/experiment-ledger.md`
- Run protocol or notebook execution order:
  `docs/research/current/run-protocol.md`
- Research-layer/evidence-utility framing:
  `docs/research/current/architecture-ontology.md`

6. Validate the notebook JSON/imports and run `git diff --check`.

## References

- Use `references/experiment-card.md` when drafting experiment proposals.
- Use `references/evidence-standards.md` when reviewing whether a method is
  ready to promote.
