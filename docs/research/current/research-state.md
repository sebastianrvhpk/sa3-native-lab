# SA3 Native Lab Research State

Status: current project snapshot for the notebook-first SA3 Native Lab direction
as of 2026-06-11.

This document answers: what the repo is for, which native objects are active,
how the helper library is organized, which claims are mature, and what evidence
is missing next.

## Direction

SA3 Native Lab is a Colab and Python research workspace for frozen SA3/SAME
latent-audio experiments. The source of truth is:

```text
colab/sa3_native_science_lab.ipynb
```

The repo keeps:

- the expanded Colab notebook,
- notebook-facing SA3/SAME helper primitives,
- current research notes,
- a Colab L4 setup runbook.

Stable Audio 3 is installed from the external upstream repository when model
weights are needed. Local code stays focused on notebook cells, helper
primitives, research notes, and Colab runtime setup.

LoRA work is externalized to [dada-bots/underfit](https://github.com/dada-bots/underfit).
This notebook can compare exported Underfit audio, checkpoints, and run notes
against frozen SA3/SAME methods.

## Evidence Labels

- `confirmed`: directly present in this repo, notebook, or a cited primary source.
- `implemented`: implemented as notebook cells or helper primitives.
- `scaffold`: present as a notebook probe or support surface, but not yet fully
  validated with real weights/listening.
- `hypothesis`: plausible behavior that still needs measurement.
- `unknown`: needs Colab execution, descriptor evidence, source inspection, or
  listening notes.

## Active Repo Surfaces

| Path | Role | Status |
|---|---|---|
| `colab/` | Expanded notebook plus Colab L4 runbook. | Active |
| `latent_audio_primitives/` | Notebook library for native objects, measurements, operators, model procedures, adapters, and evidence helpers. | Active |
| `docs/research/current/` | Current project state, method math, source context, run protocol, ledger, and backlog. | Active |
| `docs/research/current/run-protocol.md` | Operational spine for turning notebook outputs into evidence packets and decisions. | Active |

## Core Native Graph

```text
audio waveform x
  -> SAME encoder E
  -> SAME latent z0, usually B x C x T
  -> measurement, selection, intervention, memory, prompt objectives
  -> SA3 flow field v_theta(z_t, t, C(prompt))
  -> SAME decoder D
  -> decoded audio y
  -> evidence packet
  -> maturity decision
```

The research stance:

```text
freeze SA3
freeze SAME
edit prompts, soft conditioning, SAME latents, sampler states, residual activations,
or train small sidecar/control heads
measure with native flow losses, latent geometry, audio descriptors, memory rows,
control lanes, residual probes, listening notes, and experiment manifests
```

## Architecture Ontology

The project is not only a list of notebook workbenches. It has four
model-native research layers plus cross-cutting evidence utilities:

| Layer | Main question | Notebook implication |
|---|---|---|
| SAME Representation Science | What does SAME preserve, expose, erase, or make editable on its own? | Run direct-decode, bottleneck, geometry, memory, tuning-map, DSP, and control-lane experiments before invoking SA3 polish. |
| SA3 Flow and Conditioning Science | What does frozen SA3 know through prompt conditions, flow timesteps, and velocity fields? | Use shared flow probe banks, prompt variants, tuning-system prompt probes, attribution, and condition inversion as SA3-native evidence. |
| SA3 Internal Trajectory Science | What do residual streams, branch updates, condition gates, CFG/APG vectors, memory tokens, and sampler states reveal or causally control? | Run internal feature cartography first: scout, localize, select sparse-feature targets, then patch or steer only selected coordinates with audio evidence. |
| SA3-over-SAME Coupled Editing | How does SA3 read, repair, erase, or amplify SAME latents? | Compare every coupled edit against direct SAME decode and plain SA3 polish. |

Evidence utilities are not a fifth model-object layer. They are the review
system shared by all four layers: descriptors, memory rows, disagreement rows,
player annotations, manifests, and ledger decisions.

The canonical layer map is [Architecture ontology](architecture-ontology.md).

## Object Transition Workbenches

The notebook should read as a lab instrument over native-object transitions,
with each workbench moving from object to evidence to decision. The current
order follows native-object transitions: runtime and evidence surfaces first, SAME representation
next, then SA3 flow, SA3 internals, and finally coupled SA3-over-SAME editing.

| Workbench | Main transition | Main artifacts | Current status |
|---|---|---|---|
| Runtime and model boundary | upstream checkpoint -> model handle | model handles, latent rates, smoke audio | implemented |
| Evidence packet setup | output audio -> reviewable packet | player rows, descriptors, annotations, manifests | implemented |
| Audio and SAME preparation | audio -> SAME `z0` -> `LatentItem` | saved items, summaries, chunk windows | implemented |
| SAME representation bench | audio/`z0` -> summaries/geometry/lanes/tuning maps/direct decodes | descriptor, geometry, periodicity, tuning-map, control-lane, bottleneck rows | implemented |
| SAME memory and composition bench | collection -> selector -> continuation/bridge/donor | memory indices, curriculum rows, ranked bridges | implemented |
| SA3 flow and conditioning science | target `z0` -> flow probes or rendered prompt sweep -> prompt/condition score | shared probe banks, flow-loss rows, attribution, semantic prompt variants, tuning-system prompt/f0 rows, soft prompts, prompt candidates | implemented plus null-inversion scaffold |
| SA3 internal feature cartography | activation/state/branch/gate/CFG/APG -> surface rows -> selected cells/features -> bounded patch, branch intervention, sampler composition, or steer sweep -> output | contrastive residual scouts, internal surface rows, CFG/APG influence rows, sparse-feature scaffolds, residual patch specs, branch intervention specs, native trajectory-composition rows, alpha schedules, cyclic schedules | implemented as primary SA3-internal path |
| SA3-over-SAME coupled editing bench | `z0` -> edited `z0'` -> direct decode / SA3 polish | direct decodes, SA3-polished audio, deltas, source-preservation rows | implemented |
| External comparison bench | external artifacts -> evidence packet | Underfit/cross-model audio, descriptor/player rows | external/scaffold |
| Ledger and decision board | evidence packet -> maturity decision | ledger rows, promote/revise/drop decisions | template ready; no completed runs recorded |

## Claim Maturity Board

This board is a working classification before real Colab/listening runs. It
should be updated from `experiment-ledger.md`, not from speculation.

| Maturity | Current methods | What is missing |
|---|---|---|
| Microscope | tuning-map inference, flow sign diagnostic, flow attribution, loss-by-timestep, tuning-system pitch-class fit rows, geometry audit, periodicity, residual-timestep cartography, internal surface cartography, CFG/APG atlas, sparse-feature scaffold, sampler physiology, optional control-lane residual diagnostics | repeated listening evidence before control claims |
| Selector | memory index, curriculum, bridge search, prompt search, tokenizer vocabulary, tuning-map scalar target selection, tuning-system target-vs-null seed ranking, donor/source ranking ideas | evidence that rankings improve auditions |
| Intervention candidate | neighborhood renoise, selective renoise, graft, blur/filter, neural latent DSP, style profile/direction, cyclic repair, soft prompt audition | source/baseline/method packets across clips and seeds |
| High-risk intervention candidate | residual steering, branch output interventions, native trajectory composition, cyclic denoising projection, gradient guidance, posterior guidance, null-condition inversion | proof of causal movement without artifacts or fragile internals |
| Promoted method | none yet | at least repeated evidence packets and ledger decisions |
| External comparison | Underfit handoff and audio-output baseline harness | imported audio artifacts and fixed comparison packets |

## Reusable Local Modules

`latent_audio_primitives`, current role:

- Root native objects, math, measurements, and operators:
  `schema.py`, `io.py`, `latent_math.py`, `audio_descriptors.py`,
  `geometry.py`, `periodic.py`, `control_lanes.py`, `tuning_maps.py`,
  `tuning_systems.py`, `observability.py`,
  `latent_constraints.py`, `residual_probes.py`, `internal_features.py`, `latent_blur.py`,
  `latent_dsp.py`, `selective_renoise.py`, `looping.py`,
  `style.py`, `flow_prompt.py`, `prompt_semantics.py`,
  `prompt_optimization.py`, `tokenizer_vocab.py`, `index.py`,
  `curriculum.py`, `composition.py`, `guidance.py`, `trajectory.py`,
  `sampler_composition.py`, `measurement_recipes.py`, and
  `residual_features.py`.
- Model boundary: `adapters/` isolates official SA3/SAME loading, encoding,
  generation, tokenizer access, and residual-hook surfaces.
- Executable procedures: `procedures/` runs SA3/SAME flow scoring, soft prompt
  optimization, SA3 polish, selective SA3, cyclic SA3, residual extraction, and
  residual sweeps. Procedure maturity is documented in research docs and ledger
  decisions rather than encoded as package-level metadata.
- Evidence loop: `evidence/audio_player.py`, `evidence/annotations.py`,
  `evidence/disagreement.py`, `evidence/control_lane_rendering.py`, and
  `audio_descriptors.py` turn outputs into reviewable packets and decisions.

The detailed module map is [Primitive map](primitive-map.md). The object and
capability map is [Capability map](capability-map.md).

## Primitive Function Audit

- All `latent_audio_primitives/` modules compile and import under the repo
  environment.
- Public functions/classes have docstrings.
- Descriptor-target memory scoring is owned by `index.py` with memory search,
  `query_controls()`, and `query_hybrid()`.
- Native tokenizer extraction is owned by `tokenizer_vocab.py` with native
  vocabulary filtering and preview helpers.
- Residual-timestep cartography is owned by `trajectory.py` with ranked cells,
  band summaries, trajectory-derived flow probe banks, residual alpha schedules,
  cyclic mix schedules, and sampler-step record normalization.
- SA3 internal feature cartography is owned by `internal_features.py`,
  `adapters/sa3_internal_hooks.py`, and
  `procedures/internal_feature_cartography.py`: surface specs, branch/gate
  capture rows, CFG/APG condition-influence rows, memory-token summaries,
  sparse-feature target scaffolds, clean/corrupt post-block residual patch
  sweeps, and branch output intervention sweeps.
- Residual examples, steering vector containers, and layer/window/timestep
  probe rows are owned by `residual_probes.py`; residual procedures only
  collect activations and render sweeps.
- Latent scalar objectives are owned by `latent_constraints.py` so optimization
  losses are separated from evidence-packet row aggregation.
- Measurement recipes are owned by `measurement_recipes.py` with row helpers
  for SAME bottleneck tomography, flow-semantic cartography, coupled edit
  survival, control identification, source cartography, factor atlas rows,
  long-form composition rows, condition geometry, and sampler physiology.
- A synthetic smoke run covers the NumPy-only research grammar: `LatentItem`,
  summaries, geometry, periodicity, control lanes, audio descriptors, memory,
  curriculum, composition, style profiles/directions, flow-probe manifests,
  prompt semantic rows, and disagreement rows.
- Control-lane evidence now separates complete measurement from probe
  selection: the notebook exports full lane/channel artifacts, compact
  channel-region overlap audits, ranked target rows, and an internal-cartography
  target manifest before any expensive SA3-internal run is launched.
- Tuning-system evidence is owned by `tuning_systems.py`: default xenharmonic,
  non-octave, and JI limit-set prompt banks, lightweight monophonic f0 rows,
  per-system pitch-class fits, and target-vs-null selectivity rows. This is a
  prompt-conditioning microscope/selector until repeated generated outputs and
  listening notes show stable intonation movement.
- Tuning-map inference is owned by `tuning_maps.py`: audio f0 rows become pitch
  events, pitch centers, interval-ratio edges, period/generator candidates,
  Wilson-style CPS fit rows, and scalar targets for future SAME/SA3 probes.
  This is the first pitch-as-information object and must pass listening/null
  checks before any "tuning vector" language is used.
- Pitch-relation representation discovery should be audio-first: measured
  `TuningMap` fields from real or synthetic clips become targets for SAME
  latent, audio-conditioned residual, condition-state, or adapter/SAE probes.
  Prompt-based generation should receive discovered directions only after audio
  evidence shows they are readable and not just verbal style priors.

Conclusion: the current library shape is coherent. The remaining issue is not
whether these files are needed; it is whether each notebook method earns
promotion through repeated evidence packets.

## Procedure Honesty Board

Executable procedure modules are not all equally mature. They should be read as
methods under review, not as a list of promoted controls.

| Procedure | Layer | Current maturity | Promotion gate |
|---|---|---|---|
| `procedures/flow_scoring.py` | SA3 flow/conditioning | microscope / selector | Flow rankings predict generated-audio or listening outcomes across shared probe banks. |
| `procedures/soft_prompt.py` | SA3 flow/conditioning | intervention candidate | Optimized conditions, optionally weighted by trajectory-derived probe banks, beat readable prompt and audio-to-audio baselines without losing reviewability. |
| `procedures/sa3_latent_sampling.py` | SA3-over-SAME coupled editing | intervention candidate | Method packets show which SAME edits survive direct decode and SA3 polish, with sampler-step records when relevant. |
| `procedures/selective_sa3.py` | SA3-over-SAME coupled editing | intervention candidate | Source/baseline/method packets show predictable channel, donor, prompt, and sampler-step effects. |
| `procedures/cyclic_sa3.py` | SA3 internal trajectory | high-risk sampler microscope / intervention candidate | Loop metrics and auditions improve versus baselines without collapse or sampler artifacts; trajectory-derived mix schedules must beat uniform mix. |
| `procedures/residual_activation_vectors.py` | SA3 internal trajectory | microscope / selector | Prompt-derived residual examples produce cross-validated layer, sampler-timestep, and trajectory-window probe rows before steering; this is the Audioscope-style scout, not final causal evidence. |
| `procedures/audio_residual_vectors.py` | SA3 internal trajectory | high-risk microscope / selector | Audio-derived residual examples produce layer, sampler-timestep, and trajectory-window probe rows; directions still need source-leakage and alpha-sweep review. |
| `procedures/internal_feature_cartography.py` | SA3 internal trajectory | microscope / selector / intervention candidate | Internal surface capture, CFG/APG atlas, sparse-feature target scaffolds, bounded clean/corrupt post-block residual patch sweeps, and branch output intervention sweeps form the primary SA3-internal path. |
| `procedures/sampler_composition.py` | SA3 internal trajectory | high-risk intervention candidate | Runs explicit RF Euler source-latent composition with source anchoring, CFG/APG schedules, and prompt phases. |
| `procedures/control_lane_mechanistic_probe.py` | SA3 internal trajectory plus SAME representation evidence | optional microscope / selector | Control-lane residual rows remain available as a diagnostic selector, but they are not the main research path and do not prove SA3 causality. |
| `procedures/residual_sweeps.py` | SA3 internal trajectory | high-risk intervention candidate | Global or trajectory-gated alpha changes are audible, monotonic or interpretable, and not just artifact injection. |

## Artifact Graph

```text
audio files
  -> SAME encoder
  -> SAME latents
  -> latent memory / style profile / tuning maps / control lanes / DSP edits / geometry ops
  -> SAME decoder or SA3 polish
  -> audio outputs + descriptor tables + manifests + listening notes

target audio
  -> SAME z0
  -> frozen flow probe bank
  -> prompt scores / attribution rows / loss-by-timestep panels
  -> readable prompt candidates or soft prompt optimization

dataset folders
  -> pre-encoded latents + descriptors
  -> retrieval, curriculum, bridge search, donor selection
  -> notebook comparison cells

prompt pairs, labeled audio sets, or sampler condition contrasts
  -> SA3 residual / branch / gate / CFG-APG capture
  -> internal surface rows / steering vectors / sparse-feature targets
  -> bounded patch, branch, sampler-composition, or alpha sweeps
  -> generated outputs + probe reports

tuning-system prompt family
  -> SA3 rendered audio
  -> monophonic f0 rows / pitch-class lattice fits
  -> target-vs-null selectivity rows + listening notes

real-world pitch behavior
  -> tuning map
  -> scalar map fields
  -> SAME/condition/residual probe targets

audio clips with measured pitch-relation differences
  -> SAME latents and/or audio-conditioned SA3 residual states
  -> linear readability probes or SAE/adapter features
  -> later prompt-generation attachment only after audio-first evidence

LoRA/style fine-tuning need
  -> use Underfit
  -> bring back generated audio/checkpoints as comparison artifacts
```

## Runtime Assumptions

- Colab L4 is the intended notebook runtime for SA3 Medium.
- SA3 Medium uses SAME-L and requires CUDA plus FlashAttention for normal use.
- SA3 Small Music/SFX use SAME-S and are CPU-capable according to the upstream repo.
- Validation centers on notebook execution, JSON integrity, Colab smoke runs,
  descriptor deltas, manifests, and listening notes.
- Stable Audio 3 upstream/reference docs live in the external upstream repo.
- Underfit LoRA runs are validated in Underfit, then imported as audio/checkpoint
  comparison artifacts.

## Current Unknowns

- Does native flow score predict generated-audio similarity, or only
  teacher-forced vector-field agreement?
- Which logSNR bands correspond to style, structure, transient detail, or prompt
  adherence?
- Can SA3 repeatedly render audible xenharmonic or JI limit-set differences
  from prompt wording alone, or do f0-fit wins only reflect chance/root fitting?
- Can real-world tuning maps be inferred robustly enough to become SAME/SA3
  probe targets, and do these map fields exist as readable latent/residual
  subspaces?
- Which SAME latent edits survive direct decode and SA3 polish?
- Which control lanes and SAME-channel families are observable, predictable,
  and intervenable after complete-atlas review rather than top-k display alone?
- Which SA3 internal surfaces, layers, branches, CFG/APG components, and
  sampler bands produce stable causal audio controls?
- Can sampler-level guidance improve loopability or source preservation without
  artifacts?
- Which evidence panels are lightweight enough for routine Colab review?
- Which methods should graduate after the first real ledger entries?
