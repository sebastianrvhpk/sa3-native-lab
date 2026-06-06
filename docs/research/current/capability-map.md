# SA3 Native Lab Capability Map

Status: bottom-up capability and maturity map for the current notebook and
`latent_audio_primitives/` package as of 2026-06-06.

This document answers: what the repo can actually do, which native-object
transitions it supports, what evidence exists, and what proof is needed before a
method can be promoted.

Evidence labels:

- `confirmed`: directly present in the notebook, local code, or docs.
- `repo-inferred`: implied by local code paths, but not executed in this pass.
- `hypothesis`: plausible next use, not yet validated by a Colab run.
- `unknown`: needs SA3/SAME weights, artifacts, or listening evidence.

## Boundary

This repo is a notebook research instrument. It keeps:

- one expanded Colab notebook,
- a notebook-facing primitive library,
- research docs and run protocol.

Upstream runtimes and training tools remain external dependencies that the
notebook consumes through checked-out repos, audio artifacts, checkpoints, and
run notes.

## Architecture Layers

Use [Architecture ontology](architecture-ontology.md) as the canonical layer
map. The short form:

| Architecture layer | Primary native objects | Main local capabilities |
|---|---|---|
| SAME Representation Science | `x`, `E`, `D`, `z0`, `LatentItem` | encode/decode, direct decode, geometry, bottleneck stress, latent DSP, memory, control lanes |
| SA3 Flow and Conditioning Science | `z_t`, `t`, `C(p)`, `v_theta` | shared probe banks, prompt flow scoring, attribution, soft/hard/readable inversion, null/condition probes |
| SA3 Internal Trajectory Science | residual activations, sampler states, step windows | residual capture, residual feature atlas, steering sweeps, cyclic projection, guidance scaffolds |
| SA3-over-SAME Coupled Editing | edited `z0'`, init/polish/inpaint/continue paths | neighborhood renoise, selective SA3, direct decode vs polish, source-preservation checks |
| Evidence and Listening Science | descriptors, memory rows, annotations, manifests | player, annotation store, disagreement rows, ledger, static report candidates |

## Native Objects

| Object | Shape / form | Owner | Lifecycle | Evidence |
|---|---|---|---|---|
| Audio waveform `x` | waveform arrays and `.wav` paths | notebook, `audio_descriptors.py`, `evidence/audio_player.py` | loaded, previewed, described, annotated | confirmed |
| SAME latent `z0` | usually `B x C x T`, memory rows use `T x D` | upstream SAME through adapters; local `LatentItem` | encoded, edited, decoded, searched, saved | confirmed |
| SA3 flow state `z_t` | noisy/intermediate latent state | `flow_prompt.py`, `procedures/flow_scoring.py` | constructed from `z0`, timestep, and noise | confirmed |
| Flow probe bank | timestep/logSNR, noise seed/sign, velocity convention | `flow_prompt.py`, notebook cells | reused across prompt variants and flow panels | confirmed |
| Prompt condition `C(p)` | SA3 conditioner outputs or optimized tensors | upstream SA3 plus `flow_prompt.py`, `procedures/soft_prompt.py` | scored, optimized, attributed, auditioned | confirmed |
| Prompt semantic row | prompt variant, tags, flow/listening evidence | `prompt_semantics.py`, notebook cells | compares raw, readable, and flow-found language | confirmed |
| Residual activation `a_l` | layer activation tensors | `adapters/sa3_residual_hooks.py`, residual procedures | captured, contrasted, steered, summarized | confirmed |
| `LatentItem` | ID, latent, rate, prompt, descriptors, labels, metadata | `schema.py`, `io.py` | saved, loaded, indexed, clustered | confirmed |
| Control lane | time-varying values, rate, confidence, metadata | `control_lanes.py` | extracted, normalized, compared, saved, rendered | confirmed |
| Descriptor report | JSON-friendly audio statistics | `audio_descriptors.py` | computed for source/baseline/method outputs | confirmed |
| Evidence packet | source/baseline/method outputs plus rows and notes | notebook, `evidence/`, ledger | records reviewable claims | template ready |
| Disagreement row | native evidence lanes for one artifact or prompt | `evidence/disagreement.py` | surfaces conflicts before decisions | confirmed |

## Operation Matrix

| Capability | Native transition | Operation | Code altitude | Maturity | Next proof |
|---|---|---|---|---|---|
| SA3/SAME runtime access | checkpoint/audio/prompt -> model handle or latent | observe/render | adapter | confirmed | Colab smoke packet per checkpoint |
| Audio descriptors | audio -> descriptor rows | observe/compare | root measurement | confirmed | listening agreement across first runs |
| SAME summaries | `z0` -> summary vectors | observe/compare | root measurement | confirmed | nearest-memory/source-preservation checks |
| Geometry reports | `z0` collection -> PCA/covariance/transport rows | observe/select | root measurement | microscope | connect geometry movement to audition |
| Periodicity and loop metrics | audio or `z0` -> loop rows | observe/compare | root measurement | microscope | compare against loop listening notes |
| Control lanes | audio/latent -> time-varying lanes | observe/select | root measurement plus evidence | microscope/selector | lane similarity must improve retrieval or review |
| Latent memory | `LatentItem` collection -> nearest rows | select | root memory | selector | show better donor/source/novelty decisions |
| Curriculum clustering | memory collection -> clusters/heldout rows | select | root memory | selector | show clusters improve prompt or donor choices |
| Bridge/continuation ranking | memory rows -> ranked paths | select | root composition | selector | bridge scores must predict audible continuity |
| Flow prompt scoring | target `z0` -> flow losses for prompts | observe/select | root rows plus procedure | microscope/selector | test whether scores predict generated audio |
| Shared flow probe banks | logSNR/timestep controls -> reusable probe manifest | observe/compare | root rows plus procedure | confirmed | use across prompt panels and ledger packets |
| Flow attribution | prompt -> token contribution rows | observe/select | root rows plus procedure | microscope | repeat over shared probe banks |
| Prompt semantic transparency | prompt variants -> tagged flow/listening rows | observe/select | root rows plus procedure | microscope/selector | show tags explain useful prompt changes |
| Soft prompt inversion | target `z0` -> optimized condition | intervene/render | procedure | intervention candidate | audition against prompt/audio-to-audio baselines |
| Hard/readable prompt search | candidate text -> ranked prompts | select | root search plus procedure scorer | selector | compare readable rankings against listening |
| SAME latent DSP | `z0` -> DSP-edited `z0'` -> audio | intervene/render | root operator plus procedure polish | intervention candidate | direct decode vs polish evidence packets |
| Blur/filter/low-rank | `z0` -> projected `z0'` -> audio | observe/intervene | root operator plus procedure polish | microscope/intervention candidate | identify which perturbations survive audio review |
| Channel renoise/graft | source/donor `z0` -> selected-channel edit | intervene/render | root operator plus procedure | intervention candidate | source/donor/baseline packets across clips |
| Style profile/direction | collection stats -> latent edit | intervene/render | root operator | intervention candidate | nearest-memory and listening checks for copying |
| Cyclic loop repair | audio/`z0` -> rolled/repair output | intervene/render | root operator plus procedure | intervention candidate | loop metrics must match loop audition |
| Residual activation capture | prompt/audio -> residual activations | observe | adapter plus procedure | microscope | layer maps must repeat |
| Residual steering | residual vector -> patched generation | intervene/render | adapter plus procedure | high-risk candidate | alpha sweeps must move audio without artifacts |
| Residual feature atlas | activations -> feature basis/report | observe/select | root measurement plus procedure | microscope | atlas rankings must predict interventions |
| Gradient/posterior guidance | objective -> latent/sampler update | intervene/render | root operator/scaffold | high-risk candidate | objective movement must beat baselines audibly |
| External comparison | imported outputs -> evidence packet | compare | evidence/procedure | comparison | fixed task packets with descriptors and notes |
| Audio player and annotations | output paths -> audition notes | decide | evidence | confirmed | routine ledger use |
| Semantic disagreement panel | artifact/prompt rows -> evidence conflicts | decide | evidence | microscope/decision support | show conflicts change promote/revise/drop choices |

## Artifact Flow

```text
audio file
  -> SAME latent item
  -> descriptor report / latent summary / control lanes
  -> memory index / geometry / curriculum
  -> retrieval rows or donor/source candidates

target audio latent
  -> frozen SA3 flow probes
  -> prompt loss rows / attribution rows / timestep panels
  -> hard prompt candidates or soft prompt state

edited latent or sampler state
  -> direct SAME decode
  -> optional SA3 polish or intervention
  -> output audio
  -> descriptor delta + nearest-memory rows + player annotation + ledger decision

prompt pairs or labeled audio
  -> residual activation capture
  -> vector basis / steering direction / alpha sweep
  -> generated outputs and review rows
```

## Parameter Inventory

Core generation:

```text
model ID, duration, sample rate, seed, steps, CFG, init noise, negative prompt
```

Flow conditioning:

```text
velocity convention, timesteps, logSNR values, probe count, antithetic noise,
normalized MSE, cosine weight, conditional-delta weight
```

SAME representation:

```text
sigma, masks, channel ranges, temporal ranges, blur radius, rank, FFT band,
phase blend, profile/direction alpha, covariance transport alpha
```

Residual and trajectory interventions:

```text
layer indices, residual axis, vector top-k, alpha, denoising step window,
guidance scale, loss weights, cyclic projection mix
```

Dataset memory:

```text
dataset limit, crop duration, cluster count, heldout fraction, distance metric,
hybrid weights, bridge weights, lane similarity weights
```

Evidence:

```text
descriptor config, lane frame seconds, annotation tags, manifest fields,
ledger decision
```

## Workbench Shape

The notebook should keep the same local shape inside every workbench:

```text
object
baseline
method
measurement
audition
decision
```

The stable section order is:

1. Runtime and model boundary.
2. Evidence packet setup.
3. Audio and SAME object preparation.
4. SAME measurement bench.
5. SA3 flow prompt bench.
6. SAME intervention bench.
7. Residual and trajectory bench.
8. Memory and composition bench.
9. External comparison bench.
10. Ledger and promotion board.

## Unknowns And Verification Plan

- Run one small Colab packet per workbench and record it in the ledger.
- Compare direct SAME decode against SA3 polish for every representation edit.
- Cache shared flow probes before expanding prompt-search variants.
- Add nearest-memory rows to every source-preservation claim.
- Treat residual and guidance methods as microscopes until alpha/guidance sweeps
  survive listening review.
- Keep Underfit and other model outputs as imported comparison artifacts, not
  local training infrastructure.
