# SA3 Native Lab Research State

Status: current project snapshot for the notebook-first SA3 Native Lab direction
as of 2026-06-06.

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
| SAME Representation Science | What does SAME preserve, expose, erase, or make editable on its own? | Run direct-decode, bottleneck, geometry, memory, DSP, and control-lane experiments before invoking SA3 polish. |
| SA3 Flow and Conditioning Science | What does frozen SA3 know through prompt conditions, flow timesteps, and velocity fields? | Use shared flow probe banks, prompt variants, attribution, and condition inversion as SA3-native evidence. |
| SA3 Internal Trajectory Science | What do residual activations and sampler states reveal or causally control? | Keep residual/guidance work as microscope or high-risk candidate until layer/time sweeps survive audio review. |
| SA3-over-SAME Coupled Editing | How does SA3 read, repair, erase, or amplify SAME latents? | Compare every coupled edit against direct SAME decode and plain SA3 polish. |

Evidence utilities are not a fifth model-object layer. They are the review
system shared by all four layers: descriptors, memory rows, disagreement rows,
player annotations, manifests, and ledger decisions.

The canonical layer map is [Architecture ontology](architecture-ontology.md).

## Object Transition Workbenches

The notebook should read as a lab instrument over native-object transitions,
with each workbench moving from object to evidence to decision.

| Workbench | Main transition | Main artifacts | Current status |
|---|---|---|---|
| Runtime and model boundary | upstream checkpoint -> model handle | model handles, latent rates, smoke audio | implemented |
| Evidence packet setup | output audio -> reviewable packet | player rows, descriptors, annotations, manifests | implemented |
| Audio and SAME preparation | audio -> SAME `z0` -> `LatentItem` | saved items, summaries, chunk windows | implemented |
| SAME measurement bench | `z0` -> summaries/geometry/lanes | descriptor, geometry, periodicity, control-lane rows | implemented |
| SA3 flow prompt bench | target `z0` -> flow probes -> prompt/condition score | shared probe banks, flow-loss rows, attribution, semantic prompt variants, soft prompts, prompt candidates | implemented plus null-inversion scaffold |
| SAME intervention bench | `z0` -> edited `z0'` -> decode/polish | direct decodes, SA3-polished audio, deltas | implemented |
| Residual and trajectory bench | activation/state -> intervention -> output | residual vectors, alpha sweeps, guided variants | implemented plus high-risk scaffolds |
| Memory and composition bench | collection -> selector -> continuation/bridge/donor | memory indices, curriculum rows, ranked bridges | implemented |
| External comparison bench | external artifacts -> evidence packet | Underfit/cross-model audio, descriptor/player rows | external/scaffold |
| Ledger and promotion board | evidence packet -> maturity decision | ledger rows, promote/revise/drop decisions | template ready; no completed runs recorded |

## Claim Maturity Board

This board is a working classification before real Colab/listening runs. It
should be updated from `experiment-ledger.md`, not from speculation.

| Maturity | Current methods | What is missing |
|---|---|---|
| Microscope | flow sign diagnostic, flow attribution, loss-by-timestep, geometry audit, periodicity, residual feature atlas | repeated listening evidence before control claims |
| Selector | memory index, curriculum, bridge search, prompt search, tokenizer vocabulary, donor/source ranking ideas | evidence that rankings improve auditions |
| Intervention candidate | neighborhood renoise, selective renoise, graft, blur/filter, neural latent DSP, style profile/direction, cyclic repair, soft prompt audition | source/baseline/method packets across clips and seeds |
| High-risk intervention candidate | residual steering, cyclic denoising projection, gradient guidance, posterior guidance, null-condition inversion | proof of causal movement without artifacts or fragile internals |
| Promoted method | none yet | at least repeated evidence packets and ledger decisions |
| External comparison | Underfit handoff and audio-output baseline harness | imported audio artifacts and fixed comparison packets |

## Reusable Local Modules

`latent_audio_primitives`, current role:

- Root native objects, math, measurements, and operators:
  `schema.py`, `io.py`, `latent_math.py`, `audio_descriptors.py`,
  `geometry.py`, `periodic.py`, `control_lanes.py`, `observability.py`,
  `latent_blur.py`, `latent_dsp.py`, `selective_renoise.py`, `looping.py`,
  `style.py`, `flow_prompt.py`, `prompt_semantics.py`,
  `prompt_optimization.py`, `tokenizer_vocab.py`, `index.py`,
  `curriculum.py`, `composition.py`, `guidance.py`, and
  `residual_features.py`.
- Model boundary: `adapters/` isolates official SA3/SAME loading, encoding,
  generation, tokenizer access, and residual-hook surfaces.
- Executable procedures: `procedures/` runs SA3/SAME flow scoring, soft prompt
  optimization, SA3 polish, selective SA3, cyclic SA3, residual extraction, and
  residual sweeps.
- Evidence loop: `evidence/audio_player.py`, `evidence/annotations.py`,
  `evidence/disagreement.py`, `audio_descriptors.py`, and `control_lanes.py`
  turn outputs into reviewable packets and decisions.

The detailed module map is [Primitive map](primitive-map.md). The object and
capability map is [Capability map](capability-map.md).

## Primitive Function Audit

Status as of 2026-06-06:

- All `latent_audio_primitives/` modules compile and import under the repo
  environment.
- Public functions/classes have docstrings.
- Descriptor-target memory scoring is owned by `index.py`; native tokenizer
  extraction is owned by `tokenizer_vocab.py`. The former tiny support files
  were folded because they did not own separate research concepts.
- A synthetic smoke run covers the NumPy-only research grammar: `LatentItem`,
  summaries, geometry, periodicity, control lanes, audio descriptors, memory,
  curriculum, composition, style profiles/directions, flow-probe manifests,
  prompt semantic rows, and disagreement rows.

Conclusion: the current library shape is coherent. The remaining issue is not
whether these files are needed; it is whether each notebook method earns
promotion through repeated evidence packets.

## Artifact Graph

```text
audio files
  -> SAME encoder
  -> SAME latents
  -> latent memory / style profile / control lanes / DSP edits / geometry ops
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

prompt pairs or labeled audio sets
  -> SA3 residual activation capture
  -> steering vectors / residual feature basis
  -> alpha sweeps
  -> generated outputs + probe reports

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
- Which SAME latent edits survive direct decode and SA3 polish?
- Which control lanes are observable, predictable, and intervenable?
- Which residual layers produce stable causal audio controls?
- Can sampler-level guidance improve loopability or source preservation without
  artifacts?
- Which evidence panels are lightweight enough for routine Colab review?
- Which methods should graduate after the first real ledger entries?
