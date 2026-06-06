# SA3 Native Lab Research State

Status: current project snapshot for the notebook-first SA3 Native Lab direction
as of 2026-06-05.

This document answers: what exists now, what the repo is for, which science
strata and helper primitives are active, and which parts are confirmed versus
scaffolded.

## Direction

SA3 Native Lab is a Colab and Python research workspace for frozen SA3/SAME
latent experiments. The source of truth is:

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
- `scaffold`: present as a notebook probe or support surface, but not yet fully validated with real weights/listening.
- `hypothesis`: plausible behavior that still needs measurement.
- `unknown`: needs Colab execution, descriptor evidence, source inspection, or listening notes.

## Active Repo Surfaces

| Path | Role | Status |
|---|---|---|
| `colab/` | Expanded notebook plus Colab L4 runbook. | Active |
| `latent_audio_primitives/` | Notebook library for SAME/SA3 latent math, prompt scoring, controls, DSP, geometry, residual probes, memory, and descriptors. | Active |
| `docs/research/current/` | Current project state, methods/math, source context, experiment ledger, and backlog. | Active |
| `docs/research/current/run-protocol.md` | Operational run spine and evidence rules for turning notebook outputs into decisions. | Active |

## Execution Surfaces

| Surface | Entry Point | Inputs | Outputs | Notes |
|---|---|---|---|---|
| Colab notebook | `colab/sa3_native_science_lab.ipynb` | Audio files, prompts, datasets, SA3/SAME checkpoints | Audio, tables, plots, manifests, player rows, annotations | Main research instrument |
| Upstream SA3 runtime | External `Stability-AI/stable-audio-3` checkout | Prompts, audio, checkpoints | Generated audio, SAME latents, sampler state | Installed separately by notebook setup |
| Underfit comparison path | External Underfit repo/artifacts | Training runs, checkpoints, demos | Audio/checkpoint artifacts for notebook comparison | LoRA path lives outside this repo |

## Core Native Graph

```text
audio waveform x
  -> SAME encoder E
  -> SAME latent z, usually B x C x T
  -> latent operators, prompt objectives, memory summaries, residual probes
  -> SA3 flow field v_theta(z_t, t, C(prompt))
  -> SAME decoder D
  -> decoded audio y
```

The research stance:

```text
freeze SA3
freeze SAME
edit prompts, soft conditioning, SAME latents, sampler states, residual activations,
or train small sidecars/control heads
measure with native flow losses, latent geometry, audio descriptors, listening notes,
and experiment manifests
```

## Notebook Science Ontology Inventory

The active notebook is organized by native object, intervention type, and
evidence role. The first four strata are object/intervention strata; the last
two are evidence and comparison infrastructure.

| Stratum | Kind | Experiments | Main Object | Main Artifact | Status |
|---|---|---|---|---|---|
| `SAME_REPRESENTATION` | native representation stratum | neighborhood renoise; channel-selective renoise; cross-audio graft; blur bottleneck; neural latent DSP; style profile; style direction; geometry audit; latent OT style transfer; cyclic loop repair | SAME latent `z0`, latent channels, latent-time trajectories, source/donor latents, latent statistics/covariance | direct decodes, SA3-polished audio, descriptor reports, profile/direction files, geometry reports, transport comparisons | implemented |
| `SA3_FLOW_CONDITIONING` | flow-conditioning stratum | flow sign diagnostic; soft prompt inversion; soft prompt audition; hard prompt search; readable prompt search; dataset soft prompt; flow attribution; loss-by-timestep panel; null-condition inversion | SA3 conditioning `C(p)`, optimized conditioning tensors, frozen flow probes | `.pt` conditioning states, prompt candidates, flow-loss rows, attribution tables, timestep panels | implemented plus null-inversion scaffold |
| `CAUSAL_STEERING` | intervention stratum | prompt-derived residual steering; audio-derived residual steering; flow-state optimization; cyclic denoising projection; LatCH-style control head; residual feature atlas; guidance-gradient edit; audio posterior guidance | DiT residual activations, sampler state, intermediate flow state, sidecar probes, differentiable guidance objectives | steering vectors, alpha sweeps, feature atlas JSON, guided variants, probe reports | implemented plus sampler-guidance scaffolds |
| `DATASET_MEMORY_COMPOSITION` | collection/composition stratum | dataset prompt family; latent memory instrument; memory curriculum; continuation/inpainting composition; bridge search | `LatentItem` memory rows, dataset clusters, source/donor candidates, continuation paths | memory indices, curriculum rows, ranked bridges, composition outputs | implemented |
| `EVIDENCE_DECISION_PROTOCOL` | lab infrastructure | custom audio player; annotation retrieval; control lanes; combined chain; manifest/log template; experiment ledger handoff | evidence packets, descriptor/lane rows, annotations, manifests, listening notes | embedded player rows, lane JSON/SVG, manifests, promote/revise/drop decisions | implemented |
| `EXTERNAL_COMPARISON` | lab infrastructure | Underfit LoRA handoff; cross-model baseline harness | external fine-tuning artifacts and fixed comparison task packs | external audio/checkpoints/run notes, comparison reports | external/scaffold |

## Reusable Local Modules

`latent_audio_primitives`, current role:

- Model boundary: `adapters/` isolates official SA3/SAME loading, encoding,
  generation, and residual-hook surfaces.
- Native records and persistence: `schema.py`, `io.py`, `latent_math.py`,
  `index.py`, and `controls.py` define what the notebook stores, searches, and
  compares.
- Evidence and observability: `audio_descriptors.py`, `periodic.py`,
  `geometry.py`, `control_lanes.py`, `observability.py`, and
  `residual_features.py` turn audio/latent behavior into rows, metrics, lanes,
  probes, and feature bases.
- SAME representation: `latent_blur.py`, `latent_dsp.py`,
  `selective_renoise.py`, `style.py`, `geometry.py`, `periodic.py`, and
  `looping.py` probe or edit the SAME bottleneck, then measure survival.
- SA3 flow conditioning: `flow_prompt.py`, `procedures/flow_scoring.py`,
  `prompt_optimization.py`, `tokenizer_vocab.py`, and
  `procedures/soft_prompt.py` score prompts and optimized conditioning through
  frozen SA3 dynamics.
- Causal steering: `prompt_pairs.py`,
  `procedures/residual_activation_vectors.py`,
  `procedures/audio_residual_vectors.py`, `procedures/residual_sweeps.py`,
  `procedures/cyclic_sa3.py`, `guidance.py`, and `residual_features.py` test
  whether interventions causally move generated audio.
- Dataset memory and composition: `curriculum.py`, `composition.py`, `index.py`,
  `control_lanes.py`, and `audio_descriptors.py` turn collections into
  selection, continuity, and evidence tools.
- Evidence decision loop: `evidence/audio_player.py`,
  `evidence/annotations.py`, `audio_descriptors.py`, and `control_lanes.py`
  turn outputs into annotations, descriptors, lane rows, and
  promote/revise/drop decisions.

The detailed module map is [Primitive map](primitive-map.md). The bottom-up
object/capability map is [Capability map](capability-map.md).

## Artifact Graph

```text
audio files
  -> SAME encoder
  -> SAME latents
  -> latent memory / style profile / control lanes / DSP edits / geometry ops
  -> SAME decoder or SA3 polish
  -> audio outputs + descriptor tables + manifests + listening notes

prompts
  -> SA3 conditioner
  -> frozen flow field probes
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
- Validation centers on notebook execution, JSON integrity, Colab smoke runs, descriptor deltas, manifests, and listening notes.
- Stable Audio 3 upstream/reference docs live in the external upstream repo.
- Underfit LoRA runs are validated in Underfit, then imported as audio/checkpoint comparison artifacts.

## Research Discipline

- Treat latent metrics as candidate generators, not final truth.
- Separate every control into observability, predictability, and intervenability.
- Establish prompt-only and branch-and-rank baselines before residual steering, sampler guidance, or sidecar training.
- Prefer pairwise human labels for subjective qualities such as tension, section role, prompt adherence, transition quality, loop usability, and musical coherence.
- Log exact checkpoint/config/runtime details for every experiment.
- Drop a method when it cannot move a measured signal, or when it moves a measured signal but listening repeatedly rejects the output.

## Current Unknowns

- Do native flow scores predict generated-audio similarity, or only teacher-forced vector-field agreement?
- Which logSNR bands correspond to style, structure, transient detail, or prompt adherence in SA3?
- Which SAME descriptor/control lanes are observable, predictable, and intervenable?
- Which residual layers carry stable music/audio controls?
- Do sampler-level guidance probes improve audio without off-manifold artifacts?
- Can memory nearest-neighbor checks separate useful source preservation from memorization-like copying?
- Which notebook scaffolds should be promoted after Colab/listening validation?
