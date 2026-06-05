# SA3 Native Lab Research State

Status: current project snapshot for the notebook-first SA3 Native Lab direction
as of 2026-06-05.

This document answers: what exists now, what the repo is for, which notebook
modes and helper primitives are active, and which parts are confirmed versus
scaffolded.

## Direction

SA3 Native Lab is a Colab and Python research workspace for frozen SA3/SAME
latent experiments. The source of truth is:

```text
colab/sa3_same_native_experimental_modes.ipynb
```

The repo keeps:

- the expanded Colab notebook,
- notebook-facing SA3/SAME helper primitives,
- current research notes,
- a Colab L4 setup runbook.

It intentionally does not keep an app, script harness, test harness, vendored
Stable Audio 3 tree, or local product interface. Stable Audio 3 is installed
from the external upstream repository when model weights are needed.

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
| `docs/research/methods/` | Superseded once this reorganization is complete. Content is migrated into `methods-and-math.md`. | Transitional |

## Execution Surfaces

| Surface | Entry Point | Inputs | Outputs | Notes |
|---|---|---|---|---|
| Colab notebook | `colab/sa3_same_native_experimental_modes.ipynb` | Audio files, prompts, datasets, SA3/SAME checkpoints | Audio, tables, plots, manifests, player rows, annotations | Main research instrument |
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

## Notebook Mode Inventory

| Mode | Capability | Main Object | Main Artifact | Status |
|---|---|---|---|---|
| 0 | Renoise variations from existing loop/audio | SA3 init/audio latent | generated audio, manifest rows | implemented |
| 0c | Latent-selective renoise playground | SAME channel/time masks | variants, annotation search | implemented |
| 0e | Cross-audio latent channel graft | source/donor SAME channels | grafted variants | implemented |
| 0d | Latent blur playground | SAME latent trajectories | direct or polished audio | implemented |
| 0h | Neural latent DSP playground | latent gain, dynamics, FFT, PCA | audio, descriptor reports | implemented |
| 0f | Cyclic time-roll loop lab | waveform/latent boundary repair | loop previews | implemented |
| 0g | Cyclic roll inside each denoising step | sampler trajectory | loop variants | implemented |
| sign diagnostic | Flow sign diagnostic | SA3 velocity convention | diagnostic metrics | implemented |
| 1 | Audio to soft prompt | continuous conditioning | saved soft prompt | implemented |
| 1b | Generate audio from saved soft prompt | saved conditioning state | generated audio | implemented |
| 2 | Audio to babble/hard prompt | hard prompt tokens scored by SA3 flow | prompt candidates | implemented |
| 3 | Audio to readable prompt | constrained descriptor prompt axes | readable prompt | implemented |
| 4 | Dataset to soft prompt | shared conditioning vector | dataset soft prompt | implemented |
| 5 | Dataset to prompt family by SAME clustering | clusters plus text search | prompt families | implemented |
| 6 | SAME latent style profile | mean/std latent profile | profile JSON | implemented |
| 7 | SAME latent direction from positive/reference folders | dataset-level latent direction | direction JSON | implemented |
| 8 | SA3 residual steering from prompt pairs | residual activations | steering vectors, alpha sweep | implemented |
| 9 | SA3 residual steering from audio files | audio-derived residual directions | vector file, probe report | implemented |
| 10 | Flow-state optimization scaffold | intermediate flow state | optimized state | scaffold |
| 11 | Continuation/inpainting composition | mask and continuation region | extension/edit audio | implemented |
| 12 | Minimal LatCH-style control head | sidecar predictor over SAME summaries | sidecar `.pt` | implemented as probe |
| 13 | External LoRA handoff | Underfit workflow | external checkpoints/audio | external |
| 14 | Latent memory instrument | indexed `LatentItem`s | memory index | implemented |
| 15 | SAME geometry and intervention audit | latent collection | geometry report | implemented |
| 16 | Flow attribution prompt microscope | prompt tokens over shared flow probes | attribution rows | implemented |
| 17 | Loss-by-timestep flow panel | logSNR/timestep score rows | per-timestep diagnostics | implemented |
| 18 | SAME control lanes | time-varying control rows | lane JSON/SVG/table | implemented |
| 19 | Dataset memory as prompt/control curriculum | memory clusters and heldout rows | curriculum table | implemented |
| 20 | Latent OT style transfer bench | geometry/style profiles | transport comparison | implemented |
| 21 | Continuation as bridge search | transition/bridge candidates | ranked bridge table | implemented |
| 22 | Residual feature atlas | residual feature basis | layer/feature report | implemented |
| 23 | SA3 null-condition inversion probe | CFG/null conditioning path | inversion probe rows | scaffold |
| 24 | Guidance-gradient latent edit probe | differentiable latent losses | edited/polished variants | scaffold |
| 25 | Audio-to-audio posterior guidance scaffold | source/reference preservation losses | guided variants | scaffold |
| 26 | Cross-model baseline harness | fixed prompts plus external commands | comparison report | scaffold |
| combined | Combined chain scaffold | multi-method recipe | chain outputs | scaffold |
| manifest | Experiment manifest/log template | run metadata | manifest JSON | implemented |

## Reusable Local Modules

`latent_audio_primitives`, current role:

- `flow_prompt.py`: frozen SA3 flow prompt loss, logSNR timesteps, explicit velocity conventions, prompt attribution, per-timestep loss rows.
- `prompt_optimization.py`: coordinate, greedy-token, and beam hard-prompt search.
- `latent_dsp.py`: latent gain, dynamics, soft clipping, FFT EQ, phase, donor magnitude/phase, PCA gain.
- `latent_blur.py`: temporal/channel blur, low-rank projection, unsharp masking, FFT filters, SA3 polish helpers.
- `selective_renoise.py`: channel selection, masks, masked noise, grafting, sampler helpers.
- `looping.py`: cyclic roll, cyclic mix, repeated previews, loop metrics, sampler-level cyclic projection.
- `geometry.py`: PCA geometry, whitening, Mahalanobis distance, covariance transport, barycenters.
- `periodic.py`: autocorrelation, best period lag, FFT energy, spectral centroid, boundary loss.
- `style.py`: style profiles and directions.
- `residual_features.py`: residual activation basis and directions.
- `observability.py`: linear control probes and intervention effect.
- `audio_descriptors.py`: lightweight audio descriptor reports.
- `control_lanes.py`: time-varying lane extraction, normalization, comparison, SVG rendering, save/load.
- `curriculum.py`: memory clustering, heldout splits, nearest-memory rows.
- `composition.py`: transition, loop, bridge, and path ranking.
- `index.py`, `schema.py`, `io.py`: latent memory item, search, persistence.
- `colab_audio_player.py`: waveform player, annotation save/search, notebook listening bench.
- `adapters/`: thin wrappers around external SA3/SAME and residual-hook surfaces.
- `experiments/`: small notebook-facing experiment records for soft prompts, activation vectors, residual vectors, prompt pairs, and sweeps.

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
- This repo no longer keeps a separate script/test harness. Validation is notebook execution, JSON integrity, Colab smoke runs, descriptor deltas, manifests, and listening notes.
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

