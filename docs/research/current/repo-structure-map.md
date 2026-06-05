# SA3 Native Lab Repo Structure Map

Status: current as of 2026-06-05. This map reflects the notebook-first research
direction.

## Direction

SA3 Native Lab is a Colab and Python research workspace for frozen SA3/SAME
latent experiments. The source of truth is the expanded Colab notebook:

```text
colab/sa3_same_native_experimental_modes.ipynb
```

This repo grows notebook cells, helper primitives, and research notes.

LoRA work uses [dada-bots/underfit](https://github.com/dada-bots/underfit).
Notebook cells can compare exported Underfit audio, checkpoints, and run notes
alongside native SA3/SAME methods.

## Top-Level Map

| Path | Role | Status |
|---|---|---|
| `colab/` | Source notebook and Colab-facing runbook. | Active |
| `latent_audio_primitives/` | Reusable SAME/SA3 latent math, prompt scoring, controls, DSP, geometry, observability, and experiment helpers. | Active |
| `docs/research/current/` | Current notebook map, mode math, and repo structure. | Active |
| `docs/research/methods/` | Method notes that support active or near-term notebook modes. | Active reference |

## Execution Surfaces

| Surface | Entry Point | Inputs | Outputs | Notes |
|---|---|---|---|---|
| Colab notebook | `colab/sa3_same_native_experimental_modes.ipynb` | Audio files, prompts, SA3/SAME checkpoints, datasets | Audio files, tables, manifests, plots, player cells | Main research instrument |
| Upstream SA3 runtime | external `Stability-AI/stable-audio-3` checkout | Prompts/audio/checkpoints | generated audio and SAME latents | Installed separately by notebook setup |

## Python Surface

| Area | Role |
|---|---|
| `latent_audio_primitives/` | Notebook library and SA3/SAME research primitives. |
| `latent_audio_primitives/adapters/` | Thin adapters around upstream SA3/SAME and audioscope-style hooks. |
| `latent_audio_primitives/experiments/` | Small reusable experiment records and helpers for notebook cells. |

`stable_audio_3` is intentionally external. The notebook setup installs
`Stability-AI/stable-audio-3` before this repo when model weights are needed.

## Capability Map

| Capability | Primary Files | Evidence |
|---|---|---|
| SA3/SAME native flow prompt scoring | `latent_audio_primitives/flow_prompt.py`, Mode 2/16/17 notebook cells | Confirmed |
| Soft prompt optimization | `latent_audio_primitives/prompt_optimization.py`, notebook Modes 1/4 | Confirmed |
| SAME latent memory/indexing | `latent_audio_primitives/index.py`, `schema.py`, `io.py` | Confirmed |
| Latent statistics and style controls | `controls.py`, `style.py`, `geometry.py` | Confirmed |
| Selective renoise and blur/filter edits | `selective_renoise.py`, `latent_blur.py` | Confirmed |
| Neural latent DSP | `latent_dsp.py`, `docs/research/methods/neural-latent-dsp.md` | Confirmed |
| Looping and periodic operators | `looping.py`, `periodic.py` | Confirmed |
| Guidance and posterior-style edit scaffolds | `guidance.py`, notebook Modes 24/25 | Confirmed scaffold |
| Residual steering and feature discovery | `adapters/audioscope_sa3.py`, `residual_features.py` | Confirmed |
| Control lanes and curricula | `control_lanes.py`, `curriculum.py` | Confirmed |
| Audio descriptors and audits | `audio_descriptors.py`, notebook tables | Confirmed |
| Cross-model comparison harness | notebook Mode 26 | Confirmed scaffold |
| LoRA fine-tuning | Underfit training plus exported comparison artifacts | External tool |

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
```

## Markdown Inventory

### Current Project Docs

| File | Role |
|---|---|
| `README.md` | Project entrypoint and current direction. |
| `docs/research/README.md` | Research doc index and documentation rules. |
| `docs/research/current/repo-structure-map.md` | This repo map and Markdown inventory. |
| `docs/research/current/notebook-research-map-and-next-methods.md` | Current capability map, research synthesis, backlog status, and next methods. |
| `docs/research/current/native-experimental-modes-math.md` | Current notebook math and mode implementation notes. |
| `docs/research/methods/native-operators-and-measurement.md` | Active method reference for measurable SA3/SAME operators. |
| `docs/research/methods/neural-latent-dsp.md` | Active method reference for SAME latent DSP. |
| `colab/sa3_medium_l4_runbook.md` | Practical Colab L4 runtime setup and failure-mode notes. |

## Documentation Ownership

- Current notebook implementation claims belong in `docs/research/current/`.
- Reusable method notes belong in `docs/research/methods/`.
- Add broad literature material when it is distilled into current notebook
  decisions.
- Stable Audio 3 upstream/reference docs live in the external upstream repo.
- Notebook behavior should be changed directly in
  `colab/sa3_same_native_experimental_modes.ipynb`, with reusable code moved
  into `latent_audio_primitives/` when it becomes shared across cells.

## Unknowns And Verification Plan

- Full SA3/SAME audio quality claims require loaded model weights, listening
  checks, descriptor deltas, and manifest notes.
- Notebook cells that depend on GPU Colab runtimes should be smoke-tested on
  Colab L4.
- Underfit LoRA runs can be validated in Underfit and imported as
  audio/checkpoint comparison artifacts.
