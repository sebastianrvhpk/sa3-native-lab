# SA3 Native Lab App Overview

SA3 Native Lab is a local research instrument for exploring Stable Audio 3
Medium and SAME-L latents without depending on Colab cell state. The app turns
notebook experiments into typed background jobs, durable artifacts, and a
listening-first interface.

## Current Thesis

The app is a latent listening bench:

```text
prompt or source audio
  -> SA3 Medium / SAME-L runtime
  -> latent or script experiment
  -> audio, latent, or bundle artifact
  -> recipe, lineage, metrics, logs, and replayable parameters
```

The target model family is SA3 Medium. Defaults use:

- `medium` for SA3 generation and SA3 script experiments.
- `same-l` for SAME encode/decode and audio-memory/style-vector work.
- `medium-base` for LoRA training because LoRA starts from a base checkpoint.

Small checkpoints can still be selected for deliberate smoke tests, but they
are no longer the default exploration path.

## Main Surfaces

### Python API

The API in `sa3_native_lab/app/` is the source of truth for local execution.
It exposes health/status, artifact storage, recipes, jobs, generation,
SAME encode/decode, latent operators, and Colab-style script adapters.

Important endpoints:

- `GET /health`
- `GET /operators/specs`
- `GET /colab/modes`
- `POST /audio/import`
- `POST /generate/text`
- `POST /latents/encode`
- `POST /latents/decode`
- `POST /operators/run`
- `POST /experiments/run`

### Listening Bench

The React app in `frontend/` is the working interface. It supports artifact
selection, waveform inspection, audio playback, A/B comparison, MLX generation,
SAME encode/decode, latent-operator runs, Recipe Studio, Mode Atlas, and job
polling. The bench also has a Run Monitor that surfaces active jobs, percent
progress, elapsed time, and the latest backend message near the controls that
started the work.

### Operator Studio

Operator Studio exposes direct latent operators as native controls:

- cyclic roll
- latent blur
- latent DSP
- latent graft
- latent renoise

Every visible control maps to an executable request parameter. Donor-latent
selection appears only for graft or DSP modes that need a donor.

### Recipe Studio

Recipe Studio wraps the notebook/script experiments as background recipes. It
covers style vectors, style profiles, residual vectors, alpha sweeps, soft
prompts, dataset pre-encoding, and LoRA training.

## Artifact Model

Artifacts are stored under `.sa3_lab/` by default. The app currently supports:

- `audio`: WAV files with waveform peaks and playback.
- `latent`: `.npy` time-major latent arrays with shape/rate metadata.
- `bundle`: zipped script outputs such as vector folders, profiles, soft
  prompts, pre-encoded datasets, and training outputs.

Every run records a `Recipe` and `JobRecord` so results can be traced back to
operator, backend, inputs, params, model, seed, logs, and source artifacts.

## Runtime Assumptions

- The local Mac path uses MLX for practical Medium generation.
- The PyTorch path uses `torch_mps` where possible and falls back where code
  supports CPU.
- Hugging Face auth is needed for gated Stability AI weights.
- Long jobs such as LoRA training are currently submitted as background jobs
  but do not yet have pause/cancel controls.

## Current Status

Confirmed in the current codebase:

- Backend contracts and runtime defaults target SA3 Medium/SAME-L.
- `/colab/modes` keeps the notebook migration visible.
- Direct latent operators run as typed jobs over stored latent artifacts.
- Script-backed Colab experiments are reachable from Recipe Studio.
- The frontend builds and the Python test suite passes locally.

Still partial:

- Some Colab modes are mapped but not yet first-class native interactions.
- Bundle artifacts need richer browsing instead of zip-only treatment.
- Memory/query workflows exist as concepts and scripts, but not yet as a
  polished instrument surface.
- Multi-output sweeps need a dedicated result-family view.
