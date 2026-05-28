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
- `GET /readiness`
- `GET /operators/specs`
- `GET /colab/modes`
- `GET /artifacts/{artifact_id}/inspect`
- `POST /audio/import`
- `POST /generate/text`
- `POST /latents/encode`
- `POST /latents/decode`
- `POST /operators/run`
- `POST /experiments/run`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/retry`
- `POST /recipes/{recipe_id}/replay`
- `POST /recipes/{recipe_id}/fork`

### Listening Bench

The React app in `frontend/` is the working interface. It supports artifact
selection, waveform inspection, audio playback, A/B comparison, MLX generation,
SAME encode/decode, latent-operator runs, Recipe Studio, Mode Atlas, and job
polling. The bench also has a Run Monitor that surfaces active jobs, percent
progress, live event snapshots, heartbeat diagnostics, elapsed time,
cancellation, retry, and the latest backend message near the controls that
started the work. Selected artifacts can be labeled, tagged, annotated,
replayed from their recipe, and searched later from the archive. Result families
can be inspected as a compact branch surface with source references,
per-artifact playback, A/B assignment, alpha-sweep promotion controls, job
progress, replay, and fork actions. Forked recipes show changed fields and
per-parameter reset controls before submit.

Read-heavy workbench state can now be loaded through the TypeScript tRPC
control plane. This is enabled by setting `VITE_SA3_CONTROL_PLANE_URL` or by
running `uv run sa3-lab dev --with-control-plane`. The control plane currently
shapes sessions, artifacts, jobs, result families, health, readiness, operator
specs, and mode atlas data. It also exposes job lifecycle, recipe replay/fork,
artifact inspection, family loading, archive procedures, and a tRPC/SSE job
event bridge while the Python worker keeps owning model execution and artifact
file IO. The event bridge uses monotonic IDs, resume-aware sequencing, heartbeat
events, and log-tail diagnostics, but it is still backed by Python job polling
rather than a durable event store.

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
prompts, dataset pre-encoding, local latent-memory query, and LoRA training.

## Artifact Model

Artifacts are stored under `.sa3_lab/` by default. The app currently supports:

- `audio`: WAV files with waveform peaks and playback.
- `latent`: `.npy` time-major latent arrays with shape/rate metadata.
- `bundle`: zipped script outputs such as vector folders, profiles, soft
  prompts, pre-encoded datasets, and training outputs.

Every run records a `Recipe` and `JobRecord` so results can be traced back to
operator, backend, inputs, params, model, seed, logs, and source artifacts.
Artifacts can also carry user labels, notes, and tags for archive search.
Bundle artifacts can be inspected through the API and UI to reveal their file
inventory, backend-parsed JSON/NPZ summaries, parsed preview metadata, recipe,
source artifacts, and child artifacts. Jobs and artifacts with the same recipe
are grouped as result families in the right rail with run metrics when the job
reports them. Memory query bundle previews expose ranked hits that can be
selected, placed in A/B when audio, or reused as latent donors when the hit is a
latent artifact.

## Runtime Assumptions

- The local Mac path uses MLX for practical Medium generation.
- The PyTorch path uses `torch_mps` where possible and falls back where code
  supports CPU.
- Hugging Face auth is needed for gated Stability AI weights.
- `/readiness` reports artifact-root writability, backend availability, HF auth,
  Medium MLX weight presence, and SAME-L download readiness.
- Long jobs such as LoRA training are submitted as background jobs and can be
  cancelled from the app, although true resumable multi-step workflows are still
  future work.

## Current Status

Confirmed in the current codebase:

- Backend contracts and runtime defaults target SA3 Medium/SAME-L.
- `/colab/modes` keeps the notebook migration visible.
- Direct latent operators run as typed jobs over stored latent artifacts.
- Script-backed Colab experiments are reachable from Recipe Studio.
- Local latent-memory query is reachable as a CPU recipe over stored latent
  artifacts.
- Artifact annotation and archive search are implemented for labels, notes, and
  tags.
- tRPC workbench, readiness, job lifecycle, recipe replay/fork, artifact
  inspection, result-family procedures, and job-event subscriptions are
  implemented behind the control-plane launch flag.
- The frontend has live job-event snapshots, a readiness panel, a recipe fork
  editor with diffs and resets, result-family detail playback, memory-result
  reuse actions, alpha-sweep variant promotion, backend-parsed typed bundle
  inspectors, and bundle previews.
- Core app surfaces are now split into focused modules for audio playback,
  artifact display, job progress, result families, recipe forks, and bundle
  inspection.
- The frontend builds and the Python test suite passes locally.

Still partial:

- Some Colab modes are mapped but not yet first-class native interactions.
- Type-specific readers for profiles, vectors, soft prompts, training outputs,
  sweeps, and memory collections now receive backend-parsed summaries, but
  still need richer plots and reuse actions.
- Memory-query bundles expose preview rows and donor/A-B reuse actions, but
  still need richer dataset browsing, preview audio for non-local children, and
  style-reference promotion.
- Multi-output sweeps have family grouping, metrics, direct playback, explicit
  A/B promotion controls, and recipe fork deltas, but still need sweep-specific
  metric tables and sibling recipe comparison.
- Live job events now reach React through the control plane when that path is
  enabled; the bridge currently polls Python job snapshots and can later switch
  its internal source to Python WebSocket or a durable event store without
  changing the UI contract.
