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
- `GET /artifacts/{artifact_id}/bundle-file?path=...`
- `POST /artifacts/{artifact_id}/bundle-audio/promote`
- `POST /audio/import`
- `POST /generate/text`
- `POST /latents/encode`
- `POST /latents/decode`
- `POST /operators/run`
- `POST /experiments/run`
- `POST /jobs/{job_id}/cancel`
- `POST /jobs/{job_id}/retry`
- `GET /jobs/{job_id}/events/history`
- `POST /recipes/{recipe_id}/replay`
- `POST /recipes/{recipe_id}/fork`

### Listening Bench

The React app in `frontend/` is the working interface. It supports artifact
selection, waveform inspection, audio playback, region looping, playback-rate
checks, A/B comparison, MLX generation, SAME encode/decode, latent-operator
runs, Recipe Studio, Mode Atlas, and job polling. The bench also has a Run
Monitor that surfaces active jobs, percent progress, derived phase labels,
artifact counts, live event snapshots, heartbeat diagnostics, elapsed time,
cancellation, retry, the latest backend message, and recovery hints for common
failures such as gated Hugging Face access, missing MLX setup, path mistakes,
subprocess exits, and memory pressure. Selected artifacts can be labeled,
tagged, annotated, marked keeper/maybe/rejected from listening surfaces,
replayed from their recipe, inspected by kind, and searched later from the
archive. A session can be archived into the background while a
fresh session starts cleanly. Result families can be inspected as a compact
branch surface with source references, per-artifact playback, A/B assignment,
sortable alpha-sweep promotion controls, job progress, replay, and fork
actions. Forked recipes show changed fields and per-parameter reset controls
before submit.

Read-heavy workbench state can now be loaded through the TypeScript tRPC
control plane. This is enabled by setting `VITE_SA3_CONTROL_PLANE_URL` or by
running `uv run sa3-lab dev --with-control-plane`. The control plane currently
shapes sessions, artifacts, jobs, result families, health, readiness, operator
specs, and mode atlas data. It also exposes job lifecycle, recipe replay/fork,
artifact inspection, family loading, archive procedures, and a tRPC/SSE job
event bridge while the Python worker keeps owning model execution and artifact
file IO. Python now persists a per-job JSONL event journal, and the control
plane replays missed journal snapshots before resuming live polling. The event
bridge still uses polling as its live source, but refresh/reconnect no longer
depends only on transient in-memory state.

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
covers style vectors, style profiles, residual vectors, alpha sweeps, prompt
search, soft prompts, dataset pre-encoding, local latent-memory query, local
SAME geometry audit, and LoRA training.

Operator and recipe controls are partly hand-shaped for playability and partly
derived from backend `ui_fields` emitted by `/operators/specs`. The frontend
keeps the current instrument layout, then merges backend defaults, bounds,
options, artifact-kind hints, and newly discovered fields into the form model.
Select controls, integer-like numeric controls, and alpha-list fields now get
first-pass schema validation from that model. This keeps parameters such as
duration, seed, model, alpha lists, bundle paths, and backend choices
accessible while reducing drift from Python contracts.

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
inventory, embedded audio children, backend-parsed JSON/NPZ summaries, parsed
preview metadata, recipe, source artifacts, and child artifacts. Sweep and
script bundles now promote `metrics.json` values, plot/image files, prompt-search
JSON, geometry reports, and embedded WAV/FLAC/etc children into the reader
summary; image plots render inline and bundle-contained audio can be played
through the bundle-file endpoint instead of staying buried in zip contents.
Embedded bundle audio can also be promoted into a normal audio artifact, which
gives it peaks, playback, lineage, recipe provenance, annotations, and reuse in
the rest of the app. Reusable bundle types expose native Recipe Studio actions
such as use as profile, sweep vectors, use direction, use prompt in sweep, use
soft prompt, use memory, and use checkpoint. Prompt-search bundles expose
candidate prompts as a small listening bench: a candidate can be used as the
main generation prompt, sent to an alpha sweep, or rendered as MLX audio with
lineage back to the bundle. Generated candidate takes appear inline beside the
prompt, can be played immediately, assigned to A/B comparison slots, and marked
keeper/maybe/rejected with optional listening notes.
Jobs and artifacts with the same recipe are grouped as result families in the
right rail with run metrics when the job reports them; prompt-candidate
generations are grouped under their search bundle instead of scattering as
unrelated text-to-audio rows. Memory query bundle previews expose ranked hits
that can be selected, placed in A/B when audio, or reused as latent donors when
the hit is a latent artifact. Alpha sweep families can also compare sibling
sweep runs that share a vector bundle or prompt.

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
- Local SAME geometry audit is reachable as a CPU recipe over stored latent
  artifacts, producing a report bundle with variance and summary metrics.
- Local prompt search is reachable as a recipe over a selected or explicit
  target audio artifact. It supports `lexical_probe` for cheap deterministic
  wiring and `sa3_flow_probe` for an optional Medium-backed flow-loss objective.
  Prompt-search candidates can be rendered as generated audio takes with
  lineage, inline playback, A/B assignment, and saved listening decisions. CLAP
  remains queued behind the same explicit scorer field.
- Artifact annotation and archive search are implemented for labels, notes,
  tags, and durable keeper/maybe/reject listening decisions.
- tRPC workbench, readiness, job lifecycle, recipe replay/fork, artifact
  inspection, result-family procedures, and job-event subscriptions are
  implemented behind the control-plane launch flag.
- The frontend has durable job-event replay through the control plane, a
  readiness panel, a recipe fork
  editor with diffs and resets, result-family detail playback, memory-result
  reuse actions, alpha-sweep variant promotion with a compact metric table,
  metric sorting, best-candidate marking, bundle-to-recipe reuse actions, job
  phase labels, job recovery hints,
  backend-derived operator field metadata, backend-parsed typed bundle
  inspectors, bundle metrics, inline plot/image previews, and kind-specific
  artifact vitals, embedded bundle-audio playback and promotion, prompt-search
  scorer controls, candidate-family bundle reading, durable listening decision
  controls, sibling sweep comparison, and the first native geometry-audit recipe.
- Core app surfaces are now split into focused modules for audio playback,
  artifact display, job progress, result families, recipe forks, and bundle
  inspection.
- The frontend builds and the Python test suite passes locally.

Still partial:

- Some Colab modes are mapped but not yet first-class native interactions.
  Prompt-search modes now have a native recipe path and an SA3 flow-loss scorer,
  but richer descriptor deltas and comparison reports are still needed before
  calling Mode 2/3/5 parity complete.
- Type-specific readers for profiles, vectors, soft prompts, training outputs,
  sweeps, memory collections, and geometry audits now receive backend-parsed
  summaries, first-pass metrics/plot discovery, embedded image/audio rendering,
  and recipe-input actions, but still need richer domain-specific controls for
  each bundle type.
- Memory-query bundles expose preview rows and donor/A-B reuse actions, but
  still need richer dataset browsing, preview audio for non-local children, and
  style-reference promotion.
- Multi-output sweeps have family grouping, metrics, direct playback, explicit
  A/B promotion controls, recipe fork deltas, inspected metric summaries, best
  candidate marking, saved listening decisions, sort controls, a compact
  alpha/metric table, and sibling recipe comparison across separate sweep runs.
- Live job events now reach React through the control plane when that path is
  enabled; the bridge replays Python's durable job journal and can later switch
  its live source to Python WebSocket without changing the UI contract.
