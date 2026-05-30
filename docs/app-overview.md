# SA3 Native Lab App Overview

SA3 Native Lab is a local AI sound instrument for exploring Stable Audio 3
Medium and SAME-L latents without depending on Colab cell state. The app should
turn model inference into playable gestures over sounds, takes, branches, and
memory. The product rescue direction lives in `docs/product-rescue-brief.md`.

## Current Thesis

The app target is an expressive sound surface:

```text
sound, prompt, latent, or memory
  -> SA3 Medium / SAME-L runtime
  -> creative gesture
  -> take or branch
  -> playback, tuning, memory, lineage, and reuse
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

### Current Sound And Gestures

The React app in `frontend/` is the working interface. Today it supports
artifact selection, waveform inspection, audio playback, WaveSurfer-backed
waveform zoom, draggable loop regions, playback-rate checks, MLX generation,
SAME encode/decode, latent-operator runs, Advanced Gestures, Mode Atlas, and job
polling. The first product rescue implementation now reframes those paths
through Current Sound plus a gesture strip: Make, Continue, Vary, Steer,
Borrow Texture, Encode, Decode, Morph, and Remember. Selecting a gesture opens
one scoped Tune surface backed by existing forms and payload builders, while
contract/spec and mode-atlas details sit behind Inspect. Generate and SAME
controls are
now native schema-driven forms: the backend `/operators/specs` contract supplies
defaults, bounds, select options, required flags, and advanced flags, while the
frontend form model builds typed generation and encode/decode payloads. This
keeps everyday parameters such as duration, seed, model, decoder, init noise,
inpaint range, SAME chunking, prompts, and notes visible without maintaining a
separate hand-written payload surface. An Inspect activity drawer keeps active
jobs, percent progress, persisted phase labels, output counts, live event
snapshots, heartbeat diagnostics, elapsed time, cancellation, retry,
the latest backend message, and recovery hints for common failures such as gated
Hugging Face access, missing MLX setup, path mistakes, subprocess exits, and
memory pressure. Successful jobs automatically land the workbench on their
newest produced artifact when the runtime reports artifact IDs. Selected
artifacts can be labeled, tagged, annotated, marked keeper/maybe/rejected from
listening surfaces, replayed from their recipe, inspected by kind, and recovered
later with filters for decision, kind, model, gesture, branch, tag,
text, and source lineage. A session can be archived into the background while a
fresh session starts cleanly. Branches, backed by result-family records, can
currently be inspected as compact creative paths with source references, per-take playback, sortable
alpha-sweep controls, job progress, replay, and fork actions. Forked recipes
show changed fields and per-parameter reset controls
before submit. The session tray also shows a data-backed workspace pulse for
active takes, branches, job activity, listening decisions, and archive
volume. Its focus hint points to real next actions such as monitoring a run,
opening the next undecided take, recovering an archived artifact, or archiving a
crowded session. Playback markers and loop regions persist as artifact
annotations under `metadata.playback_state`, so listening cues survive reloads,
replay/fork work, and session/archive movement. Archived artifacts can now be
recovered into the active session through the archive drawer; active artifacts
can also be archived directly from the specimen, session tray, and result-family
surfaces. Both flows use the annotation contract to move `session_id` and record
source/target session metadata.

The current product loop is now explicit:

```text
Current Sound -> Gesture -> Pending Take -> Listen -> Branch / Remember / Tune
```

Remembered material is active, not just archived. Audio can be reused as Source
or Anchor, latent material can become a Borrow Texture donor, remembered
prompt/label/notes can seed Make, reusable bundles can feed existing Advanced
Gesture paths, and archived materials can be recovered into the active session.
Selected landed takes also show a `Next` affordance: audio suggests Continue,
Vary, Encode, and Remember; latent suggests Decode, Morph, Borrow Texture, and
Remember; bundles keep technical details behind Inspect while exposing real
reuse actions where a bundle path already exists.

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

### Latent Gestures

Latent Gestures expose direct latent operators as native controls:

- Roll
- Blur
- DSP/Reroute
- Borrow Texture
- Renoise

Every visible control maps to an executable request parameter. Donor-latent
selection appears only for Borrow Texture or DSP/Reroute modes that need a
donor.
Latent Gestures also have a first-pass local preset rack: named browser-local
parameter sets can be saved, reloaded, updated, or deleted per operator mode so
repeatable latent explorations do not require rebuilding a form by hand. When a
preset is selected, the UI now shows parameter and donor-latent drift from the
saved setting and can revert the current controls back to that preset.

### Advanced Gestures

Advanced Gestures wrap the notebook/script experiments as background recipes. It
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
operator, backend, inputs, params, model, seed, logs, phase, and source
artifacts. The specimen lineage thread is data-backed: it can show real source
artifacts, the recipe/job that produced the current artifact, and the result
family it belongs to. Product UI should translate this to source -> gesture ->
take -> branch/memory.
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
the rest of the app. Bundle readers now show compact workflow signals for real
available affordances: recipe actions, playable audio, lineage, plots, metrics,
prompt candidates, memory hits, sweep variants, tensors, checkpoints, and
geometry stats. Reusable bundle types expose native Advanced Gesture actions such
as use as profile, sweep vectors, use direction, use prompt in sweep, use soft
prompt, use memory, and use checkpoint. Prompt-search bundles expose candidate
prompts as a small listening bench: a candidate can be used as the main
generation prompt, sent to an alpha sweep, or rendered as MLX audio with lineage
back to the bundle. Generated candidate takes appear inline beside the prompt,
can be played immediately and marked keeper/maybe/rejected with optional
listening notes. They also show compact
descriptor deltas against the prompt-search target audio for level, brightness,
spectral motion, noise/flatness, and stereo width. A small decision-study panel
summarizes listened/generated takes by keeper/maybe/reject state and averages
the descriptor shifts for keepers so prompt-search listening notes become
research feedback rather than isolated labels. Prompt-search panels also show
first-pass prompt memory across generated takes, grouping keeper/maybe/reject
decisions by prompt text across runs.
Bundle domain cards now promote parsed sweep outputs, memory hits, vector NPZ
contents, soft-prompt tensors, and training checkpoints into native rows and
item lists. They are still inspectors rather than full editors, but they make
script-backed modes legible without pretending unavailable routing or graph
behavior exists.
Jobs and artifacts with the same recipe are grouped as branch records in the
right rail, backed by the legacy result-family contract; raw run metrics remain
inspect details when the job reports them. Prompt-candidate generations are grouped
under their search bundle instead of scattering as unrelated text-to-audio rows.
Memory query bundle previews expose ranked hits that can be selected, played
when audio, or reused as latent donors when the hit is a latent artifact.
Encoded dataset bundles expose manifest/sidecar
counts, chunk timing, prompt coverage, latent files, and reuse into LoRA
training. Alpha sweep families can also compare sibling sweep runs that share a
vector bundle or prompt.

## Runtime Assumptions

- The local Mac path uses MLX for practical Medium generation.
- The PyTorch path uses `torch_mps` where possible and falls back where code
  supports CPU.
- Hugging Face auth is needed for gated Stability AI weights.
- `/readiness` reports artifact-root writability, backend availability, HF auth,
  Medium MLX weight presence, and SAME-L download readiness.
- `uv run sa3-lab smoke-fixture --json` provides a cheap local runtime smoke:
  it creates a deterministic latent fixture, submits `latent.cyclic_roll`
  through the same job/runtime/storage path as the app, and verifies a persisted
  output artifact without gated model downloads.
- `npm run smoke:playback-session --prefix frontend` is the committed browser
  smoke for the listening/session loop: Playwright writes marker notes and loop
  cues, verifies lineage-backed wave-bus state, checks archive/recovery, and
  captures desktop/mobile screenshots.
- `npm run smoke:first-use --prefix frontend` is the committed product-health
  smoke for the first-use instrument loop: Current Sound, Gestures, Make, Tune,
  pending/failed take language, Next actions, Remember, Branch, Memory reuse as
  Source/Anchor, recovery, Settings/Inspect demotion, screenshots, and mobile
  overflow.
- `uv run sa3-lab smoke-mlx-medium --json` verifies the slow Medium/MLX path is
  gated by default; `SA3_RUN_MLX_SMOKE=1 uv run sa3-lab smoke-mlx-medium --run
  --duration 1 --steps 2 --json` submits the authenticated real model smoke.
- Long jobs such as LoRA training are submitted as background jobs and can be
  cancelled from the app, although true resumable multi-step workflows are still
  future work.

## Current Status

Confirmed in the current codebase:

- Backend contracts and runtime defaults target SA3 Medium/SAME-L.
- `/colab/modes` keeps the notebook migration visible.
- Direct latent operators run as typed jobs over stored latent artifacts.
- Script-backed Colab experiments are reachable from Advanced Gestures.
- Local latent-memory query is reachable as a CPU recipe over stored latent
  artifacts.
- Local SAME geometry audit is reachable as a CPU recipe over stored latent
  artifacts, producing a report bundle with variance and summary metrics.
- Local prompt search is reachable as a recipe over a selected or explicit
  target audio artifact. It supports `lexical_probe` for cheap deterministic
  wiring and `sa3_flow_probe` for an optional Medium-backed flow-loss objective.
  Prompt-search candidates can be rendered as generated audio takes with
  lineage, inline playback, target-vs-take descriptor deltas, saved listening
  decisions, and a first-pass decision-study summary that correlates listened
  choices with descriptor shifts. Generated prompt takes are also grouped into a
  prompt-memory summary across runs. The selectable prompt probe field exposes
  only implemented paths (`lexical_probe` and `sa3_flow_probe`).
- Artifact annotation and archive search are implemented for labels, notes,
  tags, durable keeper/maybe/reject listening decisions, artifact kind, model,
  gesture, branch, and source lineage.
- tRPC workbench, readiness, job lifecycle, recipe replay/fork, artifact
  inspection, result-family procedures, and job-event subscriptions are
  implemented behind the control-plane launch flag.
- The frontend has durable job-event replay through the control plane, a
  readiness panel, a recipe fork
  editor with diffs and resets, result-family detail playback, memory-result
  reuse actions, alpha-sweep variant promotion with a compact metric table,
  metric sorting, branch highlighting, bundle-to-recipe reuse actions, job
  phase labels, artifact landing after successful jobs, job recovery hints,
  backend-derived operator field metadata, backend-parsed typed bundle
  inspectors, bundle metrics, inline plot/image previews, and kind-specific
  artifact vitals, embedded bundle-audio playback and promotion, prompt-search
  probe controls, candidate-family bundle reading, durable listening decision
  controls, prompt-search decision summaries, prompt memory, Latent Gestures
  local presets with visible diffs, bundle workflow signals, richer domain
  cards for memory/sweep/vector/soft-prompt/training bundles, sibling sweep
  comparison, data-backed specimen lineage threads, a session workspace pulse,
  archive artifact recovery, keyboardable playback playlist navigation, local
  waveform markers, persisted playback cues, WaveSurfer zoom and draggable loop
  regions, per-marker deletion, loop-edge nudging, data-backed archive actions,
  marker notes, committed browser coverage for SessionTray
  artifact archive/recovery, and the first native geometry-audit recipe.
- Product-domain frontend models now cover memory reuse, next actions,
  pending-take landing, branch summaries, and Tune field grouping. Branch UI is
  product-language first, while raw job IDs, recipe IDs, backend details, logs,
  and material counts are behind Inspect/Settings/details.
- Artifact manifest writes are atomic, reducing transient JSON-read races during
  annotation, remember, and recovery workflows.
- Core app surfaces are now split into focused modules for audio playback,
  artifact display, job progress, branches/result families, recipe forks, and bundle
  inspection.
- The frontend builds and the Python test suite passes locally.

Still partial:

- Some Colab modes are mapped but not yet first-class native interactions.
  Prompt-search modes now have a native recipe path and an SA3 flow-loss prompt probe,
  plus first-pass target/take descriptor deltas and listened-decision
  correlation plus prompt-memory grouping across runs. Mode 2/3/5 parity still
  needs richer layer/alpha comparison reports and clearer runtime-cost notes.
- Operator presets are currently browser-local. Backend or Postgres-backed
  preset persistence and shareable preset history remain future work.
- Type-specific readers for profiles, vectors, soft prompts, training outputs,
  sweeps, memory collections, and geometry audits now receive backend-parsed
  summaries, first-pass metrics/plot discovery, embedded image/audio rendering,
  recipe-input actions, workflow hints, and first-pass domain item cards, but
  still need richer domain-specific controls for each bundle type.
- Memory-query bundles expose preview rows and donor/A-B reuse actions, but
  still need richer dataset browsing, preview audio for non-local children, and
  style-reference promotion.
- Playback is now beyond the basic browser player, with playlist navigation,
  persisted markers, marker deletion, WaveSurfer zoom, draggable loop regions,
  region persistence, marker notes, SessionTray
  archive/recovery browser coverage, and loop-edge nudging. It still needs
  richer playlist sequencing and future region-export workflows.
- Multi-output sweeps have family grouping, metrics, direct playback, recipe
  fork deltas, inspected metric summaries, branch highlighting, saved listening
  decisions, sort controls, a compact alpha/metric table, and sibling recipe
  comparison across separate sweep runs.
- Live job events now reach React through the control plane when that path is
  enabled; the bridge replays Python's durable job journal and can later switch
  its live source to Python WebSocket without changing the UI contract.
