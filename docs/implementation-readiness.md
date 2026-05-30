# Implementation Readiness And Engineering Horizon

This document is the current operating map for SA3 Native Lab before the next
implementation loop. It consolidates the app objective, current maturity,
engineering risks, test posture, and the next priority queue.

Product correction: `docs/product-rescue-brief.md` is now the authority for the
interface direction. The previous parity-dashboard gravity is explicitly not
the product target.

## App Objective

SA3 Native Lab is a local AI sound instrument for Stable Audio 3 Medium and
SAME-L. Its purpose is to make local model inference playable: start from a
sound, prompt, latent, or remembered take; perform an expressive gesture; hear a
new take; then branch, remember, tune, or continue.

The app should not become a generic dashboard, Colab parity wall, benchmark
surface, job list, or raw operator catalog. The core act is listening and
iterating:

```text
sound, prompt, latent, or memory
  -> SA3 Medium / SAME-L runtime
  -> gesture
  -> take or branch
  -> playback, memory, tuning, lineage, and reuse
```

Every visible control should map to a backend parameter, a recorded recipe
field, an artifact relationship, or an explicitly documented future capability.
The first visible language should use `sound`, `take`, `gesture`, `source`,
`anchor`, `memory`, and `branch`; engineering nouns should move behind
progressive disclosure.

## Current State

### Confirmed Strengths

- The project has a coherent split between Python runtime, TypeScript app
  control plane, React frontend, research primitives, scripts, notebooks, and
  docs.
- The target model policy is clear: SA3 `medium` for generation and SAME
  `same-l` for native latent workflows.
- The Python API owns execution, artifacts, jobs, recipes, model adapters,
  script adapters, and filesystem-backed persistence.
- The TypeScript tRPC control plane owns app-shaped reads and actions without
  replacing the Python model worker.
- The React app is a real local workbench with playback, waveform markers,
  schema-driven model controls, annotations, sessions, archive recovery, result
  grouping, prompt-search generated takes, and prompt memory.
- Backend `ui_fields` already reduce parameter drift by exposing defaults,
  bounds, options, required flags, advanced flags, and artifact-kind hints to
  the frontend.
- Generate and SAME encode/decode now use the same schema-driven RecipeFields
  control path as the rest of the instrument, with tested payload builders for
  text generation, audio-to-audio, inpaint, latent encode, and latent decode.
- Job records persist explicit phases and the frontend lands on successful
  output artifacts when artifact IDs arrive, reducing the silent-run feel.
- The specimen lineage thread is data-backed by artifacts, jobs, and result
  families instead of decorative routing. Its visible product language still
  needs to move toward source -> gesture -> take -> branch/memory.
- The session tray now has a tested workspace pulse for active takes, families,
  job state, listening decisions, archive volume, and the next real focus
  action. Archive artifacts can be recovered into the active session through a
  tested annotation/session reassignment path.
- The listening bench now has a tested recent-take playlist cursor and local
  waveform markers in addition to playback, loop, and decision controls.
  The player now supports marker deletion, relabeling after deletion, loop-edge
  nudging, marker notes, sequence-aware audition navigation, persisted playback
  annotations, WaveSurfer zoom, and draggable loop regions. The committed
  `npm run smoke:playback-session --prefix frontend` script verifies these
  interactions in a real browser with temporary API and Vite servers.
- `sa3-lab smoke-fixture` now provides a cheap local runtime smoke by creating a
  deterministic latent fixture, submitting a real background job through
  `RuntimeDispatcher`, and verifying that a latent output artifact persists.
- `sa3-lab smoke-mlx-medium` now provides an explicitly gated slow path for
  authenticated Medium/MLX runtime verification. Without `--run` or
  `SA3_RUN_MLX_SMOKE=1`, it returns `status: gated`; with the gate enabled it
  submits a real `generate.text_to_audio` Medium recipe through the same job
  manager/runtime/storage path as the app.
- Prompt search is no longer only a script adapter: it has a native recipe path,
  prompt-probe metadata, generated takes, lineage, descriptor deltas, listening
  decisions, decision summaries, and first-pass prompt memory.
- Bundle inspectors now surface parsed sweep outputs, memory hits, vector NPZs,
  encoded dataset manifests, profile aggregates, direction NPZ metadata,
  geometry artifacts, soft-prompt tensors, and training checkpoints as domain
  cards instead of only file inventory.
- Operator Studio has browser-local presets with visible diff and revert
  behavior.
- The frontend has completed a de-monolith pass. `App.tsx` is now the
  composition root for query/mutation wiring, form state, payload orchestration,
  and cross-domain handlers. Static configs, workbench/result-family model
  helpers, specimen/listening UI, session/archive UI, comparison, mode atlas,
  prompt search rack, operator preset rack, readiness/status panels, audition
  stack, and spec coverage widgets live in typed modules with clearer ownership.
- Extracted panel coverage now includes focused Testing Library tests for the
  specimen panel, session tray, status panels, prompt-search rack, operator
  preset rack, and spec-coverage widgets.
- The prompt-search contract now exposes the `separator` parameter in backend
  specs, and only currently implemented prompt probes (`lexical_probe` and
  `sa3_flow_probe`) are selectable.
- The docs already cover app overview, Colab capability mapping, triage status,
  architecture horizon, control-plane direction, improvement roadmap, and
  codebase review.

### Current Maturity

The app is in a strong runtime-prototype phase but a weak product-interface
phase. It can run meaningful workflows and preserve useful provenance, but the
visible app still feels too much like an engineering dashboard. The next work is
an interface rescue, not more visible feature parity.

The main engineering challenge is not choosing the stack. The stack direction is
sound. The main challenge is translating backend precision into a creative
instrument grammar so that every experiment can be played, tuned, remembered,
branched, and inspected without making the user operate the machinery first.

## Documentation Status

### Already Covered

- `README.md`: install, local runner, model policy, API, control-plane launch,
  and app entrypoints.
- `docs/app-overview.md`: app thesis, main surfaces, artifact model, runtime
  assumptions, and current/partial status.
- `docs/colab-capability-map.md`: migration map from notebook/script capability
  to app surface.
- `docs/app-engine-triage-loop.md`: current promoted/deferred capabilities,
  acceptance rules, and immediate queue.
- `docs/improvement-roadmap.md`: trust, exploration speed, research cognition,
  interface polish, architecture improvements, and verification plan.
- `docs/architecture-horizon.md`: current stack, near/middle/later stack
  promotions, and non-goals.
- `docs/control-plane-architecture.md`: tRPC/Python split, routers,
  deferred persistence, and control-plane queue.
- `docs/codebase-review.md`: working areas, risks, refactor suggestions, and
  verification history.

### Missing Or Still Too Fragmented

- A cell-by-cell or section-by-section notebook parity table with exact
  parameter coverage and UI affordance status.
- A normalized definition of "mode complete" for each experiment family.
- A single test-gap matrix that ties capabilities to Python tests, tRPC tests,
  frontend unit/component tests, and Playwright smoke coverage.
- A session/workspace/archive domain document that treats the creative workflow
  as a product object rather than a side effect of jobs and artifacts.
- A playback/comparison interaction brief for the next player upgrade.
- A data-persistence promotion plan that says exactly when JSON manifests stop
  being enough and Postgres/Drizzle should enter.

## Test Posture

### What The Current Tests Appear To Cover

- Python API/runtime behavior around artifacts, jobs, recipes, operators,
  app contracts, and selected experiment adapters.
- Frontend logic for bundle inspection, prompt-search helper behavior, decision
  summaries, prompt memory, operator presets, playback markers, audition
  cursors, session workspace summaries, operator-spec coverage, and form/data
  transforms.
- Control-plane procedures for workbench reads, job lifecycle, recipe actions,
  artifact inspection, family loading, archive actions, and event bridging.
- Build-level confidence for the React/Vite frontend.
- Smoke-level confidence for the local app path when explicitly run.
- Fixture-backed runtime confidence for storage, jobs, dispatcher execution,
  and output artifact persistence without gated model downloads.

### What Is Still Missing

- A comprehensive Colab parity test matrix.
- Broader contract tests proving that every backend `ui_field` used by every
  mode reaches the frontend form with the correct default, validation, and
  payload key. The generation and SAME payload builders now have focused tests,
  but full mode coverage is still incomplete.
- Component tests for all mode-specific recipe panels once they become more
  domain-specific.
- Browser/integration tests for replay/fork flows, archive review, long-session
  cleanup, and session-level archive/new-session state. The committed playback
  smoke covers annotation persistence, cue persistence,
  sequence modes, SessionTray artifact archive/recovery, screenshots, and
  mobile overflow.
- Component-level tests for mode-specific recipe panels once they become more
  domain-specific. The extracted specimen/session/status/prompt/preset/spec
  panels now have focused fixtures.
- Playback-specific browser tests for generated-take promotion, richer playlist
  sequencing, and future region-export workflows. Marker notes, loop regions,
  persisted cues, and tray archive/recovery now have committed
  browser coverage.
- Performance or runtime-cost tests for Medium/MPS prompt-search probe paths.
- Tests that protect against token or credential leakage in logs, command
  context, and error tails.

## Engineering Health

### Good Decisions

- Keeping Python as the runtime worker is the right call. It owns model
  execution, artifact IO, MLX/PyTorch adapters, and script migration.
- Adding a TypeScript tRPC control plane is valuable when it shapes app meaning
  instead of mirroring Python endpoints one-to-one.
- Keeping Postgres and pgvector out of the first slice was disciplined. They
  should enter only when session, lineage, preset, memory, or retrieval
  semantics are stable enough to persist.
- Treating visual references as interaction metaphors is correct. Routing lines
  should mean lineage or dependency; gradient cells should mean operators,
  state, or parameterized transforms.
- Defaulting to SA3 Medium and SAME-L gives the project a coherent research
  target instead of a scattered demo path.

### Risk Areas

- Frontend field schemas are still partly duplicated in TypeScript. Backend
  `ui_fields` now drive Generate, SAME, Operator Studio, and Recipe Studio
  defaults/options/bounds, but static catalogs still carry layout, labels, and
  some parameter assumptions.
- `RuntimeDispatcher` still owns too many responsibilities and should continue
  splitting into runtime-specific adapters as capability count grows.
- Script-backed experiment modes are durable and useful, but some still need
  native domain interactions rather than generic bundle readers.
- Prompt search is exciting but should remain labeled as a probe until
  Medium/MPS cost notes, layer/alpha comparisons, and listening validation are
  stronger.
- Browser-local presets are useful but not shareable or durable across machines.
- Job events are much better than silent jobs, but the live source still leans
  on polling underneath the control-plane bridge.
- The queue/archive/session model is becoming a primary creative organization
  surface through the workspace pulse, artifact recovery, and direct artifact
  archive actions, but long-session cleanup still needs deeper end-to-end
  workflows.
- The frontend root is much healthier, but the main action bands still combine
  layout, readiness, and handler wiring in `App.tsx`. Extract those only after
  the action-band state contracts are named clearly enough to avoid vague hook
  wrappers.

### Not Bad Code, But Incomplete Architecture

The current system does not read like throwaway code. It reads like a serious
research prototype that has grown quickly and now needs consolidation. The
biggest signs of immaturity are duplicated form knowledge, uneven mode depth,
large runtime hubs, and incomplete session/playback semantics. Those are normal
for this migration stage, but they should now become explicit backlog items
instead of invisible friction.

## Priority Queue

### P0: Capability Truth And Reproducibility

1. Build a notebook parity matrix that maps every Colab section/cell/function
   to a local API path, UI surface, parameters, artifacts, and tests.
2. Define "complete mode" criteria: all required params visible, advanced params
   reachable, defaults known, output artifacts typed, provenance stored, replay
   or fork available, and test coverage present.
3. Continue migrating parameter truth into backend specs and shared contracts.
   This pass expanded script adapter specs and moved Generate/SAME forms onto
   spec-derived fields; the remaining work is full mode-by-mode coverage and
   richer type-specific readers.
4. Ensure every run stores model, backend, seed, duration, prompt-probe/operator,
   source IDs, params, logs, and output artifact IDs.

### P0: Runtime Trust

1. Preserve richer safe command/stderr context for failures without exposing
   credentials.
2. Make progress stages legible for every long-running path: queued,
   preflight, model setup, scoring/sampling, decoding, writing, indexing, done.
   Job records now persist `phase`; continue adding exact phases to newly
   migrated script paths as they become native.
3. Add runtime-cost notes where Medium/MPS paths can be slow or memory-heavy.
4. Keep readiness checks honest for HF auth, weights, cache space, MLX setup,
   SAME-L access, and optional extras.

### P1: Session And Memory As Creative Workflow

1. Promote current sound, takes, branches, and memory into the app's default
   mental model. A first data-backed workspace pulse and archive recovery path
   are implemented, and the playback smoke now covers SessionTray artifact
   archive/recovery. The next step is browser-tested replay/fork and
   session-level memory/new-session behavior.
2. Reduce queue clutter by making active session, archive, generated families,
   and reusable outputs distinct surfaces.
3. Make "new session", "remember session", "recover sound", "branch", and
   "promote material" feel like native creative actions. Successful jobs now
   land on their newest artifact, but session/workspace recovery still needs a
   clearer product model.
4. Move useful archive/search actions through tRPC when client-side filtering
   stops being enough.
5. Add browser-level coverage for the extracted session tray archive/recovery
   workflow now that it has a stable component boundary.

### P1: Playback And Listening Bench

1. Upgrade playback into a creative listening instrument.
2. Add richer loop, marker, region, take-stack, and session-playlist
   behaviors where they directly improve listening decisions. Local markers,
   marker notes, playlist cursor helpers, loop-edge nudging, marker deletion,
   keyboardable audition navigation, persisted cue annotations, WaveSurfer zoom,
   draggable loop regions, and SessionTray archive/recovery
   browser coverage are implemented; richer playlist sequencing remains next.
3. Keep wavesurfer.js focused on real listening work: zoom, loop regions,
   marker editing, scrubbing, and future region annotations.
4. Tie listening decisions back into prompt-search, sweeps, memory reuse, and
   artifact recovery.

### P1: Bundle And Mode-Specific Inspectors

1. Add richer inspectors for vectors, profiles, soft prompts, prompt search,
   sweeps, memory collections, geometry reports, and training outputs. First
   domain item cards exist for sweeps, memory, vectors, soft prompts, and
   training bundles.
2. Keep zip/file inventory available, but do not make it the primary interface
   when a domain-specific summary exists.
3. Add prompt-search layer/alpha branch views and Medium/MPS probe-setting
   notes before adding another prompt objective.

### P2: Research Cognition

1. Add deeper memory/dataset browsing over encoded SAME artifacts, including
   preview audio where possible.
2. Add latent region and channel controls where operators manipulate time,
   channel, masks, or donor regions.
3. Add real lineage graph views only when graph edges correspond to actual
   recipe/source/output relationships. The current specimen thread already uses
   real artifact/job/family/A-B state; React Flow should wait until users need
   a larger interactive graph.
4. Add labelled probes and control-head recipes after the core parity and
   playback loop are more stable.

### P3: Portfolio Hardening

1. Keep docs honest about what is confirmed, partial, inferred, or speculative.
2. Add screenshots or short captures of real workflows once surfaces stabilize.
3. Add Storybook after more components are extracted and need visual QA.
4. Add Postgres/Drizzle after session, annotation, preset, lineage, and job
   event semantics are stable enough to deserve a database.
5. Add pgvector only when there is a concrete creative retrieval workflow such
   as donor suggestions, texture/rhythm search, continuation retrieval, or
   latent neighborhood browsing.

## Definition Of Done For The Next Implementation Loop

A priority slice is not complete just because a control appears in the UI. It
is complete when:

- The operation has a typed request or recipe.
- The UI exposes required and advanced parameters with trustworthy defaults.
- The backend persists full provenance.
- The run produces typed artifacts or an explicit no-output state.
- The artifact can be listened to, inspected, replayed, forked, reused, or
  archived as appropriate.
- Tests cover the contract and at least one important UI/state transform.
- Docs update the capability map and triage queue.

## Active Study Artifact

The detailed notebook parity matrix now lives in
`docs/colab-parity-matrix.md`. It is the gate for the next no-stop
implementation loop and tracks:

- notebook section
- local evidence path
- current API endpoint or script adapter
- current UI surface
- required inputs
- advanced parameters
- output artifact types
- provenance fields
- missing UI controls
- missing tests
- status: native, recipe, partial, scaffold, or deferred
- next action

The next implementation slice should keep that matrix synchronized with
`/colab/modes`, then promote the highest-leverage partial modes: prompt-search
comparison, playback-composer behavior, and session/workspace organization.
