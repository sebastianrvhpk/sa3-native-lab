# Implementation Readiness And Engineering Horizon

This document is the current operating map for SA3 Native Lab before the next
implementation loop. It consolidates the app objective, current maturity,
engineering risks, test posture, and the next priority queue.

## App Objective

SA3 Native Lab is a local research instrument for Stable Audio 3 Medium and
SAME-L. Its purpose is to turn Colab-style experimental cells into typed,
replayable, inspectable workflows that run on local hardware, especially Apple
Silicon through MLX where practical.

The app should not become a generic dashboard. The core act is listening and
iterating on latent/audio experiments:

```text
prompt or source audio
  -> SA3 Medium / SAME-L runtime
  -> recipe, operator, scorer, or script experiment
  -> artifact family
  -> playback, comparison, annotation, lineage, replay, and reuse
```

Every visible control should map to a backend parameter, a recorded recipe
field, an artifact relationship, or an explicitly documented future capability.

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
- The React app is a real listening bench with artifacts, jobs, playback, A/B,
  recipes, operators, bundle inspection, annotation, archive filters, result
  families, prompt-search generated takes, decision summaries, and prompt
  memory.
- Backend `ui_fields` already reduce parameter drift by exposing defaults,
  bounds, options, required flags, advanced flags, and artifact-kind hints to
  the frontend.
- Prompt search is no longer only a script adapter: it has a native recipe path,
  scorer metadata, generated takes, lineage, descriptor deltas, listening
  decisions, decision summaries, and first-pass prompt memory.
- Operator Studio has browser-local presets with visible diff and revert
  behavior.
- The docs already cover app overview, Colab capability mapping, triage status,
  architecture horizon, control-plane direction, improvement roadmap, and
  codebase review.

### Current Maturity

The app is in a strong prototype-to-instrument phase. It is not just a sketch,
but it is also not yet a fully complete Colab replacement. The current system
can run meaningful workflows and preserve useful provenance, but parity is still
uneven across all modes, all parameters, all bundle types, and all session-level
creative workflows.

The main engineering challenge is not choosing the stack. The stack direction is
sound. The main challenge is finishing the contracts and interaction grammar so
that every experiment can be trusted, replayed, compared, and reused.

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
  summaries, prompt memory, operator presets, and form/data transforms.
- Control-plane procedures for workbench reads, job lifecycle, recipe actions,
  artifact inspection, family loading, archive actions, and event bridging.
- Build-level confidence for the React/Vite frontend.
- Smoke-level confidence for the local app path when explicitly run.

### What Is Still Missing

- A comprehensive Colab parity test matrix.
- Contract tests proving that every backend `ui_field` used by a mode reaches
  the frontend form with the correct default, validation, and payload key.
- Component tests for all mode-specific recipe panels once they become more
  domain-specific.
- Integration tests for session/archive/replay flows as first-class workflows.
- Playback-specific tests for loop regions, A/B behavior, generated-take
  promotion, and session playlists.
- Performance or runtime-cost tests for Medium/MPS prompt-search scorer paths.
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
  `ui_fields` are reducing drift, but static catalogs still carry layout,
  labels, and some parameter assumptions.
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
- The queue/archive/session model is present but not yet the primary creative
  organization surface.

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
4. Ensure every run stores model, backend, seed, duration, scorer/operator,
   source IDs, params, logs, and output artifact IDs.

### P0: Runtime Trust

1. Preserve richer safe command/stderr context for failures without exposing
   credentials.
2. Make progress stages legible for every long-running path: queued,
   preflight, model setup, scoring/sampling, decoding, writing, indexing, done.
3. Add runtime-cost notes where Medium/MPS paths can be slow or memory-heavy.
4. Keep readiness checks honest for HF auth, weights, cache space, MLX setup,
   SAME-L access, and optional extras.

### P1: Session And Archive As Creative Workflow

1. Promote workspace/session/run/family/artifact into the app's default mental
   model.
2. Reduce queue clutter by making active session, archive, generated families,
   and reusable outputs distinct surfaces.
3. Make "new session", "archive session", "recover result", "fork run", and
   "promote artifact" feel like native creative actions.
4. Move useful archive/search actions through tRPC when client-side filtering
   stops being enough.

### P1: Playback And Listening Bench

1. Upgrade playback into a creative comparison instrument.
2. Add richer loop, marker, region, A/B, take-stack, and session-playlist
   behaviors where they directly improve listening decisions.
3. Consider wavesurfer.js when regions, zoom, marker editing, and waveform
   interaction become central enough to justify the dependency.
4. Tie listening decisions back into prompt-search, sweeps, memory reuse, and
   artifact recovery.

### P1: Bundle And Mode-Specific Inspectors

1. Add richer inspectors for vectors, profiles, soft prompts, prompt search,
   sweeps, memory collections, geometry reports, and training outputs.
2. Keep zip/file inventory available, but do not make it the primary interface
   when a domain-specific summary exists.
3. Add prompt-search layer/alpha comparisons and Medium/MPS scorer-setting
   notes before adding another scorer objective.
4. Add CLAP or hybrid scoring only when the prompt-search comparison workflow
   can actually help judge it.

### P2: Research Cognition

1. Add deeper memory/dataset browsing over encoded SAME artifacts, including
   preview audio where possible.
2. Add latent region and channel controls where operators manipulate time,
   channel, masks, or donor regions.
3. Add real lineage graph views only when graph edges correspond to actual
   recipe/source/output relationships.
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
