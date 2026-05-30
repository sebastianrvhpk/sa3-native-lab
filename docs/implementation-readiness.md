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
- The listening surface now has a tested recent-take playlist cursor and local
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
- Latent Gestures have browser-local presets with visible diff and revert
  behavior.
- The frontend has completed a de-monolith pass. `App.tsx` is now the
  composition root for query/mutation wiring, form state, payload orchestration,
  and cross-domain handlers. Static configs, workbench/result-family model
  helpers, specimen/listening UI, session/archive UI, comparison, mode atlas,
  prompt search rack, operator preset rack, readiness/status panels, audition
  stack, and spec coverage widgets live in typed modules with clearer ownership.
- The first product-domain rescue pass after that split adds explicit
  `Gesture` and `PendingTake` frontend models. Generation, SAME, latent
  operators, experiments, and remember/archive actions are now selected through
  one gesture strip and one scoped Tune surface rather than appearing as four
  simultaneous form panels. Active or interrupted jobs are translated into
  pending/failed take cards in the Takes / Branches rail.
- The priority-queue product loop pass adds product-domain models for memory
  reuse, landed-take next actions, pending-take landing, branch summaries, and
  Tune field grouping. Remembered audio/latent/bundle material can now be used
  as source, anchor, donor, prompt seed, Advanced Gesture input, or recovered
  into the active session where existing backend paths support it.
- Selected takes now expose a `Next` affordance beside Current Sound. Audio
  takes suggest Continue, Vary, Encode, and Remember; latent takes suggest
  Decode, Morph, Borrow Texture, and Remember; bundles surface Inspect plus
  available bundle-to-gesture reuse actions. These actions select the matching
  Gesture/Tune state instead of sending users back to a dashboard.
- Branches are reframed as creative paths with source, gesture, latest take,
  take count, status, do-again, branch, remember, and Inspect details. Raw job
  IDs, recipe IDs, backend fields, logs, and material counts remain available
  through Inspect/Settings/details instead of leading the primary view.
- JSON manifest writes are now atomic, which makes annotation, remember, and
  recovery less race-prone when the browser polls artifact state during a
  workflow.
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

The app is in a strong runtime-prototype phase and now has a credible
product-interface loop. It can run meaningful workflows, preserve useful
provenance, and expose the first-use path as Current Sound -> Gesture -> Pending
Take -> Listen -> Branch / Remember / Tune. The next work is not visible Colab
parity; it is sharpening listening review, Memory browsing, branch actions, and
gesture orchestration without letting backend precision dominate the first
screen again.

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
- Product-loop frontend models now have focused coverage for memory reuse,
  landed-take next actions, pending-take landing phrases, branch-card actions,
  and Tune primary/advanced field grouping.
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
  mobile overflow. The committed first-use smoke now gates the product loop:
  Current Sound, Gestures, Make, Tune, pending/failed take language, Next
  actions, Remember, Branch, Memory reuse as Source/Anchor, recovery,
  Settings/Inspect demotion, desktop and mobile screenshots, and no mobile
  horizontal overflow.
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
  `ui_fields` now drive Generate, SAME, Latent Gestures, and Advanced Gestures
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
- The frontend root is much healthier, and the main action bands now have
  named product models. `App.tsx` still owns too much action orchestration
  because mutation handlers, active gesture state, and form state are not yet
  extracted into a clean `useGestureWorkbench`-style contract. Do that only
  once memory/branch reuse semantics are clearer.

### Not Bad Code, But Incomplete Architecture

The current system does not read like throwaway code. It reads like a serious
research prototype that has grown quickly and now needs consolidation. The
biggest signs of immaturity are duplicated form knowledge, uneven mode depth,
large runtime hubs, and incomplete session/playback semantics. Those are normal
for this migration stage, but they should now become explicit backlog items
instead of invisible friction.

## Priority Queue

### Completed In This Product-Loop Pass

1. P0.1 Memory reuse: active Memory actions now cover use as Source, Anchor,
   latent donor for Borrow Texture, prompt seed, bundle-to-Advanced-Gesture
   reuse, and recovery into the current session. No vector search or new
   persistence layer was added.
2. P0.2 Landed-take next actions: selected audio, latent, bundle, pending, and
   failed states now produce context-aware `Next` actions that select the right
   Gesture/Tune state.
3. P0.3 First-use smoke: `npm run smoke:first-use --prefix frontend` is the
   product-health browser gate for Current Sound -> Gesture -> Pending Take ->
   Listen -> Branch / Remember / Tune, with desktop/mobile screenshots and
   overflow checks.
4. P0.4 Pending-take landing: pending cards now carry source IDs, produced
   outputs, landing artifact, completion phrase, branch label, suggested next
   gestures, cancel/retry/inspect availability, and failure recovery copy.
5. P0.5 Branch reframe: result-family UI is now branch-language first. Raw
   recipe/job/backend/metric details are behind Inspect.
6. P0.6 Tune scope reduction: Make, Continue/Vary, Encode/Decode, Morph, and
   Steer use product-layer primary/advanced/inspect field grouping while still
   preserving backend `ui_fields` as parameter truth.

### P1 Work Landed Where It Was Enabled By P0

1. Gesture-specific Tune layouts have a first pass for Make, Continue/Vary,
   Encode/Decode, Morph/Borrow Texture, and Steer submodes.
2. Morph uses product labels Roll, Blur, DSP/Reroute, Borrow Texture, and
   Renoise, with backend operator names retained behind Inspect.
3. Borrow Texture now explains donor availability and points to encode/import or
   recovered memory when no donor exists.
4. Remember annotations now include role, reuse intent, and decision metadata
   where the existing annotation model can persist it.
5. Inspect is more contextual: sound, gesture, branch, activity, and material
   count details are progressively disclosed instead of primary.
6. Source and Memory now share a unified product source shelf: current source,
   remembered material, donor latent, imported audio, promoted bundle audio, and
   reusable bundles all route through the same source-action path backed by the
   memory/source models.
7. The take strip is now a stronger listening queue with keep/maybe/reject,
   remember, continue, and branch actions beside playback. Row actions first
   select the clicked take so Continue/Branch/Remember and listening decisions
   route from the material the user is acting on.
8. Bundle-to-gesture promotion now comes from one shared model so Memory and
   bundle Inspect expose the same real recipe-field paths. Promoted bundle audio
   becomes normal source material after the existing backend promotion path
   creates an audio artifact.
9. Gesture orchestration now has a first named hook boundary in
   `useGestureWorkbench` for active gesture, Tune form state, donor/source
   reuse, next-action routing, prompt seeding, and bundle reuse.
10. Branch detail now has a playable trajectory cursor over landed audio takes
    and per-take Continue, Branch, Remember, and keeper/maybe/reject actions
    where existing recipes and audio artifacts support them.
11. Memory browsing has explicit stored-metadata filters for role, reuse intent,
    tags, notes, kind, listening decision, and source lineage, including
    bundle-derived audio. No vector search or new persistence layer was added.

### Deferred With Reason

1. P1.8 deeper gesture orchestration extraction is intentionally bounded:
   `useGestureWorkbench` owns active gesture, Tune form state, donor/source
   reuse, next-action routing, prompt seeding, and bundle reuse. Submit
   readiness, pending-take selection, landing side effects, archive/recover
   persistence, and preset persistence remain in `App.tsx` because they still
   coordinate React Query mutations, websocket/tRPC job events, and storage
   side effects. Pulling them into the hook would make the boundary less clear.
2. P1.9 app-native tRPC state is deferred because the frontend can now express
   current sound, gestures, pending takes, branches, and memory without adding a
   new server contract in this pass.
3. P1.10 broader contract truth cleanup remains incremental: backend
   `ui_fields` stay parameter truth; the new frontend helpers are layout and
   product overlays only.
4. P1.11 deeper workspace modeling is deferred beyond the memory/branch/pending
   pieces used by this loop.
5. P2 listening/review items are documented as next queue. Richer playlist
   review, waveform regions, mode-specific inspectors, and deeper branch-review
   workflows need more design and should not be decorative add-ons.
6. Postgres, pgvector, Temporal, React Flow, Tauri, Storybook, DuckDB,
   OpenTelemetry, and 3D remain out of scope for this pass.

### Next Queue

1. Decide whether submit readiness should ever move into a smaller action
   object, but keep mutation and landing side effects at the app edge until the
   job-event contract is simpler.
2. Turn the source shelf into a reusable picker inside Tune fields where it
   would replace generic artifact selects without hiding backend-supported
   paths.
3. Continue listening review with playlist-level keep/maybe/reject summaries,
   branch trajectory favorites, and branch-from-selected shortcuts only where
   they reduce repeated clicks.
4. Design mode-specific Inspect panels for profiles, directions, geometry,
   datasets, residual vectors, prompt search, and soft prompts using existing
   bundle summaries.
5. Revisit app-native tRPC state only if it removes frontend orchestration
   complexity; otherwise keep the current control-plane read shape.

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
