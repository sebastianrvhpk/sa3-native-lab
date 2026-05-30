# Architecture Horizon

This document captures the ideal stack direction for SA3 Native Lab. It is not
a mandate to install every tool immediately. A library is promoted when it
removes solved complexity or unlocks a real latent-audio workflow.

## Current Principle

Use modern frameworks for solved app infrastructure. Spend custom invention on
the research instrument: artifacts, recipes, listening, latent operations,
memory, lineage, and exploratory control.

## Active Core

- Python FastAPI/Pydantic runtime worker for model execution, artifacts, jobs,
  and script adapters.
- React 19, TypeScript, Vite frontend.
- TanStack Query for server state.
- Zustand for local bench state.
- React Aria Components for accessible field controls.
- TanStack Form for recipe/operator form state and validation.
- tRPC and Zod control plane for app-shaped contracts.
- wavesurfer.js for playback zoom, persisted markers, and editable loop
  regions where artifacts have real audio.
- pytest, Vitest/Testing Library, TypeScript build, and committed Playwright
  smoke scripts.

## Frontend Boundary Snapshot

`frontend/src/App.tsx` is now a composition root instead of the app's component
library. It owns high-level query/mutation wiring, backend/control-plane
switching, form state, payload orchestration, top-level layout, and cross-domain
handlers.

The extracted frontend modules now carry the current stable seams. Product
language is being corrected by `docs/product-rescue-brief.md`, so some module
names still reflect legacy implementation nouns and should not be treated as
the desired interface grammar:

- `gestureModel.ts`: product gesture definitions, source requirements,
  backend operator mappings, progress phrases, and availability reasons.
- `pendingTakeModel.ts`: product-facing translation of `JobRecord` into
  pending/failed take state for the Takes flow.
- `memoryModel.ts`: product-facing remembered material, roles, reuse intent,
  source mappings, and available memory actions over existing artifacts.
- `nextActionModel.ts`: context-aware `Next` actions for landed artifacts,
  pending/failed takes, and branches.
- `pendingTakeLandingModel.ts`: landing artifact selection, completion phrases,
  branch labels, retry/cancel/inspect availability, and failure recovery copy.
- `branchModel.ts`: branch summaries and inspect-only technical rows over the
  legacy result-family contract.
- `branchListeningModel.ts`: branch trajectory ordering and previous/next
  cursor state for landed audio takes.
- `tuneFieldGroups.ts`: product-layer primary/advanced/inspect grouping over
  backend-derived field configs.
- `sourceModel.ts`: product-facing source, anchor, donor, remembered, imported,
  take, promoted bundle audio, and bundle roles over stored material, plus
  source actions backed by existing memory reuse paths.
- `sourceShelf.tsx`: the unified Source surface for selecting and reusing
  current material, memory, donors, imports, and bundle-derived audio.
- `artifactFilters.ts`: metadata-only Memory browser filters for role, reuse
  intent, tags, notes, kind, decision, and source lineage.
- `bundleReuseModel.ts`: shared bundle-to-gesture reuse paths for profiles,
  directions, sweeps, prompt-search prompts, memory bundles, datasets, soft
  prompts, and training checkpoints where existing recipe fields support them.
- `useGestureWorkbench.ts`: named frontend orchestration hook for active
  gesture, Tune form state, donor/source reuse, next-action routing, prompt
  seeding, and bundle reuse. App-level submit mutations and persistence side
  effects still stay in `App.tsx`.
- `gestureStrip.tsx`, `tuneDrawer.tsx`, `pendingTakesPanel.tsx`: first-pass
  instrument components for choosing the next move, tuning only the active
  gesture, and keeping job state near takes instead of exposing jobs as the
  primary noun.
- `nextActionsPanel.tsx`: the Current Sound-adjacent affordance that turns a
  selected take into a suggested next gesture, remember action, or inspect path.
- `workbenchConfigs.ts`: static mode/config/default catalogs.
- `workbenchModel.ts`: pure job, result-family, event, status, and filtering
  helpers with focused tests.
- `specimenPanel.tsx`: selected artifact playback/inspection, annotation,
  bundle lazy-loading, vitals, and lineage thread.
- `sessionPanel.tsx`: session tray, workspace pulse, archive recovery, and
  artifact filters.
- `comparePanel.tsx`, `modeAtlas.tsx`, `auditionStackPanel.tsx`: anchor/source
  listening slots, Colab parity atlas kept off the primary loop, and
  sequence-aware listening stack.
- `promptSearchRack.tsx`, `operatorPresetRack.tsx`, `statusPanels.tsx`,
  `specCoverage.tsx`: reusable control/readiness widgets.
- `bundleInspector.tsx`: still lazy-loaded so bundle-heavy inspectors do not
  bloat the first interaction path.

Current focused component coverage exists for `gestureModel.ts`,
`pendingTakeModel.ts`, `memoryModel.ts`, `nextActionModel.ts`,
`tuneFieldGroups.ts`, `nextActionsPanel.tsx`, `specimenPanel.tsx`,
`sessionPanel.tsx`, `statusPanels.tsx`, `promptSearchRack.tsx`,
`operatorPresetRack.tsx`, and `specCoverage.tsx`. The next frontend extraction
should therefore target memory/branch reuse semantics and app-shaped control
plane state, not another broad component split.

Remaining frontend extraction should be driven by behavior, not cosmetics:
continue shrinking root orchestration only when gesture, pending-take, memory,
and branch state contracts are clean enough to name without hiding complexity.

## Near Horizon

| Area | Preferred Stack | Promotion Trigger |
| --- | --- | --- |
| Controls | React Aria Components | Continue replacing remaining bespoke selects, dialogs, tabs, menus, tooltips, and tables with accessible primitives that can keep the custom visual language. |
| Forms | TanStack Form | Expand the current form foundation so backend `ui_fields` drive validation, readiness, and repeatable payload building. |
| Testing | Vitest, Testing Library, MSW, Playwright | Keep expanding component and contract fixtures around extracted panels, then add MSW when mocked API/tRPC states become easier than hand-built fixtures. |
| App contract | tRPC and Zod | Harden job-event subscriptions and move family actions, archive flows, and deeper artifact inspectors behind app-level procedures when they need UI-shaped data. |

## Middle Horizon

| Area | Preferred Stack | Promotion Trigger |
| --- | --- | --- |
| Motion | Motion for React | Use causal transitions for pending take creation, take landing, sound remembered, lineage forks, and branch expansion. |
| Component lab | Storybook | Extracted panels now need visual QA across job, artifact, form, empty, archive, and error states. |
| Persistence | Drizzle and Postgres | Sessions, jobs, recipes, annotations, lineage, presets, result families, and job events outgrow JSON manifests. |
| Local Postgres | PGlite | We need lightweight local or test Postgres behavior without requiring a running database service. |
| Vectors | pgvector inside Postgres | Latent/audio summary vectors become useful for memory search, donor suggestions, continuation references, or texture/rhythm retrieval. |
| Lineage graph | React Flow | Artifact lineage and recipe graphs are real enough to inspect as graphs, not generic node editing. |
| Analytics | DuckDB first, Polars when needed | Local queries over artifact/job/result metrics or heavier Python batch transforms become common. |

## Later Horizon

| Area | Preferred Stack | Promotion Trigger |
| --- | --- | --- |
| Observability | OpenTelemetry | We need traces from UI action to tRPC to Python job to MLX/PyTorch subprocess to artifact output. |
| Durable workflows | Temporal | Multi-step jobs need crash-safe resume, retries, cancellation, dependency chains, and worker pools. |
| Monorepo | Turborepo | The repo grows into `frontend`, `apps/control-plane`, shared packages, desktop shell, and docs/site with repeated build/test tasks. |
| 3D | Three.js / React Three Fiber | Memory embeddings, clusters, trajectories, or latent neighborhoods need a real spatial map. |
| Desktop shell | Tauri | The local instrument needs installable Mac app behavior, app menus, file permissions, and process orchestration beyond localhost. |

## Explicit Non-Goals For Now

- Do not add a framework only because it is impressive.
- Do not build a graph UI before graph semantics are real.
- Do not add 3D for decoration.
- Do not introduce durable orchestration before job semantics are stable.
- Do not replace Vite with a hosted-web framework while the main product is a
  local model instrument.
- Do not make both DuckDB and Polars primary. Use DuckDB for local queries first
  and Polars for Python dataframe pipelines when the need is concrete.

## Deferred Promotion Candidates

This is not the active product queue. The current implementation queue lives in
`docs/implementation-readiness.md`; these tools should be promoted only when the
triggers above become concrete.

1. React Aria Components for accessible controls.
2. MSW for mocked API/tRPC component states after the current fixture helpers
   become repetitive.
3. More tRPC/Zod procedures for resumable job events, family actions, archive
   mutations, and backend-parsed artifact inspectors.
4. Storybook for extracted panels once component fixtures are representative.
