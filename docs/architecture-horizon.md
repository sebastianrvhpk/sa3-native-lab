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
- pytest, Node test, TypeScript build, and Playwright smoke.

## Frontend Boundary Snapshot

`frontend/src/App.tsx` is now a composition root instead of the app's component
library. It owns high-level query/mutation wiring, backend/control-plane
switching, form state, payload orchestration, top-level layout, and cross-domain
handlers.

The extracted frontend modules now carry the stable seams:

- `workbenchConfigs.ts`: static mode/config/default catalogs.
- `workbenchModel.ts`: pure job, result-family, event, status, and filtering
  helpers with focused tests.
- `specimenPanel.tsx`: selected artifact playback/inspection, annotation,
  bundle lazy-loading, vitals, and lineage thread.
- `sessionPanel.tsx`: session tray, workspace pulse, archive recovery, and
  artifact filters.
- `comparePanel.tsx`, `modeAtlas.tsx`, `auditionStackPanel.tsx`: A/B display,
  Colab parity atlas, and sequence-aware listening stack.
- `promptSearchRack.tsx`, `operatorPresetRack.tsx`, `statusPanels.tsx`,
  `specCoverage.tsx`: reusable control/readiness widgets.
- `bundleInspector.tsx`: still lazy-loaded so bundle-heavy inspectors do not
  bloat the first interaction path.

Remaining frontend extraction should be driven by behavior, not cosmetics:
extract the main generation/SAME/operator/recipe action bands only when their
state and handler contracts are clean enough to name without hiding complexity.

## Near Horizon

| Area | Preferred Stack | Promotion Trigger |
| --- | --- | --- |
| Controls | React Aria Components | Continue replacing remaining bespoke selects, dialogs, tabs, menus, tooltips, and tables with accessible primitives that can keep the custom visual language. |
| Forms | TanStack Form | Expand the current form foundation so backend `ui_fields` drive validation, readiness, and repeatable payload building. |
| Testing | Vitest, Testing Library, MSW, Playwright | Add fast unit/component tests for payload builders and mocked API/tRPC states while keeping Playwright for real app flows. |
| App contract | tRPC and Zod | Harden job-event subscriptions and move family actions, archive flows, and deeper artifact inspectors behind app-level procedures when they need UI-shaped data. |

## Middle Horizon

| Area | Preferred Stack | Promotion Trigger |
| --- | --- | --- |
| Motion | Motion for React | Use causal transitions for queued/running/produced states, lineage forks, A/B promotion, and result-family expansion. |
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

## Next Promotion Candidates

1. React Aria Components for accessible controls.
2. Vitest, Testing Library, and MSW for fast contract and component tests.
3. More tRPC/Zod procedures for resumable job events, family actions, archive
   mutations, and backend-parsed artifact inspectors.
4. Storybook for extracted panels once component fixtures are representative.
