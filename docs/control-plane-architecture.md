# Control Plane Architecture

This is the migration path from a direct React-to-FastAPI app into a hybrid
AI instrument with a TypeScript application contract and a Python model runtime.

## Thesis

The Python service remains the compute/runtime worker. It owns MLX, PyTorch,
SAME encode/decode, script adapters, artifact file IO, and long-running jobs.

The TypeScript control plane owns app meaning. It should expose instrument-level
procedures, not one-to-one mirrors of Python endpoints.

```text
React workbench
  -> tRPC control plane
    -> app-shaped procedures
    -> Python runtime worker
      -> MLX / PyTorch / SAME / scripts / artifacts
```

## First Slice

`apps/control-plane` starts with `workbench.load`, a tRPC procedure that gathers
Python runtime data and shapes it into the state the instrument actually needs:

- health and backend readiness
- sessions and active session
- session artifacts and archive artifacts
- session jobs, archive jobs, running jobs, latest job
- result families and latest family metrics
- mode atlas and operator specs
- selected artifact fallback
- artifact/job/mode counts

This is deliberately not a thin proxy. The procedure moves grouping and
readiness logic out of React and into a testable app contract.

The React frontend now has a feature-flagged tRPC client. When
`VITE_SA3_CONTROL_PLANE_URL` is set, read-heavy workbench state comes from
`workbench.load`. System readiness, job lifecycle, recipe replay/fork, artifact
inspection, and family reads are also exposed as app-shaped tRPC procedures.
Model execution and artifact file IO still belong to the Python worker. Live
job snapshots now use a `jobs.events` tRPC/SSE subscription when the control
plane is enabled. Python persists per-job JSONL snapshots and exposes them via
`/jobs/{job_id}/events/history`; the bridge replays missed journal events
before it resumes live polling. This keeps the React contract stable while
leaving room to swap the live source to the Python WebSocket stream later.
Events now include monotonic IDs, resume-aware sequencing, heartbeat
diagnostics, and log tails.
When the env var is absent, the frontend falls back to the existing direct
Python read queries, Python mutations, and Python job WebSockets.

On the UI side, result families are now treated as app objects: the right rail
can inspect a family, play its audio artifacts, assign A/B comparison slots,
show related jobs, promote alpha-sweep variants, and branch the recipe with
visible diffs. Memory-query bundle hits can also feed app actions when they
resolve to local artifacts. Bundle inspection now includes backend-parsed
summaries for JSON/NPZ outputs, metric scalars, and plot/image discovery in
addition to file inventory. Reusable bundles can populate Recipe Studio fields
for vectors, directions, profiles, memory folders, soft prompts, and LoRA
checkpoints. Operator specs also carry backend-derived `ui_fields`, which React
merges into the hand-shaped instrument forms.

The local runner can launch the full path:

```bash
uv run sa3-lab dev --with-control-plane
```

## Current Routers

| Router | Procedure | Purpose |
| --- | --- | --- |
| `system` | `readiness` | Report app/runtime readiness checks from the Python worker. |
| `workbench` | `load` | Build UI-ready workbench state, including operator `ui_fields`, from Python runtime records. |
| `jobs` | `list`, `get`, `events`, `cancel`, `retry` | App-level lifecycle actions with durable journal replay and live snapshots over Python background jobs. |
| `recipes` | `replay`, `fork` | Re-run or branch a persisted recipe without rebuilding payloads in React. |
| `artifacts` | `inspect` | Fetch artifact, recipe, sources, children, bundle file inventory, safe previews, and parsed bundle summaries. |
| `families` | `load` | Return grouped recipe result families for the active workbench scope. |
| `archive` | `search` | Search artifact annotations through the runtime worker. |
| `archive` | `annotateAndSearch` | Update annotations, then return the refreshed matching archive slice. |

The standalone server also exposes `GET /health` so the dev runner can wait for
the control plane without reaching through tRPC internals.

## Deferred On Purpose

Postgres and pgvector are not part of the first tRPC slice. They become useful
after the control plane has stable domain procedures.

Future responsibilities:

- Postgres app ledger: sessions, artifacts metadata, recipes, jobs, job events,
  annotations, lineage edges, presets, and result families.
- pgvector latent memory atlas: fixed-dimensional summaries and window vectors
  derived from SAME `.npy` artifacts, used for texture/motion/donor/continuation
  retrieval.

## Acceptance Criteria

- Any tRPC procedure must represent an app action or app-shaped query.
- Heavy model execution stays in Python.
- UI-facing procedures should validate input with Zod.
- Control-plane logic must have tests without requiring MLX/PyTorch models.
- The frontend must keep a direct-Python fallback until the control-plane path
  owns enough behavior to be the normal launch mode.
- Direct endpoint mirroring is allowed only as a temporary adapter behind a
  richer app-level procedure.

## Next Control-Plane Queue

1. Replace live polling inside `jobs.events` with a stream source while keeping
   the durable journal replay contract.
2. Promote archive annotation/search mutations to the normal UI path.
3. Move more bounded fork and recipe forms onto backend-derived `ui_fields`.
4. Promote family detail and memory-result actions into tRPC procedures where
   they need server-side shaping beyond `workbench.load`.
5. Add richer family-specific inspectors and embedded plot previews for sweeps, memory
   query bundles, and style profile bundles.
6. Evaluate Postgres only after these procedures stabilize.
