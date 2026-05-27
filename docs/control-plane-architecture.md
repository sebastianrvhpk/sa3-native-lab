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
- mode atlas and operator specs
- selected artifact fallback
- artifact/job/mode counts

This is deliberately not a thin proxy. The procedure moves grouping and
readiness logic out of React and into a testable app contract.

## Current Routers

| Router | Procedure | Purpose |
| --- | --- | --- |
| `workbench` | `load` | Build UI-ready workbench state from Python runtime records. |
| `archive` | `search` | Search artifact annotations through the runtime worker. |
| `archive` | `annotateAndSearch` | Update annotations, then return the refreshed matching archive slice. |

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
- Direct endpoint mirroring is allowed only as a temporary adapter behind a
  richer app-level procedure.
