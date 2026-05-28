# App Engine Triage Loop

This document keeps the local-app work honest as the repo moves from Colab cells
to a background Python API and instrument UI.

## Instrument Thesis

SA3 Native Lab is a latent listening bench: every action should move through a
real signal path, produce durable artifacts, and preserve the recipe needed to
re-run or fork the result.

```text
source audio / prompt
  -> model backend
  -> latent or audio operator
  -> output artifact
  -> recipe + lineage + annotation
```

## Current Triage

| Priority | Interaction / capability | Code path | Decision | Verification |
| --- | --- | --- | --- | --- |
| P0 | Health, backend readiness, operator inventory | `sa3_native_lab.app.server` + `RuntimeDispatcher` | Keep | API tests, `/health`, `/operators/specs` |
| P0 | Durable artifacts | `ArtifactStore` | Keep | latent roundtrip tests |
| P0 | Durable recipes and background jobs | `JobManager` + `Recipe` | Keep | job persistence tests |
| P0 | Medium/SAME-L default exploration path | contracts + runtime + frontend defaults | Keep | app default test, frontend build |
| P0 | Apple Silicon text/audio generation | MLX subprocess wrapper | Keep | Real MLX smoke and API job passed |
| P0 | Import audio and latent files | FastAPI multipart endpoints | Keep | API import test |
| P1 | Direct latent `.npy` blur/DSP/graft/renoise/roll | `latent_audio_primitives` wrappers | Keep as lab mode | operator job tests |
| P1 | SAME encode/decode endpoints | Torch/MPS model path | Keep | Real SAME-S MPS smoke and API jobs passed |
| P1 | Result comparison/playback surface | `frontend/` listening bench | Keep first slice | Chrome smoke, A/B slots, API operator job |
| P1 | Real audio waveform peaks | `/artifacts/{id}/peaks` + React Query waveform | Promote | storage/API tests, Chrome smoke |
| P1 | Visible artifact lineage | selected artifact source IDs in the specimen bus | Keep as truthful connective tissue | Chrome smoke |
| P1 | Colab experiment scripts as background recipes | `/experiments/run` + script-job adapter | Promote as bridge | focused app tests, frontend build |
| P1 | Native script-parameter UI | Recipe Studio schema in `frontend/` | Promote | frontend build |
| P1 | Native latent-operator parameter UI | Operator Studio schema in `frontend/` | Promote | frontend build, live UI smoke |
| P1 | Backend-derived operator field metadata | `/operators/specs` `ui_fields` + frontend merge helpers | Promote | Python contract test, frontend form tests/build |
| P1 | Colab mode parity/status inventory | `/colab/modes` + Mode Atlas | Keep as triage map | API test, frontend build |
| P1 | App-shaped tRPC workbench reads | `apps/control-plane` + feature-flagged frontend client | Promote carefully | control-plane tests, frontend build, `sa3-lab dev --with-control-plane` |
| P1 | Typed recipe replay/fork, retry, cancellation | Python API + tRPC job/recipe routers + React actions | Promote | API tests, control-plane tests, frontend build |
| P1 | Live job event snapshots | `/jobs/{id}/events` + React Query/live cache merge | Promote first slice | websocket API test, frontend build |
| P1 | Durable job-event replay | Python JSONL job journal + `/jobs/{id}/events/history` + tRPC `jobs.events` replay | Promote | API tests, control-plane tests |
| P1 | Control-plane job-event bridge | tRPC/SSE `jobs.events` subscription over durable journal replay plus Python job snapshot polling with heartbeat/resume diagnostics | Promote | control-plane tests, frontend build |
| P1 | Job recovery hints | `JobProgress` failure classifier over errors/logs | Promote | frontend unit tests |
| P1 | Runtime readiness checks | `/readiness` + `system.readiness` + readiness panel | Promote | API tests, control-plane tests, frontend tests |
| P1 | Fork-with-edited-params UI | recipe-derived fork editor with diff/reset controls | Promote | frontend tests/build |
| P1 | Bundle inspection and result families | `/artifacts/{id}/inspect` + `families.load` + result rail/detail panel | Promote | API tests, control-plane tests, frontend tests |
| P1 | Typed bundle readers | backend `bundle_summary` parser + `BundleField` readers for memory/profile/vector/sweep/soft-prompt/training outputs, metrics, plot discovery, and recipe-input reuse | Promote | API tests, frontend tests/build |
| P1 | Alpha-sweep family promotion | `FamilyDetailPanel` alpha variant band with explicit A/B promotion and compact metric table | Promote first slice | frontend tests/build |
| P2 | Memory atlas and retrieval | `LatentMemoryIndex` + `memory.query` + memory-hit reuse actions | Promote first slice | nearest-neighbor runtime test, frontend smoke |
| P2 | Residual steering and prompt search | existing scripts/experiments | Defer | model-backed recipe tests |

## Acceptance Tests Per Pass

- A capability must have a typed request/recipe.
- A run must produce or update a persistent artifact/job record.
- The response must expose whether the operation is implemented, queued, or
  explicitly not available on the current machine.
- Any visual control in the future UI must map to one of these contracts or be
  marked as speculative.

## Immediate Next Queue

1. Add embedded plot rendering for discovered bundle plot/image files.
2. Add richer kind-specific inspectors for style profiles, vectors, soft
   prompts, memory collections, sweeps, and training outputs.
3. Add sortable sweep tables and recipe comparison across sibling runs.
4. Continue shrinking frontend field drift until backend `ui_fields` can drive
   most controls without losing the instrument-specific layout.
5. Add prompt-search recipe adapters for Colab Modes 2/3/5.
6. Add geometry/control-head recipe adapters for Colab Modes 12/15.
7. Add parameter presets and recipe diffing for Operator Studio.
8. Promote MLX generation from subprocess-only to a resident worker when repeated
   generation needs lower overhead.
9. Add deeper visual lineage for multi-step artifact families once recipe jobs
   are populated enough to make the routing real.
