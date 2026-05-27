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
| P1 | Colab mode parity/status inventory | `/colab/modes` + Mode Atlas | Keep as triage map | API test, frontend build |
| P2 | Memory atlas and retrieval | `LatentMemoryIndex` + `memory.query` | Promote first slice | nearest-neighbor runtime test, frontend smoke |
| P2 | Residual steering and prompt search | existing scripts/experiments | Defer | model-backed recipe tests |

## Acceptance Tests Per Pass

- A capability must have a typed request/recipe.
- A run must produce or update a persistent artifact/job record.
- The response must expose whether the operation is implemented, queued, or
  explicitly not available on the current machine.
- Any visual control in the future UI must map to one of these contracts or be
  marked as speculative.

## Immediate Next Queue

1. Add result-family view for sweeps and multi-output script jobs.
2. Add memory result inspectors and preview/reuse flows for query bundles.
3. Add prompt-search recipe adapters for Colab Modes 2/3/5.
4. Add geometry/control-head recipe adapters for Colab Modes 12/15.
5. Add parameter presets and recipe diffing for Operator Studio.
6. Promote MLX generation from subprocess-only to a resident worker when repeated
   generation needs lower overhead.
7. Add deeper visual lineage for multi-step artifact families once recipe jobs
   are populated enough to make the routing real.
