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
| P1 | Playback/take surface | `frontend/` listening bench | Keep first slice, reframe through product rescue | Browser smoke, API operator job |
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
| P1 | Job phase and recovery feedback | `JobProgress` phase classifier and failure classifier over errors/logs | Promote | frontend unit tests |
| P1 | Runtime readiness checks | `/readiness` + `system.readiness` + readiness panel | Promote | API tests, control-plane tests, frontend tests |
| P1 | Fork-with-edited-params UI | recipe-derived fork editor with diff/reset controls | Promote | frontend tests/build |
| P1 | Bundle inspection and result families | `/artifacts/{id}/inspect` + `families.load` + result rail/detail panel | Promote | API tests, control-plane tests, frontend tests |
| P1 | Typed bundle readers | backend `bundle_summary` parser + `BundleField` readers for memory/profile/vector/sweep/soft-prompt/training outputs, metrics, plot discovery, inline plot rendering, and recipe-input reuse | Promote | API tests, frontend tests/build |
| P1 | Alpha-sweep branch surface | `FamilyDetailPanel` alpha variant band with sortable metric table, branch highlighting, and sibling sweep comparison | Promote, rename in product UI | frontend tests/build |
| P1 | Prompt-candidate take bench | Prompt-search bundle reader + text generation lineage metadata + grouped prompt-candidate branches, inline playback, and durable keeper/maybe/reject listening decisions | Promote, rename in product UI | frontend tests/build, control-plane family test |
| P1 | Prompt take descriptor deltas | `/artifacts/{target}/descriptor-comparison/{take}` + prompt-search generated-take delta strip | Promote first slice | storage/API tests, frontend tests/build |
| P1 | Prompt decision correlation | prompt-search generated-take descriptor rows + durable listening decisions | Promote first slice | frontend tests/build |
| P1 | Prompt decision memory | generated prompt-candidate takes grouped by prompt/decision across runs | Promote first slice | frontend tests/build |
| P1 | Operator Studio local presets and diffs | browser-local preset model + Operator Studio preset rack + selected-preset diff rows | Promote first slice | frontend tests/build |
| P1 | Bundle workflow signals | bundle summary signals for recipe actions, audio, lineage, plots, metrics, candidates, memory hits, tensors, checkpoints, and variants | Promote first slice | frontend tests/build |
| P1 | Session cleanup | archive-and-new session action plus searchable archive drawer | Promote | API/client tests, frontend build |
| P1 | Decision-aware artifact recovery | shared artifact filter model + session/archive filters for decision, kind, model, operator, family, source lineage, text, and tag | Promote | frontend filter tests/build, Playwright smoke |
| P1 | Kind-specific artifact vitals | specimen inspector rows for audio, latent, and bundle artifacts | Promote first slice | frontend build |
| P2 | Memory atlas and retrieval | `LatentMemoryIndex` + `memory.query` + memory-hit reuse actions | Promote first slice | nearest-neighbor runtime test, frontend smoke |
| P2 | SAME geometry audit | `geometry_report` + `experiment.geometry_audit` + bundle summary reader | Promote first slice | runtime test, frontend tests |
| P2 | Prompt search probe | `experiment.prompt_search` + `prompt_optimization` helpers + prompt-search bundle reader | Promote as probe | runtime test, frontend tests |
| P2 | SA3 flow prompt scoring | `latent_audio_primitives.flow_prompt` + `experiment.prompt_search.scorer=sa3_flow_probe` | Promote as explicit model-backed probe | flow-scorer primitive tests, runtime scorer-switch test, tiny authenticated Medium/MPS smoke |

## Acceptance Tests Per Pass

- A capability must have a typed request/recipe.
- A run must produce or update a persistent artifact/job record.
- The response must expose whether the operation is implemented, queued, or
  explicitly not available on the current machine.
- Any visual control in the future UI must map to one of these contracts or be
  marked as speculative.

## Immediate Next Queue

1. Add prompt-search sweep/layer comparisons where artifact families have enough
   examples, then document useful Medium/MPS score-sample/timestep settings.
2. Add deeper domain-specific controls for style profiles, vectors, prompt
   search, soft prompts, memory collections, sweeps, geometry audits, and
   training outputs beyond the first workflow-signal layer.
3. Promote the artifact recovery filters into tRPC/control-plane procedures
   when archive volume or remote persistence makes client-side filtering too
   heavy.
4. Add control-head recipe adapters for Mode 12 and labelled-probe extensions
   for Mode 15.
5. Continue shrinking frontend field drift until backend `ui_fields` can drive
   most controls without losing the instrument-specific layout.
7. Promote useful Operator Studio presets into backend or Postgres-backed recipe
   history.
8. Promote MLX generation from subprocess-only to a resident worker when repeated
   generation needs lower overhead.
9. Add deeper visual lineage for multi-step artifact families once recipe jobs
   are populated enough to make the routing real.
