# Codebase Review Notes

This is a lightweight review pass over the current local-app migration. It is
not a security audit; it focuses on architecture, runability, model defaults,
and the next risks for turning the Colab experiments into a proper app.

## What Is Working

- The repo has a clear split between research primitives
  (`latent_audio_primitives/`), app/runtime code (`sa3_native_lab/app/`),
  scripts (`scripts/`), notebook material (`colab/`), and the React frontend
  (`frontend/`).
- The API has typed Pydantic request/record models for artifacts, recipes, jobs,
  backends, operators, generation, encode/decode, and experiments.
- Jobs are durable: recipe, progress, logs, metrics, status, output artifact IDs,
  and errors are persisted.
- Artifacts carry useful metadata: kind, file info, audio metadata, latent
  shape/rate, source artifact IDs, recipe ID, tags, and freeform metadata.
- The frontend is no longer just a generic dashboard. It has a listening bench,
  Operator Studio, Recipe Studio, Mode Atlas, job rail, artifact rail, waveform
  peaks, region-loop playback, A/B slots, result-family inspection,
  memory-hit reuse, archive-and-new sessions, and alpha-sweep promotion plus
  recipe fork diff/reset controls.
- A first TypeScript tRPC control plane exists in `apps/control-plane`; it owns
  app-shaped workbench reads plus job lifecycle, recipe, artifact, family, and
  archive procedures plus durable-journal-aware job-event subscriptions while
  Python remains the model/runtime worker.
- Audio playback, artifact display, job progress, result families, recipe fork
  editing, and bundle inspection are now split out of `App.tsx`.
- Medium/SAME-L is now the default path across app contracts, runtime fallbacks,
  frontend defaults, README commands, docs, and tests.
- Backend operator specs now include `ui_fields`, and the frontend merges those
  fields into Operator Studio and Recipe Studio so bounds, defaults, options,
  required fields, artifact-kind hints, and backend choices are visible.
- Job progress cards now classify running phases and common failures instead
  of leaving the user with only a raw status.
- Bundle inspection promotes JSON/NPZ summaries, metric scalars, and plot/image
  discovery into typed reader rows. Reusable bundles can now populate Recipe
  Studio fields directly. Inline image plots now render from bundle artifacts,
  and alpha sweep families have sortable metrics, best-candidate marking, and
  A/B promotion from the table. Embedded bundle audio can be promoted into a
  first-class audio artifact, and prompt-search bundles are parsed into native
  reader rows and prompt reuse actions.

## Important Risks

- Frontend field schemas are still partly duplicated in TypeScript. The new
  backend `ui_fields` merge reduces drift, but the static catalogs still carry
  layout, copy, and some hand-shaped controls.
- Live job event transport now has a tRPC/SSE bridge with heartbeat and
  resume-aware IDs plus Python job-journal replay. The live source is still
  polling, so a future stream source would reduce latency and load.
- `RuntimeDispatcher` owns many responsibilities: backend status, MLX
  generation, SAME encode/decode, latent operators, script adapters, subprocess
  wrapping, and output finalization. It works, but it will become harder to test
  as more Colab modes become native.
- Bundle artifacts have backend-parsed JSON/NPZ summaries, metric rows, plot
  file discovery, inline image previews, typed UI readers, and recipe-input
  actions, but vector/profile/prompt-search/soft-prompt-specific inspectors are
  still shallow.
- `experiment.prompt_search` is useful as an app contract and workflow probe,
  but it currently uses a deterministic `lexical_probe` scorer. It should not be
  presented as true audio-text inversion until the SA3/CLAP model-backed scorer
  is implemented and validated.
- Long-running jobs have cancel/retry, but not pause/resume, priority,
  resource-aware scheduling, or resident worker reuse.
- Error messages are now transformed into first-pass recovery hints, but command
  context and safe stderr-tail preservation need more work.
- The frontend build is clean, and payload/form helper tests now exist, but
  component-level tests should expand as fields become more schema-driven.

## Suggested Refactors

1. Continue migrating field schemas toward backend `ui_fields`.
   Keep canonical operator/experiment parameter truth in Python while preserving
   the custom instrument layout in React.

2. Harden job events in tRPC.
   Job lifecycle, recipe, archive, family procedures, and a heartbeat/resume
   event bridge exist. Durable replay now exists; the next gap is replacing
   live polling with a stream source and preserving richer safe stderr context.

3. Split runtime adapters.
   Move MLX, SAME, latent operators, and script recipes into modules such as
   `runtime_mlx.py`, `runtime_same.py`, `runtime_latent.py`, and
   `runtime_scripts.py`.

4. Add payload-building tests for the frontend.
   Test default forms, donor modes, advanced params, number-list parsing, and
   Recipe Studio payloads.

5. Add artifact inspectors.
   Extend current audio/latent/bundle vitals and inline plot previews into
   domain-specific vector, profile, prompt-search, soft-prompt, training, sweep,
   and memory inspectors.

6. Add deeper job control.
   Later add pause/resume, priorities, worker pools, and resident model
   workers.

7. Harden docs around machine profiles.
   Document what runs on M1/MPS/MLX, what is realistic on CPU, and what remains
   CUDA/Colab-oriented.

## Verification Used In This Pass

- Searched model/default usage across app, runtime, frontend, tests, README,
  and MLX scripts.
- Added a default-model test for `medium`/`same-l`.
- Ran the Python test suite.
- Ran the frontend production build.
- Ran the control-plane contract tests.
- Checked whitespace with `git diff --check`.
