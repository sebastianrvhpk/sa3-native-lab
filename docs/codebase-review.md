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
  peaks, playback, and A/B slots.
- Medium/SAME-L is now the default path across app contracts, runtime fallbacks,
  frontend defaults, README commands, docs, and tests.

## Important Risks

- Frontend field schemas are still duplicated in TypeScript instead of coming
  from backend operator/experiment specs. This is the biggest drift risk.
- `RuntimeDispatcher` owns many responsibilities: backend status, MLX
  generation, SAME encode/decode, latent operators, script adapters, subprocess
  wrapping, and output finalization. It works, but it will become harder to test
  as more Colab modes become native.
- Bundle artifacts are too opaque. The app knows they exist but does not yet
  inspect vector/profile/soft-prompt contents in a native way.
- Long-running jobs do not yet have cancellation, retry, pause/resume, or
  resource-aware scheduling.
- Error messages are preserved in job records but not yet transformed into
  user-friendly recovery guidance.
- The frontend build is clean, but there are no frontend unit tests for form to
  API payload conversion. This matters now that Operator Studio is schema-like.

## Suggested Refactors

1. Add a backend schema endpoint for UI fields.
   Keep the canonical operator/experiment field definitions in Python, then
   generate or fetch frontend controls from those specs.

2. Split runtime adapters.
   Move MLX, SAME, latent operators, and script recipes into modules such as
   `runtime_mlx.py`, `runtime_same.py`, `runtime_latent.py`, and
   `runtime_scripts.py`.

3. Add payload-building tests for the frontend.
   Test default forms, donor modes, advanced params, number-list parsing, and
   Recipe Studio payloads.

4. Add artifact inspectors.
   Give bundle types readers and UI panels instead of treating every bundle as
   a zip file.

5. Add job control endpoints.
   Start with cancel/retry. Later add priorities, worker pools, and resident
   model workers.

6. Harden docs around machine profiles.
   Document what runs on M1/MPS/MLX, what is realistic on CPU, and what remains
   CUDA/Colab-oriented.

## Verification Used In This Pass

- Searched model/default usage across app, runtime, frontend, tests, README,
  and MLX scripts.
- Added a default-model test for `medium`/`same-l`.
- Ran the Python test suite.
- Ran the frontend production build.
- Checked whitespace with `git diff --check`.
