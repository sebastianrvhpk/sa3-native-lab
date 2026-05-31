# Codebase Review Notes

This is a lightweight review pass over the current local-app migration. It is
not a security audit; it focuses on architecture, runability, model defaults,
and the next risks for turning the Colab experiments into a playable local
sound instrument.

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
- The frontend has completed the product-loop rescue described in
  `docs/product-rescue-brief.md`: Current Sound, Gesture, Pending Take, Listen,
  Branch / Remember / Tune. It has a gesture strip, scoped Tune drawer, Next
  actions, pending/failed take cards, active Memory reuse, branch-language UI,
  waveform peaks, loop-region playback, marker notes, archive/recovery, and
  branch editing plus recipe diff/reset controls.
- A first TypeScript tRPC control plane exists in `apps/control-plane`; it owns
  app-shaped workbench reads plus job lifecycle, recipe, artifact, family, and
  archive procedures plus durable-journal-aware job-event subscriptions while
  Python remains the model/runtime worker.
- Audio playback, artifact display, job progress, branch/result-family views,
  recipe fork editing, bundle inspection, memory actions, next actions, pending
  landing, and Tune grouping are now split out of `App.tsx`.
- Medium/SAME-L is now the default path across app contracts, runtime fallbacks,
  frontend defaults, README commands, docs, and tests.
- Backend operator specs now include `ui_fields`, and the frontend merges those
  fields into latent gesture Tune and Advanced Gestures/Tune so bounds, defaults, options,
  required fields, artifact-kind hints, and backend choices are visible.
- Job progress cards now classify running phases and common failures instead
  of leaving the user with only a raw status.
- Bundle inspection promotes JSON/NPZ summaries, metric scalars, and plot/image
  discovery into typed reader rows. Reusable bundles can now populate existing
  Advanced Gesture fields directly. Inline image plots now render from bundle artifacts,
  and alpha sweep families have sortable metrics and branch highlighting.
  Embedded bundle audio can be promoted into a
  first-class audio artifact, and prompt-search bundles are parsed into native
  probe rows, candidate-family previews, generated-take decision summaries,
  prompt memory, workflow-signal chips, and prompt reuse actions.
- Latent gestures now have browser-local presets for repeatable latent operator
  explorations, plus selected-preset diffs for changed params and donor latents.
  This is useful immediately even before presets become durable backend records.
- The first-use browser smoke is committed as the product-health gate for
  Current Sound, Gestures, Make, Tune, Pending Takes, Next actions, Remember,
  Branch, Memory reuse/recovery, Settings/Inspect demotion, screenshots, and
  mobile overflow.

## Important Risks

- Frontend field schemas are still partly duplicated in TypeScript. The new
  backend `ui_fields` merge reduces drift, but the static catalogs still carry
  layout, copy, and some hand-shaped controls.
- `useGestureWorkbench` now owns only semantic workbench state: active gesture,
  Tune forms, donor/source reuse, next-action routing, prompt seeding, and
  bundle reuse. `App.tsx` remains the side-effect boundary for React Query
  mutations, job-event landing, archive/recover, pending-take selection, and
  submit actions. A later gesture action descriptor may help with labels,
  readiness, and disabled reasons without moving mutations into the hook.
- Live job event transport now has a tRPC/SSE bridge with heartbeat and
  resume-aware IDs plus Python job-journal replay. The live source is still
  polling, so a future stream source would reduce latency and load.
- `RuntimeDispatcher` owns many responsibilities: backend status, MLX
  generation, SAME encode/decode, latent operators, script adapters, subprocess
  wrapping, and output finalization. It works, but it will become harder to test
  as more Colab modes become native.
- Bundle artifacts have backend-parsed JSON/NPZ summaries, metric rows, plot
  file discovery, inline image previews, typed UI readers, and recipe-input
  actions plus first-pass workflow signals, but vector/profile/prompt-search/
  soft-prompt-specific inspectors are still shallow.
- `experiment.prompt_search` now has a deterministic `lexical_probe` fallback
  and an optional `sa3_flow_probe` prompt probe. It should not be presented as
  mature audio-text inversion until the real Medium/MPS path has listening
  validation, cost notes, and layer/alpha branch workflows. The candidate
  branch, decision-correlation, and prompt-memory slices exist.
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

4. Keep expanding product-model and payload tests.
   Current coverage includes memory action availability, next-action mapping,
   pending take landing, Tune field grouping, and focused panel tests. Next add
   branch-card action tests and more bundle-to-gesture reuse coverage.

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

## Recent Verification Baseline

- `npm run test --prefix frontend -- --run`
- `npm run build --prefix frontend`
- `npm run smoke:playback-session --prefix frontend`
- `npm run smoke:first-use --prefix frontend`
- `uv run pytest`
- `git diff --check`
