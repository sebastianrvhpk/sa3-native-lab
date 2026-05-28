# SA3 Native Lab Improvement Roadmap

This roadmap is a code-grounded triage queue for turning the current Colab
migration into a stronger local research app.

For the broader stack direction and promotion triggers, see
`docs/architecture-horizon.md`.

## P0: Trust And Runability

1. Add typed job-event streaming.
   Cancellation and retry now exist. The next trust step is a typed stream for
   progress, stderr tails, cancellation acknowledgement, and reconnect/resume.

2. Improve error surfacing.
   Job failures should show command, stderr tail, missing file paths, model
   auth hints, and suggested next action instead of only a failed status.

3. Add explicit environment readiness checks.
   The API should report Hugging Face auth, Medium MLX weights, SAME-L weights,
   PyTorch MPS availability, and missing optional extras as actionable checks.

4. Add fork-with-changes forms.
   Recipe replay/fork endpoints exist. The UI still needs "copy params from
   artifact", edit them inline, and submit the changed recipe as a new branch.

## P1: Exploration Speed

1. Deepen result-family views for sweeps.
   Recipe families now appear as grouped records. Next they need per-result
   playback, A/B promotion, metrics, and recipe deltas.

2. Presets for Operator Studio.
   Store named parameter sets for blur, DSP, graft, renoise, and cyclic roll.
   This would make explorations reproducible without turning everything into
   manual form entry.

3. Artifact filtering and tagging.
   Labels, tags, notes, and archive search are now implemented. Next filters
   should cover kind, model, operator, date, status, and source lineage.

4. Better bundle readers.
   Bundle file inventory now exists. Vector/profile/soft-prompt bundles should
   also expose metadata, dimensions, plots, generated audio children, and reuse
   actions without making the user inspect zip contents.

## P2: Research Cognition

1. Memory browser and query surface.
   Local latent-artifact nearest-neighbor query now exists as `memory.query`.
   The next step is a richer browser for encoded SAME datasets, preview audio,
   tags, and reuse as donors or style references.

2. Latent channel and time-region views.
   Graft/renoise masks would be more intuitive as channel/time selections,
   heatmaps, or lane views instead of only scalar forms.

3. Recipe graph and lineage map.
   Routing lines should eventually become an inspectable graph of source audio,
   latent, donor, operator, bundle, and decoded result relationships.

4. Prompt/residual comparison bench.
   Residual steering and prompt-search runs need descriptor deltas, layer/alpha
   comparisons, and generated audio grids.

## P3: Interface Polish

1. Responsive visual QA.
   Keep checking desktop and mobile screenshots for text collision, oversized
   controls, and panels that become too dense.

2. Motion for causality.
   Use motion only where it clarifies state changes: job queued to running,
   artifact produced, lineage forked, A/B promoted.

3. Stronger reference language.
   The paper, gradient-cell, and routing visual language should continue to map
   to real controls, provenance, or artifact families. Avoid adding decorative
   graph lines that imply nonexistent routing.

## Architecture Improvements

1. Move more app-shaped actions into tRPC.
   `workbench.load`, job lifecycle, recipe replay/fork, artifact inspection, and
   family reads now exist. Next should be live job events, richer fork forms,
   and archive mutations as the normal UI path. Keep the Python worker as the
   model/runtime owner.

2. Generate frontend schemas from backend operator specs.
   Operator/experiment field catalogs currently live in the frontend. A future
   typed schema endpoint would reduce drift between Python contracts and UI
   controls.

3. Keep Zod/TanStack Form validation converging.
   The first typed form foundation exists. The next step is deriving more form
   bounds and required-ness from backend operator specs to reduce drift.

4. Add persistent worker processes.
   Repeated Medium generation would benefit from a resident MLX/PyTorch worker
   instead of launching every heavy job as a fresh subprocess.

5. Split runtime adapters by capability.
   `RuntimeDispatcher` is becoming a hub for many concerns. Moving MLX, SAME,
   latent operators, and script recipes into separate adapter modules would make
   tests and future workers cleaner.

6. Add typed artifact inspectors.
   Each artifact kind should have a dedicated inspector component and backend
   reader: audio, latent, vector bundle, profile, soft prompt, training output,
   and memory collection.

## Verification Plan

For each pass:

- Run `uv run pytest`.
- Run `npm run build --prefix frontend`.
- Run `npm run test --prefix apps/control-plane`.
- Run `git diff --check`.
- Smoke the local app at `http://127.0.0.1:5173/`, preferably with
  `uv run sa3-lab dev --with-control-plane` when changing app-shaped reads.
- Confirm every new visible control maps to a backend parameter, artifact, or
  documented future capability.
