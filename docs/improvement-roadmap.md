# SA3 Native Lab Improvement Roadmap

This roadmap is a code-grounded triage queue for turning the current Colab
migration into a stronger local research app.

## P0: Trust And Runability

1. Add job cancellation and retry.
   Long-running scripts and LoRA jobs need safe interruption, resumable logs,
   and an obvious retry path from the UI.

2. Add explicit environment readiness checks.
   The API should report Hugging Face auth, Medium MLX weights, SAME-L weights,
   PyTorch MPS availability, and missing optional extras as actionable checks.

3. Add recipe replay.
   Every artifact already records its recipe; the UI should expose "run again",
   "fork with changes", and "copy params from artifact".

4. Improve error surfacing.
   Job failures should show command, stderr tail, missing file paths, model
   auth hints, and suggested next action instead of only a failed status.

## P1: Exploration Speed

1. Result-family view for sweeps.
   Alpha sweeps and multi-output script jobs should appear as one grouped
   family with per-result playback, A/B promotion, metrics, and recipe deltas.

2. Presets for Operator Studio.
   Store named parameter sets for blur, DSP, graft, renoise, and cyclic roll.
   This would make explorations reproducible without turning everything into
   manual form entry.

3. Artifact filtering and tagging.
   Add filters for kind, model, prompt, operator, tag, date, status, and source
   lineage. The artifact rail will not scale as the lab grows.

4. Better bundle readers.
   Vector/profile/soft-prompt bundles should expose their metadata, dimensions,
   plots, generated audio children, and file inventory without making the user
   inspect zip contents.

## P2: Research Cognition

1. Memory browser and query surface.
   Encoded SAME datasets should become searchable memory collections with
   nearest-neighbor retrieval, preview audio, tags, and reuse as donors or style
   references.

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

1. Generate frontend schemas from backend operator specs.
   Operator/experiment field catalogs currently live in the frontend. A future
   typed schema endpoint would reduce drift between Python contracts and UI
   controls.

2. Add Zod or equivalent frontend validation.
   The UI should validate fields before submit using the same bounds and
   required-ness as the backend.

3. Add persistent worker processes.
   Repeated Medium generation would benefit from a resident MLX/PyTorch worker
   instead of launching every heavy job as a fresh subprocess.

4. Split runtime adapters by capability.
   `RuntimeDispatcher` is becoming a hub for many concerns. Moving MLX, SAME,
   latent operators, and script recipes into separate adapter modules would make
   tests and future workers cleaner.

5. Add typed artifact inspectors.
   Each artifact kind should have a dedicated inspector component and backend
   reader: audio, latent, vector bundle, profile, soft prompt, training output,
   and memory collection.

## Verification Plan

For each pass:

- Run `uv run pytest`.
- Run `npm run build --prefix frontend`.
- Run `git diff --check`.
- Smoke the local app at `http://127.0.0.1:5173/`.
- Confirm every new visible control maps to a backend parameter, artifact, or
  documented future capability.
