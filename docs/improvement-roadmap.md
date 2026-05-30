# SA3 Native Lab Improvement Roadmap

This roadmap is a code-grounded triage queue for turning the current Colab
migration into a stronger local research app.

For the consolidated current-state, engineering-health, test-posture, and
definition-of-done document, see `docs/implementation-readiness.md`.

For the broader stack direction and promotion triggers, see
`docs/architecture-horizon.md`.

## P0: Trust And Runability

1. Harden live job-event transport.
   Typed job events now reach the UI through a tRPC/SSE control-plane bridge
   when the control plane is enabled. The bridge now emits monotonic IDs,
   resume-aware sequencing, heartbeat events, log-tail diagnostics, and
   durable Python job-journal replay. Job cards now also derive readable phases
   from current event text. The next trust step is richer command and
   stderr-tail context plus eventually replacing live polling with a stream
   source.

2. Improve error surfacing.
   Job failures now classify common failures into recovery hints for Hugging
   Face auth, missing MLX setup, path/output problems, subprocess exits, and
   memory pressure. Medium torch prompt probes also preflight Hugging Face cache
   space before the heavy checkpoint download. Next, preserve richer stderr
   tails and command context without exposing sensitive tokens.

3. Deepen environment readiness checks.
   `/readiness` now reports artifact root, HF auth, Medium MLX weights, SAME-L
   access, Hugging Face cache space, and backend availability. Next checks
   should cover exact HF license acceptance, active model cache paths, and
   optional extras.

4. Tighten fork-with-changes forms.
   The UI can fork recipe params, backend, model, seed, and notes with visible
   deltas plus reset controls. Backend `ui_fields` now provide defaults,
   bounds, options, required flags, artifact-kind hints, and advanced flags for
   Operator Studio and Recipe Studio. Select options, whole-number controls,
   and alpha lists have first-pass validation. Next, make more complex controls
   fully schema-driven without losing the custom instrument layout.

## P1: Exploration Speed

1. Playback composer.
   The result rail now has a first recent-audition stack for session audio:
   recent takes can be selected, played compactly, inspected with their
   listening decision badge, and routed directly into A/B slots. The main
   specimen player now persists markers and loop regions as artifact
   annotations, supports marker notes, uses WaveSurfer for zoom plus draggable
   loop edits, and has committed browser coverage for A/B assignment, cue
   persistence, annotation persistence, and SessionTray artifact
   archive/recovery. Next playback work should add explicit playlist/session
   sequencing.

2. Deepen result-family views for sweeps.
   Recipe families now appear with metrics, a detail panel, per-result playback,
   explicit A/B promotion, replay, fork actions, and a compact sweep metric
   table with sort controls, best-candidate marking, and durable
   keeper/maybe/reject listening decisions on playable artifacts. Sibling
   alpha-sweep families are now compared in the family detail panel when they
   share a vector bundle or prompt. Session/archive filters now surface saved
   decisions by decision, kind, model, operator, family, tag, text, and source
   lineage. Prompt-search takes now expose first-pass target-vs-take descriptor
   deltas plus a decision-study summary over keeper/maybe/reject annotations;
   prompt memory now groups generated prompt-candidate decisions across runs.
   Prompt-search bundles also compare generated takes across search runs by
   scorer, mode, model, duration, prompt variety, and listening decisions. Next
   they need richer sweep-family and layer/alpha comparisons. Recipe Studio now
   also has prompt-search presets for Mode 2 hard-token search, Mode 3 readable
   prompt search, and a small Medium flow-score check, with active scorer-cost
   guidance visible before the heavier scorer is selected. It also has
   vocabulary-set buttons, readable modifier-axis sets, and prompt-history reuse
   from previous generated takes, so Mode 2/3 prompt exploration is no longer
   only a raw parameter form.

3. Presets for Operator Studio.
   Browser-local named presets now exist for blur, DSP, graft, renoise, and
   cyclic roll. They can be saved, updated, selected, and deleted per operator
   mode. Selected presets show parameter and donor-latent diffs with a revert
   action. Next, promote useful presets into backend recipe history before
   adding a heavier database-backed preset library.

4. Artifact filtering and tagging.
   Labels, tags, notes, keeper/maybe/reject listening decisions, archive search,
   archive-and-new session cleanup, and typed local recovery filters are now
   implemented. The filter pass covers decision, kind, model, operator, result
   family, source lineage, text, and tag. Next recovery work should add date and
   job status filters only if actual archive volume makes them useful.

5. Better bundle readers.
   Bundle file inventory, JSON previews, memory-result reuse, and first-pass
   typed readers now exist. The backend now parses JSON/NPZ bundle summaries
   and promotes metric scalars plus plot/image files into reader rows. Bundle
   cards can now route profiles, vectors, directions, soft prompts, memory
   folders, prompt-search candidates, and LoRA checkpoints into Recipe Studio,
   and discovered image plots plus embedded audio files render through the
   bundle-file endpoint. Embedded bundle audio can now be promoted into normal
   audio artifacts. Prompt-search bundles now show scorer metadata, compact
   candidate families, inline generated takes, A/B assignment for candidate
   audio, target/take descriptor deltas, and saved listening decisions with
   notes. Bundle panels now show workflow signals for recipe actions, playable
   audio, lineage, plots, metrics, candidates, memory hits, variants, tensors,
   checkpoints, and geometry stats. Next, vector/profile/soft-prompt/sweep/
   geometry/prompt-search bundles should expose deeper domain-specific controls
   without making the user inspect zip contents.

## P2: Research Cognition

1. Memory browser and query surface.
   Local latent-artifact nearest-neighbor query now exists as `memory.query`.
   Bundle previews now allow selecting hits, A/B assignment for audio hits, and
   donor reuse for latent hits. The next step is a richer browser for encoded
   SAME datasets, preview audio, tags, and style-reference promotion.

2. Latent channel and time-region views.
   Graft/renoise masks would be more intuitive as channel/time selections,
   heatmaps, or lane views instead of only scalar forms.

3. Recipe graph and lineage map.
   Routing lines should eventually become an inspectable graph of source audio,
   latent, donor, operator, bundle, and decoded result relationships.

4. Prompt/residual comparison bench.
   Prompt search now exists as `experiment.prompt_search` with beam, greedy, and
   coordinate modes, `lexical_probe` fallback, and optional `sa3_flow_probe`
   scoring over Medium flow losses. The UI/spec only expose implemented
   scorers; CLAP remains future work. A tiny authenticated Medium/MPS smoke run
   succeeded locally on short target audio. Prompt candidates can now launch
   MLX text-to-audio jobs with lineage back to the search bundle; generated
   takes are grouped under the prompt-search family and can be played, sent to
   A/B beside the candidate, marked keeper/maybe/rejected with notes, and
   compared to the target through lightweight descriptor deltas. They can also be
   summarized through a decision-study panel, grouped into prompt memory across
   runs, and recovered later through decision/model/operator/family filters. The
   next research step is richer layer/alpha comparison and runtime-cost notes.
   CLAP or hybrid scoring belongs after that comparison workflow has real usage.

5. Geometry and control probes.
   `experiment.geometry_audit` now produces a local latent geometry report
   bundle from saved SAME latents. The next research step is labelled
   observability/control probes rather than a decorative node graph.

## P3: Interface Polish

0. Visual grammar.
   The first reference-vibe pass now has a canonical grammar in
   `docs/visual-design-grammar.md`. The app uses paper texture, watercolor
   gradients, pencil-line modules, gradient control cells, richer transport
   wheels, and subtle workbench flow fields. Future visual passes should keep
   lines/nodes tied to real lineage, provenance, memory, or playback state.

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
   family reads now exist, and `jobs.events` bridges live snapshots over
   tRPC/SSE with heartbeat/resume diagnostics. Next should be durable event
   history and archive mutations as the normal UI path. Keep the Python worker
   as the model/runtime owner.

2. Generate frontend schemas from backend operator specs.
   Backend operator specs now emit `ui_fields`, and the frontend merges them
   into the existing hand-shaped catalogs. The next step is reducing static
   catalog duplication while preserving instrument-specific controls and copy.

3. Keep Zod/TanStack Form validation converging.
   The first typed form foundation exists, and backend bounds/options now feed
   frontend forms. Select membership, integer controls, and alpha list parsing
   are validated. The next step is richer cross-field constraints such as donor
   requirements, source artifact kind, and mode-specific path compatibility.
   Operator presets are still browser-local; persisted preset contracts should
   wait until validation metadata and backend history semantics are stable
   enough to save.

4. Add persistent worker processes.
   Repeated Medium generation would benefit from a resident MLX/PyTorch worker
   instead of launching every heavy job as a fresh subprocess.

5. Split runtime adapters by capability.
   `RuntimeDispatcher` is becoming a hub for many concerns. Moving MLX, SAME,
   latent operators, and script recipes into separate adapter modules would make
   tests and future workers cleaner.

6. Add typed artifact inspectors.
   Bundle summaries now parse JSON/NPZ metadata, metric scalars, and plot/image
   file discovery in the backend, and reusable bundles can populate Recipe
   Studio fields. Bundle-contained audio children, prompt-search reports, and
   geometry reports are now also surfaced, and bundle audio can be promoted into
   first-class audio artifacts. The specimen panel shows kind-specific vitals for
   audio, latent, and bundle artifacts. Next each bundle kind should grow a
   dedicated inspector component with richer actions beyond the new workflow
   signals: vector bundle, profile, prompt search, soft prompt, training output,
   sweep, memory collection, and geometry audit.

7. Continue extraction only where the contract is named.
   `App.tsx` is now a composition root: configs, workbench model helpers,
   specimen, session tray, compare, mode atlas, audition stack, prompt-search
   rack, operator presets, readiness/status panels, spec coverage, playback,
   result families, recipe forks, and bundle inspection are split out. Next
   extraction targets are the generation/SAME/operator/recipe action bands, but
   only after their state and handler contracts can be named cleanly. Storybook
   and MSW scenarios are now justified for the extracted panels.

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
