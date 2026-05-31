# SA3 Native Lab Improvement Roadmap

This roadmap is a code-grounded triage queue for turning the current Colab
migration into a stronger local AI sound instrument.

The immediate product direction is governed by `docs/product-rescue-brief.md`.
The rescue pass has landed: the primary loop is now Current Sound -> Gesture ->
Pending Take -> Listen -> Branch / Remember / Tune. This roadmap now tracks
what should happen after that loop, not the old dashboard cleanup.

For the consolidated current-state, engineering-health, test-posture, and
definition-of-done document, see `docs/implementation-readiness.md`.

For the broader stack direction and promotion triggers, see
`docs/architecture-horizon.md`.

## P0: Product Loop Follow-Through

1. Keep the first-use smoke as the product-health gate.
   `npm run smoke:first-use --prefix frontend` now protects Current Sound,
   Gestures, Make, Tune, pending/failed take language, Next actions, Remember,
   Branch, Memory reuse as Source/Anchor, recovery, Settings/Inspect demotion,
   desktop/mobile screenshots, and mobile overflow.

2. Harden next-action semantics.
   The current `Next` model covers audio, latent, bundle, pending/failed take,
   and branch states. The next pass should add branch-card component tests and
   validate more bundle-to-gesture promotions as inspectors mature.

3. Expand Memory as a creative shelf.
   Active Memory reuse exists for Source, Anchor, donor, prompt seed, existing
   Advanced Gesture bundle paths, and recovery. Next, add role/reuse-intent
   filters and a fuller Memory browser before any vector retrieval work.

4. Improve the take strip into a listening queue.
   The app already has recent/lineage/open sequencing, playback decisions,
   markers, loop regions, Remember, Branch, and Continue paths. The next work is
   keep/maybe/reject review flow, continue-from-selected, and branch listening,
   not decorative timelines.

5. Keep gesture orchestration boundaries named.
   `useGestureWorkbench` is now intentionally bounded to semantic workbench
   state: active gesture, Tune form state, donor/source reuse, next-action
   routing, prompt seeding, and bundle reuse. `App.tsx` still owns React Query
   mutations, job-event landing, archive/recover, pending-take selection, and
   other side effects. If this grows again, the next extraction should be a
   small gesture action descriptor for labels, readiness, and disabled reasons,
   not a mutation-moving hook.

## P1: Trust And Runability

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
   latent gesture Tune and Advanced Gestures/Tune. Select options, whole-number controls,
   and alpha lists have first-pass validation. Next, make more complex controls
   fully schema-driven without losing the custom instrument layout.

## P2: Exploration Speed

1. Playback composer.
   The result rail now has a first recent-audition stack for session audio:
   recent takes can be selected, played compactly, inspected with their
   listening decision badge, and remembered or reused. The main
   specimen player now persists markers and loop regions as artifact
   annotations, supports marker notes, uses WaveSurfer for zoom plus draggable
   loop edits, and has committed browser coverage for cue
   persistence, annotation persistence, and SessionTray artifact
   archive/recovery. Next playback work should add explicit playlist/session
   sequencing.

2. Deepen branch views for sweeps.
   Recipe families now appear as branches with a detail panel, per-result playback,
   replay, fork actions, and a compact sweep metric
   table with sort controls, branch highlighting, and durable
   keeper/maybe/reject listening decisions on playable artifacts. Sibling
   alpha-sweep families are now compared in the family detail panel when they
   share a vector bundle or prompt. Session/archive filters now surface saved
   decisions by decision, kind, model, gesture, branch, tag, text, and source
   lineage. Prompt-search takes now expose first-pass target-vs-take descriptor
   deltas plus a decision-study summary over keeper/maybe/reject annotations;
   prompt memory now groups generated prompt-candidate decisions across runs.
   Prompt-search bundles also summarize generated takes across search runs by
   method, mode, model, duration, prompt variety, and listening decisions. Next
   they need richer sweep-family and layer/alpha branch views. Advanced Gestures/Tune now
   also has prompt-search presets for Mode 2 hard-token search, Mode 3 readable
   prompt search, and a small Medium flow-score check, with active cost
   guidance visible before the heavier probe is selected. It also has
   vocabulary-set buttons, readable modifier-axis sets, and prompt-history reuse
   from previous generated takes, so Mode 2/3 prompt exploration is no longer
   only a raw parameter form.

3. Latent presets.
   Browser-local named presets now exist for blur, DSP, graft, renoise, and
   cyclic roll. They can be saved, updated, selected, and deleted per operator
   mode. Selected presets show parameter and donor-latent diffs with a revert
   action. Next, promote useful presets into backend recipe history before
   adding a heavier database-backed preset library.

4. Artifact filtering, memory roles, and tagging.
   Labels, tags, notes, keeper/maybe/reject listening decisions, archive search,
   archive-and-new session cleanup, and typed local recovery filters are now
   implemented. The filter pass covers decision, kind, model, gesture, branch,
   source lineage, text, and tag. Memory annotations now include role, reuse
   intent, and decision metadata. Next recovery work should add role/reuse
   filters before date or job-status filters.

5. Better bundle readers.
   Bundle file inventory, JSON previews, memory-result reuse, and first-pass
   typed readers now exist. The backend now parses JSON/NPZ bundle summaries
   and promotes metric scalars plus plot/image files into reader rows. Bundle
   cards can now route profiles, vectors, directions, soft prompts, memory
   folders, prompt-search candidates, and LoRA checkpoints into Advanced Gestures/Tune,
   and discovered image plots plus embedded audio files render through the
   bundle-file endpoint. Embedded bundle audio can now be promoted into normal
   audio artifacts. Prompt-search bundles now show probe metadata, compact
   candidate families, inline generated takes, target/take descriptor deltas,
   and saved listening decisions with notes. Bundle panels now show workflow
   signals for recipe actions, playable
   audio, lineage, plots, metrics, candidates, memory hits, variants, tensors,
   checkpoints, and geometry stats. Next, vector/profile/soft-prompt/sweep/
   geometry/prompt-search bundles should expose deeper domain-specific controls
   without making the user inspect zip contents.

## P3: Research Cognition

1. Memory browser and query surface.
   Local latent-artifact nearest-neighbor query now exists as `memory.query`.
   Bundle previews now allow selecting hits, playback for audio hits, and
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
   checks over Medium flow losses. The UI/spec only expose implemented probes.
   A tiny authenticated Medium/MPS smoke run succeeded locally on short target
   audio. Prompt candidates can now launch MLX text-to-audio jobs with lineage
   back to the search bundle; generated takes are grouped under the
   prompt-search branch and can be played, marked keeper/maybe/rejected with
   notes, and described through lightweight target/take descriptor deltas. They can also be
   summarized through a decision-study panel, grouped into prompt memory across
   runs, and recovered later through decision/model/gesture/branch filters. The
   next research step is richer layer/alpha comparison and runtime-cost notes.

5. Geometry and control probes.
   `experiment.geometry_audit` now produces a local latent geometry report
   bundle from saved SAME latents. The next research step is labelled
   observability/control probes rather than a decorative node graph.

## P4: Interface Polish

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
   Use motion only where it clarifies state changes: pending take created,
   take landed, sound remembered, branch opened.

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
   branch/result-family views, recipe forks, and bundle inspection are split out. Next
   extraction targets are the generation/SAME/operator/recipe action bands, but
   only after their state and handler contracts can be named cleanly. Storybook
   and MSW scenarios are now justified for the extracted panels.

## Verification Plan

For each pass:

- Run `uv run pytest`.
- Run `npm run build --prefix frontend`.
- Run `npm run test --prefix frontend -- --run`.
- Run `npm run test --prefix apps/control-plane`.
- Run `npm run smoke:first-use --prefix frontend` for product-loop changes.
- Run `npm run smoke:playback-session --prefix frontend` for listening/session
  changes.
- Run `git diff --check`.
- Smoke the local app at `http://127.0.0.1:5173/`, preferably with
  `uv run sa3-lab dev --with-control-plane` when changing app-shaped reads.
- Confirm every new visible control maps to a backend parameter, artifact, or
  documented future capability.
