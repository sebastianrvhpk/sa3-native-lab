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
   and branch states. The gesture action descriptor now covers primary button
   labels, readiness, disabled reasons, source requirements, and "what this
   will do" copy without moving mutations out of `App.tsx`. The next pass should
   add branch-card component tests and validate more bundle-to-gesture
   promotions as inspectors mature.

3. Expand Memory as a creative shelf.
   Active Memory reuse exists for Source, Anchor, donor, prompt seed, existing
   Advanced Gesture bundle paths, and recovery. The remembered-material browser
   now sorts usable material first and exposes role, reuse intent, tags, notes,
   kind, listening decision, branch, and source lineage from existing metadata.
   Next, deepen dataset/audio preview and style-reference promotion before any
   vector retrieval work.

4. Improve the take strip into a listening queue.
   The app already has recent/lineage/open sequencing, playback decisions,
   markers, loop regions, Remember, Branch, and Continue paths. A small clarity
   pass now shows keeper/maybe/reject/open summaries for the current queue and
   branch detail, and selected takes use the same visual label in the take strip
   and branch view. Playlist export, autoplay, multi-branch listening sessions,
   and heavy review modes remain deferred because they add new listening modes
   rather than clarifying the current loop.

5. Keep gesture orchestration boundaries named.
   `useGestureWorkbench` is now intentionally bounded to semantic workbench
   state: active gesture, Tune form state, donor/source reuse, next-action
   routing, prompt seeding, and bundle reuse. `App.tsx` still owns React Query
   mutations, job-event landing, archive/recover, pending-take selection, and
   other side effects. The descriptor layer is now the approved extraction for
   labels, readiness, disabled reasons, source requirements, and intent copy;
   it must remain pure and mutation-free.

6. Keep Tune slimmer than the backend.
   Tune now uses source-aware fields, product-language labels, exact submitted
   values, and latent-region summaries for channel-mask/global latent
   transforms. Continue improving field copy only where it maps to real
   backend-supported parameters.

## P1: Trust And Runability

1. Harden live job-event transport.
   Typed job events now reach the UI through a tRPC/SSE control-plane bridge
   when the control plane is enabled. The bridge now emits monotonic IDs,
   resume-aware sequencing, heartbeat events, log-tail diagnostics, and
   durable Python job-journal replay. Job cards now also derive readable phases
   from current event text. The next trust step is richer command and
   stderr-tail context is now kept behind job drawers and Pending Take Inspect.
   The next trust step is sanitizer coverage for sensitive command details plus
   eventually replacing live polling with a stream source.

2. Improve error surfacing.
   Job failures now classify common failures into recovery hints for Hugging
   Face auth, missing MLX setup, path/output problems, subprocess exits, and
   memory pressure. Medium torch prompt probes also preflight Hugging Face cache
   space before the heavy checkpoint download. Command context and log tails are
   now available behind details surfaces; keep adding trust detail there rather
   than to the primary listening bench.

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
   method, mode, model, duration, prompt variety, and listening decisions. The
   UI now frames this as candidate listening instead of a scorer dashboard.
   Next they need richer sweep-family and layer/alpha branch views. Advanced Gestures/Tune now
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
   action. Tune now also explains supported latent-region semantics: Graft and
   Renoise expose channel masks; Blur is a global latent-time smear; DSP/Reroute
   operates in latent time, not waveform EQ. Next, promote useful presets into
   backend recipe history only after those semantics stay stable in use.

4. Artifact filtering, memory roles, and tagging.
   Labels, tags, notes, keeper/maybe/reject listening decisions, archive search,
   archive-and-new session cleanup, and typed local recovery filters are now
   implemented. The filter pass covers decision, kind, model, gesture, branch,
   source lineage, text, tag, memory role, and reuse intent. Memory annotations
   now include role, reuse intent, and decision metadata. Date/job-status
   filters remain lower priority than making remembered material easier to use
   immediately.

5. Better bundle readers.
   Bundle file inventory, JSON previews, memory-result reuse, and first-pass
   typed readers now exist. The backend now parses JSON/NPZ bundle summaries
   and promotes metric scalars plus plot/image files into reader rows. Bundle
   cards can now route profiles, vectors, directions, soft prompts, memory
   folders, and prompt-search candidates into Advanced Gestures/Tune,
   and discovered image plots plus embedded audio files render through the
   bundle-file endpoint. Embedded bundle audio can now be promoted into normal
   audio artifacts. Prompt-search bundles now show probe metadata, compact
   candidate families, inline generated takes, target/take descriptor deltas,
   and saved listening decisions with notes. Bundle panels now show workflow
   signals for recipe actions, playable
   audio, lineage, plots, metrics, candidates, memory hits, variants, tensors,
   and geometry stats. Profile, vector, prompt-search, soft-prompt,
   dataset, and geometry domain cards now expose additional parsed evidence:
   source/reference, vector shape and source pair, prompt-search probe cost/risk,
   soft-prompt loss/steps/test audio, prompt coverage and caption/chunk warnings,
   and explicit geometry experimental framing. Next, sweep/layer comparison can
   deepen only where parsed bundle summaries and backend reuse paths support it.

## Explicitly Deferred

Do not add these until a later pass has a narrower contract: saved Memory filter
presets, vector search, pgvector, similarity browser, memory atlas UI, waveform
region workflows, playlist export, autoplay queue, multi-branch listening
sessions, heavy review modes, sampler-step intervention, flow-state
optimization, control heads, macro chain UI, resident worker, moving more app
state into tRPC, or moving gesture form state server-side. Fine-tuning belongs
in `dada-bots/underfit` on Colab A100, not in this local product loop.

## P3: Research Cognition

1. Memory browser and query surface.
   Local latent-artifact nearest-neighbor query now exists as `memory.query`.
   Bundle previews now allow selecting hits, playback for audio hits, and
   donor reuse for latent hits. The app-level Memory browser now covers stored
   role, reuse, notes, tags, decisions, kind, branch, and lineage metadata. The
   next research step is a richer browser for encoded SAME datasets, preview
   audio for child material, and style-reference promotion.

2. Latent channel and time-region views.
   Graft/renoise now expose real channel-mask semantics in Tune. True bounded
   time-region masks, heatmaps, and lane views remain deferred until the backend
   accepts those masks instead of only scalar/global latent controls.

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
   next research step is richer layer/alpha comparison and Medium/MPS runtime
   cost notes after the current branch/sweep listening comparison has enough
   real use.

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
   signals: vector bundle, profile, prompt search, soft prompt, sweep, memory
   collection, and geometry audit.

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
