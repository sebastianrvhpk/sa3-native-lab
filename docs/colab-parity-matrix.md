# Colab Parity Matrix

This matrix is the implementation gate for migrating
`colab/sa3_same_native_experimental_modes.ipynb` into the local app. It is more
granular than `docs/colab-capability-map.md`: each row tracks the notebook
section or mode, the current app surface, missing controls, missing tests, and
the next action.

Evidence labels:

- `native`: first-class local API/UI behavior exists.
- `recipe`: reachable through Recipe Studio or `/experiments/run`, but still
  partly shaped like a script adapter.
- `partial`: usable pieces exist, but important notebook semantics are missing.
- `scaffold`: the repo names the path, but the app does not yet own the
  workflow.
- `deferred`: not a priority until earlier contracts are mature.

## Priority Rules

P0 work must protect runability and truth: parameters, provenance, progress,
artifact records, replay/fork, and tests.

P1 work makes iteration faster: session organization, playback, comparison,
bundle-specific inspectors, and decision memory.

P2 work expands research cognition: memory browsing, latent regions, lineage
graphs, probes, and prompt/residual comparisons.

P3 work is portfolio hardening and future architecture: Storybook, Postgres,
pgvector, resident workers, Temporal, or visual polish after the operational
meaning is real.

## Setup And Shared Infrastructure

| Notebook section | Current app surface | Status | Missing controls or risks | Missing tests | Next action |
| --- | --- | --- | --- | --- | --- |
| Colab L4 setup | `README.md`, `uv run sa3-lab dev`, `sa3-lab doctor`, MLX install notes | native | Exact HF license/cache-path diagnostics can be richer. | Readiness tests cover current checks, but not license acceptance nuance. | Keep improving `/readiness` before new heavy model paths. |
| Imports and paths | `.sa3_lab/` artifact root, app storage, sessions | native | Path-based script adapters still expose raw filesystem paths in some places. | Contract tests exist for storage; no path-compatibility matrix yet. | Add path compatibility checks per recipe family. |
| HF login and model loading | `HF_TOKEN`/HF cache, Medium/SAME-L defaults, readiness panel | native | Safe stderr and command-context preservation still needs hardening. | Token-leak regression tests are missing. | Add safe log-tail sanitizer tests before richer failure logs. |
| Medium smoke test | `/generate/text`, MLX wrapper, listening bench | native | Smoke is operational but not represented as a reusable demo fixture. | Smoke path exists outside normal unit tests. | Add a tiny deterministic demo fixture only when runtime cost is acceptable. |
| Shared helpers | API contracts, artifact store, descriptors, bundle inspection | native | Helpers are spread across runtime/adapters; `RuntimeDispatcher` remains broad. | Existing tests cover many helpers, but not all bundle-specific summaries. | Split adapters when the next mode becomes too hard to test in-place. |
| Custom Colab audio player | Listening bench, waveform peaks, region loop, A/B, decisions, recent audition stack, keyboardable playlist cursor, waveform markers, marker deletion, loop-edge nudging, artifact landing | partial | Local marker/loop editing exists; waveform zoom, draggable regions, marker notes, and persisted player annotations are still missing. | Audition stack, marker, loop-nudge, recovery, and job landing tests exist; full playback browser interaction tests are still thin. | Continue playback-composer pass with draggable regions and browser-level player tests. |
| Dataset and long-file policy | `dataset.pre_encode`, SAME encode/decode, memory query | partial | Long-file chunking rules are not consistently exposed across all dataset paths. | Dataset chunking/payload tests are partial. | Add dataset mode inspector and chunking contract rows. |

## Mode Matrix

| Mode | Notebook intent | Current API/UI surface | Status | Required / advanced parameters | Artifacts and provenance | Missing controls | Missing tests | Next action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Renoise variation / local neighborhood sampling | `generate.audio_to_audio`, Listening Bench | native | `source_artifact_id`, `prompt`, `duration_seconds`, `init_noise_level`, `steps`, `seed`, `cfg_scale`, `apg_scale`, `model`, `decoder`; rendered by spec-derived generation form | Audio artifact, recipe, source lineage, seed/model params, job phase, artifact landing | Better variation-family comparison and runtime notes | Payload builder and lineage tests exist; family comparison tests mostly cover sweeps, not this mode | Add audio-to-audio family comparison after session model cleanup |
| 0c | Latent-selective renoise | `latent.renoise`, Operator Studio | native | `source latent`, `mode`, `fraction`, `sigma`, `seed`, `backend` | Latent artifact, recipe, selected-mask metadata | Channel/time mask UI instead of scalar-only controls | Latent-region UI tests missing | P2 latent-region control pass |
| 0c-search | Annotation retrieval | Artifact annotation, archive filters | native | label, notes, tags, decision metadata, text/tag/model/operator/family filters | Updated artifact metadata and searchable archive | Rating-specific UI is not first-class | Filter tests exist, rating/search tests missing | Add only if ratings become active in workflow |
| 0e | Cross-audio latent channel graft | `latent.graft`, donor latent picker | native | `source latent`, `donor latent`, `mode`, `fraction`, `amount`, `seed`, `backend` | Latent artifact, source/donor lineage, recipe | Donor preview and channel/time region selection | Donor-region tests missing | P2 latent-region and donor audition pass |
| 0d | Latent blur playground | `latent.blur`, Operator Studio | native | `mode`, `strength`, backend, selected latent | Latent artifact, recipe | Visual preview of temporal/channel effect | Operator tests exist; perceptual/preview tests missing | Add typed inspector rows for latent transforms |
| 0h | Neural latent DSP | `latent.dsp`, Operator Studio | native | DSP mode, strength, optional donor, backend | Latent artifact, recipe, operator metadata | Mode-specific controls for FFT/PCA/channel variants | Payload tests cover helpers, not every DSP variant | Expand operator payload tests before richer UI |
| 0f | Cyclic time-roll loop lab | `latent.cyclic_roll`, MLX inpaint proxy | partial | `shift_frames`, `strength`, `symmetric`, source audio/latent, inpaint range | Latent/audio artifact, recipe | Iterative repair loop and loop-quality comparison | Loop-quality tests missing | Add loop family inspector after playback upgrade |
| 0g | Cyclic roll inside denoising trajectory | Cyclic latent roll proxy only | scaffold | roll fraction, per-step mix, sampler hooks | Planned sampler trace/audio artifact | True sampler-step intervention | No sampler-level tests | Defer until resident/sampler adapter exists |
| flow-sign | Flow sign diagnostic | Prompt-search scorer and docs mention velocity convention | scaffold | target audio, prompts, convention, timesteps/logSNR | Loss comparison JSON | Native diagnostic recipe and result panel | Diagnostic runtime tests missing | Add after prompt-search comparison path matures |
| 1 | Audio to soft prompt | `experiment.soft_prompt.optimize` | recipe | target audio, seed prompt, opt steps, duration, lr, reg, train keys, seed, velocity, model/backend | Soft-prompt bundle, recipe, logs | Better loss-curve and generated-test panel | Script adapter tests only partial | Add soft-prompt inspector with loss summary |
| 1b | Generate with soft prompt | `experiment.soft_prompt.generate` | recipe | soft prompt path, model, steps, CFG, seed, backend | Audio artifact, recipe, soft-prompt lineage | Batch seeds and compare stack | Batch generation tests missing | Add multi-seed UI after player/take stack |
| 2 | Audio to babble / hard prompt | `experiment.prompt_search` covers hard-token search | partial | target audio, vocabulary, search mode, scorer, tokens, beam, samples, seed, t/logSNR | Prompt-search bundle, prompt candidates, generated takes, run-comparison rows | Hard-token preset, scorer-cost guidance, vocabulary sets, and prompt history exist; vocabulary editing is still preset-based | Prompt-search helper, run-comparison, preset, scorer-note, vocabulary, and history tests exist | Add custom vocabulary save/share later |
| 3 | Audio to readable prompt | `experiment.prompt_search` with modifier axes | partial | target audio, modifier axes, search rounds, scorer, prefix/suffix | Prompt JSON bundle, generated takes, run-comparison rows | Readable-prompt preset, scorer-cost guidance, axis sets, and prompt history exist; axes editing is still preset-based | Preset/run-comparison/scorer-note/axis tests exist | Add custom axis save/share later |
| 4 | Dataset to soft prompt | `dataset.pre_encode` plus soft-prompt optimizer | partial | data dir, SAME model, chunking, target aggregation, opt params | Encoded dataset bundle and soft-prompt bundle | Batch dataset optimizer as one native workflow | Dataset-to-soft-prompt integration missing | Add composed recipe after dataset browser |
| 5 | Dataset to prompt family | Memory query pieces exist; clustered prompt-family recipe missing | scaffold | encoded memory, clusters, prompts per cluster, soft prompt settings | Cluster manifest and prompt/soft-prompt families | Cluster browser and prompt family generator | Missing | Add after memory/dataset browser is richer |
| 6 | SAME latent style profile | positive profile, build profile, generate profile | recipe | input/memory paths, reference path, name, alpha, prompt, generation fields | Profile bundle and audio artifact | Profile-specific inspector and profile-vs-reference comparison | Bundle tests shallow for profile semantics | Add profile inspector controls |
| 7 | SAME latent direction | audio vectors, profile build, direction/audio-direction generate | recipe | positive/ref paths, direction path, alpha/std alpha, prompt, generation fields | Direction bundle and audio artifact | Direction vector inspector and reference-delta view | Direction-specific tests shallow | Add vector/direction inspector controls |
| 8 | SA3 residual steering from prompts | prompt residual vectors, alpha sweep | recipe | axis, prompt pairs, layers, alphas, prompt, generation fields | Vector bundle and sweep family | Layer/alpha comparison and prompt-pair editor | Sweep tests exist; layer comparison missing | P1 residual layer/alpha comparison pass |
| 9 | SA3 residual steering from audio | audio residual vectors, alpha sweep | recipe | positive/negative paths, baseline, prompt, noise, layers, limit, alphas | Vector bundle and sweep family | Audio residual inspector and baseline comparison | Baseline comparison tests missing | Extend sweep/vector inspector |
| 10 | Flow-state optimization | None beyond scaffolded helper in notebook | scaffold | target latents/audio, prompt, timestep, LR, steps | Loss curve and optimized flow state | Entire native recipe, safety boundaries | Missing | Defer until prompt-search/flow diagnostics are stable |
| 11 | Inpainting / continuation composition | `generate.inpaint`, `generate.audio_to_audio` | native | source audio, prompt, duration, seed, range, init noise, model/decoder; rendered by spec-derived generation form | Audio artifact, recipe, source lineage, job phase, artifact landing | Composition timeline and continuation chain view | Generation payload and inpaint contract tests exist; timeline tests missing | Add after playback/timeline upgrade |
| 12 | LatCH-style control heads | Control-head recipe queued | scaffold | latent memory, descriptor labels, control name, training params | `control_head.pt`, metrics | Label collection, training UI, probe inspector | Missing | P2/P3 after labelled probes exist |
| 13 | LoRA scaffold | `training.lora` in Recipe Studio | recipe | encoded/raw data, model, steps, rank, adapter, precision, LR, logger, checkpointing | Checkpoint bundle, recipe, logs | Pause/resume, priority, resource scheduling, demo playback | Long-job lifecycle tests partial | Add long-job controls after runtime event hardening |
| 14 | Latent memory instrument | latent import/encode, dataset pre-encode, `memory.query` | partial | source latent, top K, metric, exclude self, encoded datasets | Memory-result bundle, ranked hits, reuse actions | Dataset browser, preview audio, style/reference promotion | Memory reuse tests first-pass only | P2 memory browser pass |
| 15 | SAME geometry and intervention audit | `experiment.geometry_audit` over local latents | partial | selected/session latents, components, limit | Geometry report bundle, variance/summary metrics | Periodicity/intervention/control probe panels | Geometry tests cover report path, not intervention UI | Add labelled geometry/control probes later |
| combined | Audio-derived prompt + residual knob + SAME style push | Chainable manually through recipes | partial | soft prompt, residual vectors, style profile, prompt, alpha/seed/model | Multiple recipes and output artifacts | Macro recipe chain UI and lineage graph | Chain tests missing | Add after session/lineage model is first-class |

## Immediate Implementation Queue

1. Keep this matrix synchronized with `/colab/modes`.
2. Add a contract test proving the runtime mode atlas still covers the notebook
   mode headings and only references known app operators.
3. Continue playback-composer behavior with draggable regions, waveform zoom,
   marker notes/persistence, and browser-level tests. Mode 2/3 presets,
   scorer-cost notes, generated-take run-comparison
   rows, vocabulary sets, axis sets, and prompt history now exist; custom
   save/share for vocabulary and axes can wait until backend preset history
   exists.
4. Expand session/workspace/archive modeling from workspace pulse and artifact
   recovery into archive review, replay/fork from archive, and long-session
   cleanup flows before Postgres so the persistence schema stores the right
   product objects.
5. Keep enriching mode-specific bundle inspectors. First domain cards exist for
   sweeps, memory hits, vector NPZs, soft-prompt tensors, and training
   checkpoints; profile and geometry workflows need the next domain pass.

## Mode Completion Contract

A mode is complete when all of the following are true:

- It has a typed request or recipe.
- Required and advanced parameters are visible with defaults, units, and bounds
  where known.
- The app records model, backend, seed, source IDs, params, logs, metrics, and
  output artifact IDs, plus persisted job phase where the runtime can report it.
- It produces typed artifacts or a clearly marked no-output diagnostic state.
- Outputs can be inspected, replayed, forked, archived, and reused where
  applicable.
- If the mode produces artifacts asynchronously, the workbench can land on the
  produced artifact when the successful job reports artifact IDs.
- The UI exposes the operation as an instrument workflow, not just a file list.
- At least one backend or contract test and one frontend/helper/UI-state test
  protects the path.

## Open Engineering Questions

- Which script-backed recipes should become true native Python operations first:
  prompt search, soft prompts, or style/vector workflows?
- Should generated takes become a formal child type inside sessions before
  playback is upgraded?
- What is the minimum useful Postgres schema once JSON manifests become too
  limiting: sessions, artifacts, recipes, jobs, annotations, lineage edges, or
  presets first?
- Which Medium/MPS scorer settings are cheap enough to recommend as defaults
  for local prompt-search exploration?
