# SA3 Native Lab Primitive Map

Status: current narrative map for the notebook-facing primitive library.

This document answers: what belongs in `latent_audio_primitives/`, how the
modules cluster, and how a reader should understand the library as one research
instrument rather than a pile of utilities.

## Evidence Labels

- `confirmed`: directly present in the repo or imported by the current notebook.
- `repo-inferred`: implied by local code paths, but not fully run in this audit.
- `unknown`: needs Colab execution with SA3/SAME weights or listening review.

## Library Thesis

`latent_audio_primitives/` is the reusable layer under the expanded Colab
notebook. It should stay small enough to read, but stable enough that notebook
cells can focus on experiments instead of reimplementing tensor bookkeeping.

The package has one job:

```text
audio/prompt/dataset
  -> SAME/SA3 objects
  -> measurable latent state
  -> prompt, edit, retrieval, or steering intervention
  -> decoded/polished audio plus descriptors, rows, annotations, and decisions
```

The upstream SA3 repo remains external. This repo keeps notebook-native
research code: scoring, measurement, latent operators, memory, probes, and
listening support.

## Primitive Contract

Each primitive should expose the lab frame, not hide it:

```text
Object: native object under study
Transition: what maps into what, or what state is compared
Operation: observe, select, intervene, render, compare, or decide
Measurement: what evidence is collected
Evidence artifact: dataclass, row, latent item, descriptor dict, audio output, or note
Maturity/decision use: microscope, selector, intervention candidate, promoted
  method, revise, drop, or unknown
```

This is why the package favors small functions, dataclasses, and JSON-friendly
rows over a new runtime framework.

## Code Altitude Layers

The library is organized by code altitude. The notebook workbenches name
native-object transitions; the package layout names what kind of code a module
owns.

```text
root module = define or transform a native object
adapter     = find or talk to external machinery
procedure   = run a research method with SA3/SAME
evidence    = audition, annotate, display, or review results
```

Research layer is separate from code altitude:

```text
SAME Representation Science can use root modules and evidence modules.
SA3 Flow and Conditioning Science usually uses root flow rows plus procedures.
SA3 Internal Trajectory Science usually uses adapters, procedures, and residual root measurements.
SA3-over-SAME Coupled Editing uses root SAME edits plus SA3 procedures.
```

Evidence utilities live in evidence modules and ledger docs. They support all
four research layers instead of forming a fifth research layer.

The research-layer ontology lives in
[Architecture ontology](architecture-ontology.md).

## Function Audit

The current library passed:

- Python compile check for all `latent_audio_primitives/` modules.
- Import smoke for every root, adapter, procedure, and evidence module under
  the repo environment.
- Public-symbol docstring audit.
- A synthetic primitive smoke over NumPy-only records, descriptors, geometry,
  periodicity, memory, composition, style, flow-probe manifests, prompt
  semantic rows, and disagreement rows.

- Descriptor-target scoring lives in `index.py` with memory search,
  `query_controls()`, and `query_hybrid()`.
- SA3 prompt-tokenizer extraction lives in `tokenizer_vocab.py` with native
  tokenizer vocabulary filtering.

The main improvement need is discoverability: module tables alone do not show
the exact notebook call grammar. Keep the following function-level map current
when primitive APIs change.

## Function-Level Call Grammar

| Research role | Primary calls | Object transition | Evidence / decision use |
|---|---|---|---|
| SAME record | `LatentItem(...)`, `LatentItem.from_channel_first(...)`, `save_item(...)`, `load_item(...)` | SAME latent tensor -> notebook memory item | artifact identity, latent rate, descriptors, metadata |
| SAME summary | `latent_summary(z)`, `boundary_summary(z, side, k)` | `z0` -> mean/std/velocity summary or boundary state | nearest-memory, bridge, source-preservation rows |
| SAME geometry | `fit_latent_geometry(items)`, `geometry_report(items)`, `mahalanobis_summary_distance(a,b,geometry)`, `covariance_transport(z, reference)` | latent collection -> PCA geometry or transported latent | microscope first; intervention only after decode/listening |
| Periodicity | `periodicity_report(z)`, `loop_boundary_loss(z)`, `latent_autocorrelation(z)` | latent/audio segment -> loop and periodic rows | loopability microscope and bridge evidence |
| Control lanes | `audio_same_control_lanes(...)`, `audio_mir_control_lanes(...)`, `active_source_mask_from_lanes(...)`, `active_source_span_from_lanes(...)`, `control_lane_summary_table(...)`, `control_lane_correlation_table(...)`, `control_lane_region_mode_families(...)`, `control_lane_region_sweep(...)`, `control_lane_region_sweep_comparison_table(...)`, `lane_region_table(...)`, `compare_control_lane_sets(...)`, `compare_lane_region_sets(...)`, `regions_for_control_lanes(...)`, `regions_from_control_lane(...)`, `control_lane_mask(...)`, `latent_channel_scores(...)`, `latent_channel_lanes(...)`, `latent_channel_correlation_table(...)`, `latent_channel_region_table(...)`, `latent_channel_region_overlap_table(...)`, `latent_channel_region_overlap_audit(...)`, `rank_channel_region_overlap_rows(...)`, `summarize_channel_region_overlap_rows(...)`, `control_lane_internal_target_manifest_rows(...)`, `latent_channel_family_table(...)`, `rank_control_lane_matches(...)`, `rank_control_lane_bridges(...)` | audio/latent trajectory -> lanes -> active span / summary rows / active-window correlations / comparison rows / typed state/event/transition/persistence/source/signed region sweeps / all-channel lanes / all-channel correlations / channel regions / overlap audits / internal-cartography target manifests / masks / retrieval rows | measurement first; full atlas and target manifests are microscope/selector evidence, not causal channel claims |
| Control-lane rendering | `control_lane_svg(...)`, `control_lane_overlay_svg(...)`, `control_lane_region_svg(...)`, `control_lane_regions_svg(...)`, `control_lane_probe_heatmap_svg(...)`, `control_lane_prediction_svg(...)`, `latent_channel_heatmap_svg(...)` | lanes / regions / probe rows / prediction rows / latent channel scores -> SVG evidence views | evidence presentation; no lane math or intervention claim |
| Descriptors | `audio_descriptor_report(audio, sample_rate)`, `descriptor_delta(a,b)` | decoded audio -> descriptor rows | evidence utility; never promotion alone |
| Tuning maps | `infer_tuning_map(...)`, `pitch_events_from_rows(...)`, `pitch_centers_from_events(...)`, `interval_edges_from_centers(...)`, `generator_candidates_from_centers(...)`, `cps_fit_rows(...)`, `tuning_map_vector_targets(...)` | audio -> f0 rows -> pitch events -> centers -> interval graph -> ratio/generator/CPS rows -> scalar probe targets | pitch-as-information microscope; future selector/probe target for SAME/SA3 vectors |
| Tuning systems | `default_tuning_systems(...)`, `tuning_prompt_specs(...)`, `pitch_track_rows(...)`, `tuning_comparison_rows(...)`, `tuning_selectivity_rows(...)` | tuning lattice + prompt seed -> SA3 output -> f0 trace -> pitch-class fit rows | SA3 prompt-conditioning microscope/selector; monophonic f0 evidence is not proof of intonation control |
| Memory | `LatentMemoryIndex(items).query(...)`, `.query_controls(...)`, `.query_hybrid(...)`, `control_score(...)` | collection + query/control target -> nearest rows | selector; requires copying/source-preservation review |
| Curriculum | `build_memory_curriculum(items, cluster_count=...)`, `nearest_memory_rows(query, items)` | collection -> clusters / nearest rows | dataset design and heldout/listening planning |
| Composition | `ranked_continuations(source, candidates)`, `ranked_bridges(start,end,candidates)`, `best_path(items,start_id,end_id)` | memory items -> continuation/bridge/path candidates | selector before audio generation |
| SAME edits | `apply_latent_blur(...)`, `apply_latent_dsp(...)`, `graft_latent_channels(...)`, `apply_style_direction(...)` | `z0 -> z0'` | intervention candidate after direct decode and polish comparison |
| Flow probes | `flow_probe_bank_from_values(...)`, `flow_probe_bank_to_manifest(...)`, `sa3_flow_losses_for_prompts(...)` | target `z0` + probe bank + prompts -> flow rows | SA3-native microscope/selector |
| Trajectory cartography | `trajectory_map_from_probe_rows(...)`, `summarize_trajectory_bands(...)`, `trajectory_cells_to_alpha_schedule(...)`, `trajectory_cells_to_flow_probe_bank(...)`, `trajectory_cells_to_cyclic_mix_schedule(...)` | residual layer/timestep rows -> ranked trajectory cells -> schedules/probe banks | microscope/selector; schedules remain intervention candidates |
| SA3 internal feature cartography | `internal_surface_table(...)`, `SA3InternalFeatureCartographer(...).capture_surfaces(...)`, `.cfg_apg_atlas(...)`, `.sparse_feature_scaffold(...)`, `.residual_patch_sweep(...)`, `.branch_intervention_sweep(...)`, `ActivationPatchSpec(...)`, `BranchInterventionSpec(...)` | prompt/source + sampler settings -> residual/branch/gate/CFG/APG rows -> selected sparse-feature targets, bounded residual patch sweeps, or branch intervention sweeps | primary SA3-internal path; microscope/selector until patch/branch/steer audio evidence exists |
| Native sampler composition | `ScalarSchedule(...)`, `PromptPhaseSpec(...)`, `SamplerCompositionPlan(...)`, `sampler_composition_step_rows(...)`, `sa3_sampler_composition_from_init_latents(...)` | source `z0` + RF schedule + prompt/guidance/anchor schedules -> composed latent output + per-step rows | sampler-state intervention candidate; needs source/baseline/method audio evidence |
| Control-lane residual diagnostics | `control_lane_layer_probe_rows(...)`, `control_lane_window_probe_rows(...)`, `control_lane_timestep_probe_rows(...)`, `control_lane_null_layer_probe_rows(...)`, `control_lane_null_margin_table(...)`, `control_lane_region_layer_probe_rows(...)`, `control_lane_region_window_probe_rows(...)`, `control_lane_region_timestep_probe_rows(...)`, `control_lane_region_null_margin_table(...)`, `control_lane_probe_prediction_table(...)`, `control_lane_region_prediction_table(...)`, `control_lane_active_direction_table(...)`, `control_lane_region_direction_table(...)`, `SA3ControlLaneProbeExtractor(...).probe_audio_path(...)` | control lanes + typed lane-region masks + captured SA3 residual activations -> continuous lane rows, typed state/event/transition/persistence region rows, null margins, prediction rows, and direction previews | optional diagnostic selector; no longer the main SA3-internal route; typed region separability is not causal until residual patches move decoded audio |
| Native tokenizer vocabulary | `native_tokenizer_vocabulary(...)`, `extract_prompt_tokenizer(...)` | SA3 conditioner/tokenizer -> hard prompt candidates | prompt-search support, not a separate adapter layer |
| Prompt semantics | `make_prompt_variants(...)`, `prompt_semantic_rows(...)`, `rank_prompt_semantic_rows(...)` | prompt variants + native evidence -> prompt rows | transparency before treating text as discovered description |
| Residual probe math | `ActivationExample`, `SteeringVectors`, `probe_layer_rows(...)`, `probe_layer_timestep_rows(...)`, `probe_layer_window_rows(...)`, `vectors_from_examples(...)` | captured residual activations -> vectors / layer rows / timestep rows / window rows | root microscope and selector math; no SA3 execution |
| Residual procedures | `SA3ActivationVectorExtractor`, `SA3AudioResidualVectorExtractor`, `SA3ControlLaneProbeExtractor`, `alpha_sweep(...)` | prompts/audio -> captured residual examples or lane-probe rows -> optional sweep outputs | SA3 execution and rendering; steering remains high-risk candidate |
| Guidance probes | `gradient_guidance_step(...)`, `combine_guidance_losses(...)` | differentiable objective -> latent/state update | scaffold until objective movement matches listening |
| Latent constraints | `LatentConstraintSpec`, `latent_constraint_value(...)`, `latent_constraint_loss(...)`, `evaluate_latent_constraints(...)` | latent tensor + constraint spec -> scalar objective / before-after rows | root objective math for guidance or optimization; high-risk until audio evidence agrees |
| Measurement recipes | `apply_bottleneck_perturbation(...)`, `bottleneck_row(...)`, `classify_edit_survival(...)`, `flow_semantic_band_rows(...)`, `condition_geometry_rows(...)`, `sampler_physiology_row(...)` | native-object transition -> JSON-friendly row / evidence packet | integrated Colab method cells; scaffold until L4 runs and ledger decisions exist |
| Evidence utilities | `display_audio_player(...)`, `save_audio_annotation(...)`, `make_disagreement_row(...)` | outputs + rows + notes -> evidence packet | review, disagreement, and ledger decisions |

## Procedure Honesty Board

The `procedures/` package contains executable methods, not promoted claims.
The rows below are research narrative and ledger guidance, not package-level
metadata.

| Procedure | Research layer | Maturity | Why it belongs in `procedures/` |
|---|---|---|---|
| `flow_scoring.py` | SA3 flow/conditioning | microscope / selector | Calls the frozen SA3 model to evaluate prompt-conditioned velocity agreement. |
| `soft_prompt.py` | SA3 flow/conditioning | intervention candidate | Optimizes SA3 conditioning tensors and then renders through SA3. |
| `sa3_latent_sampling.py` | SA3-over-SAME coupled editing | intervention candidate | Passes edited SAME latents through upstream SA3 init-latent sampling. |
| `selective_sa3.py` | SA3-over-SAME coupled editing | intervention candidate | Combines SAME channel masks with upstream SA3 variation sampling. |
| `cyclic_sa3.py` | SA3 internal trajectory | high-risk sampler microscope / intervention candidate | Inserts cyclic projections inside a sampler trajectory, optionally with trajectory-derived per-step mix schedules. |
| `residual_activation_vectors.py` | SA3 internal trajectory | microscope / selector | Runs prompt-pair SA3 generations, captures residual examples, and delegates vector/probe math to `residual_probes.py`. |
| `audio_residual_vectors.py` | SA3 internal trajectory | high-risk microscope / selector | Runs audio-conditioned SA3 paths, captures residual examples, and delegates vector/probe math to `residual_probes.py`. |
| `internal_feature_cartography.py` | SA3 internal trajectory | microscope / selector / intervention candidate | Runs source-grounded SA3 internal surface capture, CFG/APG prompt-influence rows, sparse-feature target scaffolds, bounded clean/corrupt residual patch sweeps, and branch output intervention sweeps. |
| `sampler_composition.py` | SA3 internal trajectory | high-risk intervention candidate | Runs explicit RF Euler source-latent composition with source anchoring, CFG/APG schedules, and prompt phases. |
| `control_lane_mechanistic_probe.py` | SA3 internal trajectory plus SAME representation evidence | optional microscope / selector | Runs audio-conditioned SA3 paths and lane-predictability diagnostics; kept as a selector, not as the main SA3-internal research path. |
| `residual_sweeps.py` | SA3 internal trajectory | high-risk intervention candidate | Renders residual steering sweeps for audition and descriptors. |

No row above is promoted by being present. Promotion requires repeated evidence
packets and ledger decisions.

### 1. Runtime and Model Boundary

Purpose: touch external SA3/SAME objects without making the notebook depend on
upstream internals everywhere.

| Module | Evidence | Role |
|---|---|---|
| `adapters/stable_audio3.py` | confirmed | Load/generate/encode/decode through official Stable Audio 3 and SAME wrappers; convert latents into memory items. |
| `adapters/sa3_residual_hooks.py` | confirmed | Locate SA3 DiT layers, capture residual activations, and apply residual steering vectors. |
| `adapters/sa3_internal_hooks.py` | confirmed | Capture SA3 internal branch/gate surfaces, record CFG/APG condition-influence components, summarize memory tokens, patch selected post-block residual activations, and scale/ablate/patch selected branch outputs. |

Constraint: these modules may follow upstream SA3 internals. Keep that coupling
isolated here or in a clearly named procedure.

### 2. Audio/SAME Records and Persistence

Purpose: make latents, summaries, memory entries, and saved artifacts
inspectable across cells.

| Module | Evidence | Role |
|---|---|---|
| `schema.py` | confirmed | `LatentItem` record: ID, latent array, rate, prompt, descriptors, metadata. |
| `io.py` | confirmed | Save/load latent items as notebook artifacts. |
| `latent_math.py` | confirmed | Shape normalization, summaries, distances, boundary summaries. |
| `index.py` | confirmed | Latent memory search over summaries, descriptor targets, and hybrid scores. |

Narrative role: this is the lab notebook's vocabulary for "what did we make and
how do we compare it?"

### 3. Measurement and Evidence Surfaces

Purpose: turn latent/audio behavior into rows, scores, and plots before claiming
an operator is useful.

| Module | Evidence | Role |
|---|---|---|
| `audio_descriptors.py` | confirmed | Lightweight audio descriptor reports and deltas. |
| `tuning_systems.py` | confirmed | Xenharmonic/JI tuning-system manifests, prompt sweep rows, lightweight f0 tracking, pitch-class fit comparisons, and target-vs-null selectivity rows. |
| `tuning_maps.py` | confirmed | Relational pitch-map inference: pitch events, pitch centers, interval-ratio edges, period candidates, compact generator fits, Wilson-style CPS fits, and scalar targets for later SAME/SA3 probes. |
| `periodic.py` | confirmed | Autocorrelation, periodicity, spectral centroid, and loop boundary probes. |
| `geometry.py` | confirmed | PCA, whitening, Mahalanobis distance, barycenters, covariance transport. |
| `control_lanes.py` | confirmed | Time-varying envelope/flux/motion/channel lanes, summary rows, normalization, similarity, region masks, retrieval/bridge ranking, and persistence. |
| `observability.py` | confirmed | Linear probes for whether controls are visible in latent summaries. |
| `residual_features.py` | confirmed | Residual activation bases and directions. |
| `internal_features.py` | confirmed | SA3 internal surface specs, activation summary rows, CFG/APG influence rows, sparse-feature scaffold rows, clean/corrupt patch specs, and branch intervention specs. |
| `residual_probes.py` | confirmed | Activation examples, steering-vector containers, and layer/window/timestep probe rows after residual activations are collected. |
| `prompt_semantics.py` | confirmed | Prompt variant records, semantic tags, prompt evidence rows, and manifests. |
| `latent_constraints.py` | confirmed | Scalar latent constraint specs, objective values/losses, and before-after constraint rows. |
| `measurement_recipes.py` | confirmed | Tomography perturbation specs, survival labels, flow-semantic aggregation, control/source/composition rows, condition geometry, and sampler physiology summaries. |

Narrative role: these modules keep the project honest. A control is not real
until it is measurable, audible, and repeatable.

### 4. SAME Representation Science

Purpose: probe what SAME preserves, erases, linearizes, makes retrievable, or
makes editable before claiming SA3 prompt or sampler control.

| Module | Evidence | Role |
|---|---|---|
| `latent_blur.py` | confirmed | Temporal/channel blur, low-rank projection, sharpening, FFT filters. |
| `latent_dsp.py` | confirmed | Gain, dynamics, saturation, latent-time FFT EQ/phase, donor magnitude/phase, PCA gain. |
| `selective_renoise.py` | confirmed | Channel selection, masks, masked noise, and donor-channel graft primitives. |
| `style.py` | confirmed | Style profiles, directions, profile attraction, save/load. |
| `geometry.py` | confirmed | PCA, whitening, Mahalanobis distance, barycenters, covariance transport. |
| `periodic.py` | confirmed | Autocorrelation, periodicity, spectral centroid, and loop boundary probes. |
| `looping.py` | confirmed | Cyclic latent/audio roll, loop preview, seam metrics, and inpaint bounds. |

Narrative role: this layer owns direct SAME evidence: summaries, geometry,
memory, bottleneck stress, latent DSP, style directions, loop metrics, and
control lanes.

### 5. SA3 Flow And Conditioning Science

Purpose: ask frozen SA3 what prompt or conditioning object explains a target
latent under its own flow field.

| Module | Evidence | Role |
|---|---|---|
| `flow_prompt.py` | confirmed | Flow prompt rows, reusable probe banks, logSNR timesteps, velocity convention, attribution rows, and summaries. |
| `procedures/flow_scoring.py` | confirmed | Teacher-forced frozen-SA3 flow scoring execution. |
| `prompt_optimization.py` | confirmed | Coordinate, greedy-token, and beam prompt search. |
| `tokenizer_vocab.py` | confirmed | Native tokenizer vocabulary extraction and preview. |
| `procedures/soft_prompt.py` | confirmed | Soft prompt optimization and generation hooks. |
| `prompt_semantics.py` | confirmed | Semantic prompt variants and rows for comparing raw, readable, and flow-found prompt language. |
| `tuning_maps.py` | confirmed | Audio-derived tuning-map rows that can become prompt, SAME-latent, or residual probe targets. |
| `tuning_systems.py` | confirmed | Prompt families and rendered-audio f0 evidence for xenharmonic, equal-tempered, non-octave, and JI limit-set probes. |

Narrative role: SA3-native prompt inversion by teacher-forced flow agreement.

### 6. SA3 Internal Trajectory Science

Purpose: observe residual/trajectory structure and test whether inference-time
interventions change generated audio, not just whether a signal is measurable.

| Module | Evidence | Role |
|---|---|---|
| `adapters/sa3_residual_hooks.py` | confirmed | Residual activation capture and residual steering. |
| `trajectory.py` | confirmed | Residual-timestep cartography cells, band summaries, flow-probe conversion, cyclic mix schedules, and residual alpha schedules. |
| `sampler_composition.py` | confirmed | Scalar schedules, prompt phases, sampler-composition plan rows, RF denoised-state anchoring, and RF velocity/denoised conversions. |
| `residual_probes.py` | confirmed | Residual activation examples, steering-vector math, and reusable layer/window/timestep probe rows. |
| `control_lane_probes.py` | confirmed | Continuous control-lane-to-residual linear probes, token-blocked and call-held-out CV rows, observed-call window rows, token-preserving sampler-timestep rows, null controls, null-margin rows, held-out prediction rows, and active/quiet direction previews. |
| `internal_features.py` | confirmed | SA3 internal surface inventory, branch/gate/CFG/APG row schemas, sparse-feature target selection, and patch specs. |
| `adapters/sa3_internal_hooks.py` | confirmed | Model-boundary capture for residual, branch, adaLN, memory-token, CFG/APG, and post-block residual patching surfaces. |
| `procedures/internal_feature_cartography.py` | confirmed | Notebook procedures for internal surface capture, CFG/APG atlas, sparse-feature scaffold, and clean/corrupt residual patch sweeps. |
| `procedures/residual_activation_vectors.py` | confirmed | Prompt-pair SA3 residual activation collection and extraction packaging. |
| `procedures/audio_residual_vectors.py` | confirmed | Audio-derived SA3 residual activation collection and extraction packaging. |
| `procedures/control_lane_mechanistic_probe.py` | confirmed | Audio-conditioned SA3 residual capture plus lane/layer, lane/window, lane/timestep, null-margin, prediction, and active-direction probe packaging; retained as optional diagnostic selector. |
| `prompt_pairs.py` | confirmed | Prompt-pair presets for residual steering probes. |
| `procedures/residual_sweeps.py` | confirmed | Alpha sweep generation, optional audio export, and optional trajectory-gated steering schedules. |
| `residual_features.py` | confirmed | Residual activation bases and directions. |
| `observability.py` | confirmed | Linear probes for whether candidate controls are predictable from latent summaries. |
| `guidance.py` | confirmed | Differentiable latent guidance step and loss combination. |
| `procedures/cyclic_sa3.py` | confirmed | Sampler-time cyclic roll interventions. |
| `procedures/sampler_composition.py` | confirmed | Source-latent explicit Euler RF sampler composition with source anchoring, CFG/APG choreography, and prompt phases. |

Narrative role: these are the highest-risk methods. They stay microscopes or
scaffolds until causal interventions survive audio review and baselines.

### 7. SAME Memory and Composition Bench

Purpose: turn collections into memory, donor selection, curriculum, bridges, or
composition plans without confusing source preservation with copying.

| Module | Evidence | Role |
|---|---|---|
| `index.py` | confirmed | Latent memory search over summaries, controls, and hybrid scores. |
| `curriculum.py` | confirmed | Cluster memory, pick representatives, split heldout rows, show nearest-memory evidence. |
| `composition.py` | confirmed | Continuation, loop, bridge, and path ranking. |
| `control_lanes.py` | confirmed | Lane comparison, silence confidence, spectral-flux/transient evidence, region masks, channel atlas, retrieval ranking, and bridge ranking. |
| `audio_descriptors.py` | confirmed | Descriptor summaries support donor/source comparison and novelty checks. |

Narrative role: memory is the collection-facing side of SAME representation
science. It is a selection and evidence system, not a generic bucket for every
dataset-level method.

### 8. Ledger and Decision Board

Purpose: turn many clips and many variants into decisions.

| Module | Evidence | Role |
|---|---|---|
| `evidence/audio_player.py` | confirmed | Self-contained Colab waveform player and loop audition surface. |
| `evidence/annotations.py` | confirmed | Annotation save/load/search store for listening evidence. |
| `evidence/disagreement.py` | confirmed | Native evidence disagreement rows for SAME, flow, descriptors, memory, and listening. |
| `audio_descriptors.py` | confirmed | Lightweight audio descriptor reports and deltas. |
| `control_lanes.py` | confirmed | Time-varying evidence lanes, summary rows, persistence, masks, regions, and comparison rows. |
| `evidence/control_lane_rendering.py` | confirmed | Lane overlays, region SVGs, and latent-channel heatmaps for notebook display. |
| Notebook manifest cell | confirmed | Run metadata, experiment switches, model/runtime context. |
| `docs/research/current/experiment-ledger.md` | confirmed | Listening notes and promote/revise/drop decisions. |

Narrative role: this closes the loop from method idea to listening evidence and
promote/revise/drop decisions.

## Artifact Graph

```text
audio file
  -> SAME latent item
  -> descriptor report / latent summary / control lanes
  -> memory index / geometry / curriculum
  -> retrieval rows or donor/source candidates

target audio latent
  -> frozen flow prompt probes
  -> prompt loss rows / attribution rows / timestep panels
  -> hard prompt candidates or soft prompt state

edited latent
  -> direct SAME decode
  -> optional SA3 procedure polish or sampler intervention
  -> audio output
  -> descriptor delta + player annotation + ledger decision

prompt pairs or labeled audio
  -> residual activation capture
  -> vector basis / steering direction / alpha sweep
  -> generated outputs and review rows
```

## Placement Rules

- Put pure latent/audio measurements in `audio_descriptors.py`, `latent_math.py`,
  `periodic.py`, `geometry.py`, `control_lanes.py`, `tuning_systems.py`, or
  `observability.py`.
- Put pure prompt rows, velocity conventions, logSNR/timestep conversion, and
  attribution in `flow_prompt.py`; put hard/readable prompt search in
  `prompt_optimization.py` and `tokenizer_vocab.py`; put teacher-forced SA3
  scoring execution in `procedures/flow_scoring.py`.
- Put latent edits in `latent_blur.py`, `latent_dsp.py`,
  `selective_renoise.py`, `looping.py`, `style.py`, or `guidance.py`.
- Put SA3/SAME external wrapper code in `adapters/`.
- Put executable SA3/SAME method runs in `procedures/`, and keep their current
  maturity in docs, notebook review cells, and ledger decisions.
- Put listening/display/annotation helpers in `evidence/`.
- Keep primitives as compact notebook-callable functions, dataclasses, and row
  objects with explicit inputs, outputs, and provenance.

## Maintenance Notes

- `latent_audio_primitives/__init__.py` intentionally stays small. Import from
  concrete altitude modules in the notebook.
- The notebook setup cell is grouped by code altitude. Keep future primitive
  imports in those groups so the notebook keeps reading as a lab workflow.
- Sampler-level helpers depend on upstream `stable_audio_3` internals. That is
  acceptable for research notebooks, but every such helper should live in
  `procedures/` and stay clearly labeled as SA3-version-sensitive.

## Promotion Rule

A primitive graduates from "interesting operator" to "kept method" only when it
has:

1. a clear mathematical or notebook rationale,
2. a compact API that works from cells,
3. descriptor or latent evidence,
4. listening notes in the experiment ledger,
5. a promote/revise/drop decision.
