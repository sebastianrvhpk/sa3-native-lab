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

Architecture layer is separate from code altitude:

```text
SAME Representation Science can use root modules and evidence modules.
SA3 Flow and Conditioning Science usually uses root flow rows plus procedures.
SA3 Internal Trajectory Science usually uses adapters, procedures, and residual root measurements.
SA3-over-SAME Coupled Editing uses root SAME edits plus SA3 procedures.
Evidence and Listening Science lives in evidence modules and ledger docs.
```

The architecture-layer map lives in
[Architecture ontology](architecture-ontology.md).

### 1. Runtime and Model Boundary

Purpose: touch external SA3/SAME objects without making the notebook depend on
upstream internals everywhere.

| Module | Evidence | Role |
|---|---|---|
| `adapters/stable_audio3.py` | confirmed | Load/generate/encode/decode through official Stable Audio 3 and SAME wrappers; convert latents into memory items. |
| `adapters/sa3_residual_hooks.py` | confirmed | Locate SA3 DiT layers, capture residual activations, and apply residual steering vectors. |
| `adapters/sa3_tokenizer.py` | confirmed | Find the tokenizer owned by SA3 text conditioning. |

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
| `index.py` | confirmed | Latent memory search over summaries, controls, and hybrid scores. |
| `controls.py` | confirmed | Small scoring helpers used by the memory index. |

Narrative role: this is the lab notebook's vocabulary for "what did we make and
how do we compare it?"

### 3. Measurement and Evidence Surfaces

Purpose: turn latent/audio behavior into rows, scores, and plots before claiming
an operator is useful.

| Module | Evidence | Role |
|---|---|---|
| `audio_descriptors.py` | confirmed | Lightweight audio descriptor reports and deltas. |
| `periodic.py` | confirmed | Autocorrelation, periodicity, spectral centroid, and loop boundary probes. |
| `geometry.py` | confirmed | PCA, whitening, Mahalanobis distance, barycenters, covariance transport. |
| `control_lanes.py` | confirmed | Time-varying envelope/motion/channel lanes, normalization, similarity, SVG, persistence. |
| `observability.py` | confirmed | Linear probes for whether controls are visible in latent summaries. |
| `residual_features.py` | confirmed | Residual activation bases and directions. |
| `prompt_semantics.py` | confirmed | Prompt variant records, semantic tags, prompt evidence rows, and manifests. |

Narrative role: these modules keep the project honest. A control is not real
until it is measurable, audible, and repeatable.

### 4. SAME Measurement and Intervention Bench

Purpose: probe what SAME preserves, erases, linearizes, or makes editable.

| Module | Evidence | Role |
|---|---|---|
| `latent_blur.py` | confirmed | Temporal/channel blur, low-rank projection, sharpening, FFT filters. |
| `latent_dsp.py` | confirmed | Gain, dynamics, saturation, latent-time FFT EQ/phase, donor magnitude/phase, PCA gain. |
| `selective_renoise.py` | confirmed | Channel selection, masks, masked noise, and donor-channel graft primitives. |
| `style.py` | confirmed | Style profiles, directions, profile attraction, save/load. |
| `geometry.py` | confirmed | PCA, whitening, Mahalanobis distance, barycenters, covariance transport. |
| `periodic.py` | confirmed | Autocorrelation, periodicity, spectral centroid, and loop boundary probes. |
| `looping.py` | confirmed | Cyclic latent/audio roll, loop preview, seam metrics, and inpaint bounds. |

Narrative role: this workbench asks what the SAME bottleneck itself affords before
claiming SA3 prompt or sampler control.

### 5. SA3 Flow Prompt Bench

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

Narrative role: SA3-native prompt inversion by teacher-forced flow agreement.

### 6. Residual and Trajectory Bench

Purpose: observe residual/trajectory structure and test whether inference-time
interventions change generated audio, not just whether a signal is measurable.

| Module | Evidence | Role |
|---|---|---|
| `adapters/sa3_residual_hooks.py` | confirmed | Residual activation capture and residual steering. |
| `procedures/residual_activation_vectors.py` | confirmed | SA3 activation-vector extraction from prompt pairs. |
| `procedures/audio_residual_vectors.py` | confirmed | Residual vectors from audio examples. |
| `prompt_pairs.py` | confirmed | Prompt-pair presets for residual steering probes. |
| `procedures/residual_sweeps.py` | confirmed | Alpha sweep generation and optional audio export. |
| `residual_features.py` | confirmed | Residual activation bases and directions. |
| `observability.py` | confirmed | Linear probes for whether candidate controls are predictable from latent summaries. |
| `guidance.py` | confirmed | Differentiable latent guidance step and loss combination. |
| `procedures/cyclic_sa3.py` | confirmed | Sampler-time cyclic roll interventions. |

Narrative role: these are the highest-risk methods. They stay microscopes or
scaffolds until causal interventions survive audio review and baselines.

### 7. Memory and Composition Bench

Purpose: turn collections into memory, donor selection, curriculum, bridges, or
composition plans without confusing source preservation with copying.

| Module | Evidence | Role |
|---|---|---|
| `index.py` | confirmed | Latent memory search over summaries, controls, and hybrid scores. |
| `curriculum.py` | confirmed | Cluster memory, pick representatives, split heldout rows, show nearest-memory evidence. |
| `composition.py` | confirmed | Continuation, loop, bridge, and path ranking. |
| `control_lanes.py` | confirmed | Lane similarity can support retrieval and bridge selection after evidence validation. |
| `audio_descriptors.py` | confirmed | Descriptor summaries support donor/source comparison and novelty checks. |

Narrative role: memory is a selection and evidence system, not a generic bucket
for every dataset-level method.

### 8. Ledger and Promotion Board

Purpose: turn many clips and many variants into decisions.

| Module | Evidence | Role |
|---|---|---|
| `evidence/audio_player.py` | confirmed | Self-contained Colab waveform player and loop audition surface. |
| `evidence/annotations.py` | confirmed | Annotation save/load/search store for listening evidence. |
| `evidence/disagreement.py` | confirmed | Native evidence disagreement rows for SAME, flow, descriptors, memory, and listening. |
| `audio_descriptors.py` | confirmed | Lightweight audio descriptor reports and deltas. |
| `control_lanes.py` | confirmed | Time-varying evidence lanes, SVG visualization, persistence. |
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
  `periodic.py`, `geometry.py`, `control_lanes.py`, or `observability.py`.
- Put pure prompt rows, velocity conventions, logSNR/timestep conversion, and
  attribution in `flow_prompt.py`; put hard/readable prompt search in
  `prompt_optimization.py` and `tokenizer_vocab.py`; put teacher-forced SA3
  scoring execution in `procedures/flow_scoring.py`.
- Put latent edits in `latent_blur.py`, `latent_dsp.py`,
  `selective_renoise.py`, `looping.py`, `style.py`, or `guidance.py`.
- Put SA3/SAME external wrapper code in `adapters/`.
- Put executable SA3/SAME method runs in `procedures/`.
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
