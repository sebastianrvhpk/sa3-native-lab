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
Intervention or measurement: what changes or what is reported
Evidence artifact: dataclass, row, latent item, descriptor dict, audio output, or note
Decision use: promote, revise, drop, unknown, or microscope only
```

This is why the package favors small functions, dataclasses, and JSON-friendly
rows over a new runtime framework.

## Capability Clusters

The library is not organized as a product SDK. It is organized as a set of
notebook-facing capabilities that support the science ontology.

### 1. Model Boundary

Purpose: touch external SA3/SAME objects without making the notebook depend on
upstream internals everywhere.

| Module | Evidence | Role |
|---|---|---|
| `adapters/stable_audio3.py` | confirmed | Load/generate/encode/decode through official Stable Audio 3 and SAME wrappers; convert latents into memory items. |
| `adapters/audioscope_sa3.py` | confirmed | Capture residual activations and apply audioscope-style steering vectors. |

Constraint: these modules may follow upstream SA3 internals. Keep that coupling
isolated here or in a clearly named sampler experiment.

### 2. Native Records and Persistence

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

### 3. Evidence and Observability

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

Narrative role: these modules keep the project honest. A control is not real
until it is measurable, audible, and repeatable.

### 4. SAME Representation

Purpose: probe what SAME preserves, erases, linearizes, or makes editable.

| Module | Evidence | Role |
|---|---|---|
| `latent_blur.py` | confirmed | Temporal/channel blur, low-rank projection, sharpening, FFT filters, SA3 polish from init latents. |
| `latent_dsp.py` | confirmed | Gain, dynamics, saturation, latent-time FFT EQ/phase, donor magnitude/phase, PCA gain. |
| `selective_renoise.py` | confirmed | Channel selection, masks, masked noise, grafting, selective SA3 renoise/graft. |
| `style.py` | confirmed | Style profiles, directions, profile attraction, save/load. |
| `geometry.py` | confirmed | PCA, whitening, Mahalanobis distance, barycenters, covariance transport. |
| `periodic.py` | confirmed | Autocorrelation, periodicity, spectral centroid, and loop boundary probes. |

Narrative role: this stratum asks what the SAME bottleneck itself affords
before claiming SA3 prompt or sampler control.

### 5. SA3 Flow and Conditioning

Purpose: ask frozen SA3 what prompt or conditioning object explains a target
latent under its own flow field.

| Module | Evidence | Role |
|---|---|---|
| `flow_prompt.py` | confirmed | SA3 flow prompt loss, logSNR timesteps, velocity convention, normalized MSE, cosine term, conditional-delta option, attribution rows. |
| `prompt_optimization.py` | confirmed | Coordinate, greedy-token, and beam prompt search. |
| `tokenizer_vocab.py` | confirmed | Native tokenizer vocabulary extraction and preview. |
| `experiments/soft_prompt.py` | confirmed | Soft prompt optimization and generation hooks. |

Narrative role: this is not captioning. It is SA3-native prompt inversion by
teacher-forced flow agreement.

### 6. Causal Steering

Purpose: test whether an inference-time intervention changes generated audio,
not just whether a signal is measurable.

| Module | Evidence | Role |
|---|---|---|
| `adapters/audioscope_sa3.py` | confirmed | Residual activation capture and audioscope-style residual steering. |
| `experiments/activation_vectors.py` | confirmed | SA3 activation-vector extraction from prompt pairs. |
| `experiments/audio_residual_vectors.py` | confirmed | Residual vectors from audio examples. |
| `experiments/prompt_pairs.py` | confirmed | Prompt-pair presets for residual steering probes. |
| `experiments/sa3_sweeps.py` | confirmed | Alpha sweep generation and optional audio export. |
| `residual_features.py` | confirmed | Residual activation bases and directions. |
| `observability.py` | confirmed | Linear probes for whether candidate controls are predictable from latent summaries. |
| `guidance.py` | confirmed | Differentiable latent guidance step and loss combination. |
| `looping.py` | confirmed | Sampler-time cyclic roll interventions plus loop metrics. |

Narrative role: these are the highest-risk methods. They stay microscopes or
scaffolds until causal interventions survive audio review and baselines.

### 7. Dataset Memory and Composition

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

### 8. Evidence Decision Protocol

Purpose: turn many clips and many variants into decisions.

| Module | Evidence | Role |
|---|---|---|
| `colab_audio_player.py` | confirmed | Self-contained Colab waveform player, loop audition, annotation save/search. |
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
  -> optional SA3 polish or sampler intervention
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
- Put prompt scoring/search code in `flow_prompt.py`, `prompt_optimization.py`,
  or `tokenizer_vocab.py`.
- Put latent edits in `latent_blur.py`, `latent_dsp.py`,
  `selective_renoise.py`, `looping.py`, `style.py`, or `guidance.py`.
- Put SA3/SAME external wrapper code in `adapters/`.
- Put notebook-facing experiment harnesses with model hooks in `experiments/`.
- Keep app, server, dashboard, route, session, artifact-UI, and product-control
  concepts out of this package.

## Current Structure Debt

- `latent_audio_primitives/__init__.py` is intentionally broad for notebook
  convenience, but it should remain a convenience surface, not the main mental
  model.
- The notebook setup cell is grouped by this map. Keep future primitive imports
  in those groups so the notebook keeps reading as a lab workflow.
- Sampler-level helpers depend on upstream `stable_audio_3` internals. That is
  acceptable for research notebooks, but every such helper should stay clearly
  labeled as SA3-version-sensitive.

## Promotion Rule

A primitive graduates from "interesting operator" to "kept method" only when it
has:

1. a clear mathematical or notebook rationale,
2. a compact API that works from cells,
3. descriptor or latent evidence,
4. listening notes in the experiment ledger,
5. a promote/revise/drop decision.
