# SA3 Native Lab Capability Map

Status: bottom-up capability map for the current notebook and
`latent_audio_primitives/` package as of 2026-06-05.

This document answers: what the repo can actually do, which native objects it
moves, which artifacts it produces, and which next notebook interfaces are
justified by local code and notebook evidence.

Evidence labels:

- `confirmed`: directly present in the notebook, local code, or docs.
- `repo-inferred`: implied by local code paths, but not executed in this pass.
- `hypothesis`: plausible next use, not yet validated by a Colab run.
- `unknown`: needs SA3/SAME weights, artifacts, or listening evidence.

## Boundary

This repo is a notebook research instrument. It keeps:

- one expanded Colab notebook,
- a notebook-facing primitive library,
- research docs and run protocol.

Upstream runtimes and training tools remain external dependencies that the
notebook consumes through checked-out repos, audio artifacts, checkpoints, and
run notes.

## Native Objects

| Object | Shape / Form | Owner | Lifecycle | Evidence |
|---|---|---|---|---|
| Audio waveform `x` | waveform arrays and `.wav` paths | notebook, `audio_descriptors.py`, `colab_audio_player.py` | loaded, previewed, described, annotated | confirmed |
| SAME latent `z0` | usually `B x C x T`, memory rows use `T x D` | upstream SAME through adapters; local `LatentItem` | encoded, edited, decoded, searched, saved | confirmed |
| SA3 flow state `z_t` | noisy/intermediate latent state | `flow_prompt.py`, notebook SA3 probes | constructed from `z0`, timestep, and noise | confirmed |
| Prompt condition `C(p)` | SA3 conditioner outputs or optimized tensors | upstream SA3 plus `flow_prompt.py`, `soft_prompt.py` | scored, optimized, attributed, auditioned | confirmed |
| Residual activation `a_l` | layer activation tensors | `adapters/audioscope_sa3.py`, residual experiments | captured, contrasted, steered, summarized | confirmed |
| `LatentItem` | ID, latent, rate, prompt, descriptors, labels, metadata | `schema.py`, `io.py` | saved, loaded, indexed, clustered | confirmed |
| Control lane | time-varying values, rate, confidence, metadata | `control_lanes.py` | extracted, normalized, compared, saved, rendered | confirmed |
| Descriptor report | JSON-friendly audio statistics | `audio_descriptors.py` | computed for source/baseline/method outputs | confirmed |
| Manifest / ledger row | run metadata and decision fields | notebook, `experiment-ledger.md` | records evidence packets and decisions | confirmed |

## Capability Cards

### Model Boundary

Evidence: `latent_audio_primitives/adapters/stable_audio3.py`,
`latent_audio_primitives/adapters/audioscope_sa3.py`, notebook setup cells.

I/O:

```text
prompts/audio/checkpoints
-> upstream SA3/SAME model handles
-> generation, encode/decode, residual hook access
```

Parameters: model name, device, dtype, duration, seed, steps, CFG, init noise,
hook layers.

Artifacts: audio paths, SAME latents, residual activations, sampler outputs.

Constraint: upstream internals are version-sensitive. Keep coupling isolated in
adapters or clearly labeled notebook probes.

### SAME Representation

Evidence: `latent_blur.py`, `latent_dsp.py`, `selective_renoise.py`,
`style.py`, `geometry.py`, `periodic.py`, `looping.py`, notebook cells under
`SAME_REPRESENTATION`.

I/O:

```text
audio or latent item
-> SAME latent edit / probe / statistical transform
-> direct decode or SA3 polish
-> descriptors, memory distance, listening note
```

Parameters: sigma, channel mask, latent mask, blur radius, filter band,
low-rank size, DSP gain/drive/phase, profile alpha, covariance transport alpha,
loop shift, polish noise.

Artifacts: edited latents, direct decodes, polished audio, descriptor deltas,
geometry reports, style profiles/directions, loop previews.

Affordances justified in notebook: source/donor chooser, latent edit recipe
table, direct-decode versus polish A/B player, geometry report, loop preview.

Unknowns: which SAME edits are stable controls rather than microscopes.

### SA3 Flow Conditioning

Evidence: `flow_prompt.py`, `prompt_optimization.py`, `tokenizer_vocab.py`,
`experiments/soft_prompt.py`, notebook cells under `SA3_FLOW_CONDITIONING`.

I/O:

```text
target audio
-> SAME z0
-> shared flow probe bank
-> prompt / soft condition scores
-> prompt candidates, attribution rows, timestep panels
```

Parameters: velocity convention, timesteps or logSNRs, noise seeds,
antithetic noise, normalized MSE, cosine weight, conditional-delta weight,
candidate prompts, token vocabulary, soft-prompt learning rate/steps.

Artifacts: `FlowPromptLossRow`, attribution tables, prompt rankings, soft
conditioning `.pt` files, audition outputs.

Affordances justified in notebook: prompt score table, leave-one-out token
attribution, loss-by-timestep panel, readable modifier search, soft prompt
audition controls.

Unknowns: whether teacher-forced flow agreement predicts generated audio
quality or only vector-field alignment.

### Causal Steering

Evidence: `adapters/audioscope_sa3.py`, `experiments/activation_vectors.py`,
`experiments/audio_residual_vectors.py`, `residual_features.py`,
`observability.py`, `guidance.py`, `looping.py`, notebook cells under
`CAUSAL_STEERING`.

I/O:

```text
prompt pairs / labeled audio / sampler state
-> residual vectors, feature bases, guidance gradients, cyclic projections
-> alpha sweeps or guided variants
-> descriptors, probe rows, listening decisions
```

Parameters: layer indices, prompt/audio contrast sets, alpha, top-k vectors,
guidance scale, loss weights, cyclic roll mix, denoising step window.

Artifacts: steering vectors, feature atlas JSON, alpha sweep audio, guided
variants, observability probe reports.

Affordances justified in notebook: alpha sweep player, layer/feature table,
guidance recipe JSON, cyclic projection comparison.

Unknowns: which interventions causally move audible attributes without
off-manifold artifacts.

### Dataset Memory and Composition

Evidence: `schema.py`, `io.py`, `index.py`, `curriculum.py`, `composition.py`,
`control_lanes.py`, notebook cells under `DATASET_MEMORY_COMPOSITION`.

I/O:

```text
dataset folder or saved latent memory
-> LatentItem rows, clusters, nearest neighbors, bridge candidates
-> source/donor/continuation decisions
-> audio outputs and evidence packets
```

Parameters: dataset limit, duration, cluster count, heldout fraction, hybrid
weights, bridge weights, lane similarity, donor/source constraints.

Artifacts: memory folders, curriculum JSON, nearest-memory rows, bridge
rankings, continuation outputs, source/donor comparison reports.

Affordances justified in notebook: memory search table, representative/heldout
rows, bridge candidate ranking, donor selector, novelty/source-preservation
panel.

Unknowns: whether memory distance separates useful source preservation from
copying across real datasets.

### Evidence Decision Protocol

Evidence: `colab_audio_player.py`, `audio_descriptors.py`,
`control_lanes.py`, notebook manifest/log cells, `experiment-ledger.md`.

I/O:

```text
source/baseline/method outputs
-> player rows, descriptor tables, lane panels, manifests, notes
-> promote / revise / drop / unknown / microscope-only decisions
```

Parameters: annotation labels, descriptor config, lane extraction settings,
manifest fields, run packet fields.

Artifacts: player HTML, annotation JSONL, descriptor dicts, lane JSON/SVG,
manifest rows, ledger entries.

Affordances justified in notebook: A/B/C player, annotation search, run packet
cell, descriptor/lane summary, decision template.

Unknowns: which evidence panels become too heavy for routine Colab use.

### External Comparison

Evidence: Underfit handoff cells, cross-model command harness, source context.

I/O:

```text
external audio/checkpoints/commands
-> notebook comparison rows
-> descriptor/player/ledger evidence
```

Parameters: external command templates, fixed prompts, output folder, imported
Underfit artifacts.

Artifacts: external audio outputs, comparison descriptor rows, player panels,
checkpoint/run-note references.

Constraint: this is comparison only. Training and external model management stay
outside this repo.

## Artifact Graph

```text
audio file
  -> SAME latent item
  -> descriptor report / latent summary / control lanes
  -> memory index / geometry / curriculum
  -> retrieval rows or donor/source candidates

target audio latent
  -> frozen SA3 flow probes
  -> prompt loss rows / attribution rows / timestep panels
  -> hard prompt candidates or soft prompt state

edited latent or sampler state
  -> direct SAME decode
  -> optional SA3 polish or intervention
  -> output audio
  -> descriptor delta + player annotation + ledger decision

prompt pairs or labeled audio
  -> residual activation capture
  -> vector basis / steering direction / alpha sweep
  -> generated outputs and review rows
```

## Parameter Inventory

Core generation:

```text
model ID, duration, sample rate, seed, steps, CFG, init noise, negative prompt
```

Flow conditioning:

```text
velocity convention, timesteps, logSNR values, probe count, antithetic noise,
normalized MSE, cosine weight, conditional-delta weight
```

SAME representation:

```text
sigma, masks, channel ranges, temporal ranges, blur radius, rank, FFT band,
phase blend, profile/direction alpha, covariance transport alpha
```

Causal steering:

```text
layer indices, residual axis, vector top-k, alpha, denoising step window,
guidance scale, loss weights, cyclic projection mix
```

Dataset memory:

```text
dataset limit, crop duration, cluster count, heldout fraction, distance metric,
hybrid weights, bridge weights, lane similarity weights
```

Evidence:

```text
descriptor config, lane frame seconds, annotation tags, manifest fields,
ledger decision
```

## Next Notebook Shape

The notebook should read as a lab bench organized by native objects,
interventions, and evidence:

1. Runtime and model boundary.
2. Shared evidence/player/manifest helpers.
3. Native object preparation: audio, SAME latent items, prompts, memory.
4. SAME representation experiments.
5. SA3 flow conditioning experiments.
6. Causal steering experiments.
7. Dataset memory and composition experiments.
8. External comparison imports.
9. Evidence ledger and next-action cells.

Within each stratum, cells should keep the same local shape:

```text
object
intervention
measurement
claim
decision
```

That shape is the stable grammar for future notebook cells.

## Unknowns And Verification Plan

- Run one small Colab packet per stratum and record it in the ledger.
- Compare direct SAME decode against SA3 polish for every representation edit.
- Cache shared flow probes before expanding prompt-search variants.
- Add nearest-memory rows to every source-preservation claim.
- Treat residual and guidance methods as microscopes until alpha/guidance sweeps
  survive listening review.
- Keep Underfit and other model outputs as imported comparison artifacts, not
  local training infrastructure.
