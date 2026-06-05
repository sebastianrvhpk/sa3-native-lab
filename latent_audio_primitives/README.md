# Latent Audio Primitives

This package is the notebook library for SA3 Native Lab.

The library supports one Colab-first research loop:

```text
audio/prompt/dataset
  -> SA3/SAME objects
  -> measurable latent state
  -> prompt, edit, retrieval, steering, or control probe
  -> decoded/polished audio plus descriptors, annotations, and decisions
```

## Primitive Contract

Every primitive should make these fields legible from a notebook cell:

```text
Object: native object under study
Intervention or measurement: what the primitive changes or reports
Evidence artifact: row, dataclass, audio path, latent item, descriptor, or note
Decision use: how the result supports promote / revise / drop / microscope only
```

Prefer compact functions, dataclasses, and JSON-friendly rows. Keep heavy SA3
runtime coupling in `adapters/` or clearly labeled sampler helpers. Do not hide
upstream-version-sensitive behavior behind generic abstractions.

## Module Clusters

| Cluster | Modules | Object | Research Role |
|---|---|---|---|
| Model boundary | `adapters/` | official SA3/SAME wrappers, residual hooks | isolate external runtime coupling |
| Native records and persistence | `schema.py`, `io.py`, `latent_math.py`, `index.py`, `controls.py` | latent items, summaries, memory rows | store, compare, retrieve |
| Evidence and observability | `audio_descriptors.py`, `periodic.py`, `geometry.py`, `control_lanes.py`, `observability.py`, `residual_features.py` | descriptors, latent geometry, lanes, probes, residual bases | decide whether a signal is measurable |
| SAME representation | `latent_blur.py`, `latent_dsp.py`, `selective_renoise.py`, `style.py`, `geometry.py`, `periodic.py`, `looping.py` | SAME latents, latent-time trajectories, source/donor latents | edit or probe the SAME bottleneck, then measure survival |
| SA3 flow conditioning | `flow_prompt.py`, `prompt_optimization.py`, `tokenizer_vocab.py`, `experiments/soft_prompt.py` | prompt text, prompt tokens, SA3 condition tensors, flow losses | explain target audio through frozen SA3 dynamics |
| Causal steering | `experiments/prompt_pairs.py`, `experiments/activation_vectors.py`, `experiments/audio_residual_vectors.py`, `experiments/sa3_sweeps.py`, `guidance.py`, `residual_features.py` | residual vectors, sampler states, guidance losses, alpha sweeps | test whether interventions causally move generated audio |
| Dataset memory and composition | `curriculum.py`, `composition.py`, `index.py`, `control_lanes.py`, `audio_descriptors.py` | clusters, source/donor candidates, bridges, lanes | turn collections into selection, continuity, and evidence tools |
| Evidence decision loop | `colab_audio_player.py`, `audio_descriptors.py`, `control_lanes.py` | annotations, descriptors, lane rows, player panels | turn many outputs into decisions |

The full map lives in
[`docs/research/current/primitive-map.md`](../docs/research/current/primitive-map.md).
The bottom-up object/capability map lives in
[`docs/research/current/capability-map.md`](../docs/research/current/capability-map.md).

## Maintenance Rule

Keep new code notebook-facing. A primitive should expose a compact function,
dataclass, or row object that a Colab cell can call directly. State, artifact
paths, and presentation details should remain explicit at the notebook boundary.
