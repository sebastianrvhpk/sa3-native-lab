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
| Latent objects and persistence | `schema.py`, `io.py`, `latent_math.py`, `index.py`, `controls.py` | latent items, summaries, memory rows | store, compare, retrieve |
| Measurement and observability | `audio_descriptors.py`, `periodic.py`, `geometry.py`, `control_lanes.py`, `observability.py`, `residual_features.py` | descriptors, latent geometry, lanes, probes, residual bases | decide whether a signal is measurable |
| Prompt inversion and prompt search | `flow_prompt.py`, `prompt_optimization.py`, `tokenizer_vocab.py`, `experiments/soft_prompt.py`, `experiments/prompt_pairs.py` | prompt text, prompt tokens, SA3 condition tensors, flow losses | explain target audio through frozen SA3 dynamics |
| Latent operators and interventions | `latent_blur.py`, `latent_dsp.py`, `selective_renoise.py`, `looping.py`, `style.py`, `guidance.py`, `composition.py` | SAME latents, sampler states, style profiles, bridge candidates | edit or select native objects, then measure survival |
| Dataset workflow and listening loop | `curriculum.py`, `colab_audio_player.py`, `experiments/activation_vectors.py`, `experiments/audio_residual_vectors.py`, `experiments/sa3_sweeps.py` | clusters, annotations, residual vectors, alpha sweeps | turn many outputs into decisions |

The full map lives in
[`docs/research/current/primitive-map.md`](../docs/research/current/primitive-map.md).
The bottom-up object/capability map lives in
[`docs/research/current/capability-map.md`](../docs/research/current/capability-map.md).

## Maintenance Rule

Keep new code notebook-facing. A primitive should expose a compact function,
dataclass, or row object that a Colab cell can call directly. Service contracts,
dashboard state, and product-interface plumbing stay outside the primitive
library.
