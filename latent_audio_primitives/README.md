# Latent Audio Primitives

This package is the notebook library for SA3 Native Lab. It is not an app layer,
service layer, or product SDK.

The library supports one Colab-first research loop:

```text
audio/prompt/dataset
  -> SA3/SAME objects
  -> measurable latent state
  -> prompt, edit, retrieval, steering, or control probe
  -> decoded/polished audio plus descriptors, annotations, and decisions
```

## Module Clusters

- Model boundary: `adapters/`
- Latent objects and persistence: `schema.py`, `io.py`, `latent_math.py`,
  `index.py`, `controls.py`
- Measurement and observability: `audio_descriptors.py`, `periodic.py`,
  `geometry.py`, `control_lanes.py`, `observability.py`,
  `residual_features.py`
- Prompt inversion and prompt search: `flow_prompt.py`,
  `prompt_optimization.py`, `tokenizer_vocab.py`, `experiments/soft_prompt.py`,
  `experiments/prompt_pairs.py`
- Latent operators and interventions: `latent_blur.py`, `latent_dsp.py`,
  `selective_renoise.py`, `looping.py`, `style.py`, `guidance.py`,
  `composition.py`
- Dataset workflow and listening loop: `curriculum.py`,
  `colab_audio_player.py`, `experiments/activation_vectors.py`,
  `experiments/audio_residual_vectors.py`, `experiments/sa3_sweeps.py`

The full map lives in
[`docs/research/current/primitive-map.md`](../docs/research/current/primitive-map.md).

## Maintenance Rule

Keep new code notebook-facing. A primitive should expose a compact function,
dataclass, or row object that a Colab cell can call directly. Avoid app
scaffolding, server contracts, dashboard state, or product-interface plumbing.
