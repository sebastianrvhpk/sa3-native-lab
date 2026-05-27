# Colab Capability Map

This map tracks the migration from notebook/script experiments into typed app
contracts. It is intentionally evidence-based: every row should point at a local
execution surface and the app surface that now owns it or still needs it.

## Current Coverage

Default app exploration targets SA3 `medium` and SAME `same-l`, matching the
notebook's Medium/SAME-L setup. Small checkpoints remain selectable for explicit
smoke tests, but they are not the app default.

| Capability | Evidence | Current app surface | Status |
| --- | --- | --- | --- |
| Text to audio | `optimized/mlx/sa3`, `generate.text_to_audio` | `/generate/text`, listening bench | implemented |
| Audio to audio | `optimized/mlx/sa3 --init-audio` | `/generate/audio-to-audio` | implemented |
| Inpaint | `optimized/mlx/sa3 --inpaint-range` | `/generate/inpaint` | implemented |
| SAME encode/decode | SAME adapter scripts and tests | `/latents/encode`, `/latents/decode` | implemented |
| Latent blur/DSP/graft/renoise/roll | `latent_audio_primitives` wrappers | `/operators/run`, Operator Studio | implemented with native parameter controls |
| Real waveform peaks | audio artifacts via `soundfile` | `/artifacts/{id}/peaks` | implemented |
| Audio style vectors | `scripts/extract_audio_style_vectors.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Positive style profile | `scripts/build_positive_audio_style_profile.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| SAME style profile build | `scripts/build_same_style_profile.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| SA3 style profile generation | `scripts/generate_sa3_with_style_profile.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| SA3 style direction generation | `scripts/generate_sa3_with_style_direction.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| SA3 audio direction generation | `scripts/generate_sa3_with_audio_direction.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Prompt residual vectors | `scripts/extract_sa3_vectors.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Audio residual vectors | `scripts/extract_audio_residual_vectors.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Residual alpha sweep | `scripts/run_sa3_alpha_sweep.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Soft prompt optimize/generate | `scripts/optimize_sa3_soft_prompt.py`, `scripts/generate_sa3_with_soft_prompt.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| Dataset pre-encode | `scripts/pre_encode_dataset.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls |
| LoRA training | `scripts/train_lora.py` | `/experiments/run`, Recipe Studio | script-job adapter with native controls; long-running |
| Colab mode inventory | `colab/sa3_same_native_experimental_modes.ipynb` | `/colab/modes`, Mode Atlas | typed parity/status map |

## Artifact Graph

```text
audio folder / memory folder / vector file / profile file
  -> experiment recipe
  -> background job
  -> audio artifact(s) and/or zipped bundle artifact
  -> recipe + logs + metrics + lineage
```

Script-job adapters are a bridge, not the final ideal surface. They make Colab
workflows durable and replayable first. Recipe Studio now exposes every
script-backed experiment with typed controls, advanced parameters, artifact
selectors, and selected-artifact fallbacks where the script can consume the
current artifact. The Mode Atlas keeps the whole Colab mode list visible and
marks whether each mode is native, partial, or still scaffolded.

Operator Studio exposes the direct latent operators as native controls rather
than notebook snippets: cyclic roll, blur, DSP, graft, and renoise all map to
their executable request params, with donor-latent selection shown only when the
chosen mode needs a donor.

## Next Promotion Targets

1. Result-family view for sweeps: one row per alpha, with A/B promotion.
2. Memory browser/query endpoint for encoded SAME datasets and artifact latents.
3. First-class profile/vector metadata readers instead of zipped bundle only.
4. Prompt-search recipes for Colab Modes 2/3/5.
5. Geometry/control-head recipes for Modes 12/15.
6. Operator presets and recipe diffs for repeatable latent explorations.
7. Long-job controls for LoRA training: pause/cancel is not yet implemented.
