# SA3 Native Lab Python Surface Audit

Status: current as of 2026-06-04.

This audit describes the Python surface after removing the vendored
`Stability-AI/stable-audio-3` source copy. This repo now owns the notebook,
research primitives, scripts, and tests. The SA3 runtime is an external checkout
installed by the notebook setup.

## Current Python Areas

| Area | Role |
|---|---|
| `latent_audio_primitives/` | Notebook library and tested SA3/SAME research primitives. |
| `scripts/` | Notebook-adjacent command-line helpers and validation. |
| `tests/` | Primitive and adapter tests using fake models/synthetic tensors. |

Removed upstream/reference areas:

- `stable_audio_3/`
- `optimized/mlx/`
- `docs/guides/`
- `docs/workflows/`
- `README.stable-audio-3.md`
- `LICENSE.stability-ai-stable-audio-3`

## Runtime Boundary

The repo still uses `stable_audio_3` as a runtime import, but the module now
comes from an external upstream checkout:

```text
https://github.com/Stability-AI/stable-audio-3
```

The Colab setup installs upstream SA3 first, then installs this repo:

```text
/content/stable-audio-3     -> provides stable_audio_3
/content/sa3-native-lab     -> provides latent_audio_primitives and notebook assets
```

Local scripts that load model weights require the same boundary: install the
upstream SA3 package before running them.

## Keep

### `latent_audio_primitives/`

Keep this package. The expanded notebook imports the real modules directly, and
the test suite covers the core primitive families:

- SA3/SAME adapters and audioscope hooks.
- SAME memory, schema, I/O, summaries, and retrieval.
- Flow prompt scoring and prompt optimization.
- Selective renoise, grafting, blur, latent DSP, looping, geometry, style, and
  audio vectors.
- Control lanes, curricula, observability, guidance, residual features,
  descriptors, tokenizer vocabulary, and Colab audio player helpers.

Package `__init__.py` files are needed for imports and package exports.

### `scripts/`

Keep the documented research/validation scripts:

- `validate_colab_notebook.py`: notebook safety net.
- `encode_dataset_same.py`: canonical SAME memory encoder.
- `build_same_style_profile.py`: profile/direction builder from memory folders.
- `generate_sa3_with_*`: apply soft prompts, style profiles, style directions,
  and audio directions.
- `optimize_sa3_soft_prompt.py`: soft prompt optimization wrapper.
- `extract_sa3_vectors.py`: prompt-pair residual vectors.
- `extract_audio_residual_vectors.py`: audio-pair residual vectors.
- `extract_audio_style_vectors.py`: SAME direction extraction from audio folders.
- `run_sa3_alpha_sweep.py`: residual steering sweep wrapper.
- `pre_encode_dataset.py`: captioned `.npy`/`.json` exporter that depends on
  upstream `stable_audio_3.data`.
- `_runtime.py`: helper imported by direct script execution as `_runtime`.

### `tests/`

Keep the full test suite. It is the main protection for reusable research math
without requiring model weights.

## Removed In This Cleanup

| Path | Reason |
|---|---|
| `colab/sa3_medium_l4_latent_memory.py` | Older notebook-style helper duplicated by the expanded notebook/runbook. |
| `scripts/build_positive_audio_style_profile.py` | Undocumented wrapper covered by `encode_dataset_same.py` plus `build_same_style_profile.py`. |
| `stable_audio_3/` | Upstream source now cloned/installed externally. |
| `optimized/mlx/` | Upstream/reference implementation now left to upstream SA3. |
| `docs/guides/`, `docs/workflows/`, upstream README/license/images | Upstream reference material now lives in the upstream repo. |

## Remaining Review Items

| Path | Question |
|---|---|
| `scripts/pre_encode_dataset.py` | Keep as captioned latent exporter, or fold into `encode_dataset_same.py` so all dataset encoding writes the same memory format? |

## Validation Commands

After Python-surface cleanup, run:

```bash
uv run python scripts/validate_colab_notebook.py --skip-setup
uv run pytest
python -m json.tool colab/sa3_same_native_experimental_modes.ipynb >/tmp/sa3_notebook.json
git diff --check
```
