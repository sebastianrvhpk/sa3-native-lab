# SA3 Native Lab Python Surface Audit

Status: current as of 2026-06-04.

This audit classifies tracked Python files from the current notebook-first
perspective. It separates files used by the expanded Colab notebook from
standalone research scripts, vendored upstream SA3 code, MLX reference code, and
tests.

## Counts

| Area | Files | Current role |
|---|---:|---|
| `latent_audio_primitives/` | 34 | Notebook library and tested research primitives. |
| `scripts/` | 15 | Notebook-adjacent command-line helpers and validation. |
| `stable_audio_3/` | 31 | Vendored SA3 runtime/source package. |
| `optimized/mlx/` | 16 | Separate Apple Silicon / MLX reference path. |
| `tests/` | 27 | Primitive and adapter tests. |
| `colab/` | 1 | Older standalone Colab helper script. |

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

Package `__init__.py` files are not standalone features, but they are needed for
imports and package exports.

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
- `_runtime.py`: helper imported by direct script execution as `_runtime`.

### `stable_audio_3/` Runtime

Keep the inference/model side:

- `stable_audio_3/__init__.py`
- `stable_audio_3/model.py`
- `stable_audio_3/cli.py`
- `stable_audio_3/model_configs.py`
- `stable_audio_3/loading_utils.py`
- `stable_audio_3/factory.py`
- `stable_audio_3/inference/`
- `stable_audio_3/models/`
- `stable_audio_3/data/`
- `stable_audio_3/verbose.py`

`stable_audio_3/data/` is still used by `scripts/pre_encode_dataset.py` and by
looping helpers that reuse upstream padding utilities.

Keep `stable_audio_3/models/lora/` for now because `stable_audio_3/model.py` and
`stable_audio_3/models/dit.py` import LoRA helpers as part of the upstream
runtime wrapper.

### `tests/`

Keep the full test suite. It is currently the main protection for reusable
research math without requiring model weights.

## Delete Candidates

These look safe to remove in a cleanup commit:

| File | Reason |
|---|---|
| `colab/sa3_medium_l4_latent_memory.py` | Older notebook-style standalone helper. It has no active references and duplicates the expanded notebook/runbook path. |
| `scripts/build_positive_audio_style_profile.py` | Undocumented convenience wrapper. Its behavior is covered by `encode_dataset_same.py` plus `build_same_style_profile.py`. |
| `stable_audio_3/training/diffusion.py` | No active notebook/script references; import currently fails because it imports the removed `stable_audio_3.interface` package. |
| `stable_audio_3/training/utils.py` | No active notebook/script references; import currently fails because it imports the removed `stable_audio_3.interface` package. |

The two training files are the clearest technical residue: they are neither
part of the current notebook workflow nor importable after the interface cleanup.

## Review Before Deleting

| Path | Why review first |
|---|---|
| `scripts/pre_encode_dataset.py` | Older `.npy`/`.json` latent exporter using upstream `stable_audio_3.data`. It is referenced in docs and can support captioned latent datasets, but it overlaps with the canonical `encode_dataset_same.py` memory format. |
| `optimized/mlx/` | Separate MLX reference path, not imported by the notebook. Keep if Apple Silicon inference remains useful; remove if the repo is meant to be Colab/notebook-only. |
| `stable_audio_3/data/` | Keep while `pre_encode_dataset.py` and looping helpers use it. Revisit only if those paths are removed or rewritten. |
| `stable_audio_3/models/lora/` | Runtime import dependency for the SA3 wrapper and DiT. Removing it requires intentionally stripping LoRA support from upstream model code. |

## Import Checks

Observed during audit:

```text
uv run python scripts/*.py --help
```

All script help paths completed.

```text
import stable_audio_3.training.diffusion
import stable_audio_3.training.utils
```

Both training imports fail because they reference `stable_audio_3.interface`,
which is no longer present.

## Recommended Cleanup Order

1. Remove the two direct orphans:
   - `colab/sa3_medium_l4_latent_memory.py`
   - `scripts/build_positive_audio_style_profile.py`
2. Remove `stable_audio_3/training/` because it is broken and outside the
   current notebook workflow.
3. Decide whether `scripts/pre_encode_dataset.py` remains useful as a captioned
   latent exporter or should be folded into `encode_dataset_same.py`.
4. Decide whether `optimized/mlx/` is still a desired separate reference path.

After each deletion pass, run:

```bash
uv run python scripts/validate_colab_notebook.py --skip-setup
uv run pytest
python -m json.tool colab/sa3_same_native_experimental_modes.ipynb >/tmp/sa3_notebook.json
git diff --check
```
