# SA3 Native Lab

This repository is a combined Colab/research workspace:

- official Stable Audio 3 source package in `stable_audio_3/`
- SAME/SA3 native latent-memory, steering, and prompt-inversion primitives in `latent_audio_primitives/`
- Colab notebooks in `colab/`
- research scripts in `scripts/`

The goal is exploratory research over native SA3/SAME spaces, not a finished product.

## Upstream

The Stable Audio 3 source in this repository comes from:

https://github.com/Stability-AI/stable-audio-3

The upstream README is preserved as:

`README.stable-audio-3.md`

The upstream license is preserved as:

`LICENSE.stability-ai-stable-audio-3`

## Colab L4

Push this repo to GitHub, then use the Colab notebook:

`colab/sa3_same_native_experimental_modes.ipynb`

The notebook is already configured to clone:

```python
COMBINED_REPO_URL = "https://github.com/sebastianrvhpk/sa3-native-lab.git"
```

The notebook installs this single repo. No zip upload is needed.

## Local Install

```bash
uv pip install --system -e .
```

For SA3 Medium on Linux CUDA:

```bash
uv pip install --system torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --system -e .
uv pip install --system --force-reinstall numpy==2.2.6
python -m pip uninstall -y scikit-learn sklearn
uv pip install --system flash-attn --no-build-isolation --no-cache-dir --no-deps
```

The NumPy/sklearn cleanup is mainly for Colab. It prevents Transformers'
optional sklearn import path from reaching a stale SciPy/NumPy binary stack.

## Native Spaces

```text
audio x -> SAME encoder E -> latent z
prompt p -> SA3 conditioner C(p)
SA3 DiT/flow model v_theta(z_t, t, C(p))
SAME decoder D(z) -> audio
```

The research layer focuses on:

- SAME latent memory and statistics
- audio-to-soft-prompt inversion using SA3-native flow losses
- hard/babble prompt search
- SAME latent style profiles and directions
- audioscope-style SA3 residual steering
- audio-derived residual vectors
- continuation/inpainting as composition
- LatCH-style control heads
