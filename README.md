# SA3 Native Lab

This repository is a combined Colab/research workspace:

- official Stable Audio 3 source package in `stable_audio_3/`
- SAME/SA3 native latent-memory, steering, and prompt-inversion primitives in `latent_audio_primitives/`
- Colab notebooks in `colab/`
- research scripts in `scripts/`
- current experimental math notes in `docs/research/`

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
python -m pip uninstall -y scipy scikit-learn sklearn torchvision
uv pip install --system flash-attn --no-build-isolation --no-cache-dir --no-deps
```

The NumPy/scipy/sklearn/torchvision cleanup is mainly for Colab. It prevents
Transformers' optional scientific/vision import paths from reaching stale binary
packages after Torch/NumPy are repinned. Restart the Colab runtime once after
the install phase so old binary modules are cleared from memory.

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

## Research Docs

The current repo-specific math and implementation map is:

`docs/research/native-experimental-modes-math.md`

That document covers the current Colab modes, including renoise, selective latent
renoise, blur/sharpen/filtering, cross-audio grafting, cyclic sampler mixing,
neural latent DSP, soft prompt inversion, Mode 2 beam prompt inversion, SAME
statistical controls, residual steering, LatCH-style sidecars, LoRA boundaries,
and Mode 15 geometry audits.

The seven-operator research layer is:

`docs/research/seven-better-operators.md`

It maps latent geometry, covariance transport, periodic operators, direct
guidance, prompt inversion, residual feature discovery, and control
observability to the current helper modules and Colab exposure.

The neural-latent DSP notes are:

`docs/research/neural-latent-dsp.md`

They document Mode 0h: latent dynamics, soft clipping, latent-time FFT gain and
phase operators, magnitude/phase grafting, PCA component gain, SA3 polish, and
MIR descriptor audits.

Older root-level research notes are historical context from before this repo was
consolidated around the combined SA3 Native Lab implementation.
