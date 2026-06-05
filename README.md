# SA3 Native Lab

This repository is a Colab/research workspace:

- SAME/SA3 native latent-memory, steering, and prompt-inversion primitives in `latent_audio_primitives/`
- Colab notebooks in `colab/`
- current notebook research notes in `docs/research/current/`

The goal is exploratory research over native SA3/SAME spaces.

## Upstream

Stable Audio 3 is used as an external upstream runtime:

https://github.com/Stability-AI/stable-audio-3

The Colab setup clones and installs that repo separately, then installs this
repo's notebook/primitives layer.

## Colab L4

Push this repo to GitHub, then use the Colab notebook:

`colab/sa3_same_native_experimental_modes.ipynb`

The notebook is already configured to clone:

```python
SA3_REPO_URL = "https://github.com/Stability-AI/stable-audio-3.git"
LAB_REPO_URL = "https://github.com/sebastianrvhpk/sa3-native-lab.git"
```

The notebook installs upstream SA3 plus this repo. No zip upload is needed.

## Local Install

```bash
uv pip install --system -e .
```

Install upstream Stable Audio 3 separately when running local notebook helpers
that load SA3/SAME weights:

```bash
git clone https://github.com/Stability-AI/stable-audio-3.git ../stable-audio-3
uv pip install --system -e ../stable-audio-3
uv pip install --system -e .
```

For SA3 Medium on Linux CUDA:

```bash
uv pip install --system torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --system -e ../stable-audio-3
uv pip install --system -e .
uv pip install --system --force-reinstall numpy==2.2.6
python -m pip uninstall -y scipy scikit-learn sklearn torchvision
uv pip install --system flash-attn --no-build-isolation --no-cache-dir --no-deps
```

The NumPy/scipy/sklearn/torchvision reset is mainly for Colab. It prevents
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

Start with the research documentation index:

`docs/research/README.md`

The current project state is:

`docs/research/current/research-state.md`

The current methods and math notes are:

`docs/research/current/methods-and-math.md`

That document covers SA3/SAME objects, frozen-model principles, native flow
prompt scoring, mode equations, latent DSP, geometry, control observability,
guidance, residual steering, and implementation safety notes.

The external source context is:

`docs/research/current/source-context.md`

It summarizes relevant SA3/SAME, Underfit, flow, guidance, control,
audio-generation, neural-audio, and activation-steering sources by notebook
impact.

The actual run ledger is:

`docs/research/current/experiment-ledger.md`

Use it for Colab runs, listening notes, descriptor deltas, and keep/revise/drop
decisions.

The next-work backlog is:

`docs/research/current/backlog.md`

It tracks priorities, open questions, promote/drop criteria, and implementation
order.

LoRA work uses `dada-bots/underfit`, with exported artifacts available for
notebook comparison.
