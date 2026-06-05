# Stable Audio 3 Medium on Colab L4

This runbook is for running the released `stable_audio_3` Medium checkpoint on a Colab L4 GPU, then feeding `return_latents=True` outputs into this repo's latent-memory/composition layer.

Current source facts:

- Official repo lists `medium` as `SAME-Large`, CUDA GPU, 1.4B params, max 380s.
- Official repo says Stable Audio 3 Medium requires Flash Attention 2.
- Official repo lists L4 / RTX 4090 compute capability as `8.9` for Flash Attention builds.
- Hugging Face gates `stabilityai/stable-audio-3-medium`; you must accept the model conditions and log in with a token.

Sources:

- https://github.com/Stability-AI/stable-audio-3
- https://huggingface.co/stabilityai/stable-audio-3-medium

## Colab Runtime

Use:

```text
Runtime type: Python 3
Hardware accelerator: L4 GPU
```

Then verify:

```python
!nvidia-smi
```

Expected GPU: NVIDIA L4, usually 24 GB VRAM.

The notebook `sa3_native_research_programs.ipynb` is configured as a
top-to-bottom Colab L4 setup for upstream SA3 plus this notebook repo. The
setup/model/smoke-test defaults are ON. The
only expected manual actions are:

1. Confirm `SA3_REPO_URL` points to `https://github.com/Stability-AI/stable-audio-3.git`.
2. Confirm `LAB_REPO_URL` points to `https://github.com/sebastianrvhpk/sa3-native-lab.git`.
3. Paste a Hugging Face token when prompted.

## Access Requirements

Before running:

1. Open https://huggingface.co/stabilityai/stable-audio-3-medium
2. Log into Hugging Face.
3. Accept the model/license conditions.
4. Create a Hugging Face token with read access.

In Colab:

```python
from huggingface_hub import login
login()
```

Paste the token when prompted.

## Installation Strategy

The official repo uses `uv` and pins:

```text
torch==2.7.1
torchaudio==2.7.1
CUDA 12.6 wheels on Linux
```

In Colab, prefer `uv pip install --system`, not plain `uv sync`. `uv sync`
creates/manages a project virtualenv, while the running notebook kernel uses
Colab's current Python environment. `--system` installs into the environment
the notebook is actually executing.

Colab images change over time, so the safest pattern is:

1. Clone upstream SA3 to `/content/stable-audio-3`.
2. Clone this repo to `/content/sa3-native-lab`.
3. Install both repos into the Colab Python environment with `uv pip install --system`.
4. Install Flash Attention 2 after Torch is present.
5. Restart the runtime if Colab asks.

Before Flash Attention, force the PyTorch wheel tuple expected by the SA3 repo:

```bash
pip install -U uv
uv pip install --system torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/Stability-AI/stable-audio-3.git /content/stable-audio-3
git clone https://github.com/sebastianrvhpk/sa3-native-lab.git /content/sa3-native-lab
uv pip install --system -e /content/stable-audio-3
uv pip install --system -e /content/sa3-native-lab
uv pip install --system --force-reinstall numpy==2.2.6
python -m pip uninstall -y scipy scikit-learn sklearn torchvision
```

The NumPy/scipy/sklearn/torchvision reset is Colab-specific. `uv` resolves the
SA3 dependency graph, but Colab already has optional packages outside that
graph. Transformers can opportunistically import `scipy`, `sklearn`, and
`torchvision` while loading T5Gemma; those wheels may fail after Torch/NumPy
are repinned. SA3/T5Gemma does not need them for inference, so the notebook
removes them.

The notebook intentionally restarts the Colab runtime once after the first
install. Rerun cell 0 after reconnect; it will skip installs and verify imports.

First try the wheel-enabled path:

```bash
uv pip install --system -U setuptools wheel packaging psutil ninja
TORCH_CUDA_ARCH_LIST="8.9" \
MAX_JOBS=2 \
uv pip install --system flash-attn --no-build-isolation --no-cache-dir --no-deps
```

If that starts compiling for a long time, a matching prebuilt wheel was not found. Source compile is expected to take much longer than dependency installation. Only opt into source build when you are willing to wait:

```bash
uv pip install --system -U setuptools wheel packaging psutil ninja
TORCH_CUDA_ARCH_LIST="8.9"
MAX_JOBS=2
FLASH_ATTENTION_FORCE_BUILD=TRUE
FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE
uv pip install --system flash-attn --no-build-isolation --no-binary flash-attn --force-reinstall --no-cache-dir --no-deps
```

Compilation can take a while. If Colab RAM is tight, keep `MAX_JOBS=2`. Verify:

```bash
python -c "import flash_attn; from flash_attn import flash_attn_func; print(flash_attn.__version__)"
```

## Minimal Medium Smoke Test

Use short durations first:

```python
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
latents = model.generate(
    prompt="a sparse glassy ambient loop, slow evolving texture",
    duration=10,
    steps=8,
    cfg_scale=1.0,
    seed=42,
    return_latents=True,
)

print(latents.shape)
```

Expected shape:

```text
B x 256 x T
```

At 44.1 kHz / 4096:

```text
10 sec ~= 108 latent frames
```

Because SA3 pads/adapts duration internally, exact `T` may be a little larger.

## Decode and Save Audio

```python
audio = model.model.pretransform.decode(latents)
audio = audio.float().clamp(-1, 1).cpu()

import torchaudio
torchaudio.save("sa3_medium_test.wav", audio[0], model.model.sample_rate)
```

Or generate decoded audio directly:

```python
audio = model.generate(
    prompt="a sparse glassy ambient loop, slow evolving texture",
    duration=10,
    steps=8,
    cfg_scale=1.0,
    seed=42,
)
```

## Notebook Entry Point

After the setup and smoke test work, continue in:

```text
colab/sa3_native_research_programs.ipynb
```

The notebook contains the research workflows for SAME memory, native flow prompt
scoring, latent DSP, geometry audits, residual steering, Underfit artifact
comparison, and experiment manifests. Keep this runbook focused on runtime
health; update notebook methods in the notebook and research docs.

## Practical Defaults

Start with:

```text
duration: 5-10 sec
steps: 8
cfg_scale: 1.0
batch_size: 1
return_latents: True for research
chunked_decode: True if decoding long audio
```

Scale up only after the smoke test works.

## Failure Cases

Flash Attention import failure:

```text
Check Python, Torch, CUDA, and flash_attn wheel versions.
For L4 source builds, use TORCH_CUDA_ARCH_LIST="8.9".
```

Hugging Face access failure:

```text
Accept model conditions on the model page, then login with a read token.
```

Out of memory:

```text
Reduce duration, use batch_size=1, decode chunked, restart runtime.
```

Slow first run:

```text
Expected. Model download, compilation, and first CUDA kernels are the slow path.
```

Latents shape surprise:

```text
SA3 adapts/pads duration internally. Use metadata and actual latent shape,
not only requested duration.
```
