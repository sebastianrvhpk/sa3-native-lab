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

The notebook `sa3_same_native_experimental_modes.ipynb` is configured as a
top-to-bottom Colab L4 setup for the combined Git repo. The setup/model/smoke-test defaults are ON. The
only expected manual actions are:

1. Confirm `COMBINED_REPO_URL` points to `https://github.com/sebastianrvhpk/sa3-native-lab.git`.
2. Paste a Hugging Face token when prompted.

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

1. Clone this combined repo to `/content/sa3-native-lab`.
2. Install its dependencies into the Colab Python environment with `uv pip install --system`.
3. Install Flash Attention 2 after Torch is present.
4. Restart the runtime if Colab asks.

Before Flash Attention, force the PyTorch wheel tuple expected by the SA3 repo:

```bash
pip install -U uv
uv pip install --system torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --system -e /content/sa3-native-lab
uv pip install --system --force-reinstall numpy==2.2.6
python -m pip uninstall -y scipy scikit-learn sklearn torchvision
```

The NumPy/scipy/sklearn/torchvision cleanup is Colab-specific. `uv` resolves the
SA3 dependency graph, but Colab already has optional packages outside that
graph. Transformers can opportunistically import `scipy`, `sklearn`, and
`torchvision` while loading T5Gemma; those wheels may be ABI/API-incompatible
after Torch/NumPy are repinned. SA3/T5Gemma does not need them for inference,
so the notebook removes them.

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

## Using This Repo's Adapter Layer

Once this repo is available in Colab:

```python
%pip install -e /content/sa3-native-lab
```

Then:

```python
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter
from latent_audio_primitives import LatentMemoryIndex

sa3 = StableAudio3Adapter(model=model, model_name="medium")

items = sa3.generate_items(
    prompt="a sparse glassy ambient loop, slow evolving texture",
    duration=10,
    item_id_prefix="glass-loop",
    steps=8,
    cfg_scale=1.0,
    seed=42,
)

index = LatentMemoryIndex(items)
print(items[0].shallow_metadata())
```

This converts official SA3 latents:

```text
B x C x T
```

into the project memory format:

```text
T x D
```

## First Real Memory Experiment

Generate several candidates:

```python
prompts = [
    "a sparse glassy ambient loop, slow evolving texture",
    "a dense shimmering ambient loop, bright granular movement",
    "a dark low drone with distant metallic harmonics",
]

all_items = []
for i, prompt in enumerate(prompts):
    for seed in range(3):
        all_items.extend(
            sa3.generate_items(
                prompt=prompt,
                duration=10,
                item_id_prefix=f"p{i}-seed{seed}",
                steps=8,
                cfg_scale=1.0,
                seed=seed,
            )
        )

index = LatentMemoryIndex(all_items)
results = index.query(all_items[0], top_k=5, exclude_id=all_items[0].item_id)
[(r.item_id, r.score, r.item.prompt) for r in results]
```

That gives the first real answer to:

```text
does SAME latent summary retrieval find musically related outputs?
```

## Dataset Style Direction, Not Ranking

If you want to use an arbitrary dataset to actually push generated latents,
first encode that dataset into SAME memory. For SA3 Medium, prefer `same-l`
because Medium uses SAME-Large.

```bash
python /content/sa3-native-lab/scripts/encode_dataset_same.py \
  --input /content/my_audio_dataset \
  --output /content/sa3_latent_lab/memory/my_dataset \
  --model same-l \
  --device cuda \
  --chunked
```

Build a dataset style profile:

```bash
python /content/sa3-native-lab/scripts/build_same_style_profile.py \
  --target /content/sa3_latent_lab/memory/my_dataset \
  --name my_dataset_style \
  --output /content/sa3_latent_lab/profiles/my_dataset_style.npz
```

Then generate with an actual latent-space push before decoding:

```bash
python /content/sa3-native-lab/scripts/generate_sa3_with_style_profile.py \
  --model medium \
  --profile /content/sa3_latent_lab/profiles/my_dataset_style.npz \
  --prompt "a cinematic ambient loop with evolving texture" \
  --duration 10 \
  --steps 8 \
  --cfg-scale 1.0 \
  --seed 42 \
  --alpha 0.6 \
  --save-original \
  --output /content/sa3_latent_lab/styled/my_dataset_push.wav
```

This is not branch-and-rank. It edits the generated SAME latent before decode:

```text
z = SA3(prompt)
z' = style_push(z, dataset_profile, alpha)
audio = SAME.decode(z')
```

The default style push is AdaIN-like:

```text
z' = target_std * (z - mean(z)) / std(z) + target_mean
```

with interpolation:

```text
z_final = (1 - alpha) z + alpha z'
```

For a target-minus-reference direction:

```bash
python /content/sa3-native-lab/scripts/build_same_style_profile.py \
  --target /content/sa3_latent_lab/memory/my_dataset \
  --reference /content/sa3_latent_lab/memory/reference_dataset \
  --name my_dataset_style \
  --output /content/sa3_latent_lab/profiles/my_dataset_style.npz \
  --direction-output /content/sa3_latent_lab/profiles/my_dataset_direction.npz
```

Then:

```bash
python /content/sa3-native-lab/scripts/generate_sa3_with_style_direction.py \
  --model medium \
  --direction /content/sa3_latent_lab/profiles/my_dataset_direction.npz \
  --prompt "a cinematic ambient loop with evolving texture" \
  --alpha 0.8 \
  --output /content/sa3_latent_lab/styled/my_dataset_direction.wav
```

This direction lives in SAME latent space, not SA3 residual space. It is a real
latent edit, but it happens after SA3 generation and before SAME decoding.
Sampler-time guidance is a later, more invasive step.

## Audio File Steering Vectors

You can extract two different kinds of direction from paired audio folders.

### SAME Audio Direction

This is post-generation latent editing. It is cheaper and simpler:

```text
positive folder: examples with the quality you want
negative folder: neutral/opposite/reference examples
```

Example:

```bash
python /content/sa3-native-lab/scripts/extract_audio_style_vectors.py \
  --positive /content/audio_sets/bright \
  --negative /content/audio_sets/dark \
  --output /content/sa3_latent_lab/audio_vectors/bright_minus_dark \
  --model same-l \
  --device cuda \
  --chunked
```

This writes:

```text
frame_direction.npz
summary_direction.npz
positive_memory/
negative_memory/
```

Then apply the audio-derived frame direction to SA3 output:

```bash
python /content/sa3-native-lab/scripts/generate_sa3_with_audio_direction.py \
  --model medium \
  --direction /content/sa3_latent_lab/audio_vectors/bright_minus_dark/frame_direction.npz \
  --prompt "a slow ambient piano loop" \
  --alpha 0.5 \
  --save-original \
  --output /content/sa3_latent_lab/styled/bright_audio_direction.wav
```

This is direct SAME latent steering:

```text
v_audio = mean(positive SAME frames) - mean(negative SAME frames)
z'[t] = z[t] + alpha * v_audio
```

It is not audioscope residual-stream steering. Prompt/audioscope steering edits
SA3 internals while generating; this audio-vector method edits the generated
SAME latent before decoding.

### SA3 Residual Audio Direction

This is closer to audioscope, but the contrast source is audio data rather than
prompt text. Each audio file is passed through SA3 audio-to-audio generation
while DiT residual activations are collected:

```text
audio file -> SA3 generate(init_audio=..., init_noise_level=...)
           -> collect residual h_l
positive residual mean - negative residual mean
```

Extract:

```bash
python /content/sa3-native-lab/scripts/extract_audio_residual_vectors.py \
  --model medium \
  --positive /content/audio_sets/bright \
  --prompt "audio texture" \
  --duration 6 \
  --steps 8 \
  --init-noise-level 0.35 \
  --baseline prompt \
  --output /content/sa3_latent_lab/residual_audio_vectors/bright_minus_dark
```

With `--baseline prompt`, no negative audio folder is needed. The vector is:

```text
v_l = mean(residuals from positive audio-to-audio runs)
    - mean(residuals from prompt-only baseline runs)
```

If you do have matched reference/opposite audio, use:

```bash
python /content/sa3-native-lab/scripts/extract_audio_residual_vectors.py \
  --model medium \
  --positive /content/audio_sets/bright \
  --negative /content/audio_sets/dark \
  --baseline negative_audio \
  --prompt "audio texture" \
  --duration 6 \
  --steps 8 \
  --init-noise-level 0.35 \
  --output /content/sa3_latent_lab/residual_audio_vectors/bright_minus_dark
```

This writes:

```text
residual_audio_vectors.pt
metadata.json
```

Use it like any audioscope-style steering vector:

```bash
python /content/sa3-native-lab/scripts/run_sa3_alpha_sweep.py \
  --model medium \
  --vectors /content/sa3_latent_lab/residual_audio_vectors/bright_minus_dark/residual_audio_vectors.pt \
  --prompt "a slow ambient piano loop" \
  --alphas "-8,-4,0,4,8" \
  --output /content/sa3_latent_lab/sweeps/audio_residual_bright
```

This one does steer the SA3 residual stream while generating:

```text
h_l <- h_l + alpha * v_audio_residual_l
```

Caveat: this is experimentally stronger but also more confounded. The direction
can encode source audio duration, loudness, instrumentation, and audio-to-audio
noise level unless the positive/negative sets are carefully matched.

## Audio to Prompt

There are two meanings of "optimize from audio to prompt":

1. Prompt seed/caption from audio.
   This is a text-description problem. The repo includes a deterministic
   fallback that builds a seed prompt from filenames/folder tags, and a small
   pluggable coordinate-search helper for CLAP or human-in-loop scoring.

2. Soft prompt inversion.
   This would optimize T5Gemma/text-conditioning embeddings so SA3 output
   matches a target audio. This is now scaffolded as continuous conditioning
   optimization.

The current scaffold is:

```python
from latent_audio_primitives.prompt_optimization import (
    prompt_seed_from_audio_path,
    coordinate_prompt_search,
    default_modifier_axes,
)

seed = prompt_seed_from_audio_path("/content/audio_sets/bright/my_loop.wav")

def scorer(prompt):
    # Plug in CLAP audio-text score, a caption model, or a human score here.
    return 0.0

result = coordinate_prompt_search(seed, default_modifier_axes(), scorer)
print(result.prompt)
```

### Differentiable Soft Prompt Inversion

This is closer in spirit to PEZ / "Hard Prompts Made Easy", but for SA3 we first
optimize continuous conditioning tensors rather than immediately projecting back
to discrete text. PEZ is from "Hard Prompts Made Easy: Gradient-Based Discrete
Optimization for Prompt Tuning and Discovery" (https://arxiv.org/abs/2302.03668).

Optimize from a target audio file:

```bash
python /content/sa3-native-lab/scripts/optimize_sa3_soft_prompt.py \
  --model medium \
  --target-audio /content/target.wav \
  --seed-prompt "audio texture" \
  --duration 8 \
  --optimization-steps 100 \
  --lr 0.01 \
  --output /content/sa3_latent_lab/soft_prompts/target_soft_prompt.pt
```

Generate with that optimized conditioning:

```bash
python /content/sa3-native-lab/scripts/generate_sa3_with_soft_prompt.py \
  --model medium \
  --soft-prompt /content/sa3_latent_lab/soft_prompts/target_soft_prompt.pt \
  --steps 8 \
  --cfg-scale 1.0 \
  --seed 42 \
  --output /content/sa3_latent_lab/soft_prompts/generated.wav
```

Current objective:

```text
target audio -> SAME latent z_0
z_t = (1 - t) z_0 + t eps
optimize prompt conditioning c*
so frozen SA3 predicts eps - z_0 from (z_t, t, c*)
```

This is not yet hard-token PEZ projection. It produces a reusable `.pt` soft
conditioning state for the same SA3 model/checkpoint.

## Continuation Primitive

Once you have a source audio clip:

```python
import torchaudio

source_audio = torchaudio.load("source.wav")
continued_latents = sa3.continuation_latents(
    prompt="continue as a darker, wider ambient outro",
    inpaint_audio=source_audio,
    source_duration=10.0,
    target_duration=18.0,
    steps=8,
    cfg_scale=1.0,
    seed=123,
)
```

This uses official SA3 inpainting/continuation:

```text
inpaint_mask_start_seconds = source_duration
inpaint_mask_end_seconds = target_duration
```

## audioscope-Style Steering

This repo now includes code to create vectors. To extract a quick first valence
direction:

```bash
python /content/sa3-native-lab/scripts/extract_sa3_vectors.py \
  --model medium \
  --axis valence \
  --num-pairs 2 \
  --duration 6 \
  --steps 8 \
  --cfg-scale 1.0 \
  --output /content/sa3_latent_lab/vectors/valence
```

This writes:

```text
/content/sa3_latent_lab/vectors/valence/steering_vectors.pt
/content/sa3_latent_lab/vectors/valence/metadata.json
```

Then run an alpha sweep:

```bash
python /content/sa3-native-lab/scripts/run_sa3_alpha_sweep.py \
  --model medium \
  --vectors /content/sa3_latent_lab/vectors/valence/steering_vectors.pt \
  --prompt "a cinematic ambient piano phrase" \
  --alphas "-8,-4,0,4,8" \
  --duration 8 \
  --steps 8 \
  --cfg-scale 1.0 \
  --output /content/sa3_latent_lab/sweeps/valence
```

Programmatic usage:

```python
from latent_audio_primitives.adapters.audioscope_sa3 import (
    ResidualSteerer,
    SteeringVectors,
)

vectors = SteeringVectors.load("cache/mood_vectors.pt")
steerer = ResidualSteerer(model, vectors, layer=11)

with steerer.steer(alpha=8.0):
    steered_latents = model.generate(
        prompt="a cinematic ambient piano phrase",
        duration=10,
        seed=123,
        return_latents=True,
    )
```

Important: ordinary forward hooks are fine for collecting activations, but audioscope reports hook-side mutation can be unreliable with `torch.compile`. The adapter uses monkey-patching for steering, matching audioscope's approach.

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

## Failure Modes

Flash Attention import failure:

```text
Check Python, Torch, CUDA, and flash_attn wheel compatibility.
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
