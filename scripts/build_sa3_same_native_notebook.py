from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab" / "sa3_same_native_experimental_modes.ipynb"


def lines(source: str) -> list[str]:
    source = source.strip("\n")
    return [line + "\n" for line in source.splitlines()]


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(source)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(source),
    }


cells = [
    md(
        r"""
# SAME/SA3 Native Experimental Modes on Colab L4

This notebook is a Colab L4 research instrument for Stable Audio 3 Medium and SAME.

Run the setup cells top-to-bottom on a fresh L4 runtime. The setup defaults are already ON for:

```text
SA3 Medium
Torch 2.7.1 + CUDA 12.6
FlashAttention
NumPy pinning for Colab binary compatibility
optional scipy/sklearn/torchvision removal to avoid Transformers optional import failures
one automatic runtime restart after first install
this repo's latent_audio_primitives package
Hugging Face login
model loading
one short smoke test
```

Why the dependency cleanup exists:

```text
uv resolves the requested dependency graph correctly, but Colab starts with many
preinstalled packages outside that graph. Transformers may opportunistically
import optional scipy/sklearn/torchvision modules while loading T5Gemma. Those
wheels can be ABI/API-incompatible after the NumPy/Torch stack changes. SA3 does
not need scipy, sklearn, or torchvision for inference, so the setup pins NumPy
and removes those optional import paths. Because Torch/NumPy are binary packages,
the setup intentionally restarts the Colab runtime once after the first install.
```

The goal is not to build a final app. The goal is to expose experimental primitives that can be measured, broken, repaired, and recombined.

Native spaces used here:

```text
audio waveform:           x
SAME latent:              z = E(x), z in R^{T x 256}
SA3 text conditioning:    c = C(prompt)
SA3 flow field:           v_theta(z_t, t, c)
SA3 residual stream:      a_l inside DiT block l
decoded audio:            x_hat = D(z)
```

We avoid CLAP by default. Scores and losses should come from SAME/SA3 whenever possible.
"""
    ),
    md(
        r"""
## Core Math

SAME encodes audio into a compressed continuous latent sequence:

$$
z = E(x), \qquad z \in \mathbb{R}^{T \times 256}
$$

For 44.1 kHz audio and downsampling ratio 4096:

$$
\text{latent rate} \approx \frac{44100}{4096} \approx 10.77 \text{ frames/sec}
$$

SA3 is treated here as a conditional latent generator over SAME latents:

$$
v_\theta(z_t, t, c)
$$

where:

```text
z_t = intermediate noisy/flow latent
t   = flow time
c   = SA3 text-conditioning tensor
```

For a rectified-flow-style surrogate objective, define a straight probability path between a data latent \(z_0\) and Gaussian noise \(\epsilon\):

$$
\epsilon \sim \mathcal{N}(0, I)
$$

$$
z_t = (1 - t) z_0 + t\epsilon
$$

$$
u_t = \epsilon - z_0
$$

This notebook calls that the `noise_minus_data` convention. Some samplers and codebases reverse time or target sign. The opposite convention is:

$$
u_t = z_0 - \epsilon
$$

called `data_minus_noise` below. The correct sign should be checked empirically against the released SA3 sampler/model wrapper.

A native prompt/conditioning objective is:

$$
\mathcal{L}_{flow}(c) =
\mathbb{E}_{t,\epsilon}
\left\| v_\theta(z_t,t,c) - u_t \right\|_2^2
$$

If this loss decreases, the conditioning better explains the target audio in SA3's own latent dynamics.

The exact target convention is an implementation detail. The research claim is not "SA3 must use this exact equation"; the claim is that frozen SA3 can be used as a native differentiable scorer for prompts/conditioning against SAME latents.
"""
    ),
    md(
        r"""
## Experimental Modes Covered

1. Audio -> soft prompt
2. Audio -> babble / hard prompt
3. Audio -> readable prompt
4. Dataset -> soft prompt
5. Dataset -> prompt family
6. SAME latent style profile
7. SAME latent direction
8. SA3 residual steering from prompts
9. SA3 residual steering from audio
10. Noise / trajectory optimization scaffold
11. Inpainting / continuation as composition
12. LatCH-style control heads
13. LoRA adaptation scaffold
14. Latent memory instrument

Interpretation rule:

```text
confirmed runnable primitive = code can run with installed SA3/SAME and input files
scaffold = implementation point is laid out, but needs model-specific sampler/script adaptation
unknown = should be measured before artistic trust
```
"""
    ),
    md(
        r"""
## Literature / Reference Map

Primary sources used for the design of this notebook:

| Area | Source | What it contributes here |
|---|---|---|
| SA3 architecture | [Stable Audio 3, Evans et al. 2026](https://arxiv.org/abs/2605.17991) | Variable-length latent diffusion for generation/editing, inpainting/continuation, SAME latent backbone, adversarial post-training. |
| SAME latent space | [SAME, Parker et al. 2026](https://arxiv.org/abs/2605.18613) | 4096x temporal compression, stereo audio autoencoder, semantic regularization, transformer backbone, SAME-L/S variants. |
| Released implementation | [Stability-AI/stable-audio-3](https://github.com/Stability-AI/stable-audio-3) | Official inference/training package, Medium requiring FlashAttention 2, model wrapper paths used by the adapters. |
| SA3 Medium model card | [stabilityai/stable-audio-3-medium](https://huggingface.co/stabilityai/stable-audio-3-medium) | Gated Medium weights, English text conditioning via T5Gemma, training-data/model-card details. |
| Flow matching | [Flow Matching, Lipman et al. 2022](https://arxiv.org/abs/2210.02747) | Vector-field regression view used by the prompt inversion objective. |
| Rectified flow | [Rectified Flow, Liu et al. 2022](https://arxiv.org/abs/2209.03003) | Straight-path data/noise transport intuition and low-step generation rationale. |
| Classifier guidance | [Diffusion Models Beat GANs, Dhariwal and Nichol 2021](https://arxiv.org/abs/2105.05233) | Gradient-based guidance idea: trade diversity/fidelity by adding external gradients. |
| Classifier-free guidance | [Classifier-Free Diffusion Guidance, Ho and Salimans 2022](https://arxiv.org/abs/2207.12598) | Conditional/unconditional score combination; motivates `cfg_scale` experiments. |
| Arbitrary guidance | [Universal Guidance, Bansal et al. 2023](https://arxiv.org/abs/2302.07121) | Conditioning with external guidance functions without retraining the base model. |
| Training-free guidance | [TFG, Ye et al. 2024](https://arxiv.org/abs/2409.15761) | Guidance design space and hyperparameter sensitivity; relevant to future direct sampler guidance. |
| Audio guidance gradients | [Controllable Music Production with Diffusion Models and Guidance Gradients, Levy et al. 2023](https://arxiv.org/abs/2311.00613) | Music continuation, inpainting, transitions, style transfer via sampling-time guidance. |
| Inference-time optimization | [DITTO, Novack et al. 2024](https://arxiv.org/abs/2401.12179) | Optimizing initial noise latents for control without fine-tuning. |
| LatCH | [Low-Resource Guidance for Controllable Latent Audio Diffusion, Novack et al. 2026](https://arxiv.org/abs/2603.04366) | Latent-control heads for cheap latent-space guidance over intensity, pitch, beats. |
| Live diffusion | [Live Music Diffusion Models, Novack et al. 2026](https://arxiv.org/abs/2605.22717) | Block-wise diffusion, KV-cache idea, ARC-Forcing, live interactive generation lens. |
| Hard prompt optimization | [PEZ / Hard Prompts Made Easy, Wen et al. 2023](https://arxiv.org/abs/2302.03668) | Soft-to-hard/babble prompt discovery; adapted here to SA3-native loss instead of image CLIP. |
| Residual steering | [Mood Vectors in Audio Diffusion, Camporese 2026](https://guglielmocamporese.github.io/blog/audio-mood-steering/) and [audioscope](https://github.com/guglielmocamporese/audioscope) | Non-peer-reviewed but directly relevant SA3 residual-stream steering method and torch.compile hook caveat. |
| Prior audio baselines | [MusicLM](https://arxiv.org/abs/2301.11325), [MusicGen](https://arxiv.org/abs/2306.05284), [AudioLDM](https://arxiv.org/abs/2301.12503), [Stable Audio Open](https://arxiv.org/abs/2407.14358) | Historical context for discrete-token music generation and latent-diffusion audio generation. |

Reading rule:

```text
SA3/SAME/HF/repo facts = confirmed source claims
prompt inversion / SAME style profile = experimental construction in this notebook
audioscope mood vectors = promising external experiment, not peer-reviewed ground truth
LMDM adaptation to SA3 = conceptual only unless SA3 sampler/attention code is modified and validated
```
"""
    ),
    md(
        r"""
## Confirmed Facts vs Notebook Assumptions

Confirmed from SA3/SAME/model-card sources:

```text
SA3 is a transformer-based latent diffusion family for variable-length audio generation/editing.
SA3 supports inpainting/continuation in the released pipeline.
SA3 Medium uses text conditioning through a public T5Gemma model according to the HF card.
SAME is a stereo audio autoencoder with 4096x temporal compression in the paper.
SAME-L and SAME-S are released; SA3 Medium is the target model for this Colab.
```

Notebook assumptions / experimental constructs:

```text
The flow-loss prompt inversion objective is a native surrogate, not a paper-claimed SA3 inversion method.
The velocity sign convention must be checked against the released wrapper.
SAME style profiles and SAME directions are latent edit hypotheses, not guaranteed perceptual controls.
Audio-derived residual vectors are a proposed extension of audioscope-style prompt vectors.
LatCH-style heads here are minimal scaffolds, not the full LatCH training recipe.
LMDM adaptation is conceptual unless SA3 attention/sampler code is modified and benchmarked.
```
"""
    ),
    md(
        r"""
## Method Traceability

| Notebook mode | Native object edited | Closest literature |
|---|---|---|
| 1. Audio -> soft prompt | SA3 conditioning tensor \(c\) | Flow matching, PEZ/soft prompts, DITTO-style inference optimization |
| 2. Audio -> babble prompt | hard text/token sequence \(p\) | PEZ / hard prompt optimization |
| 3. Audio -> readable prompt | constrained text prompt \(p\) | prompt search, SA3 text-conditioned generation |
| 4. Dataset -> soft prompt | shared conditioning tensor \(c_D\) | prompt tuning, dataset inversion, flow matching |
| 5. Dataset -> prompt family | cluster-conditioned \(c_k\) | latent clustering + prompt tuning |
| 6. SAME style profile | decoded SAME latent statistics | AdaIN-style statistics transfer, latent autoencoder control |
| 7. SAME direction | contrastive SAME vector | representation arithmetic / latent directions |
| 8. Prompt residual steering | DiT residual vector \(v_l\) | representation engineering, audioscope mood vectors |
| 9. Audio residual steering | audio-conditioned DiT residual vector \(v_l\) | audioscope-style steering plus audio-to-audio probing |
| 10. Trajectory optimization | noise / \(z_t\) / sampler state | DITTO, training-free guidance |
| 11. Inpainting / continuation | masked SAME latent \(z_{missing}\) | SA3 inpainting, guidance-gradient music editing |
| 12. LatCH-style control head | sidecar \(h_\psi(z)\) | LatCH, training-free guidance |
| 13. LoRA | low-rank weight delta \(\Delta W\) | lightweight adaptation/fine-tuning |
| 14. Latent memory | stored \(z, s(z), metadata\) | retrieval, dataset geometry, composition systems |
"""
    ),
    code(
        r"""
# @title 0. Colab L4 setup for the combined SA3 Native Lab repo

# Push this combined repository to GitHub, then put that URL here.
COMBINED_REPO_URL = "https://github.com/sebastianrvhpk/sa3-native-lab.git"

# Defaults are ON for Stable Audio 3 Medium experiments.
CLONE_COMBINED_REPO = True
INSTALL_REPO = True
INSTALL_TORCH_CU126 = True
INSTALL_FLASH_ATTN = True
USE_UV = True
PIN_NUMPY = True
REMOVE_TRANSFORMERS_OPTIONAL_SCIPY_SKLEARN = True
REMOVE_TRANSFORMERS_OPTIONAL_TORCHVISION = True
RESTART_RUNTIME_AFTER_FIRST_INSTALL = True
FORCE_FULL_SETUP = False

PROJECT_DIR = "/content/sa3-native-lab"
SETUP_MARKER = "/content/.sa3_native_lab_setup_complete"
FLASH_ATTN_WHEEL_URL = ""  # Optional direct wheel URL matching Python/Torch/CUDA.
FLASH_ATTN_MAX_JOBS = "2"
FLASH_ATTN_FROM_SOURCE = False

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


def run(cmd, cwd=None, env=None, check=True):
    print("+", " ".join(str(part) for part in cmd))
    proc = subprocess.run(
        [str(part) for part in cmd],
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: "
            + " ".join(str(part) for part in cmd)
        )
    return proc


def ensure_uv():
    if not USE_UV:
        return
    if shutil.which("uv"):
        return
    run([sys.executable, "-m", "pip", "install", "-U", "uv"])


def install(args, env=None):
    args = [str(arg) for arg in args]
    if USE_UV:
        ensure_uv()
        uv_env = os.environ.copy()
        uv_env.setdefault("UV_LINK_MODE", "copy")
        if env:
            uv_env.update(env)
        return run(["uv", "pip", "install", "--system", *args], env=uv_env)
    return run([sys.executable, "-m", "pip", "install", *args], env=env)


def module_available(module_name):
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def verify_import(module_name, attr=None):
    import importlib

    module = importlib.import_module(module_name)
    value = getattr(module, attr) if attr else module
    location = getattr(module, "__file__", "<built-in>")
    version = getattr(module, "__version__", "unknown")
    print(f"OK import {module_name} version={version} path={location}")
    return value


def ensure_project_dir():
    project = Path(PROJECT_DIR)
    has_project = (
        (project / "pyproject.toml").exists()
        and (project / "stable_audio_3").exists()
        and (project / "latent_audio_primitives").exists()
    )
    if has_project:
        if CLONE_COMBINED_REPO and (project / ".git").exists():
            run(["git", "-C", str(project), "pull", "--ff-only"], check=False)
        return project

    if CLONE_COMBINED_REPO:
        if not COMBINED_REPO_URL:
            raise RuntimeError(
                "Set COMBINED_REPO_URL to your GitHub repo URL. "
                "This repo must contain pyproject.toml, stable_audio_3/, and latent_audio_primitives/."
            )
        if project.exists():
            raise RuntimeError(
                f"{project} exists but is not a complete combined repo. "
                "Delete it or set PROJECT_DIR to a clean path."
            )
        run(["git", "clone", COMBINED_REPO_URL, str(project)])
        if not (
            (project / "pyproject.toml").exists()
            and (project / "stable_audio_3").exists()
            and (project / "latent_audio_primitives").exists()
        ):
            raise RuntimeError(
                f"Cloned {COMBINED_REPO_URL}, but it is missing pyproject.toml, "
                "stable_audio_3/, or latent_audio_primitives/."
            )
        return project

    raise RuntimeError(
        "PROJECT_DIR is not a complete combined repo and CLONE_COMBINED_REPO=False."
    )


setup_marker = Path(SETUP_MARKER)
setup_complete = setup_marker.exists() and not FORCE_FULL_SETUP

if INSTALL_REPO:
    PROJECT_DIR = str(ensure_project_dir())
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

if setup_complete:
    print(f"Setup marker found: {SETUP_MARKER}")
    print("Skipping dependency installation. Delete the marker or set FORCE_FULL_SETUP=True to reinstall.")

if not setup_complete and INSTALL_TORCH_CU126:
    install(
        [
            "torch==2.7.1",
            "torchaudio==2.7.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )

if not setup_complete and INSTALL_FLASH_ATTN:
    if module_available("flash_attn"):
        print("flash_attn already importable; skipping FlashAttention install.")
    else:
        if USE_UV:
            ensure_uv()
            install(["-U", "setuptools", "wheel", "packaging", "psutil", "ninja"])
        else:
            run([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel", "packaging", "psutil", "ninja"])
        env = os.environ.copy()
        env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")  # L4 / RTX 4090
        env.setdefault("MAX_JOBS", FLASH_ATTN_MAX_JOBS)
        if FLASH_ATTN_WHEEL_URL:
            install([FLASH_ATTN_WHEEL_URL], env=env)
        elif not FLASH_ATTN_FROM_SOURCE:
            # Fast path: allow flash-attn's setup to use a matching prebuilt
            # wheel when available for this Python/Torch/CUDA tuple.
            install(
                [
                    "flash-attn",
                    "--no-build-isolation",
                    "--no-cache-dir",
                    "--no-deps",
                ],
                env=env,
            )
        else:
            env.setdefault("FLASH_ATTENTION_FORCE_BUILD", "TRUE")
            env.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "FALSE")
            install(
                [
                    "flash-attn",
                    "--no-build-isolation",
                    "--no-binary",
                    "flash-attn",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "--no-deps",
                ],
                env=env,
            )
    run([sys.executable, "-c", "import flash_attn; from flash_attn import flash_attn_func; print('flash_attn', flash_attn.__version__)"])

if not setup_complete and INSTALL_REPO:
    install(["-e", PROJECT_DIR])

if not setup_complete and PIN_NUMPY:
    # The upstream lower bound is numpy>=2.2.6. On Colab, resolving to newer
    # NumPy can leave optional scipy/sklearn stacks ABI-inconsistent. Pin the
    # minimum SA3-compatible version before importing transformers/T5Gemma.
    install(["--force-reinstall", "numpy==2.2.6"])

if not setup_complete and REMOVE_TRANSFORMERS_OPTIONAL_SCIPY_SKLEARN:
    # Transformers imports scipy/sklearn opportunistically for generation/loss
    # utilities. SA3/T5Gemma does not need them, and Colab's scipy/sklearn wheels
    # can become incompatible after the NumPy resolver changes above.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "scipy", "scikit-learn", "sklearn"], check=False)

if not setup_complete and REMOVE_TRANSFORMERS_OPTIONAL_TORCHVISION:
    # Transformers also imports torchvision opportunistically from image_utils.
    # Colab's preinstalled torchvision is tied to its original Torch build and
    # can fail after pinning Torch to SA3's 2.7.1+cu126 requirement.
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchvision"], check=False)

if not setup_complete:
    setup_marker.write_text("complete\n", encoding="utf-8")
    if RESTART_RUNTIME_AFTER_FIRST_INSTALL:
        print("\n=== first install complete ===")
        print("Restarting the Colab runtime now to clear old Torch/NumPy modules.")
        print("After Colab reconnects, rerun this cell. It will skip installs and verify imports.")
        time.sleep(2)
        os.kill(os.getpid(), 9)

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

print("\n=== setup verification ===")
verify_import("torch")
verify_import("torchaudio")
verify_import("flash_attn")
verify_import("stable_audio_3")
verify_import("latent_audio_primitives")
print("Setup complete. Continue with the next cell.")
"""
    ),
    code(
        r"""
# @title 1. Imports and experiment paths

from pathlib import Path
import json
import math
import random
import shutil
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

from latent_audio_primitives import LatentMemoryIndex
from latent_audio_primitives.adapters.stable_audio3 import (
    SAMEAutoencoderAdapter,
    StableAudio3Adapter,
    latents_to_items,
)
from latent_audio_primitives.adapters.audioscope_sa3 import SteeringVectors, ResidualSteerer
from latent_audio_primitives.audio_vectors import frame_mean_direction, apply_frame_direction
from latent_audio_primitives.composition import ranked_continuations, ranked_bridges, best_path
from latent_audio_primitives.experiments.activation_vectors import SA3ActivationVectorExtractor
from latent_audio_primitives.experiments.audio_residual_vectors import SA3AudioResidualVectorExtractor
from latent_audio_primitives.experiments.prompt_pairs import DEFAULT_PROMPT_PAIRS, pairs_for_axis
from latent_audio_primitives.experiments.sa3_sweeps import alpha_sweep
from latent_audio_primitives.experiments.soft_prompt import (
    SoftPromptState,
    optimize_soft_prompt_from_latents,
    generate_with_soft_prompt,
)
from latent_audio_primitives.io import save_items, load_items
from latent_audio_primitives.latent_math import latent_summary
from latent_audio_primitives.prompt_optimization import (
    coordinate_prompt_search,
    default_modifier_axes,
    greedy_token_prompt_search,
    prompt_seed_from_audio_path,
)
from latent_audio_primitives.schema import LatentItem
from latent_audio_primitives.style import (
    fit_style_profile,
    style_direction,
    apply_profile_attraction,
    apply_style_direction,
    save_style_profile,
    load_style_profile,
    save_style_direction,
    load_style_direction,
)

DEVICE = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
WORK_DIR = Path("/content/sa3_same_native_experiments")
INPUT_DIR = WORK_DIR / "inputs"
OUTPUT_DIR = WORK_DIR / "outputs"
MEMORY_DIR = WORK_DIR / "memory"
VECTOR_DIR = WORK_DIR / "vectors"
FLOW_TARGET_CONVENTION = "noise_minus_data"  # Try "data_minus_noise" if the diagnostic favors it.

for directory in [INPUT_DIR, OUTPUT_DIR, MEMORY_DIR, VECTOR_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

print("device:", DEVICE)
print("work dir:", WORK_DIR)
print("flow target convention:", FLOW_TARGET_CONVENTION)
"""
    ),
    code(
        r"""
# @title 2. Hugging Face login and model loading

LOAD_MODELS = True
HF_LOGIN = True

SA3_MODEL_NAME = "medium"
SAME_MODEL_NAME = "same-l"  # SA3 Medium uses SAME-Large style latents.
MODEL_HALF = True

sa3_model = None
sa3 = None
same = None

if HF_LOGIN:
    from huggingface_hub import login
    login()

if LOAD_MODELS:
    if torch is None:
        raise RuntimeError("PyTorch is required before loading SA3/SAME.")
    import importlib
    import sys
    from pathlib import Path

    def import_stable_audio_model():
        importlib.invalidate_caches()
        try:
            from stable_audio_3 import StableAudioModel
            return StableAudioModel
        except ModuleNotFoundError as first_error:
            repo_dir = Path(globals().get("PROJECT_DIR", "/content/sa3-native-lab"))
            source_dir = repo_dir / "stable_audio_3"
            if source_dir.exists():
                sys.path.insert(0, str(repo_dir))
                importlib.invalidate_caches()
                try:
                    from stable_audio_3 import StableAudioModel
                    print(f"Imported stable_audio_3 from source checkout: {repo_dir}")
                    return StableAudioModel
                except ModuleNotFoundError:
                    pass
            raise ModuleNotFoundError(
                "stable_audio_3 is not importable in this notebook kernel. "
                "Rerun cell 0 with COMBINED_REPO_URL set and let it finish, "
                "or run `uv pip install --system -e /content/sa3-native-lab`. "
                f"Checked source path: {source_dir}"
            ) from first_error

    StableAudioModel = import_stable_audio_model()

    sa3_model = StableAudioModel.from_pretrained(
        SA3_MODEL_NAME,
        device=DEVICE,
        model_half=MODEL_HALF,
    )
    sa3 = StableAudio3Adapter(model=sa3_model, model_name=SA3_MODEL_NAME)
    same = SAMEAutoencoderAdapter.from_pretrained(SAME_MODEL_NAME, device=DEVICE)
    print("SA3 sample rate:", sa3.sample_rate)
    print("SA3 latent rate:", sa3.latent_rate)
else:
    print("Models not loaded. Set LOAD_MODELS=True when running on Colab L4.")


def require_models():
    if sa3_model is None or sa3 is None or same is None:
        raise RuntimeError("Set LOAD_MODELS=True and run the model-loading cell first.")
"""
    ),
    code(
        r"""
# @title 2b. SA3 Medium smoke test

RUN_SMOKE_TEST = True
SMOKE_PROMPT = "a sparse glassy ambient loop, slow evolving texture"
SMOKE_DURATION = 4.0
SMOKE_STEPS = 4
SMOKE_SEED = 42

if RUN_SMOKE_TEST:
    require_models()
    latents = sa3.generate_latents(
        prompt=SMOKE_PROMPT,
        duration=SMOKE_DURATION,
        steps=SMOKE_STEPS,
        cfg_scale=1.0,
        seed=SMOKE_SEED,
    )
    print("smoke latents:", tuple(latents.shape), latents.dtype, latents.device)
    smoke_audio = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
    smoke_path = OUTPUT_DIR / "sa3_medium_smoke.wav"
    torchaudio.save(str(smoke_path), smoke_audio[0], sa3.sample_rate)
    print("saved:", smoke_path)
    try:
        from IPython.display import Audio, display
        display(Audio(str(smoke_path)))
    except Exception as exc:
        print("Audio display skipped:", exc)
"""
    ),
    md(
        r"""
## Shared Helpers

These helpers keep the notebook focused on experiments:

```text
audio file -> SAME latent tensor
SA3 tensor B x C x T -> LatentItem memory format T x D
time-major latent -> SA3 channel-first tensor
prompt -> SA3 flow loss against target audio latent
```
"""
    ),
    code(
        r"""
# @title 3. Shared audio, latent, decode, and native flow-loss helpers

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def list_audio_files(directory, limit=None):
    paths = sorted(
        path for path in Path(directory).rglob("*")
        if path.suffix.lower() in AUDIO_EXTS
    )
    return paths[:limit] if limit else paths


def load_audio(path, target_sample_rate=None, duration=None, stereo=True):
    if torchaudio is None:
        raise RuntimeError("torchaudio is required.")
    audio, sample_rate = torchaudio.load(str(path))
    if not stereo and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if target_sample_rate and sample_rate != target_sample_rate:
        audio = torchaudio.functional.resample(audio, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    if duration is not None:
        target_samples = int(round(duration * sample_rate))
        if audio.shape[-1] < target_samples:
            pad = target_samples - audio.shape[-1]
            audio = torch.nn.functional.pad(audio, (0, pad))
        else:
            audio = audio[..., :target_samples]
    return audio, sample_rate


def item_to_sa3_tensor(item, device=None, dtype=None):
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    arr = np.asarray(item.latent, dtype=np.float32).T[None, :, :]
    return torch.from_numpy(arr).to(device=device or DEVICE, dtype=dtype or torch.float32)


def sa3_tensor_to_item(latents, item_id="latent", prompt=None, metadata=None):
    require_models()
    return latents_to_items(
        latents,
        item_id_prefix=item_id,
        latent_rate=sa3.latent_rate,
        sample_rate=sa3.sample_rate,
        prompt=prompt,
        metadata=metadata or {},
    )[0]


def decode_sa3_latents_to_file(latents, path):
    require_models()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        audio = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
    torchaudio.save(str(path), audio[0], sa3.sample_rate)
    return path


def encode_audio_file_to_item(path, item_id=None, prompt=None, duration=None):
    require_models()
    audio, sample_rate = load_audio(path, duration=duration, stereo=True)
    return same.encode_tensor(
        audio,
        sample_rate,
        item_id=item_id or Path(path).stem,
        prompt=prompt,
        chunked=False,
        metadata={"path": str(path)},
    )


def encode_audio_folder_to_items(directory, *, limit=None, duration=None, prompt_from_path=True):
    items = []
    for path in list_audio_files(directory, limit=limit):
        prompt = prompt_seed_from_audio_path(path) if prompt_from_path else None
        item = encode_audio_file_to_item(path, item_id=path.stem, prompt=prompt, duration=duration)
        items.append(item)
    return items


def pad_or_crop_channel_first(tensor, target_frames):
    if tensor.shape[-1] == target_frames:
        return tensor
    if tensor.shape[-1] > target_frames:
        return tensor[..., :target_frames]
    pad = target_frames - tensor.shape[-1]
    return torch.nn.functional.pad(tensor, (0, pad))


def items_to_latent_batch(items, *, target_frames=None, device=None, dtype=None):
    if not items:
        raise ValueError("items list is empty.")
    tensors = [item_to_sa3_tensor(item, device=device or DEVICE, dtype=dtype or torch.float32) for item in items]
    target_frames = target_frames or min(t.shape[-1] for t in tensors)
    tensors = [pad_or_crop_channel_first(t, target_frames) for t in tensors]
    return torch.cat(tensors, dim=0)


def freeze_sa3_and_same():
    require_models()
    for module in [sa3_model.model, same.autoencoder]:
        if hasattr(module, "parameters"):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        if hasattr(module, "eval"):
            module.eval()


def sa3_flow_losses_for_prompts(
    stable_model,
    target_latents,
    prompts,
    *,
    duration,
    seed=0,
    min_t=0.05,
    max_t=0.95,
    velocity_convention=None,
):
    # Native scorer:
    # L(prompt) = E || v_theta(z_t, t, C(prompt)) - u_t ||^2.
    # The sign of u_t is an implementation convention; keep it explicit.
    if torch is None:
        raise RuntimeError("PyTorch is required.")
    velocity_convention = velocity_convention or FLOW_TARGET_CONVENTION
    core = stable_model.model
    device = str(stable_model.device)
    dtype = next(core.model.parameters()).dtype
    prompts = list(prompts)
    target = target_latents.to(device=device, dtype=dtype)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and len(prompts) > 1:
        target = target.expand(len(prompts), -1, -1).contiguous()
    if target.shape[0] != len(prompts):
        raise ValueError("target batch must be 1 or match number of prompts.")

    conditioning = [{"prompt": prompt, "seconds_total": duration} for prompt in prompts]
    with torch.inference_mode():
        cond = core.conditioner(conditioning, device)
        cond = dict(cond)
        batch, channels, frames = target.shape
        cond["inpaint_mask"] = [torch.zeros((batch, 1, frames), device=device)]
        cond["inpaint_masked_input"] = [torch.zeros((batch, channels, frames), device=device, dtype=dtype)]
        cond_inputs = core.get_conditioning_inputs(cond)
        cond_inputs = {
            key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
            for key, value in cond_inputs.items()
        }
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        t = min_t + (max_t - min_t) * torch.rand(batch, device=device, generator=generator)
        noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
        t_view = t[:, None, None].to(dtype)
        z_t = (1 - t_view) * target + t_view * noise
        velocity_target = flow_velocity_target(target, noise, convention=velocity_convention)
        pred = core.model(z_t, t, **cond_inputs, cfg_scale=1.0, batch_cfg=True)
        losses = torch.mean((pred.float() - velocity_target.float()) ** 2, dim=(1, 2))
    return [float(loss.detach().cpu()) for loss in losses]


def flow_velocity_target(target, noise, *, convention):
    if convention == "noise_minus_data":
        return noise - target
    if convention == "data_minus_noise":
        return target - noise
    raise ValueError("convention must be 'noise_minus_data' or 'data_minus_noise'")


def compare_flow_velocity_conventions(stable_model, target_latents, prompts, *, duration, seed=0):
    return {
        convention: sa3_flow_losses_for_prompts(
            stable_model,
            target_latents,
            prompts,
            duration=duration,
            seed=seed,
            velocity_convention=convention,
        )
        for convention in ["noise_minus_data", "data_minus_noise"]
    }
"""
    ),
    md(
        r"""
## Flow Sign Diagnostic

Before doing serious prompt inversion, test both velocity conventions on one target audio and several candidate prompts.

This does not prove the full sampler convention, but it catches a common failure mode: optimizing against a sign-flipped target.

If one convention consistently gives lower loss for human-plausible prompts and better optimization behavior, set:

```python
FLOW_TARGET_CONVENTION = "noise_minus_data"
```

or:

```python
FLOW_TARGET_CONVENTION = "data_minus_noise"
```
"""
    ),
    code(
        r"""
# @title Optional. Flow sign diagnostic

RUN_FLOW_SIGN_DIAGNOSTIC = False

DIAGNOSTIC_AUDIO = INPUT_DIR / "target.wav"
DIAGNOSTIC_DURATION = 8.0
DIAGNOSTIC_PROMPTS = [
    "audio texture",
    "a dense bright electronic loop with shimmering harmonics",
    "a sparse dark ambient drone with low sustained tones",
]

if RUN_FLOW_SIGN_DIAGNOSTIC:
    require_models()
    diagnostic_item = encode_audio_file_to_item(
        DIAGNOSTIC_AUDIO,
        item_id="diagnostic_target",
        duration=DIAGNOSTIC_DURATION,
    )
    diagnostic_latents = item_to_sa3_tensor(
        diagnostic_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    comparison = compare_flow_velocity_conventions(
        sa3_model,
        diagnostic_latents,
        DIAGNOSTIC_PROMPTS,
        duration=DIAGNOSTIC_DURATION,
        seed=0,
    )
    print(json.dumps({k: [round(v, 6) for v in values] for k, values in comparison.items()}, indent=2))
    print("Current FLOW_TARGET_CONVENTION:", FLOW_TARGET_CONVENTION)
"""
    ),
    md(
        r"""
## Mode 1. Audio -> Soft Prompt

Object optimized:

$$
c
$$

Frozen:

```text
SAME encoder/decoder
SA3 DiT / flow model
SA3 text conditioner weights
```

Objective:

$$
c^\* = \arg\min_c
\mathbb{E}_{t,\epsilon}
\left\| v_\theta(z_t,t,c) - u_t \right\|_2^2
$$

where \(u_t\) is selected by `FLOW_TARGET_CONVENTION`.

This produces a continuous conditioning preset:

```text
audio.wav -> soft_prompt.pt
```

It is not language. It is an optimized conditioning key inside SA3.
"""
    ),
    code(
        r"""
# @title Mode 1. Audio -> soft prompt

RUN_MODE_1_AUDIO_TO_SOFT_PROMPT = False

TARGET_AUDIO = INPUT_DIR / "target.wav"
SOFT_PROMPT_PATH = OUTPUT_DIR / "mode_01_target_soft_prompt.pt"

SEED_PROMPT = "experimental audio texture"
TARGET_DURATION = 8.0
SOFT_STEPS = 100
SOFT_LR = 1e-2

if RUN_MODE_1_AUDIO_TO_SOFT_PROMPT:
    require_models()
    freeze_sa3_and_same()
    target_item = encode_audio_file_to_item(TARGET_AUDIO, item_id="target", duration=TARGET_DURATION)
    target_latents = item_to_sa3_tensor(
        target_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    soft_state = optimize_soft_prompt_from_latents(
        sa3_model,
        target_latents,
        seed_prompt=SEED_PROMPT,
        duration=TARGET_DURATION,
        optimization_steps=SOFT_STEPS,
        lr=SOFT_LR,
        train_keys=("prompt",),
        reg_weight=1e-4,
        seed=0,
        velocity_convention=FLOW_TARGET_CONVENTION,
    )
    soft_state.save(SOFT_PROMPT_PATH)
    print("saved:", SOFT_PROMPT_PATH)
    print("initial/final loss:", soft_state.losses[0], soft_state.losses[-1])
"""
    ),
    md(
        r"""
## Mode 2. Audio -> Babble / Hard Prompt

Object optimized:

$$
p = [w_1,\dots,w_N]
$$

Objective:

$$
p^\* = \arg\min_p \mathcal{L}_{flow}(C(p))
$$

The result may be linguistically strange. That is acceptable if it reliably moves SA3 into the target audio region.

This cell uses greedy hard-token-style search over a vocabulary. Later refinement can replace the word list with actual tokenizer IDs.
"""
    ),
    code(
        r"""
# @title Mode 2. Audio -> babble / hard prompt

RUN_MODE_2_AUDIO_TO_BABBLE_PROMPT = False

BABBLE_VOCAB = [
    "shimmer", "velvet", "glass", "grain", "pressure", "hollow", "dense", "thin",
    "wide", "mono", "dust", "chrome", "choir", "fold", "pulse", "drift",
    "warm", "cold", "broken", "slow", "fast", "tape", "bright", "dark",
    "sub", "spark", "rubber", "metal", "wet", "dry", "attack", "bloom",
    "loop", "riser", "drop", "intro", "ghost", "close", "distant", "aura",
]

HARD_PROMPT_JSON = OUTPUT_DIR / "mode_02_babble_prompt.json"

if RUN_MODE_2_AUDIO_TO_BABBLE_PROMPT:
    require_models()
    target_item = encode_audio_file_to_item(TARGET_AUDIO, item_id="target", duration=TARGET_DURATION)
    target_latents = item_to_sa3_tensor(
        target_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    def batch_scorer(prompts):
        losses = sa3_flow_losses_for_prompts(
            sa3_model,
            target_latents,
            prompts,
            duration=TARGET_DURATION,
            seed=123,
            velocity_convention=FLOW_TARGET_CONVENTION,
        )
        return [-loss for loss in losses]

    result = greedy_token_prompt_search(
        BABBLE_VOCAB,
        batch_scorer,
        tokens_generated=12,
        runs=4,
        token_subset=24,
        seed=0,
        higher_is_better=True,
    )
    HARD_PROMPT_JSON.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(result.prompt)
    print("score:", result.score)
    print("saved:", HARD_PROMPT_JSON)
"""
    ),
    md(
        r"""
## Mode 3. Audio -> Readable Prompt

This constrains search to interpretable descriptors.

Same native score:

$$
\text{score}(p) = -\mathcal{L}_{flow}(C(p))
$$

Tradeoff:

```text
readable prompt = easier to use
babble prompt   = often a stronger conditioning key
soft prompt     = strongest, least readable
```
"""
    ),
    code(
        r"""
# @title Mode 3. Audio -> readable prompt

RUN_MODE_3_AUDIO_TO_READABLE_PROMPT = False

READABLE_PROMPT_JSON = OUTPUT_DIR / "mode_03_readable_prompt.json"

if RUN_MODE_3_AUDIO_TO_READABLE_PROMPT:
    require_models()
    target_item = encode_audio_file_to_item(TARGET_AUDIO, item_id="target", duration=TARGET_DURATION)
    target_latents = item_to_sa3_tensor(
        target_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    seed_prompt = prompt_seed_from_audio_path(TARGET_AUDIO, extra_tags=["audio texture"])

    def scorer(prompt):
        return -sa3_flow_losses_for_prompts(
            sa3_model,
            target_latents,
            [prompt],
            duration=TARGET_DURATION,
            seed=321,
            velocity_convention=FLOW_TARGET_CONVENTION,
        )[0]

    result = coordinate_prompt_search(
        seed_prompt,
        default_modifier_axes(),
        scorer,
        rounds=3,
    )
    READABLE_PROMPT_JSON.write_text(json.dumps(result.__dict__, indent=2), encoding="utf-8")
    print(result.prompt)
    print("score:", result.score)
"""
    ),
    md(
        r"""
## Mode 4. Dataset -> Soft Prompt

Object optimized:

$$
c
$$

Target set:

$$
\{z_i = E(x_i)\}_{i=1}^N
$$

Objective:

$$
c^\* =
\arg\min_c
\mathbb{E}_{i,t,\epsilon}
\left\| v_\theta(z_{i,t},t,c) - u_{i,t} \right\|_2^2
$$

This asks: what single conditioning key makes SA3 explain this folder?
"""
    ),
    code(
        r"""
# @title Mode 4. Dataset -> soft prompt

RUN_MODE_4_DATASET_TO_SOFT_PROMPT = False

DATASET_DIR = INPUT_DIR / "dataset_positive"
DATASET_SOFT_PROMPT_PATH = OUTPUT_DIR / "mode_04_dataset_soft_prompt.pt"
DATASET_DURATION = 8.0
DATASET_LIMIT = 8

if RUN_MODE_4_DATASET_TO_SOFT_PROMPT:
    require_models()
    freeze_sa3_and_same()
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
    )
    dataset_latents = items_to_latent_batch(
        dataset_items,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    dataset_state = optimize_soft_prompt_from_latents(
        sa3_model,
        dataset_latents,
        seed_prompt="audio dataset texture",
        duration=DATASET_DURATION,
        optimization_steps=SOFT_STEPS,
        lr=SOFT_LR,
        train_keys=("prompt",),
        reg_weight=1e-4,
        seed=1,
        velocity_convention=FLOW_TARGET_CONVENTION,
    )
    dataset_state.save(DATASET_SOFT_PROMPT_PATH)
    print("saved:", DATASET_SOFT_PROMPT_PATH)
    print("initial/final loss:", dataset_state.losses[0], dataset_state.losses[-1])
"""
    ),
    md(
        r"""
## Mode 5. Dataset -> Prompt Family

If a dataset is not one thing, cluster it in SAME summary space first.

Summary used here:

$$
s(z) =
\left[
\operatorname{mean}_t(z),
\operatorname{std}_t(z),
\operatorname{mean}_t|\Delta z|
\right]
$$

Then optimize one prompt or soft prompt per cluster:

$$
c_k^\* = \arg\min_c \mathbb{E}_{i \in k} \mathcal{L}_{flow}(z_i,c)
$$
"""
    ),
    code(
        r"""
# @title Mode 5. Dataset -> prompt family by SAME clustering

RUN_MODE_5_DATASET_TO_PROMPT_FAMILY = False

PROMPT_FAMILY_DIR = OUTPUT_DIR / "mode_05_prompt_family"
CLUSTERS = 3


def kmeans_numpy(x, k, iterations=50, seed=0):
    rng = np.random.default_rng(seed)
    if len(x) < k:
        raise ValueError("not enough examples for k clusters.")
    centers = x[rng.choice(len(x), size=k, replace=False)].copy()
    labels = np.zeros(len(x), dtype=np.int64)
    for _ in range(iterations):
        distances = np.linalg.norm(x[:, None, :] - centers[None, :, :], axis=-1)
        labels = distances.argmin(axis=1)
        for cluster in range(k):
            members = x[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels, centers


if RUN_MODE_5_DATASET_TO_PROMPT_FAMILY:
    require_models()
    PROMPT_FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
    )
    summaries = np.stack([latent_summary(item) for item in dataset_items])
    labels, centers = kmeans_numpy(summaries, CLUSTERS, seed=2)

    manifest = []
    for cluster_id in range(CLUSTERS):
        cluster_items = [item for item, label in zip(dataset_items, labels) if label == cluster_id]
        if not cluster_items:
            continue
        cluster_latents = items_to_latent_batch(
            cluster_items,
            device=DEVICE,
            dtype=next(sa3_model.model.model.parameters()).dtype,
        )
        state = optimize_soft_prompt_from_latents(
            sa3_model,
            cluster_latents,
            seed_prompt=f"audio cluster {cluster_id} texture",
            duration=DATASET_DURATION,
            optimization_steps=SOFT_STEPS,
            lr=SOFT_LR,
            train_keys=("prompt",),
            seed=cluster_id,
            velocity_convention=FLOW_TARGET_CONVENTION,
        )
        path = PROMPT_FAMILY_DIR / f"cluster_{cluster_id:02d}_soft_prompt.pt"
        state.save(path)
        manifest.append({"cluster": cluster_id, "items": [item.item_id for item in cluster_items], "path": str(path)})

    (PROMPT_FAMILY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
"""
    ),
    md(
        r"""
## Mode 6. SAME Latent Style Profile

This edits the generated SAME latent before decoding.

Dataset statistics:

$$
\mu_D = \mathbb{E}_{i,t}[z_i(t)]
$$

$$
\sigma_D = \operatorname{Std}_{i,t}[z_i(t)]
$$

AdaIN-like latent profile attraction:

$$
\tilde z =
\sigma_D \frac{z-\mu_z}{\sigma_z} + \mu_D
$$

$$
z' = (1-\alpha)z + \alpha \tilde z
$$
"""
    ),
    code(
        r"""
# @title Mode 6. SAME latent style profile

RUN_MODE_6_STYLE_PROFILE = False

STYLE_PROFILE_PATH = OUTPUT_DIR / "mode_06_style_profile.npz"
STYLE_PROFILE_ALPHA = 0.75
STYLE_PROFILE_TEST_PROMPT = "a sparse evolving electronic texture, detailed, wide stereo"

if RUN_MODE_6_STYLE_PROFILE:
    require_models()
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
    )
    profile = fit_style_profile(dataset_items, name="positive_dataset")
    save_style_profile(profile, STYLE_PROFILE_PATH)

    latents = sa3.generate_latents(
        prompt=STYLE_PROFILE_TEST_PROMPT,
        duration=DATASET_DURATION,
        steps=8,
        cfg_scale=1.0,
        seed=42,
    )
    item = sa3_tensor_to_item(latents, item_id="profile_source", prompt=STYLE_PROFILE_TEST_PROMPT)
    styled_z = apply_profile_attraction(item, profile, alpha=STYLE_PROFILE_ALPHA, match_std=True)
    styled_latents = torch.from_numpy(styled_z.T[None, :, :]).to(
        device=DEVICE,
        dtype=latents.dtype,
    )
    out_path = OUTPUT_DIR / "mode_06_profile_styled.wav"
    decode_sa3_latents_to_file(styled_latents, out_path)
    print("saved:", out_path)
"""
    ),
    md(
        r"""
## Mode 7. SAME Latent Direction

A contrastive direction in SAME latent space:

$$
v = \mu_A - \mu_B
$$

Apply:

$$
z' = z + \alpha v
$$

This requires a reference/baseline set if you want a true direction. With only positive examples, use Mode 6 profile attraction instead.
"""
    ),
    code(
        r"""
# @title Mode 7. SAME latent direction from positive/reference folders

RUN_MODE_7_SAME_DIRECTION = False

REFERENCE_DIR = INPUT_DIR / "dataset_reference"
STYLE_DIRECTION_PATH = OUTPUT_DIR / "mode_07_style_direction.npz"
STYLE_DIRECTION_ALPHA = 1.0

if RUN_MODE_7_SAME_DIRECTION:
    require_models()
    positive_items = encode_audio_folder_to_items(DATASET_DIR, limit=DATASET_LIMIT, duration=DATASET_DURATION)
    reference_items = encode_audio_folder_to_items(REFERENCE_DIR, limit=DATASET_LIMIT, duration=DATASET_DURATION)
    positive_profile = fit_style_profile(positive_items, name="positive")
    reference_profile = fit_style_profile(reference_items, name="reference")
    direction = style_direction(positive_profile, reference_profile, name="positive_minus_reference")
    save_style_direction(direction, STYLE_DIRECTION_PATH)

    latents = sa3.generate_latents(
        prompt=STYLE_PROFILE_TEST_PROMPT,
        duration=DATASET_DURATION,
        steps=8,
        cfg_scale=1.0,
        seed=43,
    )
    item = sa3_tensor_to_item(latents, item_id="direction_source", prompt=STYLE_PROFILE_TEST_PROMPT)
    steered_z = apply_style_direction(item, direction, alpha=STYLE_DIRECTION_ALPHA)
    steered_latents = torch.from_numpy(steered_z.T[None, :, :]).to(device=DEVICE, dtype=latents.dtype)
    out_path = OUTPUT_DIR / "mode_07_direction_styled.wav"
    decode_sa3_latents_to_file(steered_latents, out_path)
    print("saved:", out_path)
"""
    ),
    md(
        r"""
## Mode 8. SA3 Residual Steering From Prompts

Collect hidden activations from paired prompts:

$$
a_l^+,\quad a_l^-
$$

Compute a layer-wise direction:

$$
v_l = \operatorname{mean}(a_l^+) - \operatorname{mean}(a_l^-)
$$

Patch during generation:

$$
a_l \leftarrow a_l + \alpha v_l
$$

This is audioscope-style activation steering. It is inference-time and does not update SA3 weights.
"""
    ),
    code(
        r"""
# @title Mode 8. SA3 residual steering from prompt pairs

RUN_MODE_8_PROMPT_RESIDUAL_STEERING = False

PROMPT_VECTOR_DIR = VECTOR_DIR / "mode_08_prompt_residual_valence"
PROMPT_VECTOR_AXIS = "valence"
PROMPT_VECTOR_LAYERS = None  # Example: [8, 9, 10, 11, 12]
ALPHAS = [-2.0, -1.0, 0.0, 1.0, 2.0]

if RUN_MODE_8_PROMPT_RESIDUAL_STEERING:
    require_models()
    pairs = pairs_for_axis(PROMPT_VECTOR_AXIS)
    extractor = SA3ActivationVectorExtractor(
        sa3_model,
        layer_indices=PROMPT_VECTOR_LAYERS,
        cpu_offload=True,
    )
    result = extractor.extract(
        pairs=pairs,
        duration=8.0,
        steps=8,
        cfg_scale=1.0,
        seed=100,
        normalize=True,
        probe=True,
    )
    result.save(PROMPT_VECTOR_DIR)
    print("saved vectors:", PROMPT_VECTOR_DIR)
    print("probe accuracy:", result.vectors.probe_accuracy)

    sweep_outputs = alpha_sweep(
        sa3,
        prompt="a cinematic electronic loop with clear harmony and evolving texture",
        vectors=result.vectors,
        alphas=ALPHAS,
        output_dir=OUTPUT_DIR / "mode_08_alpha_sweep",
        duration=8.0,
        steps=8,
        cfg_scale=1.0,
        seed=101,
        top_k=1,
        save_audio=True,
    )
    print(sweep_outputs)
"""
    ),
    md(
        r"""
## Mode 9. SA3 Residual Steering From Audio

Instead of positive/negative prompt pairs, use audio examples to create residual directions.

Positive audio pass:

```text
init_audio=(sample_rate, audio)
record residual activations a_l^positive
```

Baseline can be:

```text
prompt-only generation
reference audio folder
negative audio folder
```

Direction:

$$
v_l = \operatorname{mean}(a_l^{audio}) - \operatorname{mean}(a_l^{baseline})
$$
"""
    ),
    code(
        r"""
# @title Mode 9. SA3 residual steering from audio files

RUN_MODE_9_AUDIO_RESIDUAL_STEERING = False

AUDIO_VECTOR_DIR = VECTOR_DIR / "mode_09_audio_residual"
AUDIO_RESIDUAL_BASELINE = "prompt"  # "prompt" or "negative_audio"
AUDIO_RESIDUAL_PROMPT = "audio texture"
AUDIO_RESIDUAL_NOISE = 0.35

if RUN_MODE_9_AUDIO_RESIDUAL_STEERING:
    require_models()
    positive_paths = list_audio_files(DATASET_DIR, limit=DATASET_LIMIT)
    negative_paths = list_audio_files(REFERENCE_DIR, limit=DATASET_LIMIT) if AUDIO_RESIDUAL_BASELINE == "negative_audio" else None
    extractor = SA3AudioResidualVectorExtractor(
        sa3_model,
        layer_indices=PROMPT_VECTOR_LAYERS,
        cpu_offload=True,
    )
    result = extractor.extract(
        positive_paths=positive_paths,
        negative_paths=negative_paths,
        prompt=AUDIO_RESIDUAL_PROMPT,
        duration=DATASET_DURATION,
        steps=8,
        cfg_scale=1.0,
        seed=200,
        init_noise_level=AUDIO_RESIDUAL_NOISE,
        baseline_mode=AUDIO_RESIDUAL_BASELINE,
        normalize=True,
        probe=True,
    )
    result.save(AUDIO_VECTOR_DIR)
    print("saved vectors:", AUDIO_VECTOR_DIR)
    print("probe accuracy:", result.vectors.probe_accuracy)
"""
    ),
    md(
        r"""
## Mode 10. Noise / Trajectory Optimization Scaffold

This mode optimizes variables in the generation trajectory rather than prompts.

Potential trainable objects:

```text
initial noise epsilon
intermediate latent z_t
per-step guidance scales
per-layer steering alphas
```

The clean formulation requires access to SA3's sampler loop:

$$
z_{k+1} = \Phi_\theta(z_k, t_k, c)
$$

Then optimize:

$$
\epsilon^\* =
\arg\min_\epsilon
d\left(D(\operatorname{sample}_\theta(\epsilon,c)), x_{target}\right)
$$

or, natively:

$$
\epsilon^\* =
\arg\min_\epsilon
\left\| z_{generated}(\epsilon,c) - z_{target} \right\|^2
$$

This first-pass cell exposes a flow-state loss. A full differentiable sampler implementation should be added after inspecting the released SA3 sampler code on Colab.
"""
    ),
    code(
        r"""
# @title Mode 10. Flow-state optimization scaffold

RUN_MODE_10_FLOW_STATE_OPT = False


def optimize_single_flow_state(
    stable_model,
    target_latents,
    prompt,
    *,
    duration,
    steps=100,
    lr=1e-2,
    t_value=0.5,
    seed=0,
):
    # This is not a full sampler inversion.
    # It optimizes an intermediate z_t so the frozen SA3 velocity agrees with
    # the target flow velocity. It is useful for debugging trajectory losses.
    core = stable_model.model
    device = str(stable_model.device)
    dtype = next(core.model.parameters()).dtype
    target = target_latents.to(device=device, dtype=dtype)
    if target.ndim == 2:
        target = target.unsqueeze(0)
    batch, channels, frames = target.shape
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    epsilon = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)
    t = torch.full((batch,), float(t_value), device=device)
    velocity_target = flow_velocity_target(target, epsilon, convention=FLOW_TARGET_CONVENTION)
    z_t_init = (1 - t[:, None, None].to(dtype)) * target + t[:, None, None].to(dtype) * epsilon
    z_t = torch.nn.Parameter(z_t_init.detach().clone())

    conditioning = [{"prompt": prompt, "seconds_total": duration}] * batch
    cond = core.conditioner(conditioning, device)
    cond = dict(cond)
    cond["inpaint_mask"] = [torch.zeros((batch, 1, frames), device=device)]
    cond["inpaint_masked_input"] = [torch.zeros((batch, channels, frames), device=device, dtype=dtype)]
    cond_inputs = core.get_conditioning_inputs(cond)
    cond_inputs = {
        key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
        for key, value in cond_inputs.items()
    }
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([z_t], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred = core.model(z_t, t, **cond_inputs, cfg_scale=1.0, batch_cfg=True)
        loss = torch.nn.functional.mse_loss(pred.float(), velocity_target.float())
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return z_t.detach(), losses


if RUN_MODE_10_FLOW_STATE_OPT:
    require_models()
    target_item = encode_audio_file_to_item(TARGET_AUDIO, item_id="target", duration=TARGET_DURATION)
    target_latents = item_to_sa3_tensor(
        target_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    z_t_opt, losses = optimize_single_flow_state(
        sa3_model,
        target_latents,
        "audio texture",
        duration=TARGET_DURATION,
        steps=100,
        lr=1e-2,
    )
    print("initial/final loss:", losses[0], losses[-1])
"""
    ),
    md(
        r"""
## Mode 11. Inpainting / Continuation As Composition

Boundary-conditioned latent generation:

$$
p_\theta(z_{missing} \mid z_{known}, m, c)
$$

Mask merge:

$$
z = m \odot z_{known} + (1-m) \odot z_{generated}
$$

Composition operators:

```text
continue after this sound
fill a hole
bridge A to B
extend a loop
replace a section while preserving boundaries
```
"""
    ),
    code(
        r"""
# @title Mode 11. Continuation / inpainting composition

RUN_MODE_11_CONTINUATION = False

SOURCE_AUDIO = INPUT_DIR / "source.wav"
CONTINUATION_PROMPT = "continue this texture into a coherent evolving loop"
SOURCE_DURATION = 8.0
TARGET_CONTINUATION_DURATION = 16.0

if RUN_MODE_11_CONTINUATION:
    require_models()
    audio, sample_rate = load_audio(SOURCE_AUDIO, duration=SOURCE_DURATION, stereo=True)
    latents = sa3.continuation_latents(
        prompt=CONTINUATION_PROMPT,
        inpaint_audio=(sample_rate, audio),
        source_duration=SOURCE_DURATION,
        target_duration=TARGET_CONTINUATION_DURATION,
        steps=12,
        cfg_scale=1.0,
        seed=300,
    )
    out_path = OUTPUT_DIR / "mode_11_continuation.wav"
    decode_sa3_latents_to_file(latents, out_path)
    print("saved:", out_path)
"""
    ),
    md(
        r"""
## Mode 12. LatCH-Style Control Heads

LatCH-style sidecar:

$$
h_\psi(z) \rightarrow y
$$

where \(y\) can be:

```text
brightness
density
tension
loopability
section role
prompt match
```

The base model stays frozen. The control head learns to observe SAME latent properties.

Later, the head can become a guidance source:

$$
\mathcal{L}_{total}
= \mathcal{L}_{flow}
+ \lambda \mathcal{L}_{control}(h_\psi(z), y_{target})
$$
"""
    ),
    code(
        r"""
# @title Mode 12. Minimal LatCH-style control head on SAME summaries

RUN_MODE_12_CONTROL_HEAD = False

CONTROL_NAME = "brightness"


class LatentControlHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_control_head(items, control_name, *, epochs=200, lr=1e-3, device=DEVICE):
    examples = []
    labels = []
    for item in items:
        if control_name in item.descriptors:
            examples.append(latent_summary(item))
            labels.append(float(item.descriptors[control_name]))
        elif control_name in item.labels:
            examples.append(latent_summary(item))
            labels.append(float(item.labels[control_name]))
    if not examples:
        raise ValueError(f"No labels/descriptors found for control {control_name!r}.")
    x = torch.tensor(np.stack(examples), dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(labels)[:, None], dtype=torch.float32, device=device)
    model = LatentControlHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, losses


if RUN_MODE_12_CONTROL_HEAD:
    # Put controls in item.descriptors before this cell, e.g.
    # item.descriptors["brightness"] = 0.8
    items = load_items(MEMORY_DIR)
    head, losses = train_control_head(items, CONTROL_NAME)
    torch.save({"state_dict": head.state_dict(), "control": CONTROL_NAME}, OUTPUT_DIR / "mode_12_control_head.pt")
    print("initial/final loss:", losses[0], losses[-1])
"""
    ),
    md(
        r"""
## Mode 13. LoRA Adaptation Scaffold

LoRA changes weights through low-rank adapters:

$$
W' = W + \Delta W
$$

$$
\Delta W = BA
$$

with:

$$
A \in \mathbb{R}^{r \times d_{in}}, \qquad B \in \mathbb{R}^{d_{out} \times r}
$$

Use LoRA when you want a dataset/style/domain to become more native to generation.

Do not use it as the first answer for every control. Prompt inversion, SAME edits, and residual steering are better diagnostic probes.
"""
    ),
    code(
        r"""
# @title Mode 13. LoRA scaffold

RUN_MODE_13_LORA_SCAFFOLD = False

LORA_DATASET_DIR = DATASET_DIR
LORA_OUTPUT_DIR = OUTPUT_DIR / "mode_13_lora"

if RUN_MODE_13_LORA_SCAFFOLD:
    # This is a scaffold because the released SA3 LoRA script names/args should
    # be verified against the exact repo commit installed in Colab.
    # Inspect first:
    #   !find /content/sa3-native-lab -iname "*lora*" -o -iname "*train*"
    #
    # Desired training state:
    #   freeze base SA3/SAME weights
    #   train only LoRA adapter parameters
    #   log prompts, audio paths, duration, seed, loss curve
    #
    # Pseudocommand shape:
    command = [
        "python",
        str(Path(PROJECT_DIR) / "scripts" / "train_lora.py"),
        "--model", SA3_MODEL_NAME,
        "--data", str(LORA_DATASET_DIR),
        "--output", str(LORA_OUTPUT_DIR),
    ]
    print("Verify official script path/args before running:")
    print(" ".join(command))
"""
    ),
    md(
        r"""
## Mode 14. Latent Memory Instrument

Memory item:

```text
{
  latent: z,
  summary: s(z),
  prompt,
  descriptors,
  labels,
  audio path
}
```

Composition loop:

```text
retrieve -> continue -> steer -> decode -> re-encode -> store
```

This turns generated and encoded audio into reusable material.
"""
    ),
    code(
        r"""
# @title Mode 14. Latent memory instrument

RUN_MODE_14_LATENT_MEMORY = False

MEMORY_TEST_PROMPTS = [
    "a sparse glassy ambient loop, slow evolving texture",
    "a dense shimmering granular loop with bright high frequency motion",
    "a dark low drone with distant metallic harmonics",
]

if RUN_MODE_14_LATENT_MEMORY:
    require_models()
    items = []
    for prompt_index, prompt in enumerate(MEMORY_TEST_PROMPTS):
        for seed in [0, 1]:
            generated = sa3.generate_items(
                prompt=prompt,
                duration=8.0,
                item_id_prefix=f"p{prompt_index}_seed{seed}",
                steps=8,
                cfg_scale=1.0,
                seed=seed,
            )
            items.extend(generated)
    save_items(items, MEMORY_DIR)
    index = LatentMemoryIndex(items)
    query_item = items[0]
    results = index.query(query_item, top_k=5, exclude_id=query_item.item_id)
    for result in results:
        print(result.item_id, result.score, result.item.prompt)
"""
    ),
    md(
        r"""
## Combined Experiment: Audio-Derived Prompt + Residual Knob + SAME Style Push

A practical first research chain:

```text
target audio
  -> SAME latent
  -> soft prompt c*
  -> generate candidate latent
  -> residual steering alpha sweep
  -> SAME profile attraction
  -> decode
  -> store in latent memory
```

This combines:

```text
Mode 1: audio -> soft prompt
Mode 8/9: residual steering
Mode 6/7: SAME latent editing
Mode 14: memory
```
"""
    ),
    code(
        r"""
# @title Combined chain scaffold

RUN_COMBINED_CHAIN = False

if RUN_COMBINED_CHAIN:
    require_models()
    soft_state = SoftPromptState.load(SOFT_PROMPT_PATH)
    profile = load_style_profile(STYLE_PROFILE_PATH)
    vectors = SteeringVectors.load(PROMPT_VECTOR_DIR / "steering_vectors.pt")
    steerer = ResidualSteerer(sa3_model, vectors, top_k=1)

    with steerer.steer(alpha=1.0):
        latents = generate_with_soft_prompt(
            sa3_model,
            soft_state,
            steps=8,
            cfg_scale=1.0,
            seed=500,
            return_latents=True,
        )

    item = sa3_tensor_to_item(latents, item_id="combined_source", prompt="soft_prompt")
    styled_z = apply_profile_attraction(item, profile, alpha=0.5, match_std=True)
    styled_latents = torch.from_numpy(styled_z.T[None, :, :]).to(device=DEVICE, dtype=latents.dtype)
    out_path = OUTPUT_DIR / "combined_soft_residual_profile.wav"
    decode_sa3_latents_to_file(styled_latents, out_path)
    save_items([sa3_tensor_to_item(styled_latents, item_id="combined_final", prompt="soft+residual+profile")], MEMORY_DIR / "combined")
    print("saved:", out_path)
"""
    ),
    md(
        r"""
## Experiment Log Template

For each run, save:

```text
input audio paths
prompts
seeds
duration
steps
cfg scale
mode flags
vector layer ids
alpha values
loss curves
output audio paths
listening notes
failure modes
```

Minimum report:

```text
What changed?
Was the change audible?
Was it controllable with alpha/seed/prompt?
Did it generalize to another prompt or source?
Did SAME statistics move in the intended direction?
Did residual probe/loss values predict listening behavior?
```
"""
    ),
    code(
        r"""
# @title Save an experiment manifest

manifest = {
    "work_dir": str(WORK_DIR),
    "sa3_model": SA3_MODEL_NAME,
    "same_model": SAME_MODEL_NAME,
    "device": DEVICE,
    "modes": {
        "1_audio_to_soft_prompt": RUN_MODE_1_AUDIO_TO_SOFT_PROMPT,
        "2_audio_to_babble_prompt": RUN_MODE_2_AUDIO_TO_BABBLE_PROMPT,
        "3_audio_to_readable_prompt": RUN_MODE_3_AUDIO_TO_READABLE_PROMPT,
        "4_dataset_to_soft_prompt": RUN_MODE_4_DATASET_TO_SOFT_PROMPT,
        "5_dataset_to_prompt_family": RUN_MODE_5_DATASET_TO_PROMPT_FAMILY,
        "6_style_profile": RUN_MODE_6_STYLE_PROFILE,
        "7_same_direction": RUN_MODE_7_SAME_DIRECTION,
        "8_prompt_residual": RUN_MODE_8_PROMPT_RESIDUAL_STEERING,
        "9_audio_residual": RUN_MODE_9_AUDIO_RESIDUAL_STEERING,
        "10_flow_state_opt": RUN_MODE_10_FLOW_STATE_OPT,
        "11_continuation": RUN_MODE_11_CONTINUATION,
        "12_control_head": RUN_MODE_12_CONTROL_HEAD,
        "13_lora_scaffold": RUN_MODE_13_LORA_SCAFFOLD,
        "14_memory": RUN_MODE_14_LATENT_MEMORY,
        "combined_chain": RUN_COMBINED_CHAIN,
    },
}

manifest_path = OUTPUT_DIR / "experiment_manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print("saved:", manifest_path)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "L4"},
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
