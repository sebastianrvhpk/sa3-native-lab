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

0. Renoise variation / local neighborhood sampling
0c. Latent-selective renoise playground
0e. Cross-audio latent channel graft
0d. Latent blur playground
0f. Cyclic time-roll loop lab
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
| 0. Renoise variation | SAME latent neighborhood around \(E(x)\) | SDEdit/img2img-style noising, SA3 `init_audio` path |
| 0c. Latent-selective renoise | SAME channel subsets / masked init noise | latent traversal, representation probing, img2img-style variation |
| 0e. Cross-audio graft | SAME source channels mixed with donor channels | latent arithmetic, channel transplant, representation probing |
| 0d. Latent blur | SAME low-pass / low-rank / detail attenuation | representation smoothing, latent traversal, manifold projection |
| 0h. Neural latent DSP | SAME dynamics / FFT phase+gain / PCA component gain | neural signal processing, MIR auditing, manifold projection |
| 0f. Cyclic roll loop lab | time-rolled audio/SAME latents | tiling augmentation, circular optimization, SA3 inpainting |
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
from latent_audio_primitives.colab_audio_player import display_audio_player, search_audio_annotations
from latent_audio_primitives.adapters.stable_audio3 import (
    SAMEAutoencoderAdapter,
    StableAudio3Adapter,
    audio_chunk_windows,
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
from latent_audio_primitives.geometry import (
    covariance_transport,
    fit_latent_geometry,
    geometry_report,
    mahalanobis_summary_distance,
)
from latent_audio_primitives.periodic import periodicity_report
from latent_audio_primitives.observability import fit_linear_control_probe, predict_control
from latent_audio_primitives.prompt_optimization import (
    beam_token_prompt_search,
    coordinate_prompt_search,
    default_modifier_axes,
    greedy_token_prompt_search,
    prompt_seed_from_audio_path,
)
from latent_audio_primitives.schema import LatentItem
from latent_audio_primitives.latent_blur import (
    LatentBlurSpec,
    apply_latent_blur,
    sa3_sample_from_init_latents,
)
from latent_audio_primitives.latent_dsp import (
    LatentDSPSpec,
    apply_latent_dsp,
    latent_change_report,
)
from latent_audio_primitives.audio_descriptors import audio_descriptor_report, descriptor_delta
from latent_audio_primitives.looping import (
    cyclic_roll_audio,
    cyclic_roll_latents,
    frames_from_fraction,
    loop_boundary_metrics,
    repeated_loop_preview_audio,
    sa3_cyclic_roll_sample_from_init_latents,
    samples_from_fraction,
    seam_inpaint_bounds,
)
from latent_audio_primitives.selective_renoise import (
    LatentMaskSpec,
    graft_latent_channels,
    masked_latent_noise,
    select_latent_channels,
    selective_graft_sa3,
    selective_renoise_sa3,
)
from latent_audio_primitives.tokenizer_vocab import (
    native_tokenizer_vocabulary,
    preview_native_tokenizer_vocabulary,
)
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
        display_audio_player([smoke_path], title="SA3 Medium smoke test")
    except Exception as exc:
        print("Custom audio player skipped:", exc)
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


def decode_sa3_latents_to_file_cropped(latents, path, *, duration=None):
    require_models()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        audio = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
    if duration is not None:
        audio = audio[..., : int(round(duration * sa3.sample_rate))]
    torchaudio.save(str(path), audio[0], sa3.sample_rate)
    return path


def decode_sa3_latents_to_file_with_descriptors(latents, path, *, duration=None):
    path = decode_sa3_latents_to_file_cropped(latents, path, duration=duration)
    audio, sample_rate = load_audio(path, stereo=True)
    return path, audio_descriptor_report(audio, sample_rate)


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


def encode_audio_file_to_items(
    path,
    *,
    item_id=None,
    prompt=None,
    duration=None,
    use_chunks=False,
    chunk_duration=None,
    hop_duration=None,
    max_chunks=None,
    drop_last=False,
):
    require_models()
    path = Path(path)
    if use_chunks:
        chunk_duration = chunk_duration or duration
        if chunk_duration is None:
            raise ValueError("chunk_duration or duration is required when use_chunks=True.")
        return same.encode_file_chunks(
            path,
            chunk_duration=chunk_duration,
            hop_duration=hop_duration,
            max_chunks=max_chunks,
            drop_last=drop_last,
            item_id_prefix=item_id or path.stem,
            prompt=prompt,
            chunked=False,
            metadata={"path": str(path), "encoding_mode": "chunked_folder_window"},
        )
    return [encode_audio_file_to_item(path, item_id=item_id, prompt=prompt, duration=duration)]


def encode_audio_folder_to_items(
    directory,
    *,
    limit=None,
    duration=None,
    prompt_from_path=True,
    use_chunks=False,
    chunk_duration=None,
    hop_duration=None,
    max_chunks_per_file=None,
    drop_last=False,
):
    items = []
    for path in list_audio_files(directory, limit=limit):
        prompt = prompt_seed_from_audio_path(path) if prompt_from_path else None
        file_items = encode_audio_file_to_items(
            path,
            item_id=path.stem,
            prompt=prompt,
            duration=duration,
            use_chunks=use_chunks,
            chunk_duration=chunk_duration,
            hop_duration=hop_duration,
            max_chunks=max_chunks_per_file,
            drop_last=drop_last,
        )
        items.extend(file_items)
        print(f"{path.name}: {len(file_items)} item(s)")
    print(f"encoded {len(items)} latent item(s) from {directory}")
    return items


def preview_audio_folder_chunk_plan(
    directory,
    *,
    limit=None,
    chunk_duration=12.0,
    hop_duration=None,
    max_chunks_per_file=None,
    drop_last=False,
):
    if torchaudio is None:
        raise RuntimeError("torchaudio is required.")
    plan = []
    for path in list_audio_files(directory, limit=limit):
        info = torchaudio.info(str(path))
        windows = audio_chunk_windows(
            int(info.num_frames),
            int(info.sample_rate),
            chunk_duration,
            hop_duration=hop_duration,
            max_chunks=max_chunks_per_file,
            drop_last=drop_last,
        )
        plan.append(
            {
                "path": str(path),
                "source_seconds": round(info.num_frames / info.sample_rate, 3),
                "chunks": len(windows),
                "first_start": round(windows[0]["start_seconds"], 3) if windows else None,
                "last_start": round(windows[-1]["start_seconds"], 3) if windows else None,
            }
        )
    return plan


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
    score_samples=1,
    shared_noise=True,
    timestep_values=None,
    cosine_weight=0.0,
    antithetic_noise=False,
    normalize_mse=True,
    conditional_delta_weight=0.0,
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
    if timestep_values is not None:
        timestep_values = [float(value) for value in timestep_values]
        score_samples = len(timestep_values)
    else:
        score_samples = max(1, int(score_samples))
    cosine_weight = float(cosine_weight)
    conditional_delta_weight = float(conditional_delta_weight)

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

        null_cond_inputs = None
        if conditional_delta_weight:
            null_conditioning = [{"prompt": "", "seconds_total": duration} for _ in prompts]
            null_cond = core.conditioner(null_conditioning, device)
            null_cond = dict(null_cond)
            batch, channels, frames = target.shape
            null_cond["inpaint_mask"] = [torch.zeros((batch, 1, frames), device=device)]
            null_cond["inpaint_masked_input"] = [
                torch.zeros((batch, channels, frames), device=device, dtype=dtype)
            ]
            null_cond_inputs = core.get_conditioning_inputs(null_cond)
            null_cond_inputs = {
                key: value.to(dtype) if isinstance(value, torch.Tensor) and value.is_floating_point() else value
                for key, value in null_cond_inputs.items()
            }

        losses = torch.zeros((batch,), device=device, dtype=torch.float32)
        loss_terms = 0
        for sample_index in range(score_samples):
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed) + sample_index * 1009)
            if shared_noise:
                if timestep_values is not None:
                    t_scalar = torch.tensor(timestep_values[sample_index], device=device)
                else:
                    t_scalar = min_t + (max_t - min_t) * torch.rand((), device=device, generator=generator)
                t = t_scalar.expand(batch)
                noise_base = torch.randn((1, channels, frames), device=device, dtype=dtype, generator=generator)
                noise = noise_base.expand(batch, -1, -1).contiguous()
            else:
                if timestep_values is not None:
                    t = torch.full((batch,), timestep_values[sample_index], device=device)
                else:
                    t = min_t + (max_t - min_t) * torch.rand(batch, device=device, generator=generator)
                noise = torch.randn(target.shape, device=device, dtype=dtype, generator=generator)

            noise_signs = [1.0, -1.0] if antithetic_noise else [1.0]
            for noise_sign in noise_signs:
                signed_noise = noise * noise_sign
                t_view = t[:, None, None].to(dtype)
                z_t = (1 - t_view) * target + t_view * signed_noise
                velocity_target = flow_velocity_target(target, signed_noise, convention=velocity_convention)
                pred = core.model(z_t, t, **cond_inputs, cfg_scale=1.0, batch_cfg=True)
                residual = pred.float() - velocity_target.float()
                mse = torch.mean(residual ** 2, dim=(1, 2))
                if normalize_mse:
                    target_scale = torch.mean(velocity_target.float() ** 2, dim=(1, 2)).clamp_min(1e-8)
                    mse = mse / target_scale
                if cosine_weight:
                    pred_flat = pred.float().reshape(batch, -1)
                    target_flat = velocity_target.float().reshape(batch, -1)
                    cosine = torch.nn.functional.cosine_similarity(pred_flat, target_flat, dim=1, eps=1e-8)
                    mse = mse + cosine_weight * (1.0 - cosine)
                if conditional_delta_weight and null_cond_inputs is not None:
                    null_pred = core.model(z_t, t, **null_cond_inputs, cfg_scale=1.0, batch_cfg=True)
                    delta = (pred.float() - null_pred.float()).reshape(batch, -1)
                    wanted_delta = (velocity_target.float() - null_pred.float()).reshape(batch, -1)
                    delta_cosine = torch.nn.functional.cosine_similarity(
                        delta,
                        wanted_delta,
                        dim=1,
                        eps=1e-8,
                    )
                    mse = mse + conditional_delta_weight * (1.0 - delta_cosine)
                losses = losses + mse
                loss_terms += 1
        losses = losses / max(loss_terms, 1)
    return [float(loss.detach().cpu()) for loss in losses]


def timesteps_from_logsnr_values(logsnr_values):
    if not logsnr_values:
        return []
    return [float(1.0 / (1.0 + math.exp(float(value)))) for value in logsnr_values]


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
## Custom Colab Audio Player

The notebook uses a small self-contained Web Audio/canvas player instead of `IPython.display.Audio` for generated outputs.

It supports:

```text
playlist comparison
waveform seek
track loop
loop-region in/out points
volume and speed
keyboard shortcuts after clicking the player
```

The audio is embedded into the notebook output as base64, so keep playlists short when files are long.
"""
    ),
    code(
        r"""
# @title Audio player helper

def play_audio_files(
    paths,
    *,
    labels=None,
    metadata=None,
    title="Audio Player",
    peak_count=900,
    max_embed_mb=96.0,
    annotation_path=None,
):
    return display_audio_player(
        paths,
        labels=labels,
        metadata=metadata,
        title=title,
        peak_count=peak_count,
        max_embed_mb=max_embed_mb,
        annotation_path=annotation_path,
    )
"""
    ),
    md(
        r"""
## Dataset / Long-File Input Policy

For folders of long recordings, set `DATASET_USE_CHUNKS=True`.

Without chunking, each file is encoded as one item and `DATASET_DURATION=8.0` means "use the first 8 seconds of each file." With chunking, each long file becomes many SAME latent items:

$$
x_{file} \rightarrow \{x_{[0,L]}, x_{[H,H+L]}, x_{[2H,2H+L]}, \ldots\}
$$

and then:

$$
z_i = E(x_i)
$$

This is the right default for DJ sets, live sets, mixtapes, stems, and archives where the interesting distribution is spread across time.
"""
    ),
    code(
        r"""
# @title Dataset / long-file chunking controls

DATASET_DIR = INPUT_DIR / "dataset_positive"
REFERENCE_DIR = INPUT_DIR / "dataset_reference"

DATASET_DURATION = 8.0
DATASET_LIMIT = 2  # number of files before chunking; raise after the pipeline works

DATASET_USE_CHUNKS = True
DATASET_CHUNK_DURATION = 12.0
DATASET_HOP_DURATION = 12.0  # use 6.0 for 50% overlap on 12s chunks
DATASET_MAX_CHUNKS_PER_FILE = 4  # L4-safe default for optimization modes
DATASET_DROP_LAST_CHUNK = False

RUN_DATASET_CHUNK_PREVIEW = False


def dataset_effective_duration():
    return DATASET_CHUNK_DURATION if DATASET_USE_CHUNKS else DATASET_DURATION


if RUN_DATASET_CHUNK_PREVIEW:
    plan = preview_audio_folder_chunk_plan(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
    )
    print(json.dumps(plan, indent=2))
"""
    ),
    md(
        r"""
## Mode 0. Renoise Variation / Local Neighborhood Sampling

This is the Tero-style loop neighborhood primitive.

Encode an existing loop as a SAME latent, partially renoise it, then let SA3 flow it back to the data manifold:

$$
z_0 = E(x)
$$

$$
z_\sigma \approx (1-\sigma) z_0 + \sigma \epsilon,\qquad
\epsilon \sim \mathcal{N}(0,I)
$$

$$
\tilde z = \operatorname{sample}_\theta(z_\sigma, c=\emptyset)
$$

$$
\tilde x = D(\tilde z)
$$

`init_noise_level` is the local-neighborhood radius:

```text
0.10-0.30: conservative timbral/performance variants
0.40-0.60: neighborhood variants, often keeps harmony/feel
0.70-1.00: increasingly free regeneration
```

This is not text steering. It is local manifold resampling around an audio memory.
"""
    ),
    code(
        r"""
# @title Mode 0. Renoise variations from an existing loop/audio

RUN_MODE_0_RENOISE_VARIATIONS = False

VARIATION_AUDIO = INPUT_DIR / "loop.wav"
VARIATION_OUTPUT_DIR = OUTPUT_DIR / "mode_00_renoise_variations"
VARIATION_MEMORY_DIR = MEMORY_DIR / "mode_00_renoise_variations"

VARIATION_PROMPT = ""  # Empty string + cfg_scale=1.0 is a practical "no semantic prompt" baseline.
VARIATION_DURATION = 8.0
VARIATION_STEPS = 8
VARIATION_CFG_SCALE = 1.0
VARIATION_NOISE_LEVELS = [0.25, 0.5, 0.75]
VARIATION_SEEDS = [0, 1, 2]

if RUN_MODE_0_RENOISE_VARIATIONS:
    require_models()
    VARIATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = load_audio(VARIATION_AUDIO, duration=VARIATION_DURATION, stereo=True)
    items = []
    manifest = []

    for sigma in VARIATION_NOISE_LEVELS:
        for seed in VARIATION_SEEDS:
            latents = sa3.generate_latents(
                prompt=VARIATION_PROMPT,
                duration=VARIATION_DURATION,
                steps=VARIATION_STEPS,
                cfg_scale=VARIATION_CFG_SCALE,
                seed=seed,
                init_audio=(sample_rate, audio),
                init_noise_level=float(sigma),
            )
            item_id = f"renoise_sigma_{sigma:.2f}_seed_{seed}"
            out_path = VARIATION_OUTPUT_DIR / f"{item_id}.wav"
            decode_sa3_latents_to_file(latents, out_path)
            item = sa3_tensor_to_item(
                latents,
                item_id=item_id,
                prompt=VARIATION_PROMPT,
                metadata={
                    "source_audio": str(VARIATION_AUDIO),
                    "init_noise_level": float(sigma),
                    "seed": int(seed),
                    "duration": VARIATION_DURATION,
                    "steps": VARIATION_STEPS,
                    "cfg_scale": VARIATION_CFG_SCALE,
                },
            )
            items.append(item)
            manifest.append({"path": str(out_path), "init_noise_level": float(sigma), "seed": int(seed)})
            print("saved:", out_path)

    save_items(items, VARIATION_MEMORY_DIR)
    (VARIATION_OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved latent memory:", VARIATION_MEMORY_DIR)
    try:
        display_audio_player(
            [entry["path"] for entry in manifest],
            labels=[f"sigma {entry['init_noise_level']:.2f} seed {entry['seed']}" for entry in manifest],
            title="Mode 0 renoise variations",
        )
    except Exception as exc:
        print("Custom audio player skipped:", exc)
"""
    ),
    md(
        r"""
## Mode 0c. Latent-Selective Renoise Playground

This is for one short source you know well, e.g. a 9-second loop.

Instead of renoising every latent channel equally:

$$
z_\sigma = (1-\sigma)z_0 + \sigma\epsilon
$$

select a subset of SAME channels:

$$
S \subset \{0,\dots,255\}
$$

and perturb only those channels at the sampler start:

$$
\epsilon_S =
\begin{cases}
\epsilon_c, & c \in S \\
z_{0,c}, & c \notin S
\end{cases}
$$

SA3 then starts from:

$$
z_{start} = (1-\sigma)z_0 + \sigma \epsilon_S
$$

Unselected channels are not mathematically frozen during denoising; they simply start unchanged. That is enough for a first probe of emergent channel structure.

Useful listening questions:

```text
Which masks preserve rhythm?
Which masks change timbre?
Which masks change stereo/space?
Which masks create artifacts?
Are effects reproducible for the same channel set?
Do high-variance and low-variance channels form different families?
```
"""
    ),
    code(
        r"""
# @title Mode 0c. Latent-selective renoise playground

RUN_MODE_0C_LATENT_SELECTIVE_RENOISE = False

LATENT_PLAYGROUND_AUDIO = INPUT_DIR / "known_9s.wav"
LATENT_PLAYGROUND_OUTPUT_DIR = OUTPUT_DIR / "mode_00c_latent_selective_renoise"
LATENT_PLAYGROUND_DURATION = 9.0
LATENT_PLAYGROUND_PROMPT = ""
LATENT_PLAYGROUND_STEPS = 8
LATENT_PLAYGROUND_CFG = 1.0
LATENT_PLAYGROUND_NOISE = 0.40
LATENT_PLAYGROUND_BASE_SEED = 700

# Direct decode is cheap and off-manifold: useful as a microscope for SAME channels,
# but it usually sounds like damaged latent audio. Keep it off for large listening sweeps.
LATENT_PLAYGROUND_RUN_DIRECT_DECODE = False
LATENT_PLAYGROUND_RUN_SA3_SAMPLER = True
LATENT_PLAYGROUND_SAVE_PT = True
LATENT_PLAYGROUND_ANNOTATION_PATH = LATENT_PLAYGROUND_OUTPUT_DIR / "annotations.json"

# Deeper mapping sweeps. Keep these false for a first smoke run.
LATENT_PLAYGROUND_ADD_BLOCK_SWEEP = False
LATENT_PLAYGROUND_BLOCK_SIZE = 8
LATENT_PLAYGROUND_BLOCK_STRIDE = 8
LATENT_PLAYGROUND_MAX_BLOCKS = 32
LATENT_PLAYGROUND_ADD_RANDOM_SWEEP = False
LATENT_PLAYGROUND_RANDOM_FRACTIONS = [0.05, 0.10, 0.20]
LATENT_PLAYGROUND_RANDOM_MASK_SEEDS = list(range(8))
LATENT_PLAYGROUND_MAX_VARIANTS = 96

LATENT_PLAYGROUND_SPECS = [
    {"name": "random_10_s0", "mode": "random_channels", "fraction": 0.10, "seed": 0},
    {"name": "random_10_s1", "mode": "random_channels", "fraction": 0.10, "seed": 1},
    {"name": "random_25_s0", "mode": "random_channels", "fraction": 0.25, "seed": 0},
    {"name": "high_var_10", "mode": "high_variance", "fraction": 0.10, "seed": 0},
    {"name": "low_var_10", "mode": "low_variance", "fraction": 0.10, "seed": 0},
    {"name": "high_activity_10", "mode": "high_activity", "fraction": 0.10, "seed": 0},
    {"name": "block_000_16", "mode": "channel_block", "block_size": 16, "start_channel": 0},
    {"name": "block_128_16", "mode": "channel_block", "block_size": 16, "start_channel": 128},
]


def expand_latent_playground_specs(base_specs, channel_count):
    specs = list(base_specs)
    if LATENT_PLAYGROUND_ADD_BLOCK_SWEEP:
        starts = range(0, channel_count, LATENT_PLAYGROUND_BLOCK_STRIDE)
        for block_index, start in enumerate(starts):
            if block_index >= LATENT_PLAYGROUND_MAX_BLOCKS:
                break
            specs.append(
                {
                    "name": f"block_{start:03d}_{LATENT_PLAYGROUND_BLOCK_SIZE}",
                    "mode": "channel_block",
                    "block_size": LATENT_PLAYGROUND_BLOCK_SIZE,
                    "start_channel": start,
                }
            )
    if LATENT_PLAYGROUND_ADD_RANDOM_SWEEP:
        for fraction in LATENT_PLAYGROUND_RANDOM_FRACTIONS:
            for seed in LATENT_PLAYGROUND_RANDOM_MASK_SEEDS:
                percent = int(round(100 * fraction))
                specs.append(
                    {
                        "name": f"random_{percent:02d}_s{seed}",
                        "mode": "random_channels",
                        "fraction": float(fraction),
                        "seed": int(seed),
                    }
                )
    deduped = []
    seen = set()
    for spec in specs:
        name = spec["name"]
        if name not in seen:
            deduped.append(spec)
            seen.add(name)
    return deduped[:LATENT_PLAYGROUND_MAX_VARIANTS]


def latent_mask_spec_from_dict(payload):
    return LatentMaskSpec(
        name=payload["name"],
        mode=payload.get("mode", "random_channels"),
        fraction=float(payload.get("fraction", 0.25)),
        seed=int(payload.get("seed", 0)),
        channels=tuple(payload["channels"]) if payload.get("channels") is not None else None,
        start_channel=payload.get("start_channel"),
        block_size=payload.get("block_size"),
    )


def safe_audio_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:96]


if RUN_MODE_0C_LATENT_SELECTIVE_RENOISE:
    require_models()
    LATENT_PLAYGROUND_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_audio, source_sr = load_audio(
        LATENT_PLAYGROUND_AUDIO,
        duration=LATENT_PLAYGROUND_DURATION,
        stereo=True,
    )
    source_item = encode_audio_file_to_item(
        LATENT_PLAYGROUND_AUDIO,
        item_id="latent_playground_source",
        duration=LATENT_PLAYGROUND_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    player_paths = []
    player_labels = []
    player_metadata = []
    manifest = []
    source_path = LATENT_PLAYGROUND_OUTPUT_DIR / "source_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)
    player_paths.append(source_path)
    player_labels.append("source reference")
    player_metadata.append({"kind": "source_reference", "path": str(source_path)})

    expanded_specs = expand_latent_playground_specs(
        LATENT_PLAYGROUND_SPECS,
        channel_count=int(source_latents.shape[1]),
    )
    print("variants:", len(expanded_specs))

    for variant_index, spec_dict in enumerate(expanded_specs):
        spec = latent_mask_spec_from_dict(spec_dict)
        name = safe_audio_name(spec.name)
        seed = LATENT_PLAYGROUND_BASE_SEED + variant_index
        channels = select_latent_channels(source_latents, spec)
        sampler_spec = LatentMaskSpec(
            name=spec.name,
            mode=spec.mode,
            fraction=spec.fraction,
            seed=spec.seed,
            channels=tuple(channels),
            start_channel=spec.start_channel,
            block_size=spec.block_size,
        )
        entry = {
            "name": spec.name,
            "mode": spec.mode,
            "fraction": spec.fraction,
            "seed": spec.seed,
            "sampler_seed": seed,
            "noise_level": LATENT_PLAYGROUND_NOISE,
            "selected_channel_count": len(channels),
            "selected_channels": channels,
            "outputs": {},
        }

        if LATENT_PLAYGROUND_RUN_DIRECT_DECODE:
            mixed_latents = masked_latent_noise(
                source_latents,
                channels,
                sigma=LATENT_PLAYGROUND_NOISE,
                seed=seed,
            )
            direct_path = LATENT_PLAYGROUND_OUTPUT_DIR / f"direct_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                mixed_latents,
                direct_path,
                duration=LATENT_PLAYGROUND_DURATION,
            )
            entry["outputs"]["direct_same_decode"] = str(direct_path)
            player_paths.append(direct_path)
            player_labels.append(f"direct {spec.name} ({len(channels)} ch)")
            player_metadata.append(
                {
                    "kind": "direct_same_decode",
                    "variant": spec.name,
                    "mask_mode": spec.mode,
                    "noise_level": LATENT_PLAYGROUND_NOISE,
                    "selected_channel_count": len(channels),
                    "selected_channels": channels,
                    "sampler_seed": seed,
                    "path": str(direct_path),
                }
            )

        if LATENT_PLAYGROUND_RUN_SA3_SAMPLER:
            result = selective_renoise_sa3(
                sa3_model,
                source_audio,
                source_sr,
                spec=sampler_spec,
                prompt=LATENT_PLAYGROUND_PROMPT,
                duration=LATENT_PLAYGROUND_DURATION,
                steps=LATENT_PLAYGROUND_STEPS,
                cfg_scale=LATENT_PLAYGROUND_CFG,
                init_noise_level=LATENT_PLAYGROUND_NOISE,
                seed=seed,
            )
            sampled_path = LATENT_PLAYGROUND_OUTPUT_DIR / f"sa3_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                result.sampled_latents,
                sampled_path,
                duration=LATENT_PLAYGROUND_DURATION,
            )
            entry["outputs"]["sa3_sampled"] = str(sampled_path)
            player_paths.append(sampled_path)
            player_labels.append(f"SA3 {spec.name} ({len(channels)} ch)")
            player_metadata.append(
                {
                    "kind": "sa3_sampled",
                    "variant": spec.name,
                    "mask_mode": spec.mode,
                    "noise_level": LATENT_PLAYGROUND_NOISE,
                    "selected_channel_count": len(channels),
                    "selected_channels": channels,
                    "sampler_seed": seed,
                    "path": str(sampled_path),
                }
            )
            entry.update(result.metadata)
            if LATENT_PLAYGROUND_SAVE_PT:
                pt_path = LATENT_PLAYGROUND_OUTPUT_DIR / f"{name}.pt"
                torch.save(
                    {
                        "init_latents": result.init_latents.detach().cpu(),
                        "mixed_latents": result.mixed_latents.detach().cpu(),
                        "sampled_latents": result.sampled_latents.detach().cpu(),
                        "metadata": result.metadata,
                    },
                    pt_path,
                )
                entry["outputs"]["pt"] = str(pt_path)

        manifest.append(entry)
        print(spec.name, "channels:", len(channels), channels[:24])

    manifest_path = LATENT_PLAYGROUND_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        metadata=player_metadata,
        title="Mode 0c latent-selective renoise playground",
        max_embed_mb=160.0,
        annotation_path=LATENT_PLAYGROUND_ANNOTATION_PATH,
    )
"""
    ),
    md(
        r"""
## Mode 0c Annotation Retrieval

When the Mode 0c player is shown in Colab, each track has a small annotation panel:

```text
rating      numeric score, e.g. 0-5 or -5..5
tags        comma-separated labels such as drums, smear, keeper, bad-transient
use/value   a short role like loop, transition, texture, break
description free listening notes
```

The player saves these into:

```python
OUTPUT_DIR / "mode_00c_latent_selective_renoise" / "annotations.json"
```

This turns listening into a retrieval layer over SA3 interventions: you can search notes/tags,
re-open the best files, and inspect the metadata for the corresponding channel mask.
"""
    ),
    code(
        r"""
# @title Mode 0c. Search annotated variations

RUN_MODE_0C_SEARCH_ANNOTATIONS = False

ANNOTATION_SEARCH_PATH = OUTPUT_DIR / "mode_00c_latent_selective_renoise" / "annotations.json"
ANNOTATION_QUERY = ""
ANNOTATION_TAGS = []  # Example: ["keeper", "drums"]
ANNOTATION_MIN_RATING = None  # Example: 4.0
ANNOTATION_LIMIT = 24

if RUN_MODE_0C_SEARCH_ANNOTATIONS:
    matches = search_audio_annotations(
        ANNOTATION_SEARCH_PATH,
        query=ANNOTATION_QUERY,
        tags=ANNOTATION_TAGS,
        min_rating=ANNOTATION_MIN_RATING,
        limit=ANNOTATION_LIMIT,
    )
    print("matches:", len(matches))
    for item in matches:
        meta = item.get("metadata", {})
        print(
            item.get("rating"),
            item.get("tags"),
            item.get("label"),
            "variant=", meta.get("variant"),
            "kind=", meta.get("kind"),
            "channels=", meta.get("selected_channels", [])[:12],
        )
    if matches:
        play_audio_files(
            [item["path"] for item in matches],
            labels=[
                f"{item.get('rating')} | {', '.join(item.get('tags', []))} | {item.get('label')}"
                for item in matches
            ],
            metadata=[item.get("metadata", {}) for item in matches],
            title="Mode 0c annotated retrieval",
            max_embed_mb=160.0,
            annotation_path=ANNOTATION_SEARCH_PATH,
        )
"""
    ),
    md(
        r"""
## Mode 0e. Cross-Audio Latent Channel Graft

This is the donor-audio version of Mode 0c.

Instead of putting Gaussian noise into selected source channels, use another encoded audio file as the replacement direction:

$$
z^{noise}_{c,t} =
\begin{cases}
z^{donor}_{c,t}, & c \in S \\
z^{source}_{c,t}, & c \notin S
\end{cases}
$$

SA3 then starts from:

$$
z_{start} = (1-\alpha)z^{source} + \alpha z^{noise}
$$

so selected channels become:

$$
z_{start,c,t} = z^{source}_{c,t} + \alpha(z^{donor}_{c,t} - z^{source}_{c,t})
$$

and unselected channels remain exactly the source at sampler start.

Interpretation:

```text
direct graft decode = literal source/donor channel splice through SAME
SA3 graft          = source/donor splice reprojected by SA3's learned music prior
alpha              = graft amount, analogous to init_noise_level
```

This is closer to latent transplantation than variation. It is useful for asking whether a SAME channel block carries portable rhythmic, timbral, spatial, or textural information across two audios.
"""
    ),
    code(
        r"""
# @title Mode 0e. Cross-audio latent channel graft

RUN_MODE_0E_CROSS_AUDIO_GRAFT = False

LATENT_GRAFT_SOURCE_AUDIO = INPUT_DIR / "known_9s.wav"
LATENT_GRAFT_DONOR_AUDIO = INPUT_DIR / "donor_9s.wav"
LATENT_GRAFT_OUTPUT_DIR = OUTPUT_DIR / "mode_00e_cross_audio_graft"
LATENT_GRAFT_DURATION = 9.0
LATENT_GRAFT_PROMPT = ""
LATENT_GRAFT_STEPS = 8
LATENT_GRAFT_CFG = 1.0
LATENT_GRAFT_AMOUNT = 0.40
LATENT_GRAFT_BASE_SEED = 880

LATENT_GRAFT_RUN_DIRECT_DECODE = False
LATENT_GRAFT_RUN_SA3_SAMPLER = True
LATENT_GRAFT_SAVE_PT = True
LATENT_GRAFT_ANNOTATION_PATH = LATENT_GRAFT_OUTPUT_DIR / "annotations.json"

LATENT_GRAFT_SPECS = [
    {"name": "random_10_s0", "mode": "random_channels", "fraction": 0.10, "seed": 0},
    {"name": "random_10_s1", "mode": "random_channels", "fraction": 0.10, "seed": 1},
    {"name": "random_25_s0", "mode": "random_channels", "fraction": 0.25, "seed": 0},
    {"name": "high_var_10", "mode": "high_variance", "fraction": 0.10, "seed": 0},
    {"name": "high_activity_10", "mode": "high_activity", "fraction": 0.10, "seed": 0},
    {"name": "block_000_16", "mode": "channel_block", "block_size": 16, "start_channel": 0},
    {"name": "block_064_16", "mode": "channel_block", "block_size": 16, "start_channel": 64},
    {"name": "block_128_16", "mode": "channel_block", "block_size": 16, "start_channel": 128},
    {"name": "block_192_16", "mode": "channel_block", "block_size": 16, "start_channel": 192},
]


def graft_mask_spec_from_dict(payload):
    return LatentMaskSpec(
        name=payload["name"],
        mode=payload.get("mode", "random_channels"),
        fraction=float(payload.get("fraction", 0.25)),
        seed=int(payload.get("seed", 0)),
        channels=tuple(payload["channels"]) if payload.get("channels") is not None else None,
        start_channel=payload.get("start_channel"),
        block_size=payload.get("block_size"),
    )


def safe_graft_audio_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:96]


if RUN_MODE_0E_CROSS_AUDIO_GRAFT:
    require_models()
    LATENT_GRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_audio, source_sr = load_audio(
        LATENT_GRAFT_SOURCE_AUDIO,
        duration=LATENT_GRAFT_DURATION,
        stereo=True,
    )
    donor_audio, donor_sr = load_audio(
        LATENT_GRAFT_DONOR_AUDIO,
        duration=LATENT_GRAFT_DURATION,
        stereo=True,
    )
    source_item = encode_audio_file_to_item(
        LATENT_GRAFT_SOURCE_AUDIO,
        item_id="latent_graft_source",
        duration=LATENT_GRAFT_DURATION,
    )
    donor_item = encode_audio_file_to_item(
        LATENT_GRAFT_DONOR_AUDIO,
        item_id="latent_graft_donor",
        duration=LATENT_GRAFT_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )
    donor_latents = item_to_sa3_tensor(
        donor_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    player_paths = []
    player_labels = []
    player_metadata = []
    manifest = []

    source_path = LATENT_GRAFT_OUTPUT_DIR / "source_reference.wav"
    donor_path = LATENT_GRAFT_OUTPUT_DIR / "donor_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)
    torchaudio.save(str(donor_path), donor_audio.cpu(), donor_sr)
    player_paths.extend([source_path, donor_path])
    player_labels.extend(["source reference", "donor reference"])
    player_metadata.extend(
        [
            {"kind": "source_reference", "path": str(source_path)},
            {"kind": "donor_reference", "path": str(donor_path)},
        ]
    )

    for variant_index, spec_dict in enumerate(LATENT_GRAFT_SPECS):
        spec = graft_mask_spec_from_dict(spec_dict)
        name = safe_graft_audio_name(spec.name)
        seed = LATENT_GRAFT_BASE_SEED + variant_index
        channels = select_latent_channels(source_latents, spec)
        sampler_spec = LatentMaskSpec(
            name=spec.name,
            mode=spec.mode,
            fraction=spec.fraction,
            seed=spec.seed,
            channels=tuple(channels),
            start_channel=spec.start_channel,
            block_size=spec.block_size,
        )
        entry = {
            "name": spec.name,
            "mode": spec.mode,
            "fraction": spec.fraction,
            "seed": spec.seed,
            "sampler_seed": seed,
            "graft_amount": LATENT_GRAFT_AMOUNT,
            "selected_channel_count": len(channels),
            "selected_channels": channels,
            "source_audio": str(LATENT_GRAFT_SOURCE_AUDIO),
            "donor_audio": str(LATENT_GRAFT_DONOR_AUDIO),
            "outputs": {},
        }

        if LATENT_GRAFT_RUN_DIRECT_DECODE:
            grafted_latents = graft_latent_channels(
                source_latents,
                donor_latents,
                channels,
                amount=LATENT_GRAFT_AMOUNT,
            )
            direct_path = LATENT_GRAFT_OUTPUT_DIR / f"direct_graft_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                grafted_latents,
                direct_path,
                duration=LATENT_GRAFT_DURATION,
            )
            entry["outputs"]["direct_same_decode"] = str(direct_path)
            player_paths.append(direct_path)
            player_labels.append(f"direct graft {spec.name} ({len(channels)} ch)")
            player_metadata.append(
                {
                    "kind": "direct_same_graft",
                    "variant": spec.name,
                    "mask_mode": spec.mode,
                    "graft_amount": LATENT_GRAFT_AMOUNT,
                    "selected_channel_count": len(channels),
                    "selected_channels": channels,
                    "sampler_seed": seed,
                    "source_audio": str(LATENT_GRAFT_SOURCE_AUDIO),
                    "donor_audio": str(LATENT_GRAFT_DONOR_AUDIO),
                    "path": str(direct_path),
                }
            )

        if LATENT_GRAFT_RUN_SA3_SAMPLER:
            result = selective_graft_sa3(
                sa3_model,
                source_audio,
                source_sr,
                donor_audio,
                donor_sr,
                spec=sampler_spec,
                prompt=LATENT_GRAFT_PROMPT,
                duration=LATENT_GRAFT_DURATION,
                steps=LATENT_GRAFT_STEPS,
                cfg_scale=LATENT_GRAFT_CFG,
                init_noise_level=LATENT_GRAFT_AMOUNT,
                seed=seed,
            )
            sampled_path = LATENT_GRAFT_OUTPUT_DIR / f"sa3_graft_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                result.sampled_latents,
                sampled_path,
                duration=LATENT_GRAFT_DURATION,
            )
            entry["outputs"]["sa3_grafted"] = str(sampled_path)
            player_paths.append(sampled_path)
            player_labels.append(f"SA3 graft {spec.name} ({len(channels)} ch)")
            player_metadata.append(
                {
                    "kind": "sa3_grafted",
                    "variant": spec.name,
                    "mask_mode": spec.mode,
                    "graft_amount": LATENT_GRAFT_AMOUNT,
                    "selected_channel_count": len(channels),
                    "selected_channels": channels,
                    "sampler_seed": seed,
                    "source_audio": str(LATENT_GRAFT_SOURCE_AUDIO),
                    "donor_audio": str(LATENT_GRAFT_DONOR_AUDIO),
                    "path": str(sampled_path),
                }
            )
            entry.update(result.metadata)

            if LATENT_GRAFT_SAVE_PT:
                pt_path = LATENT_GRAFT_OUTPUT_DIR / f"{name}.pt"
                torch.save(
                    {
                        "source_latents": result.init_latents.detach().cpu(),
                        "donor_latents": result.donor_latents.detach().cpu(),
                        "mixed_latents": result.mixed_latents.detach().cpu(),
                        "sampled_latents": result.sampled_latents.detach().cpu(),
                        "metadata": result.metadata,
                    },
                    pt_path,
                )
                entry["outputs"]["pt"] = str(pt_path)

        manifest.append(entry)
        print(spec.name, "channels:", len(channels), channels[:24])

    manifest_path = LATENT_GRAFT_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        metadata=player_metadata,
        title="Mode 0e cross-audio latent graft",
        max_embed_mb=180.0,
        annotation_path=LATENT_GRAFT_ANNOTATION_PATH,
    )
"""
    ),
    md(
        r"""
## Mode 0d. Latent Blur Playground

Latent blur asks: what happens if we remove or smooth detail inside SAME space before decoding?

Temporal blur:

$$
z'_{c,t} = \sum_{\tau} K_\sigma(\tau) z_{c,t-\tau}
$$

Contiguous-frame box blur, closer to video blur:

$$
z'_{c,t} = \frac{1}{2r + 1}\sum_{k=-r}^{r} z_{c,t+k}
$$

One-sided motion blur:

$$
z'_{c,t} = \frac{1}{r + 1}\sum_{k=0}^{r} z_{c,t-k}
$$

or the same window looking forward in latent time. At SA3/SAME's approximate rate,
\(r=4\) means a centered window of \(9\) latent frames, roughly \(0.84\) seconds.

Channel blur:

$$
z'_{c,t} = \sum_{j} K_\sigma(j) z_{c-j,t}
$$

Low-rank blur:

$$
Z_{T \times C} \approx U_k \Sigma_k V_k^\top
$$

Detail attenuation:

$$
z' = \operatorname{blur}(z) + \gamma (z - \operatorname{blur}(z))
$$

where \(\gamma < 1\) suppresses latent detail residuals.

Latent sharpen / unsharp mask:

$$
z' = z + \beta (z - \operatorname{blur}(z))
$$

where \(\beta > 0\) amplifies detail residuals. This can emphasize transient-like or high-activity latent structure, but large values may push the latent off-manifold.

Latent FFT filters:

$$
\hat z'_{c,\omega} = g(\omega)\hat z_{c,\omega}
$$

where the FFT is over SAME latent frames, not decoded audio samples. This is a filter over latent-channel trajectories. It can damp fast latent-frame jitter before SAME decode, but it is not equivalent to an audio EQ.

Interpretation warnings:

```text
direct SAME decode = microscope, may be off-manifold or artifacty
SA3 polish        = blurred latent used as init_data with small noise, more manifold-like but less pure
channel blur      = probes whether channel index has structure; it may not
channel sharpen   = also a probe; channel adjacency is not guaranteed semantic
low-rank blur     = probes global vs fine latent components
FFT filter        = latent-time filter, not decoded-audio EQ
```
"""
    ),
    code(
        r"""
# @title Mode 0d. Latent blur playground

RUN_MODE_0D_LATENT_BLUR = False

LATENT_BLUR_AUDIO = INPUT_DIR / "known_9s.wav"
LATENT_BLUR_OUTPUT_DIR = OUTPUT_DIR / "mode_00d_latent_blur"
LATENT_BLUR_DURATION = 9.0
LATENT_BLUR_PROMPT = ""
LATENT_BLUR_STEPS = 20
LATENT_BLUR_CFG = 1.0
LATENT_BLUR_POLISH_NOISE = 0.06
LATENT_BLUR_BASE_SEED = 900

# Main harshness knobs:
# - temporal_radius: r=1 is 3 latent frames (~0.28s), r=2 is 5 frames (~0.46s),
#   r=4 is 9 frames (~0.84s), r=8 is 17 frames (~1.58s).
# - strength: 0.0 is no edit, 0.10-0.35 is gentle, 1.0 is full replacement by the blurred latent.
# - LATENT_BLUR_POLISH_NOISE: 0.03-0.08 is light SA3 projection, 0.12+ starts behaving more like regeneration.
# - LATENT_BLUR_STEPS: 8 is fast/draft, 20 is usually much cleaner for off-manifold edits.
LATENT_BLUR_RUN_DIRECT_DECODE = False
LATENT_BLUR_RUN_SA3_POLISH = True
LATENT_BLUR_SAVE_PT = True

# Optional filter on the final SA3-polished latent before SAME decode.
# This is useful when the sampler output is musically right but has latent-frame grit.
LATENT_BLUR_POST_FILTER = None
# Example:
# LATENT_BLUR_POST_FILTER = {
#     "name": "post_low_shelf_c065_g060_s025",
#     "mode": "fft_lowpass",
#     "filter_cutoff": 0.65,
#     "filter_high_gain": 0.60,
#     "strength": 0.25,
# }

# Gentle defaults. The earlier full-strength temporal blurs are intentionally not the default
# because SA3 polish often treats them as over-damaged latents.
LATENT_BLUR_SPECS = [
    {"name": "box_r1_s015", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 1, "strength": 0.15},
    {"name": "box_r1_s030", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 1, "strength": 0.30},
    {"name": "box_r2_s020", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 2, "strength": 0.20},
    {"name": "box_r2_s040", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 2, "strength": 0.40},
    {"name": "gaussian_r2_s020", "mode": "temporal", "temporal_kernel": "gaussian", "temporal_radius": 2, "strength": 0.20},
    {"name": "gaussian_r4_s015", "mode": "temporal", "temporal_kernel": "gaussian", "temporal_radius": 4, "strength": 0.15},
    {"name": "motion_past_r2_s020", "mode": "temporal", "temporal_kernel": "box", "temporal_direction": "past", "temporal_radius": 2, "strength": 0.20},
    {"name": "motion_future_r2_s020", "mode": "temporal", "temporal_kernel": "box", "temporal_direction": "future", "temporal_radius": 2, "strength": 0.20},
    {"name": "channel_r1_s020", "mode": "channel", "channel_radius": 1, "strength": 0.20},
    {"name": "time_channel_t1_c1_s020", "mode": "temporal_channel", "temporal_kernel": "box", "temporal_radius": 1, "channel_radius": 1, "strength": 0.20},
    {"name": "filter_low_shelf_c065_g060_s025", "mode": "fft_lowpass", "filter_cutoff": 0.65, "filter_high_gain": 0.60, "strength": 0.25},
    {"name": "filter_low_shelf_c045_g070_s025", "mode": "fft_lowpass", "filter_cutoff": 0.45, "filter_high_gain": 0.70, "strength": 0.25},
    {"name": "filter_band_shelf_010_070_s025", "mode": "fft_bandpass", "filter_low_cutoff": 0.10, "filter_high_cutoff": 0.70, "filter_low_gain": 0.70, "filter_high_gain": 0.60, "strength": 0.25},
    {"name": "low_rank_8", "mode": "low_rank", "rank": 8, "strength": 1.0},
    {"name": "low_rank_24", "mode": "low_rank", "rank": 24, "strength": 1.0},
    {"name": "detail_gain_050", "mode": "detail_attenuate", "temporal_radius": 4, "detail_gain": 0.50, "strength": 1.0},
    {"name": "sharpen_r2_a025", "mode": "sharpen", "temporal_kernel": "box", "temporal_radius": 2, "sharpen_amount": 0.25, "strength": 1.0},
    {"name": "sharpen_r4_a050", "mode": "sharpen", "temporal_kernel": "box", "temporal_radius": 4, "sharpen_amount": 0.50, "strength": 1.0},
    {"name": "mean_blend_010", "mode": "mean_blend", "strength": 0.10},
]

# Optional harsh probes. Use these deliberately, e.g.:
# LATENT_BLUR_SPECS = LATENT_BLUR_HEAVY_SPECS
# or:
# LATENT_BLUR_SPECS = LATENT_BLUR_SPECS + LATENT_BLUR_HEAVY_SPECS[:3]
LATENT_BLUR_HEAVY_SPECS = [
    {"name": "heavy_box_r2_s100", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 2, "strength": 1.0},
    {"name": "heavy_box_r4_s100", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 4, "strength": 1.0},
    {"name": "heavy_box_r8_s100", "mode": "temporal", "temporal_kernel": "box", "temporal_radius": 8, "strength": 1.0},
    {"name": "heavy_motion_past_r4_s100", "mode": "temporal", "temporal_kernel": "box", "temporal_direction": "past", "temporal_radius": 4, "strength": 1.0},
    {"name": "heavy_motion_future_r4_s100", "mode": "temporal", "temporal_kernel": "box", "temporal_direction": "future", "temporal_radius": 4, "strength": 1.0},
    {"name": "heavy_gaussian_r4_s100", "mode": "temporal", "temporal_kernel": "gaussian", "temporal_radius": 4, "strength": 1.0},
    {"name": "heavy_channel_r4_s100", "mode": "channel", "channel_radius": 4, "strength": 1.0},
    {"name": "heavy_time_channel_s075", "mode": "temporal_channel", "temporal_kernel": "box", "temporal_radius": 3, "channel_radius": 2, "strength": 0.75},
]


def latent_blur_spec_from_dict(payload):
    return LatentBlurSpec(
        name=payload["name"],
        mode=payload.get("mode", "temporal"),
        strength=float(payload.get("strength", 1.0)),
        temporal_radius=int(payload.get("temporal_radius", 4)),
        temporal_sigma=payload.get("temporal_sigma"),
        temporal_kernel=payload.get("temporal_kernel", "gaussian"),
        temporal_direction=payload.get("temporal_direction", "centered"),
        channel_radius=int(payload.get("channel_radius", 2)),
        channel_sigma=payload.get("channel_sigma"),
        rank=int(payload.get("rank", 16)),
        detail_gain=float(payload.get("detail_gain", 0.25)),
        sharpen_amount=float(payload.get("sharpen_amount", 0.5)),
        filter_cutoff=float(payload.get("filter_cutoff", 0.5)),
        filter_low_cutoff=float(payload.get("filter_low_cutoff", 0.1)),
        filter_high_cutoff=float(payload.get("filter_high_cutoff", 0.6)),
        filter_low_gain=float(payload.get("filter_low_gain", 0.0)),
        filter_mid_gain=float(payload.get("filter_mid_gain", 1.0)),
        filter_high_gain=float(payload.get("filter_high_gain", 0.0)),
    )


if RUN_MODE_0D_LATENT_BLUR:
    require_models()
    LATENT_BLUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_audio, source_sr = load_audio(
        LATENT_BLUR_AUDIO,
        duration=LATENT_BLUR_DURATION,
        stereo=True,
    )
    source_item = encode_audio_file_to_item(
        LATENT_BLUR_AUDIO,
        item_id="latent_blur_source",
        duration=LATENT_BLUR_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    player_paths = []
    player_labels = []
    manifest = []
    source_path = LATENT_BLUR_OUTPUT_DIR / "source_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)
    player_paths.append(source_path)
    player_labels.append("source reference")

    for variant_index, spec_dict in enumerate(LATENT_BLUR_SPECS):
        spec = latent_blur_spec_from_dict(spec_dict)
        name = safe_audio_name(spec.name)
        seed = LATENT_BLUR_BASE_SEED + variant_index
        blurred_latents = apply_latent_blur(source_latents, spec)
        entry = {
            "name": spec.name,
            "mode": spec.mode,
            "strength": spec.strength,
            "temporal_radius": spec.temporal_radius,
            "temporal_kernel": spec.temporal_kernel,
            "temporal_direction": spec.temporal_direction,
            "channel_radius": spec.channel_radius,
            "rank": spec.rank,
            "detail_gain": spec.detail_gain,
            "sharpen_amount": spec.sharpen_amount,
            "filter_cutoff": spec.filter_cutoff,
            "filter_low_cutoff": spec.filter_low_cutoff,
            "filter_high_cutoff": spec.filter_high_cutoff,
            "filter_low_gain": spec.filter_low_gain,
            "filter_mid_gain": spec.filter_mid_gain,
            "filter_high_gain": spec.filter_high_gain,
            "polish_noise": LATENT_BLUR_POLISH_NOISE,
            "polish_steps": LATENT_BLUR_STEPS,
            "seed": seed,
            "outputs": {},
        }

        if LATENT_BLUR_RUN_DIRECT_DECODE:
            direct_path = LATENT_BLUR_OUTPUT_DIR / f"direct_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                blurred_latents,
                direct_path,
                duration=LATENT_BLUR_DURATION,
            )
            entry["outputs"]["direct_same_decode"] = str(direct_path)
            player_paths.append(direct_path)
            player_labels.append(f"direct {spec.name}")

        if LATENT_BLUR_RUN_SA3_POLISH:
            polished_latents = sa3_sample_from_init_latents(
                sa3_model,
                blurred_latents,
                prompt=LATENT_BLUR_PROMPT,
                duration=LATENT_BLUR_DURATION,
                steps=LATENT_BLUR_STEPS,
                cfg_scale=LATENT_BLUR_CFG,
                init_noise_level=LATENT_BLUR_POLISH_NOISE,
                seed=seed,
            )
            if LATENT_BLUR_POST_FILTER is not None:
                post_filter_spec = latent_blur_spec_from_dict(LATENT_BLUR_POST_FILTER)
                polished_latents = apply_latent_blur(polished_latents, post_filter_spec)
                entry["post_filter"] = dict(LATENT_BLUR_POST_FILTER)
            polished_path = LATENT_BLUR_OUTPUT_DIR / f"sa3_polish_{name}.wav"
            decode_sa3_latents_to_file_cropped(
                polished_latents,
                polished_path,
                duration=LATENT_BLUR_DURATION,
            )
            entry["outputs"]["sa3_polished"] = str(polished_path)
            player_paths.append(polished_path)
            player_labels.append(f"SA3 polish {spec.name}")

            if LATENT_BLUR_SAVE_PT:
                pt_path = LATENT_BLUR_OUTPUT_DIR / f"{name}.pt"
                torch.save(
                    {
                        "source_latents": source_latents.detach().cpu(),
                        "blurred_latents": blurred_latents.detach().cpu(),
                        "polished_latents": polished_latents.detach().cpu(),
                        "spec": spec_dict,
                    },
                    pt_path,
                )
                entry["outputs"]["pt"] = str(pt_path)
        elif LATENT_BLUR_SAVE_PT:
            pt_path = LATENT_BLUR_OUTPUT_DIR / f"{name}.pt"
            torch.save(
                {
                    "source_latents": source_latents.detach().cpu(),
                    "blurred_latents": blurred_latents.detach().cpu(),
                    "spec": spec_dict,
                },
                pt_path,
            )
            entry["outputs"]["pt"] = str(pt_path)

        manifest.append(entry)
        print(spec.name, "mode:", spec.mode)

    manifest_path = LATENT_BLUR_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        title="Mode 0d latent blur playground",
        max_embed_mb=180.0,
    )
"""
    ),
    md(
        r"""
## Mode 0h. Neural Latent DSP Playground

This mode treats SAME latents as a learned 256-channel, low-rate signal:

```text
z in R^{B x 256 x T}, latent_rate ~= 10.77 Hz
```

The operators are DSP-like, but not waveform DSP:

```text
D(O_z(E(x))) != O_x(x)
```

So each operation is a hypothesis about the neural compressed representation.
The mode can direct-decode the edited latent as a microscope and/or use SA3 as
a manifold reprojector:

```text
z_source -> O_z(z_source) -> SAME decode
z_source -> O_z(z_source) -> SA3 init_data polish -> SAME decode
```

Included operators:

```text
latent gain/compression/expansion/soft clipping
latent-time FFT EQ
latent-time FFT phase shift/randomization
donor magnitude / source phase grafts
per-clip PCA component gain, like learned macro-EQ
audio descriptor audit after decode
```

The MIR descriptors are audit signals, not final truth. Use them to ask whether
an edit moved brightness, flux, loudness, stereo width, etc. in a repeatable way.
"""
    ),
    code(
        r"""
# @title Mode 0h. Neural latent DSP playground

RUN_MODE_0H_LATENT_DSP = False

LATENT_DSP_AUDIO = INPUT_DIR / "known_9s.wav"
LATENT_DSP_DONOR_AUDIO = INPUT_DIR / "donor_9s.wav"
LATENT_DSP_OUTPUT_DIR = OUTPUT_DIR / "mode_00h_latent_dsp"
LATENT_DSP_ANNOTATION_PATH = OUTPUT_DIR / "mode_00h_latent_dsp_annotations.jsonl"
LATENT_DSP_DURATION = 9.0
LATENT_DSP_PROMPT = ""
LATENT_DSP_STEPS = 20
LATENT_DSP_CFG = 1.0
LATENT_DSP_POLISH_NOISE = 0.06
LATENT_DSP_BASE_SEED = 1200
LATENT_DSP_RUN_DIRECT_DECODE = False
LATENT_DSP_RUN_SA3_POLISH = True
LATENT_DSP_SAVE_PT = True
LATENT_DSP_INCLUDE_DONOR_SPECS = False

LATENT_DSP_SPECS = [
    {
        "name": "gain_expand_115",
        "mode": "gain",
        "gain": 1.15,
        "center": "channel_mean",
        "strength": 1.0,
    },
    {
        "name": "compress_t080_r400",
        "mode": "compress",
        "threshold": 0.80,
        "ratio": 4.0,
        "center": "channel_mean",
        "strength": 1.0,
    },
    {
        "name": "expand_t130_r160",
        "mode": "expand",
        "threshold": 1.30,
        "ratio": 1.60,
        "center": "channel_mean",
        "strength": 0.70,
    },
    {
        "name": "softclip_drive160",
        "mode": "softclip",
        "drive": 1.60,
        "ceiling": 1.75,
        "center": "channel_mean",
        "strength": 1.0,
    },
    {
        "name": "fft_eq_slow_boost_fast_damp",
        "mode": "fft_eq",
        "fft_low_cutoff": 0.12,
        "fft_high_cutoff": 0.62,
        "fft_low_gain": 1.20,
        "fft_mid_gain": 1.00,
        "fft_high_gain": 0.70,
        "strength": 0.75,
    },
    {
        "name": "fft_eq_fast_push",
        "mode": "fft_eq",
        "fft_low_cutoff": 0.15,
        "fft_high_cutoff": 0.70,
        "fft_low_gain": 0.90,
        "fft_mid_gain": 1.00,
        "fft_high_gain": 1.30,
        "strength": 0.45,
    },
    {
        "name": "phase_shift_quarter",
        "mode": "fft_phase_shift",
        "phase_shift_fraction": 0.25,
        "strength": 0.40,
    },
    {
        "name": "phase_random_012",
        "mode": "fft_phase_randomize",
        "phase_random_amount": 0.12,
        "seed": 123,
        "strength": 1.0,
    },
    {
        "name": "pca_macro_1_boost",
        "mode": "pca_gain",
        "pca_component_gains": [1.20, 1.00, 1.00, 1.00, 1.00, 1.00],
        "strength": 0.70,
    },
    {
        "name": "pca_macro_top_smooth",
        "mode": "pca_gain",
        "pca_component_gains": [0.85, 0.90, 0.95, 1.00, 1.00, 1.00],
        "strength": 0.80,
    },
]

LATENT_DSP_DONOR_SPECS = [
    {
        "name": "donor_magnitude_source_phase",
        "mode": "fft_mag_phase_graft",
        "magnitude_amount": 0.50,
        "strength": 1.0,
    },
    {
        "name": "source_magnitude_donor_phase",
        "mode": "fft_phase_blend",
        "phase_blend_amount": 0.35,
        "strength": 1.0,
    },
]


def latent_dsp_spec_from_dict(payload):
    gains = payload.get("pca_component_gains")
    return LatentDSPSpec(
        name=payload["name"],
        mode=payload.get("mode", "gain"),
        strength=float(payload.get("strength", 1.0)),
        gain=float(payload.get("gain", 1.0)),
        center=payload.get("center", "channel_mean"),
        threshold=float(payload.get("threshold", 1.0)),
        ratio=float(payload.get("ratio", 4.0)),
        makeup_gain=float(payload.get("makeup_gain", 1.0)),
        drive=float(payload.get("drive", 1.0)),
        ceiling=float(payload.get("ceiling", 2.0)),
        fft_low_cutoff=float(payload.get("fft_low_cutoff", 0.15)),
        fft_high_cutoff=float(payload.get("fft_high_cutoff", 0.65)),
        fft_low_gain=float(payload.get("fft_low_gain", 1.0)),
        fft_mid_gain=float(payload.get("fft_mid_gain", 1.0)),
        fft_high_gain=float(payload.get("fft_high_gain", 1.0)),
        phase_shift_fraction=float(payload.get("phase_shift_fraction", 0.0)),
        phase_random_amount=float(payload.get("phase_random_amount", 1.0)),
        phase_blend_amount=float(payload.get("phase_blend_amount", 1.0)),
        magnitude_amount=float(payload.get("magnitude_amount", 1.0)),
        pca_rank=payload.get("pca_rank"),
        pca_component_gains=tuple(float(v) for v in gains) if gains is not None else None,
        seed=int(payload.get("seed", 0)),
    )


def latent_dsp_mode_requires_donor(mode):
    return mode.lower() in {
        "fft_mag_phase_graft",
        "mag_phase_graft",
        "magnitude_from_donor",
        "donor_magnitude",
        "fft_phase_from_donor",
        "donor_phase",
        "fft_phase_blend",
        "phase_blend",
    }


def tensor_periodicity_dict(latents, item_id):
    item = sa3_tensor_to_item(latents, item_id=item_id)
    report = periodicity_report(item)
    return {
        "best_lag": report.best_lag,
        "best_score": report.best_score,
        "boundary_l2": report.boundary_l2,
        "velocity_l2": report.velocity_l2,
        "spectral_centroid": report.spectral_centroid,
    }


if RUN_MODE_0H_LATENT_DSP:
    require_models()
    LATENT_DSP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_audio, source_sr = load_audio(
        LATENT_DSP_AUDIO,
        duration=LATENT_DSP_DURATION,
        stereo=True,
    )
    source_descriptors = audio_descriptor_report(source_audio, source_sr)
    source_item = encode_audio_file_to_item(
        LATENT_DSP_AUDIO,
        item_id="latent_dsp_source",
        duration=LATENT_DSP_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    donor_latents = None
    if Path(LATENT_DSP_DONOR_AUDIO).exists():
        donor_item = encode_audio_file_to_item(
            LATENT_DSP_DONOR_AUDIO,
            item_id="latent_dsp_donor",
            duration=LATENT_DSP_DURATION,
        )
        donor_latents = item_to_sa3_tensor(
            donor_item,
            device=DEVICE,
            dtype=next(sa3_model.model.model.parameters()).dtype,
        )

    specs = list(LATENT_DSP_SPECS)
    if LATENT_DSP_INCLUDE_DONOR_SPECS:
        specs.extend(LATENT_DSP_DONOR_SPECS)

    player_paths = []
    player_labels = []
    manifest = []
    source_path = LATENT_DSP_OUTPUT_DIR / "source_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)
    player_paths.append(source_path)
    player_labels.append("source reference")
    source_periodicity = tensor_periodicity_dict(source_latents, "latent_dsp_source_periodicity")

    for variant_index, spec_dict in enumerate(specs):
        spec = latent_dsp_spec_from_dict(spec_dict)
        if latent_dsp_mode_requires_donor(spec.mode) and donor_latents is None:
            print("skipping donor mode without donor audio:", spec.name)
            continue
        name = safe_audio_name(spec.name)
        seed = LATENT_DSP_BASE_SEED + variant_index
        edited_latents = apply_latent_dsp(source_latents, spec, donor_latents=donor_latents)
        latent_delta = latent_change_report(source_latents, edited_latents)
        entry = {
            "name": spec.name,
            "mode": spec.mode,
            "spec": dict(spec_dict),
            "polish_noise": LATENT_DSP_POLISH_NOISE,
            "polish_steps": LATENT_DSP_STEPS,
            "seed": seed,
            "latent_delta": latent_delta,
            "source_periodicity": source_periodicity,
            "edited_periodicity": tensor_periodicity_dict(edited_latents, f"{name}_edited_periodicity"),
            "source_audio_descriptors": source_descriptors,
            "outputs": {},
        }

        if LATENT_DSP_RUN_DIRECT_DECODE:
            direct_path = LATENT_DSP_OUTPUT_DIR / f"direct_{name}.wav"
            direct_path, direct_desc = decode_sa3_latents_to_file_with_descriptors(
                edited_latents,
                direct_path,
                duration=LATENT_DSP_DURATION,
            )
            entry["outputs"]["direct_same_decode"] = str(direct_path)
            entry["direct_audio_descriptors"] = direct_desc
            entry["direct_descriptor_delta"] = descriptor_delta(source_descriptors, direct_desc)
            player_paths.append(direct_path)
            player_labels.append(f"direct {spec.name}")

        if LATENT_DSP_RUN_SA3_POLISH:
            polished_latents = sa3_sample_from_init_latents(
                sa3_model,
                edited_latents,
                prompt=LATENT_DSP_PROMPT,
                duration=LATENT_DSP_DURATION,
                steps=LATENT_DSP_STEPS,
                cfg_scale=LATENT_DSP_CFG,
                init_noise_level=LATENT_DSP_POLISH_NOISE,
                seed=seed,
            )
            polished_path = LATENT_DSP_OUTPUT_DIR / f"sa3_polish_{name}.wav"
            polished_path, polished_desc = decode_sa3_latents_to_file_with_descriptors(
                polished_latents,
                polished_path,
                duration=LATENT_DSP_DURATION,
            )
            entry["outputs"]["sa3_polished"] = str(polished_path)
            entry["polished_latent_delta"] = latent_change_report(source_latents, polished_latents)
            entry["polished_periodicity"] = tensor_periodicity_dict(polished_latents, f"{name}_polished_periodicity")
            entry["polished_audio_descriptors"] = polished_desc
            entry["polished_descriptor_delta"] = descriptor_delta(source_descriptors, polished_desc)
            player_paths.append(polished_path)
            player_labels.append(f"SA3 polish {spec.name}")

            if LATENT_DSP_SAVE_PT:
                pt_path = LATENT_DSP_OUTPUT_DIR / f"{name}.pt"
                torch.save(
                    {
                        "source_latents": source_latents.detach().cpu(),
                        "donor_latents": donor_latents.detach().cpu() if donor_latents is not None else None,
                        "edited_latents": edited_latents.detach().cpu(),
                        "polished_latents": polished_latents.detach().cpu(),
                        "spec": spec_dict,
                    },
                    pt_path,
                )
                entry["outputs"]["pt"] = str(pt_path)
        elif LATENT_DSP_SAVE_PT:
            pt_path = LATENT_DSP_OUTPUT_DIR / f"{name}.pt"
            torch.save(
                {
                    "source_latents": source_latents.detach().cpu(),
                    "donor_latents": donor_latents.detach().cpu() if donor_latents is not None else None,
                    "edited_latents": edited_latents.detach().cpu(),
                    "spec": spec_dict,
                },
                pt_path,
            )
            entry["outputs"]["pt"] = str(pt_path)

        manifest.append(entry)
        print(spec.name, "mode:", spec.mode, "delta_rms:", round(latent_delta["delta_rms"], 5))

    manifest_path = LATENT_DSP_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        title="Mode 0h neural latent DSP playground",
        max_embed_mb=180.0,
        annotation_path=LATENT_DSP_ANNOTATION_PATH,
    )
"""
    ),
    md(
        r"""
## Mode 0f. Cyclic Time-Roll Loop Lab

This is the audio version of the old image-tiling trick: move the seam away from the boundary, let the model operate while that seam is inside the signal, then roll back.

Cyclic roll operator:

$$
R_s(z)_t = z_{(t-s) \bmod T}
$$

Latent roll polish:

$$
z_{rolled}=R_s(z),\quad
\tilde z_{rolled}=F_\theta(z_{rolled}),\quad
\tilde z=R_{-s}(\tilde z_{rolled})
$$

Audio seam inpaint:

```text
roll audio by half
the original loop boundary is now in the middle
inpaint a small window around that middle seam
roll audio back
repeat it several times and listen to the seam
```

This is not a perfect-loop guarantee. It is a cyclic augmentation experiment that gives SA3 a chance to repair the loop boundary as normal musical interior rather than as the edge of a file.

Important distinction:

- The single-pass probes below roll once, repair once, then unroll.
- The iterative probes roll by a schedule such as `[0.5, 0.25, 0.75]`, repair, unroll, then repeat from the repaired audio/latents. This is closer to the old image-tiling trick where the optimizer keeps seeing a different origin.
- This still is not "inside every diffusion step". Doing that would require sampler-level surgery: applying cyclic shifts, loop losses, or seam guidance inside the denoising/flow loop.
"""
    ),
    code(
        r"""
# @title Mode 0f. Cyclic time-roll loop lab

RUN_MODE_0F_CYCLIC_LOOP = False

LOOP_AUDIO = INPUT_DIR / "known_9s.wav"
LOOP_OUTPUT_DIR = OUTPUT_DIR / "mode_00f_cyclic_loop"
LOOP_DURATION = 9.0
LOOP_PROMPT = ""
LOOP_STEPS = 20
LOOP_CFG = 1.0
LOOP_BASE_SEED = 970

LOOP_SHIFT_FRACTIONS = [0.5]
LOOP_PREVIEW_REPEATS = 4

# Latent path: roll SAME latents, SA3-polish, unroll, decode.
LOOP_RUN_LATENT_ROLL_POLISH = True
LOOP_LATENT_POLISH_NOISE = 0.25

# Audio path: roll waveform, inpaint seam window, unroll waveform.
LOOP_RUN_AUDIO_ROLL_INPAINT = True
LOOP_INPAINT_WINDOW_SECONDS = 1.0

# Iterative cyclic augmentation. These work on arbitrary audio clips, not only
# pre-existing loops. They repeatedly move the temporal origin so several
# would-be seams become ordinary interior material.
LOOP_RUN_ITERATIVE_LATENT_ROLL_POLISH = False
LOOP_RUN_ITERATIVE_AUDIO_ROLL_INPAINT = False
LOOP_ITERATIONS = 3
LOOP_ITERATIVE_SHIFT_FRACTIONS = [0.5, 0.25, 0.75]
LOOP_ITERATIVE_STEPS = 12
LOOP_ITERATIVE_LATENT_POLISH_NOISE = 0.14


def safe_loop_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))[:96]


def save_loop_preview(audio_path, repeats, label):
    audio, sample_rate = torchaudio.load(str(audio_path))
    preview = repeated_loop_preview_audio(audio, repeats=repeats).cpu().float().clamp(-1, 1)
    preview_path = audio_path.with_name(f"{audio_path.stem}_x{repeats}_{safe_loop_name(label)}.wav")
    torchaudio.save(str(preview_path), preview, sample_rate)
    return preview_path


def metrics_payload(latents, *, k=8):
    metrics = loop_boundary_metrics(latents, window_frames=k)
    return {
        "state_l2": metrics.state_l2,
        "velocity_l2": metrics.velocity_l2,
        "total": metrics.total,
        "window_frames": metrics.window_frames,
    }


def cyclic_schedule_value(schedule, index):
    if not schedule:
        raise ValueError("shift schedule cannot be empty")
    return float(schedule[index % len(schedule)])


if RUN_MODE_0F_CYCLIC_LOOP:
    require_models()
    LOOP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_audio, source_sr = load_audio(
        LOOP_AUDIO,
        target_sample_rate=sa3.sample_rate,
        duration=LOOP_DURATION,
        stereo=True,
    )
    source_item = encode_audio_file_to_item(
        LOOP_AUDIO,
        item_id="loop_source",
        duration=LOOP_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    manifest = []
    player_paths = []
    player_labels = []

    source_path = LOOP_OUTPUT_DIR / "source_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)
    source_preview_path = save_loop_preview(source_path, LOOP_PREVIEW_REPEATS, "preview")
    player_paths.extend([source_path, source_preview_path])
    player_labels.extend(["source reference", f"source x{LOOP_PREVIEW_REPEATS} seam preview"])
    source_metrics = metrics_payload(source_latents)
    print("source latent loop metrics:", source_metrics)

    for index, shift_fraction in enumerate(LOOP_SHIFT_FRACTIONS):
        shift_tag = safe_loop_name(f"{shift_fraction:.3f}")
        seed = LOOP_BASE_SEED + index
        shift_frames = frames_from_fraction(source_latents, shift_fraction)
        shift_samples = samples_from_fraction(source_audio, shift_fraction)
        start_sec, end_sec = seam_inpaint_bounds(
            LOOP_DURATION,
            shift_fraction,
            LOOP_INPAINT_WINDOW_SECONDS,
        )
        entry = {
            "shift_fraction": float(shift_fraction),
            "shift_frames": int(shift_frames),
            "shift_samples": int(shift_samples),
            "seed": int(seed),
            "source_metrics": source_metrics,
            "outputs": {},
        }

        if LOOP_RUN_LATENT_ROLL_POLISH:
            rolled_latents = cyclic_roll_latents(source_latents, shift_frames)
            polished_rolled = sa3_sample_from_init_latents(
                sa3_model,
                rolled_latents,
                prompt=LOOP_PROMPT,
                duration=LOOP_DURATION,
                steps=LOOP_STEPS,
                cfg_scale=LOOP_CFG,
                init_noise_level=LOOP_LATENT_POLISH_NOISE,
                seed=seed,
            )
            unrolled_latents = cyclic_roll_latents(polished_rolled, -shift_frames)
            latent_path = LOOP_OUTPUT_DIR / f"latent_roll_polish_shift_{shift_tag}.wav"
            decode_sa3_latents_to_file_cropped(
                unrolled_latents,
                latent_path,
                duration=LOOP_DURATION,
            )
            latent_preview_path = save_loop_preview(latent_path, LOOP_PREVIEW_REPEATS, "preview")
            latent_metrics = metrics_payload(unrolled_latents)
            entry["latent_roll_polish"] = {
                "polish_noise": LOOP_LATENT_POLISH_NOISE,
                "steps": LOOP_STEPS,
                "metrics": latent_metrics,
            }
            entry["outputs"]["latent_roll_polish"] = str(latent_path)
            entry["outputs"]["latent_roll_polish_preview"] = str(latent_preview_path)
            player_paths.extend([latent_path, latent_preview_path])
            player_labels.extend(
                [
                    f"latent roll polish shift {shift_fraction:.2f}",
                    f"latent roll polish shift {shift_fraction:.2f} x{LOOP_PREVIEW_REPEATS}",
                ]
            )
            print("latent roll metrics", shift_fraction, latent_metrics)

        if LOOP_RUN_AUDIO_ROLL_INPAINT:
            rolled_audio = cyclic_roll_audio(source_audio, shift_samples)
            with torch.inference_mode():
                repaired_rolled = sa3_model.generate(
                    prompt=LOOP_PROMPT,
                    duration=LOOP_DURATION,
                    steps=LOOP_STEPS,
                    cfg_scale=LOOP_CFG,
                    seed=seed + 100,
                    inpaint_audio=(source_sr, rolled_audio),
                    inpaint_mask_start_seconds=start_sec,
                    inpaint_mask_end_seconds=end_sec,
                    truncate_output_to_duration=True,
                )
            repaired_audio = cyclic_roll_audio(repaired_rolled[0], -shift_samples).cpu().float().clamp(-1, 1)
            inpaint_path = LOOP_OUTPUT_DIR / f"audio_roll_inpaint_shift_{shift_tag}.wav"
            torchaudio.save(str(inpaint_path), repaired_audio, sa3.sample_rate)
            inpaint_preview_path = save_loop_preview(inpaint_path, LOOP_PREVIEW_REPEATS, "preview")
            entry["audio_roll_inpaint"] = {
                "inpaint_start_seconds": start_sec,
                "inpaint_end_seconds": end_sec,
                "steps": LOOP_STEPS,
            }
            entry["outputs"]["audio_roll_inpaint"] = str(inpaint_path)
            entry["outputs"]["audio_roll_inpaint_preview"] = str(inpaint_preview_path)
            player_paths.extend([inpaint_path, inpaint_preview_path])
            player_labels.extend(
                [
                    f"audio roll inpaint shift {shift_fraction:.2f}",
                    f"audio roll inpaint shift {shift_fraction:.2f} x{LOOP_PREVIEW_REPEATS}",
                ]
            )

        manifest.append(entry)

    if LOOP_RUN_ITERATIVE_LATENT_ROLL_POLISH:
        current_latents = source_latents
        trace = []
        for iteration in range(int(LOOP_ITERATIONS)):
            shift_fraction = cyclic_schedule_value(LOOP_ITERATIVE_SHIFT_FRACTIONS, iteration)
            shift_frames = frames_from_fraction(current_latents, shift_fraction)
            rolled_latents = cyclic_roll_latents(current_latents, shift_frames)
            polished_rolled = sa3_sample_from_init_latents(
                sa3_model,
                rolled_latents,
                prompt=LOOP_PROMPT,
                duration=LOOP_DURATION,
                steps=LOOP_ITERATIVE_STEPS,
                cfg_scale=LOOP_CFG,
                init_noise_level=LOOP_ITERATIVE_LATENT_POLISH_NOISE,
                seed=LOOP_BASE_SEED + 1000 + iteration,
            )
            current_latents = cyclic_roll_latents(polished_rolled, -shift_frames).detach()
            iteration_metrics = metrics_payload(current_latents)
            trace.append(
                {
                    "iteration": iteration,
                    "shift_fraction": shift_fraction,
                    "shift_frames": int(shift_frames),
                    "seed": int(LOOP_BASE_SEED + 1000 + iteration),
                    "metrics": iteration_metrics,
                }
            )
            print("iterative latent roll", iteration, shift_fraction, iteration_metrics)

        iterative_latent_path = LOOP_OUTPUT_DIR / "iterative_latent_roll_polish.wav"
        decode_sa3_latents_to_file_cropped(
            current_latents,
            iterative_latent_path,
            duration=LOOP_DURATION,
        )
        iterative_latent_preview_path = save_loop_preview(
            iterative_latent_path,
            LOOP_PREVIEW_REPEATS,
            "preview",
        )
        manifest.append(
            {
                "method": "iterative_latent_roll_polish",
                "iterations": int(LOOP_ITERATIONS),
                "shift_schedule": [float(v) for v in LOOP_ITERATIVE_SHIFT_FRACTIONS],
                "steps": int(LOOP_ITERATIVE_STEPS),
                "polish_noise": float(LOOP_ITERATIVE_LATENT_POLISH_NOISE),
                "trace": trace,
                "outputs": {
                    "audio": str(iterative_latent_path),
                    "preview": str(iterative_latent_preview_path),
                },
            }
        )
        player_paths.extend([iterative_latent_path, iterative_latent_preview_path])
        player_labels.extend(
            [
                "iterative latent roll polish",
                f"iterative latent roll polish x{LOOP_PREVIEW_REPEATS}",
            ]
        )

    if LOOP_RUN_ITERATIVE_AUDIO_ROLL_INPAINT:
        current_audio = source_audio.cpu().float()
        trace = []
        for iteration in range(int(LOOP_ITERATIONS)):
            shift_fraction = cyclic_schedule_value(LOOP_ITERATIVE_SHIFT_FRACTIONS, iteration)
            shift_samples = samples_from_fraction(current_audio, shift_fraction)
            start_sec, end_sec = seam_inpaint_bounds(
                LOOP_DURATION,
                shift_fraction,
                LOOP_INPAINT_WINDOW_SECONDS,
            )
            rolled_audio = cyclic_roll_audio(current_audio, shift_samples).cpu().float()
            with torch.inference_mode():
                repaired_rolled = sa3_model.generate(
                    prompt=LOOP_PROMPT,
                    duration=LOOP_DURATION,
                    steps=LOOP_ITERATIVE_STEPS,
                    cfg_scale=LOOP_CFG,
                    seed=LOOP_BASE_SEED + 2000 + iteration,
                    inpaint_audio=(source_sr, rolled_audio),
                    inpaint_mask_start_seconds=start_sec,
                    inpaint_mask_end_seconds=end_sec,
                    truncate_output_to_duration=True,
                )
            current_audio = cyclic_roll_audio(repaired_rolled[0], -shift_samples).cpu().float().clamp(-1, 1)
            trace.append(
                {
                    "iteration": iteration,
                    "shift_fraction": shift_fraction,
                    "shift_samples": int(shift_samples),
                    "seed": int(LOOP_BASE_SEED + 2000 + iteration),
                    "inpaint_start_seconds": float(start_sec),
                    "inpaint_end_seconds": float(end_sec),
                }
            )
            print("iterative audio roll", iteration, shift_fraction, (start_sec, end_sec))

        iterative_audio_path = LOOP_OUTPUT_DIR / "iterative_audio_roll_inpaint.wav"
        torchaudio.save(str(iterative_audio_path), current_audio.cpu(), sa3.sample_rate)
        iterative_audio_preview_path = save_loop_preview(
            iterative_audio_path,
            LOOP_PREVIEW_REPEATS,
            "preview",
        )
        manifest.append(
            {
                "method": "iterative_audio_roll_inpaint",
                "iterations": int(LOOP_ITERATIONS),
                "shift_schedule": [float(v) for v in LOOP_ITERATIVE_SHIFT_FRACTIONS],
                "steps": int(LOOP_ITERATIVE_STEPS),
                "inpaint_window_seconds": float(LOOP_INPAINT_WINDOW_SECONDS),
                "trace": trace,
                "outputs": {
                    "audio": str(iterative_audio_path),
                    "preview": str(iterative_audio_preview_path),
                },
            }
        )
        player_paths.extend([iterative_audio_path, iterative_audio_preview_path])
        player_labels.extend(
            [
                "iterative audio roll inpaint",
                f"iterative audio roll inpaint x{LOOP_PREVIEW_REPEATS}",
            ]
        )

    manifest_path = LOOP_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        title="Mode 0f cyclic time-roll loop lab",
        max_embed_mb=220.0,
    )
"""
    ),
    md(
        r"""
## Mode 0g. Cyclic Roll Inside the Denoising Trajectory

This is the closer analogue to early VQGAN/CLIP tiling tricks.

Instead of repairing an end/start boundary after the fact, we intervene during
the rectified-flow trajectory itself. Start from an arbitrary input audio clip
encoded into SAME latents:

$$
z_0 = E(x), \qquad x_T = (1-\sigma)z_0 + \sigma \epsilon
$$

The SAME decoder itself is feed-forward, so there is no decoder-time iterative
loop to scroll. The iterative signal we can actually intervene on is SA3's
denoising / rectified-flow state \(x_i\).

The mode closest to the tiling intuition is cyclic mixing:

$$
\bar x_i=\frac{1}{2}(x_i+R_s x_i)
$$

$$
x_{i+1}=\left[x_i+\Delta t_i\,v_\theta(x_i,t_i,c)\right]
\quad\text{then}\quad
x_{i+1}\leftarrow x_{i+1}+\beta(\bar x_{i+1}-x_{i+1})
$$

where \(R_s\) is a cyclic time roll, usually \(s=T/2\), and \(\beta\) is the
soft projection strength.

Unlike the earlier literal roll/unroll probe, this changes the latent state. It
also changes the latent state when `CYCLIC_STEP_INIT_NOISE = 0`, because the
projection itself still runs even if the flow update has \(\Delta t=0\).

Two older diagnostic modes remain available:

- `alternate`: literally rolls the state after every step. This is mostly a
  coordinate transform if the output is unrolled.
- `paired_average`: evaluates the velocity field at both origins and averages:

$$
v_{cyc}(x_i,t_i,c)=\frac{1}{2}\left[
v_\theta(x_i,t_i,c)+R_{-s}v_\theta(R_s x_i,t_i,c)
\right]
$$

No inpainting boundary is used here. The helper still passes zero inpaint tensors
because the SA3 model architecture expects them, but no mask region is active.
This is a sampler-level experiment, not a guaranteed loop constraint.
"""
    ),
    code(
        r"""
# @title Mode 0g. Cyclic roll inside each denoising step

RUN_MODE_0G_CYCLIC_STEP_ROLL = False

CYCLIC_STEP_AUDIO = INPUT_DIR / "known_9s.wav"
CYCLIC_STEP_OUTPUT_DIR = OUTPUT_DIR / "mode_00g_cyclic_step_roll"
CYCLIC_STEP_DURATION = 9.0
CYCLIC_STEP_PROMPT = ""
CYCLIC_STEP_STEPS = 20
CYCLIC_STEP_CFG = 1.0
CYCLIC_STEP_SEED = 1070

# Start at 0.0 to isolate the cyclic projection. Raise to 0.08-0.25 if you want
# SA3 to re-musicalize / re-denoise the projected latent.
CYCLIC_STEP_INIT_NOISE = 0.0

# 0.5 is the faithful half-scroll analogue.
CYCLIC_STEP_ROLL_FRACTION = 0.5

# "cyclic_mix" is the useful tiling-style version.
# Optional diagnostics: "alternate", "paired_average".
CYCLIC_STEP_MODES = ["cyclic_mix"]

# Soft projection strength per step:
#   0.02-0.06 = gentle
#   0.08-0.15 = audible
#   0.25+     = strong half-period collapse risk
CYCLIC_STEP_ROLL_MIXES = [0.08]
CYCLIC_STEP_MIX_EVERY_N = 1
CYCLIC_STEP_SYMMETRIC_MIX = True

# For alternate mode, unroll the final state back into the source orientation.
CYCLIC_STEP_UNROLL_OUTPUT = True
CYCLIC_STEP_ADD_REPEATED_PREVIEW = False
CYCLIC_STEP_PREVIEW_REPEATS = 2
CYCLIC_STEP_SAVE_PT = True


if RUN_MODE_0G_CYCLIC_STEP_ROLL:
    require_models()
    CYCLIC_STEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_audio, source_sr = load_audio(
        CYCLIC_STEP_AUDIO,
        target_sample_rate=sa3.sample_rate,
        duration=CYCLIC_STEP_DURATION,
        stereo=True,
    )
    source_item = encode_audio_file_to_item(
        CYCLIC_STEP_AUDIO,
        item_id="cyclic_step_source",
        duration=CYCLIC_STEP_DURATION,
    )
    source_latents = item_to_sa3_tensor(
        source_item,
        device=DEVICE,
        dtype=next(sa3_model.model.model.parameters()).dtype,
    )

    source_path = CYCLIC_STEP_OUTPUT_DIR / "source_reference.wav"
    torchaudio.save(str(source_path), source_audio.cpu(), source_sr)

    roll_frames = frames_from_fraction(source_latents, CYCLIC_STEP_ROLL_FRACTION)
    manifest = {
        "source_audio": str(CYCLIC_STEP_AUDIO),
        "duration": float(CYCLIC_STEP_DURATION),
        "prompt": CYCLIC_STEP_PROMPT,
        "steps": int(CYCLIC_STEP_STEPS),
        "cfg": float(CYCLIC_STEP_CFG),
        "init_noise": float(CYCLIC_STEP_INIT_NOISE),
        "roll_fraction": float(CYCLIC_STEP_ROLL_FRACTION),
        "roll_frames": int(roll_frames),
        "mix_every_n": int(CYCLIC_STEP_MIX_EVERY_N),
        "symmetric_mix": bool(CYCLIC_STEP_SYMMETRIC_MIX),
        "source_metrics": metrics_payload(source_latents),
        "outputs": [],
    }
    player_paths = [source_path]
    player_labels = ["source reference"]
    if CYCLIC_STEP_ADD_REPEATED_PREVIEW:
        source_preview_path = save_loop_preview(source_path, CYCLIC_STEP_PREVIEW_REPEATS, "preview")
        player_paths.append(source_preview_path)
        player_labels.append(f"source x{CYCLIC_STEP_PREVIEW_REPEATS}")

    variant_index = 0
    for mode in CYCLIC_STEP_MODES:
        for roll_mix in CYCLIC_STEP_ROLL_MIXES:
            print("running cyclic step roll:", mode, "mix", roll_mix)
            out_latents = sa3_cyclic_roll_sample_from_init_latents(
                sa3_model,
                source_latents,
                prompt=CYCLIC_STEP_PROMPT,
                duration=CYCLIC_STEP_DURATION,
                steps=CYCLIC_STEP_STEPS,
                cfg_scale=CYCLIC_STEP_CFG,
                init_noise_level=CYCLIC_STEP_INIT_NOISE,
                roll_fraction=CYCLIC_STEP_ROLL_FRACTION,
                mode=mode,
                roll_mix=roll_mix,
                mix_every_n=CYCLIC_STEP_MIX_EVERY_N,
                symmetric_mix=CYCLIC_STEP_SYMMETRIC_MIX,
                unroll_output=CYCLIC_STEP_UNROLL_OUTPUT,
                seed=CYCLIC_STEP_SEED + variant_index,
            )

            safe_mode = safe_loop_name(mode)
            mix_tag = safe_loop_name(f"{roll_mix:.3f}")
            out_path = CYCLIC_STEP_OUTPUT_DIR / f"cyclic_step_roll_{safe_mode}_mix_{mix_tag}.wav"
            decode_sa3_latents_to_file_cropped(
                out_latents,
                out_path,
                duration=CYCLIC_STEP_DURATION,
            )
            preview_path = None
            if CYCLIC_STEP_ADD_REPEATED_PREVIEW:
                preview_path = save_loop_preview(out_path, CYCLIC_STEP_PREVIEW_REPEATS, "preview")
            metrics = metrics_payload(out_latents)

            pt_path = None
            if CYCLIC_STEP_SAVE_PT:
                pt_path = CYCLIC_STEP_OUTPUT_DIR / f"cyclic_step_roll_{safe_mode}_mix_{mix_tag}.pt"
                torch.save(
                    {
                        "latents": out_latents.detach().cpu(),
                        "mode": mode,
                        "roll_mix": roll_mix,
                        "roll_fraction": CYCLIC_STEP_ROLL_FRACTION,
                        "roll_frames": roll_frames,
                        "init_noise": CYCLIC_STEP_INIT_NOISE,
                        "steps": CYCLIC_STEP_STEPS,
                        "seed": CYCLIC_STEP_SEED + variant_index,
                    },
                    pt_path,
                )

            manifest["outputs"].append(
                {
                    "mode": mode,
                    "roll_mix": float(roll_mix),
                    "audio": str(out_path),
                    "preview": str(preview_path) if preview_path else None,
                    "latents_pt": str(pt_path) if pt_path else None,
                    "metrics": metrics,
                }
            )
            player_paths.append(out_path)
            player_labels.append(f"cyclic step {mode} mix {roll_mix:.3f}")
            if preview_path is not None:
                player_paths.append(preview_path)
                player_labels.append(f"cyclic step {mode} mix {roll_mix:.3f} x{CYCLIC_STEP_PREVIEW_REPEATS}")
            print(mode, "mix", roll_mix, metrics)
            variant_index += 1

    manifest_path = CYCLIC_STEP_OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("saved manifest:", manifest_path)
    play_audio_files(
        player_paths,
        labels=player_labels,
        title="Mode 0g cyclic roll inside denoising",
        max_embed_mb=220.0,
    )
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

It is not language and it is not a SAME audio latent. It is an optimized SA3 conditioning state:

```text
soft_prompt.pt =
  conditioning dicts
  optimized conditioning tensors
  loss curve
  metadata
```

Use Mode 1b to hear what the `.pt` does. Use Mode 2 or Mode 3 if you want an actual text prompt.
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
## Mode 1b. Generate With A Soft Prompt

This consumes the `.pt` from Mode 1:

$$
c^\* \rightarrow \operatorname{sample}_\theta(\epsilon, c^\*) \rightarrow z \rightarrow D(z)
$$

The output should be judged like an instrument preset, not like a readable prompt.
"""
    ),
    code(
        r"""
# @title Mode 1b. Generate audio from a saved soft prompt

RUN_MODE_1B_GENERATE_WITH_SOFT_PROMPT = False

SOFT_PROMPT_GENERATION_DIR = OUTPUT_DIR / "mode_01b_soft_prompt_generations"
SOFT_PROMPT_GENERATION_STEPS = 8
SOFT_PROMPT_GENERATION_CFG = 1.0
SOFT_PROMPT_GENERATION_SEEDS = [10, 11, 12]

if RUN_MODE_1B_GENERATE_WITH_SOFT_PROMPT:
    require_models()
    SOFT_PROMPT_GENERATION_DIR.mkdir(parents=True, exist_ok=True)
    soft_state = SoftPromptState.load(SOFT_PROMPT_PATH)
    manifest = []
    for seed in SOFT_PROMPT_GENERATION_SEEDS:
        latents = generate_with_soft_prompt(
            sa3_model,
            soft_state,
            steps=SOFT_PROMPT_GENERATION_STEPS,
            cfg_scale=SOFT_PROMPT_GENERATION_CFG,
            seed=seed,
            return_latents=True,
        )
        out_path = SOFT_PROMPT_GENERATION_DIR / f"soft_prompt_seed_{seed}.wav"
        decode_sa3_latents_to_file(latents, out_path)
        manifest.append({"path": str(out_path), "seed": int(seed)})
        print("saved:", out_path)
    (SOFT_PROMPT_GENERATION_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        display_audio_player(
            [entry["path"] for entry in manifest],
            labels=[f"soft prompt seed {entry['seed']}" for entry in manifest],
            title="Mode 1b soft prompt generations",
        )
    except Exception as exc:
        print("Custom audio player skipped:", exc)
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

By default this cell uses candidate text pieces extracted from the tokenizer that SA3's T5Gemma prompt conditioner actually owns.

Important distinction:

```text
native tokenizer vocabulary = text/subword pieces seen by T5Gemma
SAME latent space           = continuous audio latents generated by SA3
```

So this is native to the text-conditioning path, not a symbolic music vocabulary.

Search-cost rule:

```text
search width per position      = BABBLE_TOKEN_SUBSET
beam width                    = BABBLE_BEAM_WIDTH
GPU prompts per forward chunk  = BABBLE_CANDIDATE_BATCH_SIZE
positions per run              = BABBLE_TOKENS_GENERATED
noise/timestep samples averaged= BABBLE_SCORE_SAMPLES
```

Better-math defaults:

- fixed shared log-SNR timestep/noise probe bank, so candidates are compared fairly
- antithetic noise pairs \(\epsilon,-\epsilon\), which reduce Monte Carlo variance
- normalized flow residuals, so a high-energy probe does not dominate just by scale
- beam search, so a useful token can survive even if it is not the greedy winner at one position
- MSE plus optional cosine direction loss over the SA3 velocity field
- optional conditional-delta score, comparing the prompt-specific vector-field change against the target-specific residual from the null prompt
- chunked GPU scoring, so you can raise search width without needing all candidates in VRAM at once

This is still prompt inversion through a frozen model, not a captioner. Good
results may be strange strings if those strings steer SA3's native conditioner.
"""
    ),
    code(
        r"""
# @title Mode 2. Audio -> babble / hard prompt

RUN_MODE_2_AUDIO_TO_BABBLE_PROMPT = False

USE_SA3_TOKENIZER_VOCAB = True
BABBLE_TOKENIZER_VOCAB_LIMIT = 16384
BABBLE_TOKENIZER_SCAN_LIMIT = 65536
BABBLE_REQUIRE_WORD_START = True
BABBLE_MIN_CHARS = 3
BABBLE_REJECT_STOPWORDS = True
BABBLE_AUDIO_PRIOR = True

# Search strategy:
#   "beam"   = best default; keeps multiple partial prompts alive.
#   "greedy" = older faster method.
BABBLE_SEARCH_STRATEGY = "beam"

# Search compute knobs. For beam search:
#   branch_factor is sampled per live beam at every token position.
#   beam_width keeps this many partial prompts after each position.
BABBLE_TOKEN_SUBSET = 256          # greedy width, or fallback beam branch factor
BABBLE_BEAM_WIDTH = 4
BABBLE_BRANCH_FACTOR = 256
BABBLE_CANDIDATE_BATCH_SIZE = 128
BABBLE_TOKENS_GENERATED = 16
BABBLE_GREEDY_RUNS = 6

# Native SA3 flow score. If BABBLE_LOGSNR_VALUES is non-empty, it becomes the
# timestep probe bank using t = sigmoid(-logSNR).
# These values span clean-ish, mid, and noisy regimes.
BABBLE_LOGSNR_VALUES = [2.0, 0.0, -2.0]
BABBLE_TIMESTEP_VALUES = []  # Manual override. Example: [0.12, 0.32, 0.55, 0.78]
BABBLE_SCORE_SAMPLES = 4
BABBLE_SHARED_NOISE = True
BABBLE_ANTITHETIC_NOISE = True
BABBLE_NORMALIZE_MSE = True
BABBLE_T_MIN = 0.05
BABBLE_T_MAX = 0.95
BABBLE_SCORE_SEED = 123
BABBLE_COSINE_WEIGHT = 0.25

# Expensive but conceptually useful. 0.0 disables it. Try 0.10-0.25 if you want
# to score the prompt-specific vector-field contribution relative to null text.
BABBLE_CONDITIONAL_DELTA_WEIGHT = 0.0

FALLBACK_BABBLE_VOCAB = [
    "shimmer", "velvet", "glass", "grain", "pressure", "hollow", "dense", "thin",
    "wide", "mono", "dust", "chrome", "choir", "fold", "pulse", "drift",
    "warm", "cold", "broken", "slow", "fast", "tape", "bright", "dark",
    "sub", "spark", "rubber", "metal", "wet", "dry", "attack", "bloom",
    "loop", "riser", "drop", "intro", "ghost", "close", "distant", "aura",
]

HARD_PROMPT_JSON = OUTPUT_DIR / "mode_02_babble_prompt.json"

if RUN_MODE_2_AUDIO_TO_BABBLE_PROMPT:
    require_models()
    if USE_SA3_TOKENIZER_VOCAB:
        babble_vocab = native_tokenizer_vocabulary(
            sa3_model,
            max_candidates=BABBLE_TOKENIZER_VOCAB_LIMIT,
            scan_limit=BABBLE_TOKENIZER_SCAN_LIMIT,
            min_chars=BABBLE_MIN_CHARS,
            require_word_start=BABBLE_REQUIRE_WORD_START,
            ascii_only=True,
            lowercase=True,
            reject_stopwords=BABBLE_REJECT_STOPWORDS,
            rank_by_audio_prior=BABBLE_AUDIO_PRIOR,
        )
        print(f"Using {len(babble_vocab)} native SA3/T5Gemma tokenizer-derived candidates.")
        print(preview_native_tokenizer_vocabulary(babble_vocab, columns=8, rows=4))
    else:
        babble_vocab = FALLBACK_BABBLE_VOCAB
        print(f"Using {len(babble_vocab)} fallback hand-written candidates.")

    strategy = BABBLE_SEARCH_STRATEGY.lower()
    resolved_timesteps = BABBLE_TIMESTEP_VALUES or timesteps_from_logsnr_values(BABBLE_LOGSNR_VALUES)
    effective_score_samples = len(resolved_timesteps) if resolved_timesteps else BABBLE_SCORE_SAMPLES
    antithetic_multiplier = 2 if BABBLE_ANTITHETIC_NOISE else 1
    null_multiplier = 2 if BABBLE_CONDITIONAL_DELTA_WEIGHT else 1
    greedy_width = min(BABBLE_TOKEN_SUBSET or len(babble_vocab), len(babble_vocab))
    branch_factor = min(BABBLE_BRANCH_FACTOR or greedy_width, len(babble_vocab))
    gpu_batch = BABBLE_CANDIDATE_BATCH_SIZE or greedy_width
    if strategy == "beam":
        candidates_per_position = BABBLE_BEAM_WIDTH * branch_factor
        approx_forward_chunks = (
            BABBLE_TOKENS_GENERATED
            * math.ceil(candidates_per_position / gpu_batch)
            * effective_score_samples
            * antithetic_multiplier
            * null_multiplier
        )
        approx_prompt_scores = (
            BABBLE_TOKENS_GENERATED
            * candidates_per_position
            * effective_score_samples
            * antithetic_multiplier
        )
    else:
        candidates_per_position = greedy_width
        approx_forward_chunks = (
            BABBLE_GREEDY_RUNS
            * BABBLE_TOKENS_GENERATED
            * math.ceil(greedy_width / gpu_batch)
            * effective_score_samples
            * antithetic_multiplier
            * null_multiplier
        )
        approx_prompt_scores = (
            BABBLE_GREEDY_RUNS
            * BABBLE_TOKENS_GENERATED
            * greedy_width
            * effective_score_samples
            * antithetic_multiplier
        )
    print(
        "Mode 2 budget:",
        {
            "strategy": strategy,
            "greedy_width": greedy_width,
            "beam_width": BABBLE_BEAM_WIDTH if strategy == "beam" else None,
            "branch_factor": branch_factor if strategy == "beam" else None,
            "candidates_per_position": candidates_per_position,
            "gpu_batch": gpu_batch,
            "tokens_generated": BABBLE_TOKENS_GENERATED,
            "greedy_runs": BABBLE_GREEDY_RUNS if strategy == "greedy" else None,
            "score_samples": effective_score_samples,
            "logsnr_values": BABBLE_LOGSNR_VALUES if BABBLE_LOGSNR_VALUES and not BABBLE_TIMESTEP_VALUES else None,
            "timesteps": resolved_timesteps if resolved_timesteps else "random",
            "antithetic_noise": BABBLE_ANTITHETIC_NOISE,
            "normalize_mse": BABBLE_NORMALIZE_MSE,
            "cosine_weight": BABBLE_COSINE_WEIGHT,
            "conditional_delta_weight": BABBLE_CONDITIONAL_DELTA_WEIGHT,
            "approx_prompt_scores": approx_prompt_scores,
            "approx_dit_forwards": approx_forward_chunks,
        },
    )

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
            seed=BABBLE_SCORE_SEED,
            min_t=BABBLE_T_MIN,
            max_t=BABBLE_T_MAX,
            score_samples=BABBLE_SCORE_SAMPLES,
            shared_noise=BABBLE_SHARED_NOISE,
            timestep_values=resolved_timesteps or None,
            cosine_weight=BABBLE_COSINE_WEIGHT,
            antithetic_noise=BABBLE_ANTITHETIC_NOISE,
            normalize_mse=BABBLE_NORMALIZE_MSE,
            conditional_delta_weight=BABBLE_CONDITIONAL_DELTA_WEIGHT,
            velocity_convention=FLOW_TARGET_CONVENTION,
        )
        return [-loss for loss in losses]

    if strategy == "beam":
        result = beam_token_prompt_search(
            babble_vocab,
            batch_scorer,
            tokens_generated=BABBLE_TOKENS_GENERATED,
            beam_width=BABBLE_BEAM_WIDTH,
            branch_factor=branch_factor,
            candidate_batch_size=BABBLE_CANDIDATE_BATCH_SIZE,
            seed=0,
            higher_is_better=True,
        )
    elif strategy == "greedy":
        result = greedy_token_prompt_search(
            babble_vocab,
            batch_scorer,
            tokens_generated=BABBLE_TOKENS_GENERATED,
            runs=BABBLE_GREEDY_RUNS,
            token_subset=BABBLE_TOKEN_SUBSET,
            candidate_batch_size=BABBLE_CANDIDATE_BATCH_SIZE,
            seed=0,
            higher_is_better=True,
        )
    else:
        raise ValueError("BABBLE_SEARCH_STRATEGY must be 'beam' or 'greedy'")

    payload = {
        "prompt": result.prompt,
        "score": result.score,
        "tokens": result.tokens,
        "strategy": strategy,
        "vocab_source": "sa3_t5gemma_tokenizer" if USE_SA3_TOKENIZER_VOCAB else "fallback_hand_vocab",
        "vocab_size": len(babble_vocab),
        "greedy_width": greedy_width,
        "beam_width": BABBLE_BEAM_WIDTH if strategy == "beam" else None,
        "branch_factor": branch_factor if strategy == "beam" else None,
        "candidate_batch_size": BABBLE_CANDIDATE_BATCH_SIZE,
        "tokens_generated": BABBLE_TOKENS_GENERATED,
        "greedy_runs": BABBLE_GREEDY_RUNS if strategy == "greedy" else None,
        "score_samples": effective_score_samples,
        "shared_noise": BABBLE_SHARED_NOISE,
        "logsnr_values": BABBLE_LOGSNR_VALUES if BABBLE_LOGSNR_VALUES and not BABBLE_TIMESTEP_VALUES else None,
        "timesteps": resolved_timesteps if resolved_timesteps else None,
        "t_range": [BABBLE_T_MIN, BABBLE_T_MAX] if not resolved_timesteps else None,
        "antithetic_noise": BABBLE_ANTITHETIC_NOISE,
        "normalize_mse": BABBLE_NORMALIZE_MSE,
        "cosine_weight": BABBLE_COSINE_WEIGHT,
        "conditional_delta_weight": BABBLE_CONDITIONAL_DELTA_WEIGHT,
        "final_beams": getattr(result, "beams", None),
        "history_tail": result.history[-20:],
    }
    HARD_PROMPT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
    payload = {
        "prompt": result.prompt,
        "score": result.score,
        "seed_prompt": seed_prompt,
        "history_tail": result.history[-20:],
    }
    READABLE_PROMPT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(result.prompt)
    print("score:", result.score)
    print("saved:", READABLE_PROMPT_JSON)
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

DATASET_SOFT_PROMPT_PATH = OUTPUT_DIR / "mode_04_dataset_soft_prompt.pt"

if RUN_MODE_4_DATASET_TO_SOFT_PROMPT:
    require_models()
    freeze_sa3_and_same()
    run_duration = dataset_effective_duration()
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
        use_chunks=DATASET_USE_CHUNKS,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
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
        duration=run_duration,
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
    run_duration = dataset_effective_duration()
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
        use_chunks=DATASET_USE_CHUNKS,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
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
            duration=run_duration,
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
    run_duration = dataset_effective_duration()
    dataset_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        prompt_from_path=True,
        use_chunks=DATASET_USE_CHUNKS,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
    )
    profile = fit_style_profile(dataset_items, name="positive_dataset")
    save_style_profile(profile, STYLE_PROFILE_PATH)

    latents = sa3.generate_latents(
        prompt=STYLE_PROFILE_TEST_PROMPT,
        duration=run_duration,
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
    play_audio_files([out_path], title="Mode 6 SAME profile styled output")
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

STYLE_DIRECTION_PATH = OUTPUT_DIR / "mode_07_style_direction.npz"
STYLE_DIRECTION_ALPHA = 1.0

if RUN_MODE_7_SAME_DIRECTION:
    require_models()
    run_duration = dataset_effective_duration()
    positive_items = encode_audio_folder_to_items(
        DATASET_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        use_chunks=DATASET_USE_CHUNKS,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
    )
    reference_items = encode_audio_folder_to_items(
        REFERENCE_DIR,
        limit=DATASET_LIMIT,
        duration=DATASET_DURATION,
        use_chunks=DATASET_USE_CHUNKS,
        chunk_duration=DATASET_CHUNK_DURATION,
        hop_duration=DATASET_HOP_DURATION,
        max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
        drop_last=DATASET_DROP_LAST_CHUNK,
    )
    positive_profile = fit_style_profile(positive_items, name="positive")
    reference_profile = fit_style_profile(reference_items, name="reference")
    direction = style_direction(positive_profile, reference_profile, name="positive_minus_reference")
    save_style_direction(direction, STYLE_DIRECTION_PATH)

    latents = sa3.generate_latents(
        prompt=STYLE_PROFILE_TEST_PROMPT,
        duration=run_duration,
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
    play_audio_files([out_path], title="Mode 7 SAME direction output")
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
    play_audio_files(
        [output.audio_path for output in sweep_outputs if output.audio_path],
        labels=[f"alpha {output.alpha:+.2f}" for output in sweep_outputs if output.audio_path],
        title="Mode 8 residual alpha sweep",
    )
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
    run_duration = dataset_effective_duration()
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
        duration=run_duration,
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
    play_audio_files([out_path], title="Mode 11 continuation")
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
## Mode 15. SAME Geometry and Intervention Audit

This is the first serious measurement layer for the seven better operators.

It does not try to make a new sound first. It asks what is measurable in the
current latent collection:

```text
latent geometry:     PCA spectrum, covariance, Mahalanobis distances
periodicity:         autocorrelation lags, boundary mismatch, latent FFT centroid
transport:           full-covariance Gaussian style transport probe
observability:       simple linear probes for existing descriptors/labels
```

The reason is methodological: before steering a control, check whether it is
visible in SAME latents and whether simple operators behave coherently.
"""
    ),
    code(
        r"""
# @title Mode 15. SAME geometry and intervention audit

RUN_MODE_15_GEOMETRY_AUDIT = False

GEOMETRY_AUDIT_OUTPUT = OUTPUT_DIR / "mode_15_same_geometry_audit.json"
GEOMETRY_AUDIT_SOURCE = "dataset"  # "dataset" or "memory"
GEOMETRY_AUDIT_COMPONENTS = 16
GEOMETRY_AUDIT_PERIOD_MIN_LAG = 2
GEOMETRY_AUDIT_PERIOD_MAX_LAG = 128
GEOMETRY_AUDIT_CONTROL_KEYS = []  # Example: ["brightness", "density", "usable_loop"]
GEOMETRY_AUDIT_TRANSPORT_DEMO = False


def load_geometry_audit_items():
    if GEOMETRY_AUDIT_SOURCE == "memory":
        return load_items(MEMORY_DIR)
    if GEOMETRY_AUDIT_SOURCE == "dataset":
        require_models()
        return encode_audio_directory(
            DATASET_DIR,
            limit=DATASET_LIMIT,
            duration=DATASET_DURATION,
            item_id_prefix="geometry_dataset",
            use_chunks=DATASET_USE_CHUNKS,
            chunk_duration=DATASET_CHUNK_DURATION,
            hop_duration=DATASET_HOP_DURATION,
            max_chunks_per_file=DATASET_MAX_CHUNKS_PER_FILE,
            drop_last=DATASET_DROP_LAST_CHUNK,
        )
    raise ValueError("GEOMETRY_AUDIT_SOURCE must be 'dataset' or 'memory'")


if RUN_MODE_15_GEOMETRY_AUDIT:
    items = load_geometry_audit_items()
    if len(items) < 2:
        raise ValueError("Mode 15 needs at least two latent items.")

    geometry = fit_latent_geometry(items, n_components=GEOMETRY_AUDIT_COMPONENTS)
    report = geometry_report(items, n_components=GEOMETRY_AUDIT_COMPONENTS)
    periodic = []
    for item in items:
        p = periodicity_report(
            item,
            min_lag=GEOMETRY_AUDIT_PERIOD_MIN_LAG,
            max_lag=GEOMETRY_AUDIT_PERIOD_MAX_LAG,
        )
        periodic.append(
            {
                "item_id": item.item_id,
                "best_lag": p.best_lag,
                "best_score": p.best_score,
                "boundary_l2": p.boundary_l2,
                "velocity_l2": p.velocity_l2,
                "spectral_centroid": p.spectral_centroid,
            }
        )

    distances = []
    anchor = items[0]
    for item in items[1:]:
        distances.append(
            {
                "from": anchor.item_id,
                "to": item.item_id,
                "mahalanobis_summary_distance": mahalanobis_summary_distance(anchor, item, geometry),
            }
        )

    probes = []
    for key in GEOMETRY_AUDIT_CONTROL_KEYS:
        try:
            probe = fit_linear_control_probe(items, key)
            predictions = [
                {"item_id": item.item_id, "prediction": predict_control(probe, item)}
                for item in items
            ]
            probes.append(
                {
                    "control": key,
                    "item_count": probe.item_count,
                    "train_r2": probe.r2_train,
                    "predictions": predictions,
                }
            )
        except Exception as exc:
            probes.append({"control": key, "error": str(exc)})

    transport_demo = None
    if GEOMETRY_AUDIT_TRANSPORT_DEMO and len(items) >= 2:
        transported = covariance_transport(items[0], items[1:], alpha=1.0)
        transport_demo = {
            "source": items[0].item_id,
            "reference_count": len(items) - 1,
            "source_mean_norm": float(np.linalg.norm(items[0].latent.mean(axis=0))),
            "transported_mean_norm": float(np.linalg.norm(transported.mean(axis=0))),
        }

    payload = {
        "item_count": len(items),
        "source": GEOMETRY_AUDIT_SOURCE,
        "geometry": report,
        "pca_variances": geometry.variances.astype(float).tolist(),
        "periodicity": periodic,
        "anchor_distances": distances,
        "control_probes": probes,
        "transport_demo": transport_demo,
    }
    GEOMETRY_AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GEOMETRY_AUDIT_OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2)[:6000])
    print("saved:", GEOMETRY_AUDIT_OUTPUT)
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
    play_audio_files([out_path], title="Combined chain output")
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
    "dataset_use_chunks": DATASET_USE_CHUNKS,
    "dataset_chunk_duration": DATASET_CHUNK_DURATION,
    "dataset_hop_duration": DATASET_HOP_DURATION,
    "dataset_max_chunks_per_file": DATASET_MAX_CHUNKS_PER_FILE,
    "modes": {
        "0_renoise_variations": RUN_MODE_0_RENOISE_VARIATIONS,
        "0h_neural_latent_dsp": RUN_MODE_0H_LATENT_DSP,
        "1_audio_to_soft_prompt": RUN_MODE_1_AUDIO_TO_SOFT_PROMPT,
        "1b_generate_with_soft_prompt": RUN_MODE_1B_GENERATE_WITH_SOFT_PROMPT,
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
        "15_geometry_audit": RUN_MODE_15_GEOMETRY_AUDIT,
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
