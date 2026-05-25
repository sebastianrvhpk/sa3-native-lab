"""Colab starter for Stable Audio 3 Medium on an L4 GPU.

This is written as a notebook-style Python script. In Colab, copy each section
into cells, or upload this file and run it section by section.

Assumptions:
    - You selected an L4 GPU runtime.
    - You accepted the Hugging Face conditions for
      stabilityai/stable-audio-3-medium.
    - The combined SA3 Native Lab repo is available at /content/sa3-native-lab,
      or you edit COMBINED_REPO_URL below and let this script clone it.
"""

# %% [markdown]
# # 0. Runtime check

# %%
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: str | Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True, env=env, cwd=cwd)


run(["nvidia-smi"])

# %% [markdown]
# # 1. Clone and install the combined repo plus Flash Attention
#
# Colab images change over time. This section installs this combined repo,
# which already contains the official stable_audio_3 package plus the research
# layer.

# %%
COMBINED_REPO_URL = "https://github.com/sebastianrvhpk/sa3-native-lab.git"
REPO = Path("/content/sa3-native-lab")
if not REPO.exists():
    if not COMBINED_REPO_URL:
        raise RuntimeError("Set COMBINED_REPO_URL to your combined repo URL.")
    run(["git", "clone", COMBINED_REPO_URL, str(REPO)])

run([sys.executable, "-m", "pip", "install", "-q", "uv"])
run(
    [
        "uv",
        "pip",
        "install",
        "--system",
        "torch==2.7.1",
        "torchaudio==2.7.1",
        "--index-url",
        "https://download.pytorch.org/whl/cu126",
    ]
)
run(["uv", "pip", "install", "--system", "-e", str(REPO)])

flash_env = os.environ.copy()
flash_env["TORCH_CUDA_ARCH_LIST"] = "8.9"
flash_env["MAX_JOBS"] = "2"

run(
    [
        "uv",
        "pip",
        "install",
        "--system",
        "flash-attn",
        "--no-build-isolation",
        "--no-cache-dir",
        "--no-deps",
    ],
    env=flash_env,
)

# %% [markdown]
# # 3. Hugging Face auth
#
# You must accept the model conditions on:
# https://huggingface.co/stabilityai/stable-audio-3-medium

# %%
from huggingface_hub import login

login()

# %% [markdown]
# # 4. Load Stable Audio 3 Medium

# %%
import torch
from stable_audio_3 import StableAudioModel

assert torch.cuda.is_available(), "CUDA is required for SA3 medium."

model = StableAudioModel.from_pretrained("medium", device="cuda", model_half=True)
print("sample_rate", model.model.sample_rate)
print("downsampling_ratio", model.model.pretransform.downsampling_ratio)
print("latent_rate", model.model.sample_rate / model.model.pretransform.downsampling_ratio)

# %% [markdown]
# # 5. Smoke test: generate SAME latents

# %%
prompt = "a sparse glassy ambient loop, slow evolving texture, detailed stereo field"

latents = model.generate(
    prompt=prompt,
    duration=10,
    steps=8,
    cfg_scale=1.0,
    seed=42,
    return_latents=True,
)

print("latents", tuple(latents.shape), latents.dtype, latents.device)

# %% [markdown]
# # 6. Decode and save audio

# %%
import torchaudio

with torch.inference_mode():
    audio = model.model.pretransform.decode(latents)
audio = audio.float().clamp(-1, 1).cpu()

out_path = "sa3_medium_l4_smoke.wav"
torchaudio.save(out_path, audio[0], model.model.sample_rate)
print("saved", out_path)

# %% [markdown]
# # 7. Convert SA3 latents into latent-memory items

# %%
from latent_audio_primitives import LatentMemoryIndex
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter

sa3 = StableAudio3Adapter(model=model, model_name="medium")

items = sa3.generate_items(
    prompt=prompt,
    duration=10,
    item_id_prefix="glass-loop",
    steps=8,
    cfg_scale=1.0,
    seed=43,
)

print(items[0].shallow_metadata())

index = LatentMemoryIndex(items)

# %% [markdown]
# # 8. Generate a tiny memory and retrieve neighbors

# %%
prompts = [
    "a sparse glassy ambient loop, slow evolving texture",
    "a dense shimmering ambient loop, bright granular movement",
    "a dark low drone with distant metallic harmonics",
]

memory_items = []
for prompt_index, candidate_prompt in enumerate(prompts):
    for seed in range(3):
        generated = sa3.generate_items(
            prompt=candidate_prompt,
            duration=10,
            item_id_prefix=f"p{prompt_index}-seed{seed}",
            steps=8,
            cfg_scale=1.0,
            seed=seed,
        )
        memory_items.extend(generated)

index = LatentMemoryIndex(memory_items)
query_item = memory_items[0]
results = index.query(query_item, top_k=5, exclude_id=query_item.item_id)

for result in results:
    print(result.item_id, round(result.score, 4), result.item.prompt)

# %% [markdown]
# # 9. Optional: continuation latents

# %%
# import torchaudio
# source_audio = torchaudio.load("source.wav")
# continued = sa3.continuation_latents(
#     prompt="continue as a darker, wider ambient outro",
#     inpaint_audio=source_audio,
#     source_duration=10.0,
#     target_duration=18.0,
#     steps=8,
#     cfg_scale=1.0,
#     seed=123,
# )
# print(tuple(continued.shape))

# %% [markdown]
# # 10. Optional: audioscope-style residual steering

# %%
# # Extract vectors first:
# !python /content/sa3-native-lab/scripts/extract_sa3_vectors.py \
#   --model medium \
#   --axis valence \
#   --num-pairs 2 \
#   --duration 6 \
#   --steps 8 \
#   --cfg-scale 1.0 \
#   --output /content/sa3_latent_lab/vectors/valence
#
# from latent_audio_primitives.adapters.audioscope_sa3 import ResidualSteerer, SteeringVectors
#
# vectors = SteeringVectors.load("/content/sa3_latent_lab/vectors/valence/steering_vectors.pt")
# steerer = ResidualSteerer(model, vectors, layer=11)
#
# with steerer.steer(alpha=8.0):
#     steered_latents = model.generate(
#         prompt="a cinematic ambient piano phrase",
#         duration=10,
#         seed=123,
#         return_latents=True,
#     )
# print(tuple(steered_latents.shape))

# %% [markdown]
# # 11. Optional: launch Gradio interface

# %%
# !python /content/sa3-native-lab/colab/sa3_latent_interface.py
