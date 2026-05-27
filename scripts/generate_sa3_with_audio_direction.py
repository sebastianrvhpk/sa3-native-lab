from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter
from latent_audio_primitives.audio_vectors import apply_frame_direction
from latent_audio_primitives.style import load_style_direction
from _runtime import add_torch_runtime_args, model_half_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SA3 latents, apply an audio-file-derived SAME frame direction, decode audio.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--direction", required=True, help="frame_direction.npz from extract_audio_style_vectors.py")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--save-original", action="store_true")
    add_torch_runtime_args(parser)
    args = parser.parse_args()

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=model_half_from_args(args))
    sa3 = StableAudio3Adapter(model=model, model_name=args.model)
    direction = load_style_direction(args.direction)

    latents = sa3.generate_latents(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    directed = _apply_frame_direction_to_batch(latents, direction, alpha=args.alpha)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        audio = sa3.decode_latents(directed).float().clamp(-1, 1).cpu()
    torchaudio.save(str(out_path), audio[0], sa3.sample_rate)
    print(f"saved directed audio {out_path}")

    if args.save_original:
        original_path = out_path.with_name(out_path.stem + "_original.wav")
        with torch.inference_mode():
            original = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
        torchaudio.save(str(original_path), original[0], sa3.sample_rate)
        print(f"saved original audio {original_path}")


def _apply_frame_direction_to_batch(latents, direction, *, alpha: float):
    import torch

    is_tensor = isinstance(latents, torch.Tensor)
    arr = latents.detach().float().cpu().numpy() if is_tensor else np.asarray(latents, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T, got {arr.shape}")
    directed = []
    for latent in arr:
        directed_time_major = apply_frame_direction(latent.T, direction, alpha=alpha)
        directed.append(directed_time_major.T)
    directed_arr = np.stack(directed).astype(np.float32)
    if is_tensor:
        return torch.from_numpy(directed_arr).to(device=latents.device, dtype=latents.dtype)
    return directed_arr


if __name__ == "__main__":
    main()
