from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter
from latent_audio_primitives.style import apply_profile_attraction, load_style_profile
from _runtime import add_torch_runtime_args, model_half_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SA3 latents, push them toward a SAME dataset style profile, decode audio.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--no-std", action="store_true", help="Only shift mean; do not match target std")
    parser.add_argument("--save-original", action="store_true")
    add_torch_runtime_args(parser)
    args = parser.parse_args()

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=model_half_from_args(args))
    sa3 = StableAudio3Adapter(model=model, model_name=args.model)
    profile = load_style_profile(args.profile)

    latents = sa3.generate_latents(
        prompt=args.prompt,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
    )
    styled = _apply_profile_to_batch(latents, profile, alpha=args.alpha, match_std=not args.no_std)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        audio = sa3.decode_latents(styled).float().clamp(-1, 1).cpu()
    torchaudio.save(str(out_path), audio[0], sa3.sample_rate)
    print(f"saved styled audio {out_path}")

    if args.save_original:
        original_path = out_path.with_name(out_path.stem + "_original.wav")
        with torch.inference_mode():
            original = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
        torchaudio.save(str(original_path), original[0], sa3.sample_rate)
        print(f"saved original audio {original_path}")


def _apply_profile_to_batch(latents, profile, *, alpha: float, match_std: bool):
    import torch

    is_tensor = isinstance(latents, torch.Tensor)
    arr = latents.detach().float().cpu().numpy() if is_tensor else np.asarray(latents, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"expected latents shaped B x C x T, got {arr.shape}")

    styled = []
    for latent in arr:
        time_major = latent.T
        styled_time_major = apply_profile_attraction(time_major, profile, alpha=alpha, match_std=match_std)
        styled.append(styled_time_major.T)
    styled_arr = np.stack(styled).astype(np.float32)
    if is_tensor:
        return torch.from_numpy(styled_arr).to(device=latents.device, dtype=latents.dtype)
    return styled_arr


if __name__ == "__main__":
    main()
