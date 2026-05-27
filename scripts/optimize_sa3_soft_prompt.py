from __future__ import annotations

import argparse

from latent_audio_primitives.experiments.soft_prompt import optimize_soft_prompt_from_latents
from _runtime import add_torch_runtime_args, model_half_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize SA3 continuous conditioning tensors from target audio.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--target-audio", required=True)
    parser.add_argument("--seed-prompt", default="audio texture")
    parser.add_argument("--output", required=True)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means infer target file duration")
    parser.add_argument("--optimization-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--reg-weight", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-keys", default="prompt", help="Comma-separated conditioner keys to optimize")
    parser.add_argument(
        "--velocity-convention",
        choices=["noise_minus_data", "data_minus_noise"],
        default="noise_minus_data",
        help="Rectified-flow target velocity convention to use for the inversion loss.",
    )
    add_torch_runtime_args(parser)
    args = parser.parse_args()

    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=model_half_from_args(args))
    audio, sample_rate = torchaudio.load(args.target_audio)
    duration = args.duration or (audio.shape[-1] / sample_rate)
    conditioning = [{"prompt": args.seed_prompt, "seconds_total": duration}]
    audio_sample_size = model._adapt_sample_size(conditioning, 5292032, duration_padding_sec=6.0)
    target_latents, _ = model._encode_audio_input((sample_rate, audio), audio_sample_size)

    train_keys = tuple(key.strip() for key in args.train_keys.split(",") if key.strip())
    state = optimize_soft_prompt_from_latents(
        model,
        target_latents,
        seed_prompt=args.seed_prompt,
        duration=duration,
        optimization_steps=args.optimization_steps,
        lr=args.lr,
        train_keys=train_keys,
        reg_weight=args.reg_weight,
        seed=args.seed,
        velocity_convention=args.velocity_convention,
    )
    state.metadata["target_audio"] = args.target_audio
    state.metadata["model_name"] = args.model
    state.save(args.output)
    print(f"saved soft prompt state {args.output}")
    if state.losses:
        print(f"loss_first={state.losses[0]:.6f} loss_last={state.losses[-1]:.6f}")


if __name__ == "__main__":
    main()
