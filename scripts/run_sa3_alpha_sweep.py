from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.adapters.audioscope_sa3 import SteeringVectors
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter
from latent_audio_primitives.experiments.sa3_sweeps import alpha_sweep
from _runtime import add_torch_runtime_args, model_half_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SA3 residual steering alpha sweep.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--vectors", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--alphas", default="-8,-4,0,4,8")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--output", default="outputs/sweeps")
    add_torch_runtime_args(parser)
    args = parser.parse_args()

    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=model_half_from_args(args))
    sa3 = StableAudio3Adapter(model=model, model_name=args.model)
    vectors = SteeringVectors.load(args.vectors)
    alphas = [float(value.strip()) for value in args.alphas.split(",") if value.strip()]

    outputs = alpha_sweep(
        sa3,
        prompt=args.prompt,
        vectors=vectors,
        alphas=alphas,
        output_dir=Path(args.output),
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        layer=args.layer if args.layer >= 0 else None,
        save_audio=True,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
