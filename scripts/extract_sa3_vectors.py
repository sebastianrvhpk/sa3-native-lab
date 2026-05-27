from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.experiments.activation_vectors import SA3ActivationVectorExtractor
from latent_audio_primitives.experiments.prompt_pairs import DEFAULT_PROMPT_PAIRS
from _runtime import add_torch_runtime_args, model_half_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audioscope-style SA3 residual steering vectors.")
    parser.add_argument("--model", default="medium", help="Stable Audio 3 model id, e.g. medium")
    parser.add_argument("--axis", default="valence", help="Prompt-pair axis or 'all'")
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices, blank for all layers")
    parser.add_argument("--output", default="outputs/vectors/valence")
    add_torch_runtime_args(parser)
    args = parser.parse_args()

    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device=args.device, model_half=model_half_from_args(args))
    pairs = [pair for pair in DEFAULT_PROMPT_PAIRS if args.axis == "all" or pair.axis == args.axis]
    layers = [int(value) for value in args.layers.split(",") if value.strip()] or None

    extractor = SA3ActivationVectorExtractor(model, layer_indices=layers, cpu_offload=True)
    result = extractor.extract(
        pairs=pairs,
        num_pairs=args.num_pairs,
        duration=args.duration,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        probe=True,
    )
    out_dir = Path(args.output)
    result.save(out_dir)
    print(f"saved {out_dir / 'steering_vectors.pt'}")
    print(f"best_layer={result.vectors.best_layer}")
    print(f"probe_accuracy={result.vectors.probe_accuracy}")


if __name__ == "__main__":
    main()
