from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.experiments.audio_residual_vectors import SA3AudioResidualVectorExtractor


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract SA3 residual-stream steering vectors from positive/negative audio folders.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--positive", required=True)
    parser.add_argument("--negative", default="", help="Optional negative/reference folder")
    parser.add_argument("--baseline", choices=["prompt", "negative_audio"], default="prompt")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="audio texture")
    parser.add_argument("--duration", type=float, default=0.0, help="0 means infer each file duration")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-noise-level", type=float, default=0.35)
    parser.add_argument("--layers", default="", help="Comma-separated layer indices, blank for all layers")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    from stable_audio_3 import StableAudioModel

    positive_paths = _audio_paths(args.positive, args.limit)
    negative_paths = _audio_paths(args.negative, args.limit) if args.negative else []
    if not positive_paths:
        raise SystemExit(f"no positive audio files found in {args.positive}")
    if args.baseline == "negative_audio" and not negative_paths:
        raise SystemExit(f"no negative audio files found in {args.negative}")

    layers = [int(value) for value in args.layers.split(",") if value.strip()] or None
    model = StableAudioModel.from_pretrained(args.model, device="cuda", model_half=True)
    extractor = SA3AudioResidualVectorExtractor(model, layer_indices=layers, cpu_offload=True)
    result = extractor.extract(
        positive_paths,
        negative_paths,
        prompt=args.prompt,
        duration=args.duration or None,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        init_noise_level=args.init_noise_level,
        baseline_mode=args.baseline,
        probe=True,
    )
    output = Path(args.output)
    result.save(output)
    print(f"saved {output / 'residual_audio_vectors.pt'}")
    print(f"best_layer={result.vectors.best_layer}")
    print(f"probe_accuracy={result.vectors.probe_accuracy}")


def _audio_paths(root: str | Path, limit: int) -> list[Path]:
    paths = [path for path in sorted(Path(root).rglob("*")) if path.suffix.lower() in AUDIO_EXTS]
    return paths[:limit] if limit else paths


if __name__ == "__main__":
    main()
