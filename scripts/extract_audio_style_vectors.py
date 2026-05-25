from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter
from latent_audio_primitives.audio_vectors import (
    frame_mean_direction,
    save_audio_frame_direction,
    save_summary_direction,
    summary_direction,
)
from latent_audio_primitives.io import save_items


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract steering directions from positive/negative audio folders via SAME.")
    parser.add_argument("--positive", required=True, help="Folder of positive/target audio examples")
    parser.add_argument("--negative", required=True, help="Folder of negative/reference audio examples")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--model", default="same-l", help="SAME autoencoder, e.g. same-l for SA3 medium")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--name", default="audio_direction")
    parser.add_argument("--normalize-frame", action="store_true")
    args = parser.parse_args()

    positive_paths = _audio_paths(args.positive, args.limit)
    negative_paths = _audio_paths(args.negative, args.limit)
    if not positive_paths:
        raise SystemExit(f"no positive audio files found in {args.positive}")
    if not negative_paths:
        raise SystemExit(f"no negative audio files found in {args.negative}")

    same = SAMEAutoencoderAdapter.from_pretrained(args.model, device=args.device)
    positive_items = _encode_paths(same, positive_paths, prefix="pos", chunked=args.chunked, same_model=args.model)
    negative_items = _encode_paths(same, negative_paths, prefix="neg", chunked=args.chunked, same_model=args.model)

    output = Path(args.output)
    save_items(positive_items, output / "positive_memory")
    save_items(negative_items, output / "negative_memory")

    frame_direction = frame_mean_direction(
        positive_items,
        negative_items,
        name=args.name,
        normalize=args.normalize_frame,
        metadata={
            "positive": str(Path(args.positive)),
            "negative": str(Path(args.negative)),
            "same_model": args.model,
        },
    )
    summary = summary_direction(
        positive_items,
        negative_items,
        name=args.name + "_summary",
        metadata={
            "positive": str(Path(args.positive)),
            "negative": str(Path(args.negative)),
            "same_model": args.model,
        },
    )

    save_audio_frame_direction(frame_direction, output / "frame_direction.npz")
    save_summary_direction(summary, output / "summary_direction.npz")
    print(f"saved frame direction: {output / 'frame_direction.npz'}")
    print(f"saved summary direction: {output / 'summary_direction.npz'}")
    print(f"positive_items={len(positive_items)} negative_items={len(negative_items)}")


def _audio_paths(root: str | Path, limit: int) -> list[Path]:
    paths = [path for path in sorted(Path(root).rglob("*")) if path.suffix.lower() in AUDIO_EXTS]
    return paths[:limit] if limit else paths


def _encode_paths(same: SAMEAutoencoderAdapter, paths: list[Path], *, prefix: str, chunked: bool, same_model: str):
    items = []
    for index, path in enumerate(paths):
        print(f"[{prefix} {index + 1}/{len(paths)}] {path}")
        items.append(
            same.encode_file(
                path,
                item_id=f"{prefix}_{path.stem}",
                chunked=chunked,
                metadata={"audio_path": str(path), "same_model": same_model},
            )
        )
    return items


if __name__ == "__main__":
    main()
