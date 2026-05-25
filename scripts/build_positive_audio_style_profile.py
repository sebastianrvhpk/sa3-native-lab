from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter
from latent_audio_primitives.io import save_items
from latent_audio_primitives.style import fit_style_profile, save_style_profile


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode one positive audio folder and build a SAME style profile.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--memory-output", required=True)
    parser.add_argument("--profile-output", required=True)
    parser.add_argument("--name", default="positive_style")
    parser.add_argument("--model", default="same-l")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = [path for path in sorted(Path(args.input).rglob("*")) if path.suffix.lower() in AUDIO_EXTS]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no audio files found in {args.input}")

    same = SAMEAutoencoderAdapter.from_pretrained(args.model, device=args.device)
    items = []
    for index, path in enumerate(paths):
        print(f"[{index + 1}/{len(paths)}] {path}")
        items.append(
            same.encode_file(
                path,
                item_id=path.stem,
                chunked=args.chunked,
                metadata={"audio_path": str(path), "same_model": args.model},
            )
        )

    save_items(items, args.memory_output)
    profile = fit_style_profile(
        items,
        name=args.name,
        metadata={"source_audio": str(Path(args.input)), "same_model": args.model},
    )
    save_style_profile(profile, args.profile_output)
    print(f"saved memory {args.memory_output}")
    print(f"saved profile {args.profile_output}")


if __name__ == "__main__":
    main()
