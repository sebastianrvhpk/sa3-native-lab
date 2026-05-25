from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.adapters.stable_audio3 import SAMEAutoencoderAdapter
from latent_audio_primitives.io import save_items


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode an audio dataset into SAME latent memory.")
    parser.add_argument("--input", required=True, help="Audio folder")
    parser.add_argument("--output", required=True, help="Output memory folder")
    parser.add_argument("--model", default="same-l", help="SAME autoencoder, e.g. same-l for SA3 medium")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunked", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-duration", type=float, default=0.0, help="If set, encode each file as fixed-duration windows.")
    parser.add_argument("--hop-duration", type=float, default=0.0, help="Window hop in seconds. Defaults to chunk duration.")
    parser.add_argument("--max-chunks-per-file", type=int, default=0)
    parser.add_argument("--drop-last-chunk", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input)
    paths = [path for path in sorted(input_dir.rglob("*")) if path.suffix.lower() in AUDIO_EXTS]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no audio files found in {input_dir}")

    same = SAMEAutoencoderAdapter.from_pretrained(args.model, device=args.device)
    items = []
    for index, path in enumerate(paths):
        print(f"[{index + 1}/{len(paths)}] encoding {path}")
        if args.chunk_duration:
            file_items = same.encode_file_chunks(
                path,
                chunk_duration=args.chunk_duration,
                hop_duration=args.hop_duration or None,
                max_chunks=args.max_chunks_per_file or None,
                drop_last=args.drop_last_chunk,
                item_id_prefix=path.stem,
                chunked=args.chunked,
                metadata={"dataset_path": str(path), "same_model": args.model},
            )
            items.extend(file_items)
            print(f"  -> {len(file_items)} chunks")
        else:
            item = same.encode_file(
                path,
                item_id=path.stem,
                chunked=args.chunked,
                metadata={"dataset_path": str(path), "same_model": args.model},
            )
            items.append(item)

    save_items(items, args.output)
    print(f"saved {len(items)} items to {args.output}")


if __name__ == "__main__":
    main()
