from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.io import load_items
from latent_audio_primitives.style import (
    fit_style_profile,
    save_style_direction,
    save_style_profile,
    style_direction,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SAME latent style profile/direction from memory folders.")
    parser.add_argument("--target", required=True, help="Target dataset memory folder")
    parser.add_argument("--output", required=True, help="Output .npz profile path")
    parser.add_argument("--name", default="target_style")
    parser.add_argument("--reference", default="", help="Optional reference memory folder for target-reference direction")
    parser.add_argument("--direction-output", default="", help="Optional output .npz direction path")
    args = parser.parse_args()

    target_items = load_items(args.target)
    target = fit_style_profile(
        target_items,
        name=args.name,
        metadata={"source_memory": str(Path(args.target))},
    )
    save_style_profile(target, args.output)
    print(f"saved profile {args.output}")
    print(f"items={target.item_count} dim={target.dim}")

    if args.reference:
        reference_items = load_items(args.reference)
        reference = fit_style_profile(
            reference_items,
            name="reference_style",
            metadata={"source_memory": str(Path(args.reference))},
        )
        direction = style_direction(target, reference)
        direction_output = args.direction_output or str(Path(args.output).with_name(Path(args.output).stem + "_direction.npz"))
        save_style_direction(direction, direction_output)
        print(f"saved direction {direction_output}")


if __name__ == "__main__":
    main()
