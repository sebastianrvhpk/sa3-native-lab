from __future__ import annotations

import argparse
from pathlib import Path

from latent_audio_primitives.experiments.soft_prompt import SoftPromptState, generate_with_soft_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate with an optimized SA3 soft prompt/conditioning state.")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--soft-prompt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torchaudio
    from stable_audio_3 import StableAudioModel

    model = StableAudioModel.from_pretrained(args.model, device="cuda", model_half=True)
    state = SoftPromptState.load(args.soft_prompt)
    audio = generate_with_soft_prompt(
        model,
        state,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        return_latents=False,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.float().clamp(-1, 1).cpu()
    torchaudio.save(str(out_path), audio[0], model.model.sample_rate)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
