"""Alpha-sweep helpers for auditioning SA3 residual steering vectors."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from latent_audio_primitives.adapters.sa3_residual_hooks import ResidualSteerer, SteeringVectors
from latent_audio_primitives.adapters.stable_audio3 import StableAudio3Adapter, latents_to_items
from latent_audio_primitives.io import save_items


@dataclass(slots=True)
class SweepOutput:
    """Artifact paths for one residual-steering alpha value."""

    alpha: float
    item_id: str
    latent_path: str | None = None
    audio_path: str | None = None


def alpha_sweep(
    sa3: StableAudio3Adapter,
    *,
    prompt: str,
    vectors: SteeringVectors,
    alphas: list[float],
    output_dir: str | Path,
    duration: float = 8.0,
    steps: int = 8,
    cfg_scale: float = 1.0,
    seed: int = 42,
    layer: int | None = None,
    top_k: int = 1,
    save_audio: bool = True,
    generate_kwargs: dict[str, Any] | None = None,
) -> list[SweepOutput]:
    """Generate a residual-steering alpha sweep and save latents/audio."""

    torch = _require_torch()
    torchaudio = _require_torchaudio() if save_audio else None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_kwargs = generate_kwargs or {}
    steerer = ResidualSteerer(sa3.model, vectors, layer=layer, top_k=top_k)

    outputs: list[SweepOutput] = []
    all_items = []
    for alpha in alphas:
        item_prefix = f"alpha_{_alpha_token(alpha)}"
        with steerer.steer(alpha=alpha):
            latents = sa3.generate_latents(
                prompt=prompt,
                duration=duration,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                **generate_kwargs,
            )
        items = latents_to_items(
            latents,
            item_id_prefix=item_prefix,
            latent_rate=sa3.latent_rate,
            sample_rate=sa3.sample_rate,
            prompt=prompt,
            metadata={
                "alpha": alpha,
                "seed": seed,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "duration": duration,
                "steering_layers": steerer.layer_indices,
            },
        )
        all_items.extend(items)
        audio_path = None
        if save_audio:
            audio = sa3.decode_latents(latents).float().clamp(-1, 1).cpu()
            audio_path = str(output_dir / f"{item_prefix}.wav")
            torchaudio.save(audio_path, audio[0], sa3.sample_rate)
        outputs.append(SweepOutput(alpha=alpha, item_id=items[0].item_id, audio_path=audio_path))
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    item_dirs = save_items(all_items, output_dir / "latents")
    for output, item_dir in zip(outputs, item_dirs):
        output.latent_path = str(item_dir)
    with (output_dir / "sweep.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(output) for output in outputs], f, indent=2, sort_keys=True)
    return outputs


def _alpha_token(alpha: float) -> str:
    sign = "pos" if alpha >= 0 else "neg"
    return f"{sign}{abs(alpha):.2f}".replace(".", "p")


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for SA3 sweeps.") from exc
    return torch


def _require_torchaudio():
    try:
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio is required when save_audio=True.") from exc
    return torchaudio
