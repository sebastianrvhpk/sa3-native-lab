"""Thin adapters around upstream Stable Audio 3 and SAME model wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from latent_audio_primitives.schema import LatentItem


class StableAudio3IntegrationError(RuntimeError):
    """Raised when the Stable Audio 3 adapter cannot access the real package."""


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise StableAudio3IntegrationError("PyTorch is required for Stable Audio 3 integration.") from exc
    return torch


def _require_stable_audio_3():
    try:
        from stable_audio_3 import AutoencoderModel, StableAudioModel
    except ImportError as exc:
        raise StableAudio3IntegrationError(
            "stable_audio_3 is not importable. Clone and install "
            "https://github.com/Stability-AI/stable-audio-3 or add that repository to PYTHONPATH."
        ) from exc
    return StableAudioModel, AutoencoderModel


def _tensor_to_numpy(value: Any) -> np.ndarray:
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _latent_batch_to_numpy(latents: Any) -> np.ndarray:
    arr = _tensor_to_numpy(latents)
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"expected latents shaped (B, C, T) or (C, T), got {arr.shape}")
    return arr.astype(np.float32, copy=False)


def _model_core(model: Any) -> Any:
    """Return the official ConditionedDiffusionModelWrapper when available."""

    return getattr(model, "model", model)


def _pretransform(model: Any) -> Any | None:
    if hasattr(model, "same"):
        return model.same
    core = _model_core(model)
    return getattr(core, "pretransform", None)


def _sample_rate(model: Any, model_config: dict[str, Any] | None = None) -> int:
    core = _model_core(model)
    if hasattr(core, "sample_rate"):
        return int(core.sample_rate)
    if model_config and "sample_rate" in model_config:
        return int(model_config["sample_rate"])
    raise StableAudio3IntegrationError("Could not infer Stable Audio 3 sample rate.")


def _downsampling_ratio(model: Any, fallback: int = 4096) -> int:
    pretransform = _pretransform(model)
    if pretransform is not None and hasattr(pretransform, "downsampling_ratio"):
        return int(pretransform.downsampling_ratio)
    return fallback


def stable_audio3_latent_rate(model: Any, model_config: dict[str, Any] | None = None) -> float:
    """Return the SAME latent frame rate implied by sample rate and downsampling."""

    return _sample_rate(model, model_config) / _downsampling_ratio(model)


def audio_chunk_windows(
    num_frames: int,
    sample_rate: int,
    chunk_duration: float,
    *,
    hop_duration: float | None = None,
    max_chunks: int | None = None,
    drop_last: bool = False,
) -> list[dict[str, float | int | bool]]:
    """Return deterministic audio chunk windows in sample units.

    The windows are designed for long DJ/live files: each window has a stable
    sample offset, a requested read length, and metadata in seconds. If
    ``drop_last`` is false, the final short window is kept and marked for
    padding by the caller.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if hop_duration is not None and hop_duration <= 0:
        raise ValueError("hop_duration must be positive when provided")
    if max_chunks is not None and max_chunks <= 0:
        return []

    chunk_frames = max(1, int(round(chunk_duration * sample_rate)))
    hop_frames = max(1, int(round((hop_duration if hop_duration is not None else chunk_duration) * sample_rate)))
    windows: list[dict[str, float | int | bool]] = []
    start = 0

    while start < num_frames:
        available = num_frames - start
        if available < chunk_frames and drop_last:
            break
        read_frames = min(chunk_frames, available)
        windows.append(
            {
                "chunk_index": len(windows),
                "frame_offset": start,
                "num_frames": read_frames,
                "target_num_frames": chunk_frames,
                "pad_to_target": read_frames < chunk_frames,
                "start_seconds": start / sample_rate,
                "duration_seconds": read_frames / sample_rate,
                "target_duration_seconds": chunk_frames / sample_rate,
                "hop_seconds": hop_frames / sample_rate,
            }
        )
        if max_chunks is not None and len(windows) >= max_chunks:
            break
        start += hop_frames

    return windows


def latents_to_items(
    latents: Any,
    *,
    item_id_prefix: str,
    latent_rate: float,
    sample_rate: int | None = None,
    prompt: str | list[str] | None = None,
    descriptors: list[dict[str, float]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[LatentItem]:
    """Convert official SA3/SAME latents ``(B, C, T)`` into memory items."""

    batch = _latent_batch_to_numpy(latents)
    prompts = prompt if isinstance(prompt, list) else [prompt] * batch.shape[0]
    descriptors = descriptors or [{} for _ in range(batch.shape[0])]
    if len(prompts) != batch.shape[0]:
        raise ValueError("prompt list length must match latent batch")
    if len(descriptors) != batch.shape[0]:
        raise ValueError("descriptors length must match latent batch")

    items = []
    for index, latent in enumerate(batch):
        item_metadata = dict(metadata or {})
        item_metadata["batch_index"] = index
        items.append(
            LatentItem.from_channel_first(
                item_id=f"{item_id_prefix}-{index:04d}",
                latent=latent,
                latent_rate=latent_rate,
                sample_rate=sample_rate,
                prompt=prompts[index],
                descriptors=descriptors[index],
                metadata=item_metadata,
            )
        )
    return items


@dataclass(slots=True)
class StableAudio3Adapter:
    """Adapter over the official ``stable_audio_3.StableAudioModel`` wrapper."""

    model: Any
    model_name: str | None = None
    model_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "small-music",
        *,
        device: str | None = None,
        model_half: bool = True,
    ) -> "StableAudio3Adapter":
        StableAudioModel, _ = _require_stable_audio_3()
        model = StableAudioModel.from_pretrained(model_name, device=device, model_half=model_half)
        return cls(model=model, model_name=model_name, model_config=getattr(model, "model_config", None))

    @property
    def sample_rate(self) -> int:
        return _sample_rate(self.model, self.model_config)

    @property
    def downsampling_ratio(self) -> int:
        return _downsampling_ratio(self.model)

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    def generate_latents(
        self,
        *,
        prompt: str | list[str],
        duration: float | list[float],
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        seed: int = -1,
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate SA3 SAME latents using the released ``return_latents`` path."""

        return self.model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            seed=seed,
            return_latents=True,
            **kwargs,
        )

    def generate_items(
        self,
        *,
        prompt: str | list[str],
        duration: float | list[float],
        item_id_prefix: str,
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        seed: int = -1,
        negative_prompt: str | list[str] | None = None,
        **kwargs: Any,
    ) -> list[LatentItem]:
        latents = self.generate_latents(
            prompt=prompt,
            negative_prompt=negative_prompt,
            duration=duration,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        return latents_to_items(
            latents,
            item_id_prefix=item_id_prefix,
            latent_rate=self.latent_rate,
            sample_rate=self.sample_rate,
            prompt=prompt,
            metadata={
                "source": "stable_audio_3.generate",
                "model_name": self.model_name,
                "duration": duration,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
            },
        )

    def continuation_latents(
        self,
        *,
        prompt: str,
        inpaint_audio: tuple[int, Any],
        source_duration: float,
        target_duration: float,
        steps: int = 8,
        cfg_scale: float = 1.0,
        seed: int = -1,
        **kwargs: Any,
    ) -> Any:
        """Generate a continuation by masking from source end to target end."""

        return self.model.generate(
            prompt=prompt,
            duration=target_duration,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            inpaint_audio=inpaint_audio,
            inpaint_mask_start_seconds=source_duration,
            inpaint_mask_end_seconds=target_duration,
            return_latents=True,
            **kwargs,
        )

    def decode_latents(self, latents: Any, **decode_kwargs: Any) -> Any:
        """Decode SAME latents through the model pretransform."""

        pretransform = _pretransform(self.model)
        if pretransform is None or not hasattr(pretransform, "decode"):
            raise StableAudio3IntegrationError("Loaded SA3 model does not expose a SAME pretransform decoder.")
        torch = _require_torch()
        with torch.inference_mode():
            return pretransform.decode(latents, **decode_kwargs)


@dataclass(slots=True)
class SAMEAutoencoderAdapter:
    """Adapter over the official ``stable_audio_3.AutoencoderModel`` wrapper."""

    autoencoder: Any
    model_name: str | None = None

    @classmethod
    def from_pretrained(cls, model_name: str = "same-s", *, device: str | None = None) -> "SAMEAutoencoderAdapter":
        _, AutoencoderModel = _require_stable_audio_3()
        autoencoder = AutoencoderModel.from_pretrained(model_name, device=device)
        return cls(autoencoder=autoencoder, model_name=model_name)

    @property
    def sample_rate(self) -> int:
        return int(self.autoencoder.sample_rate)

    @property
    def downsampling_ratio(self) -> int:
        return int(getattr(self.autoencoder.autoencoder, "downsampling_ratio", 4096))

    @property
    def latent_rate(self) -> float:
        return self.sample_rate / self.downsampling_ratio

    def encode_tensor(
        self,
        audio: Any,
        sample_rate: int,
        *,
        item_id: str,
        prompt: str | None = None,
        chunked: bool = False,
        descriptors: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        **encode_kwargs: Any,
    ) -> LatentItem:
        latents = self.autoencoder.encode(audio, sample_rate, chunked=chunked, **encode_kwargs)
        return latents_to_items(
            latents,
            item_id_prefix=item_id,
            latent_rate=self.latent_rate,
            sample_rate=self.sample_rate,
            prompt=prompt,
            descriptors=[descriptors or {}],
            metadata={"source": "stable_audio_3.AutoencoderModel.encode", "model_name": self.model_name, **(metadata or {})},
        )[0]

    def encode_file(
        self,
        path: str | Path,
        *,
        item_id: str | None = None,
        prompt: str | None = None,
        chunked: bool = False,
        descriptors: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        **encode_kwargs: Any,
    ) -> LatentItem:
        try:
            import torchaudio
        except ImportError as exc:
            raise StableAudio3IntegrationError("torchaudio is required for encode_file().") from exc
        path = Path(path)
        audio, sample_rate = torchaudio.load(path)
        item = self.encode_tensor(
            audio,
            sample_rate,
            item_id=item_id or path.stem,
            prompt=prompt,
            chunked=chunked,
            descriptors=descriptors,
            metadata={"path": str(path), **(metadata or {})},
            **encode_kwargs,
        )
        return item

    def encode_file_chunks(
        self,
        path: str | Path,
        *,
        chunk_duration: float,
        hop_duration: float | None = None,
        max_chunks: int | None = None,
        drop_last: bool = False,
        item_id_prefix: str | None = None,
        prompt: str | None = None,
        chunked: bool = False,
        descriptors: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        **encode_kwargs: Any,
    ) -> list[LatentItem]:
        """Encode a long audio file as multiple SAME latent chunks.

        This uses ``torchaudio.load(..., frame_offset=..., num_frames=...)`` so
        long WAV/FLAC/AIFF files can be traversed as windows instead of loading
        an entire DJ set into memory. For compressed formats where frame counts
        are unknown, torchaudio may still need to decode internally.
        """

        try:
            import torch
            import torchaudio
        except ImportError as exc:
            raise StableAudio3IntegrationError("torch and torchaudio are required for encode_file_chunks().") from exc

        path = Path(path)
        info = torchaudio.info(str(path))
        if info.num_frames <= 0:
            raise StableAudio3IntegrationError(
                f"torchaudio could not determine frame count for {path}. "
                "Convert the file to WAV/FLAC first for chunked long-file encoding."
            )

        windows = audio_chunk_windows(
            int(info.num_frames),
            int(info.sample_rate),
            chunk_duration,
            hop_duration=hop_duration,
            max_chunks=max_chunks,
            drop_last=drop_last,
        )
        prefix = item_id_prefix or path.stem
        source_duration = info.num_frames / info.sample_rate
        items: list[LatentItem] = []

        for window in windows:
            audio, sample_rate = torchaudio.load(
                str(path),
                frame_offset=int(window["frame_offset"]),
                num_frames=int(window["num_frames"]),
            )
            target_num_frames = int(window["target_num_frames"])
            if audio.shape[-1] < target_num_frames:
                audio = torch.nn.functional.pad(audio, (0, target_num_frames - audio.shape[-1]))

            chunk_index = int(window["chunk_index"])
            start_ms = int(round(float(window["start_seconds"]) * 1000))
            item_id = f"{prefix}__chunk_{chunk_index:05d}_{start_ms:010d}ms"
            item = self.encode_tensor(
                audio,
                sample_rate,
                item_id=item_id,
                prompt=prompt,
                chunked=chunked,
                descriptors=descriptors,
                metadata={
                    "path": str(path),
                    "source_path": str(path),
                    "source_num_frames": int(info.num_frames),
                    "source_duration_seconds": source_duration,
                    "chunk_index": chunk_index,
                    "chunk_start_seconds": float(window["start_seconds"]),
                    "chunk_duration_seconds": float(window["target_duration_seconds"]),
                    "chunk_read_duration_seconds": float(window["duration_seconds"]),
                    "chunk_hop_seconds": float(window["hop_seconds"]),
                    "chunk_padded": bool(window["pad_to_target"]),
                    **(metadata or {}),
                },
                **encode_kwargs,
            )
            items.append(item)

        return items

    def decode_latents(self, latents: Any, **decode_kwargs: Any) -> Any:
        return self.autoencoder.decode(latents, **decode_kwargs)
