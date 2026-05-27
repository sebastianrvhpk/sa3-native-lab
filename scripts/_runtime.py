from __future__ import annotations

import argparse


def add_torch_runtime_args(parser: argparse.ArgumentParser, *, half: bool = True) -> None:
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: cuda, mps, or cpu. Defaults to cuda -> mps -> cpu.",
    )
    if half:
        parser.add_argument(
            "--no-half",
            action="store_true",
            help="Disable fp16 model weights. Half precision is only used on CUDA.",
        )


def model_half_from_args(args: argparse.Namespace) -> bool:
    return not getattr(args, "no_half", False)


def resolve_torch_device(device: str | None = None):
    import torch

    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stable_audio_training_dtype(device):
    import torch

    return torch.bfloat16 if device.type == "cuda" else torch.float32
