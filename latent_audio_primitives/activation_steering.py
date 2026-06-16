"""Level 2: Activation steering vectors and hook injection for SA3."""

from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for activation steering.") from exc
    return torch


def extract_steering_vector(activations: list[Any], k: int = 1) -> Any:
    """Extract the top PCA steering vector from a list of collected activations."""
    torch = _require_torch()
    acts = []
    for act in activations:
        if not isinstance(act, torch.Tensor):
            act = torch.as_tensor(act)
        if act.ndim != 3:
            raise ValueError(f"expected activation shape B x C x T, got {act.shape}")
        B, C, T = act.shape
        acts.append(act.permute(0, 2, 1).reshape(B * T, C))
    all_acts = torch.cat(acts, dim=0).float()

    mean = all_acts.mean(dim=0, keepdim=True)
    centered = all_acts - mean
    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    steering_vector = Vh[0]  # First principal component
    return steering_vector


class SteeringHook:
    """Injects a steering vector into DiT block activations during forward pass."""

    def __init__(self, steering_vector: Any, scale: float = 1.0, layer_name: str | None = None):
        self.steering_vector = steering_vector
        self.scale = float(scale)
        self.layer_name = layer_name
        self.handle = None

    def register(self, module: Any) -> SteeringHook:
        self.handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def hook_fn(self, module: Any, inputs: Any, output: Any) -> Any:
        torch = _require_torch()
        if isinstance(output, tuple):
            tensor = output[0]
            other = output[1:]
        else:
            tensor = output
            other = None

        vec = self.steering_vector.to(device=tensor.device, dtype=tensor.dtype)
        if tensor.ndim == 3 and tensor.shape[1] == vec.shape[0]:
            steered = tensor + self.scale * vec.view(1, -1, 1)
        elif tensor.ndim == 3 and tensor.shape[2] == vec.shape[0]:
            steered = tensor + self.scale * vec.view(1, 1, -1)
        else:
            steered = tensor + self.scale * vec

        if other is not None:
            return (steered,) + other
        return steered
