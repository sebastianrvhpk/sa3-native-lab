"""Level 3: Latent-space Feedback Delay Networks (FDN) for high-fidelity reverb."""

from __future__ import annotations

import math
from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for latent reverb FDN.") from exc
    return torch


def latent_feedback_delay_network(
    latents: Any,
    *,
    delays: list[int] | None = None,
    feedback_gain: float = 0.5,
    mix_amount: float = 0.5,
) -> Any:
    """Pass latent channels through a unitary Feedback Delay Network (FDN).

    Splits the 256 SAME channels into 8 groups of 32, applies coprime delay
    lengths to maximize echo density, and mixes them using an orthogonal
    Hadamard matrix at each time step.
    """
    torch = _require_torch()
    x = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    was_2d = x.ndim == 2
    if was_2d:
        x = x.unsqueeze(0)

    B, C, T = x.shape
    if C % 8 != 0:
        raise ValueError(f"Channels {C} must be divisible by 8 for Group FDN")

    if delays is None:
        delays = [2, 3, 5, 7, 11, 13, 17, 19]
    if len(delays) != 8:
        raise ValueError("delays list must have exactly 8 elements")

    G = 8
    group_dim = C // G

    H = torch.tensor([
        [1,  1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1, -1,  1, -1],
        [1,  1, -1, -1,  1,  1, -1, -1],
        [1, -1, -1,  1,  1, -1, -1,  1],
        [1,  1,  1,  1, -1, -1, -1, -1],
        [1, -1,  1, -1, -1,  1, -1,  1],
        [1,  1, -1, -1, -1, -1,  1,  1],
        [1, -1, -1,  1, -1,  1,  1, -1],
    ], device=x.device, dtype=x.dtype) / math.sqrt(8.0)

    max_delay = max(delays)
    buffers = torch.zeros((G, B, group_dim, T + max_delay), device=x.device, dtype=x.dtype)
    grouped_x = x.view(B, G, group_dim, T).permute(1, 0, 2, 3)

    for t in range(T):
        delayed_states = []
        for g in range(G):
            delay = delays[g]
            delayed_states.append(buffers[g, :, :, t + max_delay - delay])

        delayed_stacked = torch.stack(delayed_states, dim=0)
        mixed_delayed = torch.matmul(H, delayed_stacked.view(G, B * group_dim)).view(G, B, group_dim)

        for g in range(G):
            buffers[g, :, :, t + max_delay] = grouped_x[g, :, :, t] + float(feedback_gain) * mixed_delayed[g]

    output_groups = []
    for g in range(G):
        output_groups.append(buffers[g, :, :, max_delay : T + max_delay])

    output_tensor = torch.stack(output_groups, dim=0)
    output_tensor = output_tensor.permute(1, 0, 2, 3).contiguous().view(B, C, T)

    res = x + float(mix_amount) * (output_tensor - x)
    if was_2d:
        res = res.squeeze(0)
    return res
