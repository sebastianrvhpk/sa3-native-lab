"""Level 1: Manifold mapping, normalizing flows, and projection networks."""

from __future__ import annotations

from typing import Any
import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """RealNVP affine coupling layer operating on latent channels."""

    def __init__(self, d: int, D: int, hidden_dim: int = 256):
        super().__init__()
        self.d = d
        self.D = D
        self.s_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, D - d),
            nn.Tanh(),
        )
        self.t_net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, D - d),
        )

    def forward(self, x: torch.Tensor, invert: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, T = x.shape
        x_perm = x.permute(0, 2, 1).reshape(B * T, C)
        x1 = x_perm[:, :self.d]
        x2 = x_perm[:, self.d:]

        s = self.s_net(x1)
        t = self.t_net(x1)

        if not invert:
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det = s.sum(dim=-1)
        else:
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            log_det = -s.sum(dim=-1)

        y_perm = torch.cat([y1, y2], dim=-1)
        y = y_perm.view(B, T, C).permute(0, 2, 1)
        return y, log_det.view(B, T).sum(dim=-1)


class LatentNormalizingFlow(nn.Module):
    """Bijective RealNVP flow mapping SAME latents to/from Gaussian spaces."""

    def __init__(self, dim: int = 256, num_layers: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            CouplingLayer(dim // 2, dim, hidden_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, invert: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        curr = x
        indices = reversed(range(self.num_layers)) if invert else range(self.num_layers)
        for idx in indices:
            layer = self.layers[idx]
            flip = (idx % 2 == 1)
            if flip:
                curr = torch.flip(curr, dims=[1])
            curr, log_det = layer(curr, invert=invert)
            log_det_total = log_det_total + log_det
            if flip:
                curr = torch.flip(curr, dims=[1])
        return curr, log_det_total


class LatentManifoldProjector(nn.Module):
    """Residual convolutional projection autoencoder to sanitize edited latents."""

    def __init__(self, channels: int = 256, bottleneck_channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, bottleneck_channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, bottleneck_channels),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(bottleneck_channels, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv1d(128, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.decoder(self.encoder(x.float())).to(dtype=x.dtype)


def project_onto_manifold(latents: Any, projector: LatentManifoldProjector) -> Any:
    """Project edited or noisy latents back onto the data manifold."""
    x = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
    was_2d = x.ndim == 2
    if was_2d:
        x = x.unsqueeze(0)
    out = projector(x)
    if was_2d:
        out = out.squeeze(0)
    return out
