"""Level 2: Cross-attention biasing and prompt token window warping."""

from __future__ import annotations

from typing import Any


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for cross-attention biasing.") from exc
    return torch


def apply_attention_bias(attn: Any, token_windows: dict[int, tuple[float, float]]) -> Any:
    """Apply time-varying bias masks to attention scores of shape B x heads x T x L.

    For frames outside a token's active window, a large negative bias is added,
    preventing the model from attending to that conditioning token at that time.
    """
    torch = _require_torch()
    if not isinstance(attn, torch.Tensor):
        attn = torch.as_tensor(attn)

    B, heads, T, L = attn.shape
    bias = torch.zeros_like(attn)

    for token_idx, (start_frac, end_frac) in token_windows.items():
        if 0 <= token_idx < L:
            start_frame = int(round(start_frac * T))
            end_frame = int(round(end_frac * T))

            if start_frame > 0:
                bias[:, :, :start_frame, token_idx] = -10000.0
            if end_frame < T:
                bias[:, :, end_frame:, token_idx] = -10000.0

    return attn + bias


class CrossAttentionBiasingHook:
    """Monkey-patch hook to intercept and bias custom attention module forward passes."""

    def __init__(self, token_windows: dict[int, tuple[float, float]]):
        self.token_windows = token_windows
        self.original_forward = None
        self.module = None

    def register(self, module: Any) -> CrossAttentionBiasingHook:
        self.module = module
        self.original_forward = module.forward
        module.forward = self.patched_forward
        return self

    def remove(self) -> None:
        if self.module is not None and self.original_forward is not None:
            self.module.forward = self.original_forward
            self.module = None
            self.original_forward = None

    def patched_forward(self, *args, **kwargs) -> Any:
        # In custom attention modules, they might call torch.nn.functional.scaled_dot_product_attention.
        # To intercept it, we can temporarily monkey-patch scaled_dot_product_attention.
        torch = _require_torch()
        orig_sdpa = torch.nn.functional.scaled_dot_product_attention

        def custom_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            # query shape: B x heads x T x head_dim
            # key shape: B x heads x L x head_dim
            # If we compute attention manually:
            B, heads, T, head_dim = query.shape
            L = key.shape[2]
            scale_val = scale if scale is not None else (1.0 / (head_dim ** 0.5))

            # Manual attention computation with biasing
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale_val
            if attn_mask is not None:
                scores = scores + attn_mask
            biased_scores = apply_attention_bias(scores, self.token_windows)
            attn_probs = torch.softmax(biased_scores.float(), dim=-1).to(query.dtype)
            if dropout_p > 0.0:
                attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p)
            return torch.matmul(attn_probs, value)

        # Temporarily override torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = custom_sdpa
        try:
            out = self.original_forward(*args, **kwargs)
        finally:
            torch.nn.functional.scaled_dot_product_attention = orig_sdpa
        return out
