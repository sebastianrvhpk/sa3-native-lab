import torch

from latent_audio_primitives.latent_blur import (
    LatentBlurSpec,
    apply_latent_blur,
    channel_blur_latents,
    detail_attenuate_latents,
    low_rank_latents,
    temporal_box_blur_latents,
    temporal_blur_latents,
)


def test_temporal_blur_preserves_shape_and_smooths_impulse():
    latents = torch.zeros(1, 2, 9)
    latents[:, :, 4] = 1.0

    blurred = temporal_blur_latents(latents, radius=2)

    assert tuple(blurred.shape) == tuple(latents.shape)
    assert blurred[0, 0, 4] < 1.0
    assert blurred[0, 0, 3] > 0.0


def test_temporal_box_blur_mixes_contiguous_frames():
    latents = torch.arange(1, 6, dtype=torch.float32).view(1, 1, 5)

    blurred = temporal_box_blur_latents(latents, radius=1)

    assert tuple(blurred.shape) == tuple(latents.shape)
    torch.testing.assert_close(blurred[0, 0, 2], torch.tensor(3.0))


def test_temporal_box_blur_can_be_trailing_or_leading():
    latents = torch.arange(1, 6, dtype=torch.float32).view(1, 1, 5)

    trailing = temporal_box_blur_latents(latents, radius=2, direction="past")
    leading = temporal_box_blur_latents(latents, radius=2, direction="future")

    torch.testing.assert_close(trailing[0, 0, 2], torch.tensor(2.0))
    torch.testing.assert_close(leading[0, 0, 2], torch.tensor(4.0))


def test_channel_blur_preserves_shape_and_smooths_channel_impulse():
    latents = torch.zeros(1, 7, 3)
    latents[:, 3, :] = 1.0

    blurred = channel_blur_latents(latents, radius=1)

    assert tuple(blurred.shape) == tuple(latents.shape)
    assert blurred[0, 3, 0] < 1.0
    assert blurred[0, 2, 0] > 0.0


def test_low_rank_latents_reduces_matrix_rank():
    torch.manual_seed(0)
    latents = torch.randn(1, 6, 10)

    blurred = low_rank_latents(latents, rank=2)
    rank = torch.linalg.matrix_rank(blurred[0].transpose(0, 1) - blurred[0].transpose(0, 1).mean(dim=0))

    assert tuple(blurred.shape) == tuple(latents.shape)
    assert int(rank) <= 2


def test_detail_attenuate_matches_blur_when_detail_gain_zero():
    latents = torch.randn(1, 3, 12)

    attenuated = detail_attenuate_latents(latents, radius=2, detail_gain=0.0)
    blurred = temporal_blur_latents(latents, radius=2)

    torch.testing.assert_close(attenuated, blurred)


def test_apply_latent_blur_strength_zero_returns_original():
    latents = torch.randn(1, 3, 12)
    spec = LatentBlurSpec(name="none", mode="temporal", temporal_radius=2, temporal_kernel="box", strength=0.0)

    out = apply_latent_blur(latents, spec)

    torch.testing.assert_close(out, latents)


def test_apply_latent_blur_mean_blend():
    latents = torch.randn(1, 3, 12)
    spec = LatentBlurSpec(name="static", mode="mean_blend", strength=1.0)

    out = apply_latent_blur(latents, spec)

    torch.testing.assert_close(out, latents.mean(dim=-1, keepdim=True).expand_as(latents))
