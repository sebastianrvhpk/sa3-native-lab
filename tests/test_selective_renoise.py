import torch

from latent_audio_primitives.selective_renoise import (
    LatentGraftResult,
    LatentMaskSpec,
    channel_mask_like,
    graft_latent_channels,
    masked_latent_noise,
    sampler_noise_from_donor_channels,
    sampler_noise_for_channels,
    select_latent_channels,
)


def test_select_latent_channels_random_is_reproducible():
    latents = torch.zeros(1, 16, 4)
    spec = LatentMaskSpec(name="random", mode="random_channels", fraction=0.25, seed=123)

    first = select_latent_channels(latents, spec)
    second = select_latent_channels(latents, spec)

    assert first == second
    assert len(first) == 4


def test_select_latent_channels_high_and_low_variance():
    latents = torch.zeros(1, 4, 5)
    latents[:, 1, :] = torch.tensor([0.0, 1.0, -1.0, 1.0, -1.0])
    latents[:, 3, :] = 7.0

    high = select_latent_channels(latents, LatentMaskSpec(name="high", mode="high_variance", fraction=0.25))
    low = select_latent_channels(latents, LatentMaskSpec(name="low", mode="low_variance", fraction=0.25))

    assert high == [1]
    assert low[0] in {0, 2, 3}


def test_channel_block_wraps_around_channel_count():
    latents = torch.zeros(1, 8, 2)
    spec = LatentMaskSpec(name="block", mode="channel_block", block_size=3, start_channel=6)

    assert select_latent_channels(latents, spec) == [0, 6, 7]


def test_channel_mask_like_has_broadcast_shape():
    latents = torch.zeros(2, 8, 5)

    mask = channel_mask_like(latents, [1, 3], level=0.5)

    assert tuple(mask.shape) == (1, 8, 1)
    assert mask[0, 1, 0] == 0.5
    assert mask[0, 2, 0] == 0.0


def test_masked_latent_noise_preserves_unselected_channels():
    latents = torch.ones(1, 4, 6)

    mixed = masked_latent_noise(latents, [1], sigma=0.5, seed=0)

    torch.testing.assert_close(mixed[:, 0, :], latents[:, 0, :])
    assert not torch.allclose(mixed[:, 1, :], latents[:, 1, :])


def test_sampler_noise_preserves_unselected_channels_before_sampler_mix():
    latents = torch.ones(1, 4, 6)

    noise = sampler_noise_for_channels(latents, [2], seed=0)

    torch.testing.assert_close(noise[:, 0, :], latents[:, 0, :])
    assert not torch.allclose(noise[:, 2, :], latents[:, 2, :])


def test_graft_latent_channels_replaces_only_selected_channels():
    source = torch.zeros(1, 4, 3)
    donor = torch.ones(1, 4, 3)

    grafted = graft_latent_channels(source, donor, [1, 3])

    torch.testing.assert_close(grafted[:, 0, :], source[:, 0, :])
    torch.testing.assert_close(grafted[:, 1, :], donor[:, 1, :])
    torch.testing.assert_close(grafted[:, 2, :], source[:, 2, :])
    torch.testing.assert_close(grafted[:, 3, :], donor[:, 3, :])


def test_graft_latent_channels_supports_partial_amount():
    source = torch.zeros(1, 2, 2)
    donor = torch.ones(1, 2, 2)

    grafted = graft_latent_channels(source, donor, [0], amount=0.25)

    torch.testing.assert_close(grafted[:, 0, :], torch.full((1, 2), 0.25))
    torch.testing.assert_close(grafted[:, 1, :], source[:, 1, :])


def test_sampler_noise_from_donor_channels_preserves_unselected_channels():
    source = torch.zeros(1, 4, 3)
    donor = torch.ones(1, 4, 3)

    sampler_noise = sampler_noise_from_donor_channels(source, donor, [2])

    torch.testing.assert_close(sampler_noise[:, 0, :], source[:, 0, :])
    torch.testing.assert_close(sampler_noise[:, 2, :], donor[:, 2, :])


def test_latent_graft_result_is_available_for_sampler_outputs():
    result = LatentGraftResult(
        sampled_latents="sampled",
        init_latents="source",
        donor_latents="donor",
        mixed_latents="mixed",
        selected_channels=[0],
        metadata={"intervention": "donor_channel_graft"},
    )

    assert result.donor_latents == "donor"
    assert result.metadata["intervention"] == "donor_channel_graft"
