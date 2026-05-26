import torch

from latent_audio_primitives.latent_dsp import (
    LatentDSPSpec,
    apply_latent_dsp,
    latent_change_report,
    latent_dynamics,
    latent_fft_eq,
    latent_fft_magnitude_phase_graft,
    latent_fft_phase_blend,
    latent_fft_phase_randomize,
    latent_fft_phase_shift,
    latent_gain,
    latent_soft_clip,
    pca_component_gain,
)


def test_latent_gain_scales_around_zero():
    latents = torch.tensor([[[1.0, -2.0, 3.0]]])

    out = latent_gain(latents, gain=2.0, center="zero")

    torch.testing.assert_close(out, latents * 2.0)


def test_latent_dynamics_compressor_reduces_large_excursion():
    latents = torch.tensor([[[0.0, 0.0, 10.0, 0.0, 0.0]]])

    compressed = latent_dynamics(latents, threshold=0.5, ratio=10.0, mode="compress")

    assert compressed[0, 0, 2].abs() < latents[0, 0, 2].abs()


def test_latent_dynamics_expander_increases_large_excursion():
    latents = torch.tensor([[[0.0, 0.0, 2.0, 0.0, 0.0]]])

    expanded = latent_dynamics(latents, threshold=0.5, ratio=2.0, mode="expand")

    assert expanded[0, 0, 2].abs() > latents[0, 0, 2].abs()


def test_latent_soft_clip_limits_extreme_value():
    latents = torch.tensor([[[0.0, 0.0, 20.0, 0.0, 0.0]]])

    clipped = latent_soft_clip(latents, drive=2.0, ceiling=1.0)

    assert clipped[0, 0, 2].abs() < latents[0, 0, 2].abs()


def test_latent_fft_eq_attenuates_fast_alternation():
    latents = torch.tensor([[[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]]])

    filtered = latent_fft_eq(latents, low_cutoff=0.1, high_cutoff=0.4, high_gain=0.0)

    assert filtered.std() < latents.std()


def test_latent_fft_phase_shift_matches_integer_roll():
    latents = torch.arange(8, dtype=torch.float32).view(1, 1, 8)

    shifted = latent_fft_phase_shift(latents, shift_fraction=0.25)

    torch.testing.assert_close(shifted, torch.roll(latents, shifts=2, dims=-1), atol=1e-5, rtol=1e-5)


def test_latent_fft_phase_randomize_amount_zero_is_identity():
    latents = torch.randn(1, 2, 16)

    out = latent_fft_phase_randomize(latents, amount=0.0, seed=123)

    torch.testing.assert_close(out, latents)


def test_latent_fft_phase_blend_amount_zero_is_identity():
    latents = torch.randn(1, 2, 16)
    donor = torch.randn(1, 2, 16)

    out = latent_fft_phase_blend(latents, donor, amount=0.0)

    torch.testing.assert_close(out, latents)


def test_latent_fft_magnitude_phase_graft_amount_zero_is_identity():
    latents = torch.randn(1, 2, 16)
    donor = torch.randn(1, 2, 16)

    out = latent_fft_magnitude_phase_graft(
        magnitude_latents=donor,
        phase_latents=latents,
        magnitude_amount=0.0,
    )

    torch.testing.assert_close(out, latents, atol=1e-5, rtol=1e-5)


def test_pca_component_gain_all_ones_is_identity():
    latents = torch.randn(1, 4, 8)

    out = pca_component_gain(latents, component_gains=[1.0, 1.0, 1.0, 1.0])

    torch.testing.assert_close(out, latents, atol=1e-5, rtol=1e-5)


def test_apply_latent_dsp_requires_donor_for_graft():
    latents = torch.randn(1, 2, 8)
    spec = LatentDSPSpec(name="graft", mode="fft_mag_phase_graft")

    try:
        apply_latent_dsp(latents, spec)
    except ValueError as exc:
        assert "donor" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_apply_latent_dsp_donor_phase_preserves_source_shape():
    latents = torch.randn(1, 2, 8)
    donor = torch.randn(1, 2, 12)
    spec = LatentDSPSpec(name="donor_phase", mode="fft_phase_from_donor")

    out = apply_latent_dsp(latents, spec, donor_latents=donor)

    assert tuple(out.shape) == tuple(latents.shape)


def test_latent_change_report_contains_delta():
    before = torch.zeros(1, 2, 8)
    after = torch.ones(1, 2, 8)

    report = latent_change_report(before, after)

    assert report["delta_rms"] > 0.0
    assert report["mean_abs_delta"] == 1.0
