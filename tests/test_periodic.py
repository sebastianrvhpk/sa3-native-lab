import numpy as np

from latent_audio_primitives.periodic import (
    best_period_lag,
    latent_fft_energy,
    latent_spectral_centroid,
    loop_boundary_loss,
    periodicity_report,
)


def test_best_period_lag_detects_repeated_pattern():
    pattern = np.array([[0.0], [1.0], [0.0], [-1.0]], dtype=np.float32)
    z = np.tile(pattern, (4, 1))

    lag, score = best_period_lag(z, min_lag=2, max_lag=8)

    assert lag == 4
    assert score > 0.5


def test_loop_boundary_loss_zero_for_matching_boundaries():
    z = np.ones((8, 2), dtype=np.float32)

    total, state, velocity = loop_boundary_loss(z, window=2)

    assert total == 0.0
    assert state == 0.0
    assert velocity == 0.0


def test_latent_fft_energy_and_centroid_are_finite():
    z = np.sin(np.linspace(0, np.pi * 4, 16, dtype=np.float32))[:, None]

    energy = latent_fft_energy(z)
    centroid = latent_spectral_centroid(z)

    assert energy.ndim == 1
    assert np.isfinite(energy).all()
    assert 0.0 <= centroid <= 1.0


def test_periodicity_report_contains_expected_fields():
    z = np.tile(np.array([[0.0], [1.0], [0.0], [-1.0]], dtype=np.float32), (3, 1))

    report = periodicity_report(z, min_lag=2, max_lag=6)

    assert report.best_lag == 4
    assert report.best_score > 0.0
