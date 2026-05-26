import numpy as np

from latent_audio_primitives.audio_descriptors import audio_descriptor_report, descriptor_delta


def test_audio_descriptor_report_detects_sine_centroid_region():
    sample_rate = 8000
    seconds = 1.0
    freq = 440.0
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * freq * t).astype(np.float32)

    report = audio_descriptor_report(audio, sample_rate)

    assert abs(report["spectral_centroid_hz"] - freq) < 80.0
    assert report["rms"] > 0.1
    assert report["duration_seconds"] == seconds


def test_audio_descriptor_report_stereo_width_changes_with_side_signal():
    sample_rate = 8000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    left = np.sin(2.0 * np.pi * 220.0 * t)
    mono = np.stack([left, left])
    wide = np.stack([left, -left])

    mono_report = audio_descriptor_report(mono, sample_rate)
    wide_report = audio_descriptor_report(wide, sample_rate)

    assert mono_report["stereo_width"] < wide_report["stereo_width"]
    assert mono_report["stereo_correlation"] > wide_report["stereo_correlation"]


def test_descriptor_delta_subtracts_shared_keys():
    before = {"rms_dbfs": -12.0, "spectral_flux": 0.1}
    after = {"rms_dbfs": -9.0, "spectral_flux": 0.4}

    delta = descriptor_delta(before, after)

    assert delta["rms_dbfs"] == 3.0
    assert delta["spectral_flux"] == 0.30000000000000004
