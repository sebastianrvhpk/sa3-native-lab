from __future__ import annotations

import math
import wave

import numpy as np

from latent_audio_primitives.colab_audio_player import (
    audio_player_html,
    load_audio_annotations,
    save_audio_annotation,
    search_audio_annotations,
    _waveform_peaks,
)


def _write_test_wav(path, *, sample_rate=8_000, seconds=0.25):
    samples = int(sample_rate * seconds)
    t = np.arange(samples, dtype=np.float32) / sample_rate
    signal = 0.5 * np.sin(2 * math.pi * 440 * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def test_waveform_peaks_are_normalized(tmp_path):
    path = tmp_path / "test.wav"
    _write_test_wav(path)

    peaks = _waveform_peaks(path, peak_count=32)

    assert len(peaks) == 32
    assert max(abs(peak) for peak in peaks) <= 1.0
    assert max(abs(peak) for peak in peaks) > 0.9


def test_audio_player_html_contains_tracks_and_controls(tmp_path):
    path = tmp_path / "test.wav"
    _write_test_wav(path)

    html = audio_player_html([path], labels=["seed 001"], title="Sweep Player", peak_count=16)

    assert "Sweep Player" in html
    assert "seed 001" in html
    assert "data:audio/" in html
    assert "<canvas" in html
    assert "Loop region" in html
    assert "const data =" in html


def test_audio_player_html_can_include_annotation_controls(tmp_path):
    path = tmp_path / "test.wav"
    _write_test_wav(path)

    html = audio_player_html(
        [path],
        labels=["seed 001"],
        metadata=[{"kind": "sa3_sampled"}],
        title="Sweep Player",
        peak_count=16,
        annotation_callback="lap.save",
    )

    assert "Save note" in html
    assert "lap.save" in html
    assert "sa3_sampled" in html


def test_audio_player_html_rejects_label_mismatch(tmp_path):
    path = tmp_path / "test.wav"
    _write_test_wav(path)

    try:
        audio_player_html([path], labels=["a", "b"])
    except ValueError as exc:
        assert "labels length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_audio_player_html_rejects_metadata_mismatch(tmp_path):
    path = tmp_path / "test.wav"
    _write_test_wav(path)

    try:
        audio_player_html([path], metadata=[{}, {}])
    except ValueError as exc:
        assert "metadata length" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_save_and_search_audio_annotations(tmp_path):
    store_path = tmp_path / "annotations.json"
    audio_path = str(tmp_path / "seed.wav")

    result = save_audio_annotation(
        store_path,
        {
            "path": audio_path,
            "label": "SA3 random_10",
            "rating": 4.5,
            "tags": ["drums", "keeper", "drums"],
            "value": "loop",
            "description": "keeps groove but mutates texture",
            "metadata": {"kind": "sa3_sampled"},
        },
    )

    assert result["ok"] is True
    loaded = load_audio_annotations(store_path)
    assert loaded["items"][0]["tags"] == ["drums", "keeper"]

    matches = search_audio_annotations(store_path, query="groove", tags=["keeper"], min_rating=4.0)
    assert len(matches) == 1
    assert matches[0]["path"] == audio_path
