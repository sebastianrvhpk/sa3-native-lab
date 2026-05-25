from latent_audio_primitives.adapters.stable_audio3 import audio_chunk_windows


def test_audio_chunk_windows_non_overlapping_keep_tail():
    windows = audio_chunk_windows(
        num_frames=10_500,
        sample_rate=1_000,
        chunk_duration=4.0,
    )

    assert [window["frame_offset"] for window in windows] == [0, 4_000, 8_000]
    assert [window["num_frames"] for window in windows] == [4_000, 4_000, 2_500]
    assert windows[-1]["pad_to_target"] is True
    assert windows[-1]["chunk_index"] == 2
    assert windows[-1]["target_num_frames"] == 4_000


def test_audio_chunk_windows_overlap_and_limit():
    windows = audio_chunk_windows(
        num_frames=20_000,
        sample_rate=1_000,
        chunk_duration=8.0,
        hop_duration=2.0,
        max_chunks=3,
    )

    assert [window["frame_offset"] for window in windows] == [0, 2_000, 4_000]
    assert all(window["num_frames"] == 8_000 for window in windows)
    assert windows[1]["hop_seconds"] == 2.0


def test_audio_chunk_windows_drop_last():
    windows = audio_chunk_windows(
        num_frames=10_500,
        sample_rate=1_000,
        chunk_duration=4.0,
        drop_last=True,
    )

    assert [window["frame_offset"] for window in windows] == [0, 4_000]
    assert all(window["pad_to_target"] is False for window in windows)


def test_audio_chunk_windows_rejects_bad_values():
    try:
        audio_chunk_windows(num_frames=1, sample_rate=0, chunk_duration=1.0)
    except ValueError as exc:
        assert "sample_rate" in str(exc)
    else:
        raise AssertionError("expected ValueError")
