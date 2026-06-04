from __future__ import annotations

import numpy as np

from latent_audio_primitives.control_lanes import (
    ControlLane,
    audio_envelope_lane,
    control_lane_distance,
    control_lane_similarity,
    control_lane_svg,
    latent_channel_energy_lane,
    latent_motion_lane,
    load_control_lanes,
    normalize_control_lane,
    resample_control_lane,
    save_control_lanes,
)


def test_latent_motion_lane_tracks_frame_changes():
    latent = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )

    lane = latent_motion_lane(latent, latent_rate=10.0)

    assert lane.name == "latent_motion_energy"
    assert lane.frames == 3
    assert lane.values[0] == 0.0
    assert lane.values[1] > 0.0
    assert lane.values[2] > lane.values[1]


def test_audio_envelope_lane_and_resample_roundtrip(tmp_path):
    sample_rate = 1000
    audio = np.concatenate([np.zeros(500), np.ones(500)]).astype(np.float32)

    lane = audio_envelope_lane(audio, sample_rate, frame_seconds=0.1, name="rms_envelope")
    normalized = normalize_control_lane(lane, mode="minmax")
    resampled = resample_control_lane(normalized, 5, target_rate_hz=10.0)
    path = save_control_lanes([resampled], tmp_path / "lanes.json")
    loaded = load_control_lanes(path)

    assert resampled.frames == 5
    assert loaded[0].name == "rms_envelope"
    assert loaded[0].values[-1] > loaded[0].values[0]


def test_lane_similarity_distance_and_svg():
    a = ControlLane("a", np.array([0.0, 1.0, 2.0], dtype=np.float32), rate_hz=1.0)
    b = ControlLane("b", np.array([0.0, 2.0, 4.0], dtype=np.float32), rate_hz=1.0)
    c = ControlLane("c", np.array([2.0, 1.0, 0.0], dtype=np.float32), rate_hz=1.0)

    assert control_lane_similarity(a, b) > 0.99
    assert control_lane_distance(a, b) < control_lane_distance(a, c)
    assert "<svg" in control_lane_svg([a, b])


def test_latent_channel_energy_lane_can_select_channels():
    latent = np.array([[1.0, 10.0], [2.0, 10.0]], dtype=np.float32)

    all_lane = latent_channel_energy_lane(latent, latent_rate=2.0)
    channel_lane = latent_channel_energy_lane(latent, latent_rate=2.0, channels=[0])

    assert all_lane.values[0] > channel_lane.values[0]
    assert channel_lane.values.tolist() == [1.0, 2.0]
