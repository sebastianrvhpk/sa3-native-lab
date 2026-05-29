import { describe, expect, it } from "vitest";

import { playbackAnnotationPayload, playbackStateFromArtifact, playbackStateSignature } from "./playbackState";
import type { ArtifactRecord } from "./types";

describe("artifact playback state", () => {
  it("parses persisted markers and loop regions from artifact metadata", () => {
    const state = playbackStateFromArtifact(
      artifact({
        playback_state: {
          markers: [
            { id: "intro", label: "Intro", time: 1.23456 },
            { time: "3.5" },
            { time: -1 },
          ],
          loop_region: { start: 0.5, end: 2.1254 },
          updated_at: "2026-05-28T22:00:00.000Z",
          source: "audio_deck",
        },
      }),
    );

    expect(state.markers).toEqual([
      { id: "intro", label: "Intro", time: 1.235 },
      { id: "marker-2-3500", label: "M2", time: 3.5 },
    ]);
    expect(state.loopRegion).toEqual({ start: 0.5, end: 2.125 });
    expect(state.source).toBe("audio_deck");
  });

  it("builds an annotation payload for persisted listening cues", () => {
    expect(
      playbackAnnotationPayload({
        markers: [{ id: "marker-1", label: "M1", time: 2.25 }],
        loopRegion: { start: 1, end: 4 },
        source: "audio_deck",
        now: "2026-05-28T22:01:00.000Z",
      }),
    ).toEqual({
      metadata: {
        playback_state: {
          markers: [{ id: "marker-1", label: "M1", time: 2.25 }],
          loop_region: { start: 1, end: 4 },
          updated_at: "2026-05-28T22:01:00.000Z",
          source: "audio_deck",
        },
        playback_marker_count: 1,
        playback_loop_region: "1.000-4.000",
      },
    });
  });

  it("creates a stable signature for local dirty-state checks", () => {
    expect(playbackStateSignature([{ id: "m", label: "M", time: 2.12345 }], { start: 1, end: 2 })).toBe(
      '{"markers":[{"id":"m","label":"M","time":2.123}],"loopRegion":{"start":1,"end":2}}',
    );
  });
});

function artifact(metadata: Record<string, unknown>): ArtifactRecord {
  return {
    artifact_id: "art_audio",
    kind: "audio",
    path: "/tmp/audio.wav",
    file: { filename: "audio.wav", byte_size: 100 },
    audio: { sample_rate: 24000, channels: 2, frames: 24000, duration_seconds: 1 },
    latent: null,
    source_artifact_ids: [],
    recipe_id: "recipe_1",
    label: "audio",
    prompt: null,
    notes: null,
    tags: [],
    metadata,
    session_id: "sess",
    created_at: "2026-05-28T10:00:00.000Z",
  };
}
