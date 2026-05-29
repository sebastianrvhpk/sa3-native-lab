import { describe, expect, it } from "vitest";

import { artifactRecoveryPayload, recoverableArchiveArtifacts } from "./sessionRecovery";
import type { ArtifactRecord } from "./types";

describe("session recovery", () => {
  it("builds an annotation payload that moves an artifact into the active session", () => {
    expect(
      artifactRecoveryPayload({
        artifact: artifact("art_old", "sess_old"),
        targetSessionId: "sess_active",
        source: "archive",
        now: "2026-05-28T20:00:00.000Z",
      }),
    ).toEqual({
      session_id: "sess_active",
      metadata: {
        recovered_from_session_id: "sess_old",
        recovered_into_session_id: "sess_active",
        recovered_artifact_at: "2026-05-28T20:00:00.000Z",
        recovery_source: "archive",
      },
    });
  });

  it("only offers archive recovery when there is an active target session", () => {
    const archived = artifact("art_old", "sess_old");
    const current = artifact("art_current", "sess_active");

    expect(recoverableArchiveArtifacts([archived, current], "sess_active").map((item) => item.artifact_id)).toEqual(["art_old"]);
    expect(recoverableArchiveArtifacts([archived], null)).toEqual([]);
  });
});

function artifact(artifactId: string, sessionId: string | null): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind: "audio",
    path: `/tmp/${artifactId}.wav`,
    file: { filename: `${artifactId}.wav`, byte_size: 100 },
    audio: { sample_rate: 24000, channels: 2, frames: 24000, duration_seconds: 1 },
    latent: null,
    source_artifact_ids: [],
    recipe_id: "recipe_1",
    label: artifactId,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: sessionId,
    created_at: "2026-05-28T10:00:00.000Z",
  };
}
