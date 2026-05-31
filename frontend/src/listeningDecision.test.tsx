import { describe, expect, it } from "vitest";

import { listeningDecision, listeningDecisionPayload, listeningDecisionSummary } from "./listeningDecision";
import type { ArtifactRecord } from "./types";

describe("listening decisions", () => {
  it("derives and persists keeper metadata as artifact annotation payload", () => {
    const artifact = audioArtifact({
      tags: ["bright", "maybe"],
      metadata: { listening_decision: "maybe" },
      notes: "old note",
    });

    expect(listeningDecision(artifact)).toBe("maybe");
    expect(
      listeningDecisionPayload({
        artifact,
        decision: "keeper",
        note: "wide useful texture",
        source: "prompt_candidate_bench",
        now: "2026-05-28T15:00:00.000Z",
      }),
    ).toEqual({
      notes: "wide useful texture",
      tags: ["bright", "keeper"],
      metadata: {
        listening_decision: "keeper",
        listening_decision_source: "prompt_candidate_bench",
        listening_decision_at: "2026-05-28T15:00:00.000Z",
        listening_decision_note: "wide useful texture",
      },
    });
  });

  it("does not overwrite notes when no listening note is provided", () => {
    expect(
      listeningDecisionPayload({
        artifact: audioArtifact({ tags: ["keeper"] }),
        decision: "rejected",
        source: "family_detail",
        now: "2026-05-28T15:00:00.000Z",
      }),
    ).toMatchObject({
      tags: ["rejected"],
      metadata: {
        listening_decision: "rejected",
        listening_decision_source: "family_detail",
      },
    });
  });

  it("summarizes playable queue decisions including open takes", () => {
    const summary = listeningDecisionSummary([
      audioArtifact({ artifact_id: "art_keep", metadata: { listening_decision: "keeper" } }),
      audioArtifact({ artifact_id: "art_maybe", metadata: { listening_decision: "maybe" } }),
      audioArtifact({ artifact_id: "art_open" }),
      audioArtifact({ artifact_id: "art_latent", kind: "latent", audio: null }),
    ]);

    expect(summary).toMatchObject({
      total: 3,
      decided: 2,
      undecided: 1,
      keeper: 1,
      maybe: 1,
      rejected: 0,
    });
    expect(summary.entries).toEqual([
      { key: "keeper", label: "keeper", count: 1 },
      { key: "maybe", label: "maybe", count: 1 },
      { key: "open", label: "open", count: 1 },
    ]);
  });
});

function audioArtifact(overrides: Partial<ArtifactRecord> = {}): ArtifactRecord {
  return {
    artifact_id: "art_take",
    kind: "audio",
    path: "/tmp/take.wav",
    file: { filename: "take.wav", media_type: "audio/wav", byte_size: 128 },
    audio: { sample_rate: 24000, channels: 1, frames: 24000, duration_seconds: 1, format: "WAV" },
    latent: null,
    source_artifact_ids: [],
    recipe_id: "recipe_take",
    label: "take",
    prompt: "bright texture",
    notes: null,
    tags: [],
    metadata: {},
    session_id: "sess_1",
    created_at: "2026-05-28T15:00:00.000Z",
    ...overrides,
  };
}
