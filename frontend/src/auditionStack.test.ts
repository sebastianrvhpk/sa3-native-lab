import { describe, expect, it } from "vitest";

import { auditionStackRows } from "./auditionStack";

describe("audition stack", () => {
  it("summarizes newest session audio artifacts for listening", () => {
    const rows = auditionStackRows([
      {
        artifact_id: "art_old",
        kind: "audio",
        path: "/tmp/old.wav",
        prompt: "old",
        recipe_id: "recipe_1",
        metadata: {},
        tags: [],
        source_artifact_ids: [],
        created_at: "2026-05-28T10:00:00.000Z",
      },
      {
        artifact_id: "art_prompt",
        kind: "audio",
        path: "/tmp/prompt.wav",
        prompt: "warm glass loop",
        recipe_id: "recipe_2",
        metadata: { generation_origin: "prompt_search_candidate" },
        tags: [],
        source_artifact_ids: [],
        created_at: "2026-05-28T11:00:00.000Z",
      },
      {
        artifact_id: "art_latent",
        kind: "latent",
        path: "/tmp/latent.npy",
        metadata: {},
        tags: [],
        source_artifact_ids: [],
        created_at: "2026-05-28T12:00:00.000Z",
      },
    ] as never);

    expect(rows).toEqual([
      expect.objectContaining({
        artifactId: "art_prompt",
        prompt: "warm glass loop",
        origin: "prompt take",
      }),
      expect.objectContaining({
        artifactId: "art_old",
        origin: "recipe output",
      }),
    ]);
  });
});
