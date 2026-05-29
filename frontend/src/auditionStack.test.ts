import { describe, expect, it } from "vitest";

import { auditionCursor, auditionKeyboardTarget, auditionPlaylist, auditionPositionLabel, auditionStackRows } from "./auditionStack";

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

  it("builds a newest-first playlist cursor for session audition", () => {
    const artifacts = [
      audio("art_old", "2026-05-28T10:00:00.000Z"),
      audio("art_mid", "2026-05-28T11:00:00.000Z"),
      audio("art_new", "2026-05-28T12:00:00.000Z"),
    ];

    const cursor = auditionCursor(artifacts, "art_mid");

    expect(cursor.playlist.map((artifact) => artifact.artifact_id)).toEqual(["art_new", "art_mid", "art_old"]);
    expect(cursor.previous?.artifact_id).toBe("art_new");
    expect(cursor.next?.artifact_id).toBe("art_old");
    expect(auditionPositionLabel(artifacts, "art_mid")).toBe("2/3");
  });

  it("maps keyboard navigation to real audition targets", () => {
    const artifacts = [
      audio("art_old", "2026-05-28T10:00:00.000Z"),
      audio("art_mid", "2026-05-28T11:00:00.000Z"),
      audio("art_new", "2026-05-28T12:00:00.000Z"),
    ];

    expect(auditionKeyboardTarget(artifacts, "art_mid", "ArrowUp")?.artifact_id).toBe("art_new");
    expect(auditionKeyboardTarget(artifacts, "art_mid", "ArrowDown")?.artifact_id).toBe("art_old");
    expect(auditionKeyboardTarget(artifacts, "art_mid", "Home")?.artifact_id).toBe("art_new");
    expect(auditionKeyboardTarget(artifacts, "art_mid", "End")?.artifact_id).toBe("art_old");
    expect(auditionKeyboardTarget(artifacts, null, "Enter")?.artifact_id).toBe("art_new");
  });

  it("can sequence session audition by open decisions and keepers", () => {
    const artifacts = [
      audio("art_rejected", "2026-05-28T12:00:00.000Z", { listening_decision: "rejected" }),
      audio("art_open", "2026-05-28T11:00:00.000Z"),
      audio("art_keeper", "2026-05-28T10:00:00.000Z", { listening_decision: "keeper" }),
    ];

    expect(auditionPlaylist(artifacts, 12, "open").map((artifact) => artifact.artifact_id)).toEqual([
      "art_open",
      "art_rejected",
      "art_keeper",
    ]);
    expect(auditionPlaylist(artifacts, 12, "keepers").map((artifact) => artifact.artifact_id)).toEqual([
      "art_keeper",
      "art_open",
      "art_rejected",
    ]);
    expect(auditionStackRows(artifacts, 12, "keepers")[0].sequence).toBe("keeper");
  });

  it("can sequence around the selected artifact lineage neighborhood", () => {
    const source = audio("art_source", "2026-05-28T09:00:00.000Z");
    const selected = audio("art_selected", "2026-05-28T10:00:00.000Z", {}, ["art_source"]);
    const child = audio("art_child", "2026-05-28T11:00:00.000Z", {}, ["art_selected"]);
    const unrelated = audio("art_unrelated", "2026-05-28T12:00:00.000Z");
    const artifacts = [unrelated, child, selected, source];

    expect(auditionPlaylist(artifacts, 12, "lineage", "art_selected").map((artifact) => artifact.artifact_id)).toEqual([
      "art_selected",
      "art_source",
      "art_child",
      "art_unrelated",
    ]);
    expect(auditionPositionLabel(artifacts, "art_selected", 12, "lineage")).toBe("1/4");
  });
});

function audio(artifactId: string, createdAt: string, metadata: Record<string, unknown> = {}, sourceArtifactIds: string[] = []) {
  return {
    artifact_id: artifactId,
    kind: "audio",
    path: `/tmp/${artifactId}.wav`,
    metadata,
    tags: [],
    source_artifact_ids: sourceArtifactIds,
    created_at: createdAt,
  } as never;
}
