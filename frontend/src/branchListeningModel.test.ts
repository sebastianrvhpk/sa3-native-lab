import { describe, expect, it } from "vitest";

import { branchListeningCursor } from "./branchListeningModel";
import { testArtifact } from "./test/fixtures";

describe("branchListeningModel", () => {
  it("orders playable branch takes as a trajectory with cursor controls", () => {
    const first = testArtifact({ artifact_id: "art_first", label: "First Take", created_at: "2026-05-28T10:00:00.000Z" });
    const second = testArtifact({ artifact_id: "art_second", label: "Second Take", created_at: "2026-05-28T10:02:00.000Z" });
    const latent = testArtifact({ artifact_id: "art_latent", kind: "latent", created_at: "2026-05-28T10:01:00.000Z" });

    const cursor = branchListeningCursor([second, latent, first], "art_second");

    expect(cursor.takes.map((artifact) => artifact.artifact_id)).toEqual(["art_first", "art_second"]);
    expect(cursor.previous?.artifact_id).toBe("art_first");
    expect(cursor.next).toBeNull();
    expect(cursor.positionLabel).toBe("2/2 · Second Take");

    const fallback = branchListeningCursor([second, first], null);
    expect(fallback.selected?.artifact_id).toBe("art_first");
    expect(fallback.next?.artifact_id).toBe("art_second");
  });
});
