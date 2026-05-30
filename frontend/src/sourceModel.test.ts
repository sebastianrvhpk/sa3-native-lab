import { describe, expect, it } from "vitest";

import { buildProductSources, productSourceFromArtifact } from "./sourceModel";
import { testArtifact } from "./test/fixtures";

describe("sourceModel", () => {
  it("describes current, anchor, source, donor, and memory roles over artifacts", () => {
    const current = testArtifact({ artifact_id: "art_current", label: "Current Loop", kind: "audio" });
    const anchor = testArtifact({ artifact_id: "art_anchor", label: "Anchor Loop", kind: "audio", session_id: null, metadata: { memory_role: "reference" } });
    const donor = testArtifact({ artifact_id: "art_donor", label: "Donor Latent", kind: "latent", session_id: null, metadata: { reuse_intent: "donor" } });

    const sources = buildProductSources([current, anchor, donor], {
      activeSessionId: "sess_1",
      currentArtifactId: "art_current",
      anchorArtifactId: "art_anchor",
      donorArtifactId: "art_donor",
    });

    expect(sources.find((source) => source.artifactId === "art_current")).toMatchObject({
      kind: "sound",
      roles: ["current", "take"],
      isCurrent: true,
      isRemembered: false,
    });
    expect(sources.find((source) => source.artifactId === "art_anchor")?.roleLabels).toEqual([
      "anchor",
      "memory: reference",
      "take",
    ]);
    expect(sources.find((source) => source.artifactId === "art_donor")).toMatchObject({
      kind: "latent",
      roles: ["donor", "remembered", "take"],
      reuseIntent: "donor",
    });
  });

  it("marks source imports and reusable bundles without raw artifact wording", () => {
    const source = productSourceFromArtifact(testArtifact({ kind: "audio", recipe_id: null }), { activeSessionId: "sess_1" });
    const bundle = productSourceFromArtifact(testArtifact({ kind: "bundle", metadata: { reuse_intent: "advanced_gesture" } }), { activeSessionId: "sess_1" });

    expect(source.roles).toContain("imported");
    expect(source.kind).toBe("sound");
    expect(bundle.roleLabels).toContain("bundle: advanced gesture");
    expect(bundle.kind).toBe("bundle");
  });
});
