import { describe, expect, it } from "vitest";

import { buildProductSources, productSourceFieldOptions, productSourceFromArtifact } from "./sourceModel";
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
    expect(sources.find((source) => source.artifactId === "art_donor")?.actions.some((action) => action.intent === "donor" && action.available)).toBe(true);
  });

  it("marks source imports, promoted bundle audio, and reusable bundles without raw artifact wording", () => {
    const source = productSourceFromArtifact(testArtifact({ kind: "audio", recipe_id: null }), { activeSessionId: "sess_1" });
    const promoted = productSourceFromArtifact(testArtifact({ kind: "audio", metadata: { promoted_from_bundle: true } }), { activeSessionId: "sess_1" });
    const bundle = productSourceFromArtifact(testArtifact({ kind: "bundle", metadata: { reuse_intent: "advanced_gesture" } }), { activeSessionId: "sess_1" });

    expect(source.roles).toContain("imported");
    expect(source.kind).toBe("sound");
    expect(promoted.roleLabels).toContain("bundle audio");
    expect(bundle.roleLabels).toContain("bundle: advanced gesture");
    expect(bundle.kind).toBe("bundle");
  });

  it("filters SourceField options by artifact kind and strict bundle use paths", () => {
    const target = testArtifact({ artifact_id: "art_audio", kind: "audio", label: "Target Sound", path: "/tmp/target.wav" });
    const donor = testArtifact({ artifact_id: "art_latent", kind: "latent", label: "Donor Latent", path: "/tmp/donor.npy" });
    const vectors = testArtifact({
      artifact_id: "art_vectors",
      kind: "bundle",
      label: "Vector Bundle",
      path: "/tmp/vectors.zip",
      metadata: { operator: "experiment.sa3_vectors.extract", script_output_path: "/runs/vectors" },
    });
    const geometry = testArtifact({
      artifact_id: "art_geometry",
      kind: "bundle",
      label: "Geometry Bundle",
      path: "/tmp/geometry.zip",
      metadata: { operator: "experiment.geometry_audit", script_output_path: "/runs/geometry" },
    });
    const sources = buildProductSources([target, donor, vectors, geometry], { activeSessionId: "sess_1" });
    const pathForField = (artifact: typeof target, fieldKey: string) => `${artifact.metadata.script_output_path ?? artifact.path}/${fieldKey}`;

    expect(productSourceFieldOptions({
      sources,
      fieldKey: "target_audio_path",
      artifactKinds: ["audio"],
      getArtifactPath: pathForField,
    }).map((option) => option.label)).toEqual(["Target Sound"]);

    expect(productSourceFieldOptions({
      sources,
      fieldKey: "donor",
      artifactKinds: ["latent"],
      valueMode: "artifact-id",
      getArtifactPath: pathForField,
    })).toEqual([
      expect.objectContaining({ label: "Donor Latent", value: "art_latent" }),
    ]);

    expect(productSourceFieldOptions({
      sources,
      fieldKey: "vectors_path",
      artifactKinds: ["bundle"],
      getArtifactPath: pathForField,
    }).map((option) => option.label)).toEqual(["Vector Bundle"]);

    expect(productSourceFieldOptions({
      sources,
      fieldKey: "profile_path",
      artifactKinds: ["bundle"],
      getArtifactPath: pathForField,
    })).toEqual([]);
  });
});
