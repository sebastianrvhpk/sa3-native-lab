import { describe, expect, it } from "vitest";

import { buildMemoryBrowserItems, memoryBrowserSummary } from "./memoryBrowserModel";
import { testArtifact, testFamily, testJob, testRecipe } from "./test/fixtures";

describe("memoryBrowserModel", () => {
  it("prioritizes remembered material that can be used right now", () => {
    const source = testArtifact({ artifact_id: "art_source", label: "Source Bell" });
    const keeper = testArtifact({
      artifact_id: "art_keeper",
      label: "Keeper Loop",
      session_id: null,
      source_artifact_ids: ["art_source"],
      notes: "works as a dry rhythmic anchor",
      metadata: { memory_role: "loop", reuse_intent: "source", listening_decision: "keeper" },
      created_at: "2026-05-28T15:00:00.000Z",
    });
    const bundle = testArtifact({
      artifact_id: "art_bundle",
      kind: "bundle",
      label: "Vector Bundle",
      session_id: null,
      metadata: { operator: "experiment.sa3_vectors.extract", bundle_kind: "vectors", promoted_from_bundle: true },
      created_at: "2026-05-28T16:00:00.000Z",
    });
    const job = testJob({ recipe: testRecipe({ recipe_id: "recipe_keeper", operator: "generate.text_to_audio" }) });
    const family = testFamily({ artifactIds: ["art_keeper"], recipe: job.recipe });

    const items = buildMemoryBrowserItems([bundle, keeper], {
      activeSessionId: "sess_1",
      artifacts: [source, bundle, keeper],
      jobs: [job],
      families: [family],
    });

    expect(items[0].artifact.artifact_id).toBe("art_keeper");
    expect(items[0].primaryAction?.intent).toBe("source");
    expect(items[0].roleLabel).toBe("loop");
    expect(items[0].reuseIntentLabel).toBe("source");
    expect(items[0].decisionLabel).toBe("keeper");
    expect(items[0].sourceLabel).toBe("Source Bell");
    expect(items[0].branchLabel).toBe("Make");
    expect(items[0].notesPreview).toMatch(/rhythmic anchor/);
    expect(items[1].lineageLabel).toBe("from bundle");
    expect(memoryBrowserSummary(items)).toMatchObject({ total: 2, usable: 2, keepers: 1 });
  });
});
