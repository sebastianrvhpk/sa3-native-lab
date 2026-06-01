import { describe, expect, it } from "vitest";

import { nextActionsForArtifact, nextActionsForBranch, nextActionsForPendingTake } from "./nextActionModel";
import { pendingTakeFromJob } from "./pendingTakeModel";
import { testArtifact, testFamily, testJob, testRecipe } from "./test/fixtures";

describe("nextActionModel", () => {
  it("maps audio, latent, and bundle artifacts to product next actions", () => {
    expect(nextActionsForArtifact(testArtifact({ kind: "audio" })).map((action) => action.id)).toEqual([
      "continue",
      "vary",
      "encode",
      "remember",
    ]);

    const latentActions = nextActionsForArtifact(testArtifact({ kind: "latent", artifact_id: "art_latent" }), {
      donorLatents: [testArtifact({ kind: "latent", artifact_id: "art_donor" })],
    });
    expect(latentActions.map((action) => action.id)).toEqual(["decode", "morph", "borrow_texture", "find_similar", "remember"]);
    expect(latentActions.find((action) => action.id === "borrow_texture")).toMatchObject({ available: true });
    expect(latentActions.find((action) => action.id === "find_similar")).toMatchObject({
      experimentMode: "memory.query",
      gestureId: "steer",
    });

    const bundleActions = nextActionsForArtifact(testArtifact({ kind: "bundle", metadata: { operator: "experiment.audio_style_vectors" } }));
    expect(bundleActions).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "inspect", label: "Inspect" }),
      expect.objectContaining({ id: "vectors_path", label: "Sweep vectors" }),
    ]));
  });

  it("explains unavailable donor actions", () => {
    const action = nextActionsForArtifact(testArtifact({ kind: "latent" }), { donorLatents: [] })
      .find((item) => item.id === "borrow_texture");

    expect(action).toMatchObject({
      available: false,
      disabledReason: "Encode or recover another latent donor first.",
    });
  });

  it("maps pending and failed takes to retry, cancel, inspect, and tune actions", () => {
    const running = pendingTakeFromJob(testJob({ status: "running" }));
    expect(nextActionsForPendingTake(running).map((action) => action.id)).toEqual(["cancel", "inspect"]);

    const failed = pendingTakeFromJob(testJob({
      status: "failed",
      error: "HF auth token missing",
      recipe: testRecipe({ operator: "generate.text_to_audio" }),
    }));
    expect(nextActionsForPendingTake(failed).map((action) => action.id)).toEqual(["retry", "tune", "inspect"]);
    expect(nextActionsForPendingTake(failed).find((action) => action.id === "tune")?.description).toMatch(/Hugging Face/);
  });

  it("suggests branch actions from the latest take", () => {
    const recipe = testRecipe({ operator: "latent.decode" });
    const family = testFamily({ recipe, operator: "latent.decode", latestArtifactId: "art_audio" });
    const actions = nextActionsForBranch(family, testArtifact({ artifact_id: "art_audio", kind: "audio" }));

    expect(actions.map((action) => action.id)).toEqual(["do_again", "continue_from_latest", "branch", "remember"]);
    expect(actions.find((action) => action.id === "continue_from_latest")).toMatchObject({ gestureId: "continue" });
  });
});
