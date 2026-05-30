import { describe, expect, it } from "vitest";

import {
  branchLabelForJob,
  completionPhraseForJob,
  landingArtifactIdForJob,
  pendingTakeLandingFromJob,
  recoverySuggestionForJob,
} from "./pendingTakeLandingModel";
import { testJob, testRecipe } from "./test/fixtures";

describe("pendingTakeLandingModel", () => {
  it("selects the newest produced artifact only after a take succeeds", () => {
    const succeeded = testJob({
      status: "succeeded",
      artifact_ids: ["art_first", "art_latest"],
      recipe: testRecipe({ operator: "generate.audio_to_audio", inputs: { source: "art_source" } }),
    });

    expect(landingArtifactIdForJob(succeeded)).toBe("art_latest");
    expect(pendingTakeLandingFromJob(succeeded)).toMatchObject({
      pendingTakeId: succeeded.job_id,
      sourceIds: ["art_source"],
      producedArtifactIds: ["art_first", "art_latest"],
      landingArtifactId: "art_latest",
      completionPhrase: "Takes landed",
      branchLabel: "2 takes",
      canCancel: false,
      canRetry: false,
      canInspect: true,
    });
  });

  it("uses product phrases for queued, failed, and cancelled takes", () => {
    const queued = testJob({ status: "queued", artifact_ids: [], recipe: testRecipe({ operator: "latent.decode" }) });
    const failed = testJob({
      status: "failed",
      artifact_ids: [],
      error: "out of memory during sampling",
      recipe: testRecipe({ operator: "generate.text_to_audio" }),
    });
    const cancelled = testJob({
      status: "cancelled",
      artifact_ids: [],
      recipe: testRecipe({ operator: "latent.blur" }),
    });

    expect(completionPhraseForJob(queued)).toBe("Waiting for a slot");
    expect(branchLabelForJob(queued)).toBe("pending take");
    expect(completionPhraseForJob(failed)).toBe("No take landed");
    expect(branchLabelForJob(failed)).toBe("needs recovery");
    expect(recoverySuggestionForJob(failed)).toBe("Reduce duration or steps in Tune, then retry.");
    expect(completionPhraseForJob(cancelled)).toBe("Take stopped before landing");
    expect(branchLabelForJob(cancelled)).toBe("stopped");
    expect(recoverySuggestionForJob(cancelled)).toBe("Retry when you are ready, or adjust Tune first.");
  });
});
