import { describe, expect, it } from "vitest";

import { pendingTakeFromJob, pendingTakesFromJobs } from "./pendingTakeModel";
import { operatorName, testJob, testRecipe } from "./test/fixtures";

describe("pendingTakeModel", () => {
  it("turns active generation jobs into pending takes", () => {
    const take = pendingTakeFromJob(testJob({
      status: "running",
      progress: 0.42,
      message: "sampling step 2/8",
      recipe: testRecipe({
        operator: "generate.text_to_audio",
        inputs: {},
      }),
    }));

    expect(take.gestureLabel).toBe("Make");
    expect(take.phrase).toBe("Making take");
    expect(take.progressPercent).toBe(42);
    expect(take.canCancel).toBe(true);
    expect(take.canRetry).toBe(false);
    expect(take.detail).toBe("sampling step 2/8");
  });

  it("uses instrument phrases for latent and experiment jobs", () => {
    expect(pendingTakeFromJob(testJob({
      status: "running",
      recipe: testRecipe({ operator: "latent.graft", inputs: { source: "art_a", donor: "art_b" } }),
    })).phrase).toBe("Borrowing texture");

    expect(pendingTakeFromJob(testJob({
      status: "queued",
      recipe: testRecipe({ operator: "latent.decode" }),
    })).phrase).toBe("Decode queued");

    expect(pendingTakeFromJob(testJob({
      status: "running",
      recipe: testRecipe({ operator: operatorName("experiment.prompt_search") }),
    })).phrase).toBe("Probing prompts");
  });

  it("keeps failed jobs retryable without exposing job as the primary noun", () => {
    const [take] = pendingTakesFromJobs([
      testJob({ status: "failed", error: "missing input", recipe: testRecipe({ operator: "latent.blur" }) }),
    ]);

    expect(take.status).toBe("failed");
    expect(take.phrase).toBe("Take failed");
    expect(take.canCancel).toBe(false);
    expect(take.canRetry).toBe(true);
    expect(take.inspect.operator).toBe("latent.blur");
  });
});
