import { describe, expect, it } from "vitest";

import { jobRecoveryHints } from "./jobProgress";
import type { JobRecord } from "./types";

describe("job recovery hints", () => {
  it("points auth failures at Hugging Face access", () => {
    expect(jobRecoveryHints(job({ error: "Hugging Face 403 forbidden for gated model" }))[0]).toMatchObject({
      title: "Auth or gated weights",
    });
  });

  it("points missing local runtime failures at MLX setup", () => {
    expect(jobRecoveryHints(job({ error: "optimized/mlx/.venv is missing; run optimized/mlx/install.sh" }))[0]).toMatchObject({
      title: "MLX setup",
    });
  });

  it("keeps cancelled jobs retryable without pretending they failed", () => {
    expect(job({ status: "cancelled" }).status).toBe("cancelled");
    expect(jobRecoveryHints(job({ status: "cancelled" }))).toEqual([
      { title: "Cancelled", detail: "The recipe is preserved; retry it when the current inputs are ready." },
    ]);
  });
});

function job(overrides: Partial<JobRecord> = {}): JobRecord {
  return {
    job_id: "job_test",
    status: "failed",
    recipe: {
      recipe_id: "recipe_test",
      operator: "generate.text_to_audio",
      backend: "mlx",
      inputs: {},
      params: {},
      model: "medium",
      seed: 7,
      notes: null,
      session_id: "sess_test",
      created_at: "2026-05-28T00:00:00.000Z",
      version: 1,
    },
    progress: 0.3,
    message: null,
    artifact_ids: [],
    metrics: {},
    logs: [],
    error: null,
    created_at: "2026-05-28T00:00:00.000Z",
    started_at: "2026-05-28T00:00:01.000Z",
    finished_at: "2026-05-28T00:00:05.000Z",
    ...overrides,
  };
}
