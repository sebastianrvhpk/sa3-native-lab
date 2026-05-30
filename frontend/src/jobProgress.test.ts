import { describe, expect, it } from "vitest";

import { jobPhase, jobRecoveryHints } from "./jobProgress";
import { landingArtifactId } from "./jobUtils";
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

  it("points Hugging Face cache failures at disk space", () => {
    expect(jobRecoveryHints(job({ error: "Internal Writer Error: Background writer channel closed after disk space warning" }))).toContainEqual(
      expect.objectContaining({ title: "Disk space" }),
    );
  });

  it("keeps cancelled jobs retryable without pretending they failed", () => {
    expect(job({ status: "cancelled" }).status).toBe("cancelled");
    expect(jobRecoveryHints(job({ status: "cancelled" }))).toEqual([
      { title: "Cancelled", detail: "The gesture is preserved; retry it when the current inputs are ready." },
    ]);
  });

  it("derives a readable running phase from the latest event text", () => {
    expect(jobPhase(job({ status: "running", message: "sampling step 4/8", logs: ["loading model", "sampling step 4/8"] }))).toEqual({
      label: "generating",
      tone: "model",
    });
    expect(jobPhase(job({ status: "running", message: "writing bundle artifact" }))).toEqual({
      label: "saving",
      tone: "io",
    });
    expect(jobPhase(job({ status: "running", message: "scoring 2 prompt candidate(s) with SA3 flow loss" }))).toEqual({
      label: "generating",
      tone: "model",
    });
  });

  it("lands on the newest produced artifact only after successful jobs", () => {
    expect(landingArtifactId(job({ status: "succeeded", artifact_ids: ["art_a", "art_b"] }))).toBe("art_b");
    expect(landingArtifactId(job({ status: "failed", artifact_ids: ["art_a"] }))).toBeNull();
    expect(landingArtifactId(job({ status: "succeeded", artifact_ids: [] }))).toBeNull();
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
