import { describe, expect, it } from "vitest";

import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord, JobStatus, Recipe } from "./types";
import {
  buildResultFamilies,
  createdAfter,
  familyStatus,
  filterFamiliesForWork,
  jobFromJobEvent,
  mergeJobRecords,
  parseJobEvent,
  primitiveMetadataValue,
  resultFamilyIdForRecipe,
} from "./workbenchModel";

describe("workbenchModel", () => {
  it("groups prompt-search candidate generations into one result family", () => {
    const firstRecipe = recipe({
      recipe_id: "recipe_candidate_a",
      inputs: { source: "art_prompt_bundle" },
      params: { metadata: { generation_origin: "prompt_search_candidate" } },
      created_at: "2026-05-27T15:00:00.000Z",
    });
    const secondRecipe = recipe({
      recipe_id: "recipe_candidate_b",
      inputs: { source: "art_prompt_bundle" },
      params: { metadata: { generation_origin: "prompt_search_candidate" } },
      created_at: "2026-05-27T15:03:00.000Z",
    });
    const families = buildResultFamilies(
      [
        artifact("art_a", "recipe_candidate_a", "2026-05-27T15:02:00.000Z"),
        artifact("art_b", "recipe_candidate_b", "2026-05-27T15:04:00.000Z"),
      ],
      [
        job("job_a", firstRecipe, "succeeded", "2026-05-27T15:01:00.000Z", ["art_a"]),
        job("job_b", secondRecipe, "succeeded", "2026-05-27T15:03:00.000Z", ["art_b"]),
      ],
    );

    expect(families).toHaveLength(1);
    expect(families[0]).toMatchObject({
      familyId: "prompt-candidates:art_prompt_bundle",
      recipeId: "recipe_candidate_b",
      status: "succeeded",
      jobIds: ["job_a", "job_b"],
      artifactIds: ["art_a", "art_b"],
      latestArtifactId: "art_b",
      artifactKinds: ["audio"],
    });
  });

  it("keeps family identity on regular recipes", () => {
    const regularRecipe = recipe({ recipe_id: "recipe_direct", inputs: {}, params: {} });

    expect(resultFamilyIdForRecipe(regularRecipe)).toBe("recipe_direct");
  });

  it("prioritizes active and terminal family statuses", () => {
    const baseRecipe = recipe();

    expect(familyStatus([job("job_running", baseRecipe, "running")])).toBe("running");
    expect(familyStatus([job("job_queued", baseRecipe, "queued"), job("job_failed", baseRecipe, "failed")])).toBe("queued");
    expect(familyStatus([job("job_failed", baseRecipe, "failed"), job("job_done", baseRecipe, "succeeded")])).toBe("failed");
    expect(familyStatus([job("job_cancelled", baseRecipe, "cancelled")])).toBe("cancelled");
    expect(familyStatus([job("job_done", baseRecipe, "succeeded")])).toBe("succeeded");
    expect(familyStatus([])).toBe("mixed");
  });

  it("filters families to the active session/work archive by recipe id", () => {
    const activeRecipe = recipe({ recipe_id: "recipe_active" });
    const archivedRecipe = recipe({ recipe_id: "recipe_archived" });
    const families = [
      family(activeRecipe, "art_active"),
      family(archivedRecipe, "art_archived"),
    ];

    expect(filterFamiliesForWork(families, [artifact("art_active", "recipe_active")], [])).toEqual([families[0]]);
    expect(filterFamiliesForWork(families, [], [job("job_archived", archivedRecipe)])).toEqual([families[1]]);
  });

  it("merges live job overlays and keeps newest jobs first", () => {
    const baseRecipe = recipe();
    const original = job("job_1", baseRecipe, "running", "2026-05-27T15:00:00.000Z");
    const overlay = { ...original, progress: 0.9, message: "almost there", created_at: "2026-05-27T15:02:00.000Z" };
    const newer = job("job_2", baseRecipe, "queued", "2026-05-27T15:05:00.000Z");

    const merged = mergeJobRecords([original, newer], [overlay]);

    expect(merged.map((item) => item.job_id)).toEqual(["job_2", "job_1"]);
    expect(merged[1]).toMatchObject({ progress: 0.9, message: "almost there" });
  });

  it("parses job events from websocket and control-plane payloads", () => {
    const baseJob = job("job_event", recipe(), "running");

    expect(parseJobEvent(JSON.stringify({ type: "snapshot", job: baseJob }))).toMatchObject({ job_id: "job_event" });
    expect(parseJobEvent(JSON.stringify({ data: { type: "snapshot", job: baseJob } }))).toMatchObject({ job_id: "job_event" });
    expect(parseJobEvent(JSON.stringify(baseJob))).toMatchObject({ job_id: "job_event" });
    expect(parseJobEvent("{not json")).toBeNull();
    expect(jobFromJobEvent({ type: "error", error: "boom" })).toBeNull();
  });

  it("keeps lenient timestamp and metadata fallbacks", () => {
    expect(createdAfter("bad-date", "2026-05-27T15:00:00.000Z")).toBe(true);
    expect(createdAfter("2026-05-27T15:01:00.000Z", "2026-05-27T15:00:00.000Z")).toBe(true);
    expect(createdAfter("2026-05-27T14:59:00.000Z", "2026-05-27T15:00:00.000Z")).toBe(false);
    expect(primitiveMetadataValue("keeper")).toBe("keeper");
    expect(primitiveMetadataValue(3)).toBe(3);
    expect(primitiveMetadataValue({ nested: true })).toBeNull();
  });
});

function family(recipeRecord: Recipe, artifactId: string): ResultFamily {
  return {
    familyId: recipeRecord.recipe_id,
    recipeId: recipeRecord.recipe_id,
    recipe: recipeRecord,
    operator: recipeRecord.operator,
    sessionId: recipeRecord.session_id ?? null,
    status: "succeeded",
    jobIds: [`job_${recipeRecord.recipe_id}`],
    artifactIds: [artifactId],
    artifactKinds: ["audio"],
    metrics: {},
    latestArtifactId: artifactId,
    createdAt: recipeRecord.created_at,
    updatedAt: recipeRecord.created_at,
  };
}

function recipe(overrides: Partial<Recipe> = {}): Recipe {
  return {
    recipe_id: overrides.recipe_id ?? "recipe_1",
    operator: overrides.operator ?? "generate.text_to_audio",
    backend: overrides.backend ?? "mlx",
    inputs: overrides.inputs ?? {},
    params: overrides.params ?? {},
    model: overrides.model ?? "medium",
    seed: overrides.seed ?? 7,
    notes: overrides.notes ?? null,
    session_id: overrides.session_id ?? "sess_1",
    created_at: overrides.created_at ?? "2026-05-27T15:00:00.000Z",
    version: overrides.version ?? 1,
  };
}

function job(
  jobId: string,
  recipeRecord: Recipe,
  status: JobStatus = "succeeded",
  createdAt = "2026-05-27T15:00:00.000Z",
  artifactIds: string[] = [],
): JobRecord {
  return {
    job_id: jobId,
    status,
    recipe: recipeRecord,
    progress: status === "succeeded" ? 1 : 0.2,
    phase: status,
    message: null,
    artifact_ids: artifactIds,
    metrics: {},
    logs: [],
    error: null,
    created_at: createdAt,
    started_at: createdAt,
    finished_at: status === "succeeded" ? createdAt : null,
  };
}

function artifact(
  artifactId: string,
  recipeId: string | null = "recipe_1",
  createdAt = "2026-05-27T15:00:00.000Z",
): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind: "audio",
    path: `/tmp/${artifactId}.wav`,
    file: { filename: `${artifactId}.wav`, media_type: "audio/wav", byte_size: 1024 },
    audio: { sample_rate: 24000, channels: 1, frames: 24000, duration_seconds: 1, format: "WAV" },
    latent: null,
    source_artifact_ids: [],
    recipe_id: recipeId,
    label: artifactId,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: "sess_1",
    created_at: createdAt,
  };
}
