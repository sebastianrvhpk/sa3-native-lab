import { describe, expect, it } from "vitest";

import { artifactLineageModel } from "./lineageModel";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord, Recipe } from "./types";

describe("artifact lineage model", () => {
  it("maps source, recipe, job, family, and compare slot into data-backed nodes", () => {
    const source = artifact("art_source", "audio", null);
    const current = artifact("art_take", "audio", "recipe_generate", [source.artifact_id]);
    const recipe = recipeRecord("recipe_generate", "generate.audio_to_audio");
    const job = jobRecord("job_generate", recipe, [current.artifact_id]);
    const family = familyRecord("family_generate", recipe, [current.artifact_id]);

    expect(
      artifactLineageModel({
        artifact: current,
        sources: [source],
        jobs: [job],
        families: [family],
        compare: { a: current.artifact_id, b: null },
      }).map((node) => [node.kind, node.label]),
    ).toEqual([
      ["source", "art_source"],
      ["recipe", "audio to audio"],
      ["job", "done"],
      ["current", "audio"],
      ["family", "1 take"],
      ["compare", "A/B A"],
    ]);
  });

  it("does not invent recipe or family nodes for imported root artifacts", () => {
    expect(
      artifactLineageModel({
        artifact: artifact("art_import", "audio", null),
        sources: [],
        jobs: [],
        families: [],
        compare: { a: null, b: null },
      }).map((node) => node.kind),
    ).toEqual(["origin", "current"]);
  });
});

function artifact(
  artifactId: string,
  kind: ArtifactRecord["kind"],
  recipeId: string | null,
  sourceIds: string[] = [],
): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind,
    path: `/tmp/${artifactId}`,
    file: null,
    audio: kind === "audio" ? { sample_rate: 44100, channels: 2, frames: 44100, duration_seconds: 1 } : null,
    latent: null,
    source_artifact_ids: sourceIds,
    recipe_id: recipeId,
    label: artifactId,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: "sess_1",
    created_at: "2026-05-28T00:00:00.000Z",
  };
}

function recipeRecord(recipeId: string, operator: Recipe["operator"]): Recipe {
  return {
    recipe_id: recipeId,
    operator,
    backend: "mlx",
    inputs: {},
    params: {},
    model: "medium",
    seed: 7,
    notes: null,
    session_id: "sess_1",
    created_at: "2026-05-28T00:00:00.000Z",
    version: 1,
  };
}

function jobRecord(jobId: string, recipe: Recipe, artifactIds: string[]): JobRecord {
  return {
    job_id: jobId,
    status: "succeeded",
    recipe,
    progress: 1,
    phase: "done",
    message: "succeeded",
    artifact_ids: artifactIds,
    metrics: {},
    logs: [],
    error: null,
    created_at: "2026-05-28T00:00:00.000Z",
    started_at: "2026-05-28T00:00:01.000Z",
    finished_at: "2026-05-28T00:00:05.000Z",
  };
}

function familyRecord(familyId: string, recipe: Recipe, artifactIds: string[]): ResultFamily {
  return {
    familyId,
    recipeId: recipe.recipe_id,
    recipe,
    operator: recipe.operator,
    sessionId: "sess_1",
    status: "succeeded",
    jobIds: ["job_generate"],
    artifactIds,
    artifactKinds: ["audio"],
    metrics: {},
    latestArtifactId: artifactIds[0] ?? null,
    createdAt: "2026-05-28T00:00:00.000Z",
    updatedAt: "2026-05-28T00:00:05.000Z",
  };
}
