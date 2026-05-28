import { describe, expect, it } from "vitest";

import { artifactFilterOptions, emptyArtifactFilters, filterArtifacts, type ArtifactFilterState } from "./artifactFilters";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord, Recipe } from "./types";

describe("artifact filters", () => {
  it("recovers listened takes by decision, kind, model, operator, family, and lineage", () => {
    const source = audioArtifact("art_source", { label: "target click", recipeId: null });
    const keeper = audioArtifact("art_keeper", {
      label: "bright take",
      recipeId: "recipe_take",
      sourceIds: ["art_bundle"],
      tags: ["keeper", "bright"],
      metadata: { listening_decision: "keeper", model: "medium" },
    });
    const rejected = audioArtifact("art_rejected", {
      label: "dull take",
      recipeId: "recipe_other",
      tags: ["rejected"],
      metadata: { listening_decision: "rejected", model: "sm-music" },
    });
    const bundle = bundleArtifact("art_bundle", {
      label: "prompt search bundle",
      recipeId: "recipe_search",
      metadata: { operator: "experiment.prompt_search", model: "medium" },
    });
    const family = resultFamily("prompt-candidates:art_bundle", "recipe_take", [keeper.artifact_id]);
    const jobs = [job("job_take", recipe("recipe_take", "generate.text_to_audio", "medium"))];

    const artifacts = [source, keeper, rejected, bundle];
    const context = { jobs, families: [family] };

    expect(filterArtifacts(artifacts, filters({ decision: "keeper" }), context).map(id)).toEqual(["art_keeper"]);
    expect(filterArtifacts(artifacts, filters({ kind: "bundle" }), context).map(id)).toEqual(["art_bundle"]);
    expect(filterArtifacts(artifacts, filters({ model: "medium", operator: "generate.text_to_audio" }), context).map(id)).toEqual(["art_keeper"]);
    expect(filterArtifacts(artifacts, filters({ familyId: family.familyId }), context).map(id)).toEqual(["art_keeper"]);
    expect(filterArtifacts(artifacts, filters({ lineage: "has_sources" }), context).map(id)).toEqual(["art_keeper"]);
    expect(filterArtifacts(artifacts, filters({ lineage: "source" }), context).map(id)).toEqual(["art_source"]);
    expect(filterArtifacts(artifacts, filters({ query: "prompt candidates" }), context).map(id)).toEqual(["art_keeper"]);
  });

  it("builds option counts from artifact metadata and job recipes", () => {
    const artifacts = [
      audioArtifact("art_keeper", {
        recipeId: "recipe_take",
        tags: ["keeper"],
        metadata: { listening_decision: "keeper" },
      }),
      audioArtifact("art_maybe", {
        recipeId: "recipe_take",
        tags: ["maybe"],
        metadata: { listening_decision: "maybe" },
      }),
    ];
    const options = artifactFilterOptions(artifacts, {
      jobs: [job("job_take", recipe("recipe_take", "generate.text_to_audio", "medium"))],
      families: [resultFamily("prompt-candidates:bundle", "recipe_take", artifacts.map(id))],
    });

    expect(options.decisions).toEqual([
      { value: "keeper", label: "keeper", count: 1 },
      { value: "maybe", label: "maybe", count: 1 },
    ]);
    expect(options.models).toEqual([{ value: "medium", label: "medium", count: 2 }]);
    expect(options.operators).toEqual([{ value: "generate.text_to_audio", label: "generate.text_to_audio", count: 2 }]);
    expect(options.families).toEqual([{ value: "prompt-candidates:bundle", label: "prompt candidates", count: 2 }]);
  });
});

function filters(overrides: Partial<ArtifactFilterState>): ArtifactFilterState {
  return { ...emptyArtifactFilters, ...overrides };
}

function id(artifact: ArtifactRecord) {
  return artifact.artifact_id;
}

function recipe(recipeId: string, operator: Recipe["operator"], model: string): Recipe {
  return {
    recipe_id: recipeId,
    operator,
    backend: "mlx",
    inputs: {},
    params: {},
    model,
    seed: 7,
    notes: null,
    session_id: "sess",
    created_at: "2026-05-28T10:00:00.000Z",
    version: 1,
  };
}

function job(jobId: string, jobRecipe: Recipe): JobRecord {
  return {
    job_id: jobId,
    status: "succeeded",
    recipe: jobRecipe,
    progress: 1,
    message: "done",
    artifact_ids: [],
    metrics: {},
    logs: [],
    error: null,
    created_at: jobRecipe.created_at,
    started_at: null,
    finished_at: "2026-05-28T10:01:00.000Z",
  };
}

function resultFamily(familyId: string, recipeId: string, artifactIds: string[]): ResultFamily {
  const familyRecipe = recipe(recipeId, "generate.text_to_audio", "medium");
  return {
    familyId,
    recipeId,
    recipe: familyRecipe,
    operator: familyRecipe.operator,
    sessionId: "sess",
    status: "succeeded",
    jobIds: ["job_take"],
    artifactIds,
    artifactKinds: ["audio"],
    metrics: {},
    latestArtifactId: artifactIds[0] ?? null,
    createdAt: "2026-05-28T10:00:00.000Z",
    updatedAt: "2026-05-28T10:01:00.000Z",
  };
}

function audioArtifact(
  artifactId: string,
  overrides: {
    label?: string | null;
    recipeId?: string | null;
    sourceIds?: string[];
    tags?: string[];
    metadata?: Record<string, unknown>;
  } = {},
): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind: "audio",
    path: `/tmp/${artifactId}.wav`,
    file: { filename: `${artifactId}.wav`, byte_size: 100 },
    audio: { sample_rate: 24000, channels: 1, frames: 24000, duration_seconds: 1 },
    latent: null,
    source_artifact_ids: overrides.sourceIds ?? [],
    recipe_id: overrides.recipeId,
    label: overrides.label ?? artifactId,
    prompt: null,
    notes: null,
    tags: overrides.tags ?? [],
    metadata: overrides.metadata ?? {},
    session_id: "sess",
    created_at: "2026-05-28T10:00:00.000Z",
  };
}

function bundleArtifact(
  artifactId: string,
  overrides: {
    label?: string | null;
    recipeId?: string | null;
    metadata?: Record<string, unknown>;
  } = {},
): ArtifactRecord {
  return {
    ...audioArtifact(artifactId, {
      label: overrides.label,
      recipeId: overrides.recipeId,
      metadata: overrides.metadata,
    }),
    kind: "bundle",
    audio: null,
    file: { filename: `${artifactId}.zip`, byte_size: 100 },
  };
}
