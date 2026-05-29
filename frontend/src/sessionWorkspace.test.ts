import { describe, expect, it } from "vitest";

import { summarizeSessionWorkspace, workspaceFocus, workspacePulseRows } from "./sessionWorkspace";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord, Recipe } from "./types";

describe("session workspace model", () => {
  it("summarizes active takes, decisions, jobs, families, and archive pressure", () => {
    const keep = audio("art_keep", "2026-05-28T10:00:00.000Z", { listening_decision: "keeper" });
    const open = audio("art_open", "2026-05-28T11:00:00.000Z");
    const latent = artifact("art_latent", "latent", "2026-05-28T12:00:00.000Z");
    const bundle = artifact("art_bundle", "bundle", "2026-05-28T13:00:00.000Z");
    const run = job("job_run", "running");
    const queued = job("job_queue", "queued");
    const failed = job("job_failed", "failed");
    const archived = audio("art_archived", "2026-05-27T10:00:00.000Z");

    const summary = summarizeSessionWorkspace({
      artifacts: [keep, open, latent, bundle],
      archivedArtifacts: [archived],
      jobs: [run, queued, failed],
      archivedJobs: [job("job_old", "succeeded")],
      runningJobs: [run, queued],
      families: [family("family_1"), family("family_2")],
      selectedId: archived.artifact_id,
    });

    expect(summary).toMatchObject({
      takes: 4,
      audioTakes: 2,
      latentArtifacts: 1,
      bundleArtifacts: 1,
      familyCount: 2,
      activeJobs: 2,
      runningJobs: 1,
      queuedJobs: 1,
      failedJobs: 1,
      archiveItems: 2,
      selectedInArchive: true,
      latestArtifactId: "art_bundle",
      latestAudioId: "art_open",
      openAudioTakes: 1,
      decisions: { keeper: 1, maybe: 0, rejected: 0, undecided: 1 },
    });
    expect(workspacePulseRows(summary)).toContainEqual({
      key: "decisions",
      label: "Decisions",
      value: "1/2",
      detail: "1 open audio take",
      tone: "decision",
    });
    expect(workspaceFocus(summary)).toEqual({
      label: "Monitor run",
      detail: "1 running, 1 queued",
      tone: "run",
    });
  });

  it("prioritizes open listening work when no job is active", () => {
    const summary = summarizeSessionWorkspace({
      artifacts: [audio("art_take", "2026-05-28T11:00:00.000Z")],
      archivedArtifacts: [],
      jobs: [job("job_done", "succeeded")],
      archivedJobs: [],
      runningJobs: [],
      families: [family("family_1")],
    });

    expect(workspaceFocus(summary)).toEqual({
      label: "Listen next",
      detail: "1 undecided audio take",
      tone: "listen",
      artifactId: "art_take",
    });
  });
});

function audio(artifactId: string, createdAt: string, metadata: Record<string, unknown> = {}): ArtifactRecord {
  return {
    ...artifact(artifactId, "audio", createdAt, metadata),
    audio: { sample_rate: 24000, channels: 2, frames: 24000, duration_seconds: 1 },
  };
}

function artifact(artifactId: string, kind: ArtifactRecord["kind"], createdAt: string, metadata: Record<string, unknown> = {}): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind,
    path: `/tmp/${artifactId}`,
    file: { filename: artifactId, byte_size: 100 },
    audio: null,
    latent: kind === "latent" ? { shape: [8, 64], latent_rate: 21.5, channel_first: false } : null,
    source_artifact_ids: [],
    recipe_id: "recipe_1",
    label: artifactId,
    prompt: null,
    notes: null,
    tags: [],
    metadata,
    session_id: "sess",
    created_at: createdAt,
  };
}

function job(jobId: string, status: JobRecord["status"]): JobRecord {
  return {
    job_id: jobId,
    status,
    recipe: recipe("recipe_1"),
    progress: status === "succeeded" ? 1 : 0.4,
    message: status,
    artifact_ids: [],
    metrics: {},
    logs: [],
    created_at: "2026-05-28T10:00:00.000Z",
  };
}

function recipe(recipeId: string): Recipe {
  return {
    recipe_id: recipeId,
    operator: "generate.text_to_audio",
    backend: "mlx",
    inputs: {},
    params: {},
    model: "medium",
    seed: 7,
    notes: null,
    session_id: "sess",
    created_at: "2026-05-28T10:00:00.000Z",
    version: 1,
  };
}

function family(familyId: string): ResultFamily {
  const familyRecipe = recipe("recipe_1");
  return {
    familyId,
    recipeId: familyRecipe.recipe_id,
    recipe: familyRecipe,
    operator: familyRecipe.operator,
    sessionId: "sess",
    status: "succeeded",
    jobIds: [],
    artifactIds: [],
    artifactKinds: ["audio"],
    metrics: {},
    latestArtifactId: null,
    createdAt: "2026-05-28T10:00:00.000Z",
    updatedAt: "2026-05-28T10:01:00.000Z",
  };
}
