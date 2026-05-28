import assert from "node:assert/strict";
import { test } from "node:test";

import type { ArtifactRecord, HealthResponse, JobRecord, NotebookMode, OperatorSpec, SessionRecord } from "./contracts.js";
import { appRouter } from "./router.js";
import { shapeWorkbenchState, type WorkbenchSnapshot } from "./workbench.js";

const now = "2026-05-27T15:00:00.000Z";

test("shapeWorkbenchState groups active session workbench data", () => {
  const session = sessionRecord("sess_active", "2026-05-27T14:00:00.000Z");
  const sessionArtifact = artifactRecord("art_session", "audio", session.session_id, "2026-05-27T14:10:00.000Z");
  const archiveArtifact = artifactRecord("art_archive", "latent", "sess_old", "2026-05-26T12:00:00.000Z");
  const runningJob = jobRecord("job_running", "running", session.session_id, "2026-05-27T14:11:00.000Z");
  const oldJob = jobRecord("job_old", "succeeded", "sess_old", "2026-05-26T12:30:00.000Z");

  const state = shapeWorkbenchState(
    snapshot({
      sessions: [session],
      artifacts: [archiveArtifact, sessionArtifact],
      jobs: [oldJob, runningJob],
    }),
    { sessionId: session.session_id, selectedArtifactId: sessionArtifact.artifact_id },
  );

  assert.equal(state.activeSession?.session_id, session.session_id);
  assert.deepEqual(state.sessionArtifacts.map((artifact) => artifact.artifact_id), [sessionArtifact.artifact_id]);
  assert.deepEqual(state.archiveArtifacts.map((artifact) => artifact.artifact_id), [archiveArtifact.artifact_id]);
  assert.deepEqual(state.runningJobs.map((job) => job.job_id), [runningJob.job_id]);
  assert.equal(state.selectedArtifact?.artifact_id, sessionArtifact.artifact_id);
  assert.equal(state.counts.audioArtifacts, 1);
  assert.equal(state.counts.latentArtifacts, 1);
  assert.deepEqual(state.readiness.readyBackends, ["mlx", "torch_mps"]);
});

test("tRPC workbench.load aggregates Python runtime calls", async () => {
  const session = sessionRecord("sess_active", "2026-05-27T14:00:00.000Z");
  const artifact = artifactRecord("art_session", "audio", session.session_id, "2026-05-27T14:10:00.000Z");
  const job = jobRecord("job_running", "running", session.session_id, "2026-05-27T14:11:00.000Z");
  const fakeFetch: typeof fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    const payloads: Record<string, unknown> = {
      "/health": healthResponse(),
      "/sessions": [session],
      "/artifacts": [artifact],
      "/jobs": [job],
      "/colab/modes": [modeRecord("2", "scaffold")],
      "/operators/specs": [operatorSpec("generate.text_to_audio")],
    };
    if (!(path in payloads)) {
      return Response.json({ detail: "not found" }, { status: 404 });
    }
    return Response.json(payloads[path]);
  };

  const caller = appRouter.createCaller({ baseUrl: "http://python.test", fetchImpl: fakeFetch });
  const state = await caller.workbench.load({ sessionId: session.session_id });

  assert.equal(state.activeSessionId, session.session_id);
  assert.equal(state.sessionArtifacts[0]?.artifact_id, artifact.artifact_id);
  assert.equal(state.runningJobs[0]?.job_id, job.job_id);
  assert.equal(state.counts.scaffoldModes, 1);
});

test("tRPC jobs and recipes procedures call Python lifecycle endpoints", async () => {
  const calls: { path: string; method: string; body: unknown }[] = [];
  const fakeFetch: typeof fetch = async (url, init) => {
    const path = new URL(String(url)).pathname;
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    calls.push({ path, method: init?.method ?? "GET", body });
    if (path === "/jobs/job_running/cancel") {
      return Response.json(jobRecord("job_running", "cancelled", "sess_active", now));
    }
    if (path === "/jobs/job_failed/retry") {
      return Response.json(jobRecord("job_retry", "queued", "sess_active", now));
    }
    if (path === "/recipes/recipe_1/replay") {
      return Response.json(jobRecord("job_replay", "queued", "sess_active", now));
    }
    if (path === "/recipes/recipe_1/fork") {
      return Response.json(jobRecord("job_fork", "queued", "sess_active", now));
    }
    return Response.json({ detail: "not found" }, { status: 404 });
  };

  const caller = appRouter.createCaller({ baseUrl: "http://python.test", fetchImpl: fakeFetch });

  assert.equal((await caller.jobs.cancel({ jobId: "job_running" })).status, "cancelled");
  assert.equal((await caller.jobs.retry({ jobId: "job_failed" })).job_id, "job_retry");
  assert.equal((await caller.recipes.replay({ recipeId: "recipe_1" })).job_id, "job_replay");
  assert.equal((await caller.recipes.fork({ recipeId: "recipe_1", params: { shift_frames: 2 } })).job_id, "job_fork");
  assert.deepEqual(calls.map((call) => [call.method, call.path]), [
    ["POST", "/jobs/job_running/cancel"],
    ["POST", "/jobs/job_failed/retry"],
    ["POST", "/recipes/recipe_1/replay"],
    ["POST", "/recipes/recipe_1/fork"],
  ]);
  assert.deepEqual(calls[3]?.body, { params: { shift_frames: 2 } });
});

function snapshot(overrides: Partial<WorkbenchSnapshot> = {}): WorkbenchSnapshot {
  return {
    health: healthResponse(),
    sessions: [],
    artifacts: [],
    jobs: [],
    modeAtlas: [modeRecord("0", "native app surface")],
    operatorSpecs: [operatorSpec("generate.text_to_audio")],
    ...overrides,
  };
}

function healthResponse(): HealthResponse {
  return {
    app: "sa3-native-lab",
    version: "test",
    artifact_root: "/tmp/lab",
    backends: [
      { backend: "mlx", available: true, loaded: false, device: "metal", details: {} },
      { backend: "torch_mps", available: true, loaded: false, device: "mps", details: {} },
      { backend: "cpu", available: false, loaded: false, details: {} },
    ],
  };
}

function sessionRecord(sessionId: string, createdAt: string): SessionRecord {
  return {
    session_id: sessionId,
    name: sessionId,
    status: "active",
    notes: null,
    metadata: {},
    created_at: createdAt,
    updated_at: createdAt,
    archived_at: null,
  };
}

function artifactRecord(artifactId: string, kind: ArtifactRecord["kind"], sessionId: string | null, createdAt: string): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind,
    path: `/tmp/${artifactId}`,
    file: null,
    audio: kind === "audio" ? { sample_rate: 44100, channels: 2, frames: 44100, duration_seconds: 1 } : null,
    latent: kind === "latent" ? { shape: [16, 256], latent_rate: 10.77, duration_seconds: 1.49, channel_first: false } : null,
    source_artifact_ids: [],
    recipe_id: null,
    label: artifactId,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: sessionId,
    created_at: createdAt,
  };
}

function jobRecord(jobId: string, status: JobRecord["status"], sessionId: string | null, createdAt: string): JobRecord {
  return {
    job_id: jobId,
    status,
    recipe: {
      recipe_id: `recipe_${jobId}`,
      operator: "generate.text_to_audio",
      backend: "mlx",
      inputs: {},
      params: {},
      model: "medium",
      seed: 7,
      notes: null,
      session_id: sessionId,
      created_at: createdAt,
      version: 1,
    },
    progress: status === "succeeded" ? 1 : 0.5,
    message: status,
    artifact_ids: [],
    metrics: {},
    logs: [],
    error: null,
    created_at: createdAt,
    started_at: status === "queued" ? null : createdAt,
    finished_at: status === "succeeded" ? now : null,
  };
}

function modeRecord(modeId: string, status: string): NotebookMode {
  return {
    mode_id: modeId,
    title: `Mode ${modeId}`,
    priority: "P2",
    maturity: "lab",
    status,
    native_surface: "test surface",
    operators: ["generate.text_to_audio"],
    scripts: [],
    inputs: [],
    outputs: [],
    notes: null,
  };
}

function operatorSpec(name: OperatorSpec["name"]): OperatorSpec {
  return {
    name,
    maturity: "core",
    backends: ["mlx"],
    inputs: [],
    params: { prompt: "str" },
    produces: ["audio"],
    status: "implemented",
  };
}
