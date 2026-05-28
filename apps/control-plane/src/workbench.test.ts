import assert from "node:assert/strict";
import { test } from "node:test";

import type { ArtifactRecord, HealthResponse, JobRecord, NotebookMode, OperatorSpec, SessionRecord } from "./contracts.js";
import { parseTrackedSequence, pollJobEvents } from "./jobEvents.js";
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
  assert.deepEqual(state.sessionResultFamilies.map((family) => family.recipeId), [runningJob.recipe.recipe_id]);
  assert.equal(state.sessionResultFamilies[0]?.status, "running");
  assert.equal(state.counts.audioArtifacts, 1);
  assert.equal(state.counts.latentArtifacts, 1);
  assert.deepEqual(state.readiness.readyBackends, ["mlx", "torch_mps"]);
});

test("shapeWorkbenchState groups prompt candidate takes under their search bundle", () => {
  const session = sessionRecord("sess_active", "2026-05-27T14:00:00.000Z");
  const bundle = artifactRecord("art_prompt_bundle", "bundle", session.session_id, "2026-05-27T14:05:00.000Z");
  const firstJob = promptCandidateJob("job_candidate_a", session.session_id, bundle.artifact_id, "2026-05-27T14:10:00.000Z");
  const secondJob = promptCandidateJob("job_candidate_b", session.session_id, bundle.artifact_id, "2026-05-27T14:11:00.000Z");
  const firstTake = {
    ...artifactRecord("art_candidate_a", "audio", session.session_id, "2026-05-27T14:12:00.000Z"),
    recipe_id: firstJob.recipe.recipe_id,
    source_artifact_ids: [bundle.artifact_id],
  };
  const secondTake = {
    ...artifactRecord("art_candidate_b", "audio", session.session_id, "2026-05-27T14:13:00.000Z"),
    recipe_id: secondJob.recipe.recipe_id,
    source_artifact_ids: [bundle.artifact_id],
  };

  const state = shapeWorkbenchState(
    snapshot({
      sessions: [session],
      artifacts: [bundle, firstTake, secondTake],
      jobs: [firstJob, secondJob],
    }),
    { sessionId: session.session_id },
  );

  assert.equal(state.sessionResultFamilies.length, 1);
  assert.equal(state.sessionResultFamilies[0]?.familyId, `prompt-candidates:${bundle.artifact_id}`);
  assert.deepEqual(new Set(state.sessionResultFamilies[0]?.artifactIds), new Set([firstTake.artifact_id, secondTake.artifact_id]));
  assert.deepEqual(new Set(state.sessionResultFamilies[0]?.jobIds), new Set([firstJob.job_id, secondJob.job_id]));
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
  assert.equal(state.sessionResultFamilies[0]?.recipeId, job.recipe.recipe_id);
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

test("pollJobEvents emits changed snapshots until terminal state", async () => {
  let calls = 0;
  const client = {
    job: async (jobId: string) => {
      calls += 1;
      if (calls === 1) return { ...jobRecord(jobId, "running", "sess_active", now), progress: 0.25 };
      return { ...jobRecord(jobId, "succeeded", "sess_active", now), progress: 1 };
    },
  };

  const events = [];
  for await (const event of pollJobEvents(client, { jobId: "job_running", intervalMs: 1 })) {
    events.push(event);
  }

  assert.deepEqual(events.map((event) => event.type), ["snapshot", "snapshot"]);
  assert.deepEqual(events.map((event) => event.type === "snapshot" ? event.job.status : "error"), ["running", "succeeded"]);
  assert.deepEqual(events.map((event) => event.sequence), [1, 2]);
  assert.deepEqual(events.map((event) => event.diagnostics.eventSource), ["python-job-snapshot-poll", "python-job-snapshot-poll"]);
  assert.deepEqual(events.map((event) => event.diagnostics.pollIntervalMs), [1, 1]);
  assert.deepEqual(events.map((event) => event.type === "snapshot" ? event.diagnostics.logTail.length : 0), [0, 0]);
});

test("pollJobEvents emits heartbeats and resumes monotonic ids", async () => {
  let calls = 0;
  const client = {
    job: async (jobId: string) => {
      calls += 1;
      if (calls < 3) return { ...jobRecord(jobId, "running", "sess_active", now), progress: 0.25, logs: ["loading"] };
      return { ...jobRecord(jobId, "succeeded", "sess_active", now), progress: 1, logs: ["loading", "done"] };
    },
  };

  const events = [];
  for await (const event of pollJobEvents(client, { jobId: "job_running", intervalMs: 1, heartbeatEveryMs: 1, lastEventId: "job_running:7" })) {
    events.push(event);
  }

  assert.deepEqual(events.map((event) => event.type), ["snapshot", "heartbeat", "snapshot"]);
  assert.deepEqual(events.map((event) => event.sequence), [8, 9, 10]);
  assert.equal(events[0]?.diagnostics.resumedFromSequence, 7);
  assert.equal(events[1]?.type === "heartbeat" ? events[1].diagnostics.unchangedPolls : 0, 1);
  assert.deepEqual(events[2]?.diagnostics.logTail, ["loading", "done"]);
});

test("pollJobEvents replays durable journal events before polling", async () => {
  let livePolls = 0;
  const running = { ...jobRecord("job_running", "running", "sess_active", now), progress: 0.25, logs: ["loading"] };
  const done = { ...jobRecord("job_running", "succeeded", "sess_active", now), progress: 1, logs: ["loading", "done"] };
  const client = {
    jobEventHistory: async (_jobId: string, after = 0) => [
      { sequence: after + 1, type: "snapshot" as const, created_at: now, job: running },
      { sequence: after + 2, type: "snapshot" as const, created_at: now, job: done },
    ],
    job: async () => {
      livePolls += 1;
      return done;
    },
  };

  const events = [];
  for await (const event of pollJobEvents(client, { jobId: "job_running", intervalMs: 1, lastEventId: "job_running:3" })) {
    events.push(event);
  }

  assert.equal(livePolls, 0);
  assert.deepEqual(events.map((event) => event.sequence), [4, 5]);
  assert.deepEqual(events.map((event) => event.diagnostics.eventSource), ["python-job-journal", "python-job-journal"]);
  assert.equal(events[0]?.diagnostics.resumedFromSequence, 3);
});

test("parseTrackedSequence accepts only ids for the same job", () => {
  assert.equal(parseTrackedSequence("job_a", "job_a:12"), 12);
  assert.equal(parseTrackedSequence("job_a", "job_b:12"), null);
  assert.equal(parseTrackedSequence("job_a", "job_a:not-a-number"), null);
});

test("tRPC system readiness calls Python readiness endpoint", async () => {
  const fakeFetch: typeof fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    if (path === "/readiness") {
      return Response.json({
        ok: true,
        warnings: 0,
        errors: 0,
        checks: [{ name: "artifact-root", status: "ok", message: "writable" }],
      });
    }
    return Response.json({ detail: "not found" }, { status: 404 });
  };
  const caller = appRouter.createCaller({ baseUrl: "http://python.test", fetchImpl: fakeFetch });

  const readiness = await caller.system.readiness();

  assert.equal(readiness.ok, true);
  assert.equal(readiness.checks[0]?.name, "artifact-root");
});

test("tRPC artifact inspection and family loading expose app-shaped records", async () => {
  const session = sessionRecord("sess_active", "2026-05-27T14:00:00.000Z");
  const artifact = {
    ...artifactRecord("art_bundle", "bundle", session.session_id, "2026-05-27T14:10:00.000Z"),
    recipe_id: "recipe_job_done",
    file: { filename: "alpha_sweep.zip", media_type: "application/zip", byte_size: 128, sha256: "abc" },
  };
  const job = {
    ...jobRecord("job_done", "succeeded", session.session_id, "2026-05-27T14:11:00.000Z"),
    artifact_ids: [artifact.artifact_id],
    metrics: { result_count: 3 },
  };
  const fakeFetch: typeof fetch = async (url) => {
    const path = new URL(String(url)).pathname;
    const payloads: Record<string, unknown> = {
      "/health": healthResponse(),
      "/sessions": [session],
      "/artifacts": [artifact],
      "/jobs": [job],
      "/colab/modes": [modeRecord("2", "native recipe")],
      "/operators/specs": [operatorSpec("experiment.alpha_sweep")],
      "/artifacts/art_bundle/inspect": {
        artifact,
        recipe: job.recipe,
        sources: [],
        children: [],
        bundle_files: [{ path: "metrics.json", byte_size: 16, compressed_size: 12 }],
        bundle_audio_files: [{ path: "take.wav", byte_size: 3200, media_type: "audio/wav", duration_seconds: 0.1 }],
        bundle_preview: { result_count: 3 },
        bundle_summary: { kind: "sweep", file_count: 1 },
      },
    };
    if (!(path in payloads)) {
      return Response.json({ detail: "not found" }, { status: 404 });
    }
    return Response.json(payloads[path]);
  };

  const caller = appRouter.createCaller({ baseUrl: "http://python.test", fetchImpl: fakeFetch });

  const inspection = await caller.artifacts.inspect({ artifactId: artifact.artifact_id });
  const families = await caller.families.load({ sessionId: session.session_id });

  assert.equal(inspection.bundle_files[0]?.path, "metrics.json");
  assert.equal(inspection.bundle_audio_files[0]?.path, "take.wav");
  assert.equal(inspection.bundle_preview.result_count, 3);
  assert.equal(families.sessionResultFamilies[0]?.latestArtifactId, artifact.artifact_id);
  assert.equal(families.sessionResultFamilies[0]?.artifactKinds[0], "bundle");
  assert.equal(families.sessionResultFamilies[0]?.metrics.result_count, 3);
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

function promptCandidateJob(jobId: string, sessionId: string | null, sourceId: string, createdAt: string): JobRecord {
  return {
    ...jobRecord(jobId, "succeeded", sessionId, createdAt),
    recipe: {
      ...jobRecord(jobId, "succeeded", sessionId, createdAt).recipe,
      inputs: { source: sourceId },
      params: {
        prompt: `candidate ${jobId}`,
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_search_bundle_id: sourceId,
          prompt_candidate_rank: jobId.endsWith("_a") ? 1 : 2,
        },
      },
    },
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
    ui_fields: [],
    produces: ["audio"],
    status: "implemented",
  };
}
