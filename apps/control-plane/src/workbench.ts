import { z } from "zod";

import type {
  ArtifactRecord,
  HealthResponse,
  JobRecord,
  NotebookMode,
  OperatorSpec,
  SessionRecord,
} from "./contracts.js";
import type { PythonClient } from "./pythonClient.js";

export const workbenchLoadInputSchema = z.object({
  sessionId: z.string().nullable().optional(),
  sessionStartedAt: z.string().datetime().nullable().optional(),
  selectedArtifactId: z.string().nullable().optional(),
});

export type WorkbenchLoadInput = z.infer<typeof workbenchLoadInputSchema>;

export interface WorkbenchSnapshot {
  health: HealthResponse;
  sessions: SessionRecord[];
  artifacts: ArtifactRecord[];
  jobs: JobRecord[];
  modeAtlas: NotebookMode[];
  operatorSpecs: OperatorSpec[];
}

export interface WorkbenchState extends WorkbenchSnapshot {
  activeSession: SessionRecord | null;
  activeSessionId: string | null;
  selectedArtifact: ArtifactRecord | null;
  sessionArtifacts: ArtifactRecord[];
  archiveArtifacts: ArtifactRecord[];
  sessionJobs: JobRecord[];
  archiveJobs: JobRecord[];
  runningJobs: JobRecord[];
  latestJob: JobRecord | null;
  counts: {
    audioArtifacts: number;
    latentArtifacts: number;
    bundleArtifacts: number;
    activeJobs: number;
    scaffoldModes: number;
  };
  readiness: {
    readyBackends: string[];
    offlineBackends: string[];
    hasMediumMlx: boolean;
    hasTorch: boolean;
  };
}

export async function loadWorkbenchState(client: PythonClient, input: WorkbenchLoadInput = {}): Promise<WorkbenchState> {
  const [health, sessions, artifacts, jobs, modeAtlas, operatorSpecs] = await Promise.all([
    client.health(),
    client.sessions(),
    client.artifacts(),
    client.jobs(),
    client.colabModes(),
    client.operatorSpecs(),
  ]);

  return shapeWorkbenchState(
    {
      health,
      sessions,
      artifacts,
      jobs,
      modeAtlas,
      operatorSpecs,
    },
    input,
  );
}

export function shapeWorkbenchState(snapshot: WorkbenchSnapshot, input: WorkbenchLoadInput = {}): WorkbenchState {
  const sessions = sortNewestBy(snapshot.sessions, (session) => session.updated_at);
  const artifacts = sortNewestBy(snapshot.artifacts, (artifact) => artifact.created_at);
  const jobs = sortNewestBy(snapshot.jobs, (job) => job.created_at);
  const activeSession = findActiveSession(sessions, input.sessionId ?? null);
  const activeSessionId = activeSession?.session_id ?? input.sessionId ?? null;
  const fallbackStartedAt = input.sessionStartedAt ?? activeSession?.created_at ?? null;
  const sessionArtifacts = activeSessionId
    ? artifacts.filter((artifact) => artifact.session_id === activeSessionId)
    : fallbackStartedAt
      ? artifacts.filter((artifact) => createdAfter(artifact.created_at, fallbackStartedAt))
      : artifacts;
  const archiveArtifacts = activeSessionId
    ? artifacts.filter((artifact) => artifact.session_id !== activeSessionId)
    : fallbackStartedAt
      ? artifacts.filter((artifact) => !createdAfter(artifact.created_at, fallbackStartedAt))
      : [];
  const sessionJobs = activeSessionId
    ? jobs.filter((job) => job.recipe.session_id === activeSessionId)
    : fallbackStartedAt
      ? jobs.filter((job) => createdAfter(job.created_at, fallbackStartedAt))
      : jobs;
  const archiveJobs = activeSessionId
    ? jobs.filter((job) => job.recipe.session_id !== activeSessionId)
    : fallbackStartedAt
      ? jobs.filter((job) => !createdAfter(job.created_at, fallbackStartedAt))
      : [];
  const runningJobs = jobs.filter(isJobActive);
  const selectedArtifact =
    artifacts.find((artifact) => artifact.artifact_id === input.selectedArtifactId) ?? sessionArtifacts[0] ?? null;
  const readyBackends = snapshot.health.backends.filter((backend) => backend.available).map((backend) => backend.backend);
  const offlineBackends = snapshot.health.backends.filter((backend) => !backend.available).map((backend) => backend.backend);

  return {
    ...snapshot,
    sessions,
    artifacts,
    jobs,
    activeSession,
    activeSessionId,
    selectedArtifact,
    sessionArtifacts,
    archiveArtifacts,
    sessionJobs,
    archiveJobs,
    runningJobs,
    latestJob: jobs[0] ?? null,
    counts: {
      audioArtifacts: artifacts.filter((artifact) => artifact.kind === "audio").length,
      latentArtifacts: artifacts.filter((artifact) => artifact.kind === "latent").length,
      bundleArtifacts: artifacts.filter((artifact) => artifact.kind === "bundle").length,
      activeJobs: runningJobs.length,
      scaffoldModes: snapshot.modeAtlas.filter((mode) => mode.status.includes("scaffold")).length,
    },
    readiness: {
      readyBackends,
      offlineBackends,
      hasMediumMlx: snapshot.health.backends.some((backend) => backend.backend === "mlx" && backend.available),
      hasTorch: snapshot.health.backends.some((backend) => backend.backend === "torch_mps" && backend.available),
    },
  };
}

function findActiveSession(sessions: SessionRecord[], sessionId: string | null): SessionRecord | null {
  if (sessionId) {
    return sessions.find((session) => session.session_id === sessionId) ?? null;
  }
  return sessions.find((session) => session.status === "active") ?? null;
}

function isJobActive(job: JobRecord): boolean {
  return job.status === "queued" || job.status === "running";
}

function createdAfter(createdAt: string, start: string): boolean {
  return new Date(createdAt).getTime() >= new Date(start).getTime();
}

function sortNewestBy<T>(items: T[], getTimestamp: (item: T) => string): T[] {
  return [...items].sort((a, b) => new Date(getTimestamp(b)).getTime() - new Date(getTimestamp(a)).getTime());
}
