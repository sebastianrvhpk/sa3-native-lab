import { z } from "zod";

import type {
  ArtifactKind,
  ArtifactRecord,
  HealthResponse,
  JobRecord,
  JobStatus,
  NotebookMode,
  OperatorSpec,
  Recipe,
  SessionRecord,
} from "./contracts.js";
import type { PythonClient } from "./pythonClient.js";

export const workbenchLoadInputSchema = z.object({
  apiBase: z.string().url().optional(),
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
  resultFamilies: ResultFamily[];
  sessionResultFamilies: ResultFamily[];
  archiveResultFamilies: ResultFamily[];
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

export interface ResultFamily {
  familyId: string;
  recipeId: string;
  recipe: Recipe;
  operator: Recipe["operator"];
  sessionId: string | null;
  status: JobStatus | "mixed";
  jobIds: string[];
  artifactIds: string[];
  artifactKinds: ArtifactKind[];
  latestArtifactId: string | null;
  createdAt: string;
  updatedAt: string;
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
  const resultFamilies = buildResultFamilies(artifacts, jobs);
  const sessionRecipeIds = new Set([
    ...sessionJobs.map((job) => job.recipe.recipe_id),
    ...sessionArtifacts.map((artifact) => artifact.recipe_id).filter((recipeId): recipeId is string => Boolean(recipeId)),
  ]);
  const archiveRecipeIds = new Set([
    ...archiveJobs.map((job) => job.recipe.recipe_id),
    ...archiveArtifacts.map((artifact) => artifact.recipe_id).filter((recipeId): recipeId is string => Boolean(recipeId)),
  ]);
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
    resultFamilies,
    sessionResultFamilies: resultFamilies.filter((family) => sessionRecipeIds.has(family.recipeId)),
    archiveResultFamilies: resultFamilies.filter((family) => archiveRecipeIds.has(family.recipeId)),
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

export function buildResultFamilies(artifacts: ArtifactRecord[], jobs: JobRecord[]): ResultFamily[] {
  const artifactsByRecipe = groupArtifactsByRecipe(artifacts);
  const families = new Map<string, { recipe: Recipe; jobs: JobRecord[]; artifacts: ArtifactRecord[] }>();

  for (const job of jobs) {
    const recipeId = job.recipe.recipe_id;
    const family = families.get(recipeId);
    if (family) {
      family.jobs.push(job);
    } else {
      families.set(recipeId, {
        recipe: job.recipe,
        jobs: [job],
        artifacts: artifactsByRecipe.get(recipeId) ?? [],
      });
    }
  }

  return [...families.entries()]
    .map(([recipeId, family]) => {
      const artifactIds = unique([
        ...family.jobs.flatMap((job) => job.artifact_ids),
        ...family.artifacts.map((artifact) => artifact.artifact_id),
      ]);
      const artifactKinds = unique(family.artifacts.map((artifact) => artifact.kind));
      const timestamps = [
        ...family.jobs.map((job) => job.finished_at ?? job.started_at ?? job.created_at),
        ...family.artifacts.map((artifact) => artifact.created_at),
      ];
      const sortedArtifacts = sortNewestBy(family.artifacts, (artifact) => artifact.created_at);
      return {
        familyId: recipeId,
        recipeId,
        recipe: family.recipe,
        operator: family.recipe.operator,
        sessionId: family.recipe.session_id ?? null,
        status: familyStatus(family.jobs),
        jobIds: family.jobs.map((job) => job.job_id),
        artifactIds,
        artifactKinds,
        latestArtifactId: sortedArtifacts[0]?.artifact_id ?? artifactIds[0] ?? null,
        createdAt: oldestTimestamp(timestamps) ?? family.recipe.created_at,
        updatedAt: newestTimestamp(timestamps) ?? family.recipe.created_at,
      } satisfies ResultFamily;
    })
    .sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
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

function familyStatus(jobs: JobRecord[]): ResultFamily["status"] {
  if (!jobs.length) return "mixed";
  if (jobs.some((job) => job.status === "running")) return "running";
  if (jobs.some((job) => job.status === "queued")) return "queued";
  if (jobs.some((job) => job.status === "failed")) return "failed";
  if (jobs.every((job) => job.status === "cancelled")) return "cancelled";
  if (jobs.every((job) => job.status === "succeeded")) return "succeeded";
  return "mixed";
}

function groupArtifactsByRecipe(artifacts: ArtifactRecord[]): Map<string, ArtifactRecord[]> {
  const groups = new Map<string, ArtifactRecord[]>();
  for (const artifact of artifacts) {
    if (!artifact.recipe_id) continue;
    const group = groups.get(artifact.recipe_id) ?? [];
    group.push(artifact);
    groups.set(artifact.recipe_id, group);
  }
  return groups;
}

function unique<T>(items: T[]): T[] {
  return [...new Set(items)];
}

function newestTimestamp(timestamps: string[]): string | null {
  return timestamps.reduce<string | null>((latest, item) => {
    if (!latest || new Date(item).getTime() > new Date(latest).getTime()) return item;
    return latest;
  }, null);
}

function oldestTimestamp(timestamps: string[]): string | null {
  return timestamps.reduce<string | null>((oldest, item) => {
    if (!oldest || new Date(item).getTime() < new Date(oldest).getTime()) return item;
    return oldest;
  }, null);
}

function createdAfter(createdAt: string, start: string): boolean {
  return new Date(createdAt).getTime() >= new Date(start).getTime();
}

function sortNewestBy<T>(items: T[], getTimestamp: (item: T) => string): T[] {
  return [...items].sort((a, b) => new Date(getTimestamp(b)).getTime() - new Date(getTimestamp(a)).getTime());
}
