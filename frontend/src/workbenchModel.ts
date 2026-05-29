import { sortNewest, sortNewestJobs } from "./artifactUtils";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord, OperatorName, Recipe } from "./types";

export function createdAfter(createdAt: string, startedAt: string) {
  const created = Date.parse(createdAt);
  const started = Date.parse(startedAt);
  if (!Number.isFinite(created) || !Number.isFinite(started)) return true;
  return created >= started;
}

export function mergeJobRecords(baseJobs: readonly JobRecord[], overlayJobs: readonly JobRecord[]) {
  const byId = new Map(baseJobs.map((job) => [job.job_id, job]));
  for (const job of overlayJobs) {
    byId.set(job.job_id, job);
  }
  return sortNewestJobs([...byId.values()]);
}

export function parseJobEvent(raw: string): JobRecord | null {
  try {
    const payload = JSON.parse(raw) as unknown;
    return jobFromJobEvent(payload);
  } catch {
    return null;
  }
}

export function jobFromJobEvent(payload: unknown): JobRecord | null {
  if (!payload || typeof payload !== "object") return null;
  const event = "data" in payload && payload.data && typeof payload.data === "object" ? payload.data : payload;
  if ("type" in event) {
    return event.type === "snapshot" && "job" in event ? (event.job as JobRecord) : null;
  }
  return "job_id" in event ? (event as JobRecord) : null;
}

export function buildResultFamilies(artifacts: readonly ArtifactRecord[], jobs: readonly JobRecord[]): ResultFamily[] {
  const artifactsByRecipe = groupArtifactsByRecipe(artifacts);
  const families = jobs.reduce<Map<string, { familyId: string; recipe: JobRecord["recipe"]; jobs: JobRecord[]; artifacts: ArtifactRecord[] }>>(
    (items, job) => {
      const recipeId = job.recipe.recipe_id;
      const familyId = resultFamilyIdForRecipe(job.recipe);
      const recipeArtifacts = artifactsByRecipe.get(recipeId) ?? [];
      const family = items.get(familyId);
      if (family) {
        family.jobs.push(job);
        family.artifacts = uniqueArtifacts([...family.artifacts, ...recipeArtifacts]);
      } else {
        items.set(familyId, {
          familyId,
          recipe: job.recipe,
          jobs: [job],
          artifacts: recipeArtifacts,
        });
      }
      return items;
    },
    new Map(),
  );
  return [...families.values()]
    .map((family) => {
      const sortedArtifacts = sortNewest(family.artifacts);
      const artifactIds = unique([
        ...family.jobs.flatMap((job) => job.artifact_ids),
        ...family.artifacts.map((artifact) => artifact.artifact_id),
      ]);
      const timestamps = [
        ...family.jobs.map((job) => job.finished_at ?? job.started_at ?? job.created_at),
        ...family.artifacts.map((artifact) => artifact.created_at),
      ];
      const sortedJobs = sortNewestJobs(family.jobs);
      const recipe = sortedJobs[0]?.recipe ?? family.recipe;
      return {
        familyId: family.familyId,
        recipeId: recipe.recipe_id,
        recipe,
        operator: recipe.operator,
        sessionId: recipe.session_id ?? null,
        status: familyStatus(family.jobs),
        jobIds: family.jobs.map((job) => job.job_id),
        artifactIds,
        artifactKinds: unique(family.artifacts.map((artifact) => artifact.kind)),
        metrics: sortedJobs[0]?.metrics ?? {},
        latestArtifactId: sortedArtifacts[0]?.artifact_id ?? artifactIds[0] ?? null,
        createdAt: oldestTimestamp(timestamps) ?? recipe.created_at,
        updatedAt: newestTimestamp(timestamps) ?? recipe.created_at,
      };
    })
    .sort((a, b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt));
}

export function resultFamilyIdForRecipe(recipe: Recipe) {
  const sourceId = typeof recipe.inputs.source === "string" ? recipe.inputs.source : "";
  const metadata = recordValue(recipe.params.metadata);
  if (recipe.operator === "generate.text_to_audio" && sourceId && metadata?.generation_origin === "prompt_search_candidate") {
    return `prompt-candidates:${sourceId}`;
  }
  return recipe.recipe_id;
}

export function filterFamiliesForWork(families: readonly ResultFamily[], artifacts: readonly ArtifactRecord[], jobs: readonly JobRecord[]) {
  const recipeIds = new Set([
    ...jobs.map((job) => job.recipe.recipe_id),
    ...artifacts.map((artifact) => artifact.recipe_id).filter((recipeId): recipeId is string => Boolean(recipeId)),
  ]);
  return families.filter((family) => recipeIds.has(family.recipeId));
}

export function groupArtifactsByRecipe(artifacts: readonly ArtifactRecord[]) {
  const groups = new Map<string, ArtifactRecord[]>();
  for (const artifact of artifacts) {
    if (!artifact.recipe_id) continue;
    const group = groups.get(artifact.recipe_id) ?? [];
    group.push(artifact);
    groups.set(artifact.recipe_id, group);
  }
  return groups;
}

export function familyStatus(jobs: readonly JobRecord[]): ResultFamily["status"] {
  if (jobs.some((job) => job.status === "running")) return "running";
  if (jobs.some((job) => job.status === "queued")) return "queued";
  if (jobs.some((job) => job.status === "failed")) return "failed";
  if (jobs.length && jobs.every((job) => job.status === "cancelled")) return "cancelled";
  if (jobs.length && jobs.every((job) => job.status === "succeeded")) return "succeeded";
  return "mixed";
}

export function unique<T>(items: readonly T[]) {
  return [...new Set(items)];
}

export function uniqueArtifacts(artifacts: readonly ArtifactRecord[]) {
  const seen = new Set<string>();
  return artifacts.filter((artifact) => {
    if (seen.has(artifact.artifact_id)) return false;
    seen.add(artifact.artifact_id);
    return true;
  });
}

export function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

export function primitiveMetadataValue(value: unknown) {
  return typeof value === "string" || typeof value === "number" || typeof value === "boolean" ? value : null;
}

export function newestTimestamp(timestamps: readonly string[]) {
  return timestamps.reduce<string | null>((latest, item) => (!latest || Date.parse(item) > Date.parse(latest) ? item : latest), null);
}

export function oldestTimestamp(timestamps: readonly string[]) {
  return timestamps.reduce<string | null>((oldest, item) => (!oldest || Date.parse(item) < Date.parse(oldest) ? item : oldest), null);
}

export function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

export function activeJobForOperator(jobs: readonly JobRecord[], operator: OperatorName) {
  return jobs.find((job) => job.recipe.operator === operator) ?? null;
}

export function statusClass(status: string) {
  if (status.includes("native")) return "native";
  if (status.includes("partial") || status.includes("chainable")) return "partial";
  return "scaffold";
}
