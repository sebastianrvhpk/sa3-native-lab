import type { ResultFamily } from "../controlPlane";
import type {
  ArtifactRecord,
  BackendName,
  JobRecord,
  JobStatus,
  OperatorName,
  OperatorSpec,
  Recipe,
  SessionRecord,
} from "../types";

export function testRecipe(overrides: Partial<Recipe> = {}): Recipe {
  return {
    recipe_id: overrides.recipe_id ?? "recipe_take",
    operator: overrides.operator ?? "generate.text_to_audio",
    backend: overrides.backend ?? "mlx",
    inputs: overrides.inputs ?? {},
    params: overrides.params ?? { prompt: "warm smoke", duration_seconds: 1 },
    model: overrides.model ?? "medium",
    seed: overrides.seed ?? 7,
    notes: overrides.notes ?? null,
    session_id: overrides.session_id ?? "sess_1",
    created_at: overrides.created_at ?? "2026-05-28T15:00:00.000Z",
    version: overrides.version ?? 1,
  };
}

export function testArtifact(overrides: Partial<ArtifactRecord> = {}): ArtifactRecord {
  const kind = overrides.kind ?? "audio";
  return {
    artifact_id: overrides.artifact_id ?? "art_take",
    kind,
    path: overrides.path ?? `/tmp/${overrides.artifact_id ?? "art_take"}.${kind === "audio" ? "wav" : "npy"}`,
    file: overrides.file ?? { filename: `${overrides.label ?? overrides.artifact_id ?? "take"}.${kind === "audio" ? "wav" : "npy"}`, media_type: kind === "audio" ? "audio/wav" : "application/octet-stream", byte_size: 1024 },
    audio: overrides.audio ?? (kind === "audio" ? { sample_rate: 24000, channels: 1, frames: 24000, duration_seconds: 1, format: "WAV" } : null),
    latent: overrides.latent ?? (kind === "latent" ? { shape: [8, 4], latent_rate: 43.07, duration_seconds: 1, channel_first: true } : null),
    source_artifact_ids: overrides.source_artifact_ids ?? [],
    recipe_id: "recipe_id" in overrides ? overrides.recipe_id : "recipe_take",
    label: "label" in overrides ? overrides.label : "Warm Smoke Take",
    prompt: "prompt" in overrides ? overrides.prompt : "warm smoke percussion",
    notes: overrides.notes ?? null,
    tags: overrides.tags ?? [],
    metadata: overrides.metadata ?? {},
    session_id: "session_id" in overrides ? overrides.session_id : "sess_1",
    created_at: overrides.created_at ?? "2026-05-28T15:00:00.000Z",
  };
}

export function testJob(
  overrides: Partial<JobRecord> & { recipe?: Recipe; status?: JobStatus } = {},
): JobRecord {
  const recipe = overrides.recipe ?? testRecipe();
  return {
    job_id: overrides.job_id ?? "job_take",
    status: overrides.status ?? "succeeded",
    recipe,
    progress: overrides.progress ?? (overrides.status === "running" ? 0.42 : 1),
    phase: overrides.phase ?? null,
    message: overrides.message ?? null,
    artifact_ids: overrides.artifact_ids ?? ["art_take"],
    metrics: overrides.metrics ?? {},
    logs: overrides.logs ?? [],
    error: overrides.error ?? null,
    created_at: overrides.created_at ?? "2026-05-28T15:00:00.000Z",
    started_at: overrides.started_at ?? "2026-05-28T15:00:00.000Z",
    finished_at: overrides.finished_at ?? (overrides.status === "running" ? null : "2026-05-28T15:00:01.000Z"),
  };
}

export function testFamily(overrides: Partial<ResultFamily> & { recipe?: Recipe } = {}): ResultFamily {
  const recipe = overrides.recipe ?? testRecipe();
  return {
    familyId: overrides.familyId ?? recipe.recipe_id,
    recipeId: overrides.recipeId ?? recipe.recipe_id,
    recipe,
    operator: overrides.operator ?? recipe.operator,
    sessionId: overrides.sessionId ?? recipe.session_id ?? null,
    status: overrides.status ?? "succeeded",
    jobIds: overrides.jobIds ?? ["job_take"],
    artifactIds: overrides.artifactIds ?? ["art_take"],
    artifactKinds: overrides.artifactKinds ?? ["audio"],
    metrics: overrides.metrics ?? {},
    latestArtifactId: overrides.latestArtifactId ?? "art_take",
    createdAt: overrides.createdAt ?? recipe.created_at,
    updatedAt: overrides.updatedAt ?? recipe.created_at,
  };
}

export function testSession(overrides: Partial<SessionRecord> = {}): SessionRecord {
  return {
    session_id: overrides.session_id ?? "sess_1",
    name: overrides.name ?? "Smoke Session",
    status: overrides.status ?? "active",
    notes: overrides.notes ?? null,
    metadata: overrides.metadata ?? {},
    created_at: overrides.created_at ?? "2026-05-28T15:00:00.000Z",
    updated_at: overrides.updated_at ?? "2026-05-28T15:00:00.000Z",
    archived_at: overrides.archived_at ?? null,
  };
}

export function testOperatorSpec(overrides: Partial<OperatorSpec> = {}): OperatorSpec {
  return {
    name: overrides.name ?? "generate.text_to_audio",
    maturity: overrides.maturity ?? "native",
    backends: overrides.backends ?? ["mlx"],
    inputs: overrides.inputs ?? [],
    params: overrides.params ?? { prompt: "warm smoke", duration_seconds: 1, seed: 7 },
    ui_fields: overrides.ui_fields ?? [],
    produces: overrides.produces ?? ["audio"],
    status: overrides.status ?? "implemented",
  };
}

export function testBackend(backend: BackendName, available = true) {
  return {
    backend,
    available,
    loaded: false,
    device: available ? backend : null,
    message: available ? null : `${backend} offline`,
    details: {},
  };
}

export function testReadiness(name: string, status: "ok" | "warn" | "error", message = name) {
  return { name, status, message, detail: `${name} detail` };
}

export function operatorName(value: string): OperatorName {
  return value as OperatorName;
}
