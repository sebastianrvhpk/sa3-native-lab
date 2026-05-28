import type {
  ArtifactInspection,
  ArtifactKind,
  ArtifactRecord,
  HealthResponse,
  JobJournalEvent,
  JobRecord,
  NotebookMode,
  OperatorSpec,
  RecipeForkPayload,
  ReadinessResponse,
  SessionRecord,
} from "./contracts.js";

export interface PythonClientOptions {
  baseUrl: string;
  fetchImpl?: typeof fetch;
}

export interface ArtifactListOptions {
  kind?: ArtifactKind;
  sessionId?: string | null;
  query?: string | null;
  tags?: string[] | null;
}

export interface ArtifactAnnotationPayload {
  label?: string | null;
  notes?: string | null;
  tags?: string[] | null;
  metadata?: Record<string, unknown> | null;
}

export class PythonApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: unknown,
  ) {
    super(message);
  }
}

export function createPythonClient({ baseUrl, fetchImpl = fetch }: PythonClientOptions) {
  const base = baseUrl.replace(/\/$/, "");

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetchImpl(`${base}${path}`, init);
    if (!response.ok) {
      let payload: unknown;
      try {
        payload = await response.json();
      } catch {
        payload = await response.text();
      }
      throw new PythonApiError(`Python API ${response.status}: ${path}`, response.status, payload);
    }
    return (await response.json()) as T;
  }

  return {
    health: () => request<HealthResponse>("/health"),
    readiness: () => request<ReadinessResponse>("/readiness"),
    operatorSpecs: () => request<OperatorSpec[]>("/operators/specs"),
    sessions: () => request<SessionRecord[]>("/sessions"),
    colabModes: () => request<NotebookMode[]>("/colab/modes"),
    artifacts: (options: ArtifactListOptions = {}) => {
      const params = new URLSearchParams();
      if (options.kind) params.set("kind", options.kind);
      if (options.sessionId) params.set("session_id", options.sessionId);
      if (options.query?.trim()) params.set("q", options.query.trim());
      if (options.tags?.length) params.set("tags", options.tags.join(","));
      const query = params.toString();
      return request<ArtifactRecord[]>(query ? `/artifacts?${query}` : "/artifacts");
    },
    inspectArtifact: (artifactId: string) => request<ArtifactInspection>(`/artifacts/${encodeURIComponent(artifactId)}/inspect`),
    jobs: () => request<JobRecord[]>("/jobs"),
    job: (jobId: string) => request<JobRecord>(`/jobs/${encodeURIComponent(jobId)}`),
    jobEventHistory: (jobId: string, after = 0, limit = 100) => {
      const params = new URLSearchParams({ after: String(after), limit: String(limit) });
      return request<JobJournalEvent[]>(`/jobs/${encodeURIComponent(jobId)}/events/history?${params.toString()}`);
    },
    cancelJob: (jobId: string) => request<JobRecord>(`/jobs/${encodeURIComponent(jobId)}/cancel`, jsonPatchOrPost("POST", {})),
    retryJob: (jobId: string) => request<JobRecord>(`/jobs/${encodeURIComponent(jobId)}/retry`, jsonPatchOrPost("POST", {})),
    replayRecipe: (recipeId: string) => request<JobRecord>(`/recipes/${encodeURIComponent(recipeId)}/replay`, jsonPatchOrPost("POST", {})),
    forkRecipe: (recipeId: string, payload: RecipeForkPayload) =>
      request<JobRecord>(`/recipes/${encodeURIComponent(recipeId)}/fork`, jsonPatchOrPost("POST", payload)),
    annotateArtifact: (artifactId: string, payload: ArtifactAnnotationPayload) =>
      request<ArtifactRecord>(`/artifacts/${encodeURIComponent(artifactId)}/annotate`, jsonPatchOrPost("POST", payload)),
  };
}

function jsonPatchOrPost(method: "PATCH" | "POST", payload: unknown): RequestInit {
  return {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}

export type PythonClient = ReturnType<typeof createPythonClient>;
