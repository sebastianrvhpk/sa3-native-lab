import type { ArtifactKind, ArtifactRecord, AudioPeaksResponse, HealthResponse, JobRecord, NotebookMode, OperatorName, OperatorSpec, SessionRecord } from "./types";

export const DEFAULT_API_BASE = import.meta.env.VITE_SA3_API_BASE ?? "http://127.0.0.1:8733";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly payload: unknown,
  ) {
    super(message);
  }
}

export interface GenerateTextPayload {
  prompt: string;
  negative_prompt?: string | null;
  duration_seconds: number;
  steps: number;
  seed?: number | null;
  cfg_scale?: number;
  apg_scale?: number;
  model: "sm-music" | "sm-sfx" | "medium";
  decoder: "same-s" | "same-l";
  backend: "mlx";
  session_id?: string | null;
}

export interface AudioToAudioPayload extends GenerateTextPayload {
  source_artifact_id: string;
  init_noise_level: number;
}

export interface InpaintPayload extends AudioToAudioPayload {
  inpaint_start_seconds: number;
  inpaint_end_seconds: number;
}

export interface LatentEncodePayload {
  source_artifact_id: string;
  model: "same-s" | "same-l";
  backend: "torch_mps" | "torch_cpu";
  chunked: boolean;
  chunk_size?: number;
  overlap?: number;
  prompt?: string | null;
  notes?: string | null;
  session_id?: string | null;
}

export interface LatentDecodePayload {
  source_artifact_id: string;
  model: "same-s" | "same-l";
  backend: "torch_mps" | "torch_cpu";
  chunked: boolean;
  chunk_size?: number;
  overlap?: number;
  notes?: string | null;
  session_id?: string | null;
}

export interface OperatorPayload {
  operator: OperatorName;
  backend: "torch_mps" | "torch_cpu";
  inputs: Record<string, string>;
  params: Record<string, unknown>;
  seed?: number | null;
  session_id?: string | null;
}

export interface ExperimentPayload {
  operator: OperatorName;
  backend: "torch_mps" | "torch_cpu" | "cpu";
  inputs: Record<string, string>;
  params: Record<string, unknown>;
  model?: string | null;
  seed?: number | null;
  notes?: string | null;
  session_id?: string | null;
}

export interface ArtifactAnnotationPayload {
  label?: string | null;
  notes?: string | null;
  tags?: string[] | null;
  metadata?: Record<string, unknown> | null;
}

export function createApi(baseUrl: string) {
  const base = baseUrl.replace(/\/$/, "");

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const response = await fetch(`${base}${path}`, init);
    if (!response.ok) {
      let payload: unknown = null;
      try {
        payload = await response.json();
      } catch {
        payload = await response.text();
      }
      throw new ApiError(`API ${response.status}: ${path}`, response.status, payload);
    }
    return (await response.json()) as T;
  }

  return {
    base,
    artifactFileUrl: (artifactId: string) => `${base}/artifacts/${artifactId}/file`,
    health: () => request<HealthResponse>("/health"),
    operatorSpecs: () => request<OperatorSpec[]>("/operators/specs"),
    sessions: () => request<SessionRecord[]>("/sessions"),
    createSession: (payload: { name?: string | null; notes?: string | null } = {}) =>
      request<SessionRecord>("/sessions", jsonPost(payload)),
    updateSession: (sessionId: string, payload: Partial<Pick<SessionRecord, "name" | "status" | "notes">>) =>
      request<SessionRecord>(`/sessions/${encodeURIComponent(sessionId)}`, { ...jsonPost(payload), method: "PATCH" }),
    colabModes: () => request<NotebookMode[]>("/colab/modes"),
    artifacts: (kind?: ArtifactKind) => request<ArtifactRecord[]>(kind ? `/artifacts?kind=${kind}` : "/artifacts"),
    audioPeaks: (artifactId: string, bins = 96) =>
      request<AudioPeaksResponse>(`/artifacts/${encodeURIComponent(artifactId)}/peaks?bins=${bins}`),
    jobs: () => request<JobRecord[]>("/jobs"),
    job: (jobId: string) => request<JobRecord>(`/jobs/${jobId}`),
    importAudio: async (file: File, label?: string, sessionId?: string | null) => {
      const data = new FormData();
      data.append("file", file);
      if (label) data.append("label", label);
      if (sessionId) data.append("session_id", sessionId);
      return request<ArtifactRecord>("/audio/import", { method: "POST", body: data });
    },
    annotateArtifact: (artifactId: string, payload: ArtifactAnnotationPayload) =>
      request<ArtifactRecord>(`/artifacts/${encodeURIComponent(artifactId)}/annotate`, jsonPost(payload)),
    generateText: (payload: GenerateTextPayload) =>
      request<JobRecord>("/generate/text", jsonPost(payload)),
    generateAudioToAudio: (payload: AudioToAudioPayload) =>
      request<JobRecord>("/generate/audio-to-audio", jsonPost(payload)),
    generateInpaint: (payload: InpaintPayload) =>
      request<JobRecord>("/generate/inpaint", jsonPost(payload)),
    encodeLatent: (payload: LatentEncodePayload) =>
      request<JobRecord>("/latents/encode", jsonPost(payload)),
    decodeLatent: (payload: LatentDecodePayload) =>
      request<JobRecord>("/latents/decode", jsonPost(payload)),
    runOperator: (payload: OperatorPayload) =>
      request<JobRecord>("/operators/run", jsonPost(payload)),
    runExperiment: (payload: ExperimentPayload) =>
      request<JobRecord>("/experiments/run", jsonPost(payload)),
  };
}

function jsonPost(payload: unknown): RequestInit {
  return {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  };
}
