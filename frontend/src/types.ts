export type ArtifactKind = "audio" | "latent" | "bundle" | "recipe" | "text";
export type BackendName = "mlx" | "torch_mps" | "torch_cpu" | "cpu";
export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type OperatorName =
  | "generate.text_to_audio"
  | "generate.audio_to_audio"
  | "generate.inpaint"
  | "latent.encode"
  | "latent.decode"
  | "latent.blur"
  | "latent.dsp"
  | "latent.graft"
  | "latent.renoise"
  | "latent.cyclic_roll"
  | "experiment.audio_style_vectors"
  | "experiment.positive_style_profile"
  | "experiment.style_profile.build"
  | "experiment.style_profile.generate"
  | "experiment.style_direction.generate"
  | "experiment.audio_direction.generate"
  | "experiment.sa3_vectors.extract"
  | "experiment.audio_residual_vectors.extract"
  | "experiment.alpha_sweep"
  | "experiment.geometry_audit"
  | "experiment.prompt_search"
  | "experiment.soft_prompt.optimize"
  | "experiment.soft_prompt.generate"
  | "dataset.pre_encode"
  | "training.lora"
  | "memory.query"
  | "artifact.promote_bundle_audio"
  | "artifact.annotate";

export interface AudioMetadata {
  sample_rate: number;
  channels: number;
  frames: number;
  duration_seconds: number;
  format?: string | null;
}

export interface AudioPeaksResponse {
  artifact_id: string;
  bins: number;
  channels: number;
  sample_rate: number;
  duration_seconds: number;
  peaks: number[];
}

export interface AudioDescriptorComparison {
  target_artifact_id: string;
  take_artifact_id: string;
  target: Record<string, number>;
  take: Record<string, number>;
  delta: Record<string, number>;
}

export interface LatentMetadata {
  shape: [number, number];
  latent_rate: number;
  duration_seconds?: number | null;
  sample_rate?: number | null;
  channel_first: boolean;
}

export interface ArtifactRecord {
  artifact_id: string;
  kind: ArtifactKind;
  path: string;
  file?: {
    filename: string;
    media_type?: string | null;
    byte_size: number;
    sha256?: string | null;
  } | null;
  audio?: AudioMetadata | null;
  latent?: LatentMetadata | null;
  source_artifact_ids: string[];
  recipe_id?: string | null;
  label?: string | null;
  prompt?: string | null;
  notes?: string | null;
  tags: string[];
  metadata: Record<string, unknown>;
  session_id?: string | null;
  created_at: string;
}

export interface BundleFileEntry {
  path: string;
  byte_size: number;
  compressed_size?: number | null;
}

export interface BundleAudioEntry {
  path: string;
  byte_size: number;
  media_type?: string | null;
  sample_rate?: number | null;
  channels?: number | null;
  frames?: number | null;
  duration_seconds?: number | null;
  format?: string | null;
}

export interface Recipe {
  recipe_id: string;
  operator: OperatorName;
  backend: BackendName;
  inputs: Record<string, string>;
  params: Record<string, unknown>;
  model?: string | null;
  seed?: number | null;
  notes?: string | null;
  session_id?: string | null;
  created_at: string;
  version: number;
}

export interface ArtifactInspection {
  artifact: ArtifactRecord;
  recipe?: Recipe | null;
  sources: ArtifactRecord[];
  children: ArtifactRecord[];
  bundle_files: BundleFileEntry[];
  bundle_audio_files: BundleAudioEntry[];
  bundle_preview: Record<string, unknown>;
  bundle_summary: Record<string, unknown>;
}

export interface SessionRecord {
  session_id: string;
  name: string;
  status: "active" | "archived";
  notes?: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  archived_at?: string | null;
}

export interface JobRecord {
  job_id: string;
  status: JobStatus;
  recipe: Recipe;
  progress: number;
  message?: string | null;
  artifact_ids: string[];
  metrics: Record<string, unknown>;
  logs: string[];
  error?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export type JobEvent = { type: "snapshot"; job: JobRecord } | { type: "error"; error: string };

export interface ModelStatus {
  backend: BackendName;
  available: boolean;
  loaded: boolean;
  device?: string | null;
  message?: string | null;
  details: Record<string, unknown>;
}

export interface OperatorFieldOption {
  value: string;
  label?: string | null;
}

export interface OperatorFieldSpec {
  key: string;
  label: string;
  type: string;
  default?: unknown;
  required: boolean;
  advanced: boolean;
  min?: number | null;
  max?: number | null;
  step?: number | null;
  options: OperatorFieldOption[];
  artifact_kinds: ArtifactKind[];
  placeholder?: string | null;
  description?: string | null;
}

export interface OperatorSpec {
  name: OperatorName;
  maturity: string;
  backends: BackendName[];
  inputs: string[];
  params: Record<string, unknown>;
  ui_fields: OperatorFieldSpec[];
  produces: ArtifactKind[];
  status: string;
}

export interface HealthResponse {
  app: string;
  version: string;
  artifact_root: string;
  backends: ModelStatus[];
}

export interface ReadinessCheck {
  name: string;
  status: "ok" | "warn" | "error" | string;
  message: string;
  detail?: string | null;
}

export interface ReadinessResponse {
  checks: ReadinessCheck[];
  ok: boolean;
  warnings: number;
  errors: number;
}

export interface NotebookMode {
  mode_id: string;
  title: string;
  priority: string;
  maturity: string;
  status: string;
  native_surface: string;
  operators: OperatorName[];
  scripts: string[];
  inputs: string[];
  outputs: string[];
  notes?: string | null;
}
