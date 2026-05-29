import type { ArtifactRecord, JobRecord } from "./types";

export function artifactName(artifact: ArtifactRecord) {
  return artifact.label || artifact.prompt || artifact.file?.filename || artifact.artifact_id;
}

export function artifactMeta(artifact: ArtifactRecord) {
  if (artifact.kind === "audio" && artifact.audio) {
    return `${formatDuration(artifact.audio.duration_seconds)} · ${artifact.audio.sample_rate} Hz`;
  }
  if (artifact.kind === "latent" && artifact.latent) {
    return `${artifact.latent.shape.join(" x ")} · ${artifact.latent.latent_rate.toFixed(2)} Hz`;
  }
  if (artifact.kind === "bundle" && artifact.file) {
    return `${formatBytes(artifact.file.byte_size)} bundle`;
  }
  return artifact.kind;
}

export function artifactShape(artifact: ArtifactRecord) {
  if (artifact.kind === "latent") return artifact.latent?.shape.join(" x ") ?? "latent";
  if (artifact.kind === "audio") return `${artifact.audio?.channels ?? 0} ch`;
  if (artifact.kind === "bundle") return artifact.file ? formatBytes(artifact.file.byte_size) : "bundle";
  return artifact.kind;
}

export function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds)) return "0s";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

export function formatPlaybackTime(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${rest}`;
}

export function formatFamilyStamp(value: string) {
  const timestamp = Date.parse(value);
  if (!Number.isFinite(timestamp) || timestamp <= 0) return "recent";
  return new Date(timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function sortNewest(artifacts: readonly ArtifactRecord[]) {
  return [...artifacts].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}

export function sortNewestJobs(jobs: readonly JobRecord[]) {
  return [...jobs].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at));
}
