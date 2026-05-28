import type { JobRecord, OperatorName } from "./types";

export function isJobActive(job: JobRecord) {
  return job.status === "queued" || job.status === "running";
}

export function progressPercent(job: JobRecord) {
  return Math.max(0, Math.min(100, Math.round((job.progress ?? 0) * 100)));
}

export function shortOperatorName(operator: OperatorName) {
  return operator.replace(/^experiment\./, "").replace(/^generate\./, "").replace(/^latent\./, "").replaceAll("_", " ");
}

export function formatJobElapsed(job: JobRecord) {
  const start = Date.parse(job.started_at ?? job.created_at);
  const end = job.finished_at ? Date.parse(job.finished_at) : Date.now();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return job.status;
  const seconds = Math.max(0, Math.round((end - start) / 1000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
}
