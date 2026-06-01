import { Check, CircleAlert, Gauge } from "lucide-react";

import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { sanitizeRuntimeText } from "./runtimeTrustModel";
import type { JobRecord, ModelStatus, ReadinessCheck } from "./types";

export function BackendPills({ backends }: { backends: ModelStatus[] }) {
  return (
    <div className="backend-pills">
      {backends.map((backend) => (
        <span key={backend.backend} className={backend.available ? "ready" : "offline"} title={sanitizeRuntimeText(backend.message, 180) || backend.backend}>
          {backend.available ? <Check size={14} /> : <CircleAlert size={14} />}
          {backend.backend}
        </span>
      ))}
    </div>
  );
}

export function RunMonitor({
  runningJobs,
  latestJob,
  eventing = false,
  onCancelJob,
  onRetryJob,
}: {
  runningJobs: JobRecord[];
  latestJob: JobRecord | null;
  eventing?: boolean;
} & JobActionHandlers) {
  const monitorJobs = runningJobs.length ? runningJobs.slice(0, 3) : latestJob ? [latestJob] : [];
  if (!monitorJobs.length) {
    return (
      <div className="run-monitor idle">
        <div>
          <span className="eyebrow">Take Status</span>
          <strong>Ready</strong>
        </div>
        <span className="monitor-state">idle</span>
      </div>
    );
  }

  const busy = runningJobs.length > 0;
  return (
    <div className={`run-monitor ${busy ? "busy" : "idle"}`}>
      <div className="run-monitor-head">
        <div>
          <span className="eyebrow">Take Status</span>
          <strong>{busy ? `${runningJobs.length} pending take${runningJobs.length === 1 ? "" : "s"}` : "Last take"}</strong>
        </div>
        <span className={`monitor-state ${eventing ? "live" : ""}`}>{busy ? (eventing ? "live progress" : "making") : latestJob?.status ?? "idle"}</span>
      </div>
      <div className="monitor-jobs">
        {monitorJobs.map((job) => (
          <JobProgress key={job.job_id} job={job} compact={monitorJobs.length > 1} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
        ))}
      </div>
    </div>
  );
}

export function InlineJobStatus({ job, onCancelJob, onRetryJob }: { job: JobRecord | null | undefined } & JobActionHandlers) {
  if (!job) return null;
  return (
    <div className="inline-job-status">
      <JobProgress job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
    </div>
  );
}

export function ReadinessPanel({ checks }: { checks: ReadinessCheck[] }) {
  const rows = priorityReadinessChecks(checks);
  const errorCount = checks.filter((check) => check.status === "error").length;
  const warnCount = checks.filter((check) => check.status === "warn").length;
  const state = errorCount ? "error" : warnCount ? "warn" : "ok";
  return (
    <details className={`readiness-panel ${state}`}>
      <summary>
        <span>
          <Gauge size={15} />
          Readiness
        </span>
        <strong>{state}</strong>
      </summary>
      <div className="readiness-list">
        {rows.map((check) => (
          <div key={check.name} className={`readiness-row ${check.status}`}>
            <span>{readinessLabel(check.name)}</span>
            <strong>{check.status}</strong>
            <small title={sanitizeRuntimeText(check.detail ?? check.message, 240)}>{readinessMessage(check)}</small>
          </div>
        ))}
      </div>
    </details>
  );
}

function readinessMessage(check: ReadinessCheck) {
  const message = sanitizeRuntimeText(check.message, 160);
  const detail = sanitizeRuntimeText(check.detail, 160);
  if (!detail || detail === message) return message;
  return `${message} · ${detail}`;
}

function priorityReadinessChecks(checks: ReadinessCheck[]) {
  const priority = ["artifact-root", "hf-auth", "mlx-medium-weights", "same-l-access", "backend:mlx", "backend:torch_mps"];
  const byName = new Map(checks.map((check) => [check.name, check]));
  const selected = priority.map((name) => byName.get(name)).filter((check): check is ReadinessCheck => Boolean(check));
  const urgent = checks.filter((check) => (check.status === "error" || check.status === "warn") && !priority.includes(check.name));
  return [...selected, ...urgent].slice(0, 7);
}

function readinessLabel(name: string) {
  return name
    .replace("backend:", "")
    .replace("hf-auth", "HF auth")
    .replace("mlx-medium-weights", "MLX medium")
    .replace("same-l-access", "SAME-L")
    .replace("artifact-root", "Artifacts");
}
