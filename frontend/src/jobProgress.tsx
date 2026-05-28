import { CircleAlert, Gauge, LoaderCircle, Repeat, X } from "lucide-react";

import { formatJobElapsed, isJobActive, progressPercent, shortOperatorName } from "./jobUtils";
import type { JobRecord } from "./types";

export interface JobActionHandlers {
  onCancelJob?: (job: JobRecord) => void;
  onRetryJob?: (job: JobRecord) => void;
}

export function JobProgress({
  job,
  compact = false,
  onCancelJob,
  onRetryJob,
}: {
  job: JobRecord;
  compact?: boolean;
} & JobActionHandlers) {
  const logLines = job.logs.slice(-12);
  const canCancel = isJobActive(job) && Boolean(onCancelJob);
  const canRetry = (job.status === "failed" || job.status === "cancelled") && Boolean(onRetryJob);
  return (
    <div className={`job-progress ${job.status} ${compact ? "compact" : ""}`}>
      <div className="job-progress-main">
        <span>{isJobActive(job) ? <LoaderCircle className="spin" size={15} /> : <Gauge size={15} />}</span>
        <strong>{shortOperatorName(job.recipe.operator)}</strong>
        <em>{progressPercent(job)}%</em>
      </div>
      <ProgressTrack job={job} />
      <div className="job-progress-meta">
        <span>{job.message ?? job.status}</span>
        <span>{formatJobElapsed(job)}</span>
      </div>
      {canCancel || canRetry ? (
        <div className="job-actions">
          {canCancel ? (
            <button type="button" onClick={() => onCancelJob?.(job)} title="Cancel job">
              <X size={13} />
              Cancel
            </button>
          ) : null}
          {canRetry ? (
            <button type="button" onClick={() => onRetryJob?.(job)} title="Retry job">
              <Repeat size={13} />
              Retry
            </button>
          ) : null}
        </div>
      ) : null}
      {!compact && (job.error || logLines.length) ? (
        <details className="job-log-drawer">
          <summary>
            <CircleAlert size={14} />
            {job.error ? "Error details" : `${logLines.length} log lines`}
          </summary>
          {job.error ? <strong>{job.error}</strong> : null}
          {logLines.length ? <pre>{logLines.join("\n")}</pre> : null}
        </details>
      ) : null}
    </div>
  );
}

function ProgressTrack({ job }: { job: JobRecord }) {
  return (
    <div className="progress-track" aria-label={`${shortOperatorName(job.recipe.operator)} ${progressPercent(job)} percent`}>
      <span style={{ width: `${progressPercent(job)}%` }} />
    </div>
  );
}
