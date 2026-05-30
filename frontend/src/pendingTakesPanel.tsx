import { LoaderCircle, Repeat, X } from "lucide-react";

import type { PendingTake } from "./pendingTakeModel";
import type { JobRecord } from "./types";

interface PendingTakesPanelProps {
  takes: readonly PendingTake[];
  onCancelJob?: (job: JobRecord) => void;
  onRetryJob?: (job: JobRecord) => void;
}

export function PendingTakesPanel({ takes, onCancelJob, onRetryJob }: PendingTakesPanelProps) {
  if (!takes.length) return null;
  const activeCount = takes.filter((take) => take.status === "queued" || take.status === "running").length;
  return (
    <section className="pending-takes-panel" aria-label="Pending takes">
      <div className="pending-takes-head">
        <span className="eyebrow">Pending Takes</span>
        <strong>{activeCount ? `${activeCount} gesture${activeCount === 1 ? "" : "s"} in progress` : "needs attention"}</strong>
      </div>
      <div className="pending-takes-list">
        {takes.map((take) => (
          <article key={take.id} className={`pending-take ${take.status}`}>
            <div className="pending-take-main">
              <LoaderCircle className={take.status === "running" || take.status === "queued" ? "spin" : ""} size={16} />
              <div>
                <strong>{take.phrase}</strong>
                <small>{take.gestureLabel} · {take.phase} · {take.elapsed}</small>
              </div>
              <em>{take.progressPercent}%</em>
            </div>
            <div className="progress-track" aria-label={`${take.phrase} ${take.progressPercent} percent`}>
              <span style={{ width: `${take.progressPercent}%` }} />
            </div>
            <p>{take.detail}</p>
            <details className="inspect-mini">
              <summary>Inspect</summary>
              <dl>
                <dt>Job</dt>
                <dd>{take.inspect.jobId}</dd>
                <dt>Gesture</dt>
                <dd>{take.inspect.operator}</dd>
                <dt>Sources</dt>
                <dd>{take.sourceIds.length ? take.sourceIds.join(", ") : "none"}</dd>
              </dl>
            </details>
            {take.canCancel || take.canRetry ? (
              <div className="pending-take-actions">
                {take.canCancel ? (
                  <button type="button" onClick={() => onCancelJob?.(take.job)}>
                    <X size={13} />
                    Cancel
                  </button>
                ) : null}
                {take.canRetry ? (
                  <button type="button" onClick={() => onRetryJob?.(take.job)}>
                    <Repeat size={13} />
                    Retry
                  </button>
                ) : null}
              </div>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  );
}
