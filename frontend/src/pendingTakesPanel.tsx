import { LoaderCircle, Repeat, X } from "lucide-react";

import { nextActionsForPendingTake } from "./nextActionModel";
import type { PendingTake } from "./pendingTakeModel";
import { safeRuntimeLogLines, sanitizeRuntimeText } from "./runtimeTrustModel";
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
          <PendingTakeCard key={take.id} take={take} onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
        ))}
      </div>
    </section>
  );
}

function PendingTakeCard({
  take,
  onCancelJob,
  onRetryJob,
}: {
  take: PendingTake;
  onCancelJob?: (job: JobRecord) => void;
  onRetryJob?: (job: JobRecord) => void;
}) {
  const logLines = safeRuntimeLogLines(take.job.logs, 4);
  const commandContext = sanitizeRuntimeText(take.job.metrics.command, 500);
  return (
    <article className={`pending-take ${take.status}`}>
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
      <p>{sanitizeRuntimeText(take.detail, 180) || take.detail}</p>
      <div className="pending-take-landing">
        <span>{take.completionPhrase}</span>
        <small>{take.recoverySuggestion ?? take.branchLabel}</small>
      </div>
      <details className="inspect-mini">
        <summary>Inspect</summary>
        <dl>
          <dt>Job</dt>
          <dd>{take.inspect.jobId}</dd>
          <dt>Gesture</dt>
          <dd>{take.inspect.operator}</dd>
          <dt>Backend</dt>
          <dd>{take.job.recipe.backend}</dd>
          <dt>Model</dt>
          <dd>{take.job.recipe.model ?? "not set"}</dd>
          <dt>Sources</dt>
          <dd>{take.sourceIds.length ? take.sourceIds.join(", ") : "none"}</dd>
          <dt>Landing</dt>
          <dd>{take.landingArtifactId ?? "not landed"}</dd>
          <dt>Outputs</dt>
          <dd>{take.producedArtifactIds.length ? take.producedArtifactIds.join(", ") : "none yet"}</dd>
          <dt>Params</dt>
          <dd>{formatInspectParams(take.job.recipe.params)}</dd>
          {commandContext ? (
            <>
              <dt>Command</dt>
              <dd>
                <code>{commandContext}</code>
              </dd>
            </>
          ) : null}
          {logLines.length ? (
            <>
              <dt>Log tail</dt>
              <dd>
                <pre>{logLines.join("\n")}</pre>
              </dd>
            </>
          ) : null}
        </dl>
      </details>
      <div className="pending-next-actions" aria-label={`${take.phrase} next actions`}>
        {nextActionsForPendingTake(take).map((action) => (
          <span key={action.id} title={action.disabledReason ?? action.description}>
            {action.label}
          </span>
        ))}
      </div>
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
  );
}

function formatInspectParams(params: Record<string, unknown>) {
  const entries = Object.entries(params)
    .filter(([key]) => key !== "metadata")
    .slice(0, 6)
    .map(([key, value]) => `${key}: ${formatInspectValue(value)}`);
  return entries.length ? entries.join(" · ") : "none";
}

function formatInspectValue(value: unknown) {
  if (typeof value === "string") return sanitizeRuntimeText(value, 140);
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const preview = value.slice(0, 4).map((item) => (typeof item === "string" ? sanitizeRuntimeText(item, 80) : String(item)));
    return `[${preview.join(", ")}${value.length > 4 ? ", ..." : ""}]`;
  }
  return value === null || value === undefined ? "none" : "set";
}
