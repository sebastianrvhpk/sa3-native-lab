import { CircleAlert, Gauge, LoaderCircle, Repeat, X } from "lucide-react";

import { formatJobElapsed, isJobActive, progressPercent, shortOperatorName } from "./jobUtils";
import type { JobRecord } from "./types";

export interface JobActionHandlers {
  onCancelJob?: (job: JobRecord) => void;
  onRetryJob?: (job: JobRecord) => void;
}

export interface JobRecoveryHint {
  title: string;
  detail: string;
}

export interface JobPhase {
  label: string;
  tone: "queued" | "running" | "io" | "model" | "done" | "failed" | "cancelled";
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
  const hints = jobRecoveryHints(job);
  const canCancel = isJobActive(job) && Boolean(onCancelJob);
  const canRetry = (job.status === "failed" || job.status === "cancelled") && Boolean(onRetryJob);
  const statusMessage = job.message ?? latestUsefulLog(job) ?? job.status;
  const phase = jobPhase(job);
  return (
    <div className={`job-progress ${job.status} ${compact ? "compact" : ""}`}>
      <div className="job-progress-main">
        <span>{isJobActive(job) ? <LoaderCircle className="spin" size={15} /> : <Gauge size={15} />}</span>
        <strong>{shortOperatorName(job.recipe.operator)}</strong>
        <span className={`job-phase ${phase.tone}`}>{phase.label}</span>
        <em>{progressPercent(job)}%</em>
      </div>
      <ProgressTrack job={job} />
      <div className="job-progress-meta">
        <span>{statusMessage}</span>
        <span>{artifactCountLabel(job)}</span>
        <span>{formatJobElapsed(job)}</span>
      </div>
      {compact && hints.length ? <RecoveryHints hints={hints.slice(0, 1)} /> : null}
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
          {hints.length ? <RecoveryHints hints={hints} /> : null}
          {logLines.length ? <pre>{logLines.join("\n")}</pre> : null}
        </details>
      ) : null}
    </div>
  );
}

export function jobPhase(job: JobRecord): JobPhase {
  if (job.status === "queued") return { label: "queued", tone: "queued" };
  if (job.status === "succeeded") return { label: "done", tone: "done" };
  if (job.status === "failed") return { label: "failed", tone: "failed" };
  if (job.status === "cancelled") return { label: "cancelled", tone: "cancelled" };
  const contractedPhase = job.phase ? phaseFromContract(job.phase) : null;
  if (contractedPhase) return contractedPhase;

  const text = [job.message, latestUsefulLog(job), ...job.logs.slice(-4)].filter(Boolean).join("\n").toLowerCase();
  if (/(sample|sampling|generate|generating|denoise|step|score|scoring|candidate)/.test(text)) {
    return { label: "generating", tone: "model" };
  }
  if (/(encode|embedding|latent|pre-encode|pre encode)/.test(text)) {
    return { label: "encoding", tone: "model" };
  }
  if (/(decode|vocoder|same)/.test(text)) {
    return { label: "decoding", tone: "model" };
  }
  if (/(download|weight|checkpoint|loading|load model|tokenizer|hugging ?face)/.test(text)) {
    return { label: "loading", tone: "model" };
  }
  if (/(save|saving|write|writing|bundle|artifact|export)/.test(text)) {
    return { label: "saving", tone: "io" };
  }
  return { label: "running", tone: "running" };
}

function phaseFromContract(phase: string): JobPhase | null {
  const normalized = phase.replace(/[_-]+/g, " ").trim().toLowerCase();
  if (!normalized || normalized === "running") return null;
  if (normalized === "preflight") return { label: "preflight", tone: "running" };
  if (normalized === "subprocess") return { label: "subprocess", tone: "running" };
  if (normalized === "io" || normalized === "saving" || normalized === "indexing") return { label: normalized, tone: "io" };
  if (normalized === "done") return { label: "done", tone: "done" };
  if (normalized === "failed") return { label: "failed", tone: "failed" };
  if (normalized === "cancelled") return { label: "cancelled", tone: "cancelled" };
  if (/(model|setup|sample|sampling|score|scoring|encode|encoding|decode|decoding|transform|analyzing)/.test(normalized)) {
    return { label: normalized, tone: "model" };
  }
  return { label: normalized.slice(0, 18), tone: "running" };
}

export function jobRecoveryHints(job: JobRecord): JobRecoveryHint[] {
  if (job.status === "cancelled") {
    return [{ title: "Cancelled", detail: "The recipe is preserved; retry it when the current inputs are ready." }];
  }
  if (job.status !== "failed") return [];

  const text = [job.error, job.message, ...job.logs].filter(Boolean).join("\n").toLowerCase();
  const hints: JobRecoveryHint[] = [];
  if (/(hugging ?face|hf_|401|403|unauthorized|forbidden|gated|token)/.test(text)) {
    hints.push({ title: "Auth or gated weights", detail: "Check the Hugging Face token and model license access, then retry." });
  }
  if (/(mlx backend|optimized\/mlx|mlx\/\.venv|sa3 wrapper|apple silicon)/.test(text)) {
    hints.push({ title: "MLX setup", detail: "Run the local install or doctor flow so the Apple Silicon wrapper and venv are available." });
  }
  if (/(no such file|filenotfound|not found|missing required parameter|did not create output|path)/.test(text)) {
    hints.push({ title: "Input or output path", detail: "Verify the selected artifact/path exists and that required bundle/audio fields are filled." });
  }
  if (/(out of memory|oom|mps|metal|allocation)/.test(text)) {
    hints.push({ title: "Memory pressure", detail: "Try shorter duration, fewer steps, smaller batch size, or a CPU fallback for non-generation jobs." });
  }
  if (/(disk space|no space|hf-cache-space|hugging face cache|background writer channel closed|writer error)/.test(text)) {
    hints.push({ title: "Disk space", detail: "Free space in the Hugging Face cache or set HF_HOME to a larger volume before retrying." });
  }
  if (/exit code/.test(text)) {
    hints.push({ title: "Subprocess failed", detail: "Open the log tail for the script output, adjust the recipe, then retry from the preserved job." });
  }
  if (!hints.length) {
    hints.push({ title: "Review log tail", detail: "The recipe and parameters were saved; inspect the final log lines, then retry after correcting inputs." });
  }
  return dedupeHints(hints);
}

function artifactCountLabel(job: JobRecord) {
  if (!job.artifact_ids.length) return "no artifacts yet";
  return `${job.artifact_ids.length} artifact${job.artifact_ids.length === 1 ? "" : "s"}`;
}

function ProgressTrack({ job }: { job: JobRecord }) {
  return (
    <div className="progress-track" aria-label={`${shortOperatorName(job.recipe.operator)} ${progressPercent(job)} percent`}>
      <span style={{ width: `${progressPercent(job)}%` }} />
    </div>
  );
}

function RecoveryHints({ hints }: { hints: readonly JobRecoveryHint[] }) {
  return (
    <div className="job-recovery-hints" aria-label="Recovery hints">
      {hints.map((hint) => (
        <span key={`${hint.title}:${hint.detail}`} title={hint.detail}>
          {hint.title}
        </span>
      ))}
    </div>
  );
}

function latestUsefulLog(job: JobRecord) {
  const latest = [...job.logs].reverse().find((line) => line.trim() && !line.includes("[heartbeat]"));
  return latest?.trim().slice(0, 180);
}

function dedupeHints(hints: JobRecoveryHint[]) {
  const seen = new Set<string>();
  return hints.filter((hint) => {
    if (seen.has(hint.title)) return false;
    seen.add(hint.title);
    return true;
  });
}
