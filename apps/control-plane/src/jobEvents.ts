import type { JobRecord } from "./contracts.js";
import type { PythonClient } from "./pythonClient.js";

export type ControlPlaneJobEvent =
  | {
      type: "snapshot";
      source: "control-plane";
      sequence: number;
      receivedAt: string;
      diagnostics: JobEventDiagnostics;
      job: JobRecord;
    }
  | {
      type: "heartbeat";
      source: "control-plane";
      sequence: number;
      receivedAt: string;
      diagnostics: JobEventDiagnostics & {
        unchangedPolls: number;
        status: JobRecord["status"];
        progress: number;
        message?: string | null;
      };
      jobId: string;
    }
  | {
      type: "error";
      source: "control-plane";
      sequence: number;
      receivedAt: string;
      diagnostics: JobEventDiagnostics;
      error: string;
    };

export interface PollJobEventsOptions {
  jobId: string;
  intervalMs?: number;
  heartbeatEveryMs?: number;
  lastEventId?: string | null;
  signal?: AbortSignal;
}

export interface JobEventDiagnostics {
  eventSource: "python-job-snapshot-poll";
  pollIntervalMs: number;
  logTail: string[];
  lastEventId?: string | null;
  resumedFromSequence?: number | null;
}

export async function* pollJobEvents(
  client: Pick<PythonClient, "job">,
  { jobId, intervalMs = 1000, heartbeatEveryMs = 10000, lastEventId = null, signal }: PollJobEventsOptions,
): AsyncGenerator<ControlPlaneJobEvent> {
  let lastSignature = "";
  let sequence = parseTrackedSequence(jobId, lastEventId) ?? 0;
  const resumedFromSequence = sequence > 0 ? sequence : null;
  let lastHeartbeatAt = Date.now();
  let unchangedPolls = 0;

  while (!signal?.aborted) {
    try {
      const job = await client.job(jobId);
      const signature = jobSnapshotSignature(job);
      const diagnostics = jobEventDiagnostics(job, { intervalMs, lastEventId, resumedFromSequence });
      if (signature !== lastSignature) {
        lastSignature = signature;
        sequence += 1;
        unchangedPolls = 0;
        lastHeartbeatAt = Date.now();
        yield {
          type: "snapshot",
          source: "control-plane",
          sequence,
          receivedAt: new Date().toISOString(),
          diagnostics,
          job,
        };
      } else {
        unchangedPolls += 1;
        const now = Date.now();
        if (heartbeatEveryMs > 0 && now - lastHeartbeatAt >= heartbeatEveryMs) {
          sequence += 1;
          lastHeartbeatAt = now;
          yield {
            type: "heartbeat",
            source: "control-plane",
            sequence,
            receivedAt: new Date().toISOString(),
            diagnostics: {
              ...diagnostics,
              unchangedPolls,
              status: job.status,
              progress: job.progress,
              message: job.message,
            },
            jobId,
          };
        }
      }
      if (!isJobActive(job)) return;
    } catch (error) {
      sequence += 1;
      yield {
        type: "error",
        source: "control-plane",
        sequence,
        receivedAt: new Date().toISOString(),
        diagnostics: {
          eventSource: "python-job-snapshot-poll",
          pollIntervalMs: intervalMs,
          logTail: [],
          lastEventId,
          resumedFromSequence,
        },
        error: error instanceof Error ? error.message : String(error),
      };
      return;
    }

    await delay(intervalMs, signal);
  }
}

function jobEventDiagnostics(
  job: JobRecord,
  {
    intervalMs,
    lastEventId,
    resumedFromSequence,
  }: {
    intervalMs: number;
    lastEventId?: string | null;
    resumedFromSequence?: number | null;
  },
): JobEventDiagnostics {
  return {
    eventSource: "python-job-snapshot-poll",
    pollIntervalMs: intervalMs,
    logTail: job.logs.slice(-6),
    lastEventId,
    resumedFromSequence,
  };
}

export function parseTrackedSequence(jobId: string, lastEventId?: string | null): number | null {
  if (!lastEventId) return null;
  const [eventJobId, rawSequence] = lastEventId.split(":");
  if (eventJobId !== jobId) return null;
  const sequence = Number(rawSequence);
  return Number.isInteger(sequence) && sequence >= 0 ? sequence : null;
}

function jobSnapshotSignature(job: JobRecord) {
  return JSON.stringify({
    status: job.status,
    progress: job.progress,
    message: job.message,
    artifact_ids: job.artifact_ids,
    metrics: job.metrics,
    error: job.error,
    logs: job.logs.slice(-4),
    started_at: job.started_at,
    finished_at: job.finished_at,
  });
}

function isJobActive(job: JobRecord) {
  return job.status === "queued" || job.status === "running";
}

function delay(ms: number, signal?: AbortSignal) {
  if (ms <= 0 || signal?.aborted) return Promise.resolve();
  return new Promise<void>((resolve) => {
    const timeout = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timeout);
        resolve();
      },
      { once: true },
    );
  });
}
