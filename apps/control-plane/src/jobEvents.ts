import type { JobRecord } from "./contracts.js";
import type { PythonClient } from "./pythonClient.js";

export type ControlPlaneJobEvent =
  | {
      type: "snapshot";
      source: "control-plane";
      sequence: number;
      receivedAt: string;
      job: JobRecord;
    }
  | {
      type: "error";
      source: "control-plane";
      sequence: number;
      receivedAt: string;
      error: string;
    };

export interface PollJobEventsOptions {
  jobId: string;
  intervalMs?: number;
  signal?: AbortSignal;
}

export async function* pollJobEvents(
  client: Pick<PythonClient, "job">,
  { jobId, intervalMs = 1000, signal }: PollJobEventsOptions,
): AsyncGenerator<ControlPlaneJobEvent> {
  let lastSignature = "";
  let sequence = 0;

  while (!signal?.aborted) {
    try {
      const job = await client.job(jobId);
      const signature = jobSnapshotSignature(job);
      if (signature !== lastSignature) {
        lastSignature = signature;
        sequence += 1;
        yield {
          type: "snapshot",
          source: "control-plane",
          sequence,
          receivedAt: new Date().toISOString(),
          job,
        };
      }
      if (!isJobActive(job)) return;
    } catch (error) {
      sequence += 1;
      yield {
        type: "error",
        source: "control-plane",
        sequence,
        receivedAt: new Date().toISOString(),
        error: error instanceof Error ? error.message : String(error),
      };
      return;
    }

    await delay(intervalMs, signal);
  }
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
