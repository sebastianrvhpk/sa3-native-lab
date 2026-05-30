import { formatJobElapsed, progressPercent } from "./jobUtils";
import { gestureById, gestureForOperator, gestureLabelForOperator } from "./gestureModel";
import { pendingTakeLandingFromJob, type PendingTakeLanding } from "./pendingTakeLandingModel";
import type { JobRecord, JobStatus, OperatorName } from "./types";

export interface PendingTake {
  id: string;
  job: JobRecord;
  gestureId: ReturnType<typeof gestureForOperator>;
  gestureLabel: string;
  status: JobStatus;
  phase: string;
  progressPercent: number;
  elapsed: string;
  sourceIds: string[];
  producedArtifactIds: string[];
  landingArtifactId: string | null;
  completionPhrase: string;
  branchLabel: string;
  suggestedNextGestureIds: PendingTakeLanding["suggestedNextGestureIds"];
  recoverySuggestion: string | null;
  landing: PendingTakeLanding;
  canCancel: boolean;
  canRetry: boolean;
  phrase: string;
  detail: string;
  inspect: {
    jobId: string;
    operator: OperatorName;
  };
}

export function pendingTakeFromJob(job: JobRecord): PendingTake {
  const gestureId = gestureForOperator(job.recipe.operator);
  const gesture = gestureById(gestureId);
  const phrase = phraseForJob(job);
  const phase = job.phase?.replace(/[_-]+/g, " ") || phaseForStatus(job.status);
  const landing = pendingTakeLandingFromJob(job);
  return {
    id: job.job_id,
    job,
    gestureId,
    gestureLabel: gestureLabelForOperator(job.recipe.operator),
    status: job.status,
    phase,
    progressPercent: progressPercent(job),
    elapsed: formatJobElapsed(job),
    sourceIds: landing.sourceIds,
    producedArtifactIds: landing.producedArtifactIds,
    landingArtifactId: landing.landingArtifactId,
    completionPhrase: landing.completionPhrase,
    branchLabel: landing.branchLabel,
    suggestedNextGestureIds: landing.suggestedNextGestureIds,
    recoverySuggestion: landing.recoverySuggestion,
    landing,
    canCancel: landing.canCancel,
    canRetry: landing.canRetry,
    phrase,
    detail: job.error ?? job.message ?? job.logs.at(-1) ?? gesture.shortIntent,
    inspect: {
      jobId: job.job_id,
      operator: job.recipe.operator,
    },
  };
}

export function pendingTakesFromJobs(jobs: readonly JobRecord[]) {
  return jobs.map(pendingTakeFromJob);
}

export function phraseForJob(job: JobRecord) {
  if (job.status === "queued") return queuedPhrase(job.recipe.operator);
  if (job.status === "failed") return "Take failed";
  if (job.status === "cancelled") return "Take cancelled";
  if (job.status === "succeeded") return completedPhrase(job.recipe.operator);
  return activePhrase(job.recipe.operator);
}

function activePhrase(operator: OperatorName) {
  if (operator === "generate.text_to_audio") return "Making take";
  if (operator === "generate.audio_to_audio" || operator === "generate.inpaint") return "Continuing sound";
  if (operator === "latent.encode") return "Encoding latent";
  if (operator === "latent.decode") return "Decoding sound";
  if (operator === "latent.graft") return "Borrowing texture";
  if (operator === "latent.renoise") return "Varying latent";
  if (operator.startsWith("latent.")) return "Morphing latent";
  if (operator === "training.lora") return "Training gesture";
  if (operator === "memory.query") return "Searching memory";
  if (operator === "experiment.prompt_search") return "Probing prompts";
  return "Steering sound";
}

function queuedPhrase(operator: OperatorName) {
  if (operator === "generate.text_to_audio") return "Make queued";
  if (operator === "generate.audio_to_audio" || operator === "generate.inpaint") return "Continue queued";
  if (operator === "latent.encode") return "Encode queued";
  if (operator === "latent.decode") return "Decode queued";
  if (operator === "latent.graft") return "Borrow queued";
  if (operator === "latent.renoise") return "Variation queued";
  if (operator.startsWith("latent.")) return "Morph queued";
  if (operator === "training.lora") return "Training queued";
  if (operator === "memory.query") return "Memory search queued";
  if (operator === "experiment.prompt_search") return "Prompt probe queued";
  return "Steer queued";
}

function completedPhrase(operator: OperatorName) {
  if (operator === "latent.encode") return "Latent encoded";
  if (operator === "latent.decode") return "Sound decoded";
  if (operator === "memory.query") return "Memory searched";
  if (operator.startsWith("experiment.") || operator === "training.lora") return "Gesture finished";
  return "Take landed";
}

function phaseForStatus(status: JobStatus) {
  if (status === "queued") return "queued";
  if (status === "running") return "running";
  if (status === "succeeded") return "done";
  return status;
}
