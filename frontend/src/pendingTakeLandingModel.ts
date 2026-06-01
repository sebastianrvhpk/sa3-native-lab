import { gestureForOperator, type GestureId } from "./gestureModel";
import type { JobRecord, JobStatus, OperatorName } from "./types";

export interface PendingTakeLanding {
  pendingTakeId: string;
  sourceIds: string[];
  producedArtifactIds: string[];
  landingArtifactId: string | null;
  completionPhrase: string;
  branchLabel: string;
  suggestedNextGestureIds: GestureId[];
  canCancel: boolean;
  canRetry: boolean;
  canInspect: boolean;
  recoverySuggestion: string | null;
}

export function pendingTakeLandingFromJob(job: JobRecord): PendingTakeLanding {
  return {
    pendingTakeId: job.job_id,
    sourceIds: Object.values(job.recipe.inputs).filter(Boolean),
    producedArtifactIds: job.artifact_ids,
    landingArtifactId: landingArtifactIdForJob(job),
    completionPhrase: completionPhraseForJob(job),
    branchLabel: branchLabelForJob(job),
    suggestedNextGestureIds: suggestedGesturesForJob(job),
    canCancel: job.status === "queued" || job.status === "running",
    canRetry: job.status === "failed" || job.status === "cancelled",
    canInspect: true,
    recoverySuggestion: recoverySuggestionForJob(job),
  };
}

export function landingArtifactIdForJob(job: JobRecord): string | null {
  if (job.status !== "succeeded") return null;
  return job.artifact_ids.at(-1) ?? null;
}

export function completionPhraseForJob(job: JobRecord): string {
  if (job.status === "queued") return "Waiting for a slot";
  if (job.status === "running") return activeCompletionPhrase(job.recipe.operator);
  if (job.status === "failed") return "No take landed";
  if (job.status === "cancelled") return "Take stopped before landing";
  return landedCompletionPhrase(job.recipe.operator, job.artifact_ids.length);
}

export function branchLabelForJob(job: JobRecord): string {
  if (job.artifact_ids.length > 1) return `${job.artifact_ids.length} takes`;
  if (job.artifact_ids.length === 1) return "1 take";
  if (job.status === "failed") return "needs recovery";
  if (job.status === "cancelled") return "stopped";
  return "pending take";
}

export function recoverySuggestionForJob(job: JobRecord): string | null {
  if (job.status !== "failed" && job.status !== "cancelled") return null;
  const text = [job.error, job.message, ...job.logs.slice(-3)].filter(Boolean).join(" ").toLowerCase();
  if (text.includes("huggingface") || text.includes("hf") || text.includes("auth") || text.includes("token")) {
    return "Check Hugging Face access in Settings, then retry.";
  }
  if (text.includes("duration") || text.includes("memory") || text.includes("oom") || text.includes("out of memory")) {
    return "Reduce duration or steps in Tune, then retry.";
  }
  if (text.includes("path") || text.includes("file") || text.includes("not found")) {
    return "Recover or reselect the source material, then retry.";
  }
  if (text.includes("mlx") || text.includes("backend") || text.includes("mps")) {
    return "Check backend readiness in Settings, then retry.";
  }
  if (job.status === "cancelled") return "Retry when you are ready, or adjust Tune first.";
  return "Inspect the failure, adjust Tune, then retry.";
}

function activeCompletionPhrase(operator: OperatorName): string {
  if (operator === "generate.text_to_audio") return "A new sound is being made";
  if (operator === "generate.audio_to_audio" || operator === "generate.inpaint") return "A continuation is being shaped";
  if (operator === "latent.encode") return "Latent material is being prepared";
  if (operator === "latent.decode") return "Audio is being decoded";
  if (operator === "latent.graft") return "Texture is being borrowed";
  if (operator.startsWith("latent.")) return "Latent material is moving";
  if (operator === "memory.query") return "Memory is being searched";
  return "The gesture is running";
}

function landedCompletionPhrase(operator: OperatorName, count: number): string {
  if (operator === "latent.encode") return "Latent landed";
  if (operator === "latent.decode") return "Sound landed";
  if (operator === "memory.query") return "Memory hits landed";
  if (operator.startsWith("experiment.")) return count > 1 ? "Branch outputs landed" : "Bundle landed";
  return count > 1 ? "Takes landed" : "Take landed";
}

function suggestedGesturesForJob(job: JobRecord): GestureId[] {
  if (job.status === "failed" || job.status === "cancelled") return [gestureForOperator(job.recipe.operator)];
  if (job.status === "queued" || job.status === "running") return [];
  const operator = job.recipe.operator;
  if (operator === "latent.encode") return ["decode", "morph", "borrow_texture", "remember"];
  if (operator === "latent.decode" || operator.startsWith("generate.")) return ["continue", "vary", "encode", "remember"];
  if (operator === "latent.graft" || operator.startsWith("latent.")) return ["decode", "morph", "remember"];
  if (operator.startsWith("experiment.") || operator === "memory.query") return ["steer", "remember"];
  return [gestureForOperator(operator), "remember"];
}
