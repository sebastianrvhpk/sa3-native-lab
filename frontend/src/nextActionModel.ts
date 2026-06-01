import type { ResultFamily } from "./controlPlane";
import { gestureForOperator, gestureLabelForOperator, type GestureId } from "./gestureModel";
import { bundleMemoryActions } from "./memoryModel";
import type { PendingTake } from "./pendingTakeModel";
import type { ArtifactRecord } from "./types";

export type NextActionKind =
  | "gesture"
  | "remember"
  | "inspect"
  | "branch"
  | "retry"
  | "cancel"
  | "tune"
  | "bundle_reuse";

export interface ProductNextAction {
  id: string;
  label: string;
  description: string;
  kind: NextActionKind;
  available: boolean;
  disabledReason: string | null;
  gestureId?: GestureId;
  generationMode?: "generate.text_to_audio" | "generate.audio_to_audio" | "generate.inpaint";
  operatorMode?: "latent.cyclic_roll" | "latent.blur" | "latent.dsp" | "latent.graft" | "latent.renoise";
  experimentMode?: string;
  fieldKey?: string;
  value?: string;
}

export function nextActionsForArtifact(
  artifact: ArtifactRecord | null,
  context: { donorLatents?: readonly ArtifactRecord[] } = {},
): ProductNextAction[] {
  if (!artifact) {
    return [
      action({
        id: "make",
        label: "Make",
        description: "Start from a prompt.",
        kind: "gesture",
        gestureId: "make",
        generationMode: "generate.text_to_audio",
      }),
    ];
  }
  if (artifact.kind === "audio") {
    return [
      action({
        id: "continue",
        label: "Continue",
        description: "Use this sound as the source for another take.",
        kind: "gesture",
        gestureId: "continue",
        generationMode: "generate.audio_to_audio",
      }),
      action({
        id: "vary",
        label: "Vary",
        description: "Make a nearby take with controlled variation.",
        kind: "gesture",
        gestureId: "vary",
        generationMode: "generate.audio_to_audio",
      }),
      action({
        id: "encode",
        label: "Encode",
        description: "Prepare this sound for latent gestures.",
        kind: "gesture",
        gestureId: "encode",
      }),
      action({
        id: "remember",
        label: "Remember",
        description: "Save this sound with a role, tags, and reuse intent.",
        kind: "remember",
        gestureId: "remember",
      }),
    ];
  }
  if (artifact.kind === "latent") {
    const donorCount = (context.donorLatents ?? []).filter((item) => item.artifact_id !== artifact.artifact_id).length;
    return [
      action({
        id: "decode",
        label: "Decode",
        description: "Hear this latent as audio.",
        kind: "gesture",
        gestureId: "decode",
      }),
      action({
        id: "morph",
        label: "Morph",
        description: "Move through roll, blur, reroute, or renoise gestures.",
        kind: "gesture",
        gestureId: "morph",
        operatorMode: "latent.cyclic_roll",
      }),
      action({
        id: "borrow_texture",
        label: "Borrow Texture",
        description: "Blend texture from another latent donor.",
        kind: "gesture",
        gestureId: "borrow_texture",
        operatorMode: "latent.graft",
        available: donorCount > 0,
        disabledReason: donorCount > 0 ? null : "Encode or recover another latent donor first.",
      }),
      action({
        id: "find_similar",
        label: "Find Similar",
        description: "Search local latent memory for nearby material.",
        kind: "gesture",
        gestureId: "steer",
        experimentMode: "memory.query",
      }),
      action({
        id: "remember",
        label: "Remember",
        description: "Save this latent with a reusable role.",
        kind: "remember",
        gestureId: "remember",
      }),
    ];
  }
  if (artifact.kind === "bundle") {
    return [
      action({
        id: "inspect",
        label: "Inspect",
        description: "Open bundle details and available use paths.",
        kind: "inspect",
      }),
      ...bundleMemoryActions(artifact).map((reuse) =>
        action({
          id: reuse.id,
          label: reuse.label,
          description: reuse.description,
          kind: "bundle_reuse",
          gestureId: reuse.gestureId ?? "steer",
          experimentMode: reuse.mode,
          fieldKey: reuse.fieldKey,
          value: reuse.value,
        }),
      ),
    ];
  }
  return [
    action({
      id: "inspect",
      label: "Inspect",
      description: "Open technical details for this material.",
      kind: "inspect",
    }),
  ];
}

export function nextActionsForPendingTake(take: PendingTake): ProductNextAction[] {
  if (take.status === "queued" || take.status === "running") {
    return [
      action({
        id: "cancel",
        label: "Cancel",
        description: "Stop this pending take.",
        kind: "cancel",
        available: take.canCancel,
        disabledReason: take.canCancel ? null : "This take can no longer be cancelled.",
      }),
      action({ id: "inspect", label: "Inspect", description: "Open progress, logs, and backend details.", kind: "inspect" }),
    ];
  }
  if (take.status === "failed" || take.status === "cancelled") {
    return [
      action({
        id: "retry",
        label: "Retry",
        description: "Run the same gesture again.",
        kind: "retry",
        available: take.canRetry,
        disabledReason: take.canRetry ? null : "Retry is not available for this take.",
      }),
      action({
        id: "tune",
        label: "Adjust Tune",
        description: take.recoverySuggestion ?? "Change parameters before trying again.",
        kind: "tune",
        gestureId: take.gestureId,
      }),
      action({ id: "inspect", label: "Inspect", description: "Open failure details and raw logs.", kind: "inspect" }),
    ];
  }
  return take.suggestedNextGestureIds.map((gestureId) =>
    action({
      id: gestureId,
      label: labelForGestureId(gestureId),
      description: "Continue from the landed take.",
      kind: gestureId === "remember" ? "remember" : "gesture",
      gestureId,
    }),
  );
}

export function nextActionsForBranch(
  branch: ResultFamily,
  latestArtifact: ArtifactRecord | null = null,
): ProductNextAction[] {
  const gestureId = gestureForOperator(branch.operator);
  return [
    action({
      id: "do_again",
      label: "Do Again",
      description: `Repeat this ${gestureLabelForOperator(branch.operator)} branch.`,
      kind: "branch",
      gestureId,
    }),
    action({
      id: "continue_from_latest",
      label: latestArtifact?.kind === "latent" ? "Morph Latest" : "Continue Latest",
      description: latestArtifact ? "Use the latest take as the next source." : "No landed take is available yet.",
      kind: "gesture",
      gestureId: latestArtifact?.kind === "latent" ? "morph" : "continue",
      available: Boolean(latestArtifact),
      disabledReason: latestArtifact ? null : "No landed take is available yet.",
    }),
    action({
      id: "branch",
      label: "Branch",
      description: "Fork the gesture settings for a new creative path.",
      kind: "branch",
      gestureId,
    }),
    action({
      id: "remember",
      label: "Remember",
      description: latestArtifact ? "Save the latest take to memory." : "No landed take is available yet.",
      kind: "remember",
      gestureId: "remember",
      available: Boolean(latestArtifact),
      disabledReason: latestArtifact ? null : "No landed take is available yet.",
    }),
  ];
}

function action(input: Omit<ProductNextAction, "available" | "disabledReason"> & Partial<Pick<ProductNextAction, "available" | "disabledReason">>): ProductNextAction {
  return {
    available: true,
    disabledReason: null,
    ...input,
  };
}

function labelForGestureId(gestureId: GestureId) {
  if (gestureId === "borrow_texture") return "Borrow Texture";
  return gestureId.slice(0, 1).toUpperCase() + gestureId.slice(1);
}
