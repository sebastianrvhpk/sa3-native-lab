import type { GestureOption } from "./gestureModel";
import { stringValue, type GenerationMode, type RecipeValue } from "./recipeFormModel";
import type { LatentOperatorMode } from "./workbenchConfigs";
import type { ArtifactRecord } from "./types";

export type GestureActionKind = "generate" | "encode" | "decode" | "operator" | "experiment" | "remember";
export type SourceRequirementStatus = "ready" | "missing" | "optional";

export interface GestureSourceRequirement {
  label: string;
  status: SourceRequirementStatus;
  detail: string;
}

export interface GestureActionDescriptor {
  kind: GestureActionKind;
  label: string;
  progressLabel: string;
  ready: boolean;
  disabledReason: string | null;
  intentCopy: string;
  sourceRequirements: GestureSourceRequirement[];
}

export interface GestureActionDescriptorInput {
  gesture: GestureOption;
  selectedArtifact: ArtifactRecord | null;
  generationMode: GenerationMode;
  generationForm: Record<string, RecipeValue>;
  generationNeedsSource: boolean;
  generationSource: ArtifactRecord | null;
  generationReady: boolean;
  generationBusy: boolean;
  encodeReady: boolean;
  encodeBusy: boolean;
  decodeReady: boolean;
  decodeBusy: boolean;
  operatorMode: LatentOperatorMode;
  operatorLabel: string;
  operatorReady: boolean;
  operatorBusy: boolean;
  operatorNeedsDonor: boolean;
  donorArtifactId: string;
  donorArtifactLabel?: string | null;
  experimentLabel: string;
  experimentReady: boolean;
  experimentBusy: boolean;
  rememberBusy: boolean;
}

export function describeGestureAction(input: GestureActionDescriptorInput): GestureActionDescriptor {
  if (input.gesture.tuneSource === "generation") return generationDescriptor(input);
  if (input.gesture.id === "encode") return encodeDescriptor(input);
  if (input.gesture.id === "decode") return decodeDescriptor(input);
  if (input.gesture.tuneSource === "operator") return operatorDescriptor(input);
  if (input.gesture.tuneSource === "experiment") return experimentDescriptor(input);
  return rememberDescriptor(input);
}

function generationDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const busy = input.generationBusy;
  const prompt = stringValue(input.generationForm.prompt);
  const label = generationButtonLabel(input.gesture.id, input.generationMode);
  const requirements: GestureSourceRequirement[] = [
    {
      label: "Prompt",
      status: prompt ? "ready" : "missing",
      detail: prompt || "Add a prompt in Tune.",
    },
  ];
  if (input.generationNeedsSource) {
    requirements.push({
      label: "Source sound",
      status: input.generationSource ? "ready" : "missing",
      detail: input.generationSource?.label ?? input.generationSource?.file?.filename ?? "Choose an audio source.",
    });
  }
  const reason = blockedReason(input.gesture.disabledReason, busy, input.generationReady, firstMissing(requirements) ?? "Check prompt, source, length, and steps.");
  return {
    kind: "generate",
    label: busy ? input.gesture.progressPhrase : label,
    progressLabel: input.gesture.progressPhrase,
    ready: !reason,
    disabledReason: reason,
    intentCopy: input.generationNeedsSource
      ? `Will use ${input.generationSource?.label ?? "the selected source"} to create a pending take.`
      : "Will make a new pending take from the prompt.",
    sourceRequirements: requirements,
  };
}

function encodeDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const requirements = [
    {
      label: "Current sound",
      status: input.selectedArtifact?.kind === "audio" ? "ready" : "missing",
      detail: input.selectedArtifact?.kind === "audio" ? artifactLabel(input.selectedArtifact) : "Select an audio sound.",
    } satisfies GestureSourceRequirement,
  ];
  const reason = blockedReason(input.gesture.disabledReason, input.encodeBusy, input.encodeReady, firstMissing(requirements) ?? "Select an audio sound.");
  return {
    kind: "encode",
    label: input.encodeBusy ? "Encoding latent" : "Encode current sound",
    progressLabel: "Encoding latent",
    ready: !reason,
    disabledReason: reason,
    intentCopy: "Will turn the current sound into SAME latent material.",
    sourceRequirements: requirements,
  };
}

function decodeDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const requirements = [
    {
      label: "Current latent",
      status: input.selectedArtifact?.kind === "latent" ? "ready" : "missing",
      detail: input.selectedArtifact?.kind === "latent" ? artifactLabel(input.selectedArtifact) : "Select latent material.",
    } satisfies GestureSourceRequirement,
  ];
  const reason = blockedReason(input.gesture.disabledReason, input.decodeBusy, input.decodeReady, firstMissing(requirements) ?? "Select latent material.");
  return {
    kind: "decode",
    label: input.decodeBusy ? "Decoding sound" : "Decode latent",
    progressLabel: "Decoding sound",
    ready: !reason,
    disabledReason: reason,
    intentCopy: "Will render the current latent as a playable sound.",
    sourceRequirements: requirements,
  };
}

function operatorDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const requirements: GestureSourceRequirement[] = [
    {
      label: "Current latent",
      status: input.selectedArtifact?.kind === "latent" ? "ready" : "missing",
      detail: input.selectedArtifact?.kind === "latent" ? artifactLabel(input.selectedArtifact) : "Select latent material.",
    },
  ];
  if (input.operatorNeedsDonor) {
    requirements.push({
      label: "Texture donor",
      status: input.donorArtifactId ? "ready" : "missing",
      detail: input.donorArtifactLabel || input.donorArtifactId || "Choose a donor latent.",
    });
  }
  const label = input.operatorMode === "latent.graft" ? "Borrow texture" : `Apply ${input.operatorLabel}`;
  const reason = blockedReason(input.gesture.disabledReason, input.operatorBusy, input.operatorReady, firstMissing(requirements) ?? "Fill required Tune fields.");
  return {
    kind: "operator",
    label: input.operatorBusy ? input.gesture.progressPhrase : label,
    progressLabel: input.gesture.progressPhrase,
    ready: !reason,
    disabledReason: reason,
    intentCopy: `Will make a latent take with ${input.operatorLabel}.`,
    sourceRequirements: requirements,
  };
}

function experimentDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const requirements: GestureSourceRequirement[] = [
    {
      label: "Tune fields",
      status: input.experimentReady ? "ready" : "missing",
      detail: input.experimentReady ? input.experimentLabel : "Fill required source or path fields.",
    },
  ];
  const reason = blockedReason(input.gesture.disabledReason, input.experimentBusy, input.experimentReady, firstMissing(requirements) ?? "Fill required Tune fields.");
  return {
    kind: "experiment",
    label: input.experimentBusy ? input.gesture.progressPhrase : runExperimentLabel(input.experimentLabel),
    progressLabel: input.gesture.progressPhrase,
    ready: !reason,
    disabledReason: reason,
    intentCopy: `Will run ${input.experimentLabel} and save the evidence as a bundle or take.`,
    sourceRequirements: requirements,
  };
}

function rememberDescriptor(input: GestureActionDescriptorInput): GestureActionDescriptor {
  const requirements = [
    {
      label: "Current material",
      status: input.selectedArtifact ? "ready" : "missing",
      detail: input.selectedArtifact ? artifactLabel(input.selectedArtifact) : "Select material to remember.",
    } satisfies GestureSourceRequirement,
  ];
  const ready = Boolean(input.selectedArtifact) && !input.rememberBusy;
  const reason = blockedReason(null, input.rememberBusy, ready, firstMissing(requirements) ?? "Select material to remember.");
  return {
    kind: "remember",
    label: input.rememberBusy ? "Remembering" : "Remember current sound",
    progressLabel: "Remembering",
    ready: !reason,
    disabledReason: reason,
    intentCopy: "Will move the current material into Memory for reuse later.",
    sourceRequirements: requirements,
  };
}

function blockedReason(
  gestureReason: string | null,
  busy: boolean,
  ready: boolean,
  fallback: string,
) {
  if (gestureReason) return gestureReason;
  if (busy) return "A take is already pending for this gesture.";
  return ready ? null : fallback;
}

function firstMissing(requirements: readonly GestureSourceRequirement[]) {
  return requirements.find((requirement) => requirement.status === "missing")?.detail ?? null;
}

function artifactLabel(artifact: ArtifactRecord) {
  return artifact.label || artifact.file?.filename || artifact.artifact_id;
}

function generationButtonLabel(gestureId: string, mode: GenerationMode) {
  if (gestureId === "vary") return "Vary sound";
  if (gestureId === "continue") return mode === "generate.inpaint" ? "Inpaint sound" : "Continue sound";
  return mode === "generate.text_to_audio" ? "Make sound" : "Make from source";
}

function runExperimentLabel(label: string) {
  if (/prompt search/i.test(label)) return "Find prompt candidates";
  if (/alpha sweep/i.test(label)) return "Render sweep";
  if (/memory/i.test(label)) return "Search memory";
  return `Run ${label}`;
}
