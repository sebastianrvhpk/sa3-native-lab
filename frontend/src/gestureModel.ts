import type { ExperimentMode, LatentOperatorMode } from "./workbenchConfigs";
import type { ArtifactKind, ArtifactRecord, OperatorName } from "./types";

export type GestureId =
  | "make"
  | "continue"
  | "vary"
  | "steer"
  | "borrow_texture"
  | "encode"
  | "decode"
  | "morph"
  | "remember";

export type GestureSourceKind = "none" | "audio" | "latent" | "bundle" | "any";
export type GestureTuneSource = "generation" | "same" | "operator" | "experiment" | "memory";

export interface GestureDefinition {
  id: GestureId;
  label: string;
  shortIntent: string;
  requiredSourceKind: GestureSourceKind;
  primaryOperators: readonly OperatorName[];
  outputKind: ArtifactKind | "memory";
  tuneSource: GestureTuneSource;
  progressPhrase: string;
  inspectOperatorName: string;
  defaultGenerationMode?: "generate.text_to_audio" | "generate.audio_to_audio" | "generate.inpaint";
  defaultOperatorMode?: LatentOperatorMode;
  defaultExperimentMode?: ExperimentMode;
}

export interface GestureOption extends GestureDefinition {
  available: boolean;
  availabilityReason: string;
  disabledReason: string | null;
}

export const gestureDefinitions: readonly GestureDefinition[] = [
  {
    id: "make",
    label: "Make",
    shortIntent: "Start fresh from a prompt.",
    requiredSourceKind: "none",
    primaryOperators: ["generate.text_to_audio"],
    outputKind: "audio",
    tuneSource: "generation",
    progressPhrase: "Making take",
    inspectOperatorName: "generate.text_to_audio",
    defaultGenerationMode: "generate.text_to_audio",
  },
  {
    id: "continue",
    label: "Continue",
    shortIntent: "Extend or reinterpret the current sound.",
    requiredSourceKind: "audio",
    primaryOperators: ["generate.audio_to_audio", "generate.inpaint"],
    outputKind: "audio",
    tuneSource: "generation",
    progressPhrase: "Continuing sound",
    inspectOperatorName: "generate.audio_to_audio",
    defaultGenerationMode: "generate.audio_to_audio",
  },
  {
    id: "vary",
    label: "Vary",
    shortIntent: "Make a nearby take without changing the idea.",
    requiredSourceKind: "audio",
    primaryOperators: ["generate.audio_to_audio"],
    outputKind: "audio",
    tuneSource: "generation",
    progressPhrase: "Varying take",
    inspectOperatorName: "generate.audio_to_audio",
    defaultGenerationMode: "generate.audio_to_audio",
  },
  {
    id: "steer",
    label: "Steer",
    shortIntent: "Push material toward a profile, direction, or prompt probe.",
    requiredSourceKind: "any",
    primaryOperators: [
      "experiment.audio_style_vectors",
      "experiment.style_direction.generate",
      "experiment.audio_direction.generate",
      "experiment.alpha_sweep",
      "experiment.prompt_search",
    ],
    outputKind: "bundle",
    tuneSource: "experiment",
    progressPhrase: "Steering sound",
    inspectOperatorName: "experiment.*",
    defaultExperimentMode: "experiment.audio_style_vectors",
  },
  {
    id: "borrow_texture",
    label: "Borrow Texture",
    shortIntent: "Use another latent as donor material.",
    requiredSourceKind: "latent",
    primaryOperators: ["latent.graft", "latent.dsp"],
    outputKind: "latent",
    tuneSource: "operator",
    progressPhrase: "Borrowing texture",
    inspectOperatorName: "latent.graft",
    defaultOperatorMode: "latent.graft",
  },
  {
    id: "encode",
    label: "Encode",
    shortIntent: "Prepare audio as latent material.",
    requiredSourceKind: "audio",
    primaryOperators: ["latent.encode"],
    outputKind: "latent",
    tuneSource: "same",
    progressPhrase: "Encoding latent",
    inspectOperatorName: "latent.encode",
  },
  {
    id: "decode",
    label: "Decode",
    shortIntent: "Hear latent material as audio.",
    requiredSourceKind: "latent",
    primaryOperators: ["latent.decode"],
    outputKind: "audio",
    tuneSource: "same",
    progressPhrase: "Decoding sound",
    inspectOperatorName: "latent.decode",
  },
  {
    id: "morph",
    label: "Morph",
    shortIntent: "Move through latent time, texture, or spectral shape.",
    requiredSourceKind: "latent",
    primaryOperators: ["latent.cyclic_roll", "latent.blur", "latent.dsp", "latent.renoise"],
    outputKind: "latent",
    tuneSource: "operator",
    progressPhrase: "Morphing latent",
    inspectOperatorName: "latent.*",
    defaultOperatorMode: "latent.cyclic_roll",
  },
  {
    id: "remember",
    label: "Remember",
    shortIntent: "Save this material to session memory.",
    requiredSourceKind: "any",
    primaryOperators: ["artifact.annotate"],
    outputKind: "memory",
    tuneSource: "memory",
    progressPhrase: "Remembering sound",
    inspectOperatorName: "artifact.annotate",
  },
] as const;

export function buildGestureOptions(selectedArtifact: ArtifactRecord | null): GestureOption[] {
  return gestureDefinitions.map((gesture) => {
    const disabledReason = disabledReasonForGesture(gesture, selectedArtifact);
    return {
      ...gesture,
      available: !disabledReason,
      availabilityReason: availabilityReasonForGesture(gesture, selectedArtifact),
      disabledReason,
    };
  });
}

export function gestureById(id: GestureId): GestureDefinition {
  return gestureDefinitions.find((gesture) => gesture.id === id) ?? gestureDefinitions[0];
}

export function disabledReasonForGesture(gesture: GestureDefinition, selectedArtifact: ArtifactRecord | null): string | null {
  if (gesture.requiredSourceKind === "none") return null;
  if (!selectedArtifact) return "Choose or import a sound first.";
  if (gesture.requiredSourceKind === "any") return null;
  if (selectedArtifact.kind !== gesture.requiredSourceKind) {
    if (gesture.requiredSourceKind === "audio") return "Needs an audio sound.";
    if (gesture.requiredSourceKind === "latent") return "Needs latent material.";
    if (gesture.requiredSourceKind === "bundle") return "Needs a reusable bundle.";
  }
  return null;
}

export function availabilityReasonForGesture(gesture: GestureDefinition, selectedArtifact: ArtifactRecord | null): string {
  const disabled = disabledReasonForGesture(gesture, selectedArtifact);
  if (disabled) return disabled;
  if (gesture.requiredSourceKind === "none") return "Ready from prompt.";
  if (gesture.id === "remember") return "Current material can be saved.";
  return "Ready for current material.";
}

export function gestureForOperator(operator: OperatorName): GestureId {
  if (operator === "generate.text_to_audio") return "make";
  if (operator === "generate.audio_to_audio" || operator === "generate.inpaint") return "continue";
  if (operator === "latent.encode") return "encode";
  if (operator === "latent.decode") return "decode";
  if (operator === "latent.graft") return "borrow_texture";
  if (operator === "latent.renoise") return "vary";
  if (operator.startsWith("latent.")) return "morph";
  if (operator === "artifact.annotate") return "remember";
  return "steer";
}

export function gestureLabelForOperator(operator: OperatorName): string {
  return gestureById(gestureForOperator(operator)).label;
}
