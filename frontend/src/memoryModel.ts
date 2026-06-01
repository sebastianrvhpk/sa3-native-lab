import { bundleReuseActionsForContext } from "./bundleReuseModel";
import { gestureForOperator, type GestureId } from "./gestureModel";
import type { ArtifactRecord, OperatorName } from "./types";

export type MemoryRole = "texture" | "loop" | "seed" | "donor" | "reference" | "keeper" | "brittle" | "reject";
export type MemoryReuseIntent =
  | "source"
  | "anchor"
  | "donor"
  | "prompt_seed"
  | "advanced_gesture"
  | "recover";

export interface MemoryItem {
  artifact: ArtifactRecord;
  role: MemoryRole | null;
  reuseIntent: MemoryReuseIntent | null;
  promptSeed: string | null;
  sourceMapping: MemorySourceMapping;
  actions: MemoryReuseAction[];
}

export interface MemorySourceMapping {
  artifactId: string;
  kind: ArtifactRecord["kind"];
  path: string;
  prompt: string | null;
  operator: OperatorName | null;
}

export interface MemoryReuseAction {
  id: string;
  label: string;
  description: string;
  intent: MemoryReuseIntent;
  available: boolean;
  disabledReason: string | null;
  gestureId?: GestureId;
  compareSlot?: "a" | "b";
  fieldKey?: string;
  mode?: string;
  value?: string;
}

export const memoryRoleOptions: readonly { value: MemoryRole; label: string }[] = [
  { value: "texture", label: "texture" },
  { value: "loop", label: "loop" },
  { value: "seed", label: "seed" },
  { value: "donor", label: "donor" },
  { value: "reference", label: "reference" },
  { value: "keeper", label: "keeper" },
  { value: "brittle", label: "brittle" },
  { value: "reject", label: "reject" },
];

export const memoryReuseIntentOptions: readonly { value: MemoryReuseIntent; label: string }[] = [
  { value: "source", label: "source" },
  { value: "anchor", label: "anchor" },
  { value: "donor", label: "donor" },
  { value: "prompt_seed", label: "prompt seed" },
  { value: "advanced_gesture", label: "advanced gesture" },
  { value: "recover", label: "recover" },
];

const memoryRoles = new Set(memoryRoleOptions.map((item) => item.value));
const memoryReuseIntents = new Set(memoryReuseIntentOptions.map((item) => item.value));

export function buildMemoryItems(
  artifacts: readonly ArtifactRecord[],
  context: { activeSessionId?: string | null } = {},
): MemoryItem[] {
  return artifacts.map((artifact) => memoryItemFromArtifact(artifact, context));
}

export function memoryItemFromArtifact(
  artifact: ArtifactRecord,
  context: { activeSessionId?: string | null } = {},
): MemoryItem {
  return {
    artifact,
    role: memoryRoleFromArtifact(artifact),
    reuseIntent: memoryReuseIntentFromArtifact(artifact),
    promptSeed: promptSeedFromMemory(artifact),
    sourceMapping: memorySourceMapping(artifact),
    actions: memoryActionsForArtifact(artifact, context),
  };
}

export function memoryRoleFromArtifact(artifact: ArtifactRecord): MemoryRole | null {
  const metadataRole = typeof artifact.metadata.memory_role === "string" ? artifact.metadata.memory_role : "";
  if (isMemoryRole(metadataRole)) return metadataRole;
  const tagRole = artifact.tags.find(isMemoryRole);
  if (tagRole) return tagRole;
  const decision = typeof artifact.metadata.listening_decision === "string" ? artifact.metadata.listening_decision : "";
  if (decision === "keeper") return "keeper";
  if (decision === "rejected") return "reject";
  return null;
}

export function memoryReuseIntentFromArtifact(artifact: ArtifactRecord): MemoryReuseIntent | null {
  const value = typeof artifact.metadata.reuse_intent === "string" ? artifact.metadata.reuse_intent : "";
  return isMemoryReuseIntent(value) ? value : null;
}

export function promptSeedFromMemory(artifact: ArtifactRecord): string | null {
  const prompt = artifact.prompt?.trim();
  if (prompt) return prompt;
  const label = artifact.label?.trim();
  if (label) return label;
  const notes = artifact.notes?.trim();
  if (notes) return notes;
  return null;
}

export function memorySourceMapping(artifact: ArtifactRecord): MemorySourceMapping {
  const operator = typeof artifact.metadata.operator === "string" ? artifact.metadata.operator as OperatorName : null;
  return {
    artifactId: artifact.artifact_id,
    kind: artifact.kind,
    path: artifact.path,
    prompt: promptSeedFromMemory(artifact),
    operator,
  };
}

export function memoryActionsForArtifact(
  artifact: ArtifactRecord,
  context: { activeSessionId?: string | null } = {},
): MemoryReuseAction[] {
  const promptSeed = promptSeedFromMemory(artifact);
  const canRecover = Boolean(context.activeSessionId && artifact.session_id !== context.activeSessionId);
  const sourceAvailable = artifact.kind === "audio" || artifact.kind === "latent";
  const actions: MemoryReuseAction[] = [
    {
      id: "source",
      label: "Use as Source",
      description: "Make this remembered material the current source.",
      intent: "source",
      available: sourceAvailable,
      disabledReason: sourceAvailable ? null : "Only audio or latent material can become the current source.",
      gestureId: artifact.kind === "audio" ? "continue" : artifact.kind === "latent" ? "morph" : undefined,
      compareSlot: artifact.kind === "audio" ? "b" : undefined,
    },
    {
      id: "anchor",
      label: "Anchor",
      description: "Pin this remembered sound as a stable listening reference.",
      intent: "anchor",
      available: artifact.kind === "audio",
      disabledReason: artifact.kind === "audio" ? null : "Only audio can be pinned as an anchor.",
      compareSlot: "a",
    },
    {
      id: "donor",
      label: "Use as Donor",
      description: "Use this latent as the donor for Borrow Texture.",
      intent: "donor",
      available: artifact.kind === "latent",
      disabledReason: artifact.kind === "latent" ? null : "Borrow Texture needs remembered latent material.",
      gestureId: "borrow_texture",
    },
    {
      id: "find_similar",
      label: "Find Similar",
      description: "Search local latent memory for material near this remembered latent.",
      intent: "advanced_gesture",
      available: artifact.kind === "latent",
      disabledReason: artifact.kind === "latent" ? null : "Memory query needs a latent source.",
      gestureId: "steer",
      mode: "memory.query",
    },
    {
      id: "prompt_seed",
      label: "Seed Prompt",
      description: "Copy remembered prompt, label, or notes into Make.",
      intent: "prompt_seed",
      available: Boolean(promptSeed),
      disabledReason: promptSeed ? null : "No prompt, label, or notes are available to seed a prompt.",
      gestureId: "make",
      value: promptSeed ?? undefined,
    },
    {
      id: "recover",
      label: "Recover",
      description: "Bring this remembered material back into the active session.",
      intent: "recover",
      available: canRecover,
      disabledReason: canRecover ? null : "Already in the active session.",
    },
  ];
  actions.push(...bundleMemoryActions(artifact));
  return dedupeMemoryActions(actions);
}

export function bundleMemoryActions(artifact: ArtifactRecord): MemoryReuseAction[] {
  if (artifact.kind !== "bundle") return [];
  const operator = typeof artifact.metadata.operator === "string" ? artifact.metadata.operator : "";
  const kind = typeof artifact.metadata.bundle_kind === "string" ? artifact.metadata.bundle_kind : "";
  const gestureId = gestureForOperator((operator || "experiment.audio_style_vectors") as OperatorName);
  return bundleReuseActionsForContext({ kind, operator, prompt: promptSeedFromMemory(artifact) }).map((action) => ({
    id: action.fieldKey,
    label: action.label,
    description: "Send this remembered bundle into an existing Advanced Gesture path.",
    intent: "advanced_gesture",
    available: true,
    disabledReason: null,
    gestureId,
    fieldKey: action.fieldKey,
    mode: action.mode,
    value: action.value,
  }));
}

export function memoryRoleLabel(role: MemoryRole | null): string {
  return role ? memoryRoleOptions.find((item) => item.value === role)?.label ?? role : "untyped";
}

function dedupeMemoryActions(actions: MemoryReuseAction[]): MemoryReuseAction[] {
  const seen = new Set<string>();
  return actions.filter((action) => {
    const key = `${action.intent}:${action.id}:${action.fieldKey ?? ""}:${action.mode ?? ""}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function isMemoryRole(value: string): value is MemoryRole {
  return memoryRoles.has(value as MemoryRole);
}

function isMemoryReuseIntent(value: string): value is MemoryReuseIntent {
  return memoryReuseIntents.has(value as MemoryReuseIntent);
}
