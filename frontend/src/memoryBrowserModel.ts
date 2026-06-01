import { listeningDecision, listeningDecisionLabel } from "./listeningDecision";
import {
  memoryActionsForArtifact,
  memoryReuseIntentFromArtifact,
  memoryRoleFromArtifact,
  memoryRoleLabel,
  type MemoryReuseAction,
} from "./memoryModel";
import { gestureLabelForOperator } from "./gestureModel";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, JobRecord } from "./types";

export interface MemoryBrowserContext {
  activeSessionId?: string | null;
  jobs?: readonly JobRecord[];
  families?: readonly ResultFamily[];
  artifacts?: readonly ArtifactRecord[];
}

export interface MemoryBrowserItem {
  artifact: ArtifactRecord;
  roleLabel: string;
  reuseIntentLabel: string | null;
  decisionLabel: string;
  kindLabel: string;
  branchLabel: string | null;
  lineageLabel: string;
  notesPreview: string | null;
  sourceLabel: string | null;
  actions: MemoryReuseAction[];
  primaryAction: MemoryReuseAction | null;
  usableNow: boolean;
}

export function buildMemoryBrowserItems(
  artifacts: readonly ArtifactRecord[],
  context: MemoryBrowserContext = {},
): MemoryBrowserItem[] {
  const index = memoryBrowserIndex(artifacts, context);
  return artifacts
    .map((artifact) => memoryBrowserItem(artifact, context, index))
    .sort(compareMemoryItems);
}

export function memoryBrowserItem(
  artifact: ArtifactRecord,
  context: MemoryBrowserContext = {},
  index = memoryBrowserIndex([artifact], context),
): MemoryBrowserItem {
  const actions = memoryActionsForArtifact(artifact, { activeSessionId: context.activeSessionId });
  const primaryAction = primaryMemoryAction(actions);
  const role = memoryRoleFromArtifact(artifact);
  const reuseIntent = memoryReuseIntentFromArtifact(artifact);
  const decision = listeningDecision(artifact);
  return {
    artifact,
    roleLabel: memoryRoleLabel(role),
    reuseIntentLabel: reuseIntent ? reuseIntent.replaceAll("_", " ") : null,
    decisionLabel: listeningDecisionLabel(decision),
    kindLabel: artifact.kind,
    branchLabel: branchLabel(artifact, index),
    lineageLabel: lineageLabel(artifact, index),
    notesPreview: notesPreview(artifact),
    sourceLabel: sourceLabel(artifact, index),
    actions,
    primaryAction,
    usableNow: Boolean(primaryAction),
  };
}

export function memoryBrowserSummary(items: readonly MemoryBrowserItem[]) {
  return {
    total: items.length,
    usable: items.filter((item) => item.usableNow).length,
    keepers: items.filter((item) => item.decisionLabel === "keeper" || item.roleLabel === "keeper").length,
    donors: items.filter((item) => item.artifact.kind === "latent" || item.reuseIntentLabel === "donor").length,
    references: items.filter((item) => item.reuseIntentLabel === "anchor" || item.roleLabel === "reference" || item.artifact.kind === "bundle").length,
  };
}

function primaryMemoryAction(actions: readonly MemoryReuseAction[]): MemoryReuseAction | null {
  const priority = ["source", "donor", "anchor", "prompt_seed", "advanced_gesture"] as const;
  return priority
    .map((intent) => actions.find((action) => action.intent === intent && action.available))
    .find((action): action is MemoryReuseAction => Boolean(action)) ?? null;
}

function memoryBrowserIndex(artifacts: readonly ArtifactRecord[], context: MemoryBrowserContext) {
  const artifactById = new Map([...(context.artifacts ?? []), ...artifacts].map((artifact) => [artifact.artifact_id, artifact]));
  const familyByArtifactId = new Map<string, ResultFamily>();
  for (const family of context.families ?? []) {
    for (const artifactId of family.artifactIds) {
      familyByArtifactId.set(artifactId, family);
    }
  }
  const recipeById = new Map<string, JobRecord["recipe"]>();
  for (const job of context.jobs ?? []) {
    recipeById.set(job.recipe.recipe_id, job.recipe);
  }
  return { artifactById, familyByArtifactId, recipeById };
}

function branchLabel(artifact: ArtifactRecord, index: ReturnType<typeof memoryBrowserIndex>) {
  const family = index.familyByArtifactId.get(artifact.artifact_id);
  if (family?.familyId.startsWith("prompt-candidates:")) return "prompt candidates";
  if (family) return gestureLabelForOperator(family.operator);
  const recipe = index.recipeById.get(artifact.recipe_id ?? "");
  return recipe ? gestureLabelForOperator(recipe.operator) : null;
}

function lineageLabel(artifact: ArtifactRecord, index: ReturnType<typeof memoryBrowserIndex>) {
  if (artifact.metadata.promoted_from_bundle === true) return "from bundle";
  if (artifact.source_artifact_ids.some((id) => index.artifactById.get(id)?.kind === "bundle")) return "from bundle";
  if (artifact.source_artifact_ids.length) return `${artifact.source_artifact_ids.length} source${artifact.source_artifact_ids.length === 1 ? "" : "s"}`;
  if (artifact.recipe_id) return "gesture take";
  return artifact.kind === "audio" ? "imported sound" : "source material";
}

function sourceLabel(artifact: ArtifactRecord, index: ReturnType<typeof memoryBrowserIndex>) {
  const source = artifact.source_artifact_ids
    .map((id) => index.artifactById.get(id))
    .find((item): item is ArtifactRecord => Boolean(item));
  if (!source) return null;
  return source.label || source.file?.filename || source.artifact_id;
}

function notesPreview(artifact: ArtifactRecord) {
  const note = artifact.notes?.trim();
  if (!note) return null;
  return note.length > 84 ? `${note.slice(0, 81)}...` : note;
}

function compareMemoryItems(left: MemoryBrowserItem, right: MemoryBrowserItem) {
  const leftScore = memoryItemScore(left);
  const rightScore = memoryItemScore(right);
  if (rightScore !== leftScore) return rightScore - leftScore;
  return Date.parse(right.artifact.created_at) - Date.parse(left.artifact.created_at);
}

function memoryItemScore(item: MemoryBrowserItem) {
  let score = 0;
  if (item.usableNow) score += 10;
  if (item.decisionLabel === "keeper") score += 5;
  if (item.roleLabel !== "untyped") score += 3;
  if (item.reuseIntentLabel) score += 3;
  if (item.notesPreview) score += 1;
  return score;
}
