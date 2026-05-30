import { artifactName, formatFamilyStamp } from "./artifactUtils";
import type { ResultFamily } from "./controlPlane";
import { gestureLabelForOperator } from "./gestureModel";
import type { ArtifactRecord } from "./types";

export interface BranchSummary {
  id: string;
  title: string;
  sourceIds: string[];
  gestureLabel: string;
  latestTake: ArtifactRecord | null;
  latestTakeLabel: string;
  takesCount: number;
  status: ResultFamily["status"];
  updatedLabel: string;
  kindLabels: string[];
  inspectRows: [string, string][];
}

export function branchSummaryForFamily(family: ResultFamily, artifacts: readonly ArtifactRecord[]): BranchSummary {
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  const latestTake = family.latestArtifactId ? artifactMap.get(family.latestArtifactId) ?? null : null;
  const sourceIds = Object.values(family.recipe.inputs).filter((value): value is string => typeof value === "string" && value.length > 0);
  const gestureLabel = gestureLabelForOperator(family.operator);
  return {
    id: family.familyId,
    title: branchTitle(family),
    sourceIds,
    gestureLabel,
    latestTake,
    latestTakeLabel: latestTake ? artifactName(latestTake) : "No landed take yet",
    takesCount: family.artifactIds.length,
    status: family.status,
    updatedLabel: formatFamilyStamp(family.updatedAt),
    kindLabels: family.artifactKinds.map((kind) => (kind === "audio" ? "sound" : kind)),
    inspectRows: branchInspectRows(family),
  };
}

export function branchTitle(family: ResultFamily): string {
  if (family.familyId.startsWith("prompt-candidates:")) return "Prompt candidates";
  return `${gestureLabelForOperator(family.operator)} branch`;
}

export function branchMeta(summary: BranchSummary): string {
  const takeWord = `take${summary.takesCount === 1 ? "" : "s"}`;
  const sourceText = summary.sourceIds.length ? `${summary.sourceIds.length} source${summary.sourceIds.length === 1 ? "" : "s"}` : "fresh source";
  return `${summary.takesCount} ${takeWord} · ${summary.gestureLabel} · ${sourceText}`;
}

function branchInspectRows(family: ResultFamily): [string, string][] {
  const rows: [string, string][] = [
    ["backend", family.recipe.backend],
    ["updated", formatFamilyStamp(family.updatedAt)],
  ];
  if (family.recipe.model) rows.push(["model", family.recipe.model]);
  if (family.recipe.seed !== null && family.recipe.seed !== undefined) rows.push(["seed", String(family.recipe.seed)]);
  rows.push(["recipe", family.recipeId], ["jobs", String(family.jobIds.length)]);
  return rows;
}
