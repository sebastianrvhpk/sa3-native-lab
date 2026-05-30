import { artifactName } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

export interface BranchListeningCursor {
  takes: ArtifactRecord[];
  selected: ArtifactRecord | null;
  selectedIndex: number;
  previous: ArtifactRecord | null;
  next: ArtifactRecord | null;
  positionLabel: string;
}

export function branchListeningCursor(
  artifacts: readonly ArtifactRecord[],
  selectedId: string | null,
): BranchListeningCursor {
  const takes = [...artifacts]
    .filter((artifact) => artifact.kind === "audio")
    .sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at));
  const exactIndex = takes.findIndex((artifact) => artifact.artifact_id === selectedId);
  const selectedIndex = exactIndex >= 0 ? exactIndex : takes.length ? 0 : -1;
  return {
    takes,
    selected: selectedIndex >= 0 ? takes[selectedIndex] : null,
    selectedIndex,
    previous: selectedIndex > 0 ? takes[selectedIndex - 1] : null,
    next: selectedIndex >= 0 && selectedIndex < takes.length - 1 ? takes[selectedIndex + 1] : null,
    positionLabel: branchListeningPositionLabel(takes, selectedIndex),
  };
}

export function branchListeningPositionLabel(takes: readonly ArtifactRecord[], selectedIndex: number): string {
  if (!takes.length) return "No playable takes";
  if (selectedIndex < 0) return `${takes.length} playable take${takes.length === 1 ? "" : "s"}`;
  return `${selectedIndex + 1}/${takes.length} · ${artifactName(takes[selectedIndex])}`;
}
