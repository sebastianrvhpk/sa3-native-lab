import { artifactMeta, artifactName, sortNewest } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

export type AuditionSequenceMode = "recent" | "oldest" | "open" | "keepers" | "lineage";

export const auditionSequenceOptions: { value: AuditionSequenceMode; label: string }[] = [
  { value: "recent", label: "Recent" },
  { value: "oldest", label: "Oldest" },
  { value: "open", label: "Open" },
  { value: "keepers", label: "Keepers" },
  { value: "lineage", label: "Lineage" },
];

export interface AuditionStackRow {
  artifactId: string;
  label: string;
  meta: string;
  prompt?: string | null;
  origin: string;
  sequence: string;
}

export function auditionStackRows(
  artifacts: readonly ArtifactRecord[],
  limit = 6,
  mode: AuditionSequenceMode = "recent",
  selectedId: string | null = null,
): AuditionStackRow[] {
  return auditionPlaylist(artifacts, limit, mode, selectedId).map((artifact, index) => ({
    artifactId: artifact.artifact_id,
    label: artifactName(artifact),
    meta: artifactMeta(artifact),
    prompt: artifact.prompt,
    origin: auditionOrigin(artifact),
    sequence: auditionSequenceLabel(artifact, mode, index),
  }));
}

export function auditionPlaylist(
  artifacts: readonly ArtifactRecord[],
  limit = 12,
  mode: AuditionSequenceMode = "recent",
  selectedId: string | null = null,
): ArtifactRecord[] {
  const audio = artifacts.filter((artifact) => artifact.kind === "audio");
  if (mode === "oldest") {
    return [...audio].sort((left, right) => Date.parse(left.created_at) - Date.parse(right.created_at)).slice(0, limit);
  }
  if (mode === "open") {
    return [...audio]
      .sort((left, right) => openRank(left) - openRank(right) || newestCompare(left, right))
      .slice(0, limit);
  }
  if (mode === "keepers") {
    return [...audio]
      .sort((left, right) => decisionRank(left) - decisionRank(right) || newestCompare(left, right))
      .slice(0, limit);
  }
  if (mode === "lineage") {
    return lineagePlaylist(audio, selectedId).slice(0, limit);
  }
  return sortNewest(audio).slice(0, limit);
}

export function auditionCursor(
  artifacts: readonly ArtifactRecord[],
  selectedId: string | null,
  limit = 12,
  mode: AuditionSequenceMode = "recent",
) {
  const playlist = auditionPlaylist(artifacts, limit, mode, selectedId);
  const selectedIndex = playlist.findIndex((artifact) => artifact.artifact_id === selectedId);
  return {
    playlist,
    selectedIndex,
    selected: selectedIndex >= 0 ? playlist[selectedIndex] : null,
    previous: selectedIndex > 0 ? playlist[selectedIndex - 1] : null,
    next: selectedIndex >= 0 && selectedIndex < playlist.length - 1 ? playlist[selectedIndex + 1] : null,
  };
}

export function auditionPositionLabel(
  artifacts: readonly ArtifactRecord[],
  selectedId: string | null,
  limit = 12,
  mode: AuditionSequenceMode = "recent",
): string {
  const cursor = auditionCursor(artifacts, selectedId, limit, mode);
  if (!cursor.playlist.length) return "0 takes";
  if (cursor.selectedIndex < 0) return `${cursor.playlist.length} takes`;
  return `${cursor.selectedIndex + 1}/${cursor.playlist.length}`;
}

export function auditionKeyboardTarget(
  artifacts: readonly ArtifactRecord[],
  selectedId: string | null,
  key: string,
  limit = 12,
  mode: AuditionSequenceMode = "recent",
): ArtifactRecord | null {
  const cursor = auditionCursor(artifacts, selectedId, limit, mode);
  if (!cursor.playlist.length) return null;
  if (key === "ArrowUp" || key === "ArrowLeft") return cursor.previous;
  if (key === "ArrowDown" || key === "ArrowRight") return cursor.next;
  if (key === "Home") return cursor.playlist[0] ?? null;
  if (key === "End") return cursor.playlist[cursor.playlist.length - 1] ?? null;
  if (!cursor.selected && (key === "Enter" || key === " ")) return cursor.playlist[0] ?? null;
  return null;
}

function auditionSequenceLabel(artifact: ArtifactRecord, mode: AuditionSequenceMode, index: number): string {
  if (mode === "open" && !listeningDecision(artifact)) return "open";
  if (mode === "keepers") return listeningDecision(artifact) ?? "open";
  if (mode === "lineage" && index === 0) return "anchor";
  if (mode === "lineage" && artifact.source_artifact_ids.length) return "source-linked";
  return mode;
}

function auditionOrigin(artifact: ArtifactRecord): string {
  const origin = artifact.metadata.generation_origin;
  if (origin === "prompt_search_candidate") return "prompt take";
  const operator = artifact.metadata.operator;
  if (typeof operator === "string" && operator.trim()) return operator.replace(/^experiment\./, "").replace(/^generate\./, "").replaceAll("_", " ");
  return artifact.recipe_id ? "recipe output" : "source audio";
}

function lineagePlaylist(audio: readonly ArtifactRecord[], selectedId: string | null): ArtifactRecord[] {
  const recent = sortNewest(audio);
  if (!selectedId) return recent;
  const selected = audio.find((artifact) => artifact.artifact_id === selectedId);
  if (!selected) return recent;
  const sourceIds = new Set(selected.source_artifact_ids);
  const childIds = new Set(audio.filter((artifact) => artifact.source_artifact_ids.includes(selectedId)).map((artifact) => artifact.artifact_id));
  const related = recent.filter(
    (artifact) =>
      artifact.artifact_id === selectedId
      || sourceIds.has(artifact.artifact_id)
      || childIds.has(artifact.artifact_id)
      || artifact.source_artifact_ids.some((sourceId) => sourceIds.has(sourceId)),
  );
  const rest = recent.filter((artifact) => !related.some((item) => item.artifact_id === artifact.artifact_id));
  return [...related.sort((left, right) => lineageRank(left, selectedId, sourceIds, childIds) - lineageRank(right, selectedId, sourceIds, childIds) || newestCompare(left, right)), ...rest];
}

function lineageRank(artifact: ArtifactRecord, selectedId: string, sourceIds: Set<string>, childIds: Set<string>): number {
  if (artifact.artifact_id === selectedId) return 0;
  if (sourceIds.has(artifact.artifact_id)) return 1;
  if (childIds.has(artifact.artifact_id)) return 2;
  return 3;
}

function listeningDecision(artifact: ArtifactRecord): string | null {
  const decision = artifact.metadata.listening_decision;
  return typeof decision === "string" && decision.trim() ? decision : null;
}

function openRank(artifact: ArtifactRecord): number {
  return listeningDecision(artifact) ? 1 : 0;
}

function decisionRank(artifact: ArtifactRecord): number {
  const decision = listeningDecision(artifact);
  if (decision === "keeper") return 0;
  if (decision === "maybe") return 1;
  if (!decision) return 2;
  if (decision === "rejected") return 3;
  return 4;
}

function newestCompare(left: ArtifactRecord, right: ArtifactRecord): number {
  return Date.parse(right.created_at) - Date.parse(left.created_at);
}
