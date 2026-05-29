import { artifactMeta, artifactName, sortNewest } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

export interface AuditionStackRow {
  artifactId: string;
  label: string;
  meta: string;
  prompt?: string | null;
  origin: string;
}

export function auditionStackRows(artifacts: readonly ArtifactRecord[], limit = 6): AuditionStackRow[] {
  return auditionPlaylist(artifacts, limit).map((artifact) => ({
    artifactId: artifact.artifact_id,
    label: artifactName(artifact),
    meta: artifactMeta(artifact),
    prompt: artifact.prompt,
    origin: auditionOrigin(artifact),
  }));
}

export function auditionPlaylist(artifacts: readonly ArtifactRecord[], limit = 12): ArtifactRecord[] {
  return sortNewest(artifacts.filter((artifact) => artifact.kind === "audio")).slice(0, limit);
}

export function auditionCursor(artifacts: readonly ArtifactRecord[], selectedId: string | null, limit = 12) {
  const playlist = auditionPlaylist(artifacts, limit);
  const selectedIndex = playlist.findIndex((artifact) => artifact.artifact_id === selectedId);
  return {
    playlist,
    selectedIndex,
    selected: selectedIndex >= 0 ? playlist[selectedIndex] : null,
    previous: selectedIndex > 0 ? playlist[selectedIndex - 1] : null,
    next: selectedIndex >= 0 && selectedIndex < playlist.length - 1 ? playlist[selectedIndex + 1] : null,
  };
}

export function auditionPositionLabel(artifacts: readonly ArtifactRecord[], selectedId: string | null, limit = 12): string {
  const cursor = auditionCursor(artifacts, selectedId, limit);
  if (!cursor.playlist.length) return "0 takes";
  if (cursor.selectedIndex < 0) return `${cursor.playlist.length} takes`;
  return `${cursor.selectedIndex + 1}/${cursor.playlist.length}`;
}

function auditionOrigin(artifact: ArtifactRecord): string {
  const origin = artifact.metadata.generation_origin;
  if (origin === "prompt_search_candidate") return "prompt take";
  const operator = artifact.metadata.operator;
  if (typeof operator === "string" && operator.trim()) return operator.replace(/^experiment\./, "").replace(/^generate\./, "").replaceAll("_", " ");
  return artifact.recipe_id ? "recipe output" : "source audio";
}
