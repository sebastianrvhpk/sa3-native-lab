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
  return sortNewest(artifacts.filter((artifact) => artifact.kind === "audio")).slice(0, limit).map((artifact) => ({
    artifactId: artifact.artifact_id,
    label: artifactName(artifact),
    meta: artifactMeta(artifact),
    prompt: artifact.prompt,
    origin: auditionOrigin(artifact),
  }));
}

function auditionOrigin(artifact: ArtifactRecord): string {
  const origin = artifact.metadata.generation_origin;
  if (origin === "prompt_search_candidate") return "prompt take";
  const operator = artifact.metadata.operator;
  if (typeof operator === "string" && operator.trim()) return operator.replace(/^experiment\./, "").replace(/^generate\./, "").replaceAll("_", " ");
  return artifact.recipe_id ? "recipe output" : "source audio";
}
