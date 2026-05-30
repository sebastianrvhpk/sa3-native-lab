import { artifactName } from "./artifactUtils";
import type { ResultFamily } from "./controlPlane";
import { shortOperatorName } from "./jobUtils";
import type { ArtifactRecord, JobRecord } from "./types";

export type LineageNodeKind = "origin" | "source" | "recipe" | "job" | "current" | "family" | "anchor";

export interface LineageNode {
  id: string;
  kind: LineageNodeKind;
  label: string;
  title: string;
  artifactId?: string;
}

export interface CompareSlots {
  a: string | null;
  b: string | null;
}

export function artifactLineageModel({
  artifact,
  sources,
  jobs,
  families,
  compare,
}: {
  artifact: ArtifactRecord;
  sources: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
}): LineageNode[] {
  const nodes: LineageNode[] = [];
  if (sources.length) {
    for (const source of sources.slice(0, 3)) {
      nodes.push({
        id: `source:${source.artifact_id}`,
        kind: "source",
        label: artifactName(source),
        title: source.artifact_id,
        artifactId: source.artifact_id,
      });
    }
  } else {
    nodes.push({ id: "origin", kind: "origin", label: "origin", title: "Imported or root artifact" });
  }

  const job = jobs.find((item) => item.recipe.recipe_id === artifact.recipe_id || item.artifact_ids.includes(artifact.artifact_id)) ?? null;
  if (job) {
    nodes.push({
      id: `recipe:${job.recipe.recipe_id}`,
      kind: "recipe",
      label: shortOperatorName(job.recipe.operator),
      title: job.recipe.recipe_id,
    });
    nodes.push({
      id: `job:${job.job_id}`,
      kind: "job",
      label: job.phase || job.status,
      title: `${job.status} · ${job.job_id}`,
    });
  } else if (artifact.recipe_id) {
    nodes.push({
      id: `recipe:${artifact.recipe_id}`,
      kind: "recipe",
      label: "recipe",
      title: artifact.recipe_id,
    });
  }

  nodes.push({
    id: `current:${artifact.artifact_id}`,
    kind: "current",
    label: artifact.kind,
    title: artifact.artifact_id,
    artifactId: artifact.artifact_id,
  });

  const family = families.find((item) => item.artifactIds.includes(artifact.artifact_id) || item.recipeId === artifact.recipe_id) ?? null;
  if (family) {
    nodes.push({
      id: `family:${family.familyId}`,
      kind: "family",
      label: `${family.artifactIds.length} take${family.artifactIds.length === 1 ? "" : "s"}`,
      title: family.familyId,
    });
  }

  const slot = compare.a === artifact.artifact_id ? "Anchor" : compare.b === artifact.artifact_id ? "Source" : null;
  if (slot) {
    nodes.push({
      id: `compare:${slot}`,
      kind: "anchor",
      label: slot,
      title: `Pinned ${slot.toLowerCase()}`,
    });
  }

  return nodes;
}
