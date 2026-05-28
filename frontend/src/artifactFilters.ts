import type { ResultFamily } from "./controlPlane";
import { listeningDecision, type ListeningDecision } from "./listeningDecision";
import type { ArtifactKind, ArtifactRecord, JobRecord, OperatorName, Recipe } from "./types";

export type ArtifactDecisionFilter = "all" | ListeningDecision | "undecided";
export type ArtifactKindFilter = "all" | ArtifactKind;
export type ArtifactLineageFilter = "all" | "source" | "derived" | "has_sources";

export interface ArtifactFilterState {
  query: string;
  tag: string;
  decision: ArtifactDecisionFilter;
  kind: ArtifactKindFilter;
  model: string;
  operator: string;
  familyId: string;
  lineage: ArtifactLineageFilter;
}

export interface ArtifactFilterOption {
  value: string;
  label: string;
  count: number;
}

export interface ArtifactFilterContext {
  jobs?: JobRecord[];
  families?: ResultFamily[];
}

export const emptyArtifactFilters: ArtifactFilterState = {
  query: "",
  tag: "",
  decision: "all",
  kind: "all",
  model: "",
  operator: "",
  familyId: "",
  lineage: "all",
};

export function artifactFiltersActive(filters: ArtifactFilterState): boolean {
  return Boolean(
    filters.query.trim()
      || filters.tag
      || filters.decision !== "all"
      || filters.kind !== "all"
      || filters.model
      || filters.operator
      || filters.familyId
      || filters.lineage !== "all",
  );
}

export function filterArtifacts(
  artifacts: ArtifactRecord[],
  filters: ArtifactFilterState,
  context: ArtifactFilterContext = {},
): ArtifactRecord[] {
  const index = artifactFilterIndex(artifacts, context);
  const needle = filters.query.trim().toLowerCase();
  const requestedTag = filters.tag.toLowerCase();
  return artifacts.filter((artifact) => {
    const recipe = index.recipeById.get(artifact.recipe_id ?? "");
    const family = index.familyByArtifactId.get(artifact.artifact_id);
    if (requestedTag && !artifact.tags.some((tag) => tag.toLowerCase() === requestedTag)) return false;
    if (!artifactDecisionMatches(artifact, filters.decision)) return false;
    if (filters.kind !== "all" && artifact.kind !== filters.kind) return false;
    if (filters.model && artifactModel(artifact, recipe) !== filters.model) return false;
    if (filters.operator && artifactOperator(artifact, recipe) !== filters.operator) return false;
    if (filters.familyId && family?.familyId !== filters.familyId) return false;
    if (!artifactLineageMatches(artifact, filters.lineage)) return false;
    if (needle && !artifactSearchText(artifact, recipe, family).includes(needle)) return false;
    return true;
  });
}

export function artifactFilterOptions(
  artifacts: ArtifactRecord[],
  context: ArtifactFilterContext = {},
) {
  const index = artifactFilterIndex(artifacts, context);
  const tags = new Map<string, number>();
  const decisions = new Map<string, number>();
  const kinds = new Map<string, number>();
  const models = new Map<string, number>();
  const operators = new Map<string, number>();
  const families = new Map<string, { label: string; count: number }>();

  for (const artifact of artifacts) {
    const recipe = index.recipeById.get(artifact.recipe_id ?? "");
    const family = index.familyByArtifactId.get(artifact.artifact_id);
    increment(kinds, artifact.kind);
    for (const tag of artifact.tags) increment(tags, tag);
    const decision = listeningDecision(artifact);
    if (decision) increment(decisions, decision);
    const model = artifactModel(artifact, recipe);
    if (model) increment(models, model);
    const operator = artifactOperator(artifact, recipe);
    if (operator) increment(operators, operator);
    if (family) {
      const existing = families.get(family.familyId);
      families.set(family.familyId, {
        label: familyFilterLabel(family),
        count: (existing?.count ?? 0) + 1,
      });
    }
  }

  return {
    tags: mapOptions(tags),
    decisions: mapOptions(decisions),
    kinds: mapOptions(kinds),
    models: mapOptions(models),
    operators: mapOptions(operators),
    families: [...families.entries()]
      .map(([value, item]) => ({ value, label: item.label, count: item.count }))
      .sort(compareOptions),
  };
}

export function artifactModel(artifact: ArtifactRecord, recipe?: Recipe): string {
  const metadataModel = primitiveString(artifact.metadata.model);
  return metadataModel || recipe?.model || "";
}

export function artifactOperator(artifact: ArtifactRecord, recipe?: Recipe): OperatorName | string {
  const metadataOperator = primitiveString(artifact.metadata.operator);
  return metadataOperator || recipe?.operator || "";
}

function artifactFilterIndex(artifacts: ArtifactRecord[], context: ArtifactFilterContext) {
  const recipeById = new Map<string, Recipe>();
  for (const job of context.jobs ?? []) {
    recipeById.set(job.recipe.recipe_id, job.recipe);
  }
  const artifactIds = new Set(artifacts.map((artifact) => artifact.artifact_id));
  const familyByArtifactId = new Map<string, ResultFamily>();
  for (const family of context.families ?? []) {
    for (const artifactId of family.artifactIds) {
      if (artifactIds.has(artifactId)) familyByArtifactId.set(artifactId, family);
    }
    recipeById.set(family.recipe.recipe_id, family.recipe);
  }
  return { recipeById, familyByArtifactId };
}

function artifactDecisionMatches(artifact: ArtifactRecord, filter: ArtifactDecisionFilter) {
  if (filter === "all") return true;
  const decision = listeningDecision(artifact);
  return filter === "undecided" ? !decision : decision === filter;
}

function artifactLineageMatches(artifact: ArtifactRecord, filter: ArtifactLineageFilter) {
  if (filter === "all") return true;
  if (filter === "source") return !artifact.recipe_id && artifact.source_artifact_ids.length === 0;
  if (filter === "derived") return Boolean(artifact.recipe_id);
  return artifact.source_artifact_ids.length > 0;
}

function artifactSearchText(artifact: ArtifactRecord, recipe: Recipe | undefined, family: ResultFamily | undefined) {
  return [
    artifact.artifact_id,
    artifact.kind,
    artifact.label,
    artifact.prompt,
    artifact.notes,
    artifact.file?.filename,
    artifactModel(artifact, recipe),
    artifactOperator(artifact, recipe),
    recipe?.backend,
    family?.familyId,
    family ? familyFilterLabel(family) : "",
    ...artifact.tags,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function familyFilterLabel(family: ResultFamily) {
  if (family.familyId.startsWith("prompt-candidates:")) return "prompt candidates";
  return String(family.operator).replace(/^experiment\./, "").replace(/^generate\./, "").replaceAll("_", " ");
}

function primitiveString(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : "";
}

function increment(map: Map<string, number>, value: string) {
  if (!value) return;
  map.set(value, (map.get(value) ?? 0) + 1);
}

function mapOptions(map: Map<string, number>): ArtifactFilterOption[] {
  return [...map.entries()]
    .map(([value, count]) => ({ value, label: value, count }))
    .sort(compareOptions);
}

function compareOptions(a: ArtifactFilterOption, b: ArtifactFilterOption) {
  if (b.count !== a.count) return b.count - a.count;
  return a.label.localeCompare(b.label);
}
