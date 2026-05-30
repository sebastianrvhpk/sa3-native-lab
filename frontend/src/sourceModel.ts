import {
  memoryActionsForArtifact,
  memoryReuseIntentFromArtifact,
  memoryRoleFromArtifact,
  memoryRoleLabel,
  type MemoryReuseAction,
  type MemoryReuseIntent,
  type MemoryRole,
} from "./memoryModel";
import { artifactMeta, artifactName, sortNewest } from "./artifactUtils";
import type { ArtifactRecord } from "./types";

export type ProductSourceKind = "sound" | "latent" | "bundle" | "material";
export type ProductSourceRole = "current" | "anchor" | "source" | "donor" | "remembered" | "imported" | "take" | "bundle" | "bundle_derived";

export interface ProductSource {
  id: string;
  artifact: ArtifactRecord;
  artifactId: string;
  label: string;
  detail: string;
  kind: ProductSourceKind;
  roles: ProductSourceRole[];
  roleLabels: string[];
  memoryRole: MemoryRole | null;
  reuseIntent: MemoryReuseIntent | null;
  actions: MemoryReuseAction[];
  isCurrent: boolean;
  isRemembered: boolean;
}

export interface ProductSourceContext {
  currentArtifactId?: string | null;
  activeSessionId?: string | null;
  anchorArtifactId?: string | null;
  sourceArtifactId?: string | null;
  donorArtifactId?: string | null;
}

export function buildProductSources(
  artifacts: readonly ArtifactRecord[],
  context: ProductSourceContext = {},
): ProductSource[] {
  return sortNewest(dedupeArtifacts(artifacts)).map((artifact) => productSourceFromArtifact(artifact, context));
}

export function productSourceFromArtifact(
  artifact: ArtifactRecord,
  context: ProductSourceContext = {},
): ProductSource {
  const memoryRole = memoryRoleFromArtifact(artifact);
  const reuseIntent = memoryReuseIntentFromArtifact(artifact);
  const roles = productSourceRoles(artifact, context);
  return {
    id: artifact.artifact_id,
    artifact,
    artifactId: artifact.artifact_id,
    label: artifactName(artifact),
    detail: sourceDetail(artifact),
    kind: sourceKind(artifact),
    roles,
    roleLabels: roles.map((role) => productSourceRoleLabel(role, memoryRole, reuseIntent)),
    memoryRole,
    reuseIntent,
    actions: memoryActionsForArtifact(artifact, { activeSessionId: context.activeSessionId }),
    isCurrent: artifact.artifact_id === context.currentArtifactId,
    isRemembered: sourceIsRemembered(artifact, context.activeSessionId),
  };
}

export function productSourceRoleLabel(
  role: ProductSourceRole,
  memoryRole: MemoryRole | null = null,
  reuseIntent: MemoryReuseIntent | null = null,
): string {
  if (role === "current") return "current";
  if (role === "anchor") return "anchor";
  if (role === "source") return "source";
  if (role === "donor") return "donor";
  if (role === "remembered") return memoryRole ? `memory: ${memoryRoleLabel(memoryRole)}` : "memory";
  if (role === "imported") return "imported";
  if (role === "bundle") return reuseIntent ? `bundle: ${reuseIntent.replaceAll("_", " ")}` : "bundle";
  if (role === "bundle_derived") return "bundle audio";
  return "take";
}

function productSourceRoles(artifact: ArtifactRecord, context: ProductSourceContext): ProductSourceRole[] {
  const roles: ProductSourceRole[] = [];
  if (artifact.artifact_id === context.currentArtifactId) roles.push("current");
  if (artifact.artifact_id === context.anchorArtifactId) roles.push("anchor");
  if (artifact.artifact_id === context.sourceArtifactId) roles.push("source");
  if (artifact.artifact_id === context.donorArtifactId) roles.push("donor");
  if (sourceIsRemembered(artifact, context.activeSessionId)) roles.push("remembered");
  if (artifact.kind === "bundle") roles.push("bundle");
  if (artifact.metadata.promoted_from_bundle === true) roles.push("bundle_derived");
  if (!artifact.recipe_id && artifact.kind === "audio") roles.push("imported");
  if (artifact.recipe_id) roles.push("take");
  return dedupeRoles(roles);
}

function sourceKind(artifact: ArtifactRecord): ProductSourceKind {
  if (artifact.kind === "audio") return "sound";
  if (artifact.kind === "latent") return "latent";
  if (artifact.kind === "bundle") return "bundle";
  return "material";
}

function sourceDetail(artifact: ArtifactRecord): string {
  if (artifact.kind === "audio") return artifactMeta(artifact);
  if (artifact.kind === "latent") return artifact.latent?.shape.length ? `${artifact.latent.shape.join(" x ")} latent` : artifactMeta(artifact);
  if (artifact.kind === "bundle") return artifactMeta(artifact);
  return artifact.kind;
}

function sourceIsRemembered(artifact: ArtifactRecord, activeSessionId?: string | null): boolean {
  if (!activeSessionId) return artifact.session_id === null;
  return artifact.session_id !== activeSessionId;
}

function dedupeRoles(roles: ProductSourceRole[]): ProductSourceRole[] {
  return [...new Set(roles)];
}

function dedupeArtifacts(artifacts: readonly ArtifactRecord[]): ArtifactRecord[] {
  const seen = new Set<string>();
  return artifacts.filter((artifact) => {
    if (seen.has(artifact.artifact_id)) return false;
    seen.add(artifact.artifact_id);
    return true;
  });
}
