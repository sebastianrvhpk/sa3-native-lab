import type { ResultFamily } from "./controlPlane";
import { listeningDecision, type ListeningDecision } from "./listeningDecision";
import type { ArtifactRecord, JobRecord } from "./types";

export type WorkspaceFocusTone = "listen" | "run" | "recover" | "archive";
export type WorkspacePulseTone = "take" | "family" | "job" | "decision" | "archive";

export interface SessionWorkspaceInput {
  artifacts: readonly ArtifactRecord[];
  archivedArtifacts: readonly ArtifactRecord[];
  jobs: readonly JobRecord[];
  archivedJobs: readonly JobRecord[];
  families: readonly ResultFamily[];
  runningJobs: readonly JobRecord[];
  selectedId?: string | null;
}

export interface DecisionCounts {
  keeper: number;
  maybe: number;
  rejected: number;
  undecided: number;
}

export interface SessionWorkspaceSummary {
  takes: number;
  audioTakes: number;
  latentArtifacts: number;
  bundleArtifacts: number;
  familyCount: number;
  sessionJobs: number;
  archivedArtifacts: number;
  archivedJobs: number;
  archiveItems: number;
  activeJobs: number;
  queuedJobs: number;
  runningJobs: number;
  failedJobs: number;
  succeededJobs: number;
  decisions: DecisionCounts;
  selectedInArchive: boolean;
  latestArtifactId: string | null;
  latestAudioId: string | null;
  openAudioTakes: number;
}

export interface WorkspacePulseRow {
  key: string;
  label: string;
  value: string;
  detail: string;
  tone: WorkspacePulseTone;
}

export interface WorkspaceFocus {
  label: string;
  detail: string;
  tone: WorkspaceFocusTone;
  artifactId?: string | null;
}

export function summarizeSessionWorkspace(input: SessionWorkspaceInput): SessionWorkspaceSummary {
  const artifacts = [...input.artifacts];
  const archivedArtifacts = [...input.archivedArtifacts];
  const activeJobs = input.runningJobs.filter((job) => job.status === "queued" || job.status === "running");
  const decisions = decisionCounts(artifacts);
  const audioArtifacts = artifacts.filter((artifact) => artifact.kind === "audio");
  const archivedIds = new Set(archivedArtifacts.map((artifact) => artifact.artifact_id));
  return {
    takes: artifacts.length,
    audioTakes: audioArtifacts.length,
    latentArtifacts: artifacts.filter((artifact) => artifact.kind === "latent").length,
    bundleArtifacts: artifacts.filter((artifact) => artifact.kind === "bundle").length,
    familyCount: input.families.length,
    sessionJobs: input.jobs.length,
    archivedArtifacts: archivedArtifacts.length,
    archivedJobs: input.archivedJobs.length,
    archiveItems: archivedArtifacts.length + input.archivedJobs.length,
    activeJobs: activeJobs.length,
    queuedJobs: activeJobs.filter((job) => job.status === "queued").length,
    runningJobs: activeJobs.filter((job) => job.status === "running").length,
    failedJobs: input.jobs.filter((job) => job.status === "failed").length,
    succeededJobs: input.jobs.filter((job) => job.status === "succeeded").length,
    decisions,
    selectedInArchive: Boolean(input.selectedId && archivedIds.has(input.selectedId)),
    latestArtifactId: newestArtifactId(artifacts),
    latestAudioId: newestArtifactId(audioArtifacts),
    openAudioTakes: audioArtifacts.filter((artifact) => !listeningDecision(artifact)).length,
  };
}

export function workspacePulseRows(summary: SessionWorkspaceSummary): WorkspacePulseRow[] {
  return [
    {
      key: "takes",
      label: "Takes",
      value: String(summary.audioTakes),
      detail: `${summary.takes} active artifact${summary.takes === 1 ? "" : "s"}`,
      tone: "take",
    },
    {
      key: "families",
      label: "Families",
      value: String(summary.familyCount),
      detail: "recipe groups",
      tone: "family",
    },
    {
      key: "jobs",
      label: "Jobs",
      value: summary.activeJobs ? String(summary.activeJobs) : String(summary.sessionJobs),
      detail: summary.activeJobs ? `${summary.runningJobs} running, ${summary.queuedJobs} queued` : `${summary.succeededJobs} done, ${summary.failedJobs} failed`,
      tone: "job",
    },
    {
      key: "decisions",
      label: "Decisions",
      value: `${summary.decisions.keeper}/${summary.audioTakes}`,
      detail: `${summary.openAudioTakes} open audio take${summary.openAudioTakes === 1 ? "" : "s"}`,
      tone: "decision",
    },
    {
      key: "archive",
      label: "Archive",
      value: String(summary.archiveItems),
      detail: `${summary.archivedArtifacts} artifacts, ${summary.archivedJobs} jobs`,
      tone: "archive",
    },
  ];
}

export function workspaceFocus(summary: SessionWorkspaceSummary): WorkspaceFocus {
  if (summary.activeJobs > 0) {
    return {
      label: "Monitor run",
      detail: `${summary.runningJobs} running, ${summary.queuedJobs} queued`,
      tone: "run",
    };
  }
  if (summary.failedJobs > 0) {
    return {
      label: "Review failure",
      detail: `${summary.failedJobs} failed job${summary.failedJobs === 1 ? "" : "s"} in this session`,
      tone: "run",
    };
  }
  if (summary.selectedInArchive) {
    return {
      label: "Recover take",
      detail: "selected from archive",
      tone: "recover",
    };
  }
  if (summary.latestAudioId && summary.openAudioTakes > 0) {
    return {
      label: "Listen next",
      detail: `${summary.openAudioTakes} undecided audio take${summary.openAudioTakes === 1 ? "" : "s"}`,
      tone: "listen",
      artifactId: summary.latestAudioId,
    };
  }
  if (summary.takes >= 12 && summary.activeJobs === 0) {
    return {
      label: "Archive ready",
      detail: `${summary.takes} active artifacts can be cleared into a new session`,
      tone: "archive",
    };
  }
  return {
    label: "Ready",
    detail: summary.familyCount ? "fork, replay, or run the next recipe" : "run a recipe to begin a family",
    tone: "listen",
    artifactId: summary.latestAudioId,
  };
}

function decisionCounts(artifacts: readonly ArtifactRecord[]): DecisionCounts {
  const counts: Record<ListeningDecision | "undecided", number> = {
    keeper: 0,
    maybe: 0,
    rejected: 0,
    undecided: 0,
  };
  for (const artifact of artifacts) {
    if (artifact.kind !== "audio") continue;
    const decision = listeningDecision(artifact);
    counts[decision ?? "undecided"] += 1;
  }
  return counts;
}

function newestArtifactId(artifacts: readonly ArtifactRecord[]): string | null {
  let newest: ArtifactRecord | null = null;
  for (const artifact of artifacts) {
    if (!newest || Date.parse(artifact.created_at) > Date.parse(newest.created_at)) {
      newest = artifact;
    }
  }
  return newest?.artifact_id ?? null;
}
