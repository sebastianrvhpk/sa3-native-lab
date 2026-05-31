import { useEffect, useState } from "react";
import { Check, CircleHelp, X } from "lucide-react";

import type { ArtifactAnnotationPayload } from "./api";
import type { ArtifactRecord } from "./types";

export type ListeningDecision = "keeper" | "maybe" | "rejected";

export type ListeningDecisionSummaryKey = ListeningDecision | "open";

export interface ListeningDecisionSummaryEntry {
  key: ListeningDecisionSummaryKey;
  label: string;
  count: number;
}

export interface ListeningDecisionSummary {
  total: number;
  decided: number;
  undecided: number;
  keeper: number;
  maybe: number;
  rejected: number;
  entries: ListeningDecisionSummaryEntry[];
}

const DECISION_TAGS = new Set(["keeper", "maybe", "rejected"]);

export function listeningDecision(artifact: ArtifactRecord): ListeningDecision | null {
  const value = artifact.metadata.listening_decision;
  return value === "keeper" || value === "maybe" || value === "rejected" ? value : null;
}

export function listeningDecisionLabel(decision: ListeningDecision | null): string {
  if (decision === "keeper") return "keeper";
  if (decision === "maybe") return "maybe";
  if (decision === "rejected") return "rejected";
  return "undecided";
}

export function listeningDecisionSummary(artifacts: readonly ArtifactRecord[]): ListeningDecisionSummary {
  const playable = artifacts.filter((artifact) => artifact.kind === "audio");
  const counts = playable.reduce(
    (items, artifact) => {
      const decision = listeningDecision(artifact);
      if (decision) items[decision] += 1;
      else items.open += 1;
      return items;
    },
    { keeper: 0, maybe: 0, rejected: 0, open: 0 },
  );
  const entries: ListeningDecisionSummaryEntry[] = [
    { key: "keeper", label: "keeper", count: counts.keeper },
    { key: "maybe", label: "maybe", count: counts.maybe },
    { key: "rejected", label: "reject", count: counts.rejected },
    { key: "open", label: "open", count: counts.open },
  ].filter((entry) => entry.count > 0);
  return {
    total: playable.length,
    decided: playable.length - counts.open,
    undecided: counts.open,
    keeper: counts.keeper,
    maybe: counts.maybe,
    rejected: counts.rejected,
    entries,
  };
}

export function listeningDecisionPayload({
  artifact,
  decision,
  note,
  source,
  now = new Date().toISOString(),
}: {
  artifact: ArtifactRecord;
  decision: ListeningDecision;
  note?: string;
  source: string;
  now?: string;
}): ArtifactAnnotationPayload {
  const tags = artifact.tags.filter((tag) => !DECISION_TAGS.has(tag));
  tags.push(decision);
  const normalizedNote = note?.trim();
  return {
    notes: normalizedNote || undefined,
    tags,
    metadata: {
      listening_decision: decision,
      listening_decision_source: source,
      listening_decision_at: now,
      ...(normalizedNote ? { listening_decision_note: normalizedNote } : {}),
    },
  };
}

export function ListeningDecisionBadge({ artifact }: { artifact: ArtifactRecord }) {
  const decision = listeningDecision(artifact);
  if (!decision) return null;
  return <span className={`listening-decision-badge ${decision}`}>{listeningDecisionLabel(decision)}</span>;
}

export function ListeningDecisionSummaryChips({
  artifacts,
  ariaLabel = "Listening decision summary",
}: {
  artifacts: readonly ArtifactRecord[];
  ariaLabel?: string;
}) {
  const summary = listeningDecisionSummary(artifacts);
  if (!summary.total) return null;
  return (
    <div className="decision-summary" aria-label={ariaLabel}>
      {summary.entries.map((entry) => (
        <i key={entry.key} className={entry.key}>
          {entry.count} {entry.label}
        </i>
      ))}
    </div>
  );
}

export function ListeningDecisionControls({
  artifact,
  source,
  compact = false,
  onDecide,
}: {
  artifact: ArtifactRecord;
  source: string;
  compact?: boolean;
  onDecide: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
}) {
  const currentDecision = listeningDecision(artifact);
  const [note, setNote] = useState(artifact.notes ?? "");

  useEffect(() => {
    setNote(artifact.notes ?? "");
  }, [artifact.artifact_id, artifact.notes]);

  const decide = (decision: ListeningDecision) => {
    onDecide(artifact.artifact_id, listeningDecisionPayload({ artifact, decision, note, source }));
  };

  return (
    <div className={`listening-decision-controls ${compact ? "compact" : ""}`} aria-label="Listening decision">
      <div className="listening-decision-buttons">
        <button type="button" className={currentDecision === "keeper" ? "active keeper" : ""} onClick={() => decide("keeper")}>
          <Check aria-hidden="true" size={13} />
          Keep
        </button>
        <button type="button" className={currentDecision === "maybe" ? "active maybe" : ""} onClick={() => decide("maybe")}>
          <CircleHelp aria-hidden="true" size={13} />
          Maybe
        </button>
        <button type="button" className={currentDecision === "rejected" ? "active rejected" : ""} onClick={() => decide("rejected")}>
          <X aria-hidden="true" size={13} />
          Reject
        </button>
      </div>
      {!compact ? (
        <input
          aria-label="Listening note"
          value={note}
          onChange={(event) => setNote(event.target.value)}
          placeholder="listening note"
        />
      ) : null}
    </div>
  );
}
