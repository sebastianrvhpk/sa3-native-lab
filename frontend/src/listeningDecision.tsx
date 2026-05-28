import { useEffect, useState } from "react";
import { Check, CircleHelp, X } from "lucide-react";

import type { ArtifactAnnotationPayload } from "./api";
import type { ArtifactRecord } from "./types";

export type ListeningDecision = "keeper" | "maybe" | "rejected";

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
