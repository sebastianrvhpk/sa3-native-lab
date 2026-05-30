import { useState } from "react";
import { SkipBack, SkipForward } from "lucide-react";

import {
  auditionCursor,
  auditionKeyboardTarget,
  auditionPositionLabel,
  auditionSequenceOptions,
  auditionStackRows,
  type AuditionSequenceMode,
} from "./auditionStack";
import { AudioDeck } from "./audioDeck";
import { ListeningDecisionBadge } from "./listeningDecision";
import type { ArtifactRecord } from "./types";

export function AuditionStackPanel({
  artifacts,
  selectedId,
  apiBase,
  onSelect,
  onCompare,
}: {
  artifacts: ArtifactRecord[];
  selectedId: string | null;
  apiBase: string;
  onSelect: (artifactId: string) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
}) {
  const [sequenceMode, setSequenceMode] = useState<AuditionSequenceMode>("recent");
  const rows = auditionStackRows(artifacts, 8, sequenceMode, selectedId);
  const cursor = auditionCursor(artifacts, selectedId, 8, sequenceMode);
  const position = auditionPositionLabel(artifacts, selectedId, 8, sequenceMode);
  if (!rows.length) return null;
  const moveSelection = (key: string) => {
    const target = auditionKeyboardTarget(artifacts, selectedId, key, 8, sequenceMode);
    if (!target) return false;
    onSelect(target.artifact_id);
    return true;
  };
  return (
    <div
      className="audition-stack"
      aria-label="Session audition stack"
      tabIndex={0}
      onKeyDown={(event) => {
        if (!moveSelection(event.key)) return;
        event.preventDefault();
      }}
    >
      <div className="audition-stack-head">
        <div>
          <span className="eyebrow">Audition</span>
          <strong>{position}</strong>
        </div>
        <label className="audition-sequence">
          <span>Sequence</span>
          <select value={sequenceMode} onChange={(event) => setSequenceMode(event.target.value as AuditionSequenceMode)}>
            {auditionSequenceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="audition-transport">
          <button type="button" disabled={!cursor.previous} onClick={() => cursor.previous && onSelect(cursor.previous.artifact_id)} title="Previous take">
            <SkipBack size={14} />
          </button>
          <button type="button" disabled={!cursor.next} onClick={() => cursor.next && onSelect(cursor.next.artifact_id)} title="Next take">
            <SkipForward size={14} />
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("a", cursor.selected.artifact_id)} title="Send selected take to A">
            A
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("b", cursor.selected.artifact_id)} title="Send selected take to B">
            B
          </button>
        </div>
      </div>
      {rows.map((row) => {
        const artifact = artifacts.find((item) => item.artifact_id === row.artifactId);
        if (!artifact) return null;
        return (
          <article key={row.artifactId} className={selectedId === row.artifactId ? "selected" : ""}>
            <button type="button" onClick={() => onSelect(row.artifactId)} title={row.prompt ?? row.label}>
              <span>{row.label}</span>
              <small>{row.sequence} · {row.origin} · {row.meta}</small>
              <ListeningDecisionBadge artifact={artifact} />
            </button>
            <AudioDeck artifact={artifact} apiBase={apiBase} compact />
            <div className="audition-stack-actions">
              <button type="button" onClick={() => onCompare("a", row.artifactId)} title="Send take to comparison slot A">A</button>
              <button type="button" onClick={() => onCompare("b", row.artifactId)} title="Send take to comparison slot B">B</button>
            </div>
          </article>
        );
      })}
    </div>
  );
}
