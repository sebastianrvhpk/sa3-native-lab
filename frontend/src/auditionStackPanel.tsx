import { useState } from "react";
import { Archive, GitFork, Route, SkipBack, SkipForward } from "lucide-react";

import {
  auditionCursor,
  auditionKeyboardTarget,
  auditionPositionLabel,
  auditionSequenceOptions,
  auditionStackRows,
  type AuditionSequenceMode,
} from "./auditionStack";
import type { ArtifactAnnotationPayload } from "./api";
import { AudioDeck } from "./audioDeck";
import { ListeningDecisionBadge, ListeningDecisionControls, ListeningDecisionSummaryChips } from "./listeningDecision";
import type { ArtifactRecord } from "./types";

export function AuditionStackPanel({
  artifacts,
  selectedId,
  apiBase,
  activeSessionId,
  archivingArtifactId,
  onSelect,
  onCompare,
  onAnnotate,
  onRemember,
  onContinue,
  onBranch,
}: {
  artifacts: ArtifactRecord[];
  selectedId: string | null;
  apiBase: string;
  activeSessionId: string | null;
  archivingArtifactId: string | null;
  onSelect: (artifactId: string) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onRemember: (artifact: ArtifactRecord) => void;
  onContinue: (artifact: ArtifactRecord) => void;
  onBranch: (artifact: ArtifactRecord) => void;
}) {
  const [sequenceMode, setSequenceMode] = useState<AuditionSequenceMode>("recent");
  const [queueAutoplay, setQueueAutoplay] = useState(false);
  const [autoPlayArtifactId, setAutoPlayArtifactId] = useState<string | null>(null);
  const rows = auditionStackRows(artifacts, 8, sequenceMode, selectedId);
  const cursor = auditionCursor(artifacts, selectedId, 8, sequenceMode);
  const position = auditionPositionLabel(artifacts, selectedId, 8, sequenceMode);
  if (!rows.length) return null;
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  const queueArtifacts = rows.map((row) => artifactMap.get(row.artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact));
  const moveSelection = (key: string) => {
    const target = auditionKeyboardTarget(artifacts, selectedId, key, 8, sequenceMode);
    if (!target) return false;
    selectTake(target.artifact_id);
    return true;
  };
  const selectTake = (artifactId: string) => {
    setQueueAutoplay(false);
    setAutoPlayArtifactId(null);
    onSelect(artifactId);
  };
  const changeSequenceMode = (mode: AuditionSequenceMode) => {
    setQueueAutoplay(false);
    setAutoPlayArtifactId(null);
    setSequenceMode(mode);
  };
  const toggleQueueAutoplay = () => {
    const next = !queueAutoplay;
    setQueueAutoplay(next);
    setAutoPlayArtifactId(next ? cursor.selected?.artifact_id ?? null : null);
  };
  const advanceAutoplay = (artifactId: string) => {
    if (!queueAutoplay || selectedId !== artifactId || !cursor.next) {
      if (!cursor.next) setQueueAutoplay(false);
      setAutoPlayArtifactId(null);
      return;
    }
    setAutoPlayArtifactId(cursor.next.artifact_id);
    onSelect(cursor.next.artifact_id);
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
          <span className="eyebrow">Takes</span>
          <strong>{position}</strong>
          <ListeningDecisionSummaryChips artifacts={queueArtifacts} ariaLabel="Queue listening decision summary" />
        </div>
        <label className="audition-sequence">
          <span>Sequence</span>
          <select value={sequenceMode} onChange={(event) => changeSequenceMode(event.target.value as AuditionSequenceMode)}>
            {auditionSequenceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <div className="audition-transport">
          <button type="button" disabled={!cursor.previous} onClick={() => cursor.previous && selectTake(cursor.previous.artifact_id)} title="Previous take">
            <SkipBack size={14} />
          </button>
          <button type="button" disabled={!cursor.next} onClick={() => cursor.next && selectTake(cursor.next.artifact_id)} title="Next take">
            <SkipForward size={14} />
          </button>
          <button
            type="button"
            disabled={!cursor.selected}
            aria-pressed={queueAutoplay}
            className={queueAutoplay ? "active" : ""}
            onClick={toggleQueueAutoplay}
            title="Play through this take queue"
          >
            Auto
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("a", cursor.selected.artifact_id)} title="Pin selected take as anchor">
            Anchor
          </button>
          <button type="button" disabled={!cursor.selected} onClick={() => cursor.selected && onCompare("b", cursor.selected.artifact_id)} title="Pin selected take as source">
            Source
          </button>
        </div>
      </div>
      {rows.map((row) => {
        const artifact = artifactMap.get(row.artifactId);
        if (!artifact) return null;
        return (
          <article key={row.artifactId} className={selectedId === row.artifactId ? "selected" : ""}>
            <button type="button" onClick={() => selectTake(row.artifactId)} title={row.prompt ?? row.label}>
              <span>{row.label}</span>
              <small>{row.sequence} · {row.origin} · {row.meta}</small>
              <ListeningDecisionBadge artifact={artifact} />
              {selectedId === row.artifactId ? <i className="selected-take-label">selected take</i> : null}
            </button>
            <AudioDeck
              artifact={artifact}
              apiBase={apiBase}
              compact
              autoPlay={queueAutoplay && autoPlayArtifactId === row.artifactId}
              onEnded={() => advanceAutoplay(row.artifactId)}
            />
            <ListeningDecisionControls
              artifact={artifact}
              source="take_strip"
              compact
              onDecide={(artifactId, payload) => {
                onSelect(artifactId);
                onAnnotate(artifactId, payload);
              }}
            />
            <div className="audition-stack-actions">
              <button type="button" onClick={() => { onSelect(row.artifactId); onCompare("a", row.artifactId); }} title="Pin take as anchor">Anchor</button>
              <button type="button" onClick={() => { onSelect(row.artifactId); onCompare("b", row.artifactId); }} title="Pin take as source">Source</button>
              <button type="button" onClick={() => { onSelect(row.artifactId); onContinue(artifact); }} title="Continue from this take">
                <Route size={13} />
                Continue
              </button>
              <button
                type="button"
                disabled={!artifact.recipe_id}
                onClick={() => { onSelect(row.artifactId); onBranch(artifact); }}
                title={artifact.recipe_id ? "Branch from this take" : "No gesture recipe to branch"}
              >
                <GitFork size={13} />
                Branch
              </button>
              <button
                type="button"
                disabled={!activeSessionId || artifact.session_id !== activeSessionId || archivingArtifactId === artifact.artifact_id}
                onClick={() => { onSelect(row.artifactId); onRemember(artifact); }}
                title="Remember this take"
              >
                <Archive size={13} />
                {archivingArtifactId === artifact.artifact_id ? "Remembering" : "Remember"}
              </button>
            </div>
          </article>
        );
      })}
    </div>
  );
}
