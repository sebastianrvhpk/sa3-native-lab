import { FlaskConical } from "lucide-react";

import { artifactName } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import type { ArtifactRecord } from "./types";

export function ComparePanel({ a, b, apiBase }: { a: ArtifactRecord | null; b: ArtifactRecord | null; apiBase: string }) {
  return (
    <div className="compare-panel">
      <div className="band-title">
        <FlaskConical size={18} />
        <span>Anchors</span>
      </div>
      <CompareSlot label="Anchor" artifact={a} apiBase={apiBase} />
      <CompareSlot label="Source" artifact={b} apiBase={apiBase} />
    </div>
  );
}

function CompareSlot({ label, artifact, apiBase }: { label: string; artifact: ArtifactRecord | null; apiBase: string }) {
  return (
    <div className="compare-slot">
      <strong>{label}</strong>
      {artifact ? (
        <div>
          <span>{artifactName(artifact)}</span>
          <AudioDeck artifact={artifact} apiBase={apiBase} compact />
        </div>
      ) : (
        <span className="muted">not pinned</span>
      )}
    </div>
  );
}
