import { Upload } from "lucide-react";

import { ArtifactIcon } from "./artifactDisplay";
import { TinyWave } from "./audioDeck";
import type { MemoryReuseAction } from "./memoryModel";
import type { ProductSource } from "./sourceModel";

export function SourceShelf({
  sources,
  selectedId,
  apiBase,
  onSelect,
  onUseAction,
}: {
  sources: ProductSource[];
  selectedId: string | null;
  apiBase: string;
  onSelect: (id: string) => void;
  onUseAction: (source: ProductSource, action: MemoryReuseAction) => void;
}) {
  if (!sources.length) {
    return (
      <div className="empty-panel">
        <Upload size={22} />
        <strong>Import audio</strong>
      </div>
    );
  }
  return (
    <div className="artifact-stack">
      {sources.map((source) => {
        const artifact = source.artifact;
        const actions = source.actions.filter((action) => action.available);
        return (
          <article
            key={artifact.artifact_id}
            className={`artifact-row-shell ${selectedId === artifact.artifact_id ? "selected" : ""}`}
          >
            <button
              type="button"
              className="artifact-row"
              onClick={() => onSelect(artifact.artifact_id)}
              aria-label={`Select ${source.label}`}
            >
              <ArtifactIcon artifact={artifact} />
              <div>
                <strong>{source.label}</strong>
                <span>{source.detail}</span>
                {source.roleLabels.length ? (
                  <small className="source-role-strip">
                    {source.roleLabels.slice(0, 3).map((role) => (
                      <i key={role}>{role}</i>
                    ))}
                  </small>
                ) : null}
              </div>
              {artifact.kind === "audio" ? <TinyWave artifact={artifact} apiBase={apiBase} /> : null}
            </button>
            {actions.length ? (
              <div className="source-action-strip" aria-label={`${source.label} source actions`}>
                {actions.slice(0, 4).map((action) => (
                  <button
                    key={`${action.intent}:${action.id}:${action.fieldKey ?? ""}`}
                    type="button"
                    title={action.description}
                    onClick={() => onUseAction(source, action)}
                  >
                    {sourceActionLabel(action)}
                  </button>
                ))}
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function sourceActionLabel(action: MemoryReuseAction): string {
  if (action.intent === "source") return "Source";
  if (action.intent === "anchor") return "Anchor";
  if (action.intent === "donor") return "Donor";
  if (action.intent === "prompt_seed") return "Prompt";
  if (action.intent === "advanced_gesture") return action.label;
  if (action.intent === "recover") return "Recover";
  return action.label;
}
