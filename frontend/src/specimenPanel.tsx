import { Fragment, Suspense, lazy, useEffect, useState } from "react";
import { Archive, Check, CircleDot, Download, GitFork, LoaderCircle, Repeat } from "lucide-react";

import type { ArtifactAnnotationPayload } from "./api";
import { artifactMeta, artifactName, artifactShape, formatDuration } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import type { PromptCandidateGenerationRequest } from "./bundleInspector";
import type { ResultFamily } from "./controlPlane";
import { shortOperatorName } from "./jobUtils";
import { artifactLineageModel, type CompareSlots } from "./lineageModel";
import type { ArtifactRecord, JobRecord, OperatorName, Recipe } from "./types";

const BundleField = lazy(() => import("./bundleInspector").then((module) => ({ default: module.BundleField })));

export function Specimen({
  artifact,
  artifacts,
  jobs,
  families,
  compare,
  apiBase,
  annotating,
  activeSessionId,
  archivingArtifactId,
  onAnnotate,
  onCompare,
  onReplayRecipe,
  onForkRecipe,
  onArchiveArtifact,
  onSelectArtifact,
  onUseAsDonor,
  onUseInRecipe,
  onUsePrompt,
  onGeneratePrompt,
  getArtifactPath,
}: {
  artifact: ArtifactRecord | null;
  artifacts: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
  apiBase: string;
  annotating: boolean;
  activeSessionId: string | null;
  archivingArtifactId: string | null;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
  onArchiveArtifact: (artifact: ArtifactRecord) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
  onUsePrompt: (prompt: string) => void;
  onGeneratePrompt: (request: PromptCandidateGenerationRequest) => void;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
}) {
  if (!artifact) {
    return (
      <div className="specimen empty">
        <CircleDot size={24} />
      </div>
    );
  }
  const fileUrl = `${apiBase}/artifacts/${artifact.artifact_id}/file`;
  const sourceArtifacts = artifact.source_artifact_ids
    .map((artifactId) => artifacts.find((item) => item.artifact_id === artifactId))
    .filter((item): item is ArtifactRecord => Boolean(item));
  const artifactRecipe = artifact.recipe_id ? jobs.find((job) => job.recipe.recipe_id === artifact.recipe_id)?.recipe ?? null : null;
  const canArchiveArtifact = Boolean(activeSessionId && artifact.session_id === activeSessionId);
  return (
    <div className="specimen">
      <div className={`wave-bus ${waveBusStateClass({ artifact, sources: sourceArtifacts, jobs, families, compare })}`}>
        {artifact.kind === "audio" ? (
          <AudioDeck artifact={artifact} apiBase={apiBase} onAnnotate={onAnnotate} persisting={annotating} />
        ) : artifact.kind === "latent" ? (
          <LatentField artifact={artifact} />
        ) : (
          <Suspense fallback={<BundleFieldFallback artifact={artifact} />}>
            <BundleField
              artifact={artifact}
              artifacts={artifacts}
              apiBase={apiBase}
              onCompare={onCompare}
              onSelectArtifact={onSelectArtifact}
              onUseAsDonor={onUseAsDonor}
              onUseInRecipe={onUseInRecipe}
              onUsePrompt={onUsePrompt}
              onGeneratePrompt={onGeneratePrompt}
              onAnnotate={onAnnotate}
              getArtifactPath={getArtifactPath}
            />
          </Suspense>
        )}
        <LineageThread
          artifact={artifact}
          sources={sourceArtifacts}
          jobs={jobs}
          families={families}
          compare={compare}
          onSelectArtifact={onSelectArtifact}
        />
      </div>
      <div className="specimen-info">
        <dl>
          <div>
            <dt>ID</dt>
            <dd>{artifact.artifact_id}</dd>
          </div>
          <div>
            <dt>Lineage</dt>
            <dd>{artifact.source_artifact_ids.length || 0}</dd>
          </div>
          <div>
            <dt>Recipe</dt>
            <dd>{artifact.recipe_id ?? "source"}</dd>
          </div>
          <div>
            <dt>Shape</dt>
            <dd>{artifactShape(artifact)}</dd>
          </div>
        </dl>
        <div className="specimen-actions">
          <button disabled={artifact.kind !== "audio"} onClick={() => onCompare("a", artifact.artifact_id)}>
            Anchor
          </button>
          <button disabled={artifact.kind !== "audio"} onClick={() => onCompare("b", artifact.artifact_id)}>
            Source
          </button>
          <button
            type="button"
            aria-label="Replay recipe"
            disabled={!artifact.recipe_id}
            onClick={() => {
              if (artifact.recipe_id) onReplayRecipe(artifact.recipe_id);
            }}
            title="Replay recipe"
          >
            <Repeat size={17} />
          </button>
          <button
            type="button"
            aria-label="Fork recipe"
            disabled={!artifactRecipe}
            onClick={() => {
              if (artifactRecipe) onForkRecipe(artifactRecipe);
            }}
            title="Fork recipe"
          >
            <GitFork size={17} />
          </button>
          <button
            type="button"
            aria-label="Archive artifact"
            disabled={!canArchiveArtifact || archivingArtifactId === artifact.artifact_id}
            onClick={() => onArchiveArtifact(artifact)}
            title="Archive artifact"
          >
            <Archive size={17} />
          </button>
          <a className="icon-link" href={fileUrl} download title="Download artifact">
            <Download size={17} />
          </a>
        </div>
        <ArtifactVitals artifact={artifact} />
        <ArtifactAnnotationPanel artifact={artifact} saving={annotating} onSave={onAnnotate} />
      </div>
    </div>
  );
}

function BundleFieldFallback({ artifact }: { artifact: ArtifactRecord }) {
  const cells = artifact.recipe_id ? 36 : 18;
  return (
    <div className="bundle-field bundle-field-loading" aria-label={`Experiment bundle ${artifactName(artifact)}`}>
      {Array.from({ length: cells }, (_, index) => (
        <span key={index} />
      ))}
      <div className="bundle-readout">
        <strong>{artifact.file?.filename ?? artifactName(artifact)}</strong>
        <span>Loading bundle inspector</span>
      </div>
    </div>
  );
}

function ArtifactVitals({ artifact }: { artifact: ArtifactRecord }) {
  const rows = artifactVitalRows(artifact);
  if (!rows.length) return null;
  return (
    <div className={`artifact-vitals ${artifact.kind}`} aria-label="Artifact inspector">
      {rows.map(([label, value]) => (
        <span key={label}>
          <small>{label}</small>
          <strong>{value}</strong>
        </span>
      ))}
    </div>
  );
}

function artifactVitalRows(artifact: ArtifactRecord): [string, string][] {
  if (artifact.kind === "audio" && artifact.audio) {
    return [
      ["duration", artifact.audio.duration_seconds ? formatDuration(artifact.audio.duration_seconds) : "unknown"],
      ["rate", `${artifact.audio.sample_rate} Hz`],
      ["channels", String(artifact.audio.channels)],
      ["frames", artifact.audio.frames.toLocaleString()],
    ];
  }
  if (artifact.kind === "latent" && artifact.latent) {
    return [
      ["shape", artifact.latent.shape.join(" x ")],
      ["latent rate", `${artifact.latent.latent_rate.toFixed(2)} Hz`],
      ["duration", artifact.latent.duration_seconds ? formatDuration(artifact.latent.duration_seconds) : "unknown"],
      ["layout", artifact.latent.channel_first ? "channel-first" : "time-first"],
    ];
  }
  if (artifact.kind === "bundle") {
    const operator = typeof artifact.metadata.operator === "string" ? artifact.metadata.operator : "bundle";
    const resultCount = typeof artifact.metadata.result_count === "number" ? String(artifact.metadata.result_count) : null;
    return [
      ["kind", "bundle"],
      ["size", artifact.file ? artifactMeta(artifact).replace(" bundle", "") : "unknown"],
      ["operator", shortOperatorName(operator as OperatorName)],
      ["results", resultCount ?? "inspect"],
    ];
  }
  return [
    ["kind", artifact.kind],
    ["sources", String(artifact.source_artifact_ids.length)],
    ["recipe", artifact.recipe_id ?? "source"],
  ];
}

function ArtifactAnnotationPanel({
  artifact,
  saving,
  onSave,
}: {
  artifact: ArtifactRecord;
  saving: boolean;
  onSave: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
}) {
  const [label, setLabel] = useState(artifact.label ?? "");
  const [tags, setTags] = useState(artifact.tags.join(", "));
  const [notes, setNotes] = useState(artifact.notes ?? "");

  useEffect(() => {
    setLabel(artifact.label ?? "");
    setTags(artifact.tags.join(", "));
    setNotes(artifact.notes ?? "");
  }, [artifact.artifact_id, artifact.label, artifact.notes, artifact.tags]);

  return (
    <div className="annotation-panel">
      <label>
        Label
        <input value={label} onChange={(event) => setLabel(event.target.value)} placeholder="keeper, brittle, needs graft..." />
      </label>
      <label>
        Tags
        <input value={tags} onChange={(event) => setTags(event.target.value)} placeholder="favorite, loop, noisy" />
      </label>
      <label>
        Notes
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="What should future-you remember?" />
      </label>
      <button
        type="button"
        className="annotation-save"
        disabled={saving}
        onClick={() =>
          onSave(artifact.artifact_id, {
            label: label.trim() || null,
            notes: notes.trim() || null,
            tags: parseTags(tags),
          })
        }
      >
        {saving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}
        Save annotation
      </button>
    </div>
  );
}

function LatentField({ artifact }: { artifact: ArtifactRecord }) {
  const shape = artifact.latent?.shape ?? [1, 1];
  const rows = Math.min(6, Math.max(2, shape[1]));
  const columns = Math.min(18, Math.max(6, shape[0]));
  return (
    <div className="latent-field" aria-label={`Latent tensor ${shape.join(" by ")}`}>
      {Array.from({ length: rows * columns }, (_, index) => (
        <span key={index} style={{ animationDelay: `${(index % columns) * 28}ms` }} />
      ))}
      <div className="latent-readout">
        <span>{shape.join(" x ")}</span>
        <span>{artifact.latent?.latent_rate.toFixed(2) ?? "0"} Hz latent</span>
      </div>
    </div>
  );
}

function LineageThread({
  artifact,
  sources,
  jobs,
  families,
  compare,
  onSelectArtifact,
}: {
  artifact: ArtifactRecord;
  sources: ArtifactRecord[];
  jobs: JobRecord[];
  families: ResultFamily[];
  compare: CompareSlots;
  onSelectArtifact: (artifactId: string | null) => void;
}) {
  const nodes = artifactLineageModel({ artifact, sources, jobs, families, compare });
  return (
    <div className="lineage-thread" aria-label="Artifact lineage">
      {nodes.map((node, index) => (
        <Fragment key={node.id}>
          {index ? <span className="thread-route" /> : null}
          {node.artifactId ? (
            <button type="button" className={`thread-node ${node.kind} ${node.kind === "current" ? artifact.kind : ""}`} title={node.title} onClick={() => onSelectArtifact(node.artifactId ?? null)}>
              {node.label}
            </button>
          ) : (
            <span className={`thread-node ${node.kind}`} title={node.title}>
              {node.label}
            </span>
          )}
        </Fragment>
      ))}
    </div>
  );
}

function waveBusStateClass({
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
}) {
  const hasJob = jobs.some((job) => job.recipe.recipe_id === artifact.recipe_id || job.artifact_ids.includes(artifact.artifact_id));
  const hasFamily = families.some((family) => family.artifactIds.includes(artifact.artifact_id) || family.recipeId === artifact.recipe_id);
  const hasCompare = compare.a === artifact.artifact_id || compare.b === artifact.artifact_id;
  return [
    sources.length ? "has-sources" : "root-source",
    hasJob ? "has-job" : "no-job",
    hasFamily ? "has-family" : "no-family",
    hasCompare ? "has-compare" : "no-compare",
    sources.length || hasJob || hasFamily || hasCompare ? "has-flow" : "quiet-flow",
  ].join(" ");
}

function parseTags(value: string) {
  const seen = new Set<string>();
  return value
    .split(/[,\n]/)
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item) => {
      const key = item.toLocaleLowerCase();
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}
