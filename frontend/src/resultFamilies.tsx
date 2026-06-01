import { useMemo, useState } from "react";
import { Archive, Gauge, GitFork, Repeat, Route, Search, SkipBack, SkipForward } from "lucide-react";

import { ArtifactIcon } from "./artifactDisplay";
import type { ArtifactAnnotationPayload } from "./api";
import { artifactMeta, artifactName, formatFamilyStamp, sortNewest, sortNewestJobs } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import { branchListeningCursor } from "./branchListeningModel";
import { branchMeta, branchSummaryForFamily } from "./branchModel";
import type { ResultFamily } from "./controlPlane";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { ListeningDecisionBadge, ListeningDecisionControls, ListeningDecisionSummaryChips } from "./listeningDecision";
import type { ArtifactRecord, JobRecord, Recipe } from "./types";

export function ResultFamilyPanel({
  families,
  artifacts,
  selectedId,
  inspectedFamilyId,
  onSelect,
  onInspectFamily,
  onReplayRecipe,
  onForkRecipe,
}: {
  families: ResultFamily[];
  artifacts: ArtifactRecord[];
  selectedId: string | null;
  inspectedFamilyId: string | null;
  onSelect: (artifactId: string | null) => void;
  onInspectFamily: (familyId: string) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
}) {
  if (!families.length) {
    return <div className="quiet-panel compact">No branches yet</div>;
  }
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  return (
    <div className="family-stack">
      {families.slice(0, 5).map((family) => {
        const summary = branchSummaryForFamily(family, artifacts);
        const latest = family.latestArtifactId ? artifactMap.get(family.latestArtifactId) ?? null : null;
        const familyArtifacts = family.artifactIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact));
        const selected = Boolean(latest && latest.artifact_id === selectedId);
        const inspected = family.familyId === inspectedFamilyId;
        return (
          <article key={family.familyId} className={`family-row ${selected ? "selected" : ""} ${inspected ? "inspected" : ""}`}>
            <button type="button" onClick={() => { onInspectFamily(family.familyId); onSelect(latest?.artifact_id ?? null); }}>
              <div>
                <strong>{summary.title}</strong>
                <span>{branchMeta(summary)}</span>
              </div>
              <span className={`family-status ${family.status}`}>{family.status}</span>
            </button>
            <div className="family-kinds" aria-label="Material kinds in branch">
              {summary.kindLabels.length ? (
                summary.kindLabels.map((kind) => (
                  <i key={kind} className={kind}>
                    {kind}
                  </i>
                ))
              ) : (
                <i>empty</i>
              )}
            </div>
            <ListeningDecisionSummaryChips artifacts={familyArtifacts} />
            <button type="button" className="family-replay" onClick={() => onInspectFamily(family.familyId)} title="Inspect branch takes and progress">
              <Search size={14} />
              Inspect
            </button>
            <button type="button" className="family-replay" onClick={() => onReplayRecipe(family.recipeId)} title="Do this branch gesture again">
              <Repeat size={14} />
              Do again
            </button>
            <button type="button" className="family-replay" onClick={() => onForkRecipe(family.recipe)} title="Branch from this path">
              <GitFork size={14} />
              Branch
            </button>
          </article>
        );
      })}
    </div>
  );
}

export function FamilyDetailPanel({
  family,
  families = [],
  artifacts,
  jobs,
  selectedId,
  apiBase,
  activeSessionId,
  archivingArtifactId,
  onSelect,
  onInspectFamily,
  onCompare,
  onAnnotate,
  onReplayRecipe,
  onForkRecipe,
  onContinueArtifact,
  onBranchArtifact,
  onArchiveArtifact,
  onCancelJob,
  onRetryJob,
}: {
  family: ResultFamily | null;
  families?: ResultFamily[];
  artifacts: ArtifactRecord[];
  jobs: JobRecord[];
  selectedId: string | null;
  apiBase: string;
  activeSessionId: string | null;
  archivingArtifactId: string | null;
  onSelect: (artifactId: string | null) => void;
  onInspectFamily?: (familyId: string) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
  onContinueArtifact?: (artifact: ArtifactRecord) => void;
  onBranchArtifact?: (artifact: ArtifactRecord) => void;
  onArchiveArtifact: (artifact: ArtifactRecord) => void;
} & JobActionHandlers) {
  if (!family) {
    return <div className="quiet-panel compact">No branch selected</div>;
  }
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  const familyArtifacts = sortNewest(family.artifactIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact)));
  const familyJobs = sortNewestJobs(family.jobIds.map((jobId) => jobs.find((job) => job.job_id === jobId)).filter((job): job is JobRecord => Boolean(job)));
  const sourceIds = Object.values(family.recipe.inputs).filter((value): value is string => typeof value === "string" && value.length > 0);
  const sourceArtifacts = sourceIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact));
  const sweepEntries = buildSweepEntries(family, familyArtifacts);
  const siblingSweeps = findSiblingSweepFamilies(family, families);
  const summary = branchSummaryForFamily(family, artifacts);
  const branchCursor = branchListeningCursor(familyArtifacts, selectedId);

  return (
    <section className="family-detail">
      <div className="family-detail-head">
        <div>
          <span className="eyebrow">Branch</span>
          <strong>{summary.title}</strong>
        </div>
        <span className={`family-status ${family.status}`}>{family.status}</span>
      </div>
      <div className="family-recipe-strip">
        <span>{summary.gestureLabel}</span>
        <span>{summary.takesCount} take{summary.takesCount === 1 ? "" : "s"}</span>
        <span>{summary.latestTakeLabel}</span>
        <span>{summary.updatedLabel}</span>
      </div>
      <ListeningDecisionSummaryChips artifacts={familyArtifacts} ariaLabel="Branch listening decision summary" />
      {sourceArtifacts.length ? (
        <div className="family-source-strip" aria-label="Branch source material">
          {sourceArtifacts.slice(0, 3).map((artifact) => (
            <button key={artifact.artifact_id} type="button" onClick={() => onSelect(artifact.artifact_id)} title={artifactName(artifact)}>
              <ArtifactIcon artifact={artifact} />
              <span>{artifactName(artifact)}</span>
            </button>
          ))}
        </div>
      ) : null}
      <div className="family-detail-actions">
        <button type="button" onClick={() => onReplayRecipe(family.recipeId)}>
          <Repeat size={14} />
          Do again
        </button>
        <button type="button" onClick={() => onForkRecipe(family.recipe)}>
          <GitFork size={14} />
          Branch
        </button>
      </div>
      {branchCursor.takes.length ? (
        <div className="branch-listening-transport" aria-label="Branch listening trajectory">
          <span>Branch trajectory</span>
          <strong>{branchCursor.positionLabel}</strong>
          <button type="button" disabled={!branchCursor.previous} onClick={() => branchCursor.previous && onSelect(branchCursor.previous.artifact_id)} title="Previous branch take">
            <SkipBack size={13} />
          </button>
          <button type="button" disabled={!branchCursor.next} onClick={() => branchCursor.next && onSelect(branchCursor.next.artifact_id)} title="Next branch take">
            <SkipForward size={13} />
          </button>
        </div>
      ) : null}
      {sweepEntries.length ? (
        <SweepFamilyBand
          entries={sweepEntries}
          selectedId={selectedId}
          onSelect={onSelect}
          onCompare={onCompare}
          onForkRecipe={() => onForkRecipe(family.recipe)}
        />
      ) : null}
      {siblingSweeps.length ? (
        <SweepSiblingComparison
          siblings={siblingSweeps}
          onInspectFamily={onInspectFamily}
          onReplayRecipe={onReplayRecipe}
          onForkRecipe={onForkRecipe}
        />
      ) : null}
      <div className="family-artifacts">
        {familyArtifacts.length ? (
          familyArtifacts.map((artifact) => (
            <article key={artifact.artifact_id} className={`family-artifact ${artifact.artifact_id === selectedId ? "selected" : ""}`}>
              <button type="button" className="family-artifact-main" onClick={() => onSelect(artifact.artifact_id)}>
                <ArtifactIcon artifact={artifact} />
                <div>
                  <strong>{artifactName(artifact)}</strong>
                  <span>{artifactMeta(artifact)}</span>
                  <ListeningDecisionBadge artifact={artifact} />
                  {artifact.artifact_id === selectedId ? <i className="selected-take-label">selected take</i> : null}
                </div>
              </button>
              {artifact.kind === "audio" ? (
                <>
                  <AudioDeck artifact={artifact} apiBase={apiBase} compact />
                  <div className="family-artifact-actions">
                    <button type="button" onClick={() => { onSelect(artifact.artifact_id); onCompare("a", artifact.artifact_id); }}>Anchor</button>
                    <button type="button" onClick={() => { onSelect(artifact.artifact_id); onCompare("b", artifact.artifact_id); }}>Source</button>
                    <button
                      type="button"
                      disabled={!onContinueArtifact}
                      onClick={() => {
                        onSelect(artifact.artifact_id);
                        onContinueArtifact?.(artifact);
                      }}
                    >
                      <Route size={13} />
                      Continue
                    </button>
                    <button
                      type="button"
                      disabled={!artifact.recipe_id || !onBranchArtifact}
                      onClick={() => {
                        onSelect(artifact.artifact_id);
                        onBranchArtifact?.(artifact);
                      }}
                    >
                      <GitFork size={13} />
                      Branch
                    </button>
                    <button
                      type="button"
                      aria-label="Remember sound"
                      disabled={!activeSessionId || artifact.session_id !== activeSessionId || archivingArtifactId === artifact.artifact_id}
                      onClick={() => onArchiveArtifact(artifact)}
                      title="Move this sound to memory"
                    >
                      <Archive size={13} />
                    </button>
                  </div>
                  <ListeningDecisionControls
                    artifact={artifact}
                    source="family_detail"
                    compact
                    onDecide={(artifactId, payload) => {
                      onSelect(artifactId);
                      onAnnotate(artifactId, payload);
                    }}
                  />
                </>
              ) : null}
            </article>
          ))
        ) : (
          <div className="quiet-panel compact">No takes landed yet</div>
        )}
      </div>
      <details className="family-job-drawer">
        <summary>
          <Gauge size={14} />
          Inspect branch
          <span>{familyJobs.length}</span>
        </summary>
        <dl className="branch-inspect-list">
          {summary.inspectRows.map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
        {familyJobs.length ? (
          <div className="family-job-list">
            {familyJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
            ))}
          </div>
        ) : null}
      </details>
    </section>
  );
}

function SweepSiblingComparison({
  siblings,
  onInspectFamily,
  onReplayRecipe,
  onForkRecipe,
}: {
  siblings: ResultFamily[];
  onInspectFamily?: (familyId: string) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
}) {
  return (
    <div className="sweep-sibling-comparison" aria-label="Sibling sweep comparison">
      <div className="sweep-family-head">
        <span>Sibling sweeps</span>
        <strong>{siblings.length} nearby</strong>
      </div>
      {siblings.slice(0, 3).map((family) => (
        <article key={family.familyId}>
          <div>
            <strong>{siblingSweepLabel(family)}</strong>
            <span>{siblingSweepMeta(family)}</span>
          </div>
          <div className="sweep-sibling-actions">
            <button type="button" disabled={!onInspectFamily} onClick={() => onInspectFamily?.(family.familyId)}>
              <Search size={13} />
              Inspect
            </button>
            <button type="button" onClick={() => onReplayRecipe(family.recipeId)}>
              <Repeat size={13} />
              Do again
            </button>
            <button type="button" onClick={() => onForkRecipe(family.recipe)}>
              <GitFork size={13} />
              Branch
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}

function SweepFamilyBand({
  entries,
  selectedId,
  onSelect,
  onCompare,
  onForkRecipe,
}: {
  entries: SweepEntry[];
  selectedId: string | null;
  onSelect: (artifactId: string | null) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onForkRecipe: () => void;
}) {
  const metricKeys = sweepMetricKeys(entries);
  const primaryMetric = metricKeys[0] ?? null;
  const [sortKey, setSortKey] = useState<SweepSortKey>("alpha");
  const sortedEntries = useMemo(() => sortSweepEntries(entries, sortKey, primaryMetric), [entries, sortKey, primaryMetric]);
  const bestArtifactId = primaryMetric ? bestSweepEntry(entries, primaryMetric)?.artifact.artifact_id ?? null : null;
  return (
    <div className="sweep-family-band" aria-label="Alpha sweep variants">
      <div className="sweep-family-head">
        <span>Alpha Sweep</span>
        <strong>{entries.length} variants</strong>
      </div>
      <ListeningDecisionSummaryChips artifacts={entries.map((entry) => entry.artifact)} ariaLabel="Sweep listening decision summary" />
      <div className="sweep-sort-controls" aria-label="Sort sweep variants">
        <button type="button" className={sortKey === "alpha" ? "active" : ""} onClick={() => setSortKey("alpha")}>
          alpha
        </button>
        {primaryMetric ? (
          <button type="button" className={sortKey === "metric" ? "active" : ""} onClick={() => setSortKey("metric")}>
            {prettyParamName(primaryMetric)}
          </button>
        ) : null}
        <button type="button" className={sortKey === "duration" ? "active" : ""} onClick={() => setSortKey("duration")}>
          duration
        </button>
      </div>
      <div className="sweep-family-grid">
        {sortedEntries.map((entry) => (
          <article
            key={entry.artifact.artifact_id}
            className={`${entry.artifact.artifact_id === bestArtifactId ? "promoted" : ""} ${entry.artifact.artifact_id === selectedId ? "selected" : ""}`}
          >
            <button type="button" className="sweep-main" onClick={() => onSelect(entry.artifact.artifact_id)} title={artifactName(entry.artifact)}>
              <span>{entry.label}</span>
              <small>{sweepDeltaLabel(entry)} · {artifactMeta(entry.artifact)}</small>
              <ListeningDecisionBadge artifact={entry.artifact} />
              {entry.artifact.artifact_id === bestArtifactId ? <i>highlight</i> : null}
              {entry.artifact.artifact_id === selectedId ? <i className="selected-take-label">selected take</i> : null}
            </button>
            <div className="sweep-actions">
              <button type="button" onClick={() => onCompare("a", entry.artifact.artifact_id)} title="Pin this sweep variant as anchor">
                Anchor
              </button>
              <button type="button" onClick={() => onCompare("b", entry.artifact.artifact_id)} title="Pin this sweep variant as source">
                Source
              </button>
              <button type="button" onClick={onForkRecipe} title="Branch from the sweep gesture">
                <GitFork size={13} />
              </button>
            </div>
          </article>
        ))}
      </div>
      <SweepMetricTable entries={sortedEntries} bestArtifactId={bestArtifactId} onSelect={onSelect} onCompare={onCompare} />
    </div>
  );
}

function SweepMetricTable({
  entries,
  bestArtifactId,
  onSelect,
  onCompare,
}: {
  entries: SweepEntry[];
  bestArtifactId: string | null;
  onSelect: (artifactId: string | null) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
}) {
  const metricKeys = sweepMetricKeys(entries);
  return (
    <div className="sweep-metric-table" aria-label="Alpha sweep metric table">
      <div className="sweep-metric-row header">
        <span>alpha</span>
        <span>take</span>
        {metricKeys.map((key) => (
          <span key={key}>{prettyParamName(key)}</span>
        ))}
        <span>duration</span>
        <span>pin</span>
      </div>
      {entries.map((entry) => (
        <div key={entry.artifact.artifact_id} className={`sweep-metric-row ${entry.artifact.artifact_id === bestArtifactId ? "promoted" : ""}`}>
          <span>{entry.alpha === null ? "n/a" : formatAlpha(entry.alpha)}</span>
          <button type="button" onClick={() => onSelect(entry.artifact.artifact_id)} title="Select sweep artifact">
            {artifactName(entry.artifact)}
          </button>
          {metricKeys.map((key) => (
            <span key={key}>{formatMetricValue(entry.metrics[key])}</span>
          ))}
          <span>{entry.artifact.audio ? `${formatMetricValue(entry.artifact.audio.duration_seconds)}s` : "n/a"}</span>
          <span className="sweep-table-actions">
            <button type="button" onClick={() => onCompare("a", entry.artifact.artifact_id)} title="Pin as anchor">
              Anchor
            </button>
            <button type="button" onClick={() => onCompare("b", entry.artifact.artifact_id)} title="Pin as source">
              Source
            </button>
          </span>
        </div>
      ))}
    </div>
  );
}

function prettyParamName(name: string) {
  return name.replaceAll("_", " ");
}

interface SweepEntry {
  artifact: ArtifactRecord;
  alpha: number | null;
  label: string;
  metrics: Record<string, number | string>;
}

type SweepSortKey = "alpha" | "metric" | "duration";

function buildSweepEntries(family: ResultFamily, artifacts: ArtifactRecord[]): SweepEntry[] {
  if (family.operator !== "experiment.alpha_sweep") return [];
  const recipeAlphas = parseAlphaList(family.recipe.params.alphas);
  return artifacts
    .filter((artifact) => artifact.kind === "audio")
    .map((artifact, index) => {
      const alpha = artifactAlpha(artifact) ?? recipeAlphas[index] ?? null;
      return {
        artifact,
        alpha,
        label: alpha === null ? sweepLabel(artifact, index) : `alpha ${formatAlpha(alpha)}`,
        metrics: artifactSweepMetrics(artifact),
      };
    })
    .sort((left, right) => {
      if (left.alpha === null && right.alpha === null) return left.artifact.created_at.localeCompare(right.artifact.created_at);
      if (left.alpha === null) return 1;
      if (right.alpha === null) return -1;
      return left.alpha - right.alpha;
    });
}

function sortSweepEntries(entries: SweepEntry[], sortKey: SweepSortKey, metricKey: string | null): SweepEntry[] {
  return [...entries].sort((left, right) => {
    if (sortKey === "metric" && metricKey) {
      return compareMetricValues(left.metrics[metricKey], right.metrics[metricKey], metricHigherIsBetter(metricKey));
    }
    if (sortKey === "duration") {
      return (right.artifact.audio?.duration_seconds ?? -1) - (left.artifact.audio?.duration_seconds ?? -1);
    }
    return compareAlpha(left, right);
  });
}

function bestSweepEntry(entries: SweepEntry[], metricKey: string): SweepEntry | null {
  const sorted = sortSweepEntries(entries.filter((entry) => entry.metrics[metricKey] !== undefined), "metric", metricKey);
  return sorted[0] ?? null;
}

function compareMetricValues(left: unknown, right: unknown, higherIsBetter: boolean) {
  const leftNumber = typeof left === "number" && Number.isFinite(left) ? left : null;
  const rightNumber = typeof right === "number" && Number.isFinite(right) ? right : null;
  if (leftNumber === null && rightNumber === null) return 0;
  if (leftNumber === null) return 1;
  if (rightNumber === null) return -1;
  return higherIsBetter ? rightNumber - leftNumber : leftNumber - rightNumber;
}

function metricHigherIsBetter(key: string) {
  return !/(loss|distance|error|wer|mae|mse|rmse)/i.test(key);
}

function compareAlpha(left: SweepEntry, right: SweepEntry) {
  if (left.alpha === null && right.alpha === null) return left.artifact.created_at.localeCompare(right.artifact.created_at);
  if (left.alpha === null) return 1;
  if (right.alpha === null) return -1;
  return left.alpha - right.alpha;
}

function artifactSweepMetrics(artifact: ArtifactRecord): Record<string, number | string> {
  const metrics: Record<string, number | string> = {};
  for (const [key, value] of Object.entries(artifact.metadata)) {
    if (key === "alpha" || key === "operator" || key === "backend" || key === "script_output_path") continue;
    if (typeof value === "number" && Number.isFinite(value)) metrics[key] = value;
    if (typeof value === "string" && value.length <= 32 && /^-?\d+(?:\.\d+)?$/.test(value)) metrics[key] = Number(value);
  }
  return metrics;
}

function sweepMetricKeys(entries: SweepEntry[]): string[] {
  const priority = ["score", "loss", "distance", "duration", "return_code"];
  const keys = [...new Set(entries.flatMap((entry) => Object.keys(entry.metrics)))];
  return keys
    .sort((left, right) => {
      const leftIndex = priority.indexOf(left);
      const rightIndex = priority.indexOf(right);
      if (leftIndex !== -1 || rightIndex !== -1) return (leftIndex === -1 ? 999 : leftIndex) - (rightIndex === -1 ? 999 : rightIndex);
      return left.localeCompare(right);
    })
    .slice(0, 3);
}

function formatMetricValue(value: unknown) {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value.toFixed(Math.abs(value) >= 10 ? 1 : 3).replace(/\.?0+$/, "");
  }
  return value === undefined || value === null || value === "" ? "n/a" : String(value);
}

function parseAlphaList(value: unknown): number[] {
  if (Array.isArray(value)) return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
  if (typeof value !== "string") return [];
  return value
    .split(/[,\s]+/)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
}

function artifactAlpha(artifact: ArtifactRecord): number | null {
  const metadataAlpha = artifact.metadata.alpha;
  if (typeof metadataAlpha === "number" && Number.isFinite(metadataAlpha)) return metadataAlpha;
  const text = [artifact.label, artifact.path, artifact.file?.filename].filter(Boolean).join(" ");
  const plain = text.match(/alpha[_\s-]*([+-]?\d+(?:\.\d+)?)/i);
  if (plain) return Number(plain[1]);
  const token = text.match(/alpha_(pos|neg)(\d+)p(\d+)/i);
  if (!token) return null;
  const sign = token[1]?.toLowerCase() === "neg" ? -1 : 1;
  const whole = Number(token[2] ?? 0);
  const decimal = Number(`0.${token[3] ?? 0}`);
  const value = sign * (whole + decimal);
  return Number.isFinite(value) ? value : null;
}

function sweepLabel(artifact: ArtifactRecord, index: number) {
  return artifact.label || artifact.file?.filename?.replace(/\.[^.]+$/, "") || `variant ${index + 1}`;
}

function sweepDeltaLabel(entry: SweepEntry) {
  if (entry.alpha === null) return "open alpha";
  if (entry.alpha === 0) return "neutral";
  return `${formatAlpha(entry.alpha)} from neutral`;
}

function formatAlpha(value: number) {
  const prefix = value > 0 ? "+" : "";
  return `${prefix}${value.toFixed(2).replace(/\.?0+$/, "")}`;
}

function findSiblingSweepFamilies(family: ResultFamily, families: ResultFamily[]): ResultFamily[] {
  if (family.operator !== "experiment.alpha_sweep") return [];
  const anchor = sweepSiblingAnchor(family.recipe);
  return families
    .filter((candidate) => candidate.familyId !== family.familyId && candidate.operator === "experiment.alpha_sweep")
    .filter((candidate) => sweepSiblingAnchorMatches(anchor, sweepSiblingAnchor(candidate.recipe)))
    .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
    .slice(0, 4);
}

function sweepSiblingAnchor(recipe: Recipe) {
  return {
    vectors: recipeInputOrParam(recipe, "vectors_path"),
    prompt: typeof recipe.params.prompt === "string" ? recipe.params.prompt.trim().toLowerCase() : "",
    model: recipe.model ?? "",
  };
}

function sweepSiblingAnchorMatches(
  left: ReturnType<typeof sweepSiblingAnchor>,
  right: ReturnType<typeof sweepSiblingAnchor>,
) {
  if (left.vectors && right.vectors) return left.vectors === right.vectors;
  if (left.prompt && right.prompt) return left.prompt === right.prompt;
  return left.model !== "" && left.model === right.model;
}

function recipeInputOrParam(recipe: Recipe, key: string) {
  const inputValue = recipe.inputs[key];
  if (typeof inputValue === "string" && inputValue.length) return inputValue;
  const paramValue = recipe.params[key];
  return typeof paramValue === "string" && paramValue.length ? paramValue : "";
}

function siblingSweepLabel(family: ResultFamily) {
  const prompt = typeof family.recipe.params.prompt === "string" ? family.recipe.params.prompt.trim() : "";
  return prompt || formatFamilyStamp(family.updatedAt);
}

function siblingSweepMeta(family: ResultFamily) {
  const alphas = parseAlphaList(family.recipe.params.alphas);
  const alphaText = alphas.length ? `${alphas.length} alphas` : "open alpha set";
  const seedText = family.recipe.seed !== null && family.recipe.seed !== undefined ? `seed ${family.recipe.seed}` : "no seed";
  return `${alphaText} · ${family.artifactIds.length} takes · ${family.recipe.model ?? family.recipe.backend} · ${seedText}`;
}
