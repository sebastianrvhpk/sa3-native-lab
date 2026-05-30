import { useMemo, useState } from "react";
import { Archive, Gauge, GitFork, Repeat, Search } from "lucide-react";

import { ArtifactIcon } from "./artifactDisplay";
import type { ArtifactAnnotationPayload } from "./api";
import { artifactMeta, artifactName, formatFamilyStamp, sortNewest, sortNewestJobs } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import type { ResultFamily } from "./controlPlane";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { shortOperatorName } from "./jobUtils";
import { listeningDecision, ListeningDecisionBadge, ListeningDecisionControls } from "./listeningDecision";
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
        const latest = family.latestArtifactId ? artifactMap.get(family.latestArtifactId) ?? null : null;
        const familyArtifacts = family.artifactIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact));
        const selected = Boolean(latest && latest.artifact_id === selectedId);
        const inspected = family.familyId === inspectedFamilyId;
        return (
          <article key={family.familyId} className={`family-row ${selected ? "selected" : ""} ${inspected ? "inspected" : ""}`}>
            <button type="button" onClick={() => { onInspectFamily(family.familyId); onSelect(latest?.artifact_id ?? null); }}>
              <div>
                <strong>{familyTitle(family)}</strong>
                <span>{familyMeta(family)}</span>
              </div>
              <span className={`family-status ${family.status}`}>{family.status}</span>
            </button>
            <div className="family-kinds" aria-label="Artifact kinds in family">
              {family.artifactKinds.length ? (
                family.artifactKinds.map((kind) => (
                  <i key={kind} className={kind}>
                    {kind}
                  </i>
                ))
              ) : (
                <i>empty</i>
              )}
            </div>
            <MetricChips metrics={family.metrics} />
            <DecisionSummary artifacts={familyArtifacts} />
            <button type="button" className="family-replay" onClick={() => onInspectFamily(family.familyId)} title="Inspect branch takes and progress">
              <Search size={14} />
              Inspect
            </button>
            <button type="button" className="family-replay" onClick={() => onReplayRecipe(family.recipeId)} title="Replay family recipe">
              <Repeat size={14} />
              Do again
            </button>
            <button type="button" className="family-replay" onClick={() => onForkRecipe(family.recipe)} title="Fork family recipe">
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

  return (
    <section className="family-detail">
      <div className="family-detail-head">
        <div>
          <span className="eyebrow">Branch</span>
          <strong>{familyTitle(family)}</strong>
        </div>
        <span className={`family-status ${family.status}`}>{family.status}</span>
      </div>
      <div className="family-recipe-strip">
        <span>{family.recipe.backend}</span>
        {family.recipe.model ? <span>{family.recipe.model}</span> : null}
        {family.recipe.seed !== null && family.recipe.seed !== undefined ? <span>seed {family.recipe.seed}</span> : null}
        <span>{formatFamilyStamp(family.updatedAt)}</span>
      </div>
      {sourceArtifacts.length ? (
        <div className="family-source-strip" aria-label="Family source artifacts">
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
      {sweepEntries.length ? (
        <SweepFamilyBand entries={sweepEntries} onSelect={onSelect} onCompare={onCompare} onForkRecipe={() => onForkRecipe(family.recipe)} />
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
                </div>
              </button>
              {artifact.kind === "audio" ? (
                <>
                  <AudioDeck artifact={artifact} apiBase={apiBase} compact />
                  <div className="family-artifact-actions">
                    <button type="button" onClick={() => onCompare("a", artifact.artifact_id)}>Anchor</button>
                    <button type="button" onClick={() => onCompare("b", artifact.artifact_id)}>Source</button>
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
                  <ListeningDecisionControls artifact={artifact} source="family_detail" compact onDecide={onAnnotate} />
                </>
              ) : null}
            </article>
          ))
        ) : (
          <div className="quiet-panel compact">No artifacts landed yet</div>
        )}
      </div>
      {familyJobs.length ? (
        <details className="family-job-drawer">
          <summary>
            <Gauge size={14} />
            Gesture history
            <span>{familyJobs.length}</span>
          </summary>
          <div className="family-job-list">
            {familyJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
            ))}
          </div>
        </details>
      ) : null}
    </section>
  );
}

function DecisionSummary({ artifacts }: { artifacts: ArtifactRecord[] }) {
  const counts = artifacts.reduce(
    (items, artifact) => {
      const decision = listeningDecision(artifact);
      if (decision) items[decision] += 1;
      return items;
    },
    { keeper: 0, maybe: 0, rejected: 0 },
  );
  const entries = [
    ["keeper", counts.keeper],
    ["maybe", counts.maybe],
    ["rejected", counts.rejected],
  ].filter(([, count]) => Number(count) > 0);
  if (!entries.length) return null;
  return (
    <div className="decision-summary" aria-label="Listening decision summary">
      {entries.map(([label, count]) => (
        <i key={label} className={String(label)}>
          {String(count)} {label}
        </i>
      ))}
    </div>
  );
}

function familyTitle(family: ResultFamily) {
  if (family.familyId.startsWith("prompt-candidates:")) return "Prompt candidates";
  return shortOperatorName(family.operator);
}

function familyMeta(family: ResultFamily) {
  const artifactWord = `take${family.artifactIds.length === 1 ? "" : "s"}`;
  const runWord = family.familyId.startsWith("prompt-candidates:")
    ? `generation${family.jobIds.length === 1 ? "" : "s"}`
    : `gesture${family.jobIds.length === 1 ? "" : "s"}`;
  return `${family.artifactIds.length} ${artifactWord} · ${family.jobIds.length} ${runWord}`;
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
          <MetricChips metrics={family.metrics} />
          <div className="sweep-sibling-actions">
            <button type="button" disabled={!onInspectFamily} onClick={() => onInspectFamily?.(family.familyId)}>
              <Search size={13} />
              Inspect
            </button>
            <button type="button" onClick={() => onReplayRecipe(family.recipeId)}>
              <Repeat size={13} />
              Replay
            </button>
            <button type="button" onClick={() => onForkRecipe(family.recipe)}>
              <GitFork size={13} />
              Fork
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}

function SweepFamilyBand({
  entries,
  onSelect,
  onCompare,
  onForkRecipe,
}: {
  entries: SweepEntry[];
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
          <article key={entry.artifact.artifact_id} className={entry.artifact.artifact_id === bestArtifactId ? "promoted" : ""}>
            <button type="button" className="sweep-main" onClick={() => onSelect(entry.artifact.artifact_id)} title={artifactName(entry.artifact)}>
              <span>{entry.label}</span>
              <small>{artifactMeta(entry.artifact)}</small>
              {entry.artifact.artifact_id === bestArtifactId ? <i>best</i> : null}
            </button>
            <div className="sweep-actions">
              <button type="button" onClick={() => onCompare("a", entry.artifact.artifact_id)} title="Pin this sweep variant as anchor">
                Anchor
              </button>
              <button type="button" onClick={() => onCompare("b", entry.artifact.artifact_id)} title="Pin this sweep variant as source">
                Source
              </button>
              <button type="button" onClick={onForkRecipe} title="Fork the sweep recipe">
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
        <span>artifact</span>
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

function MetricChips({ metrics }: { metrics: Record<string, unknown> }) {
  const rows = ["result_count", "candidate_count", "metric", "return_code"]
    .map((key) => [key, metrics[key]] as const)
    .filter(([, value]) => value !== undefined && value !== null);
  if (!rows.length) return null;
  return (
    <div className="metric-chips">
      {rows.slice(0, 4).map(([key, value]) => (
        <i key={key}>
          {prettyParamName(key)}: {String(value)}
        </i>
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
  return `${alphaText} · ${family.artifactIds.length} artifacts · ${family.recipe.model ?? family.recipe.backend} · ${seedText}`;
}
