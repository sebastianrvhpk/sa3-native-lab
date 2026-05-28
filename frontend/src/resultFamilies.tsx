import { Gauge, GitFork, Repeat, Search } from "lucide-react";

import { ArtifactIcon } from "./artifactDisplay";
import { artifactMeta, artifactName, formatFamilyStamp, sortNewest, sortNewestJobs } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import type { ResultFamily } from "./controlPlane";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { shortOperatorName } from "./jobUtils";
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
    return <div className="quiet-panel compact">No result families yet</div>;
  }
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  return (
    <div className="family-stack">
      {families.slice(0, 5).map((family) => {
        const latest = family.latestArtifactId ? artifactMap.get(family.latestArtifactId) ?? null : null;
        const selected = Boolean(latest && latest.artifact_id === selectedId);
        const inspected = family.familyId === inspectedFamilyId;
        return (
          <article key={family.familyId} className={`family-row ${selected ? "selected" : ""} ${inspected ? "inspected" : ""}`}>
            <button type="button" onClick={() => { onInspectFamily(family.familyId); onSelect(latest?.artifact_id ?? null); }}>
              <div>
                <strong>{shortOperatorName(family.operator)}</strong>
                <span>
                  {family.artifactIds.length} artifact{family.artifactIds.length === 1 ? "" : "s"} · {family.jobIds.length} run
                  {family.jobIds.length === 1 ? "" : "s"}
                </span>
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
            <button type="button" className="family-replay" onClick={() => onInspectFamily(family.familyId)} title="Inspect family artifacts and jobs">
              <Search size={14} />
              Inspect
            </button>
            <button type="button" className="family-replay" onClick={() => onReplayRecipe(family.recipeId)} title="Replay family recipe">
              <Repeat size={14} />
              Replay
            </button>
            <button type="button" className="family-replay" onClick={() => onForkRecipe(family.recipe)} title="Fork family recipe">
              <GitFork size={14} />
              Fork
            </button>
          </article>
        );
      })}
    </div>
  );
}

export function FamilyDetailPanel({
  family,
  artifacts,
  jobs,
  selectedId,
  apiBase,
  onSelect,
  onCompare,
  onReplayRecipe,
  onForkRecipe,
  onCancelJob,
  onRetryJob,
}: {
  family: ResultFamily | null;
  artifacts: ArtifactRecord[];
  jobs: JobRecord[];
  selectedId: string | null;
  apiBase: string;
  onSelect: (artifactId: string | null) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onReplayRecipe: (recipeId: string) => void;
  onForkRecipe: (recipe: Recipe) => void;
} & JobActionHandlers) {
  if (!family) {
    return <div className="quiet-panel compact">No family selected</div>;
  }
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  const familyArtifacts = sortNewest(family.artifactIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact)));
  const familyJobs = sortNewestJobs(family.jobIds.map((jobId) => jobs.find((job) => job.job_id === jobId)).filter((job): job is JobRecord => Boolean(job)));
  const sourceIds = Object.values(family.recipe.inputs).filter((value): value is string => typeof value === "string" && value.length > 0);
  const sourceArtifacts = sourceIds.map((artifactId) => artifactMap.get(artifactId)).filter((artifact): artifact is ArtifactRecord => Boolean(artifact));

  return (
    <section className="family-detail">
      <div className="family-detail-head">
        <div>
          <span className="eyebrow">Family Detail</span>
          <strong>{shortOperatorName(family.operator)}</strong>
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
          Replay
        </button>
        <button type="button" onClick={() => onForkRecipe(family.recipe)}>
          <GitFork size={14} />
          Fork
        </button>
      </div>
      <div className="family-artifacts">
        {familyArtifacts.length ? (
          familyArtifacts.map((artifact) => (
            <article key={artifact.artifact_id} className={`family-artifact ${artifact.artifact_id === selectedId ? "selected" : ""}`}>
              <button type="button" className="family-artifact-main" onClick={() => onSelect(artifact.artifact_id)}>
                <ArtifactIcon artifact={artifact} />
                <div>
                  <strong>{artifactName(artifact)}</strong>
                  <span>{artifactMeta(artifact)}</span>
                </div>
              </button>
              {artifact.kind === "audio" ? (
                <>
                  <AudioDeck artifact={artifact} apiBase={apiBase} compact />
                  <div className="family-artifact-actions">
                    <button type="button" onClick={() => onCompare("a", artifact.artifact_id)}>A</button>
                    <button type="button" onClick={() => onCompare("b", artifact.artifact_id)}>B</button>
                  </div>
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
            Jobs
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
