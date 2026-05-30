import { useMemo, useState } from "react";
import { Archive, CircleDot, Database, Gauge, LoaderCircle, Plus, Search, SlidersHorizontal, X } from "lucide-react";

import {
  artifactFilterOptions,
  artifactFiltersActive,
  emptyArtifactFilters,
  filterArtifacts,
  type ArtifactDecisionFilter,
  type ArtifactFilterOption,
  type ArtifactFilterState,
  type ArtifactKindFilter,
  type ArtifactLineageFilter,
  type ArtifactMemoryRoleFilter,
  type ArtifactReuseIntentFilter,
} from "./artifactFilters";
import { ArtifactIcon } from "./artifactDisplay";
import { artifactMeta, artifactName, sortNewest, sortNewestJobs } from "./artifactUtils";
import { TinyWave } from "./audioDeck";
import type { ResultFamily } from "./controlPlane";
import { JobProgress, type JobActionHandlers } from "./jobProgress";
import { shortOperatorName } from "./jobUtils";
import { ListeningDecisionBadge } from "./listeningDecision";
import { memoryActionsForArtifact, memoryRoleFromArtifact, memoryRoleLabel, type MemoryReuseAction } from "./memoryModel";
import { archivableSessionArtifacts, recoverableArchiveArtifacts } from "./sessionRecovery";
import { summarizeSessionWorkspace, workspaceFocus, workspacePulseRows, type WorkspaceFocus, type WorkspacePulseRow } from "./sessionWorkspace";
import type { ArtifactRecord, JobRecord, OperatorName, SessionRecord } from "./types";

export function SessionTray({
  artifacts,
  archivedArtifacts,
  jobs,
  archivedJobs,
  families,
  runningJobs,
  selectedId,
  apiBase,
  activeSessionId,
  session,
  sessionStartedAt,
  creatingSession,
  archivingSession,
  recoveringArtifactId,
  archivingArtifactId,
  onSelect,
  onStartSession,
  onArchiveSession,
  onRecoverArtifact,
  onArchiveArtifact,
  onUseMemoryAction,
  onCancelJob,
  onRetryJob,
}: {
  artifacts: ArtifactRecord[];
  archivedArtifacts: ArtifactRecord[];
  jobs: JobRecord[];
  archivedJobs: JobRecord[];
  families: ResultFamily[];
  runningJobs: JobRecord[];
  selectedId: string | null;
  apiBase: string;
  activeSessionId: string | null;
  session: SessionRecord | null;
  sessionStartedAt: string;
  creatingSession: boolean;
  archivingSession: boolean;
  recoveringArtifactId: string | null;
  archivingArtifactId: string | null;
  onSelect: (artifactId: string | null) => void;
  onStartSession: () => void;
  onArchiveSession: () => void;
  onRecoverArtifact: (artifact: ArtifactRecord) => void;
  onArchiveArtifact: (artifact: ArtifactRecord) => void;
  onUseMemoryAction?: (artifact: ArtifactRecord, action: MemoryReuseAction) => void;
} & JobActionHandlers) {
  const [artifactFilters, setArtifactFilters] = useState<ArtifactFilterState>(emptyArtifactFilters);
  const filterContext = useMemo(() => ({ jobs: [...jobs, ...archivedJobs], families }), [jobs, archivedJobs, families]);
  const filterCorpus = useMemo(() => [...artifacts, ...archivedArtifacts], [artifacts, archivedArtifacts]);
  const filterOptions = useMemo(() => artifactFilterOptions(filterCorpus, filterContext), [filterCorpus, filterContext]);
  const filterActive = artifactFiltersActive(artifactFilters);
  const filteredSessionArtifacts = useMemo(() => filterArtifacts(artifacts, artifactFilters, filterContext), [artifacts, artifactFilters, filterContext]);
  const filteredArchiveArtifacts = useMemo(() => filterArtifacts(archivedArtifacts, artifactFilters, filterContext), [archivedArtifacts, artifactFilters, filterContext]);
  const sessionArtifactRows = sortNewest(filteredSessionArtifacts).slice(0, 8);
  const sessionJobs = sortNewestJobs(jobs).slice(0, 4);
  const archiveArtifactRows = sortNewest(filteredArchiveArtifacts).slice(0, 10);
  const archiveJobRows = filterActive ? [] : sortNewestJobs(archivedJobs).slice(0, 10);
  const activeJobs = runningJobs.slice(0, 3);
  const archivableIds = useMemo(
    () => new Set(archivableSessionArtifacts(sessionArtifactRows, activeSessionId).map((artifact) => artifact.artifact_id)),
    [sessionArtifactRows, activeSessionId],
  );
  const recoverableIds = useMemo(
    () => new Set(recoverableArchiveArtifacts(archiveArtifactRows, activeSessionId).map((artifact) => artifact.artifact_id)),
    [archiveArtifactRows, activeSessionId],
  );
  const workspaceSummary = useMemo(
    () => summarizeSessionWorkspace({ artifacts, archivedArtifacts, jobs, archivedJobs, families, runningJobs, selectedId }),
    [artifacts, archivedArtifacts, jobs, archivedJobs, families, runningJobs, selectedId],
  );
  const pulseRows = useMemo(() => workspacePulseRows(workspaceSummary), [workspaceSummary]);
  const focus = useMemo(() => workspaceFocus(workspaceSummary), [workspaceSummary]);

  return (
    <div className="session-tray">
      <div className="session-head">
        <div>
          <span className="eyebrow">Session Memory</span>
          <strong>{session?.name ?? formatSessionStamp(sessionStartedAt)}</strong>
        </div>
        <div className="session-actions">
          <button type="button" className="session-new" onClick={onStartSession} title="New session" disabled={creatingSession || archivingSession}>
            {creatingSession ? <LoaderCircle className="spin" size={16} /> : <Plus size={16} />}
            {creatingSession ? "Creating" : "New"}
          </button>
          <button
            type="button"
            className="session-new"
            onClick={onArchiveSession}
            title="Remember this session and start a clean one"
            disabled={!session || session.status === "archived" || creatingSession || archivingSession || activeJobs.length > 0}
          >
            {archivingSession ? <LoaderCircle className="spin" size={16} /> : <Archive size={16} />}
            {archivingSession ? "Remembering" : "Remember"}
          </button>
        </div>
      </div>

      <WorkspacePulse rows={pulseRows} focus={focus} onSelect={onSelect} />

      {activeJobs.length ? (
        <div className="session-block">
          <span className="session-label">Pending takes</span>
          {activeJobs.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
        </div>
      ) : null}

      <ArtifactFilterPanel
        filters={artifactFilters}
        options={filterOptions}
        active={filterActive}
        onChange={setArtifactFilters}
      />

      <div className="session-block">
        <span className="session-label">
          Takes
          {filterActive ? <i>{filteredSessionArtifacts.length}/{artifacts.length}</i> : null}
        </span>
        {sessionArtifactRows.length ? (
          <div className="session-artifacts">
            {sessionArtifactRows.map((artifact) => (
              <SessionArtifactRow
                key={artifact.artifact_id}
                artifact={artifact}
                selected={artifact.artifact_id === selectedId}
                apiBase={apiBase}
                onSelect={onSelect}
                action={
                  archivableIds.has(artifact.artifact_id)
                    ? {
                        label: archivingArtifactId === artifact.artifact_id ? "Remembering" : "Remember",
                        disabled: Boolean(archivingArtifactId),
                        onAction: () => onArchiveArtifact(artifact),
                      }
                    : undefined
                }
              />
            ))}
          </div>
        ) : (
          <div className="quiet-panel compact">{filterActive ? "No matching takes" : "Fresh session"}</div>
        )}
      </div>

      {sessionJobs.length ? (
        <details className="archive-drawer">
          <summary>
            <Gauge size={15} />
            Gesture history
            <span>{jobs.length}</span>
          </summary>
          <div className="archive-list">
            {sessionJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
            ))}
          </div>
        </details>
      ) : null}

      <details className="archive-drawer">
        <summary>
          <Database size={15} />
          Memory
          <span>{filterActive ? `${filteredArchiveArtifacts.length}/${archivedArtifacts.length}` : archivedArtifacts.length + archivedJobs.length}</span>
        </summary>
        <div className="archive-list">
          {archiveArtifactRows.map((artifact) => (
            <SessionArtifactRow
              key={artifact.artifact_id}
              artifact={artifact}
              selected={artifact.artifact_id === selectedId}
              apiBase={apiBase}
              onSelect={onSelect}
              memoryActions={memoryActionsForArtifact(artifact, { activeSessionId }).filter((item) => item.intent !== "recover")}
              onMemoryAction={onUseMemoryAction ? (action) => onUseMemoryAction(artifact, action) : undefined}
              action={
                recoverableIds.has(artifact.artifact_id)
                  ? {
                      label: recoveringArtifactId === artifact.artifact_id ? "Recovering" : "Recover",
                      disabled: Boolean(recoveringArtifactId),
                      onAction: () => onRecoverArtifact(artifact),
                    }
                  : undefined
              }
            />
          ))}
          {archiveJobRows.map((job) => (
            <JobProgress key={job.job_id} job={job} compact onCancelJob={onCancelJob} onRetryJob={onRetryJob} />
          ))}
          {!archiveArtifactRows.length && !archiveJobRows.length ? <div className="quiet-panel compact">{filterActive ? "No matching takes" : "Memory empty"}</div> : null}
        </div>
      </details>
    </div>
  );
}

function WorkspacePulse({
  rows,
  focus,
  onSelect,
}: {
  rows: WorkspacePulseRow[];
  focus: WorkspaceFocus;
  onSelect: (artifactId: string | null) => void;
}) {
  return (
    <section className="workspace-pulse" aria-label="Workspace pulse">
      <div className={`workspace-focus ${focus.tone}`}>
        <span>
          <CircleDot size={14} />
          {focus.label}
        </span>
        <strong>{focus.detail}</strong>
        {focus.artifactId ? (
          <button type="button" onClick={() => onSelect(focus.artifactId ?? null)}>
            Open
          </button>
        ) : null}
      </div>
      <div className="workspace-pulse-grid">
        {rows.map((row) => (
          <div key={row.key} className={`workspace-pulse-node ${row.tone}`}>
            <span>{row.label}</span>
            <strong>{row.value}</strong>
            <small>{row.detail}</small>
          </div>
        ))}
      </div>
    </section>
  );
}

const decisionFilters: { value: ArtifactDecisionFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "keeper", label: "Keep" },
  { value: "maybe", label: "Maybe" },
  { value: "rejected", label: "Reject" },
  { value: "undecided", label: "Open" },
];

function ArtifactFilterPanel({
  filters,
  options,
  active,
  onChange,
}: {
  filters: ArtifactFilterState;
  options: ReturnType<typeof artifactFilterOptions>;
  active: boolean;
  onChange: (filters: ArtifactFilterState) => void;
}) {
  const update = (patch: Partial<ArtifactFilterState>) => onChange({ ...filters, ...patch });
  return (
    <div className="artifact-filter-panel" aria-label="Take filters">
      <div className="artifact-filter-head">
        <span className="session-label">
          <SlidersHorizontal size={13} />
          Find takes
        </span>
        {active ? (
          <button type="button" className="archive-clear" aria-label="Clear take filters" onClick={() => onChange(emptyArtifactFilters)}>
            <X size={14} />
          </button>
        ) : null}
      </div>
      <label className="artifact-filter-search">
        <Search size={14} />
        <input
          type="search"
          aria-label="Filter takes"
          value={filters.query}
          onChange={(event) => update({ query: event.target.value })}
          placeholder="label, notes, prompt, tags"
        />
      </label>
      <div className="artifact-decision-filter" aria-label="Listening decision filters">
        {decisionFilters.map((item) => (
          <button
            key={item.value}
            type="button"
            aria-pressed={filters.decision === item.value}
            className={filters.decision === item.value ? `selected ${item.value}` : item.value}
            onClick={() => update({ decision: item.value })}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="artifact-filter-selects">
        <FilterSelect
          label="Kind"
          value={filters.kind === "all" ? "" : filters.kind}
          allLabel="all kinds"
          options={options.kinds}
          onChange={(value) => update({ kind: (value || "all") as ArtifactKindFilter })}
        />
        <FilterSelect
          label="Model"
          value={filters.model}
          allLabel="any model"
          options={options.models}
          onChange={(model) => update({ model })}
        />
        <FilterSelect
          label="Gesture"
          value={filters.operator}
          allLabel="any gesture"
          options={options.operators.map((option) => ({ ...option, label: shortOperatorName(option.value as OperatorName) }))}
          onChange={(operator) => update({ operator })}
        />
        <FilterSelect
          label="Branch"
          value={filters.familyId}
          allLabel="any branch"
          options={options.families}
          onChange={(familyId) => update({ familyId })}
        />
        <FilterSelect
          label="Lineage"
          value={filters.lineage === "all" ? "" : filters.lineage}
          allLabel="any lineage"
          options={[
            { value: "source", label: "source", count: 0 },
            { value: "derived", label: "derived", count: 0 },
            { value: "has_sources", label: "has source", count: 0 },
          ]}
          onChange={(lineage) => update({ lineage: (lineage || "all") as ArtifactLineageFilter })}
        />
        <FilterSelect
          label="Role"
          value={filters.memoryRole === "all" ? "" : filters.memoryRole}
          allLabel="any role"
          options={options.memoryRoles}
          onChange={(memoryRole) => update({ memoryRole: (memoryRole || "all") as ArtifactMemoryRoleFilter })}
        />
        <FilterSelect
          label="Reuse"
          value={filters.reuseIntent === "all" ? "" : filters.reuseIntent}
          allLabel="any reuse"
          options={options.reuseIntents.map((option) => ({ ...option, label: option.label.replaceAll("_", " ") }))}
          onChange={(reuseIntent) => update({ reuseIntent: (reuseIntent || "all") as ArtifactReuseIntentFilter })}
        />
        <FilterSelect
          label="Tag"
          value={filters.tag}
          allLabel="any tag"
          options={options.tags}
          onChange={(tag) => update({ tag })}
        />
      </div>
    </div>
  );
}

function FilterSelect({
  label,
  value,
  allLabel,
  options,
  onChange,
}: {
  label: string;
  value: string;
  allLabel: string;
  options: ArtifactFilterOption[];
  onChange: (value: string) => void;
}) {
  return (
    <label>
      <span>{label}</span>
      <select aria-label={`Filter ${label.toLowerCase()}`} value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{allLabel}</option>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.count ? `${option.label} (${option.count})` : option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function SessionArtifactRow({
  artifact,
  selected,
  apiBase,
  onSelect,
  action,
  memoryActions,
  onMemoryAction,
}: {
  artifact: ArtifactRecord;
  selected: boolean;
  apiBase: string;
  onSelect: (artifactId: string | null) => void;
  action?: {
    label: string;
    disabled?: boolean;
    onAction: () => void;
  };
  memoryActions?: readonly MemoryReuseAction[];
  onMemoryAction?: (action: MemoryReuseAction) => void;
}) {
  const memoryRole = memoryRoleFromArtifact(artifact);
  return (
    <article className={`session-artifact ${selected ? "selected" : ""}`}>
      <button type="button" className="session-artifact-main" onClick={() => onSelect(artifact.artifact_id)}>
        <ArtifactIcon artifact={artifact} />
        <div>
          <strong>{artifactName(artifact)}</strong>
          <span>{artifactMeta(artifact)}</span>
          <ListeningDecisionBadge artifact={artifact} />
          {artifact.tags.length ? (
            <span className="artifact-tags">
              {artifact.tags.slice(0, 3).map((tag) => (
                <i key={tag}>#{tag}</i>
              ))}
            </span>
          ) : null}
          {memoryRole ? <span className="memory-role">role: {memoryRoleLabel(memoryRole)}</span> : null}
        </div>
        {artifact.kind === "audio" ? <TinyWave artifact={artifact} apiBase={apiBase} /> : null}
      </button>
      {memoryActions?.length ? (
        <div className="memory-reuse-actions" aria-label={`${artifact.label ?? artifact.artifact_id} memory reuse actions`}>
          {memoryActions.map((item) => (
            <button
              key={item.id}
              type="button"
              disabled={!item.available || !onMemoryAction}
              title={item.disabledReason ?? item.description}
              onClick={() => onMemoryAction?.(item)}
            >
              {item.label}
            </button>
          ))}
        </div>
      ) : null}
      {action ? (
        <button type="button" className="session-artifact-action" disabled={action.disabled} onClick={action.onAction}>
          {action.label}
        </button>
      ) : null}
    </article>
  );
}

function formatSessionStamp(value: string) {
  const started = Date.parse(value);
  if (!Number.isFinite(started) || started <= 0) return "All work";
  const date = new Date(started);
  return `Since ${date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`;
}
