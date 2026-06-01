import { useMemo } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";
import { AudioLines, FlaskConical, Sparkles } from "lucide-react";

import { createApi } from "./api";
import type { ArtifactAnnotationPayload } from "./api";
import { artifactName, sortNewest } from "./artifactUtils";
import { AudioDeck } from "./audioDeck";
import {
  listeningDecision,
  listeningDecisionLabel,
  ListeningDecisionBadge,
  ListeningDecisionControls,
  type ListeningDecision,
} from "./listeningDecision";
import type { ArtifactRecord, AudioDescriptorComparison } from "./types";

export interface PromptCandidateAction {
  rank?: number;
  prompt: string;
  score?: unknown;
  source?: string;
}

export interface PromptCandidateGenerationRequest extends PromptCandidateAction {
  bundleArtifactId: string;
  scorer?: string;
  searchMode?: string;
  searchModel?: string;
  searchDurationSeconds?: number;
}

export interface DescriptorDeltaRow {
  key: string;
  label: string;
  value: string;
  tone: "up" | "down" | "neutral";
  title: string;
}

export interface PromptDecisionCorrelationRow {
  artifactId: string;
  label: string;
  prompt?: string | null;
  decision: ListeningDecision | null;
  decisionLabel: string;
  note?: string | null;
  deltas: DescriptorDeltaRow[];
  rawDelta: Record<string, number>;
}

export interface PromptDecisionMemoryRow {
  prompt: string;
  total: number;
  listened: number;
  keeper: number;
  maybe: number;
  rejected: number;
  undecided: number;
  bundleIds: string[];
  latestArtifactId: string;
  latestAt: string;
  latestNote?: string | null;
}

export interface PromptSearchRunComparisonRow {
  bundleId: string;
  label: string;
  current: boolean;
  totalTakes: number;
  listened: number;
  keeper: number;
  maybe: number;
  rejected: number;
  undecided: number;
  promptCount: number;
  prompts: string[];
  scorer?: string;
  searchMode?: string;
  model?: string;
  durationSeconds?: number;
  latestAt: string;
  latestArtifactId?: string;
  latestNote?: string | null;
}

const DESCRIPTOR_DELTA_KEYS = [
  { key: "rms_dbfs", label: "level" },
  { key: "spectral_centroid_hz", label: "bright" },
  { key: "spectral_flux", label: "motion" },
  { key: "spectral_flatness", label: "noise" },
  { key: "stereo_width", label: "width" },
] as const;

export function PromptSearchCandidatePanel({
  summary,
  artifact,
  artifacts,
  apiBase,
  onUsePrompt,
  onGeneratePrompt,
  onAnnotate,
  onUseInRecipe,
  onCompare,
  onSelectArtifact,
}: {
  summary: Record<string, unknown> | undefined;
  artifact: ArtifactRecord;
  artifacts: ArtifactRecord[];
  apiBase: string;
  onUsePrompt: (prompt: string) => void;
  onGeneratePrompt: (request: PromptCandidateGenerationRequest) => void;
  onAnnotate: (artifactId: string, payload: ArtifactAnnotationPayload) => void;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onSelectArtifact: (artifactId: string | null) => void;
}) {
  const candidates = promptSearchCandidates(summary);
  if (!candidates.length) return null;
  const allGeneratedTakes = promptCandidateGeneratedArtifacts(artifact.artifact_id, artifacts);
  const targetArtifact = promptSearchTargetArtifact(artifact, artifacts);
  const decisionMemory = promptDecisionMemoryRows(artifacts);
  const runComparisons = promptSearchRunComparisonRows(artifact, artifacts, summary);
  return (
    <div className="prompt-candidate-panel" aria-label="Prompt search candidates">
      <div className="prompt-candidate-head">
        <strong>Candidate prompts</strong>
        <span>
          <AudioLines size={13} />
          {allGeneratedTakes.length} take{allGeneratedTakes.length === 1 ? "" : "s"}
        </span>
      </div>
      <PromptDecisionCorrelation apiBase={apiBase} target={targetArtifact} takes={allGeneratedTakes} />
      <PromptDecisionMemory rows={decisionMemory} currentBundleId={artifact.artifact_id} />
      <PromptSearchRunComparison rows={runComparisons} />
      {candidates.map((candidate) => {
        const generated = promptCandidateGeneratedArtifacts(artifact.artifact_id, artifacts, candidate.prompt);
        return (
          <article key={`${candidate.rank ?? "candidate"}:${candidate.prompt}`} className={generated.length ? "has-takes" : ""}>
            <div>
              <span>{candidate.prompt}</span>
              <small>{promptCandidateMeta(candidate)}</small>
            </div>
            <div className="prompt-candidate-actions">
              <button type="button" onClick={() => onUsePrompt(candidate.prompt)}>
                Use
              </button>
              <button type="button" onClick={() => onGeneratePrompt(promptCandidateGenerationRequest(candidate, artifact, summary))}>
                <Sparkles aria-hidden="true" size={13} />
                Generate
              </button>
              <button type="button" onClick={() => onUseInRecipe("prompt", candidate.prompt, "experiment.alpha_sweep")}>
                Sweep
              </button>
            </div>
            {generated.length ? (
              <div className="prompt-generated-takes" aria-label={`Generated takes for ${candidate.prompt}`}>
                {generated.slice(0, 3).map((take) => (
                  <div key={take.artifact_id} className="prompt-generated-take">
                    <button type="button" onClick={() => onSelectArtifact(take.artifact_id)} title={artifactName(take)}>
                      <AudioLines size={14} />
                      <span>{generatedTakeLabel(take)}</span>
                      <ListeningDecisionBadge artifact={take} />
                    </button>
                    <AudioDeck artifact={take} apiBase={apiBase} compact />
                    <PromptTakeDescriptorDelta apiBase={apiBase} target={targetArtifact} take={take} />
                    <div className="prompt-generated-actions">
                      <button type="button" onClick={() => onCompare("a", take.artifact_id)} title="Pin generated take as anchor">
                        <FlaskConical aria-hidden="true" size={12} />
                        Anchor
                      </button>
                      <button type="button" onClick={() => onCompare("b", take.artifact_id)} title="Pin generated take as source">
                        <FlaskConical aria-hidden="true" size={12} />
                        Source
                      </button>
                    </div>
                    <ListeningDecisionControls artifact={take} source="prompt_candidate_bench" onDecide={onAnnotate} />
                  </div>
                ))}
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}

function PromptSearchRunComparison({ rows }: { rows: PromptSearchRunComparisonRow[] }) {
  const visibleRows = rows.filter((row) => row.totalTakes > 0 || row.current).slice(0, 6);
  if (!visibleRows.length) return null;
  const totalRuns = visibleRows.length;
  const totalTakes = visibleRows.reduce((total, row) => total + row.totalTakes, 0);
  return (
    <section className="prompt-run-comparison" aria-label="Prompt take family comparison">
      <div>
        <strong>Prompt take families</strong>
        <span>{totalRuns} run{totalRuns === 1 ? "" : "s"} · {totalTakes} take{totalTakes === 1 ? "" : "s"}</span>
      </div>
      {visibleRows.map((row) => (
        <article key={row.bundleId} className={row.current ? "current" : ""}>
          <div>
            <strong>{row.label}</strong>
            <span>{row.keeper}K · {row.maybe}M · {row.rejected}R</span>
          </div>
          <small>{promptRunMeta(row)}</small>
          <p>{row.prompts.slice(0, 2).join(" / ") || "no generated prompts yet"}</p>
        </article>
      ))}
    </section>
  );
}

function PromptDecisionMemory({ rows, currentBundleId }: { rows: PromptDecisionMemoryRow[]; currentBundleId: string }) {
  const visibleRows = rows.filter((row) => row.listened > 0 || row.total > 1).slice(0, 5);
  if (!visibleRows.length) return null;
  return (
    <section className="prompt-decision-memory" aria-label="Prompt search decision memory">
      <div>
        <strong>Prompt memory</strong>
        <span>{visibleRows.reduce((total, row) => total + row.listened, 0)} listened</span>
      </div>
      {visibleRows.map((row) => {
        const current = row.bundleIds.includes(currentBundleId);
        return (
          <article key={row.prompt} className={current ? "current" : ""}>
            <strong>{row.prompt}</strong>
            <span>{row.keeper} keeper · {row.maybe} maybe · {row.rejected} reject</span>
            <small>{row.latestNote || `${row.total} take${row.total === 1 ? "" : "s"} across ${row.bundleIds.length} run${row.bundleIds.length === 1 ? "" : "s"}`}</small>
          </article>
        );
      })}
    </section>
  );
}

function PromptDecisionCorrelation({
  apiBase,
  target,
  takes,
}: {
  apiBase: string;
  target: ArtifactRecord | null;
  takes: ArtifactRecord[];
}) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const inspectedTakes = takes.slice(0, 12);
  const comparisonQueries = useQueries({
    queries: inspectedTakes.map((take) => ({
      queryKey: ["audio-descriptor-comparison", apiBase, target?.artifact_id, take.artifact_id],
      queryFn: () => {
        if (!target) throw new Error("prompt-search target audio unavailable");
        return api.audioDescriptorComparison(target.artifact_id, take.artifact_id);
      },
      enabled: Boolean(target),
      staleTime: 60000,
    })),
  });
  if (!takes.length) return null;
  if (!target) {
    return <div className="prompt-decision-study unavailable">target listening comparison unavailable</div>;
  }
  const comparisons = new Map<string, AudioDescriptorComparison>();
  comparisonQueries.forEach((query, index) => {
    if (query.data) comparisons.set(inspectedTakes[index].artifact_id, query.data);
  });
  const rows = promptDecisionCorrelationRows(inspectedTakes, comparisons);
  const summary = promptDecisionSummary(rows);
  const loading = comparisonQueries.some((query) => query.isLoading);
  return (
    <section className="prompt-decision-study" aria-label="Prompt candidate listening comparison">
      <div className="prompt-decision-head">
        <strong>Candidate listening</strong>
        <span>{loading ? "measuring..." : `${rows.filter((row) => row.decision).length}/${rows.length} listened`}</span>
      </div>
      <div className="prompt-decision-summary">
        {summary.map((item) => (
          <i key={item.label} className={item.tone}>
            {item.label}
            <small>{item.value}</small>
          </i>
        ))}
      </div>
      <div className="prompt-decision-rows">
        {rows.slice(0, 4).map((row) => (
          <article key={row.artifactId}>
            <div>
              <strong>{row.label}</strong>
              <span className={row.decision ?? "undecided"}>{row.decisionLabel}</span>
            </div>
            <div className="prompt-decision-deltas">
              {row.deltas.slice(0, 3).map((delta) => (
                <i key={delta.key} className={delta.tone} title={delta.title}>
                  {delta.label}
                  <small>{delta.value}</small>
                </i>
              ))}
            </div>
            {row.note ? <p>{row.note}</p> : null}
          </article>
        ))}
      </div>
    </section>
  );
}

function PromptTakeDescriptorDelta({
  apiBase,
  target,
  take,
}: {
  apiBase: string;
  target: ArtifactRecord | null;
  take: ArtifactRecord;
}) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const comparison = useQuery({
    queryKey: ["audio-descriptor-comparison", apiBase, target?.artifact_id, take.artifact_id],
    queryFn: () => {
      if (!target) throw new Error("prompt-search target audio unavailable");
      return api.audioDescriptorComparison(target.artifact_id, take.artifact_id);
    },
    enabled: Boolean(target),
    staleTime: 60000,
  });
  if (!target) {
    return <div className="descriptor-delta unavailable">target delta unavailable</div>;
  }
  if (comparison.isLoading) {
    return <div className="descriptor-delta unavailable">measuring delta...</div>;
  }
  if (comparison.isError || !comparison.data) {
    return <div className="descriptor-delta unavailable">delta unavailable</div>;
  }
  const rows = audioDescriptorDeltaRows(comparison.data);
  if (!rows.length) return null;
  return (
    <div className="descriptor-delta" aria-label={`Descriptor delta for ${artifactName(take)}`}>
      {rows.map((row) => (
        <i key={row.key} className={row.tone} title={row.title}>
          <span>{row.label}</span>
          <strong>{row.value}</strong>
        </i>
      ))}
    </div>
  );
}

export function promptSearchCandidates(summary: Record<string, unknown> | undefined): PromptCandidateAction[] {
  const promptSearch = objectValue(summary?.prompt_search);
  if (!promptSearch) return [];
  const seen = new Set<string>();
  const candidates: PromptCandidateAction[] = [];
  const add = (candidate: Record<string, unknown>, fallbackRank?: number) => {
    const prompt = typeof candidate.prompt === "string" ? candidate.prompt.trim() : "";
    if (!prompt || seen.has(prompt)) return;
    seen.add(prompt);
    candidates.push({
      rank: typeof candidate.rank === "number" ? candidate.rank : fallbackRank,
      prompt,
      score: candidate.score,
      source: typeof candidate.source === "string" ? candidate.source : undefined,
    });
  };
  add({ prompt: promptSearch.prompt, score: promptSearch.score, source: "selected", rank: 1 }, 1);
  for (const candidate of arrayOfObjects(promptSearch.families)) add(candidate);
  return candidates.slice(0, 8);
}

export function promptCandidateGeneratedArtifacts(bundleArtifactId: string, artifacts: ArtifactRecord[], prompt?: string): ArtifactRecord[] {
  return sortNewest(
    artifacts.filter((artifact) => {
      if (artifact.kind !== "audio") return false;
      if (!artifact.source_artifact_ids.includes(bundleArtifactId)) return false;
      if (prompt && artifact.prompt !== prompt) return false;
      return artifact.metadata.generation_origin === "prompt_search_candidate";
    }),
  );
}

export function promptSearchTargetArtifact(bundleArtifact: ArtifactRecord, artifacts: ArtifactRecord[]): ArtifactRecord | null {
  for (const sourceId of bundleArtifact.source_artifact_ids) {
    const source = artifacts.find((artifact) => artifact.artifact_id === sourceId);
    if (source?.kind === "audio") return source;
  }
  return null;
}

export function audioDescriptorDeltaRows(comparison: AudioDescriptorComparison): DescriptorDeltaRow[] {
  return DESCRIPTOR_DELTA_KEYS.flatMap((item) => {
    const value = comparison.delta[item.key];
    if (typeof value !== "number" || !Number.isFinite(value)) return [];
    return [{
      key: item.key,
      label: item.label,
      value: formatDescriptorDelta(item.key, value),
      tone: descriptorDeltaTone(item.key, value),
      title: `${item.label}: generated take ${formatDescriptorDelta(item.key, value)} vs target`,
    }];
  });
}

export function promptDecisionCorrelationRows(
  takes: ArtifactRecord[],
  comparisons: Map<string, AudioDescriptorComparison>,
): PromptDecisionCorrelationRow[] {
  return takes.map((take) => {
    const comparison = comparisons.get(take.artifact_id);
    const decision = listeningDecision(take);
    return {
      artifactId: take.artifact_id,
      label: generatedTakeLabel(take),
      prompt: take.prompt,
      decision,
      decisionLabel: listeningDecisionLabel(decision),
      note: listeningDecisionNote(take),
      deltas: comparison ? audioDescriptorDeltaRows(comparison) : [],
      rawDelta: comparison?.delta ?? {},
    };
  });
}

export function promptDecisionSummary(rows: PromptDecisionCorrelationRow[]): { label: string; value: string; tone: string }[] {
  const listened = rows.filter((row) => row.decision);
  const keepers = rows.filter((row) => row.decision === "keeper");
  const maybe = rows.filter((row) => row.decision === "maybe");
  const rejected = rows.filter((row) => row.decision === "rejected");
  const summary = [
    { label: "listened", value: `${listened.length}/${rows.length}`, tone: listened.length ? "neutral" : "empty" },
    { label: "keepers", value: String(keepers.length), tone: keepers.length ? "up" : "empty" },
    { label: "maybe", value: String(maybe.length), tone: maybe.length ? "neutral" : "empty" },
    { label: "rejects", value: String(rejected.length), tone: rejected.length ? "down" : "empty" },
  ];
  const keeperBright = averageDelta(keepers, "spectral_centroid_hz");
  const keeperLevel = averageDelta(keepers, "rms_dbfs");
  if (keeperBright !== null) {
    summary.push({ label: "keeper bright", value: formatDescriptorDelta("spectral_centroid_hz", keeperBright), tone: descriptorDeltaTone("spectral_centroid_hz", keeperBright) });
  }
  if (keeperLevel !== null) {
    summary.push({ label: "keeper level", value: formatDescriptorDelta("rms_dbfs", keeperLevel), tone: descriptorDeltaTone("rms_dbfs", keeperLevel) });
  }
  return summary;
}

export function promptDecisionMemoryRows(artifacts: ArtifactRecord[]): PromptDecisionMemoryRow[] {
  const rowsByPrompt = new Map<string, PromptDecisionMemoryRow>();
  for (const take of sortNewest(artifacts)) {
    if (!isPromptGeneratedTake(take)) continue;
    const prompt = take.prompt?.trim();
    if (!prompt) continue;
    const decision = listeningDecision(take);
    const bundleId = promptSearchBundleId(take);
    const existing = rowsByPrompt.get(prompt) ?? {
      prompt,
      total: 0,
      listened: 0,
      keeper: 0,
      maybe: 0,
      rejected: 0,
      undecided: 0,
      bundleIds: [],
      latestArtifactId: take.artifact_id,
      latestAt: take.created_at,
      latestNote: null,
    };
    existing.total += 1;
    if (decision) existing.listened += 1;
    if (decision === "keeper") existing.keeper += 1;
    if (decision === "maybe") existing.maybe += 1;
    if (decision === "rejected") existing.rejected += 1;
    if (!decision) existing.undecided += 1;
    if (bundleId && !existing.bundleIds.includes(bundleId)) existing.bundleIds.push(bundleId);
    if (Date.parse(take.created_at) >= Date.parse(existing.latestAt)) {
      existing.latestArtifactId = take.artifact_id;
      existing.latestAt = take.created_at;
      existing.latestNote = listeningDecisionNote(take);
    }
    rowsByPrompt.set(prompt, existing);
  }
  return [...rowsByPrompt.values()].sort((a, b) =>
    b.keeper - a.keeper
    || b.listened - a.listened
    || b.total - a.total
    || Date.parse(b.latestAt) - Date.parse(a.latestAt),
  );
}

export function promptSearchRunComparisonRows(
  currentBundle: ArtifactRecord,
  artifacts: ArtifactRecord[],
  summary?: Record<string, unknown>,
): PromptSearchRunComparisonRow[] {
  const bundles = new Map<string, ArtifactRecord>();
  for (const artifact of artifacts) {
    if (artifact.kind === "bundle") bundles.set(artifact.artifact_id, artifact);
  }
  bundles.set(currentBundle.artifact_id, currentBundle);

  const rowsByBundle = new Map<string, PromptSearchRunComparisonRow>();
  const ensureRow = (bundleId: string): PromptSearchRunComparisonRow => {
    const existing = rowsByBundle.get(bundleId);
    if (existing) return existing;
    const bundle = bundles.get(bundleId);
    const row: PromptSearchRunComparisonRow = {
      bundleId,
      label: bundle ? artifactName(bundle) : `prompt run ${shortArtifactId(bundleId)}`,
      current: bundleId === currentBundle.artifact_id,
      totalTakes: 0,
      listened: 0,
      keeper: 0,
      maybe: 0,
      rejected: 0,
      undecided: 0,
      promptCount: 0,
      prompts: [],
      latestAt: bundle?.created_at ?? "",
      latestArtifactId: undefined,
      latestNote: null,
    };
    rowsByBundle.set(bundleId, row);
    return row;
  };

  const currentRow = ensureRow(currentBundle.artifact_id);
  const promptSearch = objectValue(summary?.prompt_search);
  if (promptSearch) {
    currentRow.scorer = currentRow.scorer ?? stringValue(promptSearch.scorer);
    currentRow.searchMode = currentRow.searchMode ?? stringValue(promptSearch.search_mode);
    currentRow.model = currentRow.model ?? stringValue(promptSearch.model);
    currentRow.durationSeconds = currentRow.durationSeconds ?? numberValue(promptSearch.duration_seconds);
    const selectedPrompt = stringValue(promptSearch.prompt);
    if (selectedPrompt && !currentRow.prompts.includes(selectedPrompt)) currentRow.prompts.push(selectedPrompt);
  }

  for (const take of sortNewest(artifacts)) {
    if (!isPromptGeneratedTake(take)) continue;
    const bundleId = promptSearchBundleId(take);
    if (!bundleId) continue;
    const row = ensureRow(bundleId);
    const decision = listeningDecision(take);
    const prompt = take.prompt?.trim();
    row.totalTakes += 1;
    if (decision) row.listened += 1;
    if (decision === "keeper") row.keeper += 1;
    if (decision === "maybe") row.maybe += 1;
    if (decision === "rejected") row.rejected += 1;
    if (!decision) row.undecided += 1;
    if (prompt && !row.prompts.includes(prompt)) row.prompts.push(prompt);
    row.promptCount = row.prompts.length;
    row.scorer = row.scorer ?? stringValue(take.metadata.prompt_search_scorer);
    row.searchMode = row.searchMode ?? stringValue(take.metadata.prompt_search_mode);
    row.model = row.model ?? stringValue(take.metadata.prompt_search_model);
    row.durationSeconds = row.durationSeconds ?? numberValue(take.metadata.prompt_search_duration_seconds);
    if (!row.latestAt || Date.parse(take.created_at) >= Date.parse(row.latestAt)) {
      row.latestAt = take.created_at;
      row.latestArtifactId = take.artifact_id;
      row.latestNote = listeningDecisionNote(take);
    }
  }

  for (const row of rowsByBundle.values()) {
    row.promptCount = row.prompts.length;
  }

  return [...rowsByBundle.values()].sort((a, b) =>
    Number(b.current) - Number(a.current)
    || b.keeper - a.keeper
    || b.listened - a.listened
    || b.totalTakes - a.totalTakes
    || Date.parse(b.latestAt || "1970-01-01") - Date.parse(a.latestAt || "1970-01-01"),
  );
}

export function promptCandidateMeta(candidate: { rank?: number; source?: string; score?: unknown }) {
  const parts = [
    candidate.rank !== undefined ? `#${candidate.rank}` : "",
    typeof candidate.source === "string" ? candidate.source : "",
    candidate.score !== undefined ? formatMaybeNumber(candidate.score) : "",
  ].filter(Boolean);
  return parts.join(" · ");
}

function isPromptGeneratedTake(artifact: ArtifactRecord): boolean {
  return artifact.kind === "audio" && artifact.metadata.generation_origin === "prompt_search_candidate";
}

function promptSearchBundleId(artifact: ArtifactRecord): string | null {
  if (typeof artifact.metadata.prompt_search_bundle_id === "string") return artifact.metadata.prompt_search_bundle_id;
  return artifact.source_artifact_ids.find((sourceId) => sourceId.startsWith("art_")) ?? null;
}

function promptCandidateGenerationRequest(
  candidate: PromptCandidateAction,
  artifact: ArtifactRecord,
  summary: Record<string, unknown> | undefined,
): PromptCandidateGenerationRequest {
  const promptSearch = objectValue(summary?.prompt_search);
  return {
    ...candidate,
    bundleArtifactId: artifact.artifact_id,
    scorer: stringValue(promptSearch?.scorer),
    searchMode: stringValue(promptSearch?.search_mode),
    searchModel: stringValue(promptSearch?.model),
    searchDurationSeconds: numberValue(promptSearch?.duration_seconds),
  };
}

function generatedTakeLabel(artifact: ArtifactRecord) {
  const rank = typeof artifact.metadata.prompt_candidate_rank === "number" ? `#${artifact.metadata.prompt_candidate_rank}` : "take";
  const seed = typeof artifact.metadata.seed === "number" ? ` · seed ${artifact.metadata.seed}` : "";
  return `${rank}${seed}`;
}

function listeningDecisionNote(artifact: ArtifactRecord): string | null {
  if (typeof artifact.metadata.listening_decision_note === "string" && artifact.metadata.listening_decision_note.trim()) {
    return artifact.metadata.listening_decision_note.trim();
  }
  return artifact.notes?.trim() || null;
}

function averageDelta(rows: PromptDecisionCorrelationRow[], key: string): number | null {
  const values = rows.map((row) => row.rawDelta[key]).filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function formatDescriptorDelta(key: string, value: number): string {
  const prefix = value > 0 ? "+" : "";
  if (key === "rms_dbfs") return `${prefix}${value.toFixed(1)} dB`;
  if (key.endsWith("_hz")) return `${prefix}${Math.round(value)} Hz`;
  return `${prefix}${value.toFixed(2)}`;
}

function descriptorDeltaTone(key: string, value: number): DescriptorDeltaRow["tone"] {
  const threshold = key === "rms_dbfs" ? 0.25 : key.endsWith("_hz") ? 25 : 0.02;
  if (Math.abs(value) < threshold) return "neutral";
  return value > 0 ? "up" : "down";
}

function formatMaybeNumber(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value === undefined ? "" : String(value);
  return Number.isInteger(value) ? String(value) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function promptRunMeta(row: PromptSearchRunComparisonRow): string {
  const parts = [
    row.searchMode,
    row.scorer,
    row.model,
    row.durationSeconds !== undefined ? `${formatMaybeNumber(row.durationSeconds)}s` : "",
    row.promptCount ? `${row.promptCount} prompt${row.promptCount === 1 ? "" : "s"}` : "",
    row.listened ? `${row.listened}/${row.totalTakes} listened` : `${row.totalTakes} take${row.totalTakes === 1 ? "" : "s"}`,
  ].filter(Boolean);
  return parts.join(" · ");
}

function shortArtifactId(value: string): string {
  return value.length <= 8 ? value : value.slice(-8);
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function arrayOfObjects(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(objectValue(item))) : [];
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
