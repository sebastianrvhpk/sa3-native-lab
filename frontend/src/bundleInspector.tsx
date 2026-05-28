import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { createApi } from "./api";
import { artifactMeta, artifactName, formatBytes } from "./artifactUtils";
import type { ArtifactInspection, ArtifactRecord, BundleAudioEntry, BundleFileEntry } from "./types";

export function BundleField({
  artifact,
  artifacts,
  apiBase,
  onCompare,
  onSelectArtifact,
  onUseAsDonor,
  onUseInRecipe,
  getArtifactPath,
}: {
  artifact: ArtifactRecord;
  artifacts: ArtifactRecord[];
  apiBase: string;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
}) {
  const api = useMemo(() => createApi(apiBase), [apiBase]);
  const inspection = useQuery({
    queryKey: ["artifact-inspection", apiBase, artifact.artifact_id],
    queryFn: () => api.inspectArtifact(artifact.artifact_id),
    enabled: artifact.kind === "bundle",
    staleTime: 30000,
  });
  const fileName = artifact.file?.filename ?? artifactName(artifact);
  const cells = artifact.recipe_id ? 36 : 18;
  const bundleFiles = inspection.data?.bundle_files ?? [];
  const totalBytes = bundleFiles.reduce((total, item) => total + item.byte_size, 0);
  return (
    <div className="bundle-field" aria-label={`Experiment bundle ${fileName}`}>
      {Array.from({ length: cells }, (_, index) => (
        <span key={index} />
      ))}
      <div className="bundle-readout">
        <strong>{fileName}</strong>
        <span>
          {inspection.isLoading ? "Inspecting..." : bundleFiles.length ? `${bundleFiles.length} files · ${formatBytes(totalBytes)}` : artifact.file ? formatBytes(artifact.file.byte_size) : "bundle"}
        </span>
        {inspection.data ? <BundleReaderPanel inspection={inspection.data} apiBase={apiBase} /> : null}
        {inspection.data ? (
          <BundleReuseActions
            inspection={inspection.data}
            artifact={artifact}
            getArtifactPath={getArtifactPath}
            onUseInRecipe={onUseInRecipe}
          />
        ) : null}
        {bundleFiles.length ? (
          <div className="bundle-file-list">
            {bundleFiles.slice(0, 4).map((file) => (
              <i key={file.path}>
                {file.path}
                <small>{formatBytes(file.byte_size)}</small>
              </i>
            ))}
          </div>
        ) : null}
        {inspection.data ? (
          <BundlePreview
            preview={inspection.data.bundle_preview}
            artifacts={artifacts}
            sourceCount={inspection.data.sources.length}
            childCount={inspection.data.children.length}
            onCompare={onCompare}
            onSelectArtifact={onSelectArtifact}
            onUseAsDonor={onUseAsDonor}
          />
        ) : null}
      </div>
    </div>
  );
}

export interface BundleReuseAction {
  label: string;
  fieldKey: string;
  mode: string;
}

export function bundleReuseActions(inspection: Pick<ArtifactInspection, "artifact" | "bundle_summary">): BundleReuseAction[] {
  const kind = typeof inspection.bundle_summary.kind === "string" ? inspection.bundle_summary.kind : "";
  const operator = typeof inspection.artifact.metadata.operator === "string" ? inspection.artifact.metadata.operator : "";
  const actions: BundleReuseAction[] = [];
  if (kind === "profile" || operator.includes("style_profile")) {
    actions.push({ label: "Use as profile", fieldKey: "profile_path", mode: "experiment.style_profile.generate" });
    actions.push({ label: "Use memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" });
  }
  if (kind === "vectors" || operator.includes("vectors") || operator.includes("direction")) {
    actions.push({ label: "Sweep vectors", fieldKey: "vectors_path", mode: "experiment.alpha_sweep" });
    actions.push({ label: "Use direction", fieldKey: "direction_path", mode: "experiment.style_direction.generate" });
  }
  if (kind === "soft-prompt" || operator.includes("soft_prompt")) {
    actions.push({ label: "Use soft prompt", fieldKey: "soft_prompt_path", mode: "experiment.soft_prompt.generate" });
  }
  if (kind === "memory" || operator.includes("memory")) {
    actions.push({ label: "Use as target memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" });
    actions.push({ label: "Use as reference", fieldKey: "reference_memory_path", mode: "experiment.style_profile.build" });
  }
  if (kind === "training" || operator.includes("lora")) {
    actions.push({ label: "Use checkpoint", fieldKey: "lora_checkpoint", mode: "training.lora" });
  }
  return dedupeReuseActions(actions);
}

function BundleReuseActions({
  inspection,
  artifact,
  getArtifactPath,
  onUseInRecipe,
}: {
  inspection: ArtifactInspection;
  artifact: ArtifactRecord;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
  onUseInRecipe: (fieldKey: string, path: string, mode: string) => void;
}) {
  const actions = bundleReuseActions(inspection);
  if (!actions.length) return null;
  return (
    <div className="bundle-reuse-actions" aria-label="Bundle recipe actions">
      {actions.map((action) => (
        <button
          key={`${action.mode}:${action.fieldKey}`}
          type="button"
          onClick={() => onUseInRecipe(action.fieldKey, getArtifactPath(artifact, action.fieldKey), action.mode)}
        >
          {action.label}
        </button>
      ))}
    </div>
  );
}

function BundleReaderPanel({ inspection, apiBase }: { inspection: ArtifactInspection; apiBase: string }) {
  const descriptor = summarizeBundle(inspection.bundle_summary, inspection.bundle_preview, inspection.bundle_files);
  const rows = descriptor.rows.filter(([, value]) => value !== undefined && value !== null && value !== "");
  const plotFiles = descriptor.plotFiles ?? [];
  const domainSections = bundleDomainSections(inspection.bundle_summary);
  const api = createApi(apiBase);
  return (
    <div className={`bundle-kind-panel ${descriptor.kind}`}>
      <div>
        <strong>{descriptor.label}</strong>
        <span>{descriptor.description}</span>
      </div>
      {rows.length ? (
        <div className="bundle-kind-rows">
          {rows.slice(0, 6).map(([label, value]) => {
            const rowLabel = String(label);
            return (
              <i key={rowLabel}>
                {rowLabel}
                <small>{String(value)}</small>
              </i>
            );
          })}
        </div>
      ) : null}
      {plotFiles.length ? <PlotPreviewShell files={plotFiles} artifactId={inspection.artifact.artifact_id} getFileUrl={api.bundleFileUrl} /> : null}
      {domainSections.length ? <BundleDomainCards sections={domainSections} /> : null}
      <BundleAudioChildren artifactId={inspection.artifact.artifact_id} entries={inspection.bundle_audio_files} getFileUrl={api.bundleFileUrl} />
    </div>
  );
}

export interface BundleDomainSection {
  title: string;
  rows: [string, unknown][];
  files?: string[];
}

export function bundleDomainSections(summary: Record<string, unknown> | undefined): BundleDomainSection[] {
  if (!summary || !Object.keys(summary).length) return [];
  const sections: BundleDomainSection[] = [];
  const sweep = objectValue(summary.sweep);
  if (sweep) {
    sections.push({
      title: "Sweep",
      rows: [
        ["variants", sweep.count],
        ["alphas", Array.isArray(sweep.alphas) ? sweep.alphas.join(", ") : undefined],
        ["range", sweep.alpha_min !== undefined && sweep.alpha_max !== undefined ? `${sweep.alpha_min} to ${sweep.alpha_max}` : undefined],
      ],
      files: sweepOutputFiles(sweep),
    });
  }
  const memory = objectValue(summary.memory);
  if (memory) {
    sections.push({
      title: "Memory",
      rows: [
        ["metric", memory.metric],
        ["hits", memory.result_count],
        ["candidates", memory.candidate_count],
        ["source", memory.source_artifact_id],
      ],
    });
  }
  const vectors = objectValue(summary.vectors);
  if (vectors) {
    sections.push({
      title: "Vectors",
      rows: [
        ["best layer", vectors.best_layer],
        ["accuracy", formatMaybeNumber(vectors.probe_accuracy)],
        ["examples", vectors.example_count],
        ["layers", Array.isArray(vectors.layers) ? vectors.layers.join(", ") : vectors.layers],
      ],
    });
  }
  const geometry = objectValue(summary.geometry);
  if (geometry) {
    sections.push({
      title: "Geometry",
      rows: [
        ["latents", geometry.latent_count],
        ["components", geometry.n_components],
        ["kept variance", formatMaybeNumber(geometry.kept_variance_fraction)],
        ["summary std", formatMaybeNumber(geometry.summary_std_mean)],
      ],
    });
  }
  const profile = objectValue(summary.profile);
  if (profile) {
    for (const item of arrayOfObjects(profile.profiles).slice(0, 3)) {
      sections.push({
        title: String(item.name || item.path || "Profile"),
        rows: [
          ["path", item.path],
          ["dim", item.dim],
          ["items", item.item_count],
          ["arrays", formatArrayShapes(objectValue(item.arrays))],
        ],
      });
    }
  }
  for (const item of arrayOfObjects(summary.npz_files).slice(0, 4)) {
    sections.push({
      title: String(item.path || "NPZ"),
      rows: [
        ["kind", objectValue(item.scalars)?.kind],
        ["keys", Array.isArray(item.keys) ? item.keys.slice(0, 6).join(", ") : undefined],
        ["arrays", formatArrayShapes(objectValue(item.arrays))],
      ],
    });
  }
  const softPrompt = objectValue(summary.soft_prompt);
  if (softPrompt) {
    sections.push({
      title: "Soft Prompt",
      rows: [["tensors", Array.isArray(softPrompt.tensor_files) ? softPrompt.tensor_files.length : undefined]],
      files: arrayOfStrings(softPrompt.tensor_files),
    });
  }
  const training = objectValue(summary.training);
  if (training) {
    sections.push({
      title: "Training",
      rows: [["checkpoints", Array.isArray(training.checkpoint_files) ? training.checkpoint_files.length : undefined]],
      files: arrayOfStrings(training.checkpoint_files),
    });
  }
  return sections.filter((section) => section.rows.some(([, value]) => value !== undefined && value !== null && value !== "") || section.files?.length);
}

export function summarizeBundle(summary: Record<string, unknown> | undefined, preview: Record<string, unknown>, files: BundleFileEntry[]) {
  if (!summary || !Object.keys(summary).length) {
    return classifyBundle(preview, files);
  }
  const kind = typeof summary.kind === "string" ? summary.kind : "generic";
  const label = bundleKindLabel(kind);
  const description = bundleKindDescription(kind);
  const rows: [string, unknown][] = [
    ["files", summary.file_count],
    ["bytes", typeof summary.total_bytes === "number" ? formatBytes(summary.total_bytes) : undefined],
    ["audio", summary.audio_count],
    ["latents", summary.latent_count],
    ["npz", summary.npz_count],
  ];
  const sweep = summary.sweep && typeof summary.sweep === "object" ? (summary.sweep as Record<string, unknown>) : null;
  if (sweep) {
    rows.unshift(["alphas", Array.isArray(sweep.alphas) ? sweep.alphas.join(", ") : sweep.count]);
    rows.push(["range", sweep.alpha_min !== undefined && sweep.alpha_max !== undefined ? `${sweep.alpha_min} to ${sweep.alpha_max}` : undefined]);
  }
  const metrics = summary.metrics && typeof summary.metrics === "object" ? (summary.metrics as Record<string, unknown>) : null;
  const metricValues = metrics?.values && typeof metrics.values === "object" ? (metrics.values as Record<string, unknown>) : null;
  if (metricValues) {
    rows.push(
      ...Object.entries(metricValues)
        .slice(0, 3)
        .map(([key, value]) => [`metric ${prettyBundleKey(key)}`, formatMaybeNumber(value)] as [string, unknown]),
    );
  }
  const plots = summary.plots && typeof summary.plots === "object" ? (summary.plots as Record<string, unknown>) : null;
  const plotFiles = Array.isArray(plots?.files) ? plots.files.filter((file): file is string => typeof file === "string") : [];
  if (plots) {
    rows.push(["plots", plots.count]);
  }
  const memory = summary.memory && typeof summary.memory === "object" ? (summary.memory as Record<string, unknown>) : null;
  if (memory) {
    rows.unshift(["hits", memory.result_count], ["metric", memory.metric]);
    rows.push(["candidates", memory.candidate_count]);
  }
  const vectors = summary.vectors && typeof summary.vectors === "object" ? (summary.vectors as Record<string, unknown>) : null;
  if (vectors) {
    rows.unshift(["best layer", vectors.best_layer], ["accuracy", formatMaybeNumber(vectors.probe_accuracy)]);
  }
  const geometry = summary.geometry && typeof summary.geometry === "object" ? (summary.geometry as Record<string, unknown>) : null;
  if (geometry) {
    rows.unshift(["latents", geometry.latent_count], ["kept variance", formatMaybeNumber(geometry.kept_variance_fraction)]);
    rows.push(["components", geometry.n_components], ["dim", geometry.dim]);
  }
  const profile = summary.profile && typeof summary.profile === "object" ? (summary.profile as Record<string, unknown>) : null;
  const profiles = Array.isArray(profile?.profiles) ? profile.profiles : [];
  const firstProfile = profiles[0] && typeof profiles[0] === "object" ? (profiles[0] as Record<string, unknown>) : null;
  if (firstProfile) {
    rows.unshift(["profile", firstProfile.name], ["dim", firstProfile.dim], ["items", firstProfile.item_count]);
  }
  const softPrompt = summary.soft_prompt && typeof summary.soft_prompt === "object" ? (summary.soft_prompt as Record<string, unknown>) : null;
  if (softPrompt) {
    rows.unshift(["tensors", Array.isArray(softPrompt.tensor_files) ? softPrompt.tensor_files.length : undefined]);
  }
  const training = summary.training && typeof summary.training === "object" ? (summary.training as Record<string, unknown>) : null;
  if (training) {
    rows.unshift(["checkpoints", Array.isArray(training.checkpoint_files) ? training.checkpoint_files.length : undefined]);
  }
  return { kind, label, description, rows, plotFiles };
}

function BundleDomainCards({ sections }: { sections: BundleDomainSection[] }) {
  return (
    <div className="bundle-domain-grid" aria-label="Bundle native inspectors">
      {sections.slice(0, 6).map((section) => (
        <article key={`${section.title}:${section.files?.join("|") ?? ""}`}>
          <strong>{section.title}</strong>
          <div>
            {section.rows
              .filter(([, value]) => value !== undefined && value !== null && value !== "")
              .slice(0, 4)
              .map(([label, value]) => (
                <i key={label}>
                  {label}
                  <small>{formatDomainValue(value)}</small>
                </i>
              ))}
          </div>
          {section.files?.length ? <span>{section.files.slice(0, 3).join(" · ")}</span> : null}
        </article>
      ))}
    </div>
  );
}

function BundleAudioChildren({
  artifactId,
  entries,
  getFileUrl,
}: {
  artifactId: string;
  entries: BundleAudioEntry[];
  getFileUrl: (artifactId: string, path: string) => string;
}) {
  if (!entries.length) return null;
  return (
    <div className="bundle-audio-children" aria-label="Bundle audio children">
      <strong>Audio in bundle</strong>
      {entries.slice(0, 5).map((entry) => {
        const url = getFileUrl(artifactId, entry.path);
        return (
          <article key={entry.path}>
            <div>
              <span>{entry.path}</span>
              <small>{bundleAudioMeta(entry)}</small>
            </div>
            <audio controls preload="none" src={url} />
            <a href={url} target="_blank" rel="noreferrer" download>
              Open
            </a>
          </article>
        );
      })}
    </div>
  );
}

function bundleKindLabel(kind: string) {
  if (kind === "memory") return "Memory query";
  if (kind === "sweep") return "Sweep results";
  if (kind === "profile") return "Style profile";
  if (kind === "vectors") return "Vector bundle";
  if (kind === "geometry") return "Geometry audit";
  if (kind === "soft-prompt") return "Soft prompt";
  if (kind === "training") return "Training output";
  return "Bundle archive";
}

function bundleKindDescription(kind: string) {
  if (kind === "memory") return "Ranked latent neighbors parsed from local bundle metadata";
  if (kind === "sweep") return "Parameter branch outputs with alpha and artifact metadata";
  if (kind === "profile") return "Reusable latent/audio statistics for profile-guided generation";
  if (kind === "vectors") return "Residual or style direction vectors with parsed layers and shapes";
  if (kind === "geometry") return "Local latent geometry summary for saved SAME artifacts";
  if (kind === "soft-prompt") return "Optimized conditioning tensors for prompt continuation";
  if (kind === "training") return "Adapter checkpoints, logs, and long-running training outputs";
  return "Script output files preserved as a replayable artifact";
}

function BundlePreview({
  preview,
  artifacts,
  sourceCount,
  childCount,
  onCompare,
  onSelectArtifact,
  onUseAsDonor,
}: {
  preview: Record<string, unknown>;
  artifacts: ArtifactRecord[];
  sourceCount: number;
  childCount: number;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
}) {
  const rows = [
    ["operator", preview.operator],
    ["results", preview.result_count],
    ["metric", preview.metric],
    ["candidates", preview.candidate_count],
    ["top k", preview.top_k],
    ["sources", sourceCount],
    ["children", childCount],
  ].filter(([, value]) => value !== undefined && value !== null && value !== "");
  const results = Array.isArray(preview.results) ? preview.results.slice(0, 3) : [];
  const artifactMap = new Map(artifacts.map((artifact) => [artifact.artifact_id, artifact]));
  if (!rows.length && !results.length) return null;
  return (
    <div className="bundle-preview">
      {rows.length ? (
        <div className="metric-chips">
          {rows.slice(0, 6).map(([key, value]) => (
            <i key={String(key)}>
              {String(key)}: {String(value)}
            </i>
          ))}
        </div>
      ) : null}
      {results.length ? (
        <div className="bundle-result-list">
          {results.map((result, index) => {
            const item = result && typeof result === "object" ? (result as Record<string, unknown>) : {};
            const artifactId = typeof item.artifact_id === "string" ? item.artifact_id : "";
            const localArtifact = artifactId ? artifactMap.get(artifactId) ?? null : null;
            return (
              <article key={`${String(item.artifact_id ?? index)}-${index}`}>
                <div>
                  <strong>{localArtifact ? artifactName(localArtifact) : String(item.artifact_id ?? `result ${index + 1}`)}</strong>
                  <small>{formatBundleScore(item) || (localArtifact ? artifactMeta(localArtifact) : "memory hit")}</small>
                </div>
                <div className="memory-hit-actions">
                  <button type="button" disabled={!localArtifact} onClick={() => onSelectArtifact(localArtifact?.artifact_id ?? null)}>
                    Select
                  </button>
                  <button type="button" disabled={localArtifact?.kind !== "audio"} onClick={() => localArtifact && onCompare("a", localArtifact.artifact_id)}>
                    A
                  </button>
                  <button type="button" disabled={localArtifact?.kind !== "audio"} onClick={() => localArtifact && onCompare("b", localArtifact.artifact_id)}>
                    B
                  </button>
                  <button type="button" disabled={localArtifact?.kind !== "latent"} onClick={() => localArtifact && onUseAsDonor(localArtifact.artifact_id)}>
                    Donor
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function PlotPreviewShell({
  files,
  artifactId,
  getFileUrl,
}: {
  files: readonly string[];
  artifactId: string;
  getFileUrl: (artifactId: string, path: string) => string;
}) {
  return (
    <div className="bundle-plot-shell" aria-label="Bundle plot files">
      <strong>Plot preview</strong>
      <div>
        {files.slice(0, 4).map((file) =>
          isInlineImagePath(file) ? (
            <a key={file} href={getFileUrl(artifactId, file)} target="_blank" rel="noreferrer" title={file}>
              <img src={getFileUrl(artifactId, file)} alt={file} loading="lazy" />
              <span>{file}</span>
            </a>
          ) : (
            <a key={file} className="plot-file-link" href={getFileUrl(artifactId, file)} target="_blank" rel="noreferrer">
              {file}
            </a>
          ),
        )}
      </div>
    </div>
  );
}

function classifyBundle(preview: Record<string, unknown>, files: BundleFileEntry[]) {
  const operator = typeof preview.operator === "string" ? preview.operator : "";
  const fileNames = files.map((file) => file.path.toLowerCase());
  const hasFile = (pattern: string) => fileNames.some((name) => name.includes(pattern));
  if (operator === "memory.query" || Array.isArray(preview.results)) {
    return {
      kind: "memory",
      label: "Memory query",
      description: "Ranked latent neighbors from local SAME artifacts",
      rows: [
        ["metric", preview.metric],
        ["top k", preview.top_k],
        ["candidates", preview.candidate_count],
        ["hits", preview.result_count],
      ],
      plotFiles: [],
    };
  }
  if (operator.includes("alpha_sweep") || hasFile("metrics.json")) {
    return {
      kind: "sweep",
      label: "Sweep results",
      description: "Multi-run experiment bundle for comparing parameter branches",
      rows: [
        ["operator", operator],
        ["files", files.length],
        ["metrics", hasFile("metrics.json") ? "present" : "pending"],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  if (operator.includes("style_profile") || operator.includes("positive_style_profile") || hasFile("profile.npz")) {
    return {
      kind: "profile",
      label: "Style profile",
      description: "Reusable latent/audio statistics for profile-guided generation",
      rows: [
        ["profile", hasFile("profile.npz") ? "profile.npz" : "not found"],
        ["memory", hasFile("memory") ? "included" : "none"],
        ["files", files.length],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  if (operator.includes("vectors") || operator.includes("direction") || hasFile("direction.npz") || hasFile("vectors")) {
    return {
      kind: "vectors",
      label: "Vector bundle",
      description: "Residual or style direction vectors for steering experiments",
      rows: [
        ["operator", operator],
        ["npz", fileNames.filter((name) => name.endsWith(".npz")).length],
        ["files", files.length],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  if (operator.includes("geometry") || hasFile("geometry_report.json")) {
    return {
      kind: "geometry",
      label: "Geometry audit",
      description: "Local latent geometry summary for saved SAME artifacts",
      rows: [
        ["operator", operator],
        ["files", files.length],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  if (operator.includes("soft_prompt") || hasFile("soft_prompt.pt")) {
    return {
      kind: "soft-prompt",
      label: "Soft prompt",
      description: "Optimized conditioning tensor for prompt continuation experiments",
      rows: [
        ["tensor", hasFile("soft_prompt.pt") ? "soft_prompt.pt" : "not found"],
        ["files", files.length],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  if (operator.includes("lora") || hasFile("checkpoint") || hasFile("adapter")) {
    return {
      kind: "training",
      label: "Training output",
      description: "Long-running adapter artifacts, logs, and checkpoints",
      rows: [
        ["operator", operator],
        ["files", files.length],
      ],
      plotFiles: files.map((file) => file.path).filter(isPlotPath),
    };
  }
  return {
    kind: "generic",
    label: "Bundle archive",
    description: "Script output files preserved as a replayable artifact",
    rows: [
      ["operator", operator],
      ["files", files.length],
    ],
    plotFiles: files.map((file) => file.path).filter(isPlotPath),
  };
}

function formatBundleScore(item: Record<string, unknown>) {
  const score = typeof item.score === "number" ? item.score : null;
  const distance = typeof item.distance === "number" ? item.distance : null;
  if (score !== null) return `score ${score.toFixed(3)}`;
  if (distance !== null) return `distance ${distance.toFixed(3)}`;
  return "";
}

function formatMaybeNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3).replace(/\.?0+$/, "") : value;
}

function prettyBundleKey(key: string) {
  return key.replaceAll("_", " ");
}

function isPlotPath(path: string) {
  const value = path.toLowerCase();
  return /\.(png|jpe?g|webp|svg|pdf)$/.test(value) || value.includes("plot") || value.includes("chart");
}

function isInlineImagePath(path: string) {
  return /\.(png|jpe?g|webp|svg)$/i.test(path);
}

function objectValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
}

function arrayOfObjects(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(objectValue(item))) : [];
}

function arrayOfStrings(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function sweepOutputFiles(sweep: Record<string, unknown>): string[] {
  return arrayOfObjects(sweep.outputs)
    .flatMap((item) => [item.audio_path, item.latent_path])
    .filter((item): item is string => typeof item === "string" && item.length > 0);
}

function formatArrayShapes(arrays: Record<string, unknown> | null) {
  if (!arrays) return undefined;
  return Object.entries(arrays)
    .slice(0, 4)
    .map(([key, value]) => `${key} ${Array.isArray(value) ? value.join("x") : String(value)}`)
    .join(", ");
}

function formatDomainValue(value: unknown) {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "number") return String(formatMaybeNumber(value));
  return String(value);
}

function bundleAudioMeta(entry: BundleAudioEntry) {
  const parts = [
    entry.duration_seconds !== undefined && entry.duration_seconds !== null ? `${formatMaybeNumber(entry.duration_seconds)}s` : null,
    entry.sample_rate ? `${entry.sample_rate} Hz` : null,
    entry.channels ? `${entry.channels} ch` : null,
    formatBytes(entry.byte_size),
  ].filter(Boolean);
  return parts.join(" · ");
}

function dedupeReuseActions(actions: BundleReuseAction[]) {
  const seen = new Set<string>();
  return actions.filter((action) => {
    const key = `${action.mode}:${action.fieldKey}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}
