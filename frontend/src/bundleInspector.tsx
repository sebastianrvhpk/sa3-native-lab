import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";

import { createApi } from "./api";
import { artifactMeta, artifactName, formatBytes } from "./artifactUtils";
import type { ArtifactInspection, ArtifactRecord, BundleFileEntry } from "./types";

export function BundleField({
  artifact,
  artifacts,
  apiBase,
  onCompare,
  onSelectArtifact,
  onUseAsDonor,
}: {
  artifact: ArtifactRecord;
  artifacts: ArtifactRecord[];
  apiBase: string;
  onCompare: (slot: "a" | "b", artifactId: string | null) => void;
  onSelectArtifact: (artifactId: string | null) => void;
  onUseAsDonor: (artifactId: string) => void;
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
        {inspection.data ? <BundleReaderPanel inspection={inspection.data} /> : null}
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

function BundleReaderPanel({ inspection }: { inspection: ArtifactInspection }) {
  const descriptor = classifyBundle(inspection.bundle_preview, inspection.bundle_files);
  const rows = descriptor.rows.filter(([, value]) => value !== undefined && value !== null && value !== "");
  return (
    <div className={`bundle-kind-panel ${descriptor.kind}`}>
      <div>
        <strong>{descriptor.label}</strong>
        <span>{descriptor.description}</span>
      </div>
      {rows.length ? (
        <div className="bundle-kind-rows">
          {rows.slice(0, 4).map(([label, value]) => {
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
    </div>
  );
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
  };
}

function formatBundleScore(item: Record<string, unknown>) {
  const score = typeof item.score === "number" ? item.score : null;
  const distance = typeof item.distance === "number" ? item.distance : null;
  if (score !== null) return `score ${score.toFixed(3)}`;
  if (distance !== null) return `distance ${distance.toFixed(3)}`;
  return "";
}
