import { describe, expect, it } from "vitest";

import {
  bundleDomainSections,
  bundleReuseActions,
  promptCandidateGeneratedArtifacts,
  promptSearchCandidates,
  summarizeBundle,
} from "./bundleInspector";

describe("bundle inspector summaries", () => {
  it("promotes sweep metrics and plot counts into reader rows", () => {
    const summary = summarizeBundle(
      {
        kind: "sweep",
        file_count: 3,
        total_bytes: 128,
        audio_count: 2,
        latent_count: 0,
        npz_count: 0,
        sweep: { count: 2, alphas: [-4, 4], alpha_min: -4, alpha_max: 4 },
        metrics: { files: ["metrics.json"], values: { score: 0.8123 } },
        plots: { count: 1, files: ["plot.png"] },
      },
      {},
      [],
    );

    expect(summary.kind).toBe("sweep");
    expect(summary.rows).toContainEqual(["metric score", "0.812"]);
    expect(summary.rows).toContainEqual(["plots", 1]);
    expect(summary.plotFiles).toEqual(["plot.png"]);
  });

  it("offers recipe actions for reusable vector bundles", () => {
    expect(
      bundleReuseActions({
        artifact: {
          metadata: { operator: "experiment.sa3_vectors.extract" },
        } as never,
        bundle_summary: { kind: "vectors" },
      }),
    ).toEqual([
      { label: "Sweep vectors", fieldKey: "vectors_path", mode: "experiment.alpha_sweep" },
      { label: "Use direction", fieldKey: "direction_path", mode: "experiment.style_direction.generate" },
    ]);
  });

  it("builds native domain sections for reusable bundle payloads", () => {
    const sections = bundleDomainSections({
      kind: "vectors",
      vectors: { best_layer: 4, probe_accuracy: 0.81234, layers: [2, 4, 8] },
      npz_files: [
        {
          path: "direction.npz",
          keys: ["direction", "kind"],
          scalars: { kind: "LatentStyleDirection" },
          arrays: { direction: [64, 128] },
        },
      ],
      soft_prompt: { tensor_files: ["soft_prompt.pt"] },
    });

    expect(sections).toContainEqual({
      title: "Vectors",
      rows: [
        ["best layer", 4],
        ["accuracy", "0.812"],
        ["examples", undefined],
        ["layers", "2, 4, 8"],
      ],
    });
    expect(sections.find((section) => section.title === "direction.npz")?.rows).toContainEqual(["arrays", "direction 64x128"]);
    expect(sections.find((section) => section.title === "Soft Prompt")?.files).toEqual(["soft_prompt.pt"]);
  });

  it("summarizes geometry audit bundles", () => {
    const summary = summarizeBundle(
      {
        kind: "geometry",
        file_count: 1,
        total_bytes: 512,
        geometry: { latent_count: 4, n_components: 3, kept_variance_fraction: 0.934, dim: 64 },
      },
      {},
      [],
    );

    expect(summary.label).toBe("Geometry audit");
    expect(summary.rows).toContainEqual(["kept variance", "0.934"]);
    expect(bundleDomainSections({ geometry: { latent_count: 4, n_components: 3, kept_variance_fraction: 0.934 } })).toContainEqual({
      title: "Geometry",
      rows: [
        ["latents", 4],
        ["components", 3],
        ["kept variance", "0.934"],
        ["summary std", undefined],
      ],
    });
  });

  it("summarizes prompt search bundles and reuses the candidate prompt", () => {
    const summary = summarizeBundle(
      {
        kind: "prompt-search",
        file_count: 1,
        prompt_search: {
          path: "prompt_search.json",
          prompt: "warm granular loop",
          score: 0.82,
          search_mode: "beam",
          scorer: "sa3_flow_probe",
          model_backed: true,
          model: "medium",
          score_samples: 2,
          duration_seconds: 8,
          candidate_count: 12,
          families: [
            { rank: 1, prompt: "warm granular loop", score: 0.82, source: "selected" },
            { rank: 2, prompt: "warm granular loop shimmer", score: 0.78, source: "beam" },
          ],
        },
      },
      {},
      [],
    );

    expect(summary.label).toBe("Prompt search");
    expect(summary.rows).toContainEqual(["prompt", "warm granular loop"]);
    expect(summary.rows).toContainEqual(["scorer", "sa3_flow_probe"]);
    expect(summary.rows).toContainEqual(["model", "medium"]);
    expect(bundleDomainSections({
      prompt_search: {
        path: "prompt_search.json",
        prompt: "warm granular loop",
        score: 0.82,
        search_mode: "beam",
        scorer: "sa3_flow_probe",
        model_backed: true,
        model: "medium",
        score_samples: 2,
        duration_seconds: 8,
        families: [{ rank: 1, prompt: "warm granular loop", score: 0.82, source: "selected" }],
      },
    })).toContainEqual({
      title: "Prompt Search",
      rows: [
        ["prompt", "warm granular loop"],
        ["score", "0.82"],
        ["mode", "beam"],
        ["scorer", "sa3_flow_probe"],
        ["model-backed", "yes"],
        ["model", "medium"],
        ["samples", 2],
        ["duration", "8s"],
      ],
      files: ["prompt_search.json"],
      items: [{ label: "warm granular loop", meta: "#1 · selected · 0.82" }],
    });
    expect(
      bundleReuseActions({
        artifact: {
          metadata: { operator: "experiment.prompt_search" },
        } as never,
        bundle_summary: { kind: "prompt-search", prompt_search: { prompt: "warm granular loop" } },
      }),
    ).toEqual([{ label: "Use prompt in sweep", fieldKey: "prompt", mode: "experiment.alpha_sweep", value: "warm granular loop" }]);
    expect(
      promptSearchCandidates({
        prompt_search: {
          prompt: "warm granular loop",
          score: 0.82,
          families: [
            { rank: 1, prompt: "warm granular loop", score: 0.82, source: "selected" },
            { rank: 2, prompt: "warm granular loop shimmer", score: 0.78, source: "beam" },
          ],
        },
      }),
    ).toEqual([
      { rank: 1, prompt: "warm granular loop", score: 0.82, source: "selected" },
      { rank: 2, prompt: "warm granular loop shimmer", score: 0.78, source: "beam" },
    ]);
    expect(
      promptCandidateGeneratedArtifacts("art_prompt_bundle", [
        {
          artifact_id: "art_take",
          kind: "audio",
          source_artifact_ids: ["art_prompt_bundle"],
          prompt: "warm granular loop shimmer",
          metadata: { generation_origin: "prompt_search_candidate" },
          created_at: "2026-05-28T15:00:00.000Z",
        },
        {
          artifact_id: "art_other",
          kind: "audio",
          source_artifact_ids: ["art_prompt_bundle"],
          prompt: "other",
          metadata: { generation_origin: "manual" },
          created_at: "2026-05-28T15:01:00.000Z",
        },
      ] as never, "warm granular loop shimmer"),
    ).toEqual([
      expect.objectContaining({
        artifact_id: "art_take",
      }),
    ]);
  });
});
