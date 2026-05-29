import { describe, expect, it } from "vitest";

import {
  audioDescriptorDeltaRows,
  bundleDomainSections,
  bundleReuseActions,
  bundleWorkflowHints,
  promptCandidateGeneratedArtifacts,
  promptDecisionCorrelationRows,
  promptDecisionMemoryRows,
  promptDecisionSummary,
  promptSearchTargetArtifact,
  promptSearchCandidates,
  promptSearchRunComparisonRows,
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

  it("turns sweep, memory, vector, soft-prompt, and training summaries into domain items", () => {
    const sections = bundleDomainSections({
      sweep: {
        count: 2,
        alphas: [-2, 2],
        alpha_min: -2,
        alpha_max: 2,
        outputs: [{ alpha: -2, audio_path: "runs/a_minus.wav", latent_path: "runs/a_minus.npy" }],
      },
      memory: {
        metric: "cosine",
        top_k: 4,
        result_count: 1,
        candidate_count: 9,
        results: [{ artifact_id: "art_neighbor", score: 0.91, kind: "latent" }],
      },
      vectors: {
        npz_files: [
          {
            path: "direction.npz",
            scalars: { kind: "LatentStyleDirection" },
            arrays: { direction: [64, 128] },
          },
        ],
      },
      soft_prompt: { tensor_files: ["optim/soft_prompt.pt"] },
      training: { checkpoint_files: ["runs/lora_adapter.safetensors", "runs/checkpoint-20.pt"] },
    });

    expect(sections.find((section) => section.title === "Sweep")?.items).toEqual([
      { label: "alpha -2", meta: "a_minus.wav · a_minus.npy" },
    ]);
    expect(sections.find((section) => section.title === "Memory")?.items).toEqual([
      { label: "art_neighbor", meta: "score 0.91 · latent" },
    ]);
    expect(sections.find((section) => section.title === "Vectors")?.items).toEqual([
      { label: "direction.npz", meta: "LatentStyleDirection · direction 64x128" },
    ]);
    expect(sections.find((section) => section.title === "Soft Prompt")?.items).toEqual([
      { label: "soft_prompt.pt", meta: "conditioning tensor" },
    ]);
    expect(sections.find((section) => section.title === "Training")?.items).toEqual([
      { label: "lora_adapter.safetensors", meta: "LoRA adapter" },
      { label: "checkpoint-20.pt", meta: "checkpoint" },
    ]);
  });

  it("turns dataset manifests and profile npz files into native inspector cards", () => {
    const summary = summarizeBundle(
      {
        kind: "dataset",
        file_count: 4,
        dataset: { item_count: 2, latent_count: 2, metadata_count: 2, prompt_count: 1, model: "same-l" },
      },
      {},
      [],
    );
    const sections = bundleDomainSections({
      kind: "dataset",
      dataset: {
        item_count: 2,
        latent_count: 2,
        metadata_count: 2,
        prompt_count: 1,
        model: "same-l",
        sample_rate: 44100,
        sample_size: 88200,
        chunk_duration: 4,
        hop_duration: 2,
        source: "clips",
        latent_files: ["0000000000.npy", "0000000001.npy"],
        items: [
          {
            item_id: "clip-1",
            prompt: "soft pulse",
            source_path: "clips/soft.wav",
            sample_rate: 44100,
            chunk_duration_seconds: 4,
          },
        ],
      },
      profile: {
        profiles: [
          {
            path: "profile.npz",
            name: "target",
            dim: 64,
            item_count: 5,
            arrays: { mean: [64], std: [64] },
          },
        ],
      },
    });

    expect(summary.label).toBe("Encoded dataset");
    expect(summary.rows).toContainEqual(["items", 2]);
    expect(summary.rows).toContainEqual(["latents", 2]);
    expect(summary.rows).toContainEqual(["model", "same-l"]);
    expect(sections.find((section) => section.title === "Dataset")).toEqual({
      title: "Dataset",
      rows: [
        ["items", 2],
        ["latents", 2],
        ["metadata", 2],
        ["prompts", 1],
        ["model", "same-l"],
        ["sample rate", "44100 Hz"],
        ["sample size", 88200],
        ["chunk", "4s"],
        ["hop", "2s"],
        ["source", "clips"],
      ],
      files: ["0000000000.npy", "0000000001.npy"],
      items: [{ label: "clip-1", meta: "soft pulse · soft.wav · 4s · 44100 Hz" }],
    });
    expect(sections.find((section) => section.title === "Profiles")).toEqual({
      title: "Profiles",
      rows: [
        ["profiles", 1],
        ["items", 5],
        ["dims", "64"],
      ],
      files: ["profile.npz"],
      items: [{ label: "target", meta: "dim 64 · 5 items · mean 64, std 64" }],
    });
    expect(
      bundleReuseActions({
        artifact: { metadata: { operator: "dataset.pre_encode" } } as never,
        bundle_summary: { kind: "dataset" },
      }),
    ).toEqual([{ label: "Use encoded dataset", fieldKey: "encoded_dir", mode: "training.lora" }]);
  });

  it("summarizes bundle workflow signals from real bundle metadata", () => {
    const hints = bundleWorkflowHints({
      artifact: {
        metadata: { operator: "experiment.prompt_search" },
      } as never,
      bundle_summary: {
        kind: "prompt-search",
        metrics: { values: { score: 0.82, loss: 0.18 } },
        plots: { files: ["decision_plot.png"] },
        prompt_search: {
          prompt: "warm granular loop",
          scorer: "sa3_flow_probe",
          candidate_count: 12,
          families: [{ rank: 1, prompt: "warm granular loop", score: 0.82, source: "selected" }],
        },
      },
      bundle_preview: {},
      bundle_files: [],
      bundle_audio_files: [{ path: "take.wav", byte_size: 1024 }] as never,
      sources: [{ artifact_id: "art_target" }] as never,
      children: [{ artifact_id: "art_take", kind: "audio" }] as never,
    });

    expect(hints).toEqual(expect.arrayContaining([
      { label: "recipe actions", value: "1", tone: "reuse" },
      { label: "playable audio", value: "2", tone: "listen" },
      { label: "lineage", value: "1/1", tone: "lineage" },
      { label: "plots", value: "1", tone: "evidence" },
      { label: "metrics", value: "2", tone: "evidence" },
      { label: "candidates", value: "12", tone: "probe" },
      { label: "scorer", value: "sa3_flow_probe", tone: "probe" },
    ]));
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
        ["path", undefined],
        ["latents", 4],
        ["candidates", undefined],
        ["components", 3],
        ["kept variance", "0.934"],
        ["summary std", undefined],
        ["frames", undefined],
        ["dim", undefined],
      ],
      files: [],
      items: [],
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
    expect(
      promptSearchTargetArtifact(
        {
          artifact_id: "art_prompt_bundle",
          kind: "bundle",
          source_artifact_ids: ["art_target", "art_other"],
          metadata: {},
          tags: [],
          path: "/tmp/bundle.zip",
          created_at: "2026-05-28T15:00:00.000Z",
        },
        [
          {
            artifact_id: "art_target",
            kind: "audio",
            source_artifact_ids: [],
            metadata: {},
            tags: [],
            path: "/tmp/target.wav",
            created_at: "2026-05-28T14:00:00.000Z",
          },
          {
            artifact_id: "art_other",
            kind: "latent",
            source_artifact_ids: [],
            metadata: {},
            tags: [],
            path: "/tmp/latent.npy",
            created_at: "2026-05-28T14:01:00.000Z",
          },
        ] as never,
      ),
    ).toEqual(expect.objectContaining({ artifact_id: "art_target" }));
    expect(
      audioDescriptorDeltaRows({
        target_artifact_id: "art_target",
        take_artifact_id: "art_take",
        target: {},
        take: {},
        delta: {
          rms_dbfs: 1.25,
          spectral_centroid_hz: 220.4,
          spectral_flux: 0.005,
          spectral_flatness: -0.08,
          stereo_width: 0.03,
        },
      }),
    ).toEqual([
      expect.objectContaining({ key: "rms_dbfs", label: "level", value: "+1.3 dB", tone: "up" }),
      expect.objectContaining({ key: "spectral_centroid_hz", label: "bright", value: "+220 Hz", tone: "up" }),
      expect.objectContaining({ key: "spectral_flux", label: "motion", value: "+0.01", tone: "neutral" }),
      expect.objectContaining({ key: "spectral_flatness", label: "noise", value: "-0.08", tone: "down" }),
      expect.objectContaining({ key: "stereo_width", label: "width", value: "+0.03", tone: "up" }),
    ]);
    const takeRows = promptDecisionCorrelationRows([
      {
        artifact_id: "art_take",
        kind: "audio",
        prompt: "warm granular loop shimmer",
        source_artifact_ids: ["art_prompt_bundle"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_candidate_rank: 1,
          seed: 7,
          listening_decision: "keeper",
          listening_decision_note: "wide useful texture",
        },
        tags: ["keeper"],
        path: "/tmp/take.wav",
        created_at: "2026-05-28T15:00:00.000Z",
      },
    ] as never, new Map([
      ["art_take", {
        target_artifact_id: "art_target",
        take_artifact_id: "art_take",
        target: {},
        take: {},
        delta: { rms_dbfs: 1.25, spectral_centroid_hz: 220.4, spectral_flux: 0.01 },
      }],
    ]));
    expect(takeRows).toEqual([
      expect.objectContaining({
        artifactId: "art_take",
        label: "#1 · seed 7",
        decision: "keeper",
        note: "wide useful texture",
      }),
    ]);
    expect(promptDecisionSummary(takeRows)).toEqual(expect.arrayContaining([
      { label: "keepers", value: "1", tone: "up" },
      { label: "keeper bright", value: "+220 Hz", tone: "up" },
      { label: "keeper level", value: "+1.3 dB", tone: "up" },
    ]));
  });

  it("groups prompt-candidate decisions across search runs", () => {
    const rows = promptDecisionMemoryRows([
      {
        artifact_id: "art_take_a",
        kind: "audio",
        prompt: "warm granular loop",
        source_artifact_ids: ["art_bundle_a"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_search_bundle_id: "art_bundle_a",
          listening_decision: "keeper",
          listening_decision_note: "best texture",
        },
        tags: [],
        path: "/tmp/a.wav",
        created_at: "2026-05-28T15:00:00.000Z",
      },
      {
        artifact_id: "art_take_b",
        kind: "audio",
        prompt: "warm granular loop",
        source_artifact_ids: ["art_bundle_b"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          prompt_search_bundle_id: "art_bundle_b",
          listening_decision: "maybe",
          listening_decision_note: "good but soft",
        },
        tags: [],
        path: "/tmp/b.wav",
        created_at: "2026-05-28T15:05:00.000Z",
      },
      {
        artifact_id: "art_take_c",
        kind: "audio",
        prompt: "cold click",
        source_artifact_ids: ["art_bundle_c"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          listening_decision: "rejected",
        },
        tags: [],
        path: "/tmp/c.wav",
        created_at: "2026-05-28T15:07:00.000Z",
      },
    ] as never);

    expect(rows[0]).toEqual(expect.objectContaining({
      prompt: "warm granular loop",
      total: 2,
      listened: 2,
      keeper: 1,
      maybe: 1,
      rejected: 0,
      bundleIds: ["art_bundle_b", "art_bundle_a"],
      latestArtifactId: "art_take_b",
      latestNote: "good but soft",
    }));
    expect(rows[1]).toEqual(expect.objectContaining({ prompt: "cold click", rejected: 1 }));
  });

  it("compares prompt-search generated takes across runs", () => {
    const currentBundle = {
      artifact_id: "art_bundle_a",
      kind: "bundle",
      source_artifact_ids: ["art_target"],
      metadata: { operator: "experiment.prompt_search" },
      tags: [],
      path: "/tmp/prompt-a.zip",
      file: { filename: "prompt-a.zip", byte_size: 10 },
      created_at: "2026-05-28T14:00:00.000Z",
    };
    const rows = promptSearchRunComparisonRows(
      currentBundle as never,
      [
        currentBundle,
        {
          artifact_id: "art_bundle_b",
          kind: "bundle",
          source_artifact_ids: ["art_target"],
          metadata: { operator: "experiment.prompt_search" },
          tags: [],
          path: "/tmp/prompt-b.zip",
          file: { filename: "prompt-b.zip", byte_size: 10 },
          created_at: "2026-05-28T14:05:00.000Z",
        },
        {
          artifact_id: "art_take_a",
          kind: "audio",
          prompt: "warm granular loop",
          source_artifact_ids: ["art_bundle_a"],
          metadata: {
            generation_origin: "prompt_search_candidate",
            prompt_search_bundle_id: "art_bundle_a",
            prompt_search_scorer: "sa3_flow_probe",
            prompt_search_mode: "beam",
            prompt_search_model: "medium",
            prompt_search_duration_seconds: 8,
            listening_decision: "keeper",
          },
          tags: [],
          path: "/tmp/take-a.wav",
          created_at: "2026-05-28T15:00:00.000Z",
        },
        {
          artifact_id: "art_take_b",
          kind: "audio",
          prompt: "warm granular loop shimmer",
          source_artifact_ids: ["art_bundle_b"],
          metadata: {
            generation_origin: "prompt_search_candidate",
            prompt_search_bundle_id: "art_bundle_b",
            prompt_search_scorer: "lexical_probe",
            prompt_search_mode: "greedy",
            prompt_search_model: "medium",
            listening_decision: "maybe",
          },
          tags: [],
          path: "/tmp/take-b.wav",
          created_at: "2026-05-28T15:05:00.000Z",
        },
      ] as never,
      {
        prompt_search: {
          prompt: "warm granular loop",
          scorer: "sa3_flow_probe",
          search_mode: "beam",
          model: "medium",
          duration_seconds: 8,
        },
      },
    );

    expect(rows[0]).toEqual(expect.objectContaining({
      bundleId: "art_bundle_a",
      current: true,
      label: "prompt-a.zip",
      totalTakes: 1,
      keeper: 1,
      scorer: "sa3_flow_probe",
      searchMode: "beam",
      model: "medium",
      durationSeconds: 8,
      promptCount: 1,
      prompts: ["warm granular loop"],
    }));
    expect(rows[1]).toEqual(expect.objectContaining({
      bundleId: "art_bundle_b",
      current: false,
      totalTakes: 1,
      maybe: 1,
      scorer: "lexical_probe",
      searchMode: "greedy",
      prompts: ["warm granular loop shimmer"],
    }));
  });
});
