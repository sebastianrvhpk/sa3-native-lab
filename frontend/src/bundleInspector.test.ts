import { describe, expect, it } from "vitest";

import { bundleReuseActions, summarizeBundle } from "./bundleInspector";

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
});
