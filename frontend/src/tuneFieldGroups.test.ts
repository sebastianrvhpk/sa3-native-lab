import { describe, expect, it } from "vitest";

import { primaryFieldKeysForGesture, withTuneFieldGroups } from "./tuneFieldGroups";
import { generationCatalog, operatorCatalog, sameConfig } from "./workbenchConfigs";

describe("tuneFieldGroups", () => {
  it("keeps Make hand-shaped while leaving model and seed reachable", () => {
    expect(primaryFieldKeysForGesture({ gestureId: "make", generationMode: "generate.text_to_audio" })).toEqual([
      "prompt",
      "duration_seconds",
      "seed",
      "model",
    ]);
    const grouped = withTuneFieldGroups(generationCatalog[0], { gestureId: "make", generationMode: "generate.text_to_audio" });

    expect(grouped.fields.find((field) => field.key === "prompt")?.advanced).toBe(false);
    expect(grouped.fields.find((field) => field.key === "duration_seconds")?.advanced).toBe(false);
    expect(grouped.fields.find((field) => field.key === "decoder")?.advanced).toBe(true);
  });

  it("puts SAME chunking behind Parameters", () => {
    const grouped = withTuneFieldGroups(sameConfig, { gestureId: "encode" });

    expect(grouped.fields.find((field) => field.key === "model")?.advanced).toBe(false);
    expect(grouped.fields.find((field) => field.key === "chunk_size")?.advanced).toBe(true);
    expect(grouped.fields.find((field) => field.key === "overlap")?.advanced).toBe(true);
  });

  it("shapes Morph around the active latent move", () => {
    const blur = operatorCatalog.find((item) => item.value === "latent.blur")!;
    const grouped = withTuneFieldGroups(blur, { gestureId: "morph", operatorMode: "latent.blur" });

    expect(grouped.fields.find((field) => field.key === "mode")?.advanced).toBe(false);
    expect(grouped.fields.find((field) => field.key === "strength")?.advanced).toBe(false);
    expect(grouped.fields.find((field) => field.key === "temporal_sigma")?.advanced).toBe(true);
    expect(grouped.fields.find((field) => field.key === "backend")?.advanced).toBe(true);
  });
});
