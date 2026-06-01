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

  it("renames primary Tune fields into sound-operation language without changing keys", () => {
    const grouped = withTuneFieldGroups(generationCatalog[1], { gestureId: "continue", generationMode: "generate.audio_to_audio" });

    expect(grouped.fields.find((field) => field.key === "duration_seconds")?.label).toBe("Length");
    expect(grouped.fields.find((field) => field.key === "init_noise_level")?.label).toBe("Variation amount");
    expect(grouped.fields.find((field) => field.key === "init_noise_level")?.description).toMatch(/current sound/i);
  });

  it("keeps backend-supported channel masks visible for graft and renoise", () => {
    const graft = operatorCatalog.find((item) => item.value === "latent.graft")!;
    const renoise = operatorCatalog.find((item) => item.value === "latent.renoise")!;

    expect(withTuneFieldGroups(graft, { gestureId: "borrow_texture", operatorMode: "latent.graft" }).fields.find((field) => field.key === "channels")?.advanced).toBe(false);
    expect(withTuneFieldGroups(renoise, { gestureId: "morph", operatorMode: "latent.renoise" }).fields.find((field) => field.key === "block_size")?.advanced).toBe(false);
  });
});
