import { describe, expect, it } from "vitest";

import { describeGestureAction } from "./gestureActionDescriptor";
import { buildGestureOptions } from "./gestureModel";
import { testArtifact } from "./test/fixtures";

describe("gestureActionDescriptor", () => {
  it("explains missing source requirements for source-based generation", () => {
    const gesture = buildGestureOptions(null).find((item) => item.id === "continue")!;
    const descriptor = describeGestureAction({
      ...baseInput(),
      gesture,
      generationMode: "generate.audio_to_audio",
      generationNeedsSource: true,
      generationSource: null,
      generationReady: false,
    });

    expect(descriptor.ready).toBe(false);
    expect(descriptor.disabledReason).toMatch(/choose or import/i);
    expect(descriptor.sourceRequirements.some((item) => item.label === "Source sound" && item.status === "missing")).toBe(true);
  });

  it("describes donor requirements without owning the mutation", () => {
    const latent = testArtifact({ kind: "latent", artifact_id: "art_latent", label: "Latent Pad" });
    const gesture = buildGestureOptions(latent).find((item) => item.id === "borrow_texture")!;
    const descriptor = describeGestureAction({
      ...baseInput(),
      gesture,
      selectedArtifact: latent,
      operatorMode: "latent.graft",
      operatorLabel: "Borrow Texture",
      operatorReady: false,
      operatorNeedsDonor: true,
      donorArtifactId: "",
    });

    expect(descriptor.kind).toBe("operator");
    expect(descriptor.ready).toBe(false);
    expect(descriptor.disabledReason).toBe("Choose a donor latent.");
    expect(descriptor.intentCopy).toMatch(/latent take/i);
  });

  it("uses listening-language labels for prompt search", () => {
    const audio = testArtifact({ label: "Bell source" });
    const gesture = buildGestureOptions(audio).find((item) => item.id === "steer")!;
    const descriptor = describeGestureAction({
      ...baseInput(),
      gesture,
      selectedArtifact: audio,
      experimentLabel: "Prompt search",
      experimentReady: true,
    });

    expect(descriptor.ready).toBe(true);
    expect(descriptor.label).toBe("Find prompt candidates");
    expect(descriptor.intentCopy).toMatch(/save the evidence/i);
  });
});

function baseInput(): Parameters<typeof describeGestureAction>[0] {
  return {
    gesture: buildGestureOptions(null)[0]!,
    selectedArtifact: null,
    generationMode: "generate.text_to_audio",
    generationForm: { prompt: "soft pulse", duration_seconds: 4, steps: 8 },
    generationNeedsSource: false,
    generationSource: null,
    generationReady: true,
    generationBusy: false,
    encodeReady: false,
    encodeBusy: false,
    decodeReady: false,
    decodeBusy: false,
    operatorMode: "latent.cyclic_roll",
    operatorLabel: "Roll",
    operatorReady: false,
    operatorBusy: false,
    operatorNeedsDonor: false,
    donorArtifactId: "",
    experimentLabel: "Audio style vectors",
    experimentReady: false,
    experimentBusy: false,
    rememberBusy: false,
  };
}
