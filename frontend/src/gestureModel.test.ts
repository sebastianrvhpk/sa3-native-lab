import { describe, expect, it } from "vitest";

import { buildGestureOptions, gestureForOperator, gestureLabelForOperator } from "./gestureModel";
import { testArtifact } from "./test/fixtures";

describe("gestureModel", () => {
  it("maps backend operators to product gestures", () => {
    expect(gestureForOperator("generate.text_to_audio")).toBe("make");
    expect(gestureForOperator("generate.audio_to_audio")).toBe("continue");
    expect(gestureForOperator("latent.encode")).toBe("encode");
    expect(gestureForOperator("latent.decode")).toBe("decode");
    expect(gestureForOperator("latent.graft")).toBe("borrow_texture");
    expect(gestureForOperator("latent.blur")).toBe("morph");
    expect(gestureLabelForOperator("experiment.alpha_sweep")).toBe("Steer");
  });

  it("keeps source requirements honest", () => {
    const withoutSource = buildGestureOptions(null);
    expect(withoutSource.find((gesture) => gesture.id === "make")?.available).toBe(true);
    expect(withoutSource.find((gesture) => gesture.id === "continue")?.disabledReason).toMatch(/Choose or import/);

    const audioSource = buildGestureOptions(testArtifact({ kind: "audio" }));
    expect(audioSource.find((gesture) => gesture.id === "encode")?.available).toBe(true);
    expect(audioSource.find((gesture) => gesture.id === "decode")?.disabledReason).toBe("Needs latent material.");

    const latentSource = buildGestureOptions(testArtifact({ kind: "latent" }));
    expect(latentSource.find((gesture) => gesture.id === "morph")?.available).toBe(true);
    expect(latentSource.find((gesture) => gesture.id === "continue")?.disabledReason).toBe("Needs an audio sound.");
  });
});
