import { describe, expect, it } from "vitest";

import { bundleReuseActionsForContext } from "./bundleReuseModel";

describe("bundleReuseModel", () => {
  it("maps known bundle kinds to existing Advanced Gesture fields", () => {
    expect(bundleReuseActionsForContext({ kind: "vectors" })).toEqual([
      { label: "Sweep vectors", fieldKey: "vectors_path", mode: "experiment.alpha_sweep" },
      { label: "Use direction", fieldKey: "direction_path", mode: "experiment.style_direction.generate" },
    ]);
    expect(bundleReuseActionsForContext({ kind: "soft-prompt" })).toEqual([
      { label: "Use soft prompt", fieldKey: "soft_prompt_path", mode: "experiment.soft_prompt.generate" },
    ]);
    expect(bundleReuseActionsForContext({ kind: "dataset" })).toEqual([]);
  });

  it("carries prompt-search prompts only into the supported sweep prompt field", () => {
    expect(bundleReuseActionsForContext({ kind: "prompt-search", prompt: "warm granular loop" })).toEqual([
      { label: "Use prompt in sweep", fieldKey: "prompt", mode: "experiment.alpha_sweep", value: "warm granular loop" },
    ]);
  });

  it("dedupes kind and operator matches for the same real path", () => {
    expect(bundleReuseActionsForContext({ kind: "vectors", operator: "experiment.style_direction.generate" })).toHaveLength(2);
  });

  it("keeps profile and memory promotions on supported recipe fields", () => {
    expect(bundleReuseActionsForContext({ kind: "profile" })).toEqual([
      { label: "Use as profile", fieldKey: "profile_path", mode: "experiment.style_profile.generate" },
      { label: "Use memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" },
    ]);
    expect(bundleReuseActionsForContext({ kind: "memory" })).toEqual([
      { label: "Use as target memory", fieldKey: "target_memory_path", mode: "experiment.style_profile.build" },
      { label: "Use as reference", fieldKey: "reference_memory_path", mode: "experiment.style_profile.build" },
    ]);
    expect(bundleReuseActionsForContext({ kind: "training" })).toEqual([]);
  });
});
