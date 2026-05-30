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
    expect(bundleReuseActionsForContext({ kind: "dataset" })).toEqual([
      { label: "Use encoded dataset", fieldKey: "encoded_dir", mode: "training.lora" },
    ]);
  });

  it("carries prompt-search prompts only into the supported sweep prompt field", () => {
    expect(bundleReuseActionsForContext({ kind: "prompt-search", prompt: "warm granular loop" })).toEqual([
      { label: "Use prompt in sweep", fieldKey: "prompt", mode: "experiment.alpha_sweep", value: "warm granular loop" },
    ]);
  });

  it("dedupes kind and operator matches for the same real path", () => {
    expect(bundleReuseActionsForContext({ kind: "vectors", operator: "experiment.style_direction.generate" })).toHaveLength(2);
  });
});
