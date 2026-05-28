import { describe, expect, it } from "vitest";

import { applyPromptSearchPreset, promptSearchPresetById, promptSearchScorerNote } from "./promptSearchPresets";

describe("prompt search presets", () => {
  it("applies the Mode 2 hard-token defaults without replacing target input", () => {
    expect(applyPromptSearchPreset({ target_audio_path: "/tmp/target.wav", seed_prompt: "glass" }, "mode2-hard-token")).toMatchObject({
      target_audio_path: "/tmp/target.wav",
      seed_prompt: "glass",
      search_mode: "beam",
      scorer: "lexical_probe",
      backend: "cpu",
      tokens_generated: 4,
      score_samples: 1,
    });
  });

  it("keeps Mode 3 readable prompt defaults distinct from Medium flow scoring", () => {
    expect(promptSearchPresetById("mode3-readable")).toMatchObject({
      modeLabel: "Mode 3",
      fields: {
        search_mode: "coordinate",
        scorer: "lexical_probe",
        modifier_axes: expect.stringContaining("bright|dark"),
      },
    });
    expect(promptSearchPresetById("medium-flow-check")).toMatchObject({
      cost: "slow MPS",
      fields: {
        scorer: "sa3_flow_probe",
        backend: "torch_mps",
        duration_seconds: 6,
        score_samples: 2,
      },
    });
  });

  it("explains selected scorer maturity and cost", () => {
    expect(promptSearchScorerNote("sa3_flow_probe")).toMatchObject({
      label: "SA3 flow probe",
      cost: "slow MPS",
      maturity: "probe",
    });
    expect(promptSearchScorerNote("unknown")).toMatchObject({
      scorer: "lexical_probe",
      cost: "fast CPU",
    });
  });
});
