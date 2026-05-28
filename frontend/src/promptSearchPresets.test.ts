import { describe, expect, it } from "vitest";

import {
  applyPromptSearchAxisSet,
  applyPromptSearchPreset,
  applyPromptSearchVocabularySet,
  promptSearchHistoryRows,
  promptSearchPresetById,
  promptSearchScorerNote,
} from "./promptSearchPresets";

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

  it("applies vocabulary and axis sets to prompt-search params", () => {
    expect(applyPromptSearchVocabularySet({ vocabulary: "old" }, "texture-color")).toMatchObject({
      vocabulary: expect.stringContaining("glassy"),
    });
    expect(applyPromptSearchAxisSet({ search_mode: "beam" }, "readable-motion")).toMatchObject({
      search_mode: "coordinate",
      modifier_axes: expect.stringContaining("pulsing|drifting"),
    });
  });

  it("builds prompt history rows from generated prompt-search takes", () => {
    const rows = promptSearchHistoryRows([
      {
        artifact_id: "art_take_a",
        kind: "audio",
        prompt: "warm glass loop",
        source_artifact_ids: ["art_bundle_a"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          listening_decision: "keeper",
          listening_decision_note: "best so far",
        },
        tags: [],
        path: "/tmp/a.wav",
        created_at: "2026-05-28T15:00:00.000Z",
      },
      {
        artifact_id: "art_take_b",
        kind: "audio",
        prompt: "warm glass loop",
        source_artifact_ids: ["art_bundle_b"],
        metadata: {
          generation_origin: "prompt_search_candidate",
          listening_decision: "maybe",
        },
        tags: [],
        path: "/tmp/b.wav",
        created_at: "2026-05-28T15:05:00.000Z",
      },
      {
        artifact_id: "art_manual",
        kind: "audio",
        prompt: "manual",
        source_artifact_ids: [],
        metadata: {},
        tags: [],
        path: "/tmp/manual.wav",
        created_at: "2026-05-28T15:10:00.000Z",
      },
    ] as never);

    expect(rows).toEqual([
      expect.objectContaining({
        prompt: "warm glass loop",
        total: 2,
        keeper: 1,
        maybe: 1,
        rejected: 0,
      }),
    ]);
  });
});
