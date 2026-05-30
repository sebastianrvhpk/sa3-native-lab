import { describe, expect, it } from "vitest";

import {
  defaultExperimentForm,
  defaultGenerationForm,
  defaultOperatorForm,
  experimentCatalog,
  generationControlKeys,
  isExperimentMode,
  isLatentOperatorMode,
  operatorCatalog,
  promptSearchScorerOptions,
  sameConfig,
} from "./workbenchConfigs";

describe("workbench configs", () => {
  it("keeps catalog defaults reachable outside App", () => {
    expect(operatorCatalog.map((item) => item.value)).toContain("latent.cyclic_roll");
    expect(experimentCatalog.map((item) => item.value)).toContain("experiment.prompt_search");
    expect(sameConfig.fields.map((field) => field.key)).toEqual(["model", "chunked", "chunk_size", "overlap", "prompt", "notes"]);
  });

  it("builds default forms from extracted configs", () => {
    expect(defaultGenerationForm("generate.text_to_audio")).toMatchObject({
      prompt: "short soft percussive click",
      model: "medium",
      duration_seconds: 5,
      decoder: "same-l",
    });
    expect(defaultOperatorForm("latent.cyclic_roll")).toMatchObject({
      shift_frames: 1,
      strength: 1,
      symmetric: true,
      backend: "torch_mps",
    });
    expect(defaultExperimentForm("experiment.prompt_search")).toMatchObject({
      seed_prompt: "audio texture",
      search_mode: "beam",
      scorer: "lexical_probe",
      model: "medium",
    });
  });

  it("guards modes and exposes mode-specific generation controls", () => {
    expect(isLatentOperatorMode("latent.graft")).toBe(true);
    expect(isLatentOperatorMode("generate.text_to_audio")).toBe(false);
    expect(isExperimentMode("dataset.pre_encode")).toBe(true);
    expect(isExperimentMode("latent.blur")).toBe(false);
    expect(generationControlKeys("generate.inpaint")).toContain("inpaint_start_seconds");
    expect(generationControlKeys("generate.audio_to_audio")).toContain("init_noise_level");
    expect(generationControlKeys("generate.text_to_audio")).not.toContain("init_noise_level");
    expect(promptSearchScorerOptions.map((option) => option.value)).toEqual(["lexical_probe", "sa3_flow_probe"]);
  });
});
