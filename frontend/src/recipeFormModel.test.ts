import { describe, expect, it } from "vitest";

import {
  buildExperimentPayload,
  buildOperatorParams,
  defaultFieldForm,
  experimentReady,
  operatorUsesDonor,
  validateRecipeField,
  type ExperimentPayloadConfig,
  type OperatorPayloadConfig,
} from "./recipeFormModel";
import type { ArtifactRecord } from "./types";

describe("recipe form model", () => {
  it("builds default values from field metadata", () => {
    const form = defaultFieldForm({
      fields: [
        { key: "mode", label: "Mode", type: "select", options: [{ value: "gain", label: "gain" }] },
        { key: "enabled", label: "Enabled", type: "checkbox" },
        { key: "strength", label: "Strength", type: "range", defaultValue: 0.5 },
      ],
    });

    expect(form).toEqual({ mode: "gain", enabled: false, strength: 0.5 });
  });

  it("normalizes latent operator params for list fields and checkboxes", () => {
    const config: OperatorPayloadConfig<"latent.dsp"> = {
      value: "latent.dsp",
      defaultBackend: "torch_mps",
      fields: [
        { key: "channels", label: "Channels", type: "text" },
        { key: "pca_component_gains", label: "PCA", type: "text" },
        { key: "invert", label: "Invert", type: "checkbox" },
      ],
    };

    expect(buildOperatorParams(config, { channels: "1, 2 3.9", pca_component_gains: "0.2 1", invert: true })).toEqual({
      channels: [1, 2, 3],
      pca_component_gains: [0.2, 1],
      invert: true,
    });
  });

  it("detects donor-only DSP modes", () => {
    const config: OperatorPayloadConfig<"latent.dsp"> = {
      value: "latent.dsp",
      defaultBackend: "torch_mps",
      fields: [{ key: "mode", label: "Mode", type: "select" }],
    };

    expect(operatorUsesDonor(config, { mode: "fft_phase_blend" })).toBe(true);
    expect(operatorUsesDonor(config, { mode: "gain" })).toBe(false);
  });

  it("uses selected audio and latent artifacts as recipe inputs when configured", () => {
    const config: ExperimentPayloadConfig<"memory.query"> = {
      value: "memory.query",
      backend: "cpu",
      selectedLatentFallback: true,
      fields: [
        { key: "backend", label: "Backend", type: "select", defaultValue: "cpu" },
        { key: "top_k", label: "Top K", type: "number", defaultValue: 5 },
      ],
    };
    const latent = artifact("latent");

    expect(experimentReady(config, { backend: "cpu", top_k: 5 }, latent)).toBe(true);
    expect(buildExperimentPayload({ config, form: { backend: "cpu", top_k: 5 }, selectedArtifact: latent, sessionId: "sess_1" })).toMatchObject({
      operator: "memory.query",
      backend: "cpu",
      inputs: { source: latent.artifact_id },
      session_id: "sess_1",
      params: { top_k: 5 },
    });
  });

  it("validates required and bounded fields", () => {
    expect(validateRecipeField({ key: "input", label: "Input", type: "path", required: true }, "")).toBe("Input is required");
    expect(validateRecipeField({ key: "steps", label: "Steps", type: "number", min: 1 }, 0)).toBe("Steps must be at least 1");
    expect(validateRecipeField({ key: "steps", label: "Steps", type: "number", max: 5 }, 6)).toBe("Steps must be at most 5");
  });
});

function artifact(kind: ArtifactRecord["kind"]): ArtifactRecord {
  return {
    artifact_id: `art_${kind}`,
    kind,
    path: `/tmp/${kind}`,
    file: null,
    audio: null,
    latent: kind === "latent" ? { shape: [4, 8], latent_rate: 10.77, channel_first: false } : null,
    source_artifact_ids: [],
    recipe_id: null,
    label: kind,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: "sess_1",
    created_at: "2026-05-27T10:00:00.000Z",
  };
}
