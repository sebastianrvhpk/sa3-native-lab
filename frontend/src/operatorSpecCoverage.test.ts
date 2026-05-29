import { describe, expect, it } from "vitest";

import { missingParamKeys, specCoverageSummary, specPairCoverageSummary } from "./operatorSpecCoverage";
import type { OperatorSpec } from "./types";

describe("operator spec coverage", () => {
  it("ignores internal system params but reports real missing controls", () => {
    const spec = operatorSpec("generate.text_to_audio", {
      prompt: "str",
      duration_seconds: "float",
      metadata: "dict|null",
      model: "medium",
    });

    expect(missingParamKeys(spec, ["prompt", "duration_seconds"])).toEqual(["model"]);
    expect(specCoverageSummary(spec, ["prompt", "duration_seconds", "model"])).toMatchObject({
      status: "covered",
      paramCount: 4,
      missing: [],
    });
  });

  it("summarizes paired encode/decode specs without counting missing pending specs as covered", () => {
    expect(
      specPairCoverageSummary(
        [
          operatorSpec("latent.encode", { model: "same-s|same-l", chunked: "bool" }),
          operatorSpec("latent.decode", { model: "same-s|same-l", notes: "str|null" }),
        ],
        [["model", "chunked"], ["model"]],
      ),
    ).toMatchObject({
      status: "partial",
      paramCount: 4,
      missing: ["notes"],
    });
    expect(specPairCoverageSummary([operatorSpec("latent.encode", { model: "same-s|same-l" }), undefined], [["model"], []]).status).toBe("waiting");
  });
});

function operatorSpec(name: OperatorSpec["name"], params: Record<string, unknown>): OperatorSpec {
  return {
    name,
    maturity: "core",
    backends: ["mlx"],
    inputs: [],
    params,
    ui_fields: [],
    produces: ["audio"],
    status: "implemented",
  };
}
