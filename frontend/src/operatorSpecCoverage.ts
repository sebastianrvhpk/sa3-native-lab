import type { OperatorSpec } from "./types";

const systemParamKeys = new Set(["metadata"]);

export interface SpecCoverageSummary {
  status: "waiting" | "partial" | "covered";
  paramCount: number;
  missing: string[];
}

export function specParamKeys(spec: OperatorSpec | undefined): string[] {
  return Object.keys(spec?.params ?? {});
}

export function missingParamKeys(spec: OperatorSpec | undefined, controlledKeys: readonly string[]): string[] {
  const controlled = new Set(controlledKeys);
  return specParamKeys(spec).filter((key) => !systemParamKeys.has(key) && !controlled.has(key));
}

export function specCoverageSummary(spec: OperatorSpec | undefined, controlledKeys: readonly string[]): SpecCoverageSummary {
  const missing = missingParamKeys(spec, controlledKeys);
  return {
    status: !spec ? "waiting" : missing.length ? "partial" : "covered",
    paramCount: specParamKeys(spec).length,
    missing,
  };
}

export function specPairCoverageSummary(
  specs: readonly (OperatorSpec | undefined)[],
  controlledKeys: readonly (readonly string[])[],
): SpecCoverageSummary {
  const missing = specs.flatMap((spec, index) => missingParamKeys(spec, controlledKeys[index] ?? []));
  const readySpecs = specs.filter(Boolean) as OperatorSpec[];
  return {
    status: readySpecs.length !== specs.length ? "waiting" : missing.length ? "partial" : "covered",
    paramCount: readySpecs.reduce((count, spec) => count + specParamKeys(spec).length, 0),
    missing,
  };
}
