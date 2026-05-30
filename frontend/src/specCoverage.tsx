import { specCoverageSummary, specPairCoverageSummary } from "./operatorSpecCoverage";
import type { OperatorSpec } from "./types";

export function SpecCoverage({ spec, controlledKeys }: { spec: OperatorSpec | undefined; controlledKeys: readonly string[] }) {
  const coverage = specCoverageSummary(spec, controlledKeys);
  return (
    <div className={`spec-coverage ${coverage.status}`}>
      <span>{!spec ? "Contract pending" : coverage.missing.length ? `${coverage.missing.length} missing fields` : "Contract covered"}</span>
      <small>
        {spec ? `${coverage.paramCount} params · ${spec.backends.join(", ")} · ${spec.status}` : "waiting for /operators/specs"}
      </small>
      {coverage.missing.length ? <em title={coverage.missing.join(", ")}>{coverage.missing.slice(0, 4).join(", ")}</em> : null}
    </div>
  );
}

export function SpecCoveragePair({
  specs,
  controlledKeys,
}: {
  specs: readonly (OperatorSpec | undefined)[];
  controlledKeys: readonly (readonly string[])[];
}) {
  const coverage = specPairCoverageSummary(specs, controlledKeys);
  const readySpecs = specs.filter(Boolean) as OperatorSpec[];
  return (
    <div className={`spec-coverage ${coverage.status}`}>
      <span>{readySpecs.length !== specs.length ? "Contract pending" : coverage.missing.length ? `${coverage.missing.length} missing fields` : "Contract covered"}</span>
      <small>{readySpecs.length ? `${coverage.paramCount} params · encode/decode` : "waiting for /operators/specs"}</small>
      {coverage.missing.length ? <em title={coverage.missing.join(", ")}>{coverage.missing.slice(0, 4).join(", ")}</em> : null}
    </div>
  );
}
