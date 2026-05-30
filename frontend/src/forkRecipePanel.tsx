import { useEffect, useState } from "react";
import { GitFork, LoaderCircle, X } from "lucide-react";

import type { RecipeForkPayload } from "./api";
import { shortOperatorName } from "./jobUtils";
import type { Recipe } from "./types";

const forkBackendOptions = [
  { value: "mlx", label: "mlx" },
  { value: "torch_mps", label: "torch_mps" },
  { value: "torch_cpu", label: "torch_cpu" },
  { value: "cpu", label: "cpu" },
] as const;

export function ForkRecipePanel({
  recipe,
  submitting,
  onSubmit,
  onClose,
}: {
  recipe: Recipe;
  submitting: boolean;
  onSubmit: (payload: RecipeForkPayload) => void;
  onClose: () => void;
}) {
  const [params, setParams] = useState<Record<string, unknown>>(recipe.params);
  const [complexDrafts, setComplexDrafts] = useState<Record<string, string>>({});
  const [complexErrors, setComplexErrors] = useState<Record<string, string>>({});
  const [backend, setBackend] = useState<NonNullable<RecipeForkPayload["backend"]>>(recipe.backend);
  const [model, setModel] = useState(recipe.model ?? "");
  const [seed, setSeed] = useState(recipe.seed?.toString() ?? "");
  const [notes, setNotes] = useState(recipe.notes ?? "");

  useEffect(() => {
    setParams(recipe.params);
    setComplexDrafts({});
    setComplexErrors({});
    setBackend(recipe.backend);
    setModel(recipe.model ?? "");
    setSeed(recipe.seed?.toString() ?? "");
    setNotes(recipe.notes ?? "");
  }, [recipe]);

  const setParam = (key: string, value: unknown) => {
    setParams((current) => ({ ...current, [key]: value }));
  };

  const resetParam = (key: string) => {
    setParams((current) => ({ ...current, [key]: recipe.params[key] }));
    setComplexDrafts((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
    setComplexErrors((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  };

  const resetCore = () => {
    setBackend(recipe.backend);
    setModel(recipe.model ?? "");
    setSeed(recipe.seed?.toString() ?? "");
    setNotes(recipe.notes ?? "");
  };

  const resetAll = () => {
    resetCore();
    setParams(recipe.params);
    setComplexDrafts({});
    setComplexErrors({});
  };

  const paramChanged = (key: string, value: unknown) =>
    !recipeValuesEqual(recipeValueForDiff(value, complexDrafts[key]), recipe.params[key]);
  const seedForDiff = seed.trim() ? Number(seed) : null;
  const coreDiffs = [
    backend !== recipe.backend ? "backend" : null,
    (model.trim() || "") !== (recipe.model ?? "") ? "model" : null,
    (Number.isFinite(seedForDiff) ? seedForDiff : null) !== (recipe.seed ?? null) ? "seed" : null,
    (notes.trim() || "") !== (recipe.notes ?? "") ? "notes" : null,
  ].filter((item): item is string => Boolean(item));
  const paramDiffs = Object.entries(params)
    .filter(([key, value]) => paramChanged(key, value))
    .map(([key]) => prettyParamName(key));
  const diffLabels = [...coreDiffs, ...paramDiffs];

  const submit = () => {
    const parsedParams = { ...params };
    const errors: Record<string, string> = {};
    for (const [key, draft] of Object.entries(complexDrafts)) {
      try {
        parsedParams[key] = JSON.parse(draft);
      } catch {
        errors[key] = "Invalid JSON";
      }
    }
    setComplexErrors(errors);
    if (Object.keys(errors).length) return;
    const parsedSeed = seed.trim() ? Number(seed) : null;
    onSubmit({
      backend,
      model: model.trim() || null,
      seed: Number.isFinite(parsedSeed) ? parsedSeed : null,
      notes: notes.trim() || null,
      params: parsedParams,
    });
  };

  return (
    <section className="fork-panel">
      <div className="fork-head">
        <div>
          <span className="eyebrow">Branch Gesture</span>
          <strong>{shortOperatorName(recipe.operator)}</strong>
        </div>
        <button type="button" onClick={onClose} title="Close branch editor">
          <X size={15} />
        </button>
      </div>
      <div className="fork-core-grid">
        <label className={backend !== recipe.backend ? "changed" : ""}>
          <ForkFieldHead label="Backend" changed={backend !== recipe.backend} onReset={() => setBackend(recipe.backend)} />
          <select value={backend} onChange={(event) => setBackend(event.target.value as NonNullable<RecipeForkPayload["backend"]>)}>
            {forkBackendOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className={(model.trim() || "") !== (recipe.model ?? "") ? "changed" : ""}>
          <ForkFieldHead label="Model" changed={(model.trim() || "") !== (recipe.model ?? "")} onReset={() => setModel(recipe.model ?? "")} />
          <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="default" />
        </label>
        <label className={(Number.isFinite(seedForDiff) ? seedForDiff : null) !== (recipe.seed ?? null) ? "changed" : ""}>
          <ForkFieldHead
            label="Seed"
            changed={(Number.isFinite(seedForDiff) ? seedForDiff : null) !== (recipe.seed ?? null)}
            onReset={() => setSeed(recipe.seed?.toString() ?? "")}
          />
          <input type="number" value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="random" />
        </label>
      </div>
      <RecipeDiffSummary labels={diffLabels} onReset={resetAll} />
      <div className="fork-param-grid">
        {Object.entries(params).length ? (
          Object.entries(params).map(([key, value]) => (
            <ForkParamControl
              key={key}
              name={key}
              value={value}
              draft={complexDrafts[key]}
              error={complexErrors[key]}
              changed={paramChanged(key, value)}
              onChange={(next) => setParam(key, next)}
              onDraft={(draft) => setComplexDrafts((current) => ({ ...current, [key]: draft }))}
              onReset={() => resetParam(key)}
            />
          ))
        ) : (
          <div className="quiet-panel compact">No gesture parameters</div>
        )}
      </div>
      <label className={`fork-notes ${(notes.trim() || "") !== (recipe.notes ?? "") ? "changed" : ""}`}>
        <ForkFieldHead label="Notes" changed={(notes.trim() || "") !== (recipe.notes ?? "")} onReset={() => setNotes(recipe.notes ?? "")} />
        <textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="why this branch exists" />
      </label>
      <button type="button" className="fork-submit" disabled={submitting} onClick={submit}>
        {submitting ? <LoaderCircle className="spin" size={15} /> : <GitFork size={15} />}
        Branch with changes
      </button>
    </section>
  );
}

function RecipeDiffSummary({ labels, onReset }: { labels: string[]; onReset: () => void }) {
  if (!labels.length) {
    return <div className="recipe-diff-summary idle">No gesture changes yet</div>;
  }
  return (
    <div className="recipe-diff-summary">
      <div>
        <strong>{labels.length} changed</strong>
        <span>{labels.slice(0, 8).join(", ")}</span>
      </div>
      <button type="button" onClick={onReset}>
        Reset all
      </button>
    </div>
  );
}

function ForkFieldHead({ label, changed, onReset }: { label: string; changed?: boolean; onReset?: () => void }) {
  return (
    <span className="fork-field-head">
      <span>{label}</span>
      {changed && onReset ? (
        <button
          type="button"
          className="fork-reset"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onReset();
          }}
        >
          Reset
        </button>
      ) : null}
    </span>
  );
}

function ForkParamControl({
  name,
  value,
  draft,
  error,
  changed,
  onChange,
  onDraft,
  onReset,
}: {
  name: string;
  value: unknown;
  draft?: string;
  error?: string;
  changed?: boolean;
  onChange: (value: unknown) => void;
  onDraft: (value: string) => void;
  onReset: () => void;
}) {
  if (typeof value === "boolean") {
    return (
      <label className={`fork-param checkbox ${changed ? "changed" : ""}`}>
        <ForkFieldHead label={prettyParamName(name)} changed={changed} onReset={onReset} />
        <input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} />
      </label>
    );
  }
  if (typeof value === "number") {
    return (
      <label className={`fork-param ${changed ? "changed" : ""}`}>
        <ForkFieldHead label={prettyParamName(name)} changed={changed} onReset={onReset} />
        <input type="number" value={value} onChange={(event) => onChange(Number(event.target.value))} />
      </label>
    );
  }
  if (typeof value === "string") {
    return (
      <label className={`fork-param ${changed ? "changed" : ""}`}>
        <ForkFieldHead label={prettyParamName(name)} changed={changed} onReset={onReset} />
        <input value={value} onChange={(event) => onChange(event.target.value)} />
      </label>
    );
  }
  return (
    <label className={`fork-param complex ${error ? "invalid" : ""} ${changed ? "changed" : ""}`}>
      <ForkFieldHead label={prettyParamName(name)} changed={changed} onReset={onReset} />
      <textarea value={draft ?? JSON.stringify(value, null, 2)} onChange={(event) => onDraft(event.target.value)} />
      {error ? <small>{error}</small> : null}
    </label>
  );
}

function prettyParamName(name: string) {
  return name.replaceAll("_", " ");
}

function recipeValueForDiff(value: unknown, draft?: string) {
  if (draft === undefined) return value;
  try {
    return JSON.parse(draft) as unknown;
  } catch {
    return { invalidDraft: draft };
  }
}

function recipeValuesEqual(left: unknown, right: unknown) {
  return stableRecipeStringify(left) === stableRecipeStringify(right);
}

function stableRecipeStringify(value: unknown): string {
  try {
    return JSON.stringify(sortRecipeValue(value));
  } catch {
    return String(value);
  }
}

function sortRecipeValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortRecipeValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, item]) => [key, sortRecipeValue(item)]),
  );
}
