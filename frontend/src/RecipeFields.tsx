import { useForm } from "@tanstack/react-form";
import { SlidersHorizontal } from "lucide-react";
import {
  Checkbox as AriaCheckbox,
  Input as AriaInput,
  Label as AriaLabel,
  NumberField as AriaNumberField,
  Slider as AriaSlider,
  SliderThumb,
  SliderTrack,
  TextField as AriaTextField,
} from "react-aria-components";

import { SourceField } from "./SourceField";
import {
  validateRecipeField,
  type FieldConfig,
  type RecipeField,
  type RecipeValue,
} from "./recipeFormModel";
import { buildProductSources, type ProductSource } from "./sourceModel";
import type { ArtifactRecord } from "./types";

interface RecipeFieldsProps {
  config: FieldConfig;
  form: Record<string, RecipeValue>;
  artifacts: ArtifactRecord[];
  sources?: ProductSource[];
  selectedArtifact: ArtifactRecord | null;
  onChange: (key: string, value: RecipeValue) => void;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
  getArtifactLabel: (artifact: ArtifactRecord) => string;
}

export function RecipeFields({
  config,
  form,
  artifacts,
  sources,
  selectedArtifact,
  onChange,
  getArtifactPath,
  getArtifactLabel,
}: RecipeFieldsProps) {
  const formController = useForm({ defaultValues: form });
  const sourceOptions = sources ?? buildProductSources(artifacts);
  const coreFields = config.fields.filter((field) => !field.advanced);
  const advancedFields = config.fields.filter((field) => field.advanced);
  const renderControl = (field: RecipeField) => (
    <formController.Field
      key={field.key}
      name={field.key}
      validators={{
        onChange: ({ value }) => validateRecipeField(field, value as RecipeValue | undefined),
      }}
    >
      {(fieldApi) => {
        const error = fieldApi.state.meta.errors[0]?.toString();
        return (
          <RecipeFieldControl
            field={field}
            value={form[field.key]}
            error={error}
            artifacts={artifacts}
            sources={sourceOptions}
            selectedArtifact={selectedArtifact}
            getArtifactPath={getArtifactPath}
            getArtifactLabel={getArtifactLabel}
            onChange={(value) => {
              fieldApi.handleChange(value);
              onChange(field.key, value);
            }}
          />
        );
      }}
    </formController.Field>
  );

  return (
    <div className="recipe-fields">
      {coreFields.map(renderControl)}
      {advancedFields.length ? (
        <details className="recipe-advanced">
          <summary>
            <SlidersHorizontal size={15} />
            Parameters
          </summary>
          <div className="recipe-fields advanced">{advancedFields.map(renderControl)}</div>
        </details>
      ) : null}
    </div>
  );
}

interface RecipeFieldControlProps {
  field: RecipeField;
  value: RecipeValue | undefined;
  error?: string;
  artifacts: ArtifactRecord[];
  sources: ProductSource[];
  selectedArtifact: ArtifactRecord | null;
  onChange: (value: RecipeValue) => void;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
  getArtifactLabel: (artifact: ArtifactRecord) => string;
}

function RecipeFieldControl({
  field,
  value,
  error,
  artifacts,
  sources,
  selectedArtifact,
  onChange,
  getArtifactPath,
  getArtifactLabel,
}: RecipeFieldControlProps) {
  if (field.type === "checkbox") {
    return (
      <AriaCheckbox className="field-checkbox aria-checkbox" isSelected={Boolean(value)} onChange={onChange}>
        <span className="checkbox-box" />
        <span title={field.description}>{field.label}</span>
        <FieldError error={error} />
      </AriaCheckbox>
    );
  }

  if (field.type === "select") {
    return (
      <label className="control-cell" title={field.description}>
        {field.label}
        <select value={String(value ?? "")} onChange={(event) => onChange(event.target.value)}>
          {field.options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <FieldError error={error} />
      </label>
    );
  }

  if (field.type === "number") {
    return (
      <AriaNumberField
        className="control-cell"
        minValue={field.min}
        maxValue={field.max}
        step={field.step}
        value={toNumberValue(value)}
        onChange={(next) => onChange(Number.isFinite(next) ? next : "")}
      >
        <AriaLabel>{field.label}</AriaLabel>
        <AriaInput title={field.description} />
        <FieldError error={error} />
      </AriaNumberField>
    );
  }

  if (field.type === "range") {
    const numericValue = toNumberValue(value, Number(field.defaultValue ?? field.min ?? 0));
    return (
      <AriaSlider
        className="control-cell range-field aria-range-field"
        minValue={field.min}
        maxValue={field.max}
        step={field.step ?? 0.01}
        value={numericValue}
        onChange={(next) => onChange(Array.isArray(next) ? next[0] : next)}
      >
        <span title={field.description}>
          <AriaLabel>{field.label}</AriaLabel>
          <strong>{formatControlNumber(numericValue)}</strong>
        </span>
        <div className="range-pair">
          <SliderTrack className="aria-slider-track">
            <SliderThumb className="aria-slider-thumb" />
          </SliderTrack>
          <input
            type="number"
            min={field.min}
            max={field.max}
            step={field.step ?? "any"}
            value={numericValue}
            onChange={(event) => onChange(event.target.value === "" ? "" : Number(event.target.value))}
          />
        </div>
        <FieldError error={error} />
      </AriaSlider>
    );
  }

  if (field.type === "artifact-path") {
    return (
      <SourceField
        label={field.label}
        description={field.description}
        placeholder={field.placeholder}
        value={String(value ?? "")}
        sources={sources}
        selectedArtifact={selectedArtifact}
        artifactKinds={field.artifactKinds}
        fieldKey={field.key}
        getArtifactPath={getArtifactPath}
        onChange={onChange}
        error={<FieldError error={error} />}
      />
    );
  }

  return (
    <AriaTextField className="control-cell" value={String(value ?? "")} onChange={onChange}>
      <AriaLabel>{field.label}</AriaLabel>
      <AriaInput placeholder={field.placeholder} title={field.description} />
      <FieldError error={error} />
    </AriaTextField>
  );
}

function FieldError({ error }: { error?: string }) {
  return error ? <small className="field-error">{error}</small> : null;
}

function toNumberValue(value: RecipeValue | undefined, fallback = 0) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatControlNumber(value: number) {
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/\.?0+$/, "");
}
