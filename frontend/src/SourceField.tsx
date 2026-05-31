import { Route } from "lucide-react";
import type { ReactNode } from "react";
import {
  Button as AriaButton,
  Input as AriaInput,
  Label as AriaLabel,
  TextField as AriaTextField,
} from "react-aria-components";

import {
  productSourceFieldOptions,
  type ProductSource,
  type SourceFieldValueMode,
} from "./sourceModel";
import type { ArtifactKind, ArtifactRecord } from "./types";

export function SourcePicker({
  label,
  ariaLabel,
  description,
  value,
  sources,
  selectedArtifact,
  artifactKinds,
  fieldKey,
  valueMode = "path",
  getArtifactPath,
  onChange,
  emptyLabel = "Source shelf",
  selectedButtonLabel = "Current",
  showSelectedButton = true,
  showExactValue = true,
  className = "",
}: {
  label: string;
  ariaLabel?: string;
  description?: string;
  value: string;
  sources: readonly ProductSource[];
  selectedArtifact: ArtifactRecord | null;
  artifactKinds?: readonly ArtifactKind[];
  fieldKey: string;
  valueMode?: SourceFieldValueMode;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
  onChange: (value: string) => void;
  emptyLabel?: string;
  selectedButtonLabel?: string;
  showSelectedButton?: boolean;
  showExactValue?: boolean;
  className?: string;
}) {
  const options = productSourceFieldOptions({
    sources,
    fieldKey,
    artifactKinds,
    valueMode,
    getArtifactPath,
  });
  const current = options.find((option) => option.value === value) ?? null;
  const selectedOption = selectedArtifact
    ? options.find((option) => option.artifact.artifact_id === selectedArtifact.artifact_id) ?? null
    : null;
  const exactValue = value.trim();

  return (
    <div className={`source-picker ${className}`} title={description}>
      <div className="source-picker-head">
        <span>{label}</span>
        {current?.roleLabels.length ? (
          <small>
            {current.roleLabels.slice(0, 2).map((role) => (
              <i key={role}>{role}</i>
            ))}
          </small>
        ) : null}
      </div>
      <div className="source-picker-actions">
        <select
          aria-label={ariaLabel ?? label}
          value={current?.value ?? ""}
          onChange={(event) => onChange(event.target.value)}
        >
          <option value="">{emptyLabel}</option>
          {options.map((option) => (
            <option key={`${option.artifact.artifact_id}:${option.value}`} value={option.value}>
              {option.label} · {option.detail}
            </option>
          ))}
        </select>
        {showSelectedButton ? (
          <AriaButton
            type="button"
            isDisabled={!selectedOption}
            onPress={() => selectedOption && onChange(selectedOption.value)}
          >
            <Route size={15} />
            {selectedButtonLabel}
          </AriaButton>
        ) : null}
      </div>
      {current ? (
        <small className="source-picker-current">
          Using {current.label}
        </small>
      ) : showExactValue && exactValue ? (
        <small className="source-picker-current">
          Exact value: <code>{exactValue}</code>
        </small>
      ) : !options.length ? (
        <small className="source-picker-current">No matching source yet</small>
      ) : null}
    </div>
  );
}

export function SourceField({
  label,
  description,
  placeholder,
  value,
  sources,
  selectedArtifact,
  artifactKinds,
  fieldKey,
  getArtifactPath,
  onChange,
  error,
}: {
  label: string;
  description?: string;
  placeholder?: string;
  value: string;
  sources: readonly ProductSource[];
  selectedArtifact: ArtifactRecord | null;
  artifactKinds?: readonly ArtifactKind[];
  fieldKey: string;
  getArtifactPath: (artifact: ArtifactRecord, fieldKey: string) => string;
  onChange: (value: string) => void;
  error?: ReactNode;
}) {
  return (
    <AriaTextField className="control-cell path-field source-field" value={value} onChange={onChange}>
      <AriaLabel>{label}</AriaLabel>
      <AriaInput placeholder={placeholder} title={description} />
      <SourcePicker
        label="Source shelf"
        ariaLabel={`${label} source`}
        description={description}
        value={value}
        sources={sources}
        selectedArtifact={selectedArtifact}
        artifactKinds={artifactKinds}
        fieldKey={fieldKey}
        getArtifactPath={getArtifactPath}
        onChange={onChange}
        showExactValue={false}
      />
      {error}
    </AriaTextField>
  );
}
