import { Plus, Repeat, X } from "lucide-react";

import type { OperatorPreset, OperatorPresetDiffRow } from "./operatorPresets";
import type { RecipeValue } from "./recipeFormModel";

export function OperatorPresetRack({
  presets,
  selectedPreset,
  selectedPresetId,
  presetName,
  diffRows,
  onSelectPreset,
  onChangePresetName,
  onSavePreset,
  onResetPreset,
  onDeletePreset,
}: {
  presets: OperatorPreset[];
  selectedPreset: OperatorPreset | null;
  selectedPresetId: string;
  presetName: string;
  diffRows: OperatorPresetDiffRow[];
  onSelectPreset: (presetId: string) => void;
  onChangePresetName: (name: string) => void;
  onSavePreset: () => void;
  onResetPreset: () => void;
  onDeletePreset: () => void;
}) {
  const diffStatus = !selectedPreset ? "empty" : diffRows.length ? "changed" : "clean";
  return (
    <div className="operator-preset-stack">
      <div className="operator-preset-rack" aria-label="Operator presets">
        <label>
          Preset
          <select value={selectedPresetId} onChange={(event) => onSelectPreset(event.target.value)}>
            <option value="">New preset</option>
            {presets.map((preset) => (
              <option key={preset.id} value={preset.id}>
                {preset.name}
              </option>
            ))}
          </select>
        </label>
        <label>
          Name
          <input
            value={presetName}
            onChange={(event) => onChangePresetName(event.target.value)}
            placeholder={presets.length ? "save current params" : "name this setting"}
          />
        </label>
        <button type="button" onClick={onSavePreset} title="Save current operator parameters">
          <Plus aria-hidden="true" size={13} />
          Save
        </button>
        <button type="button" onClick={onResetPreset} disabled={!selectedPreset || !diffRows.length} title="Revert current parameters to the selected preset">
          <Repeat aria-hidden="true" size={13} />
          Revert
        </button>
        <button type="button" onClick={onDeletePreset} disabled={!selectedPresetId} title="Delete selected operator preset">
          <X aria-hidden="true" size={13} />
          Delete
        </button>
      </div>
      <div className={`operator-preset-diff ${diffStatus}`} aria-label="Operator preset diff">
        <div className="operator-preset-diff-head">
          <strong>{!selectedPreset ? "Preset diff" : diffRows.length ? `${diffRows.length} unsaved change${diffRows.length === 1 ? "" : "s"}` : "Matches preset"}</strong>
          <small>{selectedPreset ? selectedPreset.name : "save or select a preset to compare params"}</small>
        </div>
        {diffRows.length ? (
          <div className="operator-preset-diff-list">
            {diffRows.slice(0, 4).map((row) => (
              <span key={row.key} className={row.status} title={`${formatPresetValue(row.presetValue)} -> ${formatPresetValue(row.currentValue)}`}>
                <b>{row.label}</b>
                <i>{formatPresetValue(row.presetValue)}</i>
                <em>{formatPresetValue(row.currentValue)}</em>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function formatPresetValue(value: RecipeValue | null): string {
  if (value === null || value === "") return "none";
  if (typeof value === "number") return Number.isInteger(value) ? value.toString() : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  if (typeof value === "boolean") return value ? "on" : "off";
  if (value.length > 18) return `${value.slice(0, 15)}...`;
  return value;
}
