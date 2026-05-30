import { Search, SlidersHorizontal, Wand2 } from "lucide-react";

import {
  promptSearchScorerNote,
  type PromptSearchAxisSet,
  type PromptSearchHistoryRow,
  type PromptSearchPreset,
  type PromptSearchVocabularySet,
} from "./promptSearchPresets";
import type { RecipeValue } from "./recipeFormModel";

export function PromptSearchPresetRack({
  presets,
  vocabularySets,
  axisSets,
  historyRows,
  scorer,
  onApply,
  onApplyVocabulary,
  onApplyAxis,
  onUseHistoryPrompt,
}: {
  presets: readonly PromptSearchPreset[];
  vocabularySets: readonly PromptSearchVocabularySet[];
  axisSets: readonly PromptSearchAxisSet[];
  historyRows: readonly PromptSearchHistoryRow[];
  scorer: RecipeValue | undefined;
  onApply: (presetId: string) => void;
  onApplyVocabulary: (setId: string) => void;
  onApplyAxis: (setId: string) => void;
  onUseHistoryPrompt: (prompt: string) => void;
}) {
  const note = promptSearchScorerNote(scorer);
  return (
    <div className="prompt-search-guide">
      <div className="prompt-search-preset-rack" aria-label="Prompt search presets">
        {presets.map((preset) => (
          <button key={preset.id} type="button" onClick={() => onApply(preset.id)} title={preset.intent}>
            <Search aria-hidden="true" size={13} />
            <span>{preset.label}</span>
            <small>{preset.modeLabel} · {preset.cost}</small>
          </button>
        ))}
      </div>
      <div className={`prompt-search-scorer-note ${note.maturity}`}>
        <strong>{note.label}</strong>
        <span>{note.cost}</span>
        <p>{note.guidance}</p>
      </div>
      <div className="prompt-search-token-tools" aria-label="Prompt search vocabulary tools">
        <div>
          <strong>Vocabulary</strong>
          <span>{vocabularySets.length} sets</span>
        </div>
        <div className="prompt-search-tool-buttons">
          {vocabularySets.map((set) => (
            <button key={set.id} type="button" onClick={() => onApplyVocabulary(set.id)} title={set.terms}>
              <Wand2 aria-hidden="true" size={13} />
              <span>{set.label}</span>
              <small>{set.focus}</small>
            </button>
          ))}
        </div>
        <div>
          <strong>Axes</strong>
          <span>Mode 3</span>
        </div>
        <div className="prompt-search-tool-buttons">
          {axisSets.map((set) => (
            <button key={set.id} type="button" onClick={() => onApplyAxis(set.id)} title={set.axes}>
              <SlidersHorizontal aria-hidden="true" size={13} />
              <span>{set.label}</span>
              <small>{set.focus}</small>
            </button>
          ))}
        </div>
      </div>
      {historyRows.length ? (
        <div className="prompt-search-history" aria-label="Prompt search history">
          <div>
            <strong>Prompt history</strong>
            <span>{historyRows.length} prompts</span>
          </div>
          {historyRows.slice(0, 4).map((row) => (
            <button key={row.prompt} type="button" onClick={() => onUseHistoryPrompt(row.prompt)} title={row.latestNote ?? row.prompt}>
              <span>{row.prompt}</span>
              <small>{row.keeper}K · {row.maybe}M · {row.rejected}R · {row.total} take{row.total === 1 ? "" : "s"}</small>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}
