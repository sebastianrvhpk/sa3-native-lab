import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PromptSearchPresetRack } from "./promptSearchRack";
import {
  promptSearchAxisSets,
  promptSearchPresets,
  promptSearchVocabularySets,
  type PromptSearchHistoryRow,
} from "./promptSearchPresets";

describe("PromptSearchPresetRack", () => {
  it("applies presets, token tools, axes, and prompt history from visible controls", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const onApplyVocabulary = vi.fn();
    const onApplyAxis = vi.fn();
    const onUseHistoryPrompt = vi.fn();
    const historyRows: PromptSearchHistoryRow[] = [{
      prompt: "warm glass rhythm",
      total: 4,
      keeper: 2,
      maybe: 1,
      rejected: 1,
      latestAt: "2026-05-28T15:00:00.000Z",
      latestNote: "worked as a seed phrase",
    }];

    render(
      <PromptSearchPresetRack
        presets={promptSearchPresets}
        vocabularySets={promptSearchVocabularySets}
        axisSets={promptSearchAxisSets}
        historyRows={historyRows}
        scorer="sa3_flow_probe"
        onApply={onApply}
        onApplyVocabulary={onApplyVocabulary}
        onApplyAxis={onApplyAxis}
        onUseHistoryPrompt={onUseHistoryPrompt}
      />,
    );

    expect(screen.getByText("SA3 flow probe")).toBeInTheDocument();
    expect(screen.getByText("slow MPS")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /Hard-token probe/i }));
    await user.click(screen.getByRole("button", { name: /Texture color/i }));
    await user.click(screen.getByRole("button", { name: /Readable timbre/i }));
    await user.click(screen.getByRole("button", { name: /warm glass rhythm/i }));

    expect(onApply).toHaveBeenCalledWith("mode2-hard-token");
    expect(onApplyVocabulary).toHaveBeenCalledWith("texture-color");
    expect(onApplyAxis).toHaveBeenCalledWith("readable-timbre");
    expect(onUseHistoryPrompt).toHaveBeenCalledWith("warm glass rhythm");
  });

  it("hides history when no prompt history is available", () => {
    render(
      <PromptSearchPresetRack
        presets={promptSearchPresets.slice(0, 1)}
        vocabularySets={[]}
        axisSets={[]}
        historyRows={[]}
        scorer="lexical_probe"
        onApply={vi.fn()}
        onApplyVocabulary={vi.fn()}
        onApplyAxis={vi.fn()}
        onUseHistoryPrompt={vi.fn()}
      />,
    );

    expect(screen.getByText("Lexical probe")).toBeInTheDocument();
    expect(screen.queryByLabelText("Prompt search history")).not.toBeInTheDocument();
  });
});
