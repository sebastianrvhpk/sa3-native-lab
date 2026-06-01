import { useState } from "react";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AuditionStackPanel } from "./auditionStackPanel";
import { testArtifact } from "./test/fixtures";

vi.mock("./audioDeck", () => ({
  AudioDeck: ({
    artifact,
    autoPlay,
    onEnded,
  }: {
    artifact: { artifact_id: string };
    autoPlay?: boolean;
    onEnded?: () => void;
  }) => (
    <button type="button" data-testid={`deck-${artifact.artifact_id}`} data-autoplay={autoPlay ? "true" : "false"} onClick={() => onEnded?.()}>
      end {artifact.artifact_id}
    </button>
  ),
}));

describe("AuditionStackPanel", () => {
  it("turns the take strip into a listening queue with decide, remember, continue, and branch actions", async () => {
    const user = userEvent.setup();
    const take = testArtifact({ artifact_id: "art_take", label: "Queue Take", recipe_id: "recipe_take" });
    const onAnnotate = vi.fn();
    const onRemember = vi.fn();
    const onContinue = vi.fn();
    const onBranch = vi.fn();
    const onSelect = vi.fn();

    render(
      <AuditionStackPanel
        artifacts={[take]}
        selectedId="art_take"
        apiBase="http://api.test"
        activeSessionId="sess_1"
        archivingArtifactId={null}
        onSelect={onSelect}
        onCompare={vi.fn()}
        onAnnotate={onAnnotate}
        onRemember={onRemember}
        onContinue={onContinue}
        onBranch={onBranch}
      />,
    );

    const row = screen.getByText("Queue Take").closest("article");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText("selected take")).toBeInTheDocument();
    await user.click(within(row as HTMLElement).getByRole("button", { name: /Keep/ }));
    await user.click(within(row as HTMLElement).getByRole("button", { name: /Continue/ }));
    await user.click(within(row as HTMLElement).getByRole("button", { name: /Branch/ }));
    await user.click(within(row as HTMLElement).getByRole("button", { name: /Remember/ }));

    expect(onAnnotate).toHaveBeenCalledWith("art_take", expect.objectContaining({
      metadata: expect.objectContaining({ listening_decision: "keeper", listening_decision_source: "take_strip" }),
    }));
    expect(onSelect).toHaveBeenCalledWith("art_take");
    expect(onContinue).toHaveBeenCalledWith(take);
    expect(onBranch).toHaveBeenCalledWith(take);
    expect(onRemember).toHaveBeenCalledWith(take);
  });

  it("keeps branch and remember disabled when the take has no active recipe or session", () => {
    const imported = testArtifact({ artifact_id: "art_import", label: "Imported Loop", recipe_id: null, session_id: null });

    render(
      <AuditionStackPanel
        artifacts={[imported]}
        selectedId="art_import"
        apiBase="http://api.test"
        activeSessionId="sess_1"
        archivingArtifactId={null}
        onSelect={vi.fn()}
        onCompare={vi.fn()}
        onAnnotate={vi.fn()}
        onRemember={vi.fn()}
        onContinue={vi.fn()}
        onBranch={vi.fn()}
      />,
    );

    const row = screen.getByText("Imported Loop").closest("article");
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByRole("button", { name: /Branch/ })).toBeDisabled();
    expect(within(row as HTMLElement).getByRole("button", { name: /Remember/ })).toBeDisabled();
  });

  it("summarizes listening decisions for the visible queue", () => {
    render(
      <AuditionStackPanel
        artifacts={[
          testArtifact({ artifact_id: "art_keep", label: "Keep", metadata: { listening_decision: "keeper" } }),
          testArtifact({ artifact_id: "art_maybe", label: "Maybe", metadata: { listening_decision: "maybe" } }),
          testArtifact({ artifact_id: "art_open", label: "Open" }),
        ]}
        selectedId="art_open"
        apiBase="http://api.test"
        activeSessionId="sess_1"
        archivingArtifactId={null}
        onSelect={vi.fn()}
        onCompare={vi.fn()}
        onAnnotate={vi.fn()}
        onRemember={vi.fn()}
        onContinue={vi.fn()}
        onBranch={vi.fn()}
      />,
    );

    const summary = screen.getByLabelText("Queue listening decision summary");
    expect(summary).toHaveTextContent("1 keeper");
    expect(summary).toHaveTextContent("1 maybe");
    expect(summary).toHaveTextContent("1 open");
  });

  it("plays forward through the current visible queue without creating a playlist mode", async () => {
    const user = userEvent.setup();
    const artifacts = [
      testArtifact({ artifact_id: "art_new", label: "Newest", created_at: "2026-01-03T00:00:00Z" }),
      testArtifact({ artifact_id: "art_mid", label: "Middle", created_at: "2026-01-02T00:00:00Z" }),
      testArtifact({ artifact_id: "art_old", label: "Oldest", created_at: "2026-01-01T00:00:00Z" }),
    ];

    function Harness() {
      const [selectedId, setSelectedId] = useState("art_new");
      return (
        <AuditionStackPanel
          artifacts={artifacts}
          selectedId={selectedId}
          apiBase="http://api.test"
          activeSessionId="sess_1"
          archivingArtifactId={null}
          onSelect={setSelectedId}
          onCompare={vi.fn()}
          onAnnotate={vi.fn()}
          onRemember={vi.fn()}
          onContinue={vi.fn()}
          onBranch={vi.fn()}
        />
      );
    }

    render(<Harness />);

    const auto = screen.getByRole("button", { name: "Auto" });
    await user.click(auto);
    expect(auto).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("deck-art_new")).toHaveAttribute("data-autoplay", "true");

    await user.click(screen.getByTestId("deck-art_new"));

    expect(screen.getByText("Middle").closest("article")).toHaveClass("selected");
    expect(screen.getByTestId("deck-art_mid")).toHaveAttribute("data-autoplay", "true");
    expect(screen.getByTestId("deck-art_new")).toHaveAttribute("data-autoplay", "false");
  });
});
