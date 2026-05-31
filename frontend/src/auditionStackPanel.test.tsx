import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AuditionStackPanel } from "./auditionStackPanel";
import { testArtifact } from "./test/fixtures";

vi.mock("./audioDeck", () => ({
  AudioDeck: ({ artifact }: { artifact: { artifact_id: string } }) => <div data-testid={`deck-${artifact.artifact_id}`} />,
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
});
