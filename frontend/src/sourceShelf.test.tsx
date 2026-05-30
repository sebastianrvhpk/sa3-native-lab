import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { buildProductSources } from "./sourceModel";
import { SourceShelf } from "./sourceShelf";
import { testArtifact } from "./test/fixtures";

vi.mock("./audioDeck", () => ({
  TinyWave: ({ artifact }: { artifact: { artifact_id: string } }) => <span data-testid={`tiny-wave-${artifact.artifact_id}`} />,
}));

describe("SourceShelf", () => {
  it("routes active, remembered, donor, and recovered source actions through one surface", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onUseAction = vi.fn();
    const active = testArtifact({ artifact_id: "art_current", label: "Current Loop" });
    const memory = testArtifact({ artifact_id: "art_memory", label: "Memory Loop", session_id: null, metadata: { memory_role: "loop" } });
    const donor = testArtifact({ artifact_id: "art_donor", label: "Donor Latent", kind: "latent", session_id: null });
    const sources = buildProductSources([active, memory, donor], {
      activeSessionId: "sess_1",
      currentArtifactId: "art_current",
    });

    render(
      <SourceShelf
        sources={sources}
        selectedId="art_current"
        apiBase="http://api.test"
        onSelect={onSelect}
        onUseAction={onUseAction}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Select Memory Loop" }));
    const memoryRow = screen.getByText("Memory Loop").closest("article");
    const donorRow = screen.getByText("Donor Latent").closest("article");
    expect(memoryRow).not.toBeNull();
    expect(donorRow).not.toBeNull();
    await user.click(within(memoryRow as HTMLElement).getByRole("button", { name: "Source" }));
    await user.click(within(memoryRow as HTMLElement).getByRole("button", { name: "Recover" }));
    await user.click(within(donorRow as HTMLElement).getByRole("button", { name: "Donor" }));

    expect(onSelect).toHaveBeenCalledWith("art_memory");
    expect(onUseAction).toHaveBeenCalledWith(expect.objectContaining({ artifactId: "art_memory" }), expect.objectContaining({ intent: "source" }));
    expect(onUseAction).toHaveBeenCalledWith(expect.objectContaining({ artifactId: "art_memory" }), expect.objectContaining({ intent: "recover" }));
    expect(onUseAction).toHaveBeenCalledWith(expect.objectContaining({ artifactId: "art_donor" }), expect.objectContaining({ intent: "donor" }));
  });
});
