import type { ComponentProps } from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { Specimen } from "./specimenPanel";
import { testArtifact, testFamily, testJob, testRecipe } from "./test/fixtures";

vi.mock("./audioDeck", () => ({
  AudioDeck: ({ artifact }: { artifact: { label?: string | null; artifact_id: string } }) => (
    <div data-testid="audio-deck">{artifact.label ?? artifact.artifact_id}</div>
  ),
}));

describe("Specimen", () => {
  it("surfaces artifact context and routes playback actions to the app", async () => {
    const user = userEvent.setup();
    const recipe = testRecipe({ recipe_id: "recipe_take" });
    const artifact = testArtifact({ artifact_id: "art_take", recipe_id: recipe.recipe_id, label: "Warm Smoke Take", tags: ["favorite"] });
    const onAnnotate = vi.fn();
    const onCompare = vi.fn();
    const onReplayRecipe = vi.fn();
    const onForkRecipe = vi.fn();
    const onArchiveArtifact = vi.fn();
    const onSelectArtifact = vi.fn();

    render(
      <Specimen
        artifact={artifact}
        artifacts={[artifact]}
        jobs={[testJob({ recipe })]}
        families={[testFamily({ recipe })]}
        compare={{ a: null, b: null }}
        apiBase="http://api.test"
        annotating={false}
        activeSessionId="sess_1"
        archivingArtifactId={null}
        onAnnotate={onAnnotate}
        onCompare={onCompare}
        onReplayRecipe={onReplayRecipe}
        onForkRecipe={onForkRecipe}
        onArchiveArtifact={onArchiveArtifact}
        onSelectArtifact={onSelectArtifact}
        onUseAsDonor={vi.fn()}
        onUseInRecipe={vi.fn()}
        onUsePrompt={vi.fn()}
        onGeneratePrompt={vi.fn()}
        getArtifactPath={(item) => item.path}
      />,
    );

    expect(screen.getByText("Warm Smoke Take")).toBeInTheDocument();
    expect(screen.getByText("recipe_take")).toBeInTheDocument();
    expect(screen.getByText("1.0s")).toBeInTheDocument();
    expect(screen.getByLabelText("Artifact lineage")).toHaveTextContent("take");

    await user.click(screen.getByRole("button", { name: "Anchor" }));
    await user.click(screen.getByRole("button", { name: "Source" }));
    await user.click(screen.getByRole("button", { name: "Do again" }));
    await user.click(screen.getByRole("button", { name: "Branch" }));
    await user.click(screen.getByRole("button", { name: "Remember sound" }));

    expect(onCompare).toHaveBeenCalledWith("a", "art_take");
    expect(onCompare).toHaveBeenCalledWith("b", "art_take");
    expect(onReplayRecipe).toHaveBeenCalledWith("recipe_take");
    expect(onForkRecipe).toHaveBeenCalledWith(recipe);
    expect(onArchiveArtifact).toHaveBeenCalledWith(artifact);
  });

  it("saves trimmed annotations and de-duplicates tags", async () => {
    const user = userEvent.setup();
    const onAnnotate = vi.fn();

    renderSpecimen({ onAnnotate });

    await user.clear(screen.getByLabelText("Label"));
    await user.type(screen.getByLabelText("Label"), "  keeper take  ");
    await user.clear(screen.getByLabelText("Tags"));
    await user.type(screen.getByLabelText("Tags"), "favorite, loop, Favorite, noisy");
    await user.type(screen.getByLabelText("Notes"), "  brittle opening  ");
    await user.click(screen.getByRole("button", { name: /save annotation/i }));

    expect(onAnnotate).toHaveBeenCalledWith("art_take", {
      label: "keeper take",
      notes: "brittle opening",
      tags: ["favorite", "loop", "noisy"],
    });
  });

  it("resets annotation form state when the selected artifact changes", async () => {
    const user = userEvent.setup();
    const onAnnotate = vi.fn();
    const first = testArtifact({ artifact_id: "art_take", label: "Warm Smoke Take", tags: ["warm"], notes: "first note" });
    const second = testArtifact({ artifact_id: "art_second", label: "Second Take", tags: ["wide"], notes: "second note" });
    const view = renderSpecimen({ artifact: first, onAnnotate });

    await user.clear(screen.getByLabelText("Label"));
    await user.type(screen.getByLabelText("Label"), "edited locally");

    view.rerender(specimenElement({ artifact: second, onAnnotate }));

    expect(screen.getByLabelText("Label")).toHaveValue("Second Take");
    expect(screen.getByLabelText("Tags")).toHaveValue("wide");
    expect(screen.getByLabelText("Notes")).toHaveValue("second note");
    await user.click(screen.getByRole("button", { name: /save annotation/i }));
    expect(onAnnotate).toHaveBeenLastCalledWith("art_second", expect.objectContaining({ label: "Second Take" }));
  });

  it("disables archive when the selected artifact is outside the active session", () => {
    renderSpecimen({ artifact: testArtifact({ session_id: null }) });

    expect(screen.getByRole("button", { name: "Remember sound" })).toBeDisabled();
  });
});

interface SpecimenFixtureOptions {
  artifact?: ReturnType<typeof testArtifact>;
  onAnnotate?: ComponentProps<typeof Specimen>["onAnnotate"];
}

function renderSpecimen(overrides: SpecimenFixtureOptions = {}) {
  return render(specimenElement(overrides));
}

function specimenElement({
  artifact = testArtifact({ artifact_id: "art_take", recipe_id: "recipe_take", label: "Warm Smoke Take", tags: ["favorite"], notes: null }),
  onAnnotate = vi.fn(),
}: SpecimenFixtureOptions = {}) {
  const recipe = testRecipe({ recipe_id: artifact.recipe_id ?? "recipe_take" });
  return (
    <Specimen
      artifact={artifact}
      artifacts={[artifact]}
      jobs={[testJob({ recipe, artifact_ids: [artifact.artifact_id] })]}
      families={[testFamily({ recipe, artifactIds: [artifact.artifact_id] })]}
      compare={{ a: null, b: null }}
      apiBase="http://api.test"
      annotating={false}
      activeSessionId="sess_1"
      archivingArtifactId={null}
      onAnnotate={onAnnotate}
      onCompare={vi.fn()}
      onReplayRecipe={vi.fn()}
      onForkRecipe={vi.fn()}
      onArchiveArtifact={vi.fn()}
      onSelectArtifact={vi.fn()}
      onUseAsDonor={vi.fn()}
      onUseInRecipe={vi.fn()}
      onUsePrompt={vi.fn()}
      onGeneratePrompt={vi.fn()}
      getArtifactPath={(item) => item.path}
    />
  );
}
