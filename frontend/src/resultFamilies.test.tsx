import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FamilyDetailPanel } from "./resultFamilies";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, Recipe } from "./types";

describe("FamilyDetailPanel", () => {
  it("surfaces alpha sweep variants with A/B promotion actions", async () => {
    const user = userEvent.setup();
    const onCompare = vi.fn();
    const onSelect = vi.fn();
    const onForkRecipe = vi.fn();
    const family = sweepFamily();
    const artifacts = [
      audioArtifact("art_pos", "alpha_pos4p00.wav", "2026-05-27T15:02:00.000Z", { score: 0.91 }),
      audioArtifact("art_neg", "alpha_neg4p00.wav", "2026-05-27T15:01:00.000Z", { score: 0.32 }),
    ];

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <FamilyDetailPanel
          family={family}
          artifacts={artifacts}
          jobs={[]}
          selectedId={null}
          apiBase="http://api.test"
          onSelect={onSelect}
          onCompare={onCompare}
          onReplayRecipe={vi.fn()}
          onForkRecipe={onForkRecipe}
        />
      </QueryClientProvider>,
    );

    expect(screen.getByLabelText("Alpha sweep variants")).toBeInTheDocument();
    expect(screen.getByLabelText("Alpha sweep metric table")).toBeInTheDocument();
    expect(screen.getByText("alpha -4")).toBeInTheDocument();
    expect(screen.getByText("alpha +4")).toBeInTheDocument();
    expect(screen.getByText("0.32")).toBeInTheDocument();

    const negativeVariant = screen.getByText("alpha -4").closest("article");
    expect(negativeVariant).not.toBeNull();
    await user.click(within(negativeVariant as HTMLElement).getByRole("button", { name: "A" }));
    await user.click(within(negativeVariant as HTMLElement).getByTitle("Fork the sweep recipe"));

    expect(onCompare).toHaveBeenCalledWith("a", "art_neg");
    expect(onForkRecipe).toHaveBeenCalledWith(family.recipe);
  });
});

function sweepFamily(): ResultFamily {
  const recipe: Recipe = {
    recipe_id: "recipe_sweep",
    operator: "experiment.alpha_sweep",
    backend: "torch_mps",
    inputs: {},
    params: { alphas: "-4,4", prompt: "glass rhythm" },
    model: "medium",
    seed: 7,
    notes: null,
    session_id: "sess_1",
    created_at: "2026-05-27T15:00:00.000Z",
    version: 1,
  };
  return {
    familyId: recipe.recipe_id,
    recipeId: recipe.recipe_id,
    recipe,
    operator: recipe.operator,
    sessionId: "sess_1",
    status: "succeeded",
    jobIds: [],
    artifactIds: ["art_pos", "art_neg"],
    artifactKinds: ["audio"],
    latestArtifactId: "art_pos",
    metrics: {},
    createdAt: recipe.created_at,
    updatedAt: "2026-05-27T15:02:00.000Z",
  };
}

function audioArtifact(artifactId: string, filename: string, createdAt: string, metadata: Record<string, unknown> = {}): ArtifactRecord {
  return {
    artifact_id: artifactId,
    kind: "audio",
    path: `/tmp/${filename}`,
    file: { filename, media_type: "audio/wav", byte_size: 128 },
    audio: { sample_rate: 24000, channels: 1, frames: 24000, duration_seconds: 1, format: "WAV" },
    latent: null,
    source_artifact_ids: [],
    recipe_id: "recipe_sweep",
    label: filename.replace(".wav", ""),
    prompt: "glass rhythm",
    notes: null,
    tags: [],
    metadata,
    session_id: "sess_1",
    created_at: createdAt,
  };
}
