import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { FamilyDetailPanel, ResultFamilyPanel } from "./resultFamilies";
import type { ResultFamily } from "./controlPlane";
import type { ArtifactRecord, Recipe } from "./types";

describe("FamilyDetailPanel", () => {
  it("labels grouped prompt candidate generations as takes", () => {
    const family = promptCandidateFamily();

    render(
      <ResultFamilyPanel
        families={[family]}
        artifacts={[]}
        selectedId={null}
        inspectedFamilyId={family.familyId}
        onSelect={vi.fn()}
        onInspectFamily={vi.fn()}
        onReplayRecipe={vi.fn()}
        onForkRecipe={vi.fn()}
      />,
    );

    expect(screen.getByText("Prompt candidates")).toBeInTheDocument();
    expect(screen.getByText("2 takes · Make · 1 source")).toBeInTheDocument();
  });

  it("routes branch card inspect, repeat, and branch actions", async () => {
    const user = userEvent.setup();
    const family = sweepFamily();
    const latest = audioArtifact("art_pos", "alpha_pos4p00.wav", "2026-05-27T15:02:00.000Z");
    const onSelect = vi.fn();
    const onInspectFamily = vi.fn();
    const onReplayRecipe = vi.fn();
    const onForkRecipe = vi.fn();

    render(
      <ResultFamilyPanel
        families={[family]}
        artifacts={[latest]}
        selectedId={null}
        inspectedFamilyId={null}
        onSelect={onSelect}
        onInspectFamily={onInspectFamily}
        onReplayRecipe={onReplayRecipe}
        onForkRecipe={onForkRecipe}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Steer branch/i }));
    await user.click(screen.getByRole("button", { name: "Inspect" }));
    await user.click(screen.getByRole("button", { name: "Do again" }));
    await user.click(screen.getByRole("button", { name: "Branch" }));

    expect(onInspectFamily).toHaveBeenCalledWith(family.familyId);
    expect(onSelect).toHaveBeenCalledWith("art_pos");
    expect(onReplayRecipe).toHaveBeenCalledWith(family.recipeId);
    expect(onForkRecipe).toHaveBeenCalledWith(family.recipe);
  });

  it("surfaces alpha sweep variants with anchor actions", async () => {
    const user = userEvent.setup();
    const onCompare = vi.fn();
    const onSelect = vi.fn();
    const onInspectFamily = vi.fn();
    const onForkRecipe = vi.fn();
    const onAnnotate = vi.fn();
    const family = sweepFamily();
    const sibling = sweepFamily({
      recipe_id: "recipe_sibling",
      params: { alphas: "-8,0,8", prompt: "glass rhythm", vectors_path: "art_vectors" },
      seed: 9,
      created_at: "2026-05-27T16:00:00.000Z",
      updatedAt: "2026-05-27T16:02:00.000Z",
    });
    const artifacts = [
      audioArtifact("art_pos", "alpha_pos4p00.wav", "2026-05-27T15:02:00.000Z", { score: 0.91 }),
      audioArtifact("art_neg", "alpha_neg4p00.wav", "2026-05-27T15:01:00.000Z", { score: 0.32 }),
    ];

    render(
      <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
        <FamilyDetailPanel
          family={family}
          families={[family, sibling]}
          artifacts={artifacts}
          jobs={[]}
          selectedId={null}
          apiBase="http://api.test"
          activeSessionId="sess_1"
          archivingArtifactId={null}
          onSelect={onSelect}
          onInspectFamily={onInspectFamily}
          onCompare={onCompare}
          onAnnotate={onAnnotate}
          onReplayRecipe={vi.fn()}
          onForkRecipe={onForkRecipe}
          onArchiveArtifact={vi.fn()}
        />
      </QueryClientProvider>,
    );

    expect(screen.getByLabelText("Alpha sweep variants")).toBeInTheDocument();
    expect(screen.getByLabelText("Alpha sweep metric table")).toBeInTheDocument();
    expect(screen.getByText("alpha -4")).toBeInTheDocument();
    expect(screen.getByText("alpha +4")).toBeInTheDocument();
    expect(screen.getByText("0.32")).toBeInTheDocument();
    expect(screen.getByText("highlight")).toBeInTheDocument();
    expect(screen.getByLabelText("Sibling sweep comparison")).toBeInTheDocument();
    expect(screen.getByText("3 alphas · 2 takes · medium · seed 9")).toBeInTheDocument();

    const negativeVariant = screen.getByText("alpha -4").closest("article");
    expect(negativeVariant).not.toBeNull();
    await user.click(within(negativeVariant as HTMLElement).getByRole("button", { name: "Anchor" }));
    const negativeArtifact = screen.getAllByText("alpha_neg4p00")
      .map((element) => element.closest(".family-artifact"))
      .find((element): element is HTMLElement => Boolean(element));
    expect(negativeArtifact).not.toBeNull();
    await user.click(within(negativeArtifact as HTMLElement).getByRole("button", { name: /keep/i }));
    await user.click(within(negativeVariant as HTMLElement).getByTitle("Branch from the sweep gesture"));
    await user.click(within(screen.getByLabelText("Sort sweep variants")).getByRole("button", { name: "score" }));
    await user.click(within(screen.getByLabelText("Sibling sweep comparison")).getByRole("button", { name: /inspect/i }));

    const tableRows = within(screen.getByLabelText("Alpha sweep metric table")).getAllByTitle("Select sweep artifact");
    expect(tableRows[0]).toHaveTextContent("alpha_pos4p00");

    expect(onCompare).toHaveBeenCalledWith("a", "art_neg");
    expect(onAnnotate).toHaveBeenCalledWith("art_neg", expect.objectContaining({
      tags: ["keeper"],
      metadata: expect.objectContaining({ listening_decision: "keeper", listening_decision_source: "family_detail" }),
    }));
    expect(onInspectFamily).toHaveBeenCalledWith("recipe_sibling");
    expect(onForkRecipe).toHaveBeenCalledWith(family.recipe);
  });
});

function sweepFamily(
  overrides: Partial<Recipe> & { updatedAt?: string } = {},
): ResultFamily {
  const recipe: Recipe = {
    recipe_id: overrides.recipe_id ?? "recipe_sweep",
    operator: "experiment.alpha_sweep",
    backend: "torch_mps",
    inputs: {},
    params: overrides.params ?? { alphas: "-4,4", prompt: "glass rhythm", vectors_path: "art_vectors" },
    model: overrides.model ?? "medium",
    seed: overrides.seed ?? 7,
    notes: null,
    session_id: "sess_1",
    created_at: overrides.created_at ?? "2026-05-27T15:00:00.000Z",
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
    updatedAt: overrides.updatedAt ?? "2026-05-27T15:02:00.000Z",
  };
}

function promptCandidateFamily(): ResultFamily {
  const recipe: Recipe = {
    recipe_id: "recipe_candidate_latest",
    operator: "generate.text_to_audio",
    backend: "mlx",
    inputs: { source: "art_prompt_bundle" },
    params: { metadata: { generation_origin: "prompt_search_candidate" } },
    model: "medium",
    seed: 7,
    notes: null,
    session_id: "sess_1",
    created_at: "2026-05-27T15:00:00.000Z",
    version: 1,
  };
  return {
    familyId: "prompt-candidates:art_prompt_bundle",
    recipeId: recipe.recipe_id,
    recipe,
    operator: recipe.operator,
    sessionId: "sess_1",
    status: "succeeded",
    jobIds: ["job_a", "job_b"],
    artifactIds: ["art_a", "art_b"],
    artifactKinds: ["audio"],
    latestArtifactId: "art_b",
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
