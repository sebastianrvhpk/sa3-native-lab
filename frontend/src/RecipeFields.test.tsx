import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";

import { RecipeFields } from "./RecipeFields";
import type { ArtifactRecord } from "./types";

describe("RecipeFields", () => {
  it("renders accessible fields and reports changes", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();

    render(<RecipeFieldsHarness onChange={onChange} />);

    await user.type(screen.getByLabelText("Input folder"), "/tmp/audio");

    expect(onChange).toHaveBeenCalledWith("input_path", expect.stringContaining("/tmp/audio"));
  });

  it("can fill artifact-path fields from the selected artifact", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const selected = artifact("bundle");

    render(
      <RecipeFields
        config={{
          fields: [{ key: "profile_path", label: "Profile", type: "artifact-path", artifactKinds: ["bundle"] }],
        }}
        form={{ profile_path: "" }}
        artifacts={[selected]}
        selectedArtifact={selected}
        onChange={onChange}
        getArtifactPath={(artifact, fieldKey) => `${artifact.path}/${fieldKey}.npz`}
        getArtifactLabel={(artifact) => artifact.label ?? artifact.artifact_id}
      />,
    );

    await user.click(screen.getByRole("button", { name: /selected/i }));

    expect(onChange).toHaveBeenCalledWith("profile_path", "/tmp/bundle/profile_path.npz");
  });
});

function RecipeFieldsHarness({ onChange }: { onChange: (key: string, value: string | number | boolean) => void }) {
  const [form, setForm] = useState({ input_path: "", limit: 3 });
  return (
    <RecipeFields
      config={{
        fields: [
          { key: "input_path", label: "Input folder", type: "path", required: true },
          { key: "limit", label: "Limit", type: "number", min: 1, defaultValue: 3 },
        ],
      }}
      form={form}
      artifacts={[]}
      selectedArtifact={null}
      onChange={(key, value) => {
        setForm((current) => ({ ...current, [key]: value }));
        onChange(key, value);
      }}
      getArtifactPath={(artifact) => artifact.path}
      getArtifactLabel={(artifact) => artifact.label ?? artifact.artifact_id}
    />
  );
}

function artifact(kind: ArtifactRecord["kind"]): ArtifactRecord {
  return {
    artifact_id: `art_${kind}`,
    kind,
    path: `/tmp/${kind}`,
    file: null,
    audio: null,
    latent: null,
    source_artifact_ids: [],
    recipe_id: null,
    label: `${kind} artifact`,
    prompt: null,
    notes: null,
    tags: [],
    metadata: {},
    session_id: "sess_1",
    created_at: "2026-05-27T10:00:00.000Z",
  };
}
