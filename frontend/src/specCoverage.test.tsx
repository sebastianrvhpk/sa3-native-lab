import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SpecCoverage, SpecCoveragePair } from "./specCoverage";
import { testOperatorSpec } from "./test/fixtures";

describe("SpecCoverage", () => {
  it("reports covered and missing operator params in user-facing copy", () => {
    const spec = testOperatorSpec({
      params: { prompt: "str", duration_seconds: "float", seed: "int", metadata: "dict|null" },
    });

    const view = render(<SpecCoverage spec={spec} controlledKeys={["prompt", "duration_seconds", "seed"]} />);
    expect(screen.getByText("Spec covered")).toBeInTheDocument();
    expect(screen.getByText("4 params · mlx · implemented")).toBeInTheDocument();

    view.rerender(<SpecCoverage spec={spec} controlledKeys={["prompt"]} />);
    expect(screen.getByText("2 missing params")).toBeInTheDocument();
    expect(screen.getByText("duration_seconds, seed")).toBeInTheDocument();
  });

  it("keeps paired SAME specs pending until both contracts are available", () => {
    const encode = testOperatorSpec({
      name: "latent.encode",
      params: { model: "same-s|same-l", chunked: "bool" },
      produces: ["latent"],
    });
    const decode = testOperatorSpec({
      name: "latent.decode",
      params: { model: "same-s|same-l", notes: "str|null" },
      produces: ["audio"],
    });

    const view = render(<SpecCoveragePair specs={[encode, undefined]} controlledKeys={[["model", "chunked"], []]} />);
    expect(screen.getByText("Spec pending")).toBeInTheDocument();

    view.rerender(<SpecCoveragePair specs={[encode, decode]} controlledKeys={[["model", "chunked"], ["model"]]} />);
    expect(screen.getByText("1 missing params")).toBeInTheDocument();
    expect(screen.getByText("notes")).toBeInTheDocument();
  });
});
