import { afterAll, afterEach, beforeAll, describe, expect, it } from "vitest";
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";

import { createApi } from "./api";

const server = setupServer(
  http.get("http://api.test/health", () =>
    HttpResponse.json({
      app: "sa3-native-lab",
      version: "test",
      artifact_root: "/tmp/lab",
      backends: [{ backend: "mlx", available: true, loaded: false, details: {} }],
    }),
  ),
  http.get("http://api.test/artifacts/art_bundle/inspect", () =>
    HttpResponse.json({
      artifact: {
        artifact_id: "art_bundle",
        kind: "bundle",
        path: "/tmp/art_bundle/bundle.zip",
        file: { filename: "bundle.zip", media_type: "application/zip", byte_size: 32 },
        source_artifact_ids: [],
        recipe_id: "recipe_1",
        tags: [],
        metadata: {},
        created_at: "2026-05-27T15:00:00.000Z",
      },
      recipe: null,
      sources: [],
      children: [],
      bundle_files: [{ path: "metrics.json", byte_size: 16, compressed_size: 12 }],
      bundle_preview: { result_count: 2 },
      bundle_summary: { kind: "sweep", file_count: 1 },
    }),
  ),
  http.get("http://api.test/readiness", () =>
    HttpResponse.json({
      ok: true,
      warnings: 0,
      errors: 0,
      checks: [{ name: "artifact-root", status: "ok", message: "/tmp/lab" }],
    }),
  ),
  http.post("http://api.test/recipes/recipe_1/fork", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      job_id: "job_fork",
      status: "queued",
      recipe: {
        recipe_id: "recipe_fork",
        operator: "latent.cyclic_roll",
        backend: payload.backend ?? "torch_mps",
        inputs: {},
        params: payload.params ?? {},
        seed: payload.seed ?? null,
        session_id: payload.session_id ?? null,
        created_at: "2026-05-27T15:00:00.000Z",
        version: 1,
      },
      progress: 0,
      artifact_ids: [],
      metrics: {},
      logs: [],
      created_at: "2026-05-27T15:00:00.000Z",
    });
  }),
  http.patch("http://api.test/sessions/sess_1", async ({ request }) => {
    const payload = (await request.json()) as Record<string, unknown>;
    return HttpResponse.json({
      session_id: "sess_1",
      name: payload.name ?? "session one",
      status: payload.status ?? "active",
      notes: payload.notes ?? null,
      metadata: {},
      created_at: "2026-05-27T15:00:00.000Z",
      updated_at: "2026-05-27T15:05:00.000Z",
      archived_at: payload.status === "archived" ? "2026-05-27T15:05:00.000Z" : null,
    });
  }),
);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe("createApi", () => {
  it("reads mocked API responses through MSW", async () => {
    await expect(createApi("http://api.test").health()).resolves.toMatchObject({
      artifact_root: "/tmp/lab",
      backends: [{ backend: "mlx", available: true }],
    });
  });

  it("inspects bundle file inventories", async () => {
    await expect(createApi("http://api.test").inspectArtifact("art_bundle")).resolves.toMatchObject({
      artifact: { artifact_id: "art_bundle" },
      bundle_files: [{ path: "metrics.json", byte_size: 16 }],
      bundle_preview: { result_count: 2 },
    });
  });

  it("reads readiness checks", async () => {
    await expect(createApi("http://api.test").readiness()).resolves.toMatchObject({
      ok: true,
      checks: [{ name: "artifact-root" }],
    });
  });

  it("forks recipes with edited params", async () => {
    await expect(
      createApi("http://api.test").forkRecipe("recipe_1", {
        backend: "torch_mps",
        params: { shift_frames: 4 },
        seed: 11,
      }),
    ).resolves.toMatchObject({
      job_id: "job_fork",
      recipe: { params: { shift_frames: 4 }, seed: 11 },
    });
  });

  it("updates session archive state", async () => {
    await expect(createApi("http://api.test").updateSession("sess_1", { status: "archived" })).resolves.toMatchObject({
      session_id: "sess_1",
      status: "archived",
      archived_at: "2026-05-27T15:05:00.000Z",
    });
  });
});
