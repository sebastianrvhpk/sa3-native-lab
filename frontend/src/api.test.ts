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
    }),
  ),
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
    });
  });
});
