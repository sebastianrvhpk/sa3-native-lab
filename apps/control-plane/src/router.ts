import { initTRPC } from "@trpc/server";
import { z } from "zod";

import type { ArtifactAnnotationPayload } from "./pythonClient.js";
import { createPythonClient, type PythonClientOptions } from "./pythonClient.js";
import { loadWorkbenchState, workbenchLoadInputSchema } from "./workbench.js";

export interface ControlPlaneContext extends PythonClientOptions {}

const t = initTRPC.context<ControlPlaneContext>().create();

const archiveSearchInputSchema = z.object({
  query: z.string().optional().default(""),
  tags: z.array(z.string()).optional().default([]),
  sessionId: z.string().nullable().optional(),
});

const annotateInputSchema = z.object({
  artifactId: z.string().min(1),
  label: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
  tags: z.array(z.string()).nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
});

export const appRouter = t.router({
  workbench: t.router({
    load: t.procedure
      .input(workbenchLoadInputSchema.optional().default({}))
      .query(({ ctx, input }) => {
        const { apiBase: baseUrl = ctx.baseUrl, ...workbenchInput } = input;
        return loadWorkbenchState(createPythonClient({ ...ctx, baseUrl }), workbenchInput);
      }),
  }),
  archive: t.router({
    search: t.procedure.input(archiveSearchInputSchema).query(({ ctx, input }) =>
      createPythonClient(ctx).artifacts({
        query: input.query,
        tags: input.tags,
        sessionId: input.sessionId,
      }),
    ),
    annotateAndSearch: t.procedure.input(annotateInputSchema).mutation(async ({ ctx, input }) => {
      const client = createPythonClient(ctx);
      const payload: ArtifactAnnotationPayload = {
        label: input.label,
        notes: input.notes,
        tags: input.tags,
        metadata: input.metadata,
      };
      const artifact = await client.annotateArtifact(input.artifactId, payload);
      const matches = await client.artifacts({ query: artifact.label ?? "", tags: artifact.tags });
      return { artifact, matches };
    }),
  }),
});

export type AppRouter = typeof appRouter;
