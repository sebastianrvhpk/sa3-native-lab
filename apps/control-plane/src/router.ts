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

const jobIdInputSchema = z.object({
  jobId: z.string().min(1),
});

const recipeIdInputSchema = z.object({
  recipeId: z.string().min(1),
});

const recipeForkInputSchema = recipeIdInputSchema.extend({
  inputs: z.record(z.string(), z.string()).nullable().optional(),
  params: z.record(z.string(), z.unknown()).nullable().optional(),
  backend: z.enum(["mlx", "torch_mps", "torch_cpu", "cpu"]).nullable().optional(),
  model: z.string().nullable().optional(),
  seed: z.number().int().nullable().optional(),
  notes: z.string().nullable().optional(),
  session_id: z.string().nullable().optional(),
});

const annotateInputSchema = z.object({
  artifactId: z.string().min(1),
  label: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
  tags: z.array(z.string()).nullable().optional(),
  metadata: z.record(z.string(), z.unknown()).nullable().optional(),
});

const artifactIdInputSchema = z.object({
  artifactId: z.string().min(1),
});

export const appRouter = t.router({
  system: t.router({
    readiness: t.procedure.query(({ ctx }) => createPythonClient(ctx).readiness()),
  }),
  workbench: t.router({
    load: t.procedure
      .input(workbenchLoadInputSchema.optional().default({}))
      .query(({ ctx, input }) => {
        const { apiBase: baseUrl = ctx.baseUrl, ...workbenchInput } = input;
        return loadWorkbenchState(createPythonClient({ ...ctx, baseUrl }), workbenchInput);
      }),
  }),
  jobs: t.router({
    list: t.procedure.query(({ ctx }) => createPythonClient(ctx).jobs()),
    get: t.procedure.input(jobIdInputSchema).query(({ ctx, input }) => createPythonClient(ctx).job(input.jobId)),
    cancel: t.procedure.input(jobIdInputSchema).mutation(({ ctx, input }) => createPythonClient(ctx).cancelJob(input.jobId)),
    retry: t.procedure.input(jobIdInputSchema).mutation(({ ctx, input }) => createPythonClient(ctx).retryJob(input.jobId)),
  }),
  recipes: t.router({
    replay: t.procedure.input(recipeIdInputSchema).mutation(({ ctx, input }) => createPythonClient(ctx).replayRecipe(input.recipeId)),
    fork: t.procedure.input(recipeForkInputSchema).mutation(({ ctx, input }) => {
      const { recipeId, ...payload } = input;
      return createPythonClient(ctx).forkRecipe(recipeId, payload);
    }),
  }),
  artifacts: t.router({
    inspect: t.procedure.input(artifactIdInputSchema).query(({ ctx, input }) => createPythonClient(ctx).inspectArtifact(input.artifactId)),
  }),
  families: t.router({
    load: t.procedure
      .input(workbenchLoadInputSchema.optional().default({}))
      .query(async ({ ctx, input }) => {
        const { apiBase: baseUrl = ctx.baseUrl, ...workbenchInput } = input;
        const state = await loadWorkbenchState(createPythonClient({ ...ctx, baseUrl }), workbenchInput);
        return {
          resultFamilies: state.resultFamilies,
          sessionResultFamilies: state.sessionResultFamilies,
          archiveResultFamilies: state.archiveResultFamilies,
        };
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
