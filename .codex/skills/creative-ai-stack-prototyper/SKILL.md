---
name: creative-ai-stack-prototyper
description: Prototype and implement AI-native creative tools with a modern TypeScript stack after an interaction grammar is accepted. Use when building or refactoring web interfaces using React 19, TypeScript, React Router, Zustand, TanStack Query, React Aria, Tailwind, Framer Motion, React Flow/@xyflow, React Three Fiber, tRPC or Hono-style APIs, Cloudflare or Supabase or Drizzle patterns, Zod validation, and research-demo quality testing. Do not use as the first step for ambiguous creative/research UI; first use an interface-direction skill to avoid generic dashboards.
---

# Creative AI Stack Prototyper

## Overview

Use this skill after the code capability map, interface architecture, and interaction grammar are grounded. It turns an accepted research-instrument concept into a maintainable TypeScript prototype without pretending the prototype is a finished product.

Prefer a thin, working vertical slice over a broad visual mock that is disconnected from data and artifacts.

Do not start implementation from schemas and API endpoints alone. If the request lacks a central surface, screen-state model, and visual semantics, stop and ask to run `research-instrument-interface-director` first.

## Stack Defaults

Use these defaults unless the repo already has stronger conventions:

- Frontend: React 19, TypeScript, Vite, React Router v7.
- State: Zustand for local session, canvas, and interaction state.
- Server state: TanStack Query for fetched artifacts, jobs, runs, and metadata.
- Controls: React Aria Components for accessible sliders, buttons, tabs, dialogs, menus, and forms.
- Styling: Tailwind CSS with explicit design tokens; avoid one-off inline visual systems.
- Motion: Framer Motion for state changes that clarify causality, not decoration.
- Graph/canvas: `@xyflow/react` only when nodes or edges represent real graph, routing, lineage, or workflow state.
- 3D/spatial: React Three Fiber only when spatial latent exploration or audio visualization genuinely needs it.
- API: tRPC or Hono-style endpoints with Zod schemas at every boundary.
- Persistence: Drizzle, Supabase/Postgres, D1, KV, or Durable Objects only when the workflow actually needs saved state, collaboration, queues, or realtime coordination.

## Implementation Workflow

0. Confirm the design brief.
   - Instrument thesis.
   - Central surface.
   - Interaction grammar.
   - Screen-state model.
   - Visual semantics.
   - Explicit anti-dashboard constraints.
   - If these are missing, do not implement UI yet.

1. Define the domain model.
   - Types for artifacts, runs, recipes, operators, audio assets, latents, metrics, notes, and lineage.
   - Zod schemas for persisted or API-crossing data.
   - Use mock data shaped like real artifacts before wiring heavy models.

2. Build a vertical slice around the central surface.
   - One source artifact.
   - One operator family.
   - One result artifact.
   - One comparison/listening or inspection surface.
   - One metadata/recipe inspector.
   - One repeatable execution path, even if backed by a mock API.

3. Add interaction depth incrementally.
   - Start with deterministic state and reproducible fixtures.
   - Add async jobs, generated artifacts, and persistence after the UI proves useful.
   - Introduce graph/canvas interactions only when they reduce complexity.

4. Keep the backend honest.
   - Begin mock-first or local-file-first when model execution is expensive.
   - Add typed endpoints for artifact listing, run creation, run status, artifact read, annotation write, and export.
   - Keep model runtime concerns behind service boundaries so UI iteration remains fast.

5. Verify like a research tool.
   - Typecheck, lint, test, and build.
   - Open the UI and inspect desktop and mobile layouts.
   - Verify no text overlaps, controls map to real parameters, audio paths load, and metadata is visible.
   - Add small tests for schema transforms and operator UI state.

## Anti-Patterns

- Building a node editor before proving there is a graph.
- Building a DAW timeline when the task is artifact exploration.
- Building a dashboard because the data model has many resource types.
- Making raw files, JSON sidecars, readiness cards, or settings forms the first object of attention.
- Adding reference-inspired colors, glyphs, texture, or dots after the layout is already generic.
- Hiding provenance behind pretty cards.
- Using animation to compensate for unclear state.
- Persisting everything before the research loop is understood.
- Showing controls that do not map to code, math, or executable plans.

## Pre-Implementation Gate

Before writing app code, verify these are true:

- The first viewport has one primary research action.
- The interface does not use a default admin dashboard layout unless explicitly justified.
- Infrastructure state is visible but not dominant.
- Generated artifacts are grouped by meaningful families, not raw file type alone.
- Empty states teach the next action.
- Every visible control maps to a real parameter, artifact, or accepted future capability.

Load `references/stack-decision-matrix.md` when choosing between UI architecture options.
