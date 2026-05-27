# Creative AI Stack Decision Matrix

## UI Architecture

| Need | Default | Use when | Avoid when |
| --- | --- | --- | --- |
| App shell | React Router | Multiple pages, loaders, shareable routes | Single embedded widget |
| Local state | Zustand | Session, canvas, player, selection state | Data is remote and cacheable |
| Remote state | TanStack Query | Runs, artifacts, job status, server metadata | Pure local transforms |
| Accessible controls | React Aria | Sliders, menus, dialogs, tabs, forms | Custom control is truly novel |
| Graph canvas | @xyflow/react | Real nodes/edges, lineage, routing, pipelines | Decorative network visuals |
| Motion | Framer Motion | State transition clarity | Decorative motion only |
| 3D | React Three Fiber | Spatial latent maps or model-space navigation | Flat UI explains it better |

## Backend Architecture

| Need | Candidate | Notes |
| --- | --- | --- |
| Typed RPC | tRPC | Strong TS end-to-end fit |
| Lightweight API | Hono | Good for Workers and simple HTTP boundaries |
| Validation | Zod | Use at API and persistence boundaries |
| Relational data | Postgres/Supabase/Drizzle | Runs, artifacts, users, annotations |
| Local-first prototype | JSON/files/sqlite | Good before collaboration or cloud |
| Edge state | KV/D1/Durable Objects | Use only for real deployment needs |
| Realtime/collab | Durable Objects or Supabase realtime | Add after single-user workflow works |

## Prototype Stages

1. Static fixtures and typed domain model.
2. Local mock API.
3. Real artifact browser.
4. Job execution boundary.
5. Persistence and annotations.
6. Collaboration or cloud deployment.

Do not skip stages unless the repo already has the required infrastructure.
