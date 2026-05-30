# Product Rescue Brief

This brief is the product reset for SA3 Native Lab. It intentionally overrides
the previous dashboard/parity gravity in the interface while preserving the
useful backend work: local execution, typed recipes, durable artifacts, job
events, sessions, and replayable model state.

## North Star

SA3 Native Lab should feel like a local AI sound instrument:

```text
start from a sound, prompt, latent, or remembered take
  -> choose one expressive gesture
  -> hear a new take
  -> keep playing, branch, remember, or tune deeper
```

The primary product question is:

```text
What do I want to do with this sound next?
```

The app is not a Colab parity dashboard, benchmark lab, operator catalog, job
queue, artifact database, or A/B evaluator. Those concepts may remain in the
implementation, but they should not be the first product language.

## Product Diagnosis

The current app has a reasonably healthy code split and useful local runtime
contracts, but the visible interface is still organized around engineering
objects:

- artifacts
- jobs
- operators
- result families
- specs
- recipes
- bundle inspectors
- A/B slots
- scorer choices
- backend readiness

That makes the app feel like a control room for experiments instead of an
instrument for creative exploration. The backend can keep its precision. The
front of the app needs a different grammar.

## New Object Model

Use these words in product UI:

| Product word | Meaning | Implementation backing |
| --- | --- | --- |
| Sound | The current audible material or selected latent/audio object | `ArtifactRecord` |
| Take | A generated or imported playable result | Audio artifact plus recipe/job |
| Gesture | A creative operation the user performs on the current sound | Generation, SAME, latent operator, script recipe |
| Source | Material used to start or continue a gesture | Source artifact, prompt, latent, bundle path |
| Anchor | A pinned sound used as a stable reference or starting point | Stored artifact id, formerly compare slot idea |
| Memory | Saved material worth recovering later | Archived/session artifact plus labels/tags |
| Branch | A cluster of related takes from one creative move | Result family/recipe group |
| Tuning | Advanced parameters for a gesture | Backend `ui_fields`, form state |

Avoid these words in primary UI:

- artifact
- job
- operator
- result family
- spec coverage
- scorer
- A/B
- benchmark
- best candidate

These may appear in advanced/developer drawers when they are useful for trust,
debugging, or reproducibility.

## Central Sound Surface

The first screen should be organized around one dominant surface:

1. Current sound: waveform/player/latent field with clear selected material.
2. Gesture wheel or gesture strip: Generate, Continue, Vary, Steer, Borrow,
   Encode, Decode, Morph, Remember.
3. Take strip: recent output takes attached to the current session and source.
4. Memory shelf: pinned/remembered material, hidden until needed.
5. Tuning drawer: parameters for the selected gesture.
6. Inspect drawer: lineage, recipe, file, logs, backend details, and raw bundle
   structure.

The sound should be visually and interactively larger than every control panel.
Progress should appear on the sound/take that is being made, not primarily as a
job card elsewhere.

## First 60 Seconds

The first minute should work like this:

1. The user lands on an empty sound surface with three clear starts:
   `Prompt`, `Import`, `Open memory`.
2. If they type a prompt, the primary action is `Make sound`.
3. While the model runs, a pending take appears in the take strip with phase,
   elapsed time, cancel, and any real percent.
4. When finished, the take becomes the current sound and starts as playable
   material.
5. The UI asks what to do next through gestures:
   `Vary`, `Continue`, `Steer`, `Borrow texture`, `Remember`, `Tune`.
6. Advanced model/runtime details are available, but not required to continue.

If imported audio is the start:

1. The audio becomes the current sound.
2. The first suggestions are `Continue`, `Vary`, `Encode`, `Remember`.
3. Encoding is described as preparing the sound for latent gestures, not as a
   backend mode.

## Primary Gestures

| Gesture | User intent | Implementation paths |
| --- | --- | --- |
| Make | Create a new sound from prompt or source | `generate.text_to_audio`, audio-to-audio, inpaint |
| Continue | Extend or reinterpret current audio | audio-to-audio, inpaint, future continuation affordance |
| Vary | Produce nearby takes without changing intent | seed, init noise, alpha sweep, latent perturbations |
| Steer | Push the sound toward a direction | style vectors, residual vectors, profile/direction bundles |
| Borrow texture | Use another latent/sound as donor material | latent graft, latent DSP donor, memory hit reuse |
| Encode | Turn audio into latent material | SAME encode |
| Decode | Hear latent material | SAME decode |
| Morph | Move through latent/time/style space | cyclic roll, blur, DSP, future region controls |
| Remember | Save material for later use | labels, tags, archive/memory shelf |
| Tune | Reveal advanced parameters for the active gesture | `ui_fields`, recipe forms |
| Inspect | Reveal provenance and diagnostics | lineage, recipe, logs, bundle inspector |

## Progress Model

Current product problem:

- Jobs are visible as a separate system object.
- The user has to understand that a job may eventually create an artifact.
- A busy job rail makes the app feel administrative.

Rescue model:

- A queued/running job is displayed as a pending take.
- Pending takes live beside the sound they will become.
- Job logs and retry/cancel live inside the pending take or an inspect drawer.
- The global run monitor becomes a compact safety strip, not a main panel.

Implementation can keep `JobRecord`; UI should say `Making take`,
`Encoding latent`, `Decoding sound`, `Steering`, `Borrowing texture`, or
another gesture-specific phrase.

## Advanced Controls

Advanced controls stay available through progressive disclosure:

- `Tune`: duration, seed, model, backend, steps, guidance, init noise,
  chunking, alpha lists, masks, paths, search settings.
- `Inspect`: IDs, files, recipe JSON, logs, specs, bundle contents.
- `Readiness`: backend/HF/MLX/SAME status, moved to settings or a small status
  popover.

Advanced does not mean hidden forever. It means the user chooses to tune after
understanding the gesture.

## Concepts To Remove Or Reframe

| Current concept | Decision | Replacement |
| --- | --- | --- |
| CLAP | Delete from product language | No replacement until a compatible, useful workflow exists |
| A/B comparison | Delete as primary model | Anchors, pinned sources, memory, take strip |
| Result family | Reframe | Branch |
| Artifact | Reframe in product UI | Sound, latent, bundle, material |
| Job | Reframe in product UI | Pending take / gesture in progress |
| Operator Studio | Reframe | Gestures / latent moves |
| Recipe Studio | Reframe | Advanced gestures / experiments |
| Spec coverage | Move to dev/readiness | Contract health drawer |
| Scorer | Reframe or hide | Prompt probe / search method in Tune |
| Best candidate | Remove evaluative framing | Interesting take / remembered take / branch highlight |
| Archive | Reframe | Memory shelf |

## Element-By-Element Audit

| Current visible element | Decision | Target behavior |
| --- | --- | --- |
| Brand/header | Keep, simplify | Brand plus compact local status; API field moves to settings |
| Backend pills | Move behind readiness | Status dot/popover unless setup is broken |
| API input | Move behind settings | Useful for development, harmful in first contact |
| Source rail | Reframe | `Start` / `Sources` / `Memory`, with import as a primary start action |
| Artifact stack | Rename/reframe | Material shelf with sounds, latents, remembered takes |
| Listening Bench heading | Keep/reword | `Current sound` or selected sound name |
| Specimen waveform/player | Keep and enlarge | Central sound surface with transport, loop, markers, decisions |
| Specimen ID/recipe/shape vitals | Progressive disclosure | Move to Inspect drawer |
| A/B buttons on specimen | Delete/reframe | Pin as anchor/source, or remove until anchor workflow is real |
| Replay/fork recipe buttons | Reframe | `Do again` / `Branch` inside Inspect or gesture context |
| Artifact annotation panel | Keep/reframe | `Remember this sound`: name, notes, tags, decision |
| Run Monitor | Demote | Pending take strip and compact global activity |
| Generate band | Reframe as gesture | `Make` / `Continue` with Tune drawer |
| SAME band | Reframe as gesture | `Encode` / `Decode`, explain as prepare/hear latent |
| Operator Studio | Reframe | `Latent gestures`: Roll, Blur, Graft, Texture, Renoise |
| Operator presets | Keep later | Gesture presets inside Tune, not first-screen admin |
| Recipe Studio | Move behind experiments | Advanced gestures only after first playable loop |
| Mode Atlas | Move to dev/parity | Useful for migration, harmful in product UI |
| Result rail | Reframe | `Takes` and `Branches`, not result family administration |
| Readiness panel | Move | Setup/status popover, not right rail default |
| ResultFamilyPanel | Reframe | Branch strip grouped by creative move |
| FamilyDetailPanel | Reframe | Branch detail for takes, not metrics-first |
| SessionTray | Reframe | Memory/session shelf; keep new session and recall |
| ComparePanel | Delete/reframe | Anchors or pinned sources, no A/B language |
| AuditionStackPanel | Keep/reframe | Take strip / playlist; it should be central, not buried |
| Mini counts | Delete from primary | Debug/status drawer only |
| Bundle inspector | Keep but hide by default | Inspect drawer and typed reuse actions |
| Prompt search rack | Hide under gesture tuning | Prompt exploration, not scorer dashboard |
| SpecCoverage widgets | Move to dev drawer | Not product UI |

## New Information Architecture

```text
App
  Header
    Brand
    Local status
    Settings

  Instrument
    Current Sound Surface
      Player / waveform / latent preview
      Pending take overlay
      Remember controls
      Inspect drawer

    Gesture Strip
      Make
      Continue
      Vary
      Steer
      Borrow texture
      Encode
      Decode
      Morph

    Gesture Tuning Drawer
      Selected gesture controls
      Advanced params
      Backend/model settings

    Takes
      Recent takes
      Pending takes
      Branches
      Do again / branch / remember

    Memory
      Anchors
      Sources
      Remembered takes
      Archive recovery

  Developer / Readiness
    API base
    backend readiness
    spec coverage
    raw jobs
    mode atlas
    raw artifact metadata
```

## Backend Complexity That Disappears

The backend can keep precise nouns, but the UI should translate them:

| Backend | Product UI |
| --- | --- |
| `ArtifactRecord` | Sound, latent, material, take |
| `JobRecord` | Pending take / gesture progress |
| `Recipe` | Gesture settings / branch recipe |
| `OperatorName` | Gesture id |
| `ResultFamily` | Branch |
| `ui_fields` | Tune controls |
| Bundle files | Inspectable material / reusable source |
| `/readiness` | Local status |
| `/operators/specs` | Contract health / tuning metadata |

## P0/P1/P2 Implementation Queue

### P0: Product Language And Main Surface

1. Remove CLAP from frontend product notes and docs. Leave only a generic
   unsupported runtime guard if needed.
2. Remove A/B language from primary UI. Replace the visible compare surface
   with a neutral `Anchors` or `Pinned sources` model, or remove it until the
   anchor workflow is useful.
3. Rename primary product surfaces:
   `Listening Bench` -> `Current sound`,
   `Result Family` -> `Branches`,
   `Audition` -> `Takes`,
   `Archive` -> `Memory`,
   `Operator Studio` -> `Latent gestures`,
   `Recipe Studio` -> `Experiments` or `Advanced gestures`.
4. Demote spec coverage, readiness, mode atlas, raw job cards, and API settings
   behind progressive disclosure.
5. Promote the take strip and current sound surface so they dominate first
   contact.
6. Attach progress to pending takes, with global run monitor reduced to a
   compact emergency/status strip.

### P1: Gesture Workflow

1. Build a `GestureStrip` component that selects the active creative act.
2. Build a `GestureTuningDrawer` that renders the existing schema-driven
   controls for the selected gesture.
3. Group generation/SAME/operator/experiment payload builders behind gesture
   view models without hiding backend parameter truth.
4. Convert result families into branch/take language across tests and docs.
5. Convert archive/session interactions into memory/shelf language.
6. Add browser smoke for the first 60 seconds: prompt -> pending take -> landed
   sound -> vary/remember/tune.

### P2: Instrument Feel

1. Add tasteful motion only for causal transitions: pending take created,
   take landed, sound remembered, branch opened.
2. Refine visual motifs so lines connect real source -> gesture -> take
   relationships.
3. Add a component lab only after the new product components exist.
4. Revisit deeper memory/vector search once memory has real product semantics.

## Migration Plan

### Current status after first-screen rescue

As of the first interface rescue implementation pass, slices 1 and 2 are
partially implemented in the app:

- Primary visible language now centers on Current Sound, Sources, Takes /
  Branches, Session Memory, Latent Gestures, Advanced Gestures, and Anchors.
- API base, backend readiness, contract/spec coverage, material counts, mode
  atlas, and raw metadata are moved behind Settings, Inspect, or developer
  disclosure affordances.
- The current sound/player and take strip are promoted above raw branch/admin
  panels.
- CLAP and A/B are not product-facing concepts; remaining mentions in this
  document describe the audit history and migration rationale.

Still open:

- A real gesture strip/tuning drawer should replace the remaining form-heavy
  action wall.
- Running jobs should become pending takes inside the take/branch flow instead
  of relying on a compact global status strip.
- Memory and anchors need deeper reuse actions, not just better names.
- The visual motif should keep getting closer to source -> gesture -> take ->
  memory relationships instead of decorative dashboard framing.

### Current status after gesture model pass

The app now has frontend product-domain models for `Gesture` and `PendingTake`.
The primary action area is no longer four simultaneous panels for generation,
SAME, latent operators, and script experiments. The first screen exposes a
single gesture strip:

```text
Make / Continue / Vary / Steer / Borrow Texture / Encode / Decode / Morph / Remember
```

Selecting a gesture reveals a scoped Tune surface backed by the existing
schema-driven form and payload builders. Raw contract/spec and mode atlas
details remain reachable through Inspect rather than being first-order product
controls. Job records are translated into pending/failed take cards in the
Takes / Branches flow, while the compact global Take Status remains only a
safety/status strip.

Still open:

- Gesture selection is now real, but deeper gesture chaining is not: a finished
  take should suggest the next useful gestures from its kind, source, and
  lineage.
- Memory is still mostly archive/recovery plus anchors. It needs reuse actions
  such as use as source, donor, prompt seed, or branch context where the
  backend already supports those paths.
- Tune is scoped to one gesture, but some gestures still expose dense
  parameter sets inherited from backend/script forms.
- Pending takes currently translate jobs in the UI; the tRPC/workbench control
  plane should eventually return pending-take-shaped state directly.

### Current status after priority-queue product loop

The core loop is now implemented in the frontend product layer:

```text
Current Sound -> Gesture -> Pending Take -> Listen -> Branch / Remember / Tune
```

New domain helpers translate backend records into instrument concepts:

- `memoryModel.ts`: remembered material, role/reuse intent, source mapping,
  available reuse actions, and disabled reasons.
- `nextActionModel.ts`: context-aware actions for selected audio, latent,
  bundle, pending/failed take, and branch state.
- `pendingTakeLandingModel.ts`: landing artifact, completion phrase, branch
  label, recovery suggestion, and next-gesture hints for pending take cards.
- `branchModel.ts`: branch summaries and inspect-only technical rows for
  former result-family groups.
- `tuneFieldGroups.ts`: primary, advanced, and inspect-only Tune grouping over
  existing backend-derived field configs.

The remaining open items above are partially resolved: landed takes now suggest
next gestures, Memory can be reused as source/anchor/donor/prompt seed or
recovered, branch UI is creative-path language first, and Tune exposes fewer
primary fields for Make, Continue/Vary, Encode/Decode, Morph, and Steer. The
unfinished work is deeper orchestration extraction, richer listening review,
memory browsing/filtering, and bundle-specific promotion after the action
semantics stabilize.

The product-health browser gate is now
`npm run smoke:first-use --prefix frontend`. It verifies the first-use path,
Memory reuse, recovery, Branch, Remember, Settings/Inspect demotion, desktop and
mobile screenshots, and mobile overflow.

### Slice 1: Language correction

- Remove CLAP from product docs/frontend notes.
- Replace A/B text/buttons with anchors or remove visible compare panel.
- Update tests that encode the old evaluation language.

### Slice 2: First-screen hierarchy

- Change labels and ordering so current sound, takes, and gestures are primary.
- Move readiness/spec/API/dev material into details/settings.
- Keep behavior equivalent while improving product meaning.

### Slice 3: Gesture strip

- Extract a gesture model from existing modes.
- Let each gesture choose the right existing form/payload path.
- Keep advanced params in a tuning drawer.

### Slice 4: Pending takes

- Render running jobs as pending takes in the take strip.
- Keep cancel/retry/logs available from each pending take.
- Reduce the global job monitor.

### Slice 5: Memory and anchors

- Replace archive/compare mental model with memory/anchor/source actions.
- Let remembered material be reused as source, donor, prompt seed, or branch
  context where the backend already supports it.

### Slice 6: Visual refinement

- Rebalance layout around the central sound surface.
- Keep the reference vibe tactile and playful, but make every strong motif
  correspond to a real interaction or state.

## Definition Of Done For The Rescue

- A new user can make or import a sound and understand the next action within
  sixty seconds.
- The main UI speaks in sound/take/gesture/memory language.
- Raw artifacts, jobs, specs, operators, and backend details are still
  available but not dominant.
- CLAP is not present in product language.
- A/B is not a primary concept.
- Progress is attached to takes.
- Advanced parameters are reachable through Tune.
- Tests protect the new product language and first-use flow.
