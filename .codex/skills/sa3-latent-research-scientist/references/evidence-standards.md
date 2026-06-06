# Evidence Standards

## Claim Maturity Ladder

1. **Idea**: plausible relation to SA3/SAME math or architecture.
2. **Microscope**: exposes structure in a native object, but makes no control
   claim.
3. **Selector**: ranks prompts, seeds, donors, channels, clusters, or recipes.
4. **Intervention candidate**: changes a native object and shows nontrivial
   descriptor, latent, flow, memory, or audition movement.
5. **Repeatability**: behavior survives at least two clips, seeds, prompts, or
   dataset clusters.
6. **Promoted method**: has rationale, notebook-facing use, measurements,
   listening notes, and a clear promote/revise/drop decision.

## Red Flags

- The method only makes a tensor change but no audible or measurable change.
- Descriptor gains are mostly clipping, noise, silence, loudness, or artifacts.
- The method increases nearest-memory/source similarity when novelty is claimed.
- A prompt/control result depends on one seed with no repeated probe.
- A sampler-internal method hides a version-sensitive SA3 assumption.
- A doc claims a control exists before measurement and listening.
- A root primitive quietly calls upstream SA3/SAME when it should be a
  `procedures/` helper.
- An adapter starts making research decisions instead of isolating external
  runtime access.
- A selector is described as a control without decoded evidence.
- A notebook cell produces impressive audio but no baseline or artifact rows.

## Keep As Microscope

Keep a method as a microscope when it reveals useful structure but does not yet
make reliable audio control. Examples: flow loss panels, null-condition probes,
residual feature maps, geometry reports, and diagnostic roll/periodicity tests.

## Minimum Evidence Packet

```text
Object:
Transition:
Operation:
Workbench:
Maturity before:
Maturity after:
Altitude:
Baseline:
Method output:
Measurements:
Artifact paths:
Descriptor/latent/flow rows:
Listening notes:
Repeatability check:
Decision:
Next action:
```
