# Evidence Standards

## Promotion Ladder

1. **Idea**: plausible relation to SA3/SAME math or architecture.
2. **Probe**: local primitive or notebook cell exposes it.
3. **Measurement**: descriptor, latent, flow, geometry, control, or memory rows
   show nontrivial movement.
4. **Audition**: decoded or polished audio is reviewed with notes.
5. **Repeatability**: behavior survives at least two clips, seeds, prompts, or
   dataset clusters.
6. **Decision**: promote, revise, drop, or keep as microscope only.

## Red Flags

- The method only makes a tensor change but no audible or measurable change.
- Descriptor gains are mostly clipping, noise, silence, loudness, or artifacts.
- The method increases nearest-memory/source similarity when novelty is claimed.
- A prompt/control result depends on one seed with no repeated probe.
- A sampler-internal method hides a version-sensitive SA3 assumption.
- A doc claims a control exists before measurement and listening.

## Keep As Microscope

Keep a method as a microscope when it reveals useful structure but does not yet
make reliable audio control. Examples: flow loss panels, null-condition probes,
residual feature maps, geometry reports, and diagnostic roll/periodicity tests.
