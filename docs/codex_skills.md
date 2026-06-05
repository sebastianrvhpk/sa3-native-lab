# Codex Skills

This repo vendors the project-specific Codex skills under:

```text
.codex/skills/
```

They are included so another machine can restore the same notebook research
workflow.

## Install On Another Machine

From the repo root:

```bash
mkdir -p ~/.codex/skills
cp -R .codex/skills/* ~/.codex/skills/
```

Then start a new Codex session. The skills should be available by name, for example:

```text
Use $sa3-notebook-steward
Use $sa3-same-primitive-researcher
Use $sa3-research-map-curator
```

## Included Skills

- `sa3-notebook-steward`: edit and smoke-check the expanded Colab notebook.
- `sa3-same-primitive-researcher`: implement notebook-native SA3/SAME latent-audio helpers.
- `sa3-research-map-curator`: keep current research docs and maps aligned.

The usual workflow is:

```text
$sa3-research-map-curator
$sa3-same-primitive-researcher
$sa3-notebook-steward
```
