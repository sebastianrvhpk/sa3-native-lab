# Codex Skills

This repo vendors the project-specific Codex skills under:

```text
.codex/skills/
```

They are included so another machine can restore the same research/design workflow.

## Install On Another Machine

From the repo root:

```bash
mkdir -p ~/.codex/skills
cp -R .codex/skills/* ~/.codex/skills/
```

Then start a new Codex session. The skills should be available by name, for example:

```text
Use $research-instrument-interface-director
Use $visual-reference-digestion
Use $creative-ai-stack-prototyper
```

## Included Skills

- `codebase-capability-cartographer`
- `latent-audio-interface-architect`
- `creative-ai-stack-prototyper`
- `interaction-triage-loop`
- `portfolio-research-demo-hardener`
- `research-instrument-interface-director`
- `visual-reference-digestion`

The recommended restart flow is:

```text
$codebase-capability-cartographer
$visual-reference-digestion
$research-instrument-interface-director
$interaction-triage-loop
$creative-ai-stack-prototyper
$portfolio-research-demo-hardener
```
