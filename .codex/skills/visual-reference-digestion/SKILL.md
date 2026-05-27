---
name: visual-reference-digestion
description: Translate visual references, moodboards, screenshots, style text, and design examples into functional interface rules before UI implementation. Use when Codex receives reference images or aesthetic descriptions for creative/research software and must digest them into layout principles, interaction metaphors, visual semantics, anti-patterns, and acceptance criteria instead of copying decoration.
---

# Visual Reference Digestion

## Overview

Use this skill when visual references are provided. The goal is to extract operational lessons from references, not to skin an existing dashboard.

References must become functional rules: what objects exist, how they relate, what gestures they imply, and what state changes should be visible.

## Workflow

1. Separate literal features from transferable principles.
   - Literal: color, texture, line style, shape.
   - Transferable: density, grouping, flow, rhythm, hierarchy, reveal, tactility, annotation, signal routing.

2. Map each motif to a system function.
   - If a motif has no system function, mark it decorative and defer it.
   - If it could mislead, forbid it.

3. Produce visual semantics.
   - Color must encode role, status, family, or selection.
   - Shape must encode object kind or operation family.
   - Lines must encode real provenance, lineage, routing, or dependency.
   - Texture must encode research notebook/lab surface only if it stays subtle.

4. Produce layout rules.
   - Decide what the first viewport should teach.
   - Decide the main focal object.
   - Decide what recedes into drawers/details.
   - Decide when density helps and when it becomes clutter.

5. Produce interaction rules.
   - What can be dragged, selected, sent, pinned, forked, compared, annotated, or expanded?
   - What transitions clarify cause and effect?
   - What empty states invite the next useful action?

6. Produce anti-copy rules.
   - State exactly what should not be copied.
   - State what would make the result look like a bad imitation.

## Required Output

- Reference reading.
- Motif-to-function table.
- Visual semantics.
- Layout rules.
- Interaction rules.
- Anti-copy rules.
- Implementation acceptance criteria.

Use `references/reference-digestion-template.md` when a strict structure is useful.
