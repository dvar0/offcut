---
name: compare-generation
description: Test a model, LoRA, prompt, or generation setting with controlled comparisons, including whether a style transfers across subjects. Use when the user asks to experiment or compare results.
---

Identify the question being tested before changing settings. Keep a baseline recipe and image ID. For a prompt or LoRA comparison, use the same exact seed, dimensions, route, and steps while changing the intended variable. For seed exploration, leave the prompt and other settings fixed and use `generate_image(fresh_seed=true)`.

Inspect active LoRA notes for prompting suggestions. Do not alter guidance on a distilled route, and do not inflate steps or adapter strength as a generic quality fix. A requested adapter-strength comparison is an experiment; a wide schema bound is not a quality recommendation.

Use the brief to retain the question and baseline when the experiment spans turns. Name each treatment in `change_note`, review the actual image, and compare candidates by ID. A reused frame is not another sample. A failure to render a detail is an observation; an explanation about training bias or randomness remains a hypothesis unless the comparison supports it.

For art-style transfer, distinguish paint/texture/edge behavior from the original subject's colors, scenery, and composition. When the user requests testing before saving, use contrasting subjects that expose leakage and keep trial wording separate from the saved library entry until a revision is requested. For scene recipes, preserve the requested composition rather than applying the art-style exclusion rule.

Stay within the requested variants. Report which image answers which treatment and the visible tradeoffs. Preserve the best candidate or restore the baseline when the user requests it; do not declare a universal model limitation from a few unsuccessful samples.
