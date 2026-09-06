---
name: save-scene
description: Save or revise a reusable scene recipe when the user wants to keep composition, camera, motion, or an environment while changing the subject. Use art-style saving instead when only the medium or mark-making should transfer.
---

A scene recipe preserves how the scene is arranged. It can keep a vortex, a waterfall between islands, camera placement, focus, motion, and spatial relationships. These are allowed ingredients, not content that must be removed to make an art style.

Read the current prompt and inspect the image the user wants to capture. Separate what the user wants to repeat from what they want to replace. Refer to the replaceable character as “the subject”; avoid making a reusable composition depend on a specific character's hair, costume, or weapon unless requested. Keep named locations or objects when they define the requested recipe. Do not add art-medium language already supplied by the LoRA.

For example, a falling-vortex recipe can retain a camera alongside the subject, a close side view, a violet vortex, vertical motion streaks, and a sharp face against blurred surroundings. Removing the vortex would lose the requested recipe. A universal applicability test such as “does this work for a fisherman repairing nets?” is inappropriate here: a scene recipe deliberately asks future subjects to participate in its scene.

Use `list_styles` to check for an existing recipe and `inspect_style` before revising one. Save with `save_style(kind="scene", ...)` or change an existing entry with `update_style(kind="scene", ...)`. Keep the text concise, roughly 40–100 words when sufficient. Link the reference image and use the description to explain what the recipe repeats. Save only when requested, and edit the named entry in place rather than adding near-duplicates.

Confirm the saved name and show the wording. The recipe appears alongside art styles in the library and can be enabled by the user. An active recipe is automatically prefixed; do not duplicate it in the prompt. To use an inactive recipe, adapt its wording to the user's subject. Offer a cover if helpful, using the existing shared cover recipe only when the user requests it.
