---
name: save-style
description: Save, capture, or revise a reusable art style in the library by separating medium and mark-making from the reference subject. For reusable camera, motion, environment, or scene composition, use save-scene.
---

# Saving a style

A saved style is a reusable description of **how an image was made**, not of **what it shows**. It
is stored once and later prefixed onto prompts about completely different subjects, so every word
in it has to survive that move. "A white Moorish palace among cypress trees" is a subject. "1920s
travel-poster lithography with stippled ink texture and a peach-and-cobalt palette" is a style.

These exclusions apply to `kind="art"`. If the user asks to retain a particular composition,
environment, or motion as a recipe, load `save-scene` and save `kind="scene"` instead. Do not
remove the very scene the user asked to preserve to fit an art-style rule.

## Read the image for these

Work through the axes below and keep only the ones the image actually commits to. A style that
names four things confidently beats one that names ten things vaguely.

- **Medium and process.** Oil on canvas, gouache, screen print, lithograph, cel animation, 35mm
  photograph, CGI render, pencil on toned paper. Include the process artifacts that come with it:
  plate registration, halftone dots, canvas tooth, film grain, chromatic aberration.
- **Mark-making and edge quality.** Visible brushwork or invisible blending; hard graphic edges or
  soft atmospheric ones; hatching, stippling, dry-brush scumbling, hard cel shading.
- **Texture and surface.** Woven paper, ink bleed, grain, noise, scan artifacts, varnish crackle.
- **Palette behavior.** Not just which colors, but how they act: limited two-tone duotone, muted
  earth tones with one saturated accent, high-key pastels, crushed blacks with warm highlights.
- **Lighting logic.** Flat and even, single hard key, golden-hour rim, luminous atmospheric haze,
  strong chiaroscuro.
- **Era or movement**, when the image genuinely reads as one: Art Nouveau, Bauhaus, Ukiyo-e,
  Soviet constructivist, 1970s airbrush sci-fi, Y2K chrome.
- **Compositional habits** that recur in the style rather than in this one picture: deep flattened
  perspective, decorative border framing, extreme negative space, isometric staging.

## Exclude

- The subject, characters, animals, or named people.
- Props, objects, wardrobe, architecture, and vegetation.
- The setting, location, weather, or time of day, unless the light itself is the style.
- This image's one-off composition: who stands where, what the camera happens to be pointing at.
- Any text visible in the image.

## Write it

- One paragraph, roughly 25 to 60 words. Comma-separated phrases read better here than full
  sentences.
- Lead with medium and era, then technique and texture, then palette and light.
- Name real artists, studios, or movements only when the resemblance is genuine and specific.
- Do not open with "a", "an", or "image of" — the text is prefixed onto another prompt, so it has
  to join cleanly.

**Test it before saving:** read your style text followed by an unrelated prompt, such as *"a lone
fisherman mending nets on a harbour wall"*. If the result still describes a coherent picture, the
style is clean. If it fights the new subject or smuggles in the old one, cut whatever leaked.

## Then call save_style

- `name` — short and human, two to four words, how the user would ask for it later ("Deco Travel
  Poster", "Muted Risograph"). Names are unique and case-insensitive.
- `description` — one sentence on when to reach for this style. This is what you and the user read
  when choosing between styles later, so make it discriminating, not decorative.
- `style_text` — the paragraph you just wrote.
- `reference_image_id` — the image you read the style from, whenever there is one.

Confirm the saved name back to the user and quote the style text so they can correct it.

## Editing a saved style

When the user asks to change a style they already have, edit it in place with `update_style` instead
of saving a near-duplicate — three slightly different "Muted Risograph" entries is a library nobody
can choose from. Only edit when asked: never rewrite a saved style on your own initiative to make a
prompt fit, and never edit a style the user did not point you at.

- Call `inspect_style` first, so the edit starts from what is actually saved rather than your memory
  of saving it.
- Pass only the fields that change; omitted fields keep their value. A changed `style_text` is the
  whole rewritten paragraph, not a diff.
- The rewritten text still answers to every rule above, including the subject-leak test.
- A board with the style active picks up the new wording at its next generation; which boards have it
  active is the user's control and does not change.

After editing, confirm what changed and quote the resulting style text.

## Giving a style a cover

Every library entry can carry one cover frame, rendered from a single recipe the user configures in
settings: one prompt, one seed, one size, shared by every entry. That is the whole point of it. A
cover generated any other way is just a picture; a cover generated through the recipe sits beside
the others in the grid and shows what this style does that the one next to it does not.

`generate_cover` renders it and attaches it. Nothing about it comes from the current board, and the
board's prompt, size and LoRA stack are left exactly as they were.

- After saving a new style, offer a cover in one line and call `generate_cover` when the user agrees.
  Do not generate one unasked — it takes the GPU, and the board may be mid-thought.
- Do not replace a cover the entry already has unless the user asks for a new one.
- Do not offer to change the recipe's prompt or seed to flatter a particular style. A recipe that
  varies per entry stops being a comparison, which is the one thing it is for.
