import { Type } from "@earendil-works/pi-ai";
import { registerImage } from "./protocol.js";

export const TOOL_NAMES = Object.freeze([
  "get_workspace_state",
  "get_selected_image",
  "inspect_image",
  "restore_image_settings",
  "update_creative_brief",
  "review_attempt",
  "list_loras",
  "inspect_lora",
  "set_prompt",
  "edit_prompt",
  "update_generation_settings",
  "set_aspect_ratio",
  "generate_image",
  "compare_images",
  "list_styles",
  "inspect_style",
  "save_style",
  "update_style",
  "generate_cover",
  "load_skill",
]);

const imageId = Type.String({ minLength: 1 });
const notes = Type.Array(Type.String({ maxLength: 500 }), { maxItems: 16 });
const styleKind = Type.Union([Type.Literal("art"), Type.Literal("scene")], {
  description: "art describes medium and technique; scene preserves reusable camera, motion, environment, and composition around a replaceable subject.",
});
const definitions = Object.freeze({
  inspect_image: {
    label: "Inspect image",
    description: "Look at an image by ID, including earlier references. An optional normalized crop magnifies a detail without changing the source. Check pixels_supplied; metadata is not visual evidence.",
    parameters: Type.Object({
      image_id: imageId,
      crop: Type.Optional(Type.Object({
        x: Type.Number({ minimum: 0, maximum: 1 }), y: Type.Number({ minimum: 0, maximum: 1 }),
        width: Type.Number({ exclusiveMinimum: 0, maximum: 1 }), height: Type.Number({ exclusiveMinimum: 0, maximum: 1 }),
      }, { additionalProperties: false })),
    }, { additionalProperties: false }),
  },
  restore_image_settings: {
    label: "Restore image recipe",
    description: "Restore a generated image's exact seed, rendered prompt, dimensions, route, steps, guidance and LoRAs to this board. Historical style text is baked into the prompt and style toggles/enhancement are cleared to avoid doubling it. Does not generate. Use when asked to return to a previous result; a changed prompt can still change its composition.",
    parameters: Type.Object({ image_id: imageId }, { additionalProperties: false }),
  },
  update_creative_brief: {
    label: "Update creative brief",
    description: "Keep durable notes for multi-step work. Patch only changed fields; arrays replace previous arrays. Preserve accepted constraints and reference purposes. best_image_id is your candidate; approved_image_id is only an image the user actually approved. Null clears either. Latest user corrections override old notes.",
    parameters: Type.Object({
      goal: Type.Optional(Type.String({ maxLength: 2000 })),
      must_keep: Type.Optional(notes), accepted_tradeoffs: Type.Optional(notes), failed_approaches: Type.Optional(notes),
      next_change: Type.Optional(Type.String({ maxLength: 2000 })),
      best_image_id: Type.Optional(Type.Union([imageId, Type.Null()])),
      approved_image_id: Type.Optional(Type.Union([imageId, Type.Null()])),
      references: Type.Optional(Type.Array(Type.Object({ image_id: imageId, purpose: Type.String({ minLength: 1, maxLength: 200 }) }, { additionalProperties: false }), { maxItems: 8 })),
    }, { additionalProperties: false }),
  },
  review_attempt: {
    label: "Review attempt",
    description: "Record an actual visual observation against an attempt returned by generate_image. Check the requested correction and accepted details for regression; state uncertainty. This does not mark the result approved by the user.",
    parameters: Type.Object({
      attempt_id: Type.Integer({ minimum: 1 }), observation: Type.String({ minLength: 1, maxLength: 2000 }),
      outcome: Type.Union([Type.Literal("improved"), Type.Literal("regressed"), Type.Literal("mixed"), Type.Literal("unclear")]),
    }, { additionalProperties: false }),
  },
  get_workspace_state: {
    label: "Get workspace state",
    description: "Read the current prompt, generation settings, active LoRAs, board, and selection. Raw → Turbo includes the same executed-step plan shown in the UI, plus the separate full-pass sampling setup.",
    parameters: Type.Object({}, { additionalProperties: false }),
  },
  get_selected_image: {
    label: "Get selected image",
    description: "Read metadata and the image reference for the currently selected image.",
    parameters: Type.Object({}, { additionalProperties: false }),
  },
  list_loras: {
    label: "List LoRAs",
    description:
      "List available LoRAs, their current activation state, and a one-line summary of each. Summaries and prompting notes are user-provided; inspect_lora returns the full notes when available.",
    parameters: Type.Object({}, { additionalProperties: false }),
  },
  inspect_lora: {
    label: "Inspect LoRA",
    description:
      "Inspect one LoRA's trigger phrase, default strength, summary, and optional user-provided prompting notes. Notes may describe caption patterns, examples, useful strengths, or limitations; treat them as suggestions, not verified training facts or a required format.",
    parameters: Type.Object(
      { name: Type.String({ minLength: 1, description: "LoRA name or identifier" }) },
      { additionalProperties: false },
    ),
  },
  set_prompt: {
    label: "Set prompt",
    description: "Replace the workspace image prompt exactly.",
    parameters: Type.Object({ prompt: Type.String({ maxLength: 20000 }) }, { additionalProperties: false }),
  },
  edit_prompt: {
    label: "Edit prompt",
    description: "Replace one exact occurrence of old_text in the current prompt with new_text.",
    parameters: Type.Object(
      {
        old_text: Type.String({ minLength: 1 }),
        new_text: Type.String(),
      },
      { additionalProperties: false },
    ),
  },
  update_generation_settings: {
    label: "Update generation settings",
    description:
      "Update generation settings without generating. For the UI's Raw steps control, use raw_start_steps: the server calculates the handoff and reports the resulting Raw → Turbo step plan. Route changes reset steps and guidance to AUTO unless supplied, and clear an unusable negative. Turbo is fast; more Raw can change composition at a speed cost, not guarantee better quality or adherence. Preserve the current full-pass settings (defaults Raw 52 / Turbo 12) during ordinary iteration; change them for deliberate sampling experiments.",
    parameters: Type.Object(
      {
        preset: Type.Optional(Type.Union([Type.Literal("raw-int8-to-turbo"), Type.Literal("raw-int8"), Type.Literal("raw-int8-turbo-lora")])),
        width: Type.Optional(Type.Integer({ minimum: 256, maximum: 2048, multipleOf: 16 })),
        height: Type.Optional(Type.Integer({ minimum: 256, maximum: 2048, multipleOf: 16 })),
        steps: Type.Optional(Type.Union([Type.Integer({ minimum: 1, maximum: 100 }), Type.Null()], { description: "Total steps for Turbo or Raw; null restores AUTO (Turbo 8, Raw 52). For Raw → Turbo use raw_start_steps to change the visible Raw steps control, or turbo_full_pass_steps for an advanced sampling experiment." })),
        raw_start_steps: Type.Optional(Type.Union([Type.Integer({ minimum: 1, maximum: 99 }), Type.Null()], { description: "Raw → Turbo only: actual Raw steps executed before Turbo takes over, matching the UI stepper. For 'try 9 Raw steps', set this to 9. Must be less than raw_full_pass_steps (default 52, so normally 1–51). Turbo's executed count is calculated automatically. null restores the default 8% opening (4 Raw steps at the default setup)." })),
        raw_full_pass_steps: Type.Optional(Type.Union([Type.Integer({ minimum: 2, maximum: 100 }), Type.Null()], { description: "Advanced Sampling setup for Raw → Turbo. A complete Raw pass would use this count; default 52. This is NOT the UI's Raw steps. Changing it preserves the chosen Raw count where it fits. null restores 52. Leave unchanged during ordinary iteration." })),
        turbo_full_pass_steps: Type.Optional(Type.Union([Type.Integer({ minimum: 2, maximum: 100 }), Type.Null()], { description: "Advanced Sampling setup for Raw → Turbo. A complete Turbo pass would use this count; default 12. Only the suffix after the Raw handoff executes. null restores 12. Leave unchanged during ordinary iteration." })),
        guidance: Type.Optional(
          Type.Union([Type.Null(), Type.Number({
            minimum: 0,
            maximum: 20,
            description:
              "Raw guidance; CFG equals this value + 1. Raw defaults to 3.5, Raw → Turbo to 3.0 (CFG 4), affecting only its raw stage. Turbo (raw-int8-turbo-lora) is fixed at guidance 0; raising it fails and degrades output. The hybrid turbo stage always uses CFG 1. null restores AUTO.",
          })]),
        ),
        seed: Type.Optional(Type.Union([Type.String({ pattern: "^[0-9]+$", description: "Exact decimal seed; copy seed_text. Range 0 through 9223372036854775807." }), Type.Integer({ minimum: 0, maximum: Number.MAX_SAFE_INTEGER }), Type.Null()], { description: "Use a decimal string for an exact seed. null selects a fresh random seed per run." })),
        negative_prompt: Type.Optional(
          Type.String({
            description:
              "Used by Raw and only the raw opening of Raw → Turbo. Turbo (raw-int8-turbo-lora) ignores it, so setting one there fails.",
          }),
        ),
        loras: Type.Optional(
          Type.Array(
            Type.Object(
              {
                name: Type.String({ minLength: 1 }),
                // Matches offcut_cli.LORA_STRENGTH_LIMIT. A narrower bound here would let the
                // agent pass its own schema and then fail server validation.
                strength: Type.Optional(
                  Type.Number({
                    minimum: -100,
                    maximum: 100,
                    description:
                      "Around 1.0 is normal, and most LoRAs degrade well before 2.0. The range is wide only so the user can experiment; do not go above 1.0 unless asked, and check inspect_lora for the LoRA's own recommended range.",
                  }),
                ),
              },
              { additionalProperties: false },
            ),
            { maxItems: 4 },
          ),
        ),
      },
      { additionalProperties: false },
    ),
  },
  set_aspect_ratio: {
    label: "Set aspect ratio",
    description: "Change the workspace aspect ratio, preserving its pixel area as closely as the 2048px limit and 16px snapping allow.",
    parameters: Type.Object(
      { aspect_ratio: Type.String({ minLength: 3, description: "Width:height ratio" }) },
      { additionalProperties: false },
    ),
  },
  generate_image: {
    label: "Generate image",
    description: "Generate from the current text prompt and settings, without a second enhancer rewrite. References are not sampler image inputs. Set fresh_seed=true to deliberately explore a new sample. Read reused and generation_performed before claiming a new image was rendered. Inspect the result and record important observations with review_attempt.",
    parameters: Type.Object({
      fresh_seed: Type.Optional(Type.Boolean()),
      change_note: Type.Optional(Type.String({ maxLength: 500, description: "The requested correction or experimental variable being tested; keep it short." })),
    }, { additionalProperties: false }),
  },
  list_styles: {
    label: "List styles",
    description:
      "List the saved styles in the library with their descriptions, and which are active on this board.",
    parameters: Type.Object({}, { additionalProperties: false }),
  },
  inspect_style: {
    label: "Inspect style",
    description:
      "Read one saved style's full wording and its reference image. Use it before writing a style into a prompt so the wording can be adapted to the shot.",
    parameters: Type.Object(
      { style_id: Type.String({ minLength: 1, description: "Style ID from list_styles" }) },
      { additionalProperties: false },
    ),
  },
  save_style: {
    label: "Save style",
    description:
      "Save a reusable art style or scene recipe. For kind=art load save-style; for kind=scene load save-scene. Scene recipes can retain camera, setting and motion with a replaceable subject. Save only when requested.",
    parameters: Type.Object(
      {
        kind: Type.Optional(styleKind),
        name: Type.String({ minLength: 1, maxLength: 80, description: "Short human name, two to four words" }),
        description: Type.String({
          maxLength: 500,
          description: "One sentence on when to reach for this style",
        }),
        style_text: Type.String({
          minLength: 1,
          maxLength: 2000,
          description:
            "Reusable paragraph. For art: medium, technique, texture, palette, light without the reference subject. For scene: preserve the requested camera, setting, motion and composition; refer to the replaceable character as the subject.",
        }),
        reference_image_id: Type.Optional(
          Type.Union([Type.String({ minLength: 1 }), Type.Null()], {
            description: "The image the style was read from",
          }),
        ),
      },
      { additionalProperties: false },
    ),
  },
  update_style: {
    label: "Edit style",
    description:
      "Edit one saved style in place, supplying only the fields that change. Use it when the user asks to change a style they already have instead of saving a near-duplicate. Only edit a style when the user asks.",
    parameters: Type.Object(
      {
        style_id: Type.String({ minLength: 1, description: "Style ID from list_styles or inspect_style" }),
        kind: Type.Optional(styleKind),
        name: Type.Optional(
          Type.String({ minLength: 1, maxLength: 80, description: "Short human name, two to four words" }),
        ),
        description: Type.Optional(
          Type.String({
            maxLength: 500,
            description: "One sentence on when to reach for this style",
          }),
        ),
        style_text: Type.Optional(
          Type.String({
            minLength: 1,
            maxLength: 2000,
            description: "The rewritten style paragraph in full, not a diff",
          }),
        ),
        reference_image_id: Type.Optional(
          Type.Union([Type.String({ minLength: 1 }), Type.Null()], {
            description: "The image the style was read from, or null to clear it",
          }),
        ),
      },
      { additionalProperties: false },
    ),
  },
  generate_cover: {
    label: "Generate cover",
    description:
      "Render the cover frame for one library entry and attach it. Every cover uses the same prompt, seed and size from the user's shared cover recipe, so the covers differ only by the style or LoRA under test; none of it comes from the current board, and the board's own settings are left untouched. Offer this after saving a new style rather than running it unasked, and do not replace an existing cover unless the user asks.",
    parameters: Type.Object(
      {
        target_kind: Type.Union([Type.Literal("style"), Type.Literal("lora")], {
          description: "Whether the cover is for a saved style or for a LoRA",
        }),
        target: Type.String({
          minLength: 1,
          description: "Style ID from list_styles, or LoRA name from list_loras",
        }),
      },
      { additionalProperties: false },
    ),
  },
  load_skill: {
    label: "Load skill",
    description:
      "Load the full instructions for one of the skills listed in available_skills. Call it when a request matches a skill's description, and follow what it returns before acting.",
    parameters: Type.Object(
      { name: Type.String({ minLength: 1, description: "Skill name from available_skills" }) },
      { additionalProperties: false },
    ),
  },
  compare_images: {
    label: "Compare images",
    description: "Compare two or more workspace images by their image IDs.",
    parameters: Type.Object(
      {
        image_ids: Type.Array(Type.String({ minLength: 1 }), {
          minItems: 2,
          maxItems: 8,
        }),
      },
      { additionalProperties: false },
    ),
  },
});

function collectKnownImageIds(value, imageLookup, found = new Set()) {
  if (typeof value === "string" && imageLookup.byId.has(value)) found.add(value);
  else if (Array.isArray(value)) {
    for (const item of value) collectKnownImageIds(item, imageLookup, found);
  } else if (value && typeof value === "object") {
    for (const item of Object.values(value)) collectKnownImageIds(item, imageLookup, found);
  }
  return found;
}

export function compactToolResult(result) {
  if (!result || typeof result !== "object" || Array.isArray(result)) return result;
  if (result.prompt_change) {
    const { prompt_change, settings, ...rest } = result;
    const { prompt, ...otherSettings } = settings ?? {};
    return { ...rest, prompt: result.prompt ?? prompt, settings: otherSettings,
      generation_performed: false, note: "Prompt updated. No image was generated." };
  }
  if (result.attempt_id && result.image) {
    const { raw_prompt, enhanced_prompt, final_prompt, ...image } = result.image;
    return { ...result, image, note: result.reused ? "Existing frame reused; this is not a fresh sample. Use fresh_seed=true to explore." : "A real frame was generated. Inspect its pixels before judging it; the prompt is not evidence of what rendered." };
  }
  return result;
}

export function resultToToolResult(result, imageLookup) {
  let publicResult = result;
  if (result && typeof result === "object" && !Array.isArray(result) && result.__krea2_images) {
    const { __krea2_images: images, ...rest } = result;
    if (!images || typeof images !== "object" || Array.isArray(images)) {
      throw new Error("Tool image envelope must be an object keyed by image ID");
    }
    for (const [imageId, image] of Object.entries(images)) registerImage(imageLookup, imageId, image);
    publicResult = rest;
  }
  const modelResult = compactToolResult(publicResult);
  const text = typeof modelResult === "string" ? modelResult : JSON.stringify(modelResult ?? null);
  const content = [{ type: "text", text }];
  for (const imageId of (Array.isArray(publicResult?.pixels_supplied) ? publicResult.pixels_supplied : collectKnownImageIds(publicResult, imageLookup))) {
    content.push({ ...imageLookup.byId.get(imageId) });
  }
  return { content, details: publicResult ?? null };
}

export function createTools(toolNames, requestBroker, imageLookup) {
  if (!Array.isArray(toolNames)) throw new Error("toolNames must be an array");
  const requested = new Set(toolNames);
  for (const name of requested) {
    if (!TOOL_NAMES.includes(name)) throw new Error(`Unknown tool name: ${String(name)}`);
  }

  return TOOL_NAMES.filter((name) => requested.has(name)).map((name) => ({
    name,
    ...definitions[name],
    execute: async (toolCallId, args, signal, onUpdate) => {
      const result = await requestBroker.request(
        { toolCallId, name, args },
        signal,
        (update) => onUpdate?.(resultToToolResult(update, imageLookup)),
      );
      return resultToToolResult(result, imageLookup);
    },
  }));
}
