import assert from "node:assert/strict";
import test from "node:test";
import { validateToolArguments } from "@earendil-works/pi-ai";
import { createTools, resultToToolResult, TOOL_NAMES } from "../src/tools.js";
import { createImageLookup } from "../src/protocol.js";
const tools = createTools(TOOL_NAMES, { request() {} }, createImageLookup({}));
const validate = (name, args) => validateToolArguments(tools.find((tool) => tool.name === name), { name, arguments: args });
test("exact seed strings and AUTO survive tool validation", () => {
  const args = { seed: "9007199254740993", steps: null, guidance: null };
  assert.deepEqual(validate("update_generation_settings", args), args);
  assert.deepEqual(validate("update_generation_settings", { seed: null }), { seed: null });
  for (const invalid of [{ width: 4096 }, { width: 240 }, { height: 64 }, { height: 1000 }, { steps: 101 }, { preset: "unknown" }, { loras: Array.from({ length: 5 }, () => ({ name: "lora" })) }]) {
    assert.throws(() => validate("update_generation_settings", invalid));
  }
});
test("prompt receipts keep one prompt and full UI details", () => {
  const prompt = "A spatial scene. ".repeat(100);
  const details = { prompt, settings: { prompt, width: 1024 }, prompt_change: { before: prompt, after: prompt, removed: "old", added: "new" } };
  const result = resultToToolResult(details, createImageLookup({}));
  const receipt = JSON.parse(result.content[0].text);
  assert.equal(receipt.prompt, prompt);
  assert.equal(receipt.generation_performed, false);
  assert.equal(receipt.settings.width, 1024);
  assert.equal(receipt.settings.prompt, undefined);
  assert.equal(result.details.prompt_change.after, prompt);
  assert.ok(result.content[0].text.length < JSON.stringify(details).length / 2);
});
test("Raw UI steps and full-pass setup have distinct tool arguments", () => {
  const args = { preset: "raw-int8-to-turbo", raw_start_steps: 9 };
  assert.deepEqual(validate("update_generation_settings", args), args);
  const setup = { raw_full_pass_steps: 60, turbo_full_pass_steps: 15 };
  assert.deepEqual(validate("update_generation_settings", setup), setup);
  const reset = { raw_start_steps: null, raw_full_pass_steps: null, turbo_full_pass_steps: null };
  assert.deepEqual(validate("update_generation_settings", reset), reset);
  // Pi coerces scalar types before validation (as it does for the existing numeric tools).
  // The Python boundary tests cover malformed raw requests; here test the adapter's bounds.
  for (const value of [100, "nine"]) {
    assert.throws(() => validate("update_generation_settings", { raw_start_steps: value }));
  }
  const properties = tools.find(tool => tool.name === "update_generation_settings").parameters.properties;
  assert.equal(properties.raw_steps, undefined);
  assert.equal(properties.raw_portion, undefined);
});
test("the step plan survives model receipt compaction without losing the no-generation receipt", () => {
  const plan = { raw_start_steps: 9, turbo_finish_steps: 9, label: "9 Raw steps → 9 Turbo steps" };
  const result = resultToToolResult({ generation_performed: false,
    settings: { raw_start_steps: 9, sampling_plan: plan, sampling_setup: { raw_full_pass_steps: 52, turbo_full_pass_steps: 12 } },
  }, createImageLookup({}));
  const receipt = JSON.parse(result.content[0].text);
  assert.equal(receipt.generation_performed, false);
  assert.deepEqual(receipt.settings.sampling_plan, plan);
});
test("generation receipts distinguish reused frames and retain transcript metadata", () => {
  const result = resultToToolResult({ image_id: "frame", attempt_id: 1, reused: true, generation_performed: false,
    image: { id: "frame", seed: "9223372036854775807", width: 1024, height: 1024, raw_prompt: "fox", final_prompt: "ink, fox" } }, createImageLookup({}));
  const receipt = JSON.parse(result.content[0].text);
  assert.equal(receipt.generation_performed, false);
  assert.equal(receipt.image.seed, "9223372036854775807");
  assert.equal(receipt.image.final_prompt, undefined);
  assert.equal(result.details.image.final_prompt, "ink, fox");
});
test("a crop attaches its declared preview rather than every known reference", () => {
  const lookup = createImageLookup({ original: { data: "b3JpZw==", mimeType: "image/jpeg" } });
  const result = resultToToolResult({ image: { id: "original" }, preview_id: "crop", pixels_supplied: ["crop"],
    __krea2_images: { crop: { data: "Y3JvcA==", mimeType: "image/jpeg" } } }, lookup);
  assert.equal(result.content.length, 2);
  assert.equal(result.content[1].data, "Y3JvcA==");
  assert.equal(lookup.byId.get("original").data, "b3JpZw==");
});
test("brief patches can clear approval separately from the best candidate", () => {
  const args = { approved_image_id: null, best_image_id: "candidate" };
  assert.deepEqual(validate("update_creative_brief", args), args);
  assert.equal(validate("save_style", { kind: "scene", name: "Vortex", description: "Camera follows", style_text: "the subject falling through a vortex" }).kind, "scene");
});
