import assert from "node:assert/strict";
import test from "node:test";
import { convertMessages } from "@earendil-works/pi-ai/api/openai-completions";
import { buildModel } from "../src/model.js";
import { createImageLookup, liftToolResultImages, TOOL_IMAGE_TAG } from "../src/protocol.js";

const CONNECTION = { protocol: "opencode-go", base_url: "https://opencode.ai/zen/go/v1" };
const PIXELS = { data: "AAAA", mimeType: "image/jpeg" };

function lookup() {
  return createImageLookup({ "image-1": PIXELS, "image-2": { data: "BBBB", mimeType: "image/jpeg" } });
}

function toolResult(toolCallId, toolName, text, images = []) {
  return {
    role: "toolResult",
    toolCallId,
    toolName,
    isError: false,
    content: [{ type: "text", text }, ...images.map((image) => ({ type: "image", ...image }))],
  };
}

function conversation(model, results) {
  return [
    { role: "user", content: "make a ps2 render" },
    {
      role: "assistant",
      api: model.api,
      provider: model.provider,
      model: model.id,
      content: results.map((result) => ({
        type: "toolCall",
        id: result.toolCallId,
        name: result.toolName,
        arguments: {},
      })),
    },
    ...results,
  ];
}

test("a tool-result image reaches OpenAI completions labelled as tool output", () => {
  const model = buildModel(CONNECTION, "glm-5.3-flash");
  const messages = conversation(model, [
    toolResult("call_1", "generate_image", '{"image":{"id":"image-1"}}', [PIXELS]),
  ]);
  const context = { systemPrompt: "SYS", tools: [], messages: liftToolResultImages(model, messages, lookup()) };
  const params = convertMessages(model, context, model.compat ?? {}, {});

  const [tool, carrier] = params.slice(-2);
  assert.equal(tool.role, "tool");
  assert.equal(tool.tool_call_id, "call_1");
  assert.equal(tool.content, '{"image":{"id":"image-1"}}');

  // The role is forced by the wire format; the text is what has to deny the user turn.
  assert.equal(carrier.role, "user");
  const [notice, image] = carrier.content;
  assert.match(notice.text, new RegExp(`^<${TOOL_IMAGE_TAG} tools="generate_image" image_ids="image-1">`));
  assert.match(notice.text, /not a message from the user/);
  assert.equal(image.image_url.url, "data:image/jpeg;base64,AAAA");

  // pi-ai's own hoist must not have fired: it would label the pixels far more weakly.
  assert.equal(JSON.stringify(params).includes("Attached image(s) from tool result:"), false);
});

test("consecutive tool results share one carrier so no user message splits them", () => {
  const model = buildModel(CONNECTION, "glm-5.3-flash");
  const messages = conversation(model, [
    toolResult("call_1", "generate_image", '{"image":{"id":"image-1"}}', [PIXELS]),
    toolResult("call_2", "compare_images", '{"ok":true}', [{ data: "BBBB", mimeType: "image/jpeg" }]),
  ]);
  const params = convertMessages(
    model,
    { systemPrompt: "SYS", tools: [], messages: liftToolResultImages(model, messages, lookup()) },
    model.compat ?? {},
    {},
  );

  const roles = params.map((message) => message.role);
  assert.deepEqual(roles.slice(-3), ["tool", "tool", "user"]);
  const carrier = params.at(-1);
  assert.match(carrier.content[0].text, /tools="generate_image compare_images"/);
  assert.match(carrier.content[0].text, /image_ids="image-1 image-2"/);
  assert.equal(carrier.content.filter((block) => block.type === "image_url").length, 2);
});

test("a text-free tool result keeps a tool message the API will accept", () => {
  const model = buildModel(CONNECTION, "glm-5.3-flash");
  const result = { role: "toolResult", toolCallId: "call_1", toolName: "generate_image", isError: false, content: [{ type: "image", ...PIXELS }] };
  const [lifted] = liftToolResultImages(model, [result], lookup());
  assert.deepEqual(lifted.content, [{ type: "text", text: "(image returned below)" }]);
});

test("protocols that carry images natively are left untouched", () => {
  const messages = [toolResult("call_1", "generate_image", "{}", [PIXELS])];
  for (const modelId of ["grok-4.6", "qwen3.8-flash"]) {
    const model = buildModel(CONNECTION, modelId);
    assert.notEqual(model.api, "openai-completions");
    assert.equal(liftToolResultImages(model, messages, lookup()), messages);
  }
});

test("a text-only model is left to the adapter rather than handed pixels", () => {
  const model = buildModel(CONNECTION, "deepseek-v4-pro");
  assert.equal(model.api, "openai-completions");
  assert.deepEqual(model.input, ["text"]);
  const messages = [toolResult("call_1", "generate_image", "{}", [PIXELS])];
  assert.equal(liftToolResultImages(model, messages, lookup()), messages);
});

test("a conversation with no tool-result image is returned unchanged", () => {
  const model = buildModel(CONNECTION, "glm-5.3-flash");
  const messages = [{ role: "user", content: "hello" }, toolResult("call_1", "set_prompt", "{}")];
  assert.equal(liftToolResultImages(model, messages, lookup()), messages);
});
