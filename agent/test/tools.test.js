import assert from "node:assert/strict";
import test from "node:test";
import { createImageLookup } from "../src/protocol.js";
import { createTools, resultToToolResult, TOOL_NAMES } from "../src/tools.js";

test("selected tools retain canonical stable order", () => {
  const tools = createTools(
    ["compare_images", "get_workspace_state", "set_prompt"],
    { request() {} },
    createImageLookup({}),
  );
  assert.deepEqual(
    tools.map((tool) => tool.name),
    TOOL_NAMES.filter((name) => ["compare_images", "get_workspace_state", "set_prompt"].includes(name)),
  );
  assert.throws(
    () => createTools(["shell"], { request() {} }, createImageLookup({})),
    /Unknown tool name/,
  );
});

test("Python results become text plus referenced initial images", () => {
  const images = createImageLookup({ generated: { data: "YWJj", mimeType: "image/png" } });
  const result = resultToToolResult({ image_ids: ["generated"], seed: 42 }, images);
  assert.deepEqual(JSON.parse(result.content[0].text), { image_ids: ["generated"], seed: 42 });
  assert.deepEqual(result.content[1], { type: "image", data: "YWJj", mimeType: "image/png" });
});

test("Python can add generated images without exposing base64 in tool text or details", () => {
  const images = createImageLookup({});
  const result = resultToToolResult(
    {
      image_id: "new-frame",
      seed: 8,
      __krea2_images: {
        "new-frame": { data: "cGl4ZWxz", mimeType: "image/jpeg" },
      },
    },
    images,
  );
  assert.deepEqual(JSON.parse(result.content[0].text), { image_id: "new-frame", seed: 8 });
  assert.deepEqual(result.content[1], {
    type: "image",
    data: "cGl4ZWxz",
    mimeType: "image/jpeg",
  });
  assert.deepEqual(result.details, { image_id: "new-frame", seed: 8 });
  assert.equal(result.content[0].text.includes("cGl4ZWxz"), false);
});
