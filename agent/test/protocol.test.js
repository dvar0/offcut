import assert from "node:assert/strict";
import test from "node:test";
import {
  createImageLookup,
  formatJsonLine,
  inflateImageRefs,
  parseJsonLine,
  sanitizeImages,
} from "../src/protocol.js";

test("JSONL helpers parse objects and emit exactly one line", () => {
  assert.deepEqual(parseJsonLine('{"type":"abort"}'), { type: "abort" });
  assert.equal(formatJsonLine({ type: "done" }), '{"type":"done"}\n');
  assert.throws(() => parseJsonLine("not json"), /Invalid JSONL/);
  assert.throws(() => parseJsonLine("[]"), /must be an object/);
});

test("image references inflate for Pi and sanitize without base64", () => {
  const images = createImageLookup({ hero: { data: "c2VjcmV0", mimeType: "image/png" } });
  const inflated = inflateImageRefs(
    { role: "user", content: [{ type: "imageRef", imageId: "hero" }] },
    images,
  );
  assert.deepEqual(inflated.content[0], {
    type: "image",
    data: "c2VjcmV0",
    mimeType: "image/png",
  });
  const event = sanitizeImages({ type: "event", message: inflated }, images);
  assert.deepEqual(event.message.content[0], { type: "imageRef", imageId: "hero" });
  assert.equal(JSON.stringify(event).includes("c2VjcmV0"), false);
});

test("unknown streamed images get deterministic opaque references", () => {
  const images = createImageLookup({});
  const first = sanitizeImages({ type: "image", data: "abc", mimeType: "image/webp" }, images);
  const second = sanitizeImages({ type: "image", data: "abc", mimeType: "image/webp" }, images);
  assert.match(first.imageId, /^sha256:[a-f0-9]{64}$/);
  assert.deepEqual(first, second);
});
