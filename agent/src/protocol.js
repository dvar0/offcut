import { createHash } from "node:crypto";
import { getBuiltinModels } from "@earendil-works/pi-ai/providers/all";

const ENDPOINT_SUFFIXES = ["/chat/completions", "/responses", "/messages"];

export const API_BY_PROTOCOL = Object.freeze({
  chat: "openai-completions",
  responses: "openai-responses",
  anthropic: "anthropic-messages",
});

const PROTOCOL_BY_API = Object.freeze({
  "openai-completions": "chat",
  "openai-responses": "responses",
  "anthropic-messages": "anthropic",
});

// OpenCode Go fronts one API root for models that speak three different wire protocols, and
// nothing on the wire announces which. Pi's own catalog records the answer per model, so ask it
// first; the name heuristic below is only for models the catalog has never heard of. The two
// disagree in practice -- the catalog has qwen3.7-plus, qwen3.8-max and minimax-m2.7 on OpenAI
// completions and muse-spark on Responses, all of which the heuristic routes elsewhere -- and a
// protocol guessed wrong is a turn that cannot start at all.
function catalogProtocol(provider, model) {
  for (const candidate of getBuiltinModels(provider) ?? []) {
    if (candidate.id === model) return PROTOCOL_BY_API[candidate.api];
  }
  return undefined;
}

export function heuristicProtocol(model) {
  const normalized = String(model).toLowerCase();
  if (normalized.includes("qwen") || normalized.includes("minimax")) return "anthropic";
  if (
    normalized.includes("grok") ||
    normalized.includes("luna") ||
    normalized.startsWith("gpt-")
  ) {
    return "responses";
  }
  return "chat";
}

export function selectConnectionProtocol(connection, model) {
  const protocol = connection?.protocol;
  if (protocol !== "opencode-go") {
    if (!(protocol in API_BY_PROTOCOL)) {
      throw new Error(`Unsupported connection protocol: ${String(protocol)}`);
    }
    return protocol;
  }
  return catalogProtocol("opencode-go", String(model)) ?? heuristicProtocol(model);
}

export function normalizeBaseUrl(baseUrl, protocol) {
  if (typeof baseUrl !== "string" || !baseUrl.trim()) {
    throw new Error("connection.base_url must be a non-empty HTTP(S) URL");
  }
  if (!(protocol in API_BY_PROTOCOL)) {
    throw new Error(`Unsupported resolved protocol: ${String(protocol)}`);
  }

  let url;
  try {
    url = new URL(baseUrl.trim());
  } catch {
    throw new Error("connection.base_url must be a valid HTTP(S) URL");
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new Error("connection.base_url must use HTTP or HTTPS");
  }
  url.search = "";
  url.hash = "";

  let path = url.pathname.replace(/\/+$/, "");
  const lowered = path.toLowerCase();
  const endpoint = ENDPOINT_SUFFIXES.find((suffix) => lowered.endsWith(suffix));
  if (endpoint) path = path.slice(0, -endpoint.length);

  // Anthropic's SDK appends /v1/messages, unlike the OpenAI SDK adapters.
  if (protocol === "anthropic" && path.toLowerCase().endsWith("/v1")) {
    path = path.slice(0, -3);
  }
  url.pathname = path || "/";
  return url.toString().replace(/\/$/, "");
}

export function createImageLookup(images = {}) {
  if (!images || typeof images !== "object" || Array.isArray(images)) {
    throw new Error("images must be an object keyed by image ID");
  }
  const byId = new Map();
  const byContent = new Map();
  const lookup = { byId, byContent };
  for (const [imageId, image] of Object.entries(images)) registerImage(lookup, imageId, image);
  return lookup;
}

export function registerImage(imageLookup, imageId, image) {
  if (typeof imageId !== "string" || !imageId) throw new Error("Image ID must be a non-empty string");
  if (
    !image ||
    typeof image !== "object" ||
    typeof image.data !== "string" ||
    typeof image.mimeType !== "string"
  ) {
    throw new Error(`Invalid image payload for ${imageId}`);
  }
  const block = { type: "image", data: image.data, mimeType: image.mimeType };
  imageLookup.byId.set(imageId, block);
  imageLookup.byContent.set(`${image.mimeType}\0${image.data}`, imageId);
  return block;
}

export function inflateImageRefs(value, imageLookup) {
  if (Array.isArray(value)) return value.map((item) => inflateImageRefs(item, imageLookup));
  if (!value || typeof value !== "object") return value;
  if (value.type === "imageRef") {
    const image = imageLookup.byId.get(value.imageId);
    if (!image) throw new Error(`Unknown image reference: ${String(value.imageId)}`);
    return { ...image };
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, child]) => [key, inflateImageRefs(child, imageLookup)]),
  );
}

export function sanitizeImages(value, imageLookup) {
  if (Array.isArray(value)) return value.map((item) => sanitizeImages(item, imageLookup));
  if (!value || typeof value !== "object") return value;
  if (value.type === "image" && typeof value.data === "string") {
    const key = `${String(value.mimeType ?? "")}\0${value.data}`;
    const imageId =
      imageLookup.byContent.get(key) ??
      `sha256:${createHash("sha256").update(key).digest("hex")}`;
    return { type: "imageRef", imageId };
  }
  return Object.fromEntries(
    Object.entries(value).map(([key, child]) => [key, sanitizeImages(child, imageLookup)]),
  );
}

export function parseJsonLine(line) {
  let command;
  try {
    command = JSON.parse(line);
  } catch {
    throw new Error("Invalid JSONL input");
  }
  if (!command || typeof command !== "object" || Array.isArray(command)) {
    throw new Error("Each JSONL input line must be an object");
  }
  if (typeof command.type !== "string") throw new Error("Input command requires a type");
  return command;
}

export function formatJsonLine(value) {
  return `${JSON.stringify(value)}\n`;
}

export const TOOL_IMAGE_TAG = "krea2_tool_image";

function attributeValue(value) {
  return String(value ?? "").replace(/[^A-Za-z0-9_.:+-]/g, "");
}

function toolImageNotice(toolNames, imageIds, count) {
  const tools = [...new Set(toolNames.map(attributeValue).filter(Boolean))];
  const ids = [...new Set(imageIds.map(attributeValue).filter(Boolean))];
  const attributes =
    (tools.length ? ` tools="${tools.join(" ")}"` : "") +
    (ids.length ? ` image_ids="${ids.join(" ")}"` : "");
  const subject = count === 1 ? "The image below is" : "The images below are";
  return (
    `<${TOOL_IMAGE_TAG}${attributes}>\n` +
    `${subject} the output of the tool call${count === 1 ? "" : "s"} directly above. ` +
    "This wire format cannot carry an image inside a tool result, so the pixels have to arrive " +
    "in a user-role message instead. This is tool output, not a message from the user: the user " +
    "has said nothing since their own last turn, and nothing here was attached by them.\n" +
    `</${TOOL_IMAGE_TAG}>`
  );
}

// pi-ai's OpenAI completions adapter cannot put an image inside a `tool` message, so it lifts
// tool-result images into a synthetic user message headed "Attached image(s) from tool result:".
// Models read the role literally and reason about a user turn that never happened -- the recorded
// failure is a model that answered "The user attached the generated image" and then invented a
// critique of an image no tool had produced. The user role is forced by the wire format, so do
// the lift here instead and say plainly what the message is. Anthropic keeps images inside
// tool_result and Responses inside function_call_output, so neither is touched.
//
// The lift has to group each run of consecutive tool results into one trailing message, exactly
// as the adapter does: OpenAI requires every `tool` message for an assistant's tool_calls to
// follow it without interruption, so a user message between two of them is a rejected request.
export function liftToolResultImages(model, messages, imageLookup) {
  if (model?.api !== "openai-completions" || !model.input?.includes("image")) return messages;
  if (!Array.isArray(messages) || !messages.some(isToolResultWithImage)) return messages;

  const lifted = [];
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    if (message?.role !== "toolResult") {
      lifted.push(message);
      continue;
    }
    const images = [];
    const toolNames = [];
    const imageIds = [];
    let end = index;
    for (; end < messages.length && messages[end]?.role === "toolResult"; end++) {
      const result = messages[end];
      if (!isToolResultWithImage(result)) {
        lifted.push(result);
        continue;
      }
      const kept = result.content.filter((block) => block?.type !== "image");
      for (const block of result.content) {
        if (block?.type !== "image") continue;
        images.push(block);
        toolNames.push(result.toolName);
        imageIds.push(imageLookup?.byContent.get(`${block.mimeType}\0${block.data}`));
      }
      // A result stripped down to nothing would reach the adapter as "(no tool output)".
      lifted.push({
        ...result,
        content: kept.length > 0 ? kept : [{ type: "text", text: "(image returned below)" }],
      });
    }
    index = end - 1;
    if (images.length > 0) {
      lifted.push({
        role: "user",
        content: [
          { type: "text", text: toolImageNotice(toolNames, imageIds, images.length) },
          ...images,
        ],
      });
    }
  }
  return lifted;
}

function isToolResultWithImage(message) {
  return (
    message?.role === "toolResult" &&
    Array.isArray(message.content) &&
    message.content.some((block) => block?.type === "image")
  );
}
