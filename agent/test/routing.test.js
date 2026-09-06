import assert from "node:assert/strict";
import test from "node:test";
import { buildModel, describeModel, inferModelCapabilities } from "../src/model.js";
import { heuristicProtocol, normalizeBaseUrl, selectConnectionProtocol } from "../src/protocol.js";

test("OpenCode Go selection asks the catalog before guessing from the name", () => {
  const connection = { protocol: "opencode-go" };
  // The name heuristic would route all four of these by their vendor prefix alone.
  assert.equal(selectConnectionProtocol(connection, "qwen3.8-max"), "chat");
  assert.equal(selectConnectionProtocol(connection, "minimax-m2.7"), "chat");
  assert.equal(selectConnectionProtocol(connection, "qwen3.8-flash"), "anthropic");
  assert.equal(selectConnectionProtocol(connection, "muse-spark-1.2-contributor"), "responses");
  assert.equal(selectConnectionProtocol(connection, "glm-5.3-flash"), "chat");
});

test("models the catalog has never heard of still fall back to the name heuristic", () => {
  const connection = { protocol: "opencode-go" };
  for (const [model, expected] of [
    ["qwen3-coder", "anthropic"],
    ["MiniMax-M2", "anthropic"],
    ["gpt-luna", "responses"],
    ["grok-4", "responses"],
    ["glm-5.2-unreleased", "chat"],
  ]) {
    assert.equal(selectConnectionProtocol(connection, model), expected);
    assert.equal(heuristicProtocol(model), expected);
  }
});

test("saved API roots normalize for each SDK", () => {
  assert.equal(normalizeBaseUrl("https://host.example/api/v1/", "chat"), "https://host.example/api/v1");
  assert.equal(
    normalizeBaseUrl("https://host.example/api/v1/chat/completions", "chat"),
    "https://host.example/api/v1",
  );
  assert.equal(
    normalizeBaseUrl("https://host.example/api/v1/responses", "responses"),
    "https://host.example/api/v1",
  );
  assert.equal(
    normalizeBaseUrl("https://host.example/api/v1/messages", "anthropic"),
    "https://host.example/api",
  );
  assert.equal(normalizeBaseUrl("https://api.anthropic.com", "anthropic"), "https://api.anthropic.com");
});

test("model construction reuses catalog capabilities and overrides endpoint", () => {
  const model = buildModel(
    { protocol: "opencode-go", base_url: "https://custom.example/root/v1" },
    "gpt-5.6-luna",
  );
  assert.equal(model.api, "openai-responses");
  assert.equal(model.provider, "opencode-go");
  assert.equal(model.baseUrl, "https://custom.example/root/v1");
  assert.deepEqual(model.input, ["text", "image"]);
  assert.equal(model.reasoning, true);
});

test("OpenCode Go GLM 5.3 Flash uses catalog capabilities and Z.ai thinking compat", () => {
  const connection = { protocol: "opencode-go", base_url: "https://opencode.ai/zen/go/v1" };
  const model = buildModel(connection, "glm-5.3-flash");
  assert.deepEqual(model.input, ["text", "image"]);
  assert.equal(model.reasoning, true);
  assert.equal(model.compat.supportsStore, false);
  assert.equal(model.compat.supportsDeveloperRole, false);
  assert.equal(model.compat.maxTokensField, "max_tokens");
  assert.equal(model.compat.thinkingFormat, "zai");
  assert.equal(buildModel(connection, "glm-5.3").compat.thinkingFormat, "zai");
  assert.deepEqual(describeModel(connection, "glm-5.3-flash"), {
    api: "openai-completions",
    vision: true,
    reasoning: true,
    supportedThinkingLevels: ["low", "high", "max"],
    recommendedThinkingLevel: "high",
  });
});

test("catalog capabilities survive a connection protocol override", () => {
  const model = buildModel(
    { protocol: "chat", base_url: "https://custom.example/v1" },
    "gpt-5.6-luna",
  );
  assert.equal(model.api, "openai-completions");
  assert.deepEqual(model.input, ["text", "image"]);
  assert.equal(model.reasoning, true);
  assert.equal(model.compat, undefined);
});

test("unknown model capability heuristics remain conservative", () => {
  assert.deepEqual(inferModelCapabilities("plain-text-model"), {
    input: ["text"],
    reasoning: false,
  });
  assert.deepEqual(inferModelCapabilities("vendor/qwen3-vl"), {
    input: ["text", "image"],
    reasoning: true,
  });
});
