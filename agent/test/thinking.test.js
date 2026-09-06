import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { validateTurn } from "../src/bridge.js";

function validTurn(thinkingLevel) {
  return {
    type: "turn",
    systemPrompt: "",
    sessionId: "test-session",
    connection: { protocol: "chat", base_url: "https://example.test/v1", api_key: "secret" },
    model: "test-model",
    messages: [],
    prompt: "test",
    promptImageIds: [],
    thinkingLevel,
  };
}

test("turn validation accepts every supported thinking level", () => {
  for (const level of ["off", "minimal", "low", "medium", "high", "xhigh", "max"]) {
    assert.doesNotThrow(() => validateTurn(validTurn(level)));
  }
});

test("turn validation rejects missing, non-string, and unknown thinking levels", () => {
  assert.throws(() => validateTurn(validTurn(undefined)), /thinkingLevel must be one of/);
  assert.throws(() => validateTurn(validTurn(1)), /thinkingLevel must be one of/);
  assert.throws(() => validateTurn(validTurn("extreme")), /thinkingLevel must be one of/);
});

test("describe-model reads one object from stdin without exposing connection secrets", () => {
  const input = {
    connection: {
      protocol: "opencode-go",
      base_url: "https://opencode.ai/zen/go/v1",
      api_key: "must-not-appear",
    },
    model: "glm-5.3-flash",
  };
  const result = spawnSync(process.execPath, ["src/describe-model.js"], {
    cwd: new URL("..", import.meta.url),
    input: JSON.stringify(input),
    encoding: "utf8",
  });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout), {
    api: "openai-completions",
    vision: true,
    reasoning: true,
    supportedThinkingLevels: ["low", "high", "max"],
    recommendedThinkingLevel: "high",
  });
  assert.equal(result.stdout.includes(input.connection.api_key), false);
});
