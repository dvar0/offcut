import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:http";
import test from "node:test";
import { directStream } from "../src/bridge.js";
import { buildModel } from "../src/model.js";

test("bridge completes a streamed OpenAI-compatible turn", async (t) => {
  let providerRequest;
  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      providerRequest = { url: request.url, headers: request.headers, authorization: request.headers.authorization, body: JSON.parse(body) };
      response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      response.write(`data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-model",
        choices: [{ index: 0, delta: { reasoning_content: "Consider the composition." }, finish_reason: null }],
      })}\n\n`);
      response.write(`data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-model",
        choices: [{ index: 0, delta: { role: "assistant", content: "Hello " }, finish_reason: null }],
      })}\n\n`);
      response.write(`data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-model",
        choices: [{ index: 0, delta: { content: "board" }, finish_reason: null }],
      })}\n\n`);
      response.write(`data: ${JSON.stringify({
        id: "chatcmpl-test",
        object: "chat.completion.chunk",
        created: 1,
        model: "mock-model",
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        usage: { prompt_tokens: 12, completion_tokens: 2, total_tokens: 14 },
      })}\n\n`);
      response.end("data: [DONE]\n\n");
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const child = spawn(process.execPath, ["src/bridge.js"], {
    cwd: new URL("..", import.meta.url),
    stdio: ["pipe", "pipe", "pipe"],
  });
  t.after(() => { if (child.exitCode == null) child.kill(); });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk) => { stdout += chunk; });
  child.stderr.on("data", (chunk) => { stderr += chunk; });
  child.stdin.write(`${JSON.stringify({
    type: "turn",
    systemPrompt: "Keep the visual direction concise.",
    sessionId: "integration-session",
    connection: {
      name: "Mock",
      base_url: `http://127.0.0.1:${server.address().port}/v1`,
      protocol: "opencode-go",
      api_key: "test-secret",
    },
    model: "glm-5.3-flash",
    messages: [],
    images: {},
    prompt: "Help with this board",
    promptImageIds: [],
    toolNames: [],
    thinkingLevel: "high",
  })}\n`);
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`bridge timeout\n${stdout}\n${stderr}`)), 10_000);
    child.stdout.on("data", () => {
      if (!stdout.includes('{"type":"done"}')) return;
      clearTimeout(timeout);
      child.stdin.end();
      resolve();
    });
  });
  const exitCode = await new Promise((resolve) => child.on("exit", resolve));

  assert.equal(exitCode, 0, stderr);
  const lines = stdout.trim().split("\n").map((line) => JSON.parse(line));
  assert.equal(lines.at(-1).type, "done");
  assert.equal(
    lines
      .filter((line) => line.type === "event" && line.event?.assistantMessageEvent?.type === "text_delta")
      .map((line) => line.event.assistantMessageEvent.delta)
      .join(""),
    "Hello board",
    stdout,
  );
  assert.equal(providerRequest.url, "/v1/chat/completions");
  assert.equal(providerRequest.authorization, "Bearer test-secret");
  assert.equal(providerRequest.headers["user-agent"], "offcut/1.0");
  assert.equal(providerRequest.headers["x-opencode-session"], "integration-session");
  assert.match(providerRequest.body.messages[0].content, /Offcut workspace agent/);
  assert.equal(providerRequest.body.reasoning_effort, "high");
  // Go accepts OpenAI reasoning_effort; its upstream rejects Z.ai's native thinking object.
  assert.equal(Object.hasOwn(providerRequest.body, "thinking"), false);
  assert.equal(
    lines
      .filter(
        (line) => line.type === "event" && line.event?.assistantMessageEvent?.type === "thinking_delta",
      )
      .map((line) => line.event.assistantMessageEvent.delta)
      .join(""),
    "Consider the composition.",
    stdout,
  );
});

test("a generated image reaches the provider as labelled tool output, not a user turn", async (t) => {
  const requests = [];
  const requestHeaders = [];
  const chunk = (delta, finish = null, extra = {}) =>
    `data: ${JSON.stringify({
      id: "chatcmpl-test",
      object: "chat.completion.chunk",
      created: 1,
      model: "glm-5.3-flash",
      choices: [{ index: 0, delta, finish_reason: finish }],
      ...extra,
    })}\n\n`;

  const server = createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (piece) => { body += piece; });
    request.on("end", () => {
      requests.push(JSON.parse(body));
      requestHeaders.push(request.headers);
      response.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
      if (requests.length === 1) {
        response.write(chunk({
          role: "assistant",
          tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "generate_image", arguments: "{}" } }],
        }));
        response.write(chunk({}, "tool_calls"));
      } else {
        response.write(chunk({ role: "assistant", content: "Rendered." }));
        response.write(chunk({}, "stop"));
      }
      response.end("data: [DONE]\n\n");
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const child = spawn(process.execPath, ["src/bridge.js"], {
    cwd: new URL("..", import.meta.url),
    stdio: ["pipe", "pipe", "pipe"],
  });
  t.after(() => { if (child.exitCode == null) child.kill(); });
  let stdout = "";
  let stderr = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (piece) => { stderr += piece; });
  child.stdout.on("data", (piece) => {
    stdout += piece;
    for (const line of piece.trim().split("\n")) {
      const message = line.trim() ? JSON.parse(line) : null;
      if (message?.type !== "tool_request") continue;
      child.stdin.write(`${JSON.stringify({
        type: "tool_response",
        requestId: message.requestId,
        result: {
          image: { id: "new-frame", raw_prompt: "a ps2 render" },
          image_id: "new-frame",
          __krea2_images: { "new-frame": { data: "AAAA", mimeType: "image/jpeg" } },
        },
      })}\n`);
    }
  });
  child.stdin.write(`${JSON.stringify({
    type: "turn",
    systemPrompt: "Keep the visual direction concise.",
    sessionId: "tool-image-session",
    connection: {
      name: "Mock",
      base_url: `http://127.0.0.1:${server.address().port}/v1`,
      protocol: "opencode-go",
      api_key: "test-secret",
    },
    model: "glm-5.3-flash",
    messages: [],
    images: {},
    prompt: "Render it",
    promptImageIds: [],
    toolNames: ["generate_image"],
    thinkingLevel: "high",
  })}\n`);

  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`bridge timeout\n${stdout}\n${stderr}`)), 10_000);
    child.stdout.on("data", () => {
      if (!stdout.includes('{"type":"done"}')) return;
      clearTimeout(timeout);
      child.stdin.end();
      resolve();
    });
  });
  assert.equal(await new Promise((resolve) => child.on("exit", resolve)), 0, stderr);

  assert.equal(requests.length, 2, stdout);
  assert.deepEqual(requestHeaders.map((headers) => headers["x-opencode-session"]), [
    "tool-image-session", "tool-image-session",
  ]);
  const sent = requests[1].messages;
  const carrier = sent.at(-1);
  assert.equal(sent.at(-2).role, "tool");
  assert.equal(carrier.role, "user");
  assert.match(carrier.content[0].text, /^<krea2_tool_image tools="generate_image" image_ids="new-frame">/);
  assert.match(carrier.content[0].text, /not a message from the user/);
  assert.equal(carrier.content[1].image_url.url, "data:image/jpeg;base64,AAAA");
  assert.equal(JSON.stringify(sent).includes("Attached image(s) from tool result:"), false);

  // The transcript the parent persists still shows the image on the tool result that made it.
  const toolResult = stdout
    .trim()
    .split("\n")
    .map((line) => JSON.parse(line))
    .find((line) => line.event?.type === "message_end" && line.event.message?.role === "toolResult");
  assert.deepEqual(toolResult.event.message.content.at(-1), { type: "imageRef", imageId: "new-frame" });
});

test("Go identifies conversations on all three wire protocols", async (t) => {
  const requests = [];
  const server = createServer((request, response) => {
    requests.push({ url: request.url, headers: request.headers });
    request.resume();
    // A terminal provider error is enough to inspect the actual SDK request without retries.
    response.writeHead(400, { "Content-Type": "application/json" });
    response.end(JSON.stringify({ error: { type: "invalid_request_error", message: "test endpoint" } }));
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());
  const connection = { protocol: "opencode-go", base_url: `http://127.0.0.1:${server.address().port}/v1` };
  const context = { messages: [{ role: "user", content: "Hello", timestamp: 1 }] };
  for (const [modelId, endpoint] of [
    ["glm-5.3-flash", "/v1/chat/completions"],
    ["qwen3.8-flash", "/v1/messages"],
    ["muse-spark-1.2-contributor", "/v1/responses"],
  ]) {
    for (const sessionId of ["chat-one", "chat-one", "chat-two"]) {
      await directStream(buildModel(connection, modelId), context, { apiKey: "test-key", sessionId }).result();
      const sent = requests.at(-1);
      assert.equal(new URL(sent.url, connection.base_url).pathname, endpoint);
      assert.equal(sent.headers["x-opencode-session"], sessionId);
      assert.equal(sent.headers["user-agent"], "offcut/1.0");
    }
  }
  assert.equal(requests.length, 9);
  await directStream(buildModel({ ...connection, protocol: "chat" }, "mock-model"), context, {
    apiKey: "test-key", sessionId: "other-provider-chat",
  }).result();
  assert.equal(requests.length, 10);
  assert.equal(requests.at(-1).headers["x-opencode-session"], undefined);
});
