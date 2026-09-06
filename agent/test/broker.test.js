import assert from "node:assert/strict";
import test from "node:test";
import { ToolRequestBroker } from "../src/bridge.js";

test("tool broker forwards repeated updates before the final result", async () => {
  const sent = [];
  const updates = [];
  const broker = new ToolRequestBroker((message) => sent.push(message));
  const resultPromise = broker.request(
    { toolCallId: "call-7", name: "generate_image", args: {} },
    undefined,
    (update) => updates.push(update),
  );

  assert.deepEqual(sent, [
    {
      type: "tool_request",
      requestId: "1",
      toolCallId: "call-7",
      name: "generate_image",
      args: {},
    },
  ]);
  broker.respond({ type: "tool_response", requestId: "1", update: { progress: 0.2 } });
  broker.respond({ type: "tool_response", requestId: "1", update: { progress: 0.8 } });
  broker.respond({ type: "tool_response", requestId: "1", result: { image_id: "new" } });

  assert.deepEqual(updates, [{ progress: 0.2 }, { progress: 0.8 }]);
  assert.deepEqual(await resultPromise, { image_id: "new" });
});

test("tool broker rejects pending requests on abort", async () => {
  const broker = new ToolRequestBroker(() => {});
  const resultPromise = broker.request(
    { toolCallId: "call-8", name: "get_workspace_state", args: {} },
    undefined,
    () => {},
  );
  broker.abortAll();
  await assert.rejects(resultPromise, { name: "AbortError", message: "Agent turn aborted" });
});
