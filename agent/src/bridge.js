#!/usr/bin/env node
import { fileURLToPath } from "node:url";
import { createInterface } from "node:readline";
import { Agent } from "@earendil-works/pi-agent-core";
import { streamSimple as streamAnthropic } from "@earendil-works/pi-ai/api/anthropic-messages";
import { streamSimple as streamChat } from "@earendil-works/pi-ai/api/openai-completions";
import { streamSimple as streamResponses } from "@earendil-works/pi-ai/api/openai-responses";
import { buildModel } from "./model.js";
import {
  createImageLookup,
  formatJsonLine,
  inflateImageRefs,
  liftToolResultImages,
  parseJsonLine,
  sanitizeImages,
} from "./protocol.js";
import { createTools } from "./tools.js";

const THINKING_LEVELS = new Set(["off", "minimal", "low", "medium", "high", "xhigh", "max"]);

export const SYSTEM_PREFIX = `You are the Offcut workspace agent.
Use only the provided Offcut tools to inspect or change application state.
Do not claim a change or image generation succeeded until its tool returns successfully.`;

function errorText(error) {
  return error instanceof Error ? error.message : String(error);
}

function abortError() {
  const error = new Error("Agent turn aborted");
  error.name = "AbortError";
  return error;
}

export class ToolRequestBroker {
  constructor(send) {
    this.send = send;
    this.nextId = 1;
    this.pending = new Map();
  }

  request({ toolCallId, name, args }, signal, onUpdate) {
    if (signal?.aborted) return Promise.reject(abortError());
    const requestId = String(this.nextId++);
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        this.pending.delete(requestId);
        reject(abortError());
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      this.pending.set(requestId, { resolve, reject, onUpdate, signal, onAbort });
      this.send({ type: "tool_request", requestId, toolCallId, name, args });
    });
  }

  respond(command) {
    const pending = this.pending.get(String(command.requestId));
    if (!pending) throw new Error(`Unknown tool response requestId: ${String(command.requestId)}`);

    if (Object.hasOwn(command, "update")) pending.onUpdate?.(command.update);
    if (!Object.hasOwn(command, "result") && !Object.hasOwn(command, "error")) return;

    this.pending.delete(String(command.requestId));
    pending.signal?.removeEventListener("abort", pending.onAbort);
    if (Object.hasOwn(command, "error") && command.error != null) {
      pending.reject(new Error(errorText(command.error)));
    } else {
      pending.resolve(command.result);
    }
  }

  abortAll() {
    for (const [requestId, pending] of this.pending) {
      this.pending.delete(requestId);
      pending.signal?.removeEventListener("abort", pending.onAbort);
      pending.reject(abortError());
    }
  }
}

export function directStream(model, context, options) {
  // These low-level Pi adapters do not add Go's conversation header. Keep it here so
  // every protocol and every tool-loop continuation uses the persisted chat ID.
  if (model.provider === "opencode-go") {
    options = {
      ...options,
      headers: {
        ...options?.headers,
        "User-Agent": "offcut/1.0",
        ...(options?.sessionId ? { "x-opencode-session": options.sessionId } : {}),
      },
    };
  }
  if (model.api === "openai-completions") return streamChat(model, context, options);
  if (model.api === "openai-responses") return streamResponses(model, context, options);
  if (model.api === "anthropic-messages") return streamAnthropic(model, context, options);
  throw new Error(`Unsupported Pi API: ${model.api}`);
}

export function validateTurn(command) {
  if (command.type !== "turn") throw new Error("The first input command must have type=turn");
  if (typeof command.systemPrompt !== "string") throw new Error("systemPrompt must be a string");
  if (typeof command.sessionId !== "string" || !command.sessionId) {
    throw new Error("sessionId must be a non-empty string");
  }
  if (!command.connection || typeof command.connection !== "object") {
    throw new Error("connection must be an object");
  }
  if (typeof command.connection.api_key !== "string") {
    throw new Error("connection.api_key must be a string");
  }
  if (!Array.isArray(command.messages)) throw new Error("messages must be an array");
  if (typeof command.prompt !== "string") throw new Error("prompt must be a string");
  if (!Array.isArray(command.promptImageIds)) throw new Error("promptImageIds must be an array");
  if (typeof command.thinkingLevel !== "string" || !THINKING_LEVELS.has(command.thinkingLevel)) {
    throw new Error("thinkingLevel must be one of: off, minimal, low, medium, high, xhigh, max");
  }
}

export class JsonlAgentBridge {
  constructor({ input = process.stdin, output = process.stdout } = {}) {
    this.output = output;
    this.readline = createInterface({ input, crlfDelay: Infinity });
    this.broker = new ToolRequestBroker((message) => this.send(message));
    this.agent = undefined;
    this.started = false;
    this.finished = false;
  }

  send(message) {
    this.output.write(formatJsonLine(message));
  }

  async handle(command) {
    if (!this.started) {
      this.started = true;
      validateTurn(command);
      await this.runTurn(command);
      return;
    }
    if (this.finished) throw new Error("Agent turn is already finished");
    if (command.type === "tool_response") {
      this.broker.respond(command);
      return;
    }
    if (command.type === "abort") {
      this.agent?.abort();
      this.broker.abortAll();
      return;
    }
    throw new Error(`Unsupported input command during turn: ${String(command.type)}`);
  }

  async runTurn(command) {
    const imageLookup = createImageLookup(command.images);
    const messages = inflateImageRefs(command.messages, imageLookup);
    const promptImages = command.promptImageIds.map((imageId) => {
      const image = imageLookup.byId.get(imageId);
      if (!image) throw new Error(`Unknown prompt image reference: ${String(imageId)}`);
      return { ...image };
    });
    const model = buildModel(command.connection, command.model);
    const tools = createTools(command.toolNames, this.broker, imageLookup);
    const systemPrompt = command.systemPrompt
      ? `${SYSTEM_PREFIX}\n\n${command.systemPrompt}`
      : SYSTEM_PREFIX;

    this.agent = new Agent({
      initialState: {
        systemPrompt,
        model,
        messages,
        tools,
        thinkingLevel: command.thinkingLevel,
      },
      // The lift is a property of the wire format, not of the conversation, so it happens on the
      // way to the provider and never touches the agent's own message list: what is emitted,
      // persisted and replayed stays the plain tool result that produced the image.
      streamFn: (streamModel, context, options) =>
        directStream(
          streamModel,
          { ...context, messages: liftToolResultImages(streamModel, context.messages, imageLookup) },
          options,
        ),
      getApiKey: () => command.connection.api_key,
      sessionId: command.sessionId,
    });
    this.agent.subscribe((event) => {
      if (!this.finished) {
        this.send({ type: "event", event: sanitizeImages(event, imageLookup) });
      }
    });
    await this.agent.prompt(command.prompt, promptImages);
  }

  start() {
    this.readline.on("line", (line) => {
      if (!line.trim()) return;
      let command;
      try {
        command = parseJsonLine(line);
      } catch (error) {
        this.finishError(error);
        return;
      }
      void this.handle(command).then(
        () => {
          if (command.type === "turn") this.finishDone();
        },
        (error) => this.finishError(error),
      );
    });
    this.readline.on("close", () => {
      if (this.started && !this.finished) {
        this.agent?.abort();
        this.broker.abortAll();
      }
    });
  }

  finishDone() {
    if (this.finished) return;
    this.finished = true;
    this.send({ type: "done" });
    this.readline.close();
  }

  finishError(error) {
    if (this.finished) return;
    this.finished = true;
    this.agent?.abort();
    this.broker.abortAll();
    this.send({ type: "error", error: errorText(error) });
    this.readline.close();
  }
}

const isMain = process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1];
if (isMain) new JsonlAgentBridge().start();
