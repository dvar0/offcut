import { openImageViewer } from "./images.js";
import { renderPromptChange } from "./prompt-diff.js";
import { $, copyText, findImage, formatCost, frameSeed, state } from "./shared.js";

export function messageText(message) {
  if (typeof message?.content === "string") return message.content;
  if (typeof message?.text === "string") return message.text;
  if (typeof message?.message === "string") return message.message;
  if (Array.isArray(message?.content)) return message.content.map((part) => {
    if (typeof part === "string") return part;
    return part?.text || part?.content || "";
  }).filter(Boolean).join("\n\n");
  return "";
}

export function messageImages(message) {
  const images = [];
  const candidates = [message?.image, ...(message?.images || []), ...(message?.attachments || [])].filter(Boolean);
  for (const candidate of candidates) {
    if (typeof candidate === "string") images.push(state.boardImages.some((image) => image.id === candidate) ? { id: candidate } : { image_url: candidate });
    else if (candidate.type === "image" || candidate.image_url || candidate.url || candidate.image_id || candidate.id) images.push(candidate);
  }
  if (message?.image_id && !images.some((image) => (image.image_id || image.id) === message.image_id)) {
    images.push({ image_id: message.image_id, image_url: message.image_url, prompt: message.prompt });
  }
  return images;
}

// A turn is a sequence of prose, reasoning, and tool calls in the order the model
// produced them. The server sends that order as `blocks`; older payloads and locally
// built messages only carry the flattened fields, so rebuild an approximate order.
export function messageBlocks(message) {
  if (!message) return [];
  if (Array.isArray(message.blocks)) return message.blocks;
  const blocks = [];
  const reasoning = typeof message.reasoning === "string" ? message.reasoning : "";
  if (reasoning || message.reasoning_active) {
    blocks.push({
      type: "reasoning",
      reasoning,
      active: Boolean(message.reasoning_active),
      started_at: message.reasoning_started_at,
      ended_at: message.reasoning_ended_at,
      duration_ms: message.reasoning_duration_ms ?? message.thinking_duration_ms ?? message.reasoningDurationMs,
    });
  }
  const text = messageText(message);
  if (text) blocks.push({ type: "text", text });
  const tools = message.tools || message.tool_calls || (message.tool ? [message.tool] : []);
  for (const tool of tools) blocks.push({ type: "tool", tool });
  return blocks;
}

// Tool calls stream in on their own events without a content index, so the block a
// delta belongs to is the one still open at the tail of the turn.
export function openStreamBlock(message, type, seed = {}, contentIndex = null) {
  message.blocks ||= [];
  const last = message.blocks[message.blocks.length - 1];
  if (last && last.type === type && (contentIndex == null || last.content_index == null || last.content_index === contentIndex)) {
    if (contentIndex != null) last.content_index = contentIndex;
    return last;
  }
  const block = { type, content_index: contentIndex, ...seed };
  message.blocks.push(block);
  return block;
}

export function lastStreamBlock(message, type) {
  const blocks = message?.blocks || [];
  for (let index = blocks.length - 1; index >= 0; index -= 1) if (blocks[index].type === type) return blocks[index];
  return null;
}

// Keep the flattened fields in step with the blocks so everything that reads a
// finished message (persistence fallbacks, copy, usage) still sees the whole turn.
export function syncStreamMirrors(message) {
  const blocks = message.blocks || [];
  message.content = blocks.filter((block) => block.type === "text").map((block) => block.text).join("\n\n");
  message.reasoning = blocks.filter((block) => block.type === "reasoning").map((block) => block.reasoning).join("\n\n");
  message.tools = blocks.filter((block) => block.type === "tool").map((block) => block.tool);
  const active = blocks.some((block) => block.type === "reasoning" && block.active);
  message.reasoning_active = active;
}

function appendInlineMarkdown(container, source) {
  let text = String(source || "");
  const token = /(`[^`\n]+`|\[[^\]\n]+\]\([^\s)]+\)|\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_)/;
  while (text) {
    const match = token.exec(text);
    if (!match) {
      container.append(document.createTextNode(text));
      break;
    }
    if (match.index) container.append(document.createTextNode(text.slice(0, match.index)));
    const value = match[0];
    if (value.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = value.slice(1, -1);
      container.append(code);
    } else if (value.startsWith("[")) {
      const parts = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(value);
      let url = null;
      try { url = /^(?:https?:|mailto:)/i.test(parts[2]) ? new URL(parts[2]) : null; } catch (_) { /* Render invalid links as text. */ }
      if (url && ["http:", "https:", "mailto:"].includes(url.protocol)) {
        const link = document.createElement("a");
        link.href = url.href;
        link.rel = "noopener noreferrer";
        if (url.protocol !== "mailto:") link.target = "_blank";
        appendInlineMarkdown(link, parts[1]);
        container.append(link);
      } else container.append(document.createTextNode(value));
    } else {
      const strong = value.startsWith("**") || value.startsWith("__");
      const emphasis = document.createElement(strong ? "strong" : "em");
      appendInlineMarkdown(emphasis, value.slice(strong ? 2 : 1, strong ? -2 : -1));
      container.append(emphasis);
    }
    text = text.slice(match.index + value.length);
  }
}

function appendMarkdown(container, source) {
  const lines = String(source || "").replaceAll("\r\n", "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    const fence = /^\s*```([\w.+-]{0,20})\s*$/.exec(line);
    if (fence) {
      const codeLines = [];
      index += 1;
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) codeLines.push(lines[index++]);
      if (index < lines.length) index += 1;
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      if (fence[1]) code.dataset.language = fence[1];
      code.textContent = codeLines.join("\n");
      pre.append(code);
      container.append(pre);
      continue;
    }
    const heading = /^(#{1,6})\s+(.+)$/.exec(line);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      appendInlineMarkdown(element, heading[2]);
      container.append(element);
      index += 1;
      continue;
    }
    if (index + 1 < lines.length && line.includes("|") && /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1])) {
      const cells = (value) => value.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
      const headings = cells(line);
      const alignments = cells(lines[index + 1]).map((cell) => cell.startsWith(":") && cell.endsWith(":") ? "center" : cell.endsWith(":") ? "right" : "left");
      const table = document.createElement("table");
      const head = document.createElement("thead");
      const headRow = document.createElement("tr");
      for (const [cellIndex, value] of headings.entries()) {
        const cell = document.createElement("th");
        cell.style.textAlign = alignments[cellIndex] || "left";
        appendInlineMarkdown(cell, value);
        headRow.append(cell);
      }
      head.append(headRow);
      table.append(head);
      index += 2;
      const body = document.createElement("tbody");
      while (index < lines.length && lines[index].trim() && lines[index].includes("|")) {
        const row = document.createElement("tr");
        for (const [cellIndex, value] of cells(lines[index]).entries()) {
          const cell = document.createElement("td");
          cell.style.textAlign = alignments[cellIndex] || "left";
          appendInlineMarkdown(cell, value);
          row.append(cell);
        }
        body.append(row);
        index += 1;
      }
      table.append(body);
      container.append(table);
      continue;
    }
    const listMatch = /^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/.exec(line);
    if (listMatch) {
      const ordered = Boolean(listMatch[2]);
      const list = document.createElement(ordered ? "ol" : "ul");
      while (index < lines.length) {
        const itemMatch = /^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/.exec(lines[index]);
        if (!itemMatch || Boolean(itemMatch[2]) !== ordered) break;
        const item = document.createElement("li");
        appendInlineMarkdown(item, itemMatch[3]);
        list.append(item);
        index += 1;
      }
      container.append(list);
      continue;
    }
    if (/^\s*>/.test(line)) {
      const quote = document.createElement("blockquote");
      const quoteLines = [];
      while (index < lines.length && /^\s*>/.test(lines[index])) quoteLines.push(lines[index++].replace(/^\s*>\s?/, ""));
      appendMarkdown(quote, quoteLines.join("\n"));
      container.append(quote);
      continue;
    }
    const paragraphLines = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim()
      && !/^\s*```/.test(lines[index]) && !/^#{1,6}\s/.test(lines[index])
      && !/^\s*(?:[-+*]|\d+[.)])\s+/.test(lines[index]) && !/^\s*>/.test(lines[index])) {
      paragraphLines.push(lines[index].trim());
      index += 1;
    }
    const paragraph = document.createElement("p");
    appendInlineMarkdown(paragraph, paragraphLines.join("\n"));
    container.append(paragraph);
  }
}

// A pasted reference carries no prompt, route, or seed, so it renders as the picture alone.
// Anything this app generated has that record, and it goes underneath in a collapsed panel
// rather than a truncated line, so the image itself keeps the width.
function imageCardMetadata(image) {
  const rows = [];
  const push = (label, value) => { if (value || value === 0) rows.push([label, String(value)]); };
  push("ROUTE", image.preset);
  if (image.width && image.height) push("FRAME", `${image.width} x ${image.height}`);
  push("SEED", frameSeed(image));
  push("STEPS", image.steps);
  push("GUIDANCE", image.guidance);
  const loras = (image.loras || []).filter((item) => item && item.name);
  if (loras.length) push("LORAS", loras.map((item) => `${item.name} @ ${item.strength ?? 1}`).join(", "));
  // Images generated before the route gate could record a negative the sampler discarded.
  // Showing it unqualified would credit it with an influence it never had.
  if (image.negative_prompt) {
    const honored = image.preset === "raw-int8";
    push("NEGATIVE", honored ? image.negative_prompt : `${image.negative_prompt}  (ignored on ${image.preset})`);
  }
  return rows;
}

function hasImageRecord(image) {
  return Boolean(image.raw_prompt || image.final_prompt || image.enhanced_prompt || image.seed || image.preset);
}

function createImageCard(image) {
  const resolved = findImage(image);
  const imageId = resolved.image_id || resolved.id || "";
  const card = document.createElement("div");
  card.className = "message-image";
  const button = document.createElement("button");
  button.type = "button";
  button.className = "message-image-frame";
  button.title = "View full screen";
  button.setAttribute("aria-haspopup", "dialog");
  const img = document.createElement("img");
  img.src = resolved.image_url || resolved.url || "";
  // An <img> whose load failed paints its alt text inside the frame. The prompt is a whole
  // paragraph, so it filled the box and read as the description overlapping the picture. Keep
  // this short: the prompt is already under the image, in the meta panel and in the summary.
  img.alt = hasImageRecord(resolved) ? "Generated image" : "Attached image";
  img.loading = "lazy";
  // An image with no intrinsic size reserves nothing until its first byte lands, so the box
  // starts at the fallback height and jumps to the real one as it decodes. These give the box
  // its final height before a single byte of the bitmap is needed.
  if (resolved.width && resolved.height) {
    img.width = resolved.width;
    img.height = resolved.height;
  } else {
    img.style.minHeight = "180px";
  }
  button.append(img);
  button.addEventListener("click", () => openImageViewer(resolved));
  card.append(button);

  if (!hasImageRecord(resolved)) {
    if (imageId) {
      const bare = document.createElement("b");
      bare.className = "message-image-id";
      bare.textContent = `REFERENCE ${imageId.slice(0, 8)}`;
      bare.title = imageId;
      card.append(bare);
    }
    return card;
  }

  const details = document.createElement("details");
  details.className = "message-image-meta";
  const summary = document.createElement("summary");
  const label = document.createElement("b");
  label.textContent = imageId ? `IMAGE ${imageId.slice(0, 8)}` : "IMAGE";
  label.title = imageId;
  const preview = document.createElement("span");
  preview.textContent = resolved.raw_prompt || resolved.final_prompt || "No prompt recorded";
  summary.append(label, preview);
  details.append(summary);

  const body = document.createElement("div");
  for (const [name, value] of imageCardMetadata(resolved)) {
    const row = document.createElement("p");
    const key = document.createElement("i");
    key.textContent = name;
    const held = document.createElement("span");
    held.textContent = value;
    row.append(key, held);
    body.append(row);
  }
  for (const [name, value] of [["PROMPT", resolved.raw_prompt], ["ENHANCED", resolved.enhanced_prompt], ["FINAL", resolved.final_prompt]]) {
    if (!value || value === resolved.raw_prompt && name !== "PROMPT") continue;
    const heading = document.createElement("i");
    heading.className = "message-image-prompt-label";
    heading.textContent = name;
    const text = document.createElement("p");
    text.className = "message-image-prompt";
    text.textContent = value;
    body.append(heading, text);
  }
  details.append(body);
  card.append(details);
  return card;
}

function usageText(usage) {
  if (!usage) return "";
  const input = usage.input ?? usage.input_tokens ?? usage.prompt_tokens;
  const output = usage.output ?? usage.output_tokens ?? usage.completion_tokens;
  const pieces = [];
  if (input != null) pieces.push(`${input} in`);
  if (output != null) pieces.push(`${output} out`);
  if (usage.cacheRead) pieces.push(`${usage.cacheRead} cached`);
  if (usage.cacheWrite) pieces.push(`${usage.cacheWrite} cache write`);
  if (usage.reasoning) pieces.push(`${usage.reasoning} reasoning`);
  const cost = typeof usage.cost === "object" ? usage.cost?.total : usage.cost;
  if (cost != null) pieces.push(formatCost(cost));
  return pieces.join(" · ");
}

export function messageUiId(message) {
  if (!message._uiId) message._uiId = message.id ? String(message.id) : `local-${state.nextMessageUiId++}`;
  return message._uiId;
}

function reasoningDuration(block) {
  const milliseconds = block.duration_ms ?? block.reasoning_duration_ms ?? block.thinking_duration_ms ?? block.reasoningDurationMs;
  if (Number.isFinite(Number(milliseconds))) return Number(milliseconds);
  const startedAt = block.started_at ?? block.reasoning_started_at;
  const endedAt = block.ended_at ?? block.reasoning_ended_at;
  if (startedAt && endedAt) {
    const duration = new Date(endedAt) - new Date(startedAt);
    if (Number.isFinite(duration) && duration >= 0) return duration;
  }
  return null;
}

function formatThinkingDuration(milliseconds) {
  const seconds = Math.max(0, milliseconds / 1000);
  if (seconds < 60) return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)}S`;
  return `${Math.floor(seconds / 60)}M ${Math.round(seconds % 60)}S`;
}

// Every token rebuilds the streaming turn's node, and a fresh element starts its CSS
// animations over from zero. A looping indicator therefore never reached the end of its
// cycle while deltas were landing, which is what made the thinking sheen stutter and
// restart. A negative delay drops each new node into the phase the old one was in, so the
// loop reads as continuous across the rebuilds. Durations must match styles.css.
function continueAnimation(element, epoch, durationMs) {
  const elapsed = Math.max(0, Date.now() - epoch);
  element.style.animationDelay = `-${elapsed % durationMs}ms`;
}

// The label cycles rather than sitting still: a word that changes says the turn is alive even
// when the ripple is easy to miss. They name the work of directing a shot, since that is what
// the model is doing while this row is open.
const THINKING_WORDS = ["COMPOSING", "FRAMING", "VISUALIZING", "SKETCHING", "LIGHTING", "STAGING", "IMAGINING", "DIRECTING", "PICTURING", "REFRAMING"];

const THINKING_WAVE_MS = 1400;

// Two whole ripples per word, so a swap always lands on a crest boundary rather than cutting
// one mid-travel the way an interval unrelated to the wave used to.
const THINKING_WORD_MS = THINKING_WAVE_MS * 2;

// Wide enough that a word's letters span the flat tail of the keyframe: at a shorter stagger
// every letter was at rest at once for a beat between ripples, which read as another stall.
const THINKING_WAVE_STAGGER_MS = 70;

const CHAT_PROGRESS_MS = 1600;

const DOT_MATRIX_MS = 960; // Must match dot-matrix-spin in styles.css.

// Row-major cells of the 3x3 grid, in the order the light travels round the ring. Index 4 is
// the middle, which never lights: it is the still point that makes the ring read as a ring.
const DOT_MATRIX_RING = [0, 1, 2, 5, 8, 7, 6, 3];

// The activity indicator: eight cells of a small LED grid lit in ring order, so it is a dot
// matrix and a spinner at once. It replaced a 1px rule that sprang between two widths, which
// in a column of horizontal rules read as a rendering fault rather than as progress. Phases
// come off the turn clock the same way the reasoning ripple's do, so the rebuild on every
// streamed token drops each cell back into the phase its predecessor was in.
export function createDotMatrix(epoch, { small = false } = {}) {
  const matrix = document.createElement("span");
  matrix.className = small ? "dot-matrix dot-matrix-sm" : "dot-matrix";
  matrix.setAttribute("aria-hidden", "true");
  const elapsed = Math.max(0, Date.now() - epoch);
  const step = DOT_MATRIX_MS / DOT_MATRIX_RING.length;
  for (let index = 0; index < 9; index += 1) {
    const cell = document.createElement("i");
    const ring = DOT_MATRIX_RING.indexOf(index);
    if (ring < 0) cell.className = "dot-matrix-core";
    else cell.style.animationDelay = `-${(((elapsed - ring * step) % DOT_MATRIX_MS) + DOT_MATRIX_MS) % DOT_MATRIX_MS}ms`;
    matrix.append(cell);
  }
  return matrix;
}

// One inline-block per letter, each trailing the one before it, so the crest travels left to
// right across the word. Spaces stay bare text: an inline-block space collapses to nothing.
function paintThinkingLabel(label, epoch) {
  const elapsed = Math.max(0, Date.now() - epoch);
  const word = THINKING_WORDS[Math.floor(elapsed / THINKING_WORD_MS) % THINKING_WORDS.length];
  label.replaceChildren();
  for (const [index, character] of [...word].entries()) {
    if (character === " ") {
      label.append(character);
      continue;
    }
    const cell = document.createElement("span");
    cell.textContent = character;
    const offset = (elapsed - index * THINKING_WAVE_STAGGER_MS) % THINKING_WAVE_MS;
    cell.style.animationDelay = `-${(offset + THINKING_WAVE_MS) % THINKING_WAVE_MS}ms`;
    label.append(cell);
  }
}

function renderReasoning(block, id) {
  const reasoning = typeof block.reasoning === "string" ? block.reasoning : "";
  const active = Boolean(block.active);
  if (!reasoning && !active) return null;
  const details = document.createElement("details");
  details.className = `turn-step thinking-row${active ? " active" : ""}`;
  details.open = state.expandedReasoning.has(id);
  details.addEventListener("toggle", () => {
    if (details.open) state.expandedReasoning.add(id);
    else state.expandedReasoning.delete(id);
  });
  const summary = document.createElement("summary");
  const label = document.createElement("b");
  const duration = reasoningDuration(block);
  if (active) {
    // Anchored to when this block opened, so the word and the ripple advance with the turn
    // rather than with however often the transcript happens to repaint.
    paintThinkingLabel(label, Date.parse(block.started_at || "") || Date.now());
  } else {
    label.textContent = duration != null ? `THOUGHT FOR ${formatThinkingDuration(duration)}` : "REASONING";
  }
  const tail = document.createElement("span");
  tail.textContent = reasoning.trim().split(/\n+/).filter(Boolean).at(-1) || "Working through the request";
  summary.append(label, tail);
  if (active) summary.append(createDotMatrix(Date.parse(block.started_at || "") || state.turnStartedAt, { small: true }));
  details.append(summary);
  if (reasoning) {
    const full = document.createElement("div");
    full.className = "step-body";
    appendMarkdown(full, reasoning);
    details.append(full);
  }
  return details;
}

function renderModeChange(change) {
  const row = document.createElement("div");
  row.className = "mode-event";
  const label = document.createElement("b");
  label.textContent = "MODE";
  const change_ = document.createElement("span");
  change_.textContent = `${String(change.from || "").toUpperCase()} → ${String(change.to || "").toUpperCase()}`;
  row.append(label, change_);
  return row;
}

function renderWorkspaceChange(change) {
  const details = document.createElement("details");
  details.className = "turn-step workspace-event";
  const summary = document.createElement("summary");
  const removed = Array.isArray(change.removed) ? change.removed : change.removed ? [change.removed] : [];
  const added = Array.isArray(change.added) ? change.added : change.added ? [change.added] : [];
  const label = document.createElement("b");
  label.textContent = "WORKSPACE";
  const tail = document.createElement("span");
  tail.textContent = `${removed.length} removed · ${added.length} added`;
  summary.append(label, tail);
  const body = document.createElement("div");
  body.className = "step-body";
  // settings_revision counts board autosaves, not prompt versions, so it means nothing to a
  // reader; the removed/added lines below say what actually happened.
  if (change.actor) body.append(`Actor: ${change.actor}`);
  for (const item of removed) {
    const line = document.createElement("p");
    line.className = "prompt-removed";
    line.textContent = `- ${typeof item === "string" ? item : JSON.stringify(item)}`;
    body.append(line);
  }
  for (const item of added) {
    const line = document.createElement("p");
    line.className = "prompt-added";
    line.textContent = `+ ${typeof item === "string" ? item : JSON.stringify(item)}`;
    body.append(line);
  }
  details.append(summary, body);
  return details;
}

function renderToolRow(tool, toolId) {
  const row = document.createElement("details");
  const active = tool.status === "running" || tool.status === "started";
  row.className = `turn-step tool-row${active ? " active" : ""}`;
  row.open = state.expandedTools.has(toolId);
  row.addEventListener("toggle", () => {
    if (row.open) state.expandedTools.add(toolId);
    else state.expandedTools.delete(toolId);
  });
  const summary = document.createElement("summary");
  const name = document.createElement("b");
  name.textContent = String(tool.name || tool.tool_name || "TOOL").replaceAll("_", " ").toUpperCase();
  const detail = document.createElement("span");
  detail.className = "step-body";
  const detailValue = tool.detail || tool.output || tool.result || tool.status || "Working";
  detail.textContent = typeof detailValue === "string" ? detailValue : JSON.stringify(detailValue, null, 2);
  // A collapsed call used to show its name and nothing else, so a run of them was a column of
  // labels. The first line of the result rides in the summary the way reasoning's does, which
  // is usually the whole answer and saves opening the row at all.
  const tail = document.createElement("span");
  tail.textContent = detail.textContent.trim().split(/\n+/).find(Boolean) || "";
  summary.append(name, tail);
  if (active) summary.append(createDotMatrix(state.turnStartedAt, { small: true }));
  row.append(summary, detail);
  return row;
}

function renderMessage(message, { streaming = false } = {}) {
  const role = message.role || message.type || "assistant";
  const article = document.createElement("article");
  article.className = `message message-${role === "user" ? "user" : role === "system" || role === "error" ? "system" : "assistant"}`;
  const body = document.createElement("div");
  body.className = "message-body";
  // Only notices carry a label. The user bubble and the bare assistant column already
  // read as two different voices, so an ASSISTANT header is noise on every turn.
  if (role === "system" || role === "error") {
    const label = document.createElement("span");
    label.className = "message-role";
    label.textContent = "NOTICE";
    body.append(label);
  }
  const uiId = messageUiId(message);
  const text = messageText(message);
  // The same edit reaches the column twice over a turn: once on the streamed workspace patch,
  // once on the settled tool result that replaces it. Draw it where it first appears, once.
  const seenChanges = new Set();
  const appendChange = (change) => {
    const identity = [change.before, change.after, change.revision].join("\u0000");
    if (seenChanges.has(identity)) return;
    seenChanges.add(identity);
    body.append(renderPromptChange(change));
  };
  // Walk the turn in the order the model produced it, so prose written after a tool
  // call stays below that call instead of being hoisted above it.
  let lastTextNode = null;
  for (const [index, block] of messageBlocks(message).entries()) {
    if (block.type === "reasoning") {
      const reasoning = renderReasoning(block, `${uiId}:r${index}`);
      if (reasoning) body.append(reasoning);
    } else if (block.type === "text" && block.text) {
      const before = body.lastElementChild;
      appendMarkdown(body, block.text);
      if (body.lastElementChild !== before) lastTextNode = body.lastElementChild;
    } else if (block.type === "tool" && block.tool) {
      body.append(renderToolRow(block.tool, `${uiId}:${block.tool.id || index}`));
      if (block.tool.prompt_change) appendChange(block.tool.prompt_change);
      // The generated image belongs to the call that made it. Hanging it off the tool block
      // rather than a block of its own is what keeps it in place when message_end settles the
      // turn, and what lets it come back on reload from the persisted tool call.
      if (block.tool.image_id) body.append(createImageCard({ image_id: block.tool.image_id }));
      if (block.tool.comparison_image_ids?.length) {
        const comparison = document.createElement("div");
        comparison.className = "chat-comparison";
        for (const id of block.tool.comparison_image_ids) comparison.append(createImageCard({ image_id: id }));
        body.append(comparison);
      }
    } else if (block.type === "prompt_change" && block.change) {
      appendChange(block.change);
    } else if (block.type === "image" && block.image) {
      body.append(createImageCard(block.image));
    }
  }
  if (role === "user" && message.mode_change) body.append(renderModeChange(message.mode_change));
  if (role === "user" && message.workspace_change) body.append(renderWorkspaceChange(message.workspace_change));
  for (const image of messageImages(message)) body.append(createImageCard(image));
  if (message.error) {
    const notice = document.createElement("p");
    notice.textContent = `Model request failed: ${message.error}`;
    body.append(notice);
  }
  if (message.interrupted) {
    const notice = document.createElement("p");
    notice.textContent = "Stopped";
    body.append(notice);
  }
  // The model is between blocks: the turn started but nothing has arrived, or a tool
  // finished and the next token has not landed. Keep a heartbeat on screen either way.
  if (streaming) {
    const blocks = messageBlocks(message);
    const tail = blocks[blocks.length - 1];
    if (!message.generating && (!tail || tail.type === "tool" || (tail.type === "reasoning" && !tail.active))) {
      const waiting = document.createElement("div");
      waiting.className = "response-waiting";
      const label = document.createElement("span");
      label.textContent = tail ? "WORKING" : "WAITING FOR MODEL";
      waiting.append(createDotMatrix(state.turnStartedAt), label);
      body.append(waiting);
    } else if (tail?.type === "text" && lastTextNode) {
      lastTextNode.classList.add("streaming-tail");
    }
  }
  if (message.generating) {
    const progress = document.createElement("div");
    progress.className = "chat-generation";
    const head = document.createElement("span");
    const detail = document.createElement("span");
    detail.textContent = message.generation_detail || "Generating image";
    head.append(createDotMatrix(state.turnStartedAt), detail);
    const bar = document.createElement("i");
    continueAnimation(bar, state.turnStartedAt, CHAT_PROGRESS_MS);
    progress.append(head, bar);
    body.append(progress);
  }
  if (text) {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "message-copy";
    copy.textContent = "COPY";
    // The button survives the rebuilds now, so the text it writes is read at click time
    // rather than captured from the turn as it stood when the node was made.
    copy.addEventListener("click", async () => {
      const value = state.streamMessage?._uiId === uiId ? messageText(state.streamMessage) : text;
      try { await copyText(value); copy.textContent = "COPIED"; }
      catch (_) { copy.textContent = "COPY FAILED"; }
    });
    body.append(copy);
  }
  const usage = usageText(message.usage || message.token_usage);
  if (usage) {
    const details = document.createElement("div");
    details.className = "message-usage";
    details.textContent = usage;
    body.append(details);
  }
  article.append(body);
  return article;
}

// A rebuilt turn gets fresh <img> elements, and a fresh element has no bitmap until its own
// load resolves — so a card that was on screen a moment ago goes back to being an empty frame,
// or to painting its alt text if that load fails. The streaming turn is rebuilt on every
// animation frame, which made that race run sixty times a second. Carry the elements that
// already hold a picture into the new node instead. Only those: one still loading, or one that
// failed, is left behind so the fresh element gets another attempt.
function adoptLoadedImages(previous, next) {
  const loaded = new Map();
  for (const img of previous.querySelectorAll("img")) {
    const source = img.getAttribute("src");
    if (source && img.naturalWidth > 0 && !loaded.has(source)) loaded.set(source, img);
  }
  if (!loaded.size) return;
  for (const img of next.querySelectorAll("img")) {
    const live = loaded.get(img.getAttribute("src"));
    if (!live) continue;
    loaded.delete(img.getAttribute("src"));
    // The record can gain dimensions between rebuilds, so the fresh element's sizing wins.
    live.alt = img.alt;
    live.style.minHeight = img.style.minHeight;
    if (img.hasAttribute("width")) live.width = img.width;
    if (img.hasAttribute("height")) live.height = img.height;
    img.replaceWith(live);
  }
}

export function renderMessages() {
  const list = $("#messageList");
  const next = document.createDocumentFragment();
  for (const message of state.chatMessages) next.append(renderMessage(message));
  state.streamNode = null;
  if (state.streamMessage) {
    state.streamNode = renderMessage(state.streamMessage, { streaming: state.chatStreaming });
    next.append(state.streamNode);
  }
  adoptLoadedImages(list, next);
  list.replaceChildren(next);
  afterMessagesRendered();
}

function sameAttributes(a, b) {
  if (a.attributes.length !== b.attributes.length) return false;
  for (const attribute of a.attributes) if (b.getAttribute(attribute.name) !== attribute.value) return false;
  return true;
}

// Handing the column a whole new node also throws away the state the browser keeps outside the
// markup, and the hover that lights the copy button and the usage row is part of it: sixty
// rebuilds a second left that row strobing for the length of a turn. Splice in only the run of
// children whose markup actually changed, matched from both ends so a block growing in the
// middle does not count as a change to everything below it.
function patchElement(current, next) {
  if (current.tagName !== next.tagName || !sameAttributes(current, next)) return false;
  // A node whose children are not all elements carries text of its own, and a leaf that got
  // this far differs in that text: neither can be reconciled from the child list alone.
  if (!current.children.length || !next.children.length) return false;
  if (current.childNodes.length !== current.children.length) return false;
  if (next.childNodes.length !== next.children.length) return false;
  const mine = [...current.children];
  const theirs = [...next.children];
  let head = 0;
  while (head < mine.length && head < theirs.length && mine[head].outerHTML === theirs[head].outerHTML) head += 1;
  let tail = 0;
  while (head + tail < mine.length && head + tail < theirs.length
    && mine[mine.length - 1 - tail].outerHTML === theirs[theirs.length - 1 - tail].outerHTML) tail += 1;
  const stale = mine.slice(head, mine.length - tail);
  const fresh = theirs.slice(head, theirs.length - tail);
  if (stale.length === 1 && fresh.length === 1 && patchElement(stale[0], fresh[0])) return true;
  const before = mine[mine.length - tail] || null;
  for (const node of stale) node.remove();
  for (const node of fresh) current.insertBefore(node, before);
  return true;
}

// Only the streaming turn changes between tokens. Replacing that one node leaves the
// rest of the transcript untouched, so scroll position and text selection survive.
function renderStreamMessage() {
  if (!state.streamMessage || !state.streamNode?.isConnected) {
    renderMessages();
    return;
  }
  const next = renderMessage(state.streamMessage, { streaming: state.chatStreaming });
  adoptLoadedImages(state.streamNode, next);
  if (!patchElement(state.streamNode, next)) {
    state.streamNode.replaceWith(next);
    state.streamNode = next;
  }
  afterMessagesRendered();
}

function afterMessagesRendered() {
  if (state.chatFollowing) scrollChatToBottom(false);
  else $("#jumpLatest").hidden = false;
}

export function scheduleMessagesRender({ streamOnly = false } = {}) {
  if (!streamOnly) state.streamRenderOnly = false;
  if (state.messageRenderFrame != null) return;
  if (streamOnly) state.streamRenderOnly = true;
  state.messageRenderFrame = requestAnimationFrame(() => {
    state.messageRenderFrame = null;
    const onlyStream = state.streamRenderOnly;
    state.streamRenderOnly = false;
    if (onlyStream) renderStreamMessage();
    else renderMessages();
  });
}

export function appendLocalMessage(text, role = "system") {
  state.chatMessages.push({ role, content: text, local: true });
  renderMessages();
}

const CHAT_FOLLOW_SLACK = 56;

function chatDistanceFromBottom() {
  const scroller = $("#messageScroller");
  return scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight;
}

function latestMessageScrollTarget() {
  const scroller = $("#messageScroller");
  // The bottom spacer already parks the newest content clear of the composer, so the
  // follow target is simply the end of the scroller. A fixed gap under the live text
  // means the target stops sliding as the response grows.
  return Math.max(0, scroller.scrollHeight - scroller.clientHeight);
}

export function scrollChatToBottom(smooth = true) {
  const scroller = $("#messageScroller");
  state.chatFollowing = true;
  state.autoScrolling = true;
  state.lastScrollHeight = scroller.scrollHeight;
  scroller.scrollTo({ top: latestMessageScrollTarget(), behavior: smooth ? "smooth" : "auto" });
  state.lastScrollTop = scroller.scrollTop;
  requestAnimationFrame(() => { state.autoScrolling = false; });
  $("#jumpLatest").hidden = true;
}

// Following is released by an actual upward move, not by inferring intent from a
// position that the growing response keeps changing underneath the reader.
function releaseChatFollow() {
  if (!state.chatFollowing) return;
  state.chatFollowing = false;
  $("#jumpLatest").hidden = false;
}

export function initChatRendering() {
  $("#jumpLatest").addEventListener("click", () => scrollChatToBottom());
  $("#messageScroller").addEventListener("scroll", () => {
    const scroller = $("#messageScroller");
    const top = scroller.scrollTop;
    const previous = state.lastScrollTop;
    state.lastScrollTop = top;
    // A turn shrinks its own column repeatedly: the progress bar goes when the image lands, the
    // waiting heartbeat comes and goes. Each shrink clamps scrollTop toward the bottom and fires
    // this event, and reading that as "they are at the bottom" re-arms follow and steals the view
    // back from a reader who had scrolled up. Only a scroll against stable content says that.
    const height = scroller.scrollHeight;
    const shrank = height < state.lastScrollHeight;
    state.lastScrollHeight = height;
    if (!shrank && chatDistanceFromBottom() <= CHAT_FOLLOW_SLACK) {
      state.chatFollowing = true;
      $("#jumpLatest").hidden = true;
      return;
    }
    // A scrollbar drag or a momentum fling upward is the same intent as a wheel up.
    if (top < previous - 1 && !state.autoScrolling) state.chatFollowing = false;
    $("#jumpLatest").hidden = state.chatFollowing;
  }, { passive: true });
  $("#messageScroller").addEventListener("wheel", (event) => {
    if (event.deltaY < 0) releaseChatFollow();
  }, { passive: true });
  $("#messageScroller").addEventListener("touchstart", (event) => {
    state.touchAnchorY = event.touches[0]?.clientY ?? 0;
  }, { passive: true });
  $("#messageScroller").addEventListener("touchmove", (event) => {
    const y = event.touches[0]?.clientY ?? 0;
    if (y > (state.touchAnchorY ?? y) + 4) releaseChatFollow();
    state.touchAnchorY = y;
  }, { passive: true });
  $("#messageScroller").addEventListener("keydown", (event) => {
    if (["ArrowUp", "PageUp", "Home"].includes(event.key)) releaseChatFollow();
  });
}
