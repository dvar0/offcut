import { renderMessages } from "./chat-render.js";
import { applySettings, generationPayload } from "./generation.js";
import { $, api, showChatNotice, state } from "./shared.js";

export function renderPromptChange(change) {
  change = {
    ...change,
    before: change.before ?? change.removed ?? "",
    after: change.after ?? change.added ?? "",
  };
  const block = document.createElement("div");
  block.className = "prompt-change";
  const heading = document.createElement("div");
  const title = document.createElement("b");
  title.textContent = "PROMPT CHANGE";
  const undo = document.createElement("button");
  undo.type = "button";
  undo.textContent = "UNDO";
  const currentPrompt = $("#prompt").value;
  undo.disabled = currentPrompt !== change.after;
  undo.title = undo.disabled ? "The prompt has changed since this edit" : "Restore the previous prompt";
  undo.addEventListener("click", () => undoPromptChange(change, undo));
  heading.append(title, undo);
  const removed = document.createElement("pre");
  removed.className = "prompt-removed";
  removed.textContent = `- ${change.removed ?? change.before ?? ""}`;
  const added = document.createElement("pre");
  added.className = "prompt-added";
  added.textContent = `+ ${change.added ?? change.after ?? ""}`;
  block.append(heading, removed, added);
  return block;
}

// Prompt diffs are computed here rather than taken from the payload's flat removed/added strings,
// because the highlight needs to know *where* in the new prompt each change landed. Both producers
// of prompt_change always send the full before/after text, so the client has everything it needs.
const PROMPT_DIFF_TOKEN_LIMIT = 800;

const PROMPT_DIFF_BRIDGE_CHARS = 24;

const PROMPT_PEEK_WIDTH = 250;

// Words, not characters: a character diff splits mid-word and highlights unreadable fragments.
// Whitespace runs are tokens too, so every offset lands on a word boundary.
function tokenizePrompt(text) {
  const tokens = [];
  const pattern = /\s+|\S+/g;
  let match;
  while ((match = pattern.exec(text)) !== null) tokens.push({ text: match[0], start: match.index });
  return tokens;
}

function tokenOpcodes(a, b) {
  const shorter = Math.min(a.length, b.length);
  let head = 0;
  while (head < shorter && a[head].text === b[head].text) head += 1;
  let tail = 0;
  while (tail < shorter - head && a[a.length - 1 - tail].text === b[b.length - 1 - tail].text) tail += 1;
  const left = a.slice(head, a.length - tail);
  const right = b.slice(head, b.length - tail);
  const n = left.length;
  const m = right.length;
  if (!n && !m) return [];
  // A full rewrite leaves nothing to align, and the table would be enormous. One coarse hunk is
  // still truthful: everything in this span is new.
  if (n > PROMPT_DIFF_TOKEN_LIMIT || m > PROMPT_DIFF_TOKEN_LIMIT) {
    const op = n && m ? "replace" : n ? "delete" : "insert";
    return [{ op, aStart: head, aEnd: head + n, bStart: head, bEnd: head + m }];
  }
  const width = m + 1;
  const table = new Uint32Array((n + 1) * width);
  for (let i = n - 1; i >= 0; i -= 1) {
    for (let j = m - 1; j >= 0; j -= 1) {
      table[i * width + j] = left[i].text === right[j].text
        ? table[(i + 1) * width + j + 1] + 1
        : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
  }
  const runs = [];
  const emit = (op, aStart, aEnd, bStart, bEnd) => {
    const last = runs[runs.length - 1];
    if (last && last.op === op) {
      last.aEnd = aEnd;
      last.bEnd = bEnd;
      return;
    }
    runs.push({ op, aStart, aEnd, bStart, bEnd });
  };
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (left[i].text === right[j].text) {
      emit("equal", i, i + 1, j, j + 1);
      i += 1;
      j += 1;
    } else if (table[(i + 1) * width + j] >= table[i * width + j + 1]) {
      emit("delete", i, i + 1, j, j);
      i += 1;
    } else {
      emit("insert", i, i, j, j + 1);
      j += 1;
    }
  }
  if (i < n) emit("delete", i, n, j, j);
  if (j < m) emit("insert", i, i, j, m);
  // Left alone, LCS shatters a rewritten clause into a hunk per word, because it happily anchors
  // on the spaces and the odd short word that survived the rewrite. Reading five amber shards
  // with untouched words peeking between them is worse than reading the clause as one change, so
  // adjacent hunks absorb the gap between them when that gap is smaller than the changes either
  // side of it. Two bounds keep it honest: the relative one stops two small independent word
  // swaps from collapsing into one, and the absolute cap means no more than a couple of words of
  // untouched text can ever end up inside a highlight. Line breaks never bridge, so edits in
  // separate paragraphs stay separate.
  const spanWidth = (tokens, start, end) => (end > start ? tokens[end - 1].start + tokens[end - 1].text.length - tokens[start].start : 0);
  const size = (run) => Math.max(spanWidth(left, run.aStart, run.aEnd), spanWidth(right, run.bStart, run.bEnd));
  const merged = [];
  for (const run of runs) {
    if (run.op === "equal") continue;
    const previous = merged[merged.length - 1];
    if (previous) {
      const gap = right.slice(previous.bEnd, run.bStart).map((token) => token.text).join("");
      const bridgeable = Math.min(size(previous) + size(run), PROMPT_DIFF_BRIDGE_CHARS);
      if (!gap.includes("\n") && gap.trim().length < bridgeable) {
        previous.aEnd = run.aEnd;
        previous.bEnd = run.bEnd;
        continue;
      }
    }
    merged.push({ ...run });
  }
  for (const run of merged) {
    // Read the kind back off the spans: coalescing can turn a delete and an insert into a swap,
    // and can also join two inserts, so the original labels no longer describe the result.
    const cut = run.aEnd > run.aStart;
    const grew = run.bEnd > run.bStart;
    run.op = cut && grew ? "replace" : cut ? "delete" : "insert";
    run.aStart += head;
    run.aEnd += head;
    run.bStart += head;
    run.bEnd += head;
  }
  return merged;
}

// Hunks are expressed in `after` offsets: that is the text the highlight backdrop draws.
// A pure deletion collapses to start === end, the point where the wedge goes.
function diffPromptWords(before, after) {
  const a = tokenizePrompt(before);
  const b = tokenizePrompt(after);
  const span = (tokens, start, end, text) => {
    if (end <= start) return [start < tokens.length ? tokens[start].start : text.length, null];
    const from = tokens[start].start;
    const to = tokens[end - 1].start + tokens[end - 1].text.length;
    return [from, to];
  };
  const hunks = [];
  for (const run of tokenOpcodes(a, b)) {
    let [start, end] = span(b, run.bStart, run.bEnd, after);
    if (end === null) end = start;
    // Tokenizing on whitespace means a hunk usually carries a leading or trailing space. Trimming
    // it keeps the highlight tight around the words that actually changed.
    const slice = after.slice(start, end);
    const lead = slice.length - slice.trimStart().length;
    const trail = slice.length - slice.trimEnd().length;
    if (lead + trail < slice.length) {
      start += lead;
      end -= trail;
    }
    const [removedFrom, removedTo] = span(a, run.aStart, run.aEnd, before);
    const removed = removedTo === null ? "" : before.slice(removedFrom, removedTo).trim();
    if (start === end && !removed) continue;
    // A run whose new text trimmed away to nothing is a deletion whatever the opcode called it.
    hunks.push({ op: start === end ? "delete" : run.op, start, end, removed });
  }
  return hunks;
}

function promptPeek() {
  const wrap = $("#promptWrap");
  let peek = wrap.querySelector(".pc-peek");
  if (!peek) {
    peek = document.createElement("div");
    peek.className = "pc-peek";
    peek.hidden = true;
    wrap.append(peek);
  }
  return peek;
}

// A pinned peek was opened by a click and outlives the pointer, so hovering another wedge or
// leaving this one must not take it away; only another click, a step, or losing the diff can.
function showPromptPeek(anchor, label, text, duration = 0, pinned = false) {
  const peek = promptPeek();
  const heading = document.createElement("b");
  heading.textContent = label;
  peek.replaceChildren(heading, document.createTextNode(text));
  const box = anchor.getBoundingClientRect();
  const frame = $("#promptWrap").getBoundingClientRect();
  peek.style.left = `${Math.max(0, Math.min(box.left - frame.left, frame.width - PROMPT_PEEK_WIDTH))}px`;
  peek.style.top = `${box.bottom - frame.top + 6}px`;
  peek.classList.toggle("pc-pinned", pinned);
  peek.hidden = false;
  state.promptPeekPinned = pinned;
  window.clearTimeout(state.promptPeekTimer);
  if (duration) state.promptPeekTimer = window.setTimeout(() => hidePromptPeek(true), duration);
}

function hidePromptPeek(force = false) {
  if (state.promptPeekPinned && !force) return;
  window.clearTimeout(state.promptPeekTimer);
  state.promptPeekPinned = false;
  const peek = $("#promptWrap").querySelector(".pc-peek");
  if (peek) peek.hidden = true;
}

function renderPromptHighlight(text, hunks) {
  const host = $("#promptHighlight");
  host.replaceChildren();
  const marks = [];
  let cursor = 0;
  for (const hunk of hunks) {
    if (hunk.start > cursor) host.append(document.createTextNode(text.slice(cursor, hunk.start)));
    if (hunk.op === "delete") {
      const cut = document.createElement("span");
      cut.className = "pc-cut";
      const glyph = document.createElement("i");
      glyph.setAttribute("aria-hidden", "true");
      glyph.title = "Click to keep the removed text on screen";
      glyph.addEventListener("pointerenter", () => {
        // A pinned peek was asked for; drifting the pointer over a neighbouring wedge on the way
        // to the text must not swap it out from under the reader.
        if (!state.promptPeekPinned) showPromptPeek(glyph, "REMOVED", hunk.removed);
      });
      glyph.addEventListener("pointerleave", () => hidePromptPeek());
      // Hovering a wedge to read what went is fine for a word; it is useless for a removed
      // clause you want to keep in view while you retype it. A click leaves it up.
      glyph.addEventListener("click", () => {
        const wasPinned = state.promptPeekPinned;
        hidePromptPeek(true);
        if (!wasPinned) showPromptPeek(glyph, "REMOVED", hunk.removed, 0, true);
      });
      cut.append(glyph);
      host.append(cut);
      marks.push(cut);
    } else {
      const mark = document.createElement("mark");
      mark.className = hunk.op === "replace" ? "pc-swap" : "pc-add";
      mark.textContent = text.slice(hunk.start, hunk.end);
      host.append(mark);
      marks.push(mark);
    }
    cursor = Math.max(cursor, hunk.end);
  }
  // The trailing newline mirrors how a textarea reserves a line box after a final newline;
  // without it the backdrop is one line shorter and scrolls out of step at the bottom.
  host.append(document.createTextNode(`${text.slice(cursor)}\n`));
  return marks;
}

// The backdrop mirrors the textarea's box, but CSS cannot see the width a scrollbar steals from
// the text column. Without this the two copies of the prompt wrap at different words.
function syncPromptHighlightMetrics() {
  const prompt = $("#prompt");
  const gutter = Math.max(0, prompt.offsetWidth - prompt.clientWidth - 2);
  $("#promptHighlight").style.paddingRight = `${10 + gutter}px`;
  syncPromptHighlightScroll();
}

function syncPromptHighlightScroll() {
  $("#promptHighlight").scrollTop = $("#prompt").scrollTop;
}

// Drops what is painted, not the turn baseline: a mid-turn patch clears the old hunks on its way
// to drawing wider ones. Use forgetPromptTurn() when the turn's starting text stops being a
// truthful "before" — the user typed, the board changed, the edit was undone.
export function clearPromptDiff() {
  hidePromptPeek(true);
  state.latestPromptChange = null;
  state.promptDiffHunks = [];
  state.promptDiffMarks = [];
  state.promptDiffIndex = -1;
  $("#promptWrap").classList.remove("diff-active");
  $("#promptHighlight").replaceChildren();
  $("#promptSyncDiff").hidden = true;
}

export function forgetPromptTurn() {
  state.promptTurnBefore = null;
  clearPromptDiff();
}

// Each prompt-editing tool call arrives as its own workspace patch, so a turn that rewrites the
// prompt and then tidies one clause lands as two. Showing only the last one hid the rewrite
// behind the tidy-up, so the box anchors on the text as it stood when the turn first touched it
// and widens from there. The transcript still lists every step separately.
export function accumulatePromptTurnChange(step) {
  const before = state.promptTurnBefore ?? step.before ?? step.removed ?? "";
  state.promptTurnBefore = before;
  // The step's own `removed`/`added` summary describes one edit, not the widened span, so it is
  // dropped rather than carried forward; the bar counts words off the recomputed hunks.
  return { before, after: step.after ?? step.added ?? "", revision: step.revision, actor: step.actor };
}

function stepPromptDiff(direction) {
  const marks = state.promptDiffMarks;
  if (!marks.length) return;
  state.promptDiffIndex = (((state.promptDiffIndex + direction) % marks.length) + marks.length) % marks.length;
  const mark = marks[state.promptDiffIndex];
  const prompt = $("#prompt");
  // The backdrop shares the textarea's geometry, so a mark's offset within it is the scroll
  // position that brings the same words into view in the textarea.
  prompt.scrollTop = Math.max(0, mark.offsetTop - prompt.clientHeight / 2);
  syncPromptHighlightScroll();
  mark.classList.remove("pc-flash");
  void mark.offsetWidth;
  mark.classList.add("pc-flash");
  const hunk = state.promptDiffHunks[state.promptDiffIndex];
  showPromptCutText(hunk);
  if (hunk?.removed) showPromptPeek(mark, hunk.op === "delete" ? "REMOVED" : "REPLACED", hunk.removed, 2600, false);
  else hidePromptPeek(true);
}

// The peek is anchored to a mark and times out. The bar slot is the durable copy: whatever text
// this hunk took out stays readable there for as long as you are looking at that hunk. The full
// string goes in; the slot ellipsizes it to whatever the panel width allows, so a wider window
// simply shows more of it, and the tooltip and the peek both still hold all of it.
function showPromptCutText(hunk) {
  const slot = $("#promptSyncDiff").querySelector(".pc-cut-text");
  if (!slot) return;
  const flat = (hunk?.removed || "").replace(/\s+/g, " ").trim();
  slot.textContent = flat ? `− “${flat}”` : "";
  slot.hidden = !flat;
  slot.title = flat;
}

function countPromptWords(text) {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

function renderPromptDiffBar(change, hunks) {
  const bar = document.createElement("div");
  bar.className = "prompt-diff-bar";
  const title = document.createElement("b");
  title.textContent = "PROMPT CHANGE";
  // Characters counted the keystrokes, which told you nothing about the edit: swapping "red" for
  // "crimson" read as +7 \u22123. Words describe what the model actually did to the prompt.
  let added = 0;
  let removed = 0;
  for (const hunk of hunks) {
    added += countPromptWords(change.after.slice(hunk.start, hunk.end));
    removed += countPromptWords(hunk.removed);
  }
  const stat = document.createElement("span");
  stat.className = "pc-stat";
  for (const [count, sign, tone] of [[added, "+", "pc-plus"], [removed, "\u2212", "pc-minus"]]) {
    if (!count) continue;
    const part = document.createElement("span");
    part.className = tone;
    part.textContent = `${sign}${count}w`;
    stat.append(part);
  }
  const step = (label, direction, hint) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "pc-step";
    button.textContent = label;
    button.title = hint;
    button.disabled = state.promptDiffMarks.length < 2;
    button.addEventListener("click", () => stepPromptDiff(direction));
    return button;
  };
  const undo = document.createElement("button");
  undo.type = "button";
  undo.textContent = "UNDO";
  undo.disabled = $("#prompt").value !== change.after;
  undo.title = undo.disabled ? "The prompt has changed since this edit" : "Restore the previous prompt";
  undo.addEventListener("click", () => undoPromptChange(change, undo));
  const dismiss = document.createElement("button");
  dismiss.type = "button";
  dismiss.className = "pc-dismiss";
  dismiss.textContent = "✕";
  dismiss.title = "Dismiss this diff";
  dismiss.addEventListener("click", forgetPromptTurn);
  bar.append(title, stat, step("‹", -1, "Previous change"), step("›", 1, "Next change"), undo, dismiss);
  return bar;
}

// Its own row rather than a slot in the bar: the panel is a fixed width and the bar's controls
// already fill it, so quoted text there truncates to a few useless characters.
function renderPromptCutLine() {
  const line = document.createElement("div");
  line.className = "pc-cut-text";
  line.hidden = true;
  return line;
}

export function showPromptSyncChange(change) {
  const normalized = {
    ...change,
    before: change.before ?? change.removed ?? "",
    after: change.after ?? change.added ?? "",
  };
  hidePromptPeek(true);
  // The counts only need the two texts, so they are computed either way. Hunk *offsets* describe
  // `after` alone, so they are painted only when the box really holds that text; otherwise the
  // summary strip stands on its own rather than highlighting the wrong words.
  const hunks = diffPromptWords(normalized.before, normalized.after);
  const aligned = $("#prompt").value === normalized.after;
  state.latestPromptChange = normalized;
  state.promptDiffHunks = aligned ? hunks : [];
  state.promptDiffMarks = aligned && hunks.length ? renderPromptHighlight(normalized.after, hunks) : [];
  state.promptDiffIndex = -1;
  $("#promptWrap").classList.toggle("diff-active", state.promptDiffMarks.length > 0);
  if (!state.promptDiffMarks.length) $("#promptHighlight").replaceChildren();
  syncPromptHighlightMetrics();
  const host = $("#promptSyncDiff");
  host.replaceChildren(renderPromptDiffBar(normalized, hunks), renderPromptCutLine());
  host.hidden = false;
  // One removal has no ambiguity about which hunk the bar is quoting, so show it without making
  // the user step to it first. Several, and the slot waits until stepping picks one.
  const cuts = state.promptDiffHunks.filter((hunk) => hunk.removed);
  if (cuts.length === 1) showPromptCutText(cuts[0]);
  const section = $(".prompt-section");
  section.classList.remove("workspace-updated");
  void section.offsetWidth;
  section.classList.add("workspace-updated");
  window.clearTimeout(state.promptEmphasisTimer);
  state.promptEmphasisTimer = window.setTimeout(() => section.classList.remove("workspace-updated"), 1800);
}

async function undoPromptChange(change, button) {
  if (!state.currentBoardId || $("#prompt").value !== change.after) {
    showChatNotice("This prompt changed after that edit, so it cannot be undone safely.");
    renderMessages();
    return;
  }
  button.disabled = true;
  try {
    const settings = { ...generationPayload(), prompt: change.before };
    const result = await api(`/api/boards/${encodeURIComponent(state.currentBoardId)}`, {
      method: "POST",
      body: JSON.stringify({ settings, expected_settings_revision: state.boardRevision }),
    });
    state.boardRevision = Number(result.settings_revision ?? result.board?.settings_revision ?? state.boardRevision + 1);
    const board = state.boards.find((item) => item.id === state.currentBoardId);
    if (board) {
      board.settings = settings;
      board.settings_revision = state.boardRevision;
    }
    state.applyingSettings = true;
    applySettings(settings);
    state.applyingSettings = false;
    showChatNotice("Prompt edit undone.");
    forgetPromptTurn();
    renderMessages();
  } catch (error) {
    showChatNotice(`Could not undo prompt edit: ${error.message}`);
    renderMessages();
  }
}

export function initPromptDiff() {
  $("#prompt").addEventListener("input", (event) => {
    $("#characterCount").textContent = `${event.target.value.length} / 20K`;
    // Once the text moves the hunk offsets describe words that are no longer there, so the diff
    // goes rather than pointing at the wrong places. The turn baseline goes with it: after a hand
    // edit it would credit the model with the user's own words.
    if (state.latestPromptChange || state.promptTurnBefore !== null) forgetPromptTurn();
  });
  $("#prompt").addEventListener("scroll", syncPromptHighlightScroll);
  // A pinned peek ignores the pointer leaving its wedge, so putting the caret in the box is the
  // other way out of it — otherwise it would sit over the text until the diff itself went.
  $("#prompt").addEventListener("pointerdown", () => hidePromptPeek(true));
  // Catches both a drag on the textarea's resize handle and the scrollbar appearing or vanishing.
  new ResizeObserver(syncPromptHighlightMetrics).observe($("#prompt"));
}
