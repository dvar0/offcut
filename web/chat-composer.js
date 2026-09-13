import { loadBoardChats, openChat, renameChat, renderConversationHeader, sendChatTurn } from "./chat.js";
import { appendLocalMessage, createDotMatrix, messageBlocks, messageImages } from "./chat-render.js";
import { knownCapabilities } from "./connections.js";
import { setAgentGenerating } from "./generation.js";
import { gridColumnWidth, thumbnailUrl } from "./image-layout.js";
import { loadBoardImages } from "./images.js";
import { $, api, lookupImage, rememberImage, showChatNotice, state } from "./shared.js";

export function renderComposerContext() {
  const context = $("#composerContext");
  if (!context) return;
  context.replaceChildren();
  const currentPending = state.currentImage && state.chatAttachments.includes(state.currentImage.id);
  if (currentPending) {
    const img = document.createElement("img");
    img.src = thumbnailUrl(state.currentImage, 34);
    img.alt = "";
    const label = document.createElement("span");
    const strong = document.createElement("b");
    const imageId = state.currentImage.id || "";
    const hint = state.currentImage.raw_prompt || state.currentImage.final_prompt || "Selected board frame";
    strong.textContent = "Pending current image";
    label.append(strong, document.createElement("br"), `${hint.slice(0, 72)} · ${imageId.slice(0, 8)}`);
    label.title = imageId;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.ariaLabel = "Remove current image from the next turn";
    remove.addEventListener("click", () => removeChatAttachment(imageId));
    context.append(img, label, remove);
  } else {
    const label = document.createElement("span");
    label.textContent = state.currentImage ? "Current image is not attached to the next turn." : "No selected image. Board settings are still included.";
    context.append(label);
  }
  renderAttachmentTray();
  updateAttachmentControls();
}

function addChatAttachment(imageId) {
  if (!attachmentsAllowed()) {
    $("#composerStatus").textContent = "THIS MODEL DOES NOT ACCEPT IMAGE ATTACHMENTS";
    return;
  }
  if (!imageId || state.chatAttachments.includes(imageId)) {
    $("#composerStatus").textContent = imageId ? "IMAGE ALREADY PENDING" : "";
    return;
  }
  if (state.chatAttachments.length >= 8) {
    $("#composerStatus").textContent = "MAXIMUM 8 IMAGES PER TURN";
    return;
  }
  if (!lookupImage(imageId)) {
    $("#composerStatus").textContent = "THAT IMAGE IS NO LONGER AVAILABLE";
    return;
  }
  state.chatAttachments.push(imageId);
  $("#composerStatus").textContent = "";
  renderComposerContext();
}

export function removeChatAttachment(imageId) {
  state.chatAttachments = state.chatAttachments.filter((id) => id !== imageId);
  renderComposerContext();
}

export function attachmentsAllowed() {
  if (!state.currentChat) return true;
  return knownCapabilities(state.currentChat.connection_id, state.currentChat.model)?.vision !== false;
}

function updateAttachmentControls() {
  const allowed = attachmentsAllowed();
  const busy = state.chatStreaming;
  $("#attachCurrentImage").disabled = busy || !allowed || !state.currentImage || state.chatAttachments.includes(state.currentImage?.id) || state.chatAttachments.length >= 8;
  $("#uploadImages").disabled = busy || !allowed || state.chatAttachments.length >= 8;
  $("#imageUploadInput").disabled = busy || !allowed || state.chatAttachments.length >= 8;
  $("#pickFromBoard").disabled = busy || !allowed || state.chatAttachments.length >= 8;
  if (!allowed && !busy) $("#composerStatus").textContent = "THIS MODEL DOES NOT ACCEPT IMAGE ATTACHMENTS";
}

function renderAttachmentTray() {
  const tray = $("#attachmentTray");
  if (!tray) return;
  state.chatAttachments = [...new Set(state.chatAttachments)].filter((id) => lookupImage(id)).slice(0, 8);
  tray.replaceChildren();
  tray.hidden = !state.chatAttachments.some((id) => id !== state.currentImage?.id);
  for (const imageId of state.chatAttachments) {
    if (imageId === state.currentImage?.id) continue;
    const image = lookupImage(imageId);
    const item = document.createElement("div");
    item.className = "attachment";
    const img = document.createElement("img");
    img.src = thumbnailUrl(image, 42);
    img.alt = "Attached frame";
    img.title = image.raw_prompt || image.id;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.ariaLabel = `Remove image ${image.id}`;
    remove.addEventListener("click", () => {
      removeChatAttachment(imageId);
    });
    item.append(img, remove);
    tray.append(item);
  }
}

async function uploadReferenceImages(files) {
  const acceptedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
  const candidates = [...files].filter((file) => file && acceptedTypes.has(file.type));
  if (!attachmentsAllowed()) return updateAttachmentControls();
  if (!candidates.length) {
    $("#composerStatus").textContent = "USE JPEG, PNG, OR WEBP IMAGES";
    return;
  }
  const tooLarge = candidates.find((file) => file.size > 20 * 1024 * 1024);
  if (tooLarge) {
    $("#composerStatus").textContent = `${tooLarge.name || "IMAGE"} EXCEEDS 20MB`;
    return;
  }
  const available = 8 - state.chatAttachments.length;
  if (available <= 0) {
    $("#composerStatus").textContent = "MAXIMUM 8 IMAGES PER TURN";
    return;
  }
  const uploading = candidates.slice(0, available);
  if (uploading.length < candidates.length) $("#composerStatus").textContent = `UPLOADING ${uploading.length}; MAXIMUM 8 IMAGES PER TURN`;
  else $("#composerStatus").textContent = `UPLOADING ${uploading.length} IMAGE${uploading.length === 1 ? "" : "S"}`;
  const uploadedIds = [];
  try {
    for (const [index, file] of uploading.entries()) {
      $("#composerStatus").textContent = `UPLOADING ${index + 1} / ${uploading.length}`;
      const response = await fetch(`/api/reference-images?board_id=${encodeURIComponent(state.currentBoardId)}&label=${encodeURIComponent(file.name || `pasted-image-${index + 1}`)}`, {
        method: "POST",
        headers: { "Content-Type": file.type },
        body: file,
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || `Upload failed (${response.status})`);
      const image = data.image || data;
      if (!image.id) throw new Error("Upload response did not include an image ID");
      uploadedIds.push(image.id);
    }
    await loadBoardImages(null);
    for (const id of uploadedIds) addChatAttachment(id);
    $("#composerStatus").textContent = `${uploadedIds.length} IMAGE${uploadedIds.length === 1 ? "" : "S"} READY`;
  } catch (error) {
    if (uploadedIds.length) {
      await loadBoardImages(null);
      for (const id of uploadedIds) addChatAttachment(id);
    }
    $("#composerStatus").textContent = `UPLOAD FAILED: ${error.message}`;
  } finally {
    $("#imageUploadInput").value = "";
  }
}

export function resizeChatInput() {
  const input = $("#chatInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
}

function formatTurnElapsed(milliseconds) {
  const seconds = Math.floor(Math.max(0, milliseconds) / 1000);
  if (seconds < 60) return `${seconds}S`;
  return `${Math.floor(seconds / 60)}M ${String(seconds % 60).padStart(2, "0")}S`;
}

export function toolPhaseLabel(tool) {
  return String(tool.name || tool.tool_name || "TOOL").replaceAll("_", " ").toUpperCase();
}

// The composer keeps a running account of what the model is doing. It sits in a fixed
// spot, so it stays readable while scrolled back through the transcript.
function paintTurnStatus() {
  if (!state.chatStreaming) return;
  const elapsed = formatTurnElapsed(Date.now() - state.turnStartedAt);
  const label = document.createElement("span");
  label.textContent = `${state.turnPhase || "WORKING"} · ${elapsed}`;
  $("#composerStatus").replaceChildren(createDotMatrix(state.turnStartedAt, { small: true }), label);
}

export function setTurnPhase(phase) {
  state.turnPhase = phase;
  paintTurnStatus();
}

export function setChatStreaming(active) {
  state.chatStreaming = active;
  // Pinned for the life of the turn: its events describe the board it was sent from, and the
  // user is free to be looking at another one by the time they arrive.
  state.streamBoardId = active ? state.currentBoardId : null;
  $("#sendChat").hidden = active;
  $("#stopChat").hidden = !active;
  $("#stopChat").disabled = false;
  // The input stays enabled while a turn streams so a follow-up can be drafted
  // mid-reply. Sending is still blocked until the turn settles (the server
  // rejects overlapping turns), but the draft is kept.
  const input = $("#chatInput");
  input.disabled = false;
  if (!input.dataset.placeholder) input.dataset.placeholder = input.placeholder || "";
  input.placeholder = active ? "Type your next message…" : (input.dataset.placeholder || "Message this board…");
  $("#chatBrief").disabled = active;
  $("#chatModel").disabled = active;
  $("#chatReasoning").disabled = active || $("#chatReasoning").hidden;
  $("#composerProgress").hidden = !active;
  $("#chatComposer").classList.toggle("streaming", active);
  window.clearInterval(state.turnPhaseTimer);
  state.turnPhaseTimer = null;
  if (active) {
    state.turnStartedAt = Date.now();
    state.turnPhase = "SENDING";
    paintTurnStatus();
    state.turnPhaseTimer = window.setInterval(paintTurnStatus, 1000);
  } else {
    // A turn that stops, errors, or is aborted mid-generation never sends the completion
    // event, so release the left column here rather than leaving it stuck on GENERATING.
    setAgentGenerating(false);
    state.turnPhase = "";
    $("#composerStatus").replaceChildren();
  }
  updateAttachmentControls();
}

// One registry drives the executor and the "/" palette together, so a command can never be
// listed without working or work without being listed. Commands the server owns (/compact,
// skills) carry no run function and fall through to the turn as ordinary message text.
const CHAT_COMMANDS = [
  { name: "/new", summary: "Start a fresh chat on this board" },
  { name: "/clear", summary: "Start a fresh chat on this board" },
  { name: "/compact", args: "[focus]", summary: "Summarize the conversation so far and continue with less context" },
  { name: "/context", summary: "Show the board, selection, pending attachments, and mode" },
  { name: "/rename", args: "<name>", summary: "Rename this chat" },
];

function availableCommands() {
  return [
    ...CHAT_COMMANDS,
    ...state.skills.map((skill) => ({ name: `/${skill.name}`, summary: skill.description, skill: true })),
  ];
}

export async function runClientCommand(message) {
  const [command, ...rest] = message.trim().split(/\s+/);
  if (command === "/rename") {
    if (!rest.length) appendLocalMessage("Use /rename followed by a chat name.");
    else await renameChat(state.currentChat, rest.join(" "));
    return true;
  }
  if (command === "/context") {
    const board = state.boards.find((item) => item.id === state.currentBoardId);
    const image = state.currentImage ? `${state.currentImage.id} (${state.currentImage.raw_prompt || "no prompt"})` : "none";
    const extras = state.chatAttachments.length ? state.chatAttachments.join(", ") : "none";
    appendLocalMessage(`Current context\n\nBoard: ${board?.name || state.currentBoardId}\nSelected image: ${image}\nPending attachments: ${extras}\nMode: ${"create"}\nWorkspace revision: ${state.boardRevision}`);
    return true;
  }
  if (command === "/new" || command === "/clear") {
    const chat = state.currentChat;
    try {
      const result = await api("/api/chats", {
        method: "POST",
        body: JSON.stringify({ board_id: state.currentBoardId, connection_id: chat.connection_id, model: chat.model, permission_mode: "create", reasoning_effort: chat.reasoning_effort || null }),
      });
      await loadBoardChats();
      await openChat((result.chat || result).id);
    } catch (error) { appendLocalMessage(error.message); }
    return true;
  }
  return false;
}

// The palette and the board picker both want the strip directly above the textarea, so they
// share one slot and only one can hold it. Anything that opens closes whatever was there.
export function closeComposerOverlay() {
  if (!state.overlay) return;
  state.overlay = null;
  state.commandQuery = null;
  const overlay = $("#composerOverlay");
  overlay.hidden = true;
  overlay.replaceChildren();
}

function openComposerOverlay(kind) {
  if (state.overlay && state.overlay !== kind) closeComposerOverlay();
  state.overlay = kind;
  const overlay = $("#composerOverlay");
  overlay.hidden = false;
  overlay.dataset.kind = kind;
  return overlay;
}

// Only while the whole field is still one unbroken "/token": once a space is typed the user is
// writing arguments, and a list that keeps hovering over the text is in the way.
function commandQuery() {
  const value = $("#chatInput").value;
  const match = /^\/(\S*)$/.exec(value);
  return match ? match[1].toLowerCase() : null;
}

function syncCommandPalette() {
  const query = commandQuery();
  if (query === null) {
    if (state.overlay === "commands") closeComposerOverlay();
    return;
  }
  const matches = availableCommands().filter((command) => command.name.slice(1).toLowerCase().startsWith(query));
  if (!matches.length) {
    if (state.overlay === "commands") closeComposerOverlay();
    return;
  }
  if (state.commandQuery !== query) state.commandIndex = 0;
  state.commandQuery = query;
  state.commandMatches = matches;
  state.commandIndex = Math.max(0, Math.min(state.commandIndex, matches.length - 1));
  renderCommandPalette();
}

function renderCommandPalette() {
  const overlay = openComposerOverlay("commands");
  overlay.replaceChildren();
  const list = document.createElement("div");
  list.className = "command-list";
  list.setAttribute("role", "listbox");
  for (const [index, command] of state.commandMatches.entries()) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "command-row";
    row.setAttribute("role", "option");
    row.setAttribute("aria-selected", String(index === state.commandIndex));
    if (index === state.commandIndex) row.classList.add("active");
    const name = document.createElement("b");
    name.textContent = command.args ? `${command.name} ${command.args}` : command.name;
    const summary = document.createElement("span");
    summary.textContent = command.summary || "";
    row.append(name, summary);
    if (command.skill) {
      const tag = document.createElement("i");
      tag.textContent = "SKILL";
      row.append(tag);
    }
    // Pointer down, not click: the textarea must not lose focus before the value is rewritten.
    row.addEventListener("mousedown", (event) => { event.preventDefault(); acceptCommand(index); });
    row.addEventListener("mousemove", () => {
      if (state.commandIndex === index) return;
      state.commandIndex = index;
      renderCommandPalette();
    });
    list.append(row);
  }
  const hint = document.createElement("p");
  hint.className = "command-hint";
  hint.textContent = "UP DOWN TO NAVIGATE · ENTER TO SELECT · ESC TO DISMISS";
  overlay.append(list, hint);
}

function moveCommandSelection(delta) {
  const count = state.commandMatches.length;
  if (!count) return;
  state.commandIndex = (state.commandIndex + delta + count) % count;
  renderCommandPalette();
  const active = $("#composerOverlay").querySelector(".command-row.active");
  if (active) active.scrollIntoView({ block: "nearest" });
}

function acceptCommand(index = state.commandIndex) {
  const command = state.commandMatches[index];
  if (!command) return;
  const input = $("#chatInput");
  closeComposerOverlay();
  // A command that takes arguments leaves the caret after a trailing space so the next
  // keystroke is the argument; one that takes none is already a complete message.
  input.value = command.args ? `${command.name} ` : command.name;
  input.focus();
  input.setSelectionRange(input.value.length, input.value.length);
  resizeChatInput();
  if (!command.args) $("#chatComposer").requestSubmit();
}

export async function loadChatSkills() {
  try {
    const data = await api("/api/skills");
    state.skills = Array.isArray(data.skills) ? data.skills : [];
  } catch (_) {
    state.skills = [];
  }
}

export async function pickerImages(boardId) {
  if (state.pickerCache.has(boardId)) return state.pickerCache.get(boardId);
  const data = await api(`/api/images?board_id=${encodeURIComponent(boardId)}`);
  const images = data.images || [];
  state.pickerCache.set(boardId, images);
  return images;
}

async function openBoardPicker() {
  if (state.overlay === "board") return closeComposerOverlay();
  if (!attachmentsAllowed()) {
    $("#composerStatus").textContent = "THIS MODEL DOES NOT ACCEPT IMAGE ATTACHMENTS";
    return;
  }
  state.pickerBoardId = state.pickerBoardId || state.currentBoardId;
  state.pickerSelection = new Set();
  openComposerOverlay("board");
  await renderBoardPicker();
}

async function renderBoardPicker() {
  const overlay = openComposerOverlay("board");
  overlay.replaceChildren();

  const head = document.createElement("div");
  head.className = "picker-head";
  const select = document.createElement("select");
  select.ariaLabel = "Board to pick images from";
  for (const board of state.boards) {
    const option = document.createElement("option");
    option.value = board.id;
    option.textContent = `${board.name}${board.image_count ? ` (${board.image_count})` : ""}`;
    if (board.id === state.pickerBoardId) option.selected = true;
    select.append(option);
  }
  // The selection is deliberately not cleared here: picking two frames from one board and a
  // third from another is the reason this exists.
  select.addEventListener("change", async () => {
    state.pickerBoardId = select.value;
    await renderBoardPicker();
  });
  const close = document.createElement("button");
  close.type = "button";
  close.className = "picker-close";
  close.textContent = "×";
  close.ariaLabel = "Close the board picker";
  close.addEventListener("click", closeComposerOverlay);
  head.append(select, close);

  const grid = document.createElement("div");
  grid.className = "picker-grid";
  const loading = document.createElement("p");
  loading.className = "picker-empty";
  loading.textContent = "LOADING";
  grid.append(loading);

  const foot = document.createElement("div");
  foot.className = "picker-foot";
  const count = document.createElement("span");
  const confirm = document.createElement("button");
  confirm.type = "button";
  confirm.className = "picker-confirm";
  const refreshFoot = () => {
    const chosen = state.pickerSelection.size;
    const room = Math.max(0, 8 - state.chatAttachments.length);
    count.textContent = chosen ? `${chosen} SELECTED · ${room} SLOT${room === 1 ? "" : "S"} FREE` : `${room} SLOT${room === 1 ? "" : "S"} FREE`;
    confirm.textContent = chosen ? `ADD ${chosen}` : "ADD";
    confirm.disabled = !chosen || chosen > room;
  };
  confirm.addEventListener("click", () => {
    for (const id of state.pickerSelection) {
      if (state.chatAttachments.length >= 8) break;
      addChatAttachment(id);
    }
    closeComposerOverlay();
  });
  foot.append(count, confirm);
  refreshFoot();
  overlay.append(head, grid, foot);

  let images = [];
  try {
    images = await pickerImages(state.pickerBoardId);
  } catch (error) {
    grid.replaceChildren();
    const failed = document.createElement("p");
    failed.className = "picker-empty";
    failed.textContent = error.message;
    grid.append(failed);
    return;
  }
  if (state.overlay !== "board") return;
  const cellWidth = gridColumnWidth(grid, 74);
  grid.replaceChildren();
  if (!images.length) {
    const empty = document.createElement("p");
    empty.className = "picker-empty";
    empty.textContent = "THIS BOARD HAS NO IMAGES";
    grid.append(empty);
    return;
  }
  for (const image of images) {
    rememberImage(image);
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "picker-cell";
    if (state.pickerSelection.has(image.id)) cell.classList.add("chosen");
    if (state.chatAttachments.includes(image.id)) {
      cell.classList.add("attached");
      cell.disabled = true;
      cell.title = "Already attached to this turn";
    }
    cell.setAttribute("aria-pressed", String(state.pickerSelection.has(image.id)));
    const img = document.createElement("img");
    img.src = thumbnailUrl(image, cellWidth);
    img.alt = "Board frame";
    img.loading = "lazy";
    const mark = document.createElement("i");
    cell.append(img, mark);
    cell.addEventListener("click", () => {
      if (state.pickerSelection.has(image.id)) state.pickerSelection.delete(image.id);
      else state.pickerSelection.add(image.id);
      cell.classList.toggle("chosen", state.pickerSelection.has(image.id));
      cell.setAttribute("aria-pressed", String(state.pickerSelection.has(image.id)));
      refreshFoot();
    });
    grid.append(cell);
  }
}

function briefImages() {
  const ids = new Set(state.boardImages.map((image) => image.id));
  const brief = state.currentChat?.creative_brief || {};
  for (const id of [brief.best_image_id, brief.approved_image_id, ...(brief.references || []).map((r) => r.image_id)]) if (id) ids.add(id);
  for (const message of state.chatMessages) {
    for (const image of messageImages(message)) ids.add(image.image_id || image.id);
    for (const block of messageBlocks(message)) {
      if (block.tool?.image_id) ids.add(block.tool.image_id);
      for (const id of block.tool?.comparison_image_ids || []) ids.add(id);
    }
  }
  return [...ids].filter(Boolean).map((id) => lookupImage(id) || { id });
}

function fillBriefImageSelect(select, selected = null) {
  select.replaceChildren(new Option("None", ""));
  for (const image of briefImages()) {
    const description = (image.raw_prompt || image.final_prompt || "Frame").slice(0, 65);
    select.add(new Option(`${description}${image.width ? ` · ${image.width}×${image.height}` : ""}`, image.id));
  }
  if (selected && ![...select.options].some((option) => option.value === selected)) select.add(new Option(`${selected.slice(0, 8)} · unavailable frame`, selected));
  select.value = selected || "";
  const preview = document.createElement("img");
  preview.className = "brief-frame-preview";
  preview.alt = "Reference frame";
  preview.width = 96; preview.height = 96;
  const paint = () => {
    preview.hidden = !select.value;
    if (select.value) preview.src = thumbnailUrl(lookupImage(select.value) || { id: select.value }, 96);
    else preview.removeAttribute("src");
  };
  select.parentElement?.querySelector(".brief-frame-preview")?.remove();
  select.after(preview);
  select.onchange = paint;
  paint();
}

function addBriefReference(reference = {}) {
  const list = $("#briefReferences");
  if (list.children.length >= 8) return;
  const row = document.createElement("div");
  row.className = "brief-reference";
  const select = document.createElement("select");
  select.ariaLabel = "Reference image";
  const purpose = document.createElement("input");
  purpose.placeholder = "Purpose: layout, tower detail, art style…";
  purpose.ariaLabel = "Reference purpose";
  purpose.maxLength = 200;
  purpose.value = reference.purpose || "";
  const remove = document.createElement("button");
  remove.type = "button"; remove.textContent = "×"; remove.ariaLabel = "Remove reference pin";
  remove.addEventListener("click", () => row.remove());
  row.append(select, purpose, remove); list.append(row);
  fillBriefImageSelect(select, reference.image_id || state.currentImage?.id);
}

export async function openBriefDialog() {
  if (!state.currentChat || state.chatStreaming) return;
  const chatId = state.currentChat.id;
  try {
    const data = await api(`/api/chats/${encodeURIComponent(chatId)}`);
    if (state.currentChat?.id !== chatId) return;
    Object.assign(state.currentChat, data.chat);
    const brief = data.chat.creative_brief || {};
    $("#briefGoal").value = brief.goal || "";
    $("#briefKeep").value = (brief.must_keep || []).join("\n");
    $("#briefTradeoffs").value = (brief.accepted_tradeoffs || []).join("\n");
    $("#briefNext").value = brief.next_change || "";
    $("#briefFailed").value = (brief.failed_approaches || []).join("\n");
    $("#briefLimit").value = data.chat.generation_limit || "";
    fillBriefImageSelect($("#briefApproved"), brief.approved_image_id);
    fillBriefImageSelect($("#briefBest"), brief.best_image_id);
    $("#briefReferences").replaceChildren();
    for (const reference of brief.references || []) addBriefReference(reference);
    const attempts = $("#briefAttempts"); attempts.replaceChildren();
    for (const attempt of data.attempts || []) {
      const row = document.createElement("p");
      row.className = "style-guidance";
      row.textContent = `#${attempt.id} · ${attempt.outcome}${attempt.reused ? " · reused frame" : ""} · ${attempt.change_note || "Generation"}${attempt.observation ? " — " + attempt.observation : ""}`;
      attempts.append(row);
    }
    $("#briefError").textContent = "";
    $("#briefDialog").dataset.chatId = chatId;
    $("#briefDialog").showModal();
  } catch (error) { showChatNotice(error.message); }
}

export function initChatComposer() {
  $("#chatComposer").addEventListener("submit", sendChatTurn);
  $("#chatInput").addEventListener("input", () => { resizeChatInput(); syncCommandPalette(); });
  $("#chatInput").addEventListener("keydown", (event) => {
    // The palette owns the arrow keys, Enter, and Escape while it is open; without that, Enter
    // would send a bare "/re" instead of completing it.
    if (state.overlay === "commands" && !event.isComposing) {
      if (event.key === "ArrowDown") { event.preventDefault(); return moveCommandSelection(1); }
      if (event.key === "ArrowUp") { event.preventDefault(); return moveCommandSelection(-1); }
      if (event.key === "Enter" || event.key === "Tab") { event.preventDefault(); return acceptCommand(); }
      if (event.key === "Escape") { event.preventDefault(); return closeComposerOverlay(); }
    }
    if (state.overlay === "board" && event.key === "Escape") {
      event.preventDefault();
      return closeComposerOverlay();
    }
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("#chatComposer").requestSubmit();
    }
  });
  $("#pickFromBoard").addEventListener("click", () => { openBoardPicker(); });
  $("#attachCurrentImage").addEventListener("click", () => {
    if (state.currentImage) addChatAttachment(state.currentImage.id);
    else $("#composerStatus").textContent = "SELECT AN IMAGE FIRST";
  });
  $("#uploadImages").addEventListener("click", () => $("#imageUploadInput").click());
  $("#imageUploadInput").addEventListener("change", (event) => uploadReferenceImages(event.target.files));
  $("#chatInput").addEventListener("paste", (event) => {
    const files = [...(event.clipboardData?.files || [])].filter((file) => file.type.startsWith("image/"));
    if (!files.length) return;
    event.preventDefault();
    uploadReferenceImages(files);
  });
  for (const type of ["dragenter", "dragover"]) {
    $("#chatComposer").addEventListener(type, (event) => {
      const types = Array.from(event.dataTransfer.types);
      if (!types.includes("application/x-offcut-image-id") && !types.includes("Files")) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      $("#chatComposer").classList.add("drag-over");
    });
  }
  $("#chatComposer").addEventListener("dragleave", (event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) event.currentTarget.classList.remove("drag-over");
  });
  $("#chatComposer").addEventListener("drop", async (event) => {
    const imageId = event.dataTransfer.getData("application/x-offcut-image-id");
    const files = [...(event.dataTransfer.files || [])];
    if (!imageId && !files.length) return;
    event.preventDefault();
    $("#chatComposer").classList.remove("drag-over");
    if (imageId) addChatAttachment(imageId);
    else await uploadReferenceImages(files);
  });
  $("#briefAddReference").addEventListener("click", () => addBriefReference());
  $("#closeBrief").addEventListener("click", () => $("#briefDialog").close());
  $("#briefForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.chatStreaming || state.currentChat?.id !== $("#briefDialog").dataset.chatId) return;
    const lines = (id) => $(id).value.split("\n").map((line) => line.trim()).filter(Boolean);
    const references = [...$("#briefReferences").children].map((row) => ({ image_id: row.querySelector("select").value, purpose: row.querySelector("input").value.trim() }));
    const patch = { generation_limit: Number($("#briefLimit").value), creative_brief: {
      goal: $("#briefGoal").value, must_keep: lines("#briefKeep"), accepted_tradeoffs: lines("#briefTradeoffs"),
      next_change: $("#briefNext").value, failed_approaches: lines("#briefFailed"),
      approved_image_id: $("#briefApproved").value || null, best_image_id: $("#briefBest").value || null, references,
    }};
    try {
      const data = await api(`/api/chats/${encodeURIComponent(state.currentChat.id)}`, { method: "POST", body: JSON.stringify(patch) });
      Object.assign(state.currentChat, data.chat);
      renderConversationHeader();
      $("#briefDialog").close();
    } catch (error) { $("#briefError").textContent = error.message; }
  });
}
