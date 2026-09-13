import {
  attachmentsAllowed,
  closeComposerOverlay,
  openBriefDialog,
  renderComposerContext,
  resizeChatInput,
  runClientCommand,
  setChatStreaming,
  setTurnPhase,
  toolPhaseLabel,
} from "./chat-composer.js";
import {
  appendLocalMessage,
  lastStreamBlock,
  messageBlocks,
  messageImages,
  messageText,
  messageUiId,
  openStreamBlock,
  renderMessages,
  scheduleMessagesRender,
  scrollChatToBottom,
  syncStreamMirrors,
} from "./chat-render.js";
import {
  capabilityKey,
  defaultChatConnection,
  keyedConnections,
  knownCapabilities,
  loadModelCapabilities,
  populateReasoningControl,
} from "./connections.js";
import {
  applySettings,
  generationPayload,
  markFresh,
  setAgentGenerating,
  updateProgress,
} from "./generation.js";
import { loadBoardImages, selectImage } from "./images.js";
import { loadStyles } from "./library.js";
import { accumulatePromptTurnChange, showPromptSyncChange } from "./prompt-diff.js";
import {
  $,
  api,
  findImage,
  formatChatDate,
  formatCost,
  lookupImage,
  rememberImage,
  showChatNotice,
  state,
} from "./shared.js";
import { flushBoardDraft, populateBoardSelectors, renderBoards } from "./workspace.js";

function renderConversationCapabilities() {
  const chat = state.currentChat;
  if (!chat) return null;
  const capabilities = knownCapabilities(chat.connection_id, chat.model);
  const effort = populateReasoningControl($("#chatReasoning"), capabilities, chat.reasoning_effort);
  renderComposerContext();
  return effort;
}

async function refreshConversationCapabilities(updateEffort = false) {
  const chat = state.currentChat;
  if (!chat) return;
  try {
    const capabilities = await loadModelCapabilities(chat.connection_id, chat.model);
    if (state.currentChat?.id !== chat.id || state.currentChat.model !== chat.model) return;
    const effort = populateReasoningControl($("#chatReasoning"), capabilities, chat.reasoning_effort);
    renderComposerContext();
    if (updateEffort && effort !== chat.reasoning_effort) await updateCurrentChat({ reasoning_effort: effort });
  } catch (error) {
    $("#composerStatus").textContent = `CAPABILITIES UNAVAILABLE: ${error.message}`;
  }
}

export function showChatList() {
  $("#chatListView").hidden = false;
  $("#conversation").hidden = true;
  state.chatFollowing = true;
}

export async function loadBoardChats(boardId = state.currentBoardId) {
  if (!boardId) return;
  try {
    const data = await api(`/api/chats?board_id=${encodeURIComponent(boardId)}`);
    if (boardId !== state.currentBoardId) return;
    state.chats = data.chats || [];
    showChatNotice("");
    renderChatList();
  } catch (error) {
    state.chats = [];
    renderChatList();
    showChatNotice(error.message);
  }
}

function chatPreview(chat) {
  return chat.preview || chat.last_message || chat.last_message_preview || chat.latest_preview || chat.summary || "No messages yet.";
}

function renderChatList() {
  const list = $("#chatList");
  list.replaceChildren();
  const board = state.boards.find((item) => item.id === state.currentBoardId);
  $("#chatBoardName").textContent = board?.name || "";
  $("#chatEmpty").hidden = state.chats.length > 0;
  for (const chat of state.chats) {
    const card = document.createElement("article");
    card.className = "chat-card";
    card.tabIndex = 0;
    const title = document.createElement("h3");
    title.textContent = chat.title || "Untitled chat";
    const preview = document.createElement("p");
    preview.textContent = chatPreview(chat);
    const meta = document.createElement("div");
    meta.className = "chat-card-meta";
    const date = document.createElement("span");
    date.textContent = formatChatDate(chat.updated_at || chat.created_at);
    const model = document.createElement("span");
    model.textContent = chat.model || "";
    meta.append(date, model);
    if (chat.running) {
      // The turn outlives the connection that started it, so a chat can still be working
      // when the list is drawn after a refresh.
      const running = document.createElement("span");
      running.className = "chat-card-running";
      running.textContent = "WORKING";
      meta.append(running);
    }
    const costValue = chat.cost ?? chat.usage?.cost ?? chat.total_cost;
    if (costValue !== undefined && costValue !== null) {
      const cost = document.createElement("span");
      cost.textContent = formatCost(costValue);
      meta.append(cost);
    }
    const actions = document.createElement("div");
    actions.className = "chat-card-actions";
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "RENAME";
    rename.ariaLabel = `Rename ${chat.title || "chat"}`;
    rename.addEventListener("click", (event) => { event.stopPropagation(); renameChat(chat); });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "DELETE";
    remove.ariaLabel = `Delete ${chat.title || "chat"}`;
    remove.addEventListener("click", (event) => { event.stopPropagation(); deleteChat(chat); });
    actions.append(rename, remove);
    card.append(title, preview, meta, actions);
    card.addEventListener("click", () => openChat(chat.id));
    card.addEventListener("keydown", (event) => { if (event.key === "Enter") openChat(chat.id); });
    list.append(card);
  }
}

// Model and reasoning effort are left out of the request on purpose: the server resolves both from
// the profile, so the default a model does not offer is corrected there rather than guessed here.
async function createChat() {
  const connection = defaultChatConnection();
  if (!connection) return showChatNotice("Add an API key and sync models in Connections before starting a chat.");
  showChatNotice("");
  try {
    const result = await api("/api/chats", {
      method: "POST",
      body: JSON.stringify({ board_id: state.currentBoardId, connection_id: connection.id, permission_mode: "create" }),
    });
    const chat = result.chat || result;
    localStorage.setItem("offcut.chatConnection", connection.id);
    await loadBoardChats();
    await openChat(chat.id);
  } catch (error) { showChatNotice(error.message); }
}

export async function openChat(chatId) {
  try {
    const data = await api(`/api/chats/${encodeURIComponent(chatId)}`);
    state.currentChat = data.chat || state.chats.find((chat) => chat.id === chatId) || { id: chatId };
    state.chatMessages = data.messages || state.currentChat.messages || [];
    state.chatAttempts = data.attempts || [];
    state.chatAttachments = state.chatMessages.length || !state.currentImage ? [] : [state.currentImage.id];
    state.chatFollowing = true;
    $("#chatListView").hidden = true;
    $("#conversation").hidden = false;
    renderConversationHeader();
    renderMessages();
    renderComposerContext();
    refreshConversationCapabilities();
    resizeChatInput();
    requestAnimationFrame(() => { scrollChatToBottom(false); $("#chatInput").focus(); });
    // A reopened chat can reference images from other boards, which boardImages will never
    // hold. Their records are fetched once and the transcript repainted, so an attachment keeps
    // its route, frame, and seed instead of degrading to a bare picture.
    hydrateTranscriptImages();
    // A turn that was running when the page refreshed never stopped: the server held it
    // running, the transcript above hides its assistant messages, and the replayed events
    // rebuild the live column below exactly as the original connection saw it.
    if (data.chat?.running) resumeChatTurn(chatId);
  } catch (error) { showChatNotice(error.message); }
}

async function hydrateTranscriptImages() {
  const wanted = new Set();
  const want = (id) => { if (id && !lookupImage(id)) wanted.add(id); };
  for (const message of state.chatMessages) {
    for (const candidate of messageImages(message)) want(candidate.image_id || candidate.id);
    for (const block of messageBlocks(message)) {
      if (block.type === "tool" && block.tool?.image_id) want(block.tool.image_id);
      for (const id of block.tool?.comparison_image_ids || []) want(id);
    }
  }
  if (!wanted.size) return;
  const fetched = await Promise.all([...wanted].map(async (id) => {
    try { return await api(`/api/images/${encodeURIComponent(id)}`); } catch (_) { return null; }
  }));
  const found = fetched.filter(Boolean);
  if (!found.length) return;
  for (const image of found) rememberImage(image);
  renderMessages();
}

export async function renameChat(chat = state.currentChat, requestedName = null) {
  if (!chat) return;
  const name = requestedName ?? prompt("Rename chat", chat.title || "Untitled chat");
  if (!name?.trim()) return;
  try {
    const result = await api(`/api/chats/${encodeURIComponent(chat.id)}`, { method: "POST", body: JSON.stringify({ title: name.trim() }) });
    Object.assign(chat, result.chat || result, { title: name.trim() });
    if (state.currentChat?.id === chat.id) renderConversationHeader();
    await loadBoardChats();
  } catch (error) { showChatNotice(error.message); }
}

async function deleteChat(chat) {
  if (!confirm(`Delete “${chat.title || "Untitled chat"}”?`)) return;
  try {
    await api(`/api/chats/${encodeURIComponent(chat.id)}/delete`, { method: "POST", body: "{}" });
    if (state.currentChat?.id === chat.id) { state.currentChat = null; showChatList(); }
    await loadBoardChats();
  } catch (error) { showChatNotice(error.message); }
}

export function renderConversationHeader() {
  const chat = state.currentChat;
  if (!chat) return;
  $("#conversationTitle").textContent = chat.title || "Untitled chat";
  const connection = state.connections.find((item) => item.id === chat.connection_id);
  $("#conversationMeta").textContent = connection?.name || "Connected model";
  $("#chatBrief").title = chat.generation_limit
    ? `Creative brief · up to ${chat.generation_limit} generations per reply`
    : "Creative brief · unlimited generations";
  const modelSelect = $("#chatModel");
  modelSelect.replaceChildren();
  // With the setup screen gone this is the only place a chat's backend profile can still change,
  // so it lists every keyed profile rather than only the one the chat was started on. Options are
  // keyed by profile and model together, because the same model name can be served by two.
  const profiles = keyedConnections();
  if (connection && !profiles.some((item) => item.id === connection.id)) profiles.unshift(connection);
  if (!profiles.length) profiles.push({ id: chat.connection_id, name: connection?.name || "Connected model", models: [] });
  for (const profile of profiles) {
    const models = [...(profile.models || [])];
    // A chat can outlive the sync that dropped its model; keep it selectable rather than
    // silently repointing the chat at a neighbour.
    if (profile.id === chat.connection_id && chat.model && !models.includes(chat.model)) models.unshift(chat.model);
    if (!models.length) continue;
    const group = profiles.length > 1 ? document.createElement("optgroup") : modelSelect;
    if (group !== modelSelect) group.label = profile.name;
    for (const model of models) {
      const option = document.createElement("option");
      option.value = capabilityKey(profile.id, model);
      option.textContent = model;
      group.append(option);
    }
    if (group !== modelSelect) modelSelect.append(group);
  }
  const selected = capabilityKey(chat.connection_id, chat.model);
  modelSelect.value = [...modelSelect.options].some((option) => option.value === selected)
    ? selected
    : (modelSelect.options[0]?.value || "");
  modelSelect.title = chat.model || "";
  renderConversationCapabilities();
}

async function updateCurrentChat(patch) {
  if (!state.currentChat || state.chatStreaming) return false;
  const previous = { ...state.currentChat };
  Object.assign(state.currentChat, patch);
  renderConversationHeader();
  try {
    const result = await api(`/api/chats/${encodeURIComponent(state.currentChat.id)}`, { method: "POST", body: JSON.stringify(patch) });
    Object.assign(state.currentChat, result.chat || result);
    if (patch.connection_id) localStorage.setItem("offcut.chatConnection", patch.connection_id);
    renderConversationHeader();
    await loadBoardChats();
    return true;
  } catch (error) {
    state.currentChat = previous;
    renderConversationHeader();
    appendLocalMessage(error.message, "system");
    return false;
  }
}

async function changeCurrentChatModel(value) {
  if (!state.currentChat || state.chatStreaming) return;
  const [connectionId, model] = String(value).split("\u0000");
  const previous = { connection_id: state.currentChat.connection_id, model: state.currentChat.model };
  if (!model || (connectionId === previous.connection_id && model === previous.model)) return;
  Object.assign(state.currentChat, { connection_id: connectionId, model });
  renderConversationHeader();
  try {
    const capabilities = await loadModelCapabilities(connectionId, model);
    if (state.currentChat?.model !== model || state.currentChat?.connection_id !== connectionId) return;
    const levels = capabilities.reasoning ? capabilities.supportedThinkingLevels || [] : [];
    const current = String(state.currentChat.reasoning_effort || "").toLowerCase();
    // An effort the new model does not offer is sent as null rather than guessed at: the server
    // resolves it from the profile default and falls back to what this model recommends.
    const effort = levels.includes(current) ? current : null;
    const saved = await updateCurrentChat({ connection_id: connectionId, model, reasoning_effort: effort });
    if (!saved && state.currentChat) {
      Object.assign(state.currentChat, previous);
      renderConversationHeader();
    }
  } catch (error) {
    Object.assign(state.currentChat, previous);
    renderConversationHeader();
    appendLocalMessage(`Could not change model: ${error.message}`, "system");
  }
}

export async function sendChatTurn(event) {
  event?.preventDefault();
  if (!state.currentChat) return;
  const input = $("#chatInput");
  if (state.chatStreaming) {
    // The server refuses overlapping turns, so the draft stays put until this
    // one settles. Only nudge when there is something typed; an empty Enter
    // during streaming should stay silent.
    if (input.value.trim()) showChatNotice("Still replying — your draft is kept; send again when it finishes.");
    return;
  }
  const message = input.value.trim();
  if (!message) return;
  closeComposerOverlay();
  if (await runClientCommand(message)) {
    input.value = "";
    resizeChatInput();
    return;
  }
  if (!attachmentsAllowed() && state.chatAttachments.length) {
    $("#composerStatus").textContent = "SWITCH TO A VISION MODEL OR REMOVE PENDING IMAGES";
    return;
  }
  const pending = [...new Set(state.chatAttachments)].slice(0, 8);
  const selectedImageId = pending.includes(state.currentImage?.id) ? state.currentImage.id : null;
  const attachments = pending.filter((id) => id !== selectedImageId);
  const controller = new AbortController();
  const localUser = { role: "user", content: message, attachments: pending.map((id) => findImage({ id })) };
  let initiated = false;
  let optimistic = false;
  try {
    await flushBoardDraft();
    // A new turn diffs from the prompt as it stands now, not from where the last turn started.
    // The previous turn's marks stay up until this one actually edits the prompt.
    state.promptTurnBefore = null;
    setChatStreaming(true);
    state.chatAbortController = controller;
    input.value = "";
    resizeChatInput();
    state.chatAttachments = [];
    renderComposerContext();
    state.chatMessages.push(localUser);
    state.streamMessage = { role: "assistant", content: "", tools: [], blocks: [] };
    state.streamBlockStart = 0;
    state.chatFollowing = true;
    optimistic = true;
    renderMessages();
    const response = await fetch(`/api/chats/${encodeURIComponent(state.currentChat.id)}/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        selected_image_id: selectedImageId,
        workspace_selected_image_id: state.currentImage?.id || null,
        attachment_ids: attachments,
        workspace: generationPayload(),
        workspace_revision: state.boardRevision,
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Chat request failed (${response.status})`);
    }
    if (!response.body) throw new Error("The chat response did not include a stream.");
    initiated = true;
    scheduleMessagesRender();
    await readChatStream(response.body);
  } catch (error) {
    if (!initiated && optimistic) {
      state.chatMessages = state.chatMessages.filter((item) => item !== localUser);
      state.streamMessage = null;
      input.value = message;
      resizeChatInput();
      state.chatAttachments = [...new Set([...pending, ...state.chatAttachments])].slice(0, 8);
      renderComposerContext();
      renderMessages();
    }
    if (error.name !== "AbortError" && !initiated) {
      appendLocalMessage(error.message || "The model could not complete this turn.");
    } else if (error.name !== "AbortError") {
      showChatNotice(error.message || "The model could not complete this turn.");
    }
  } finally {
    state.chatAbortController = null;
    if (initiated) await refreshCurrentChat();
    if (state.chatStreaming) setChatStreaming(false);
  }
}

async function readChatStream(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = done ? "" : lines.pop();
    for (const rawLine of lines) {
      const line = rawLine.trim().replace(/^data:\s*/, "");
      if (!line || line === "[DONE]") continue;
      let event;
      try { event = JSON.parse(line); }
      catch (_) { continue; }
      await handleChatEvent(event);
    }
    if (done) break;
  }
  if (buffer.trim()) {
    let event;
    try { event = JSON.parse(buffer.trim().replace(/^data:\s*/, "")); }
    catch (_) { event = null; }
    if (event) await handleChatEvent(event);
  }
}

function eventType(event) {
  return event.type || event.event || event.kind || "";
}

function eventText(event) {
  if (typeof event.delta === "string") return event.delta;
  if (typeof event.text === "string") return event.text;
  if (typeof event.content === "string") return event.content;
  if (typeof event.delta?.text === "string") return event.delta.text;
  return "";
}

function ensureStreamMessage(role = "assistant") {
  if (!state.streamMessage) state.streamMessage = { role, content: "", tools: [], blocks: [] };
  state.streamMessage.blocks ||= [];
  return state.streamMessage;
}

// Tool rows, generated images, and workspace patches arrive between assistant messages, so
// they belong to the turn rather than to any one message. Moving the boundary past them keeps
// the next message_end from settling over them.
// The image belongs to the generate_image call still open at the tail of the turn.
function lastGenerationCall(message) {
  const blocks = message?.blocks || [];
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index];
    if (block.type !== "tool" || !block.tool) continue;
    if (block.tool.name !== "generate_image") continue;
    return block.tool.image_id ? null : block.tool;
  }
  return null;
}

function appendTurnBlock(message, block) {
  message.blocks.push(block);
  state.streamBlockStart = message.blocks.length;
  return block;
}

// A turn edits the workspace of the board it was sent from. The form on screen is that board's
// only while the user stays there, so the settings a turn reports are recorded against the
// board that owns them and painted into the panel only when the two are still the same one.
function adoptTurnSettings(settings, revision) {
  const boardId = state.streamBoardId || state.currentBoardId;
  const board = state.boards.find((item) => item.id === boardId);
  const current = boardId === state.currentBoardId;
  if (settings) {
    if (board) board.settings = settings;
    if (current) {
      state.applyingSettings = true;
      applySettings(settings);
      state.applyingSettings = false;
    }
  }
  if (revision != null) {
    if (board) board.settings_revision = Number(revision);
    // The revision guards this tab's own writes against the board it is editing. Adopting another
    // board's would make the next draft save claim a revision the server never handed out.
    if (current) state.boardRevision = Number(revision);
  }
}

async function handleChatEvent(event) {
  const type = eventType(event);
  if (type === "creative_brief") {
    if (state.currentChat) state.currentChat.creative_brief = event.brief;
  } else if (type === "message_start") {
    const message = ensureStreamMessage(event.role || "assistant");
    const previousUiId = messageUiId(message);
    message.role = event.role || message.role || "assistant";
    message.id = event.id || event.message?.id || message.id;
    if (message.id && previousUiId !== String(message.id)) {
      // Reasoning panels are keyed by message id, so carry any open ones across.
      for (const key of [...state.expandedReasoning]) {
        if (!key.startsWith(`${previousUiId}:`)) continue;
        state.expandedReasoning.delete(key);
        state.expandedReasoning.add(`${message.id}:${key.slice(previousUiId.length + 1)}`);
      }
      message._uiId = String(message.id);
    }
    // One agent turn is several assistant messages with tool calls between them, but the
    // transcript shows it as one column. Remember where this message's blocks begin so its
    // message_end can settle just those and leave the earlier ones in place.
    state.streamBlockStart = message.blocks.length;
  } else if (type === "thinking_start") {
    const message = ensureStreamMessage();
    const block = openStreamBlock(message, "reasoning", { reasoning: "", active: true }, event.content_index);
    block.active = true;
    block.started_at ||= event.started_at || new Date().toISOString();
    setTurnPhase("THINKING");
  } else if (type === "thinking_delta") {
    const message = ensureStreamMessage();
    const block = openStreamBlock(message, "reasoning", { reasoning: "", active: true }, event.content_index);
    block.active = true;
    block.started_at ||= new Date().toISOString();
    block.reasoning = `${block.reasoning || ""}${eventText(event)}`;
    setTurnPhase("THINKING");
  } else if (type === "thinking_end") {
    const message = ensureStreamMessage();
    const block = lastStreamBlock(message, "reasoning");
    if (block) {
      block.active = false;
      block.ended_at = event.ended_at || new Date().toISOString();
      if (event.duration_ms != null) block.duration_ms = event.duration_ms;
      if (event.reasoning && !block.reasoning) block.reasoning = event.reasoning;
    }
  } else if (type === "text_delta" || type === "content_delta" || type === "message_delta" || type === "delta") {
    const message = ensureStreamMessage();
    const opening = !lastStreamBlock(message, "text");
    const block = openStreamBlock(message, "text", { text: "" }, event.content_index);
    if (opening) {
      // Prose has started, so collapse the reasoning that led to it.
      for (const [position, item] of message.blocks.entries()) {
        if (item.type !== "reasoning") continue;
        item.active = false;
        state.expandedReasoning.delete(`${messageUiId(message)}:r${position}`);
      }
    }
    block.text = `${block.text || ""}${eventText(event)}`;
    setTurnPhase("WRITING");
  } else if (type === "tool_start" || type === "tool_update" || type === "tool_end") {
    const message = ensureStreamMessage();
    const id = event.tool_call_id || event.tool_id || event.id || event.name || event.tool_name;
    let block = message.blocks.find((item) => item.type === "tool" && item.tool.id === id);
    if (!block) {
      block = appendTurnBlock(message, { type: "tool", tool: { id, name: event.name || event.tool_name || event.tool?.name || "tool" } });
    }
    Object.assign(block.tool, event.tool || {}, {
      ...(event.image_id ? { image_id: event.image_id } : {}),
      ...(event.comparison_image_ids ? { comparison_image_ids: event.comparison_image_ids } : {}),
      status: type === "tool_end" ? (event.status || "complete") : "running",
      detail: event.detail || event.message || event.output || event.result || block.tool.detail || "Working",
    });
    setTurnPhase(type === "tool_end" ? "WORKING" : toolPhaseLabel(block.tool));
  } else if (type === "styles_changed") {
    // The agent just wrote to the library, so the sidebar and the styles page refresh without
    // waiting for a reload. The board's active list is untouched: turning a style on is the
    // user's call, and save_style deliberately cannot make one.
    loadStyles();
  } else if (type === "workspace_patch") {
    // Widen the turn's diff before applySettings sees the new prompt: its stale-prompt guard
    // clears the painted hunks, and the baseline has to already be recorded to rebuild them.
    const turnChange = event.prompt_change ? accumulatePromptTurnChange(event.prompt_change) : null;
    const settings = event.settings || event.patch?.settings || event.workspace?.settings;
    adoptTurnSettings(settings, event.revision ?? event.settings_revision ?? event.workspace_revision);
    if (turnChange) {
      // The transcript keeps each edit as its own row; only the box collapses them into one span.
      // The patch arrives while the tool that made it is still running, so hang the diff off that
      // tool call: that is where the settled turn puts it, and the row would otherwise drift to
      // the bottom of the column as the rest of the turn streamed in above it.
      const message = ensureStreamMessage();
      const tool = [...message.blocks].reverse().find((block) => block.type === "tool" && block.tool.status === "running");
      if (tool) tool.tool.prompt_change = event.prompt_change;
      else appendTurnBlock(message, { type: "prompt_change", change: event.prompt_change });
      showPromptSyncChange(turnChange);
    }
    setTurnPhase("WORKSPACE UPDATED");
  } else if (type === "generation" || type === "image" || type === "generation_complete") {
    const message = ensureStreamMessage();
    const image = event.image || { image_id: event.image_id, image_url: event.image_url, prompt: event.prompt };
    if (image.image_id || image.id || image.image_url || image.url) {
      const key = image.image_id || image.id;
      // Remembered before anything renders so the card has the full record on its first paint
      // instead of growing a metadata panel once loadBoardImages resolves.
      if (image.id || image.image_id) rememberImage({ ...image, id: key });
      const call = lastGenerationCall(message);
      if (call) call.image_id = key;
      else if (!message.blocks.some((item) => item.type === "image" && (item.image.image_id || item.image.id) === key)) {
        appendTurnBlock(message, { type: "image", image });
      }
      message.generating = false;
      setAgentGenerating(false);
      // The agent's frame lands on the turn's board, so the rail only repaints for someone who
      // is still standing on it; loadBoardImages drops the call otherwise.
      markFresh(image.image_id || image.id || null);
      await loadBoardImages(image.image_id || image.id || null, state.streamBoardId || state.currentBoardId);
      setTurnPhase("WORKING");
    } else {
      message.generating = true;
      message.generation_detail = event.detail || event.status || "Generating image";
      setAgentGenerating(true);
      if (state.agentGenerating) updateProgress(event);
      setTurnPhase("GENERATING IMAGE");
    }
  } else if (type === "message_end") {
    // The server sends this assistant message settled, with its blocks in true production
    // order, but it describes that one message while the column holds the whole turn. Splice
    // it over its own blocks only: replacing all of them dropped the earlier tool calls, and
    // their prompt diffs then reappeared under the closing prose instead of beside the call
    // that made them. Tool detail and diffs come from events the settled message cannot carry,
    // so carry those across by tool call id.
    const previous = state.streamMessage?.blocks || [];
    const start = Math.min(state.streamBlockStart, previous.length);
    const carried = new Map();
    for (const block of previous) if (block.type === "tool" && block.tool?.id) carried.set(block.tool.id, block.tool);
    if (event.message) state.streamMessage = { ...state.streamMessage, ...event.message };
    // A payload without blocks says nothing about order, so the streamed ones stand.
    if (state.streamMessage && Array.isArray(event.message?.blocks)) {
      const settled = event.message.blocks;
      for (const block of settled) {
        if (block.type !== "tool" || !block.tool) continue;
        const known = carried.get(block.tool.id);
        if (known) Object.assign(block.tool, known);
      }
      state.streamMessage.blocks = [...previous.slice(0, start), ...settled];
      state.streamBlockStart = state.streamMessage.blocks.length;
    }
    if (event.usage && state.streamMessage) state.streamMessage.usage = event.usage;
    if (state.streamMessage) {
      state.streamMessage.reasoning_active = false;
      for (const block of state.streamMessage.blocks || []) if (block.type === "reasoning") block.active = false;
    }
  } else if (type === "error") {
    throw new Error(event.error?.message || event.error || event.message || "The model reported an error.");
  } else if (type === "done") {
    adoptTurnSettings(event.settings || event.workspace?.settings, event.revision ?? event.workspace_revision ?? event.settings_revision);
    if (event.image_id || event.image) {
      markFresh(event.image_id || event.image?.id || event.image?.image_id || null);
      await loadBoardImages(event.image_id || event.image?.id || event.image?.image_id || null, state.streamBoardId || state.currentBoardId);
    }
  } else if (event.text || event.delta) {
    const message = ensureStreamMessage();
    const block = openStreamBlock(message, "text", { text: "" }, event.content_index);
    block.text = `${block.text || ""}${eventText(event)}`;
    setTurnPhase("WRITING");
  }
  if (state.streamMessage) syncStreamMirrors(state.streamMessage);
  scheduleMessagesRender({ streamOnly: true });
}

async function refreshCurrentChat() {
  const chatId = state.currentChat?.id;
  if (!chatId) return;
  try {
    const [data, boardData, imageData, chatData] = await Promise.all([
      api(`/api/chats/${encodeURIComponent(chatId)}`),
      api("/api/boards"),
      api(`/api/images?board_id=${encodeURIComponent(state.currentBoardId)}`),
      api(`/api/chats?board_id=${encodeURIComponent(state.currentBoardId)}`),
    ]);
    if (state.currentChat?.id !== chatId) return;
    state.boards = boardData.boards || state.boards;
    const board = state.boards.find((item) => item.id === state.currentBoardId);
    if (board) {
      state.boardRevision = Number(board.settings_revision || 0);
      if (board.settings) {
        state.applyingSettings = true;
        applySettings(board.settings);
        state.applyingSettings = false;
      }
    }
    state.currentChat = data.chat || state.currentChat;
    state.chatMessages = data.messages || state.currentChat.messages || state.chatMessages;
    state.boardImages = imageData.images || state.boardImages;
    state.chats = chatData.chats || state.chats;
    state.streamMessage = null;
    populateBoardSelectors();
    renderBoards();
    renderChatList();
    const known = state.boardImages.find((image) => image.id === state.currentImage?.id);
    selectImage(known || (state.canvasDismissed ? null : state.boardImages[0] || null));
    renderConversationHeader();
    renderMessages();
    refreshConversationCapabilities();
  } catch (error) {
    if (state.streamMessage && (messageText(state.streamMessage) || state.streamMessage.reasoning || state.streamMessage.tools?.length)) state.chatMessages.push(state.streamMessage);
    state.streamMessage = null;
    renderMessages();
    showChatNotice(`Could not refresh final messages: ${error.message}`);
  }
}

async function stopChatTurn() {
  if (!state.currentChat || !state.chatStreaming) return;
  $("#stopChat").disabled = true;
  setTurnPhase("STOPPING");
  try {
    await api(`/api/chats/${encodeURIComponent(state.currentChat.id)}/abort`, { method: "POST", body: "{}" });
    // Keep reading until the server has saved the interrupted turn and closed it.
    // Refreshing before that hides the turn behind its live-replay watermark.
  } catch (error) {
    showChatNotice(`Could not stop the turn: ${error.message}`);
    $("#stopChat").disabled = false;
  }
}

// Reattach to a turn the server is still running after a refresh or a reopened tab. The
// endpoint replays the buffered events from the turn's first token, so the column picks up
// mid-stream rather than waiting for the settlement, and Stop still works through the same
// abort controller the live path uses.
async function resumeChatTurn(chatId) {
  if (state.currentChat?.id !== chatId || state.chatStreaming) return;
  const controller = new AbortController();
  state.chatAbortController = controller;
  state.streamMessage = { role: "assistant", content: "", tools: [], blocks: [] };
  state.streamBlockStart = 0;
  state.chatFollowing = true;
  setChatStreaming(true);
  renderMessages();
  try {
    const response = await fetch(`/api/chats/${encodeURIComponent(chatId)}/events`, { signal: controller.signal });
    if (!response.ok) {
      // A 409 means the turn finished between opening the chat and attaching; the transcript
      // refresh in the finally block settles whatever it completed.
      if (response.status !== 409) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Could not rejoin the running turn.");
      }
    } else if (!response.body) {
      throw new Error("The chat response did not include a stream.");
    } else {
      await readChatStream(response.body);
    }
  } catch (error) {
    if (error.name !== "AbortError") showChatNotice(error.message || "Could not rejoin the running turn.");
  } finally {
    state.chatAbortController = null;
    await refreshCurrentChat();
    if (state.chatStreaming) setChatStreaming(false);
  }
}

export function initChat() {
  $("#newChatButton").addEventListener("click", createChat);
  $("#conversationBack").addEventListener("click", async () => { showChatList(); await loadBoardChats(); });
  $("#conversationRename").addEventListener("click", () => renameChat());
  $("#chatBrief").addEventListener("click", openBriefDialog);
  $("#chatModel").addEventListener("change", (event) => changeCurrentChatModel(event.target.value));
  $("#chatReasoning").addEventListener("change", (event) => updateCurrentChat({ reasoning_effort: event.target.value }));
  $("#stopChat").addEventListener("click", stopChatTurn);
}
