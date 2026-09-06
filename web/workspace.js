import { loadBoardChats, showChatList } from "./chat.js";
import { renderComposerContext } from "./chat-composer.js";
import { applySettings, generationPayload, paintGeneratingBoard } from "./generation.js";
import { gridColumnWidth, thumbnailUrl } from "./image-layout.js";
import { loadBoardImages, loadGallery } from "./images.js";
import { loadLoras, loadStyles } from "./library.js";
import { forgetPromptTurn } from "./prompt-diff.js";
import { loadCoverRecipe, renderSettings } from "./settings.js";
import { $, $$, api, debounce, showChatNotice, state } from "./shared.js";

export function currentPage() {
  const segment = location.pathname.split("/").filter(Boolean)[0] || "create";
  if (segment === "connections") return "settings";
  return ["create", "gallery", "styles", "boards", "settings"].includes(segment) ? segment : "create";
}

export function navigate(page, boardId = null) {
  const path = page === "create" && boardId ? `/create/${encodeURIComponent(boardId)}` : `/${page}`;
  history.pushState({}, "", path);
  showPage();
}

export async function showPage() {
  if (location.pathname === "/connections" || location.pathname.startsWith("/connections/")) {
    history.replaceState({}, "", "/settings");
  }
  const page = currentPage();
  $$("[data-page]").forEach((element) => { element.hidden = element.dataset.page !== page; });
  $$(".primary-nav a").forEach((link) => link.classList.toggle("active", link.dataset.route === page));
  if (page === "gallery") await loadGallery();
  // Both halves of the library, since a cover set from the gallery lands on either one.
  if (page === "styles") await Promise.all([loadStyles(), loadLoras()]);
  if (page === "boards") renderBoards();
  if (page === "settings") {
    renderSettings();
    if (!state.coverRecipe) loadCoverRecipe();
  }
  if (page === "create") {
    const routeBoard = decodeURIComponent(location.pathname.split("/")[2] || "");
    if (routeBoard && routeBoard !== state.currentBoardId && state.boards.some((board) => board.id === routeBoard)) {
      await setActiveBoard(routeBoard, false);
    }
  }
}

export function populateBoardSelectors() {
  const active = $("#activeBoard");
  const move = $("#moveBoard");
  const gallery = $("#galleryBoard");
  const activeValue = state.currentBoardId;
  const moveValue = move.value;
  const galleryValue = gallery.value;
  active.replaceChildren();
  move.innerHTML = '<option value="">SELECT BOARD</option>';
  gallery.innerHTML = '<option value="">ALL BOARDS</option>';
  for (const board of state.boards) {
    for (const select of [active, move, gallery]) {
      const option = document.createElement("option");
      option.value = board.id;
      option.textContent = board.name.toUpperCase();
      select.append(option);
    }
  }
  if (activeValue) active.value = activeValue;
  if (state.boards.some((board) => board.id === moveValue)) move.value = moveValue;
  if (state.boards.some((board) => board.id === galleryValue)) gallery.value = galleryValue;
}

export async function refreshBoards() {
  const data = await api("/api/boards");
  state.boards = data.boards;
  const current = state.boards.find((board) => board.id === state.currentBoardId);
  if (current) state.boardRevision = Number(current.settings_revision || state.boardRevision || 0);
  populateBoardSelectors();
  renderBoards();
  // Names come from this listing, so a board renamed mid-run repaints the run's line with it.
  paintGeneratingBoard();
}

// A run is the engine's, not the panel's: where its frames land was decided by the payload
// snapshot taken when it started, so leaving the board it is working on cannot misfile anything
// and is not blocked. What does have to move is everything keyed to the board on screen — the
// draft, and any turn streaming into the board being left.
export async function setActiveBoard(boardId, updateRoute = true) {
  const leaving = state.currentBoardId && boardId !== state.currentBoardId;
  // The turn keeps running on the server, which buffers its events; reopening the chat replays
  // them from the first token. Only this tab's reader lets go, because the state it writes into
  // is about to be cleared out from under it. Detached first so the draft flushed below is the
  // form as it stands rather than one the stream is still editing.
  if (leaving && state.chatStreaming) state.chatAbortController?.abort();
  if (leaving) await flushBoardDraft();
  state.currentBoardId = boardId;
  state.draftError = null;
  localStorage.setItem("offcut.activeBoard", boardId);
  $("#activeBoard").value = boardId;
  // A run left behind on another board has to say so from here on, and stop saying it on return.
  paintGeneratingBoard();
  const board = state.boards.find((item) => item.id === boardId);
  if (!board) return;
  state.boardRevision = Number(board.settings_revision || 0);
  // Another board's prompt is not a continuation of this turn's edits.
  forgetPromptTurn();
  state.applyingSettings = true;
  if (board.settings && Object.keys(board.settings).length) applySettings(board.settings);
  state.applyingSettings = false;
  state.currentChat = null;
  state.chatMessages = [];
  state.chatAttachments = [];
  state.canvasDismissed = false;
  state.railQuery = "";
  $("#railSearch").value = "";
  showChatList();
  await Promise.all([loadBoardImages(null, boardId), loadBoardChats(boardId)]);
  renderComposerContext();
  if (updateRoute && currentPage() === "create") history.replaceState({}, "", `/create/${encodeURIComponent(boardId)}`);
}

export function scheduleBoardDraft() {
  if (state.applyingSettings || !state.currentBoardId) return;
  window.clearTimeout(state.draftTimer);
  state.draftTimer = window.setTimeout(() => persistBoardDraft().catch(() => {}), 500);
}

async function persistBoardDraft() {
  window.clearTimeout(state.draftTimer);
  state.draftTimer = null;
  if (!state.currentBoardId) return;
  if (state.draftPromise) await state.draftPromise;
  const boardId = state.currentBoardId;
  const settings = generationPayload();
  // Every board write bumps settings_revision, and the form autosaves on each burst of input, so
  // re-selecting a value or re-firing change on an untouched control used to cost a revision for
  // nothing. Skip a write that would store exactly what the server last confirmed.
  const serialized = JSON.stringify(settings);
  if (serialized === state.draftSaved) return;
  const expectedRevision = state.boardRevision;
  state.draftPromise = api(`/api/boards/${encodeURIComponent(boardId)}`, {
    method: "POST",
    body: JSON.stringify({ settings, expected_settings_revision: expectedRevision }),
  }).then((result) => {
    if (boardId !== state.currentBoardId) return;
    state.draftSaved = serialized;
    state.boardRevision = Number(result.settings_revision ?? result.board?.settings_revision ?? expectedRevision + 1);
    const board = state.boards.find((item) => item.id === boardId);
    if (board) {
      board.settings = settings;
      board.settings_revision = state.boardRevision;
    }
    state.draftError = null;
    showChatNotice("");
  }).catch(async (error) => {
    state.draftError = error;
    state.draftSaved = null;
    if (boardId !== state.currentBoardId) throw error;
    try {
      const data = await api("/api/boards");
      state.boards = data.boards;
      const fresh = state.boards.find((item) => item.id === boardId);
      if (fresh) state.boardRevision = Number(fresh.settings_revision || 0);
      populateBoardSelectors();
      renderBoards();
    } catch (_) { /* Keep the local controls intact if refresh also fails. */ }
    showChatNotice(`Board draft was not saved: ${error.message}. Your local controls were kept.`);
    throw error;
  }).finally(() => { state.draftPromise = null; });
  await state.draftPromise;
}

export async function flushBoardDraft() {
  if (state.draftTimer) await persistBoardDraft();
  else if (state.draftPromise) await state.draftPromise;
  else if (state.draftError) await persistBoardDraft();
}

function setInspectorTab(tab) {
  const chat = tab === "chat";
  $("#imagesTab").ariaSelected = String(!chat);
  $("#chatTab").ariaSelected = String(chat);
  $("#imagesTab").tabIndex = chat ? -1 : 0;
  $("#chatTab").tabIndex = chat ? 0 : -1;
  $("#imagesPanel").hidden = chat;
  $("#chatPanel").hidden = !chat;
  localStorage.setItem("offcut.inspectorTab", tab);
  if (chat) renderComposerContext();
}

function initInspectorResize() {
  const stored = Number(localStorage.getItem("offcut.inspectorWidth"));
  const handle = $("#inspectorResize");
  const availableMaximum = () => {
    if (matchMedia("(max-width: 820px)").matches) return 640;
    const controls = window.innerWidth <= 1100 ? 300 : 330;
    const canvas = window.innerWidth <= 1100 ? 320 : 360;
    return Math.max(280, Math.min(640, window.innerWidth - controls - canvas));
  };
  // The width the user dragged to, kept separately from the width that fits right now. A narrow
  // window only constrains what gets rendered; folding that constraint back into the stored
  // preference is what used to leave the panel stuck at the smaller size after the window grew
  // back, because there was no longer any record of what the user had actually asked for.
  let desired = Math.max(280, Math.min(640, stored || 360));
  const applyWidth = () => {
    const bounded = Math.max(280, Math.min(availableMaximum(), desired));
    document.documentElement.style.setProperty("--inspector-width", `${bounded}px`);
    handle.setAttribute("aria-valuenow", String(bounded));
    return bounded;
  };
  const resize = (width) => {
    desired = Math.max(280, Math.min(640, Math.round(width)));
    localStorage.setItem("offcut.inspectorWidth", String(desired));
    applyWidth();
  };
  applyWidth();
  handle.addEventListener("pointerdown", (event) => {
    if (matchMedia("(max-width: 820px)").matches) return;
    event.preventDefault();
    handle.setPointerCapture(event.pointerId);
    $("#inspector").classList.add("resizing");
  });
  handle.addEventListener("pointermove", (event) => {
    if (!handle.hasPointerCapture(event.pointerId)) return;
    resize(window.innerWidth - event.clientX);
  });
  const finish = (event) => {
    if (handle.hasPointerCapture(event.pointerId)) handle.releasePointerCapture(event.pointerId);
    $("#inspector").classList.remove("resizing");
  };
  handle.addEventListener("pointerup", finish);
  handle.addEventListener("pointercancel", finish);
  handle.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const current = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--inspector-width"), 10) || 360;
    if (event.key === "Home") resize(280);
    else if (event.key === "End") resize(640);
    else resize(current + (event.key === "ArrowLeft" ? 16 : -16));
  });
  window.addEventListener("resize", debounce(applyWidth, 100));
}

export function renderBoards() {
  const grid = $("#boardsGrid");
  // A cover holds up to four tiles in a 2x2, so each spans half the card.
  const coverWidth = gridColumnWidth(grid, 320) / 2;
  grid.replaceChildren();
  state.boards.forEach((board, index) => {
    const card = $("#boardCardTemplate").content.firstElementChild.cloneNode(true);
    $(".board-number", card).textContent = String(index + 1).padStart(2, "0");
    $("h2", card).textContent = board.name.toUpperCase();
    $("p", card).textContent = board.description || "No description yet.";
    $(".board-frame-count", card).textContent = `${board.image_count} FRAMES`;
    $(".board-favorite-count", card).textContent = `${board.favorite_count} FAVORITES`;
    const cover = $(".board-cover", card);
    const empty = $(".board-cover span", card);
    for (const [coverIndex, url] of (board.cover_urls || (board.cover_url ? [board.cover_url] : [])).entries()) {
      const image = document.createElement("img");
      image.src = thumbnailUrl({ image_url: url }, coverWidth);
      image.alt = `${board.name} board image ${coverIndex + 1}`;
      image.loading = "lazy";
      cover.append(image);
      empty.hidden = true;
    }
    const open = async () => { await setActiveBoard(board.id, false); navigate("create", board.id); };
    $(".board-cover", card).addEventListener("click", open);
    $(".open-board", card).addEventListener("click", open);
    $(".rename-board", card).addEventListener("click", () => renameBoard(board));
    const deleteButton = $(".delete-board", card);
    deleteButton.hidden = board.id === "inbox";
    deleteButton.addEventListener("click", () => deleteBoard(board));
    grid.append(card);
  });
}

async function createBoard(event) {
  event.preventDefault();
  $("#boardError").textContent = "";
  try {
    const board = await api("/api/boards", { method: "POST", body: JSON.stringify({ name: $("#boardName").value, description: $("#boardDescription").value }) });
    $("#boardDialog").close();
    $("#boardForm").reset();
    await refreshBoards();
    await setActiveBoard(board.id, false);
    navigate("create", board.id);
  } catch (error) { $("#boardError").textContent = error.message; }
}

async function renameBoard(board) {
  const name = prompt("Rename board", board.name);
  if (!name || name.trim() === board.name) return;
  await api(`/api/boards/${encodeURIComponent(board.id)}`, { method: "POST", body: JSON.stringify({ name }) });
  await refreshBoards();
}

async function deleteBoard(board) {
  if (!confirm(`Delete “${board.name}”? Its images will move to Inbox.`)) return;
  await api(`/api/boards/${encodeURIComponent(board.id)}/delete`, { method: "POST", body: "{}" });
  await refreshBoards();
  if (state.currentBoardId === board.id) await setActiveBoard("inbox", false);
}

export function initWorkspace() {
  $$("[data-route]").forEach((link) => link.addEventListener("click", (event) => { event.preventDefault(); navigate(link.dataset.route, link.dataset.route === "create" ? state.currentBoardId : null); }));
  window.addEventListener("popstate", showPage);
  $("#activeBoard").addEventListener("change", (event) => setActiveBoard(event.target.value));
  $("#quickBoardButton").addEventListener("click", () => $("#boardDialog").showModal());
  $("#newBoardButton").addEventListener("click", () => $("#boardDialog").showModal());
  $("#closeBoardDialog").addEventListener("click", () => $("#boardDialog").close());
  $("#boardForm").addEventListener("submit", createBoard);
  $("#imagesTab").addEventListener("click", () => setInspectorTab("images"));
  $("#chatTab").addEventListener("click", () => setInspectorTab("chat"));
  for (const type of ["dragenter", "dragover"]) {
    $("#chatTab").addEventListener(type, (event) => {
      if (!Array.from(event.dataTransfer.types).includes("application/x-offcut-image-id")) return;
      event.preventDefault();
      setInspectorTab("chat");
    });
  }
  for (const tab of [$("#imagesTab"), $("#chatTab")]) {
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const target = tab === $("#imagesTab") ? $("#chatTab") : $("#imagesTab");
      setInspectorTab(target === $("#chatTab") ? "chat" : "images");
      target.focus();
    });
  }
  $("#generateForm").addEventListener("input", scheduleBoardDraft);
  $("#generateForm").addEventListener("change", scheduleBoardDraft);
  $("#randomSeed").addEventListener("click", scheduleBoardDraft);
  initInspectorResize();
  setInspectorTab(localStorage.getItem("offcut.inspectorTab") === "chat" ? "chat" : "images");
}
