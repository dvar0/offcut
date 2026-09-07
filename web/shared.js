export const $ = (selector, root = document) => root.querySelector(selector);

export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const state = {
  presets: {},
  boards: [],
  connections: [],
  loras: [],
  selectedLoras: [],
  styles: [],
  selectedStyles: [],
  editingStyle: null,
  editingLora: null,
  loraCoverId: null,
  libraryKind: "all",
  libraryQuery: "",
  pickerKind: "all",
  pickerQuery: "",
  coverBoardId: null,
  coverPickerResolve: null,
  coverRecipe: null,
  coverRunning: false,
  coverStopping: false,
  coverRun: null,
  boardImages: [],
  currentBoardId: null,
  currentImage: null,
  canvasDismissed: false,
  galleryImages: [],
  detailImage: null,
  runtimeReady: false,
  activeEngine: null,
  runtimeInitializing: false,
  runtimeTimer: null,
  presetReady: {},
  generating: false,
  agentGenerating: false,
  // The board a run is writing into, which is not necessarily the one on screen: a run is free
  // to keep going while the user works somewhere else, so every path that reacts to a finished
  // frame checks against this rather than against whatever board is current when it lands.
  generatingBoardId: null,
  frameDetailsOpen: false,
  batchCount: 1,
  batchIndex: 0,
  batchTotal: 0,
  stopping: false,
  progressTimer: null,
  progressStarted: 0,
  railFavorites: false,
  railQuery: "",
  galleryFavorites: false,
  // The frames the newest run produced. A batch is otherwise only told apart from what was
  // already on the board by counting it, so each run marks its own and clears the run before it.
  freshImageIds: new Set(),
  draggedImageId: null,
  aspectLocked: true,
  aspectRatio: 1,
  boardRevision: 0,
  draftTimer: null,
  draftPromise: null,
  draftError: null,
  draftSaved: null,
  applyingSettings: false,
  chats: [],
  currentChat: null,
  chatMessages: [],
  chatAttachments: [],
  // Attachments may reference any board, so the records behind them outlive the board that is
  // currently loaded. Board scoping still governs the workspace; this only governs rendering.
  knownImages: new Map(),
  skills: [],
  overlay: null,
  commandQuery: null,
  commandMatches: [],
  commandIndex: 0,
  pickerBoardId: null,
  pickerSelection: new Set(),
  pickerCache: new Map(),
  chatStreaming: false,
  // The board the streaming turn belongs to. A turn writes settings and frames into the board it
  // was sent from, so its events are matched against this and not against the current board.
  streamBoardId: null,
  chatAbortController: null,
  chatFollowing: true,
  streamMessage: null,
  streamBlockStart: 0,
  streamNode: null,
  modelCapabilities: new Map(),
  expandedReasoning: new Set(),
  expandedTools: new Set(),
  nextMessageUiId: 1,
  messageRenderFrame: null,
  streamRenderOnly: false,
  autoScrolling: false,
  lastScrollTop: 0,
  lastScrollHeight: 0,
  turnStartedAt: 0,
  turnPhase: "",
  turnPhaseTimer: null,
  latestPromptChange: null,
  promptTurnBefore: null,
  promptDiffHunks: [],
  promptDiffMarks: [],
  promptDiffIndex: -1,
  promptEmphasisTimer: null,
  promptPeekTimer: null,
  promptPeekPinned: false,
};

export async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({ error: `Invalid response (${response.status})` }));
  if (!response.ok) {
    const error = new Error(data.error || `Request failed (${response.status})`);
    error.status = response.status;
    error.data = data;
    throw error;
  }
  return data;
}

export function escapeHtml(value) {
  const span = document.createElement("span");
  span.textContent = String(value ?? "");
  return span.innerHTML;
}

export function presetLabel(preset) {
  if (preset === "turbo-int8") return "TURBO (LEGACY CHECKPOINT)";
  return state.presets[preset]?.label?.toUpperCase() || String(preset || "").toUpperCase();
}

// Hybrid guidance belongs only to its raw opening; Turbo is always fixed at CFG 1.
export function frameUsesGuidance(image) {
  return routeUsesGuidance(image.preset);
}

// One copy of the rule, because the cover recipe disables its own guidance field on exactly the
// same routes and a second literal here would drift from this one.
export function routeUsesGuidance(preset) {
  return preset === "raw-int8" || preset === "raw-int8-to-turbo";
}

export function isReference(image) {
  return image.kind === "reference" || image.preset === "reference";
}

// A seed above 2^53 does not survive JSON.parse, so the record carries the exact digits as text
// and everything that shows one, copies one, or puts one back in the form reads that instead.
export function frameSeed(image) {
  return image.seed_text || String(image.seed ?? "");
}

export function showChatNotice(message = "") {
  const notice = $("#chatNotice");
  notice.textContent = message;
  notice.hidden = !message;
  if (state.currentChat && !state.chatStreaming) $("#composerStatus").textContent = message;
}

export function showFormError(message) { $("#formError").textContent = message; }

export function formatChatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return "";
  const sameYear = date.getFullYear() === new Date().getFullYear();
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", ...(sameYear ? {} : { year: "numeric" }) });
}

export function formatCost(value) {
  const amount = Number(value);
  return Number.isFinite(amount) ? `$${amount < 0.01 ? amount.toFixed(4) : amount.toFixed(2)}` : String(value);
}

export function rememberImage(image) {
  if (image?.id) state.knownImages.set(image.id, image);
  return image;
}

export function lookupImage(id) {
  if (!id) return null;
  return state.boardImages.find((item) => item.id === id) || state.knownImages.get(id) || null;
}

export function findImage(image) {
  const id = image?.image_id || image?.id;
  return lookupImage(id) || image || {};
}

export function formatDate(value) {
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "--" : date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" }).toUpperCase();
}

export function debounce(callback, wait = 250) {
  let timeout;
  return (...args) => { clearTimeout(timeout); timeout = setTimeout(() => callback(...args), wait); };
}

// The editors hold typed work, so nothing inside the box dismisses them — only a click on the
// blurred area outside it. The backdrop reports the dialog itself as the click target, which is
// also what a click on the dialog's own padding looks like, so the box's rectangle is what tells
// the two apart.
export function dismissOnBackdrop(dialog) {
  dialog.addEventListener("click", (event) => {
    if (event.target !== dialog) return;
    const box = dialog.getBoundingClientRect();
    if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
  });
}
