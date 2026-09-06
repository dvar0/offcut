import { removeChatAttachment, renderComposerContext } from "./chat-composer.js";
import { reuseImage, reusePrompt } from "./generation.js";
import { GALLERY_PACK, justifiedRailSizes, packContentWidth, thumbnailUrl } from "./image-layout.js";
import { loraLabel } from "./library.js";
import {
  $,
  api,
  debounce,
  findImage,
  formatDate,
  frameSeed,
  frameUsesGuidance,
  isReference,
  presetLabel,
  rememberImage,
  showFormError,
  state,
} from "./shared.js";
import { refreshBoards } from "./workspace.js";

export async function loadBoardImages(preferredImageId = null, requestedBoardId = state.currentBoardId) {
  if (!requestedBoardId) return;
  const data = await api(`/api/images?board_id=${encodeURIComponent(requestedBoardId)}`);
  if (requestedBoardId !== state.currentBoardId) return;
  state.boardImages = data.images;
  state.pickerCache.set(requestedBoardId, data.images);
  for (const image of data.images) rememberImage(image);
  renderBoardThumbnails();
  const selected = state.boardImages.find((image) => image.id === (preferredImageId || state.currentImage?.id));
  if (!selected && state.canvasDismissed) selectImage(null);
  else selectImage(selected || state.boardImages[0] || null);
}

function visibleBoardImages() {
  const query = state.railQuery.trim().toLowerCase();
  return state.boardImages.filter((image) => {
    if (state.railFavorites && !image.favorite) return false;
    if (!query) return true;
    return [image.raw_prompt, image.enhanced_prompt, image.final_prompt].some((text) => (text || "").toLowerCase().includes(query));
  });
}

// Rebuilding the whole rail on every generation threw away every decoded thumbnail and asked
// for all of them again. Tiles are kept by image id and re-ordered in place instead: moving a
// node inside the document does not restart its image load.
const TRASH_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"><path d="M2.9 4.3h10.2M6.3 4.3V2.7h3.4v1.6M4.4 4.3l.6 8.5a1 1 0 0 0 1 .9h4a1 1 0 0 0 1-.9l.6-8.5M6.7 6.6v4.6M9.3 6.6v4.6"/></svg>';

// A tile is a button, so its delete control cannot be one: it is a span that swallows the click
// before the tile's own handler opens the frame.
function tileDeleteControl(remove) {
  const control = document.createElement("span");
  control.className = "tile-delete";
  control.setAttribute("role", "button");
  control.setAttribute("aria-label", "Delete this frame");
  control.title = "Delete this frame";
  control.innerHTML = TRASH_ICON;
  control.addEventListener("click", (event) => { event.stopPropagation(); remove(); });
  return control;
}

// Deleting a frame removes the file as well as the row, so nothing here comes back.
async function deleteImage(image, successorId = null) {
  if (!image) return false;
  if (!confirm(`Delete this frame permanently?\n\n${image.raw_prompt || "Untitled frame"}`)) return false;
  try {
    await api(`/api/images/${encodeURIComponent(image.id)}/delete`, { method: "POST", body: "{}" });
  } catch (error) {
    showFormError(error.message);
    return false;
  }
  const wasCurrent = state.currentImage?.id === image.id;
  forgetImage(image.id);
  renderBoardThumbnails();
  renderGallery();
  if (wasCurrent) selectImage(state.boardImages.find((item) => item.id === successorId) || state.boardImages[0] || null);
  await refreshBoards();
  return true;
}

function forgetImage(imageId) {
  state.boardImages = state.boardImages.filter((image) => image.id !== imageId);
  state.galleryImages = state.galleryImages.filter((image) => image.id !== imageId);
  for (const [boardId, images] of state.pickerCache) {
    state.pickerCache.set(boardId, images.filter((image) => image.id !== imageId));
  }
  state.knownImages.delete(imageId);
  state.freshImageIds.delete(imageId);
  state.pickerSelection.delete(imageId);
  railTiles.delete(imageId);
  galleryTiles.delete(imageId);
  if (state.chatAttachments.includes(imageId)) removeChatAttachment(imageId);
  if (state.detailImage?.id === imageId) state.detailImage = null;
  if (state.currentImage?.id === imageId) state.currentImage = null;
}

// The frame that takes the deleted one's place on the canvas is its neighbour in the rail.
function deleteBoardFrame(image) {
  const ids = visibleBoardImages().map((item) => item.id);
  const index = ids.indexOf(image.id);
  return deleteImage(image, ids[index + 1] || ids[index - 1] || null);
}

const railTiles = new Map();

function railTile(image, requestSize) {
  let button = railTiles.get(image.id);
  if (!button) {
    button = document.createElement("button");
    button.type = "button";
    button.dataset.imageId = image.id;
    const img = document.createElement("img");
    img.loading = "lazy";
    // Short, for the same reason as the transcript card: a failed thumbnail paints its alt.
    img.alt = "Board frame";
    button.append(img);
    // The handlers outlive any one render, so they read the record the tile currently holds
    // rather than the one it was first built from.
    button.addEventListener("click", () => {
      state.freshImageIds.delete(button.imageRecord.id);
      selectImage(button.imageRecord);
    });
    button.addEventListener("dragstart", (event) => {
      state.draggedImageId = button.imageRecord.id;
      event.dataTransfer.effectAllowed = "copyMove";
      event.dataTransfer.setData("application/x-offcut-image-id", button.imageRecord.id);
      event.dataTransfer.setData("text/plain", button.imageRecord.id);
      button.classList.add("dragging");
    });
    button.addEventListener("dragend", () => { state.draggedImageId = null; button.classList.remove("dragging"); });
    button.addEventListener("dragover", (event) => event.preventDefault());
    button.addEventListener("drop", (event) => reorderDrop(event, button.imageRecord.id));
    button.append(tileDeleteControl(() => deleteBoardFrame(button.imageRecord)));
    railTiles.set(image.id, button);
  }
  button.imageRecord = image;
  button.className = `thumbnail${image.favorite ? " favorite" : ""}${state.freshImageIds.has(image.id) ? " fresh" : ""}${state.currentImage?.id === image.id ? " selected" : ""}`;
  button.draggable = !state.railFavorites && !state.railQuery.trim();
  button.title = image.raw_prompt || "";
  const img = button.querySelector("img");
  // The request bounds the short side, so a true-ratio tile asks for its smaller dimension.
  const source = thumbnailUrl(image, requestSize);
  if (img.getAttribute("src") !== source) img.src = source;
  const badge = button.querySelector(".thumbnail-kind");
  if (image.kind === "reference" && !badge) {
    const kind = document.createElement("span");
    kind.className = "thumbnail-kind";
    kind.textContent = "REFERENCE";
    button.append(kind);
  } else if (image.kind !== "reference" && badge) {
    badge.remove();
  }
  return button;
}

export function renderBoardThumbnails() {
  const images = visibleBoardImages();
  const searching = Boolean(state.railQuery.trim());
  $("#railEmpty").hidden = images.length > 0;
  $("#railEmpty").textContent = searching ? "NO FRAMES MATCH THIS SEARCH" : "NO FRAMES IN THIS BOARD";
  $("#railHint").textContent = searching ? `${images.length} MATCHING ${images.length === 1 ? "FRAME" : "FRAMES"}` : "DRAG TO REORDER OR DROP INTO CHAT";
  const grid = $("#boardThumbnails");
  const sizes = justifiedRailSizes(images, packContentWidth(grid, 300));
  const tiles = images.map((image) => {
    const { width, height } = sizes.get(image.id);
    const tile = railTile(image, Math.min(width, height));
    const wide = `${width}px`;
    const tall = `${height}px`;
    if (tile.style.width !== wide) tile.style.width = wide;
    if (tile.style.height !== tall) tile.style.height = tall;
    return tile;
  });
  const shown = new Set(images.map((image) => image.id));
  for (const id of railTiles.keys()) if (!shown.has(id)) railTiles.delete(id);
  // selectImage repaints the rail to move the highlight. Re-inserting every tile to say
  // nothing changed costs a full layout of the column, so only touch the grid when the set
  // or the order actually moved. The tiles still carry the packed size of the last render,
  // and the width guards above leave those writes alone when the rail has not been resized.
  const settled = tiles.length === grid.children.length && tiles.every((tile, index) => grid.children[index] === tile);
  if (!settled) grid.replaceChildren(...tiles);
}

async function reorderDrop(event, targetId) {
  event.preventDefault();
  const draggedId = state.draggedImageId;
  if (!draggedId || draggedId === targetId) return;
  const ids = visibleBoardImages().map((image) => image.id);
  const from = ids.indexOf(draggedId);
  const to = ids.indexOf(targetId);
  ids.splice(from, 1);
  ids.splice(to, 0, draggedId);
  await api(`/api/boards/${encodeURIComponent(state.currentBoardId)}/reorder`, { method: "POST", body: JSON.stringify({ image_ids: ids }) });
  await loadBoardImages(draggedId);
}

export function selectImage(image) {
  state.currentImage = image;
  const hasImage = Boolean(image);
  if (hasImage) state.canvasDismissed = false;
  $("#canvasEmpty").hidden = hasImage;
  $("#selectedImage").hidden = !hasImage;
  $("#clearSelectedImage").hidden = !hasImage;
  $("#frameBar").hidden = !hasImage;
  $("#frameDetails").hidden = !hasImage || !state.frameDetailsOpen;
  if (!image) {
    renderBoardThumbnails();
    renderComposerContext();
    return;
  }
  $("#selectedImage").src = image.image_url;
  $("#selectedImage").alt = image.raw_prompt;
  $("#framePrompt").textContent = image.raw_prompt || "UNTITLED FRAME";
  $("#framePrompt").title = image.raw_prompt || "";
  const favorite = Boolean(image.favorite);
  const favoriteButton = $("#favoriteButton");
  favoriteButton.classList.toggle("active", favorite);
  favoriteButton.setAttribute("aria-pressed", String(favorite));
  favoriteButton.title = favorite ? "Remove from favorites" : "Favorite this frame";
  $("span", favoriteButton).textContent = favorite ? "★" : "☆";
  $("#moveBoard").value = "";
  $("#downloadButton").href = image.image_url;
  $("#reuseButton").hidden = isReference(image);
  $("#reusePromptButton").hidden = isReference(image);
  renderFrameGlance(image);
  renderFrameDetails(image);
  renderBoardThumbnails();
  renderComposerContext();
}

// The strip beside the prompt carries only what gets asked for constantly while comparing two
// frames. Everything else — the shape, the date, the recipe, the three prompts — is one click
// away, because reading it is a deliberate act rather than a glance, and a panel that shows
// everything at once is one nobody reads at all.
function renderFrameGlance(image) {
  const glance = $("#frameGlance");
  glance.replaceChildren();
  const route = document.createElement("b");
  route.textContent = presetLabel(image.preset);
  glance.append(route);
  // A pasted reference was never sampled, so it has no steps and no seed — the zeroes the row
  // would print are the absence of a run, not a record of one.
  if (isReference(image)) {
    const frame = document.createElement("span");
    frame.textContent = `${image.width} × ${image.height}`;
    glance.append(frame);
    return;
  }
  const steps = document.createElement("span");
  steps.textContent = `${image.steps} STEPS`;
  const seed = document.createElement("button");
  seed.type = "button";
  seed.className = "frame-seed";
  seed.textContent = frameSeed(image);
  seed.title = "Copy this seed";
  seed.addEventListener("click", () => copySeed(seed, frameSeed(image)));
  glance.append(steps, seed);
}

async function copySeed(button, seed) {
  try {
    await navigator.clipboard.writeText(String(seed));
    button.classList.add("copied");
    window.setTimeout(() => button.classList.remove("copied"), 900);
  } catch (_) { /* Clipboard permission is the browser's to refuse; the seed is on screen anyway. */ }
}

function specRow(label, value) {
  const row = document.createElement("div");
  const name = document.createElement("span");
  name.textContent = label;
  const body = document.createElement("b");
  body.textContent = value;
  row.append(name, body);
  return row;
}

function recipeChip(label, detail, kind) {
  const chip = document.createElement("span");
  chip.className = `recipe-chip ${kind}`;
  const name = document.createElement("b");
  name.textContent = label;
  chip.append(name);
  if (detail) {
    const strength = document.createElement("i");
    strength.textContent = detail;
    chip.append(strength);
  }
  return chip;
}

function recipeGroup(title, chips, empty) {
  const group = document.createElement("div");
  group.className = "recipe-group";
  const heading = document.createElement("span");
  heading.textContent = title;
  group.append(heading);
  const row = document.createElement("div");
  if (chips.length) row.append(...chips);
  else {
    const none = document.createElement("em");
    none.textContent = empty;
    row.append(none);
  }
  group.append(row);
  return group;
}

// A frame is worth reusing only if the record says what actually went into it, so the panel
// answers the questions the four-cell strip could not: how many steps, which adapters at which
// strength, which styles were switched on, and what the enhancer did to the words.
function renderFrameDetails(image) {
  const reference = isReference(image);
  $("#frameRecipe").hidden = reference;
  $("#framePrompts").hidden = reference;
  const spec = $("#frameSpec");
  spec.replaceChildren();
  spec.append(
    specRow("ROUTE", presetLabel(image.preset)),
    specRow("FRAME", `${image.width} × ${image.height}`),
  );
  if (reference) {
    spec.append(specRow("BOARD", image.board_name || "--"), specRow("CREATED", formatDate(image.created_at)));
    return;
  }
  spec.append(specRow("STEPS", String(image.steps)));
  if (frameUsesGuidance(image)) spec.append(specRow("GUIDANCE", Number(image.guidance).toFixed(1)));
  spec.append(
    specRow("SEED", frameSeed(image)),
    specRow("BOARD", image.board_name || "--"),
    specRow("CREATED", formatDate(image.created_at)),
  );
  if (image.enhance && image.enhancer_model) spec.append(specRow("ENHANCER", image.enhancer_model));

  const recipe = $("#frameRecipe");
  recipe.replaceChildren();
  const loras = (image.loras || []).map((lora) => {
    const known = state.loras.find((item) => item.name === lora.name);
    return recipeChip(known ? loraLabel(known) : (lora.name || "LORA"), `× ${Number(lora.strength ?? 1)}`, "trained");
  });
  // Older frames recorded only the ids, and a style can be renamed or deleted after the fact, so
  // the names written into the frame win and the live library is only the fallback.
  const recorded = image.metadata?.style_details;
  const styles = Array.isArray(recorded) && recorded.length
    ? recorded.map((style) => recipeChip(style.name || "STYLE", "", "written"))
    : (image.metadata?.styles || []).map((id) => {
      const known = state.styles.find((style) => style.id === id);
      return recipeChip(known ? known.name : "REMOVED STYLE", "", "written");
    });
  recipe.append(
    recipeGroup("LORAS", loras, "NONE"),
    recipeGroup("STYLES", styles, "NONE"),
  );
  if (frameUsesGuidance(image) && image.negative_prompt) {
    const negative = document.createElement("div");
    negative.className = "recipe-group";
    const heading = document.createElement("span");
    heading.textContent = "NEGATIVE";
    const body = document.createElement("p");
    body.textContent = image.negative_prompt;
    negative.append(heading, body);
    recipe.append(negative);
  }

  $("#recordRaw").textContent = image.raw_prompt || "--";
  $("#recordEnhanced").textContent = image.enhanced_prompt || "NOT USED";
  $("#recordFinal").textContent = image.final_prompt || "--";
  $("#recordEnhancedLabel").classList.toggle("muted", !image.enhanced_prompt);
}

function toggleFrameDetails(open = !state.frameDetailsOpen) {
  state.frameDetailsOpen = open;
  $("#frameDetails").hidden = !open || !state.currentImage;
  $("#frameDetailsToggle").setAttribute("aria-expanded", String(open));
  $("#frameDetailsToggle").classList.toggle("open", open);
}

function clearSelectedImage() {
  if (!state.currentImage) return;
  const imageId = state.currentImage.id;
  state.canvasDismissed = true;
  selectImage(null);
  if (state.chatAttachments.includes(imageId)) removeChatAttachment(imageId);
}

export function openImageViewer(source = state.currentImage) {
  const resolved = source && (source.image_url || source.url) ? source : findImage(source || {});
  const url = resolved.image_url || resolved.url;
  if (!url) return;
  const image = $("#imageViewerImage");
  image.src = url;
  image.alt = resolved.raw_prompt || resolved.prompt || "Selected generation";
  $("#imageViewerDialog").showModal();
}

async function toggleFavorite(image = state.currentImage) {
  if (!image) return;
  const updated = await api(`/api/images/${encodeURIComponent(image.id)}`, { method: "POST", body: JSON.stringify({ favorite: !image.favorite }) });
  const local = state.boardImages.find((item) => item.id === updated.id);
  if (local) Object.assign(local, updated);
  if (state.detailImage?.id === updated.id) state.detailImage = updated;
  if (state.currentImage?.id === updated.id) selectImage(local || updated);
  await refreshBoards();
}

async function moveSelectedImage() {
  const boardId = $("#moveBoard").value;
  if (!state.currentImage || !boardId || boardId === state.currentImage.board_id) return;
  await api(`/api/images/${encodeURIComponent(state.currentImage.id)}`, { method: "POST", body: JSON.stringify({ board_id: boardId }) });
  await refreshBoards();
  await loadBoardImages();
}

export async function loadGallery() {
  const params = new URLSearchParams();
  if ($("#galleryBoard").value) params.set("board_id", $("#galleryBoard").value);
  if ($("#gallerySearch").value.trim()) params.set("q", $("#gallerySearch").value.trim());
  if (state.galleryFavorites) params.set("favorite", "1");
  const data = await api(`/api/images?${params}`);
  state.galleryImages = data.images;
  renderGallery();
}

// Kept by image id for the same reason the rail keeps its tiles: a re-pack after a window resize
// moves nodes around, and moving a node does not restart its image load, while rebuilding it does.
const galleryTiles = new Map();

function galleryTile(image, requestSize) {
  let card = galleryTiles.get(image.id);
  if (!card) {
    card = document.createElement("button");
    card.type = "button";
    card.className = "gallery-card";
    const img = document.createElement("img");
    img.loading = "lazy";
    // Short: a failed thumbnail paints its alt inside the frame, and the prompt would fill it.
    img.alt = "Gallery frame";
    const meta = document.createElement("span");
    meta.className = "gallery-card-meta";
    meta.append(document.createElement("b"), document.createElement("small"));
    card.append(img, meta, tileDeleteControl(() => deleteImage(card.imageRecord)));
    card.addEventListener("click", () => openGalleryDetail(card.imageRecord));
    galleryTiles.set(image.id, card);
  }
  card.imageRecord = image;
  // The prompt is the card's title rather than its body: it is what a hover answers and what the
  // dialog opens onto, so the grid stays a grid of pictures.
  card.title = image.raw_prompt || "";
  const img = card.querySelector("img");
  const source = thumbnailUrl(image, requestSize);
  if (img.getAttribute("src") !== source) img.src = source;
  $(".gallery-card-meta b", card).textContent = image.board_name.toUpperCase();
  $(".gallery-card-meta small", card).textContent = `${image.width} × ${image.height}`;
  const star = card.querySelector("i");
  if (image.favorite && !star) {
    const mark = document.createElement("i");
    mark.textContent = "★";
    card.append(mark);
  } else if (!image.favorite && star) {
    star.remove();
  }
  return card;
}

function renderGallery() {
  const grid = $("#galleryGrid");
  const images = state.galleryImages;
  $("#galleryCount").textContent = images.length;
  $("#galleryEmpty").hidden = images.length > 0;
  const sizes = justifiedRailSizes(images, packContentWidth(grid, 900), GALLERY_PACK);
  const cards = images.map((image) => {
    const { width, height } = sizes.get(image.id);
    // The request bounds the short side, so a true-ratio tile asks for its smaller dimension.
    const card = galleryTile(image, Math.min(width, height));
    const wide = `${width}px`;
    const tall = `${height}px`;
    if (card.style.width !== wide) card.style.width = wide;
    if (card.style.height !== tall) card.style.height = tall;
    return card;
  });
  const shown = new Set(images.map((image) => image.id));
  for (const id of galleryTiles.keys()) if (!shown.has(id)) galleryTiles.delete(id);
  const settled = cards.length === grid.children.length && cards.every((card, index) => grid.children[index] === card);
  if (!settled) grid.replaceChildren(...cards);
}

function openGalleryDetail(image) {
  state.detailImage = image;
  $("#detailImage").src = image.image_url;
  $("#detailImage").alt = image.raw_prompt;
  $("#detailBoard").textContent = image.board_name.toUpperCase();
  $("#detailPrompt").textContent = image.raw_prompt || "Untitled frame";
  $("#detailFavorite").textContent = `${image.favorite ? "★" : "☆"} ${image.favorite ? "FAVORITED" : "FAVORITE"}`;
  $("#detailDownload").href = image.image_url;
  $("#galleryDetailDialog").showModal();
}

export function initImages() {
  // The rail packs its rows against the width it actually has, and three things change that: the
  // inspector is dragged, the rail's scrollbar comes and goes with the board's length, and the
  // images panel starts hidden. Any of them leaves a packed-for-the-wrong-width rail wrapping its
  // last tile onto a line of its own, so watch the box and re-pack. The rAF hop coalesces the
  // continuous fire of a drag into one render per frame.
  new ResizeObserver(() => requestAnimationFrame(renderBoardThumbnails)).observe($("#boardThumbnails"));
  // The gallery packs against the page column, which the window resize and the scrollbar both move.
  new ResizeObserver(() => requestAnimationFrame(renderGallery)).observe($("#galleryGrid"));
  $("#clearSelectedImage").addEventListener("click", clearSelectedImage);
  $("#selectedImage").addEventListener("click", () => openImageViewer(state.currentImage));
  $("#selectedImage").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openImageViewer();
    }
  });
  $("#closeImageViewer").addEventListener("click", () => $("#imageViewerDialog").close());
  $("#imageViewerDialog").addEventListener("click", (event) => event.currentTarget.close());
  $("#favoriteButton").addEventListener("click", () => toggleFavorite());
  $("#deleteFrameButton").addEventListener("click", () => { if (state.currentImage) deleteBoardFrame(state.currentImage); });
  $("#reuseButton").addEventListener("click", () => reuseImage());
  $("#reusePromptButton").addEventListener("click", () => reusePrompt());
  $("#frameDetailsToggle").addEventListener("click", () => toggleFrameDetails());
  // The prompt is clipped to one line in the bar, so the bar itself is the way to the full text --
  // and it toggles, because a press that opens something has to be the press that closes it.
  // Anything that does its own job on click keeps it; the toggle button has its own handler above.
  $("#frameBar").addEventListener("click", (event) => {
    if (event.target.closest("button, a, select, input, textarea, label")) return;
    toggleFrameDetails();
  });
  $("#moveBoard").addEventListener("change", moveSelectedImage);
  $("#railSearch").addEventListener("input", (event) => { state.railQuery = event.target.value; renderBoardThumbnails(); });
  $("#railFavoriteFilter").addEventListener("click", () => { state.railFavorites = !state.railFavorites; $("#railFavoriteFilter").classList.toggle("active", state.railFavorites); renderBoardThumbnails(); });
  $("#gallerySearch").addEventListener("input", debounce(loadGallery));
  $("#galleryBoard").addEventListener("change", loadGallery);
  $("#galleryFavorites").addEventListener("click", () => { state.galleryFavorites = !state.galleryFavorites; $("#galleryFavorites").classList.toggle("active", state.galleryFavorites); loadGallery(); });
  $("#closeDetailDialog").addEventListener("click", () => $("#galleryDetailDialog").close());
  $("#galleryDetailDialog").addEventListener("click", (event) => {
    if (event.target === event.currentTarget) event.currentTarget.close();
  });
  $("#detailDelete").addEventListener("click", async () => {
    if (await deleteImage(state.detailImage)) $("#galleryDetailDialog").close();
  });
  $("#detailFavorite").addEventListener("click", async () => { await toggleFavorite(state.detailImage); await loadGallery(); const refreshed = state.galleryImages.find((image) => image.id === state.detailImage.id) || state.detailImage; openGalleryDetail(refreshed); });
  $("#detailReuse").addEventListener("click", () => { $("#galleryDetailDialog").close(); reuseImage(state.detailImage); });
  $("#detailReusePrompt").addEventListener("click", () => { $("#galleryDetailDialog").close(); reusePrompt(state.detailImage); });
}
