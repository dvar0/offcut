import { pickerImages } from "./chat-composer.js";
import { paintGenerationState, pollProgress, updateProgress } from "./generation.js";
import { gridColumnWidth, thumbnailUrl } from "./image-layout.js";
import { openImageViewer } from "./images.js";
import { $, $$, api, dismissOnBackdrop, showFormError, state } from "./shared.js";
import { navigate, scheduleBoardDraft } from "./workspace.js";

// Both "use this frame as a cover" controls list the whole library, so one entry can be given a
// cover from the canvas or from the gallery without opening its editor.
export function populateCoverTargets() {
  for (const selector of ["#coverTarget", "#detailCoverTarget"]) {
    const select = $(selector);
    if (!select) continue;
    select.replaceChildren();
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "CHOOSE…";
    select.append(placeholder);
    for (const entry of libraryEntries()) {
      const option = document.createElement("option");
      option.value = entry.key;
      option.textContent = `${entry.kind === "trained" ? "◆" : "✎"} ${entry.name}`;
      select.append(option);
    }
    select.value = "";
  }
}

// A LoRA and a saved style are different machinery — one is a file of weights the sampler patches
// into the model and carries a strength, the other is text prefixed onto the prompt — but choosing
// between them is one decision, "make it look like this". So they are listed together and told
// apart by a badge rather than by living in separate sections. Everything below works in terms of
// these entries; the two selection lists underneath stay separate because the payload needs them
// that way.
const LIBRARY_BADGES = { trained: "◆ TRAINED", written: "✎ WRITTEN" };

// Mirrors offcut_cli.LORA_STRENGTH_LIMIT and offcut_store.DEFAULT_LORA_STRENGTH. A field wider than
// the server's bound just fails on submit; a narrower one silently forbids a value the generator
// would have taken, so these two have to move together with the Python.
const LORA_STRENGTH_LIMIT = 100;

const DEFAULT_LORA_STRENGTH = 0.8;

export function loraLabel(lora) {
  return lora.display_name || lora.name.replace(/^offcut_(raw_)?/, "").replaceAll("_", " ").toUpperCase();
}

function libraryEntries() {
  const byName = (first, second) => first.name.localeCompare(second.name, undefined, { sensitivity: "base" });
  const trained = state.loras.map((lora) => ({
    kind: "trained",
    key: `lora:${lora.name}`,
    id: lora.name,
    name: loraLabel(lora),
    detail: lora.notes || lora.summary || lora.trigger || `${lora.name}.safetensors`,
    search: [lora.name, lora.display_name, lora.trigger, lora.summary, lora.notes].filter(Boolean).join(" ").toLowerCase(),
    cover: lora.cover_url,
    lora,
  })).sort(byName);
  const written = state.styles.map((style) => ({
    kind: "written",
    badge: style.kind === "scene" ? "SCENE RECIPE" : "✎ WRITTEN",
    key: `style:${style.id}`,
    id: style.id,
    name: style.name,
    detail: style.description || style.style_text,
    search: [style.name, style.description, style.style_text].filter(Boolean).join(" ").toLowerCase(),
    cover: style.reference_url,
    style,
  })).sort(byName);
  // Grouped rather than interleaved so the kind filters hide a contiguous run and nothing appears
  // to jump position when one is switched on.
  return [...trained, ...written];
}

function findEntry(key) {
  return libraryEntries().find((entry) => entry.key === key) || null;
}

function filterEntries(entries, kind, query) {
  const needle = query.trim().toLowerCase();
  return entries.filter((entry) => (kind === "all" || entry.kind === kind)
    && (!needle || entry.search.includes(needle)));
}

function entryActive(entry) {
  return entry.kind === "trained"
    ? state.selectedLoras.some((item) => item.name === entry.id)
    : state.selectedStyles.includes(entry.id);
}

// Turning something on is the whole interaction: there is no separate commit step, because the
// one this replaced let a chosen LoRA and a typed strength sit in the picker looking selected
// while generation quietly ignored them.
function toggleEntry(entry) {
  if (entry.kind === "trained") {
    const index = state.selectedLoras.findIndex((item) => item.name === entry.id);
    if (index >= 0) state.selectedLoras.splice(index, 1);
    else state.selectedLoras.push({ name: entry.id, strength: loraDefaultStrength(entry.lora) });
  } else if (state.selectedStyles.includes(entry.id)) {
    state.selectedStyles = state.selectedStyles.filter((id) => id !== entry.id);
  } else {
    state.selectedStyles.push(entry.id);
  }
  renderActiveLibrary();
  scheduleBoardDraft();
}

function loraDefaultStrength(lora) {
  const value = Number(lora?.default_strength);
  return Number.isFinite(value) ? value : DEFAULT_LORA_STRENGTH;
}

function coverThumbnail(entry, size) {
  if (!entry.cover) return null;
  const image = document.createElement("img");
  image.src = thumbnailUrl({ image_url: entry.cover }, size);
  image.alt = "";
  image.loading = "lazy";
  return image;
}

// At picker size the placeholder is a large empty square, so it says why it is empty rather than
// leaving the entry looking like a style whose cover failed to load.
function coverPlaceholder(entry, hint = false) {
  const blank = document.createElement("i");
  blank.textContent = entry.name.slice(0, 2).toUpperCase();
  if (!hint) return blank;
  const wrap = document.createElement("span");
  wrap.className = "cover-placeholder";
  const note = document.createElement("small");
  note.textContent = "NO COVER";
  wrap.append(blank, note);
  return wrap;
}

function openEntryEditor(entry) {
  if (entry.kind === "trained") openLoraDialog(entry.lora);
  else openStyleDialog(entry.style);
}

// The panel shows only what is switched on. Everything else lives one click away in the picker,
// which is the only place with room to draw a cover big enough to recognise.
export function renderActiveLibrary() {
  const list = $("#activeLibrary");
  if (!list) return;
  state.selectedStyles = state.selectedStyles.filter((id) => state.styles.some((style) => style.id === id));
  state.selectedLoras = state.selectedLoras.filter((item) => state.loras.some((lora) => lora.name === item.name));
  const entries = libraryEntries();
  const active = [
    ...state.selectedLoras.map((stack) => ({
      entry: entries.find((item) => item.kind === "trained" && item.id === stack.name),
      stack,
    })),
    ...state.selectedStyles.map((id) => ({ entry: entries.find((item) => item.kind === "written" && item.id === id) })),
  ].filter((row) => row.entry);
  list.replaceChildren();
  for (const { entry, stack } of active) {
    const chip = document.createElement("div");
    chip.className = `library-chip ${entry.kind}`;
    const open = document.createElement("button");
    open.type = "button";
    open.className = "library-chip-open";
    open.title = `Edit ${entry.name}`;
    open.append(coverThumbnail(entry, 34) || coverPlaceholder(entry));
    const copy = document.createElement("span");
    const name = document.createElement("b");
    name.textContent = entry.name;
    const detail = document.createElement("small");
    detail.textContent = entry.kind === "trained" ? (entry.lora.trigger || entry.detail) : entry.detail;
    copy.append(name, detail);
    open.append(copy);
    open.addEventListener("click", () => openEntryEditor(entry));
    chip.append(open);
    if (entry.kind === "trained") {
      const strength = document.createElement("input");
      strength.type = "number";
      strength.className = "library-chip-strength";
      strength.min = String(-LORA_STRENGTH_LIMIT);
      strength.max = String(LORA_STRENGTH_LIMIT);
      strength.step = "0.05";
      strength.value = String(stack.strength);
      strength.ariaLabel = `${entry.name} strength`;
      strength.title = "Around 1.0 is normal; most LoRAs degrade well before 2.0. The range is wide on purpose, for experimenting.";
      strength.addEventListener("input", () => {
        const value = Number(strength.value);
        // A half-typed "-" or "" is not a strength yet. Leaving the stack alone until the field
        // parses keeps a draft save from writing a NaN the server would reject.
        if (!Number.isFinite(value)) return;
        stack.strength = value;
        scheduleBoardDraft();
      });
      chip.append(strength);
    }
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "library-chip-remove";
    remove.textContent = "×";
    remove.ariaLabel = `Turn off ${entry.name}`;
    remove.addEventListener("click", () => toggleEntry(entry));
    chip.append(remove);
    list.append(chip);
  }
  $("#styleCount").textContent = `${active.length} ACTIVE`;
}

export async function loadLoras() {
  try {
    const data = await api("/api/loras");
    state.loras = data.loras || [];
    $("#loraDirectories").replaceChildren(...(data.directories || []).map((directory) => {
      const item = document.createElement("li");
      item.textContent = directory;
      return item;
    }));
  } catch (error) {
    showFormError(error.message);
  }
  renderActiveLibrary();
  renderLibraryPage();
  populateCoverTargets();
}

export async function loadStyles() {
  try {
    const data = await api("/api/styles");
    state.styles = data.styles || [];
  } catch (error) {
    state.styles = [];
    showFormError(error.message);
  }
  renderActiveLibrary();
  renderLibraryPage();
  populateCoverTargets();
}

function syncKindButtons(group, kind) {
  for (const button of $$("button", group)) button.classList.toggle("active", button.dataset.kind === kind);
}

function openLibraryPicker() {
  $("#pickerSearch").value = state.pickerQuery;
  syncKindButtons($("#pickerFilter"), state.pickerKind);
  // The dialog opens before its grid renders. A closed <dialog> lays nothing out, so tile
  // heights would read as zero and fitPickerGrid would collapse the window; opening first also
  // hands gridColumnWidth a real column width on the first open instead of the 150px stand-in.
  $("#libraryPickerDialog").showModal();
  renderLibraryPicker();
}

function renderLibraryPicker() {
  const grid = $("#pickerGrid");
  if (!grid) return;
  const entries = filterEntries(libraryEntries(), state.pickerKind, state.pickerQuery);
  grid.replaceChildren();
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "picker-empty";
    empty.textContent = state.styles.length + state.loras.length ? "NOTHING MATCHES THIS FILTER" : "NOTHING IN THE LIBRARY YET";
    grid.append(empty);
  }
  const cellWidth = gridColumnWidth(grid, 212);
  for (const entry of entries) {
    const tile = document.createElement("div");
    tile.className = `library-tile ${entry.kind}`;
    tile.classList.toggle("active", entryActive(entry));
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "library-tile-toggle";
    toggle.setAttribute("aria-pressed", String(entryActive(entry)));
    toggle.title = entry.kind === "written" ? entry.style.style_text : (entry.lora.summary || entry.lora.trigger || entry.name);
    const figure = document.createElement("div");
    figure.className = "library-tile-figure";
    figure.append(coverThumbnail(entry, cellWidth) || coverPlaceholder(entry, true));
    const badge = document.createElement("em");
    badge.className = "library-badge";
    badge.textContent = entry.badge || LIBRARY_BADGES[entry.kind];
    const mark = document.createElement("i");
    mark.className = "library-tile-mark";
    figure.append(badge, mark);
    const copy = document.createElement("span");
    const name = document.createElement("b");
    name.textContent = entry.name;
    const detail = document.createElement("small");
    detail.textContent = entry.detail;
    copy.append(name, detail);
    toggle.append(figure, copy);
    // Only the tile itself is re-drawn on a toggle: repainting the grid would restart every
    // image load and throw away the scroll position mid-browse.
    toggle.addEventListener("click", () => {
      toggleEntry(entry);
      const active = entryActive(entry);
      tile.classList.toggle("active", active);
      toggle.setAttribute("aria-pressed", String(active));
      refreshPickerCount();
    });
    tile.append(toggle);
    const actions = document.createElement("div");
    actions.className = "library-tile-actions";
    if (entry.kind === "written") {
      const insert = document.createElement("button");
      insert.type = "button";
      insert.textContent = "INSERT";
      insert.title = "Write this style into the prompt so it can be edited";
      insert.addEventListener("click", () => {
        insertStyleIntoPrompt(entry.style);
        $("#libraryPickerDialog").close();
      });
      actions.append(insert);
    }
    const edit = document.createElement("button");
    edit.type = "button";
    edit.textContent = "EDIT";
    edit.addEventListener("click", () => openEntryEditor(entry));
    actions.append(edit);
    tile.append(actions);
    grid.append(tile);
  }
  fitPickerGrid(grid);
  refreshPickerCount();
}

// The picker window is fixed at two rows of tiles and scrolls for anything beyond, so the
// dialog holds one height no matter how the kind filter or the search thins the list. A row's
// height rides on the figure's aspect ratio, so it moves with the dialog's width and is
// measured off a laid-out tile rather than hardcoded. An empty result keeps the height the
// last populated render set, so "NOTHING MATCHES THIS FILTER" does not collapse the window
// either.
const PICKER_ROWS = 2;

const PICKER_ROW_GAP = 10; // mirrors the gap on .library-picker-grid

const PICKER_GRID_PADDING = 4; // 2px top + 2px bottom on .library-picker-grid

function pickerGridHeight(tile, rows) {
  return rows * tile.offsetHeight + (rows - 1) * PICKER_ROW_GAP + PICKER_GRID_PADDING;
}

function fitPickerGrid(grid) {
  const tile = $(".library-tile", grid);
  if (!tile) return;
  // A list shorter than the window would otherwise reserve two rows of tall tiles and leave the
  // second one as blank dialog, so the window is the smaller of the cap and what is in it.
  const columns = getComputedStyle(grid).gridTemplateColumns.split(" ").filter(Boolean).length || 1;
  const wanted = Math.max(1, Math.min(PICKER_ROWS, Math.ceil($$(".library-tile", grid).length / columns)));
  grid.style.height = `${pickerGridHeight(tile, wanted)}px`;
  // On a short screen the dialog's own cap is less than the window asked for, and the grid is a
  // shrinkable flex item, so it gets what is left rather than what it set — which puts the fold
  // through the middle of a tile. Re-fit to whole rows of whatever was actually granted.
  const granted = grid.clientHeight - PICKER_GRID_PADDING + PICKER_ROW_GAP;
  const rows = Math.max(1, Math.min(wanted, Math.floor(granted / (tile.offsetHeight + PICKER_ROW_GAP))));
  if (rows !== wanted) grid.style.height = `${pickerGridHeight(tile, rows)}px`;
}

function refreshPickerCount() {
  const count = state.selectedLoras.length + state.selectedStyles.length;
  $("#pickerCount").textContent = `${count} ACTIVE`;
}

function insertStyleIntoPrompt(style) {
  const prompt = $("#prompt");
  const existing = prompt.value.trim();
  if (existing.toLowerCase().startsWith(style.style_text.toLowerCase())) return;
  prompt.value = existing ? `${style.style_text}, ${existing}` : style.style_text;
  $("#characterCount").textContent = `${prompt.value.length} / 20K`;
  scheduleBoardDraft();
}

function renderLibraryPage() {
  const grid = $("#stylesGrid");
  if (!grid) return;
  const all = libraryEntries();
  const entries = filterEntries(all, state.libraryKind, state.libraryQuery);
  const empty = $("#stylesEmpty");
  empty.hidden = Boolean(entries.length);
  empty.textContent = all.length ? "NOTHING MATCHES THIS FILTER" : "NOTHING IN THE LIBRARY YET";
  grid.replaceChildren();
  const cellWidth = gridColumnWidth(grid, 320);
  for (const entry of entries) {
    const card = document.createElement("article");
    card.className = `style-card ${entry.kind}`;
    const figure = document.createElement("div");
    figure.className = "style-card-figure";
    const cover = coverThumbnail(entry, cellWidth);
    if (cover) {
      cover.alt = `${entry.name} cover`;
      cover.addEventListener("click", () => openImageViewer({ image_url: entry.cover, raw_prompt: entry.name }));
      figure.append(cover);
    } else {
      const blank = document.createElement("span");
      blank.textContent = "NO COVER";
      figure.append(blank);
    }
    const badge = document.createElement("em");
    badge.className = "library-badge";
    badge.textContent = entry.badge || LIBRARY_BADGES[entry.kind];
    figure.append(badge);
    const copy = document.createElement("div");
    copy.className = "style-card-copy";
    const name = document.createElement("h2");
    name.textContent = entry.name;
    const description = document.createElement("p");
    description.className = "style-card-description";
    const text = document.createElement("p");
    text.className = "style-card-text";
    const origin = document.createElement("b");
    if (entry.kind === "written") {
      description.textContent = entry.style.description || "No description";
      text.textContent = entry.style.style_text;
      origin.textContent = entry.style.source === "agent" ? "SAVED BY THE AGENT" : "SAVED BY YOU";
    } else {
      description.textContent = entry.lora.notes || entry.lora.summary || "No notes yet";
      // The trigger is the one thing about a LoRA that changes the prompt, and it is prepended
      // whether or not it is written out, so it is worth showing on the card.
      text.textContent = entry.lora.trigger ? `Trigger: ${entry.lora.trigger}` : "No trigger phrase";
      origin.textContent = `${entry.lora.name}.safetensors · DEFAULT ${loraDefaultStrength(entry.lora).toFixed(2)}`;
    }
    copy.append(name, description, text, origin);
    const actions = document.createElement("div");
    actions.className = "style-card-actions";
    const use = document.createElement("button");
    use.type = "button";
    const paintUse = () => {
      const active = entryActive(entry);
      use.textContent = active ? "IN USE" : "USE";
      use.classList.toggle("active", active);
    };
    paintUse();
    use.addEventListener("click", () => { toggleEntry(entry); paintUse(); });
    const edit = document.createElement("button");
    edit.type = "button";
    edit.textContent = "EDIT";
    edit.addEventListener("click", () => openEntryEditor(entry));
    const coverButton = document.createElement("button");
    coverButton.type = "button";
    coverButton.textContent = entry.cover ? "NEW COVER" : "COVER";
    coverButton.title = entry.cover
      ? "Replace this cover with a fresh one from the recipe"
      : "Generate this entry's cover from the shared recipe";
    coverButton.addEventListener("click", async () => {
      if (state.coverRunning || state.generating) return;
      coverButton.disabled = true;
      coverButton.textContent = "…";
      try {
        await generateCover(entry);
        await Promise.all([loadStyles(), loadLoras()]);
        renderLibraryPage();
      } catch (error) {
        coverButton.disabled = false;
        coverButton.textContent = entry.cover ? "NEW COVER" : "COVER";
        const status = $("#coverRunStatus");
        status.hidden = false;
        status.textContent = error.data?.cancelled ? "STOPPED." : error.message;
      }
    });
    actions.append(use, edit, coverButton);
    // A LoRA has no delete: the file on disk is what makes it exist, and a profile is only the
    // cover and notes hung off it.
    if (entry.kind === "written") {
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "DELETE";
      remove.addEventListener("click", async () => {
        if (!confirm(`Delete the style "${entry.name}"?`)) return;
        try {
          await api(`/api/styles/${encodeURIComponent(entry.id)}/delete`, { method: "POST", body: "{}" });
          state.selectedStyles = state.selectedStyles.filter((id) => id !== entry.id);
          await loadStyles();
          scheduleBoardDraft();
        } catch (error) { showFormError(error.message); }
      });
      actions.append(remove);
    }
    card.append(figure, copy, actions);
    grid.append(card);
  }
}

// One cover, generated on the server from the shared recipe. The browser sends only which entry
// it is for: everything the sampler reads lives in the recipe, which is what makes two covers
// comparable.
async function generateCover(entry) {
  const target = entry.kind === "written" ? entry.id : entry.lora.name;
  const result = await api("/api/covers", {
    method: "POST",
    body: JSON.stringify({ target_kind: entry.kind === "written" ? "style" : "lora", target }),
  });
  return result;
}

// The queue lives in this tab exactly as the create panel's batch does: the engine takes one frame
// at a time, so a run is a loop here, and a stop drops the rest of the queue here while
// interrupting the frame in flight on the server.
async function runCoverBatch(mode) {
  if (state.coverRunning || state.generating) return;
  const status = $("#coverRunStatus");
  const entries = libraryEntries().filter((entry) => mode === "all" || !entry.cover);
  if (!entries.length) {
    status.hidden = false;
    status.textContent = mode === "all" ? "THE LIBRARY IS EMPTY." : "EVERY ENTRY ALREADY HAS A COVER.";
    return;
  }
  const label = mode === "all" ? "REGENERATE ALL" : "FILL MISSING COVERS";
  if (!confirm(`${label}: generate ${entries.length} cover${entries.length === 1 ? "" : "s"} from the current recipe?`)) return;
  status.hidden = true;
  status.textContent = "";
  let done = 0;
  setCoverGenerating(true, { total: entries.length, index: 0, name: entries[0].name });
  try {
    for (const [index, entry] of entries.entries()) {
      // Either stop ends the run: the one on this strip, or the create panel's, which is live
      // because a cover run drives the same engine and the same panel state.
      if (state.coverStopping || state.stopping) break;
      state.coverRun = { total: entries.length, index, name: entry.name };
      paintCoverProgress();
      await generateCover(entry);
      done += 1;
      await Promise.all([loadStyles(), loadLoras()]);
      renderLibraryPage();
    }
    status.hidden = false;
    status.textContent = done < entries.length
      ? `STOPPED AFTER ${done} OF ${entries.length}.`
      : `DONE · ${done} COVER${done === 1 ? "" : "S"}.`;
  } catch (error) {
    status.hidden = false;
    status.textContent = error.data?.cancelled
      ? `STOPPED AFTER ${done} OF ${entries.length}.`
      : `FAILED AFTER ${done} OF ${entries.length}: ${error.message}`;
  } finally {
    setCoverGenerating(false);
  }
}

// A cover run is the engine, so it drives the same panel state every other path does: the Create
// panel's generate button goes dead, its progress bar fills, and the runtime pill reads GENERATING.
// Without this, Create looked idle while the GPU was busy and a second press queued behind a run
// the user could not see.
function setCoverGenerating(active, run = null) {
  state.coverRunning = active;
  state.coverStopping = false;
  state.coverRun = active ? run : null;
  paintCoverButtons();
  paintGenerationState(active);
  $("#coverProgress").hidden = !active;
  if (active) {
    updateProgress({ stage: "queued", fraction: 0, detail: "Submitting cover" });
    state.progressTimer = window.setInterval(pollProgress, 350);
  } else {
    window.clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
}

// Two readings of the same run: the engine's own fraction for the frame in flight, and how far
// through the queue that frame is. The bar shows the second, because a wall of covers is judged by
// how many are left rather than by how far into one of them the sampler is.
export function paintCoverProgress(progress = null) {
  const run = state.coverRun;
  if (!run) return;
  const frame = Math.max(0, Math.min(Number(progress?.fraction || 0), 1));
  const fraction = (run.index + frame) / run.total;
  $("#coverProgressStage").textContent = state.coverStopping || state.stopping
    ? "STOPPING"
    : String(progress?.stage || "queued").replaceAll("_", " ").toUpperCase();
  $("#coverProgressCount").textContent = `${run.index + 1} / ${run.total}`;
  $("#coverProgressBar").style.width = `${fraction * 100}%`;
  $("#coverProgressDetail").textContent = `${run.name} — ${progress?.detail || "Queued"}`;
  const elapsed = Math.max(0, performance.now() - state.progressStarted) / 1000;
  $("#coverProgressElapsed").textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${(elapsed % 60).toFixed(1).padStart(4, "0")}`;
}

// While a run is going the first button is the only way out of it, so it becomes the stop and the
// second one goes dead rather than queueing a second pass behind the first.
function paintCoverButtons() {
  const running = state.coverRunning;
  const fill = $("#fillCoversButton");
  const all = $("#regenerateCoversButton");
  if (!fill || !all) return;
  fill.textContent = running ? (state.coverStopping ? "STOPPING…" : "STOP") : "FILL MISSING";
  fill.disabled = running && state.coverStopping;
  all.textContent = running ? "RUNNING…" : "REGENERATE ALL";
  all.disabled = running;
}

// Same two halves as the create panel's stop: drop the rest of the queue here, interrupt the frame
// in flight there.
async function stopCoverBatch() {
  if (!state.coverRunning) return;
  state.coverStopping = true;
  paintCoverButtons();
  try {
    await api("/api/generate/cancel", { method: "POST", body: "{}" });
  } catch (_) {
    // Nothing was sampling yet; the loop still stops after the frame it is on.
  }
}

function openStyleDialog(style = null, referenceImage = null) {
  state.editingStyle = style;
  $("#styleDialogTitle").textContent = style ? "EDIT STYLE" : "SAVE A STYLE";
  $("#styleDialogEyebrow").textContent = style ? "STYLE LIBRARY" : "NEW STYLE";
  $("#styleName").value = style?.name || "";
  $("#styleKind").value = style?.kind || "art";
  $("#styleDescription").value = style?.description || "";
  $("#styleText").value = style?.style_text || "";
  $("#styleError").textContent = "";
  state.styleReferenceId = referenceImage?.id || style?.reference_image_id || null;
  renderCoverPreview($("#styleReference"), referenceImage?.image_url || style?.reference_url || null);
  paintStyleGuidance();
  $("#styleDialog").showModal();
}

// LoRA prompting metadata is user-authored and shared by the CLI, agent, and enhancer.
function openLoraDialog(lora) {
  state.editingLora = lora;
  state.loraCoverId = lora.cover_image_id || null;
  $("#loraDialogTitle").textContent = loraLabel(lora);
  $("#loraDisplayName").value = lora.display_name || "";
  $("#loraDefaultStrength").value = String(loraDefaultStrength(lora));
  $("#loraNotes").value = lora.notes || "";
  $("#loraTrigger").value = lora.trigger || "";
  $("#loraSummary").value = lora.summary || "";
  $("#loraPromptingNotes").value = lora.prompting_notes || "";
  $("#loraError").textContent = "";
  renderCoverPreview($("#loraCover"), lora.cover_url);
  const facts = $("#loraFacts");
  facts.replaceChildren();
  const rows = [["FILE", lora.path || `${lora.name}.safetensors`]];
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("span");
    term.textContent = label;
    const detail = document.createElement("p");
    detail.textContent = value;
    row.append(term, detail);
    facts.append(row);
  }
  $("#loraDialog").showModal();
}

function renderCoverPreview(node, url) {
  const image = $("img", node);
  const empty = $("span", node);
  image.hidden = !url;
  empty.hidden = Boolean(url);
  if (url) image.src = url;
  else image.removeAttribute("src");
}

// Opened on top of an editor that is itself a modal dialog, so it resolves rather than calling
// back: the editor awaits a frame and keeps every other field it was holding.
function openCoverPicker(eyebrow) {
  $("#coverPickerEyebrow").textContent = eyebrow;
  state.coverBoardId = [state.coverBoardId, state.currentBoardId, state.boards[0]?.id]
    .find((id) => state.boards.some((board) => board.id === id)) || null;
  const select = $("#coverPickerBoard");
  select.replaceChildren();
  for (const board of state.boards) {
    const option = document.createElement("option");
    option.value = board.id;
    option.textContent = `${board.name.toUpperCase()}${board.image_count ? ` (${board.image_count})` : ""}`;
    if (board.id === state.coverBoardId) option.selected = true;
    select.append(option);
  }
  renderCoverPickerGrid();
  $("#coverPickerDialog").showModal();
  return new Promise((resolve) => { state.coverPickerResolve = resolve; });
}

function resolveCoverPicker(image) {
  const resolve = state.coverPickerResolve;
  state.coverPickerResolve = null;
  $("#coverPickerDialog").close();
  if (resolve) resolve(image);
}

async function renderCoverPickerGrid() {
  const grid = $("#coverPickerGrid");
  grid.replaceChildren();
  const loading = document.createElement("p");
  loading.className = "picker-empty";
  loading.textContent = "LOADING";
  grid.append(loading);
  let images = [];
  try {
    images = state.coverBoardId ? await pickerImages(state.coverBoardId) : [];
  } catch (error) {
    loading.textContent = error.message;
    return;
  }
  const cellWidth = gridColumnWidth(grid, 96);
  grid.replaceChildren();
  if (!images.length) {
    const empty = document.createElement("p");
    empty.className = "picker-empty";
    empty.textContent = "THIS BOARD HAS NO FRAMES";
    grid.append(empty);
    return;
  }
  for (const image of images) {
    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "picker-cell";
    const thumb = document.createElement("img");
    thumb.src = thumbnailUrl(image, cellWidth);
    thumb.alt = "Board frame";
    thumb.loading = "lazy";
    cell.append(thumb);
    cell.addEventListener("click", () => resolveCoverPicker(image));
    grid.append(cell);
  }
}

// Setting a cover from a frame is the reverse of the editors: the frame is already chosen and the
// select names which library entry should wear it.
async function setLibraryCover(select, image) {
  const entry = findEntry(select.value);
  select.value = "";
  if (!entry || !image) return;
  const placeholder = select.options[0];
  try {
    if (entry.kind === "trained") {
      await api(`/api/loras/${encodeURIComponent(entry.id)}`, {
        method: "POST",
        body: JSON.stringify({ cover_image_id: image.id }),
      });
      await loadLoras();
    } else {
      await api(`/api/styles/${encodeURIComponent(entry.id)}`, {
        method: "POST",
        body: JSON.stringify({ reference_image_id: image.id }),
      });
      await loadStyles();
    }
    // populateCoverTargets rebuilt the options, so the flash lands on the fresh placeholder.
    const fresh = select.options[0];
    fresh.textContent = `COVER SET · ${entry.name}`;
    window.setTimeout(() => { fresh.textContent = "CHOOSE…"; }, 2000);
  } catch (error) {
    showFormError(error.message);
    placeholder.textContent = "CHOOSE…";
  }
}

function paintStyleGuidance() {
  const scene = $("#styleKind").value === "scene";
  $("#styleDialogTitle").textContent = `${state.editingStyle ? "EDIT" : "SAVE"} ${scene ? "SCENE RECIPE" : "ART STYLE"}`;
  $("#styleForm button[type=submit]").textContent = scene ? "SAVE RECIPE" : "SAVE STYLE";
  $("#styleGuidance").textContent = scene
    ? "Keep reusable camera, motion, setting, and composition. Describe the replaceable character as ‘the subject’. Leave the art medium to the LoRA or a separate art style."
    : "Describe medium, technique, texture, palette, and light without copying the reference subject or setting. This text will be prefixed onto unrelated prompts.";
  $("#styleText").placeholder = scene ? "Camera alongside the subject, falling through a violet vortex…" : "Medium, technique, texture, palette, and light";
}

export function initLibrary() {
  $("#browseLibrary").addEventListener("click", openLibraryPicker);
  $("#newStyleButton").addEventListener("click", () => openStyleDialog());
  $("#fillCoversButton").addEventListener("click", () => (state.coverRunning ? stopCoverBatch() : runCoverBatch("missing")));
  $("#regenerateCoversButton").addEventListener("click", () => runCoverBatch("all"));
  $("#closeStyleDialog").addEventListener("click", () => $("#styleDialog").close());
  $("#closeLoraDialog").addEventListener("click", () => $("#loraDialog").close());
  dismissOnBackdrop($("#styleDialog"));
  dismissOnBackdrop($("#loraDialog"));
  $("#closeLibraryPicker").addEventListener("click", () => $("#libraryPickerDialog").close());
  $("#pickerDone").addEventListener("click", () => $("#libraryPickerDialog").close());
  $("#pickerManage").addEventListener("click", () => { $("#libraryPickerDialog").close(); navigate("styles"); });
  // Clicking away ends the browse the way it ends the image viewer, but the picker is full of
  // things worth clicking: a tile, the kind filters, the search, and the footer buttons all keep
  // it open, while the blurred area around the box and the blank space the fixed two-row window
  // leaves under a short list close it. A backdrop click arrives with the dialog itself as the
  // target, so the dialog is not exempted the way a plain outside-the-box rule would exempt it.
  $("#libraryPickerDialog").addEventListener("click", (event) => {
    if (event.target.closest("button, input, label, select, textarea, .library-tile")) return;
    event.currentTarget.close();
  });
  $("#pickerSearch").addEventListener("input", (event) => { state.pickerQuery = event.target.value; renderLibraryPicker(); });
  $$("#pickerFilter button").forEach((button) => button.addEventListener("click", () => {
    state.pickerKind = button.dataset.kind;
    syncKindButtons($("#pickerFilter"), state.pickerKind);
    renderLibraryPicker();
  }));
  // A resized window re-wraps the picker's tiles at a new width, which moves the row height its
  // two-row window was measured from. Only a width change refits: this observer also fires when
  // fitPickerGrid sets the height, and the grid's scrollbar-gutter keeps the scrollbar itself
  // from changing the width back.
  let pickerGridWidth = 0;
  new ResizeObserver((entries) => {
    const width = entries[entries.length - 1].contentRect.width;
    if (width === pickerGridWidth) return;
    pickerGridWidth = width;
    if (width > 0) fitPickerGrid($("#pickerGrid"));
  }).observe($("#pickerGrid"));
  $("#librarySearch").addEventListener("input", (event) => { state.libraryQuery = event.target.value; renderLibraryPage(); });
  $$("#libraryFilter button").forEach((button) => button.addEventListener("click", () => {
    state.libraryKind = button.dataset.kind;
    syncKindButtons($("#libraryFilter"), state.libraryKind);
    renderLibraryPage();
  }));
  $("#coverTarget").addEventListener("change", (event) => setLibraryCover(event.target, state.currentImage));
  $("#detailCoverTarget").addEventListener("change", (event) => setLibraryCover(event.target, state.detailImage));
  $("#closeCoverPicker").addEventListener("click", () => resolveCoverPicker(null));
  $("#coverPickerDialog").addEventListener("close", () => {
    // Escape and the backdrop both land here rather than in resolveCoverPicker, so the promise is
    // settled from one place and an editor is never left awaiting a frame that will not arrive.
    const resolve = state.coverPickerResolve;
    state.coverPickerResolve = null;
    if (resolve) resolve(null);
  });
  $("#coverPickerBoard").addEventListener("change", (event) => {
    state.coverBoardId = event.target.value;
    renderCoverPickerGrid();
  });
  $("#styleReferencePick").addEventListener("click", async () => {
    const image = await openCoverPicker("STYLE COVER");
    if (!image) return;
    state.styleReferenceId = image.id;
    renderCoverPreview($("#styleReference"), image.image_url);
  });
  $("#styleReferenceClear").addEventListener("click", () => {
    state.styleReferenceId = null;
    renderCoverPreview($("#styleReference"), null);
  });
  $("#closeLoraDialog").addEventListener("click", () => $("#loraDialog").close());
  $("#loraCoverPick").addEventListener("click", async () => {
    const image = await openCoverPicker("LORA COVER");
    if (!image) return;
    state.loraCoverId = image.id;
    renderCoverPreview($("#loraCover"), image.image_url);
  });
  $("#loraCoverClear").addEventListener("click", () => {
    state.loraCoverId = null;
    renderCoverPreview($("#loraCover"), null);
  });
  $("#loraForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const lora = state.editingLora;
    if (!lora) return;
    try {
      await api(`/api/loras/${encodeURIComponent(lora.name)}`, {
        method: "POST",
        body: JSON.stringify({
          display_name: $("#loraDisplayName").value,
          notes: $("#loraNotes").value,
          trigger: $("#loraTrigger").value,
          summary: $("#loraSummary").value,
          prompting_notes: $("#loraPromptingNotes").value,
          default_strength: Number($("#loraDefaultStrength").value),
          cover_image_id: state.loraCoverId,
        }),
      });
      $("#loraDialog").close();
      await loadLoras();
      // The picker is often what is underneath, and it holds a snapshot of the entries it drew.
      if ($("#libraryPickerDialog").open) renderLibraryPicker();
    } catch (error) {
      $("#loraError").textContent = error.message;
    }
  });
  $("#detailSaveStyle").addEventListener("click", () => {
    $("#galleryDetailDialog").close();
    openStyleDialog(null, state.detailImage);
  });
  $("#styleForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = {
      name: $("#styleName").value,
      description: $("#styleDescription").value,
      style_text: $("#styleText").value,
      kind: $("#styleKind").value,
      reference_image_id: state.styleReferenceId || null,
    };
    const editing = state.editingStyle;
    try {
      await api(editing ? `/api/styles/${encodeURIComponent(editing.id)}` : "/api/styles", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      $("#styleDialog").close();
      await loadStyles();
      if ($("#libraryPickerDialog").open) renderLibraryPicker();
    } catch (error) {
      $("#styleError").textContent = error.message;
    }
  });
  $("#styleKind").addEventListener("change", paintStyleGuidance);
  $("#refreshLoras").addEventListener("click", loadLoras);
}
