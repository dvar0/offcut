import { populateConnectionSelectors } from "./connections.js";
import { loadBoardImages, renderBoardThumbnails } from "./images.js";
import { paintCoverProgress, renderActiveLibrary } from "./library.js";
import { clearPromptDiff, forgetPromptTurn } from "./prompt-diff.js";
import { $, $$, api, frameSeed, frameUsesGuidance, showFormError, state } from "./shared.js";
import {
  currentPage,
  flushBoardDraft,
  navigate,
  refreshBoards,
  scheduleBoardDraft,
  setActiveBoard,
} from "./workspace.js";

export function setRuntime(text, mode = "") {
  $("#runtimeText").textContent = text;
  $("#runtimePill").className = `runtime-pill ${mode}`;
}

export function updateRuntime() {
  if (!state.runtimeReady) return setRuntime("RUNTIME UNAVAILABLE", "error");
  if (state.generating) return setRuntime("GENERATING", "busy");
  if (state.runtimeInitializing) return setRuntime("STARTING RUNTIME", "busy");
  setRuntime(state.activeEngine ? "RTX 4090 READY" : "ENGINE COLD", state.activeEngine ? "" : "cold");
}

export function watchRuntime() {
  window.clearTimeout(state.runtimeTimer);
  if (!state.runtimeInitializing) return;
  state.runtimeTimer = window.setTimeout(async () => {
    try {
      const status = await api("/api/status");
      state.runtimeInitializing = status.runtime_initializing;
      state.activeEngine = status.active_engine;
      updateRuntime();
    } catch (_) { /* The next page load will retry if the service restarted. */ }
    watchRuntime();
  }, 1000);
}

function selectedPreset() {
  return $("input[name=preset]:checked").value;
}

function updateRoute() {
  const preset = selectedPreset();
  // The field is disabled rather than cleared so switching back to raw restores what was typed,
  // but the draft and the generation payload both drop it while a distilled route is selected.
  $("#negativePrompt").disabled = preset !== "raw-int8";
  $("#negativePrompt").placeholder = preset === "raw-int8" ? "What to steer away from" : "Raw route only";
  // A distilled route samples at cfg 1.0 by design; any other guidance ruins the frame rather
  // than steering it. Disabled the same way as the negative prompt, and dropped from the payload
  // below, so switching back to raw restores whatever was typed.
  $("#guidance").disabled = preset !== "raw-int8";
  $("#guidance").placeholder = preset === "raw-int8" ? "AUTO" : "FIXED AT 0";
  $("#generateButton").disabled = state.generating || !routeReady();
  updateRuntime();
}

export function routeReady() {
  return state.runtimeReady && state.presetReady[selectedPreset()] !== false;
}

// Shape and size are the two things worth thinking about separately, so the ratio row fixes the
// shape and the megapixel row fixes the area. Neither replaces the width and height fields: those
// stay authoritative, and both rows simply light up when they describe what is in them.
// Wide enough that every shipped ratio preset reads as the 1 MP frame it is — 1024x1024 is 4.9%
// over, 1344x768 is 3.2% over — and tight enough that a hand-typed size well between two targets
// lights neither. The readout underneath always states the real figure either way.
const MEGAPIXEL_MATCH = 0.06;

const NAMED_RATIOS = [[1, 1], [5, 4], [4, 3], [3, 2], [16, 10], [16, 9], [2, 1]];

function snapDimension(value) {
  return Math.max(256, Math.min(2048, Math.round(value / 16) * 16));
}

function currentRatio() {
  // A lit ratio button is the ratio the user asked for; the snapped fields are only an
  // approximation of it, and reading back from them would drift the shape a little on every
  // megapixel change.
  const active = $("#ratioGrid button.active");
  if (active) return Number(active.dataset.width) / Number(active.dataset.height);
  const ratio = Number($("#width").value) / Number($("#height").value);
  return Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
}

function setMegapixels(megapixels) {
  const ratio = currentRatio();
  const pixels = megapixels * 1_000_000;
  let width = Math.sqrt(pixels * ratio);
  let height = pixels / width;
  // A shape extreme enough to push a side past the 256..2048 bounds gives up megapixels rather
  // than the shape, which is the half the user picked deliberately.
  const shrink = Math.min(1, 2048 / Math.max(width, height));
  const grow = Math.max(1, 256 / Math.min(width, height));
  const scale = shrink < 1 ? shrink : grow;
  $("#width").value = snapDimension(width * scale);
  $("#height").value = snapDimension(height * scale);
  if (state.aspectLocked) state.aspectRatio = Number($("#width").value) / Number($("#height").value) || 1;
  syncFrameControls();
  scheduleBoardDraft();
}

function setResolution(button) {
  const megapixels = activeMegapixels() ?? currentMegapixels();
  $("#width").value = button.dataset.width;
  $("#height").value = button.dataset.height;
  state.aspectRatio = Number(button.dataset.width) / Number(button.dataset.height);
  $$("#ratioGrid button").forEach((item) => item.classList.toggle("active", item === button));
  // The preset dimensions are all about a megapixel; re-solving for whatever size was already
  // chosen means picking a shape does not silently resize the frame as well.
  if (megapixels) setMegapixels(megapixels);
  else syncFrameControls();
  scheduleBoardDraft();
}

function currentMegapixels() {
  const pixels = Number($("#width").value) * Number($("#height").value);
  return Number.isFinite(pixels) && pixels > 0 ? pixels / 1_000_000 : 0;
}

function activeMegapixels() {
  const actual = currentMegapixels();
  if (!actual) return null;
  const target = $$("#megapixelGrid button")
    .map((button) => Number(button.dataset.megapixels))
    .find((value) => Math.abs(actual - value) / value <= MEGAPIXEL_MATCH);
  return target ?? null;
}

function ratioLabel() {
  const width = Number($("#width").value);
  const height = Number($("#height").value);
  if (!(width > 0 && height > 0)) return "";
  const portrait = height > width;
  const value = portrait ? height / width : width / height;
  let best = null;
  for (const [wide, tall] of NAMED_RATIOS) {
    const distance = Math.abs(value - wide / tall);
    if (!best || distance < best.distance) best = { wide, tall, distance };
  }
  // 4% is wide enough to name 1216x832 as 3:2 and still far short of the gap between any two
  // entries in the list, the closest of which are 16:10 and 16:9 at 11% apart.
  if (best.distance / value > 0.04) return portrait ? `1:${value.toFixed(2)}` : `${value.toFixed(2)}:1`;
  return portrait ? `${best.tall}:${best.wide}` : `${best.wide}:${best.tall}`;
}

function syncFrameControls() {
  const ratio = Number($("#width").value) / Number($("#height").value);
  $$("#ratioGrid button").forEach((button) => button.classList.toggle(
    "active",
    Math.abs(Number(button.dataset.width) / Number(button.dataset.height) - ratio) < 0.015,
  ));
  const megapixels = activeMegapixels();
  $$("#megapixelGrid button").forEach((button) => button.classList.toggle(
    "active",
    Number(button.dataset.megapixels) === megapixels,
  ));
  const actual = currentMegapixels();
  const label = ratioLabel();
  $("#frameReadout").textContent = actual ? `${actual.toFixed(2)} MP${label ? ` · ${label}` : ""}` : "";
}

function toggleAspectLock() {
  state.aspectLocked = !state.aspectLocked;
  if (state.aspectLocked) state.aspectRatio = Number($("#width").value) / Number($("#height").value) || 1;
  $("#ratioLock").classList.toggle("active", state.aspectLocked);
  $("#ratioLock").ariaPressed = String(state.aspectLocked);
  $("#ratioLock").ariaLabel = state.aspectLocked ? "Unlock aspect ratio" : "Lock aspect ratio";
  $("#ratioLock").title = $("#ratioLock").ariaLabel;
}

function updateLockedDimension(source) {
  const value = Number(source.value);
  if (state.aspectLocked && state.aspectRatio && Number.isFinite(value) && value > 0) {
    const target = source.id === "width" ? $("#height") : $("#width");
    target.value = snapDimension(source.id === "width" ? value / state.aspectRatio : value * state.aspectRatio);
  }
  syncFrameControls();
}

export function applySettings(settings) {
  // Diff hunks are offsets into one exact prompt string. Any other prompt landing in the box —
  // a board switch, a reused image, an undo — makes them point at the wrong words. A second
  // prompt edit inside one turn lands here too, but it clears only the painted hunks: the turn
  // baseline is separate state, so the caller rebuilds the diff from the turn's start.
  if (state.latestPromptChange && settings.prompt !== state.latestPromptChange.after) clearPromptDiff();
  // Settings arriving from elsewhere mean the autosave baseline no longer describes what this
  // board holds on the server. Forgetting it makes the next edit write unconditionally, which is
  // the safe direction: a redundant write costs a revision, a skipped one loses the edit.
  state.draftSaved = null;
  if (typeof settings.prompt === "string") $("#prompt").value = settings.prompt;
  const preset = settings.preset || "turbo-int8";
  const radio = $(`input[name=preset][value="${CSS.escape(preset)}"]`);
  if (radio) radio.checked = true;
  if (settings.width) $("#width").value = settings.width;
  if (settings.height) $("#height").value = settings.height;
  state.aspectRatio = Number($("#width").value) / Number($("#height").value) || 1;
  syncFrameControls();
  $("#steps").value = settings.steps ?? "";
  $("#guidance").value = settings.guidance ?? "";
  $("#seed").value = settings.seed ?? "";
  $("#negativePrompt").value = settings.negative_prompt || "";
  const enhancerConnection = state.connections.find((connection) => connection.id === settings.connection_id);
  const enhancerValid = Boolean(
    settings.enhance
    && enhancerConnection
    && enhancerConnection.models.includes(settings.enhancer_model)
  );
  $("#enhance").checked = enhancerValid;
  $("#enhancerControls").hidden = !$("#enhance").checked;
  state.selectedLoras = (settings.loras || []).map((lora) => ({
    name: lora.name || String(lora.path || "").split("/").pop().replace(/\.safetensors$/, ""),
    strength: Number(lora.strength ?? 1),
  })).filter((lora) => state.loras.some((available) => available.name === lora.name));
  state.selectedStyles = (settings.styles || []).filter((id) => typeof id === "string");
  renderActiveLibrary();
  populateConnectionSelectors(enhancerValid ? settings.connection_id : null, enhancerValid ? settings.enhancer_model : null);
  updateRoute();
  syncBatchHint();
  $("#characterCount").textContent = `${$("#prompt").value.length} / 20K`;
}

// Two different intentions wear the same word. "Reuse settings" is "make another one of these",
// which wants the whole recipe including the seed; "reuse prompt" is "these words, my controls",
// which is the more common of the two and used to mean copying the text out by hand.
export async function reuseImage(image = state.currentImage) {
  if (!image) return;
  if (image.board_id !== state.currentBoardId) await setActiveBoard(image.board_id, false);
  applySettings({
    prompt: image.raw_prompt,
    preset: image.preset,
    width: image.width,
    height: image.height,
    steps: image.steps,
    // Carrying a stored 0.0 onto a distilled route would be refused by the server, and it was
    // never a chosen value there in the first place.
    guidance: frameUsesGuidance(image) ? image.guidance : null,
    negative_prompt: image.negative_prompt,
    loras: image.loras,
    styles: image.metadata?.styles,
    enhance: image.enhance,
    connection_id: image.connection_id,
    enhancer_model: image.enhancer_model,
  });
  $("#seed").value = frameSeed(image);
  syncBatchHint();
  scheduleBoardDraft();
  navigate("create", image.board_id);
  $("#prompt").focus();
}

export async function reusePrompt(image = state.currentImage) {
  if (!image) return;
  const prompt = $("#prompt");
  prompt.value = image.raw_prompt || "";
  // applySettings owns this clean-up for the whole-recipe path; the prompt-only path has to do
  // it too, because a diff painted against the previous text points at words that are gone.
  if (state.latestPromptChange || state.promptTurnBefore !== null) forgetPromptTurn();
  $("#characterCount").textContent = `${prompt.value.length} / 20K`;
  scheduleBoardDraft();
  navigate("create", state.currentBoardId);
  prompt.focus();
}

export function generationPayload() {
  const seed = $("#seed").value.trim();
  const steps = $("#steps").value.trim();
  const guidance = $("#guidance").value.trim();
  return {
    board_id: state.currentBoardId,
    prompt: $("#prompt").value,
    preset: selectedPreset(),
    width: Number($("#width").value),
    height: Number($("#height").value),
    // Sent as text for the same reason it comes back as text: Number() would round anything above
    // 2^53 and silently sample a different frame than the pinned seed asked for.
    seed: seed || null,
    steps: steps ? Number(steps) : null,
    guidance: guidance && selectedPreset() === "raw-int8" ? Number(guidance) : null,
    negative_prompt: selectedPreset() === "raw-int8" ? $("#negativePrompt").value : "",
    loras: state.selectedLoras,
    styles: state.selectedStyles,
    enhance: $("#enhance").checked,
    connection_id: $("#enhance").checked ? $("#enhancerConnection").value : null,
    enhancer_model: $("#enhance").checked ? $("#enhancerModel").value : "",
  };
}

const BATCH_LIMIT = 8;

// Seeing what a prompt does across seeds means pressing GENERATE and waiting, over and over. The
// queue is the browser's rather than the server's: the engine still takes one frame at a time, so
// a list of runs held here needs no scheduler, survives nothing it should survive, and stops the
// instant the tab says stop. Each frame lands in the rail as it finishes.
function setBatchCount(count) {
  state.batchCount = Math.max(1, Math.min(Number(count) || 1, BATCH_LIMIT));
  for (const button of $$("#batchOptions button")) {
    button.classList.toggle("active", Number(button.dataset.count) === state.batchCount);
  }
  syncBatchHint();
  paintGenerateLabel();
}

function syncBatchHint() {
  const hint = $("#batchHint");
  if (state.batchCount === 1) hint.textContent = "ONE FRAME";
  else hint.textContent = $("#seed").value.trim()
    ? `${state.batchCount} FRAMES · SEED +1 EACH`
    : `${state.batchCount} FRAMES · NEW SEED EACH`;
}

// A pinned seed run four times makes the same frame four times, which is never what a batch was
// for. The run the press asked for keeps the exact seed and the extras walk forward from it, so
// the set stays reproducible. An empty field already means the server rolls a fresh seed per run.
const SEED_LIMIT = 2n ** 63n; // mirrors offcut_cli.validate_generation_values

function batchSeed(seed, index) {
  if (index === 0 || seed === null) return seed;
  try {
    const stepped = BigInt(seed) + BigInt(index);
    // Walking off the end of the range would be refused by the server mid-batch, so the run that
    // would overflow rolls a fresh seed instead of failing the rest of the queue.
    return stepped >= 0n && stepped < SEED_LIMIT ? String(stepped) : null;
  } catch (_) {
    return null;
  }
}

// Marking is per run, not per frame: a run takes the marks off the frames before it and puts
// them on its own, so the dots always answer "which of these just arrived" and never accumulate.
function beginFreshRun() {
  if (!state.freshImageIds.size) return;
  state.freshImageIds.clear();
  renderBoardThumbnails();
}

export function markFresh(imageId) {
  if (imageId) state.freshImageIds.add(imageId);
}

function paintGenerateLabel() {
  const running = state.generating || state.agentGenerating;
  if (running && state.batchTotal > 1) {
    $("#generateLabel").textContent = `GENERATING ${state.batchIndex + 1}/${state.batchTotal}`;
  } else if (running) {
    $("#generateLabel").textContent = "GENERATING";
  } else {
    $("#generateLabel").textContent = state.batchCount > 1 ? `GENERATE ×${state.batchCount}` : "GENERATE";
  }
}

async function generate(event) {
  event?.preventDefault();
  if (state.generating) return;
  showFormError("");
  state.stopping = false;
  state.batchIndex = 0;
  state.batchTotal = state.batchCount;
  beginFreshRun();
  try {
    await flushBoardDraft();
    const targetBoardId = state.currentBoardId;
    state.generatingBoardId = targetBoardId;
    setGenerating(true);
    // Snapshotted once: a batch runs with the settings that were on screen when it was started,
    // not with whatever the panel drifts to while it works.
    const basePayload = generationPayload();
    for (let index = 0; index < state.batchTotal; index += 1) {
      if (state.stopping) break;
      state.batchIndex = index;
      state.progressStarted = performance.now();
      paintGenerateLabel();
      updateProgress({
        stage: "queued",
        fraction: 0,
        detail: state.batchTotal > 1 ? `Queued frame ${index + 1} of ${state.batchTotal}` : "Submitting generation",
      });
      const payload = { ...basePayload, seed: batchSeed(basePayload.seed, index) };
      const result = await api("/api/generate", { method: "POST", body: JSON.stringify(payload) });
      state.activeEngine = result.image.preset;
      await refreshBoards();
      // Follow the frame only for someone who never left: the board a run lands on is normally
      // the one it was started from, and this catches the case where the server filed it
      // elsewhere. Someone who has since moved on gets left where they are — dragging them back
      // once per finished frame is what made staying put the only usable option.
      if (state.currentBoardId === targetBoardId && result.image.board_id !== targetBoardId) {
        await setActiveBoard(result.image.board_id, false);
      }
      // No-ops when the rail is showing another board; the frames are there on the way back.
      markFresh(result.image_id);
      await loadBoardImages(result.image_id, targetBoardId);
    }
  } catch (error) {
    // A stop the user asked for is not a failure to report as one. Anything else is.
    if (!error.data?.cancelled) showFormError(error.message);
  } finally {
    state.stopping = false;
    state.batchTotal = 0;
    setGenerating(false);
  }
}

// The queue lives in this tab and the run lives in the server, so a press has to end both: the
// remaining frames are dropped here and the frame in flight is interrupted there.
async function stopGeneration() {
  if (!state.generating && !state.agentGenerating) return;
  state.stopping = true;
  $("#stopButton").disabled = true;
  $("#progressStage").textContent = "STOPPING";
  try {
    await api("/api/generate/cancel", { method: "POST", body: "{}" });
  } catch (error) {
    showFormError(error.message);
  }
}

// The engine can be driven from the form, from another tab, or by the chat agent's
// generate_image tool. Whoever started it, the left column has to look the same, so the
// panel state lives here and each path only supplies its own source of progress.
export function paintGenerationState(active) {
  state.generating = active;
  $("#generateButton").disabled = active || !routeReady();
  paintGenerateLabel();
  $("#generationProgress").hidden = !active;
  $("#stopButton").hidden = !active;
  $("#stopButton").disabled = !active || state.stopping;
  paintGeneratingBoard();
  if (active) {
    state.progressStarted = performance.now();
    setRuntime("GENERATING", "busy");
  } else {
    state.generatingBoardId = null;
    updateRuntime();
  }
}

// Leaving a running board is allowed, so the panel has to say which board is still being written
// to — otherwise a bar filling next to someone else's frames reads as this board's. It only
// appears once the two differ, and it goes back, because that is the only thing anyone wants
// from it. A cover run belongs to the library rather than to a board and shows nothing.
export function paintGeneratingBoard() {
  const line = $("#progressBoard");
  if (!line) return;
  const busy = state.generating || state.agentGenerating;
  const boardId = state.generatingBoardId;
  const elsewhere = busy && boardId && boardId !== state.currentBoardId && !state.coverRunning;
  line.hidden = !elsewhere;
  if (!elsewhere) return;
  const board = state.boards.find((item) => item.id === boardId);
  line.textContent = `↩ GENERATING IN ${(board?.name || boardId).toUpperCase()}`;
}

function setGenerating(active) {
  paintGenerationState(active);
  if (active) {
    updateProgress({ stage: "queued", fraction: 0, detail: "Submitting generation" });
    state.progressTimer = window.setInterval(pollProgress, 350);
  } else {
    window.clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
}

// The chat stream already carries the engine's progress, so the agent's run drives the
// same panel straight from those events instead of opening a second poll against /api/progress.
export function setAgentGenerating(active) {
  if (active === state.agentGenerating) return;
  if (active && state.generating) return;
  state.agentGenerating = active;
  if (active) {
    state.generatingBoardId = state.streamBoardId || state.currentBoardId;
    beginFreshRun();
  }
  paintGenerationState(active);
  if (active) updateProgress({ stage: "queued", fraction: 0, detail: "Agent generation queued" });
}

export function resumeProgress() {
  paintGenerationState(true);
  state.progressTimer = window.setInterval(async () => {
    try {
      const progress = await api("/api/progress");
      updateProgress(progress);
      if (!progress.active) {
        const status = await api("/api/status");
        state.activeEngine = status.active_engine;
        // Read before setGenerating clears it: the frame belongs to the run's board, which is
        // not the current one for anyone who moved on while the adopted run finished.
        const landed = state.generatingBoardId || progress.board_id || state.currentBoardId;
        setGenerating(false);
        await refreshBoards();
        markFresh(progress.image_id);
        await loadBoardImages(progress.image_id || null, landed);
      }
    } catch (_) { /* Retry until the local server responds. */ }
  }, 350);
}

export async function pollProgress() {
  try { updateProgress(await api("/api/progress")); } catch (_) { /* POST remains authoritative. */ }
}

export function updateProgress(progress) {
  // A run started in this tab already recorded its own board and that snapshot stays
  // authoritative. This fills the one case with no snapshot to read: a reload landing on a run
  // the previous page started, where the server's board is the only record of it left.
  if (!state.generatingBoardId && progress.board_id) {
    state.generatingBoardId = progress.board_id;
    paintGeneratingBoard();
  }
  const fraction = Math.max(0, Math.min(Number(progress.fraction || 0), 1));
  $("#progressStage").textContent = String(progress.stage || "working").replaceAll("_", " ").toUpperCase();
  $("#progressPercent").textContent = `${Math.round(fraction * 100)}%`;
  $("#progressBar").style.width = `${fraction * 100}%`;
  // Say what the engine is busy with, not just that it is: a user who wanders onto Create mid-run
  // otherwise sees a dead generate button with no reason attached to it.
  $("#progressDetail").textContent = state.coverRun
    ? `Library cover ${state.coverRun.index + 1} of ${state.coverRun.total} — ${state.coverRun.name}`
    : (progress.detail || "Working");
  const elapsed = Math.max(0, performance.now() - state.progressStarted) / 1000;
  $("#progressElapsed").textContent = `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${(elapsed % 60).toFixed(1).padStart(4, "0")}`;
  if (state.coverRunning) paintCoverProgress(progress);
}

export function initGeneration() {
  $("#enhance").addEventListener("change", (event) => { $("#enhancerControls").hidden = !event.target.checked; });
  $$("input[name=preset]").forEach((input) => input.addEventListener("change", updateRoute));
  $$("#ratioGrid button").forEach((button) => button.addEventListener("click", () => setResolution(button)));
  $$("#megapixelGrid button").forEach((button) => button.addEventListener("click", () => setMegapixels(Number(button.dataset.megapixels))));
  $("#ratioLock").addEventListener("click", toggleAspectLock);
  for (const input of [$("#width"), $("#height")]) input.addEventListener("input", () => updateLockedDimension(input));
  syncFrameControls();
  $("#randomSeed").addEventListener("click", () => { $("#seed").value = String(Math.floor(Math.random() * Number.MAX_SAFE_INTEGER)); syncBatchHint(); });
  $("#seed").addEventListener("input", syncBatchHint);
  $$("#batchOptions button").forEach((button) => button.addEventListener("click", () => setBatchCount(button.dataset.count)));
  setBatchCount(state.batchCount);
  $("#generateForm").addEventListener("submit", generate);
  $("#stopButton").addEventListener("click", stopGeneration);
  $("#progressBoard").addEventListener("click", () => {
    if (state.generatingBoardId) navigate("create", state.generatingBoardId);
  });
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && event.key === "Enter" && currentPage() === "create" && !event.target.closest?.("#chatPanel")) generate(event);
  });
}
