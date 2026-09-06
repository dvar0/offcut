import { initChat } from "./chat.js";
import { initChatComposer, loadChatSkills } from "./chat-composer.js";
import { initChatRendering } from "./chat-render.js";
import { initConnections, populateConnectionSelectors } from "./connections.js";
import {
  initGeneration,
  resumeProgress,
  routeReady,
  setRuntime,
  updateRuntime,
  watchRuntime,
} from "./generation.js";
import { initImages } from "./images.js";
import { initLibrary, loadStyles, populateCoverTargets } from "./library.js";
import { initPromptDiff } from "./prompt-diff.js";
import { initSettings, loadCoverRecipe, renderSettings } from "./settings.js";
import { $, api, showFormError, state } from "./shared.js";
import {
  initWorkspace,
  populateBoardSelectors,
  renderBoards,
  setActiveBoard,
  showPage,
} from "./workspace.js";

async function bootstrap() {
  try {
    const [status, boardData, connectionData, loraData] = await Promise.all([
      api("/api/status"), api("/api/boards"), api("/api/connections"), api("/api/loras"),
    ]);
    // The library renders from both lists at once, so the LoRAs are in state before loadStyles
    // paints anything with them.
    state.loras = loraData.loras;
    await Promise.all([loadChatSkills(), loadStyles()]);
    state.presets = status.presets;
    state.runtimeReady = status.ready;
    state.activeEngine = status.active_engine;
    state.runtimeInitializing = status.runtime_initializing;
    state.presetReady = status.preset_ready || {};
    state.boards = boardData.boards;
    state.connections = connectionData.connections;
    populateBoardSelectors();
    populateCoverTargets();
    populateConnectionSelectors();
    renderBoards();
    renderSettings();
    loadCoverRecipe();
    const routeBoard = decodeURIComponent(location.pathname.split("/")[2] || "");
    const storedBoard = localStorage.getItem("offcut.activeBoard");
    const initialBoard = [routeBoard, storedBoard, "inbox"].find((id) => state.boards.some((board) => board.id === id)) || state.boards[0]?.id;
    if (initialBoard) await setActiveBoard(initialBoard, false);
    if (status.ready) updateRuntime();
    else setRuntime(`MISSING ${status.missing.join(", ")}`, "error");
    $("#generateButton").disabled = !routeReady() || status.busy;
    if (status.busy) resumeProgress();
    watchRuntime();
    await showPage();
  } catch (error) {
    setRuntime("RUNTIME UNAVAILABLE", "error");
    showFormError(error.message);
  }
}

initWorkspace();
initImages();
initLibrary();
initGeneration();
initConnections();
initSettings();
initChat();
initChatRendering();
initChatComposer();
initPromptDiff();

bootstrap();
