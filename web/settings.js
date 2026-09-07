import { renderConnections } from "./connections.js";
import { $, $$, api, escapeHtml, presetLabel, routeUsesGuidance, state } from "./shared.js";
import { currentPage } from "./workspace.js";
import { bindHybridControls } from "./sampling.js";

let coverHybridControls;

export function renderSettings() {
  renderAppearance();
  renderConnections();
  renderCoverRecipe();
}

// The cover recipe is app state, not board state, so it is fetched once and kept: the settings
// page, the per-card buttons and the batch run all read the same copy.
export async function loadCoverRecipe() {
  try {
    const settings = await api("/api/settings");
    state.coverRecipe = settings.cover || null;
  } catch (_) {
    state.coverRecipe = null;
  }
  renderCoverRecipe();
}

function renderCoverRecipe() {
  const form = $("#coverRecipeForm");
  if (!form) return;
  const recipe = state.coverRecipe;
  const summary = $("#coverSummary");
  if (!recipe) {
    if (summary) summary.textContent = "UNAVAILABLE";
    return;
  }
  $("#coverPrompt").value = recipe.prompt;
  $("#coverSeed").value = recipe.seed;
  $("#coverPreset").value = recipe.preset;
  $("#coverWidth").value = recipe.width;
  $("#coverHeight").value = recipe.height;
  $("#coverSteps").value = recipe.steps;
  $("#coverTurboSteps").value = recipe.preset === "raw-int8-to-turbo" ? recipe.steps : 12;
  $("#coverGuidance").value = recipe.guidance;
  $("#coverRawPortion").value = recipe.raw_portion ?? 8;
  $("#coverRawSteps").value = recipe.raw_steps ?? 52;
  $("#coverLoraStrength").value = recipe.lora_strength;
  syncCoverGuidance();
  if (summary) {
    summary.innerHTML = `${escapeHtml(presetLabel(recipe.preset))} · ${recipe.width}×${recipe.height} · SEED <b>${escapeHtml(recipe.seed)}</b> · LORA ${Number(recipe.lora_strength).toFixed(2)}`;
  }
}

// The same rule the create panel follows: a distilled route has no usable guidance dial, and the
// server refuses a non-default value on one, so the field is disabled rather than left to fail.
function syncCoverGuidance() {
  const field = $("#coverGuidanceField");
  const input = $("#coverGuidance");
  if (!field || !input) return;
  const usesGuidance = routeUsesGuidance($("#coverPreset").value);
  input.disabled = !usesGuidance;
  field.classList.toggle("disabled", !usesGuidance);
  if (!usesGuidance) input.value = "0";
  const hybrid = $("#coverPreset").value === "raw-int8-to-turbo";
  field.querySelector("span").textContent = hybrid ? "RAW GUIDANCE" : "GUIDANCE";
  $("#coverStepsField").hidden = hybrid;
  $("#coverSteps").disabled = hybrid;
  $("#coverHybridControls").hidden = !hybrid;
  for (const id of ["coverRawCount", "coverRawSteps", "coverTurboSteps"]) $(`#${id}`).disabled = !hybrid;
  coverHybridControls?.refresh();
}

async function saveCoverRecipe(event) {
  event?.preventDefault();
  const error = $("#coverRecipeError");
  const status = $("#coverRecipeStatus");
  error.textContent = "";
  status.textContent = "";
  try {
    const settings = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify({
        cover: {
          prompt: $("#coverPrompt").value,
          seed: $("#coverSeed").value.trim(),
          preset: $("#coverPreset").value,
          width: Number($("#coverWidth").value),
          height: Number($("#coverHeight").value),
          steps: Number($($("#coverPreset").value === "raw-int8-to-turbo" ? "#coverTurboSteps" : "#coverSteps").value),
          guidance: Number($("#coverGuidance").value),
          ...($("#coverPreset").value === "raw-int8-to-turbo" ? {
            raw_portion: Number($("#coverRawPortion").value),
            raw_steps: Number($("#coverRawSteps").value),
          } : {}),
          lora_strength: Number($("#coverLoraStrength").value),
        },
      }),
    });
    state.coverRecipe = settings.cover;
    renderCoverRecipe();
    status.textContent = "SAVED. EXISTING COVERS ARE UNCHANGED UNTIL YOU REGENERATE THEM.";
  } catch (failure) {
    error.textContent = failure.message;
  }
}

function appearanceSummaryText(appearance, theme) {
  const label = (theme?.label || "theme").toUpperCase();
  if (appearance.mode === "manual") return `LOCKED · <b>${escapeHtml(label)}</b>`;
  const savedDark = KreaThemes.THEMES[appearance.dark];
  const savedLight = KreaThemes.THEMES[appearance.light];
  return `SYSTEM · <b>${escapeHtml(label)}</b> · ${escapeHtml((savedDark?.label || appearance.dark).toUpperCase())} / ${escapeHtml((savedLight?.label || appearance.light).toUpperCase())}`;
}

function appearanceStatusText(appearance, theme) {
  const label = (theme?.label || "theme").toUpperCase();
  const system = KreaThemes.systemPrefersDark() ? "DARK" : "LIGHT";
  if (appearance.mode === "manual") return `LOCKED TO <b>${escapeHtml(label)}</b>`;
  const savedDark = KreaThemes.THEMES[appearance.dark];
  const savedLight = KreaThemes.THEMES[appearance.light];
  return `SYSTEM IS ${system} · SHOWING <b>${escapeHtml(label)}</b> · DARK ${escapeHtml((savedDark?.label || appearance.dark).toUpperCase())} · LIGHT ${escapeHtml((savedLight?.label || appearance.light).toUpperCase())}`;
}

function themeSwatches(theme) {
  return `<span class="theme-swatches">${theme.swatches.map((color) => `<i style="background:${escapeHtml(color)}"></i>`).join("")}</span>`;
}

function themeCard(theme, active) {
  return `<button class="theme-card${active ? " active" : ""}" type="button" data-theme-id="${escapeHtml(theme.id)}" aria-pressed="${active ? "true" : "false"}"><span class="theme-card-top">${themeSwatches(theme)}<b>${escapeHtml(theme.label.toUpperCase())}</b><small>${escapeHtml(theme.scheme.toUpperCase())}</small></span><p>${escapeHtml(theme.description)}</p></button>`;
}

function renderAppearance() {
  const families = $("#themeFamilies");
  const status = $("#appearanceStatus");
  const summary = $("#appearanceSummary");
  if (!families || !window.KreaThemes) return;
  const appearance = KreaThemes.loadAppearance();
  const activeId = KreaThemes.resolveThemeId(appearance);
  const activeTheme = KreaThemes.THEMES[activeId];
  const systemDark = KreaThemes.systemPrefersDark();
  $$("#appearanceMode input").forEach((input) => { input.checked = input.value === appearance.mode; });
  status.innerHTML = appearanceStatusText(appearance, activeTheme);
  if (summary) summary.innerHTML = appearanceSummaryText(appearance, activeTheme);
  families.replaceChildren();
  for (const family of KreaThemes.FAMILIES) {
    const dark = KreaThemes.THEMES[family.dark];
    const light = KreaThemes.THEMES[family.light];
    const section = document.createElement("section");
    section.className = "theme-family";
    const darkActive = appearance.mode === "manual" ? activeId === dark.id : appearance.dark === dark.id;
    const lightActive = appearance.mode === "manual" ? activeId === light.id : appearance.light === light.id;
    section.innerHTML = `<div class="theme-family-head"><h3>${escapeHtml(family.label.toUpperCase())}</h3><p>${escapeHtml(family.description)}</p><button class="text-button" type="button" data-family="${escapeHtml(family.id)}">USE PAIR</button></div><div class="theme-pair">${themeCard(dark, darkActive)}${themeCard(light, lightActive)}</div>`;
    $("[data-family]", section).addEventListener("click", () => {
      KreaThemes.setAppearance({ mode: appearance.mode, dark: family.dark, light: family.light, manual: systemDark ? family.dark : family.light });
      renderAppearance();
    });
    $$("[data-theme-id]", section).forEach((button) => {
      button.addEventListener("click", () => {
        const theme = KreaThemes.THEMES[button.dataset.themeId];
        if (!theme) return;
        if (appearance.mode === "manual") KreaThemes.setAppearance({ manual: theme.id });
        else if (theme.scheme === "dark") KreaThemes.setAppearance({ dark: theme.id });
        else KreaThemes.setAppearance({ light: theme.id });
        renderAppearance();
      });
    });
    families.append(section);
  }
}

export function initSettings() {
  coverHybridControls = bindHybridControls({
    count: "coverRawCount", portion: "coverRawPortion", rawDensity: "coverRawSteps", turboDensity: "coverTurboSteps",
    width: "coverWidth", height: "coverHeight", readout: "coverHybridReadout", minus: "coverRawCountMinus", plus: "coverRawCountPlus",
  });
  $("#coverRecipeForm").addEventListener("submit", saveCoverRecipe);
  $("#coverPreset").addEventListener("change", () => {
    const preset = $("#coverPreset").value;
    $("#coverSteps").value = preset === "raw-int8" ? 52 : preset === "raw-int8-to-turbo" ? 12 : 8;
    $("#coverTurboSteps").value = 12;
    $("#coverGuidance").value = preset === "raw-int8" ? 3.5 : preset === "raw-int8-to-turbo" ? 3 : 0;
    syncCoverGuidance();
  });
  // The link navigates through the generic [data-route] handler; this only makes sure the panel it
  // is pointing at is open when the page arrives.
  $("#coverRecipeLink").addEventListener("click", () => $("#coverPanel")?.setAttribute("open", ""));
  $("#coverRestorePrompt").addEventListener("click", () => {
    if (state.coverRecipe?.default_prompt) $("#coverPrompt").value = state.coverRecipe.default_prompt;
  });
  $$("#appearanceMode input").forEach((input) => input.addEventListener("change", () => {
    if (!input.checked) return;
    KreaThemes.setAppearance({ mode: input.value });
    renderAppearance();
  }));
  window.addEventListener("offcut-appearance", () => {
    if (currentPage() === "settings") renderAppearance();
  });
}
