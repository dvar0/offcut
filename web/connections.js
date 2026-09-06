import { $, api, escapeHtml, state } from "./shared.js";

export function populateConnectionSelectors(preferredConnection = null, preferredModel = null) {
  const connectionSelect = $("#enhancerConnection");
  const selected = preferredConnection || connectionSelect.value;
  connectionSelect.replaceChildren();
  for (const connection of state.connections) {
    const option = document.createElement("option");
    option.value = connection.id;
    option.textContent = connection.name.toUpperCase();
    option.disabled = !connection.has_api_key;
    connectionSelect.append(option);
  }
  if (selected && state.connections.some((connection) => connection.id === selected)) connectionSelect.value = selected;
  populateEnhancerModels(preferredModel);
}

function populateEnhancerModels(preferred = null) {
  const connection = state.connections.find((item) => item.id === $("#enhancerConnection").value);
  const select = $("#enhancerModel");
  const previous = preferred || select.value;
  select.replaceChildren();
  for (const model of connection?.models || []) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.append(option);
  }
  if (previous && (connection?.models || []).includes(previous)) select.value = previous;
  if (!select.options.length) {
    const option = document.createElement("option");
    option.textContent = "SYNC MODELS IN CONNECTIONS";
    option.value = "";
    select.append(option);
  }
}

export function keyedConnections() {
  return state.connections.filter((connection) => connection.has_api_key && (connection.models || []).length);
}

// There is one working mode left and the model and effort now come from the backend profile, so a
// new chat has nothing left to ask about. The profile it opens on is the last one a chat used,
// which keeps a multi-profile setup reachable without a screen in front of every chat.
export function defaultChatConnection() {
  const keyed = keyedConnections();
  const remembered = localStorage.getItem("offcut.chatConnection");
  return keyed.find((connection) => connection.id === remembered) || keyed[0] || null;
}

export function capabilityKey(connectionId, model) {
  return `${connectionId || ""}\u0000${model || ""}`;
}

export async function loadModelCapabilities(connectionId, model) {
  if (!connectionId || !model) return null;
  const key = capabilityKey(connectionId, model);
  const cached = state.modelCapabilities.get(key);
  if (cached) return cached instanceof Promise ? cached : Promise.resolve(cached);
  const request = api(`/api/chat-model-capabilities?connection_id=${encodeURIComponent(connectionId)}&model=${encodeURIComponent(model)}`)
    .then((capabilities) => {
      const normalized = {
        ...capabilities,
        supportedThinkingLevels: (capabilities.supportedThinkingLevels || []).map((level) => String(level).toLowerCase()),
        recommendedThinkingLevel: capabilities.recommendedThinkingLevel ? String(capabilities.recommendedThinkingLevel).toLowerCase() : null,
      };
      state.modelCapabilities.set(key, normalized);
      return normalized;
    })
    .catch((error) => {
      state.modelCapabilities.delete(key);
      throw error;
    });
  state.modelCapabilities.set(key, request);
  return request;
}

export function knownCapabilities(connectionId, model) {
  const value = state.modelCapabilities.get(capabilityKey(connectionId, model));
  return value && !(value instanceof Promise) ? value : null;
}

export function populateReasoningControl(select, capabilities, currentValue = "", { lockWhileStreaming = true } = {}) {
  select.replaceChildren();
  const levels = capabilities?.reasoning ? capabilities.supportedThinkingLevels || [] : [];
  select.hidden = !levels.length;
  select.disabled = !levels.length || (lockWhileStreaming && state.chatStreaming);
  for (const level of levels) {
    const option = document.createElement("option");
    option.value = level;
    option.textContent = level.toUpperCase();
    select.append(option);
  }
  const preferred = levels.includes(String(currentValue || "").toLowerCase())
    ? String(currentValue).toLowerCase()
    : capabilities?.recommendedThinkingLevel;
  if (preferred && levels.includes(preferred)) select.value = preferred;
  else if (levels.length) select.value = levels[0];
  return select.value || null;
}

// What a new chat on this profile will actually open with, spelled out rather than implied: an
// unset default is not "nothing", it is the first synced model, and saying so is what stops the
// blank-looking editor from reading as a failed save.
function connectionDefaultSummary(connection) {
  const models = connection.models || [];
  if (!models.length) return "NEW CHATS: NO MODELS SYNCED YET";
  const explicit = models.includes(connection.default_model);
  const model = explicit ? connection.default_model : models[0];
  const effort = connection.default_reasoning ? ` \u00b7 ${connection.default_reasoning.toUpperCase()}` : "";
  return `NEW CHATS: ${model}${effort}${explicit ? "" : " (AUTO)"}`;
}

export function renderConnections() {
  const list = $("#connectionList");
  const summary = $("#connectionsSummary");
  list.replaceChildren();
  const count = state.connections.length;
  if (summary) {
    summary.textContent = count ? `${count} PROFILE${count === 1 ? "" : "S"}` : "NO PROFILES";
  }
  for (const connection of state.connections) {
    const card = document.createElement("article");
    card.className = "connection-card";
    // The default is hoisted to the front so it survives the eight-chip cut, and marked, because
    // the card is where you look to find out what a new chat on this profile will open with.
    const ordered = connection.models.includes(connection.default_model)
      ? [connection.default_model, ...connection.models.filter((model) => model !== connection.default_model)]
      : connection.models;
    const models = ordered.slice(0, 8)
      .map((model) => `<i class="${model === connection.default_model ? "is-default" : ""}">${escapeHtml(model)}</i>`)
      .join("");
    const keyLabel = connection.has_api_key ? `KEY / ${String(connection.key_storage || "available").toUpperCase()}` : "NO KEY";
    card.innerHTML = `<div><h2>${escapeHtml(connection.name.toUpperCase())}</h2><p>${escapeHtml(connection.base_url)}</p><small>${escapeHtml(connection.protocol.toUpperCase())} / <span class="connection-status ${connection.has_api_key ? "" : "missing"}">${escapeHtml(keyLabel)}</span></small><div class="connection-default">${escapeHtml(connectionDefaultSummary(connection))}</div><div class="model-chips">${models || "<i>NO MODELS SYNCED</i>"}</div></div><div class="connection-card-actions"><button data-action="edit">EDIT / DEFAULTS</button><button data-action="sync">SYNC MODELS</button><button data-action="test">TEST</button><button data-action="delete">DELETE</button></div>`;
    $("[data-action=edit]", card).addEventListener("click", () => editConnection(connection));
    $("[data-action=sync]", card).addEventListener("click", () => syncConnection(connection, card));
    $("[data-action=test]", card).addEventListener("click", () => testConnection(connection, card));
    $("[data-action=delete]", card).addEventListener("click", () => deleteConnection(connection));
    list.append(card);
  }
  // Leave an edit in progress alone; otherwise give the blank editor a populated model list.
  if (!$("#connectionId").value) populateConnectionDefaults(null);
}

function clearConnectionForm() {
  $("#connectionForm").reset();
  $("#connectionId").value = "";
  $("#connectionFormMode").textContent = "NEW CONNECTION";
  $("#connectionKeyStatus").textContent = "NOT STORED";
  setConnectionStatus("");
  populateConnectionDefaults(null);
}

// The default model is what a new chat on this profile opens with. A profile with nothing synced
// yet still saves; the server falls back to the first model the next sync brings in.
function populateConnectionDefaults(connection) {
  const select = $("#connectionDefaultModel");
  const models = connection?.models || [];
  select.replaceChildren();
  const fallback = document.createElement("option");
  fallback.value = "";
  fallback.textContent = models.length ? `AUTO \u2014 ${models[0]}` : "NO MODELS SYNCED YET";
  select.append(fallback);
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model;
    option.textContent = model;
    select.append(option);
  }
  select.value = models.includes(connection?.default_model) ? connection.default_model : "";
  updateConnectionDefaultReasoning(connection?.default_reasoning || "");
}

// Effort names are not shared across models -- one offers off/low/medium/high, another low/high/max
// -- so the levels come from the same capability probe the chat uses, read against whichever model
// this profile would actually start on. A default the next model does not offer is not an error:
// the server swaps it for that model's recommended level when the chat is created.
async function updateConnectionDefaultReasoning(current = $("#connectionDefaultReasoning").value) {
  const wrap = $("#connectionDefaultReasoningWrap");
  const select = $("#connectionDefaultReasoning");
  const connectionId = $("#connectionId").value;
  const connection = state.connections.find((item) => item.id === connectionId);
  const model = $("#connectionDefaultModel").value || (connection?.models || [])[0] || "";
  wrap.hidden = true;
  delete wrap.dataset.pending;
  if (!connectionId || !model) return;
  // Saving while the probe is still out would read a control nothing has filled in yet, and an
  // empty one reads as "no default". The flag makes the save omit the field instead of clearing it.
  wrap.dataset.pending = "1";
  try {
    const capabilities = await loadModelCapabilities(connectionId, model);
    if ($("#connectionId").value !== connectionId || ($("#connectionDefaultModel").value || (connection?.models || [])[0] || "") !== model) return;
    populateReasoningControl(select, capabilities, current, { lockWhileStreaming: false });
    wrap.hidden = select.hidden;
    delete wrap.dataset.pending;
  } catch (error) {
    setConnectionStatus(`Could not inspect model capabilities: ${error.message}`, true);
  }
}

function setConnectionStatus(message, isError = false) {
  const field = $("#connectionError");
  field.textContent = message;
  field.classList.toggle("ok", Boolean(message) && !isError);
}

function editConnection(connection, { scroll = true } = {}) {
  $("#connectionsPanel")?.setAttribute("open", "");
  $("#connectionId").value = connection.id;
  $("#connectionName").value = connection.name;
  $("#connectionUrl").value = connection.base_url;
  $("#connectionProtocol").value = connection.protocol;
  $("#connectionKey").value = "";
  $("#connectionFormMode").textContent = "EDIT CONNECTION";
  $("#connectionKeyStatus").textContent = connection.has_api_key ? `KEY / ${String(connection.key_storage || "available").toUpperCase()}` : "NOT STORED";
  populateConnectionDefaults(connection);
  if (scroll) $("#connectionForm").scrollIntoView({ behavior: "smooth", block: "start" });
}

// undefined leaves the stored default untouched; "" is a deliberate clear. The difference matters
// while the capability probe is unresolved, where the visible control means nothing yet.
function reasoningDefaultForSave() {
  const wrap = $("#connectionDefaultReasoningWrap");
  if (wrap.dataset.pending) return undefined;
  return wrap.hidden ? "" : $("#connectionDefaultReasoning").value;
}

async function saveConnection(event) {
  event.preventDefault();
  setConnectionStatus("");
  try {
    const payload = {
      id: $("#connectionId").value || undefined,
      name: $("#connectionName").value,
      base_url: $("#connectionUrl").value,
      protocol: $("#connectionProtocol").value,
      api_key: $("#connectionKey").value || undefined,
      default_model: $("#connectionDefaultModel").value,
      default_reasoning: reasoningDefaultForSave(),
    };
    const saved = await api("/api/connections", { method: "POST", body: JSON.stringify(payload) });
    state.connections = (await api("/api/connections")).connections;
    renderConnections();
    populateConnectionSelectors(saved.id);
    // Resetting to a blank NEW CONNECTION here read as a failed save: the defaults you just chose
    // vanished and the model list went back to "no models synced". Stay on the saved profile and
    // repaint it from the server's copy, so what is on screen is what was stored. CLEAR is how you
    // get to a blank form.
    editConnection(state.connections.find((item) => item.id === saved.id) || saved, { scroll: false });
    setConnectionStatus(`SAVED · ${connectionDefaultSummary(saved)}`);
  } catch (error) { setConnectionStatus(error.message, true); }
}

async function syncConnection(connection, card) {
  const button = $("[data-action=sync]", card);
  button.disabled = true; button.textContent = "SYNCING...";
  try {
    await api(`/api/connections/${encodeURIComponent(connection.id)}/models`, { method: "POST", body: "{}" });
    state.connections = (await api("/api/connections")).connections;
    renderConnections(); populateConnectionSelectors(connection.id);
    if ($("#connectionId").value === connection.id) {
      populateConnectionDefaults(state.connections.find((item) => item.id === connection.id));
    }
  } catch (error) { alert(error.message); }
  finally { button.disabled = false; }
}

async function testConnection(connection, card) {
  const button = $("[data-action=test]", card);
  button.disabled = true; button.textContent = "TESTING...";
  try {
    const result = await api(`/api/connections/${encodeURIComponent(connection.id)}/test`, { method: "POST", body: "{}" });
    alert(`Connection succeeded. ${result.model_count} models found.`);
  } catch (error) { alert(error.message); }
  finally { button.disabled = false; button.textContent = "TEST"; }
}

async function deleteConnection(connection) {
  if (!confirm(`Delete connection “${connection.name}”?`)) return;
  await api(`/api/connections/${encodeURIComponent(connection.id)}/delete`, { method: "POST", body: "{}" });
  state.connections = (await api("/api/connections")).connections;
  renderConnections(); populateConnectionSelectors();
}

export function initConnections() {
  $("#enhancerConnection").addEventListener("change", () => populateEnhancerModels());
  $("#clearConnectionForm").addEventListener("click", clearConnectionForm);
  $("#connectionForm").addEventListener("submit", saveConnection);
  $("#connectionDefaultModel").addEventListener("change", () => updateConnectionDefaultReasoning(""));
}
