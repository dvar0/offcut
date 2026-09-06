import { getBuiltinModels, getBuiltinProviders } from "@earendil-works/pi-ai/providers/all";
import { getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { API_BY_PROTOCOL, normalizeBaseUrl, selectConnectionProtocol } from "./protocol.js";

const EMPTY_COST = Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });

function catalogMatch(modelId, api, preferredProvider) {
  const providers = [preferredProvider, ...getBuiltinProviders()].filter(
    (provider, index, all) => provider && all.indexOf(provider) === index,
  );
  let crossApiMatch;
  for (const provider of providers) {
    for (const candidate of getBuiltinModels(provider)) {
      if (candidate.id !== modelId) continue;
      if (candidate.api === api) return candidate;
      crossApiMatch ??= candidate;
    }
  }
  return crossApiMatch;
}

export function inferModelCapabilities(modelId) {
  const id = String(modelId).toLowerCase();
  const vision =
    /(^|[-/_.])(vision|vl)([-/_.]|$)/.test(id) ||
    /gpt-(4o|4\.1|5)/.test(id) ||
    /claude-(3|4|5)/.test(id) ||
    /gemini|pixtral|grok-(2-vision|4)/.test(id);
  const reasoning =
    /(^|[-/_.])(o1|o3|o4)([-/_.]|$)/.test(id) ||
    /gpt-5|claude-(3|4|5)|qwen3|qwq|deepseek[-_/.:]?r1|reason|thinking/.test(id);
  return { input: vision ? ["text", "image"] : ["text"], reasoning };
}

export function buildModel(connection, modelId) {
  if (typeof modelId !== "string" || !modelId.trim()) {
    throw new Error("model must be a non-empty string");
  }
  const protocol = selectConnectionProtocol(connection, modelId);
  const api = API_BY_PROTOCOL[protocol];
  const preferredProvider = connection.protocol === "opencode-go" ? "opencode-go" : undefined;
  const catalog = catalogMatch(modelId, api, preferredProvider);
  const { compat: catalogCompat, ...catalogMetadata } = catalog ?? {};
  const inferred = inferModelCapabilities(modelId);
  const provider = connection.protocol === "opencode-go" ? "opencode-go" : "offcut-custom";
  const isOpenCodeGoGlm53 =
    provider === "opencode-go" && (modelId === "glm-5.3" || modelId === "glm-5.3-flash");
  const compat =
    catalog?.api === api && catalogCompat
      ? { ...catalogCompat, ...(isOpenCodeGoGlm53 ? { thinkingFormat: "zai" } : {}) }
      : undefined;

  return {
    ...catalogMetadata,
    id: modelId,
    name: catalog?.name ?? modelId,
    api,
    provider,
    baseUrl: normalizeBaseUrl(connection.base_url, protocol),
    reasoning: catalog?.reasoning ?? inferred.reasoning,
    input: catalog?.input ?? inferred.input,
    cost: catalog?.cost ?? EMPTY_COST,
    contextWindow: catalog?.contextWindow ?? 128_000,
    maxTokens: catalog?.maxTokens ?? 16_384,
    ...(compat ? { compat } : {}),
  };
}

export function describeModel(connection, modelId) {
  const model = buildModel(connection, modelId);
  const supportedThinkingLevels = getSupportedThinkingLevels(model);
  const recommendedThinkingLevel = supportedThinkingLevels.includes("high")
    ? "high"
    : supportedThinkingLevels.includes("medium")
      ? "medium"
      : supportedThinkingLevels.at(-1) ?? "off";

  return {
    api: model.api,
    vision: model.input.includes("image"),
    reasoning: model.reasoning,
    supportedThinkingLevels,
    recommendedThinkingLevel,
  };
}
