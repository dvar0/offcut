/* Theme catalog for Offcut.
 *
 * Adding a theme:
 *   1. Add a [data-theme='id'] palette in themes.css
 *   2. Register it here (THEMES + a family dark/light id)
 *
 * Removing a theme: delete those two entries. Saved ids that vanish fall back
 * to DEFAULT_APPEARANCE.
 *
 * Changing the default: edit DEFAULT_APPEARANCE. `dark` / `light` are used when
 * mode is "system"; `manual` is used when the user locks a single theme.
 */
(function (global) {
  const STORAGE_KEY = "offcut.appearance";

  const DEFAULT_APPEARANCE = {
    mode: "system",
    dark: "periwinkle-owl-dark",
    light: "periwinkle-owl",
    manual: "periwinkle-owl-dark",
  };

  const FAMILIES = [
    {
      id: "offcut",
      label: "Offcut",
      description: "Warm bronze on paper or near-black.",
      dark: "offcut-dark",
      light: "offcut-light",
    },
    {
      id: "oracle",
      label: "Oracle",
      description: "Amber phosphor on black, paired with a pale blue terminal.",
      dark: "oracle",
      light: "nous",
    },
    {
      id: "everforest",
      label: "Everforest",
      description: "Muted greens and stone, dark or parchment.",
      dark: "everforest-dark",
      light: "everforest-light",
    },
    {
      id: "periwinkle",
      label: "Periwinkle Owl",
      description: "Dusty violet-blue, dusk or daylight.",
      dark: "periwinkle-owl-dark",
      light: "periwinkle-owl",
    },
    {
      id: "astral-rose",
      label: "Astral Rose",
      description: "Magenta on plum, or rose ink on blush paper.",
      dark: "astral-rose",
      light: "astral-rose-light",
    },
  ];

  const THEMES = {
    "offcut-dark": {
      id: "offcut-dark",
      family: "offcut",
      scheme: "dark",
      label: "Offcut Dark",
      description: "Bronze on near-black.",
      swatches: ["#050505", "#d9842f", "#f4f1ea"],
    },
    "offcut-light": {
      id: "offcut-light",
      family: "offcut",
      scheme: "light",
      label: "Offcut Light",
      description: "The same bronze, on warm paper.",
      swatches: ["#f3efe6", "#b56a1f", "#1c1b18"],
    },
    oracle: {
      id: "oracle",
      family: "oracle",
      scheme: "dark",
      label: "Oracle Amber",
      description: "Phosphor amber and a cold blue secondary.",
      swatches: ["#050505", "#ffb000", "#0000ff"],
    },
    nous: {
      id: "nous",
      family: "oracle",
      scheme: "light",
      label: "Nous Terminal",
      description: "Ice-blue paper and a steel caption.",
      swatches: ["#f6fbff", "#0088cc", "#0f3550"],
    },
    "everforest-dark": {
      id: "everforest-dark",
      family: "everforest",
      scheme: "dark",
      label: "Everforest Dark",
      description: "Moss on charcoal.",
      swatches: ["#2d353b", "#a7c080", "#7fbbb3"],
    },
    "everforest-light": {
      id: "everforest-light",
      family: "everforest",
      scheme: "light",
      label: "Everforest Light",
      description: "Leaf on cream parchment.",
      swatches: ["#fdf6e3", "#8da101", "#3a94c5"],
    },
    "periwinkle-owl-dark": {
      id: "periwinkle-owl-dark",
      family: "periwinkle",
      scheme: "dark",
      label: "Periwinkle Dark",
      description: "Slate dusk and dusty lilac.",
      swatches: ["#283045", "#92a3d3", "#9f9adb"],
    },
    "periwinkle-owl": {
      id: "periwinkle-owl",
      family: "periwinkle",
      scheme: "light",
      label: "Periwinkle Light",
      description: "Lavender paper and owl-grey ink.",
      swatches: ["#e8ebf8", "#9f9adb", "#5567a0"],
    },
    "astral-rose": {
      id: "astral-rose",
      family: "astral-rose",
      scheme: "dark",
      label: "Astral Rose Dark",
      description: "Hot pink on plum-black.",
      swatches: ["#0c080f", "#e8579a", "#a07ac8"],
    },
    "astral-rose-light": {
      id: "astral-rose-light",
      family: "astral-rose",
      scheme: "light",
      label: "Astral Rose Light",
      description: "Rose ink on blush paper.",
      swatches: ["#fdf2f7", "#c23e8a", "#7b4faf"],
    },
  };

  function cloneDefaults() {
    return {
      mode: DEFAULT_APPEARANCE.mode,
      dark: DEFAULT_APPEARANCE.dark,
      light: DEFAULT_APPEARANCE.light,
      manual: DEFAULT_APPEARANCE.manual,
    };
  }

  function normalize(raw) {
    const next = cloneDefaults();
    if (!raw || typeof raw !== "object") return next;
    if (raw.mode === "manual" || raw.mode === "system") next.mode = raw.mode;
    if (THEMES[raw.dark] && THEMES[raw.dark].scheme === "dark") next.dark = raw.dark;
    if (THEMES[raw.light] && THEMES[raw.light].scheme === "light") next.light = raw.light;
    if (THEMES[raw.manual]) next.manual = raw.manual;
    return next;
  }

  function loadAppearance() {
    try {
      const raw = global.localStorage.getItem(STORAGE_KEY);
      return normalize(raw ? JSON.parse(raw) : {});
    } catch (_) {
      return cloneDefaults();
    }
  }

  function saveAppearance(appearance) {
    const next = normalize(appearance);
    try {
      global.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    } catch (_) { /* Private mode can block storage; the session still applies. */ }
    return next;
  }

  function systemPrefersDark() {
    return Boolean(global.matchMedia && global.matchMedia("(prefers-color-scheme: dark)").matches);
  }

  function resolveThemeId(appearance) {
    const prefs = normalize(appearance);
    if (prefs.mode === "manual") return THEMES[prefs.manual] ? prefs.manual : DEFAULT_APPEARANCE.manual;
    return systemPrefersDark() ? prefs.dark : prefs.light;
  }

  function applyTheme(id) {
    const theme = THEMES[id] || THEMES[DEFAULT_APPEARANCE.manual];
    const root = document.documentElement;
    root.dataset.theme = theme.id;
    root.dataset.scheme = theme.scheme;
    root.style.colorScheme = theme.scheme;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = theme.swatches[0];
    return theme;
  }

  function applyAppearance(appearance) {
    return applyTheme(resolveThemeId(appearance));
  }

  function setAppearance(patch) {
    const next = saveAppearance({ ...loadAppearance(), ...patch });
    const theme = applyAppearance(next);
    global.dispatchEvent(new CustomEvent("offcut-appearance", { detail: { appearance: next, theme } }));
    return { appearance: next, theme };
  }

  function listenToSystem() {
    if (!global.matchMedia) return;
    const media = global.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => {
      const appearance = loadAppearance();
      if (appearance.mode !== "system") return;
      applyAppearance(appearance);
      global.dispatchEvent(new CustomEvent("offcut-appearance", { detail: { appearance, theme: THEMES[resolveThemeId(appearance)] } }));
    };
    if (media.addEventListener) media.addEventListener("change", onChange);
    else media.addListener(onChange);
  }

  function themesForScheme(scheme) {
    return Object.values(THEMES).filter((theme) => theme.scheme === scheme);
  }

  global.KreaThemes = {
    STORAGE_KEY,
    DEFAULT_APPEARANCE,
    FAMILIES,
    THEMES,
    loadAppearance,
    saveAppearance,
    setAppearance,
    resolveThemeId,
    applyTheme,
    applyAppearance,
    systemPrefersDark,
    themesForScheme,
  };

  applyAppearance(loadAppearance());
  listenToSystem();
})(window);
