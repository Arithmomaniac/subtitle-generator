/**
 * Alpine.js app component for subtitle-generator.
 * Wires services + VM into reactive state.
 */

import { createApi } from "./services.js";
import { deriveSubtitleVM, deriveSourcesVM, buildSettingsVM } from "./subtitle-vm.js";

const SETTINGS_KEY = "subtitle-gen-settings";

/** Load settings from localStorage. */
function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY)) || {};
  } catch {
    return {};
  }
}

/**
 * Create the Alpine x-data component.
 * @returns {object} Alpine data object
 */
export function createApp() {
  const api = createApi();

  return {
    // ── Reactive state ──
    mode: "detecting",
    settingsVis: {},
    loading: false,
    jacketLoading: false,

    // Settings
    tone: "",
    remix: true,
    remixProb: 0.8,
    minSim: 0.1,
    model: "gpt-5.4-mini",
    deepResearch: false,
    settingsOpen: true,

    // Subtitle display
    subtitle: { slots: [], fullText: "", remixed: false, similarity: null },
    sources: [],
    hasSubtitle: false,

    // Jacket/prompt
    jacket: null,
    prompt: null,
    toneTier: null,

    // ── Init ──
    async init() {
      const saved = loadSettings();
      if (saved.tone) this.tone = saved.tone;
      if (saved.remix === false) this.remix = false;
      if (saved.remixProb != null) this.remixProb = saved.remixProb;
      if (saved.minSim != null) this.minSim = saved.minSim;
      if (saved.model) this.model = saved.model;
      if (saved.deepResearch) this.deepResearch = true;

      const h = await api.health();
      this.mode = h.error ? "azure" : (h.mode || "local");
      this.settingsVis = buildSettingsVM(this.mode);
    },

    // ── Settings persistence ──
    saveSettings() {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        tone: this.tone,
        remix: this.remix,
        remixProb: this.remixProb,
        minSim: this.minSim,
        model: this.model,
        deepResearch: this.deepResearch,
      }));
    },

    get modeBadgeClass() {
      return this.mode === "local" ? "mode-local" : "mode-azure";
    },

    get modeBadgeText() {
      if (this.mode === "detecting") return "detecting...";
      return this.mode === "local" ? "Local Mode" : "Web Mode";
    },

    // ── Generate ──
    async generate() {
      this.loading = true;
      this.jacket = null;
      this.prompt = null;

      const result = await api.generate({
        tone: this.tone || null,
        remixProb: this.remix ? this.remixProb : 0,
        minSim: this.minSim,
      });

      if (result.error) {
        alert("Error: " + result.error);
      } else {
        this._rawSub = result;
        this.subtitle = deriveSubtitleVM(result);
        this.sources = deriveSourcesVM(result);
        this.hasSubtitle = true;
      }
      this.loading = false;
    },

    // ── Jacket ──
    async buildPrompt() {
      await this._doJacket(true);
    },

    async generateJacket() {
      await this._doJacket(false);
    },

    async _doJacket(dryRun) {
      if (!this.hasSubtitle) { alert("Generate a subtitle first."); return; }
      this.jacketLoading = true;

      const result = await api.jacket({
        subtitle: this.subtitle.fullText,
        model: this.model,
        deepResearch: this.deepResearch,
        dryRun,
      });

      if (result.error) {
        alert("Error: " + result.error);
      } else {
        this.prompt = result.prompt;
        this.toneTier = result.tone_tier;
        this.jacket = result.result;
      }
      this.jacketLoading = false;
    },

    // ── Clipboard ──
    async copySubtitle() {
      if (!this.hasSubtitle) return;
      await navigator.clipboard.writeText(this.subtitle.fullText);
      this._flashCopy("copyBtn", "Copied!");
    },

    async copyPrompt() {
      if (!this.prompt) return;
      await navigator.clipboard.writeText(this.prompt);
      this._flashCopy("copyPromptBtn", "Copied!");
    },

    _flashCopy(refName, text) {
      const el = this.$refs[refName];
      if (!el) return;
      const orig = el.textContent;
      el.textContent = text;
      setTimeout(() => (el.textContent = orig), 1500);
    },
  };
}
