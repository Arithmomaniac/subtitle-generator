/**
 * Alpine.js app component for subtitle-generator.
 * Wires services + VM into reactive state.
 */

import { createApi } from "./services.js";
import { deriveSubtitleVM, deriveSourcesVM, buildSettingsVM, cleanJacketMarkdown } from "./subtitle-vm.js";

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
  const apiBase = document.querySelector('meta[name="api-base"]')?.content || "";
  const api = createApi(apiBase);

  return {
    // ── Reactive state ──
    mode: "detecting",
    settingsVis: {},
    loading: false,
    jacketLoading: false,
    jacketProgress: "",

    // Settings
    tone: "",
    model: "gpt-5.4-mini",
    settingsOpen: true,
    availableModels: [],

    // Subtitle display
    subtitle: { slots: [], fullText: "", remixed: false, similarity: null },
    sources: [],
    hasSubtitle: false,

    // Jacket/prompt
    jacket: null,
    jacketHtml: "",
    jacketMd: "",
    prompt: null,
    toneTier: null,

    // ── Init ──
    async init() {
      const saved = loadSettings();
      if (saved.tone) this.tone = saved.tone;
      if (saved.model) this.model = saved.model;

      const h = await api.health();
      this.mode = h.error ? "azure" : (h.mode || "local");
      this.settingsVis = buildSettingsVM(this.mode);

      // Fetch available models in local mode
      if (this.mode === "local") {
        const m = await api.models();
        if (m.models && m.models.length > 0) {
          this.availableModels = m.models;
          // If saved model not in list, fall back to first
          if (!m.models.some(x => x.id === this.model)) {
            this.model = m.models[0].id;
          }
        }
      }
    },

    // ── Settings persistence ──
    saveSettings() {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify({
        tone: this.tone,
        model: this.model,
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
      this._setJacket(null);
      this.prompt = null;

      const result = await api.generate({
        tone: this.tone || null,
      });

      if (result.error) {
        alert("Error: " + result.error);
      } else {
        this._rawSub = result;
        this.subtitle = deriveSubtitleVM(result);
        this.sources = deriveSourcesVM(result);
        this.hasSubtitle = true;
        this.ratingSubmitted = false;
        this.ratingToneRevealed = false;
        this.selectedTone = null;
        this.selectedTags = [];
      }
      this.loading = false;
    },

    // ── Rating ──
    ratingSubmitted: false,
    ratingToneRevealed: false,
    selectedTone: null,
    ratingSystemTone: null,
    ratingScore: null,
    selectedTags: [],
    tagsExpanded: sessionStorage.getItem('tagsExpanded') === '1',

    toggleTag(tag) {
      const idx = this.selectedTags.indexOf(tag);
      if (idx >= 0) this.selectedTags.splice(idx, 1);
      else this.selectedTags.push(tag);
    },

    async submitRating(thumbs) {
      if (!this.hasSubtitle || this.ratingSubmitted) return;
      const body = {
        subtitle: this.subtitle.fullText,
        thumbs,
        tone_override: this.selectedTone,
        system_tone: this.tone || null,
        tags: this.selectedTags.length ? this.selectedTags : undefined,
      };
      await api.rate(body);
      this.ratingSubmitted = true;
      this.ratingToneRevealed = true;
      if (this.selectedTags.length) {
        this.tagsExpanded = true;
        sessionStorage.setItem('tagsExpanded', '1');
      }
    },

    selectTone(tone) {
      this.selectedTone = this.selectedTone === tone ? null : tone;
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
      this.jacketProgress = "";

      if (dryRun) {
        // Simple JSON request for prompt-only
        const result = await api.jacket({
          subtitle: this.subtitle.fullText,
          model: this.model,
          dryRun: true,
        });
        if (result.error) {
          alert("Error: " + result.error);
        } else {
          this.prompt = result.prompt;
          this.toneTier = result.tone_tier;
          this._setJacket(result.result);
        }
        this.jacketLoading = false;
        return;
      }

      // SSE streaming for full jacket generation
      try {
        const body = JSON.stringify({
          subtitle: this.subtitle.fullText,
          model: this.model,
          dry_run: false,
        });
        const response = await fetch(apiBase + "/api/jacket", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          // Parse SSE events from buffer
          // SSE spec: events are separated by blank lines, data can span multiple "data:" lines
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop(); // keep incomplete block
          for (const block of blocks) {
            if (!block.trim()) continue;
            let eventType = null;
            const dataLines = [];
            for (const line of block.split("\n")) {
              if (line.startsWith("event: ")) eventType = line.slice(7);
              else if (line.startsWith("data: ")) dataLines.push(line.slice(6));
              else if (line.startsWith("data:")) dataLines.push(line.slice(5));
            }
            if (!eventType || dataLines.length === 0) continue;
            const data = dataLines.join("\n");
            if (eventType === "progress") {
              this.jacketProgress = data;
            } else if (eventType === "result") {
              try {
                const parsed = JSON.parse(data);
                this.prompt = parsed.prompt;
                this.toneTier = parsed.tone_tier;
                this._setJacket(parsed.result);
              } catch { /* partial JSON, wait for more */ }
            } else if (eventType === "error") {
              alert("Error: " + data);
            }
          }
        }
      } catch (e) {
        alert("Failed: " + e.message);
      }
      this.jacketLoading = false;
      this.jacketProgress = "";
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

    async copyJacketMd() {
      if (!this.jacketMd) return;
      await navigator.clipboard.writeText(this.jacketMd);
      this._flashCopy("copyMdBtn", "Copied!");
    },

    async copyJacketHtml() {
      if (!this.jacketHtml) return;
      // Copy as rich text (HTML) to clipboard
      try {
        const blob = new Blob([this.jacketHtml], { type: "text/html" });
        await navigator.clipboard.write([new ClipboardItem({ "text/html": blob })]);
      } catch {
        await navigator.clipboard.writeText(this.jacketHtml);
      }
      this._flashCopy("copyHtmlBtn", "Copied!");
    },

    // ── Helpers ──
    _setJacket(raw) {
      if (!raw) {
        this.jacket = null;
        this.jacketMd = "";
        this.jacketHtml = "";
        return;
      }
      this.jacketMd = cleanJacketMarkdown(raw);
      this.jacket = raw;
      // Render markdown to HTML via marked.js (loaded from CDN)
      if (typeof marked !== "undefined" && marked.parse) {
        this.jacketHtml = marked.parse(this.jacketMd);
      } else {
        this.jacketHtml = this.jacketMd;
      }
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
