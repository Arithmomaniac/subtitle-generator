/**
 * Alpine.js app component for spot-check page.
 * Loads batches of tone-targeted subtitles and collects tier ratings.
 */

const API_BASE = document.querySelector('meta[name="api-base"]')?.content || "";

async function post(path, body) {
  const r = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ error: r.statusText }));
    return { error: err.error || r.statusText };
  }
  return r.json();
}

export function spotCheckApp() {
  return {
    // ── State ──
    phase: "idle",         // idle | rating | reveal | summary
    loading: false,
    submitting: false,
    samplesPerTier: 2,

    items: [],             // current batch items from API
    currentIndex: 0,
    results: [],           // results for current batch
    currentTags: [],       // tags being toggled in reveal phase
    currentResult: null,   // last rating response (for reveal)

    // Session-wide stats (persists across batches)
    sessionStats: { total: 0, correct: 0, skipped: 0 },

    // ── Computed ──
    get currentItem() {
      return this.items[this.currentIndex] || null;
    },

    get batchStats() {
      const rated = this.results.filter(r => r.match !== null).length;
      const correct = this.results.filter(r => r.match === true).length;
      const skipped = this.results.filter(r => r.match === null).length;
      return {
        rated,
        correct,
        skipped,
        accuracy: rated > 0 ? correct / rated : 0,
      };
    },

    get sessionAccuracy() {
      return this.sessionStats.total > 0
        ? this.sessionStats.correct / this.sessionStats.total
        : 0;
    },

    get batchMismatches() {
      return this.results
        .filter(r => r.match === false)
        .map(r => ({ text: r.text, felt: r.felt_tier, target: r.target_tier }));
    },

    // ── Init ──
    init() {
      // Restore session stats from sessionStorage
      try {
        const saved = JSON.parse(sessionStorage.getItem("spot-check-session"));
        if (saved) this.sessionStats = saved;
      } catch {}
    },

    _saveSession() {
      sessionStorage.setItem("spot-check-session", JSON.stringify(this.sessionStats));
    },

    // ── Load batch ──
    async loadBatch() {
      this.loading = true;
      const result = await post("/api/spot-check/batch", {
        samples_per_tier: this.samplesPerTier,
      });

      if (result.error) {
        alert("Error loading batch: " + result.error);
        this.loading = false;
        return;
      }

      this.items = result.items;
      this.currentIndex = 0;
      this.results = [];
      this.currentTags = [];
      this.currentResult = null;
      this.phase = "rating";
      this.loading = false;
    },

    // ── Rate tier ──
    async rateTier(tier) {
      if (this.submitting || !this.currentItem) return;
      this.submitting = true;

      const result = await post("/api/spot-check/rate", {
        sample_id: this.currentItem.sample_id,
        felt_tier: tier,
      });

      if (result.error) {
        alert("Error submitting rating: " + result.error);
        this.submitting = false;
        return;
      }

      // Store result for reveal + summary
      const entry = {
        ...result,
        felt_tier: tier,
        text: this.currentItem.text,
        tags: [],
      };
      this.results.push(entry);
      this.currentResult = entry;

      // Update session stats
      this.sessionStats.total++;
      if (result.match) this.sessionStats.correct++;
      this._saveSession();

      this.submitting = false;
      this.phase = "reveal";
    },

    // ── Skip ──
    async skip() {
      if (this.submitting || !this.currentItem) return;
      this.submitting = true;

      const result = await post("/api/spot-check/rate", {
        sample_id: this.currentItem.sample_id,
        skipped: true,
      });

      if (result.error) {
        alert("Error submitting skip: " + result.error);
        this.submitting = false;
        return;
      }

      const entry = {
        ...result,
        felt_tier: null,
        text: this.currentItem.text,
        tags: [],
      };
      this.results.push(entry);
      this.currentResult = entry;

      this.sessionStats.skipped++;
      this._saveSession();

      this.submitting = false;
      this.phase = "reveal";
    },

    // ── Tags (during reveal phase) ──
    toggleTag(tag) {
      const idx = this.currentTags.indexOf(tag);
      if (idx >= 0) {
        this.currentTags.splice(idx, 1);
      } else {
        this.currentTags.push(tag);
      }
      // Update the result entry so tags are tracked
      if (this.currentResult) {
        this.currentResult.tags = [...this.currentTags];
      }
    },

    // ── Next (advance from reveal) ──
    async next() {
      // If tags were toggled during reveal, re-submit with tags
      if (this.currentTags.length > 0 && this.currentResult) {
        await post("/api/spot-check/rate", {
          sample_id: this.currentItem.sample_id,
          felt_tier: this.currentResult.felt_tier,
          skipped: this.currentResult.match === null,
          tags: this.currentTags,
        });
      }

      this.currentTags = [];
      this.currentResult = null;
      this.currentIndex++;

      if (this.currentIndex >= this.items.length) {
        this.phase = "summary";
      } else {
        this.phase = "rating";
      }
    },

    // ── Keyboard handler ──
    handleKey(event) {
      // Ignore when focus is in form controls
      const tag = event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      // Ignore key repeats
      if (event.repeat) return;

      const key = event.key.toLowerCase();

      if (this.phase === "rating") {
        if (key === "p") { this.rateTier("pop"); event.preventDefault(); }
        else if (key === "m") { this.rateTier("mainstream"); event.preventDefault(); }
        else if (key === "n") { this.rateTier("niche"); event.preventDefault(); }
        else if (key === "s") { this.skip(); event.preventDefault(); }
      } else if (this.phase === "reveal") {
        if (key === "f") { this.toggleTag("funny"); event.preventDefault(); }
        else if (key === "g") { this.toggleTag("grammar"); event.preventDefault(); }
        else if (key === "c") { this.toggleTag("contradiction"); event.preventDefault(); }
        else if (key === "b") { this.toggleTag("boring"); event.preventDefault(); }
        else if (key === "enter") { this.next(); event.preventDefault(); }
      }
    },
  };
}
