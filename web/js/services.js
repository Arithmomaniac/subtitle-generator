/**
 * API service layer for subtitle-generator.
 * Pure IO — no DOM, no framework. Injectable fetch for testability.
 */

/** Track a custom event to App Insights (no-op if SDK not loaded). */
export function trackEvent(name, props) {
  try { if (typeof window !== "undefined" && window.appInsights) window.appInsights.trackEvent({ name, properties: props }); } catch(e) {}
}

/** Track a custom metric to App Insights. */
export function trackMetric(name, value, props) {
  try { if (typeof window !== "undefined" && window.appInsights) window.appInsights.trackMetric({ name, average: value }, props); } catch(e) {}
}

/**
 * Create an API client.
 * @param {string} baseUrl - API base URL (e.g., "" for same-origin, "http://localhost:8742")
 * @param {function} [fetchFn=fetch] - fetch implementation (inject for testing)
 * @returns {{ generate, jacket, health }}
 */
export function createApi(baseUrl = "", fetchFn = fetch) {
  async function post(path, body) {
    const r = await fetchFn(baseUrl + path, {
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

  return {
    /** Generate a subtitle. Returns API response or {error}. */
    async generate({ tone } = {}) {
      const t0 = performance.now();
      const result = await post("/api/generate", { tone: tone || null });
      const durationMs = Math.round(performance.now() - t0);
      trackMetric("GenerateDuration", durationMs, { tone: tone || "any", remixed: String(!!result.remixed) });
      if (result.error) {
        trackEvent("GenerateError", { error: result.error, durationMs: String(durationMs) });
      } else {
        trackEvent("GenerateSuccess", { durationMs: String(durationMs), remixed: String(result.remixed), tone: tone || "any" });
      }
      return result;
    },

    /** Build jacket prompt and optionally run LLM. Returns {prompt, tone_tier, result} or {error}. */
    async jacket({ subtitle, model, dryRun = true } = {}) {
      const t0 = performance.now();
      const result = await post("/api/jacket", {
        subtitle,
        model: model || "gpt-5.4-mini",
        dry_run: dryRun,
      });
      const durationMs = Math.round(performance.now() - t0);
      trackMetric("JacketDuration", durationMs, { dryRun: String(dryRun), model: model || "gpt-5.4-mini" });
      return result;
    },

    /** Check server health. Returns {ok, mode} or {error}. */
    async health() {
      try {
        const r = await fetchFn(baseUrl + "/api/health");
        if (!r.ok) return { error: r.statusText };
        return r.json();
      } catch {
        return { error: "unreachable" };
      }
    },

    /** List available LLM models (local mode only). Returns {models: [{id, name, cost}]} or {error}. */
    async models() {
      try {
        const r = await fetchFn(baseUrl + "/api/models");
        if (!r.ok) return { error: r.statusText };
        return r.json();
      } catch {
        return { error: "unreachable" };
      }
    },

    /** Submit a human rating for a subtitle. */
    async rate({ subtitle, thumbs, tone_override, system_tone, free_text } = {}) {
      const result = await post("/api/rate", { subtitle, thumbs, tone_override, system_tone, free_text });
      trackEvent("RateSubtitle", { thumbs: String(thumbs), tone_override: tone_override || "none" });
      return result;
    },
  };
}
