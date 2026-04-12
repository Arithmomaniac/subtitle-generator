/**
 * API service layer for subtitle-generator.
 * Pure IO — no DOM, no framework. Injectable fetch for testability.
 */

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
      return post("/api/generate", {
        tone: tone || null,
      });
    },

    /** Build jacket prompt and optionally run LLM. Returns {prompt, tone_tier, result} or {error}. */
    async jacket({ subtitle, model, deepResearch, dryRun = true } = {}) {
      return post("/api/jacket", {
        subtitle,
        model: model || "gpt-5.4-mini",
        deep_research: !!deepResearch,
        dry_run: dryRun,
      });
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
  };
}
