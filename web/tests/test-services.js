/**
 * Unit tests for services.js — zero-framework pattern with mock fetch.
 * Run with: node web/tests/test-services.js
 */

import { createApi } from "../js/services.js";

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (condition) { passed++; }
  else { failed++; console.error("  FAIL:", msg); }
}

async function test(name, fn) {
  try { await fn(); }
  catch (e) { failed++; console.error("  FAIL:", name, e.message); }
}

/** Create a mock fetch that returns a canned response. */
function mockFetch(status, body) {
  return async (url, opts) => ({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? "OK" : "Error",
    json: async () => body,
  });
}

// ── health ──

await test("health returns mode on success", async () => {
  const api = createApi("", mockFetch(200, { ok: true, mode: "local" }));
  const r = await api.health();
  assert(r.ok === true, "ok should be true");
  assert(r.mode === "local", "mode should be local");
});

await test("health returns error on failure", async () => {
  const api = createApi("", mockFetch(500, { error: "down" }));
  const r = await api.health();
  assert(r.error !== undefined, "should have error");
});

await test("health returns error on network failure", async () => {
  const api = createApi("", async () => { throw new Error("network"); });
  const r = await api.health();
  assert(r.error === "unreachable", "should be unreachable");
});

// ── generate ──

await test("generate sends correct body", async () => {
  let capturedBody;
  const mockF = async (url, opts) => {
    capturedBody = JSON.parse(opts.body);
    return { ok: true, json: async () => ({ text: "A, B, and the C of D" }) };
  };
  const api = createApi("", mockF);
  await api.generate({ tone: "pop" });
  assert(capturedBody.tone === "pop", "tone sent");
});

await test("generate with no options sends null tone", async () => {
  let capturedBody;
  const mockF = async (url, opts) => {
    capturedBody = JSON.parse(opts.body);
    return { ok: true, json: async () => ({}) };
  };
  const api = createApi("", mockF);
  await api.generate();
  assert(capturedBody.tone === null, "tone null");
});

await test("generate returns error on HTTP failure", async () => {
  const api = createApi("", mockFetch(400, { error: "bad input" }));
  const r = await api.generate({ tone: "pop" });
  assert(r.error === "bad input", "error message forwarded");
});

// ── jacket ──

await test("jacket sends correct body for dry_run", async () => {
  let capturedBody;
  const mockF = async (url, opts) => {
    capturedBody = JSON.parse(opts.body);
    return { ok: true, json: async () => ({ prompt: "...", tone_tier: "pop", result: null }) };
  };
  const api = createApi("", mockF);
  await api.jacket({ subtitle: "test subtitle", dryRun: true });
  assert(capturedBody.subtitle === "test subtitle", "subtitle sent");
  assert(capturedBody.dry_run === true, "dry_run sent");
  assert(capturedBody.deep_research === false, "deep_research default false");
});

await test("jacket sends correct body for full generation", async () => {
  let capturedBody;
  const mockF = async (url, opts) => {
    capturedBody = JSON.parse(opts.body);
    return { ok: true, json: async () => ({ prompt: "...", result: "## Title" }) };
  };
  const api = createApi("", mockF);
  await api.jacket({ subtitle: "test", model: "gpt-4.1", deepResearch: true, dryRun: false });
  assert(capturedBody.model === "gpt-4.1", "model sent");
  assert(capturedBody.deep_research === true, "deep_research sent");
  assert(capturedBody.dry_run === false, "dry_run false");
});

// ── baseUrl ──

await test("baseUrl is prepended to paths", async () => {
  let capturedUrl;
  const mockF = async (url, opts) => {
    capturedUrl = url;
    return { ok: true, json: async () => ({}) };
  };
  const api = createApi("http://localhost:9999", mockF);
  await api.health();
  assert(capturedUrl === "http://localhost:9999/api/health", `URL: ${capturedUrl}`);
});

// ── Summary ──
console.log(`\n${passed} passed, ${failed} failed (${passed + failed} total)`);
process.exit(failed > 0 ? 1 : 0);
