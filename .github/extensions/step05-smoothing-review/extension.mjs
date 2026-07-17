// Extension: step05-smoothing-review
// Step 5 semantic-smoothing human-review cockpit. Reads the candidate feed
// (generated-artifacts/.../step05_review_feed.json), lets the reviewer rate each
// boosted filler and record an overall accept/reject/iterate decision, then
// persists the submission via the `ingest-smoothing-ratings` CLI (ratings -> DB,
// decision -> committed feedback/step05-smoothing/decision.json).
//
// Analysis-only: nothing here touches the served distribution.

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { joinSession, createCanvas } from "@github/copilot-sdk/extension";

const EXT_DIR = dirname(fileURLToPath(import.meta.url));
// .github/extensions/step05-smoothing-review -> repo root is three levels up.
const REPO_ROOT = resolve(EXT_DIR, "..", "..", "..");
const FEED_PATH = join(
    REPO_ROOT,
    "generated-artifacts",
    "tier-slot-distribution",
    "step05_review_feed.json",
);
const SUBMISSION_PATH = join(
    REPO_ROOT,
    "generated-artifacts",
    "tier-slot-distribution",
    "step05_review_submission.json",
);
const DECISION_PATH = join(REPO_ROOT, "feedback", "step05-smoothing", "decision.json");

const servers = new Map();

async function readFeed() {
    const raw = await readFile(FEED_PATH, "utf-8");
    return JSON.parse(raw);
}

// Persist the submission JSON, then ingest it (ratings -> DB, decision -> JSON)
// via the tested Python CLI. The submission file persists either way, so a
// failed ingest can be retried manually with the same command.
async function saveSubmission(submission, log) {
    await mkdir(dirname(SUBMISSION_PATH), { recursive: true });
    await writeFile(SUBMISSION_PATH, JSON.stringify(submission, null, 2) + "\n", "utf-8");

    return await new Promise((resolvePromise) => {
        const child = spawn(
            "uv",
            [
                "run",
                "--no-sync",
                "subtitle-gen",
                "ingest-smoothing-ratings",
                "--submission",
                SUBMISSION_PATH,
                "--decision-path",
                DECISION_PATH,
            ],
            { cwd: REPO_ROOT, shell: process.platform === "win32" },
        );
        let stderr = "";
        child.stderr.on("data", (d) => (stderr += d.toString()));
        child.on("error", (err) => {
            resolvePromise({ ok: false, error: String(err), submission: SUBMISSION_PATH });
        });
        child.on("close", (code) => {
            if (code === 0) {
                resolvePromise({ ok: true, decision: DECISION_PATH });
            } else {
                if (log) log(`ingest failed (${code}): ${stderr}`, { level: "error" });
                resolvePromise({
                    ok: false,
                    error: stderr || `exit ${code}`,
                    submission: SUBMISSION_PATH,
                });
            }
        });
    });
}

function renderHtml() {
    // The page fetches the feed from /api/feed and posts to /api/save on this
    // same local server, so all data flow stays on loopback. The UI leads with
    // plain-language context (an example subtitle, the tier "feel", familiar
    // similar words, source books) and tucks ML internals into a details toggle.
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Step 5 smoothing review</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem 7rem; line-height: 1.45; max-width: 60rem; }
    h1 { font-size: 1.25rem; margin: 0 0 .35rem; }
    .intro { font-size: .9rem; opacity: .85; margin-bottom: .6rem; }
    .legend { font-size: .78rem; opacity: .7; margin-bottom: 1rem; }
    .legend b { opacity: .9; }
    .cand { border: 1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius: 10px; padding: .8rem .95rem; margin-bottom: .85rem; }
    .subtitle { font-size: 1.05rem; margin: 0 0 .45rem; }
    .subtitle .w { font-weight: 700; background: color-mix(in srgb, gold 35%, transparent); padding: 0 .15rem; border-radius: 3px; }
    .why { font-size: .9rem; margin: .15rem 0; }
    .why .tier { font-weight: 600; }
    .sim { font-size: .88rem; opacity: .9; }
    .src { font-size: .82rem; opacity: .75; font-style: italic; margin-top: .2rem; }
    .opts { display: flex; gap: .4rem; flex-wrap: wrap; margin-top: .6rem; }
    .opts label { border: 1px solid color-mix(in srgb, currentColor 30%, transparent); border-radius: 999px; padding: .2rem .7rem; font-size: .85rem; cursor: pointer; user-select: none; }
    .opts input { margin-right: .35rem; }
    .opts label:has(input:checked) { background: color-mix(in srgb, currentColor 14%, transparent); border-color: currentColor; }
    .notes { width: 100%; margin-top: .4rem; font: inherit; padding: .3rem .45rem; box-sizing: border-box; }
    details { margin-top: .45rem; font-size: .78rem; opacity: .65; }
    details code { font-variant-numeric: tabular-nums; }
    .bar { position: fixed; left: 0; right: 0; bottom: 0; padding: .65rem 1.25rem; background: Canvas; border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
    .bar select, .bar input, .bar button { font: inherit; padding: .35rem .55rem; }
    .bar input[type=text] { flex: 1; min-width: 12rem; }
    button { cursor: pointer; border-radius: 6px; }
    #status { font-size: .85rem; }
    #progress { font-size: .8rem; opacity: .7; }
  </style>
</head>
<body>
  <h1>Which words fit which subtitle style?</h1>
  <div class="intro">
    The generator builds subtitles like <i>“Greed, Grit, and the Rise of the American Dream.”</i>
    It’s learning which words feel right for each <b>style tier</b>. For each suggestion below,
    the system wants to make a word a more likely choice for a tier because it resembles words
    already common there — <b>your call is whether it actually fits</b> that style and slot.
    Rate the ones you have a view on; you don’t have to rate them all.
  </div>
  <div class="legend">
    Tiers: <b>Mass-market</b> (bestseller / BookTok) ·
    <b>Broad trade</b> (book-club / NPR) ·
    <b>Scholarly</b> (academic / specialty).
  </div>
  <div id="meta" class="legend">Loading…</div>
  <div id="list"></div>

  <div class="bar">
    <span id="progress"></span>
    <label>Overall:
      <select id="overall">
        <option value="">— choose —</option>
        <option value="accept">accept (use this smoothing)</option>
        <option value="iterate">iterate (try another variant)</option>
        <option value="reject">reject (don’t smooth)</option>
      </select>
    </label>
    <input type="text" id="summary" placeholder="One-line reason for your overall call (required)" />
    <input type="text" id="reviewer" placeholder="your name" style="max-width:7rem" />
    <button id="save">Save review</button>
    <span id="status"></span>
  </div>

<script>
// value = stored enum; label = plain-language meaning shown to the reviewer.
const OPTIONS = [
  ["plausible_repair", "Good fit"],
  ["semantic_bleed", "Wrong vibe / off-topic"],
  ["too_generic", "Too generic / bland"],
  ["needs_context", "Can’t tell"],
];
let FEED = null;

function esc(s) { return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }

function exampleHtml(text, word) {
  // Highlight the candidate word within the example subtitle.
  const i = text.toLowerCase().indexOf(String(word).toLowerCase());
  if (i < 0) return esc(text);
  return esc(text.slice(0, i)) + '<span class="w">' + esc(text.slice(i, i + word.length)) + '</span>' + esc(text.slice(i + word.length));
}

function updateProgress() {
  const rated = FEED.candidates.filter((_, i) => document.querySelector(\`input[name="d\${i}"]:checked\`)).length;
  document.getElementById("progress").textContent = \`\${rated}/\${FEED.candidates.length} rated\`;
}

async function load() {
  const res = await fetch("/api/feed");
  if (!res.ok) { document.getElementById("meta").textContent = "No feed found. Run build-smoothing-review-feed first."; return; }
  FEED = await res.json();
  document.getElementById("meta").textContent =
    \`\${FEED.candidate_count} suggestions from variant “\${FEED.variant}”.\`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  FEED.candidates.forEach((c, i) => {
    const x = c.context || {};
    const word = c.display_filler;
    const similar = (x.similar_words || []).slice(0, 5).join(", ");
    const sources = (x.source_titles || []);
    const div = document.createElement("div");
    div.className = "cand";
    div.innerHTML =
      \`<p class="subtitle">\${exampleHtml(x.example_subtitle || word, word)}</p>\` +
      \`<p class="why">Style: <span class="tier">\${esc(x.tier_label || c.tier)}</span> — \${esc(x.tier_blurb || "")}<br>\` +
      \`Role: \${esc(x.slot_label || c.slot_type)}. Smoothing would make <b>\${esc(word)}</b> \${esc(x.lift_phrase || "more likely")} here.</p>\` +
      (similar ? \`<p class="sim">Suggested because it resembles: \${esc(similar)}.</p>\` : "") +
      (sources.length ? \`<p class="src">From books like: \${sources.map(esc).join("; ")}</p>\` : "") +
      \`<div class="opts">\` + OPTIONS.map(([v, lbl]) =>
        \`<label><input type="radio" name="d\${i}" value="\${v}" />\${lbl}</label>\`).join("") + \`</div>\` +
      \`<input class="notes" id="n\${i}" placeholder="notes (optional)" />\` +
      \`<details><summary>stats</summary><code>p \${Number(c.base_p).toExponential(2)} → \${Number(c.smoothed_p).toExponential(2)} · soft \${c.evidence.soft} · src \${c.evidence.src} · \${(c.flags||[]).join(", ")}</code></details>\`;
    list.appendChild(div);
  });
  list.addEventListener("change", updateProgress);
  updateProgress();
}

async function save() {
  const status = document.getElementById("status");
  const overall = document.getElementById("overall").value;
  const summary = document.getElementById("summary").value.trim();
  const reviewer = document.getElementById("reviewer").value.trim() || null;
  if (!overall) { status.textContent = "Choose an overall decision."; return; }
  if (!summary) { status.textContent = "Add a one-line reason."; return; }
  const ratings = [];
  FEED.candidates.forEach((c, i) => {
    const sel = document.querySelector(\`input[name="d\${i}"]:checked\`);
    if (!sel) return;
    const notes = document.getElementById("n" + i).value.trim() || null;
    ratings.push({ slot_type: c.slot_type, tier: c.tier, filler: c.filler,
      base_p: c.base_p, smoothed_p: c.smoothed_p, delta: c.delta,
      evidence: c.evidence, decision: sel.value, notes });
  });
  if (!ratings.length) { status.textContent = "Rate at least one suggestion first."; return; }
  status.textContent = "Saving…";
  const payload = { run_id: FEED.run_id, variant: FEED.variant,
    vector_source: FEED.vector_source, reviewer, ratings,
    overall: { decision: overall, summary } };
  const res = await fetch("/api/save", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const out = await res.json();
  status.textContent = out.ok
    ? \`Saved \${ratings.length} ratings + your “\${overall}” decision.\`
    : \`Submission written but ingest failed: \${out.error}\`;
}

document.getElementById("save").addEventListener("click", save);
load();
</script>
</body>
</html>`;
}

async function startServer(log) {
    const server = createServer(async (req, res) => {
        try {
            if (req.url === "/api/feed") {
                const feed = await readFeed().catch(() => null);
                res.statusCode = feed ? 200 : 404;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(feed || { error: "no feed" }));
                return;
            }
            if (req.url === "/api/save" && req.method === "POST") {
                let body = "";
                for await (const chunk of req) body += chunk;
                const submission = JSON.parse(body);
                const result = await saveSubmission(submission, log);
                res.statusCode = result.ok ? 200 : 500;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(result));
                return;
            }
            res.setHeader("Content-Type", "text/html; charset=utf-8");
            res.end(renderHtml());
        } catch (err) {
            res.statusCode = 500;
            res.setHeader("Content-Type", "application/json");
            res.end(JSON.stringify({ ok: false, error: String(err) }));
        }
    });
    await new Promise((r) => server.listen(0, "127.0.0.1", r));
    const address = server.address();
    const port = typeof address === "object" && address ? address.port : 0;
    return { server, url: `http://127.0.0.1:${port}/` };
}

const session = await joinSession({
    canvases: [
        createCanvas({
            id: "step05-smoothing-review",
            displayName: "Step 5 smoothing review",
            description: "Rate semantic-smoothing candidate moves and record the accept/reject/iterate decision.",
            actions: [
                {
                    name: "get_feed_summary",
                    description: "Return the current review feed's variant, run_id, and candidate count.",
                    handler: async () => {
                        const feed = await readFeed().catch(() => null);
                        if (!feed) return { ok: false, error: "no feed at " + FEED_PATH };
                        return {
                            ok: true,
                            variant: feed.variant,
                            run_id: feed.run_id,
                            candidate_count: feed.candidate_count,
                        };
                    },
                },
            ],
            open: async (ctx) => {
                let entry = servers.get(ctx.instanceId);
                if (!entry) {
                    entry = await startServer((m, o) => session.log(m, o));
                    servers.set(ctx.instanceId, entry);
                }
                return { title: "Step 5 smoothing review", url: entry.url };
            },
            onClose: async (ctx) => {
                const entry = servers.get(ctx.instanceId);
                if (entry) {
                    servers.delete(ctx.instanceId);
                    await new Promise((r) => entry.server.close(() => r()));
                }
            },
        }),
    ],
});
