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
    // same local server, so all data flow stays on loopback.
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Step 5 smoothing review</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem 6rem; line-height: 1.4; }
    h1 { font-size: 1.2rem; margin: 0 0 .25rem; }
    .meta { opacity: .7; font-size: .85rem; margin-bottom: 1rem; }
    .cand { border: 1px solid color-mix(in srgb, currentColor 18%, transparent); border-radius: 8px; padding: .65rem .8rem; margin-bottom: .7rem; }
    .cand h3 { font-size: .95rem; margin: 0 0 .3rem; }
    .move { font-variant-numeric: tabular-nums; opacity: .85; font-size: .85rem; }
    .flags { font-size: .72rem; opacity: .7; }
    .nbrs { font-size: .78rem; opacity: .8; margin: .35rem 0; padding-left: 1rem; }
    .opts { display: flex; gap: .35rem; flex-wrap: wrap; margin-top: .4rem; }
    .opts label { border: 1px solid color-mix(in srgb, currentColor 25%, transparent); border-radius: 999px; padding: .12rem .55rem; font-size: .8rem; cursor: pointer; }
    .opts input { margin-right: .3rem; }
    .notes { width: 100%; margin-top: .35rem; font: inherit; padding: .25rem .4rem; box-sizing: border-box; }
    .bar { position: fixed; left: 0; right: 0; bottom: 0; padding: .6rem 1.25rem; background: Canvas; border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
    .bar select, .bar input, .bar button { font: inherit; padding: .3rem .5rem; }
    .bar input[type=text] { flex: 1; min-width: 12rem; }
    button { cursor: pointer; border-radius: 6px; }
    #status { font-size: .82rem; }
  </style>
</head>
<body>
  <h1>Step 5 semantic-smoothing review</h1>
  <div class="meta" id="meta">Loading feed…</div>
  <div id="list"></div>

  <div class="bar">
    <label>Overall:
      <select id="overall">
        <option value="">— choose —</option>
        <option value="accept">accept</option>
        <option value="iterate">iterate</option>
        <option value="reject">reject</option>
      </select>
    </label>
    <input type="text" id="summary" placeholder="One-line rationale for the overall decision (required)" />
    <input type="text" id="reviewer" placeholder="reviewer" style="max-width:8rem" />
    <button id="save">Save review</button>
    <span id="status"></span>
  </div>

<script>
const DECISIONS = ["plausible_repair", "semantic_bleed", "too_generic", "needs_context"];
let FEED = null;

function fmt(n) { return Number(n).toFixed(6); }

async function load() {
  const res = await fetch("/api/feed");
  if (!res.ok) { document.getElementById("meta").textContent = "No feed found. Run build-smoothing-review-feed first."; return; }
  FEED = await res.json();
  document.getElementById("meta").textContent =
    \`variant \${FEED.variant} · run_id \${FEED.run_id} · \${FEED.candidate_count} candidates · vectors \${FEED.vector_source}\`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  FEED.candidates.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "cand";
    const nbrs = (c.nearest_contributors || []).map(n =>
      \`\${n.display_filler} (sim \${n.similarity}, p \${n.p}, src \${n.src})\`).join("; ");
    div.innerHTML =
      \`<h3>\${c.slot_type} · <b>\${c.tier}</b> · \${c.display_filler}</h3>\` +
      \`<div class="move">\${fmt(c.base_p)} → \${fmt(c.smoothed_p)} (Δ +\${fmt(c.delta)}) · soft \${c.evidence.soft} · src \${c.evidence.src}</div>\` +
      \`<div class="flags">\${(c.flags||[]).join(" · ")}</div>\` +
      (nbrs ? \`<div class="nbrs">← \${nbrs}</div>\` : "") +
      \`<div class="opts">\` + DECISIONS.map(d =>
        \`<label><input type="radio" name="d\${i}" value="\${d}" />\${d}</label>\`).join("") + \`</div>\` +
      \`<input class="notes" id="n\${i}" placeholder="notes (optional)" />\`;
    list.appendChild(div);
  });
}

async function save() {
  const status = document.getElementById("status");
  const overall = document.getElementById("overall").value;
  const summary = document.getElementById("summary").value.trim();
  const reviewer = document.getElementById("reviewer").value.trim() || null;
  if (!overall) { status.textContent = "Choose an overall decision."; return; }
  if (!summary) { status.textContent = "Add a one-line rationale."; return; }
  const ratings = [];
  FEED.candidates.forEach((c, i) => {
    const sel = document.querySelector(\`input[name="d\${i}"]:checked\`);
    if (!sel) return;
    const notes = document.getElementById("n" + i).value.trim() || null;
    ratings.push({ slot_type: c.slot_type, tier: c.tier, filler: c.filler,
      base_p: c.base_p, smoothed_p: c.smoothed_p, delta: c.delta,
      evidence: c.evidence, decision: sel.value, notes });
  });
  status.textContent = "Saving…";
  const payload = { run_id: FEED.run_id, variant: FEED.variant,
    vector_source: FEED.vector_source, reviewer, ratings,
    overall: { decision: overall, summary } };
  const res = await fetch("/api/save", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const out = await res.json();
  status.textContent = out.ok
    ? \`Saved \${ratings.length} ratings + decision (\${overall}).\`
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
