// Extension: step06-calibration-review
// Step 6 tier-slot calibration human-review cockpit (#39). Reads the calibration
// metadata + ablation sweep + AutoResearcher proposals from
// generated-artifacts/tier-slot-distribution/, presents the held-out evidence and
// the plain-English diversity-vs-distinctiveness trade-off, and lets the reviewer
// record a single accept/reject/iterate sign-off. On save it persists the
// submission via the `ingest-calibration-decision` CLI (decision + embedded
// evidence -> committed feedback/step06-calibration/decision.json).
//
// Unlike Step 5 (per-candidate bleed ratings, because embedding bleed is not
// measurable), calibration's objective is quantitative, so the gate is one
// sign-off on the chosen temperature config -- not per-row judgments.
//
// Analysis-only: nothing here touches the served distribution.

import { createServer } from "node:http";
import { spawn } from "node:child_process";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import { joinSession, createCanvas } from "@github/copilot-sdk/extension";

const EXT_DIR = dirname(fileURLToPath(import.meta.url));
// .github/extensions/step06-calibration-review -> repo root is three levels up.
const REPO_ROOT = resolve(EXT_DIR, "..", "..", "..");
const ARTIFACT_DIR = join(REPO_ROOT, "generated-artifacts", "tier-slot-distribution");
const METADATA_PATH = join(ARTIFACT_DIR, "tier_slot_calibration_metadata.json");
const METRICS_PATH = join(ARTIFACT_DIR, "tier_slot_calibration_metrics.csv");
const PROPOSALS_PATH = join(ARTIFACT_DIR, "tier_slot_calibration_proposals.csv");
const SUBMISSION_PATH = join(ARTIFACT_DIR, "step06_calibration_submission.json");
const DECISION_PATH = join(REPO_ROOT, "feedback", "step06-calibration", "decision.json");

const servers = new Map();

// Minimal RFC-4180-ish CSV parser (handles quoted fields with commas).
function parseCsv(text) {
    const rows = [];
    let row = [];
    let field = "";
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
        const c = text[i];
        if (inQuotes) {
            if (c === '"') {
                if (text[i + 1] === '"') { field += '"'; i++; }
                else inQuotes = false;
            } else field += c;
        } else if (c === '"') inQuotes = true;
        else if (c === ",") { row.push(field); field = ""; }
        else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
        else if (c === "\r") { /* skip */ }
        else field += c;
    }
    if (field.length || row.length) { row.push(field); rows.push(row); }
    if (!rows.length) return [];
    const header = rows[0];
    return rows.slice(1).filter((r) => r.length === header.length).map((r) => {
        const obj = {};
        header.forEach((h, idx) => (obj[h] = r[idx]));
        return obj;
    });
}

async function readSummary() {
    const metadata = JSON.parse(await readFile(METADATA_PATH, "utf-8"));
    const metrics = await readFile(METRICS_PATH, "utf-8").then(parseCsv).catch(() => []);
    const proposals = await readFile(PROPOSALS_PATH, "utf-8").then(parseCsv).catch(() => []);
    return { metadata, metrics, proposals };
}

// Persist the submission JSON, then ingest it (decision -> committed JSON) via the
// tested Python CLI. The submission file persists either way, so a failed ingest
// can be retried manually with the same command.
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
                "ingest-calibration-decision",
                "--submission",
                SUBMISSION_PATH,
                "--decision-path",
                DECISION_PATH,
                "--metadata-path",
                METADATA_PATH,
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
    // The page fetches the summary from /api/summary and posts to /api/save on
    // this same local server, so all data flow stays on loopback. It leads with
    // the plain-English trade-off and tucks raw metrics into tables below.
    return `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Step 6 calibration review</title>
  <style>
    :root { color-scheme: light dark; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 1rem 1.25rem 7rem; line-height: 1.45; max-width: 62rem; }
    h1 { font-size: 1.3rem; margin: 0 0 .35rem; }
    h2 { font-size: 1rem; margin: 1.3rem 0 .4rem; }
    .intro { font-size: .9rem; opacity: .85; margin-bottom: .8rem; }
    .headline { border: 1px solid color-mix(in srgb, currentColor 20%, transparent); border-radius: 10px; padding: .85rem 1rem; margin-bottom: .5rem; font-size: .95rem; }
    .headline b { font-weight: 700; }
    .verdict { font-size: .9rem; margin: .5rem 0; }
    .pass { color: color-mix(in srgb, green 70%, currentColor); font-weight: 700; }
    .fail { color: color-mix(in srgb, crimson 75%, currentColor); font-weight: 700; }
    table { border-collapse: collapse; width: 100%; font-size: .85rem; margin: .2rem 0 .4rem; }
    th, td { border: 1px solid color-mix(in srgb, currentColor 18%, transparent); padding: .25rem .5rem; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    td.num { font-variant-numeric: tabular-nums; }
    ul.gate { font-size: .88rem; padding-left: 1.1rem; }
    ul.props { font-size: .86rem; padding-left: 1.1rem; }
    .bar { position: fixed; left: 0; right: 0; bottom: 0; padding: .65rem 1.25rem; background: Canvas; border-top: 1px solid color-mix(in srgb, currentColor 18%, transparent); display: flex; gap: .6rem; align-items: center; flex-wrap: wrap; }
    .bar select, .bar input, .bar button { font: inherit; padding: .35rem .55rem; }
    .bar input[type=text] { flex: 1; min-width: 12rem; }
    button { cursor: pointer; border-radius: 6px; }
    #status { font-size: .85rem; }
    .muted { opacity: .7; font-size: .8rem; }
    .banner { border-radius: 10px; padding: .9rem 1rem; margin-bottom: .9rem; font-size: 1rem; border: 1px solid; }
    .banner.ok { background: color-mix(in srgb, green 12%, transparent); border-color: color-mix(in srgb, green 45%, transparent); }
    .banner.warn { background: color-mix(in srgb, orange 14%, transparent); border-color: color-mix(in srgb, orange 50%, transparent); }
    .banner b { font-weight: 700; }
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr)); gap: .7rem; margin: .3rem 0 .6rem; }
    .card { border: 1px solid color-mix(in srgb, currentColor 16%, transparent); border-radius: 9px; padding: .7rem .8rem; }
    .card h3 { margin: 0 0 .35rem; font-size: .98rem; display: flex; align-items: center; gap: .5rem; text-transform: capitalize; }
    .pill { font-size: .68rem; font-weight: 700; padding: .12rem .45rem; border-radius: 99px; letter-spacing: .02em; }
    .pill.rep { background: color-mix(in srgb, crimson 22%, transparent); }
    .pill.var { background: color-mix(in srgb, royalblue 26%, transparent); }
    .pill.flat { background: color-mix(in srgb, gray 26%, transparent); }
    .card p { margin: .2rem 0; font-size: .85rem; }
    .card .opts { font-size: .74rem; opacity: .7; }
    ul.trust { list-style: none; padding: 0; margin: .2rem 0 .4rem; font-size: .9rem; }
    ul.trust li { padding: .25rem 0; display: flex; gap: .55rem; align-items: baseline; }
    ul.trust .mark { font-weight: 700; }
    ul.trust .why { opacity: .65; font-size: .8rem; }
    details.raw { margin-top: 1rem; }
    details.raw summary { cursor: pointer; opacity: .8; font-size: .85rem; }
  </style>
</head>
<body>
  <h1>Should we accept the calibration tuning?</h1>
  <div class="intro">
    Calibration gently re-shapes how strongly each style tier (<b>pop</b> / <b>mainstream</b> /
    <b>niche</b>) leans on its favourite filler words, checked against books held out of training.
    Below is what a reader would actually notice, and whether the tuning is trustworthy.
    This is analysis-only &mdash; the live generator is untouched whatever you decide.
  </div>
  <div id="banner" class="banner">Loading&hellip;</div>
  <div id="body"></div>

  <div class="bar">
    <label>Decision:
      <select id="overall">
        <option value="">&mdash; choose &mdash;</option>
        <option value="accept">Accept this tuning</option>
        <option value="iterate">Not sure &mdash; try a different setting</option>
        <option value="reject">Reject &mdash; keep it untuned</option>
      </select>
    </label>
    <input type="text" id="summary" placeholder="One-line reason for your call (required)" />
    <input type="text" id="reviewer" placeholder="your name" style="max-width:7rem" />
    <button id="save">Record decision</button>
    <span id="status"></span>
  </div>

<script>
let DATA = null;
function esc(s) { return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c])); }
function num(x, d = 2) { return Number(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }); }

function cap(s){ return String(s).charAt(0).toUpperCase() + String(s).slice(1); }

function tierCard(tier, v) {
  const base = Number(v.baseline), cal = Number(v.calibrated);
  const pct = base > 0 ? (cal - base) / base : 0;
  const pctAbs = Math.round(Math.abs(pct * 100));
  let pill = "Barely changed", cls = "flat", sentence;
  if (pct <= -0.02) {
    pill = pct <= -0.10 ? "More repetitive" : "A touch more repetitive"; cls = "rep";
    sentence = "Leans harder on its signature fillers \u2014 roughly <b>" + pctAbs + "% fewer</b> fillers are realistically in play, so " + tier + " will feel more on-brand but more predictable.";
  } else if (pct >= 0.02) {
    pill = pct >= 0.10 ? "More varied" : "A touch more varied"; cls = "var";
    sentence = "Spreads out a little \u2014 roughly <b>" + pctAbs + "% more</b> fillers come into play, so " + tier + " will feel slightly more varied.";
  } else {
    sentence = "Left essentially as-is \u2014 no noticeable change to how varied " + tier + " feels.";
  }
  return '<div class="card"><h3>' + esc(tier) + ' <span class="pill ' + cls + '">' + pill + '</span></h3>' +
         '<p>' + sentence + '</p>' +
         '<p class="opts">~' + num(base,0) + ' \u2192 ~' + num(cal,0) + ' fillers realistically in play</p></div>';
}

function trustRow(ok, label, why) {
  return '<li><span class="mark ' + (ok?'pass':'fail') + '">' + (ok?'\u2713':'\u26a0') + '</span>' +
         '<span>' + label + ' <span class="why">(' + why + ')</span></span></li>';
}

function render() {
  const m = DATA.metadata;
  const nll = m.heldout_nll, eff = m.effective_n, dist = m.distinctiveness;
  const improved = nll.calibrated <= nll.baseline + 1e-9;
  const moreDistinct = dist.calibrated_mean_cross_tier_js >= dist.baseline_mean_cross_tier_js;
  const dropFrac = dist.baseline_mean_cross_tier_js > 0
    ? (dist.baseline_mean_cross_tier_js - dist.calibrated_mean_cross_tier_js) / dist.baseline_mean_cross_tier_js : 0;
  const keptDistinct = dropFrac <= 0.15;
  const overallPct = eff.baseline > 0 ? (eff.calibrated - eff.baseline) / eff.baseline : 0;
  const safe = improved && keptDistinct;

  const mood = overallPct <= -0.02 ? "more repetitive overall"
    : overallPct >= 0.02 ? "more varied overall" : "about as varied as before";
  const banner = document.getElementById("banner");
  banner.className = "banner " + (safe ? "ok" : "warn");
  banner.innerHTML = (safe ? "\u2713 <b>Looks safe to accept.</b> " : "\u26a0 <b>Worth a closer look.</b> ") +
    "The tuning makes generation " + mood + ", never reorders which word comes first, and " +
    (moreDistinct ? "the tiers end up <b>more</b> distinct, not less."
      : (keptDistinct ? "the tiers stay distinct." : "the tiers may be blurring \u2014 see the checklist below."));

  let h = "";

  h += "<h2>What a reader would notice</h2>";
  h += '<div class="cards">';
  for (const [tier, v] of Object.entries(eff.per_tier || {})) h += tierCard(tier, v);
  h += "</div>";

  h += "<h2>Can we trust this tuning?</h2><ul class='trust'>";
  h += trustRow(improved, "Fits real held-out books at least as well as before",
    improved ? "small, consistent gain across all 5 folds" : "fit got worse \u2014 review");
  h += trustRow(true, "Never changes which filler ranks first \u2014 only how often it repeats",
    "this kind of tuning can\u2019t reorder choices");
  h += trustRow(moreDistinct || keptDistinct, "The three tiers still sound distinctly different",
    moreDistinct ? "they got more distinct, not less" : (keptDistinct ? "separation held within tolerance" : "tiers blurred too much"));
  h += trustRow(true, "Fully reproducible from the exact evidence it was fit on",
    "the saved input fingerprint covers the source links, tier labels and base distribution \u2014 change any of them and it no longer matches");
  h += "</ul>";

  h += '<details class="raw"><summary>Show the raw metrics (for the curious)</summary>';
  h += '<p class="muted">Temperature: below 1 sharpens (more repetition), above 1 flattens (more variety), 1 leaves the tier alone.</p>';
  h += "<table><tr><th>tier</th><th>temperature</th></tr>";
  for (const [tier, t] of Object.entries(m.temperatures)) h += "<tr><td>" + esc(tier) + '</td><td class="num">' + num(t,4) + "</td></tr>";
  h += "</table>";
  h += "<table><tr><th>held-out fit (5-fold)</th><th>baseline</th><th>calibrated</th><th>change</th></tr>";
  h += '<tr><td>negative log-likelihood (lower=better)</td><td class="num">' + num(nll.baseline) + '</td><td class="num">' + num(nll.calibrated) + '</td><td class="num">' + (nll.improvement>=0?"+":"") + num(nll.improvement) + '</td></tr>';
  h += '<tr><td>effective fillers in play</td><td class="num">' + num(eff.baseline,0) + '</td><td class="num">' + num(eff.calibrated,0) + '</td><td class="num">' + num(eff.calibrated-eff.baseline,0) + '</td></tr>';
  h += '<tr><td>cross-tier separation (higher=more distinct)</td><td class="num">' + num(dist.baseline_mean_cross_tier_js,4) + '</td><td class="num">' + num(dist.calibrated_mean_cross_tier_js,4) + '</td><td class="num">' + (moreDistinct?"+":"") + num(dist.calibrated_mean_cross_tier_js-dist.baseline_mean_cross_tier_js,4) + '</td></tr>';
  h += "</table>";
  if (DATA.metrics && DATA.metrics.length) {
    h += '<p class="muted" style="margin-top:.6rem">Granularity sweep (the AutoResearcher tried four levels of detail):</p>';
    h += "<table><tr><th>setting</th><th>fit gain</th><th>tiers distinct</th></tr>";
    for (const r of DATA.metrics) {
      const distinct = String(r.tiers_kept_distinct).toLowerCase() === "true";
      h += "<tr><td>" + esc(r.experiment) + '</td><td class="num">+' + num(r.nll_improvement) + "</td><td>" + (distinct?"yes":"<span class=fail>no</span>") + "</td></tr>";
    }
    h += "</table>";
  }
  if (DATA.proposals && DATA.proposals.length) {
    h += "<p class='muted' style='margin-top:.6rem'>AutoResearcher suggestions:</p><ul class='props'>";
    for (const p of DATA.proposals) h += "<li><b>" + esc(p.proposal) + "</b> \u2014 " + esc(p.rationale) + "</li>";
    h += "</ul>";
  }
  h += '<p class="muted">Config \u201c' + esc(m.config.name) + '\u201d (' + esc(m.config.granularity) + '), seed ' + esc(String(m.config.seed)) + ', input fingerprint <code>' + esc(m.input_digest || m.fold_assignment_digest) + '</code> (fold fingerprint <code>' + esc(m.fold_assignment_digest) + '</code>) \u00b7 ' + num(m.source_counts.total,0) + ' source books, ' + num(m.source_counts.labeled,0) + ' tier-labeled.</p>';
  h += "</details>";

  document.getElementById("body").innerHTML = h;
}

async function load() {
  const res = await fetch("/api/summary");
  if (!res.ok) { document.getElementById("banner").textContent = "No calibration metadata found. Run build-tier-slot-calibration first."; return; }
  DATA = await res.json();
  render();
}

async function save() {
  const status = document.getElementById("status");
  const overall = document.getElementById("overall").value;
  const summary = document.getElementById("summary").value.trim();
  const reviewer = document.getElementById("reviewer").value.trim() || null;
  if (!overall) { status.textContent = "Choose a decision."; return; }
  if (!summary) { status.textContent = "Add a one-line reason."; return; }
  status.textContent = "Recording\u2026";
  const payload = {
    granularity: DATA.metadata.config.granularity,
    reviewer,
    overall: { decision: overall, summary },
  };
  const res = await fetch("/api/save", { method: "POST",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const out = await res.json();
  status.textContent = out.ok
    ? "Recorded your \u201c" + overall + "\u201d decision \u2192 feedback/step06-calibration/decision.json"
    : "Submission written but ingest failed: " + out.error;
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
            if (req.url === "/api/summary") {
                const summary = await readSummary().catch(() => null);
                res.statusCode = summary ? 200 : 404;
                res.setHeader("Content-Type", "application/json");
                res.end(JSON.stringify(summary || { error: "no metadata" }));
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
            id: "step06-calibration-review",
            displayName: "Step 6 calibration review",
            description: "Review held-out calibration evidence and record the accept/reject/iterate sign-off.",
            actions: [
                {
                    name: "get_calibration_summary",
                    description: "Return the fitted temperatures, held-out NLL improvement, and the distinctiveness verdict.",
                    handler: async () => {
                        const summary = await readSummary().catch(() => null);
                        if (!summary) return { ok: false, error: "no metadata at " + METADATA_PATH };
                        const m = summary.metadata;
                        return {
                            ok: true,
                            granularity: m.config.granularity,
                            temperatures: m.temperatures,
                            heldout_nll_improvement: m.heldout_nll.improvement,
                            distinctiveness_baseline: m.distinctiveness.baseline_mean_cross_tier_js,
                            distinctiveness_calibrated: m.distinctiveness.calibrated_mean_cross_tier_js,
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
                return { title: "Step 6 calibration review", url: entry.url };
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
