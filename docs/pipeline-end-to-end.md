# End-to-end pipeline walkthrough

This walkthrough explains how subtitle-generator moves from raw catalog data to
runtime generation, browser verification, and deployment. It is the operational
companion to `subtitle-gen validate-pipeline` and the schema contracts in
`src\subtitle_generator`.

## Core idea

The project now has two separate runtime responsibilities:

| Responsibility | Owner | Runtime effect |
|---|---|---|
| **Filler universe** | `build-slots`, popularity enrichment, remix precompute | Decides which real source-derived slot fillers are valid at all. |
| **Tier categorization** | Offline book-model artifacts and `slot_filler_model_scores` | Scores each existing filler for pop/mainstream/niche generation. |
| **Generation** | `generate.py` | Samples existing fillers using frequency plus the requested tier score. |
| **Literal guardrail** | `generate.py` | Filters broken artifacts without rejecting funny absurdity. |

The generator does not invent new slot fillers. It recombines validated real
pieces. The learned model changes which pieces surface for a requested tier; it
does not add new text to the pool.

## What changed in the learned-tier rewrite

The old runtime tiering path used frequency, popularity, and tone-target
heuristics directly. During the ML rewrite, an intermediate scalar
`book_model_score` framing was tried and rejected: it compressed pop,
mainstream, and niche behavior onto a single line and then blended that scalar
back into the legacy popularity path. That was not the desired behavior.

The final design is pure categorization:

| Piece | Final behavior |
|---|---|
| `score_pop` | Probability-like score that a filler belongs in pop-accessible generation. |
| `score_mainstream` | Probability-like score for mainstream generation. |
| `score_niche` | Probability-like score for niche generation. |
| Requested tier `T` | Generation weights existing fillers roughly as `sqrt(freq) * score_T`. |

The model classifies existing fillers; it does not generate or approve new text.
This keeps the strict filler universe, source attribution, article handling, and
remix safety gates intact while letting the requested tier choose a different
slice of that same universe.

The rewrite also narrowed the quality guardrail. "Funny nonsense" such as
`Emissions Trading` or `Second Indochina War` is allowed when it is grammatical
and usable as nonfiction-title absurdity. Only literal artifacts are blocked,
such as bare adjective-shaped final objects (`Christian`), typo/acronym
artifacts (`Imf`), run-together initials (`H.G.W.ells`), suffix artifacts
(`Con Men, Jr`), and known standalone artifacts (`Xcalibur`).

## Runtime data flow

```text
raw records
  -> source candidates
  -> validated pattern matches
  -> slot fillers
  -> popularity/remix/model-score enrichment
  -> mini DB
  -> CLI, local web, Azure Functions
```

The most important runtime tables are:

| Table | Purpose |
|---|---|
| `subtitles` | Raw and normalized source-title/subtitle rows in the full local DB. |
| `pattern_matches` | Clean parsed source candidates that passed slot validation. |
| `slot_fillers` | Canonical runtime filler inventory, frequency, popularity, article, and remix state. |
| `slot_filler_sources` | Source attribution from fillers back to source books. |
| `slot_filler_model_scores` | Learned pop/mainstream/niche probabilities for each deployed filler. |
| `config` | Runtime tuning values, article thresholds, remix constants, and generation ratios. |

Only a subset is deployed. The mini DB contains the runtime-facing state:
`slot_fillers`, `slot_filler_model_scores`, `sources`, and `config`.

## Full rebuild sequence

Run this when rebuilding the corpus from raw third-party data:

```powershell
uv run subtitle-gen download --parts all
uv run subtitle-gen extract
uv run subtitle-gen download-ol
uv run subtitle-gen extract-ol
uv run subtitle-gen build-slots
uv run subtitle-gen download-popularity
uv run subtitle-gen populate-popularity
uv run subtitle-gen precompute-vectors
uv run subtitle-gen validate-pipeline
```

`validate-pipeline` is read-only. It checks schema, required config values,
popularity coverage, remix precompute state, model IDs, and serving readiness.

## Source ingestion

`extract` and `extract-ol` normalize Library of Congress and Open Library records
into `subtitles`.

Important columns:

| Column | Meaning |
|---|---|
| `title` | Display title from the source record. |
| `subtitle` | Display subtitle from the source record; empty for title-derived candidates. |
| `candidate_text` | The text `build-slots` parses. |
| `candidate_source` | `subtitle` or `title`. |
| `isbn`, `lccn`, `source_file`, `lang` | Identifier and provenance metadata. |

Extraction repairs obvious title/subtitle duplication when it can recover a real
split. Rows that cannot be repaired are rejected before they can become runtime
fillers.

## Slot extraction

`build-slots` looks for:

```text
X, Y, and [the/a/an] Z of [the/a/an] W
```

The regex match is only a candidate. `slots.py` then applies stricter gates:

| Gate | Behavior |
|---|---|
| List shape | Accept exactly two or three source list clauses; generated output still uses two. |
| List item validation | Reject the whole candidate if any original list item is malformed. |
| Action noun validation | Reject weak, truncated, or implausible action nouns. |
| Of-object validation | Reject malformed final objects and bad starts such as `using`, `with`, or `for`. |
| Source repair | Preserve title-only source display for title-derived candidates. |

Accepted source candidates create:

| Output | Meaning |
|---|---|
| `pattern_matches` | Parsed source rows with list items, action noun, of-object, and observed articles. |
| `slot_fillers` | Deduplicated strict fillers by slot type. |
| `slot_filler_sources` | Links from fillers to source subtitles/books. |

## Popularity enrichment

Popularity is still useful as source evidence and as a fallback signal, but it is
no longer the primary deployed tier classifier when model scores are present.

`populate-popularity` maps source data to Open Library work keys, combines demand
signals, and pushes work-level scores down to fillers:

| Source | Signal |
|---|---|
| Seattle Public Library | Checkout demand. |
| Goodreads / UCSD Book Graph | Rating count and engagement. |
| NYT | Bestseller appearances. |
| Ottawa / library data | Library demand and holdings. |
| Trove Australia | Library breadth via `holdingsCount`. |
| Open Library edition count | Prior/confidence signal, not a direct popularity vote. |

The resulting filler columns remain:

| Column | Meaning |
|---|---|
| `popularity_score` | Historical/fallback source popularity score. |
| `popularity_level` | Coarse availability of source popularity evidence. |
| `popularity_confidence` | Confidence in the popularity score. |

## Remix precompute

`precompute-vectors` prepares runtime-safe remix information for multi-word
of-objects. It stores scalar data so serving can evaluate remix coherence without
loading spaCy or full vectors.

Key fields:

| Field | Meaning |
|---|---|
| `remix_type` | Type 1 compound noun phrase or Type 2 prepositional phrase. |
| `remix_prep` | Preposition for Type 2 remixes. |
| `remix_word_count` | Original phrase length. |
| `centroid_dot`, `norm_sq`, `token_count` | Scalar vector approximation fields. |
| `centroid_norm`, `avg_cross_sim_t1`, `avg_cross_sim_t2` | Config constants for runtime similarity approximation. |

Runtime remixing composes candidate parts and rejects low-similarity combinations
with the scalar approximation. It does not load build-time NLP dependencies.

## Book-model tier categorization

The learned tier model is offline-only. It consumes source-book features and
exports runtime scores for existing fillers.

Typical flow:

```powershell
uv sync --extra ml --extra tune

# Optional offline enrichment from Open Library and LOC raw data.
uv run subtitle-gen build-book-metadata

# Build source-book features and labels.
uv run subtitle-gen build-book-features

# Train local models and distill exportable students.
uv run subtitle-gen train-book-model
uv run subtitle-gen train-book-model-torch
uv run subtitle-gen distill-book-model

# Roll student predictions up to slot fillers.
uv run subtitle-gen shadow-book-model

# Optional: generate fixed samples for human/ad hoc LLM review.
uv run subtitle-gen categorization-gate --dry-run

# Install the selected export-slot rollup into the full local DB.
uv run subtitle-gen install-book-model-scores `
  --input generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv
```

`scripts\run-book-model-pipeline.ps1` wraps the same commands in step-gated
PowerShell. It defaults to the safe inventory step; expensive training,
installation, export, mini-DB build, validation, and optional review sampling
are explicit steps or switches:

```powershell
pwsh -File scripts\run-book-model-pipeline.ps1 -Steps Inventory
pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps Features,Torch,Distill,Shadow,CategorizationGate `
  -PlanOnly
pwsh -File scripts\run-book-model-pipeline.ps1 `
  -Steps CategorizationGate `
  -ReviewGates
```

The review gates are optional sanity-check tooling. They do not grade the real
books, train the model, compute deployed weights, or block export by themselves.
The core weight-production path is labeled source books -> features/labels ->
train/distill -> shadow rollups -> install/export scores.

The two optional review gates answer different questions:

| Command | Purpose | Status |
|---|---|---|
| `deployment-gate` | Legacy scalar/blend strategy comparison from the rejected intermediate design. | Kept for historical comparison and regression checks. |
| `categorization-gate` | Pure per-tier categorization comparison against legacy sampling for requested pop/mainstream/niche tiers. | Preferred gate for learned-tier runtime changes. |

The installed table is:

| Column | Meaning |
|---|---|
| `slot_filler_id` | References `slot_fillers.id`. |
| `score_pop` | Learned probability-like score for pop generation. |
| `score_mainstream` | Learned probability-like score for mainstream generation. |
| `score_niche` | Learned probability-like score for niche generation. |
| `model_tier` | Highest-scoring tier. |
| `source_prediction_count` | Number of source predictions contributing to the rollup. |

`install-book-model-scores` replaces the table atomically. If the CSV is missing
required columns or contains invalid rows, the old score table is left intact.

The rollup CSV from `shadow-book-model`, especially
`generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv`,
is the source for installed runtime scores. The generated reports, predictions,
and gate outputs under `generated-artifacts\book-model\` are local review
artifacts and are intentionally not deployment inputs.

## Export and mini DB

After rebuilding slot, popularity, remix, or model-score state, regenerate the
tracked deployment data:

```powershell
uv run subtitle-gen export-data -o api\data
uv run subtitle-gen build-db -d api\data -o api\data\subtitles.mini.db
```

Tracked deployment inputs:

| File | Purpose |
|---|---|
| `api\data\slot_fillers.csv` | Strict filler inventory and runtime scalar state. |
| `api\data\slot_filler_model_scores.csv` | Learned runtime tier probabilities. |
| `api\data\sources.csv` | Source-book attribution. |
| `api\data\config.csv` | Runtime configuration. |
| `api\data\source_tier_labels.csv` | Offline source-label evidence exported for training/evaluation continuity. |

Ignored local/CI-built artifacts:

| Path | Purpose |
|---|---|
| `api\data\subtitles.mini.db` | Built SQLite artifact for local serving and Azure Functions; CI rebuilds it from tracked CSVs in `.github\workflows\deploy.yml`. |
| `generated-artifacts\` | Local reports, features, predictions, rollups, and gate outputs. |

If `slot_filler_model_scores.csv` exists, `build-db` requires one score row for
every exported slot filler. Partial coverage is rejected so deployment cannot
silently mix learned-tier sampling with legacy popularity sampling.

## Runtime generation

Generation loads strict candidates for:

| Slot | Source |
|---|---|
| `list_item` | Two sampled list fillers. |
| `action_noun` | One sampled action noun. |
| `of_object` | One sampled final object, optionally remixed. |

When a requested tier has model scores, sampling uses:

```text
weight = sqrt(freq) * score_for_requested_tier
```

When model scores are unavailable, generation falls back to the legacy
frequency/popularity/tone-target weighting. This keeps local development and
older DBs usable, but deployment is expected to include complete model scores.

The literal-bad guardrail applies in both paths. It filters known broken artifact
shapes such as:

| Rejected shape | Why |
|---|---|
| bare final object `Christian` | Adjective-shaped incomplete object. |
| `Imf` | Acronym/typo artifact. |
| `H.G.W.ells` | Run-together initials artifact. |
| `Con Men, Jr` | Suffix artifact that does not parse as a final object. |

It does not reject valid absurdity such as `Emissions Trading` or `Second
Indochina War`.

## Runtime tier classification

`compute_tier_evidence()` classifies generated subtitles.

| DB state | Classifier behavior |
|---|---|
| Complete `slot_filler_model_scores` | Average per-slot learned tier probabilities and choose the highest tier with deterministic tie-breaking. |
| No model scores | Fall back to the legacy blended frequency/popularity score and calibrated thresholds. |

The classifier also returns compatibility evidence such as per-slot details,
accessibility score, lower-tail score, and demand confidence for UI/debugging.

## Serving surfaces

```text
CLI -> generate.py / jacket.py
local web -> serve.py -> handlers.py -> shared runtime modules
Azure Functions -> api\function_app.py -> handlers.py -> shared runtime modules
```

`handlers.py` is the shared API boundary for:

| Handler | Purpose |
|---|---|
| `handle_generate` | Generate a subtitle and return slot/source/remix metadata. |
| `handle_jacket` | Build a jacket prompt; local mode can also stream LLM output. |
| `handle_rate` | Store human feedback. |
| `handle_health` | Health and mode response. |

The frontend is intentionally thin. It renders API results, shows sources and
remix parts, captures feedback, and builds jacket prompts. Runtime decisions stay
server-side.

## Local browser verification

Install browser dependencies once:

```powershell
uv sync --extra deploy --extra tune --extra e2e
uv run playwright install --with-deps chromium
```

Run the local e2e gate:

```powershell
pwsh -File scripts\run-local-e2e.ps1
```

The script:

1. Starts `subtitle-gen serve --no-open` on `http://127.0.0.1:8742`.
2. Waits for readiness.
3. Captures `home-before.png`.
4. Runs `tests\test_e2e.py`.
5. Captures `home-after.png`.
6. Runs `tests\test_e2e_spot_check.py`.
7. Captures `spot-check-after.png`.
8. Stops the server and leaves logs/screenshots in `test-results\local-e2e`.

The home-page e2e covers generation, quality tags, sources, jacket prompt
building, settings, hard tier-filter regeneration, remix display, mobile layout,
and deployed App Insights telemetry. The spot-check e2e covers local-only batch
rating, keyboard shortcuts, tag toggles, skip flow, summary, and navigation.

## Deployment

Deployment is handled by GitHub Actions:

| Workflow | Trigger | Responsibilities |
|---|---|---|
| `.github\workflows\deploy-infra.yml` | Push to `infra/**` or manual dispatch | Deploy Azure resources, static website setup, monitoring, alerts, and optional RBAC role assignments. |
| `.github\workflows\deploy.yml` | Push to `master` touching `web/**`, `api/**`, `src/**`, or manual dispatch | Build the mini DB from CSVs, deploy Azure Functions, upload the static website, smoke-test the API, and run deployed e2e. |

Required GitHub configuration:

| Name | Type |
|---|---|
| `AZURE_CLIENT_ID` | Secret |
| `AZURE_TENANT_ID` | Secret |
| `AZURE_SUBSCRIPTION_ID` | Secret |
| `AZURE_FUNCTIONAPP_NAME` | Variable |
| `ALERT_EMAIL` | Optional secret or workflow input |

For the current production site, deployed e2e targets:

```text
https://subtitlegenst.z13.web.core.windows.net
```

## Feedback and tuning

Human feedback is stored, reviewed, and used to guide future work. It is not
immediately applied to runtime generation.

```powershell
uv run subtitle-gen review
uv run subtitle-gen generate --review
uv sync --extra deploy
uv run subtitle-gen pull-ratings --account subtitlegenst --since 2026-05-01
```

Source-title labels are separate from generated-subtitle ratings:

```powershell
uv sync --extra tune
uv run subtitle-gen classify-source-tiers --dry-run --limit 20
uv run subtitle-gen classify-source-tiers --limit 200 --batch-size 10
uv run subtitle-gen source-tier-distribution
```

The source-label workflow writes `pattern_matches.llm_market_tier*` and exports
`api\data\source_tier_labels.csv`. Those labels can feed offline training and
evaluation, but they are not read directly by the deployed mini DB.

## Quality gate before deployment

Run this before creating or merging a deployment PR:

```powershell
uv run subtitle-gen validate-pipeline
uv run ruff check
uv run ty check
uv run pytest -q
pwsh -File scripts\run-local-e2e.ps1
```

Review the diff, bump the package version, create a feature branch, open a PR,
wait for CI, then merge only after all checks are green.
