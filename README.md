# subtitle-generator

Generate bizarre book subtitles in the pop-nonfiction pattern — *"X, Y, and the Z of W"* — by mining real parts from the Library of Congress MARC database and Open Library, then recombining them slot-machine style. Supports article variants (*"and a/an Z"*) and articles before of-objects (*"of the W"*) based on corpus statistics.

Optionally generate a **full book jacket** with title, back cover copy, trade journal reviews, and endorsement blurbs from real people — powered by the GitHub Copilot SDK.

## Examples

**Random subtitles:**
```
Jefferson, Repression, and the Category of Scripture in Lurianic Kabbala
Greed, Grit, and the Rise of the American Dream
Faith, Hope, and a Healthy Dose of Laughter
Celebrity Culture, Theology, and the Collapse of New England
```

**Full book jacket** (with `--jacket`):

> **Holy Nation**
> *Professionals, Pagan Authors, and the Sacramental Vision of the Nation State*
>
> *Publishers Weekly* — "This compact, argument-driven study contends that modern political life cannot be understood apart from its spiritual assumptions..."
>
> *Ross Douthat* (NYT columnist) — "A sharp and unusually serious book about the truth everyone keeps trying to avoid..."

## How it works

1. **Download** ~25M MARC records from the LOC bulk distribution (43 files, ~9 GB) and/or ~35M edition records from Open Library (~9.2 GB)
2. **Extract** 11M+ English source rows into SQLite (with cross-source deduplication and repair of repeated title/subtitle corruption). Records can enter as subtitle-derived candidates or title-derived candidates when the title itself matches the subtitle pattern.
3. **Pattern match** candidate text matching "X, Y, and [the/a/an] Z of [the/a/an] W" using regex + spaCy NLP validation. Source subtitles may contribute either two or three list clauses; generated subtitles still use two list slots.
4. **Decompose** into typed slots: list items, action nouns, of-objects — plus sub-parts (modifiers, heads, prepositional complements) for remixing. Articles (a/an/the) are stripped and stored separately for re-insertion at generation time.
5. **Score and tune** fillers with popularity, tone, article, and remix parameters stored in SQLite config rows with Python contract views.
6. **Generate** by randomly drawing one filler per slot — weighted by corpus frequency, popularity, and tone targets. Multi-word of-objects can be remixed into novel combinations (e.g., "New York" + "kitsch" from different books)
7. **Jacket** (optional) — send the subtitle to an LLM (via Copilot SDK) to generate a full book jacket with trade journal reviews and endorsement blurbs from real people

## Setup

```bash
git clone https://github.com/Arithmomaniac/subtitle-generator.git
cd subtitle-generator
uv sync
```

## Pipeline

Run these in order to build the database from scratch:

```bash
uv run subtitle-gen download --parts all       # LOC MARC (~9 GB)
uv run subtitle-gen extract                     # parse into SQLite
uv run subtitle-gen download-ol                 # Open Library (~9.2 GB)
uv run subtitle-gen extract-ol                  # parse + deduplicate
uv run subtitle-gen build-slots                 # extract slot fillers
uv run subtitle-gen download-popularity         # SPL, Goodreads, Ottawa, Trove, etc.
uv run subtitle-gen populate-popularity         # compute composite scores
uv run subtitle-gen precompute-vectors          # remix scalar/vector state
uv run subtitle-gen validate-pipeline           # read-only readiness checks
uv run subtitle-gen export-data -o api\data     # write runtime CSV artifacts
uv run subtitle-gen build-db -d api\data -o api\data\subtitles.mini.db
pwsh -File scripts\run-local-e2e.ps1            # browser/API verification
```

### Pipeline contracts and validation

The pipeline has explicit contract modules for the cross-stage state that feeds generation and serving:

| Stage | Inputs | Outputs / contracts |
|---|---|---|
| Source ingestion | LOC MARC files and Open Library dumps | `subtitles` rows with ISBN/source metadata plus `candidate_text` / `candidate_source` provenance |
| Slot extraction | Title/subtitle pattern candidates plus spaCy validation | `pattern_matches` and strict `slot_fillers` candidates |
| Popularity scoring | SPL, Open Library, Goodreads, NYT, Ottawa/library, Trove, and corpus frequency signals | `popularity_data.composite_score`, filler `popularity_score`, and calibrated threshold config values |
| Remix precompute | Strict of-object fillers, spaCy vectors, article statistics | remix classifications, vector/scalar columns, embedding config keys |
| Tuning | Generated samples, human ratings, LLM ratings, and strict proposal schemas | accepted config changes, rollback-capable proposal records, and rating snapshots |
| Runtime/serving | Validated SQLite state and request parameters | `GeneratedSubtitle`, `subtitle_to_dict()`, CLI output, local HTTP, and Azure Functions payloads |

Use `uv run subtitle-gen validate-pipeline` before tuning, serving, or exporting. It is read-only and fails non-zero when required tables/columns, config values, remix precompute state, popularity coverage, model IDs, or serving handlers are not ready.

### Slot extraction quality gates

`build-slots` treats the broad regex match as a candidate, not as proof that a
source row is usable. It parses `subtitles.candidate_text`, while `subtitles.title`
and `subtitles.subtitle` remain source-display metadata. Title-derived rows keep
an empty source subtitle, so exports and runtime sources render as title-only
rather than `Title: Title`. A candidate is accepted only after these gates:

| Gate | Behavior |
|---|---|
| List shape | Accept exactly two or three original list clauses before the final `and the/a/an ... of ...` clause. Reject four or more clauses. |
| List item validation | Reject the whole candidate if any original list clause fails cleanup, artifact, weak/jargon, truncation, or spaCy noun/name validation. The pipeline no longer silently drops bad list items and keeps the rest. |
| Action/object validation | Reject weak action nouns, malformed of-objects, and SEO/prepositional object starts such as `using`, `with`, or `for`. |
| Title/subtitle repair | Repair common `Title: Subtitle` duplication before validation. `Title: Subtitle Title: Subtitle` and `Title: Subtitle Subtitle` are uncorrupted to `Title` / `Subtitle` when possible; unrepairable repeated rows are rejected. |

`pattern_matches` contains the clean NLP-validated matches that survived these
gates. Downstream source attribution, filler popularity, article statistics,
calibration, remix precompute, export, and serving should be rebuilt from that
clean set after slot-filter or candidate-ingestion changes.

Model and weight state is grouped by purpose rather than treated as one loose dictionary:

| Family | Examples |
|---|---|
| LLM models | rating model `github_copilot/gpt-5.4-mini`, proposal model `github_copilot/gpt-5.4`, jacket model `gpt-5.4-mini`, responses-only model family |
| Sampling and tone | weighted sample spread, bias floor, tone targets, tier centers, accessibility thresholds |
| Popularity | source weights for SPL/Open Library/Goodreads/NYT/library/Trove/frequency, exponent, blend defaults, slot multipliers |
| Article and remix | article frequency thresholds, remix heuristic threshold, double-`of` rejection toggle, calibrated remix probability and similarity thresholds |

## Usage

### CLI

```bash
uv run subtitle-gen generate                    # 10 random subtitles
uv run subtitle-gen generate --sources          # show source books
uv run subtitle-gen generate --tone pop         # bias toward accessible
uv run subtitle-gen generate --jacket           # subtitle + full jacket
uv run subtitle-gen jacket "sturgeon, caviar, and the geography of desire"
```

Run `subtitle-gen <command> --help` for full options on any command.

Trove Australia is an optional API-keyed popularity source. Set
`TROVE_API_KEY` or pass `--trove-api-key`, then run a small resumable sample
with `uv run subtitle-gen download-popularity --sources trove --trove-limit 100`.
For rebuilds after slot-source changes, use `--trove-target-mode slot-sources`
so Trove only refreshes ISBNs attached to current strict valid sources.
The lookup stores `holdingsCount` as Australian library breadth; physical copy
counts are marked as proxies unless Trove exposes exact copy fields.

### Web app

**Live demo:** [subtitlegenst.z13.web.core.windows.net](https://subtitlegenst.z13.web.core.windows.net/)

```bash
uv run subtitle-gen serve                       # start on localhost:8742
```

The web app provides an interactive UI with:
- Tone selection and settings panel
- Color-coded slot display with remix sub-parts
- Jacket generation with live progress streaming
- Rendered markdown output with Copy Markdown / Copy HTML buttons
- Dynamic model picker (queries available Copilot SDK models)

The frontend is a thin Alpine.js client (`web/index.html`) calling the Python API — all generation logic stays server-side.

### Local browser verification

```bash
uv sync --extra e2e
uv run playwright install --with-deps chromium
pwsh -File scripts/run-local-e2e.ps1
```

The local e2e script starts the web app on `http://127.0.0.1:8742`, runs the Playwright tests, captures screenshots, and writes logs/artifacts to `test-results/local-e2e/`.

### Deployment

The web app supports two modes:

| | Local | Deployed |
|---|---|---|
| **Frontend** | Served by `subtitle-gen serve` | Azure Blob Storage static website |
| **Backend** | stdlib HTTP server | Azure Functions (Flex Consumption) |
| **Database** | Full 3 GB SQLite | Mini DB built from CSVs (no vectors) |
| **Jacket** | Full LLM generation | Prompt-only (copy to your LLM) |
| **Monitoring** | -- | App Insights + Log Analytics + email alerts |

**Infrastructure as code** (Bicep): `infra/main.bicep` creates all Azure resources (storage, function app, monitoring, alerts).

**Data pipeline**: slot data is exported as CSV files (tracked in Git), and the mini SQLite DB is built from them at deploy time:

```bash
uv run subtitle-gen export-data                 # dump CSVs (after rebuilding slots)
uv run subtitle-gen build-db                    # build SQLite from CSVs (CI does this)
```

**Deploy**:
1. Configure OIDC: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` as GitHub secrets
2. Set `AZURE_FUNCTIONAPP_NAME` as a GitHub variable
3. Run `deploy-infra.yml` workflow (creates Azure resources)
4. Run `deploy.yml` workflow (deploys function app + frontend)

### Tone tiers

The jacket prompt auto-adapts based on the subtitle's accessibility score (derived from filler corpus frequency):

| Tier | Score | Voice | Examples |
|------|-------|-------|----------|
| **pop** | > 1.0 | Airport bookstore (Gladwell, Pollan, Bryson) | Race, Power, America |
| **mainstream** | 0.5-1.0 | Indie bookstore (Solnit, Mishra, Sheldrake) | Tolkien, Brooklyn |
| **niche** | < 0.5 | University press crossover (Princeton, Yale) | Helmontian Chymistry |

### Remixing

Multi-word of-objects (e.g., "Lurianic Kabbala", "Jews in America") are decomposed into sub-parts and can be recombined into novel pairings. This is enabled by default; use `--no-remix` for original of-objects only.

Run `subtitle-gen calibrate-remix --help` to auto-tune remix parameters with LLM-based rating.

### Tuning

The system includes an autoresearch-inspired tuning loop that uses LLM evaluation to optimize tunable parameters (tone targets, sampling spread, popularity weights, article thresholds, etc.):

```bash
uv run subtitle-gen tune                        # full pipeline (remix + tone)
uv run subtitle-gen tune --phase tone           # tone parameters only
uv run subtitle-gen tune --spot-check           # with human spot-checks
uv run subtitle-gen tune --show-results         # view experiment history
```

Human feedback can also be collected interactively and fed into the tuning loop:

```bash
uv run subtitle-gen review                      # rate 20 subtitles
uv run subtitle-gen generate --review           # rate while generating
```

Source-title tier labels can be populated separately for calibration/evaluation:

```bash
uv sync --extra tune
uv run subtitle-gen classify-source-tiers --dry-run --limit 20
uv run subtitle-gen classify-source-tiers --limit 200 --batch-size 10
uv run subtitle-gen source-tier-distribution
```

`classify-source-tiers` stores labels on `pattern_matches.llm_market_tier*` and
exports `api/data/source_tier_labels.csv` keyed by stable `subtitle_id` plus the
current `pattern_match_id`. It uses the shared pop/mainstream/niche taxonomy
from `market_tiers.py`, with source-label wording distinct from jacket-tone
wording. By default, it uses hosted Responses `web_search` once per source
title; the rationale includes whether the evidence was an exact match,
weak/adjacent match, or no reliable match.
Pass `--no-web-search` to fall back to title/subtitle-only structured labeling.
The reusable Copilot MCP bridge lives in `subtitle_generator.copilot_web_search`
as a plain importable module for future scripts/tools; no FastMCP server is
required for this workflow.

Use `--candidate-source title` or `--candidate-source subtitle` when you need a
targeted labeling batch. `source-tier-distribution` reports the title/subtitle
breakdown and the combined post-rebuild distribution used for calibration.

Tuning goals and parameter bounds are documented in `tuning_goals.md`.

## Commands

| Command | Description |
|---|---|
| `download` | Download LOC MARC bulk data files |
| `download-ol` | Download Open Library editions dump |
| `extract` | Parse MARC files into SQLite |
| `extract-ol` | Parse Open Library dump (deduplicates against LOC) |
| `analyze` | POS-tag subtitles, extract structural templates |
| `build-slots` | Extract clean slot fillers (regex + NLP validated), article stats, and remix sub-parts |
| `generate` | Random subtitle generation (+ optional jacket, review) |
| `jacket` | Standalone jacket generation |
| `calibrate-remix` | Auto-tune remix parameters via LLM rating |
| `classify-source-tiers` | LLM-label real source-title market tiers for calibration/evaluation |
| `source-tier-distribution` | Report source-tier label coverage and the combined post-rebuild calibration mix |
| `tune` | Autoresearch tuning loop (remix + tone parameters) |
| `review` | Interactive subtitle rating session |
| `precompute-vectors` | Recompute remix vector/scalar state after slot extraction changes |
| `serve` | Start the web app locally |
| `export-db` | Export mini SQLite directly from full DB |
| `export-data` | Export slot data as CSV files (for Git) |
| `build-db` | Build mini SQLite from CSV files (for CI) |
| `patterns` | Show discovered subtitle patterns by frequency |
| `slots` | Show available slot fillers |
| `download-popularity` | Download all popularity data sources (SPL, Goodreads, Ottawa, NYT, Trove) |
| `populate-popularity` | Build ISBN mappings + compute composite popularity scores |
| `validate-pipeline` | Run read-only pipeline readiness checks |

## Architecture

```
src/subtitle_generator/
  generate.py          # subtitle generation with remix + article logic
  jacket.py            # jacket prompt construction + LLM execution
  slots.py             # slot extraction + decomposition + article stats
  source_validation.py # shared title/subtitle repair and source corruption checks
  config.py            # centralized tuning parameters (20 params, DB-overridable)
  calibrate.py         # LLM-based remix parameter tuning
  tune.py              # autoresearch tuning loop (Karpathy-inspired)
  eval_harness.py      # evaluation infrastructure (rating, tone separation, composite)
  feedback.py          # human feedback collection + summarization for tuning
  serve.py             # local HTTP server (stdlib)
  export_db.py         # mini DB export for deployment
  parameter_state.py   # typed views over model IDs and tunable parameter families
  pipeline_validation.py # read-only pipeline readiness checks
  remix_state.py       # remix precompute contracts and runtime context
  schema_contracts.py  # stage-aware SQLite schema contracts
  tuning_state.py      # tuning proposal, decision, and rollback state records
  cli.py               # Click CLI entry point
api/
  function_app.py      # Azure Functions v2 (same Python modules)
web/
  index.html           # Alpine.js frontend (thin client)
  js/services.js       # API layer (injectable fetch)
  js/subtitle-vm.js    # Pure view-model functions
  js/app.js            # Alpine x-data component
```

## Tech stack

- **Python 3.13** with [uv](https://docs.astral.sh/uv/)
- **pymarc** — MARC record parsing
- **spaCy** (`en_core_web_md`) — NLP at build time (POS tagging, NER, word vectors for remix precomputation)
- **SQLite** — subtitle storage and slot filler tables
- **click** — CLI framework
- **GitHub Copilot SDK** — LLM for jacket generation
- **litellm** + **pydantic** — structured LLM output for tuning and calibration (optional, lazy-imported)
- **Alpine.js** — reactive frontend (CDN, no build step)
- **marked.js** — markdown rendering (CDN)

## Data sources

### Library of Congress MARC (2016)

[Library of Congress MARC Distribution Services](https://www.loc.gov/cds/products/marcDist.php) — Books All, 2016 retrospective conversion, UTF-8 encoding. ~25M records across 43 files. Free and open access.

### Open Library

[Open Library bulk data dumps](https://openlibrary.org/developers/dumps) — ~35M edition records with a dedicated `subtitle` field (when present). Broader coverage including post-2016 books.

## License

MIT
