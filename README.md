> Created/edited by GitHub Copilot with human review/feedback by Avi Levin.

# subtitle-generator

Generate strange pop-nonfiction subtitles in the pattern:

```text
X, Y, and the Z of W
```

The project mines real book titles and subtitles from Library of Congress and Open Library data, extracts reusable slot fillers, classifies those fillers into pop/mainstream/niche tiers, and recombines them into new subtitles. The intended result is coherent grammar with surreal nonfiction energy: "Genius of Emissions Trading" is good; broken artifacts like "the Challenge of Christian" are not.

## Current runtime model

Generation now separates these responsibilities:

| Concern | Runtime behavior |
|---|---|
| **Build evidence** | Source labels, learned book-model probabilities, confidence, and priors contribute evidence to the tier-slot artifact. |
| **Generation** | The default runtime samples directly from normalized `P(filler | tier, slot_type)` over the validated filler universe. |
| **Rollback** | The former `sqrt(freq) * score_T` path remains available explicitly as `legacy`. |
| **Guardrail** | A narrow literal-artifact filter removes broken strings and malformed final objects without rejecting funny conceptual collisions. |

The deployment inputs include the tracked
`api\data\tier_slot_filler_distribution_v1.csv` artifact and
`generation_runtime_mode=artifact` config. The ignored mini DB at
`api\data\subtitles.mini.db` is built from tracked CSVs locally and in CI, so
deployed Azure Functions do not need the full local SQLite database.

## Examples

```text
Jefferson, Repression, and the Category of Scripture in Lurianic Kabbala
Greed, Grit, and the Rise of the American Dream
Faith, Hope, and a Healthy Dose of Laughter
Celebrity Culture, Theology, and the Collapse of New England
Hamas, Hard Power, and the Genius of Emissions Trading
```

The optional jacket generator turns a subtitle into a full book-jacket prompt or local LLM-backed jacket:

```text
Holy Nation
Professionals, Pagan Authors, and the Sacramental Vision of the Nation State
```

## Setup

```powershell
git clone https://github.com/Arithmomaniac/subtitle-generator.git
Set-Location subtitle-generator
uv sync
```

Optional extras:

| Extra | Use |
|---|---|
| `uv sync --extra e2e` | Playwright browser verification |
| `uv sync --extra tune` | LiteLLM/Pydantic structured LLM tuning and review tools |
| `uv sync --extra deploy` | Azure Table Storage rating sync |
| `uv sync --extra ml` | Torch book-model training/distillation |

## Daily use

```powershell
uv run subtitle-gen generate
uv run subtitle-gen generate --tone pop
uv run subtitle-gen generate --tone mainstream --sources
uv run subtitle-gen generate --tone niche --jacket
uv run subtitle-gen generate --runtime legacy --tone pop  # rollback comparison
uv run subtitle-gen jacket "sturgeon, caviar, and the geography of desire"
uv run subtitle-gen serve --no-open
```

The local web app runs on `http://127.0.0.1:8742` by default. It exposes the same generation path as the CLI, plus source display, rating tags, prompt building, remix details, and a local-only spot-check page.

## Build pipeline

Run the full data pipeline only when rebuilding the corpus from raw data:

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

The pipeline stages are contract-checked. `validate-pipeline` is read-only and verifies schema, config, remix precompute state, popularity coverage, model IDs, and serving readiness.

## Book-model evidence workflow

The offline book model produces filler-level tier evidence. Those scores feed the
anchored tier-slot artifact and remain installed for legacy rollback and
compatibility classification.

```powershell
uv sync --extra ml --extra tune

# Build training artifacts from labeled source books and enrichment data.
uv run subtitle-gen build-book-features

# Train/evaluate richer local models and export distilled runtime scores.
uv run subtitle-gen train-book-model-torch
uv run subtitle-gen distill-book-model
uv run subtitle-gen shadow-book-model

# Optional: generate fixed samples for human/ad hoc LLM review.
uv run subtitle-gen categorization-gate --dry-run

# Install the selected export-slot rollup into the local DB.
uv run subtitle-gen install-book-model-scores `
  --input generated-artifacts\book-model\shadow-rollups\filler_book_rollups_export-slot.csv
```

For repeatable runs, use the checked-in step runner:

```powershell
pwsh -File scripts\run-book-model-pipeline.ps1 -Steps Inventory
pwsh -File scripts\run-book-model-pipeline.ps1 -Steps Features,Torch,CalibrateRuntimeTierModel,Distill,Shadow,CategorizationGate -PlanOnly
```

The evidence/rollback table is `slot_filler_model_scores`:

| Column | Meaning |
|---|---|
| `slot_filler_id` | References `slot_fillers.id`. |
| `score_pop` | Probability-like score that this filler belongs in pop-accessible generation. |
| `score_mainstream` | Probability-like score for mainstream generation. |
| `score_niche` | Probability-like score for niche generation. |
| `model_tier` | Highest-scoring learned tier. |
| `source_prediction_count` | Number of source-book predictions that contributed to the rollup. |

Mini DB builds reject partial model-score coverage when
`slot_filler_model_scores.csv` is present. This keeps artifact rebuilds and
legacy rollback from silently using partially scored fillers.

## Export and deployment artifacts

After rebuilding slots, popularity, remix vectors, or model scores, regenerate the tracked runtime data:

```powershell
uv run subtitle-gen build-tier-slot-distribution
uv run subtitle-gen install-tier-slot-runtime `
  --artifact generated-artifacts\tier-slot-distribution\tier_slot_filler_distribution_v1.csv
uv run subtitle-gen export-data -o api\data
uv run subtitle-gen build-db -d api\data -o api\data\subtitles.mini.db
```

Tracked deployment inputs:

| Path | Purpose |
|---|---|
| `api\data\slot_fillers.csv` | Validated strict filler universe and runtime scalar state. |
| `api\data\tier_slot_filler_distribution_v1.csv` | Default anchored tier-slot generation probabilities. |
| `api\data\slot_filler_model_scores.csv` | Learned tier probabilities retained for artifact builds and legacy rollback. |
| `api\data\sources.csv` | Source-book attribution for generated slots. |
| `api\data\config.csv` | Runtime tuning/config values. |
| `api\data\source_tier_labels.csv` | Offline source-label evidence for training/evaluation continuity. |

Ignored build outputs:

| Path | Purpose |
|---|---|
| `api\data\subtitles.mini.db` | Built SQLite artifact used by local serving and Azure Functions; CI rebuilds it from CSVs with worker-readable permissions. |
| `generated-artifacts\` | Local reports, model features, predictions, rollups, and gate outputs. |

The configured default runtime is `artifact`. Use `--runtime legacy`,
`SUBTITLE_GEN_RUNTIME_MODE=legacy`, or
`subtitle-gen set-runtime-default --mode legacy` to roll back.

Azure Flex mounts the released package read-only. The Function worker copies the
packaged mini DB once into temporary storage and opens that immutable copy
read-only; local serving continues to use its writable DB for local ratings.

## Local verification

Run these before deploying runtime or web changes:

```powershell
uv run subtitle-gen validate-pipeline
uv run ruff check
uv run pytest -q
uv sync --extra deploy --extra tune --extra e2e
uv run playwright install --with-deps chromium
pwsh -File scripts\run-local-e2e.ps1
```

`scripts\run-local-e2e.ps1` starts `subtitle-gen serve --no-open`, waits for readiness, captures before/after screenshots, runs the home-page flow, runs the local spot-check flow, and writes artifacts to `test-results\local-e2e\`.

## Browser e2e coverage

| File | Scope |
|---|---|
| `tests\test_e2e.py` | Home page, mode badge, generation, quality tags, sources, jacket prompt, copy button, settings, tier-filtered regeneration, GitHub link, remix display, mobile overflow, and deployed App Insights telemetry. |
| `tests\test_e2e_spot_check.py` | Local spot-check page, batch loading, tier rating payloads, keyboard shortcuts, tag toggles, skip flow, summary, load more, hints, and back link. |
| `scripts\run-local-e2e.ps1` | Local server lifecycle, readiness wait, screenshots, failure capture, and test orchestration. |

To run against deployment manually:

```powershell
$env:BASE_URL = "https://subtitlegenst.z13.web.core.windows.net"
uv run python tests\test_e2e.py
```

The spot-check test is local-only because those endpoints are not deployed.

## GitHub Actions deployment

Deployment is split into infrastructure and application workflows:

| Workflow | Trigger | What it does |
|---|---|---|
| `.github\workflows\deploy-infra.yml` | Push to `infra/**` or manual dispatch | Deploys Bicep resources, static website hosting, monitoring, alerts, and optional RBAC role assignments. |
| `.github\workflows\deploy.yml` | Push to `master` touching `web/**`, `api/**`, `src/**`, or manual dispatch | Builds the mini DB from CSVs, deploys Azure Functions, uploads the static web app, runs smoke tests, and runs deployed e2e. |

Required GitHub configuration:

| Name | Type | Purpose |
|---|---|---|
| `AZURE_CLIENT_ID` | Secret | OIDC client ID for Azure login. |
| `AZURE_TENANT_ID` | Secret | Azure tenant. |
| `AZURE_SUBSCRIPTION_ID` | Secret | Azure subscription. |
| `AZURE_FUNCTIONAPP_NAME` | Variable | Function app name, for example `subtitlegen-func`. |
| `ALERT_EMAIL` | Secret or workflow input | Optional Azure Monitor alert recipient. |

The deploy workflow derives the static website storage account from the function app name. For the current production site, e2e targets `https://subtitlegenst.z13.web.core.windows.net`.

## Feedback and tuning

Interactive feedback is stored locally and can be synced from Azure Table Storage when deploy dependencies are installed:

```powershell
uv run subtitle-gen review
uv run subtitle-gen generate --review
uv sync --extra deploy
uv run subtitle-gen pull-ratings --account subtitlegenst --since 2026-05-01
```

Source-title tier labels are separate from generated-subtitle ratings:

```powershell
uv sync --extra tune
uv run subtitle-gen classify-source-tiers --dry-run --limit 20
uv run subtitle-gen classify-source-tiers --limit 200 --batch-size 10
uv run subtitle-gen source-tier-distribution
```

The source-label workflow writes `pattern_matches.llm_market_tier*` and exports `api\data\source_tier_labels.csv`.

## Architecture map

```text
src\subtitle_generator\
  cli.py                         Click command surface
  generate.py                    Runtime subtitle generation, remixing, guardrails
  tiering.py                     Runtime pop/mainstream/niche evidence
  slots.py                       Slot extraction and source cleanup gates
  export_db.py                   CSV export and mini DB build
  schema_contracts.py            Full and mini DB schema contracts
  pipeline_validation.py         Read-only pipeline readiness checks
  book_model_artifacts.py        Offline feature/label artifact builder
  book_model_torch.py            Torch teacher training
  book_model_distillation.py     Exportable runtime student models
  book_model_shadow.py           Slot-filler rollups for runtime install
  jacket.py                      Jacket prompt construction and local LLM execution
  serve.py                       Local stdlib HTTP server
  handlers.py                    Shared local/Azure request handlers

api\
  function_app.py                Azure Functions entry point
  data\                          Tracked CSVs and built mini DB

web\
  index.html                     Static Alpine.js app
  js\                            Browser services/view-model modules
```

## Tech stack

- Python 3.13 and `uv`
- SQLite for local corpus state and deployment mini DBs
- spaCy for build-time NLP and remix precompute
- Torch for optional offline book-tier modeling
- Click for CLI commands
- GitHub Copilot SDK for local jacket generation
- LiteLLM/Pydantic for structured tuning/modeling reviews
- Playwright for browser e2e
- Alpine.js and marked.js for the static frontend
- Azure Functions, Blob static website hosting, Table Storage, and App Insights for deployment

## Data sources

| Source | Use |
|---|---|
| Library of Congress MARC Books All 2016 | Large source-title/subtitle corpus. |
| Open Library dumps | Additional edition metadata and subtitle/title candidates. |
| SPL, Goodreads, NYT, Ottawa/library, Trove | Popularity/enrichment signals for offline training and historical scoring. |

## License

MIT
