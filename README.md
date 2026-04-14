# subtitle-generator

Generate bizarre book subtitles in the pop-nonfiction pattern — *"X, Y, and the Z of W"* — by mining real parts from the Library of Congress MARC database and Open Library, then recombining them slot-machine style.

Optionally generate a **full book jacket** with title, back cover copy, trade journal reviews, and endorsement blurbs from real people — powered by the GitHub Copilot SDK.

## Examples

**Random subtitles:**
```
Jefferson, Repression, and the Category of Scripture in Lurianic Kabbala
UFOs, Rising Powers, and the Bicentennial History of Performance
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
2. **Extract** 11M+ English subtitles into SQLite (with cross-source deduplication)
3. **Pattern match** subtitles matching "X, Y, and the Z of W" using regex + spaCy NLP validation
4. **Decompose** into typed slots: list items, action nouns, of-objects — plus sub-parts (modifiers, heads, prepositional complements) for remixing
5. **Generate** by randomly drawing one filler per slot — weighted by sqrt(corpus frequency). Multi-word of-objects can be remixed into novel combinations (e.g., "New York" + "kitsch" from different books)
6. **Jacket** (optional) — send the subtitle to an LLM (via Copilot SDK) to generate a full book jacket with trade journal reviews and endorsement blurbs from real people

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
```

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

## Commands

| Command | Description |
|---|---|
| `download` | Download LOC MARC bulk data files |
| `download-ol` | Download Open Library editions dump |
| `extract` | Parse MARC files into SQLite |
| `extract-ol` | Parse Open Library dump (deduplicates against LOC) |
| `analyze` | POS-tag subtitles, extract structural templates |
| `build-slots` | Extract slot fillers (regex + NLP validated) + precompute vectors |
| `generate` | Random subtitle generation (+ optional jacket) |
| `jacket` | Standalone jacket generation |
| `calibrate-remix` | Auto-tune remix parameters via LLM rating |
| `precompute-vectors` | Recompute remix vector decomposition (included in `build-slots`) |
| `serve` | Start the web app locally |
| `export-db` | Export mini SQLite directly from full DB |
| `export-data` | Export slot data as CSV files (for Git) |
| `build-db` | Build mini SQLite from CSV files (for CI) |
| `patterns` | Show discovered subtitle patterns by frequency |
| `slots` | Show available slot fillers |

## Architecture

```
src/subtitle_generator/
  generate.py          # subtitle generation with remix + locked slots
  jacket.py            # jacket prompt construction + LLM execution
  slots.py             # slot extraction + decomposition
  calibrate.py         # LLM-based remix parameter tuning
  serve.py             # local HTTP server (stdlib)
  export_db.py         # mini DB export for deployment
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
- **Alpine.js** — reactive frontend (CDN, no build step)
- **marked.js** — markdown rendering (CDN)

## Data sources

### Library of Congress MARC (2016)

[Library of Congress MARC Distribution Services](https://www.loc.gov/cds/products/marcDist.php) — Books All, 2016 retrospective conversion, UTF-8 encoding. ~25M records across 43 files. Free and open access.

### Open Library

[Open Library bulk data dumps](https://openlibrary.org/developers/dumps) — ~35M edition records with a dedicated `subtitle` field (when present). Broader coverage including post-2016 books.

## License

MIT
