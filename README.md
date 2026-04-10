# subtitle-generator

Generate bizarre book subtitles in the pop-nonfiction pattern — *"X, Y, and the Z of W"* — by mining real parts from the Library of Congress MARC database and Open Library, then recombining them slot-machine style.

Optionally generate a **full book jacket** with title, back cover copy, trade journal reviews, and endorsement blurbs from real people — powered by the GitHub Copilot SDK.

## Examples

**Random subtitles:**
```
Jefferson, Repression, and the category of Scripture in Lurianic Kabbala
UFOs, rising powers, and the bicentennial history of performance
celebrity culture, theology, and the collapse of New England
```

**Full book jacket** (with `--jacket`):

> **Holy Nation**
> *professionals, pagan authors, and the sacramental vision of the nation state*
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
6. **Jacket** (optional) — send the subtitle to an LLM (via Copilot SDK) to generate a full book jacket with web-search-enriched concept, trade journal reviews, and endorsement blurbs from real people

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

```bash
uv run subtitle-gen generate                    # 10 random subtitles
uv run subtitle-gen generate --sources          # show source books
uv run subtitle-gen generate --tone pop         # bias toward accessible
uv run subtitle-gen generate --jacket           # subtitle + full jacket
uv run subtitle-gen jacket "sturgeon, caviar, and the geography of desire"
```

Run `subtitle-gen <command> --help` for full options on any command.

### Tone tiers

The jacket prompt auto-adapts based on the subtitle's accessibility score (derived from filler corpus frequency):

| Tier | Score | Voice | Examples |
|------|-------|-------|----------|
| **pop** | > 1.0 | Airport bookstore (Gladwell, Pollan, Bryson) | Race, Power, America |
| **mainstream** | 0.5–1.0 | Indie bookstore (Solnit, Mishra, Sheldrake) | Tolkien, Brooklyn |
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
| `build-slots` | Extract slot fillers (regex + NLP validated) |
| `generate` | Random subtitle generation (+ optional jacket) |
| `jacket` | Standalone jacket generation |
| `calibrate-remix` | Auto-tune remix parameters via LLM rating |
| `patterns` | Show discovered subtitle patterns by frequency |
| `slots` | Show available slot fillers |

## Tech stack

- **Python 3.13** with [uv](https://docs.astral.sh/uv/)
- **pymarc** — MARC record parsing
- **spaCy** (`en_core_web_md`) — NLP (POS tagging, NER, word vectors)
- **SQLite** — subtitle storage and slot filler tables
- **click** — CLI framework
- **GitHub Copilot SDK** — LLM + web search for jacket generation

## Data sources

### Library of Congress MARC (2016)

[Library of Congress MARC Distribution Services](https://www.loc.gov/cds/products/marcDist.php) — Books All, 2016 retrospective conversion, UTF-8 encoding. ~25M records across 43 files. Free and open access.

### Open Library

[Open Library bulk data dumps](https://openlibrary.org/developers/dumps) — ~35M edition records with a dedicated `subtitle` field (when present). Broader coverage including post-2016 books.

## License

MIT
