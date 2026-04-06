# 📚 subtitle-generator

Generate bizarre book subtitles in the pop-nonfiction pattern — *"X, Y, and the Z of W"* — by mining real parts from the Library of Congress MARC database, then recombining them slot-machine style.

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

1. **Download** ~25M MARC records from the LOC bulk distribution (43 files, ~9 GB)
2. **Extract** 2.4M English subtitles from MARC field 245$b into SQLite
3. **Pattern match** subtitles matching "X, Y, and the Z of W" using regex + spaCy NLP validation
4. **Decompose** into typed slots: list items, action nouns, of-objects
5. **Generate** by randomly drawing one filler per slot — uniform random, no weighting
6. **Jacket** (optional) — send the subtitle to an LLM (via Copilot SDK) to generate a full book jacket with web-search-enhanced endorsement blurbs

### Strict vs Loose mode

- **Strict** (~3K list items, ~670 action nouns, ~1.9K of-objects) — fillers from NLP-validated tricolon subtitles only
- **Loose** (~7K / ~2.7K / ~3.9K) — expanded from the full 2.4M corpus with two-pass tuning (rule-based + vector similarity)

## Setup

```bash
# Clone and install
git clone https://github.com/Arithmomaniac/subtitle-generator.git
cd subtitle-generator
uv sync

# Download LOC MARC data (~9 GB, takes a while)
uv run subtitle-gen download --parts all

# Extract subtitles into SQLite
uv run subtitle-gen extract

# Build slot fillers (strict mode)
uv run subtitle-gen build-slots

# Optional: expand with loose mode
uv run subtitle-gen build-slots --loose

# Optional: tune loose mode quality
uv run subtitle-gen tune
```

## Usage

```bash
# Generate 10 random subtitles
uv run subtitle-gen generate

# Generate with loose (expanded) pool
uv run subtitle-gen generate --loose

# Show source books for each filler
uv run subtitle-gen generate --sources

# Generate a full book jacket (title, back cover, reviews, blurbs)
uv run subtitle-gen generate --jacket

# Standalone jacket command — custom subtitle
uv run subtitle-gen jacket "sturgeon, caviar, and the geography of desire"

# Jacket with a specific model
uv run subtitle-gen jacket --model gpt-4.1

# All the flags
uv run subtitle-gen generate --jacket --loose --sources --model claude-haiku-4.5 -n 3
```

### Available jacket models (sub-1x cost)

| Model | Cost | Speed | Reliability |
|---|---|---|---|
| `gpt-5.4-mini` (default) | 0.33x | ~22s | ✅ All sections |
| `gpt-4.1` | Free | ~87s | ✅ All sections |
| `claude-haiku-4.5` | 0.33x | ~40s | ⚠️ May merge sections |
| `gpt-5-mini` | Free | ~67s | ⚠️ May skip blurbs |

## Commands

| Command | Description |
|---|---|
| `download` | Download LOC MARC bulk data files |
| `extract` | Parse MARC files → SQLite subtitles table |
| `analyze` | POS-tag subtitles, extract structural templates |
| `build-slots` | Extract slot fillers (regex + NLP validated) |
| `generate` | Random subtitle generation (+ optional `--jacket`) |
| `jacket` | Standalone jacket generation (custom or random subtitle) |
| `tune` | Two-pass quality tuning for loose mode |
| `patterns` | Show discovered subtitle patterns by frequency |
| `slots` | Show available slot fillers |

## Tech stack

- **Python 3.13** with [uv](https://docs.astral.sh/uv/)
- **pymarc** — MARC record parsing
- **spaCy** — NLP (POS tagging, noun validation, word vectors)
- **SQLite** — subtitle storage and slot filler tables
- **click** — CLI framework
- **GitHub Copilot SDK** — LLM + web search for jacket generation

## Data source

[Library of Congress MARC Distribution Services](https://www.loc.gov/cds/products/marcDist.php) — Books All, 2016 retrospective conversion, UTF-8 encoding. ~25M records across 43 files. Free and open access.

## License

MIT
