# subtitle-generator

Generate bizarre book subtitles by mixing and matching real parts from Library of Congress MARC records.

Extracts subtitles from LOC bulk MARC data, decomposes them into structural patterns using NLP (spaCy POS tagging), then recombines slot-fillers slot-machine style for surprising results.

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_sm
```

## Usage

```bash
uv run subtitle-gen --help
```
