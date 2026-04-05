"""Analyze subtitles: POS-tag with spaCy and extract structural templates."""

import json
import sqlite3
from collections import Counter
from pathlib import Path

import click
import spacy

from subtitle_generator.extract import DB_PATH

# Function words to keep literal in templates (lowercased)
LITERAL_TOKENS = {
    "and", "or", "the", "a", "an", "of", "in", "for", "to", "from",
    "with", "on", "at", "by", "as", "into", "through", "between",
    "about", "against", "across", "among", "beyond", "during",
}

# Punctuation to keep literal
LITERAL_PUNCT = {",", ";", ":", "-", "--", "—"}

BATCH_SIZE = 1000
NLP_BATCH_SIZE = 500


def _load_nlp():
    """Load spaCy model with only the components we need."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return nlp


def _token_to_template(token) -> str:
    """Convert a spaCy token to its template representation."""
    text_lower = token.text.lower()

    # Keep punctuation literal
    if token.pos_ == "PUNCT" and token.text in LITERAL_PUNCT:
        return token.text

    # Keep function words literal
    if text_lower in LITERAL_TOKENS:
        return text_lower

    # Everything else becomes its POS tag
    return token.pos_


def _subtitle_to_template(doc) -> str:
    """Convert a spaCy Doc to a structural template string."""
    parts = []
    for token in doc:
        if token.is_space:
            continue
        part = _token_to_template(token)
        parts.append(part)

    # Collapse consecutive identical POS tags: PROPN PROPN → PROPN+
    collapsed = []
    for part in parts:
        if collapsed and collapsed[-1].rstrip("+") == part and part.isupper():
            if not collapsed[-1].endswith("+"):
                collapsed[-1] = part + "+"
        else:
            collapsed.append(part)

    return " ".join(collapsed)


def _extract_slot_data(doc, template: str) -> str:
    """Extract the actual words filling each slot position as JSON."""
    slots = []
    template_parts = template.split()
    token_idx = 0
    tokens = [t for t in doc if not t.is_space]

    for tpl_part in template_parts:
        if token_idx >= len(tokens):
            break

        is_multi = tpl_part.endswith("+")
        base_pos = tpl_part.rstrip("+")

        if tpl_part in LITERAL_PUNCT or tpl_part in LITERAL_TOKENS:
            # Literal — skip the token
            token_idx += 1
            continue

        if is_multi:
            # Consume all consecutive tokens of this POS
            words = []
            while token_idx < len(tokens) and tokens[token_idx].pos_ == base_pos:
                words.append(tokens[token_idx].text)
                token_idx += 1
            slots.append({"pos": base_pos, "text": " ".join(words), "multi": True})
        else:
            tok = tokens[token_idx]
            slots.append({"pos": tok.pos_, "text": tok.text, "multi": False})
            token_idx += 1

    return json.dumps(slots)


def ensure_analysis_tables(conn: sqlite3.Connection):
    """Create analysis tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyzed_subtitles (
            subtitle_id INTEGER PRIMARY KEY,
            template TEXT NOT NULL,
            slot_data TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template TEXT UNIQUE NOT NULL,
            count INTEGER DEFAULT 0,
            example_subtitle TEXT,
            example_title TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_analyzed_template
        ON analyzed_subtitles(template)
    """)
    conn.commit()


def analyze_subtitles(
    conn: sqlite3.Connection,
    limit: int | None = None,
    offset: int = 0,
) -> int:
    """POS-tag subtitles and extract structural templates.

    Returns number of subtitles analyzed.
    """
    nlp = _load_nlp()
    ensure_analysis_tables(conn)

    # Check how many are already analyzed
    already_done = conn.execute("SELECT COUNT(*) FROM analyzed_subtitles").fetchone()[0]
    if already_done > 0 and offset == 0:
        offset = already_done
        click.echo(f"Resuming from offset {offset:,} (already analyzed)")

    # Load subtitle texts
    query = "SELECT id, subtitle, title FROM subtitles ORDER BY id LIMIT ? OFFSET ?"
    batch_limit = limit or 999_999_999
    rows = conn.execute(query, (batch_limit, offset)).fetchall()

    if not rows:
        click.echo("No subtitles to analyze.")
        return 0

    click.echo(f"Analyzing {len(rows):,} subtitles with spaCy...")

    analyzed = 0
    batch_rows = []
    batch_texts = []

    for i, (sid, subtitle, title) in enumerate(rows):
        batch_rows.append((sid, subtitle, title))
        batch_texts.append(subtitle)

        if len(batch_texts) >= NLP_BATCH_SIZE or i == len(rows) - 1:
            docs = list(nlp.pipe(batch_texts))
            insert_batch = []
            for (sid, subtitle, title), doc in zip(batch_rows, docs):
                template = _subtitle_to_template(doc)
                slot_data = _extract_slot_data(doc, template)
                insert_batch.append((sid, template, slot_data))

            conn.executemany(
                "INSERT OR IGNORE INTO analyzed_subtitles (subtitle_id, template, slot_data) "
                "VALUES (?, ?, ?)",
                insert_batch,
            )
            conn.commit()
            analyzed += len(insert_batch)
            batch_rows.clear()
            batch_texts.clear()

            if analyzed % 10000 == 0:
                click.echo(f"  ...analyzed {analyzed:,} subtitles")

    click.echo(f"Analyzed {analyzed:,} subtitles total.")
    return analyzed


def build_pattern_index(conn: sqlite3.Connection):
    """Aggregate templates into the patterns table with counts and examples."""
    click.echo("Building pattern index...")
    conn.execute("DELETE FROM patterns")
    conn.execute("""
        INSERT INTO patterns (template, count, example_subtitle, example_title)
        SELECT
            a.template,
            COUNT(*) as cnt,
            s.subtitle,
            s.title
        FROM analyzed_subtitles a
        JOIN subtitles s ON s.id = a.subtitle_id
        GROUP BY a.template
        ORDER BY cnt DESC
    """)
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
    top = conn.execute("SELECT COUNT(*) FROM patterns WHERE count >= 10").fetchone()[0]
    click.echo(f"Found {total:,} unique templates ({top:,} with 10+ occurrences)")
