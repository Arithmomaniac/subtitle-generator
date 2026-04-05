"""Extract slots from the 'X, Y, and the Z of W' subtitle pattern.

Uses regex to find the structural pattern in raw text, then spaCy to
validate that each slot filler is the right type of phrase.
"""

import json
import re
import sqlite3

import click
import spacy

# Regex to match: "A, B[,] and the C of D"
PATTERN_RE = re.compile(
    r"^(?P<list_part>.+,\s*.+?)\s*,?\s+and\s+the\s+(?P<action>.+?)\s+of\s+(?P<object>.+)$",
    re.IGNORECASE,
)

NOISE_WORDS = {"hearing", "subcommittee", "committee", "congress", "session"}

# Deverbal/process suffixes for action noun validation
ACTION_SUFFIXES = (
    "ing", "tion", "sion", "ment", "ance", "ence", "ery", "ure",
    "al", "sis", "ry", "cy", "th", "se", "ge",
)
# Common action nouns without obvious suffixes
ACTION_WHITELIST = {
    "rise", "fall", "fate", "birth", "death", "dawn", "age", "quest",
    "end", "loss", "cost", "role", "rule", "roots", "price",
    "cult", "myth", "dream", "ghost", "world", "life", "saga", "war",
    "art", "soul", "heart", "spirit", "genius", "secret", "mystery",
    "power", "future", "nature", "origins", "legacy", "promise",
    "triumph", "crisis", "limits", "paradox", "dilemma", "logic",
    "pursuit", "specter", "collapse", "plight", "impact", "source",
    "demise", "advent", "roots", "fabric", "drama", "gospel",
}


def _load_nlp():
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


def _is_noise(subtitle: str) -> bool:
    lower = subtitle.lower()
    return any(w in lower for w in NOISE_WORDS)


def _split_list_items(list_part: str) -> list[str]:
    return [item.strip() for item in list_part.split(",") if item.strip()]


def _is_valid_action(phrase: str, nlp) -> bool:
    """Action noun must be a process/event noun (making, rise, collapse, etc.)."""
    words = phrase.lower().split()
    if not words or len(words) > 3:
        return False
    head = words[-1]
    if head in ACTION_WHITELIST:
        return True
    if any(head.endswith(s) for s in ACTION_SUFFIXES):
        return True
    # Check lemma via spaCy
    doc = nlp(phrase)
    if doc:
        lemma = doc[-1].lemma_.lower()
        if lemma in ACTION_WHITELIST:
            return True
        if any(lemma.endswith(s) for s in ACTION_SUFFIXES):
            return True
    return False


def _is_valid_list_item(phrase: str, nlp) -> bool:
    """List items should be punchy nouns/names, 1-3 words."""
    words = phrase.split()
    if not (1 <= len(words) <= 3):
        return False
    if phrase.lower().startswith(("and ", "or ", "with ", "from ", "to ", "the ")):
        return False
    if re.match(r"^\d", phrase) or re.search(r"\d{4}", phrase):
        return False
    doc = nlp(phrase)
    return any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in doc)


def _is_valid_object(phrase: str, nlp) -> bool:
    """The 'of X' part should be a meaningful NP, 1-6 words."""
    words = phrase.split()
    if not (1 <= len(words) <= 6):
        return False
    if re.search(r"\d{4}", phrase):
        return False
    doc = nlp(phrase)
    return any(t.pos_ in ("NOUN", "PROPN") for t in doc)


def extract_pattern_matches(conn: sqlite3.Connection) -> list[dict]:
    """Find all subtitles matching X, Y, and the Z of W."""
    rows = conn.execute(
        "SELECT id, title, subtitle FROM subtitles "
        "WHERE subtitle LIKE '%, % and the % of %'"
    ).fetchall()

    matches = []
    for sid, title, subtitle in rows:
        if _is_noise(subtitle):
            continue
        m = PATTERN_RE.match(subtitle)
        if not m:
            continue

        list_part = m.group("list_part")
        action = m.group("action").strip()
        obj = re.sub(r"[\s]*[/:;,.]\s*$", "", m.group("object")).strip()

        items = _split_list_items(list_part)
        if len(items) < 2:
            continue

        matches.append({
            "subtitle_id": sid,
            "title": title,
            "subtitle": subtitle,
            "list_items": items,
            "action_noun": action,
            "of_object": obj,
        })

    return matches


def ensure_slot_tables(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pattern_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subtitle_id INTEGER UNIQUE,
            title TEXT,
            subtitle TEXT,
            list_items_json TEXT,
            action_noun TEXT,
            of_object TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS slot_fillers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            source_subtitle_id INTEGER,
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_slot_type ON slot_fillers(slot_type)")
    conn.commit()


def build_slots(conn: sqlite3.Connection):
    """Extract pattern matches and build slot filler tables with NLP validation."""
    ensure_slot_tables(conn)
    conn.execute("DELETE FROM pattern_matches")
    conn.execute("DELETE FROM slot_fillers")
    conn.commit()

    click.echo("Loading spaCy model...")
    nlp = _load_nlp()

    click.echo("Finding pattern matches...")
    matches = extract_pattern_matches(conn)
    click.echo(f"Found {len(matches):,} raw matches")

    conn.executemany(
        "INSERT OR IGNORE INTO pattern_matches "
        "(subtitle_id, title, subtitle, list_items_json, action_noun, of_object) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (m["subtitle_id"], m["title"], m["subtitle"],
             json.dumps(m["list_items"]), m["action_noun"], m["of_object"])
            for m in matches
        ],
    )
    conn.commit()

    # NLP-validated slot extraction
    list_items_seen = set()
    action_nouns_seen = set()
    of_objects_seen = set()
    clean_matches = 0

    for i, m in enumerate(matches):
        action = m["action_noun"]
        obj = m["of_object"]

        if not _is_valid_action(action, nlp):
            continue
        if not _is_valid_object(obj, nlp):
            continue

        valid_items = []
        for item in m["list_items"]:
            cleaned = re.sub(r"[\s]*[/:;,.]\s*$", "", item).strip()
            if cleaned and _is_valid_list_item(cleaned, nlp):
                valid_items.append(cleaned)

        if len(valid_items) < 2:
            continue

        clean_matches += 1
        for item in valid_items:
            list_items_seen.add(item)
        action_nouns_seen.add(action)
        of_objects_seen.add(obj)

        if (i + 1) % 2000 == 0:
            click.echo(f"  ...validated {i + 1:,} / {len(matches):,}")

    click.echo(f"Clean matches (NLP-validated): {clean_matches:,}")

    filler_rows = (
        [("list_item", x, None) for x in list_items_seen]
        + [("action_noun", x, None) for x in action_nouns_seen]
        + [("of_object", x, None) for x in of_objects_seen]
    )
    conn.executemany(
        "INSERT OR IGNORE INTO slot_fillers (slot_type, filler, source_subtitle_id) "
        "VALUES (?, ?, ?)",
        filler_rows,
    )
    conn.commit()

    for slot_type in ["list_item", "action_noun", "of_object"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ?", (slot_type,)
        ).fetchone()[0]
        click.echo(f"  {slot_type}: {count:,} unique fillers")

    click.echo("Slot extraction complete!")
