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
    # Reject parens, quotes, special chars
    if re.search(r"""['"()\[\]\u2018\u2019\u201c\u201d]""", phrase):
        return False
    # Reject ALL-CAPS words
    if any(w.isupper() and len(w) > 1 for w in phrase.split()):
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
    # Reject compound fragments ("Loans and deposits")
    if " and " in phrase.lower() or " or " in phrase.lower():
        return False
    if re.match(r"^\d", phrase) or re.search(r"\d{4}", phrase):
        return False
    # Reject embedded quotes, possessives, parens, and brackets
    if re.search(r"""['"()\[\]\u2018\u2019\u201c\u201d]""", phrase):
        return False
    # Reject ALL-CAPS words (catalog artifacts)
    if any(w.isupper() and len(w) > 1 for w in words):
        return False
    # Reject abbreviations (1-2 letter uppercase words that aren't common)
    if any(len(w) <= 2 and w.isupper() and w not in ("US", "UK", "EU", "AI") for w in words):
        return False
    doc = nlp(phrase)
    # Must contain at least one real noun (ADJ alone isn't enough)
    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return False
    # Reject phrases led by function words (prepositions, conjunctions, determiners, verbs)
    if doc[0].pos_ in ("ADP", "CCONJ", "SCONJ", "DET", "VERB", "AUX", "PART", "ADV"):
        return False
    return True


def _is_valid_object(phrase: str, nlp) -> bool:
    """The 'of X' part should be a meaningful NP, 1-5 words."""
    words = phrase.split()
    if not (1 <= len(words) <= 5):
        return False
    if re.search(r"\d{4}", phrase):
        return False
    # Reject internal commas (trailing location patterns like "Oaxaca, Mexico")
    if "," in phrase:
        return False
    # Reject possessives, embedded quotes, parens, brackets
    if re.search(r"""['()\[\]\u2018\u2019\u201c\u201d]""", phrase):
        return False
    if '"' in phrase:
        return False
    # Reject ALL-CAPS words (catalog artifacts)
    if any(w.isupper() and len(w) > 1 for w in words):
        return False
    # Reject non-English articles at start
    if words[0].lower() in ("der", "die", "das", "les", "la", "le", "el", "los", "las", "des", "du"):
        return False
    doc = nlp(phrase)
    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return False
    # Reject determiner-led phrases ("Every Christian", "a new species")
    if doc[0].pos_ == "DET":
        return False
    # Reject if contains pronouns ("himself", "its")
    if any(t.pos_ == "PRON" for t in doc):
        return False
    return True


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
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            UNIQUE(slot_type, filler)
        )
    """)
    # Migration: add freq column if missing (pre-existing databases)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slot_fillers)")}
    if "freq" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN freq INTEGER NOT NULL DEFAULT 1")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_slot_type ON slot_fillers(slot_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_slot_mode ON slot_fillers(mode)")
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
    list_items_seen: dict[str, tuple[int, int]] = {}  # filler → (source_subtitle_id, count)
    action_nouns_seen: dict[str, tuple[int, int]] = {}
    of_objects_seen: dict[str, tuple[int, int]] = {}
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
        sid = m["subtitle_id"]
        for item in valid_items:
            prev = list_items_seen.get(item)
            list_items_seen[item] = (prev[0] if prev else sid, (prev[1] if prev else 0) + 1)
        prev = action_nouns_seen.get(action)
        action_nouns_seen[action] = (prev[0] if prev else sid, (prev[1] if prev else 0) + 1)
        prev = of_objects_seen.get(obj)
        of_objects_seen[obj] = (prev[0] if prev else sid, (prev[1] if prev else 0) + 1)

        if (i + 1) % 2000 == 0:
            click.echo(f"  ...validated {i + 1:,} / {len(matches):,}")

    click.echo(f"Clean matches (NLP-validated): {clean_matches:,}")

    filler_rows = (
        [("list_item", x, "strict", sid, freq) for x, (sid, freq) in list_items_seen.items()]
        + [("action_noun", x, "strict", sid, freq) for x, (sid, freq) in action_nouns_seen.items()]
        + [("of_object", x, "strict", sid, freq) for x, (sid, freq) in of_objects_seen.items()]
    )
    conn.executemany(
        "INSERT OR IGNORE INTO slot_fillers (slot_type, filler, mode, source_subtitle_id, freq) "
        "VALUES (?, ?, ?, ?, ?)",
        filler_rows,
    )
    conn.commit()

    for slot_type in ["list_item", "action_noun", "of_object"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ?", (slot_type,)
        ).fetchone()[0]
        click.echo(f"  {slot_type}: {count:,} unique fillers")

    click.echo("Strict slot extraction complete!")


# Boilerplate words to reject in loose mode
_BOILERPLATE = {
    "study", "introduction", "survey", "review", "report", "analysis",
    "proceedings", "bibliography", "handbook", "manual", "guide",
    "volume", "edition", "series", "supplement", "appendix", "index",
    "catalog", "catalogue", "directory", "register", "yearbook",
    "dissertation", "thesis", "monograph",
}


def _mine_the_x_of_y(conn: sqlite3.Connection, nlp) -> tuple[set, set]:
    """Mine 'the X of Y' from ALL subtitles for action nouns and of-objects.
    Uses cheap heuristics only — no per-item nlp() calls. Tune pass cleans up later.
    """
    the_x_of_y_re = re.compile(
        r"the\s+(.{2,30}?)\s+of\s+(.{2,60}?)(?:\s*[,:;.]|$)", re.IGNORECASE,
    )
    rows = conn.execute("SELECT id, subtitle FROM subtitles").fetchall()

    action_nouns = set()
    of_objects = set()

    for i, (sid, subtitle) in enumerate(rows):
        for m in the_x_of_y_re.finditer(subtitle):
            action = m.group(1).strip()
            obj = re.sub(r"[\s]*[/:;,.]\s*$", "", m.group(2)).strip()

            # Cheap action noun filter: suffix/whitelist + length + boilerplate
            if action.lower() in _BOILERPLATE:
                continue
            action_words = action.split()
            if len(action_words) > 3:
                continue
            head = action_words[-1].lower()
            if head in ACTION_WHITELIST or any(head.endswith(s) for s in ACTION_SUFFIXES):
                action_nouns.add(action)

            # Cheap of-object filter: length + has alpha + no dates
            obj_words = obj.split()
            if 1 <= len(obj_words) <= 6 and len(obj) >= 2:
                if not re.search(r"\d{4}", obj):
                    of_objects.add(obj)

        if (i + 1) % 500_000 == 0:
            click.echo(f"  ...scanned {i + 1:,} / {len(rows):,}")

    return action_nouns, of_objects


def _mine_comma_lists(conn: sqlite3.Connection, nlp) -> set:
    """Mine list items from 'X, Y, and Z' subtitles.
    Uses cheap heuristics only — tune pass cleans up later.
    """
    rows = conn.execute(
        "SELECT subtitle FROM subtitles WHERE subtitle LIKE '%, %, and %'"
    ).fetchall()

    list_re = re.compile(r"^(.+),\s+and\s+", re.IGNORECASE)
    items = set()

    for (subtitle,) in rows:
        m = list_re.match(subtitle)
        if not m:
            continue
        list_part = m.group(1)
        for piece in list_part.split(","):
            cleaned = re.sub(r"[\s]*[/:;,.]\s*$", "", piece).strip()
            if not cleaned or cleaned.lower() in _BOILERPLATE:
                continue
            words = cleaned.split()
            # Cheap filter: 1-3 words, no leading prepositions, no dates
            if not (1 <= len(words) <= 3):
                continue
            first = words[0].lower()
            if first in ("of", "in", "on", "for", "with", "from", "to", "and",
                          "or", "by", "at", "as", "including", "containing"):
                continue
            if re.search(r"\d{4}", cleaned):
                continue
            items.add(cleaned)

    return items


def build_loose_slots(conn: sqlite3.Connection):
    """Expand slots by mining the full 2.4M subtitle corpus."""
    ensure_slot_tables(conn)

    # Remove old loose fillers only
    conn.execute("DELETE FROM slot_fillers WHERE mode = 'loose'")
    conn.commit()

    click.echo("Loading spaCy model...")
    nlp = _load_nlp()

    click.echo("Mining 'the X of Y' from all subtitles (action nouns + of-objects)...")
    action_nouns, of_objects = _mine_the_x_of_y(conn, nlp)
    click.echo(f"  Found {len(action_nouns):,} action nouns, {len(of_objects):,} of-objects")

    click.echo("Mining comma-list subtitles (list items)...")
    list_items = _mine_comma_lists(conn, nlp)
    click.echo(f"  Found {len(list_items):,} list items")

    # Exclude items already in strict set
    existing = set(
        r[0] for r in conn.execute("SELECT filler FROM slot_fillers WHERE mode = 'strict'")
    )
    action_nouns -= existing
    of_objects -= existing
    list_items -= existing

    filler_rows = (
        [("list_item", x, "loose", None) for x in list_items]
        + [("action_noun", x, "loose", None) for x in action_nouns]
        + [("of_object", x, "loose", None) for x in of_objects]
    )
    conn.executemany(
        "INSERT OR IGNORE INTO slot_fillers (slot_type, filler, mode, source_subtitle_id) "
        "VALUES (?, ?, ?, ?)",
        filler_rows,
    )
    conn.commit()

    for slot_type in ["list_item", "action_noun", "of_object"]:
        strict = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict'",
            (slot_type,),
        ).fetchone()[0]
        loose = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'",
            (slot_type,),
        ).fetchone()[0]
        click.echo(f"  {slot_type}: {strict:,} strict + {loose:,} loose = {strict + loose:,} total")

    click.echo("Loose slot expansion complete!")
