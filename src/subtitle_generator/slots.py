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

# Generic/bland words that don't carry pop-nonfiction punch
_WEAK_FILLERS = {
    "things", "factors", "indicators", "regions", "aspects", "elements",
    "issues", "items", "matters", "topics", "areas", "units", "features",
    "conditions", "situations", "circumstances", "parameters", "variables",
    "data", "results", "outcomes", "findings", "processes", "procedures",
    "methods", "techniques", "approaches", "activities", "operations",
    "developments", "achievements", "productions", "perceptions",
}

# MARC catalog / library-science terms that leak from the data source
_MARC_JARGON = {
    "bibliographies", "bibliography", "monograph", "monographs",
    "proceedings", "dissertations", "yearbook", "yearbooks",
    "catalog", "catalogue", "catalogues", "catalogs",
    "supplements", "appendix", "appendices", "abstracts",
    "periodicals", "serials", "pamphlets", "leaflets",
    "lead ores", "defects", "specimens", "reagents",
}


def _normalize_spacing(phrase: str) -> str:
    """Fix common spacing artifacts from MARC data extraction.
    'U. S.' → 'U.S.', 'D. C' → 'D.C.', etc.
    """
    # Fix spaced-out initials: "U. S." → "U.S."
    phrase = re.sub(r"\b([A-Z])\.\s+([A-Z])\.?", r"\1.\2.", phrase)
    # Fix spaced-out initials without trailing dot: "D. C" → "D.C."
    phrase = re.sub(r"\b([A-Z])\.\s+([A-Z])\b", r"\1.\2.", phrase)
    return phrase.strip()


def _has_encoding_artifacts(phrase: str) -> bool:
    """Reject fillers with mojibake or non-ASCII noise from MARC data."""
    # Common mojibake characters: ¿, Ã, Â, Æ, ï¿½, replacement char
    if re.search(r"[¿\ufffd\u00c3\u00c2\u00c6]", phrase):
        return True
    # Non-ASCII that isn't common diacritics (é, ñ, ü, etc.)
    for ch in phrase:
        if ord(ch) > 127:
            # Allow common Latin diacritics (À-ÿ range minus control chars)
            if not ("\u00c0" <= ch <= "\u024f" or ch in "\u2013\u2014"):
                return True
    return False


def _is_truncated(phrase: str) -> bool:
    """Detect fillers cut off mid-word (e.g. 'Independent Fil')."""
    words = phrase.split()
    if not words:
        return False
    last = words[-1]
    # Short lowercase fragment at end (2-3 chars, not a real word pattern)
    if len(last) <= 3 and last[0].isupper() and not last.isupper():
        # Likely a truncated word like "Fil", "Rel", "Gov"
        # Allow real short words by checking if it looks like a fragment
        if len(last) <= 2 and len(words) > 1:
            return True
    # Ends with a lone capital letter (truncation artifact)
    if len(last) == 1 and last.isupper() and len(words) > 1:
        return True
    return False


def _is_weak_or_jargon(phrase: str) -> bool:
    """Reject generic abstractions and MARC catalog jargon."""
    lower = phrase.lower().strip()
    if lower in _WEAK_FILLERS or lower in _MARC_JARGON:
        return True
    # Also check individual words for MARC jargon in multi-word phrases
    for word in lower.split():
        if word in _MARC_JARGON:
            return True
    return False

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
    return spacy.load("en_core_web_sm", disable=["lemmatizer"])


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
    # Orthographic checks (can't express via POS)
    if any(_is_all_caps_noise(w) for w in phrase.split()):
        return False
    head = words[-1]
    if head in ACTION_WHITELIST:
        return True
    if any(head.endswith(s) for s in ACTION_SUFFIXES):
        return True
    # Check lemma via spaCy
    doc = nlp(phrase)
    # Reject non-content tokens (quotes, parens, etc.)
    if any(t.pos_ == "PUNCT" and t.text != "-" for t in doc):
        return False
    if doc:
        lemma = doc[-1].lemma_.lower()
        if lemma in ACTION_WHITELIST:
            return True
        if any(lemma.endswith(s) for s in ACTION_SUFFIXES):
            return True
    return False


def _is_all_caps_noise(word: str) -> bool:
    """Reject full all-caps words 5+ letters (catalog noise like GOLF, SAMUEL).
    Allow short acronyms (CIA, NBA, BBC) — they're real pop-nonfiction terms."""
    return word.isupper() and len(word) >= 5


# POS tags allowed in list items: only content words (+ hyphens)
_LIST_ITEM_ALLOWED_POS = {"NOUN", "PROPN", "ADJ"}


def _is_valid_list_item(phrase: str, nlp) -> bool:
    """List items should be punchy nouns/names, 1-3 words."""
    words = phrase.split()
    if not (1 <= len(words) <= 3):
        return False
    # Orthographic checks (can't express via POS)
    if re.search(r"\d{4}", phrase):
        return False
    if any(_is_all_caps_noise(w) for w in words):
        return False
    doc = nlp(phrase)
    # Every token must be a content word (allow hyphens as connecting punctuation)
    for t in doc:
        if t.pos_ not in _LIST_ITEM_ALLOWED_POS and not (t.pos_ == "PUNCT" and t.text == "-"):
            return False
    # Must contain at least one real noun
    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return False
    return True


# POS tags rejected in of-objects (more permissive than list items since
# prepositional modifiers like "knowledge in Late Antiquity" are valid)
_OF_OBJECT_REJECTED_POS = {"DET", "CCONJ", "SCONJ", "VERB", "AUX", "PRON", "INTJ", "X"}


def _is_valid_object(phrase: str, nlp) -> bool:
    """The 'of X' part should be a meaningful NP, 1-5 words."""
    words = phrase.split()
    if not (1 <= len(words) <= 5):
        return False
    # Orthographic checks (can't express via POS)
    if re.search(r"\d{4}", phrase):
        return False
    if any(_is_all_caps_noise(w) for w in words):
        return False
    doc = nlp(phrase)
    for t in doc:
        if t.pos_ in _OF_OBJECT_REJECTED_POS:
            return False
        if t.pos_ == "PUNCT" and t.text != "-":
            return False
    # Must contain at least one real noun
    if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
        return False
    return True


# --- Of-object decomposition ---

def _decompose_compound(phrase: str, doc) -> tuple[str, str, str, str] | None:
    """Decompose a 2-3 word compound NP into (modifier, modifier_pos, head, head_pos).

    Uses dependency parsing: ROOT = head noun, everything before ROOT = modifier.
    Returns None if phrase is excluded or doesn't fit the pattern.
    """
    tokens = [t for t in doc if not t.is_space]
    words = phrase.split()
    if len(words) not in (2, 3):
        return None

    # Exclusion: PERSON entity
    if any(e.label_ == "PERSON" for e in doc.ents):
        return None

    # Find ROOT
    roots = [t for t in tokens if t.dep_ == "ROOT"]
    if not roots:
        return None
    root = roots[0]

    # ROOT must be a noun
    if root.pos_ not in ("NOUN", "PROPN"):
        return None

    # ROOT must be the last content token
    if root != tokens[-1]:
        return None

    # Exclusion: HEAD is NUM/ordinal
    if root.pos_ == "NUM" or root.text in ("I", "II", "III", "IV", "V"):
        return None

    modifier_tokens = tokens[:-1]
    if not modifier_tokens:
        return None

    # Exclusion: full-phrase GPE (e.g., "New York" as a 2-word of-object)
    if len(words) == 2:
        for ent in doc.ents:
            if ent.label_ == "GPE" and ent.start == 0 and ent.end == len(tokens):
                return None

    # Exclusion: NOUN+NOUN for 2-word (compound nouns not freely composable)
    if len(words) == 2 and all(t.pos_ == "NOUN" for t in tokens):
        return None

    modifier = " ".join(t.text for t in modifier_tokens)
    modifier_pos = "+".join(t.pos_ for t in modifier_tokens)
    head = root.text
    head_pos = root.pos_

    # Reconstruction guard: modifier + head must equal original tokens
    reconstructed = " ".join(t.text for t in modifier_tokens) + " " + head
    original = " ".join(t.text for t in tokens)
    if reconstructed != original:
        return None

    return modifier, modifier_pos, head, head_pos


def _decompose_prepositional(phrase: str, doc) -> tuple[str, str, str, str, str] | None:
    """Decompose a prepositional NP into (topic, topic_pos, prep, complement, complement_pos).

    Split at first ADP: everything before = topic, the prep, everything after = complement.
    Returns None if no valid split found.
    """
    tokens = [t for t in doc if not t.is_space]

    # Find first ADP
    adp_idx = None
    for i, t in enumerate(tokens):
        if t.pos_ == "ADP":
            adp_idx = i
            break
    if adp_idx is None:
        return None

    topic_tokens = tokens[:adp_idx]
    prep_token = tokens[adp_idx]
    complement_tokens = tokens[adp_idx + 1:]

    # Both sides must be non-empty and have at least one noun
    if not topic_tokens or not complement_tokens:
        return None
    if not any(t.pos_ in ("NOUN", "PROPN") for t in topic_tokens):
        return None
    if not any(t.pos_ in ("NOUN", "PROPN") for t in complement_tokens):
        return None

    topic = " ".join(t.text for t in topic_tokens)
    topic_pos = "+".join(t.pos_ for t in topic_tokens)
    prep = prep_token.text.lower()
    complement = " ".join(t.text for t in complement_tokens)
    complement_pos = "+".join(t.pos_ for t in complement_tokens)

    return topic, topic_pos, prep, complement, complement_pos


def extract_pattern_matches(conn: sqlite3.Connection) -> list[dict]:
    """Find all subtitles matching X, Y, and the Z of W.

    Open Library records without any institutional identifier (ISBN, LCCN)
    are excluded — they tend to be dissertations, pamphlets, or government
    reports whose language doesn't reflect real published-book subtitles.
    """
    rows = conn.execute(
        "SELECT id, title, subtitle FROM subtitles "
        "WHERE subtitle LIKE '%, % and the % of %' "
        "AND NOT ("
        "  source_file = 'openlibrary'"
        "  AND (isbn IS NULL OR isbn = '')"
        "  AND (lccn IS NULL OR lccn = '')"
        ")"
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
            pos_tag TEXT,
            prep TEXT,
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    # Migration: add columns if missing (pre-existing databases)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slot_fillers)")}
    if "freq" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN freq INTEGER NOT NULL DEFAULT 1")
    if "pos_tag" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN pos_tag TEXT")
    if "prep" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN prep TEXT")
    if "remix_type" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN remix_type TEXT")
    if "remix_prep" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN remix_prep TEXT")
    if "remix_word_count" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN remix_word_count INTEGER")
    if "vector_sum" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN vector_sum BLOB")
    if "token_count" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN token_count INTEGER")
    if "centroid_dot" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN centroid_dot REAL")
    if "norm_sq" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN norm_sq REAL")
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
        action = _normalize_spacing(m["action_noun"])
        obj = _normalize_spacing(m["of_object"])

        if _has_encoding_artifacts(action) or _is_truncated(action) or _is_weak_or_jargon(action):
            continue
        if not _is_valid_action(action, nlp):
            continue
        if _has_encoding_artifacts(obj) or _is_truncated(obj) or _is_weak_or_jargon(obj):
            continue
        if not _is_valid_object(obj, nlp):
            continue

        valid_items = []
        for item in m["list_items"]:
            cleaned = re.sub(r"[\s]*[/:;,.]\s*$", "", item).strip()
            cleaned = _normalize_spacing(cleaned)
            if not cleaned or _has_encoding_artifacts(cleaned) or _is_truncated(cleaned) or _is_weak_or_jargon(cleaned):
                continue
            if _is_valid_list_item(cleaned, nlp):
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

    # --- Decompose of-objects into sub-parts for remixing ---
    _decompose_of_objects(conn, nlp, of_objects_seen)


def _case_merge_key(filler: str) -> str:
    """Key for case-insensitive merging."""
    return filler.lower()


def _decompose_of_objects(
    conn: sqlite3.Connection,
    nlp,
    of_objects_seen: dict[str, tuple[int, int]],
):
    """Decompose validated of-objects into sub-parts for remixing.

    Type 1 (compound): 2-3 word NPs without prepositions → of_modifier + of_head
    Type 2 (prepositional): NPs with a preposition → of_topic + of_complement
    """
    click.echo("\nDecomposing of-objects for remixing...")

    # Accumulate sub-parts: key → {filler: (source_sid, freq, pos_tag, prep)}
    # Using case-insensitive merge: lower(filler) → (canonical_filler, source_sid, total_freq, pos_tag, prep)
    modifiers: dict[str, list] = {}   # lower → [canonical, sid, freq, pos_tag]
    heads: dict[str, list] = {}
    topics: dict[str, list] = {}      # lower → [canonical, sid, freq, pos_tag, prep]
    complements: dict[str, list] = {}

    # Batch parse all of-objects for decomposition
    of_objects = list(of_objects_seen.keys())
    of_meta = {obj: of_objects_seen[obj] for obj in of_objects}
    docs = list(nlp.pipe(of_objects, batch_size=500))

    type1_count = 0
    type2_count = 0

    for obj, doc in zip(of_objects, docs):
        sid, freq = of_meta[obj]
        words = obj.split()

        # Try prepositional decomposition first (any word count with ADP)
        prep_result = _decompose_prepositional(obj, doc)
        if prep_result:
            topic, topic_pos, prep, complement, complement_pos = prep_result
            type2_count += 1

            # Accumulate topic (case-insensitive merge, sum freq)
            tk = _case_merge_key(topic)
            if tk in topics:
                existing = topics[tk]
                existing[2] += freq  # sum freq
            else:
                topics[tk] = [topic, sid, freq, topic_pos, prep]

            # Accumulate complement
            ck = _case_merge_key(complement)
            if ck in complements:
                existing = complements[ck]
                existing[2] += freq
            else:
                complements[ck] = [complement, sid, freq, complement_pos, prep]
            continue

        # Try compound decomposition (2-3 words, no prep)
        if len(words) in (2, 3):
            comp_result = _decompose_compound(obj, doc)
            if comp_result:
                modifier, modifier_pos, head, head_pos = comp_result
                type1_count += 1

                # Accumulate modifier (case-insensitive merge, sum freq)
                mk = _case_merge_key(modifier)
                if mk in modifiers:
                    existing = modifiers[mk]
                    existing[2] += freq
                else:
                    modifiers[mk] = [modifier, sid, freq, modifier_pos]

                # Accumulate head
                hk = _case_merge_key(head)
                if hk in heads:
                    existing = heads[hk]
                    existing[2] += freq
                else:
                    heads[hk] = [head, sid, freq, head_pos]

    click.echo(f"  Type 1 (compound): {type1_count:,} of-objects decomposed")
    click.echo(f"  Type 2 (prepositional): {type2_count:,} of-objects decomposed")

    # Insert decomposed sub-parts
    rows = []
    for data in modifiers.values():
        canonical, sid, freq, pos_tag = data
        rows.append(("of_modifier", canonical, "strict", sid, freq, pos_tag, None))
    for data in heads.values():
        canonical, sid, freq, pos_tag = data
        rows.append(("of_head", canonical, "strict", sid, freq, pos_tag, None))
    for data in topics.values():
        canonical, sid, freq, pos_tag, prep = data
        rows.append(("of_topic", canonical, "strict", sid, freq, pos_tag, prep))
    for data in complements.values():
        canonical, sid, freq, pos_tag, prep = data
        rows.append(("of_complement", canonical, "strict", sid, freq, pos_tag, prep))

    conn.executemany(
        "INSERT OR IGNORE INTO slot_fillers "
        "(slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()

    # Report counts
    for slot_type in ["of_modifier", "of_head", "of_topic", "of_complement"]:
        count = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ?", (slot_type,)
        ).fetchone()[0]
        click.echo(f"  {slot_type}: {count:,} unique fillers")

    # Store POS pattern distributions in config table for generation
    _store_decomposition_config(conn)

    click.echo("Of-object decomposition complete!")


def _store_decomposition_config(conn: sqlite3.Connection):
    """Compute and store POS pattern distributions and prep group sizes."""
    # Clear stale config keys
    conn.execute("DELETE FROM config WHERE key LIKE 'remix_%'")

    # Type 1: POS pattern distributions by output word count
    # We need modifier_pos + head_pos patterns, grouped by total word count.
    # Since modifiers and heads are stored separately, we reconstruct the pattern
    # from the modifier's pos_tag (which encodes the full modifier POS, e.g. "ADJ" or
    # "PROPN+PROPN") combined with the head's pos_tag.
    # We compute this from the of_modifier rows grouped by pos_tag and word count.
    for bucket in (2, 3):
        mod_word_count = bucket - 1  # head is always 1 word
        mod_space_count = mod_word_count - 1
        rows = conn.execute(
            "SELECT pos_tag, SUM(freq) as total_freq "
            "FROM slot_fillers "
            "WHERE slot_type = 'of_modifier' "
            "AND length(filler) - length(replace(filler, ' ', '')) = ? "
            "GROUP BY pos_tag ORDER BY total_freq DESC",
            (mod_space_count,),
        ).fetchall()
        # Also get head POS distribution
        head_rows = conn.execute(
            "SELECT pos_tag, SUM(freq) as total_freq "
            "FROM slot_fillers WHERE slot_type = 'of_head' "
            "GROUP BY pos_tag ORDER BY total_freq DESC"
        ).fetchall()
        if rows:
            mod_patterns = {pat: freq for pat, freq in rows}
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (f"remix_mod_pos_{bucket}word", json.dumps(mod_patterns)),
            )
        if head_rows:
            head_patterns = {pat: freq for pat, freq in head_rows}
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                ("remix_head_pos", json.dumps(head_patterns)),
            )

    # Type 2: prep group sizes
    rows = conn.execute(
        "SELECT prep, COUNT(*) FROM slot_fillers "
        "WHERE slot_type = 'of_topic' GROUP BY prep ORDER BY COUNT(*) DESC"
    ).fetchall()
    if rows:
        prep_groups = {prep: count for prep, count in rows}
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            ("remix_prep_groups", json.dumps(prep_groups)),
        )

    conn.commit()

