"""Generate bizarre subtitles by randomly combining slot fillers."""

import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass, field

import click


@dataclass
class GeneratedSubtitle:
    """A generated subtitle with its component fillers."""
    text: str
    item1: str
    item2: str
    action_noun: str
    of_object: str
    remixed: bool = False
    remix_parts: dict = field(default_factory=dict)


def _weighted_sample(
    rows: list[tuple[str, int]], k: int,
    rng: random.Random | None = None,
    tone_target: float | None = None,
) -> list[str]:
    """Pick k unique fillers weighted by sqrt(freq), optionally biased by tone.

    When tone_target is set, applies a log-space Gaussian bias that boosts
    fillers near the target frequency and suppresses distant ones.
    tone_target is a log10 score: ~1.5 for pop, ~0.75 for mainstream, ~0.25 for niche.
    """
    fillers = [r[0] for r in rows]
    weights = [math.sqrt(r[1]) for r in rows]

    if tone_target is not None:
        spread = 0.4
        for i, (_, freq) in enumerate(rows):
            filler_score = math.log10(1 + freq)
            bias = math.exp(-((filler_score - tone_target) / spread) ** 2)
            # Aggressive blend: near-zero floor so distant fillers are strongly suppressed
            weights[i] *= (0.05 + 0.95 * bias)

    chosen = []
    # Weighted sampling without replacement
    for _ in range(k):
        pick = (rng or random).choices(fillers, weights=weights, k=1)[0]
        idx = fillers.index(pick)
        chosen.append(pick)
        fillers.pop(idx)
        weights.pop(idx)
    return chosen


# Tone target scores for filler biasing (log10 scale), per slot type.
# of_object has a much thinner pop tail, so its targets are lower.
TONE_TARGETS = {
    "pop": {"list_item": 1.5, "action_noun": 1.5, "of_object": 1.0},
    "mainstream": {"list_item": 1.0, "action_noun": 1.0, "of_object": 0.8},
    "niche": {"list_item": 0.25, "action_noun": 0.25, "of_object": 0.25},
}

# --- Remix infrastructure ---

# Module-level cache for remix context (lazy-loaded)
_remix_ctx: dict | None = None


def _load_remix_context(conn: sqlite3.Connection) -> dict:
    """Lazy-load everything needed for remixing: spaCy model, centroid, config."""
    global _remix_ctx
    if _remix_ctx is not None:
        return _remix_ctx

    import numpy as np
    import spacy

    nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])

    # Compute of-object centroid from strict fillers
    rows = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = 'of_object' AND mode = 'strict'"
    ).fetchall()
    vectors = []
    for (filler,) in rows:
        doc = nlp(filler)
        if doc.has_vector and doc.vector_norm > 0:
            vectors.append(doc.vector)
    centroid = np.mean(vectors, axis=0) if vectors else None

    # Load config
    config = {}
    for key, value in conn.execute("SELECT key, value FROM config WHERE key LIKE 'remix_%'"):
        config[key] = json.loads(value)

    _remix_ctx = {"nlp": nlp, "centroid": centroid, "config": config}
    return _remix_ctx


def _cosine_sim(v1, v2) -> float:
    import numpy as np
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _classify_for_remix(phrase: str, doc) -> tuple[str, int] | tuple[str, str, int] | None:
    """Classify an atomic of-object for remixing.

    Returns:
        ("type1", word_count) for compound NPs
        ("type2", prep, word_count) for prepositional NPs
        None if not remixable
    """
    words = phrase.split()
    word_count = len(words)
    tokens = [t for t in doc if not t.is_space]

    # 1-word or 4+ word compound: never remix
    if word_count == 1:
        return None

    # Check for preposition → Type 2
    for t in tokens:
        if t.pos_ == "ADP":
            prep = t.text.lower()
            return ("type2", prep, word_count)

    # 2-3 word compound → Type 1 (if not excluded)
    if word_count in (2, 3):
        # Check exclusions (same as _decompose_compound)
        if any(e.label_ == "PERSON" for e in doc.ents):
            return None
        roots = [t for t in tokens if t.dep_ == "ROOT"]
        if not roots or roots[0].pos_ not in ("NOUN", "PROPN"):
            return None
        if roots[0] != tokens[-1]:
            return None
        if word_count == 2 and all(t.pos_ == "NOUN" for t in tokens):
            return None
        if word_count == 2:
            for ent in doc.ents:
                if ent.label_ == "GPE" and ent.start == 0 and ent.end == len(tokens):
                    return None
        return ("type1", word_count)

    # 4+ word without prep: skip
    return None


def compose_compound(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    mode: str,
    tone_target: dict[str, float] | None,
    ctx: dict,
    word_count: int,
) -> tuple[str, dict] | None:
    """Compose a Type 1 remixed of-object (modifier + head).

    Returns (composed_text, parts_dict) or None if composition fails.
    """
    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"
    mod_word_count = word_count - 1  # head is always 1 word
    mod_space_count = mod_word_count - 1

    # Get modifier POS distribution for this bucket
    config_key = f"remix_mod_pos_{word_count}word"
    mod_pos_weights = ctx["config"].get(config_key, {})
    if not mod_pos_weights:
        return None

    # Sample a modifier POS tag
    pos_tags = list(mod_pos_weights.keys())
    pos_freqs = list(mod_pos_weights.values())
    chosen_mod_pos = (rng or random).choices(pos_tags, weights=pos_freqs, k=1)[0]

    # Draw modifier with matching POS and word count
    mod_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers "
        f"WHERE slot_type = 'of_modifier' AND pos_tag = ? "
        f"AND length(filler) - length(replace(filler, ' ', '')) = ? {mode_filter}",
        (chosen_mod_pos, mod_space_count),
    ).fetchall()
    if not mod_rows:
        return None

    # Draw head
    head_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers "
        f"WHERE slot_type = 'of_head' {mode_filter}",
    ).fetchall()
    if not head_rows:
        return None

    mod_target = tone_target.get("of_object") if tone_target else None
    modifier = _weighted_sample(mod_rows, 1, rng, mod_target)[0]
    head = _weighted_sample(head_rows, 1, rng, mod_target)[0]
    composed = f"{modifier} {head}"

    parts = {"modifier": modifier, "head": head}
    return composed, parts


def compose_prepositional(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    mode: str,
    tone_target: dict[str, float] | None,
    ctx: dict,
    prep: str,
    word_count: int,
) -> tuple[str, dict] | None:
    """Compose a Type 2 remixed of-object (topic + prep + complement).

    Returns (composed_text, parts_dict) or None if composition fails.
    Enforces strict bucket word-count matching.
    """
    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"

    topic_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers "
        f"WHERE slot_type = 'of_topic' AND prep = ? {mode_filter}",
        (prep,),
    ).fetchall()
    comp_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers "
        f"WHERE slot_type = 'of_complement' AND prep = ? {mode_filter}",
        (prep,),
    ).fetchall()
    if not topic_rows or not comp_rows:
        return None

    obj_target = tone_target.get("of_object") if tone_target else None
    topic = _weighted_sample(topic_rows, 1, rng, obj_target)[0]
    complement = _weighted_sample(comp_rows, 1, rng, obj_target)[0]
    composed = f"{topic} {prep} {complement}"

    # Strict bucket: verify word count matches
    if len(composed.split()) != word_count:
        return None

    parts = {"topic": topic, "prep": prep, "complement": complement}
    return composed, parts


def generate_subtitle(
    conn: sqlite3.Connection, seed: int | None = None, mode: str = "strict",
    tone_target: dict[str, float] | None = None,
    remix_prob: float = 0.0, min_sim: float = 0.0,
) -> GeneratedSubtitle:
    """Generate one random subtitle in the 'X, Y, and the Z of W' pattern.

    tone_target maps slot_type → log10 target score for filler biasing.
    remix_prob: probability of remixing a multi-word of-object (0.0 = never, 1.0 = always).
    min_sim: minimum cosine similarity for embedding coherence filter.
    """
    rng = None
    if seed is not None:
        rng = random.Random(seed)

    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"

    list_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers WHERE slot_type = 'list_item' {mode_filter}"
    ).fetchall()
    action_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers WHERE slot_type = 'action_noun' {mode_filter}"
    ).fetchall()
    obj_rows = conn.execute(
        f"SELECT filler, freq FROM slot_fillers WHERE slot_type = 'of_object' {mode_filter}"
    ).fetchall()

    if len(list_rows) < 2 or not action_rows or not obj_rows:
        return GeneratedSubtitle(
            text="(not enough fillers — run 'build-slots' first)",
            item1="", item2="", action_noun="", of_object="",
        )

    list_target = tone_target.get("list_item") if tone_target else None
    action_target = tone_target.get("action_noun") if tone_target else None
    obj_target = tone_target.get("of_object") if tone_target else None

    items = _weighted_sample(list_rows, 2, rng, list_target)
    action_noun = _weighted_sample(action_rows, 1, rng, action_target)[0]
    of_object = _weighted_sample(obj_rows, 1, rng, obj_target)[0]

    remixed = False
    remix_parts = {}

    # Remix attempt: if multi-word and coin flip succeeds
    if remix_prob > 0 and len(of_object.split()) >= 2:
        should_remix = (rng or random).random() < remix_prob
        if should_remix:
            result = _try_remix(conn, rng, mode, tone_target, of_object, min_sim)
            if result:
                of_object, remix_parts = result
                remixed = True

    return GeneratedSubtitle(
        text=f"{items[0]}, {items[1]}, and the {action_noun} of {of_object}",
        item1=items[0],
        item2=items[1],
        action_noun=action_noun,
        of_object=of_object,
        remixed=remixed,
        remix_parts=remix_parts,
    )


def _try_remix(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    mode: str,
    tone_target: dict[str, float] | None,
    original_of_object: str,
    min_sim: float,
    max_retries: int = 5,
) -> tuple[str, dict] | None:
    """Attempt to remix an of-object. Returns (composed_text, parts) or None."""
    ctx = _load_remix_context(conn)
    nlp = ctx["nlp"]
    centroid = ctx["centroid"]

    doc = nlp(original_of_object)
    classification = _classify_for_remix(original_of_object, doc)
    if classification is None:
        return None

    for _ in range(max_retries):
        if classification[0] == "type1":
            _, word_count = classification
            result = compose_compound(conn, rng, mode, tone_target, ctx, word_count)
        else:
            _, prep, word_count = classification
            result = compose_prepositional(conn, rng, mode, tone_target, ctx, prep, word_count)

        if result is None:
            continue

        composed, parts = result

        # Embedding coherence check
        if centroid is not None and min_sim > 0:
            composed_doc = nlp(composed)
            if composed_doc.has_vector and composed_doc.vector_norm > 0:
                sim = _cosine_sim(centroid, composed_doc.vector)
                if sim < min_sim:
                    continue

        return composed, parts

    return None


def find_source(conn: sqlite3.Connection, filler: str, slot_type: str = "of_object") -> tuple[str, str] | None:
    """Find the real book a slot filler was extracted from.

    First tries the exact source_subtitle_id linkage from slot_fillers,
    then falls back to a random LIKE search.
    Returns (description, source_tag) where source_tag is 'LOC' or 'OL'.
    """
    # Try exact source via slot_fillers → subtitles join (scoped to slot_type)
    row = conn.execute(
        "SELECT s.title, s.subtitle, s.source_file "
        "FROM slot_fillers sf "
        "JOIN subtitles s ON s.id = sf.source_subtitle_id "
        "WHERE sf.filler = ? AND sf.slot_type = ? AND sf.source_subtitle_id IS NOT NULL "
        "LIMIT 1",
        (filler, slot_type),
    ).fetchone()

    # Fallback: substring search (for loose fillers without source linkage)
    if not row:
        escaped = filler.replace("'", "''")
        row = conn.execute(
            "SELECT title, subtitle, source_file FROM subtitles "
            f"WHERE subtitle LIKE '%{escaped}%' ORDER BY RANDOM() LIMIT 1"
        ).fetchone()
    if row:
        title = (row[0] or "").strip().rstrip(" /:")
        subtitle = (row[1] or "").strip().rstrip(" /:")
        source_file = row[2] or ""
        tag = "OL" if source_file == "openlibrary" else "LOC"
        desc = f"{title}: {subtitle}" if title and subtitle else (title or subtitle)
        return desc, tag
    return None


def format_sources(conn: sqlite3.Connection, sub: GeneratedSubtitle) -> str:
    """Look up source books for each filler and format as markdown."""
    fillers = [
        ("List item 1", sub.item1, "list_item"),
        ("List item 2", sub.item2, "list_item"),
        ("Action noun", sub.action_noun, "action_noun"),
    ]

    if sub.remixed and sub.remix_parts:
        # Show individual remix parts
        if "modifier" in sub.remix_parts:
            fillers.append(("Of-modifier", sub.remix_parts["modifier"], "of_modifier"))
            fillers.append(("Of-head", sub.remix_parts["head"], "of_head"))
        elif "topic" in sub.remix_parts:
            fillers.append(("Of-topic", sub.remix_parts["topic"], "of_topic"))
            fillers.append(("Of-complement", sub.remix_parts["complement"], "of_complement"))
    else:
        fillers.append(("Of-object", sub.of_object, "of_object"))

    lines = ["", "---", "**Sources:**"]
    for label, filler, slot_type in fillers:
        result = find_source(conn, filler, slot_type)
        if result:
            desc, tag = result
            lines.append(f"- *{label}* \"{filler}\" ← [{tag}] {desc}")
        else:
            lines.append(f"- *{label}* \"{filler}\" ← (source not found)")
    if sub.remixed:
        lines.append(f"- *(remixed from: \"{sub.of_object}\")*")
    return "\n".join(lines)


def slot_stats(conn: sqlite3.Connection, mode: str = "strict") -> dict:
    """Get counts per slot type for a given mode."""
    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"
    rows = conn.execute(
        f"SELECT slot_type, COUNT(*) FROM slot_fillers WHERE 1=1 {mode_filter} GROUP BY slot_type"
    ).fetchall()
    return dict(rows)
