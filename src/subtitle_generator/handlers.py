"""Shared API handler logic used by both serve.py (local) and function_app.py (Azure).

All handlers accept a parsed JSON body dict and return (status_code, body_dict).
The transport layer (stdlib http.server or Azure Functions) wraps these into
its own response type.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.generate import (
    GeneratedSubtitle,
    TierFilterError,
    find_source,
    generate_subtitle_matching_tiers,
)
from subtitle_generator.jacket import (
    TONE_HIGH,
    TONE_LOW,
    TONE_MEDIUM,
    build_jacket_prompt,
    generate_jacket,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
VALID_TONES = set(TONE_CHOICES.keys())


@dataclass(frozen=True)
class RatingPayload:
    subtitle: str
    thumbs: int | None
    tone_override: str | None
    free_text: str | None
    system_tone: str | None
    tags: list[str] | None
    source: str
    prompt_generated: bool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection to the subtitle DB."""
    if db_path is None:
        db_path = os.environ.get(
            "DB_PATH",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "db" / "subtitles.db"),
        )
    return sqlite3.connect(db_path)


def parse_tone(tone_str: str | None) -> set[str] | None:
    """Parse a comma-separated tone string into a set of valid tier names."""
    if not tone_str:
        return None
    tones = {t.strip().lower() for t in tone_str.split(",")}
    invalid = tones - VALID_TONES
    if invalid:
        raise ValueError(
            f"Invalid tone(s): {', '.join(invalid)}. Choose from: pop, mainstream, niche"
        )
    return tones


def build_sources(conn: sqlite3.Connection, sub: GeneratedSubtitle) -> dict:
    """Look up source books for each filler and return as a dict."""
    fillers: list[tuple[str, str, str]] = [
        ("item1", sub.item1, "list_item"),
        ("item2", sub.item2, "list_item"),
        ("action_noun", sub.action_noun, "action_noun"),
    ]

    if sub.remixed and sub.remix_parts:
        if "modifier" in sub.remix_parts:
            fillers.append(("of_modifier", sub.remix_parts["modifier"], "of_modifier"))
            fillers.append(("of_head", sub.remix_parts["head"], "of_head"))
        elif "topic" in sub.remix_parts:
            fillers.append(("of_topic", sub.remix_parts["topic"], "of_topic"))
            fillers.append(("of_complement", sub.remix_parts["complement"], "of_complement"))
    else:
        fillers.append(("of_object", sub.of_object, "of_object"))

    sources: dict[str, dict] = {}
    for key, filler, slot_type in fillers:
        result = find_source(conn, filler, slot_type)
        if result:
            desc, tag = result
            sources[key] = {"title": desc, "tag": tag}
        else:
            sources[key] = {"title": None, "tag": None}
    return sources


def subtitle_to_dict(sub: GeneratedSubtitle, sources: dict) -> dict:
    """Convert a GeneratedSubtitle + sources into the API response dict."""
    return {
        "text": sub.text,
        "item1": sub.item1,
        "item2": sub.item2,
        "action_noun": sub.action_noun,
        "of_object": sub.of_object,
        "remixed": sub.remixed,
        "remix_parts": sub.remix_parts,
        "remix_similarity": sub.remix_similarity,
        "of_article": sub.of_article,
        "action_article": sub.action_article,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# Handlers — return (status_code, body_dict)
# ---------------------------------------------------------------------------


def handle_health() -> tuple[int, dict]:
    mode = os.environ.get("SUBTITLE_GEN_MODE", "local")
    return 200, {"ok": True, "mode": mode}


def handle_generate(body: dict) -> tuple[int, dict]:
    """Generate a subtitle. Body may contain tone, remix_prob, and min_sim."""
    tone_str = body.get("tone")
    remix_prob = body.get("remix_prob")
    min_sim = body.get("min_sim")

    try:
        tone_set = parse_tone(tone_str)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    conn = get_db()
    try:
        if remix_prob is None:
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'remix_calibrated_remix_prob'"
            ).fetchone()
            remix_prob = float(row[0]) if row else 0.8
        if min_sim is None:
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'remix_calibrated_min_sim'"
            ).fetchone()
            min_sim = float(row[0]) if row else 0.1

        try:
            sub = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=tone_set,
                remix_prob=remix_prob,
                min_sim=min_sim,
            )
        except TierFilterError as exc:
            return 422, {"error": str(exc)}
        sources = build_sources(conn, sub)
        return 200, subtitle_to_dict(sub, sources)
    finally:
        conn.close()


def handle_jacket(body: dict) -> tuple[int, dict]:
    """Build jacket prompt and optionally generate. Body must contain subtitle."""
    subtitle = body.get("subtitle")
    if not subtitle or not isinstance(subtitle, str):
        return 400, {"error": "subtitle is required and must be a non-empty string"}

    model = body.get("model", "gpt-5.4-mini")
    dry_run = bool(body.get("dry_run", True))
    remix_parts = body.get("remix_parts")
    if remix_parts is not None and not isinstance(remix_parts, dict):
        return 400, {"error": "remix_parts must be an object when provided"}

    conn = get_db()
    try:
        system_prompt, user_prompt, tone_tier = build_jacket_prompt(
            subtitle,
            conn=conn,
            remix_parts=remix_parts,
        )
        prompt_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

        result_text = None
        if not dry_run:
            result_text = generate_jacket(
                subtitle,
                model=model,
                conn=conn,
                remix_parts=remix_parts,
            )

        return 200, {
            "prompt": prompt_text,
            "tone_tier": tone_tier,
            "result": result_text,
        }
    finally:
        conn.close()


def validate_rating_body(body: dict) -> tuple[int, dict, RatingPayload | None]:
    """Validate a rating request body without writing to local storage."""
    subtitle = body.get("subtitle")
    if not subtitle or not isinstance(subtitle, str):
        return 400, {"error": "subtitle is required and must be a non-empty string"}, None

    thumbs = body.get("thumbs")
    if thumbs is not None:
        if thumbs not in (1, -1):
            return 400, {"error": "thumbs must be 1 (up) or -1 (down)"}, None

    tone_override = body.get("tone_override")
    if tone_override and tone_override not in ("pop", "mainstream", "niche"):
        return 400, {"error": "tone_override must be pop, mainstream, or niche"}, None

    free_text = body.get("free_text")
    system_tone = body.get("system_tone")

    tags = body.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return 400, {"error": "tags must be an array of strings"}, None
        from subtitle_generator.feedback import VALID_RATING_TAGS, normalize_rating_tags

        tags = normalize_rating_tags(tags)
        invalid_tags = set(tags) - VALID_RATING_TAGS
        if invalid_tags:
            return 400, {"error": f"Invalid tags: {', '.join(invalid_tags)}"}, None

    source = body.get("_source", "web_user")
    if not isinstance(source, str) or not source:
        source = "web_user"

    prompt_generated = body.get("prompt_generated", False)
    if not isinstance(prompt_generated, bool):
        return 400, {"error": "prompt_generated must be a boolean"}, None

    return 200, {"status": "validated"}, RatingPayload(
        subtitle=subtitle,
        thumbs=thumbs,
        tone_override=tone_override,
        free_text=free_text if free_text else None,
        system_tone=system_tone,
        tags=tags,
        source=source,
        prompt_generated=prompt_generated,
    )


def store_rating_payload(payload: RatingPayload) -> int:
    """Store a validated rating payload in the local SQLite database."""
    from subtitle_generator.feedback import store_rating

    conn = get_db()
    try:
        return store_rating(
            conn,
            payload.subtitle,
            system_tone=payload.system_tone,
            thumbs=payload.thumbs,
            tone_override=payload.tone_override,
            free_text=payload.free_text,
            tags=payload.tags,
            source=payload.source,
            prompt_generated=payload.prompt_generated,
        )
    finally:
        conn.close()


def handle_rate(body: dict) -> tuple[int, dict]:
    """Store a human rating for a subtitle."""
    status, resp, payload = validate_rating_body(body)
    if status != 200:
        return status, resp

    assert payload is not None
    row_id = store_rating_payload(payload)
    return 200, {"id": row_id, "status": "saved"}
