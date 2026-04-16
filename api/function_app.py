"""Azure Functions v2 (Python) app wrapping subtitle-generator."""

import json
import os
import sqlite3
import sys
from pathlib import Path

# Add the src directory so subtitle_generator is importable (local dev)
# In Azure, subtitle_generator/ is copied alongside function_app.py
_src_path = Path(__file__).parent.parent / "src"
if _src_path.is_dir():
    sys.path.insert(0, str(_src_path))

import azure.functions as func

from subtitle_generator.config import get_tone_targets
from subtitle_generator.generate import (
    GeneratedSubtitle,
    find_source,
    generate_subtitle,
)
from subtitle_generator.jacket import (
    TONE_HIGH,
    TONE_LOW,
    TONE_MEDIUM,
    build_jacket_prompt,
    generate_jacket,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

_TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
_VALID_TONES = set(_TONE_CHOICES.keys())
_VALID_LOCK_KEYS = {
    "item1", "item2", "action_noun", "of_object",
    "of_modifier", "of_head", "of_topic", "of_complement",
}


def _get_db() -> sqlite3.Connection:
    db_path = os.environ.get(
        "DB_PATH",
        str(Path(__file__).parent.parent / "data" / "db" / "subtitles.db"),
    )
    return sqlite3.connect(db_path)


def _json_response(body: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body),
        status_code=status_code,
        mimetype="application/json",
    )


def _error(msg: str, status_code: int = 400) -> func.HttpResponse:
    return _json_response({"error": msg}, status_code)


def _parse_tone(tone_str: str | None) -> set[str] | None:
    """Parse a comma-separated tone string into a set of valid tier names."""
    if not tone_str:
        return None
    tones = {t.strip().lower() for t in tone_str.split(",")}
    invalid = tones - _VALID_TONES
    if invalid:
        raise ValueError(
            f"Invalid tone(s): {', '.join(invalid)}. Choose from: pop, mainstream, niche"
        )
    return tones


def _build_sources(conn: sqlite3.Connection, sub: GeneratedSubtitle) -> dict:
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


# ── POST /api/generate ──────────────────────────────────────────────


@app.route(route="generate", methods=["POST"])
def generate(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        tone_str = body.get("tone")
        remix_prob = body.get("remix_prob")
        min_sim = body.get("min_sim")
        locks = body.get("locks")

        # Validate tone
        try:
            tone_set = _parse_tone(tone_str)
        except ValueError as exc:
            return _error(str(exc))

        # Validate locks
        if locks is not None:
            if not isinstance(locks, dict):
                return _error("locks must be an object mapping slot keys to values")
            invalid_keys = set(locks.keys()) - _VALID_LOCK_KEYS
            if invalid_keys:
                return _error(f"Invalid lock keys: {', '.join(invalid_keys)}")

        conn = _get_db()
        try:
            # Resolve remix defaults from DB config (same logic as cli.py)
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

            # Build tone target
            tone_target = None
            if tone_set:
                targets = get_tone_targets(conn)
                merged: dict[str, float] = {}
                for slot in ("list_item", "action_noun", "of_object"):
                    merged[slot] = sum(targets[t][slot] for t in tone_set) / len(tone_set)
                tone_target = merged

            sub = generate_subtitle(
                conn,
                tone_target=tone_target,
                remix_prob=remix_prob,
                min_sim=min_sim,
                locks=locks,
            )

            sources = _build_sources(conn, sub)

            result = {
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
            return _json_response(result)
        finally:
            conn.close()

    except ValueError as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Internal error: {exc}", 500)


# ── POST /api/jacket ────────────────────────────────────────────────


@app.route(route="jacket", methods=["POST"])
def jacket(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            return _error("Request body must be valid JSON")

        subtitle = body.get("subtitle")
        if not subtitle or not isinstance(subtitle, str):
            return _error("subtitle is required and must be a non-empty string")

        model = body.get("model", "gpt-5.4-mini")
        dry_run = bool(body.get("dry_run", True))

        conn = _get_db()
        try:
            system_prompt, user_prompt, tone_tier = build_jacket_prompt(subtitle, conn=conn)
            prompt_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

            result_text = None
            if not dry_run:
                result_text = generate_jacket(
                    subtitle,
                    model=model,
                    conn=conn,
                )

            return _json_response({
                "prompt": prompt_text,
                "tone_tier": tone_tier,
                "result": result_text,
            })
        finally:
            conn.close()

    except Exception as exc:
        return _error(f"Internal error: {exc}", 500)


# ── GET /api/health ─────────────────────────────────────────────────


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    mode = os.environ.get("SUBTITLE_GEN_MODE", "local")
    return _json_response({"ok": True, "mode": mode})
