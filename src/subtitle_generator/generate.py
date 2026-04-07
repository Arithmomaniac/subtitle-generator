"""Generate bizarre subtitles by randomly combining slot fillers."""

import random
import re
import sqlite3
from dataclasses import dataclass

import click


@dataclass
class GeneratedSubtitle:
    """A generated subtitle with its component fillers."""
    text: str
    item1: str
    item2: str
    action_noun: str
    of_object: str


def generate_subtitle(
    conn: sqlite3.Connection, seed: int | None = None, mode: str = "strict"
) -> GeneratedSubtitle:
    """Generate one random subtitle in the 'X, Y, and the Z of W' pattern."""
    if seed is not None:
        random.seed(seed)

    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"

    list_items = conn.execute(
        f"SELECT filler FROM slot_fillers WHERE slot_type = 'list_item' {mode_filter} ORDER BY RANDOM() LIMIT 2"
    ).fetchall()
    action = conn.execute(
        f"SELECT filler FROM slot_fillers WHERE slot_type = 'action_noun' {mode_filter} ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    obj = conn.execute(
        f"SELECT filler FROM slot_fillers WHERE slot_type = 'of_object' {mode_filter} ORDER BY RANDOM() LIMIT 1"
    ).fetchone()

    if len(list_items) < 2 or not action or not obj:
        return GeneratedSubtitle(
            text="(not enough fillers — run 'build-slots' first)",
            item1="", item2="", action_noun="", of_object="",
        )

    item1 = list_items[0][0]
    item2 = list_items[1][0]
    action_noun = action[0]
    of_object = obj[0]

    return GeneratedSubtitle(
        text=f"{item1}, {item2}, and the {action_noun} of {of_object}",
        item1=item1,
        item2=item2,
        action_noun=action_noun,
        of_object=of_object,
    )


def find_source(conn: sqlite3.Connection, filler: str) -> tuple[str, str] | None:
    """Find the real book a slot filler was extracted from.

    First tries the exact source_subtitle_id linkage from slot_fillers,
    then falls back to a random LIKE search.
    Returns (description, source_tag) where source_tag is 'LOC' or 'OL'.
    """
    # Try exact source via slot_fillers → subtitles join
    row = conn.execute(
        "SELECT s.title, s.subtitle, s.source_file "
        "FROM slot_fillers sf "
        "JOIN subtitles s ON s.id = sf.source_subtitle_id "
        "WHERE sf.filler = ? AND sf.source_subtitle_id IS NOT NULL "
        "LIMIT 1",
        (filler,),
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
        ("List item 1", sub.item1),
        ("List item 2", sub.item2),
        ("Action noun", sub.action_noun),
        ("Of-object", sub.of_object),
    ]
    lines = ["", "---", "**Sources:**"]
    for label, filler in fillers:
        result = find_source(conn, filler)
        if result:
            desc, tag = result
            lines.append(f"- *{label}* \"{filler}\" ← [{tag}] {desc}")
        else:
            lines.append(f"- *{label}* \"{filler}\" ← (source not found)")
    return "\n".join(lines)


def slot_stats(conn: sqlite3.Connection, mode: str = "strict") -> dict:
    """Get counts per slot type for a given mode."""
    mode_filter = "" if mode == "loose" else "AND mode = 'strict'"
    rows = conn.execute(
        f"SELECT slot_type, COUNT(*) FROM slot_fillers WHERE 1=1 {mode_filter} GROUP BY slot_type"
    ).fetchall()
    return dict(rows)
