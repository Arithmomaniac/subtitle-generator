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


def find_source(conn: sqlite3.Connection, filler: str) -> str | None:
    """Find a real book (title + subtitle) that contains this filler text."""
    # Search for the filler as a substring in subtitles (case-insensitive)
    escaped = filler.replace("'", "''")
    row = conn.execute(
        "SELECT title, subtitle FROM subtitles "
        f"WHERE subtitle LIKE '%{escaped}%' LIMIT 1"
    ).fetchone()
    if row:
        title = (row[0] or "").strip().rstrip(" /:")
        subtitle = (row[1] or "").strip().rstrip(" /:")
        if title and subtitle:
            return f"{title}: {subtitle}"
        return title or subtitle
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
        source = find_source(conn, filler)
        if source:
            lines.append(f"- *{label}* \"{filler}\" ← {source}")
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
