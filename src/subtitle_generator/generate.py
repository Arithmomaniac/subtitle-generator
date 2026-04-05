"""Generate bizarre subtitles by randomly combining slot fillers."""

import random
import sqlite3

import click


def generate_subtitle(conn: sqlite3.Connection, seed: int | None = None) -> str:
    """Generate one random subtitle in the 'X, Y, and the Z of W' pattern."""
    if seed is not None:
        random.seed(seed)

    # Pick two random list items
    list_items = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = 'list_item' ORDER BY RANDOM() LIMIT 2"
    ).fetchall()
    # Pick one action noun
    action = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = 'action_noun' ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    # Pick one of-object
    obj = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = 'of_object' ORDER BY RANDOM() LIMIT 1"
    ).fetchone()

    if len(list_items) < 2 or not action or not obj:
        return "(not enough fillers — run 'build-slots' first)"

    item1 = list_items[0][0]
    item2 = list_items[1][0]
    action_noun = action[0]
    of_object = obj[0]

    return f"{item1}, {item2}, and the {action_noun} of {of_object}"


def slot_stats(conn: sqlite3.Connection) -> dict:
    """Get counts per slot type."""
    rows = conn.execute(
        "SELECT slot_type, COUNT(*) FROM slot_fillers GROUP BY slot_type"
    ).fetchall()
    return dict(rows)
