"""Shared runtime eligibility rules for strict filler support."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict

_BAD_ONE_WORD_OBJECTS = frozenset({"christian", "imf"})
_BAD_STANDALONE_FILLERS = frozenset({"jr", "sr", "xcalibur"})


def normalize_filler(filler: str) -> str:
    return " ".join(filler.split()).casefold()


def filler_key(slot_type: str, filler: str) -> str:
    return f"{slot_type}\0{normalize_filler(filler)}"


def is_runtime_eligible_strict_filler(slot_type: str, filler: str) -> bool:
    lower = normalize_filler(filler)
    if not lower:
        return False
    if lower in _BAD_STANDALONE_FILLERS:
        return False
    if re.search(r",\s*(?:jr|sr)\.?$", lower):
        return False
    if slot_type == "of_object" and lower in _BAD_ONE_WORD_OBJECTS:
        return False
    if re.search(r"(?:[a-z]\.){2,}[a-z]", lower):
        return False
    return True


def filter_runtime_eligible_rows(
    slot_type: str,
    rows: list[tuple],
    *,
    filler_index: int = 0,
) -> list[tuple]:
    return [
        row
        for row in rows
        if is_runtime_eligible_strict_filler(slot_type, str(row[filler_index]))
    ]


def load_runtime_eligible_strict_fillers(
    conn: sqlite3.Connection,
) -> dict[str, set[str]]:
    rows = conn.execute(
        """
        SELECT slot_type, filler
        FROM slot_fillers
        WHERE mode = 'strict'
        ORDER BY slot_type, filler
        """
    ).fetchall()
    eligible: dict[str, set[str]] = defaultdict(set)
    for slot_type_raw, filler_raw in rows:
        slot_type = str(slot_type_raw)
        filler = str(filler_raw)
        if is_runtime_eligible_strict_filler(slot_type, filler):
            eligible[slot_type].add(filler)
    return dict(eligible)


def load_runtime_eligible_strict_filler_keys(
    conn: sqlite3.Connection,
) -> dict[str, set[str]]:
    eligible = load_runtime_eligible_strict_fillers(conn)
    return {
        slot_type: {filler_key(slot_type, filler) for filler in fillers}
        for slot_type, fillers in eligible.items()
    }
