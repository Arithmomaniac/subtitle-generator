"""Build ISBN normalization and filler↔source mapping tables.

Creates:
1. isbn_aliases — canonical ISBN-13 + OL work_key for every ISBN in our DB
2. slot_filler_sources — many-to-many mapping of fillers to ALL source subtitles
"""

import json
import re
import sqlite3
import time
import argparse
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("data/db/subtitles.db")
OL_LOOKUP_PATH = Path("data/ol_edition_lookup.json")


def isbn10_to_isbn13(isbn10: str) -> str | None:
    """Convert ISBN-10 to ISBN-13."""
    isbn10 = re.sub(r"[\s-]", "", isbn10)
    if len(isbn10) != 10:
        return None
    # ISBN-10 last char can be X (=10), but we only need first 9 digits for conversion
    base = "978" + isbn10[:9]
    if not base.isdigit():
        return None
    total = sum(int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(base))
    check = (10 - (total % 10)) % 10
    return base + str(check)


def normalize_isbn(raw: str) -> str:
    """Strip hyphens/spaces, normalize to ISBN-13 if possible."""
    clean = re.sub(r"[\s-]", "", raw.strip())
    if not clean:
        return clean
    if len(clean) == 10 and clean[:9].isdigit():
        converted = isbn10_to_isbn13(clean)
        return converted if converted else clean
    return clean


def build_isbn_aliases(conn: sqlite3.Connection, ol_lookup: dict):
    """Build isbn_aliases table mapping every ISBN to canonical ISBN-13 + work_key."""
    print("Building isbn_aliases table...")

    conn.execute("DROP TABLE IF EXISTS isbn_aliases")
    conn.execute("""
        CREATE TABLE isbn_aliases (
            isbn TEXT PRIMARY KEY,
            canonical_isbn TEXT NOT NULL,
            work_key TEXT
        )
    """)
    conn.execute("CREATE INDEX idx_isbn_aliases_canonical ON isbn_aliases(canonical_isbn)")
    conn.execute("CREATE INDEX idx_isbn_aliases_work ON isbn_aliases(work_key)")

    # Get all ISBNs from subtitles
    rows = conn.execute(
        "SELECT DISTINCT isbn FROM subtitles WHERE isbn IS NOT NULL AND isbn != ''"
    ).fetchall()
    print(f"  Unique ISBNs in subtitles: {len(rows):,}")

    batch = []
    work_found = 0
    for (isbn_raw,) in rows:
        isbn = isbn_raw.strip()
        canonical = normalize_isbn(isbn)
        work_key = None

        # Try exact match in OL lookup
        ol_entry = ol_lookup.get(isbn)
        if ol_entry:
            work_key = ol_entry["work_key"]
            work_found += 1
        elif canonical != isbn:
            # Try canonical form
            ol_entry = ol_lookup.get(canonical)
            if ol_entry:
                work_key = ol_entry["work_key"]
                work_found += 1

        batch.append((isbn, canonical, work_key))

        if len(batch) >= 50000:
            conn.executemany(
                "INSERT OR IGNORE INTO isbn_aliases VALUES (?, ?, ?)", batch
            )
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO isbn_aliases VALUES (?, ?, ?)", batch
        )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM isbn_aliases").fetchone()[0]
    with_work = conn.execute(
        "SELECT COUNT(*) FROM isbn_aliases WHERE work_key IS NOT NULL"
    ).fetchone()[0]
    print(f"  isbn_aliases: {total:,} entries, {with_work:,} with work_key ({with_work * 100 // total}%)")


def build_filler_sources(conn: sqlite3.Connection):
    """Build slot_filler_sources many-to-many table.

    For each slot filler, find ALL subtitles that contributed it
    (not just the one stored in source_subtitle_id).
    """
    print("\nBuilding slot_filler_sources table...")

    conn.execute("DROP TABLE IF EXISTS slot_filler_sources")
    conn.execute("""
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL,
            PRIMARY KEY (slot_filler_id, subtitle_id)
        )
    """)

    # Get all fillers with their types
    fillers = conn.execute(
        "SELECT id, slot_type, filler FROM slot_fillers WHERE mode = 'strict'"
    ).fetchall()
    print(f"  Slot fillers to map: {len(fillers):,}")

    # Get all pattern matches (these have the actual slot→subtitle mappings)
    # pattern_matches stores: subtitle_id, list_items_json, action_noun, of_object
    print("  Loading pattern matches...")
    matches = conn.execute(
        "SELECT subtitle_id, list_items_json, action_noun, of_object FROM pattern_matches"
    ).fetchall()
    print(f"  Pattern matches: {len(matches):,}")

    # Build reverse index: filler_text → set of subtitle_ids, grouped by slot type
    # This is more accurate than source_subtitle_id which only keeps the first
    list_item_map: dict[str, set[int]] = defaultdict(set)
    action_noun_map: dict[str, set[int]] = defaultdict(set)
    of_object_map: dict[str, set[int]] = defaultdict(set)

    for sid, list_json, action, obj in matches:
        if action:
            action_noun_map[action].add(sid)
        if obj:
            of_object_map[obj].add(sid)
        if list_json:
            try:
                items = json.loads(list_json)
                for item in items:
                    list_item_map[item].add(sid)
            except (json.JSONDecodeError, TypeError):
                pass

    # Map fillers to their source subtitles
    batch = []
    mapped = 0
    multi_source = 0

    for filler_id, slot_type, filler_text in fillers:
        if slot_type == "list_item":
            sids = list_item_map.get(filler_text, set())
        elif slot_type == "action_noun":
            sids = action_noun_map.get(filler_text, set())
        elif slot_type == "of_object":
            sids = of_object_map.get(filler_text, set())
        else:
            # Sub-parts (of_modifier, of_head, etc.) don't have direct pattern_match entries
            # Fall back to source_subtitle_id
            sid_row = conn.execute(
                "SELECT source_subtitle_id FROM slot_fillers WHERE id = ?", (filler_id,)
            ).fetchone()
            sids = {sid_row[0]} if sid_row and sid_row[0] else set()

        if sids:
            mapped += 1
            if len(sids) > 1:
                multi_source += 1
            for sid in sids:
                batch.append((filler_id, sid))

        if len(batch) >= 50000:
            conn.executemany(
                "INSERT OR IGNORE INTO slot_filler_sources VALUES (?, ?)", batch
            )
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT OR IGNORE INTO slot_filler_sources VALUES (?, ?)", batch
        )
    conn.commit()

    total_links = conn.execute("SELECT COUNT(*) FROM slot_filler_sources").fetchone()[0]
    print(f"  Mapped: {mapped:,} fillers, {total_links:,} total links")
    print(f"  Multi-source fillers: {multi_source:,} (had >1 source subtitle)")

    # Stats by slot type
    for stype in ["list_item", "action_noun", "of_object", "of_modifier", "of_head", "of_topic", "of_complement"]:
        row = conn.execute("""
            SELECT COUNT(DISTINCT sfs.slot_filler_id), COUNT(*)
            FROM slot_filler_sources sfs
            JOIN slot_fillers sf ON sfs.slot_filler_id = sf.id
            WHERE sf.slot_type = ?
        """, (stype,)).fetchone()
        if row[0] > 0:
            avg_sources = row[1] / row[0]
            print(f"    {stype}: {row[0]:,} fillers, {row[1]:,} links (avg {avg_sources:.1f} sources/filler)")


def report_isbn_coverage(conn: sqlite3.Connection):
    """Report how many filler sources have ISBN-linked subtitles."""
    print("\nISBN coverage via slot_filler_sources:")

    for stype in ["list_item", "action_noun", "of_object"]:
        total = conn.execute("""
            SELECT COUNT(DISTINCT sf.id) FROM slot_fillers sf WHERE sf.slot_type = ? AND sf.mode = 'strict'
        """, (stype,)).fetchone()[0]

        with_isbn = conn.execute("""
            SELECT COUNT(DISTINCT sf.id)
            FROM slot_fillers sf
            JOIN slot_filler_sources sfs ON sfs.slot_filler_id = sf.id
            JOIN subtitles s ON sfs.subtitle_id = s.id
            WHERE sf.slot_type = ? AND sf.mode = 'strict'
            AND s.isbn IS NOT NULL AND s.isbn != ''
        """, (stype,)).fetchone()[0]

        with_work = conn.execute("""
            SELECT COUNT(DISTINCT sf.id)
            FROM slot_fillers sf
            JOIN slot_filler_sources sfs ON sfs.slot_filler_id = sf.id
            JOIN subtitles s ON sfs.subtitle_id = s.id
            JOIN isbn_aliases ia ON ia.isbn = s.isbn
            WHERE sf.slot_type = ? AND sf.mode = 'strict'
            AND ia.work_key IS NOT NULL
        """, (stype,)).fetchone()[0]

        print(f"  {stype}: {total:,} total, {with_isbn:,} with ISBN ({with_isbn*100//total}%), "
              f"{with_work:,} with work_key ({with_work*100//total}%)")


def main():
    parser = argparse.ArgumentParser(description="Build ISBN aliases and filler-source mappings")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to subtitles.db")
    args = parser.parse_args()

    print("Loading OL edition lookup...")
    with open(OL_LOOKUP_PATH) as f:
        ol_lookup = json.load(f)
    print(f"  {len(ol_lookup):,} entries")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    start = time.time()
    build_isbn_aliases(conn, ol_lookup)
    build_filler_sources(conn)
    report_isbn_coverage(conn)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    conn.close()


if __name__ == "__main__":
    main()
