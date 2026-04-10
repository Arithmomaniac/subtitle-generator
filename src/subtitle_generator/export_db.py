"""Export a minimal SQLite database for web/API deployment."""

import sqlite3
from pathlib import Path


def export_mini_db(source_conn: sqlite3.Connection, output_path: Path) -> dict:
    """Create a minimal SQLite DB for web deployment.

    Copies slot_fillers and config tables verbatim, then builds a
    pre-joined ``sources`` lookup table so the web app can show source
    books without shipping the 3 GB subtitles table.

    Returns stats dict: {table: row_count, ...}.
    """
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dest = sqlite3.connect(str(output_path))
    stats: dict[str, int] = {}

    # -- slot_fillers (same schema as source) --
    dest.execute("""
        CREATE TABLE slot_fillers (
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
    rows = source_conn.execute("SELECT id, slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep FROM slot_fillers").fetchall()
    dest.executemany(
        "INSERT INTO slot_fillers (id, slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    stats["slot_fillers"] = len(rows)

    # -- config --
    dest.execute("""
        CREATE TABLE config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    rows = source_conn.execute("SELECT key, value FROM config").fetchall()
    dest.executemany("INSERT INTO config (key, value) VALUES (?, ?)", rows)
    stats["config"] = len(rows)

    # -- sources (pre-joined lookup: slot_filler → source book) --
    dest.execute("""
        CREATE TABLE sources (
            slot_filler_id INTEGER NOT NULL,
            title TEXT,
            subtitle_text TEXT,
            source_tag TEXT,
            FOREIGN KEY (slot_filler_id) REFERENCES slot_fillers(id)
        )
    """)
    source_rows = source_conn.execute("""
        SELECT sf.id,
               s.title,
               s.subtitle,
               CASE WHEN s.source_file = 'openlibrary' THEN 'OL' ELSE 'LOC' END
        FROM slot_fillers sf
        JOIN subtitles s ON s.id = sf.source_subtitle_id
        WHERE sf.source_subtitle_id IS NOT NULL
    """).fetchall()
    dest.executemany(
        "INSERT INTO sources (slot_filler_id, title, subtitle_text, source_tag) VALUES (?, ?, ?, ?)",
        source_rows,
    )
    stats["sources"] = len(source_rows)

    # -- indexes matching the queries in generate.py and jacket.py --
    dest.execute("CREATE INDEX idx_sf_slot_type ON slot_fillers(slot_type)")
    dest.execute("CREATE INDEX idx_sf_slot_type_pos ON slot_fillers(slot_type, pos_tag)")
    dest.execute("CREATE INDEX idx_sf_slot_type_prep ON slot_fillers(slot_type, prep)")
    dest.execute("CREATE INDEX idx_sf_filler ON slot_fillers(filler)")
    dest.execute("CREATE INDEX idx_sources_filler ON sources(slot_filler_id)")

    dest.commit()
    dest.execute("VACUUM")
    dest.close()

    return stats
