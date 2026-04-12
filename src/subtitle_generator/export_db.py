"""Export data for deployment and build mini SQLite from exported data."""

import base64
import csv
import sqlite3
from pathlib import Path


def export_data(source_conn: sqlite3.Connection, output_dir: Path) -> dict:
    """Export slot_fillers, config, and sources as CSV files.

    These text files are committed to the repo and used by ``build_mini_db``
    in CI to construct the SQLite deployment artifact.

    Returns stats dict: {filename: row_count, ...}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    # -- slot_fillers (with vector and remix columns) --
    rows = source_conn.execute(
        "SELECT id, slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep, "
        "remix_type, remix_prep, remix_word_count, vector_sum, token_count "
        "FROM slot_fillers"
    ).fetchall()
    path = output_dir / "slot_fillers.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "slot_type", "filler", "mode", "source_subtitle_id", "freq",
            "pos_tag", "prep", "remix_type", "remix_prep", "remix_word_count",
            "vector_sum_b64", "token_count",
        ])
        for row in rows:
            row = list(row)
            # Encode vector BLOB as base64 for CSV transport
            if row[11] is not None:
                row[11] = base64.b64encode(row[11]).decode("ascii")
            else:
                row[11] = ""
            w.writerow(row)
    stats["slot_fillers.csv"] = len(rows)

    # -- config --
    rows = source_conn.execute("SELECT key, value FROM config").fetchall()
    path = output_dir / "config.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)
    stats["config.csv"] = len(rows)

    # -- sources (pre-joined: slot_filler -> source book) --
    rows = source_conn.execute(
        "SELECT sf.id, s.title, s.subtitle, "
        "CASE WHEN s.source_file = 'openlibrary' THEN 'OL' ELSE 'LOC' END "
        "FROM slot_fillers sf "
        "JOIN subtitles s ON s.id = sf.source_subtitle_id "
        "WHERE sf.source_subtitle_id IS NOT NULL"
    ).fetchall()
    path = output_dir / "sources.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slot_filler_id", "title", "subtitle_text", "source_tag"])
        w.writerows(rows)
    stats["sources.csv"] = len(rows)

    return stats


def build_mini_db(data_dir: Path, output_path: Path) -> dict:
    """Build a minimal SQLite DB from exported CSV files.

    Reads slot_fillers.csv, config.csv, and sources.csv from ``data_dir``,
    creates an indexed SQLite database at ``output_path``.

    Returns stats dict: {table: row_count, ...}.
    """
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(output_path))
    stats: dict[str, int] = {}

    # -- slot_fillers (with vector + remix columns) --
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            vector_sum BLOB,
            token_count INTEGER,
            UNIQUE(slot_type, filler)
        )
    """)
    sf_path = data_dir / "slot_fillers.csv"
    with open(sf_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Decode base64 vector back to BLOB
            vec_b64 = row.get("vector_sum_b64", "") or row.get("vector_sum", "")
            vec_blob = base64.b64decode(vec_b64) if vec_b64 else None
            rows.append((
                int(row["id"]), row["slot_type"], row["filler"], row["mode"],
                int(row["source_subtitle_id"]) if row["source_subtitle_id"] else None,
                int(row["freq"]), row["pos_tag"] or None, row["prep"] or None,
                row.get("remix_type") or None,
                row.get("remix_prep") or None,
                int(row["remix_word_count"]) if row.get("remix_word_count") else None,
                vec_blob,
                int(row["token_count"]) if row.get("token_count") else None,
            ))
        conn.executemany(
            "INSERT INTO slot_fillers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        stats["slot_fillers"] = len(rows)

    # -- config --
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    cfg_path = data_dir / "config.csv"
    with open(cfg_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["key"], row["value"]) for row in reader]
        conn.executemany("INSERT INTO config VALUES (?, ?)", rows)
        stats["config"] = len(rows)

    # -- sources --
    conn.execute("""
        CREATE TABLE sources (
            slot_filler_id INTEGER NOT NULL,
            title TEXT,
            subtitle_text TEXT,
            source_tag TEXT,
            FOREIGN KEY (slot_filler_id) REFERENCES slot_fillers(id)
        )
    """)
    src_path = data_dir / "sources.csv"
    with open(src_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(int(row["slot_filler_id"]), row["title"], row["subtitle_text"], row["source_tag"]) for row in reader]
        conn.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)", rows)
        stats["sources"] = len(rows)

    # -- indexes --
    conn.execute("CREATE INDEX idx_sf_slot_type ON slot_fillers(slot_type)")
    conn.execute("CREATE INDEX idx_sf_slot_type_pos ON slot_fillers(slot_type, pos_tag)")
    conn.execute("CREATE INDEX idx_sf_slot_type_prep ON slot_fillers(slot_type, prep)")
    conn.execute("CREATE INDEX idx_sf_filler ON slot_fillers(filler)")
    conn.execute("CREATE INDEX idx_sources_filler ON sources(slot_filler_id)")

    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    return stats


# Keep backward compat for existing export-db CLI command
def export_mini_db(source_conn: sqlite3.Connection, output_path: Path) -> dict:
    """Create a minimal SQLite DB directly from the full DB (legacy one-step)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        export_data(source_conn, tmp_dir)
        return build_mini_db(tmp_dir, output_path)
