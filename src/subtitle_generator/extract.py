"""Extract subtitles from MARC records and store in SQLite."""

import re
import sqlite3
from pathlib import Path

import click
from pymarc import MARCReader

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "db" / "subtitles.db"


def get_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Get a connection to the subtitles database, creating tables if needed."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            subtitle TEXT NOT NULL,
            lang TEXT,
            lccn TEXT,
            source_file TEXT
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subtitles_lang ON subtitles(lang)
    """)
    conn.commit()
    return conn


def _clean_subtitle(raw: str) -> str | None:
    """Clean and normalize a MARC 245$b subtitle value."""
    s = raw.strip()
    # Strip trailing MARC punctuation: / : ; .
    s = re.sub(r"[\s]*[/:;.]\s*$", "", s)
    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Skip very short subtitles (< 5 chars) — likely noise
    if len(s) < 5:
        return None
    return s


def _get_language(record) -> str | None:
    """Extract 3-letter language code from MARC 008 field (positions 35-37)."""
    field_008 = record.get("008")
    if field_008:
        raw = field_008.data if hasattr(field_008, "data") else str(field_008)
        if len(raw) >= 38:
            return raw[35:38].strip()
    return None


def extract_from_file(
    mrc_path: Path, conn: sqlite3.Connection, english_only: bool = True
) -> tuple[int, int]:
    """Extract subtitles from a single .mrc file.

    Returns (records_scanned, subtitles_found).
    """
    records_scanned = 0
    subtitles_found = 0
    batch = []
    source = mrc_path.name

    with open(mrc_path, "rb") as f:
        reader = MARCReader(f, to_unicode=True, force_utf8=False, utf8_handling="replace")
        for record in reader:
            if record is None:
                continue
            records_scanned += 1

            if english_only:
                lang = _get_language(record)
                if lang and lang != "eng":
                    continue
            else:
                lang = _get_language(record)

            # Get 245$b (subtitle / remainder of title)
            field_245 = record.get("245")
            if not field_245:
                continue
            subtitle_raw = field_245.get("b")
            if not subtitle_raw:
                continue

            subtitle = _clean_subtitle(subtitle_raw)
            if not subtitle:
                continue

            title = field_245.get("a", "")
            title = re.sub(r"[\s]*[/:;.]\s*$", "", title).strip()

            lccn_field = record.get("010")
            lccn = lccn_field.get("a", "").strip() if lccn_field else None

            batch.append((title, subtitle, lang, lccn, source))
            subtitles_found += 1

            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO subtitles (title, subtitle, lang, lccn, source_file) "
                    "VALUES (?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            if records_scanned % 50000 == 0:
                click.echo(f"  ...scanned {records_scanned:,} records, found {subtitles_found:,} subtitles")

    if batch:
        conn.executemany(
            "INSERT INTO subtitles (title, subtitle, lang, lccn, source_file) "
            "VALUES (?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    return records_scanned, subtitles_found
