"""Extract subtitles from MARC records and store in SQLite."""

import re
import sqlite3
from pathlib import Path

import click
from pymarc import MARCReader

from subtitle_generator.feedback import ensure_ratings_table
from subtitle_generator.source_validation import (
    clean_title_and_subtitle,
    looks_like_subtitle_pattern,
)

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
            subtitle TEXT NOT NULL DEFAULT '',
            lang TEXT,
            lccn TEXT,
            source_file TEXT,
            isbn TEXT,
            candidate_text TEXT,
            candidate_source TEXT
        )
    """)
    ensure_subtitle_candidate_columns(conn)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subtitles_lang ON subtitles(lang)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_subtitles_isbn ON subtitles(isbn)
    """)
    ensure_ratings_table(conn)
    conn.commit()
    return conn


def ensure_subtitle_candidate_columns(conn: sqlite3.Connection) -> None:
    """Add source-candidate columns and backfill legacy subtitle-derived rows."""

    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(subtitles)").fetchall()
    }
    if "isbn" not in cols:
        conn.execute("ALTER TABLE subtitles ADD COLUMN isbn TEXT")
    if "candidate_text" not in cols:
        conn.execute("ALTER TABLE subtitles ADD COLUMN candidate_text TEXT")
    if "candidate_source" not in cols:
        conn.execute("ALTER TABLE subtitles ADD COLUMN candidate_source TEXT")
    conn.execute(
        """
        UPDATE subtitles
        SET candidate_text = subtitle
        WHERE candidate_text IS NULL
          AND subtitle IS NOT NULL
          AND subtitle != ''
        """
    )
    conn.execute(
        """
        UPDATE subtitles
        SET candidate_source = 'subtitle'
        WHERE candidate_source IS NULL
          AND candidate_text IS NOT NULL
          AND candidate_text != ''
        """
    )
    conn.commit()


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


def _clean_title(raw: str | None) -> str:
    """Clean and normalize a MARC title value."""

    return re.sub(r"[\s]*[/:;.]\s*$", "", raw or "").strip()


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

            field_245 = record.get("245")
            if not field_245:
                continue

            title = _clean_title(field_245.get("a", ""))

            # Prefer 245$b when present. If absent, admit title-only records
            # whose 245$a already has the generator source pattern.
            subtitle_raw = field_245.get("b")
            subtitle = _clean_subtitle(subtitle_raw) if subtitle_raw else None
            if subtitle:
                cleaned = clean_title_and_subtitle(title, subtitle)
                if cleaned is None:
                    continue
                title, subtitle = cleaned
                candidate_text = subtitle
                candidate_source = "subtitle"
            elif looks_like_subtitle_pattern(title):
                subtitle = ""
                candidate_text = title
                candidate_source = "title"
            else:
                continue

            lccn_field = record.get("010")
            lccn = lccn_field.get("a", "").strip() if lccn_field else None

            batch.append((title, subtitle, lang, lccn, source, candidate_text, candidate_source))
            subtitles_found += 1

            if len(batch) >= 5000:
                conn.executemany(
                    "INSERT INTO subtitles "
                    "(title, subtitle, lang, lccn, source_file, candidate_text, candidate_source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            if records_scanned % 50000 == 0:
                click.echo(f"  ...scanned {records_scanned:,} records, found {subtitles_found:,} subtitles")

    if batch:
        conn.executemany(
            "INSERT INTO subtitles "
            "(title, subtitle, lang, lccn, source_file, candidate_text, candidate_source) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    return records_scanned, subtitles_found
