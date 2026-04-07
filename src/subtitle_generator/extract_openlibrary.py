"""Download and extract subtitles from Open Library edition dumps."""

import gzip
import json
import re
import sqlite3
from pathlib import Path
from urllib.request import Request, urlopen

import click

from subtitle_generator.extract import DATA_DIR, _clean_subtitle

OL_DUMP_URL = "https://openlibrary.org/data/ol_dump_editions_latest.txt.gz"
OL_DUMP_FILENAME = "ol_dump_editions_latest.txt.gz"
OL_RAW_DIR = DATA_DIR / "raw"
OL_DUMP_PATH = OL_RAW_DIR / OL_DUMP_FILENAME

USER_AGENT = "subtitle-generator/0.1.0 (research project; Open Library bulk data)"
CHUNK_SIZE = 1024 * 1024  # 1 MB
BATCH_SIZE = 5000


def download_ol_dump(force: bool = False) -> Path:
    """Download the Open Library editions dump (~9.2 GB compressed)."""
    OL_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if OL_DUMP_PATH.exists() and not force:
        click.echo(f"OL dump already downloaded at {OL_DUMP_PATH}")
        return OL_DUMP_PATH

    click.echo(f"Downloading Open Library editions dump from {OL_DUMP_URL}")
    click.echo("This is ~9.2 GB and will take a while...")
    req = Request(OL_DUMP_URL, headers={"User-Agent": USER_AGENT})
    with urlopen(req) as response, open(OL_DUMP_PATH, "wb") as f_out:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            f_out.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded * 100 // total
                mb = downloaded / (1024 * 1024)
                total_mb = total / (1024 * 1024)
                print(f"\r  {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)
            else:
                mb = downloaded / (1024 * 1024)
                print(f"\r  {mb:.0f} MB", end="", flush=True)
        print()  # newline after progress

    click.echo(f"Download complete: {OL_DUMP_PATH}")
    return OL_DUMP_PATH


def _map_ol_language(languages: list[dict] | None) -> str | None:
    """Map Open Library language keys to ISO 639-3 codes.

    OL format: [{"key": "/languages/eng"}] → "eng"
    """
    if not languages:
        return None
    # Take the first language
    first = languages[0]
    key = first.get("key", "")
    # Extract code from "/languages/eng" → "eng"
    if key.startswith("/languages/"):
        return key.split("/")[-1]
    return None


def _clean_ol_subtitle(raw: str) -> str | None:
    """Clean an Open Library subtitle value.

    OL subtitles are generally cleaner than MARC 245$b (no trailing
    punctuation conventions), but we still normalize whitespace and
    skip very short values.
    """
    s = raw.strip()
    # Normalize internal whitespace
    s = re.sub(r"\s+", " ", s).strip()
    # Strip trailing punctuation that occasionally appears
    s = re.sub(r"[\s]*[/:;.]\s*$", "", s)
    if len(s) < 5:
        return None
    return s


def _normalize_lccn(raw: str) -> str:
    """Normalize LCCN for dedup comparison.

    Strips whitespace, hyphens, and leading zeros from the suffix.
    E.g. "  89-83818 " → "8983818"
    """
    return re.sub(r"[\s\-]", "", raw).strip()


def _build_existing_lccns(conn: sqlite3.Connection) -> set[str]:
    """Build a set of normalized existing LCCNs from the subtitles table."""
    rows = conn.execute(
        "SELECT DISTINCT lccn FROM subtitles WHERE lccn IS NOT NULL AND lccn != ''"
    ).fetchall()
    return {_normalize_lccn(r[0]) for r in rows if _normalize_lccn(r[0])}


def _build_existing_isbns(conn: sqlite3.Connection) -> set[str]:
    """Build a set of existing ISBNs from the subtitles table for dedup."""
    rows = conn.execute(
        "SELECT DISTINCT isbn FROM subtitles WHERE isbn IS NOT NULL AND isbn != ''"
    ).fetchall()
    return {r[0].strip() for r in rows}


def ensure_isbn_column(conn: sqlite3.Connection):
    """Add isbn column to subtitles table if it doesn't exist."""
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info(subtitles)").fetchall()
    }
    if "isbn" not in cols:
        conn.execute("ALTER TABLE subtitles ADD COLUMN isbn TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_subtitles_isbn ON subtitles(isbn)")
        conn.commit()


def extract_from_ol_dump(
    conn: sqlite3.Connection,
    dump_path: Path | None = None,
    english_only: bool = True,
    dedup: bool = True,
) -> tuple[int, int, int]:
    """Extract subtitles from the Open Library editions dump.

    Returns (lines_scanned, subtitles_found, duplicates_skipped).
    """
    path = dump_path or OL_DUMP_PATH
    if not path.exists():
        raise FileNotFoundError(f"OL dump not found at {path}. Run 'download-ol' first.")

    ensure_isbn_column(conn)

    # Idempotency: clear any previous OL extraction before re-running
    existing_ol = conn.execute(
        "SELECT COUNT(*) FROM subtitles WHERE source_file = 'openlibrary'"
    ).fetchone()[0]
    if existing_ol > 0:
        click.echo(f"Clearing {existing_ol:,} existing Open Library rows (re-run)...")
        conn.execute("DELETE FROM subtitles WHERE source_file = 'openlibrary'")
        conn.commit()

    # Build dedup sets
    existing_lccns: set[str] = set()
    existing_isbns: set[str] = set()
    seen_work_keys: set[str] = set()

    if dedup:
        click.echo("Building dedup index from existing records...")
        existing_lccns = _build_existing_lccns(conn)
        existing_isbns = _build_existing_isbns(conn)
        click.echo(
            f"  {len(existing_lccns):,} existing LCCNs, "
            f"{len(existing_isbns):,} existing ISBNs"
        )

    lines_scanned = 0
    subtitles_found = 0
    duplicates_skipped = 0
    batch: list[tuple] = []

    click.echo(f"Extracting from {path.name} (streaming gzip)...")

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_scanned += 1

            # TSV format: type\tkey\trevision\tlast_modified\tJSON
            parts = line.split("\t", 4)
            if len(parts) < 5:
                continue

            record_type = parts[0].strip()
            if record_type != "/type/edition":
                continue

            try:
                data = json.loads(parts[4])
            except (json.JSONDecodeError, IndexError):
                continue

            # Must have a subtitle
            subtitle_raw = data.get("subtitle")
            if not subtitle_raw or not isinstance(subtitle_raw, str):
                continue

            # Language filter
            lang = _map_ol_language(data.get("languages"))
            if english_only and lang != "eng":
                continue

            # Clean subtitle
            subtitle = _clean_ol_subtitle(subtitle_raw)
            if not subtitle:
                continue

            title = (data.get("title") or "").strip()

            # Extract identifiers
            lccn_list = data.get("lccn", [])
            lccn = lccn_list[0].strip() if lccn_list else None

            isbn_10 = data.get("isbn_10", [])
            isbn_13 = data.get("isbn_13", [])
            isbn = (isbn_13[0] if isbn_13 else isbn_10[0] if isbn_10 else None)
            if isbn:
                isbn = isbn.strip()

            # Deduplication
            if dedup:
                # Skip if normalized LCCN matches existing LOC record
                if lccn and _normalize_lccn(lccn) in existing_lccns:
                    duplicates_skipped += 1
                    continue

                # Skip if ISBN matches existing record
                if isbn and isbn in existing_isbns:
                    duplicates_skipped += 1
                    continue

                # Within OL: one edition per work
                works = data.get("works", [])
                if works:
                    work_key = works[0].get("key", "")
                    if work_key:
                        if work_key in seen_work_keys:
                            duplicates_skipped += 1
                            continue
                        seen_work_keys.add(work_key)

            batch.append((title, subtitle, lang, lccn, "openlibrary", isbn))
            subtitles_found += 1

            if len(batch) >= BATCH_SIZE:
                conn.executemany(
                    "INSERT INTO subtitles (title, subtitle, lang, lccn, source_file, isbn) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
                batch.clear()

            if lines_scanned % 1_000_000 == 0:
                click.echo(
                    f"  ...{lines_scanned:,} lines scanned, "
                    f"{subtitles_found:,} subtitles, "
                    f"{duplicates_skipped:,} duplicates skipped"
                )

    if batch:
        conn.executemany(
            "INSERT INTO subtitles (title, subtitle, lang, lccn, source_file, isbn) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            batch,
        )
        conn.commit()

    return lines_scanned, subtitles_found, duplicates_skipped
