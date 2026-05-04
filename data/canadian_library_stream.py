"""Download Canadian public library data and build ISBN→popularity lookup.

Downloads Most Requested/Popular titles from Ottawa and Edmonton public
libraries, resolves titles to ISBNs (via Open Library when needed),
and produces a single lookup JSON.

Output: data/canadian_library_lookup.json
        (ISBN → {source, title, author, holds_count, year})
"""

import argparse
import csv
import io
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "subtitle-generator/0.1 (research project)"

# --- Ottawa ---
OTTAWA_DATASETS = {
    2023: "5b1dc0e2c68946f6b290a42199c12688",
    2024: "5357f98636314cafb3ad85a1458f6ad5",
    2025: "923f2c10812d471ea64a747c6d885391",
}
OTTAWA_CSV_URL = (
    "https://open.ottawa.ca/api/download/v1/items/{item_id}/csv?layers=0"
)
OTTAWA_LOCAL = "data/raw/ottawa_most_requested_{year}.csv"

# --- Edmonton ---
EDMONTON_CSV_URL = (
    "https://data.edmonton.ca/api/views/qdgm-hex6/rows.csv?accessType=DOWNLOAD"
)
EDMONTON_LOCAL = Path("data/raw/edmonton_popular_books.csv")

OUTPUT = Path("data/canadian_library_lookup.json")

# ISBN-13 pattern (may have hyphens)
ISBN13_RE = re.compile(r"97[89]\d{10}")
# Strip trailing format notes like "(hardcover)"
ISBN_CLEAN_RE = re.compile(r"^([\d\-]+)")


def _request(url: str, timeout: int = 30) -> bytes:
    """Fetch URL with User-Agent header."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    resp = urlopen(req, timeout=timeout)
    data = resp.read()
    resp.close()
    return data


# ── Download ──────────────────────────────────────────────────────────


def download_edmonton() -> Path:
    """Download Edmonton popular books CSV."""
    print("Downloading Edmonton popular books...")
    EDMONTON_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    data = _request(EDMONTON_CSV_URL, timeout=120)
    EDMONTON_LOCAL.write_bytes(data)
    size_mb = len(data) / 1e6
    print(f"  Saved {EDMONTON_LOCAL} ({size_mb:.1f} MB)")
    return EDMONTON_LOCAL


def download_ottawa(year: int) -> Path:
    """Download Ottawa most-requested-titles CSV for a given year."""
    item_id = OTTAWA_DATASETS.get(year)
    if not item_id:
        raise ValueError(
            f"No Ottawa dataset ID for {year}. "
            f"Known years: {sorted(OTTAWA_DATASETS)}"
        )
    url = OTTAWA_CSV_URL.format(item_id=item_id)
    dest = Path(OTTAWA_LOCAL.format(year=year))
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Ottawa {year} most-requested titles...")
    data = _request(url, timeout=120)
    dest.write_bytes(data)
    size_mb = len(data) / 1e6
    print(f"  Saved {dest} ({size_mb:.1f} MB)")
    return dest


# ── Parse ─────────────────────────────────────────────────────────────


def _clean_isbn(raw: str) -> str | None:
    """Extract a clean ISBN-13 from a possibly messy field.

    Handles: '9781250178633 (hardcover)', '9.78E+12', '0735211299', etc.
    Returns the first ISBN-13 found, or None.
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()

    # Excel scientific notation (e.g. '9.78E+12') loses precision —
    # the original digits are irrecoverable, so skip these.
    if "E+" in raw.upper() or "E-" in raw.upper():
        return None

    # Strip format notes: '9781250178633 (hardcover)' → '9781250178633'
    m = ISBN_CLEAN_RE.match(raw)
    if m:
        digits = m.group(1).replace("-", "")
        if ISBN13_RE.match(digits):
            return digits

    # Fallback: scan for any ISBN-13 in the string
    found = ISBN13_RE.search(raw.replace("-", ""))
    if found:
        return found.group()

    return None


def _clean_title(raw: str) -> str:
    """Strip trailing slash/punctuation from catalog titles."""
    title = raw.strip().rstrip("/").strip()
    # Remove trailing " /" or " :" patterns
    title = re.sub(r"\s*[/:]\s*$", "", title)
    return title


def _clean_author(raw: str) -> str:
    """Normalise author from 'Last, First,' → 'First Last'."""
    author = raw.strip().rstrip(",").strip()
    if "," in author:
        parts = [p.strip() for p in author.split(",", 1)]
        if len(parts) == 2 and parts[1]:
            author = f"{parts[1]} {parts[0]}"
    return author


def parse_ottawa(csv_path: Path) -> list[dict]:
    """Parse Ottawa most-requested-titles CSV.

    Returns list of {isbn, title, author, holds_count, year}.
    """
    results = []
    rows = 0
    with_isbn = 0

    # Handle BOM
    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        rows += 1
        title = _clean_title(row.get("Item_Title", "") or row.get("Title", ""))
        author = _clean_author(row.get("Author", ""))
        holds_str = row.get("Holds", "0") or "0"
        year_str = row.get("Hold_Created_Year", "")

        try:
            holds = int(holds_str)
        except (ValueError, TypeError):
            holds = 0

        try:
            year = int(year_str) if year_str else 0
        except (ValueError, TypeError):
            year = 0

        isbn = _clean_isbn(row.get("ISBN", ""))
        if isbn:
            with_isbn += 1

        results.append({
            "isbn": isbn,
            "title": title,
            "author": author,
            "holds_count": holds,
            "year": year,
            "source": "ottawa",
        })

    print(f"  Ottawa: {rows:,} rows, {with_isbn:,} with ISBN-13")
    return results


def parse_edmonton(csv_path: Path) -> list[dict]:
    """Parse Edmonton popular-books CSV.

    Aggregates by title+author across branches/dates, summing holds.
    Returns list of {title, author, holds_count, year}.
    """
    agg: dict[tuple[str, str], dict] = {}
    rows = 0

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            raw_title = row.get("Title", "")
            author_raw = row.get("Author", "")
            holds_str = row.get("Number of Holds", "0") or "0"
            date_str = row.get("As of Date", "")

            # Title often has author appended: "The girl on the train / Paula Hawkins"
            if " / " in raw_title:
                title = _clean_title(raw_title.split(" / ")[0])
            else:
                title = _clean_title(raw_title)

            # Author is "Last First" in Edmonton (no comma)
            author = author_raw.strip()

            try:
                holds = int(holds_str)
            except (ValueError, TypeError):
                holds = 0

            # Extract year from date like "Mar 16, 2015"
            year = 0
            if date_str:
                m = re.search(r"(\d{4})", date_str)
                if m:
                    year = int(m.group(1))

            key = (title.lower(), author.lower())
            if key in agg:
                agg[key]["holds_count"] += holds
                agg[key]["appearances"] += 1
                if year:
                    agg[key]["years"].add(year)
            else:
                agg[key] = {
                    "title": title,
                    "author": author,
                    "holds_count": holds,
                    "appearances": 1,
                    "years": {year} if year else set(),
                    "source": "edmonton",
                }

    results = []
    for entry in agg.values():
        years = sorted(entry.pop("years"))
        entry["year"] = max(years) if years else 0
        results.append(entry)

    print(f"  Edmonton: {rows:,} rows → {len(results):,} unique titles")
    return results


# ── ISBN resolution via local subtitles DB ────────────────────────────


DB_PATH = Path("data/db/subtitles.db")
_isbn_conn = None


def _get_isbn_conn():
    """Get or create a shared DB connection for ISBN lookups."""
    global _isbn_conn
    if _isbn_conn is None:
        import sqlite3
        _isbn_conn = sqlite3.connect(str(DB_PATH))
    return _isbn_conn


def lookup_isbn_local(title: str, author: str) -> str | None:
    """Resolve title+author to ISBN via our local subtitles database.

    Uses COLLATE NOCASE index for fast case-insensitive matching.
    """
    if not DB_PATH.exists():
        return None

    conn = _get_isbn_conn()

    row = conn.execute(
        "SELECT isbn FROM subtitles "
        "WHERE title = ? COLLATE NOCASE AND isbn IS NOT NULL "
        "LIMIT 1",
        (title,),
    ).fetchone()
    if row and row[0]:
        return row[0].strip()

    row = conn.execute(
        "SELECT isbn FROM subtitles "
        "WHERE title LIKE ? COLLATE NOCASE AND isbn IS NOT NULL "
        "LIMIT 1",
        (title + "%",),
    ).fetchone()
    if row and row[0]:
        return row[0].strip()

    return None


def resolve_isbns(
    entries: list[dict], delay: float = 0.0
) -> tuple[dict[str, dict], int]:
    """Resolve entries without ISBNs via local subtitles DB.

    Returns (isbn_lookup_dict, skipped_count).
    """
    lookup: dict[str, dict] = {}
    skipped = 0
    local_lookups = 0
    local_hits = 0

    for i, entry in enumerate(entries):
        isbn = entry.get("isbn")

        if not isbn:
            local_lookups += 1
            if local_lookups % 500 == 0:
                print(
                    f"    Local lookups: {local_lookups} done, "
                    f"{local_hits} hits, processing {i+1}/{len(entries)}..."
                )
            isbn = lookup_isbn_local(entry["title"], entry["author"])
            if isbn:
                local_hits += 1
            else:
                skipped += 1
                continue

        # Deduplicate: keep entry with highest holds
        if isbn in lookup:
            if entry["holds_count"] > lookup[isbn]["holds_count"]:
                lookup[isbn] = _make_record(entry, isbn)
        else:
            lookup[isbn] = _make_record(entry, isbn)

    if local_lookups:
        print(
            f"    Local DB resolution: {local_lookups} lookups, "
            f"{local_hits} hits, {skipped} unresolved"
        )

    return lookup, skipped


def _make_record(entry: dict, isbn: str) -> dict:
    """Build a lookup record from a parsed entry."""
    rec = {
        "source": entry["source"],
        "title": entry["title"],
        "author": entry["author"],
        "holds_count": entry["holds_count"],
        "year": entry["year"],
    }
    if "appearances" in entry:
        rec["appearances"] = entry["appearances"]
    return rec


def build_lookup_no_isbn(entries: list[dict]) -> dict[str, dict]:
    """Build a title-keyed lookup (skip ISBN resolution)."""
    lookup: dict[str, dict] = {}

    for entry in entries:
        isbn = entry.get("isbn")
        if isbn:
            key = isbn
        else:
            key = f"{entry['title']}|{entry['author']}"

        if key in lookup:
            if entry["holds_count"] > lookup[key]["holds_count"]:
                lookup[key] = _make_record(entry, key)
        else:
            lookup[key] = _make_record(entry, key)

    return lookup


# ── CLI ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Download and process Canadian public library popularity data."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download raw data files",
    )
    parser.add_argument(
        "--parse-only",
        action="store_true",
        help="Parse existing files without downloading",
    )
    parser.add_argument(
        "--skip-isbn-lookup",
        action="store_true",
        help="Skip OL ISBN resolution (faster, mixed key output)",
    )
    parser.add_argument(
        "--ottawa-years",
        nargs="*",
        type=int,
        default=[2023, 2024, 2025],
        help="Ottawa dataset years to process (default: 2023 2024 2025)",
    )
    args = parser.parse_args()

    # Phase 1: Download
    if args.download and not args.parse_only:
        print("Phase 1: Download\n")
        try:
            download_edmonton()
        except (HTTPError, URLError, OSError) as e:
            print(f"  Edmonton download failed: {e}")

        for year in args.ottawa_years:
            try:
                download_ottawa(year)
            except (HTTPError, URLError, OSError, ValueError) as e:
                print(f"  Ottawa {year} download failed: {e}")
        print()

    # Phase 2: Parse
    print("Phase 2: Parse\n")
    all_entries: list[dict] = []

    # Edmonton
    if EDMONTON_LOCAL.exists():
        edmonton = parse_edmonton(EDMONTON_LOCAL)
        all_entries.extend(edmonton)
    else:
        print(f"  Edmonton CSV not found at {EDMONTON_LOCAL} (use --download)")

    # Ottawa
    for year in args.ottawa_years:
        path = Path(OTTAWA_LOCAL.format(year=year))
        if path.exists():
            ottawa = parse_ottawa(path)
            all_entries.extend(ottawa)
        else:
            print(f"  Ottawa {year} CSV not found at {path} (use --download)")

    if not all_entries:
        print("\nNo data to process. Use --download first.")
        return

    print(f"\nTotal entries: {len(all_entries):,}")

    # Phase 3: Build lookup
    print("\nPhase 3: Build lookup\n")
    if args.skip_isbn_lookup:
        lookup = build_lookup_no_isbn(all_entries)
        print("  Built title/ISBN-keyed lookup (no OL resolution)")
    else:
        lookup, skipped = resolve_isbns(all_entries)
        print(f"  Skipped (no ISBN): {skipped:,}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(lookup, f, indent=2)
    print(f"\nSaved {len(lookup):,} entries to {OUTPUT}")

    # Summary stats
    sources = defaultdict(int)
    for v in lookup.values():
        sources[v["source"]] += 1
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count:,} entries")

    if lookup:
        holds = sorted(v["holds_count"] for v in lookup.values())
        n = len(holds)
        print(
            f"\nHolds distribution: min={holds[0]}, "
            f"median={holds[n//2]}, max={holds[-1]}"
        )


if __name__ == "__main__":
    main()
