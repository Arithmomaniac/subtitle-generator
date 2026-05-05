"""Extract edition counts per work from OL editions dump.

Streams the gzipped dump, extracts ISBN + work_key pairs,
groups by work_key to count editions, then joins to our subtitles DB.
"""

import gzip
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

DUMP_PATH = Path("data/raw/ol_dump_editions_latest.txt.gz")
DB_PATH = Path("data/db/subtitles.db")

# OL dump format: type\tkey\trevision\tlast_modified\tjson
# We want: json.works[0].key, json.isbn_13, json.isbn_10


def stream_editions(dump_path: Path, max_lines: int = 0):
    """Stream edition records, yielding (work_key, isbns) tuples."""
    count = 0
    with gzip.open(dump_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            count += 1
            if max_lines and count > max_lines:
                break
            if count % 2_000_000 == 0:
                print(f"  ...processed {count:,} lines", flush=True)

            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                data = json.loads(parts[4])
            except (json.JSONDecodeError, IndexError):
                continue

            # Get work key
            works = data.get("works", [])
            if not works:
                continue
            work_key = works[0].get("key", "")
            if not work_key:
                continue

            # Get ISBNs
            isbns = []
            for isbn in data.get("isbn_13", []):
                if isinstance(isbn, str) and isbn.strip():
                    isbns.append(isbn.strip())
            for isbn in data.get("isbn_10", []):
                if isinstance(isbn, str) and isbn.strip():
                    isbns.append(isbn.strip())

            if isbns:
                yield work_key, isbns

    print(f"  Total lines processed: {count:,}")


def main():
    print(f"Streaming OL dump: {DUMP_PATH}")
    print(f"File size: {DUMP_PATH.stat().st_size / 1e9:.1f} GB")

    # Phase 1: Stream dump, collect work_key → set of ISBNs, and ISBN → work_key
    work_edition_count: dict[str, int] = defaultdict(int)
    isbn_to_work: dict[str, str] = {}

    start = time.time()
    max_lines = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    for work_key, isbns in stream_editions(DUMP_PATH, max_lines=max_lines):
        work_edition_count[work_key] += 1
        for isbn in isbns:
            isbn_to_work[isbn] = work_key

    elapsed = time.time() - start
    print(f"\nExtraction complete in {elapsed:.0f}s")
    print(f"  Works with editions: {len(work_edition_count):,}")
    print(f"  ISBNs mapped to works: {len(isbn_to_work):,}")

    # Distribution of edition counts
    counts = sorted(work_edition_count.values(), reverse=True)
    print("\nEdition count distribution:")
    print(f"  1 edition: {sum(1 for c in counts if c == 1):,}")
    print(f"  2-5 editions: {sum(1 for c in counts if 2 <= c <= 5):,}")
    print(f"  6-20 editions: {sum(1 for c in counts if 6 <= c <= 20):,}")
    print(f"  21-100 editions: {sum(1 for c in counts if 21 <= c <= 100):,}")
    print(f"  100+ editions: {sum(1 for c in counts if c > 100):,}")
    if counts:
        print(f"  max: {counts[0]:,}")

    # Phase 2: Join to our subtitles DB
    print("\nJoining to subtitles DB...")
    conn = sqlite3.connect(str(DB_PATH))

    # Get all ISBNs from our subtitles
    our_isbns = conn.execute(
        "SELECT DISTINCT isbn FROM subtitles WHERE isbn IS NOT NULL AND isbn != ''"
    ).fetchall()
    our_isbn_set = {r[0].strip() for r in our_isbns}
    print(f"  Our ISBNs: {len(our_isbn_set):,}")

    # Find overlap
    matched = 0
    edition_counts_for_matched = []
    for isbn in our_isbn_set:
        work = isbn_to_work.get(isbn)
        if work:
            ec = work_edition_count[work]
            edition_counts_for_matched.append(ec)
            matched += 1

    print(f"  Matched ISBNs: {matched:,} ({matched * 100 // len(our_isbn_set)}%)")

    if edition_counts_for_matched:
        edition_counts_for_matched.sort()
        n = len(edition_counts_for_matched)
        print("\n  Edition count distribution (matched ISBNs):")
        print(f"    min={edition_counts_for_matched[0]}")
        print(f"    median={edition_counts_for_matched[n // 2]}")
        print(f"    mean={sum(edition_counts_for_matched) // n}")
        print(f"    p90={edition_counts_for_matched[int(n * 0.9)]}")
        print(f"    max={edition_counts_for_matched[-1]}")

    # Phase 3: Check slot filler coverage
    filler_isbns = conn.execute("""
        SELECT DISTINCT s.isbn, sf.filler, sf.slot_type
        FROM slot_fillers sf
        JOIN subtitles s ON sf.source_subtitle_id = s.id
        WHERE s.isbn IS NOT NULL AND s.isbn != ''
    """).fetchall()

    filler_matched = 0
    for isbn, filler, stype in filler_isbns:
        if isbn.strip() in isbn_to_work:
            filler_matched += 1

    print(f"\n  Slot filler ISBN coverage: {filler_matched:,} / {len(filler_isbns):,} "
          f"({filler_matched * 100 // len(filler_isbns) if filler_isbns else 0}%)")

    # Save lookup for later use
    output = Path("data/ol_edition_lookup.json")
    lookup = {}
    for isbn in our_isbn_set:
        work = isbn_to_work.get(isbn)
        if work:
            lookup[isbn] = {"work_key": work, "edition_count": work_edition_count[work]}
    with open(output, "w") as f:
        json.dump(lookup, f)
    print(f"\n  Saved {len(lookup):,} ISBN→edition lookups to {output}")

    conn.close()


if __name__ == "__main__":
    main()
