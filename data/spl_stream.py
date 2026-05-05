"""Download SPL checkouts CSV and extract ISBN→checkout aggregates.

Downloads the ~11 GB CSV in chunks with retry/resume support,
then parses it to aggregate checkouts by ISBN and optionally by word.

Output: data/spl_checkout_lookup.json  (ISBN → {total_checkouts, years_active, pub_year})
        data/spl_word_popularity.json  (word → {total_checkouts, book_count}) [with --words]
"""

import csv
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen

SPL_CSV_URL = "https://zenodo.org/records/15792698/files/spl-monthly-checkouts_2005-2025.csv?download=1"
SPL_LOCAL = Path("data/raw/spl_checkouts.csv")
OUTPUT_ISBN = Path("data/spl_checkout_lookup.json")
OUTPUT_WORD = Path("data/spl_word_popularity.json")

CHUNK_SIZE = 1024 * 1024  # 1 MB
ISBN_SPLIT = re.compile(r"[,\s]+")
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def download_with_resume(url: str, dest: Path, max_retries: int = 10):
    """Download a large file with resume support on connection reset."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = dest.stat().st_size if dest.exists() else 0

    for attempt in range(max_retries):
        try:
            headers = {"User-Agent": "subtitle-generator/0.1 (research project)"}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"  Resuming from {downloaded / 1e9:.2f} GB (attempt {attempt + 1})")

            req = Request(url, headers=headers)
            resp = urlopen(req, timeout=60)

            # Check if server supports range
            status = resp.status
            if downloaded > 0 and status != 206:
                print(f"  Server doesn't support Range (status {status}), restarting...")
                downloaded = 0
                dest.unlink(missing_ok=True)
                resp.close()
                continue

            total = int(resp.headers.get("Content-Length", 0))
            if status == 206:
                # Content-Length is remaining bytes
                total += downloaded

            mode = "ab" if downloaded > 0 else "wb"
            with open(dest, mode) as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded % (100 * CHUNK_SIZE) == 0:
                        pct = downloaded * 100 / total if total else 0
                        print(f"  {downloaded / 1e9:.1f} GB ({pct:.0f}%)", flush=True)

            resp.close()
            print(f"  Download complete: {downloaded / 1e9:.1f} GB")
            return

        except (ConnectionResetError, OSError, TimeoutError) as e:
            print(f"  Connection error after {downloaded / 1e9:.2f} GB: {e}")
            time.sleep(5 * (attempt + 1))  # exponential-ish backoff

    raise RuntimeError(f"Failed to download after {max_retries} retries")


def parse_local_csv(csv_path: Path, save_words: bool = False):
    """Parse the downloaded CSV and aggregate by ISBN and optionally word."""
    print(f"\nParsing {csv_path} ({csv_path.stat().st_size / 1e9:.1f} GB)...")

    isbn_checkouts: dict[str, int] = defaultdict(int)
    isbn_years: dict[str, set] = defaultdict(set)
    isbn_pub_year: dict[str, str] = {}  # track earliest pub year per ISBN

    word_checkouts: dict[str, int] = defaultdict(int)
    word_books: dict[str, set] = defaultdict(set)

    rows_processed = 0
    rows_with_isbn = 0
    start = time.time()

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_processed += 1

            isbn_raw = row.get("isbn", row.get("ISBN", "")) or ""
            checkouts_str = row.get("checkouts", row.get("Checkouts", "0")) or "0"
            year = row.get("checkoutyear", row.get("CheckoutYear", "")) or ""
            pub_year = row.get("publicationyear", row.get("PublicationYear", "")) or ""

            try:
                checkouts = int(checkouts_str)
            except (ValueError, TypeError):
                continue

            if isbn_raw.strip() and isbn_raw.strip() != "NA":
                isbns = [i.strip() for i in ISBN_SPLIT.split(isbn_raw) if i.strip()]
                if isbns:
                    rows_with_isbn += 1
                    for isbn in isbns:
                        isbn_checkouts[isbn] += checkouts
                        isbn_years[isbn].add(year)
                        # Keep earliest publication year
                        if pub_year and isbn not in isbn_pub_year:
                            isbn_pub_year[isbn] = pub_year.strip().rstrip(".")

            if save_words:
                title = row.get("title", row.get("Title", "")) or ""
                words = WORD_RE.findall(title.lower())
                title_key = title.lower()[:80]
                for w in set(words):
                    word_checkouts[w] += checkouts
                    word_books[w].add(title_key)

            if rows_processed % 5_000_000 == 0:
                elapsed = time.time() - start
                print(
                    f"  {rows_processed:,} rows | "
                    f"{rows_with_isbn:,} with ISBN | "
                    f"{len(isbn_checkouts):,} unique ISBNs | "
                    f"{elapsed:.0f}s"
                )

    elapsed = time.time() - start
    print("\n--- Parse Complete ---")
    print(f"Total rows: {rows_processed:,}")
    print(f"Rows with ISBN: {rows_with_isbn:,} ({rows_with_isbn * 100 // max(rows_processed, 1)}%)")
    print(f"Unique ISBNs: {len(isbn_checkouts):,}")
    print(f"Time: {elapsed:.0f}s")

    # Save ISBN lookup
    isbn_lookup = {}
    for isbn, total in isbn_checkouts.items():
        isbn_lookup[isbn] = {
            "total_checkouts": total,
            "years_active": len(isbn_years[isbn]),
            "pub_year": isbn_pub_year.get(isbn, ""),
        }
    with open(OUTPUT_ISBN, "w") as f:
        json.dump(isbn_lookup, f)
    print(f"Saved {len(isbn_lookup):,} ISBN lookups to {OUTPUT_ISBN}")

    # Distribution
    if isbn_lookup:
        vals = sorted(v["total_checkouts"] for v in isbn_lookup.values())
        n = len(vals)
        print("\nCheckout distribution:")
        print(f"  min={vals[0]}, median={vals[n//2]}, mean={sum(vals)//n}, "
              f"p90={vals[int(n*0.9)]}, max={vals[-1]}")

    if save_words:
        word_lookup = {}
        for word, total in word_checkouts.items():
            word_lookup[word] = {
                "total_checkouts": total,
                "book_count": len(word_books[word]),
            }
        with open(OUTPUT_WORD, "w") as f:
            json.dump(word_lookup, f)
        print(f"Saved {len(word_lookup):,} word lookups to {OUTPUT_WORD}")


def main():
    save_words = "--words" in sys.argv
    skip_download = "--parse-only" in sys.argv

    if not skip_download:
        print("Phase 1: Download SPL CSV")
        download_with_resume(SPL_CSV_URL, SPL_LOCAL)

    print("\nPhase 2: Parse and aggregate")
    parse_local_csv(SPL_LOCAL, save_words=save_words)


if __name__ == "__main__":
    main()
