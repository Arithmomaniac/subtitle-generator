"""Download Goodreads ratings data and build ISBN→ratings lookup.

Supports two modes:
  --mode 10k   Download Goodbooks-10k CSV (top 10,000 most-rated books).
  --mode full  Download the UCSD Goodreads Book Graph (~2.36M books, ~2GB gz).
  --mode merge (default) Download both, merge, keeping higher ratings_count per ISBN.

Output: data/goodreads_lookup.json  (ISBN → {ratings_count, work_ratings_count, average_rating, title})
"""

import argparse
import csv
import gzip
import json
import math
import time
from pathlib import Path
from urllib.request import Request, urlopen, urlretrieve

GOODBOOKS_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv"
RAW_CSV = Path("data/raw/goodbooks_10k.csv")

FULL_GZ_URLS = [
    "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/goodreads_books.json.gz",
    "https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/goodreads/goodreads_books.json.gz",
]
RAW_GZ = Path("data/raw/goodreads_books.json.gz")

OUTPUT = Path("data/goodreads_lookup.json")
CHUNK_SIZE = 1024 * 1024  # 1 MB


def download_csv():
    """Download the Goodbooks-10k books.csv if not already present."""
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"  Already downloaded: {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Downloading {GOODBOOKS_URL} ...")
    urlretrieve(GOODBOOKS_URL, RAW_CSV)
    print(f"  Saved to {RAW_CSV} ({RAW_CSV.stat().st_size / 1e6:.1f} MB)")


def download_gz_with_resume(max_retries: int = 10):
    """Download the full UCSD Goodreads gzipped JSONL with resume support."""
    RAW_GZ.parent.mkdir(parents=True, exist_ok=True)
    downloaded = RAW_GZ.stat().st_size if RAW_GZ.exists() else 0

    for url in FULL_GZ_URLS:
        print(f"  Trying {url}")
        for attempt in range(max_retries):
            try:
                headers = {"User-Agent": "subtitle-generator/0.1 (research project)"}
                if downloaded > 0:
                    headers["Range"] = f"bytes={downloaded}-"
                    print(f"  Resuming from {downloaded / 1e9:.2f} GB (attempt {attempt + 1})")

                req = Request(url, headers=headers)
                resp = urlopen(req, timeout=120)

                status = resp.status
                if downloaded > 0 and status != 206:
                    print(f"  Server doesn't support Range (status {status}), restarting...")
                    downloaded = 0
                    RAW_GZ.unlink(missing_ok=True)
                    resp.close()
                    continue

                total = int(resp.headers.get("Content-Length", 0))
                if status == 206:
                    total += downloaded

                mode = "ab" if downloaded > 0 else "wb"
                last_report = time.time()
                with open(RAW_GZ, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_report >= 10:
                            pct = downloaded * 100 / total if total else 0
                            print(f"  {downloaded / 1e9:.2f} GB ({pct:.0f}%)", flush=True)
                            last_report = now

                resp.close()
                print(f"  Download complete: {downloaded / 1e9:.2f} GB")
                return

            except (ConnectionResetError, OSError, TimeoutError) as e:
                # 404/403 means wrong URL — skip to next immediately
                err_str = str(e)
                if "404" in err_str or "403" in err_str:
                    print(f"  HTTP {err_str} — skipping to next URL")
                    break
                print(f"  Connection error after {downloaded / 1e9:.2f} GB: {e}")
                time.sleep(5 * (attempt + 1))

        print(f"  All retries exhausted for {url}, trying next URL...")

    raise RuntimeError("Failed to download from all URLs")


def isbn13_from_float(raw: str) -> str:
    """Convert isbn13 values like '9.78043902348e+12' to proper 13-digit strings."""
    try:
        val = float(raw)
        if math.isnan(val) or val <= 0:
            return ""
        return str(int(val)).zfill(13)
    except (ValueError, TypeError, OverflowError):
        return ""


def parse_10k_csv() -> dict[str, dict]:
    """Parse the Goodbooks-10k CSV and return ISBN→ratings lookup."""
    print(f"\nParsing {RAW_CSV} ...")
    lookup: dict[str, dict] = {}
    total_rows = 0
    skipped_no_isbn = 0

    with open(RAW_CSV, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            isbn10 = (row.get("isbn") or "").strip()
            isbn13_raw = (row.get("isbn13") or "").strip()
            isbn13 = isbn13_from_float(isbn13_raw)

            if not isbn10 and not isbn13:
                skipped_no_isbn += 1
                continue

            try:
                ratings_count = int(row["ratings_count"])
                work_ratings_count = int(row["work_ratings_count"])
                average_rating = float(row["average_rating"])
            except (ValueError, KeyError):
                continue

            title = row.get("title") or row.get("original_title") or ""
            entry = {
                "ratings_count": ratings_count,
                "work_ratings_count": work_ratings_count,
                "average_rating": average_rating,
                "title": title,
            }

            if isbn13:
                lookup[isbn13] = entry
            if isbn10:
                lookup[isbn10] = entry

    print(f"  10k CSV: {total_rows:,} rows → {len(lookup):,} ISBN keys "
          f"(skipped {skipped_no_isbn:,} without ISBN)")
    return lookup


def parse_full_goodreads(gz_path: Path) -> dict[str, dict]:
    """Stream the gzipped JSONL file and extract ISBN→ratings mappings."""
    print(f"\nParsing {gz_path} ({gz_path.stat().st_size / 1e9:.2f} GB compressed)...")
    lookup: dict[str, dict] = {}
    count = 0
    skipped_no_isbn = 0
    skipped_no_ratings = 0
    start = time.time()

    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            count += 1
            if count % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  ...processed {count:,} lines, {len(lookup):,} with ISBN "
                      f"({elapsed:.0f}s)", flush=True)
            try:
                data = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            isbn13 = (data.get("isbn13") or "").strip()
            isbn10 = (data.get("isbn") or "").strip()

            if not isbn13 and not isbn10:
                skipped_no_isbn += 1
                continue

            ratings = int(data.get("ratings_count") or 0)
            if ratings == 0:
                skipped_no_ratings += 1
                continue

            avg = float(data.get("average_rating") or 0)
            title = data.get("title", "")
            work_ratings = int(
                data.get("work_ratings_count") or data.get("ratings_count") or 0
            )

            entry = {
                "ratings_count": ratings,
                "work_ratings_count": work_ratings,
                "average_rating": avg,
                "title": title,
            }

            if isbn13:
                lookup[isbn13] = entry
            if isbn10:
                lookup[isbn10] = entry

    elapsed = time.time() - start
    print(f"  Full dataset: {count:,} lines → {len(lookup):,} ISBN keys "
          f"(no ISBN: {skipped_no_isbn:,}, no ratings: {skipped_no_ratings:,}) "
          f"in {elapsed:.0f}s")
    return lookup


def merge_lookups(base: dict[str, dict], overlay: dict[str, dict]) -> dict[str, dict]:
    """Merge two lookups, keeping the entry with higher ratings_count per ISBN."""
    merged = dict(base)
    new_from_overlay = 0
    updated_from_overlay = 0
    for isbn, entry in overlay.items():
        if isbn not in merged:
            merged[isbn] = entry
            new_from_overlay += 1
        elif entry["ratings_count"] > merged[isbn]["ratings_count"]:
            merged[isbn] = entry
            updated_from_overlay += 1
    print(f"  Merge: {new_from_overlay:,} new ISBNs from overlay, "
          f"{updated_from_overlay:,} updated with higher ratings")
    return merged


def print_stats(lookup: dict[str, dict], lookup_10k: dict | None, lookup_full: dict | None):
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("FINAL LOOKUP STATS")
    print(f"{'='*60}")
    print(f"Total entries: {len(lookup):,}")
    if lookup_10k is not None:
        print(f"From 10k dataset: {len(lookup_10k):,}")
    if lookup_full is not None:
        print(f"From full dataset: {len(lookup_full):,}")

    if lookup:
        vals = sorted(v["ratings_count"] for v in lookup.values())
        n = len(vals)
        print("\nRatings count distribution:")
        print(f"  min={vals[0]:,}, median={vals[n//2]:,}, mean={sum(vals)//n:,}, "
              f"p90={vals[int(n*0.9)]:,}, max={vals[-1]:,}")

    # Check overlap with SPL checkout data
    spl_path = Path("data/spl_checkout_lookup.json")
    if spl_path.exists():
        spl = json.load(open(spl_path))
        overlap = set(lookup.keys()) & set(spl.keys())
        print(f"\nISBN overlap with SPL checkouts: {len(overlap):,}")

    # Top 10 by ratings_count (deduplicate by title)
    seen_titles: set[str] = set()
    top: list[tuple[str, dict]] = []
    for isbn, d in sorted(lookup.items(), key=lambda x: -x[1]["ratings_count"]):
        title = d["title"]
        if title in seen_titles:
            continue
        seen_titles.add(title)
        top.append((isbn, d))
        if len(top) >= 10:
            break

    print("\nTop 10 books by ratings_count:")
    for isbn, d in top:
        print(f"  {isbn}: {d['title']} ({d['ratings_count']:,} ratings, "
              f"avg {d['average_rating']})")


def main():
    parser = argparse.ArgumentParser(description="Build Goodreads ISBN→ratings lookup")
    parser.add_argument(
        "--mode",
        choices=["10k", "full", "merge"],
        default="merge",
        help="10k=Goodbooks-10k only, full=UCSD full dataset only, "
             "merge=both merged (default)",
    )
    args = parser.parse_args()

    lookup_10k = None
    lookup_full = None

    if args.mode in ("10k", "merge"):
        print("Phase 1a: Download Goodbooks-10k CSV")
        download_csv()
        lookup_10k = parse_10k_csv()

    if args.mode in ("full", "merge"):
        print("\nPhase 1b: Download UCSD full Goodreads dataset")
        if RAW_GZ.exists():
            size_gb = RAW_GZ.stat().st_size / 1e9
            print(f"  Already downloaded: {RAW_GZ} ({size_gb:.2f} GB)")
        else:
            download_gz_with_resume()
        lookup_full = parse_full_goodreads(RAW_GZ)

    # Build final lookup
    print("\nPhase 2: Build merged lookup")
    if args.mode == "merge" and lookup_10k and lookup_full:
        lookup = merge_lookups(lookup_full, lookup_10k)
    elif lookup_full is not None:
        lookup = lookup_full
    elif lookup_10k is not None:
        lookup = lookup_10k
    else:
        raise RuntimeError("No data sources produced results")

    # Save
    print(f"\nPhase 3: Save to {OUTPUT}")
    with open(OUTPUT, "w") as f:
        json.dump(lookup, f)
    print(f"  Saved {len(lookup):,} entries ({OUTPUT.stat().st_size / 1e6:.1f} MB)")

    print_stats(lookup, lookup_10k, lookup_full)


if __name__ == "__main__":
    main()
