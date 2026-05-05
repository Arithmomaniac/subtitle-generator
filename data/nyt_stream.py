"""Pull historical NYT bestseller data and aggregate ISBN popularity.

Iterates through weekly NYT Books API bestseller lists from 2008 to present,
collecting ISBN-level data with checkpointing for resumable multi-day runs.

Rate limits: 5 req/min, 500 req/day (free tier).
Expected ~7 days for a full historical pull.

Output: data/nyt_bestseller_lookup.json  (ISBN → {title, author, weeks_on_list, ...})
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta
from urllib.error import HTTPError
from pathlib import Path
from urllib.request import Request, urlopen

NYT_API_KEY_ENV = "NYT_API_KEY"
LISTS = [
    "combined-print-and-e-book-nonfiction",
    "hardcover-nonfiction",
]
START_DATE = "2008-06-01"
CHECKPOINT_PATH = Path("data/nyt_checkpoint.json")
OUTPUT_PATH = Path("data/nyt_bestseller_lookup.json")
REQUEST_DELAY = 12  # seconds between requests (5/min limit)
DAILY_LIMIT = 490  # stay under 500/day with margin


def load_checkpoint() -> dict:
    """Load checkpoint from disk, or return a fresh one."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {
        "last_completed": None,
        "requests_today": 0,
        "today_date": date.today().isoformat(),
        "total_requests": 0,
        "books_collected": {},
    }


def save_checkpoint(cp: dict):
    """Atomically write checkpoint to disk."""
    tmp = CHECKPOINT_PATH.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(cp, f)
    tmp.replace(CHECKPOINT_PATH)


def build_schedule() -> list[tuple[str, str]]:
    """Build the full (date, list) schedule in chronological order.

    Each entry is a Sunday date string paired with a list name.
    The NYT API accepts any date and returns the list for that week,
    but we step by 7 days to avoid duplicate weeks.
    """
    schedule = []
    current = datetime.strptime(START_DATE, "%Y-%m-%d").date()
    # Align to the nearest Sunday
    days_until_sunday = (6 - current.weekday()) % 7
    current += timedelta(days=days_until_sunday)
    today = date.today()

    while current <= today:
        date_str = current.isoformat()
        for list_name in LISTS:
            schedule.append((date_str, list_name))
        current += timedelta(days=7)

    return schedule


def find_resume_index(schedule: list[tuple[str, str]], cp: dict) -> int:
    """Find where to resume in the schedule based on checkpoint."""
    last = cp.get("last_completed")
    if not last:
        return 0
    target = (last["date"], last["list"])
    for i, entry in enumerate(schedule):
        if entry == target:
            return i + 1  # resume after the last completed
    return 0


def reset_daily_counter_if_needed(cp: dict):
    """Reset the daily request counter if the date has changed."""
    today = date.today().isoformat()
    if cp["today_date"] != today:
        cp["requests_today"] = 0
        cp["today_date"] = today


def fetch_list(api_key: str, list_date: str, list_name: str) -> dict | None:
    """Fetch a single bestseller list from the NYT API.

    Returns parsed JSON on success, None on 404 (list doesn't exist for date).
    Raises on rate limit (429) or other HTTP errors.
    """
    url = (
        f"https://api.nytimes.com/svc/books/v3/lists/"
        f"{list_date}/{list_name}.json?api-key={api_key}"
    )
    req = Request(url, headers={"User-Agent": "subtitle-generator/0.1 (research project)"})

    try:
        resp = urlopen(req, timeout=30)
        raw = resp.read()
        resp.close()
        return json.loads(raw)
    except HTTPError as e:
        if e.code == 404:
            return None  # list didn't exist for this date
        raise


def merge_book(books: dict, isbn: str, title: str, author: str,
               rank: int, weeks: int, list_name: str, list_date: str):
    """Merge a single book record into the aggregate."""
    if isbn in books:
        entry = books[isbn]
        entry["weeks_on_list"] += 1
        entry["peak_rank"] = min(entry["peak_rank"], rank)
        if list_name not in entry["list_categories"]:
            entry["list_categories"].append(list_name)
        if list_date < entry["first_seen"]:
            entry["first_seen"] = list_date
        if list_date > entry["last_seen"]:
            entry["last_seen"] = list_date
    else:
        books[isbn] = {
            "title": title,
            "author": author,
            "weeks_on_list": 1,
            "peak_rank": rank,
            "list_categories": [list_name],
            "first_seen": list_date,
            "last_seen": list_date,
        }


def process_response(data: dict, books: dict, list_name: str, list_date: str):
    """Extract books from API response and merge into aggregate."""
    results = data.get("results", {})
    book_list = results.get("books", [])
    for book in book_list:
        title = book.get("title", "")
        author = book.get("author", "")
        rank = book.get("rank", 999)
        weeks = book.get("weeks_on_list", 0)
        isbns = book.get("isbns", [])

        # Collect all ISBN variants for this book
        isbn13s = []
        isbn10s = []
        for isbn_entry in isbns:
            i13 = (isbn_entry.get("isbn13") or "").strip()
            i10 = (isbn_entry.get("isbn10") or "").strip()
            if i13:
                isbn13s.append(i13)
            if i10:
                isbn10s.append(i10)

        # Also check top-level isbn fields
        primary_isbn13 = (book.get("primary_isbn13") or "").strip()
        primary_isbn10 = (book.get("primary_isbn10") or "").strip()
        if primary_isbn13 and primary_isbn13 not in isbn13s:
            isbn13s.append(primary_isbn13)
        if primary_isbn10 and primary_isbn10 not in isbn10s:
            isbn10s.append(primary_isbn10)

        # Merge under isbn13 first, then isbn10
        all_isbns = isbn13s + isbn10s
        if not all_isbns:
            continue

        for isbn in all_isbns:
            merge_book(books, isbn, title, author, rank, weeks, list_name, list_date)


def run_ingestion(api_key: str, max_requests: int | None = None):
    """Main ingestion loop with checkpointing and rate limiting."""
    cp = load_checkpoint()
    reset_daily_counter_if_needed(cp)

    schedule = build_schedule()
    start_idx = find_resume_index(schedule, cp)

    if start_idx >= len(schedule):
        print("All weeks already processed. Nothing to do.")
        save_checkpoint(cp)
        return

    print(f"Schedule: {len(schedule)} total (list, week) combinations")
    print(f"Resuming from index {start_idx} ({len(schedule) - start_idx} remaining)")
    print(f"Requests today: {cp['requests_today']} / {DAILY_LIMIT}")
    print(f"Total requests so far: {cp['total_requests']}")
    print(f"Books collected: {len(cp['books_collected']):,}")
    print()

    books = cp["books_collected"]
    requests_made = 0

    for i in range(start_idx, len(schedule)):
        list_date, list_name = schedule[i]

        # Check daily limit
        reset_daily_counter_if_needed(cp)
        if cp["requests_today"] >= DAILY_LIMIT:
            print(f"\nDaily limit reached ({DAILY_LIMIT} requests).")
            print("Save checkpoint and re-run tomorrow.")
            save_checkpoint(cp)
            return

        # Check max requests (for testing)
        if max_requests is not None and requests_made >= max_requests:
            print(f"\nMax requests ({max_requests}) reached.")
            save_checkpoint(cp)
            return

        # Rate-limit delay (skip before the first request of a session)
        if requests_made > 0:
            time.sleep(REQUEST_DELAY)

        # Fetch
        try:
            data = fetch_list(api_key, list_date, list_name)
            cp["requests_today"] += 1
            cp["total_requests"] += 1
            requests_made += 1

            if data is None:
                # List didn't exist for this date — skip
                pass
            else:
                process_response(data, books, list_name, list_date)

            cp["last_completed"] = {"list": list_name, "date": list_date}
            save_checkpoint(cp)

            remaining = len(schedule) - i - 1
            if requests_made % 10 == 0 or requests_made == 1:
                print(
                    f"  [{cp['total_requests']:,}] {list_date} / {list_name} | "
                    f"{len(books):,} books | {remaining:,} remaining"
                )

        except HTTPError as e:
            if e.code == 429:
                print(f"  Rate limited (429) at {list_date}/{list_name}. Backing off 60s...")
                save_checkpoint(cp)
                time.sleep(60)
                # Retry this same index by not updating last_completed
                continue
            else:
                print(f"  HTTP {e.code} for {list_date}/{list_name}: {e.reason}. Skipping.")
                cp["last_completed"] = {"list": list_name, "date": list_date}
                save_checkpoint(cp)

        except Exception as e:
            print(f"  Error for {list_date}/{list_name}: {e}. Skipping.")
            cp["last_completed"] = {"list": list_name, "date": list_date}
            save_checkpoint(cp)

    print("\n--- Ingestion Complete ---")
    print(f"Total requests: {cp['total_requests']:,}")
    print(f"Books collected: {len(books):,}")
    save_checkpoint(cp)


def show_status():
    """Show checkpoint status and estimated time remaining."""
    cp = load_checkpoint()
    books = cp["books_collected"]

    schedule = build_schedule()
    start_idx = find_resume_index(schedule, cp)
    remaining = len(schedule) - start_idx

    print("--- NYT Bestseller Ingestion Status ---")
    print(f"Total requests made:  {cp['total_requests']:,}")
    print(f"Requests today:       {cp['requests_today']} / {DAILY_LIMIT}")
    print(f"Books collected:      {len(books):,}")
    print(f"Schedule entries:     {len(schedule):,} total, {remaining:,} remaining")

    last = cp.get("last_completed")
    if last:
        print(f"Last checkpoint:      {last['date']} / {last['list']}")
    else:
        print("Last checkpoint:      (not started)")

    if remaining > 0:
        days_remaining = math.ceil(remaining / DAILY_LIMIT)
        print(f"Estimated days left:  ~{days_remaining}")
    else:
        print("Status:               COMPLETE")


def export_data():
    """Export current books_collected from checkpoint to output JSON."""
    cp = load_checkpoint()
    books = cp["books_collected"]
    if not books:
        print("No books collected yet. Nothing to export.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(books, f, indent=2)
    print(f"Exported {len(books):,} ISBN entries to {OUTPUT_PATH}")


def main():
    parser = argparse.ArgumentParser(description="NYT Books API bestseller ingestion")
    parser.add_argument(
        "--api-key", default=None,
        help="NYT API key (or set NYT_API_KEY env var)",
    )
    parser.add_argument(
        "--max-requests", type=int, default=None,
        help="Stop after N requests (for testing)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show checkpoint status and exit",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Export current collected data to output JSON",
    )
    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.export:
        export_data()
        return

    api_key = args.api_key or os.environ.get(NYT_API_KEY_ENV)
    if not api_key:
        print(f"Error: provide --api-key or set {NYT_API_KEY_ENV} env var.", file=sys.stderr)
        sys.exit(1)

    run_ingestion(api_key, max_requests=args.max_requests)


if __name__ == "__main__":
    main()
