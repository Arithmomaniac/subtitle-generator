"""Download Trove Australia book holdings and build an ISBN popularity lookup.

Output: data/trove_holdings_lookup.json
        (ISBN -> Trove holdings/library-count popularity metadata)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_BASE = "https://api.trove.nla.gov.au/v3"
USER_AGENT = "subtitle-generator/0.5.0 (research project; Trove holdings popularity)"
OUTPUT_PATH = Path("data/trove_holdings_lookup.json")
CHECKPOINT_PATH = Path("data/trove_checkpoint.json")
OL_LOOKUP_PATH = Path("data/ol_edition_lookup.json")
DB_PATH = Path("data/db/subtitles.db")

ISBN_TEXT_RE = re.compile(r"isbn|identifier|isxn", re.IGNORECASE)


class TroveQuotaExceeded(RuntimeError):
    """Raised when a run-local request budget has been consumed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _as_int(value, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.replace(",", ""))
        except ValueError:
            return default
    return default


def _isbn13_checksum(digits: str) -> int:
    total = sum((1 if idx % 2 == 0 else 3) * int(ch) for idx, ch in enumerate(digits[:12]))
    return (10 - (total % 10)) % 10


def _is_valid_isbn13(value: str) -> bool:
    return (
        len(value) == 13
        and value.isdigit()
        and value.startswith(("978", "979"))
        and _isbn13_checksum(value) == int(value[-1])
    )


def _is_valid_isbn10(value: str) -> bool:
    if len(value) != 10 or not value[:9].isdigit() or value[-1] not in "0123456789X":
        return False
    total = 0
    for idx, ch in enumerate(value):
        digit = 10 if ch == "X" else int(ch)
        total += (10 - idx) * digit
    return total % 11 == 0


def isbn10_to_isbn13(value: str) -> str | None:
    """Convert a valid ISBN-10 to ISBN-13."""

    clean = normalize_isbn(value)
    if not clean or len(clean) != 10:
        return None
    prefix = "978" + clean[:9]
    return prefix + str(_isbn13_checksum(prefix + "0"))


def normalize_isbn(value: str | int | None) -> str | None:
    """Return a normalized ISBN-10 or ISBN-13 without punctuation."""

    if value is None:
        return None
    clean = re.sub(r"[^0-9Xx]", "", str(value)).upper()
    if _is_valid_isbn13(clean) or _is_valid_isbn10(clean):
        return clean
    return None


def isbn_variants(value: str | int | None) -> set[str]:
    """Return normalized ISBN variants useful for matching existing aliases."""

    normalized = normalize_isbn(value)
    if not normalized:
        return set()
    variants = {normalized}
    if len(normalized) == 10:
        converted = isbn10_to_isbn13(normalized)
        if converted:
            variants.add(converted)
    return variants


def _scan_isbns(text: str) -> set[str]:
    """Extract validated ISBN-10/13 values from a string."""

    compact = re.sub(r"[^0-9Xx]", "", text).upper()
    found: set[str] = set()
    for size in (13, 10):
        if len(compact) < size:
            continue
        for idx in range(0, len(compact) - size + 1):
            found.update(isbn_variants(compact[idx : idx + size]))
    return found


def extract_isbns(value, in_identifier_context: bool = False) -> set[str]:
    """Recursively collect ISBN-like values from Trove metadata."""

    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_context = in_identifier_context or bool(ISBN_TEXT_RE.search(str(key)))
            found.update(extract_isbns(child, child_context))
    elif isinstance(value, list):
        for child in value:
            found.update(extract_isbns(child, in_identifier_context))
    elif isinstance(value, (str, int)):
        text = str(value)
        if in_identifier_context or ISBN_TEXT_RE.search(text):
            found.update(_scan_isbns(text))
    return found


def _first_text(value) -> str:
    for item in _as_list(value):
        if isinstance(item, str) and item.strip():
            return item.strip()
        if isinstance(item, dict):
            for key in ("value", "name"):
                text = item.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return ""


def _work_id(work: dict) -> str:
    value = work.get("id") or work.get("workId") or work.get("@id")
    if isinstance(value, str):
        return value
    return str(value) if value is not None else ""


def _holding_entries(work: dict) -> list[dict]:
    return [item for item in _as_list(work.get("holding")) if isinstance(item, dict)]


def extract_library_codes(work: dict) -> list[str]:
    """Return distinct library identifiers from full-work holding entries."""

    codes: set[str] = set()
    for holding in _holding_entries(work):
        nuc = _first_text(holding.get("nuc"))
        if nuc:
            codes.add(nuc)
            continue
        name = _first_text(holding.get("name")) or _first_text(holding.get("contributor"))
        if name:
            codes.add(name)
    return sorted(codes)


def _version_holding_count(work: dict) -> int:
    """Return the strongest version-level holdings signal without double-counting."""

    counts = []
    for version in _as_list(work.get("version")):
        if isinstance(version, dict):
            counts.append(_as_int(version.get("holdingsCount")))
    return max(counts, default=0)


def normalize_work(
    work: dict,
    *,
    queried_isbn: str | None = None,
    include_libraries: bool = False,
    checked_at: str | None = None,
) -> dict[str, dict]:
    """Normalize one Trove work into ISBN-keyed lookup entries."""

    isbns = extract_isbns(work)
    if queried_isbn:
        isbns.update(isbn_variants(queried_isbn))
    if not isbns:
        return {}

    libraries = extract_library_codes(work)
    holdings_count = _as_int(work.get("holdingsCount"))
    holding_count = holdings_count or len(libraries) or len(_holding_entries(work))
    library_count = holdings_count or len(libraries) or holding_count
    copy_count = holding_count
    work_id = _work_id(work)
    checked_at = checked_at or _utc_now()

    entry = {
        "source": "trove",
        "trove_work_id": work_id,
        "title": _first_text(work.get("title")),
        "author": _first_text(work.get("contributor")),
        "issued": _first_text(work.get("issued")),
        "trove_url": _first_text(work.get("troveUrl") or work.get("url")),
        "library_count": library_count,
        "holding_count": holding_count,
        "copy_count": copy_count,
        "copy_count_is_exact": False,
        "copy_count_basis": "holdings_count_proxy",
        "version_count": _as_int(work.get("versionCount")),
        "version_holding_count": _version_holding_count(work),
        "last_checked": checked_at,
    }
    if include_libraries:
        entry["libraries"] = libraries

    return {isbn: dict(entry) for isbn in sorted(isbns)}


def _records_from_result(payload: dict) -> list[dict]:
    works: list[dict] = []
    for category in _as_list(payload.get("category")):
        if not isinstance(category, dict):
            continue
        records = category.get("records")
        if isinstance(records, dict):
            for work in _as_list(records.get("work")):
                if isinstance(work, dict):
                    works.append(work)
    return works


def _next_start(payload: dict) -> str | None:
    for category in _as_list(payload.get("category")):
        if isinstance(category, dict) and category.get("nextStart"):
            return str(category["nextStart"])
    return None


class RateLimiter:
    """Fixed-window limiter compatible with Trove's calls-per-minute quota."""

    def __init__(self, rate_per_minute: int, state: dict | None = None):
        self.rate_per_minute = rate_per_minute
        state = state or {}
        self.window_started_at = float(state.get("window_started_at_epoch") or 0.0)
        self.requests_in_window = int(state.get("requests_in_current_window") or 0)
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            if now - self.window_started_at >= 60:
                self.window_started_at = now
                self.requests_in_window = 0
            if self.requests_in_window >= self.rate_per_minute:
                sleep_for = max(0.0, 60 - (now - self.window_started_at))
                if sleep_for > 0:
                    time.sleep(sleep_for)
                self.window_started_at = time.time()
                self.requests_in_window = 0
            self.requests_in_window += 1

    def checkpoint_state(self) -> dict:
        with self._lock:
            return {
                "window_started_at_epoch": self.window_started_at,
                "requests_in_current_window": self.requests_in_window,
            }


class TroveClient:
    def __init__(
        self,
        api_key: str,
        *,
        rate_per_minute: int,
        quota_per_minute: int,
        max_requests: int | None,
        checkpoint: dict,
    ):
        if rate_per_minute > quota_per_minute:
            raise ValueError(
                f"rate_per_minute ({rate_per_minute}) cannot exceed quota_per_minute ({quota_per_minute})"
            )
        self.api_key = api_key
        self.max_requests = max_requests
        self.checkpoint = checkpoint
        self.requests_made = int(checkpoint.get("requests_made") or 0)
        self.limiter = RateLimiter(rate_per_minute, checkpoint)
        self._request_lock = threading.Lock()

    def _persist_request_state(self) -> None:
        self.checkpoint["requests_made"] = self.requests_made
        self.checkpoint.update(self.limiter.checkpoint_state())

    def get_json(self, path: str, params: dict[str, str | int | bool], *, retries: int = 6) -> dict:
        request_params = dict(params)
        request_params["key"] = self.api_key
        request_params.setdefault("encoding", "json")
        url = f"{API_BASE}/{path.lstrip('/')}?{urlencode(request_params)}"
        redacted_url = url.replace(self.api_key, "<TROVE_API_KEY>")

        for attempt in range(retries + 1):
            self.limiter.wait()
            with self._request_lock:
                if self.max_requests is not None and self.requests_made >= self.max_requests:
                    raise TroveQuotaExceeded(f"Reached --max-requests={self.max_requests}")
                self.requests_made += 1
                self._persist_request_state()
            req = Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": USER_AGENT,
                },
            )
            try:
                with urlopen(req, timeout=90) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code == 429 and attempt < retries:
                    time.sleep(60)
                    continue
                if exc.code in {500, 502, 503, 504} and attempt < retries:
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise RuntimeError(f"Trove request failed ({exc.code}) for {redacted_url}") from exc
            except (URLError, TimeoutError, ConnectionError) as exc:
                if attempt < retries:
                    time.sleep(min(60, 2 ** attempt))
                    continue
                raise RuntimeError(f"Trove request failed for {redacted_url}") from exc

        raise RuntimeError(f"Trove request retries exhausted for {redacted_url}")

    def search(self, query: str, *, n: int = 5) -> list[dict]:
        payload = self.get_json(
            "result",
            {
                "category": "book",
                "q": query,
                "n": n,
            },
        )
        return _records_from_result(payload)

    def bulk_page(self, *, cursor: str = "*", n: int = 100, full: bool = False) -> tuple[list[dict], str | None]:
        params: dict[str, str | int | bool] = {
            "category": "book",
            "bulkHarvest": "true",
            "n": n,
            "s": cursor,
        }
        if full:
            params["reclevel"] = "full"
            params["include"] = "workversions"
        payload = self.get_json("result", params)
        return _records_from_result(payload), _next_start(payload)

    def work(self, work_id: str) -> dict:
        return self.get_json(
            f"work/{work_id}",
            {
                "reclevel": "full",
                "include": "holdings,workversions",
            },
        )


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _load_isbns_from_json_lookup(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    isbns: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            isbns.update(isbn_variants(key))
            isbns.update(extract_isbns(value))
    return isbns


def _load_isbns_from_text(path: Path) -> set[str]:
    isbns: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        isbns.update(isbn_variants(line.strip()))
    return isbns


def _load_isbns_from_db(path: Path, target_mode: str = "all") -> set[str]:
    if not path.exists():
        return set()
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        isbns: set[str] = set()
        if target_mode == "slot-sources":
            required = {
                "slot_filler_sources",
                "slot_fillers",
                "subtitles",
                "pattern_matches",
            }
            if not required <= tables:
                return set()
            pattern_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(pattern_matches)")
            }
            if not {"subtitle_id", "list_items_json"} <= pattern_columns:
                return set()
            for (isbn,) in conn.execute("""
                SELECT DISTINCT s.isbn
                FROM slot_filler_sources sfs
                JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
                JOIN subtitles s ON s.id = sfs.subtitle_id
                JOIN pattern_matches pm ON pm.subtitle_id = sfs.subtitle_id
                WHERE sf.mode = 'strict'
                  AND s.isbn IS NOT NULL
                  AND s.isbn != ''
                  AND json_valid(pm.list_items_json)
                  AND json_array_length(pm.list_items_json) IN (2, 3)
            """):
                isbns.update(isbn_variants(isbn))
            return isbns

        if "isbn_aliases" in tables:
            for (isbn,) in conn.execute("SELECT isbn FROM isbn_aliases"):
                isbns.update(isbn_variants(isbn))
        if "subtitles" in tables:
            for (isbn,) in conn.execute("SELECT isbn FROM subtitles"):
                isbns.update(isbn_variants(isbn))
        return isbns
    finally:
        conn.close()


def load_target_isbns(
    paths: list[Path],
    db_path: Path | None,
    include_ol: bool = True,
    db_target_mode: str = "all",
) -> set[str]:
    isbns: set[str] = set()
    for path in paths:
        if path.suffix.lower() == ".json":
            isbns.update(_load_isbns_from_json_lookup(path))
        else:
            isbns.update(_load_isbns_from_text(path))
    if include_ol:
        isbns.update(_load_isbns_from_json_lookup(OL_LOOKUP_PATH))
    if db_path:
        isbns.update(_load_isbns_from_db(db_path, db_target_mode))
    return isbns


def _is_fresh(entry: dict, refresh_days: int) -> bool:
    if refresh_days <= 0:
        return False
    raw = entry.get("last_checked")
    if not isinstance(raw, str) or not raw:
        return False
    try:
        checked = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (datetime.now(timezone.utc) - checked).days < refresh_days


def _best_work(works: list[dict]) -> dict | None:
    if not works:
        return None
    return max(works, key=lambda work: _as_int(work.get("holdingsCount")))


def _merge_entries(lookup: dict[str, dict], entries: dict[str, dict]) -> int:
    changed = 0
    for isbn, entry in entries.items():
        existing = lookup.get(isbn)
        if not existing or entry.get("library_count", 0) >= existing.get("library_count", 0):
            lookup[isbn] = entry
            changed += 1
    return changed


def fetch_isbn_entry(
    client: TroveClient,
    isbn: str,
    *,
    fetch_details: bool,
    include_libraries: bool,
) -> tuple[str, dict[str, dict], str | None]:
    """Fetch one ISBN and return normalized lookup entries or a failure string."""

    try:
        works = client.search(f"isbn:{isbn}", n=5)
        if not works:
            works = client.search(f"identifier:{isbn}", n=5)
        work = _best_work(works)
        if not work:
            return isbn, {}, None
        if fetch_details or include_libraries:
            work_id = _work_id(work)
            if work_id:
                work = client.work(work_id)
        return isbn, normalize_work(work, queried_isbn=isbn, include_libraries=include_libraries), None
    except TroveQuotaExceeded:
        raise
    except RuntimeError as exc:
        return isbn, {}, str(exc)


def build_lookup_from_csv(
    csv_path: Path,
    *,
    target_isbns: set[str] | None = None,
    include_libraries: bool = False,
) -> dict[str, dict]:
    """Build lookup entries from a manually exported Trove bulk-download CSV."""

    lookup: dict[str, dict] = {}
    text = csv_path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.DictReader(text.splitlines())
    if not reader.fieldnames:
        return lookup

    isbn_cols = [name for name in reader.fieldnames if ISBN_TEXT_RE.search(name)]
    id_cols = [name for name in reader.fieldnames if re.search(r"\b(work|trove).*id\b|\bid\b", name, re.IGNORECASE)]
    count_cols = [
        name
        for name in reader.fieldnames
        if re.search(r"holdings?count|libraries|library_count", name, re.IGNORECASE)
    ]
    title_cols = [name for name in reader.fieldnames if re.search(r"title", name, re.IGNORECASE)]
    author_cols = [name for name in reader.fieldnames if re.search(r"author|contributor|creator", name, re.IGNORECASE)]

    checked_at = _utc_now()
    for row in reader:
        row_isbns: set[str] = set()
        for col in isbn_cols:
            row_isbns.update(extract_isbns({col: row.get(col)}))
        if target_isbns is not None:
            row_isbns &= target_isbns
        if not row_isbns:
            continue
        library_count = max([_as_int(row.get(col)) for col in count_cols], default=0)
        entry = {
            "source": "trove",
            "trove_work_id": _first_text([row.get(col, "") for col in id_cols]),
            "title": _first_text([row.get(col, "") for col in title_cols]),
            "author": _first_text([row.get(col, "") for col in author_cols]),
            "library_count": library_count,
            "holding_count": library_count,
            "copy_count": library_count,
            "copy_count_is_exact": False,
            "copy_count_basis": "holdings_count_proxy",
            "last_checked": checked_at,
        }
        if include_libraries:
            entry["libraries"] = []
        _merge_entries(lookup, {isbn: dict(entry) for isbn in row_isbns})
    return lookup


def download_by_isbn(
    client: TroveClient,
    target_isbns: set[str],
    lookup: dict[str, dict],
    checkpoint: dict,
    *,
    limit: int | None,
    refresh_days: int,
    fetch_details: bool,
    include_libraries: bool,
    save_every: int,
    workers: int,
    output_path: Path,
    checkpoint_path: Path,
) -> int:
    processed = set(checkpoint.get("processed_isbns") or [])
    failed = dict(checkpoint.get("failed_isbns") or {})
    added = 0
    candidates: list[str] = []
    for isbn in sorted(target_isbns):
        if isbn in processed:
            continue
        if isbn in lookup and _is_fresh(lookup[isbn], refresh_days):
            processed.add(isbn)
            continue
        candidates.append(isbn)
        if limit is not None and len(candidates) >= limit:
            break

    def sync_checkpoint() -> None:
        checkpoint["processed_isbns"] = sorted(processed)
        checkpoint["failed_isbns"] = failed

    def persist(checked: int) -> None:
        sync_checkpoint()
        save_json(output_path, lookup)
        save_json(checkpoint_path, checkpoint)
        print(f"  Checked {checked:,} ISBNs this run; lookup has {len(lookup):,} entries")

    checked = 0
    try:
        if workers <= 1:
            for isbn in candidates:
                isbn, entries, failure = fetch_isbn_entry(
                    client,
                    isbn,
                    fetch_details=fetch_details,
                    include_libraries=include_libraries,
                )
                if failure:
                    failed[isbn] = failure
                    print(f"  Failed {isbn}: {failure}", file=sys.stderr)
                else:
                    added += _merge_entries(lookup, entries)
                    processed.add(isbn)
                checked += 1
                if checked % save_every == 0:
                    persist(checked)
        else:
            executor = ThreadPoolExecutor(max_workers=workers)
            futures = [
                executor.submit(
                    fetch_isbn_entry,
                    client,
                    isbn,
                    fetch_details=fetch_details,
                    include_libraries=include_libraries,
                )
                for isbn in candidates
            ]
            try:
                for future in as_completed(futures):
                    isbn, entries, failure = future.result()
                    if failure:
                        failed[isbn] = failure
                        print(f"  Failed {isbn}: {failure}", file=sys.stderr)
                    else:
                        added += _merge_entries(lookup, entries)
                        processed.add(isbn)
                    checked += 1
                    if checked % save_every == 0:
                        persist(checked)
            except TroveQuotaExceeded:
                for future in futures:
                    future.cancel()
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
    finally:
        sync_checkpoint()
        save_json(output_path, lookup)
        save_json(checkpoint_path, checkpoint)
    return added


def download_bulk_pages(
    client: TroveClient,
    lookup: dict[str, dict],
    checkpoint: dict,
    *,
    target_isbns: set[str] | None,
    pages: int,
    page_size: int,
    full: bool,
    include_libraries: bool,
    output_path: Path,
    checkpoint_path: Path,
) -> int:
    cursor = str(checkpoint.get("bulk_cursor") or "*")
    added = 0
    for page in range(pages):
        works, next_start = client.bulk_page(cursor=cursor, n=page_size, full=full)
        if full and not next_start:
            _, next_start = client.bulk_page(cursor=cursor, n=page_size, full=False)
        for work in works:
            entries = normalize_work(work, include_libraries=include_libraries)
            if target_isbns is not None:
                entries = {isbn: entry for isbn, entry in entries.items() if isbn in target_isbns}
            added += _merge_entries(lookup, entries)
        checkpoint["bulk_cursor"] = next_start
        save_json(output_path, lookup)
        save_json(checkpoint_path, checkpoint)
        print(f"  Bulk page {page + 1:,}/{pages:,}: {len(works):,} works, lookup has {len(lookup):,} entries")
        if not next_start:
            break
        cursor = next_start
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Trove holdings into an ISBN lookup")
    parser.add_argument("--api-key", default=os.environ.get("TROVE_API_KEY"), help="Trove API key or TROVE_API_KEY")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--target-isbns", type=Path, action="append", default=[], help="Text or JSON file containing target ISBNs")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="Optional subtitles DB used to load target ISBNs")
    parser.add_argument(
        "--db-target-mode",
        choices=["all", "slot-sources"],
        default="all",
        help="Which ISBNs to load from --db",
    )
    parser.add_argument("--no-ol-targets", action="store_true", help="Do not load target ISBNs from data/ol_edition_lookup.json")
    parser.add_argument("--limit", type=int, default=None, help="Maximum ISBNs to check this run")
    parser.add_argument("--max-requests", type=int, default=None, help="Maximum Trove API requests this run")
    parser.add_argument("--rate-per-minute", type=int, default=180, help="Request rate, capped by --quota-per-minute")
    parser.add_argument("--quota-per-minute", type=int, default=200, help="Approved Trove API quota")
    parser.add_argument("--refresh-days", type=int, default=30, help="Skip lookup entries checked more recently than this")
    parser.add_argument("--include-libraries", action="store_true", help="Persist distinct library NUC/name list")
    parser.add_argument("--fetch-details", action="store_true", help="Fetch full work records for matched ISBNs")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent targeted ISBN workers")
    parser.add_argument("--bulk-pages", type=int, default=0, help="Also harvest this many bulk result pages")
    parser.add_argument("--bulk-full", action="store_true", help="Use reclevel=full&include=workversions for bulk pages")
    parser.add_argument("--bulk-page-size", type=int, default=100)
    parser.add_argument("--csv", type=Path, action="append", default=[], help="Import manually exported Trove bulk CSV before API calls")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint")
    args = parser.parse_args()

    if not args.api_key and (args.limit != 0 or args.bulk_pages):
        raise SystemExit("Trove requires --api-key or TROVE_API_KEY")

    target_isbns = load_target_isbns(
        args.target_isbns,
        args.db,
        include_ol=not args.no_ol_targets,
        db_target_mode=args.db_target_mode,
    )
    lookup = load_json(args.output, {})
    checkpoint = {} if args.no_resume else load_json(args.checkpoint, {})

    for csv_path in args.csv:
        imported = build_lookup_from_csv(
            csv_path,
            target_isbns=target_isbns or None,
            include_libraries=args.include_libraries,
        )
        _merge_entries(lookup, imported)
        print(f"Imported {len(imported):,} entries from {csv_path}")

    if args.api_key:
        client = TroveClient(
            args.api_key,
            rate_per_minute=args.rate_per_minute,
            quota_per_minute=args.quota_per_minute,
            max_requests=args.max_requests,
            checkpoint=checkpoint,
        )
        try:
            if args.bulk_pages:
                download_bulk_pages(
                    client,
                    lookup,
                    checkpoint,
                    target_isbns=target_isbns or None,
                    pages=args.bulk_pages,
                    page_size=args.bulk_page_size,
                    full=args.bulk_full,
                    include_libraries=args.include_libraries,
                    output_path=args.output,
                    checkpoint_path=args.checkpoint,
                )
            if target_isbns and args.limit != 0:
                download_by_isbn(
                    client,
                    target_isbns,
                    lookup,
                    checkpoint,
                    limit=args.limit,
                    refresh_days=args.refresh_days,
                    fetch_details=args.fetch_details,
                    include_libraries=args.include_libraries,
                    save_every=args.save_every,
                    workers=max(args.workers, 1),
                    output_path=args.output,
                    checkpoint_path=args.checkpoint,
                )
        except TroveQuotaExceeded as exc:
            save_json(args.output, lookup)
            save_json(args.checkpoint, checkpoint)
            print(f"Stopped: {exc}", file=sys.stderr)

    save_json(args.output, lookup)
    save_json(args.checkpoint, checkpoint)
    print(f"Done. Trove lookup: {len(lookup):,} ISBN entries -> {args.output}")


if __name__ == "__main__":
    main()
