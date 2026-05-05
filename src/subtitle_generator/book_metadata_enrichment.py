"""Offline metadata sidecar builder for rich book-tier modeling."""

from __future__ import annotations

import csv
import gzip
import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from pymarc import MARCReader

from subtitle_generator.extract import DATA_DIR

METADATA_COLUMNS = (
    "subtitle_id",
    "metadata_source",
    "work_key",
    "edition_key",
    "publisher_text",
    "publisher_count",
    "publish_year",
    "edition_name",
    "physical_format",
    "physical_dimensions",
    "physical_description",
    "number_of_pages",
    "is_hardcover",
    "is_paperback",
    "is_ebook",
    "is_large_print",
    "subject_text",
    "subject_count",
    "author_count",
    "loc_call_number",
    "dewey_decimal",
    "has_physical_format_metadata",
    "has_publisher_metadata",
    "has_subject_metadata",
)

OL_DUMP_PATH = DATA_DIR / "raw" / "ol_dump_editions_latest.txt.gz"
LOC_RAW_DIR = DATA_DIR / "raw"


@dataclass(frozen=True)
class BookMetadataBuildResult:
    metadata_path: Path
    report_path: Path
    target_count: int
    enriched_count: int


@dataclass
class BookMetadataRow:
    subtitle_id: int
    metadata_source: str = ""
    work_key: str = ""
    edition_key: str = ""
    publisher_text: str = ""
    publisher_count: int = 0
    publish_year: int | None = None
    edition_name: str = ""
    physical_format: str = ""
    physical_dimensions: str = ""
    physical_description: str = ""
    number_of_pages: int | None = None
    is_hardcover: int = 0
    is_paperback: int = 0
    is_ebook: int = 0
    is_large_print: int = 0
    subject_text: str = ""
    subject_count: int = 0
    author_count: int = 0
    loc_call_number: str = ""
    dewey_decimal: str = ""
    has_physical_format_metadata: int = 0
    has_publisher_metadata: int = 0
    has_subject_metadata: int = 0


@dataclass(frozen=True)
class _SourceTarget:
    subtitle_id: int
    source_file: str
    isbn: str
    lccn: str


def build_book_metadata_artifact(
    conn: sqlite3.Connection,
    *,
    output_dir: Path,
    ol_dump_path: Path = OL_DUMP_PATH,
    loc_raw_dir: Path = LOC_RAW_DIR,
    max_ol_lines: int = 0,
    max_loc_records_per_file: int = 0,
) -> BookMetadataBuildResult:
    """Build an offline metadata sidecar from raw OL and LOC sources.

    The sidecar is for training/evaluation only. None of these fields are
    required by the runtime CSV or mini-DB export.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    targets = _load_targets(conn)
    rows = {target.subtitle_id: BookMetadataRow(target.subtitle_id) for target in targets}

    _enrich_from_openlibrary(
        rows,
        targets,
        ol_dump_path=ol_dump_path,
        max_lines=max_ol_lines,
    )
    _enrich_from_loc_marc(
        rows,
        targets,
        raw_dir=loc_raw_dir,
        max_records_per_file=max_loc_records_per_file,
    )

    ordered_rows = tuple(rows[subtitle_id] for subtitle_id in sorted(rows))
    metadata_path = output_dir / "book_metadata.csv"
    report_path = output_dir / "book_metadata_report.md"
    _write_metadata_csv(metadata_path, ordered_rows)
    report_path.write_text(
        format_metadata_report(ordered_rows, metadata_path),
        encoding="utf-8",
    )
    enriched_count = sum(1 for row in ordered_rows if row.metadata_source)
    return BookMetadataBuildResult(
        metadata_path=metadata_path,
        report_path=report_path,
        target_count=len(targets),
        enriched_count=enriched_count,
    )


def format_metadata_report(
    rows: tuple[BookMetadataRow, ...],
    metadata_path: Path,
) -> str:
    total = len(rows)
    source_counts: dict[str, int] = {}
    for row in rows:
        key = row.metadata_source or "missing"
        source_counts[key] = source_counts.get(key, 0) + 1
    physical = sum(row.has_physical_format_metadata for row in rows)
    publisher = sum(row.has_publisher_metadata for row in rows)
    subject = sum(row.has_subject_metadata for row in rows)
    lines = [
        "# Book metadata sidecar",
        "",
        "This artifact enriches offline book-tier training with raw LOC/Open "
        "Library metadata that is intentionally not part of the runtime export.",
        "",
        "## Outputs",
        "",
        f"- Metadata: `{metadata_path}` ({total:,} rows)",
        "",
        "## Coverage",
        "",
        f"- Metadata source: {_format_counts(source_counts)}",
        f"- Physical/format metadata: {_format_coverage(physical, total)}",
        f"- Publisher metadata: {_format_coverage(publisher, total)}",
        f"- Subject metadata: {_format_coverage(subject, total)}",
        "",
        "## Gate notes",
        "",
        "- These columns are allowed for rich offline training and teacher-model "
        "evaluation.",
        "- Downstream runtime/export distillation must not depend on these raw "
        "metadata columns being available.",
    ]
    return "\n".join(lines)


def load_book_metadata_rows(path: Path | None) -> dict[int, dict[str, str]]:
    """Load a generated metadata sidecar keyed by subtitle_id."""

    if path is None:
        return {}
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            int(row["subtitle_id"]): row
            for row in csv.DictReader(handle)
            if row.get("subtitle_id")
        }


def _load_targets(conn: sqlite3.Connection) -> tuple[_SourceTarget, ...]:
    rows = conn.execute(
        """
        SELECT DISTINCT
               s.id,
               COALESCE(s.source_file, ''),
               COALESCE(s.isbn, ''),
               COALESCE(s.lccn, '')
        FROM pattern_matches pm
        JOIN subtitles s ON s.id = pm.subtitle_id
        ORDER BY s.id
        """
    ).fetchall()
    return tuple(
        _SourceTarget(
            subtitle_id=row[0],
            source_file=row[1],
            isbn=_normalize_isbn(row[2]),
            lccn=_normalize_lccn(row[3]),
        )
        for row in rows
    )


def _enrich_from_openlibrary(
    rows: dict[int, BookMetadataRow],
    targets: tuple[_SourceTarget, ...],
    *,
    ol_dump_path: Path,
    max_lines: int,
) -> None:
    isbn_to_targets: dict[str, list[_SourceTarget]] = {}
    for target in targets:
        if target.source_file == "openlibrary" and target.isbn:
            isbn_to_targets.setdefault(target.isbn, []).append(target)
    if not isbn_to_targets or not ol_dump_path.exists():
        return

    with gzip.open(ol_dump_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line_index, line in enumerate(handle, start=1):
            if max_lines and line_index > max_lines:
                break
            parts = line.split("\t", 4)
            if len(parts) < 5 or parts[0].strip() != "/type/edition":
                continue
            try:
                data = json.loads(parts[4])
            except json.JSONDecodeError:
                continue
            record_isbns = {
                _normalize_isbn(isbn)
                for isbn in [
                    *data.get("isbn_13", []),
                    *data.get("isbn_10", []),
                ]
                if isinstance(isbn, str)
            }
            matched_targets = [
                target
                for isbn in record_isbns
                for target in isbn_to_targets.get(isbn, [])
            ]
            if not matched_targets:
                continue
            for target in matched_targets:
                rows[target.subtitle_id] = _merge_prefer_existing(
                    rows[target.subtitle_id],
                    _metadata_from_ol(target.subtitle_id, data, parts[1].strip()),
                )
            for isbn in record_isbns:
                isbn_to_targets.pop(isbn, None)
            if not isbn_to_targets:
                break


def _metadata_from_ol(
    subtitle_id: int,
    data: dict,
    edition_key: str,
) -> BookMetadataRow:
    publishers = _string_list(data.get("publishers"))
    subjects = _string_list(data.get("subjects"))
    physical_format = _clean_text(data.get("physical_format", ""))
    physical_dimensions = _clean_text(data.get("physical_dimensions", ""))
    pages = _int_or_none(data.get("number_of_pages"))
    format_text = " ".join(
        value
        for value in (physical_format, physical_dimensions, str(pages or ""))
        if value
    )
    works = data.get("works", [])
    work_key = ""
    if works and isinstance(works[0], dict):
        work_key = _clean_text(works[0].get("key", ""))
    authors = data.get("authors", [])
    return BookMetadataRow(
        subtitle_id=subtitle_id,
        metadata_source="openlibrary_dump",
        work_key=work_key,
        edition_key=edition_key,
        publisher_text=" | ".join(publishers[:5]),
        publisher_count=len(publishers),
        publish_year=_extract_year(_clean_text(data.get("publish_date", ""))),
        edition_name=_clean_text(data.get("edition_name", "")),
        physical_format=physical_format,
        physical_dimensions=physical_dimensions,
        number_of_pages=pages,
        is_hardcover=_contains_any(format_text, ("hardcover", "hardback")),
        is_paperback=_contains_any(format_text, ("paperback", "softcover")),
        is_ebook=_contains_any(format_text, ("ebook", "e-book", "electronic")),
        is_large_print=_contains_any(format_text, ("large print",)),
        subject_text=" | ".join(subjects[:20]),
        subject_count=len(subjects),
        author_count=len(authors) if isinstance(authors, list) else 0,
        has_physical_format_metadata=int(bool(physical_format or physical_dimensions or pages)),
        has_publisher_metadata=int(bool(publishers)),
        has_subject_metadata=int(bool(subjects)),
    )


def _enrich_from_loc_marc(
    rows: dict[int, BookMetadataRow],
    targets: tuple[_SourceTarget, ...],
    *,
    raw_dir: Path,
    max_records_per_file: int,
) -> None:
    targets_by_file: dict[str, dict[str, list[_SourceTarget]]] = {}
    for target in targets:
        if target.source_file and target.source_file != "openlibrary" and target.lccn:
            by_lccn = targets_by_file.setdefault(target.source_file, {})
            by_lccn.setdefault(target.lccn, []).append(target)
    if not targets_by_file:
        return

    for source_file, by_lccn in sorted(targets_by_file.items()):
        marc_path = raw_dir / source_file
        if not marc_path.exists():
            continue
        seen = 0
        with open(marc_path, "rb") as handle:
            reader = MARCReader(
                handle,
                to_unicode=True,
                force_utf8=False,
                utf8_handling="replace",
            )
            for record in reader:
                if record is None:
                    continue
                seen += 1
                if max_records_per_file and seen > max_records_per_file:
                    break
                lccn = _normalize_lccn(_first_subfield(record, "010", "a"))
                matched = by_lccn.get(lccn)
                if not matched:
                    continue
                for target in matched:
                    rows[target.subtitle_id] = _merge_prefer_existing(
                        rows[target.subtitle_id],
                        _metadata_from_marc(target.subtitle_id, record),
                    )
                by_lccn.pop(lccn, None)
                if not by_lccn:
                    break


def _metadata_from_marc(subtitle_id: int, record) -> BookMetadataRow:
    publishers = _marc_subfields(record, ("264", "260"), "b")
    subjects = _marc_subjects(record)
    physical_description = _clean_text(_first_subfield(record, "300", "a"))
    edition_name = _clean_text(_first_subfield(record, "250", "a"))
    loc_call = _clean_text(
        " ".join(
            value
            for value in (
                _first_subfield(record, "050", "a"),
                _first_subfield(record, "050", "b"),
            )
            if value
        )
    )
    dewey = _clean_text(_first_subfield(record, "082", "a"))
    author_count = sum(1 for field in record.get_fields("100", "700") if field)
    return BookMetadataRow(
        subtitle_id=subtitle_id,
        metadata_source="loc_marc",
        publisher_text=" | ".join(publishers[:5]),
        publisher_count=len(publishers),
        publish_year=_extract_year(
            _first_subfield(record, "264", "c")
            or _first_subfield(record, "260", "c")
            or _control_field_008_year(record)
        ),
        edition_name=edition_name,
        physical_description=physical_description,
        number_of_pages=_extract_pages(physical_description),
        subject_text=" | ".join(subjects[:20]),
        subject_count=len(subjects),
        author_count=author_count,
        loc_call_number=loc_call,
        dewey_decimal=dewey,
        has_physical_format_metadata=int(bool(physical_description)),
        has_publisher_metadata=int(bool(publishers)),
        has_subject_metadata=int(bool(subjects)),
    )


def _merge_prefer_existing(
    existing: BookMetadataRow,
    incoming: BookMetadataRow,
) -> BookMetadataRow:
    if not existing.metadata_source:
        return incoming
    for column in METADATA_COLUMNS:
        if column == "subtitle_id":
            continue
        current = getattr(existing, column)
        new = getattr(incoming, column)
        if current in ("", 0, None) and new not in ("", 0, None):
            setattr(existing, column, new)
    if incoming.metadata_source and incoming.metadata_source not in existing.metadata_source:
        existing.metadata_source = f"{existing.metadata_source}+{incoming.metadata_source}"
    return existing


def _write_metadata_csv(path: Path, rows: tuple[BookMetadataRow, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _marc_subfields(record, tags: tuple[str, ...], code: str) -> list[str]:
    values: list[str] = []
    for field in record.get_fields(*tags):
        value = _clean_text(field.get(code, ""))
        if value:
            values.append(value)
    return values


def _marc_subjects(record) -> list[str]:
    subjects: list[str] = []
    for field in record.get_fields("600", "610", "611", "630", "650", "651", "655"):
        pieces = [
            _clean_text(field.get(code, ""))
            for code in ("a", "x", "y", "z", "v")
        ]
        text = " -- ".join(piece for piece in pieces if piece)
        if text:
            subjects.append(text)
    return subjects


def _first_subfield(record, tag: str, code: str) -> str:
    field = record.get(tag)
    if not field:
        return ""
    return _clean_text(field.get(code, ""))


def _control_field_008_year(record) -> str:
    field = record.get("008")
    raw = field.data if field and hasattr(field, "data") else ""
    return raw[7:11] if len(raw) >= 11 else ""


def _string_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [
        cleaned
        for item in value
        if isinstance(item, str) and (cleaned := _clean_text(item))
    ]


def _clean_text(value) -> str:
    if value is None:
        return ""
    text = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(r"[\s,;:/]+$", "", text).strip()


def _normalize_isbn(raw: str) -> str:
    return re.sub(r"[\s-]", "", (raw or "").strip())


def _normalize_lccn(raw: str) -> str:
    return re.sub(r"[\s-]", "", (raw or "").strip())


def _extract_year(raw: str) -> int | None:
    match = re.search(r"(1[5-9]\d{2}|20\d{2})", raw or "")
    return int(match.group(1)) if match else None


def _extract_pages(raw: str) -> int | None:
    match = re.search(r"\b(\d{1,5})\s*(?:p\.|pages?)\b", raw or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def _int_or_none(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _contains_any(text: str, needles: Iterable[str]) -> int:
    lower = text.lower()
    return int(any(needle in lower for needle in needles))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(
        f"{key}={value:,}"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _format_coverage(count: int, total: int) -> str:
    if not total:
        return "0/0"
    return f"{count:,}/{total:,} ({count / total:.1%})"
