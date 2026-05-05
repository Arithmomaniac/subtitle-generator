"""Read-only inventory for offline book-model planning."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from pathlib import Path


FULL_DB_TABLES = (
    "subtitles",
    "pattern_matches",
    "slot_fillers",
    "slot_filler_sources",
    "isbn_aliases",
    "popularity_data",
    "config",
    "human_ratings",
)
MINI_DB_TABLES = (
    "slot_fillers",
    "sources",
    "config",
)
EXPORTABLE_SLOT_FILLER_COLUMNS = (
    "id",
    "slot_type",
    "filler",
    "mode",
    "source_subtitle_id",
    "freq",
    "pos_tag",
    "prep",
    "remix_type",
    "remix_prep",
    "remix_word_count",
    "centroid_dot",
    "norm_sq",
    "token_count",
    "popularity_score",
    "popularity_level",
    "popularity_confidence",
)
EXPORTABLE_SOURCE_COLUMNS = (
    "slot_filler_id",
    "title",
    "subtitle_text",
    "source_tag",
)
OFFLINE_ONLY_TABLES = (
    "subtitles",
    "pattern_matches",
    "slot_filler_sources",
    "isbn_aliases",
    "popularity_data",
)
DESIRED_MODEL_ARTIFACTS = (
    "book_features",
    "book_labels",
    "book_predictions",
    "slot_export_model",
    "filler_book_rollups",
)


@dataclass(frozen=True)
class TableInventory:
    name: str
    exists: bool
    row_count: int | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseInventory:
    label: str
    path: Path
    exists: bool
    tables: tuple[TableInventory, ...]
    pattern_label_counts: tuple[tuple[str, int], ...]
    candidate_source_counts: tuple[tuple[str, int], ...]
    slot_type_counts: tuple[tuple[str, int], ...]
    source_tag_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CsvInventory:
    path: Path
    exists: bool
    row_count: int | None
    columns: tuple[str, ...]


@dataclass(frozen=True)
class BookModelInventory:
    databases: tuple[DatabaseInventory, ...]
    csv_exports: tuple[CsvInventory, ...]
    exportable_slot_columns: tuple[str, ...]
    exportable_source_columns: tuple[str, ...]
    missing_model_artifacts: tuple[str, ...]


def _connect_readonly(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> tuple[str, ...]:
    if not _table_exists(conn, table):
        return ()
    return tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))


def _count_rows(conn: sqlite3.Connection, table: str) -> int | None:
    if not _table_exists(conn, table):
        return None
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def _group_counts(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    *,
    where: str | None = None,
) -> tuple[tuple[str, int], ...]:
    columns = set(_columns(conn, table))
    if column not in columns:
        return ()
    where_clause = f"WHERE {where}" if where else ""
    rows = conn.execute(
        f"""
        SELECT COALESCE(CAST({column} AS TEXT), 'NULL') AS value, COUNT(*)
        FROM {table}
        {where_clause}
        GROUP BY value
        ORDER BY COUNT(*) DESC, value
        """
    ).fetchall()
    return tuple((str(row[0]), int(row[1])) for row in rows)


def inspect_database(
    path: Path,
    *,
    label: str,
    expected_tables: tuple[str, ...],
) -> DatabaseInventory:
    """Return read-only inventory for a SQLite database."""

    if not path.exists():
        return DatabaseInventory(
            label=label,
            path=path,
            exists=False,
            tables=tuple(
                TableInventory(table, False, None, ())
                for table in expected_tables
            ),
            pattern_label_counts=(),
            candidate_source_counts=(),
            slot_type_counts=(),
            source_tag_counts=(),
        )

    with _connect_readonly(path) as conn:
        tables = tuple(
            TableInventory(
                name=table,
                exists=_table_exists(conn, table),
                row_count=_count_rows(conn, table),
                columns=_columns(conn, table),
            )
            for table in expected_tables
        )
        pattern_label_counts = _group_counts(
            conn,
            "pattern_matches",
            "llm_market_tier",
        )
        candidate_source_counts = _group_counts(
            conn,
            "pattern_matches",
            "candidate_source",
        )
        slot_type_counts = _group_counts(
            conn,
            "slot_fillers",
            "slot_type",
            where="mode = 'strict'" if "mode" in _columns(conn, "slot_fillers") else None,
        )
        source_tag_counts = _group_counts(conn, "sources", "source_tag")
    return DatabaseInventory(
        label=label,
        path=path,
        exists=True,
        tables=tables,
        pattern_label_counts=pattern_label_counts,
        candidate_source_counts=candidate_source_counts,
        slot_type_counts=slot_type_counts,
        source_tag_counts=source_tag_counts,
    )


def inspect_csv(path: Path) -> CsvInventory:
    """Return header and row-count inventory for an exported CSV file."""

    if not path.exists():
        return CsvInventory(path=path, exists=False, row_count=None, columns=())
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            columns = tuple(next(reader))
        except StopIteration:
            return CsvInventory(path=path, exists=True, row_count=0, columns=())
        row_count = sum(1 for _ in reader)
    return CsvInventory(
        path=path,
        exists=True,
        row_count=row_count,
        columns=columns,
    )


def build_inventory(
    *,
    full_db: Path,
    mini_db: Path,
    api_db: Path,
    export_dir: Path,
) -> BookModelInventory:
    """Build the book-model inventory across local DB and export surfaces."""

    databases = (
        inspect_database(full_db, label="full local DB", expected_tables=FULL_DB_TABLES),
        inspect_database(mini_db, label="mini DB", expected_tables=MINI_DB_TABLES),
        inspect_database(api_db, label="API DB", expected_tables=MINI_DB_TABLES),
    )
    csv_exports = tuple(
        inspect_csv(export_dir / name)
        for name in (
            "slot_fillers.csv",
            "sources.csv",
            "config.csv",
            "source_tier_labels.csv",
        )
    )
    available_tables = {
        table.name
        for db in databases
        for table in db.tables
        if table.exists
    }
    missing_model_artifacts = tuple(
        artifact for artifact in DESIRED_MODEL_ARTIFACTS
        if artifact not in available_tables
    )
    return BookModelInventory(
        databases=databases,
        csv_exports=csv_exports,
        exportable_slot_columns=EXPORTABLE_SLOT_FILLER_COLUMNS,
        exportable_source_columns=EXPORTABLE_SOURCE_COLUMNS,
        missing_model_artifacts=missing_model_artifacts,
    )


def format_inventory_markdown(inventory: BookModelInventory) -> str:
    """Format the inventory as a markdown report."""

    lines: list[str] = [
        "# Book-model inventory",
        "",
        "This read-only report answers the first export-field inventory gate for "
        "offline book categorization. It distinguishes full/offline-only inputs "
        "from fields already present in export or serving artifacts.",
        "",
        "## Database surfaces",
        "",
        "| Surface | Path | Table | Rows | Columns |",
        "|---|---|---|---:|---|",
    ]
    for db in inventory.databases:
        if not db.exists:
            lines.append(f"| {db.label} | `{db.path}` | missing database |  |  |")
            continue
        for table in db.tables:
            row_count = "" if table.row_count is None else f"{table.row_count:,}"
            columns = ", ".join(table.columns) if table.columns else ""
            table_name = table.name if table.exists else f"{table.name} (missing)"
            lines.append(
                f"| {db.label} | `{db.path}` | `{table_name}` | "
                f"{row_count} | {columns} |"
            )

    lines.extend([
        "",
        "## Label and source coverage",
        "",
    ])
    for db in inventory.databases:
        if not db.exists:
            continue
        lines.append(f"### {db.label}")
        _append_counts(lines, "LLM market tiers", db.pattern_label_counts)
        _append_counts(lines, "Pattern candidate source", db.candidate_source_counts)
        _append_counts(lines, "Strict slot types", db.slot_type_counts)
        _append_counts(lines, "Export source tags", db.source_tag_counts)
        lines.append("")

    lines.extend([
        "## CSV export surfaces",
        "",
        "| CSV | Rows | Columns |",
        "|---|---:|---|",
    ])
    for csv_inventory in inventory.csv_exports:
        rows = (
            "missing"
            if not csv_inventory.exists
            else f"{csv_inventory.row_count:,}"
        )
        lines.append(
            f"| `{csv_inventory.path}` | {rows} | "
            f"{', '.join(csv_inventory.columns)} |"
        )

    lines.extend([
        "",
        "## Exportable feature spec, initial",
        "",
        "The current exportable slot surface is limited to:",
        "",
        ", ".join(f"`{column}`" for column in inventory.exportable_slot_columns),
        "",
        "The current exportable source surface is limited to:",
        "",
        ", ".join(f"`{column}`" for column in inventory.exportable_source_columns),
        "",
        "Offline-only source/modeling surfaces include:",
        "",
        ", ".join(f"`{table}`" for table in OFFLINE_ONLY_TABLES),
        "",
        "Missing model artifacts:",
        "",
        ", ".join(f"`{artifact}`" for artifact in inventory.missing_model_artifacts),
        "",
        "## OQ1 preliminary answer",
        "",
        "- Exportable fields currently cover strict filler identity, slot type, "
        "frequency, source link, POS/preposition/remix metadata, scalar remix "
        "features, popularity score/level/confidence, source title/subtitle text, "
        "and source tag.",
        "- Full source-row fields, source-tier labels, ISBN/work-key linkage, "
        "popularity component data, and candidate-source role are offline-only "
        "unless an export change is explicitly made.",
        "- No `book_features`, `book_labels`, `book_predictions`, "
        "`slot_export_model`, or `filler_book_rollups` artifact exists yet, so "
        "D8 should stay unresolved until those artifacts and the accuracy-gap "
        "report exist.",
        "- Any runtime-safe compact model must be reproducible from the exportable "
        "slot/source columns above plus explicitly exported derived scalar "
        "features; raw embeddings and full source-row joins are not currently "
        "runtime-safe.",
        "",
    ])
    return "\n".join(lines)


def _append_counts(
    lines: list[str],
    title: str,
    counts: tuple[tuple[str, int], ...],
) -> None:
    if not counts:
        lines.append(f"- {title}: not available")
        return
    rendered = ", ".join(f"{value}={count:,}" for value, count in counts)
    lines.append(f"- {title}: {rendered}")
