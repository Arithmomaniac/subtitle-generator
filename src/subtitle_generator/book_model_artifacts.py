"""Offline book-model feature and label artifact builder."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

from subtitle_generator.book_metadata_enrichment import load_book_metadata_rows


FEATURE_COLUMNS = (
    "pattern_match_id",
    "subtitle_id",
    "title",
    "subtitle_text",
    "candidate_source",
    "source_file",
    "source_group",
    "lang",
    "candidate_source_is_title",
    "source_group_is_ol",
    "lang_is_eng",
    "has_isbn",
    "has_lccn",
    "has_work_key",
    "has_popularity_data",
    "work_key",
    "work_popularity_score",
    "spl_checkouts",
    "spl_years",
    "ol_edition_count",
    "checkouts_per_year",
    "editions_per_decade",
    "gr_ratings_count",
    "gr_average_rating",
    "nyt_weeks_on_list",
    "library_appearances",
    "trove_library_count",
    "trove_holding_count",
    "trove_copy_count",
    "slot_source_link_count",
    "distinct_strict_filler_count",
    "max_filler_popularity_score",
    "avg_filler_popularity_score",
    "title_length_chars",
    "subtitle_length_chars",
    "list_item_count",
    "list_items_text",
    "list_item_pair_text",
    "action_noun",
    "of_object",
    "action_object_pair_text",
    "slot_frame_text",
    "metadata_source",
    "metadata_work_key",
    "metadata_edition_key",
    "metadata_publisher_text",
    "metadata_publisher_count",
    "metadata_publish_year",
    "metadata_edition_name",
    "metadata_physical_format",
    "metadata_physical_dimensions",
    "metadata_physical_description",
    "metadata_number_of_pages",
    "metadata_is_hardcover",
    "metadata_is_paperback",
    "metadata_is_ebook",
    "metadata_is_large_print",
    "metadata_subject_text",
    "metadata_subject_count",
    "metadata_author_count",
    "metadata_loc_call_number",
    "metadata_dewey_decimal",
    "has_physical_format_metadata",
    "has_publisher_metadata",
    "has_subject_metadata",
)
LABEL_COLUMNS = (
    "pattern_match_id",
    "subtitle_id",
    "llm_market_tier",
    "llm_market_tier_confidence",
    "llm_market_tier_rationale",
    "has_llm_market_tier",
    "popularity_comparison_score",
    "label_target",
    "label_confidence",
    "label_source",
)


@dataclass(frozen=True)
class BookFeatureRow:
    pattern_match_id: int
    subtitle_id: int
    title: str
    subtitle_text: str
    candidate_source: str
    source_file: str
    source_group: str
    lang: str
    candidate_source_is_title: int
    source_group_is_ol: int
    lang_is_eng: int
    has_isbn: int
    has_lccn: int
    has_work_key: int
    has_popularity_data: int
    work_key: str
    work_popularity_score: float | None
    spl_checkouts: int
    spl_years: int
    ol_edition_count: int
    checkouts_per_year: float | None
    editions_per_decade: float | None
    gr_ratings_count: int
    gr_average_rating: float | None
    nyt_weeks_on_list: int
    library_appearances: int
    trove_library_count: int
    trove_holding_count: int
    trove_copy_count: int
    slot_source_link_count: int
    distinct_strict_filler_count: int
    max_filler_popularity_score: float | None
    avg_filler_popularity_score: float | None
    title_length_chars: int
    subtitle_length_chars: int
    list_item_count: int
    list_items_text: str
    list_item_pair_text: str
    action_noun: str
    of_object: str
    action_object_pair_text: str
    slot_frame_text: str
    metadata_source: str
    metadata_work_key: str
    metadata_edition_key: str
    metadata_publisher_text: str
    metadata_publisher_count: int
    metadata_publish_year: int | None
    metadata_edition_name: str
    metadata_physical_format: str
    metadata_physical_dimensions: str
    metadata_physical_description: str
    metadata_number_of_pages: int | None
    metadata_is_hardcover: int
    metadata_is_paperback: int
    metadata_is_ebook: int
    metadata_is_large_print: int
    metadata_subject_text: str
    metadata_subject_count: int
    metadata_author_count: int
    metadata_loc_call_number: str
    metadata_dewey_decimal: str
    has_physical_format_metadata: int
    has_publisher_metadata: int
    has_subject_metadata: int


@dataclass(frozen=True)
class BookLabelRow:
    pattern_match_id: int
    subtitle_id: int
    llm_market_tier: str
    llm_market_tier_confidence: float | None
    llm_market_tier_rationale: str
    has_llm_market_tier: int
    popularity_comparison_score: float | None
    label_target: str
    label_confidence: float | None
    label_source: str


@dataclass(frozen=True)
class ArtifactBuildResult:
    features_path: Path
    labels_path: Path
    report_path: Path
    feature_count: int
    label_count: int


def build_book_model_artifacts(
    conn: sqlite3.Connection,
    output_dir: Path,
    metadata_path: Path | None = None,
) -> ArtifactBuildResult:
    """Build local book feature/label CSVs and a coverage report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    features = tuple(load_book_feature_rows(conn, metadata_path=metadata_path))
    labels = tuple(build_label_rows(features, conn))

    features_path = output_dir / "book_features.csv"
    labels_path = output_dir / "book_labels.csv"
    report_path = output_dir / "book_feature_label_report.md"

    _write_dataclass_csv(features_path, FEATURE_COLUMNS, features)
    _write_dataclass_csv(labels_path, LABEL_COLUMNS, labels)
    report_path.write_text(
        format_feature_label_report(features, labels, features_path, labels_path),
        encoding="utf-8",
    )
    return ArtifactBuildResult(
        features_path=features_path,
        labels_path=labels_path,
        report_path=report_path,
        feature_count=len(features),
        label_count=len(labels),
    )

def load_book_feature_rows(
    conn: sqlite3.Connection,
    *,
    metadata_path: Path | None = None,
) -> list[BookFeatureRow]:
    """Load one feature row per generator-relevant pattern match."""

    _require_tables(conn, ("pattern_matches", "subtitles"))
    has_slot_sources = _table_exists(conn, "slot_filler_sources")
    has_isbn_aliases = _table_exists(conn, "isbn_aliases")
    has_popularity = has_isbn_aliases and _table_exists(conn, "popularity_data")
    has_slot_fillers = _table_exists(conn, "slot_fillers")
    popularity_columns = _columns(conn, "popularity_data")
    metadata_rows = load_book_metadata_rows(metadata_path)

    subtitle_expr = _source_subtitle_expr(_columns(conn, "pattern_matches"))
    slot_metrics_cte = _slot_metrics_cte(has_slot_sources, has_slot_fillers)
    isbn_join = "LEFT JOIN isbn_aliases ia ON ia.isbn = s.isbn" if has_isbn_aliases else ""
    popularity_join = (
        "LEFT JOIN popularity_data pd ON pd.work_key = ia.work_key"
        if has_popularity
        else ""
    )
    has_work_key_expr = (
        "CASE WHEN COALESCE(ia.work_key, '') <> '' THEN 1 ELSE 0 END"
        if has_isbn_aliases
        else "0"
    )
    has_popularity_expr = (
        "CASE WHEN pd.work_key IS NOT NULL THEN 1 ELSE 0 END"
        if has_popularity
        else "0"
    )
    work_key_expr = "COALESCE(ia.work_key, '')" if has_isbn_aliases else "''"
    popularity_score_expr = (
        _column_expr(popularity_columns, "pd", "composite_score", "NULL")
        if has_popularity
        else "NULL"
    )
    popularity_exprs = {
        column: _column_expr(popularity_columns, "pd", column, default)
        if has_popularity
        else default
        for column, default in {
            "spl_checkouts": "0",
            "spl_years": "0",
            "ol_edition_count": "0",
            "checkouts_per_year": "NULL",
            "editions_per_decade": "NULL",
            "gr_ratings_count": "0",
            "gr_average_rating": "NULL",
            "nyt_weeks_on_list": "0",
            "library_appearances": "0",
            "trove_library_count": "0",
            "trove_holding_count": "0",
            "trove_copy_count": "0",
        }.items()
    }
    query = f"""
    {slot_metrics_cte}
    SELECT
        pm.id,
        pm.subtitle_id,
        COALESCE(pm.title, s.title, ''),
        {subtitle_expr},
        COALESCE(NULLIF(pm.candidate_source, ''), s.candidate_source, 'subtitle'),
        COALESCE(s.source_file, ''),
        CASE WHEN s.source_file = 'openlibrary' THEN 'OL' ELSE 'LOC' END,
        COALESCE(s.lang, ''),
        CASE WHEN COALESCE(s.isbn, '') <> '' THEN 1 ELSE 0 END,
        CASE WHEN COALESCE(s.lccn, '') <> '' THEN 1 ELSE 0 END,
        {has_work_key_expr},
        {has_popularity_expr},
        {work_key_expr},
        {popularity_score_expr},
        {popularity_exprs["spl_checkouts"]},
        {popularity_exprs["spl_years"]},
        {popularity_exprs["ol_edition_count"]},
        {popularity_exprs["checkouts_per_year"]},
        {popularity_exprs["editions_per_decade"]},
        {popularity_exprs["gr_ratings_count"]},
        {popularity_exprs["gr_average_rating"]},
        {popularity_exprs["nyt_weeks_on_list"]},
        {popularity_exprs["library_appearances"]},
        {popularity_exprs["trove_library_count"]},
        {popularity_exprs["trove_holding_count"]},
        {popularity_exprs["trove_copy_count"]},
        COALESCE(sm.slot_source_link_count, 0),
        COALESCE(sm.distinct_strict_filler_count, 0),
        sm.max_filler_popularity_score,
        sm.avg_filler_popularity_score,
        COALESCE(pm.list_items_json, ''),
        COALESCE(pm.action_noun, ''),
        COALESCE(pm.of_object, '')
    FROM pattern_matches pm
    JOIN subtitles s ON s.id = pm.subtitle_id
    {isbn_join}
    {popularity_join}
    LEFT JOIN slot_metrics sm ON sm.subtitle_id = pm.subtitle_id
    ORDER BY pm.id
    """
    rows = conn.execute(query).fetchall()
    features: list[BookFeatureRow] = []
    for row in rows:
        title = row[2] or ""
        subtitle = row[3] or ""
        list_items = _list_items(row[30])
        list_items_text = " | ".join(list_items)
        list_item_pair_text = " || ".join(list_items[:2])
        action_noun = row[31] or ""
        of_object = row[32] or ""
        action_object_pair_text = _pair_text(action_noun, of_object)
        slot_frame_text = _slot_frame_text(list_items, action_noun, of_object)
        metadata = metadata_rows.get(row[1], {})
        features.append(BookFeatureRow(
            pattern_match_id=row[0],
            subtitle_id=row[1],
            title=title,
            subtitle_text=subtitle,
            candidate_source=row[4] or "subtitle",
            source_file=row[5] or "",
            source_group=row[6] or "",
            lang=row[7] or "",
            candidate_source_is_title=int((row[4] or "").lower() == "title"),
            source_group_is_ol=int((row[6] or "").upper() == "OL"),
            lang_is_eng=int((row[7] or "").lower() == "eng"),
            has_isbn=row[8],
            has_lccn=row[9],
            has_work_key=row[10] if has_isbn_aliases else 0,
            has_popularity_data=row[11] if has_popularity else 0,
            work_key=row[12] if has_isbn_aliases else "",
            work_popularity_score=row[13] if has_popularity else None,
            spl_checkouts=row[14] if has_popularity else 0,
            spl_years=row[15] if has_popularity else 0,
            ol_edition_count=row[16] if has_popularity else 0,
            checkouts_per_year=row[17] if has_popularity else None,
            editions_per_decade=row[18] if has_popularity else None,
            gr_ratings_count=row[19] if has_popularity else 0,
            gr_average_rating=row[20] if has_popularity else None,
            nyt_weeks_on_list=row[21] if has_popularity else 0,
            library_appearances=row[22] if has_popularity else 0,
            trove_library_count=row[23] if has_popularity else 0,
            trove_holding_count=row[24] if has_popularity else 0,
            trove_copy_count=row[25] if has_popularity else 0,
            slot_source_link_count=row[26] if has_slot_sources else 0,
            distinct_strict_filler_count=row[27] if has_slot_fillers else 0,
            max_filler_popularity_score=row[28] if has_slot_fillers else None,
            avg_filler_popularity_score=row[29] if has_slot_fillers else None,
            title_length_chars=len(title),
            subtitle_length_chars=len(subtitle),
            list_item_count=len(list_items),
            list_items_text=list_items_text,
            list_item_pair_text=list_item_pair_text,
            action_noun=action_noun,
            of_object=of_object,
            action_object_pair_text=action_object_pair_text,
            slot_frame_text=slot_frame_text,
            metadata_source=metadata.get("metadata_source", ""),
            metadata_work_key=metadata.get("work_key", ""),
            metadata_edition_key=metadata.get("edition_key", ""),
            metadata_publisher_text=metadata.get("publisher_text", ""),
            metadata_publisher_count=_int(metadata.get("publisher_count")),
            metadata_publish_year=_optional_int(metadata.get("publish_year")),
            metadata_edition_name=metadata.get("edition_name", ""),
            metadata_physical_format=metadata.get("physical_format", ""),
            metadata_physical_dimensions=metadata.get("physical_dimensions", ""),
            metadata_physical_description=metadata.get("physical_description", ""),
            metadata_number_of_pages=_optional_int(metadata.get("number_of_pages")),
            metadata_is_hardcover=_int(metadata.get("is_hardcover")),
            metadata_is_paperback=_int(metadata.get("is_paperback")),
            metadata_is_ebook=_int(metadata.get("is_ebook")),
            metadata_is_large_print=_int(metadata.get("is_large_print")),
            metadata_subject_text=metadata.get("subject_text", ""),
            metadata_subject_count=_int(metadata.get("subject_count")),
            metadata_author_count=_int(metadata.get("author_count")),
            metadata_loc_call_number=metadata.get("loc_call_number", ""),
            metadata_dewey_decimal=metadata.get("dewey_decimal", ""),
            has_physical_format_metadata=_int(
                metadata.get("has_physical_format_metadata")
            ),
            has_publisher_metadata=_int(metadata.get("has_publisher_metadata")),
            has_subject_metadata=_int(metadata.get("has_subject_metadata")),
        ))
    return features


def build_label_rows(
    features: tuple[BookFeatureRow, ...],
    conn: sqlite3.Connection,
) -> list[BookLabelRow]:
    """Build separated target/comparison label rows for feature rows."""

    pm_columns = _columns(conn, "pattern_matches")
    if not {
        "llm_market_tier",
        "llm_market_tier_confidence",
        "llm_market_tier_rationale",
    } <= pm_columns:
        labels_by_pm: dict[int, tuple[str, float | None, str]] = {}
    else:
        rows = conn.execute(
            """
            SELECT id, COALESCE(llm_market_tier, ''),
                   llm_market_tier_confidence,
                   COALESCE(llm_market_tier_rationale, '')
            FROM pattern_matches
            """
        ).fetchall()
        labels_by_pm = {row[0]: (row[1], row[2], row[3]) for row in rows}

    label_rows: list[BookLabelRow] = []
    for feature in features:
        tier, confidence, rationale = labels_by_pm.get(
            feature.pattern_match_id,
            ("", None, ""),
        )
        has_llm_label = int(tier in {"pop", "mainstream", "niche"})
        label_rows.append(BookLabelRow(
            pattern_match_id=feature.pattern_match_id,
            subtitle_id=feature.subtitle_id,
            llm_market_tier=tier,
            llm_market_tier_confidence=confidence,
            llm_market_tier_rationale=rationale,
            has_llm_market_tier=has_llm_label,
            popularity_comparison_score=feature.work_popularity_score,
            label_target=tier if has_llm_label else "",
            label_confidence=confidence if has_llm_label else None,
            label_source="llm_market_tier" if has_llm_label else "",
        ))
    return label_rows


def format_feature_label_report(
    features: tuple[BookFeatureRow, ...],
    labels: tuple[BookLabelRow, ...],
    features_path: Path,
    labels_path: Path,
) -> str:
    """Format coverage and label semantics for the generated artifacts."""

    label_counts = _count_values(
        label.llm_market_tier if label.has_llm_market_tier else "unlabeled"
        for label in labels
    )
    source_counts = _count_values(feature.candidate_source for feature in features)
    source_group_counts = _count_values(feature.source_group for feature in features)
    missingness = {
        "has_isbn": sum(feature.has_isbn for feature in features),
        "has_lccn": sum(feature.has_lccn for feature in features),
        "has_work_key": sum(feature.has_work_key for feature in features),
        "has_popularity_data": sum(feature.has_popularity_data for feature in features),
        "has_physical_format_metadata": sum(
            feature.has_physical_format_metadata for feature in features
        ),
        "has_publisher_metadata": sum(feature.has_publisher_metadata for feature in features),
        "has_subject_metadata": sum(feature.has_subject_metadata for feature in features),
    }
    total = len(features)
    lines = [
        "# Book feature and label artifacts",
        "",
        "These artifacts are local offline modeling inputs. They do not change "
        "runtime generation, export data, or browser/API behavior.",
        "",
        "## Outputs",
        "",
        f"- Features: `{features_path}` ({total:,} rows)",
        f"- Labels: `{labels_path}` ({len(labels):,} rows)",
        "",
        "## Coverage",
        "",
        f"- Feature rows: {total:,}",
        f"- Candidate source: {_format_counts(source_counts)}",
        f"- Source group: {_format_counts(source_group_counts)}",
        f"- Identifier/popularity coverage: {_format_coverage(missingness, total)}",
        f"- LLM market-tier labels: {_format_counts(label_counts)}",
        "",
        "## Label semantics",
        "",
        "- `label_target` is populated only from `pattern_matches.llm_market_tier`.",
        "- `label_confidence` is the associated LLM confidence value.",
        "- `popularity_comparison_score` is a comparison signal, not ground truth.",
        "- `human_ratings` are deliberately excluded because they describe "
        "generated subtitle outputs, not source-book market tier.",
        "- Raw LOC/Open Library metadata is offline-only enrichment; it is useful "
        "for rich training but cannot be assumed available after CSV export.",
        "",
        "## Gate notes",
        "",
    ]
    pop_count = label_counts.get("pop", 0)
    if pop_count < 100:
        lines.append(
            f"- Pop remains underrepresented ({pop_count:,} labels); downstream "
            "models should use class weighting, report macro metrics, or continue "
            "targeted labeling before treating pop performance as stable."
        )
    else:
        lines.append("- Label coverage meets the initial per-tier threshold.")
    return "\n".join(lines)


def _slot_metrics_cte(
    has_slot_sources: bool,
    has_slot_fillers: bool,
) -> str:
    if not has_slot_sources or not has_slot_fillers:
        return """
        WITH slot_metrics AS (
            SELECT
                NULL AS subtitle_id,
                0 AS slot_source_link_count,
                0 AS distinct_strict_filler_count,
                NULL AS max_filler_popularity_score,
                NULL AS avg_filler_popularity_score
            WHERE 0
        )
        """
    return """
    WITH slot_metrics AS (
        SELECT
            sfs.subtitle_id,
            COUNT(*) AS slot_source_link_count,
            COUNT(DISTINCT sf.id) AS distinct_strict_filler_count,
            MAX(sf.popularity_score) AS max_filler_popularity_score,
            AVG(sf.popularity_score) AS avg_filler_popularity_score
        FROM slot_filler_sources sfs
        JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
        WHERE COALESCE(sf.mode, 'strict') = 'strict'
        GROUP BY sfs.subtitle_id
    )
    """


def _source_subtitle_expr(columns: set[str]) -> str:
    if "candidate_source" in columns:
        return (
            "CASE WHEN pm.candidate_source = 'title' "
            "THEN '' ELSE COALESCE(pm.subtitle, s.subtitle, '') END"
        )
    return "COALESCE(pm.subtitle, s.subtitle, '')"


def _list_item_count(list_items_json: str) -> int:
    return len(_list_items(list_items_json))


def _list_items(list_items_json: str) -> list[str]:
    if not list_items_json:
        return []
    try:
        parsed = json.loads(list_items_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _pair_text(left: str, right: str) -> str:
    if not left or not right:
        return ""
    return f"{left} || {right}"


def _slot_frame_text(
    list_items: list[str],
    action_noun: str,
    of_object: str,
) -> str:
    parts = [*list_items[:2], action_noun, of_object]
    return " || ".join(part for part in parts if part)


def _column_expr(columns: set[str], alias: str, column: str, default: str) -> str:
    if column in columns:
        return f"{alias}.{column}"
    return default


def _int(value: str | int | None) -> int:
    if value in (None, ""):
        return 0
    return int(value)


def _optional_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def _write_dataclass_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: tuple[object, ...],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _require_tables(conn: sqlite3.Connection, tables: tuple[str, ...]) -> None:
    missing = [table for table in tables if not _table_exists(conn, table)]
    if missing:
        raise RuntimeError(
            "Missing required table(s): " + ", ".join(sorted(missing))
        )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value:,}" for key, value in counts.items())


def _format_coverage(counts: dict[str, int], total: int) -> str:
    if total == 0:
        return "none"
    return ", ".join(
        f"{key}={value:,}/{total:,} ({value / total:.1%})"
        for key, value in counts.items()
    )
