"""SQLite schema contracts for pipeline stage readiness checks."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class TableContract:
    """Required columns for a table owned or consumed by a pipeline stage."""

    stage: str
    table: str
    columns: frozenset[str]


@dataclass(frozen=True)
class SchemaIssue:
    """A missing table or column reported with stage context."""

    stage: str
    table: str
    column: str | None
    message: str


SCHEMA_CONTRACTS: tuple[TableContract, ...] = (
    TableContract(
        stage="source_ingestion",
        table="subtitles",
        columns=frozenset({
            "id", "title", "subtitle", "lang", "lccn", "source_file", "isbn",
            "candidate_text", "candidate_source",
        }),
    ),
    TableContract(
        stage="internal_slots",
        table="pattern_matches",
        columns=frozenset({
            "id", "subtitle_id", "title", "subtitle", "list_items_json",
            "action_noun", "of_object", "of_article", "action_article",
            "candidate_source", "llm_market_tier", "llm_market_tier_confidence",
            "llm_market_tier_rationale",
        }),
    ),
    TableContract(
        stage="internal_slots",
        table="slot_fillers",
        columns=frozenset({
            "id", "slot_type", "filler", "mode", "source_subtitle_id", "freq",
            "pos_tag", "prep", "remix_type", "remix_prep", "remix_word_count",
            "vector_sum", "token_count", "centroid_dot", "norm_sq",
            "popularity_score", "popularity_level", "popularity_confidence",
        }),
    ),
    TableContract(
        stage="popularity_scoring",
        table="popularity_data",
        columns=frozenset({
            "work_key", "spl_checkouts", "spl_years", "spl_earliest_pub_year",
            "ol_edition_count", "checkouts_per_year", "editions_per_decade",
            "gr_ratings_count", "gr_average_rating", "nyt_weeks_on_list",
            "nyt_peak_rank", "library_appearances", "trove_library_count",
            "trove_holding_count", "trove_copy_count",
            "trove_copy_count_is_exact", "composite_score",
        }),
    ),
    TableContract(
        stage="model_weight_state",
        table="config",
        columns=frozenset({"key", "value"}),
    ),
    TableContract(
        stage="tuning_feedback",
        table="human_ratings",
        columns=frozenset({
            "id", "subtitle", "system_tone", "thumbs", "tone_override",
            "free_text", "interpreted", "config_snapshot", "created_at",
            "tags", "source", "prompt_generated",
        }),
    ),
)

MINI_DB_SCHEMA_CONTRACTS: tuple[TableContract, ...] = (
    # Serving used to rely on the generated mini DB shape implicitly. Tiering now
    # consumes popularity level/confidence at runtime, so the mini DB needs its own
    # explicit slot_filler contract instead of inheriting only the full-DB checks.
    TableContract(
        stage="serving",
        table="slot_fillers",
        columns=frozenset({
            "id", "slot_type", "filler", "mode", "source_subtitle_id", "freq",
            "pos_tag", "prep", "remix_type", "remix_prep", "remix_word_count",
            "vector_sum", "token_count", "centroid_dot", "norm_sq",
            "popularity_score", "popularity_level", "popularity_confidence",
        }),
    ),
    TableContract(
        stage="model_weight_state",
        table="config",
        columns=frozenset({"key", "value"}),
    ),
    TableContract(
        stage="serving_sources",
        table="sources",
        columns=frozenset({
            "slot_filler_id", "title", "subtitle_text", "source_tag",
        }),
    ),
    TableContract(
        stage="serving_model_scores",
        table="slot_filler_model_scores",
        columns=frozenset({
            "slot_filler_id", "score_pop", "score_mainstream", "score_niche",
            "model_tier", "source_prediction_count",
        }),
    ),
)

REQUIRED_TABLES_BY_STAGE: dict[str, tuple[str, ...]] = {
    stage: tuple(contract.table for contract in SCHEMA_CONTRACTS if contract.stage == stage)
    for stage in sorted({contract.stage for contract in SCHEMA_CONTRACTS})
}


def get_contract(table: str) -> TableContract:
    """Return the schema contract for ``table``."""

    for contract in SCHEMA_CONTRACTS:
        if contract.table == table:
            return contract
    raise KeyError(f"No schema contract for table: {table}")


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    """Return column names for ``table``, or an empty set if it is missing."""

    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def validate_schema(
    conn: sqlite3.Connection,
    contracts: tuple[TableContract, ...] = SCHEMA_CONTRACTS,
) -> list[SchemaIssue]:
    """Validate required tables/columns and return stage-aware issues."""

    issues: list[SchemaIssue] = []
    existing_tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    for contract in contracts:
        if contract.table not in existing_tables:
            issues.append(SchemaIssue(
                stage=contract.stage,
                table=contract.table,
                column=None,
                message=(
                    f"{contract.stage}: missing required table "
                    f"{contract.table!r}"
                ),
            ))
            continue

        actual_columns = table_columns(conn, contract.table)
        for column in sorted(contract.columns - actual_columns):
            issues.append(SchemaIssue(
                stage=contract.stage,
                table=contract.table,
                column=column,
                message=(
                    f"{contract.stage}: table {contract.table!r} is missing "
                    f"required column {column!r}"
                ),
            ))
    return issues


def assert_schema_valid(conn: sqlite3.Connection) -> None:
    """Raise ``RuntimeError`` with stage-aware messages when schema is invalid."""

    issues = validate_schema(conn)
    if issues:
        raise RuntimeError("\n".join(issue.message for issue in issues))
