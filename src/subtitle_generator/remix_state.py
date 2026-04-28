"""Remix precompute contracts and runtime context."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any


REQUIRED_PRECOMPUTE_CONFIG_KEYS = frozenset({
    "embedding_version",
    "centroid_norm",
    "avg_cross_sim_t1",
    "avg_cross_sim_t2",
})

REQUIRED_REMIX_COLUMNS = frozenset({
    "remix_type",
    "remix_prep",
    "remix_word_count",
    "vector_sum",
    "token_count",
    "centroid_dot",
    "norm_sq",
})


@dataclass(frozen=True)
class RemixPrecomputeIssue:
    stage: str
    field: str
    message: str


@dataclass
class RemixRuntimeContext:
    precomputed: bool
    config: dict[str, Any] = field(default_factory=dict)
    article_stats_of: dict[str, dict[str, int]] = field(default_factory=dict)
    article_stats_action: dict[str, dict[str, int]] = field(default_factory=dict)
    centroid_norm: float | None = None
    avg_cross_sim_t1: float | None = None
    avg_cross_sim_t2: float | None = None
    filler_scalars: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    nlp: Any = None
    centroid: Any = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def validate_remix_precompute_state(
    conn: sqlite3.Connection,
    expected_embedding_version: str,
) -> list[RemixPrecomputeIssue]:
    """Validate the DB state required for precomputed runtime remixing."""

    issues: list[RemixPrecomputeIssue] = []

    tables = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if "config" not in tables:
        return [RemixPrecomputeIssue(
            stage="remix_precompute",
            field="config",
            message="remix_precompute: missing required table 'config'",
        )]
    if "slot_fillers" not in tables:
        issues.append(RemixPrecomputeIssue(
            stage="remix_precompute",
            field="slot_fillers",
            message="remix_precompute: missing required table 'slot_fillers'",
        ))
        return issues

    config_values = dict(conn.execute("SELECT key, value FROM config").fetchall())
    for key in sorted(REQUIRED_PRECOMPUTE_CONFIG_KEYS - config_values.keys()):
        issues.append(RemixPrecomputeIssue(
            stage="remix_precompute",
            field=key,
            message=f"remix_precompute: missing required config key {key!r}",
        ))

    actual_version = config_values.get("embedding_version")
    if actual_version is not None and str(actual_version) != expected_embedding_version:
        issues.append(RemixPrecomputeIssue(
            stage="remix_precompute",
            field="embedding_version",
            message=(
                "remix_precompute: embedding_version "
                f"{actual_version!r} does not match expected {expected_embedding_version!r}"
            ),
        ))

    columns = _table_columns(conn, "slot_fillers")
    for column in sorted(REQUIRED_REMIX_COLUMNS - columns):
        issues.append(RemixPrecomputeIssue(
            stage="remix_precompute",
            field=column,
            message=f"remix_precompute: slot_fillers is missing column {column!r}",
        ))

    return issues


def assert_remix_precompute_state(
    conn: sqlite3.Connection,
    expected_embedding_version: str,
) -> None:
    issues = validate_remix_precompute_state(conn, expected_embedding_version)
    if issues:
        raise RuntimeError("\n".join(issue.message for issue in issues))
