"""Read-only readiness checks for the subtitle generation pipeline."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from subtitle_generator.config import ALL_TUNABLE_PARAMS
from subtitle_generator.parameter_state import get_model_registry
from subtitle_generator.remix_state import validate_remix_precompute_state
from subtitle_generator.schema_contracts import validate_schema


@dataclass(frozen=True)
class ValidationIssue:
    stage: str
    check: str
    message: str


@dataclass(frozen=True)
class PipelineValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _validate_parameter_config(conn: sqlite3.Connection) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not _table_exists(conn, "config"):
        return [ValidationIssue(
            stage="model_weight_state",
            check="config",
            message="model_weight_state: missing required table 'config'",
        )]

    config_columns = _columns(conn, "config")
    if not {"key", "value"} <= config_columns:
        return issues

    for key, value in conn.execute("SELECT key, value FROM config").fetchall():
        if key in ALL_TUNABLE_PARAMS:
            try:
                float(value)
            except (TypeError, ValueError):
                issues.append(ValidationIssue(
                    stage="model_weight_state",
                    check="config_value",
                    message=f"model_weight_state: config key {key!r} is not numeric",
                ))
    return issues


def _validate_model_registry(conn: sqlite3.Connection) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    registry = get_model_registry()
    known_models = {registry.rater, registry.proposer, registry.jacket}
    known_models.update(registry.responses_only)
    if not registry.rater or not registry.proposer or not registry.jacket:
        issues.append(ValidationIssue(
            stage="model_weight_state",
            check="model_registry",
            message="model_weight_state: model registry contains an empty model id",
        ))

    if not _table_exists(conn, "config") or not {"key", "value"} <= _columns(conn, "config"):
        return issues

    for key, value in conn.execute("SELECT key, value FROM config").fetchall():
        if key.endswith("_model") and value not in known_models:
            issues.append(ValidationIssue(
                stage="model_weight_state",
                check="model_registry",
                message=f"model_weight_state: unknown configured model id {value!r}",
            ))
    return issues


def _validate_popularity_coverage(conn: sqlite3.Connection) -> list[ValidationIssue]:
    if not _table_exists(conn, "slot_fillers"):
        return [ValidationIssue(
            stage="popularity_scoring",
            check="coverage",
            message="popularity_scoring: missing required table 'slot_fillers'",
        )]

    required_columns = {"slot_type", "mode"}
    actual_columns = _columns(conn, "slot_fillers")
    if not required_columns <= actual_columns:
        missing = ", ".join(sorted(required_columns - actual_columns))
        return [ValidationIssue(
            stage="popularity_scoring",
            check="coverage",
            message=f"popularity_scoring: slot_fillers is missing columns: {missing}",
        )]

    issues: list[ValidationIssue] = []
    evidence_columns = {"popularity_score", "popularity_level", "popularity_confidence"}
    missing_evidence_columns = evidence_columns - actual_columns
    if missing_evidence_columns:
        missing = ", ".join(sorted(missing_evidence_columns))
        issues.append(ValidationIssue(
            stage="popularity_scoring",
            check="coverage",
            message=f"popularity_scoring: slot_fillers is missing columns: {missing}",
        ))

    strict_count = conn.execute(
        "SELECT COUNT(*) FROM slot_fillers WHERE mode = 'strict'"
    ).fetchone()[0]
    if strict_count == 0:
        issues.append(ValidationIssue(
            stage="popularity_scoring",
            check="coverage",
            message="popularity_scoring: no strict slot fillers are available",
        ))
        return issues

    if "popularity_score" in actual_columns:
        if "popularity_level" in actual_columns:
            missing_scores = conn.execute(
                """
                SELECT COUNT(*) FROM slot_fillers
                WHERE mode = 'strict'
                  AND popularity_score IS NULL
                  AND COALESCE(popularity_level, 1) <> 0
                """
            ).fetchone()[0]
        else:
            missing_scores = conn.execute(
                """
                SELECT COUNT(*) FROM slot_fillers
                WHERE mode = 'strict' AND popularity_score IS NULL
                """
            ).fetchone()[0]
        if missing_scores:
            issues.append(ValidationIssue(
                stage="popularity_scoring",
                check="coverage",
                message=(
                    "popularity_scoring: "
                    f"{missing_scores} strict non-Level-0 slot fillers lack popularity_score"
                ),
            ))

    if {"popularity_level", "popularity_confidence"} <= actual_columns:
        missing_evidence = conn.execute(
            """
            SELECT COUNT(*) FROM slot_fillers
            WHERE mode = 'strict'
              AND (popularity_level IS NULL OR popularity_confidence IS NULL)
            """
        ).fetchone()[0]
        if missing_evidence:
            issues.append(ValidationIssue(
                stage="popularity_scoring",
                check="coverage",
                message=(
                    "popularity_scoring: "
                    f"{missing_evidence} strict slot fillers lack popularity level/confidence"
                ),
            ))

    slot_counts = dict(conn.execute(
        """
        SELECT slot_type, COUNT(*) FROM slot_fillers
        WHERE mode = 'strict'
        GROUP BY slot_type
        """
    ).fetchall())
    for slot_type in ("list_item", "action_noun", "of_object"):
        if slot_counts.get(slot_type, 0) == 0:
            issues.append(ValidationIssue(
                stage="serving",
                check="runtime_candidates",
                message=f"serving: no strict {slot_type!r} candidates are available",
            ))
    return issues


def _validate_serving_contracts() -> list[ValidationIssue]:
    from subtitle_generator import handlers

    issues: list[ValidationIssue] = []
    for name in ("subtitle_to_dict", "handle_generate", "handle_jacket", "handle_rate"):
        if not callable(getattr(handlers, name, None)):
            issues.append(ValidationIssue(
                stage="serving",
                check="handler_contract",
                message=f"serving: missing callable handlers.{name}",
            ))
    return issues


def validate_pipeline(
    conn: sqlite3.Connection,
    expected_embedding_version: str = "2",
) -> PipelineValidationReport:
    """Validate pipeline readiness without mutating DB state or generating subtitles."""

    issues: list[ValidationIssue] = []
    issues.extend(
        ValidationIssue(issue.stage, "schema", issue.message)
        for issue in validate_schema(conn)
    )
    issues.extend(_validate_parameter_config(conn))
    issues.extend(_validate_model_registry(conn))
    issues.extend(
        ValidationIssue(issue.stage, "remix_precompute", issue.message)
        for issue in validate_remix_precompute_state(conn, expected_embedding_version)
    )
    issues.extend(_validate_popularity_coverage(conn))
    issues.extend(_validate_serving_contracts())
    return PipelineValidationReport(issues=tuple(issues))


def format_validation_report(report: PipelineValidationReport) -> str:
    if report.ok:
        return "Pipeline validation passed."

    lines = ["Pipeline validation failed:"]
    for issue in report.issues:
        lines.append(f"- [{issue.stage}/{issue.check}] {issue.message}")
    return "\n".join(lines)
