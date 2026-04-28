"""State records and config mutation helpers for tuning."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from subtitle_generator.config import ALL_TUNABLE_PARAMS, invalidate_config_cache


@dataclass(frozen=True)
class ConfigChange:
    """A proposed config mutation with enough state to roll back."""

    param: str
    old_value: float
    new_value: float


@dataclass(frozen=True)
class ProposalDecision:
    """Evaluation outcome for a proposed tuning change."""

    change: ConfigChange
    reasoning: str
    status: str
    quality_before: float
    separation_before: float
    composite_before: float
    quality_after: float
    separation_after: float
    composite_after: float


def write_config_param(conn: sqlite3.Connection, param: str, value: float) -> None:
    """Persist a config value, deleting rows that match the default."""

    if value == ALL_TUNABLE_PARAMS[param]:
        conn.execute("DELETE FROM config WHERE key = ?", (param,))
    else:
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (param, str(value)),
        )
    conn.commit()
    invalidate_config_cache()


def apply_config_change(conn: sqlite3.Connection, change: ConfigChange) -> None:
    write_config_param(conn, change.param, change.new_value)


def revert_config_change(conn: sqlite3.Connection, change: ConfigChange) -> None:
    write_config_param(conn, change.param, change.old_value)


def record_decision(
    change: ConfigChange,
    reasoning: str,
    status: str,
    before: tuple[float, float, float],
    after: tuple[float, float, float],
) -> ProposalDecision:
    return ProposalDecision(
        change=change,
        reasoning=reasoning,
        status=status,
        quality_before=before[0],
        separation_before=before[1],
        composite_before=before[2],
        quality_after=after[0],
        separation_after=after[1],
        composite_after=after[2],
    )
