"""Human-feedback contract for Step 5 semantic-smoothing review (#38).

Judgment-heavy ML-rewrite steps must produce a *durable* ratings artifact, not
an ephemeral "a human looked at it" packet (see the epic's human-feedback
policy). Smoothing candidates are NOT regenerable judgments, so per-candidate
ratings live in the working DB (mirroring ``feedback.py``'s ``human_ratings``
pattern) and the overall accept/reject/iterate decision is written to a small
committed ``decision.json`` as the exit-gate evidence.

This module is analysis-only and never touches the served distribution.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Per-candidate review verdicts (kind-1 ratings that define the AutoResearcher
# objective for Step 5).
VALID_SMOOTHING_DECISIONS = frozenset({
    "plausible_repair",  # boosted filler feels better/more varied for the tier/slot
    "semantic_bleed",    # topic-close but wrong tier/style/register/humour
    "too_generic",       # boost moves toward bland academic/generic language
    "needs_context",     # hard to judge without source titles or samples
})

# Overall gate verdicts (kind-3 sign-off recorded in decision.json).
VALID_OVERALL_DECISIONS = frozenset({"accept", "reject", "iterate"})

DECISION_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# DB table: per-candidate smoothing ratings
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS smoothing_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    vector_source TEXT NOT NULL,
    slot_type TEXT NOT NULL,
    tier TEXT NOT NULL,
    filler TEXT NOT NULL,
    base_p REAL,
    smoothed_p REAL,
    delta REAL,
    evidence_soft REAL,
    evidence_src INTEGER,
    evidence_anchored REAL,
    decision TEXT NOT NULL,
    notes TEXT,
    reviewer TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


def ensure_smoothing_ratings_table(conn: sqlite3.Connection) -> None:
    """Create the smoothing_ratings table if it doesn't exist."""
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()


def store_smoothing_rating(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    variant: str,
    vector_source: str,
    slot_type: str,
    tier: str,
    filler: str,
    decision: str,
    base_p: float | None = None,
    smoothed_p: float | None = None,
    delta: float | None = None,
    evidence_soft: float | None = None,
    evidence_src: int | None = None,
    evidence_anchored: float | None = None,
    notes: str | None = None,
    reviewer: str | None = None,
) -> int:
    """Store one per-candidate smoothing rating. Returns the row id.

    ``run_id`` ties the rating to the exact candidate feed it judged, so a
    rating is always replayable against the inputs that produced the candidate.
    """
    if decision not in VALID_SMOOTHING_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_SMOOTHING_DECISIONS)}, got {decision!r}"
        )
    ensure_smoothing_ratings_table(conn)
    cur = conn.execute(
        """INSERT INTO smoothing_ratings
           (run_id, variant, vector_source, slot_type, tier, filler,
            base_p, smoothed_p, delta, evidence_soft, evidence_src,
            evidence_anchored, decision, notes, reviewer)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_id, variant, vector_source, slot_type, tier, filler,
            base_p, smoothed_p, delta, evidence_soft, evidence_src,
            evidence_anchored, decision, notes, reviewer,
        ),
    )
    conn.commit()
    return cur.lastrowid


def summarize_smoothing_ratings(
    conn: sqlite3.Connection,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate smoothing ratings into a per-variant / per-tier summary.

    This is the signal the Step 5 AutoResearcher consumes as its objective:
    which variants/tiers earned ``plausible_repair`` vs ``semantic_bleed`` etc.
    Optionally scope to a single ``run_id``. Read-only safe: if the table does
    not exist yet (no ratings recorded), returns an empty summary.
    """
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='smoothing_ratings'"
    ).fetchone()
    if table_exists is None:
        return {
            "total": 0,
            "runs": [],
            "variants": [],
            "by_decision": {},
            "by_variant": {},
            "by_variant_tier": {},
        }
    query = (
        "SELECT run_id, variant, vector_source, slot_type, tier, decision "
        "FROM smoothing_ratings"
    )
    params: tuple[Any, ...] = ()
    if run_id is not None:
        query += " WHERE run_id = ?"
        params = (run_id,)
    rows = conn.execute(query, params).fetchall()

    total = len(rows)
    by_decision: Counter[str] = Counter()
    by_variant: dict[str, Counter[str]] = {}
    by_variant_tier: dict[tuple[str, str], Counter[str]] = {}
    variants: set[str] = set()
    runs: set[str] = set()
    for run, variant, _vector_source, _slot, tier, decision in rows:
        by_decision[decision] += 1
        by_variant.setdefault(variant, Counter())[decision] += 1
        by_variant_tier.setdefault((variant, tier), Counter())[decision] += 1
        variants.add(variant)
        runs.add(run)

    return {
        "total": total,
        "runs": sorted(runs),
        "variants": sorted(variants),
        "by_decision": dict(by_decision),
        "by_variant": {
            variant: dict(counts) for variant, counts in sorted(by_variant.items())
        },
        "by_variant_tier": {
            f"{variant}|{tier}": dict(counts)
            for (variant, tier), counts in sorted(by_variant_tier.items())
        },
    }


# ---------------------------------------------------------------------------
# decision.json: committed overall accept/reject/iterate gate evidence
# ---------------------------------------------------------------------------

def build_decision_record(
    *,
    run_id: str,
    variant: str,
    vector_source: str,
    decision: str,
    summary: str,
    reviewer: str | None = None,
    ratings_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the overall Step 5 decision record (validated)."""
    if decision not in VALID_OVERALL_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_OVERALL_DECISIONS)}, got {decision!r}"
        )
    if not summary.strip():
        raise ValueError("summary must be non-empty")
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "run_id": run_id,
        "variant": variant,
        "vector_source": vector_source,
        "decision": decision,
        "summary": summary,
        "reviewer": reviewer,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ratings_summary": ratings_summary or {},
    }


def write_decision_record(path: Path, record: dict[str, Any]) -> None:
    """Persist the committed decision record as pretty JSON."""
    validate_decision_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_decision_record(path: Path) -> dict[str, Any]:
    """Read and validate a committed decision record."""
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_decision_record(record)
    return record


def validate_decision_record(record: dict[str, Any]) -> None:
    """Validate a decision record's required fields and enums."""
    required = {"schema_version", "run_id", "variant", "decision", "summary"}
    missing = required - record.keys()
    if missing:
        raise ValueError(f"decision record missing fields: {sorted(missing)}")
    if record["decision"] not in VALID_OVERALL_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_OVERALL_DECISIONS)}, "
            f"got {record['decision']!r}"
        )
    if not str(record["summary"]).strip():
        raise ValueError("summary must be non-empty")
