"""Human-sign-off contract for Step 6 calibration review (#39).

Like Step 5's smoothing review, an `autoresearcher`-labeled step must close with a
*durable* human decision, not an ephemeral "a human looked at it". Calibration's
objective is quantitative (held-out likelihood / ECE), so -- unlike Step 5's
per-candidate bleed ratings -- the review packet is a single accept/iterate
sign-off on the chosen temperature config and its diversity-vs-distinctiveness
trade-off. The decision plus the quantitative evidence is written to a small
committed ``decision.json`` as the exit-gate evidence.

This module is analysis-only and never touches the served distribution.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_OVERALL_DECISIONS = frozenset({"accept", "reject", "iterate"})

DECISION_SCHEMA_VERSION = 1


def _temperatures_match(
    submitted: dict[str, float], authoritative: dict[str, float]
) -> bool:
    """Whether a submitted temperature map matches the reviewed metadata map."""

    if submitted.keys() != authoritative.keys():
        return False
    return all(
        math.isclose(
            float(submitted[key]), float(authoritative[key]), rel_tol=1e-9, abs_tol=1e-12
        )
        for key in authoritative
    )


def build_decision_record(
    *,
    granularity: str,
    decision: str,
    summary: str,
    temperatures: dict[str, float],
    metadata_digest: str | None = None,
    metrics: dict[str, Any] | None = None,
    reviewer: str | None = None,
) -> dict[str, Any]:
    """Construct the validated Step 6 calibration decision record."""

    if decision not in VALID_OVERALL_DECISIONS:
        raise ValueError(
            f"decision must be one of {sorted(VALID_OVERALL_DECISIONS)}, got {decision!r}"
        )
    if not summary.strip():
        raise ValueError("summary must be non-empty")
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "granularity": granularity,
        "decision": decision,
        "summary": summary,
        "temperatures": temperatures,
        "metadata_digest": metadata_digest,
        "metrics": metrics or {},
        "reviewer": reviewer,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def validate_decision_record(record: dict[str, Any]) -> None:
    """Validate a decision record's required fields and enums."""

    required = {"schema_version", "granularity", "decision", "summary", "temperatures"}
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


def write_decision_record(path: Path, record: dict[str, Any]) -> None:
    """Persist the committed calibration decision record as pretty JSON."""

    validate_decision_record(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def read_decision_record(path: Path) -> dict[str, Any]:
    """Read and validate a committed calibration decision record."""

    record = json.loads(path.read_text(encoding="utf-8"))
    validate_decision_record(record)
    return record


def ingest_decision(
    submission: dict[str, Any],
    *,
    decision_path: Path,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    """Persist a calibration sign-off submission to ``decision.json``.

    ``submission`` is the small payload a reviewer (or the review surface) posts::

        {
          "granularity": "per_tier",
          "reviewer": "avi",
          "overall": {"decision": "accept|reject|iterate", "summary": "..."}
        }

    Temperatures and metrics are read from the calibration metadata when a
    ``metadata_path`` is given, so the decision is tied to exactly what it judged.
    The metadata temperatures are **authoritative**: a submission may omit
    temperatures (they are then filled from metadata), but if it supplies
    temperatures that disagree with the reviewed metadata the decision is
    rejected -- a sign-off must not silently record a config other than the one
    that produced the evidence.

    The temperatures recorded here are *calibration* temperatures (the shape
    correction baked into the sidecar distribution). They are NOT Step 7's
    runtime *sampling* temperature; see ``tier_slot_calibration`` for the split.
    """

    overall = submission.get("overall") or {}
    temperatures: dict[str, float] = dict(submission.get("temperatures", {}) or {})
    metadata_digest: str | None = None
    metrics: dict[str, Any] = {}
    granularity = str(submission.get("granularity", "per_tier"))

    if metadata_path is not None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata_temps = dict(metadata.get("temperatures", {}) or {})
        submitted_temps = dict(submission.get("temperatures", {}) or {})
        if submitted_temps and not _temperatures_match(submitted_temps, metadata_temps):
            raise ValueError(
                "submission temperatures "
                f"{submitted_temps} do not match the reviewed calibration metadata "
                f"temperatures {metadata_temps}; the metadata is authoritative -- omit "
                "submission temperatures or correct them to sign off on the reviewed config"
            )
        temperatures = metadata_temps
        metadata_digest = metadata.get("input_digest") or metadata.get(
            "fold_assignment_digest"
        )
        granularity = str(metadata.get("config", {}).get("granularity", granularity))
        metrics = {
            "heldout_nll": metadata.get("heldout_nll", {}),
            "reliability_ece": metadata.get("reliability_ece", {}),
            "distinctiveness": metadata.get("distinctiveness", {}),
            "ranking": metadata.get("ranking", {}),
        }

    record = build_decision_record(
        granularity=granularity,
        decision=str(overall["decision"]),
        summary=str(overall.get("summary", "")),
        temperatures=temperatures,
        metadata_digest=metadata_digest,
        metrics=metrics,
        reviewer=submission.get("reviewer"),
    )
    write_decision_record(decision_path, record)
    return {
        "decision": record["decision"],
        "granularity": record["granularity"],
        "decision_path": str(decision_path),
    }
