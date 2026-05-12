"""Diagnostics for real-title tier calibration."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from subtitle_generator.tiering import compute_tier_evidence


@dataclass(frozen=True)
class SourceTierLabelCase:
    title: str
    subtitle: str
    expected_tier: str
    rationale: str
    source: str


@dataclass(frozen=True)
class RealTitleTierResult:
    title: str
    subtitle: str
    expected_tier: str
    predicted_tier: str
    accessibility_score: float
    lower_tail_score: float
    demand_confidence: float
    rationale: str

    @property
    def correct(self) -> bool:
        return self.expected_tier == self.predicted_tier

    @property
    def pop_guardrail_ok(self) -> bool:
        if self.expected_tier == "pop":
            return self.predicted_tier != "niche"
        return self.predicted_tier != "pop"


def _source_subtitle_expr(columns: set[str]) -> str:
    if "candidate_source" in columns:
        return (
            "CASE WHEN candidate_source = 'title' "
            "THEN '' ELSE COALESCE(subtitle, '') END"
        )
    return "COALESCE(subtitle, '')"


def load_source_tier_label_cases(
    conn: sqlite3.Connection,
    *,
    min_confidence: float = 0.0,
) -> tuple[SourceTierLabelCase, ...]:
    """Load real source-title labels populated on pattern_matches rows."""

    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pattern_matches'"
    ).fetchone()
    if not table_exists:
        return ()

    columns = {row[1] for row in conn.execute("PRAGMA table_info(pattern_matches)")}
    required = {
        "llm_market_tier",
        "llm_market_tier_confidence",
        "llm_market_tier_rationale",
    }
    if not required <= columns:
        return ()
    subtitle_expr = _source_subtitle_expr(columns)

    rows = conn.execute(
        f"""
        SELECT
            COALESCE(title, ''),
            {subtitle_expr},
            llm_market_tier,
            COALESCE(llm_market_tier_rationale, ''),
            COALESCE(llm_market_tier_confidence, 0.0),
            id
        FROM pattern_matches
        WHERE llm_market_tier IN ('pop', 'mainstream', 'niche')
          AND COALESCE(llm_market_tier_confidence, 0.0) >= ?
        ORDER BY id
        """,
        (min_confidence,),
    ).fetchall()
    return tuple(
        SourceTierLabelCase(
            title=row[0],
            subtitle=row[1],
            expected_tier=row[2],
            rationale=row[3],
            source=f"pattern_matches/{row[5]}",
        )
        for row in rows
    )


def evaluate_real_title_tiers(
    conn: sqlite3.Connection,
    cases: tuple[SourceTierLabelCase, ...] | None = None,
    *,
    min_confidence: float = 0.0,
) -> list[RealTitleTierResult]:
    """Classify labeled real source titles with evidence details."""

    if cases is None:
        cases = load_source_tier_label_cases(conn, min_confidence=min_confidence)
    results: list[RealTitleTierResult] = []
    for case in cases:
        evidence = compute_tier_evidence(case.subtitle, conn)
        results.append(RealTitleTierResult(
            title=case.title,
            subtitle=case.subtitle,
            expected_tier=case.expected_tier,
            predicted_tier=evidence.tier,
            accessibility_score=evidence.accessibility_score,
            lower_tail_score=evidence.lower_tail_score,
            demand_confidence=evidence.demand_confidence,
            rationale=case.rationale,
        ))
    return results


def measure_real_title_tier_accuracy(conn: sqlite3.Connection) -> float:
    """Return real-title label accuracy as a 0-1 metric."""

    results = evaluate_real_title_tiers(conn)
    if not results:
        return 0.0
    return sum(1 for result in results if result.correct) / len(results)


def measure_real_title_tier_pop_guardrail(conn: sqlite3.Connection) -> float:
    """Return the label score for avoiding false-pop and false-niche-pop calls."""

    score, _count = score_real_title_tier_pop_guardrail(conn)
    return score


def score_real_title_tier_pop_guardrail(conn: sqlite3.Connection) -> tuple[float, int]:
    """Return the pop guardrail score and number of source labels evaluated."""

    results = evaluate_real_title_tiers(conn)
    if not results:
        return 1.0, 0
    score = sum(1 for result in results if result.pop_guardrail_ok) / len(results)
    return score, len(results)


def format_real_title_tier_report(conn: sqlite3.Connection) -> str:
    """Format the real-title tier diagnostic report."""

    results = evaluate_real_title_tiers(conn)
    correct = sum(1 for result in results if result.correct)
    guardrail_ok = sum(1 for result in results if result.pop_guardrail_ok)
    total = len(results)
    confusion: dict[tuple[str, str], int] = {}
    for result in results:
        key = (result.expected_tier, result.predicted_tier)
        confusion[key] = confusion.get(key, 0) + 1

    lines = [
        "# Real-title tier diagnostic",
        "",
    ]
    if not results:
        lines.extend([
            "No `pattern_matches.llm_market_tier` labels found.",
            "",
            "Populate source-title tier labels in the full database before using this diagnostic.",
        ])
        return "\n".join(lines)

    lines.extend([
        f"Exact accuracy: {correct}/{total} ({(correct / total) if total else 0.0:.0%})",
        f"Pop guardrail: {guardrail_ok}/{total} ({(guardrail_ok / total) if total else 0.0:.0%})",
        "",
        "## Confusion matrix",
        "",
        "| Expected | Predicted | Count |",
        "|---|---|---:|",
    ])
    for (expected, predicted), count in sorted(confusion.items()):
        lines.append(f"| {expected} | {predicted} | {count} |")

    lines.extend([
        "",
        "## Cases",
        "",
        (
            "| Result | Pop guardrail | Expected | Predicted | Title | "
            "Accessibility | Lower tail | Demand |"
        ),
        "|---|---|---|---|---|---:|---:|---:|",
    ])
    for result in results:
        marker = "PASS" if result.correct else "FAIL"
        pop_marker = "PASS" if result.pop_guardrail_ok else "FAIL"
        lines.append(
            f"| {marker} | {pop_marker} | {result.expected_tier} | {result.predicted_tier} | "
            f"{result.title} | {result.accessibility_score:.3f} | "
            f"{result.lower_tail_score:.3f} | {result.demand_confidence:.3f} |"
        )
    return "\n".join(lines)

