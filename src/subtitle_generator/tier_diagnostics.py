"""Diagnostics for real-title tier calibration."""

from __future__ import annotations

import sqlite3
from collections import Counter
from dataclasses import dataclass

from subtitle_generator.config import invalidate_config_cache, load_tuning_config
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


@dataclass(frozen=True)
class TierGateCalibration:
    label_count: int
    label_distribution: dict[str, int]
    target_generation_ratios: dict[str, float]
    baseline_exact_accuracy: float
    baseline_macro_accuracy: float
    baseline_pop_recall: float
    baseline_pop_guardrail: float
    baseline_confusion: dict[tuple[str, str], int]
    exact_accuracy: float
    macro_accuracy: float
    pop_recall: float
    pop_guardrail: float
    accessibility_threshold_pop: float
    accessibility_threshold_mainstream: float
    tier_pop_min_lower_tail: float
    tier_pop_min_demand_confidence: float
    tier_center_pop: float
    tier_center_mainstream: float
    tier_center_niche: float
    confusion: dict[tuple[str, str], int]

    def threshold_config_values(self) -> dict[str, float]:
        return {
            "accessibility_threshold_pop": self.accessibility_threshold_pop,
            "accessibility_threshold_mainstream": self.accessibility_threshold_mainstream,
            "tier_pop_min_lower_tail": self.tier_pop_min_lower_tail,
            "tier_pop_min_demand_confidence": self.tier_pop_min_demand_confidence,
        }

    def generation_ratio_config_values(self) -> dict[str, float]:
        return {
            f"generation_tier_ratio_{tier}": self.target_generation_ratios[tier]
            for tier in ("pop", "mainstream", "niche")
        }

    def generation_center_config_values(self) -> dict[str, float]:
        return {
            "tier_center_pop": self.tier_center_pop,
            "tier_center_mainstream": self.tier_center_mainstream,
            "tier_center_niche": self.tier_center_niche,
        }

    def as_config_values(self) -> dict[str, float]:
        return {
            **self.threshold_config_values(),
            **self.generation_ratio_config_values(),
            **self.generation_center_config_values(),
        }


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


def suggest_tier_gate_config(
    conn: sqlite3.Connection,
    *,
    min_confidence: float = 0.0,
) -> TierGateCalibration | None:
    """Search deterministic tier gates that best match source-title labels."""

    results = evaluate_real_title_tiers(conn, min_confidence=min_confidence)
    if not results:
        return None

    current = load_tuning_config(conn)
    accessibility_values = [result.accessibility_score for result in results]
    lower_tail_values = [result.lower_tail_score for result in results]
    demand_values = [result.demand_confidence for result in results]

    pop_thresholds = _candidate_thresholds(
        accessibility_values,
        current["accessibility_threshold_pop"],
    )
    mainstream_thresholds = _candidate_thresholds(
        accessibility_values,
        current["accessibility_threshold_mainstream"],
    )
    lower_tail_thresholds = _candidate_thresholds(
        lower_tail_values,
        current["tier_pop_min_lower_tail"],
    )
    demand_thresholds = _candidate_thresholds(
        demand_values,
        current["tier_pop_min_demand_confidence"],
    )

    baseline = _score_gate_candidate(
        results,
        pop_threshold=current["accessibility_threshold_pop"],
        mainstream_threshold=current["accessibility_threshold_mainstream"],
        lower_tail_threshold=current["tier_pop_min_lower_tail"],
        demand_threshold=current["tier_pop_min_demand_confidence"],
    )
    baseline_objective = _gate_objective(baseline)
    best: tuple[float, float, float, float, float, TierGateCalibration] | None = (
        baseline_objective,
        baseline.macro_accuracy,
        baseline.pop_recall,
        baseline.pop_guardrail,
        baseline.exact_accuracy,
        baseline,
    )
    for pop_threshold in pop_thresholds:
        for mainstream_threshold in mainstream_thresholds:
            if mainstream_threshold > pop_threshold:
                continue
            for lower_tail_threshold in lower_tail_thresholds:
                for demand_threshold in demand_thresholds:
                    calibration = _score_gate_candidate(
                        results,
                        pop_threshold=pop_threshold,
                        mainstream_threshold=mainstream_threshold,
                        lower_tail_threshold=lower_tail_threshold,
                        demand_threshold=demand_threshold,
                    )
                    if _has_expected_pop(results) and calibration.pop_recall <= 0.0:
                        continue
                    if (
                        calibration.exact_accuracy < baseline.exact_accuracy
                        or calibration.macro_accuracy < baseline.macro_accuracy
                        or calibration.pop_recall < baseline.pop_recall
                        or calibration.pop_guardrail < baseline.pop_guardrail
                    ):
                        continue
                    objective = _gate_objective(calibration)
                    tie_breaker = (
                        objective,
                        calibration.macro_accuracy,
                        calibration.pop_recall,
                        calibration.pop_guardrail,
                        calibration.exact_accuracy,
                    )
                    if best is None or tie_breaker > best[:5]:
                        best = (*tie_breaker, calibration)
    return _with_source_label_context(best[5], baseline, results, current)


def _gate_objective(calibration: TierGateCalibration) -> float:
    return (
        (0.55 * calibration.macro_accuracy)
        + (0.2 * calibration.pop_guardrail)
        + (0.15 * calibration.exact_accuracy)
        + (0.1 * calibration.pop_recall)
    )


def apply_tier_gate_calibration(
    conn: sqlite3.Connection,
    calibration: TierGateCalibration,
) -> None:
    """Persist suggested tier gates to config."""

    conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
    for key, value in calibration.as_config_values().items():
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, str(round(value, 4))),
        )
    conn.commit()
    invalidate_config_cache()


def _candidate_thresholds(values: list[float], current_value: float) -> tuple[float, ...]:
    if not values:
        return (current_value,)
    sorted_values = sorted(values)
    n = len(sorted_values)
    candidates = {0.0, current_value}
    for percentile in range(0, 101, 5):
        idx = min(int(n * percentile / 100), n - 1)
        value = sorted_values[idx]
        candidates.add(value)
        if value < sorted_values[-1]:
            candidates.add(max(0.0, value + 0.0001))
    return tuple(sorted(round(candidate, 4) for candidate in candidates))


def _score_gate_candidate(
    results: list[RealTitleTierResult],
    *,
    pop_threshold: float,
    mainstream_threshold: float,
    lower_tail_threshold: float,
    demand_threshold: float,
) -> TierGateCalibration:
    confusion: Counter[tuple[str, str]] = Counter()
    correct = 0
    guardrail_ok = 0
    tier_totals: Counter[str] = Counter()
    tier_correct: Counter[str] = Counter()
    for result in results:
        predicted = _classify_with_gate_values(
            accessibility_score=result.accessibility_score,
            lower_tail_score=result.lower_tail_score,
            demand_confidence=result.demand_confidence,
            pop_threshold=pop_threshold,
            mainstream_threshold=mainstream_threshold,
            lower_tail_threshold=lower_tail_threshold,
            demand_threshold=demand_threshold,
        )
        confusion[(result.expected_tier, predicted)] += 1
        tier_totals[result.expected_tier] += 1
        if predicted == result.expected_tier:
            correct += 1
            tier_correct[result.expected_tier] += 1
        if (
            predicted != "niche"
            if result.expected_tier == "pop"
            else predicted != "pop"
        ):
            guardrail_ok += 1
    total = len(results)
    macro_accuracy = sum(
        tier_correct[tier] / tier_totals[tier]
        for tier in ("pop", "mainstream", "niche")
        if tier_totals[tier]
    ) / len(tier_totals)
    pop_recall = (
        tier_correct["pop"] / tier_totals["pop"]
        if tier_totals["pop"]
        else 1.0
    )
    return TierGateCalibration(
        label_count=total,
        label_distribution={},
        target_generation_ratios={},
        baseline_exact_accuracy=correct / total,
        baseline_macro_accuracy=macro_accuracy,
        baseline_pop_recall=pop_recall,
        baseline_pop_guardrail=guardrail_ok / total,
        baseline_confusion=dict(confusion),
        exact_accuracy=correct / total,
        macro_accuracy=macro_accuracy,
        pop_recall=pop_recall,
        pop_guardrail=guardrail_ok / total,
        accessibility_threshold_pop=pop_threshold,
        accessibility_threshold_mainstream=mainstream_threshold,
        tier_pop_min_lower_tail=lower_tail_threshold,
        tier_pop_min_demand_confidence=demand_threshold,
        tier_center_pop=0.0,
        tier_center_mainstream=0.0,
        tier_center_niche=0.0,
        confusion=dict(confusion),
    )


def _with_source_label_context(
    calibration: TierGateCalibration,
    baseline: TierGateCalibration,
    results: list[RealTitleTierResult],
    current: dict[str, float],
) -> TierGateCalibration:
    distribution = _label_distribution(results)
    ratios = _target_generation_ratios(distribution)
    return TierGateCalibration(
        label_count=calibration.label_count,
        label_distribution=distribution,
        target_generation_ratios=ratios,
        baseline_exact_accuracy=baseline.exact_accuracy,
        baseline_macro_accuracy=baseline.macro_accuracy,
        baseline_pop_recall=baseline.pop_recall,
        baseline_pop_guardrail=baseline.pop_guardrail,
        baseline_confusion=baseline.confusion,
        exact_accuracy=calibration.exact_accuracy,
        macro_accuracy=calibration.macro_accuracy,
        pop_recall=calibration.pop_recall,
        pop_guardrail=calibration.pop_guardrail,
        accessibility_threshold_pop=calibration.accessibility_threshold_pop,
        accessibility_threshold_mainstream=calibration.accessibility_threshold_mainstream,
        tier_pop_min_lower_tail=calibration.tier_pop_min_lower_tail,
        tier_pop_min_demand_confidence=calibration.tier_pop_min_demand_confidence,
        tier_center_pop=current["tier_center_pop"],
        tier_center_mainstream=current["tier_center_mainstream"],
        tier_center_niche=current["tier_center_niche"],
        confusion=calibration.confusion,
    )


def _label_distribution(results: list[RealTitleTierResult]) -> dict[str, int]:
    counts = Counter(result.expected_tier for result in results)
    return {tier: counts[tier] for tier in ("pop", "mainstream", "niche")}


def _target_generation_ratios(distribution: dict[str, int]) -> dict[str, float]:
    total = sum(distribution.values())
    if total <= 0:
        return {"pop": 0.0, "mainstream": 1.0, "niche": 0.0}
    return {
        tier: distribution[tier] / total
        for tier in ("pop", "mainstream", "niche")
    }


def _has_expected_pop(results: list[RealTitleTierResult]) -> bool:
    return any(result.expected_tier == "pop" for result in results)


def _classify_with_gate_values(
    *,
    accessibility_score: float,
    lower_tail_score: float,
    demand_confidence: float,
    pop_threshold: float,
    mainstream_threshold: float,
    lower_tail_threshold: float,
    demand_threshold: float,
) -> str:
    if (
        accessibility_score >= pop_threshold
        and lower_tail_score >= lower_tail_threshold
        and demand_confidence >= demand_threshold
    ):
        return "pop"
    if accessibility_score < mainstream_threshold:
        return "niche"
    return "mainstream"


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


def format_tier_gate_calibration_report(
    conn: sqlite3.Connection,
    *,
    min_confidence: float = 0.0,
    calibration: TierGateCalibration | None = None,
) -> str:
    """Format suggested deterministic tier gates from source-title labels."""

    calibration = calibration or suggest_tier_gate_config(
        conn,
        min_confidence=min_confidence,
    )
    lines = [
        "# Tier-gate calibration suggestion",
        "",
    ]
    if calibration is None:
        lines.extend([
            "No `pattern_matches.llm_market_tier` labels found.",
            "",
            "Populate source-title tier labels before calibrating tier gates.",
        ])
        return "\n".join(lines)

    current = load_tuning_config(conn)
    lines.extend([
        f"Labels: {calibration.label_count}",
        "",
        "## Source-label distribution",
        "",
        "| Tier | Labels | Target generation ratio |",
        "|---|---:|---:|",
    ])
    for tier in ("pop", "mainstream", "niche"):
        lines.append(
            f"| {tier} | {calibration.label_distribution[tier]} | "
            f"{calibration.target_generation_ratios[tier]:.4f} |"
        )

    lines.extend([
        "",
        "## Fit metrics",
        "",
        "| Metric | Baseline | Suggested |",
        "|---|---:|---:|",
        (
            f"| Exact accuracy | {calibration.baseline_exact_accuracy:.0%} | "
            f"{calibration.exact_accuracy:.0%} |"
        ),
        (
            f"| Macro accuracy | {calibration.baseline_macro_accuracy:.0%} | "
            f"{calibration.macro_accuracy:.0%} |"
        ),
        (
            f"| Pop recall | {calibration.baseline_pop_recall:.0%} | "
            f"{calibration.pop_recall:.0%} |"
        ),
        (
            f"| Pop guardrail | {calibration.baseline_pop_guardrail:.0%} | "
            f"{calibration.pop_guardrail:.0%} |"
        ),
        "",
        f"Suggested exact accuracy: {calibration.exact_accuracy:.0%}",
        f"Suggested macro accuracy: {calibration.macro_accuracy:.0%}",
        f"Suggested pop recall: {calibration.pop_recall:.0%}",
        f"Suggested pop guardrail: {calibration.pop_guardrail:.0%}",
        "",
        "## Suggested config",
        "",
        "| Key | Current | Suggested |",
        "|---|---:|---:|",
    ])
    for key, value in calibration.as_config_values().items():
        lines.append(f"| {key} | {current[key]:.4f} | {value:.4f} |")

    lines.extend([
        "",
        "## Baseline confusion matrix",
        "",
        "| Expected | Predicted | Count |",
        "|---|---|---:|",
    ])
    for (expected, predicted), count in sorted(calibration.baseline_confusion.items()):
        lines.append(f"| {expected} | {predicted} | {count} |")

    lines.extend([
        "",
        "## Suggested confusion matrix",
        "",
        "| Expected | Predicted | Count |",
        "|---|---|---:|",
    ])
    for (expected, predicted), count in sorted(calibration.confusion.items()):
        lines.append(f"| {expected} | {predicted} | {count} |")
    return "\n".join(lines)
