"""Constrained calibration for popularity source weights.

This module learns the ratios between source-specific popularity signals while
preserving the runtime contract: downstream generation still consumes one
collapsed ``popularity_score``.
"""

from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from subtitle_generator.config import ALL_TUNABLE_PARAMS, invalidate_config_cache

DEMAND_SOURCES = ("spl", "goodreads", "library", "nyt", "trove")
CONFIG_KEYS = {
    "spl": "pop_weight_spl",
    "goodreads": "pop_weight_gr",
    "library": "pop_weight_library",
    "nyt": "pop_weight_nyt",
    "trove": "pop_weight_trove",
}
TARGET_MODES = ("accessibility", "pop-only")


@dataclass(frozen=True)
class CalibrationExample:
    pattern_match_id: str
    title: str
    subtitle_text: str
    target: float
    ol_norm: float
    source_scores: dict[str, float]
    source_present: dict[str, bool]
    current_score: float | None


@dataclass(frozen=True)
class WeightFit:
    weights: dict[str, float]
    mse: float


@dataclass(frozen=True)
class PopularityCalibrationResult:
    report_path: Path
    learned_weights: dict[str, float]
    current_weights: dict[str, float]
    learned_mse: float
    current_mse: float
    example_count: int
    applied: bool


def calibrate_popularity_weights(
    *,
    features_path: Path,
    teacher_predictions_path: Path,
    output_dir: Path,
    db_path: Path | None = None,
    apply: bool = False,
    target_mode: str = "accessibility",
    regularization: float = 0.01,
    min_weight_share: float = 0.02,
) -> PopularityCalibrationResult:
    """Fit demand-source weights against teacher predictions and write a report."""

    if target_mode not in TARGET_MODES:
        raise RuntimeError(
            "Unknown target mode. Expected one of: " + ", ".join(TARGET_MODES)
        )
    if not 0 <= min_weight_share < 1 / len(DEMAND_SOURCES):
        raise RuntimeError(
            f"min_weight_share must be >= 0 and < {1 / len(DEMAND_SOURCES):.3f}"
        )

    feature_rows = _read_csv_by_id(features_path)
    teacher_rows = _read_csv_by_id(teacher_predictions_path)
    shared_ids = sorted(set(feature_rows) & set(teacher_rows), key=_sort_key)
    if not shared_ids:
        raise RuntimeError("No shared pattern_match_id values found between inputs")

    percentiles = _build_percentile_models(feature_rows.values())
    examples = [
        _build_example(feature_rows[pattern_id], teacher_rows[pattern_id], percentiles, target_mode)
        for pattern_id in shared_ids
    ]
    examples = [example for example in examples if _has_teacher_scores(teacher_rows[example.pattern_match_id])]
    if not examples:
        raise RuntimeError("No examples with teacher score_pop/score_mainstream/score_niche values found")

    current_weights = _current_demand_weights()
    current_fit = WeightFit(
        weights=current_weights,
        mse=_objective(examples, current_weights, current_weights, regularization=0.0),
    )
    learned_fit = _fit_weights(
        examples,
        initial_weights=current_weights,
        regularization=regularization,
        min_weight_share=min_weight_share,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "popularity_weight_calibration.md"
    report_path.write_text(
        _format_report(
            examples=examples,
            current_fit=current_fit,
            learned_fit=learned_fit,
            target_mode=target_mode,
            regularization=regularization,
            min_weight_share=min_weight_share,
            db_path=db_path,
            applied=apply,
        ),
        encoding="utf-8",
    )

    if apply:
        if db_path is None:
            raise RuntimeError("--apply requires --db")
        _apply_weights(db_path, learned_fit.weights)

    return PopularityCalibrationResult(
        report_path=report_path,
        learned_weights=learned_fit.weights,
        current_weights=current_fit.weights,
        learned_mse=learned_fit.mse,
        current_mse=current_fit.mse,
        example_count=len(examples),
        applied=apply,
    )


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["pattern_match_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (0, value)


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def _log1p_values(rows, column: str) -> list[float]:
    values = [math.log10(1 + max(0.0, _float(row, column))) for row in rows]
    values = [value for value in values if value > 0]
    return sorted(values)


def _build_percentile_models(rows) -> dict[str, list[float]]:
    rows = list(rows)
    return {
        "spl": _log1p_values(rows, "checkouts_per_year"),
        "goodreads": _log1p_values(rows, "gr_ratings_count"),
        "library": _log1p_values(rows, "library_appearances"),
        "trove": _log1p_values(rows, "trove_library_count"),
        "open_library": _log1p_values(rows, "ol_edition_count"),
    }


def _percentile(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    lo = 0
    hi = len(sorted_values)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_values[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(sorted_values)


def _teacher_target(row: dict[str, str], target_mode: str) -> float:
    pop = _float(row, "score_pop")
    if target_mode == "pop-only":
        return pop
    mainstream = _float(row, "score_mainstream")
    return pop + 0.5 * mainstream


def _has_teacher_scores(row: dict[str, str]) -> bool:
    return any(row.get(key) not in (None, "") for key in ("score_pop", "score_mainstream", "score_niche"))


def _build_example(
    feature_row: dict[str, str],
    teacher_row: dict[str, str],
    percentiles: dict[str, list[float]],
    target_mode: str,
) -> CalibrationExample:
    spl_raw = _float(feature_row, "checkouts_per_year")
    gr_raw = _float(feature_row, "gr_ratings_count")
    library_raw = _float(feature_row, "library_appearances")
    trove_raw = _float(feature_row, "trove_library_count")
    nyt_weeks = _float(feature_row, "nyt_weeks_on_list")
    ol_raw = _float(feature_row, "ol_edition_count")

    source_present = {
        "spl": spl_raw > 0,
        "goodreads": gr_raw > 0,
        "library": library_raw > 0,
        "trove": trove_raw > 0,
        "nyt": nyt_weeks > 0,
    }
    source_scores = {
        "spl": _percentile(percentiles["spl"], math.log10(1 + spl_raw)),
        "goodreads": _percentile(percentiles["goodreads"], math.log10(1 + gr_raw)),
        "library": _percentile(percentiles["library"], math.log10(1 + library_raw)),
        "trove": _percentile(percentiles["trove"], math.log10(1 + trove_raw)),
        "nyt": min(1.0, 0.8 + 0.2 * math.log10(1 + nyt_weeks) / 2.0) if nyt_weeks > 0 else 0.0,
    }
    current_score = (
        _float(feature_row, "work_popularity_score")
        if feature_row.get("work_popularity_score") not in (None, "")
        else None
    )
    return CalibrationExample(
        pattern_match_id=feature_row["pattern_match_id"],
        title=feature_row.get("title", ""),
        subtitle_text=feature_row.get("subtitle_text", ""),
        target=_teacher_target(teacher_row, target_mode),
        ol_norm=_percentile(percentiles["open_library"], math.log10(1 + max(ol_raw, 1.0))),
        source_scores=source_scores,
        source_present=source_present,
        current_score=current_score,
    )


def _current_demand_weights() -> dict[str, float]:
    return {
        source: float(ALL_TUNABLE_PARAMS[CONFIG_KEYS[source]])
        for source in DEMAND_SOURCES
    }


def _normalize(
    weights: dict[str, float],
    total: float = 1.0,
    *,
    min_weight_share: float = 0.0,
) -> dict[str, float]:
    floor = total * min_weight_share
    floor_total = floor * len(DEMAND_SOURCES)
    remaining_total = total - floor_total
    if remaining_total < 0:
        raise RuntimeError("Minimum weight share is too large")
    clamped = {source: max(0.0, weights[source]) for source in DEMAND_SOURCES}
    weight_sum = sum(clamped.values())
    if weight_sum <= 0:
        equal = total / len(DEMAND_SOURCES)
        return {source: equal for source in DEMAND_SOURCES}
    return {
        source: floor + remaining_total * clamped[source] / weight_sum
        for source in DEMAND_SOURCES
    }


def _composite_score(example: CalibrationExample, weights: dict[str, float]) -> float:
    total_possible = sum(weights.values())
    if total_possible <= 0:
        return example.ol_norm
    total_present = sum(
        weights[source]
        for source in DEMAND_SOURCES
        if example.source_present[source]
    )
    if total_present > 0:
        demand_score = sum(
            weights[source] * example.source_scores[source]
            for source in DEMAND_SOURCES
            if example.source_present[source]
        ) / total_present
    else:
        demand_score = 0.0
    confidence = min(total_present / total_possible, 1.0)
    composite = confidence * demand_score + (1 - confidence) * example.ol_norm
    return min(composite, 0.5) if confidence == 0 else composite


def _objective(
    examples: list[CalibrationExample],
    weights: dict[str, float],
    prior_weights: dict[str, float],
    *,
    regularization: float,
) -> float:
    mse = mean(
        (_composite_score(example, weights) - example.target) ** 2
        for example in examples
    )
    if regularization <= 0:
        return mse
    weights_norm = _normalize(weights)
    prior_norm = _normalize(prior_weights)
    ridge = sum(
        (weights_norm[source] - prior_norm[source]) ** 2
        for source in DEMAND_SOURCES
    )
    return mse + regularization * ridge


def _fit_weights(
    examples: list[CalibrationExample],
    *,
    initial_weights: dict[str, float],
    regularization: float,
    min_weight_share: float,
) -> WeightFit:
    total = sum(initial_weights.values())
    starts = [
        _normalize(initial_weights, total, min_weight_share=min_weight_share),
        {source: total / len(DEMAND_SOURCES) for source in DEMAND_SOURCES},
    ]
    for source in DEMAND_SOURCES:
        starts.append(
            _normalize(
                {candidate: (total if candidate == source else 0.0) for candidate in DEMAND_SOURCES},
                total,
                min_weight_share=min_weight_share,
            )
        )

    weights = min(
        starts,
        key=lambda candidate: _objective(
            examples,
            candidate,
            initial_weights,
            regularization=regularization,
        ),
    )
    best = _objective(examples, weights, initial_weights, regularization=regularization)
    step = total / 4

    while step > total / 1000:
        improved = False
        for source_from in DEMAND_SOURCES:
            if weights[source_from] <= 0:
                continue
            for source_to in DEMAND_SOURCES:
                if source_from == source_to:
                    continue
                shift = min(step, weights[source_from])
                candidate = dict(weights)
                candidate[source_from] -= shift
                candidate[source_to] += shift
                candidate = _normalize(
                    candidate,
                    total,
                    min_weight_share=min_weight_share,
                )
                score = _objective(
                    examples,
                    candidate,
                    initial_weights,
                    regularization=regularization,
                )
                if score + 1e-12 < best:
                    weights = candidate
                    best = score
                    improved = True
        if not improved:
            step /= 2

    learned_mse = _objective(examples, weights, initial_weights, regularization=0.0)
    return WeightFit(
        weights=_normalize(weights, total, min_weight_share=min_weight_share),
        mse=learned_mse,
    )


def _apply_weights(db_path: Path, weights: dict[str, float]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
        for source, weight in weights.items():
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (CONFIG_KEYS[source], f"{weight:.8f}"),
            )
        conn.commit()
    finally:
        conn.close()
    invalidate_config_cache()


def _format_report(
    *,
    examples: list[CalibrationExample],
    current_fit: WeightFit,
    learned_fit: WeightFit,
    target_mode: str,
    regularization: float,
    min_weight_share: float,
    db_path: Path | None,
    applied: bool,
) -> str:
    equal_weights = {source: sum(current_fit.weights.values()) / len(DEMAND_SOURCES) for source in DEMAND_SOURCES}
    fits = [
        ("current", current_fit),
        ("learned", learned_fit),
        ("equal", WeightFit(equal_weights, _objective(examples, equal_weights, current_fit.weights, regularization=0.0))),
    ]
    for source in DEMAND_SOURCES:
        source_only = {candidate: (sum(current_fit.weights.values()) if candidate == source else 0.0) for candidate in DEMAND_SOURCES}
        fits.append((f"{source}-only", WeightFit(source_only, _objective(examples, source_only, current_fit.weights, regularization=0.0))))

    coverage = {
        source: sum(1 for example in examples if example.source_present[source])
        for source in DEMAND_SOURCES
    }
    deltas = sorted(
        (
            (
                abs(_composite_score(example, learned_fit.weights) - _composite_score(example, current_fit.weights)),
                example,
                _composite_score(example, current_fit.weights),
                _composite_score(example, learned_fit.weights),
            )
            for example in examples
        ),
        key=lambda item: (item[0], _sort_key(item[1].pattern_match_id)),
    )[-10:]

    lines = [
        "# Popularity weight calibration",
        "",
        f"- Examples: {len(examples):,}",
        f"- Target mode: `{target_mode}`",
        f"- Regularization toward current weights: {regularization:g}",
        f"- Minimum learned source share: {min_weight_share:g}",
        f"- Applied to DB: {'yes' if applied else 'no'}",
        f"- DB: `{db_path}`" if db_path else "- DB: not provided",
        "",
        "## Important interpretation notes",
        "",
        "- The learned values only calibrate demand-source ratios used before the single runtime `popularity_score` is rebuilt.",
        "- OpenLibrary edition count remains the fallback/confidence signal in the current popularity formula; `pop_weight_ol` is not rewritten by this calibration.",
        "- If the teacher was trained with `feature-set all` or `feature-set popularity`, it already saw existing popularity features, so this report should be read as teacher-alignment calibration rather than independent ground truth.",
        "",
        "## Weights",
        "",
        "| Source | Current | Learned | Coverage |",
        "|---|---:|---:|---:|",
    ]
    for source in DEMAND_SOURCES:
        lines.append(
            f"| {source} | {current_fit.weights[source]:.4f} | {learned_fit.weights[source]:.4f} | {coverage[source]:,} |"
        )

    lines.extend([
        "",
        "## Fit comparison",
        "",
        "| Model | MSE |",
        "|---|---:|",
    ])
    for label, fit in fits:
        lines.append(f"| {label} | {fit.mse:.6f} |")

    lines.extend([
        "",
        "## Largest score deltas",
        "",
        "| Pattern | Current | Learned | Target | Title |",
        "|---|---:|---:|---:|---|",
    ])
    for _, example, current_score, learned_score in reversed(deltas):
        title = (example.title or example.subtitle_text).replace("|", "\\|")
        lines.append(
            f"| {example.pattern_match_id} | {current_score:.3f} | {learned_score:.3f} | {example.target:.3f} | {title[:80]} |"
        )
    lines.append("")
    return "\n".join(lines)
