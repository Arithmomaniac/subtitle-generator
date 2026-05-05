"""Deployment-gate review for book-model scoring strategies."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from titlecase import titlecase

from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL

RiskLabel = Literal["nonsensical", "intriguing_or_funny", "acceptable"]
Winner = Literal["current", "candidate", "tie"]
StrategyReviewFn = Callable[
    [tuple["StrategyComparison", ...], str],
    tuple["StrategyReview", ...],
]

SAMPLE_COLUMNS = (
    "id",
    "seed",
    "student",
    "strategy",
    "current_subtitle",
    "candidate_subtitle",
)
REVIEW_COLUMNS = (
    *SAMPLE_COLUMNS,
    "winner",
    "current_risk",
    "candidate_risk",
    "tier_match_winner",
    "rationale",
)
STRATEGIES = (
    ("blend-70-current", 0.7),
    ("blend-50", 0.5),
    ("blend-70-model", 0.3),
    ("model-only", 0.0),
)


@dataclass(frozen=True)
class StrategyComparison:
    id: int
    seed: int
    student: str
    strategy: str
    current_subtitle: str
    candidate_subtitle: str


@dataclass(frozen=True)
class StrategyReview:
    id: int
    winner: Winner
    current_risk: RiskLabel
    candidate_risk: RiskLabel
    tier_match_winner: Winner
    rationale: str


@dataclass(frozen=True)
class DeploymentGateResult:
    samples_path: Path
    reviews_path: Path
    report_path: Path
    comparison_count: int
    reviewed_count: int


@dataclass(frozen=True)
class _Rollup:
    slot_type: str
    filler: str
    freq: int
    current_popularity_score: float
    current_tier_score: float
    book_model_score: float


def run_deployment_gate_review(
    *,
    rollup_paths: dict[str, Path],
    output_dir: Path,
    sample_count: int = 24,
    random_seed: int = 20260505,
    model: str = DEFAULT_RATER_MODEL,
    dry_run: bool = False,
    reviewer: StrategyReviewFn | None = None,
) -> DeploymentGateResult:
    """Generate strategy samples and optionally review them with an LLM."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = tuple(_build_comparisons(
        rollup_paths=rollup_paths,
        sample_count=sample_count,
        random_seed=random_seed,
    ))
    reviews = () if dry_run else (reviewer or _review_with_llm)(comparisons, model)
    if reviews:
        _assert_review_ids_match(comparisons, reviews)
    samples_path = output_dir / "deployment_gate_samples.csv"
    reviews_path = output_dir / "deployment_gate_reviews.csv"
    report_path = output_dir / "deployment_gate_report.md"
    _write_samples(samples_path, comparisons)
    _write_reviews(reviews_path, comparisons, reviews)
    report_path.write_text(
        format_deployment_gate_report(
            comparisons=comparisons,
            reviews=reviews,
            samples_path=samples_path,
            reviews_path=reviews_path,
            dry_run=dry_run,
        ),
        encoding="utf-8",
    )
    return DeploymentGateResult(
        samples_path=samples_path,
        reviews_path=reviews_path,
        report_path=report_path,
        comparison_count=len(comparisons),
        reviewed_count=len(reviews),
    )


def format_deployment_gate_report(
    *,
    comparisons: tuple[StrategyComparison, ...],
    reviews: tuple[StrategyReview, ...],
    samples_path: Path,
    reviews_path: Path,
    dry_run: bool,
) -> str:
    lines = [
        "# Book-model deployment gate",
        "",
        "This report compares current popularity sampling with candidate "
        "model/blend strategies using fixed seeds. It is offline-only.",
        "",
        "## Outputs",
        "",
        f"- Samples: `{samples_path}` ({len(comparisons):,} comparisons)",
        f"- Reviews: `{reviews_path}` ({len(reviews):,} reviewed)",
        "",
    ]
    if dry_run:
        lines.extend([
            "## Status",
            "",
            "- Dry run only; no LLM review was requested.",
        ])
        return "\n".join(lines)

    review_by_comparison = {review.id: review for review in reviews}
    by_strategy: dict[tuple[str, str], list[StrategyReview]] = defaultdict(list)
    for comparison in comparisons:
        review = review_by_comparison.get(comparison.id)
        if review:
            by_strategy[(comparison.student, comparison.strategy)].append(review)
    lines.extend([
        "## Review summary",
        "",
        "| Student | Strategy | Candidate win | Tie | Current win | Candidate nonsensical | Current nonsensical |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for (student, strategy), strategy_reviews in sorted(by_strategy.items()):
        counts = Counter(review.winner for review in strategy_reviews)
        candidate_bad = sum(
            review.candidate_risk == "nonsensical" for review in strategy_reviews
        )
        current_bad = sum(
            review.current_risk == "nonsensical" for review in strategy_reviews
        )
        total = max(1, len(strategy_reviews))
        lines.append(
            f"| {student} | {strategy} | "
            f"{counts.get('candidate', 0) / total:.1%} | "
            f"{counts.get('tie', 0) / total:.1%} | "
            f"{counts.get('current', 0) / total:.1%} | "
            f"{candidate_bad / total:.1%} | {current_bad / total:.1%} |"
        )
    lines.extend(["", "## Representative candidate wins", ""])
    for comparison in comparisons:
        review = review_by_comparison.get(comparison.id)
        if not review or review.winner != "candidate":
            continue
        lines.append(f"- `{comparison.student}` / `{comparison.strategy}` seed `{comparison.seed}`")
        lines.append(f"  - current: {comparison.current_subtitle}")
        lines.append(f"  - candidate: {comparison.candidate_subtitle}")
        lines.append(f"  - rationale: {review.rationale}")
        if len([line for line in lines if line.startswith("- `")]) >= 8:
            break
    lines.extend(["", "## Recommendation", "", _recommendation(comparisons, reviews)])
    return "\n".join(lines)


def build_review_prompt(comparisons: tuple[StrategyComparison, ...]) -> str:
    rendered = "\n".join(
        f"{comparison.id}. student={comparison.student}; strategy={comparison.strategy}; "
        f"current={comparison.current_subtitle!r}; candidate={comparison.candidate_subtitle!r}"
        for comparison in comparisons
    )
    return f"""Compare generated book-subtitle candidates.

For each id, compare current scoring against the candidate strategy.
Do not punish a subtitle merely for being funny, surprising, niche, or odd.
Prefer the subtitle that is more coherent, more evocative, and more plausibly marketable as a bizarre nonfiction book subtitle.

Return:
- winner: current, candidate, or tie
- current_risk and candidate_risk: acceptable, intriguing_or_funny, or nonsensical
- tier_match_winner: current, candidate, or tie for which better fits a pop/mainstream/niche subtitle feel
- short rationale

Comparisons:
{rendered}
"""


def _build_comparisons(
    *,
    rollup_paths: dict[str, Path],
    sample_count: int,
    random_seed: int,
) -> list[StrategyComparison]:
    comparisons: list[StrategyComparison] = []
    next_id = 1
    for student, path in rollup_paths.items():
        rollups = _read_rollups(path)
        by_type: dict[str, list[_Rollup]] = defaultdict(list)
        for row in rollups:
            by_type[row.slot_type].append(row)
        for offset in range(sample_count):
            seed = random_seed + offset
            current = _sample_subtitle(
                by_type,
                random.Random(seed),
                current_weight=1.0,
            )
            for strategy, current_weight in STRATEGIES:
                candidate = _sample_subtitle(
                    by_type,
                    random.Random(seed),
                    current_weight=current_weight,
                )
                comparisons.append(StrategyComparison(
                    id=next_id,
                    seed=seed,
                    student=student,
                    strategy=strategy,
                    current_subtitle=current,
                    candidate_subtitle=candidate,
                ))
                next_id += 1
    return comparisons


def _sample_subtitle(
    by_type: dict[str, list[_Rollup]],
    rng: random.Random,
    *,
    current_weight: float,
) -> str:
    item_rows = _weighted_rows(by_type["list_item"], 2, rng, current_weight=current_weight)
    action = _weighted_rows(by_type["action_noun"], 1, rng, current_weight=current_weight)[0]
    obj = _weighted_rows(by_type["of_object"], 1, rng, current_weight=current_weight)[0]
    return titlecase(
        f"{item_rows[0].filler}, {item_rows[1].filler}, and the "
        f"{action.filler} of {obj.filler}"
    )


def _weighted_rows(
    rows: list[_Rollup],
    count: int,
    rng: random.Random,
    *,
    current_weight: float,
) -> list[_Rollup]:
    pool = list(rows)
    chosen: list[_Rollup] = []
    for _ in range(count):
        weights = [
            max(0.001, _blend_score(row, current_weight)) * max(1.0, row.freq ** 0.5)
            for row in pool
        ]
        pick = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def _blend_score(row: _Rollup, current_weight: float) -> float:
    if current_weight >= 1.0:
        return row.current_popularity_score
    return (
        current_weight * row.current_tier_score
        + (1 - current_weight) * row.book_model_score
    )


def _review_with_llm(
    comparisons: tuple[StrategyComparison, ...],
    model: str,
) -> tuple[StrategyReview, ...]:
    try:
        from pydantic import BaseModel
        from subtitle_generator.eval_harness import structured_completion
    except ImportError as exc:
        raise RuntimeError(
            "Deployment-gate LLM review requires optional tune dependencies. "
            "Run `uv sync --extra tune` first."
        ) from exc

    class _ReviewModel(BaseModel):
        id: int
        winner: Literal["current", "candidate", "tie"]
        current_risk: Literal["nonsensical", "intriguing_or_funny", "acceptable"]
        candidate_risk: Literal["nonsensical", "intriguing_or_funny", "acceptable"]
        tier_match_winner: Literal["current", "candidate", "tie"]
        rationale: str

    class _ReviewBatch(BaseModel):
        reviews: list[_ReviewModel]

    reviews: list[StrategyReview] = []
    for start in range(0, len(comparisons), 20):
        batch = comparisons[start:start + 20]
        result = structured_completion(
            model=model,
            messages=[{"role": "user", "content": build_review_prompt(batch)}],
            schema=_ReviewBatch,
            temperature=0,
            max_tokens=4096,
            timeout=120,
        )
        reviews.extend(
            StrategyReview(
                id=review.id,
                winner=review.winner,
                current_risk=review.current_risk,
                candidate_risk=review.candidate_risk,
                tier_match_winner=review.tier_match_winner,
                rationale=" ".join(review.rationale.split()),
            )
            for review in result.reviews
        )
    output = tuple(reviews)
    _assert_review_ids_match(comparisons, output)
    return output


def _read_rollups(path: Path) -> tuple[_Rollup, ...]:
    with open(path, newline="", encoding="utf-8") as handle:
        return tuple(
            _Rollup(
                slot_type=row["slot_type"],
                filler=row["filler"],
                freq=int(row["freq"]),
                current_popularity_score=float(row["current_popularity_score"]),
                current_tier_score=float(row["current_tier_score"]),
                book_model_score=float(row["book_model_score"]),
            )
            for row in csv.DictReader(handle)
        )


def _write_samples(path: Path, comparisons: tuple[StrategyComparison, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({
                "id": comparison.id,
                "seed": comparison.seed,
                "student": comparison.student,
                "strategy": comparison.strategy,
                "current_subtitle": comparison.current_subtitle,
                "candidate_subtitle": comparison.candidate_subtitle,
            })


def _write_reviews(
    path: Path,
    comparisons: tuple[StrategyComparison, ...],
    reviews: tuple[StrategyReview, ...],
) -> None:
    comparison_by_id = {comparison.id: comparison for comparison in comparisons}
    review_by_id = {review.id: review for review in reviews}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for comparison in comparisons:
            review = review_by_id.get(comparison.id)
            row = {
                "id": comparison.id,
                "seed": comparison.seed,
                "student": comparison.student,
                "strategy": comparison.strategy,
                "current_subtitle": comparison.current_subtitle,
                "candidate_subtitle": comparison.candidate_subtitle,
                "winner": "",
                "current_risk": "",
                "candidate_risk": "",
                "tier_match_winner": "",
                "rationale": "",
            }
            if review:
                row.update({
                    "winner": review.winner,
                    "current_risk": review.current_risk,
                    "candidate_risk": review.candidate_risk,
                    "tier_match_winner": review.tier_match_winner,
                    "rationale": review.rationale,
                })
            if comparison.id not in comparison_by_id:
                raise RuntimeError(f"Unknown comparison id {comparison.id}")
            writer.writerow(row)


def _assert_review_ids_match(
    comparisons: tuple[StrategyComparison, ...],
    reviews: tuple[StrategyReview, ...],
) -> None:
    expected = Counter(comparison.id for comparison in comparisons)
    actual = Counter(review.id for review in reviews)
    if actual != expected:
        raise RuntimeError(
            "Deployment-gate review did not match requested ids: "
            f"expected {dict(sorted(expected.items()))}, got {dict(sorted(actual.items()))}"
        )


def _recommendation(
    comparisons: tuple[StrategyComparison, ...],
    reviews: tuple[StrategyReview, ...],
) -> str:
    if not reviews:
        return "- No reviewed rows are available."
    comparison_by_id = {comparison.id: comparison for comparison in comparisons}
    by_strategy: dict[tuple[str, str], list[StrategyReview]] = defaultdict(list)
    for review in reviews:
        comparison = comparison_by_id.get(review.id)
        if comparison:
            by_strategy[(comparison.student, comparison.strategy)].append(review)
    eligible: list[tuple[str, str, float, float, float]] = []
    for (student, strategy), strategy_reviews in by_strategy.items():
        total = len(strategy_reviews)
        if not total:
            continue
        candidate_win_rate = sum(review.winner == "candidate" for review in strategy_reviews) / total
        candidate_bad = sum(review.candidate_risk == "nonsensical" for review in strategy_reviews) / total
        current_bad = sum(review.current_risk == "nonsensical" for review in strategy_reviews) / total
        if (
            candidate_win_rate >= 0.60
            and candidate_bad <= 0.20
            and candidate_bad <= current_bad + 0.05
        ):
            eligible.append((student, strategy, candidate_win_rate, candidate_bad, current_bad))
    if eligible:
        student, strategy, win_rate, candidate_bad, current_bad = max(
            eligible,
            key=lambda item: (item[2], -item[3]),
        )
        return (
            f"- `{student}` / `{strategy}` is the only deployment-gate pass: "
            f"candidate win rate {win_rate:.1%}, candidate nonsensical rate "
            f"{candidate_bad:.1%}, current nonsensical rate {current_bad:.1%}. "
            "Use it only for a flagged blend/rerank integration experiment; "
            "do not ship full replacement without runtime validation."
        )
    return (
        "- Candidate strategies are not clearly safer/better than current. Keep "
        "diagnostics or run a narrower blend/rerank experiment before runtime integration."
    )
