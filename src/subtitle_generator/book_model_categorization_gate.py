"""Pure-categorization gate for learned book-model tier probabilities."""

from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, cast

from titlecase import titlecase

from subtitle_generator.generate import _is_literal_bad_filler, _weighted_sample
from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL

TIERS = ("pop", "mainstream", "niche")
RiskLabel = Literal["literal_bad", "good_absurd", "weak_but_allowed", "acceptable"]
Winner = Literal["current", "candidate", "tie"]
CategorizationReviewFn = Callable[
    [tuple["CategorizationComparison", ...], str],
    tuple["CategorizationReview", ...],
]

SAMPLE_COLUMNS = (
    "id",
    "seed",
    "student",
    "requested_tier",
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


@dataclass(frozen=True)
class CategorizationComparison:
    id: int
    seed: int
    student: str
    requested_tier: str
    current_subtitle: str
    candidate_subtitle: str


@dataclass(frozen=True)
class CategorizationReview:
    id: int
    winner: Winner
    current_risk: RiskLabel
    candidate_risk: RiskLabel
    tier_match_winner: Winner
    rationale: str


@dataclass(frozen=True)
class CategorizationGateResult:
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
    avg_score_pop: float
    avg_score_mainstream: float
    avg_score_niche: float


def run_categorization_gate_review(
    *,
    rollup_paths: dict[str, Path],
    output_dir: Path,
    samples_per_tier: int = 12,
    random_seed: int = 20260505,
    model: str = DEFAULT_RATER_MODEL,
    dry_run: bool = False,
    reviewer: CategorizationReviewFn | None = None,
) -> CategorizationGateResult:
    """Generate pure-categorization samples and optionally review them with an LLM."""

    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = tuple(_build_comparisons(
        rollup_paths=rollup_paths,
        samples_per_tier=samples_per_tier,
        random_seed=random_seed,
    ))
    reviews = () if dry_run else (reviewer or _review_with_llm)(comparisons, model)
    if reviews:
        _assert_review_ids_match(comparisons, reviews)
    samples_path = output_dir / "categorization_gate_samples.csv"
    reviews_path = output_dir / "categorization_gate_reviews.csv"
    report_path = output_dir / "categorization_gate_report.md"
    _write_samples(samples_path, comparisons)
    _write_reviews(reviews_path, comparisons, reviews)
    report_path.write_text(
        format_categorization_gate_report(
            comparisons=comparisons,
            reviews=reviews,
            samples_path=samples_path,
            reviews_path=reviews_path,
            dry_run=dry_run,
        ),
        encoding="utf-8",
    )
    return CategorizationGateResult(
        samples_path=samples_path,
        reviews_path=reviews_path,
        report_path=report_path,
        comparison_count=len(comparisons),
        reviewed_count=len(reviews),
    )


def format_categorization_gate_report(
    *,
    comparisons: tuple[CategorizationComparison, ...],
    reviews: tuple[CategorizationReview, ...],
    samples_path: Path,
    reviews_path: Path,
    dry_run: bool,
) -> str:
    lines = [
        "# Book-model pure categorization gate",
        "",
        "This report compares frequency-only sampling against learned tier "
        "categorization using fixed seeds. The candidate path weights existing "
        "fillers by the requested tier probability; it does not introduce new "
        "filler text.",
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
    by_student_tier: dict[tuple[str, str], list[CategorizationReview]] = defaultdict(list)
    for comparison in comparisons:
        review = review_by_comparison.get(comparison.id)
        if review:
            by_student_tier[(comparison.student, comparison.requested_tier)].append(review)

    lines.extend([
        "## Review summary",
        "",
        "| Student | Requested tier | Candidate win | Tie | Current win | Candidate literal-bad | Current literal-bad |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for (student, tier), tier_reviews in sorted(by_student_tier.items()):
        counts = Counter(review.winner for review in tier_reviews)
        candidate_bad = sum(review.candidate_risk == "literal_bad" for review in tier_reviews)
        current_bad = sum(review.current_risk == "literal_bad" for review in tier_reviews)
        total = max(1, len(tier_reviews))
        lines.append(
            f"| {student} | {tier} | "
            f"{counts.get('candidate', 0) / total:.1%} | "
            f"{counts.get('tie', 0) / total:.1%} | "
            f"{counts.get('current', 0) / total:.1%} | "
            f"{candidate_bad / total:.1%} | {current_bad / total:.1%} |"
        )

    lines.extend(["", "## Representative candidate wins", ""])
    win_count = 0
    comparison_by_id = {comparison.id: comparison for comparison in comparisons}
    for review in reviews:
        if review.winner != "candidate":
            continue
        comparison = comparison_by_id.get(review.id)
        if not comparison:
            continue
        lines.append(
            f"- `{comparison.student}` / `{comparison.requested_tier}` seed `{comparison.seed}`"
        )
        lines.append(f"  - current: {comparison.current_subtitle}")
        lines.append(f"  - candidate: {comparison.candidate_subtitle}")
        lines.append(f"  - rationale: {review.rationale}")
        win_count += 1
        if win_count >= 8:
            break
    lines.extend(["", "## Recommendation", "", _recommendation(comparisons, reviews)])
    return "\n".join(lines)


def build_review_prompt(comparisons: tuple[CategorizationComparison, ...]) -> str:
    rendered = "\n".join(
        f"{comparison.id}. requested_tier={comparison.requested_tier}; "
        f"student={comparison.student}; current={comparison.current_subtitle!r}; "
        f"candidate={comparison.candidate_subtitle!r}"
        for comparison in comparisons
    )
    return f"""Compare generated book-subtitle candidates for learned tier categorization.

For each id, the requested_tier is the target market tier. The candidate subtitle
was sampled with learned per-filler probabilities for that tier. The current
subtitle was sampled with frequency-only weighting.

Do not punish a subtitle merely for being funny, surprising, niche, or oddly
specific. Bizarre-but-grammatical nonfiction energy is desired. Mark literal_bad
only for broken syntax, typo/acronym artifacts, malformed final objects, or text
that cannot work as "the X of Y".

Return:
- winner: current, candidate, or tie
- current_risk and candidate_risk: literal_bad, good_absurd, weak_but_allowed, or acceptable
- tier_match_winner: current, candidate, or tie for which better fits requested_tier
- short rationale

Comparisons:
{rendered}
"""


def _build_comparisons(
    *,
    rollup_paths: dict[str, Path],
    samples_per_tier: int,
    random_seed: int,
) -> list[CategorizationComparison]:
    comparisons: list[CategorizationComparison] = []
    next_id = 1
    for student, path in rollup_paths.items():
        rollups = _read_rollups(path)
        by_type: dict[str, list[_Rollup]] = defaultdict(list)
        for row in rollups:
            if not _is_literal_bad_filler(row.slot_type, row.filler):
                by_type[row.slot_type].append(row)
        _assert_required_slots(by_type, student)
        for tier_index, tier in enumerate(TIERS):
            for offset in range(samples_per_tier):
                seed = random_seed + (tier_index * samples_per_tier) + offset
                current = _sample_subtitle(by_type, random.Random(seed), tier, candidate=False)
                candidate = _sample_subtitle(by_type, random.Random(seed), tier, candidate=True)
                comparisons.append(CategorizationComparison(
                    id=next_id,
                    seed=seed,
                    student=student,
                    requested_tier=tier,
                    current_subtitle=current,
                    candidate_subtitle=candidate,
                ))
                next_id += 1
    return comparisons


def _sample_subtitle(
    by_type: dict[str, list[_Rollup]],
    rng: random.Random,
    requested_tier: str,
    *,
    candidate: bool,
) -> str:
    items = _sample_slot(by_type["list_item"], 2, rng, requested_tier, candidate=candidate)
    action = _sample_slot(by_type["action_noun"], 1, rng, requested_tier, candidate=candidate)[0]
    obj = _sample_slot(by_type["of_object"], 1, rng, requested_tier, candidate=candidate)[0]
    return titlecase(f"{items[0]}, {items[1]}, and the {action} of {obj}")


def _sample_slot(
    rows: list[_Rollup],
    count: int,
    rng: random.Random,
    requested_tier: str,
    *,
    candidate: bool,
) -> list[str]:
    if candidate:
        weighted_rows = [
            (
                row.filler,
                row.freq,
                row.current_popularity_score,
                row.avg_score_pop,
                row.avg_score_mainstream,
                row.avg_score_niche,
            )
            for row in rows
        ]
        return _weighted_sample(weighted_rows, count, rng, model_tier=requested_tier)

    weighted_rows = [
        (row.filler, row.freq, row.current_popularity_score)
        for row in rows
    ]
    return _weighted_sample(weighted_rows, count, rng)


def _review_with_llm(
    comparisons: tuple[CategorizationComparison, ...],
    model: str,
) -> tuple[CategorizationReview, ...]:
    try:
        from pydantic import BaseModel
        from subtitle_generator.eval_harness import structured_completion
    except ImportError as exc:
        raise RuntimeError(
            "Categorization-gate LLM review requires optional tune dependencies. "
            "Run `uv sync --extra tune` first."
        ) from exc

    class _ReviewModel(BaseModel):
        id: int
        winner: Literal["current", "candidate", "tie"]
        current_risk: Literal[
            "literal_bad", "good_absurd", "weak_but_allowed", "acceptable"
        ]
        candidate_risk: Literal[
            "literal_bad", "good_absurd", "weak_but_allowed", "acceptable"
        ]
        tier_match_winner: Literal["current", "candidate", "tie"]
        rationale: str

    class _ReviewBatch(BaseModel):
        reviews: list[_ReviewModel]

    reviews: list[CategorizationReview] = []
    for start in range(0, len(comparisons), 20):
        batch = comparisons[start:start + 20]
        result = cast(_ReviewBatch, structured_completion(
            model=model,
            messages=[{"role": "user", "content": build_review_prompt(batch)}],
            schema=_ReviewBatch,
            temperature=0,
            max_tokens=4096,
            timeout=120,
        ))
        reviews.extend(
            CategorizationReview(
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
        reader = csv.DictReader(handle)
        required_columns = {
            "slot_type",
            "filler",
            "freq",
            "current_popularity_score",
            "avg_score_pop",
            "avg_score_mainstream",
            "avg_score_niche",
        }
        missing = required_columns - set(reader.fieldnames or ())
        if missing:
            raise RuntimeError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return tuple(
            _Rollup(
                slot_type=row["slot_type"],
                filler=row["filler"],
                freq=int(row["freq"]),
                current_popularity_score=float(row["current_popularity_score"]),
                avg_score_pop=float(row["avg_score_pop"]),
                avg_score_mainstream=float(row["avg_score_mainstream"]),
                avg_score_niche=float(row["avg_score_niche"]),
            )
            for row in reader
        )


def _write_samples(
    path: Path,
    comparisons: tuple[CategorizationComparison, ...],
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_COLUMNS)
        writer.writeheader()
        for comparison in comparisons:
            writer.writerow({
                "id": comparison.id,
                "seed": comparison.seed,
                "student": comparison.student,
                "requested_tier": comparison.requested_tier,
                "current_subtitle": comparison.current_subtitle,
                "candidate_subtitle": comparison.candidate_subtitle,
            })


def _write_reviews(
    path: Path,
    comparisons: tuple[CategorizationComparison, ...],
    reviews: tuple[CategorizationReview, ...],
) -> None:
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
                "requested_tier": comparison.requested_tier,
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
            writer.writerow(row)


def _assert_review_ids_match(
    comparisons: tuple[CategorizationComparison, ...],
    reviews: tuple[CategorizationReview, ...],
) -> None:
    expected = Counter(comparison.id for comparison in comparisons)
    actual = Counter(review.id for review in reviews)
    if actual != expected:
        raise RuntimeError(
            "Categorization-gate review did not match requested ids: "
            f"expected {dict(sorted(expected.items()))}, got {dict(sorted(actual.items()))}"
        )


def _assert_required_slots(
    by_type: dict[str, list[_Rollup]],
    student: str,
) -> None:
    missing = [
        slot_type for slot_type, count in (
            ("list_item", 2),
            ("action_noun", 1),
            ("of_object", 1),
        )
        if len(by_type.get(slot_type, ())) < count
    ]
    if missing:
        raise RuntimeError(
            f"{student} rollups do not contain enough rows for: {', '.join(missing)}"
        )


def _recommendation(
    comparisons: tuple[CategorizationComparison, ...],
    reviews: tuple[CategorizationReview, ...],
) -> str:
    if not reviews:
        return "- No reviewed rows are available."
    comparison_by_id = {comparison.id: comparison for comparison in comparisons}
    by_student: dict[str, list[CategorizationReview]] = defaultdict(list)
    for review in reviews:
        comparison = comparison_by_id.get(review.id)
        if comparison:
            by_student[comparison.student].append(review)

    eligible: list[tuple[str, float, float]] = []
    for student, student_reviews in by_student.items():
        total = len(student_reviews)
        candidate_win_rate = sum(
            review.winner == "candidate" for review in student_reviews
        ) / total
        candidate_bad = sum(
            review.candidate_risk == "literal_bad" for review in student_reviews
        ) / total
        if candidate_win_rate >= 0.55 and candidate_bad <= 0.10:
            eligible.append((student, candidate_win_rate, candidate_bad))
    if eligible:
        student, win_rate, bad_rate = max(eligible, key=lambda item: (item[1], -item[2]))
        return (
            f"- `{student}` passes the pure-categorization gate: candidate win "
            f"rate {win_rate:.1%}, candidate literal-bad rate {bad_rate:.1%}. "
            "Install its rollup only after export/build validation."
        )
    return (
        "- No student clearly passes the pure-categorization gate. Keep current "
        "runtime scores or collect more reviewed comparisons before changing deployment data."
    )
