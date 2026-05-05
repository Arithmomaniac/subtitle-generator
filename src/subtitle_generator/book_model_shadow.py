"""Shadow rollups from book predictions onto slot fillers."""

from __future__ import annotations

import csv
import random
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from titlecase import titlecase

from subtitle_generator.book_model_baseline import TIERS

ROLLUP_COLUMNS = (
    "slot_filler_id",
    "slot_type",
    "filler",
    "freq",
    "current_popularity_score",
    "current_popularity_level",
    "current_tier_score",
    "source_prediction_count",
    "avg_score_pop",
    "avg_score_mainstream",
    "avg_score_niche",
    "book_model_score",
    "book_model_tier",
    "score_delta",
    "tier_changed",
)
_TIER_VALUE = {"pop": 1.0, "mainstream": 0.55, "niche": 0.1}
_TIER_THRESHOLDS = (("pop", 0.72), ("mainstream", 0.34), ("niche", 0.0))


@dataclass(frozen=True)
class ShadowInput:
    label: str
    predictions_path: Path


@dataclass(frozen=True)
class ShadowResult:
    report_path: Path
    rollup_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _RollupRow:
    slot_filler_id: int
    slot_type: str
    filler: str
    freq: int
    current_popularity_score: float
    current_popularity_level: str
    current_tier_score: float
    source_prediction_count: int
    avg_score_pop: float
    avg_score_mainstream: float
    avg_score_niche: float
    book_model_score: float
    book_model_tier: str
    score_delta: float
    tier_changed: int


def build_shadow_rollups(
    conn: sqlite3.Connection,
    *,
    output_dir: Path,
    prediction_inputs: tuple[ShadowInput, ...],
    sample_count: int = 12,
    random_seed: int = 20260505,
) -> ShadowResult:
    """Build filler rollups and a shadow comparison report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rollups_by_label: dict[str, tuple[_RollupRow, ...]] = {}
    rollup_paths: list[Path] = []
    for prediction_input in prediction_inputs:
        predictions = _read_predictions_by_subtitle(prediction_input.predictions_path)
        rollups = tuple(_load_rollups(conn, predictions))
        rollups_by_label[prediction_input.label] = rollups
        safe_label = _safe_label(prediction_input.label)
        path = output_dir / f"filler_book_rollups_{safe_label}.csv"
        _write_rollups(path, rollups)
        rollup_paths.append(path)
    report = format_shadow_report(
        rollups_by_label,
        sample_count=sample_count,
        random_seed=random_seed,
    )
    report_path = output_dir / "book_shadow_rollup_report.md"
    report_path.write_text(report, encoding="utf-8")
    return ShadowResult(report_path=report_path, rollup_paths=tuple(rollup_paths))


def format_shadow_report(
    rollups_by_label: dict[str, tuple[_RollupRow, ...]],
    *,
    sample_count: int,
    random_seed: int,
) -> str:
    lines = [
        "# Book-model shadow rollup report",
        "",
        "This report aggregates source-book predictions onto strict slot fillers "
        "and compares model-weighted sampling against current popularity scoring. "
        "It is offline-only and does not change generation defaults.",
        "",
        "## Rollup summary",
        "",
        "| Student | Rollups | With predictions | Tier changes | Pearson current/model | Model tiers |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for label, rollups in rollups_by_label.items():
        predicted = sum(row.source_prediction_count > 0 for row in rollups)
        tier_changes = sum(row.tier_changed for row in rollups)
        corr = _pearson(
            [row.current_tier_score for row in rollups],
            [row.book_model_score for row in rollups],
        )
        tiers = Counter(row.book_model_tier for row in rollups)
        lines.append(
            f"| {label} | {len(rollups):,} | {predicted:,} | {tier_changes:,} | "
        f"{corr:.3f} | {_format_counts(tiers)} |"
        )
    lines.extend(["", "## Largest filler score changes", ""])
    for label, rollups in rollups_by_label.items():
        lines.extend([f"### {label}", ""])
        for row in _largest_changes(rollups, limit=12):
            lines.append(
                f"- `{row.slot_type}` **{row.filler}**: "
                f"{row.current_popularity_level}({row.current_tier_score:.2f}) -> "
                f"{row.book_model_score:.3f}/{row.book_model_tier} "
                f"(sources={row.source_prediction_count}, delta={row.score_delta:+.3f})"
            )
        lines.append("")
    lines.extend([
        "## Fixed-seed shadow samples",
        "",
        "Each pair uses the same seed and candidate pool. `current` weights by "
        "`slot_fillers.popularity_score`; `model` weights by the rollup score.",
        "",
    ])
    for label, rollups in rollups_by_label.items():
        lines.extend([f"### {label}", ""])
        samples = _shadow_samples(
            rollups,
            sample_count=sample_count,
            random_seed=random_seed,
        )
        for sample in samples:
            lines.append(f"- Seed `{sample['seed']}`")
            lines.append(f"  - current: {sample['current']}")
            lines.append(f"  - model: {sample['model']}")
        lines.append("")
    lines.extend([
        "## Gate notes",
        "",
        "- Raw `slot_fillers.popularity_score` is still used for current-sampling "
        "weights, but tier deltas/correlation compare against the current "
        "popularity level mapped onto the model tier-value scale.",
        "- Rollups with no source predictions fall back to the current popularity tier.",
        "- This report decides what to inspect next; deployment still requires a "
        "separate integration decision and browser/API validation if runtime changes are made.",
    ])
    return "\n".join(lines)


def _load_rollups(
    conn: sqlite3.Connection,
    predictions_by_subtitle: dict[int, dict[str, float]],
) -> list[_RollupRow]:
    rows = conn.execute(
        """
        SELECT
            sf.id,
            sf.slot_type,
            sf.filler,
            COALESCE(sf.freq, 0),
            COALESCE(sf.popularity_score, 0.1),
            COALESCE(sf.popularity_level, ''),
            sfs.subtitle_id
        FROM slot_fillers sf
        LEFT JOIN slot_filler_sources sfs ON sfs.slot_filler_id = sf.id
        WHERE COALESCE(sf.mode, 'strict') = 'strict'
        ORDER BY sf.id
        """
    ).fetchall()
    grouped: dict[int, dict] = {}
    for row in rows:
        filler_id = int(row[0])
        entry = grouped.setdefault(
            filler_id,
            {
                "slot_filler_id": filler_id,
                "slot_type": row[1] or "",
                "filler": row[2] or "",
                "freq": int(row[3] or 0),
                "current_popularity_score": float(row[4] or 0.1),
                "current_popularity_level": row[5] or "",
                "scores": [],
            },
        )
        subtitle_id = row[6]
        if subtitle_id is not None and int(subtitle_id) in predictions_by_subtitle:
            entry["scores"].append(predictions_by_subtitle[int(subtitle_id)])
    rollups: list[_RollupRow] = []
    for entry in grouped.values():
        scores = entry["scores"]
        if scores:
            avg_scores = {
                tier: sum(score[tier] for score in scores) / len(scores)
                for tier in TIERS
            }
            model_score = sum(avg_scores[tier] * _TIER_VALUE[tier] for tier in TIERS)
            model_tier = max(avg_scores, key=avg_scores.get)
        else:
            model_score = entry["current_popularity_score"]
            model_tier = _tier_from_score(model_score)
            avg_scores = {
                "pop": 1.0 if model_tier == "pop" else 0.0,
                "mainstream": 1.0 if model_tier == "mainstream" else 0.0,
                "niche": 1.0 if model_tier == "niche" else 0.0,
            }
        current_tier = _normalize_current_tier(
            entry["current_popularity_level"],
            entry["current_popularity_score"],
        )
        current_tier_score = _TIER_VALUE.get(current_tier, _TIER_VALUE["niche"])
        rollups.append(_RollupRow(
            slot_filler_id=entry["slot_filler_id"],
            slot_type=entry["slot_type"],
            filler=entry["filler"],
            freq=entry["freq"],
            current_popularity_score=entry["current_popularity_score"],
            current_popularity_level=current_tier,
            current_tier_score=current_tier_score,
            source_prediction_count=len(scores),
            avg_score_pop=avg_scores["pop"],
            avg_score_mainstream=avg_scores["mainstream"],
            avg_score_niche=avg_scores["niche"],
            book_model_score=model_score,
            book_model_tier=model_tier,
            score_delta=model_score - current_tier_score,
            tier_changed=int(model_tier != current_tier),
        ))
    return rollups


def _read_predictions_by_subtitle(path: Path) -> dict[int, dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        predictions: dict[int, dict[str, float]] = {}
        for row in csv.DictReader(handle):
            subtitle_id = int(row["subtitle_id"])
            predictions[subtitle_id] = {
                "pop": float(row["score_pop"]),
                "mainstream": float(row["score_mainstream"]),
                "niche": float(row["score_niche"]),
            }
    return predictions


def _write_rollups(path: Path, rollups: tuple[_RollupRow, ...]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROLLUP_COLUMNS)
        writer.writeheader()
        for row in rollups:
            writer.writerow({
                "slot_filler_id": row.slot_filler_id,
                "slot_type": row.slot_type,
                "filler": row.filler,
                "freq": row.freq,
                "current_popularity_score": f"{row.current_popularity_score:.6f}",
                "current_popularity_level": row.current_popularity_level,
                "current_tier_score": f"{row.current_tier_score:.6f}",
                "source_prediction_count": row.source_prediction_count,
                "avg_score_pop": f"{row.avg_score_pop:.6f}",
                "avg_score_mainstream": f"{row.avg_score_mainstream:.6f}",
                "avg_score_niche": f"{row.avg_score_niche:.6f}",
                "book_model_score": f"{row.book_model_score:.6f}",
                "book_model_tier": row.book_model_tier,
                "score_delta": f"{row.score_delta:.6f}",
                "tier_changed": row.tier_changed,
            })


def _shadow_samples(
    rollups: tuple[_RollupRow, ...],
    *,
    sample_count: int,
    random_seed: int,
) -> list[dict[str, str]]:
    by_type: dict[str, list[_RollupRow]] = defaultdict(list)
    for row in rollups:
        by_type[row.slot_type].append(row)
    samples: list[dict[str, str]] = []
    for offset in range(sample_count):
        seed = random_seed + offset
        current_rng = random.Random(seed)
        model_rng = random.Random(seed)
        current = _sample_subtitle(by_type, current_rng, score_attr="current_popularity_score")
        model = _sample_subtitle(by_type, model_rng, score_attr="book_model_score")
        samples.append({"seed": str(seed), "current": current, "model": model})
    return samples


def _sample_subtitle(
    by_type: dict[str, list[_RollupRow]],
    rng: random.Random,
    *,
    score_attr: str,
) -> str:
    item_rows = _weighted_rows(by_type["list_item"], 2, rng, score_attr=score_attr)
    action = _weighted_rows(by_type["action_noun"], 1, rng, score_attr=score_attr)[0]
    obj = _weighted_rows(by_type["of_object"], 1, rng, score_attr=score_attr)[0]
    return titlecase(
        f"{item_rows[0].filler}, {item_rows[1].filler}, and the "
        f"{action.filler} of {obj.filler}"
    )


def _weighted_rows(
    rows: list[_RollupRow],
    count: int,
    rng: random.Random,
    *,
    score_attr: str,
) -> list[_RollupRow]:
    pool = list(rows)
    chosen: list[_RollupRow] = []
    for _ in range(count):
        weights = [
            max(0.001, float(getattr(row, score_attr))) * max(1.0, row.freq ** 0.5)
            for row in pool
        ]
        pick = rng.choices(pool, weights=weights, k=1)[0]
        chosen.append(pick)
        pool.remove(pick)
    return chosen


def _largest_changes(
    rollups: tuple[_RollupRow, ...],
    *,
    limit: int,
) -> list[_RollupRow]:
    predicted = [row for row in rollups if row.source_prediction_count > 0]
    return sorted(predicted, key=lambda row: abs(row.score_delta), reverse=True)[:limit]


def _tier_from_score(score: float) -> str:
    for tier, threshold in _TIER_THRESHOLDS:
        if score >= threshold:
            return tier
    return "niche"


def _normalize_current_tier(level: str, score: float) -> str:
    if level in TIERS:
        return level
    return _tier_from_score(score)


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_denom = sum((a - left_mean) ** 2 for a in left) ** 0.5
    right_denom = sum((b - right_mean) ** 2 for b in right) ** 0.5
    denom = left_denom * right_denom
    return numerator / denom if denom else 0.0


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{tier}={counts.get(tier, 0):,}" for tier in TIERS)


def _safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
