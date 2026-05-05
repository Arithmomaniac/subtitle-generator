"""Interpretable baseline trainer for offline book-tier predictions."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

TIERS = ("pop", "mainstream", "niche")
_TOKEN_RE = re.compile(r"[a-z][a-z']+")
NUMERIC_FEATURES = (
    "candidate_source_is_title",
    "source_group_is_ol",
    "lang_is_eng",
    "has_isbn",
    "has_lccn",
    "has_work_key",
    "has_popularity_data",
    "work_popularity_score",
    "spl_checkouts",
    "spl_years",
    "ol_edition_count",
    "checkouts_per_year",
    "editions_per_decade",
    "gr_ratings_count",
    "gr_average_rating",
    "nyt_weeks_on_list",
    "library_appearances",
    "trove_library_count",
    "trove_holding_count",
    "trove_copy_count",
    "slot_source_link_count",
    "distinct_strict_filler_count",
    "max_filler_popularity_score",
    "avg_filler_popularity_score",
    "title_length_chars",
    "subtitle_length_chars",
    "list_item_count",
    "metadata_publisher_count",
    "metadata_publish_year",
    "metadata_number_of_pages",
    "metadata_is_hardcover",
    "metadata_is_paperback",
    "metadata_is_ebook",
    "metadata_is_large_print",
    "metadata_subject_count",
    "metadata_author_count",
    "has_physical_format_metadata",
    "has_publisher_metadata",
    "has_subject_metadata",
)
TEXT_FEATURE_COLUMNS = (
    "title",
    "subtitle_text",
    "list_items_text",
    "list_item_pair_text",
    "action_noun",
    "of_object",
    "action_object_pair_text",
    "slot_frame_text",
    "metadata_publisher_text",
    "metadata_edition_name",
    "metadata_physical_format",
    "metadata_physical_dimensions",
    "metadata_physical_description",
    "metadata_subject_text",
    "metadata_loc_call_number",
    "metadata_dewey_decimal",
)


@dataclass(frozen=True)
class TrainingResult:
    predictions_path: Path
    report_path: Path
    prediction_count: int
    labeled_count: int
    validation_count: int
    validation_accuracy: float
    validation_macro_accuracy: float


def train_book_tier_baseline(
    *,
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
) -> TrainingResult:
    """Train a nearest-centroid baseline and write predictions/report."""

    output_dir.mkdir(parents=True, exist_ok=True)
    features = _read_csv_by_id(features_path)
    labels = _read_csv_by_id(labels_path)
    labeled_ids = [
        pattern_id for pattern_id, row in labels.items()
        if row.get("label_target") in TIERS
    ]
    if not labeled_ids:
        raise RuntimeError("No labeled rows available for baseline training.")

    train_ids, validation_ids = _train_validation_split(labeled_ids, labels)
    if not validation_ids:
        validation_ids = train_ids
    scaler = _fit_scaler(features, train_ids)
    centroids = _fit_centroids(features, labels, train_ids, scaler)
    text_model = _fit_text_model(features, labels, train_ids)
    predictions = [
        _predict_row(
            pattern_id,
            features[pattern_id],
            labels.get(pattern_id, {}),
            scaler,
            centroids,
            text_model,
        )
        for pattern_id in sorted(features, key=int)
    ]
    metrics = _evaluate_predictions(predictions, validation_ids, labels)

    predictions_path = output_dir / "book_predictions.csv"
    report_path = output_dir / "book_baseline_report.md"
    _write_predictions(predictions_path, predictions)
    report_path.write_text(
        format_baseline_report(
            predictions=predictions,
            labels=labels,
            validation_ids=validation_ids,
            metrics=metrics,
            predictions_path=predictions_path,
        ),
        encoding="utf-8",
    )
    return TrainingResult(
        predictions_path=predictions_path,
        report_path=report_path,
        prediction_count=len(predictions),
        labeled_count=len(labeled_ids),
        validation_count=len(validation_ids),
        validation_accuracy=metrics["accuracy"],
        validation_macro_accuracy=metrics["macro_accuracy"],
    )

def format_baseline_report(
    *,
    predictions: list[dict[str, str]],
    labels: dict[str, dict[str, str]],
    validation_ids: list[str],
    metrics: dict,
    predictions_path: Path,
) -> str:
    label_counts = Counter(
        row.get("label_target", "")
        for row in labels.values()
        if row.get("label_target") in TIERS
    )
    predicted_counts = Counter(row["predicted_tier"] for row in predictions)
    lines = [
        "# Book-tier baseline report",
        "",
        "This is an interpretable text-plus-centroid baseline over local "
        "`book_features` and `book_labels`. It is a calibration artifact, not "
        "a runtime model.",
        "",
        "## Outputs",
        "",
        f"- Predictions: `{predictions_path}` ({len(predictions):,} rows)",
        "",
        "## Label balance",
        "",
        _format_counts(label_counts),
        "",
        "## Validation",
        "",
        f"- Validation rows: {len(validation_ids):,}",
        f"- Exact accuracy: {metrics['accuracy']:.3f}",
        f"- Macro accuracy: {metrics['macro_accuracy']:.3f}",
        "- Per-tier accuracy: " + _format_float_counts(metrics["per_tier_accuracy"]),
        "- Confusion: " + _format_confusion(metrics["confusion"]),
        "",
        "## Prediction distribution",
        "",
        _format_counts(predicted_counts),
        "",
        "## Gate notes",
        "",
    ]
    if label_counts.get("pop", 0) < 100:
        lines.append(
            "- Pop validation is still label-limited; treat pop recall/precision as "
            "directional until more pop labels exist or a stronger acquisition "
            "strategy is added."
        )
    lines.append(
        "- This baseline uses interpretable scalar/provenance features, offline "
        "raw LOC/Open Library metadata when supplied, and a smoothed token model "
        "over source text plus slot-interaction phrases; it does not yet use full "
        "embeddings or a learned exportable distillation model."
    )
    return "\n".join(lines)


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {
            row["pattern_match_id"]: row
            for row in csv.DictReader(handle)
        }
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _train_validation_split(
    labeled_ids: list[str],
    labels: dict[str, dict[str, str]],
) -> tuple[list[str], list[str]]:
    by_tier: dict[str, list[str]] = {tier: [] for tier in TIERS}
    for pattern_id in labeled_ids:
        by_tier[labels[pattern_id]["label_target"]].append(pattern_id)
    train: list[str] = []
    validation: list[str] = []
    for tier_ids in by_tier.values():
        tier_ids = sorted(tier_ids, key=int)
        for index, pattern_id in enumerate(tier_ids):
            if index % 5 == 0 and len(tier_ids) >= 5:
                validation.append(pattern_id)
            else:
                train.append(pattern_id)
    return train, validation


def _fit_scaler(
    features: dict[str, dict[str, str]],
    train_ids: list[str],
) -> dict[str, tuple[float, float]]:
    scaler: dict[str, tuple[float, float]] = {}
    for feature_name in NUMERIC_FEATURES:
        values = [
            _float(features[pattern_id].get(feature_name))
            for pattern_id in train_ids
        ]
        low = min(values)
        high = max(values)
        scaler[feature_name] = (low, high)
    return scaler


def _fit_centroids(
    features: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    train_ids: list[str],
    scaler: dict[str, tuple[float, float]],
) -> dict[str, list[float]]:
    vectors: dict[str, list[list[float]]] = {tier: [] for tier in TIERS}
    for pattern_id in train_ids:
        tier = labels[pattern_id]["label_target"]
        vectors[tier].append(_vectorize(features[pattern_id], scaler))
    centroids: dict[str, list[float]] = {}
    for tier, tier_vectors in vectors.items():
        if not tier_vectors:
            centroids[tier] = [0.0] * len(NUMERIC_FEATURES)
            continue
        centroids[tier] = [
            sum(vector[index] for vector in tier_vectors) / len(tier_vectors)
            for index in range(len(NUMERIC_FEATURES))
        ]
    return centroids


def _fit_text_model(
    features: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    train_ids: list[str],
) -> dict:
    token_counts: dict[str, Counter[str]] = {tier: Counter() for tier in TIERS}
    totals: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for pattern_id in train_ids:
        tier = labels[pattern_id]["label_target"]
        tokens = _tokens(features[pattern_id])
        token_counts[tier].update(tokens)
        totals[tier] += len(tokens)
        vocabulary.update(tokens)
    return {
        "token_counts": token_counts,
        "totals": totals,
        "vocabulary_size": max(1, len(vocabulary)),
    }


def _predict_row(
    pattern_id: str,
    feature_row: dict[str, str],
    label_row: dict[str, str],
    scaler: dict[str, tuple[float, float]],
    centroids: dict[str, list[float]],
    text_model: dict,
) -> dict[str, str]:
    vector = _vectorize(feature_row, scaler)
    distances = {
        tier: _euclidean_distance(vector, centroid)
        for tier, centroid in centroids.items()
    }
    scores = _combined_scores(feature_row, distances, text_model)
    predicted = max(scores, key=scores.get)
    return {
        "pattern_match_id": pattern_id,
        "subtitle_id": feature_row["subtitle_id"],
        "predicted_tier": predicted,
        "prediction_confidence": f"{scores[predicted]:.6f}",
        "score_pop": f"{scores['pop']:.6f}",
        "score_mainstream": f"{scores['mainstream']:.6f}",
        "score_niche": f"{scores['niche']:.6f}",
        "label_target": label_row.get("label_target", ""),
        "label_confidence": label_row.get("label_confidence", ""),
        "candidate_source": feature_row.get("candidate_source", ""),
        "source_group": feature_row.get("source_group", ""),
    }


def _vectorize(
    row: dict[str, str],
    scaler: dict[str, tuple[float, float]],
) -> list[float]:
    vector = []
    for feature_name in NUMERIC_FEATURES:
        value = _float(row.get(feature_name))
        low, high = scaler[feature_name]
        if high == low:
            vector.append(0.0)
        else:
            vector.append((value - low) / (high - low))
    return vector


def _evaluate_predictions(
    predictions: list[dict[str, str]],
    validation_ids: list[str],
    labels: dict[str, dict[str, str]],
) -> dict:
    prediction_by_id = {row["pattern_match_id"]: row for row in predictions}
    confusion: Counter[tuple[str, str]] = Counter()
    per_tier_total: Counter[str] = Counter()
    per_tier_correct: Counter[str] = Counter()
    correct = 0
    for pattern_id in validation_ids:
        expected = labels[pattern_id]["label_target"]
        predicted = prediction_by_id[pattern_id]["predicted_tier"]
        confusion[(expected, predicted)] += 1
        per_tier_total[expected] += 1
        if expected == predicted:
            correct += 1
            per_tier_correct[expected] += 1
    validation_count = max(1, len(validation_ids))
    per_tier_accuracy = {
        tier: (
            per_tier_correct[tier] / per_tier_total[tier]
            if per_tier_total[tier]
            else 0.0
        )
        for tier in TIERS
    }
    return {
        "accuracy": correct / validation_count,
        "macro_accuracy": sum(per_tier_accuracy.values()) / len(TIERS),
        "per_tier_accuracy": per_tier_accuracy,
        "confusion": dict(confusion),
    }


def _write_predictions(path: Path, predictions: list[dict[str, str]]) -> None:
    fieldnames = (
        "pattern_match_id",
        "subtitle_id",
        "predicted_tier",
        "prediction_confidence",
        "score_pop",
        "score_mainstream",
        "score_niche",
        "label_target",
        "label_confidence",
        "candidate_source",
        "source_group",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _distance_scores(distances: dict[str, float]) -> dict[str, float]:
    return _softmax({
        tier: -distance
        for tier, distance in distances.items()
    })


def _combined_scores(
    row: dict[str, str],
    distances: dict[str, float],
    text_model: dict,
) -> dict[str, float]:
    tokens = _tokens(row)
    token_counts: dict[str, Counter[str]] = text_model["token_counts"]
    totals: Counter[str] = text_model["totals"]
    vocabulary_size: int = text_model["vocabulary_size"]
    log_scores: dict[str, float] = {}
    for tier in TIERS:
        # Uniform class prior: the report should expose class separability, not
        # reproduce the current niche-heavy label distribution.
        log_score = math.log(1 / len(TIERS))
        denominator = totals[tier] + vocabulary_size
        for token in tokens:
            log_score += math.log((token_counts[tier][token] + 1) / denominator)
        log_score += -distances[tier]
        log_scores[tier] = log_score
    return _softmax(log_scores)


def _softmax(log_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(log_scores.values())
    raw = {
        tier: math.exp(score - max_score)
        for tier, score in log_scores.items()
    }
    total = sum(raw.values()) or 1.0
    return {tier: value / total for tier, value in raw.items()}


def _tokens(row: dict[str, str]) -> list[str]:
    text = " ".join(
        row.get(column, "")
        for column in TEXT_FEATURE_COLUMNS
    ).lower()
    tokens = _TOKEN_RE.findall(text)
    tokens.extend(_interaction_tokens(row))
    return tokens


def _interaction_tokens(row: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    for column in ("list_item_pair_text", "action_object_pair_text", "slot_frame_text"):
        parts = [
            _normalize_phrase(part)
            for part in row.get(column, "").split("||")
        ]
        parts = [part for part in parts if part]
        if len(parts) >= 2:
            tokens.append(f"{column}:{'__'.join(parts[:4])}")
    for column in ("metadata_physical_format", "metadata_loc_call_number"):
        phrase = _normalize_phrase(row.get(column, ""))
        if phrase:
            tokens.append(f"{column}:{phrase}")
    return tokens


def _normalize_phrase(value: str) -> str:
    words = _TOKEN_RE.findall((value or "").lower())
    return "_".join(words[:6])


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(
        f"{tier}={counts.get(tier, 0):,}"
        for tier in TIERS
    )


def _format_float_counts(counts: dict[str, float]) -> str:
    return ", ".join(
        f"{tier}={counts.get(tier, 0.0):.3f}"
        for tier in TIERS
    )


def _format_confusion(confusion: dict[tuple[str, str], int]) -> str:
    if not confusion:
        return "none"
    return ", ".join(
        f"{expected}->{predicted}={count:,}"
        for (expected, predicted), count in sorted(confusion.items())
    )
