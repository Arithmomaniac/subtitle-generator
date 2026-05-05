"""Exportable student-model distillation reports."""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.book_model_baseline import TIERS
from subtitle_generator.book_model_torch import train_book_tier_torch


@dataclass(frozen=True)
class DistillationResult:
    teacher_labels_path: Path
    student_predictions_path: Path
    student_report_path: Path
    distillation_report_path: Path
    agreement: float
    validation_macro_accuracy: float


def distill_exportable_book_model(
    *,
    features_path: Path,
    labels_path: Path,
    teacher_predictions_path: Path,
    output_dir: Path,
    feature_set: str = "export-slot",
    device: str = "auto",
    epochs: int = 300,
    learning_rate: float = 0.02,
    hidden_dim: int = 64,
    hash_dim: int = 2048,
) -> DistillationResult:
    """Train an export-focused student from rich teacher predictions."""

    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_predictions = _read_csv_by_id(teacher_predictions_path)
    teacher_labels_path = output_dir / "book_teacher_labels.csv"
    _write_teacher_labels(teacher_labels_path, teacher_predictions)

    student_result = train_book_tier_torch(
        features_path=features_path,
        labels_path=teacher_labels_path,
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        hash_dim=hash_dim,
        device=device,
        feature_set=feature_set,
        semantic_vectors="none",
    )
    student_predictions = _read_csv_by_id(student_result.predictions_path)
    true_labels = _read_csv_by_id(labels_path)
    labeled_ids = [
        pattern_id for pattern_id, row in true_labels.items()
        if row.get("label_target") in TIERS
        and pattern_id in teacher_predictions
        and pattern_id in student_predictions
    ]
    _, validation_ids = _train_validation_split(labeled_ids, true_labels)
    if not validation_ids:
        validation_ids = labeled_ids
    agreement_metrics = _agreement_metrics(student_predictions, teacher_predictions)
    validation_metrics = _validation_metrics(
        student_predictions,
        true_labels,
        validation_ids,
    )
    teacher_validation_metrics = _validation_metrics(
        teacher_predictions,
        true_labels,
        validation_ids,
    )
    report = format_distillation_report(
        feature_set=feature_set,
        teacher_predictions_path=teacher_predictions_path,
        student_predictions_path=student_result.predictions_path,
        teacher_labels_path=teacher_labels_path,
        agreement_metrics=agreement_metrics,
        validation_metrics=validation_metrics,
        teacher_validation_metrics=teacher_validation_metrics,
        validation_count=len(validation_ids),
    )
    distillation_report_path = output_dir / "book_distillation_report.md"
    distillation_report_path.write_text(report, encoding="utf-8")
    return DistillationResult(
        teacher_labels_path=teacher_labels_path,
        student_predictions_path=student_result.predictions_path,
        student_report_path=student_result.report_path,
        distillation_report_path=distillation_report_path,
        agreement=agreement_metrics["agreement"],
        validation_macro_accuracy=validation_metrics["macro_accuracy"],
    )


def format_distillation_report(
    *,
    feature_set: str,
    teacher_predictions_path: Path,
    student_predictions_path: Path,
    teacher_labels_path: Path,
    agreement_metrics: dict,
    validation_metrics: dict,
    teacher_validation_metrics: dict,
    validation_count: int,
) -> str:
    return "\n".join([
        "# Exportable book-model distillation report",
        "",
        "This report evaluates whether a compact student model can approximate "
        "the rich offline teacher without raw LOC/Open Library metadata or "
        "semantic vectors at runtime.",
        "",
        "## Inputs and outputs",
        "",
        f"- Feature set: `{feature_set}`",
        f"- Teacher predictions: `{teacher_predictions_path}`",
        f"- Teacher labels for student training: `{teacher_labels_path}`",
        f"- Student predictions: `{student_predictions_path}`",
        "",
        "## Teacher/student agreement",
        "",
        f"- All-row agreement: {agreement_metrics['agreement']:.3f}",
        "- Agreement by teacher tier: "
        + _format_float_counts(agreement_metrics["by_teacher_tier"]),
        "- Student prediction distribution: "
        + _format_counts(agreement_metrics["student_counts"]),
        "- Teacher prediction distribution: "
        + _format_counts(agreement_metrics["teacher_counts"]),
        "",
        "## Validation against original LLM labels",
        "",
        f"- Validation rows: {validation_count:,}",
        f"- Teacher exact/macro: {teacher_validation_metrics['accuracy']:.3f} / "
        f"{teacher_validation_metrics['macro_accuracy']:.3f}",
        f"- Student exact/macro: {validation_metrics['accuracy']:.3f} / "
        f"{validation_metrics['macro_accuracy']:.3f}",
        "- Student per-tier accuracy: "
        + _format_float_counts(validation_metrics["per_tier_accuracy"]),
        "- Student confusion: " + _format_confusion(validation_metrics["confusion"]),
        "",
        "## Recommendation",
        "",
        _recommendation(agreement_metrics, validation_metrics, teacher_validation_metrics),
    ])


def _recommendation(
    agreement_metrics: dict,
    validation_metrics: dict,
    teacher_validation_metrics: dict,
) -> str:
    macro_gap = (
        teacher_validation_metrics["macro_accuracy"]
        - validation_metrics["macro_accuracy"]
    )
    if agreement_metrics["agreement"] >= 0.75 and macro_gap <= 0.08:
        return (
            "- The exportable student is a plausible candidate for shadow rollups; "
            "continue to filler aggregation and fixed-seed comparison."
        )
    return (
        "- The exportable student loses too much teacher/label signal for "
        "deployment. Use it for diagnostics, or add explicit exported scalar "
        "features before shadow rollups."
    )


def _write_teacher_labels(
    path: Path,
    teacher_predictions: dict[str, dict[str, str]],
) -> None:
    fieldnames = ("pattern_match_id", "subtitle_id", "label_target", "label_confidence")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pattern_id in sorted(teacher_predictions, key=int):
            row = teacher_predictions[pattern_id]
            writer.writerow({
                "pattern_match_id": pattern_id,
                "subtitle_id": row.get("subtitle_id", ""),
                "label_target": row.get("predicted_tier", ""),
                "label_confidence": row.get("prediction_confidence", ""),
            })


def _agreement_metrics(
    student_predictions: dict[str, dict[str, str]],
    teacher_predictions: dict[str, dict[str, str]],
) -> dict:
    shared_ids = sorted(
        set(student_predictions) & set(teacher_predictions),
        key=int,
    )
    correct = 0
    teacher_counts: Counter[str] = Counter()
    student_counts: Counter[str] = Counter()
    by_tier_total: Counter[str] = Counter()
    by_tier_correct: Counter[str] = Counter()
    for pattern_id in shared_ids:
        teacher = teacher_predictions[pattern_id]["predicted_tier"]
        student = student_predictions[pattern_id]["predicted_tier"]
        teacher_counts[teacher] += 1
        student_counts[student] += 1
        by_tier_total[teacher] += 1
        if teacher == student:
            correct += 1
            by_tier_correct[teacher] += 1
    by_teacher_tier = {
        tier: (
            by_tier_correct[tier] / by_tier_total[tier]
            if by_tier_total[tier]
            else 0.0
        )
        for tier in TIERS
    }
    return {
        "agreement": correct / max(1, len(shared_ids)),
        "by_teacher_tier": by_teacher_tier,
        "teacher_counts": teacher_counts,
        "student_counts": student_counts,
    }


def _validation_metrics(
    predictions: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    validation_ids: list[str],
) -> dict:
    confusion: Counter[tuple[str, str]] = Counter()
    per_tier_total: Counter[str] = Counter()
    per_tier_correct: Counter[str] = Counter()
    correct = 0
    for pattern_id in validation_ids:
        expected = labels[pattern_id]["label_target"]
        predicted = predictions[pattern_id]["predicted_tier"]
        confusion[(expected, predicted)] += 1
        per_tier_total[expected] += 1
        if expected == predicted:
            correct += 1
            per_tier_correct[expected] += 1
    per_tier_accuracy = {
        tier: (
            per_tier_correct[tier] / per_tier_total[tier]
            if per_tier_total[tier]
            else 0.0
        )
        for tier in TIERS
    }
    return {
        "accuracy": correct / max(1, len(validation_ids)),
        "macro_accuracy": sum(per_tier_accuracy.values()) / len(TIERS),
        "per_tier_accuracy": per_tier_accuracy,
        "confusion": dict(confusion),
    }


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["pattern_match_id"]: row for row in csv.DictReader(handle)}
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


def _format_counts(counts: Counter[str]) -> str:
    return ", ".join(f"{tier}={counts.get(tier, 0):,}" for tier in TIERS)


def _format_float_counts(counts: dict[str, float]) -> str:
    return ", ".join(f"{tier}={counts.get(tier, 0.0):.3f}" for tier in TIERS)


def _format_confusion(confusion: dict[tuple[str, str], int]) -> str:
    if not confusion:
        return "none"
    return ", ".join(
        f"{expected}->{predicted}={count:,}"
        for (expected, predicted), count in sorted(confusion.items())
    )
