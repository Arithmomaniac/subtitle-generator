"""Calibrate final assembled-subtitle tier classifier coefficients."""

from __future__ import annotations

import csv
import math
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.book_model_artifacts import build_book_model_artifacts
from subtitle_generator.book_model_popularity_calibration import (
    PopularityCalibrationResult,
    calibrate_popularity_weights,
)
from subtitle_generator.config import invalidate_config_cache, load_tuning_config
from subtitle_generator.tiering import (
    TIER_NAMES,
    _lookup_slot_evidence,
    _slot_model_scores,
    parse_subtitle_slots,
)


@dataclass(frozen=True)
class TierClassifierExample:
    pattern_match_id: str
    subtitle: str
    target: list[float]
    model_scores: list[float]
    popularity: float
    interactions: list[float]
    frequency_score: float


@dataclass(frozen=True)
class TierClassifierCalibrationResult:
    report_path: Path
    current_mse: float
    learned_mse: float
    validation_mse: float
    example_count: int
    applied: bool


@dataclass(frozen=True)
class RuntimeTierModelCalibrationResult:
    report_path: Path
    popularity: PopularityCalibrationResult
    classifier: TierClassifierCalibrationResult
    applied: bool


def calibrate_runtime_tier_model(
    *,
    features_path: Path,
    teacher_predictions_path: Path,
    db_path: Path,
    output_dir: Path,
    rollup_path: Path | None = None,
    metadata_path: Path | None = None,
    apply: bool = False,
    popularity_epochs: int = 300,
    classifier_epochs: int = 500,
    device: str = "cpu",
) -> RuntimeTierModelCalibrationResult:
    """Run the single runtime tier-model calibration step."""

    popularity = calibrate_popularity_weights(
        features_path=features_path,
        teacher_predictions_path=teacher_predictions_path,
        output_dir=output_dir / "popularity",
        db_path=db_path,
        apply=apply,
        epochs=popularity_epochs,
        device=device,
    )
    if apply:
        script_path = Path(__file__).resolve().parents[2] / "data" / "populate_popularity.py"
        subprocess.run(
            [sys.executable, str(script_path), "--db", str(db_path)],
            check=True,
            cwd=script_path.parents[1],
        )
        conn = sqlite3.connect(db_path)
        try:
            features_path = build_book_model_artifacts(
                conn,
                features_path.parent,
                metadata_path=metadata_path,
            ).features_path
        finally:
            conn.close()
    classifier = calibrate_tier_classifier_weights(
        features_path=features_path,
        teacher_predictions_path=teacher_predictions_path,
        db_path=db_path,
        output_dir=output_dir / "classifier",
        rollup_path=rollup_path,
        apply=apply,
        epochs=classifier_epochs,
        device=device,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "runtime_tier_model_calibration.md"
    report_path.write_text(
        _format_runtime_report(popularity, classifier, apply, db_path),
        encoding="utf-8",
    )
    return RuntimeTierModelCalibrationResult(
        report_path=report_path,
        popularity=popularity,
        classifier=classifier,
        applied=apply,
    )


def calibrate_tier_classifier_weights(
    *,
    features_path: Path,
    teacher_predictions_path: Path,
    db_path: Path,
    output_dir: Path,
    rollup_path: Path | None = None,
    apply: bool = False,
    epochs: int = 500,
    learning_rate: float = 0.03,
    regularization: float = 0.001,
    device: str = "cpu",
) -> TierClassifierCalibrationResult:
    """Fit runtime config coefficients for post-selection tier classification."""

    torch = _import_torch()
    resolved_device = _resolve_device(torch, device)
    torch.manual_seed(20260511)

    conn = sqlite3.connect(db_path)
    try:
        cfg = load_tuning_config(conn)
        feature_rows = _read_csv_by_id(features_path)
        teacher_rows = _read_csv_by_id(teacher_predictions_path)
        rollup_scores = _read_rollup_scores(rollup_path) if rollup_path else None
        shared_ids = sorted(set(feature_rows) & set(teacher_rows), key=_sort_key)
        examples = [
            example
            for pattern_id in shared_ids
            if (
                example := _build_example(
                    conn,
                    cfg,
                    pattern_id,
                    feature_rows[pattern_id],
                    teacher_rows[pattern_id],
                    rollup_scores,
                )
            )
            is not None
        ]
    finally:
        conn.close()
    if not examples:
        raise RuntimeError("No examples had complete slot model evidence and teacher scores.")

    train_examples, validation_examples = _train_validation_split(examples)
    learned = _fit_coefficients(
        torch,
        train_examples=train_examples,
        validation_examples=validation_examples,
        epochs=epochs,
        learning_rate=learning_rate,
        regularization=regularization,
        device=resolved_device,
    )
    current_mse = _mse(examples, _current_probabilities)
    learned_mse = _mse(examples, lambda example: _learned_probabilities(example, learned))
    validation_mse = _mse(
        validation_examples,
        lambda example: _learned_probabilities(example, learned),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "tier_classifier_calibration.md"
    report_path.write_text(
        _format_report(
            examples=examples,
            current_mse=current_mse,
            learned_mse=learned_mse,
            validation_mse=validation_mse,
            coefficients=learned,
            epochs=epochs,
            regularization=regularization,
            device=str(resolved_device),
            applied=apply,
            db_path=db_path,
        ),
        encoding="utf-8",
    )
    if apply:
        _apply_coefficients(db_path, learned)
    return TierClassifierCalibrationResult(
        report_path=report_path,
        current_mse=current_mse,
        learned_mse=learned_mse,
        validation_mse=validation_mse,
        example_count=len(examples),
        applied=apply,
    )


def _build_example(
    conn: sqlite3.Connection,
    cfg: dict[str, float],
    pattern_id: str,
    feature_row: dict[str, str],
    teacher_row: dict[str, str],
    rollup_scores: dict[tuple[str, str], dict[str, float]] | None,
) -> TierClassifierExample | None:
    target = [_float(teacher_row.get(f"score_{tier}")) for tier in TIER_NAMES]
    total = sum(target)
    if total <= 0:
        return None
    target = [value / total for value in target]
    subtitle = _canonical_subtitle(feature_row)
    slots = parse_subtitle_slots(subtitle)
    if not slots:
        return None
    blend = cfg["pop_classification_blend"]
    missing_default = cfg["pop_missing_default"]
    evidence = [
        _lookup_slot_evidence(conn, slot, blend, missing_default)
        for slot in slots
    ]
    slot_scores = [
        _rollup_model_scores(rollup_scores, slot)
        if rollup_scores is not None
        else _slot_model_scores(slot)
        for slot in evidence
    ]
    if any(not scores for scores in slot_scores):
        return None
    popularity_values = [
        slot.popularity_score if slot.popularity_score is not None else missing_default
        for slot in evidence
    ]
    popularity = sum(popularity_values) / len(popularity_values)
    model_scores = [
        sum(scores[tier] for scores in slot_scores) / len(slot_scores)
        for tier in TIER_NAMES
    ]
    interactions = [
        sum(
            popularity_values[index] * slot_scores[index][tier]
            for index in range(len(slot_scores))
        ) / len(slot_scores)
        for tier in TIER_NAMES
    ]
    frequency_score = sum(slot.frequency_score for slot in evidence) / len(evidence)
    return TierClassifierExample(
        pattern_match_id=pattern_id,
        subtitle=subtitle,
        target=target,
        model_scores=model_scores,
        popularity=popularity,
        interactions=interactions,
        frequency_score=frequency_score,
    )


def _canonical_subtitle(row: dict[str, str]) -> str:
    list_items = [item.strip() for item in row.get("list_items_text", "").split("|")]
    list_items = [item for item in list_items if item]
    action = row.get("action_noun", "").strip()
    obj = row.get("of_object", "").strip()
    if len(list_items) >= 2 and action and obj:
        return f"{list_items[0]}, {list_items[1]}, and the {action} of {obj}"
    return row.get("subtitle_text") or row.get("title") or ""


def _fit_coefficients(
    torch,
    *,
    train_examples: list[TierClassifierExample],
    validation_examples: list[TierClassifierExample],
    epochs: int,
    learning_rate: float,
    regularization: float,
    device,
) -> dict[str, float]:
    train = _tensor_bundle(torch, train_examples, device)
    validation = _tensor_bundle(torch, validation_examples, device)
    intercept = torch.zeros(len(TIER_NAMES), dtype=torch.float32, device=device, requires_grad=True)
    model_weight = torch.tensor(1.0, dtype=torch.float32, device=device, requires_grad=True)
    popularity_weight = torch.zeros(len(TIER_NAMES), dtype=torch.float32, device=device, requires_grad=True)
    interaction_weight = torch.zeros(len(TIER_NAMES), dtype=torch.float32, device=device, requires_grad=True)
    frequency_weight = torch.zeros(len(TIER_NAMES), dtype=torch.float32, device=device, requires_grad=True)
    params = [intercept, model_weight, popularity_weight, interaction_weight, frequency_weight]
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    best_state = None
    best_validation = float("inf")
    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = _predict_tensor(
            torch,
            train,
            intercept,
            model_weight,
            popularity_weight,
            interaction_weight,
            frequency_weight,
        )
        loss = torch.mean((prediction - train["target"]) ** 2)
        loss = loss + regularization * (
            torch.sum(intercept ** 2)
            + (model_weight - 1.0) ** 2
            + torch.sum(popularity_weight ** 2)
            + torch.sum(interaction_weight ** 2)
            + torch.sum(frequency_weight ** 2)
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation_prediction = _predict_tensor(
                torch,
                validation,
                intercept,
                model_weight,
                popularity_weight,
                interaction_weight,
                frequency_weight,
            )
            validation_loss = torch.mean((validation_prediction - validation["target"]) ** 2)
            validation_value = float(validation_loss.item())
        if validation_value < best_validation:
            best_validation = validation_value
            best_state = [param.detach().clone() for param in params]
    if best_state is not None:
        with torch.no_grad():
            for param, value in zip(params, best_state, strict=True):
                param.copy_(value)
    return _coefficients_from_tensors(
        intercept,
        model_weight,
        popularity_weight,
        interaction_weight,
        frequency_weight,
    )


def _tensor_bundle(torch, examples: list[TierClassifierExample], device) -> dict[str, object]:
    return {
        "model_scores": torch.tensor(
            [example.model_scores for example in examples],
            dtype=torch.float32,
            device=device,
        ),
        "popularity": torch.tensor(
            [[example.popularity] for example in examples],
            dtype=torch.float32,
            device=device,
        ),
        "interactions": torch.tensor(
            [example.interactions for example in examples],
            dtype=torch.float32,
            device=device,
        ),
        "frequency": torch.tensor(
            [[example.frequency_score] for example in examples],
            dtype=torch.float32,
            device=device,
        ),
        "target": torch.tensor(
            [example.target for example in examples],
            dtype=torch.float32,
            device=device,
        ),
    }


def _predict_tensor(
    torch,
    bundle: dict[str, object],
    intercept,
    model_weight,
    popularity_weight,
    interaction_weight,
    frequency_weight,
):
    logits = (
        intercept
        + model_weight * bundle["model_scores"]
        + bundle["popularity"] * popularity_weight
        + bundle["interactions"] * interaction_weight
        + bundle["frequency"] * frequency_weight
    )
    return torch.softmax(logits, dim=1)


def _coefficients_from_tensors(
    intercept,
    model_weight,
    popularity_weight,
    interaction_weight,
    frequency_weight,
) -> dict[str, float]:
    coefficients = {
        "tier_classifier_model_score_weight": float(model_weight.detach().cpu().item()),
        "tier_classifier_temperature": 1.0,
    }
    for index, tier in enumerate(TIER_NAMES):
        coefficients[f"tier_classifier_intercept_{tier}"] = float(
            intercept[index].detach().cpu().item()
        )
        coefficients[f"tier_classifier_popularity_weight_{tier}"] = float(
            popularity_weight[index].detach().cpu().item()
        )
        coefficients[f"tier_classifier_popularity_interaction_{tier}"] = float(
            interaction_weight[index].detach().cpu().item()
        )
        coefficients[f"tier_classifier_frequency_weight_{tier}"] = float(
            frequency_weight[index].detach().cpu().item()
        )
    return coefficients


def _current_probabilities(example: TierClassifierExample) -> list[float]:
    return example.model_scores


def _learned_probabilities(
    example: TierClassifierExample,
    coefficients: dict[str, float],
) -> list[float]:
    logits = []
    for index, tier in enumerate(TIER_NAMES):
        logits.append(
            coefficients[f"tier_classifier_intercept_{tier}"]
            + coefficients["tier_classifier_model_score_weight"] * example.model_scores[index]
            + coefficients[f"tier_classifier_popularity_weight_{tier}"] * example.popularity
            + coefficients[f"tier_classifier_popularity_interaction_{tier}"]
            * example.interactions[index]
            + coefficients[f"tier_classifier_frequency_weight_{tier}"]
            * example.frequency_score
        )
    return _softmax(logits)


def _softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    values = [math.exp(value - max_logit) for value in logits]
    total = sum(values)
    return [value / total for value in values]


def _mse(examples: list[TierClassifierExample], predictor) -> float:
    if not examples:
        return 0.0
    return sum(
        (prediction - target) ** 2
        for example in examples
        for prediction, target in zip(predictor(example), example.target, strict=True)
    ) / (len(examples) * len(TIER_NAMES))


def _train_validation_split(
    examples: list[TierClassifierExample],
) -> tuple[list[TierClassifierExample], list[TierClassifierExample]]:
    train = []
    validation = []
    for index, example in enumerate(sorted(examples, key=lambda item: _sort_key(item.pattern_match_id))):
        if index % 5 == 0:
            validation.append(example)
        else:
            train.append(example)
    return train or examples, validation or examples


def _apply_coefficients(db_path: Path, coefficients: dict[str, float]) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
        for key, value in coefficients.items():
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (key, f"{value:.8f}"),
            )
        for slot_key in (
            "tier_classifier_slot_weight_list_item",
            "tier_classifier_slot_weight_action_noun",
            "tier_classifier_slot_weight_of_object",
        ):
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, '1')",
                (slot_key,),
            )
        conn.commit()
    finally:
        conn.close()
    invalidate_config_cache()


def _read_rollup_scores(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {
            (row["slot_type"], row["filler"].casefold()): {
                "pop": float(row["avg_score_pop"]),
                "mainstream": float(row["avg_score_mainstream"]),
                "niche": float(row["avg_score_niche"]),
            }
            for row in csv.DictReader(handle)
        }


def _rollup_model_scores(
    rollup_scores: dict[tuple[str, str], dict[str, float]] | None,
    slot,
) -> dict[str, float]:
    if rollup_scores is None:
        return {}
    return rollup_scores.get((slot.slot_type, slot.filler.casefold()), {})


def _format_report(
    *,
    examples: list[TierClassifierExample],
    current_mse: float,
    learned_mse: float,
    validation_mse: float,
    coefficients: dict[str, float],
    epochs: int,
    regularization: float,
    device: str,
    applied: bool,
    db_path: Path,
) -> str:
    lines = [
        "# Tier classifier calibration",
        "",
        f"- Examples: {len(examples):,}",
        f"- Training epochs: {epochs:,}",
        f"- Regularization toward neutral coefficients: {regularization:g}",
        f"- Device: `{device}`",
        f"- Applied to DB: {'yes' if applied else 'no'}",
        f"- DB: `{db_path}`",
        "",
        "## Fit comparison",
        "",
        "| Model | MSE |",
        "|---|---:|",
        f"| current mean slot probabilities | {current_mse:.6f} |",
        f"| learned runtime classifier | {learned_mse:.6f} |",
        f"| learned runtime classifier validation | {validation_mse:.6f} |",
        "",
        "## Coefficients",
        "",
        "| Config key | Value |",
        "|---|---:|",
    ]
    for key in sorted(coefficients):
        lines.append(f"| {key} | {coefficients[key]:.8f} |")
    return "\n".join(lines) + "\n"


def _format_runtime_report(
    popularity: PopularityCalibrationResult,
    classifier: TierClassifierCalibrationResult,
    applied: bool,
    db_path: Path,
) -> str:
    return "\n".join([
        "# Runtime tier model calibration",
        "",
        "This is the single pipeline calibration step for runtime tiering. It learns",
        "the collapsed popularity source ratios, refreshes runtime popularity scores",
        "when applied, and learns final assembled-subtitle classifier coefficients.",
        "",
        f"- Applied to DB: {'yes' if applied else 'no'}",
        f"- DB: `{db_path}`",
        "",
        "## MSEs",
        "",
        "| Component | Baseline MSE | Learned MSE | Validation MSE |",
        "|---|---:|---:|---:|",
        (
            "| Popularity scalar | "
            f"{popularity.current_mse:.6f} | {popularity.learned_mse:.6f} | n/a |"
        ),
        (
            "| Final classifier | "
            f"{classifier.current_mse:.6f} | {classifier.learned_mse:.6f} | "
            f"{classifier.validation_mse:.6f} |"
        ),
        "",
        "## Reports",
        "",
        f"- Popularity source report: `{popularity.report_path}`",
        f"- Final classifier report: `{classifier.report_path}`",
    ]) + "\n"


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["pattern_match_id"]: row for row in csv.DictReader(handle)}


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _sort_key(value: str):
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for calibrate-tier-classifier-weights. "
            "Install it with `uv sync --extra ml`."
        ) from exc
    return torch


def _resolve_device(torch, requested: str):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build cannot use CUDA.")
    return torch.device(requested)
