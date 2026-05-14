"""PyTorch trainer for offline book-tier predictions."""

from __future__ import annotations

import csv
import copy
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.book_model_baseline import (
    NUMERIC_FEATURES,
    TEXT_FEATURE_COLUMNS,
    TIERS,
)

HASH_DIM = 2048
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.02
DEFAULT_HIDDEN_DIM = 64
_TOKEN_RE = re.compile(r"[a-z][a-z']+")
BASE_NUMERIC_FEATURES = (
    "candidate_source_is_title",
    "source_group_is_ol",
    "lang_is_eng",
    "has_isbn",
    "has_lccn",
    "has_work_key",
    "has_popularity_data",
    "work_popularity_score",
    "slot_source_link_count",
    "distinct_strict_filler_count",
    "max_filler_popularity_score",
    "avg_filler_popularity_score",
    "max_filler_frequency_score",
    "avg_filler_frequency_score",
    "title_length_chars",
    "subtitle_length_chars",
    "list_item_count",
)
POPULARITY_NUMERIC_FEATURES = (
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
)
METADATA_NUMERIC_FEATURES = (
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
BASE_TEXT_COLUMNS = ("title", "subtitle_text", "action_noun", "of_object")
EXPORT_TEXT_COLUMNS = ("title", "subtitle_text")
INTERACTION_TEXT_COLUMNS = (
    "list_items_text",
    "list_item_pair_text",
    "action_object_pair_text",
    "slot_frame_text",
)
METADATA_TEXT_COLUMNS = (
    "metadata_publisher_text",
    "metadata_edition_name",
    "metadata_physical_format",
    "metadata_physical_dimensions",
    "metadata_physical_description",
    "metadata_subject_text",
    "metadata_loc_call_number",
    "metadata_dewey_decimal",
)
EXPORT_CURRENT_NUMERIC_FEATURES = (
    "source_group_is_ol",
    "title_length_chars",
    "subtitle_length_chars",
)
EXPORT_SLOT_NUMERIC_FEATURES = (
    *EXPORT_CURRENT_NUMERIC_FEATURES,
    "slot_source_link_count",
    "distinct_strict_filler_count",
    "max_filler_popularity_score",
    "avg_filler_popularity_score",
    "max_filler_frequency_score",
    "avg_filler_frequency_score",
)
FEATURE_SET_CHOICES = (
    "export-current",
    "export-slot",
    "persisted",
    "popularity",
    "interactions",
    "metadata",
    "all",
)
SEMANTIC_VECTOR_CHOICES = ("none", "spacy")


@dataclass(frozen=True)
class TorchTrainingResult:
    predictions_path: Path
    report_path: Path
    prediction_count: int
    labeled_count: int
    validation_count: int
    validation_accuracy: float
    validation_macro_accuracy: float
    training_device: str


@dataclass(frozen=True)
class _PreparedData:
    pattern_ids: list[str]
    feature_rows: dict[str, dict[str, str]]
    labels: dict[str, dict[str, str]]
    train_ids: list[str]
    validation_ids: list[str]
    x_all: object
    y_train: object
    train_indices: object
    validation_indices: object
    y_validation: object
    sample_weights: object
    class_weights: object
    numeric_features: tuple[str, ...]
    text_columns: tuple[str, ...]
    semantic_vector_mode: str
    semantic_vector_dim: int


def train_book_tier_torch(
    *,
    features_path: Path,
    labels_path: Path,
    output_dir: Path,
    epochs: int = DEFAULT_EPOCHS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    hash_dim: int = HASH_DIM,
    random_seed: int = 20260505,
    device: str = "auto",
    feature_set: str = "all",
    semantic_vectors: str = "none",
) -> TorchTrainingResult:
    """Train a gradient-descent book-tier classifier and write predictions/report."""

    torch = _import_torch()
    if semantic_vectors not in SEMANTIC_VECTOR_CHOICES:
        raise RuntimeError(
            "Unknown semantic vector mode. Expected one of: "
            + ", ".join(SEMANTIC_VECTOR_CHOICES)
        )
    numeric_features, text_columns = _feature_columns(feature_set)
    resolved_device = _resolve_device(torch, device)
    torch.manual_seed(random_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = _read_csv_by_id(features_path)
    labels = _read_csv_by_id(labels_path)
    labeled_ids = [
        pattern_id for pattern_id, row in labels.items()
        if row.get("label_target") in TIERS and pattern_id in features
    ]
    if not labeled_ids:
        raise RuntimeError("No labeled rows available for Torch training.")

    train_ids, validation_ids = _train_validation_split(labeled_ids, labels)
    if not validation_ids:
        validation_ids = train_ids

    prepared = _prepare_data(
        torch,
        features=features,
        labels=labels,
        train_ids=train_ids,
        validation_ids=validation_ids,
        hash_dim=hash_dim,
        device=resolved_device,
        numeric_features=numeric_features,
        text_columns=text_columns,
        semantic_vector_mode=semantic_vectors,
    )
    model = _BookTierNet(
        torch=torch,
        input_dim=prepared.x_all.shape[1],
        hidden_dim=hidden_dim,
        device=resolved_device,
    )
    history = _fit_model(
        torch,
        model=model,
        prepared=prepared,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    predictions = _predict_all(torch, model, prepared)
    metrics = _evaluate_predictions(predictions, validation_ids, labels)

    predictions_path = output_dir / "book_torch_predictions.csv"
    report_path = output_dir / "book_torch_report.md"
    _write_predictions(predictions_path, predictions)
    report_path.write_text(
        format_torch_report(
            predictions=predictions,
            labels=labels,
            validation_ids=validation_ids,
            metrics=metrics,
            predictions_path=predictions_path,
            epochs=epochs,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            hash_dim=hash_dim,
            training_device=str(resolved_device),
            feature_set=feature_set,
            numeric_feature_count=len(numeric_features),
            text_column_count=len(text_columns),
            semantic_vector_mode=semantic_vectors,
            semantic_vector_dim=prepared.semantic_vector_dim,
            history=history,
        ),
        encoding="utf-8",
    )
    return TorchTrainingResult(
        predictions_path=predictions_path,
        report_path=report_path,
        prediction_count=len(predictions),
        labeled_count=len(labeled_ids),
        validation_count=len(validation_ids),
        validation_accuracy=metrics["accuracy"],
        validation_macro_accuracy=metrics["macro_accuracy"],
        training_device=str(resolved_device),
    )


def format_torch_report(
    *,
    predictions: list[dict[str, str]],
    labels: dict[str, dict[str, str]],
    validation_ids: list[str],
    metrics: dict,
    predictions_path: Path,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    hash_dim: int,
    training_device: str,
    feature_set: str,
    numeric_feature_count: int,
    text_column_count: int,
    semantic_vector_mode: str,
    semantic_vector_dim: int,
    history: dict[str, float],
) -> str:
    label_counts = Counter(
        row.get("label_target", "")
        for row in labels.values()
        if row.get("label_target") in TIERS
    )
    predicted_counts = Counter(row["predicted_tier"] for row in predictions)
    lines = [
        "# Book-tier Torch model report",
        "",
        "This is a gradient-descent trained offline model over `book_features` "
        "and `book_labels`. It is still a research artifact, not a runtime model.",
        "",
        "## Outputs",
        "",
        f"- Predictions: `{predictions_path}` ({len(predictions):,} rows)",
        "",
        "## Training setup",
        "",
        f"- Epochs: {epochs:,}",
        f"- Learning rate: {learning_rate:g}",
        f"- Hidden dimension: {hidden_dim:,}",
        f"- Hashed text dimensions: {hash_dim:,}",
        f"- Training device: `{training_device}`",
        f"- Feature set: `{feature_set}`",
        f"- Numeric features: {numeric_feature_count:,}",
        f"- Text columns: {text_column_count:,}",
        f"- Semantic vector mode: `{semantic_vector_mode}`",
        f"- Semantic vector dimensions: {semantic_vector_dim:,}",
        f"- Final weighted training loss: {history['final_loss']:.4f}",
        f"- Best epoch: {history['best_epoch']:.0f}",
        f"- Best validation macro during training: {history['best_macro_accuracy']:.3f}",
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
        "- This model uses offline-rich features when present. Runtime work still "
        "requires exportable distillation or a durable feature export decision.",
        "- Class and sample weights are used so the current niche-heavy labels do "
        "not dominate the loss completely.",
    ]
    if label_counts.get("pop", 0) < 100:
        lines.append(
            "- Pop remains under-labeled; pop metrics are useful for direction but "
            "not yet stable enough for deployment decisions."
        )
    return "\n".join(lines)


class _BookTierNet:
    def __init__(self, *, torch, input_dim: int, hidden_dim: int, device) -> None:
        self.module = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(TIERS)),
        ).to(device)

    def __call__(self, x):
        return self.module(x)

    def parameters(self):
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()


def _fit_model(
    torch,
    *,
    model: _BookTierNet,
    prepared: _PreparedData,
    epochs: int,
    learning_rate: float,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(
        weight=prepared.class_weights,
        reduction="none",
    )
    x_train = prepared.x_all[prepared.train_indices]
    model.train()
    final_loss = 0.0
    best_state = copy.deepcopy(model.module.state_dict())
    best_macro_accuracy = -1.0
    best_accuracy = -1.0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        losses = criterion(logits, prepared.y_train)
        loss = (losses * prepared.sample_weights).mean()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
        accuracy, macro_accuracy = _torch_validation_metrics(
            torch,
            model,
            prepared,
        )
        if (macro_accuracy, accuracy) > (best_macro_accuracy, best_accuracy):
            best_macro_accuracy = macro_accuracy
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.module.state_dict())
    model.module.load_state_dict(best_state)
    return {
        "final_loss": final_loss,
        "best_epoch": float(best_epoch),
        "best_accuracy": best_accuracy,
        "best_macro_accuracy": best_macro_accuracy,
    }


def _torch_validation_metrics(
    torch,
    model: _BookTierNet,
    prepared: _PreparedData,
) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(prepared.x_all[prepared.validation_indices])
        predicted = torch.argmax(logits, dim=1)
    correct = predicted.eq(prepared.y_validation)
    accuracy = float(correct.float().mean().item())
    per_tier: list[float] = []
    for tier_index in range(len(TIERS)):
        mask = prepared.y_validation.eq(tier_index)
        if bool(mask.any().item()):
            per_tier.append(float(correct[mask].float().mean().item()))
        else:
            per_tier.append(0.0)
    return accuracy, sum(per_tier) / len(per_tier)


def _predict_all(torch, model: _BookTierNet, prepared: _PreparedData) -> list[dict[str, str]]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(prepared.x_all), dim=1).cpu()
    predictions: list[dict[str, str]] = []
    for row_index, pattern_id in enumerate(prepared.pattern_ids):
        scores = {
            tier: float(probabilities[row_index, tier_index].item())
            for tier_index, tier in enumerate(TIERS)
        }
        predicted = max(scores, key=scores.get)
        feature_row = prepared.feature_rows[pattern_id]
        label_row = prepared.labels.get(pattern_id, {})
        predictions.append({
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
        })
    return predictions


def _prepare_data(
    torch,
    *,
    features: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    train_ids: list[str],
    validation_ids: list[str],
    hash_dim: int,
    device,
    numeric_features: tuple[str, ...],
    text_columns: tuple[str, ...],
    semantic_vector_mode: str,
) -> _PreparedData:
    pattern_ids = sorted(features, key=int)
    scaler = _fit_numeric_scaler(features, train_ids, numeric_features=numeric_features)
    semantic_vectors = _semantic_vectors(
        features,
        pattern_ids,
        text_columns=text_columns,
        mode=semantic_vector_mode,
    )
    vectors = [
        _vectorize_row(
            features[pattern_id],
            scaler,
            hash_dim=hash_dim,
            numeric_features=numeric_features,
            text_columns=text_columns,
            semantic_vector=semantic_vectors[row_index],
        )
        for row_index, pattern_id in enumerate(pattern_ids)
    ]
    train_index_by_id = {pattern_id: index for index, pattern_id in enumerate(pattern_ids)}
    train_indices_list = [train_index_by_id[pattern_id] for pattern_id in train_ids]
    validation_indices_list = [
        train_index_by_id[pattern_id] for pattern_id in validation_ids
    ]
    tier_to_index = {tier: index for index, tier in enumerate(TIERS)}
    y_train_values = [tier_to_index[labels[pattern_id]["label_target"]] for pattern_id in train_ids]
    label_counts = Counter(labels[pattern_id]["label_target"] for pattern_id in train_ids)
    class_weights = [
        len(train_ids) / (len(TIERS) * max(1, label_counts[tier]))
        for tier in TIERS
    ]
    sample_weights = [
        max(0.25, min(1.0, _float(labels[pattern_id].get("label_confidence")) or 1.0))
        for pattern_id in train_ids
    ]
    return _PreparedData(
        pattern_ids=pattern_ids,
        feature_rows=features,
        labels=labels,
        train_ids=train_ids,
        validation_ids=validation_ids,
        x_all=torch.tensor(vectors, dtype=torch.float32, device=device),
        y_train=torch.tensor(y_train_values, dtype=torch.long, device=device),
        train_indices=torch.tensor(train_indices_list, dtype=torch.long, device=device),
        validation_indices=torch.tensor(
            validation_indices_list,
            dtype=torch.long,
            device=device,
        ),
        y_validation=torch.tensor(
            [
                tier_to_index[labels[pattern_id]["label_target"]]
                for pattern_id in validation_ids
            ],
            dtype=torch.long,
            device=device,
        ),
        sample_weights=torch.tensor(sample_weights, dtype=torch.float32, device=device),
        class_weights=torch.tensor(class_weights, dtype=torch.float32, device=device),
        numeric_features=numeric_features,
        text_columns=text_columns,
        semantic_vector_mode=semantic_vector_mode,
        semantic_vector_dim=len(semantic_vectors[0]) if semantic_vectors else 0,
    )


def _fit_numeric_scaler(
    features: dict[str, dict[str, str]],
    train_ids: list[str],
    *,
    numeric_features: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    scaler: dict[str, tuple[float, float]] = {}
    for feature_name in numeric_features:
        values = [_float(features[pattern_id].get(feature_name)) for pattern_id in train_ids]
        low = min(values)
        high = max(values)
        scaler[feature_name] = (low, high)
    return scaler


def _vectorize_row(
    row: dict[str, str],
    scaler: dict[str, tuple[float, float]],
    *,
    hash_dim: int,
    numeric_features: tuple[str, ...],
    text_columns: tuple[str, ...],
    semantic_vector: list[float],
) -> list[float]:
    vector = []
    for feature_name in numeric_features:
        value = _float(row.get(feature_name))
        low, high = scaler[feature_name]
        vector.append(0.0 if high == low else (value - low) / (high - low))
    hashed = [0.0] * hash_dim
    for token in _tokens(row, text_columns=text_columns):
        hashed[_hash_token(token, hash_dim)] += 1.0
    norm = math.sqrt(sum(value * value for value in hashed)) or 1.0
    vector.extend(value / norm for value in hashed)
    vector.extend(semantic_vector)
    return vector


def _semantic_vectors(
    features: dict[str, dict[str, str]],
    pattern_ids: list[str],
    *,
    text_columns: tuple[str, ...],
    mode: str,
) -> list[list[float]]:
    if mode == "none":
        return [[] for _ in pattern_ids]
    if mode != "spacy":
        raise RuntimeError(
            "Unknown semantic vector mode. Expected one of: "
            + ", ".join(SEMANTIC_VECTOR_CHOICES)
        )
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is required for --semantic-vectors spacy."
        ) from exc
    try:
        nlp = spacy.load(
            "en_core_web_md",
            disable=["parser", "tagger", "ner", "lemmatizer", "attribute_ruler"],
        )
    except OSError as exc:
        raise RuntimeError(
            "The en_core_web_md spaCy model is required for semantic vectors."
        ) from exc
    vector_dim = int(nlp.vocab.vectors_length)
    texts = [
        _row_text(features[pattern_id], text_columns=text_columns)
        for pattern_id in pattern_ids
    ]
    vectors: list[list[float]] = []
    for doc in nlp.pipe(texts, batch_size=128):
        if vector_dim == 0 or doc.vector_norm == 0:
            vectors.append([0.0] * vector_dim)
            continue
        vector = doc.vector / doc.vector_norm
        vectors.append([float(value) for value in vector])
    return vectors


def _row_text(row: dict[str, str], *, text_columns: tuple[str, ...]) -> str:
    return " ".join(row.get(column, "") for column in text_columns)


def _tokens(row: dict[str, str], *, text_columns: tuple[str, ...]) -> list[str]:
    text = _row_text(row, text_columns=text_columns).lower()
    tokens = _TOKEN_RE.findall(text)
    if set(INTERACTION_TEXT_COLUMNS) & set(text_columns):
        tokens.extend(_interaction_tokens(row))
    return tokens


def _interaction_tokens(row: dict[str, str]) -> list[str]:
    tokens: list[str] = []
    for column in ("list_item_pair_text", "action_object_pair_text", "slot_frame_text"):
        parts = [_normalize_phrase(part) for part in row.get(column, "").split("||")]
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


def _hash_token(token: str, hash_dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % hash_dim


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


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for train-book-model-torch. "
            "Install it with `uv sync --extra ml`."
        ) from exc
    return torch


def _feature_columns(feature_set: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if feature_set not in FEATURE_SET_CHOICES:
        raise RuntimeError(
            "Unknown feature set. Expected one of: " + ", ".join(FEATURE_SET_CHOICES)
        )
    if feature_set == "persisted":
        return BASE_NUMERIC_FEATURES, BASE_TEXT_COLUMNS
    if feature_set == "export-current":
        return EXPORT_CURRENT_NUMERIC_FEATURES, EXPORT_TEXT_COLUMNS
    if feature_set == "export-slot":
        return EXPORT_SLOT_NUMERIC_FEATURES, EXPORT_TEXT_COLUMNS
    if feature_set == "popularity":
        return (
            (*BASE_NUMERIC_FEATURES, *POPULARITY_NUMERIC_FEATURES),
            BASE_TEXT_COLUMNS,
        )
    if feature_set == "interactions":
        return (
            BASE_NUMERIC_FEATURES,
            (*BASE_TEXT_COLUMNS, *INTERACTION_TEXT_COLUMNS),
        )
    if feature_set == "metadata":
        return (
            (*BASE_NUMERIC_FEATURES, *METADATA_NUMERIC_FEATURES),
            (*BASE_TEXT_COLUMNS, *METADATA_TEXT_COLUMNS),
        )
    return NUMERIC_FEATURES, TEXT_FEATURE_COLUMNS


def _resolve_device(torch, requested: str):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled torch build or use --device cpu."
        )
    return torch.device(requested)
