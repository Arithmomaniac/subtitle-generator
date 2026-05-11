"""Constrained student for learning popularity source ratios.

The runtime still consumes one collapsed ``popularity_score``. This trainer
learns the source ratios that feed that scalar, while allowing the scalar to
interact with non-popularity text, slot, and metadata features.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from subtitle_generator.config import ALL_TUNABLE_PARAMS, invalidate_config_cache

SOURCES = ("spl", "open_library", "goodreads", "library", "nyt", "trove")
CONFIG_KEYS = {
    "spl": "pop_weight_spl",
    "open_library": "pop_weight_ol",
    "goodreads": "pop_weight_gr",
    "library": "pop_weight_library",
    "nyt": "pop_weight_nyt",
    "trove": "pop_weight_trove",
}
SOURCE_COLUMNS = {
    "spl": "checkouts_per_year",
    "open_library": "ol_edition_count",
    "goodreads": "gr_ratings_count",
    "library": "library_appearances",
    "trove": "trove_library_count",
}
TEXT_COLUMNS = (
    "title",
    "subtitle_text",
    "action_noun",
    "of_object",
    "list_items_text",
    "list_item_pair_text",
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
EXCLUDED_NUMERIC_COLUMNS = {
    "pattern_match_id",
    "subtitle_id",
    "has_popularity_data",
    "work_popularity_score",
    "popularity_comparison_score",
    "max_filler_popularity_score",
    "avg_filler_popularity_score",
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
    "trove_copy_count_is_exact",
}
TARGET_MODES = ("accessibility", "pop-only")
_TOKEN_RE = re.compile(r"[a-z][a-z']+")


@dataclass(frozen=True)
class CalibrationExample:
    pattern_match_id: str
    title: str
    subtitle_text: str
    target: float
    source_scores: list[float]
    base_features: list[float]


@dataclass(frozen=True)
class FitMetrics:
    weights: dict[str, float]
    scalar_mse: float
    student_mse: float | None = None


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
    epochs: int = 300,
    learning_rate: float = 0.03,
    hidden_dim: int = 32,
    hash_dim: int = 256,
    scalar_loss_weight: float = 0.25,
    device: str = "cpu",
) -> PopularityCalibrationResult:
    """Train the constrained popularity-block student and write a report."""

    if target_mode not in TARGET_MODES:
        raise RuntimeError(
            "Unknown target mode. Expected one of: " + ", ".join(TARGET_MODES)
        )
    if not 0 <= min_weight_share < 1 / len(SOURCES):
        raise RuntimeError(f"min_weight_share must be >= 0 and < {1 / len(SOURCES):.3f}")

    torch = _import_torch()
    torch.manual_seed(20260510)

    feature_rows = _read_csv_by_id(features_path)
    teacher_rows = _read_csv_by_id(teacher_predictions_path)
    shared_ids = sorted(set(feature_rows) & set(teacher_rows), key=_sort_key)
    if not shared_ids:
        raise RuntimeError("No shared pattern_match_id values found between inputs")

    percentiles = _build_percentile_models(feature_rows.values())
    numeric_columns = _infer_base_numeric_columns(feature_rows.values())
    scaler = _fit_numeric_scaler(feature_rows, shared_ids, numeric_columns)
    examples = [
        _build_example(
            feature_rows[pattern_id],
            teacher_rows[pattern_id],
            percentiles,
            numeric_columns,
            scaler,
            target_mode,
            hash_dim,
        )
        for pattern_id in shared_ids
        if _has_teacher_scores(teacher_rows[pattern_id])
    ]
    if not examples:
        raise RuntimeError(
            "No examples with teacher score_pop/score_mainstream/score_niche values found"
        )

    current_weights = _current_weights()
    current_metrics = FitMetrics(
        weights=current_weights,
        scalar_mse=_scalar_mse(examples, current_weights),
    )
    learned_metrics = _train_constrained_student(
        torch,
        examples,
        current_weights=current_weights,
        regularization=regularization,
        min_weight_share=min_weight_share,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        scalar_loss_weight=scalar_loss_weight,
        requested_device=device,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "popularity_weight_calibration.md"
    report_path.write_text(
        _format_report(
            examples=examples,
            current_metrics=current_metrics,
            learned_metrics=learned_metrics,
            target_mode=target_mode,
            regularization=regularization,
            min_weight_share=min_weight_share,
            scalar_loss_weight=scalar_loss_weight,
            epochs=epochs,
            hidden_dim=hidden_dim,
            hash_dim=hash_dim,
            numeric_columns=numeric_columns,
            db_path=db_path,
            applied=apply,
        ),
        encoding="utf-8",
    )

    if apply:
        if db_path is None:
            raise RuntimeError("--apply requires --db")
        _apply_weights(db_path, learned_metrics.weights)

    return PopularityCalibrationResult(
        report_path=report_path,
        learned_weights=learned_metrics.weights,
        current_weights=current_weights,
        learned_mse=learned_metrics.scalar_mse,
        current_mse=current_metrics.scalar_mse,
        example_count=len(examples),
        applied=apply,
    )


class _ConstrainedPopularityStudent:
    def __init__(
        self,
        torch,
        *,
        base_dim: int,
        hidden_dim: int,
        initial_weights: dict[str, float],
        min_weight_share: float,
        device,
    ):
        self.torch = torch
        nn = torch.nn
        self.total_weight = sum(initial_weights.values())
        initial_share = _weight_shares(initial_weights)
        floor = min_weight_share
        adjusted = [
            max(1e-6, (initial_share[source] - floor) / (1 - floor * len(SOURCES)))
            for source in SOURCES
        ]
        adjusted_sum = sum(adjusted)
        logits = [math.log(value / adjusted_sum) for value in adjusted]

        class Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.source_logits = nn.Parameter(
                    torch.tensor(logits, dtype=torch.float32, device=device)
                )
                self.base_net = nn.Sequential(
                    nn.Linear(base_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.interaction_net = nn.Sequential(
                    nn.Linear(base_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.popularity_bias = nn.Parameter(torch.tensor(0.0, device=device))
                self.min_weight_share = min_weight_share

            def source_shares(self):
                soft = torch.softmax(self.source_logits, dim=0)
                floor_total = self.min_weight_share * len(SOURCES)
                return self.min_weight_share + (1 - floor_total) * soft

            def forward(self, base_x, source_x):
                popularity_scalar = source_x.matmul(self.source_shares())
                base_score = self.base_net(base_x).squeeze(1)
                interaction_score = self.interaction_net(base_x).squeeze(1)
                prediction = base_score + popularity_scalar * (
                    self.popularity_bias + interaction_score
                )
                return prediction, popularity_scalar

        self.module = Module()

    def weights(self) -> dict[str, float]:
        shares = self.module.source_shares().detach().cpu().tolist()
        return {
            source: self.total_weight * float(shares[index])
            for index, source in enumerate(SOURCES)
        }


def _train_constrained_student(
    torch,
    examples: list[CalibrationExample],
    *,
    current_weights: dict[str, float],
    regularization: float,
    min_weight_share: float,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    scalar_loss_weight: float,
    requested_device: str,
) -> FitMetrics:
    device = torch.device(requested_device)
    base_x = torch.tensor([example.base_features for example in examples], dtype=torch.float32, device=device)
    source_x = torch.tensor([example.source_scores for example in examples], dtype=torch.float32, device=device)
    target = torch.tensor([example.target for example in examples], dtype=torch.float32, device=device)

    model = _ConstrainedPopularityStudent(
        torch,
        base_dim=base_x.shape[1],
        hidden_dim=hidden_dim,
        initial_weights=current_weights,
        min_weight_share=min_weight_share,
        device=device,
    )
    optimizer = torch.optim.Adam(model.module.parameters(), lr=learning_rate)
    prior = torch.tensor(
        [_weight_shares(current_weights)[source] for source in SOURCES],
        dtype=torch.float32,
        device=device,
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction, popularity_scalar = model.module(base_x, source_x)
        shares = model.module.source_shares()
        loss = torch.mean((prediction - target) ** 2)
        loss = loss + scalar_loss_weight * torch.mean((popularity_scalar - target) ** 2)
        loss = loss + regularization * torch.sum((shares - prior) ** 2)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        prediction, popularity_scalar = model.module(base_x, source_x)
        student_mse = float(torch.mean((prediction - target) ** 2).item())
        scalar_mse = float(torch.mean((popularity_scalar - target) ** 2).item())
    return FitMetrics(
        weights=model.weights(),
        scalar_mse=scalar_mse,
        student_mse=student_mse,
    )


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for calibrate-popularity-weights. "
            "Install it with `uv sync --extra ml`."
        ) from exc
    return torch


def _read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = {row["pattern_match_id"]: row for row in csv.DictReader(handle)}
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def _sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (0, value)


def _float(value: str | None) -> float:
    if value in (None, ""):
        return 0.0
    return float(value)


def _is_float(value: str | None) -> bool:
    if value in (None, ""):
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _log1p_values(rows, column: str) -> list[float]:
    values = [math.log10(1 + max(0.0, _float(row.get(column)))) for row in rows]
    values = [value for value in values if value > 0]
    return sorted(values)


def _build_percentile_models(rows) -> dict[str, list[float]]:
    rows = list(rows)
    return {
        "spl": _log1p_values(rows, "checkouts_per_year"),
        "open_library": _log1p_values(rows, "ol_edition_count"),
        "goodreads": _log1p_values(rows, "gr_ratings_count"),
        "library": _log1p_values(rows, "library_appearances"),
        "trove": _log1p_values(rows, "trove_library_count"),
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
    pop = _float(row.get("score_pop"))
    if target_mode == "pop-only":
        return pop
    return pop + 0.5 * _float(row.get("score_mainstream"))


def _has_teacher_scores(row: dict[str, str]) -> bool:
    return any(
        row.get(key) not in (None, "")
        for key in ("score_pop", "score_mainstream", "score_niche")
    )


def _infer_base_numeric_columns(rows) -> tuple[str, ...]:
    rows = list(rows)
    columns: set[str] = set()
    for row in rows:
        columns.update(row)
    numeric_columns = []
    for column in sorted(columns):
        if column in EXCLUDED_NUMERIC_COLUMNS or column in TEXT_COLUMNS:
            continue
        if column in {"candidate_source", "source_group"}:
            continue
        values = [row.get(column) for row in rows[:200]]
        if any(_is_float(value) for value in values):
            numeric_columns.append(column)
    return tuple(numeric_columns)


def _fit_numeric_scaler(
    rows: dict[str, dict[str, str]],
    pattern_ids: list[str],
    numeric_columns: tuple[str, ...],
) -> dict[str, tuple[float, float]]:
    scaler = {}
    for column in numeric_columns:
        values = [_float(rows[pattern_id].get(column)) for pattern_id in pattern_ids]
        scaler[column] = (min(values), max(values))
    return scaler


def _build_example(
    feature_row: dict[str, str],
    teacher_row: dict[str, str],
    percentiles: dict[str, list[float]],
    numeric_columns: tuple[str, ...],
    scaler: dict[str, tuple[float, float]],
    target_mode: str,
    hash_dim: int,
) -> CalibrationExample:
    source_scores = [
        _source_score(feature_row, percentiles, source)
        for source in SOURCES
    ]
    base_features = _base_numeric_features(feature_row, numeric_columns, scaler)
    base_features.extend(_hashed_text_features(feature_row, hash_dim))
    return CalibrationExample(
        pattern_match_id=feature_row["pattern_match_id"],
        title=feature_row.get("title", ""),
        subtitle_text=feature_row.get("subtitle_text", ""),
        target=_teacher_target(teacher_row, target_mode),
        source_scores=source_scores,
        base_features=base_features,
    )


def _source_score(
    row: dict[str, str],
    percentiles: dict[str, list[float]],
    source: str,
) -> float:
    if source == "nyt":
        weeks = _float(row.get("nyt_weeks_on_list"))
        return min(1.0, 0.8 + 0.2 * math.log10(1 + weeks) / 2.0) if weeks > 0 else 0.0
    column = SOURCE_COLUMNS[source]
    value = max(0.0, _float(row.get(column)))
    return _percentile(percentiles[source], math.log10(1 + value))


def _base_numeric_features(
    row: dict[str, str],
    numeric_columns: tuple[str, ...],
    scaler: dict[str, tuple[float, float]],
) -> list[float]:
    features = []
    for column in numeric_columns:
        low, high = scaler[column]
        value = _float(row.get(column))
        features.append(0.0 if high == low else (value - low) / (high - low))
    return features


def _hashed_text_features(row: dict[str, str], hash_dim: int) -> list[float]:
    hashed = [0.0] * hash_dim
    text = " ".join(row.get(column, "") for column in TEXT_COLUMNS).lower()
    for token in _TOKEN_RE.findall(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        hashed[int.from_bytes(digest, "big") % hash_dim] += 1.0
    norm = math.sqrt(sum(value * value for value in hashed)) or 1.0
    return [value / norm for value in hashed]


def _current_weights() -> dict[str, float]:
    return {
        source: float(ALL_TUNABLE_PARAMS[CONFIG_KEYS[source]])
        for source in SOURCES
    }


def _weight_shares(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, weights[source]) for source in SOURCES)
    if total <= 0:
        return {source: 1 / len(SOURCES) for source in SOURCES}
    return {source: max(0.0, weights[source]) / total for source in SOURCES}


def _scalar_score(example: CalibrationExample, weights: dict[str, float]) -> float:
    shares = _weight_shares(weights)
    return sum(
        shares[source] * example.source_scores[index]
        for index, source in enumerate(SOURCES)
    )


def _scalar_mse(examples: list[CalibrationExample], weights: dict[str, float]) -> float:
    return mean(
        (_scalar_score(example, weights) - example.target) ** 2
        for example in examples
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
    current_metrics: FitMetrics,
    learned_metrics: FitMetrics,
    target_mode: str,
    regularization: float,
    min_weight_share: float,
    scalar_loss_weight: float,
    epochs: int,
    hidden_dim: int,
    hash_dim: int,
    numeric_columns: tuple[str, ...],
    db_path: Path | None,
    applied: bool,
) -> str:
    equal_weights = {
        source: sum(current_metrics.weights.values()) / len(SOURCES)
        for source in SOURCES
    }
    fits = [
        ("current scalar", current_metrics.scalar_mse),
        ("learned scalar", learned_metrics.scalar_mse),
        ("learned constrained student", learned_metrics.student_mse or 0.0),
        ("equal scalar", _scalar_mse(examples, equal_weights)),
    ]
    for source in SOURCES:
        source_only = {
            candidate: (sum(current_metrics.weights.values()) if candidate == source else 0.0)
            for candidate in SOURCES
        }
        fits.append((f"{source}-only scalar", _scalar_mse(examples, source_only)))

    deltas = sorted(
        (
            (
                abs(_scalar_score(example, learned_metrics.weights) - _scalar_score(example, current_metrics.weights)),
                example,
                _scalar_score(example, current_metrics.weights),
                _scalar_score(example, learned_metrics.weights),
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
        f"- Training epochs: {epochs:,}",
        f"- Hidden dimension: {hidden_dim}",
        f"- Text hash dimension: {hash_dim}",
        f"- Base numeric features: {len(numeric_columns)}",
        f"- Regularization toward current source shares: {regularization:g}",
        f"- Auxiliary scalar-alignment loss weight: {scalar_loss_weight:g}",
        f"- Minimum learned source share: {min_weight_share:g}",
        f"- Applied to DB: {'yes' if applied else 'no'}",
        f"- DB: `{db_path}`" if db_path else "- DB: not provided",
        "",
        "## Model shape",
        "",
        "The constrained student computes one shared popularity scalar:",
        "",
        "`popularity_scalar = sum(source_weight_i * normalized_source_i)`",
        "",
        "The scalar then enters the prediction model directly and through shared",
        "interactions with text, slot, and metadata features. Source-specific values",
        "do not get separate text/slot/metadata interaction weights, so the learned",
        "source coefficients remain collapsible into runtime `pop_weight_*` values.",
        "",
        "## Weights",
        "",
        "| Source | Current | Learned |",
        "|---|---:|---:|",
    ]
    for source in SOURCES:
        lines.append(
            f"| {source} | {current_metrics.weights[source]:.4f} | {learned_metrics.weights[source]:.4f} |"
        )

    lines.extend([
        "",
        "## Fit comparison",
        "",
        "| Model | MSE |",
        "|---|---:|",
    ])
    for label, mse in fits:
        lines.append(f"| {label} | {mse:.6f} |")

    lines.extend([
        "",
        "## Largest scalar deltas",
        "",
        "| Pattern | Current scalar | Learned scalar | Target | Title |",
        "|---|---:|---:|---:|---|",
    ])
    for _, example, current_score, learned_score in reversed(deltas):
        title = (example.title or example.subtitle_text).replace("|", "\\|")
        lines.append(
            f"| {example.pattern_match_id} | {current_score:.3f} | {learned_score:.3f} | {example.target:.3f} | {title[:80]} |"
        )
    lines.append("")
    return "\n".join(lines)
