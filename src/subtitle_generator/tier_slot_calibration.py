"""Calibrate the served-shaped tier-slot filler distributions (epic #33, step 6).

The Step 3-5 builder emits sums-to-1 ``P(filler | tier, slot_type)`` distributions,
but nothing checks whether the *shape* of those probabilities matches reality. A
distribution can be perfectly normalized yet too **sharp** (over-concentrated ->
repetitive generation) or too **flat** (tiers lose distinctive vocabulary). This
module calibrates that shape against held-out source/filler evidence using
**tier-specific temperature scaling** and reports the result in plain English.

In plain English: each tier has a *confidence dial* for how hard it leans on its
favourite words in a slot. Too low -> repetitive; too high -> the tiers blur into
one generic voice. Calibration tunes that dial per tier by hiding a slice of the
source books and picking the dial that best predicts the words those hidden books
actually used. Temperature is monotonic, so it only changes *how often* words are
picked, never *which* ranks first.

Everything here is analysis-only and side-by-side: the served runtime artifact
``tier_slot_filler_distribution_v1.csv`` is never modified. Calibrated output goes
to a ``*.calibrated.csv`` sidecar plus versioned, replayable metadata.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from subtitle_generator.schema_contracts import (
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    TIER_SLOT_FILLER_DISTRIBUTION_TIERS,
)
from subtitle_generator.tier_slot_distribution import (
    DISTRIBUTION_COLUMNS,
    DistributionInputs,
    _format_csv_value,
    _js_divergence,
    _validate_rows,
    build_anchored_rows,
    load_distribution_inputs,
    load_strict_source_links,
)

TIERS = TIER_SLOT_FILLER_DISTRIBUTION_TIERS
CALIBRATION_METADATA_SCHEMA_VERSION = 1

# Temperature search bounds. The upper bound is the distinctiveness guardrail:
# temperatures above this are not allowed to flatten a tier so far that it loses
# its signature vocabulary. T=1.0 is always evaluated, so calibration can choose
# "leave it alone" whenever scaling does not improve the held-out fit.
DEFAULT_TEMPERATURE_MIN = 0.25
DEFAULT_TEMPERATURE_MAX = 3.0

GroupKey = tuple[str, str]  # (slot_type, tier)
CellKey = tuple[str, str, str]  # (slot_type, tier, filler)


# ---------------------------------------------------------------------------
# Config + result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationConfig:
    """One calibration experiment.

    ``granularity`` controls how many temperatures are fitted:
      - ``none``           -> T=1 everywhere (the uncalibrated baseline).
      - ``global``         -> a single temperature shared by every group.
      - ``per_tier``       -> one temperature per tier (the issue's first lever).
      - ``per_tier_slot``  -> one temperature per (tier, slot_type) group.
    """

    name: str
    granularity: str = "per_tier"
    folds: int = 5
    seed: int = 20260612
    alpha: float = 0.5
    inferred_source_weight: float = 1.0
    temperature_min: float = DEFAULT_TEMPERATURE_MIN
    temperature_max: float = DEFAULT_TEMPERATURE_MAX
    artifact_version: str = "tier_slot_filler_distribution_v1"


@dataclass(frozen=True)
class CalibrationResult:
    distribution_path: Path
    metadata_path: Path
    report_path: Path
    row_count: int
    temperatures: dict[str, float]


@dataclass(frozen=True)
class CalibrationAblationResult:
    report_path: Path
    metrics_path: Path
    experiment_count: int


@dataclass(frozen=True)
class CalibrationAutoResearcherResult:
    report_path: Path
    proposals_path: Path
    ablation_result: CalibrationAblationResult


@dataclass
class _FoldData:
    """Per-fold cross-validation state.

    ``group_base`` carries the complement-trained (analysis-only) rows so the
    held-out fold can be scored against a distribution that never saw it.
    """

    index: int
    train_ids: set[int]
    heldout_ids: set[int]
    group_base: dict[GroupKey, list[dict[str, object]]]
    counts: dict[CellKey, int]


def default_calibration_experiments() -> tuple[CalibrationConfig, ...]:
    """Bounded calibration sweep for the AutoResearcher loop.

    Temperature scaling is evaluated first (issue #39); isotonic/Platt are only
    worth adding if temperature is insufficient, so they are intentionally left
    out of the default grid and documented as the staged fallback.
    """

    return (
        CalibrationConfig("baseline_T1", "none"),
        CalibrationConfig("global_temperature", "global"),
        CalibrationConfig("per_tier_temperature", "per_tier"),
        CalibrationConfig("per_tier_slot_temperature", "per_tier_slot"),
    )


# ---------------------------------------------------------------------------
# Held-out source split + evidence
# ---------------------------------------------------------------------------


def assign_source_folds(
    subtitle_ids: set[int] | list[int],
    *,
    folds: int,
    seed: int,
) -> dict[int, int]:
    """Deterministically assign each source to a cross-validation fold.

    A seeded hash keeps the split replayable: the same ``seed`` + ``folds``
    always reproduce the same assignment, which is what makes the calibration
    parameters replayable per the exit gate.
    """

    if folds < 2:
        raise RuntimeError("folds must be at least 2")
    assignment: dict[int, int] = {}
    for subtitle_id in sorted(subtitle_ids):
        digest = hashlib.sha256(f"{seed}:{subtitle_id}".encode()).hexdigest()
        assignment[int(subtitle_id)] = int(digest, 16) % folds
    return assignment


def heldout_evidence_counts(
    inputs: DistributionInputs,
    links: list[tuple[int, int]],
    heldout_ids: set[int],
) -> dict[CellKey, int]:
    """Count held-out ``(slot_type, tier, filler)`` occurrences.

    A held-out source labeled tier ``T`` contributes each strict filler it used
    as one sample from ``P(filler | T, slot)`` -- the proxy validation target.
    Only sources with a usable hard tier label are scored.
    """

    counts: Counter[CellKey] = Counter()
    for filler_id, subtitle_id in links:
        if subtitle_id not in heldout_ids:
            continue
        label = inputs.source_labels.get(subtitle_id)
        if label is None or label.tier not in TIERS:
            continue
        filler_key = inputs.filler_id_to_key.get(filler_id)
        if filler_key is None:
            continue
        filler = inputs.fillers.get(filler_key)
        if filler is None:
            continue
        counts[(filler.slot_type, label.tier, filler.filler)] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------


def apply_temperature(probabilities: dict[str, float], temperature: float) -> dict[str, float]:
    """Temperature-scale a categorical distribution and renormalize.

    ``q_i = p_i^(1/T) / sum_j p_j^(1/T)``. ``T>1`` flattens (more variety),
    ``T<1`` sharpens (more repetition), ``T=1`` is identity. Monotonic in ``p``,
    so the ranking of fillers within the group is preserved.
    """

    if temperature <= 0:
        raise RuntimeError("temperature must be positive")
    if temperature == 1.0:
        return dict(probabilities)
    exponent = 1.0 / temperature
    powered = {
        filler: (prob**exponent if prob > 0 else 0.0)
        for filler, prob in probabilities.items()
    }
    total = sum(powered.values())
    if total <= 0:
        return dict(probabilities)
    return {filler: value / total for filler, value in powered.items()}


def _group_probabilities(rows: list[dict[str, object]]) -> dict[GroupKey, dict[str, float]]:
    grouped: dict[GroupKey, dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["slot_type"]), str(row["tier"]))
        grouped[key][str(row["filler"])] = float(row["probability"])
    return grouped


def _group_rows(rows: list[dict[str, object]]) -> dict[GroupKey, list[dict[str, object]]]:
    grouped: dict[GroupKey, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    return grouped


def _group_nll(
    base: dict[str, float],
    counts: dict[CellKey, int],
    group: GroupKey,
    temperature: float,
) -> float:
    """Negative log-likelihood of a group's held-out fillers under temperature."""

    if not base:
        return 0.0
    scaled = apply_temperature(base, temperature)
    slot_type, tier = group
    total = 0.0
    for filler, count in counts.items():
        if filler[0] != slot_type or filler[1] != tier:
            continue
        prob = scaled.get(filler[2], 0.0)
        if count:
            total += -count * math.log(prob) if prob > 0 else count * 1e9
    return total


def _minimize_temperature(objective, lo: float, hi: float) -> float:
    """Bounded 1-D search for the temperature minimizing ``objective``.

    Deterministic coarse log-grid plus a golden-section style refine. ``T=1.0``
    is always a candidate so calibration can decline to change a group.
    """

    candidates = [1.0]
    grid = 24
    ratio = hi / lo
    candidates.extend(lo * ratio ** (index / grid) for index in range(grid + 1))
    best_temperature = min(candidates, key=objective)
    best_value = objective(best_temperature)

    # Local refine around the best grid point, in log space.
    step = ratio ** (1.0 / grid)
    left = max(lo, best_temperature / step)
    right = min(hi, best_temperature * step)
    for _ in range(48):
        if right / left < 1.0001:
            break
        mid_left = left * (right / left) ** (1.0 / 3.0)
        mid_right = left * (right / left) ** (2.0 / 3.0)
        if objective(mid_left) <= objective(mid_right):
            right = mid_right
        else:
            left = mid_left
    refined = math.sqrt(left * right)
    if objective(refined) < best_value:
        return refined
    return best_temperature


def _fit_temperatures(
    folds: list[_FoldData],
    granularity: str,
    *,
    groups: list[GroupKey],
    temperature_min: float,
    temperature_max: float,
) -> dict[GroupKey, float]:
    """Fit temperatures by minimizing summed cross-validation held-out NLL."""

    if granularity == "none":
        return {group: 1.0 for group in groups}

    def fold_nll(group_subset: list[GroupKey], temperature: float) -> float:
        total = 0.0
        for fold in folds:
            for group in group_subset:
                base = _group_probabilities_for(fold.group_base.get(group, []))
                total += _group_nll(base, fold.counts, group, temperature)
        return total

    if granularity == "global":
        temperature = _minimize_temperature(
            lambda value: fold_nll(groups, value),
            temperature_min,
            temperature_max,
        )
        return {group: temperature for group in groups}

    if granularity == "per_tier":
        result: dict[GroupKey, float] = {}
        for tier in TIERS:
            tier_groups = [group for group in groups if group[1] == tier]
            if not tier_groups:
                continue
            temperature = _minimize_temperature(
                lambda value, tier_groups=tier_groups: fold_nll(tier_groups, value),
                temperature_min,
                temperature_max,
            )
            for group in tier_groups:
                result[group] = temperature
        return result

    if granularity == "per_tier_slot":
        result = {}
        for group in groups:
            result[group] = _minimize_temperature(
                lambda value, group=group: fold_nll([group], value),
                temperature_min,
                temperature_max,
            )
        return result

    raise RuntimeError(f"Unknown calibration granularity: {granularity}")


def _group_probabilities_for(rows: list[dict[str, object]]) -> dict[str, float]:
    return {str(row["filler"]): float(row["probability"]) for row in rows}


# ---------------------------------------------------------------------------
# Cross-validation preparation
# ---------------------------------------------------------------------------


def _prepare_folds(
    conn: sqlite3.Connection,
    inputs: DistributionInputs,
    links: list[tuple[int, int]],
    *,
    folds: int,
    seed: int,
    alpha: float,
    inferred_source_weight: float,
    artifact_version: str,
) -> tuple[list[_FoldData], dict[int, int]]:
    source_ids = {subtitle_id for _filler_id, subtitle_id in links}
    assignment = assign_source_folds(source_ids, folds=folds, seed=seed)
    fold_to_ids: dict[int, set[int]] = defaultdict(set)
    for subtitle_id, fold_index in assignment.items():
        fold_to_ids[fold_index].add(subtitle_id)

    prepared: list[_FoldData] = []
    for fold_index in range(folds):
        heldout_ids = fold_to_ids.get(fold_index, set())
        if not heldout_ids:
            continue
        train_ids = source_ids - heldout_ids
        train_rows = build_anchored_rows(
            conn,
            inputs,
            include_subtitle_ids=train_ids,
            alpha=alpha,
            inferred_source_weight=inferred_source_weight,
            artifact_version=artifact_version,
        )
        prepared.append(
            _FoldData(
                index=fold_index,
                train_ids=train_ids,
                heldout_ids=heldout_ids,
                group_base=_group_rows(train_rows),
                counts=heldout_evidence_counts(inputs, links, heldout_ids),
            )
        )
    if not prepared:
        raise RuntimeError("No held-out folds carried evidence; cannot calibrate")
    return prepared, assignment


# ---------------------------------------------------------------------------
# Metrics: held-out likelihood, reliability/ECE, ranking, distance
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    slot_type: str
    tier: str
    predicted: float
    empirical: float
    count: int
    group_total: int
    source_count: int
    head: bool


def _collect_cells(
    folds: list[_FoldData],
    temperatures: dict[GroupKey, float],
    *,
    head_fraction: float = 0.2,
) -> list[_Cell]:
    """Pool per-fold, per-cell calibration observations across the CV folds."""

    cells: list[_Cell] = []
    for fold in folds:
        for group, rows in fold.group_base.items():
            slot_type, tier = group
            group_total = sum(
                count
                for (cell_slot, cell_tier, _filler), count in fold.counts.items()
                if cell_slot == slot_type and cell_tier == tier
            )
            if group_total <= 0:
                continue
            base = {str(row["filler"]): float(row["probability"]) for row in rows}
            scaled = apply_temperature(base, temperatures.get(group, 1.0))
            ranked = sorted(base, key=lambda filler: base[filler], reverse=True)
            head_size = max(1, int(round(len(ranked) * head_fraction)))
            head_set = set(ranked[:head_size])
            source_counts = {
                str(row["filler"]): int(row.get("source_count", 0) or 0) for row in rows
            }
            for filler, predicted in scaled.items():
                count = fold.counts.get((slot_type, tier, filler), 0)
                cells.append(
                    _Cell(
                        slot_type=slot_type,
                        tier=tier,
                        predicted=predicted,
                        empirical=count / group_total,
                        count=count,
                        group_total=group_total,
                        source_count=source_counts.get(filler, 0),
                        head=filler in head_set,
                    )
                )
    return cells


def _heldout_nll(cells: list[_Cell]) -> float:
    total = 0.0
    for cell in cells:
        if cell.count:
            total += (
                -cell.count * math.log(cell.predicted)
                if cell.predicted > 0
                else cell.count * 1e9
            )
    return total


def _expected_calibration_error(cells: list[_Cell], *, bins: int = 10) -> float:
    """ECE-style reliability gap: binned predicted prob vs held-out frequency.

    Each cell is weighted by its group's held-out evidence so well-attested
    ``(tier, slot)`` groups dominate. Within each predicted-probability bin we
    compare the evidence-weighted mean predicted mass to the evidence-weighted
    mean empirical frequency; ECE is the weighted average gap across bins.
    """

    weighted = [(cell, float(cell.group_total)) for cell in cells]
    total_weight = sum(weight for _cell, weight in weighted)
    if total_weight <= 0:
        return 0.0
    binned: dict[int, list[tuple[_Cell, float]]] = defaultdict(list)
    for cell, weight in weighted:
        index = min(bins - 1, int(cell.predicted * bins))
        binned[index].append((cell, weight))
    ece = 0.0
    for entries in binned.values():
        bin_weight = sum(weight for _cell, weight in entries)
        if bin_weight <= 0:
            continue
        mean_pred = sum(cell.predicted * weight for cell, weight in entries) / bin_weight
        mean_emp = sum(cell.empirical * weight for cell, weight in entries) / bin_weight
        ece += (bin_weight / total_weight) * abs(mean_pred - mean_emp)
    return ece


def _top1_hit_rate(folds: list[_FoldData], temperatures: dict[GroupKey, float]) -> float:
    """Fraction of held-out groups whose argmax filler matches the held-out mode.

    Temperature scaling is monotonic, so the argmax is identical before and
    after calibration -- this is reported to show ranking is preserved.
    """

    hits = 0
    total = 0
    for fold in folds:
        group_counts: dict[GroupKey, Counter[str]] = defaultdict(Counter)
        for (slot_type, tier, filler), count in fold.counts.items():
            group_counts[(slot_type, tier)][filler] += count
        for group, counter in group_counts.items():
            rows = fold.group_base.get(group, [])
            if not rows or not counter:
                continue
            base = {str(row["filler"]): float(row["probability"]) for row in rows}
            predicted_top = max(base, key=lambda filler: base[filler])
            empirical_top = counter.most_common(1)[0][0]
            total += 1
            if predicted_top == empirical_top:
                hits += 1
    return hits / total if total else 0.0


def _entropy(probabilities: list[float]) -> float:
    return -sum(prob * math.log(prob) for prob in probabilities if prob > 0)


def _distinctiveness(group_probs: dict[GroupKey, dict[str, float]]) -> float:
    """Mean cross-tier JS divergence per slot (higher = tiers more distinct)."""

    slots = {slot for slot, _tier in group_probs}
    divergences: list[float] = []
    for slot in slots:
        tier_vectors = {
            tier: group_probs.get((slot, tier), {})
            for tier in TIERS
            if (slot, tier) in group_probs
        }
        present = [tier for tier in TIERS if tier_vectors.get(tier)]
        for left_index in range(len(present)):
            for right_index in range(left_index + 1, len(present)):
                left = tier_vectors[present[left_index]]
                right = tier_vectors[present[right_index]]
                fillers = sorted(set(left) | set(right))
                if not fillers:
                    continue
                divergences.append(
                    _js_divergence(
                        [left.get(filler, 0.0) for filler in fillers],
                        [right.get(filler, 0.0) for filler in fillers],
                    )
                )
    return sum(divergences) / len(divergences) if divergences else 0.0


@dataclass
class _CalibrationMetrics:
    temperatures: dict[GroupKey, float]
    heldout_nll_baseline: float
    heldout_nll_calibrated: float
    ece_baseline: float
    ece_calibrated: float
    top1_hit_rate: float
    mean_effective_n_baseline: float
    mean_effective_n_calibrated: float
    distinctiveness_baseline: float
    distinctiveness_calibrated: float
    slices: dict[str, dict[str, float]]
    per_tier_effective_n: dict[str, dict[str, float]]


def _slice_ece(cells: list[_Cell]) -> dict[str, dict[str, float]]:
    """ECE broken out by tier, slot type, head vs tail, and evidence strength."""

    slices: dict[str, dict[str, float]] = {}

    def record(group_label: str, key: str, subset: list[_Cell]) -> None:
        if subset:
            slices.setdefault(group_label, {})[key] = _expected_calibration_error(subset)

    for tier in TIERS:
        record("tier", tier, [cell for cell in cells if cell.tier == tier])
    for slot in sorted({cell.slot_type for cell in cells}):
        record("slot_type", slot, [cell for cell in cells if cell.slot_type == slot])
    record("filler_mass", "head", [cell for cell in cells if cell.head])
    record("filler_mass", "tail", [cell for cell in cells if not cell.head])
    record("evidence", "high (src>=3)", [cell for cell in cells if cell.source_count >= 3])
    record("evidence", "low (src<3)", [cell for cell in cells if cell.source_count < 3])
    return slices


def _compute_metrics(
    folds: list[_FoldData],
    full_group_probs: dict[GroupKey, dict[str, float]],
    temperatures: dict[GroupKey, float],
) -> _CalibrationMetrics:
    baseline_temps = {group: 1.0 for group in temperatures}
    base_cells = _collect_cells(folds, baseline_temps)
    cal_cells = _collect_cells(folds, temperatures)

    calibrated_full = {
        group: apply_temperature(probs, temperatures.get(group, 1.0))
        for group, probs in full_group_probs.items()
    }

    per_tier_effective_n: dict[str, dict[str, float]] = {}
    base_effective: list[float] = []
    cal_effective: list[float] = []
    for tier in TIERS:
        base_values: list[float] = []
        cal_values: list[float] = []
        for group, probs in full_group_probs.items():
            if group[1] != tier:
                continue
            base_values.append(math.exp(_entropy(list(probs.values()))))
            cal_values.append(
                math.exp(_entropy(list(calibrated_full[group].values())))
            )
        if base_values:
            per_tier_effective_n[tier] = {
                "baseline": sum(base_values) / len(base_values),
                "calibrated": sum(cal_values) / len(cal_values),
            }
            base_effective.extend(base_values)
            cal_effective.extend(cal_values)

    return _CalibrationMetrics(
        temperatures=temperatures,
        heldout_nll_baseline=_heldout_nll(base_cells),
        heldout_nll_calibrated=_heldout_nll(cal_cells),
        ece_baseline=_expected_calibration_error(base_cells),
        ece_calibrated=_expected_calibration_error(cal_cells),
        top1_hit_rate=_top1_hit_rate(folds, temperatures),
        mean_effective_n_baseline=(
            sum(base_effective) / len(base_effective) if base_effective else 0.0
        ),
        mean_effective_n_calibrated=(
            sum(cal_effective) / len(cal_effective) if cal_effective else 0.0
        ),
        distinctiveness_baseline=_distinctiveness(full_group_probs),
        distinctiveness_calibrated=_distinctiveness(calibrated_full),
        slices=_slice_ece(cal_cells),
        per_tier_effective_n=per_tier_effective_n,
    )


# ---------------------------------------------------------------------------
# Calibrated artifact + metadata
# ---------------------------------------------------------------------------


def _calibrated_rows(
    full_rows: list[dict[str, object]],
    temperatures: dict[GroupKey, float],
) -> list[dict[str, object]]:
    grouped = _group_rows(full_rows)
    calibrated: list[dict[str, object]] = []
    for group, rows in grouped.items():
        temperature = temperatures.get(group, 1.0)
        base = {str(row["filler"]): float(row["probability"]) for row in rows}
        scaled = apply_temperature(base, temperature)
        for row in rows:
            new_row = dict(row)
            probability = scaled.get(str(row["filler"]), 0.0)
            new_row["probability"] = probability
            new_row["log_probability"] = (
                math.log(probability) if probability > 0 else float("-inf")
            )
            new_row["calibration_temperature"] = temperature
            calibrated.append(new_row)
    return calibrated


def _write_calibrated(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISTRIBUTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {column: _format_csv_value(row[column]) for column in DISTRIBUTION_COLUMNS}
            )


def _temperatures_by_label(temperatures: dict[GroupKey, float], granularity: str) -> dict[str, float]:
    if granularity == "per_tier_slot":
        return {f"{tier}|{slot}": value for (slot, tier), value in sorted(temperatures.items())}
    by_tier: dict[str, float] = {}
    for (_slot, tier), value in temperatures.items():
        by_tier[tier] = value
    return {tier: by_tier[tier] for tier in TIERS if tier in by_tier}


def _fold_digest(assignment: dict[int, int]) -> str:
    payload = ";".join(f"{sid}:{fold}" for sid, fold in sorted(assignment.items()))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _build_metadata(
    config: CalibrationConfig,
    assignment: dict[int, int],
    inputs: DistributionInputs,
    metrics: _CalibrationMetrics,
) -> dict[str, object]:
    labeled = sum(
        1
        for subtitle_id in assignment
        if (label := inputs.source_labels.get(subtitle_id)) is not None
        and label.tier in TIERS
    )
    per_fold = Counter(assignment.values())
    return {
        "schema_version": CALIBRATION_METADATA_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "name": config.name,
            "granularity": config.granularity,
            "folds": config.folds,
            "seed": config.seed,
            "alpha": config.alpha,
            "inferred_source_weight": config.inferred_source_weight,
            "temperature_min": config.temperature_min,
            "temperature_max": config.temperature_max,
            "artifact_version": config.artifact_version,
        },
        "fold_assignment_digest": _fold_digest(assignment),
        "source_counts": {
            "total": len(assignment),
            "labeled": labeled,
            "per_fold": {str(fold): count for fold, count in sorted(per_fold.items())},
        },
        "temperatures": _temperatures_by_label(metrics.temperatures, config.granularity),
        "heldout_nll": {
            "baseline": metrics.heldout_nll_baseline,
            "calibrated": metrics.heldout_nll_calibrated,
            "improvement": metrics.heldout_nll_baseline - metrics.heldout_nll_calibrated,
        },
        "reliability_ece": {
            "baseline": metrics.ece_baseline,
            "calibrated": metrics.ece_calibrated,
        },
        "ranking": {"top1_hit_rate": metrics.top1_hit_rate, "ranking_preserved": True},
        "effective_n": {
            "baseline": metrics.mean_effective_n_baseline,
            "calibrated": metrics.mean_effective_n_calibrated,
            "per_tier": metrics.per_tier_effective_n,
        },
        "distinctiveness": {
            "baseline_mean_cross_tier_js": metrics.distinctiveness_baseline,
            "calibrated_mean_cross_tier_js": metrics.distinctiveness_calibrated,
        },
        "ece_slices": metrics.slices,
    }


# ---------------------------------------------------------------------------
# Plain-English helpers + report
# ---------------------------------------------------------------------------


def _behaviour_phrase(metrics: _CalibrationMetrics) -> str:
    base = metrics.mean_effective_n_baseline
    cal = metrics.mean_effective_n_calibrated
    if base <= 0:
        return "left the distributions unchanged"
    delta = (cal - base) / base
    if delta > 0.02:
        return (
            "made generation **more varied** (flatter distributions, more fillers "
            "share the mass)"
        )
    if delta < -0.02:
        return (
            "made generation **more repetitive** (sharper distributions, the top "
            "fillers carry more mass)"
        )
    return (
        "left variety roughly unchanged and **merely re-balanced** the mass "
        "(ranking preserved)"
    )


def calibration_verdict(metrics: _CalibrationMetrics) -> dict[str, object]:
    """Summarize the exit-gate signals as a small machine-readable verdict."""

    improved = metrics.heldout_nll_calibrated <= metrics.heldout_nll_baseline + 1e-9
    distinct_drop = (
        (metrics.distinctiveness_baseline - metrics.distinctiveness_calibrated)
        / metrics.distinctiveness_baseline
        if metrics.distinctiveness_baseline > 0
        else 0.0
    )
    return {
        "heldout_likelihood_improved_or_preserved": improved,
        "ranking_preserved": True,
        "distinctiveness_drop_fraction": distinct_drop,
        "tiers_kept_distinct": distinct_drop <= 0.15,
        "behaviour": _behaviour_phrase(metrics),
    }


def _format_report(
    config: CalibrationConfig,
    metrics: _CalibrationMetrics,
    metadata: dict[str, object],
    *,
    distribution_path: Path,
    metadata_path: Path,
) -> str:
    verdict = calibration_verdict(metrics)
    nll_delta = metrics.heldout_nll_baseline - metrics.heldout_nll_calibrated
    lines: list[str] = []
    lines.append("# Tier-slot distribution calibration (step 6, #39)")
    lines.append("")
    lines.append(
        "Analysis-only. The served `tier_slot_filler_distribution_v1.csv` is "
        "untouched; calibrated probabilities are written side-by-side to "
        f"`{distribution_path.name}` with replayable metadata in "
        f"`{metadata_path.name}`."
    )
    lines.append("")
    lines.append("## In plain English")
    lines.append("")
    lines.append(
        "Each tier has a *confidence dial* for how hard it leans on its favourite "
        "words. Calibration tuned that dial against books held out of training."
    )
    lines.append(f"- Calibration {verdict['behaviour']}.")
    lines.append(
        "- It **only changed how often words are picked, never which ranks first** "
        f"(top-1 held-out match rate {metrics.top1_hit_rate:.0%}, identical before "
        "and after by construction)."
    )
    kept = "yes" if verdict["tiers_kept_distinct"] else "**NO -- review**"
    lines.append(
        "- Did the tiers stay distinct (pop vs mainstream vs niche)? "
        f"{kept} (cross-tier separation moved "
        f"{metrics.distinctiveness_baseline:.4f} -> "
        f"{metrics.distinctiveness_calibrated:.4f})."
    )
    lines.append("")
    lines.append("## Fitted temperatures")
    lines.append("")
    lines.append("`T<1` sharpens (more repetition); `T>1` flattens (more variety).")
    lines.append("")
    lines.append("| group | temperature |")
    lines.append("|---|---|")
    for label, value in metadata["temperatures"].items():  # type: ignore[index]
        lines.append(f"| {label} | {value:.4f} |")
    lines.append("")
    lines.append("## Held-out fit (5-fold cross-validation)")
    lines.append("")
    lines.append("| metric | baseline (T=1) | calibrated | change |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| held-out NLL | {metrics.heldout_nll_baseline:.2f} | "
        f"{metrics.heldout_nll_calibrated:.2f} | {nll_delta:+.2f} |"
    )
    lines.append(
        f"| reliability ECE | {metrics.ece_baseline:.6f} | "
        f"{metrics.ece_calibrated:.6f} | "
        f"{metrics.ece_calibrated - metrics.ece_baseline:+.6f} |"
    )
    lines.append(
        f"| mean effective-N | {metrics.mean_effective_n_baseline:.2f} | "
        f"{metrics.mean_effective_n_calibrated:.2f} | "
        f"{metrics.mean_effective_n_calibrated - metrics.mean_effective_n_baseline:+.2f} |"
    )
    lines.append("")
    lines.append("## Reliability (ECE) by slice")
    lines.append("")
    lines.append(
        "Lower is better. ECE compares predicted mass to held-out frequency, "
        "evidence-weighted, for the calibrated distribution. Absolute gaps are "
        "tiny because most fillers sit in the long tail; the calibration win is "
        "in held-out likelihood and shape, not in moving an already-small ECE."
    )
    for group_label, entries in metrics.slices.items():
        lines.append("")
        lines.append(f"**{group_label}**")
        lines.append("")
        lines.append("| bucket | ECE |")
        lines.append("|---|---|")
        for key, value in entries.items():
            lines.append(f"| {key} | {value:.6f} |")
    lines.append("")
    lines.append("## Per-tier variety (effective-N over the served-shaped build)")
    lines.append("")
    lines.append("| tier | baseline | calibrated |")
    lines.append("|---|---|---|")
    for tier, values in metrics.per_tier_effective_n.items():
        lines.append(
            f"| {tier} | {values['baseline']:.2f} | {values['calibrated']:.2f} |"
        )
    lines.append("")
    lines.append("## Exit-gate check")
    lines.append("")
    likelihood = (
        "PASS" if verdict["heldout_likelihood_improved_or_preserved"] else "FAIL"
    )
    distinct = "PASS" if verdict["tiers_kept_distinct"] else "FAIL"
    lines.append(
        f"- Held-out likelihood improved or preserved: **{likelihood}** "
        f"({nll_delta:+.2f} NLL)."
    )
    lines.append("- Ranking preserved (temperature is monotonic): **PASS**.")
    lines.append(
        f"- Tiers keep distinctive vocabulary: **{distinct}** "
        f"({verdict['distinctiveness_drop_fraction']:.1%} cross-tier drop)."
    )
    lines.append(
        "- Calibration parameters versioned and replayable: **PASS** "
        f"(`{metadata_path.name}`, fold digest "
        f"`{metadata['fold_assignment_digest']}`)."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def build_tier_slot_calibration(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    config: CalibrationConfig | None = None,
) -> CalibrationResult:
    """Fit calibration temperatures and emit the calibrated side-by-side artifact."""

    config = config or CalibrationConfig("per_tier_temperature", "per_tier")
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_distribution_inputs(conn)
    links = load_strict_source_links(conn)
    folds, assignment = _prepare_folds(
        conn,
        inputs,
        links,
        folds=config.folds,
        seed=config.seed,
        alpha=config.alpha,
        inferred_source_weight=config.inferred_source_weight,
        artifact_version=config.artifact_version,
    )
    full_rows = build_anchored_rows(
        conn,
        inputs,
        include_subtitle_ids=None,
        alpha=config.alpha,
        inferred_source_weight=config.inferred_source_weight,
        artifact_version=config.artifact_version,
    )
    groups = sorted(_group_probabilities(full_rows))
    temperatures = _fit_temperatures(
        folds,
        config.granularity,
        groups=groups,
        temperature_min=config.temperature_min,
        temperature_max=config.temperature_max,
    )
    metrics = _compute_metrics(folds, _group_probabilities(full_rows), temperatures)

    calibrated = _calibrated_rows(full_rows, temperatures)
    distribution_path = (
        output_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.calibrated.csv"
    )
    _write_calibrated(distribution_path, calibrated)
    _validate_rows(conn, calibrated)

    metadata = _build_metadata(config, assignment, inputs, metrics)
    metadata_path = output_dir / "tier_slot_calibration_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "tier_slot_calibration_report.md"
    report_path.write_text(
        _format_report(
            config,
            metrics,
            metadata,
            distribution_path=distribution_path,
            metadata_path=metadata_path,
        ),
        encoding="utf-8",
    )
    return CalibrationResult(
        distribution_path=distribution_path,
        metadata_path=metadata_path,
        report_path=report_path,
        row_count=len(calibrated),
        temperatures=_temperatures_by_label(temperatures, config.granularity),
    )


def run_calibration_ablation(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    configs: tuple[CalibrationConfig, ...] | None = None,
    folds: int = 5,
    seed: int = 20260612,
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
) -> CalibrationAblationResult:
    """Sweep calibration granularities over one shared CV split."""

    output_dir.mkdir(parents=True, exist_ok=True)
    configs = configs or default_calibration_experiments()

    inputs = load_distribution_inputs(conn)
    links = load_strict_source_links(conn)
    fold_data, _assignment = _prepare_folds(
        conn,
        inputs,
        links,
        folds=folds,
        seed=seed,
        alpha=alpha,
        inferred_source_weight=inferred_source_weight,
        artifact_version=artifact_version,
    )
    full_rows = build_anchored_rows(
        conn,
        inputs,
        include_subtitle_ids=None,
        alpha=alpha,
        inferred_source_weight=inferred_source_weight,
        artifact_version=artifact_version,
    )
    full_group_probs = _group_probabilities(full_rows)
    groups = sorted(full_group_probs)

    metrics_rows: list[dict[str, object]] = []
    for config in configs:
        temperatures = _fit_temperatures(
            fold_data,
            config.granularity,
            groups=groups,
            temperature_min=config.temperature_min,
            temperature_max=config.temperature_max,
        )
        metrics = _compute_metrics(fold_data, full_group_probs, temperatures)
        verdict = calibration_verdict(metrics)
        temps = list(temperatures.values())
        metrics_rows.append(
            {
                "experiment": config.name,
                "granularity": config.granularity,
                "heldout_nll_baseline": round(metrics.heldout_nll_baseline, 6),
                "heldout_nll_calibrated": round(metrics.heldout_nll_calibrated, 6),
                "nll_improvement": round(
                    metrics.heldout_nll_baseline - metrics.heldout_nll_calibrated, 6
                ),
                "ece_baseline": round(metrics.ece_baseline, 6),
                "ece_calibrated": round(metrics.ece_calibrated, 6),
                "top1_hit_rate": round(metrics.top1_hit_rate, 6),
                "effective_n_baseline": round(metrics.mean_effective_n_baseline, 6),
                "effective_n_calibrated": round(metrics.mean_effective_n_calibrated, 6),
                "distinctiveness_baseline": round(metrics.distinctiveness_baseline, 6),
                "distinctiveness_calibrated": round(metrics.distinctiveness_calibrated, 6),
                "distinctiveness_drop": round(
                    float(verdict["distinctiveness_drop_fraction"]), 6
                ),
                "min_temperature": round(min(temps), 6) if temps else 1.0,
                "max_temperature": round(max(temps), 6) if temps else 1.0,
                "tiers_kept_distinct": bool(verdict["tiers_kept_distinct"]),
            }
        )

    metrics_path = output_dir / "tier_slot_calibration_metrics.csv"
    _write_metrics(metrics_path, metrics_rows)
    report_path = output_dir / "tier_slot_calibration_ablation_report.md"
    report_path.write_text(
        _format_ablation_report(metrics_rows), encoding="utf-8"
    )
    return CalibrationAblationResult(
        report_path=report_path,
        metrics_path=metrics_path,
        experiment_count=len(configs),
    )


def run_calibration_autoresearcher(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    configs: tuple[CalibrationConfig, ...] | None = None,
    folds: int = 5,
    seed: int = 20260612,
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
) -> CalibrationAutoResearcherResult:
    """Run the calibration sweep, then propose the next config + review packet.

    Deterministic and local: no external LLM, no serving change. Calibration's
    objective (held-out likelihood) is quantitative, so the proposals are driven
    by the swept metrics rather than human bleed ratings.
    """

    ablation = run_calibration_ablation(
        conn,
        output_dir,
        configs=configs,
        folds=folds,
        seed=seed,
        alpha=alpha,
        inferred_source_weight=inferred_source_weight,
        artifact_version=artifact_version,
    )
    metrics_rows = _read_metrics(ablation.metrics_path)
    proposals = _next_calibration_proposals(metrics_rows)

    proposals_path = output_dir / "tier_slot_calibration_proposals.csv"
    _write_proposals(proposals_path, proposals)
    report_path = output_dir / "tier_slot_calibration_autoresearcher_report.md"
    report_path.write_text(
        _format_autoresearcher_report(metrics_rows, proposals), encoding="utf-8"
    )
    return CalibrationAutoResearcherResult(
        report_path=report_path,
        proposals_path=proposals_path,
        ablation_result=ablation,
    )


# ---------------------------------------------------------------------------
# Ablation / autoresearcher IO helpers
# ---------------------------------------------------------------------------


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    float_fields = {
        "heldout_nll_baseline",
        "heldout_nll_calibrated",
        "nll_improvement",
        "ece_baseline",
        "ece_calibrated",
        "top1_hit_rate",
        "effective_n_baseline",
        "effective_n_calibrated",
        "distinctiveness_baseline",
        "distinctiveness_calibrated",
        "distinctiveness_drop",
        "min_temperature",
        "max_temperature",
    }
    for row in rows:
        for field_name in float_fields:
            row[field_name] = float(row[field_name])
        row["tiers_kept_distinct"] = str(row["tiers_kept_distinct"]).lower() == "true"
    return rows


def _next_calibration_proposals(
    metrics_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    proposals: list[dict[str, object]] = []
    candidates = [row for row in metrics_rows if row["granularity"] != "none"]
    if not candidates:
        return proposals

    safe = [row for row in candidates if row["tiers_kept_distinct"]]
    pool = safe or candidates
    best = max(pool, key=lambda row: row["nll_improvement"])

    if best["nll_improvement"] <= 1e-6:
        proposals.append(
            {
                "priority": 1,
                "proposal": "accept_baseline",
                "rationale": (
                    "Temperature scaling did not improve held-out likelihood; the "
                    "uncalibrated distribution is already well-shaped. Keep T=1."
                ),
            }
        )
    else:
        proposals.append(
            {
                "priority": 1,
                "proposal": f"adopt:{best['experiment']}",
                "rationale": (
                    f"Best held-out NLL improvement ({best['nll_improvement']:+.2f}) "
                    f"while keeping tiers distinct (drop "
                    f"{best['distinctiveness_drop']:.1%})."
                ),
            }
        )

    if any(not row["tiers_kept_distinct"] for row in candidates):
        proposals.append(
            {
                "priority": 2,
                "proposal": "lower_temperature_cap",
                "rationale": (
                    "At least one config flattened tiers past the distinctiveness "
                    "guardrail; tighten temperature_max before re-sweeping."
                ),
            }
        )

    if best["ece_calibrated"] >= best["ece_baseline"] - 1e-6:
        proposals.append(
            {
                "priority": 3,
                "proposal": "consider_isotonic_or_platt",
                "rationale": (
                    "Temperature left reliability (ECE) roughly unchanged; if a "
                    "lower ECE is required, evaluate isotonic/Platt as the staged "
                    "fallback (issue #39)."
                ),
            }
        )
    return proposals


def _write_proposals(path: Path, proposals: list[dict[str, object]]) -> None:
    fieldnames = ["priority", "proposal", "rationale"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for proposal in proposals:
            writer.writerow(proposal)


def _format_ablation_report(metrics_rows: list[dict[str, object]]) -> str:
    lines = ["# Tier-slot calibration ablation (step 6, #39)", ""]
    lines.append(
        "Bounded sweep of temperature-scaling granularities over one shared "
        "5-fold held-out split. Analysis-only; nothing here changes serving."
    )
    lines.append("")
    lines.append(
        "| experiment | NLL improvement | ECE (base->cal) | effective-N "
        "(base->cal) | cross-tier drop | tiers distinct |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in metrics_rows:
        lines.append(
            f"| {row['experiment']} | {float(row['nll_improvement']):+.2f} | "
            f"{float(row['ece_baseline']):.4f}->{float(row['ece_calibrated']):.4f} | "
            f"{float(row['effective_n_baseline']):.2f}->"
            f"{float(row['effective_n_calibrated']):.2f} | "
            f"{float(row['distinctiveness_drop']):.1%} | "
            f"{'yes' if row['tiers_kept_distinct'] else 'NO'} |"
        )
    lines.append("")
    lines.append(
        "Higher NLL improvement is better; lower ECE is better; effective-N up = "
        "more variety, down = more repetition."
    )
    lines.append("")
    return "\n".join(lines)


def _format_autoresearcher_report(
    metrics_rows: list[dict[str, object]],
    proposals: list[dict[str, object]],
) -> str:
    lines = ["# Tier-slot calibration AutoResearcher (step 6, #39)", ""]
    lines.append(
        "Deterministic local loop: swept the calibration grid, scored each on "
        "held-out evidence, and proposed the next step. No external LLM, no "
        "serving change."
    )
    lines.append("")
    lines.append("## Swept results")
    lines.append("")
    lines.append("| experiment | NLL improvement | ECE calibrated | tiers distinct |")
    lines.append("|---|---|---|---|")
    for row in metrics_rows:
        lines.append(
            f"| {row['experiment']} | {float(row['nll_improvement']):+.2f} | "
            f"{float(row['ece_calibrated']):.4f} | "
            f"{'yes' if row['tiers_kept_distinct'] else 'NO'} |"
        )
    lines.append("")
    lines.append("## Proposals for the next round")
    lines.append("")
    if not proposals:
        lines.append("_No proposals (no calibration variants were evaluated)._")
    else:
        for proposal in proposals:
            lines.append(f"- **{proposal['proposal']}** -- {proposal['rationale']}")
    lines.append("")
    lines.append(
        "A human approves the chosen config and trade-off via "
        "`subtitle-gen ingest-calibration-decision` before this step closes."
    )
    lines.append("")
    return "\n".join(lines)
