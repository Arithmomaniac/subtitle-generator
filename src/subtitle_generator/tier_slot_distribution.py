"""Build tier-conditioned filler distributions from source/filler evidence."""

from __future__ import annotations

import csv
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.schema_contracts import (
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    TIER_SLOT_FILLER_DISTRIBUTION_TIERS,
    validate_tier_slot_distribution,
)

TIERS = TIER_SLOT_FILLER_DISTRIBUTION_TIERS
DISTRIBUTION_COLUMNS = (
    "slot_type",
    "tier",
    "filler",
    "display_filler",
    "probability",
    "log_probability",
    "soft_count",
    "prior_count",
    "evidence_count",
    "source_count",
    "anchored_source_count",
    "inferred_source_count",
    "anchored_soft_count",
    "inferred_soft_count",
    "teacher_confidence_mean",
    "frequency",
    "popularity_score",
    "semantic_smoothing_mass",
    "calibration_temperature",
    "artifact_version",
)


@dataclass(frozen=True)
class TierSlotDistributionResult:
    distribution_path: Path
    report_path: Path
    row_count: int


@dataclass(frozen=True)
class _Filler:
    id: str
    slot_type: str
    filler: str
    display_filler: str
    frequency: int
    popularity_score: float | None
    scores: dict[str, float]


@dataclass(frozen=True)
class _SourceLabel:
    tier: str | None
    confidence: float | None


@dataclass
class _EvidenceCell:
    anchored_soft_count: float = 0.0
    inferred_soft_count: float = 0.0
    anchored_sources: set[int] | None = None
    inferred_sources: set[int] | None = None
    confidence_sum: float = 0.0
    confidence_count: int = 0

    def __post_init__(self) -> None:
        if self.anchored_sources is None:
            self.anchored_sources = set()
        if self.inferred_sources is None:
            self.inferred_sources = set()

    @property
    def soft_count(self) -> float:
        return self.anchored_soft_count + self.inferred_soft_count

    @property
    def anchored_source_count(self) -> int:
        return len(self.anchored_sources or ())

    @property
    def inferred_source_count(self) -> int:
        return len(self.inferred_sources or ())

    @property
    def source_count(self) -> int:
        return self.anchored_source_count + self.inferred_source_count

    @property
    def teacher_confidence_mean(self) -> float | None:
        if self.confidence_count == 0:
            return None
        return self.confidence_sum / self.confidence_count


def build_tier_slot_distribution(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
) -> TierSlotDistributionResult:
    """Build a transparent empirical-Bayes tier-slot distribution artifact."""

    if alpha < 0:
        raise RuntimeError("alpha must be nonnegative")
    if inferred_source_weight < 0:
        raise RuntimeError("inferred_source_weight must be nonnegative")
    if not artifact_version:
        raise RuntimeError("artifact_version must be nonempty")

    output_dir.mkdir(parents=True, exist_ok=True)
    fillers, filler_id_to_key = _load_fillers(conn)
    source_labels = _load_source_labels(conn)
    source_fallbacks = _load_source_fallback_vectors(conn)
    global_fallback = _global_fallback_vector(fillers.values())
    residual_priors = _label_residual_priors(source_labels.values())
    cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=source_labels,
        source_fallbacks=source_fallbacks,
        global_fallback=global_fallback,
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
    )
    rows = _distribution_rows(
        fillers=fillers,
        cells=cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )

    distribution_path = output_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv"
    _write_distribution(distribution_path, rows)
    _validate_rows(conn, rows)

    report_path = output_dir / "tier_slot_distribution_report.md"
    report_path.write_text(
        _format_report(
            rows=rows,
            fillers=fillers,
            source_labels=source_labels,
            residual_priors=residual_priors,
            alpha=alpha,
            inferred_source_weight=inferred_source_weight,
            artifact_version=artifact_version,
        ),
        encoding="utf-8",
    )
    return TierSlotDistributionResult(
        distribution_path=distribution_path,
        report_path=report_path,
        row_count=len(rows),
    )


def _load_fillers(conn: sqlite3.Connection) -> tuple[dict[str, _Filler], dict[int, str]]:
    rows = conn.execute(
        """
        SELECT
            sf.id,
            sf.slot_type,
            sf.filler,
            sf.freq,
            sf.popularity_score,
            COALESCE(ms.score_pop, 0.0),
            COALESCE(ms.score_mainstream, 0.0),
            COALESCE(ms.score_niche, 0.0)
        FROM slot_fillers sf
        LEFT JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
        WHERE sf.mode = 'strict'
        ORDER BY sf.id
        """
    ).fetchall()
    grouped: dict[str, dict[str, object]] = {}
    filler_id_to_key: dict[int, str] = {}
    for row in rows:
        filler_id = int(row[0])
        slot_type = row[1]
        original_filler = row[2]
        frequency = int(row[3] or 0)
        weight = max(1, frequency)
        key = _filler_key(slot_type, original_filler)
        filler_id_to_key[filler_id] = key
        entry = grouped.setdefault(
            key,
            {
                "slot_type": slot_type,
                "filler": _normalize_filler(original_filler),
                "display_candidates": [],
                "frequency": 0,
                "popularity_weighted_sum": 0.0,
                "popularity_weight": 0,
                "score_weighted_sum": {tier: 0.0 for tier in TIERS},
                "score_weight": 0,
            },
        )
        entry["display_candidates"].append((frequency, original_filler))
        entry["frequency"] += frequency
        if row[4] is not None:
            entry["popularity_weighted_sum"] += float(row[4]) * weight
            entry["popularity_weight"] += weight
        scores = _normalize({
            "pop": float(row[5] or 0.0),
            "mainstream": float(row[6] or 0.0),
            "niche": float(row[7] or 0.0),
        })
        for tier in TIERS:
            entry["score_weighted_sum"][tier] += scores[tier] * weight
        entry["score_weight"] += weight

    fillers: dict[str, _Filler] = {}
    for key, entry in grouped.items():
        display_filler = sorted(
            list(entry["display_candidates"]),
            key=lambda item: (-int(item[0]), str(item[1]).lower(), str(item[1])),
        )[0][1]
        popularity_weight = int(entry["popularity_weight"])
        score_weight = int(entry["score_weight"])
        fillers[key] = _Filler(
            id=key,
            slot_type=str(entry["slot_type"]),
            filler=str(entry["filler"]),
            display_filler=str(display_filler),
            frequency=int(entry["frequency"]),
            popularity_score=(
                float(entry["popularity_weighted_sum"]) / popularity_weight
                if popularity_weight
                else None
            ),
            scores=_normalize({
                tier: float(entry["score_weighted_sum"][tier]) / score_weight
                for tier in TIERS
            }),
        )
    return fillers, filler_id_to_key


def _load_source_labels(conn: sqlite3.Connection) -> dict[int, _SourceLabel]:
    rows = conn.execute(
        """
        SELECT subtitle_id, llm_market_tier, llm_market_tier_confidence
        FROM pattern_matches
        WHERE subtitle_id IS NOT NULL
        """
    ).fetchall()
    labels: dict[int, _SourceLabel] = {}
    for subtitle_id, tier, confidence in rows:
        labels[int(subtitle_id)] = _SourceLabel(
            tier=tier if tier in TIERS else None,
            confidence=_clamp_confidence(confidence),
        )
    return labels


def _load_source_fallback_vectors(conn: sqlite3.Connection) -> dict[int, dict[str, float]]:
    rows = conn.execute(
        """
        SELECT
            sfs.subtitle_id,
            COALESCE(ms.score_pop, 0.0),
            COALESCE(ms.score_mainstream, 0.0),
            COALESCE(ms.score_niche, 0.0)
        FROM slot_filler_sources sfs
        JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
        LEFT JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
        WHERE sf.mode = 'strict'
        """
    ).fetchall()
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for subtitle_id, pop, mainstream, niche in rows:
        grouped[int(subtitle_id)].append(_normalize({
            "pop": float(pop or 0.0),
            "mainstream": float(mainstream or 0.0),
            "niche": float(niche or 0.0),
        }))
    return {
        subtitle_id: _normalize({
            tier: sum(vector[tier] for vector in vectors) / len(vectors)
            for tier in TIERS
        })
        for subtitle_id, vectors in grouped.items()
        if vectors
    }


def _global_fallback_vector(fillers: list[_Filler] | tuple[_Filler, ...] | object) -> dict[str, float]:
    filler_list = list(fillers)
    if not filler_list:
        return {tier: 1.0 / len(TIERS) for tier in TIERS}
    return _normalize({
        tier: sum(filler.scores[tier] for filler in filler_list) / len(filler_list)
        for tier in TIERS
    })


def _label_residual_priors(labels: list[_SourceLabel] | tuple[_SourceLabel, ...] | object) -> dict[str, dict[str, float]]:
    counts = Counter(label.tier for label in labels if label.tier in TIERS)
    result: dict[str, dict[str, float]] = {}
    for anchor in TIERS:
        other_tiers = [tier for tier in TIERS if tier != anchor]
        total = sum(counts[tier] for tier in other_tiers)
        if total <= 0:
            result[anchor] = {tier: 1.0 / len(other_tiers) for tier in other_tiers}
        else:
            result[anchor] = {tier: counts[tier] / total for tier in other_tiers}
    return result


def _build_evidence_cells(
    conn: sqlite3.Connection,
    *,
    fillers: dict[str, _Filler],
    filler_id_to_key: dict[int, str],
    source_labels: dict[int, _SourceLabel],
    source_fallbacks: dict[int, dict[str, float]],
    global_fallback: dict[str, float],
    residual_priors: dict[str, dict[str, float]],
    inferred_source_weight: float,
) -> dict[tuple[str, str], _EvidenceCell]:
    cells: dict[tuple[str, str], _EvidenceCell] = defaultdict(_EvidenceCell)
    links = conn.execute(
        """
        SELECT sfs.slot_filler_id, sfs.subtitle_id
        FROM slot_filler_sources sfs
        JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
        WHERE sf.mode = 'strict'
        ORDER BY sfs.slot_filler_id, sfs.subtitle_id
        """
    ).fetchall()
    for filler_id_raw, subtitle_id_raw in links:
        filler_id = int(filler_id_raw)
        subtitle_id = int(subtitle_id_raw)
        filler_key = filler_id_to_key.get(filler_id)
        if filler_key not in fillers:
            continue
        label = source_labels.get(subtitle_id)
        if label and label.tier in TIERS and label.confidence is not None:
            _add_labeled_source(
                cells,
                filler_id=filler_key,
                subtitle_id=subtitle_id,
                label=label,
                residual_prior=residual_priors[label.tier],
                inferred_source_weight=inferred_source_weight,
            )
        else:
            vector = source_fallbacks.get(subtitle_id, global_fallback)
            _add_inferred_source(
                cells,
                filler_id=filler_key,
                subtitle_id=subtitle_id,
                vector=vector,
                weight=inferred_source_weight,
            )
    return cells


def _add_labeled_source(
    cells: dict[tuple[str, str], _EvidenceCell],
    *,
    filler_id: str,
    subtitle_id: int,
    label: _SourceLabel,
    residual_prior: dict[str, float],
    inferred_source_weight: float,
) -> None:
    assert label.tier is not None and label.confidence is not None
    confidence = label.confidence
    anchored = cells[(filler_id, label.tier)]
    anchored.anchored_soft_count += confidence
    anchored.anchored_sources.add(subtitle_id)
    anchored.confidence_sum += confidence
    anchored.confidence_count += 1
    residual = max(0.0, 1.0 - confidence) * inferred_source_weight
    for tier, share in residual_prior.items():
        if residual <= 0:
            continue
        cell = cells[(filler_id, tier)]
        cell.inferred_soft_count += residual * share
        cell.inferred_sources.add(subtitle_id)
        cell.confidence_sum += confidence
        cell.confidence_count += 1


def _add_inferred_source(
    cells: dict[tuple[str, str], _EvidenceCell],
    *,
    filler_id: str,
    subtitle_id: int,
    vector: dict[str, float],
    weight: float,
) -> None:
    if weight <= 0:
        return
    for tier in TIERS:
        mass = vector[tier] * weight
        if mass <= 0:
            continue
        cell = cells[(filler_id, tier)]
        cell.inferred_soft_count += mass
        cell.inferred_sources.add(subtitle_id)


def _distribution_rows(
    *,
    fillers: dict[str, _Filler],
    cells: dict[tuple[str, str], _EvidenceCell],
    alpha: float,
    artifact_version: str,
) -> list[dict[str, object]]:
    raw_rows: list[dict[str, object]] = []
    group_totals: dict[tuple[str, str], float] = defaultdict(float)
    for filler in fillers.values():
        for tier in TIERS:
            cell = cells[(filler.id, tier)]
            evidence_count = cell.soft_count + alpha
            row = {
                "slot_type": filler.slot_type,
                "tier": tier,
                "filler": filler.filler,
                "display_filler": filler.display_filler,
                "probability": 0.0,
                "log_probability": 0.0,
                "soft_count": cell.soft_count,
                "prior_count": alpha,
                "evidence_count": evidence_count,
                "source_count": cell.source_count,
                "anchored_source_count": cell.anchored_source_count,
                "inferred_source_count": cell.inferred_source_count,
                "anchored_soft_count": cell.anchored_soft_count,
                "inferred_soft_count": cell.inferred_soft_count,
                "teacher_confidence_mean": cell.teacher_confidence_mean,
                "frequency": filler.frequency,
                "popularity_score": filler.popularity_score,
                "semantic_smoothing_mass": 0.0,
                "calibration_temperature": 1.0,
                "artifact_version": artifact_version,
            }
            raw_rows.append(row)
            group_totals[(filler.slot_type, tier)] += evidence_count
    for row in raw_rows:
        total = group_totals[(str(row["slot_type"]), str(row["tier"]))]
        probability = float(row["evidence_count"]) / total if total > 0 else 0.0
        row["probability"] = probability
        row["log_probability"] = math.log(probability) if probability > 0 else float("-inf")
    return raw_rows


def _write_distribution(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISTRIBUTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: _format_csv_value(row[column])
                for column in DISTRIBUTION_COLUMNS
            })


def _validate_rows(conn: sqlite3.Connection, rows: list[dict[str, object]]) -> None:
    validation = sqlite3.connect(":memory:")
    try:
        validation.execute(
            """
            CREATE TABLE slot_fillers (
                id INTEGER PRIMARY KEY,
                slot_type TEXT NOT NULL,
                filler TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'strict'
            )
            """
        )
        validation.executemany(
            "INSERT INTO slot_fillers VALUES (?, ?, ?, 'strict')",
            conn.execute(
                """
                SELECT id, slot_type, filler
                FROM slot_fillers
                WHERE mode = 'strict'
                """
            ).fetchall(),
        )
        validation.execute(f"""
            CREATE TABLE {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} (
                slot_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                filler TEXT NOT NULL,
                display_filler TEXT NOT NULL,
                probability REAL NOT NULL,
                log_probability REAL NOT NULL,
                soft_count REAL NOT NULL,
                prior_count REAL NOT NULL,
                evidence_count REAL NOT NULL,
                source_count INTEGER NOT NULL,
                anchored_source_count INTEGER NOT NULL,
                inferred_source_count INTEGER NOT NULL,
                anchored_soft_count REAL NOT NULL,
                inferred_soft_count REAL NOT NULL,
                teacher_confidence_mean REAL,
                frequency INTEGER NOT NULL,
                popularity_score REAL,
                semantic_smoothing_mass REAL NOT NULL,
                calibration_temperature REAL NOT NULL,
                artifact_version TEXT NOT NULL
            )
        """)
        validation.executemany(
            f"""
            INSERT INTO {TIER_SLOT_FILLER_DISTRIBUTION_TABLE}
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                tuple(row[column] for column in DISTRIBUTION_COLUMNS)
                for row in rows
            ],
        )
        issues = validate_tier_slot_distribution(validation)
        if issues:
            raise RuntimeError("\n".join(issue.message for issue in issues))
    finally:
        validation.close()


def _format_report(
    *,
    rows: list[dict[str, object]],
    fillers: dict[str, _Filler],
    source_labels: dict[int, _SourceLabel],
    residual_priors: dict[str, dict[str, float]],
    alpha: float,
    inferred_source_weight: float,
    artifact_version: str,
) -> str:
    label_counts = Counter(
        label.tier if label.tier in TIERS else "unlabeled"
        for label in source_labels.values()
    )
    group_summary = _group_summary(rows)
    mass_summary = _mass_summary(rows)
    top_rows = _top_rows(rows)
    current_comparison = _current_rollup_comparison(rows, fillers)
    lines = [
        "# Tier-slot distribution report",
        "",
        "This report builds the first empirical-Bayes baseline for "
        "$P(\\mathrm{filler} \\mid \\mathrm{tier}, \\mathrm{slot\\_type})$.",
        "",
        "## Build settings",
        "",
        "| Setting | Value |",
        "|---|---:|",
        f"| artifact_version | `{artifact_version}` |",
        f"| alpha | {alpha:.6g} |",
        f"| inferred_source_weight | {inferred_source_weight:.6g} |",
        "",
        "## Source label coverage",
        "",
        "| Source label | Sources |",
        "|---|---:|",
    ]
    for label, count in sorted(label_counts.items()):
        lines.append(f"| {label} | {count:,} |")
    lines.extend([
        "",
        "LLM-labeled sources anchor the labeled tier at "
        "`llm_market_tier_confidence`. Residual mass is allocated across the "
        "other tiers using the label-marginal residual prior below. Unlabeled "
        "sources use the current score-table fallback and are counted as inferred "
        "evidence.",
        "",
        "## Residual priors for labeled sources",
        "",
        "| Anchored label | Residual split |",
        "|---|---|",
    ])
    for anchor in TIERS:
        split = ", ".join(
            f"{tier}={share:.4f}"
            for tier, share in residual_priors[anchor].items()
        )
        lines.append(f"| {anchor} | {split} |")
    lines.extend([
        "",
        "## Group summary",
        "",
        "| Slot type | Tier | Rows | Probability mass | Entropy | Effective N | Prior-only rows | Anchored mass | Inferred mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in group_summary:
        lines.append(
            f"| {row['slot_type']} | {row['tier']} | {row['rows']:,} | "
            f"{row['probability_mass']:.6f} | {row['entropy']:.3f} | "
            f"{row['effective_n']:.1f} | {row['prior_only_rows']:,} | "
            f"{row['anchored_mass']:.3f} | {row['inferred_mass']:.3f} |"
        )
    lines.extend([
        "",
        "## Anchored vs inferred mass by tier",
        "",
        "| Tier | Anchored soft count | Inferred soft count | Prior count |",
        "|---|---:|---:|---:|",
    ])
    for tier in TIERS:
        row = mass_summary[tier]
        lines.append(
            f"| {tier} | {row['anchored']:.3f} | {row['inferred']:.3f} | "
            f"{row['prior']:.3f} |"
        )
    lines.extend([
        "",
        "## Current rollup comparison",
        "",
        "This compares the new normalized distribution with the current runtime "
        "requested-tier weighting baseline:",
        "",
        "$$",
        "w(f, T) = \\sqrt{\\mathrm{freq}(f)} \\cdot \\max(\\mathrm{score}_T(f), 0.001)",
        "$$",
        "",
        "| Slot type | Tier | JS divergence | Top-20 overlap | Biggest probability increases | Biggest probability decreases |",
        "|---|---:|---:|---:|---|---|",
    ])
    for row in current_comparison:
        lines.append(
            f"| {row['slot_type']} | {row['tier']} | "
            f"{row['js_divergence']:.6f} | {row['top20_overlap']}/20 | "
            f"{row['increases']} | {row['decreases']} |"
        )
    lines.extend([
        "",
        "## Top probabilities by tier and slot",
        "",
        "| Slot type | Tier | Top fillers |",
        "|---|---|---|",
    ])
    for (slot_type, tier), ranked in top_rows.items():
        formatted = "; ".join(
            f"{row['display_filler']} [{row['filler']}] "
            f"(p={float(row['probability']):.5f}, "
            f"soft={float(row['soft_count']):.3f}, "
            f"anch={float(row['anchored_soft_count']):.3f}, "
            f"inf={float(row['inferred_soft_count']):.3f})"
            for row in ranked[:8]
        )
        lines.append(f"| {slot_type} | {tier} | {formatted} |")
    lines.extend([
        "",
        "## Caveats",
        "",
        "- This is a transparent first baseline, not the final served distribution.",
        "- `alpha` and `inferred_source_weight` are exposed for reproducibility "
        "and later sweeps, but this step does not tune them in isolation. "
        "Serious AutoResearcher-style tuning should wait until semantic "
        "smoothing, calibration, and runtime sample review are part of the loop.",
        "- Semantic smoothing is not applied yet; `semantic_smoothing_mass` is zero.",
        "- Calibration is not applied yet; `calibration_temperature` is one.",
        "- Unlabeled sources use the current score table as a bootstrap fallback, "
        "so inferred evidence should be interpreted as compatibility evidence, "
        "not independent teacher evidence.",
    ])
    return "\n".join(lines)


def _group_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    summary = []
    for (slot_type, tier), group_rows in sorted(grouped.items()):
        probabilities = [float(row["probability"]) for row in group_rows]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        summary.append({
            "slot_type": slot_type,
            "tier": tier,
            "rows": len(group_rows),
            "probability_mass": sum(probabilities),
            "entropy": entropy,
            "effective_n": math.exp(entropy),
            "prior_only_rows": sum(float(row["soft_count"]) == 0 for row in group_rows),
            "anchored_mass": sum(float(row["anchored_soft_count"]) for row in group_rows),
            "inferred_mass": sum(float(row["inferred_soft_count"]) for row in group_rows),
        })
    return summary


def _mass_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary = {
        tier: {"anchored": 0.0, "inferred": 0.0, "prior": 0.0}
        for tier in TIERS
    }
    for row in rows:
        tier = str(row["tier"])
        summary[tier]["anchored"] += float(row["anchored_soft_count"])
        summary[tier]["inferred"] += float(row["inferred_soft_count"])
        summary[tier]["prior"] += float(row["prior_count"])
    return summary


def _current_rollup_comparison(
    rows: list[dict[str, object]],
    fillers: dict[str, _Filler],
) -> list[dict[str, object]]:
    filler_lookup = {
        (filler.slot_type, filler.filler): filler
        for filler in fillers.values()
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    comparison: list[dict[str, object]] = []
    for (slot_type, tier), group_rows in sorted(grouped.items()):
        current_weights = []
        for row in group_rows:
            filler = filler_lookup[(slot_type, str(row["filler"]))]
            current_weights.append(
                math.sqrt(max(0, filler.frequency))
                * max(filler.scores[tier], 0.001)
            )
        current_total = sum(current_weights)
        if current_total <= 0:
            current_probabilities = [1.0 / len(group_rows) for _ in group_rows]
        else:
            current_probabilities = [
                weight / current_total for weight in current_weights
            ]
        deltas = []
        for row, current_probability in zip(group_rows, current_probabilities):
            new_probability = float(row["probability"])
            deltas.append({
                "filler": str(row["filler"]),
                "display_filler": str(row["display_filler"]),
                "new_probability": new_probability,
                "current_probability": current_probability,
                "delta": new_probability - current_probability,
            })
        new_top = {
            item["filler"]
            for item in sorted(
                deltas,
                key=lambda item: (-item["new_probability"], item["filler"].lower()),
            )[:20]
        }
        current_top = {
            item["filler"]
            for item in sorted(
                deltas,
                key=lambda item: (-item["current_probability"], item["filler"].lower()),
            )[:20]
        }
        comparison.append({
            "slot_type": slot_type,
            "tier": tier,
            "js_divergence": _js_divergence(
                [float(row["probability"]) for row in group_rows],
                current_probabilities,
            ),
            "top20_overlap": len(new_top & current_top),
            "increases": _format_delta_examples(
                sorted(deltas, key=lambda item: (-item["delta"], item["filler"].lower()))[:5]
            ),
            "decreases": _format_delta_examples(
                sorted(deltas, key=lambda item: (item["delta"], item["filler"].lower()))[:5]
            ),
        })
    return comparison


def _js_divergence(left: list[float], right: list[float]) -> float:
    midpoint = [(l_val + r_val) / 2.0 for l_val, r_val in zip(left, right)]
    return (_kl_divergence(left, midpoint) + _kl_divergence(right, midpoint)) / 2.0


def _kl_divergence(left: list[float], right: list[float]) -> float:
    total = 0.0
    for l_val, r_val in zip(left, right):
        if l_val > 0 and r_val > 0:
            total += l_val * math.log(l_val / r_val)
    return total


def _format_delta_examples(examples: list[dict[str, object]]) -> str:
    return "; ".join(
        f"{item['display_filler']} [{item['filler']}] "
        f"({float(item['current_probability']):.5f} -> "
        f"{float(item['new_probability']):.5f}, "
        f"{float(item['delta']):+.5f})"
        for item in examples
    )


def _top_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    return {
        key: sorted(
            group_rows,
            key=lambda row: (-float(row["probability"]), str(row["filler"]).lower()),
        )
        for key, group_rows in sorted(grouped.items())
    }


def _normalize(values: dict[str, float]) -> dict[str, float]:
    cleaned = {tier: max(0.0, float(values.get(tier, 0.0))) for tier in TIERS}
    total = sum(cleaned.values())
    if total <= 0:
        return {tier: 1.0 / len(TIERS) for tier in TIERS}
    return {tier: cleaned[tier] / total for tier in TIERS}


def _normalize_filler(filler: str) -> str:
    return " ".join(filler.split()).casefold()


def _filler_key(slot_type: str, filler: str) -> str:
    return f"{slot_type}\0{_normalize_filler(filler)}"


def _clamp_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.12g}"
    return value
