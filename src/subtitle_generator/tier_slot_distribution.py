"""Build tier-conditioned filler distributions from source/filler evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from subtitle_generator.schema_contracts import (
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    TIER_SLOT_FILLER_DISTRIBUTION_TIERS,
    validate_tier_slot_distribution,
)
from subtitle_generator.slots import (
    _is_valid_action,
    _is_valid_list_item,
    _is_valid_object,
    _load_nlp,
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
class SemanticSmoothingAblationResult:
    report_path: Path
    metrics_path: Path
    experiment_count: int


@dataclass(frozen=True)
class SemanticSmoothingAutoResearcherResult:
    report_path: Path
    proposals_path: Path
    ablation_result: SemanticSmoothingAblationResult


@dataclass(frozen=True)
class SmoothingReviewFeedResult:
    feed_path: Path
    run_id: str
    variant: str
    candidate_count: int


@dataclass(frozen=True)
class SmoothingExperimentConfig:
    name: str
    variant: str
    neighbor_count: int = 10
    shrinkage: float = 0.5
    evidence_gate: str = "none"
    max_borrowed_mass: float = 0.10
    vector_transform: str = "raw"


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
    reliability_sum: float = 0.0
    reliability_count: int = 0

    def __post_init__(self) -> None:
        if self.anchored_sources is None:
            self.anchored_sources = set()
        if self.inferred_sources is None:
            self.inferred_sources = set()

    @property
    def soft_count(self) -> float:
        return self.anchored_soft_count + self.inferred_soft_count

    @property
    def reliability_mean(self) -> float | None:
        if self.reliability_count == 0:
            return None
        return self.reliability_sum / self.reliability_count

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
    reliability_exponent: float = 1.0,
    unlabeled_reliability: float = 0.70,
    artifact_version: str = "tier_slot_filler_distribution_v1",
) -> TierSlotDistributionResult:
    """Build a transparent empirical-Bayes tier-slot distribution artifact.

    The emitted served artifact is anchored-only. Source reliability weighting
    is computed for analysis and written to a sidecar CSV
    (``<artifact>.confidence_weighted.csv``) plus report sections; it does not
    change the served distribution.
    """

    if alpha < 0:
        raise RuntimeError("alpha must be nonnegative")
    if inferred_source_weight < 0:
        raise RuntimeError("inferred_source_weight must be nonnegative")
    if not math.isfinite(reliability_exponent) or reliability_exponent <= 0:
        raise RuntimeError("reliability_exponent must be a finite positive value")
    if not 0.0 <= unlabeled_reliability <= 1.0:
        raise RuntimeError("unlabeled_reliability must be in [0, 1]")
    if not artifact_version:
        raise RuntimeError("artifact_version must be nonempty")

    output_dir.mkdir(parents=True, exist_ok=True)
    fillers, filler_id_to_key = _load_fillers(conn)
    source_labels = _load_source_labels(conn)
    source_fallbacks = _load_source_fallback_vectors(conn)
    global_fallback = _global_fallback_vector(fillers.values())
    residual_priors = _label_residual_priors(source_labels.values())
    evidence_source_ids = _evidence_source_ids(conn)
    scored_source_ids = _scored_source_ids(conn)
    reliability_weights = _source_reliability_weights(
        source_labels,
        evidence_source_ids,
        exponent=reliability_exponent,
        unlabeled_reliability=unlabeled_reliability,
    )
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
    hard_label_cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=_hard_label_source_labels(source_labels),
        source_fallbacks=source_fallbacks,
        global_fallback=global_fallback,
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
    )
    weighted_cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=source_labels,
        source_fallbacks=source_fallbacks,
        global_fallback=global_fallback,
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
        reliability_weights=reliability_weights,
        residual_from_teacher_vector=True,
        scored_source_ids=scored_source_ids,
    )
    # Same reliability magnitudes as ``weighted_cells`` but with the corpus-prior
    # residual direction -- a report-only baseline to isolate the Step 4b
    # source-aware residual movement. Not written to disk.
    weighted_corpus_residual_cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=source_labels,
        source_fallbacks=source_fallbacks,
        global_fallback=global_fallback,
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
        reliability_weights=reliability_weights,
        residual_from_teacher_vector=False,
    )
    rows = _distribution_rows(
        fillers=fillers,
        cells=cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )
    hard_label_rows = _distribution_rows(
        fillers=fillers,
        cells=hard_label_cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )
    weighted_rows = _distribution_rows(
        fillers=fillers,
        cells=weighted_cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )
    weighted_corpus_residual_rows = _distribution_rows(
        fillers=fillers,
        cells=weighted_corpus_residual_cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )

    distribution_path = output_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv"
    _write_distribution(distribution_path, rows)
    _validate_rows(conn, rows)

    weighted_path = (
        output_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.confidence_weighted.csv"
    )
    _write_distribution(weighted_path, weighted_rows)
    _validate_rows(conn, weighted_rows)

    report_path = output_dir / "tier_slot_distribution_report.md"
    report_path.write_text(
        _format_report(
            rows=rows,
            hard_label_rows=hard_label_rows,
            weighted_rows=weighted_rows,
            weighted_corpus_residual_rows=weighted_corpus_residual_rows,
            fillers=fillers,
            source_labels=source_labels,
            source_fallbacks=source_fallbacks,
            evidence_source_ids=evidence_source_ids,
            reliability_weights=reliability_weights,
            residual_priors=residual_priors,
            alpha=alpha,
            inferred_source_weight=inferred_source_weight,
            reliability_exponent=reliability_exponent,
            unlabeled_reliability=unlabeled_reliability,
            weighted_path=weighted_path,
            artifact_version=artifact_version,
        ),
        encoding="utf-8",
    )
    return TierSlotDistributionResult(
        distribution_path=distribution_path,
        report_path=report_path,
        row_count=len(rows),
    )


def run_semantic_smoothing_ablation(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
    vector_source: str = "offline_spacy",
    configs: tuple[SmoothingExperimentConfig, ...] | None = None,
) -> SemanticSmoothingAblationResult:
    """Run bounded semantic smoothing experiments without changing serving."""

    output_dir.mkdir(parents=True, exist_ok=True)
    fillers, filler_id_to_key = _load_fillers(conn)
    source_labels = _load_source_labels(conn)
    source_fallbacks = _load_source_fallback_vectors(conn)
    residual_priors = _label_residual_priors(source_labels.values())
    cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=source_labels,
        source_fallbacks=source_fallbacks,
        global_fallback=_global_fallback_vector(fillers.values()),
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
    )
    base_rows = _distribution_rows(
        fillers=fillers,
        cells=cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )
    configs = configs or default_smoothing_experiments()
    vectors, vector_source_counts = _load_smoothing_vectors(
        conn,
        fillers,
        output_dir / "tier_slot_embedding_cache.csv",
        vector_source=vector_source,
    )
    vector_cache: dict[str, dict[str, list[float]]] = {}
    metrics: list[dict[str, object]] = []
    experiment_outputs: list[tuple[SmoothingExperimentConfig, list[dict[str, object]]]] = []
    for config in configs:
        transformed_vectors = vector_cache.setdefault(
            config.vector_transform,
            _transform_vectors(vectors, config.vector_transform),
        )
        smoothed_rows = _apply_smoothing(base_rows, fillers, transformed_vectors, config)
        metrics.extend(_smoothing_metrics(config, base_rows, smoothed_rows))
        experiment_outputs.append((config, smoothed_rows))

    metrics_path = output_dir / "semantic_smoothing_metrics.csv"
    _write_smoothing_metrics(metrics_path, metrics)
    report_path = output_dir / "semantic_smoothing_ablation_report.md"
    report_path.write_text(
        _format_smoothing_report(
            base_rows=base_rows,
            experiment_outputs=experiment_outputs,
            metrics=metrics,
            vector_coverage=_vector_coverage(fillers, vectors),
            vector_source_counts=vector_source_counts,
        ),
        encoding="utf-8",
    )
    return SemanticSmoothingAblationResult(
        report_path=report_path,
        metrics_path=metrics_path,
        experiment_count=len(configs),
    )


def run_semantic_smoothing_autoresearcher(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
    vector_source: str = "offline_spacy",
    configs: tuple[SmoothingExperimentConfig, ...] | None = None,
) -> SemanticSmoothingAutoResearcherResult:
    """Run the deterministic local Step 5 AutoResearcher harvesting loop.

    This intentionally does not call an external LLM or mutate serving defaults.
    It executes/refreshes the bounded ablation, inspects the resulting metrics,
    and writes a structured next-round proposal packet for human or LLM review.
    """

    ablation_result = run_semantic_smoothing_ablation(
        conn,
        output_dir,
        alpha=alpha,
        inferred_source_weight=inferred_source_weight,
        artifact_version=artifact_version,
        vector_source=vector_source,
        configs=configs,
    )
    metrics = _read_smoothing_metrics(ablation_result.metrics_path)
    findings = _autoresearcher_findings(metrics)
    proposals = _next_smoothing_proposals(metrics, findings)

    proposals_path = output_dir / "semantic_smoothing_autoresearcher_proposals.csv"
    _write_autoresearcher_proposals(proposals_path, proposals)
    report_path = output_dir / "semantic_smoothing_autoresearcher_report.md"
    report_path.write_text(
        _format_autoresearcher_report(
            ablation_result=ablation_result,
            metrics=metrics,
            findings=findings,
            proposals=proposals,
        ),
        encoding="utf-8",
    )
    return SemanticSmoothingAutoResearcherResult(
        report_path=report_path,
        proposals_path=proposals_path,
        ablation_result=ablation_result,
    )


SMOOTHING_REVIEW_FEED_SCHEMA_VERSION = 1


def build_smoothing_review_feed(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    variant_name: str = "knn10_m0_5_cap0_10",
    alpha: float = 0.5,
    inferred_source_weight: float = 1.0,
    artifact_version: str = "tier_slot_filler_distribution_v1",
    vector_source: str = "offline_spacy",
    limit: int = 60,
    neighbor_limit: int = 5,
    configs: tuple[SmoothingExperimentConfig, ...] | None = None,
) -> SmoothingReviewFeedResult:
    """Emit a candidate feed of rate-worthy smoothing moves for human review.

    Replaces the throwaway review-packet generator with a committed, deterministic
    producer. For one chosen smoothing ``variant_name`` it ranks the largest
    base->smoothed probability moves on valid ML fillers, attaches the evidence
    and the nearest semantic contributors that drove each move, flags
    repair/bleed candidates, and writes ``step05_review_feed.json``. The feed is
    analysis-only and never affects the served distribution.

    The ``run_id`` is a content hash of the variant, vector source, and the
    selected candidates' base/smoothed probabilities, so an unchanged build is
    reproducible and a rating can always be tied to the exact candidates it
    judged.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    configs = configs or default_smoothing_experiments()
    config = next((c for c in configs if c.name == variant_name), None)
    if config is None:
        raise RuntimeError(
            f"Unknown smoothing variant {variant_name!r}; available: "
            f"{', '.join(c.name for c in configs)}"
        )
    if config.variant == "none":
        raise RuntimeError("variant 'none' has no smoothing moves to review")

    fillers, filler_id_to_key = _load_fillers(conn)
    source_labels = _load_source_labels(conn)
    source_fallbacks = _load_source_fallback_vectors(conn)
    residual_priors = _label_residual_priors(source_labels.values())
    cells = _build_evidence_cells(
        conn,
        fillers=fillers,
        filler_id_to_key=filler_id_to_key,
        source_labels=source_labels,
        source_fallbacks=source_fallbacks,
        global_fallback=_global_fallback_vector(fillers.values()),
        residual_priors=residual_priors,
        inferred_source_weight=inferred_source_weight,
    )
    base_rows = _distribution_rows(
        fillers=fillers,
        cells=cells,
        alpha=alpha,
        artifact_version=artifact_version,
    )
    vectors, _vector_source_counts = _load_smoothing_vectors(
        conn,
        fillers,
        output_dir / "tier_slot_embedding_cache.csv",
        vector_source=vector_source,
    )
    transformed_vectors = _transform_vectors(vectors, config.vector_transform)
    smoothed_rows = _apply_smoothing(base_rows, fillers, transformed_vectors, config)

    candidates = _select_smoothing_candidates(
        base_rows,
        smoothed_rows,
        transformed_vectors,
        config,
        limit=limit,
        neighbor_limit=neighbor_limit,
    )
    run_id = _smoothing_feed_run_id(config.name, vector_source, candidates)
    feed = {
        "schema_version": SMOOTHING_REVIEW_FEED_SCHEMA_VERSION,
        "run_id": run_id,
        "variant": config.name,
        "vector_source": vector_source,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }
    feed_path = output_dir / "step05_review_feed.json"
    feed_path.write_text(json.dumps(feed, indent=2) + "\n", encoding="utf-8")
    return SmoothingReviewFeedResult(
        feed_path=feed_path,
        run_id=run_id,
        variant=config.name,
        candidate_count=len(candidates),
    )


def _smoothing_feed_run_id(
    variant: str,
    vector_source: str,
    candidates: list[dict[str, object]],
) -> str:
    """Deterministic content hash tying ratings to the exact candidate set."""
    payload = {
        "variant": variant,
        "vector_source": vector_source,
        "candidates": [
            [
                c["slot_type"], c["tier"], c["filler"],
                round(float(c["base_p"]), 9), round(float(c["smoothed_p"]), 9),
            ]
            for c in candidates
        ],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _select_smoothing_candidates(
    base_rows: list[dict[str, object]],
    smoothed_rows: list[dict[str, object]],
    vectors: dict[str, list[float]],
    config: SmoothingExperimentConfig,
    *,
    limit: int,
    neighbor_limit: int,
    min_abs_delta: float = 1e-9,
) -> list[dict[str, object]]:
    """Rank the largest valid-ML smoothing *boosts* and attach evidence.

    The review decision enum (plausible_repair / semantic_bleed / too_generic) is
    entirely about *boosted* fillers -- "is this borrowed mass a good repair or
    topical bleed?". Pure |delta| ranking would instead surface head demotions
    (high-probability fillers shedding mass), which carry no such decision. So we
    keep positive-delta moves and rank by descending boost, matching the user's
    pop/mainstream-repair priority.
    """
    base_lookup = {
        (str(r["slot_type"]), str(r["tier"]), str(r["filler"])): r for r in base_rows
    }
    moves: list[dict[str, object]] = []
    for srow in smoothed_rows:
        key = (str(srow["slot_type"]), str(srow["tier"]), str(srow["filler"]))
        base = base_lookup.get(key)
        if base is None:
            continue
        slot_type = key[0]
        if not _is_valid_ml_slot_filler(slot_type, str(srow["display_filler"])):
            continue
        base_p = float(base["probability"])
        smoothed_p = float(srow["probability"])
        delta = smoothed_p - base_p
        if delta <= min_abs_delta:
            continue
        src = int(base["source_count"])
        anchored = float(base["anchored_soft_count"])
        flags: list[str] = ["repair_candidate"]
        if src <= 1:
            flags.append("low_source_support")
        if anchored <= 0:
            flags.append("no_anchored_same_tier")
        moves.append({
            "slot_type": slot_type,
            "tier": key[1],
            "filler": key[2],
            "display_filler": str(srow["display_filler"]),
            "base_p": base_p,
            "smoothed_p": smoothed_p,
            "delta": delta,
            "evidence": {
                "soft": float(base["soft_count"]),
                "src": src,
                "anchored": anchored,
            },
            "flags": flags,
        })
    moves.sort(key=lambda m: (-float(m["delta"]), str(m["filler"]).lower()))
    selected = moves[:limit]

    neighbors = _smoothing_candidate_neighbors(
        selected, base_rows, vectors, config, neighbor_limit=neighbor_limit
    )
    for move in selected:
        nkey = (str(move["slot_type"]), str(move["tier"]), str(move["filler"]))
        move["nearest_contributors"] = neighbors.get(nkey, [])
    return selected


def _smoothing_candidate_neighbors(
    candidates: list[dict[str, object]],
    base_rows: list[dict[str, object]],
    vectors: dict[str, list[float]],
    config: SmoothingExperimentConfig,
    *,
    neighbor_limit: int,
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    """For each candidate, the top semantic contributors that drove its move.

    Mirrors the neighbor selection in ``_semantic_prior_for_group`` but captures
    the contributing fillers (similarity + their probability/evidence) instead of
    only the blended prior. Returns empty lists for variants without semantic
    neighbors (e.g. ``uniform_prior``).
    """
    if config.variant not in {"generic_embedding_kNN", "tier_evidence_filtered_kNN"}:
        return {}
    import numpy as np

    evidence_gate = (
        config.evidence_gate
        if config.variant == "tier_evidence_filtered_kNN"
        else "none"
    )
    rows_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in base_rows:
        rows_by_group[(str(row["slot_type"]), str(row["tier"]))].append(row)

    wanted: dict[tuple[str, str], set[str]] = defaultdict(set)
    for cand in candidates:
        wanted[(str(cand["slot_type"]), str(cand["tier"]))].add(str(cand["filler"]))

    result: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for (slot_type, tier), targets in wanted.items():
        group_rows = rows_by_group.get((slot_type, tier), [])
        keys = [
            str(row["filler"])
            for row in group_rows
            if _filler_key(slot_type, str(row["filler"])) in vectors
            and _is_valid_ml_slot_filler(slot_type, str(row["display_filler"]))
        ]
        if not keys or config.neighbor_count <= 0:
            continue
        row_by_filler = {str(row["filler"]): row for row in group_rows}
        evidence_allowed = {
            str(row["filler"])
            for row in group_rows
            if _passes_evidence_gate(row, evidence_gate)
            and _is_valid_ml_slot_filler(slot_type, str(row["display_filler"]))
        }
        matrix = np.array(
            [vectors[_filler_key(slot_type, key)] for key in keys],
            dtype=np.float64,
        )
        similarities = matrix @ matrix.T
        key_index = {key: index for index, key in enumerate(keys)}
        for target in targets:
            if target not in key_index:
                continue
            index = key_index[target]
            contributors: list[tuple[float, str]] = []
            for other_index, similarity in enumerate(similarities[index]):
                other_key = keys[other_index]
                if other_key == target or similarity <= 0:
                    continue
                if other_key not in evidence_allowed:
                    continue
                contributors.append((float(similarity), other_key))
            contributors.sort(reverse=True)
            top = []
            for similarity, other_key in contributors[:neighbor_limit]:
                nrow = row_by_filler.get(other_key, {})
                top.append({
                    "filler": other_key,
                    "display_filler": str(nrow.get("display_filler", other_key)),
                    "similarity": round(similarity, 4),
                    "p": round(float(nrow.get("probability", 0.0)), 6),
                    "soft": round(float(nrow.get("soft_count", 0.0)), 3),
                    "src": int(nrow.get("source_count", 0)),
                })
            result[(slot_type, tier, target)] = top
    return result


def default_smoothing_experiments() -> tuple[SmoothingExperimentConfig, ...]:
    return (
        SmoothingExperimentConfig("none", "none", 0, 0.0, "none", 0.0),
        # Small manual spread for ordinary numeric knobs.
        SmoothingExperimentConfig("uniform_m0_1_cap0_05", "uniform_prior", 0, 0.1, "none", 0.05),
        SmoothingExperimentConfig("uniform_m0_5_cap0_10", "uniform_prior", 0, 0.5, "none", 0.10),
        SmoothingExperimentConfig("uniform_m1_0_cap0_20", "uniform_prior", 0, 1.0, "none", 0.20),
        SmoothingExperimentConfig("knn5_m0_5_cap0_10", "generic_embedding_kNN", 5, 0.5, "none", 0.10),
        SmoothingExperimentConfig("knn10_m0_5_cap0_10", "generic_embedding_kNN", 10, 0.5, "none", 0.10),
        SmoothingExperimentConfig("knn25_m0_5_cap0_20", "generic_embedding_kNN", 25, 0.5, "none", 0.20),
        SmoothingExperimentConfig("knn10_m0_5_source2_cap0_10", "tier_evidence_filtered_kNN", 10, 0.5, "source_count>=2", 0.10),
        SmoothingExperimentConfig("knn10_m0_5_anchor_cap0_10", "tier_evidence_filtered_kNN", 10, 0.5, "anchored_mass", 0.10),
        # Hypothesis batch: generic space may be too topical; transform the vector space.
        SmoothingExperimentConfig("centered_knn10_m0_5_cap0_10", "generic_embedding_kNN", 10, 0.5, "none", 0.10, "global_center"),
        SmoothingExperimentConfig("slot_centered_knn10_m0_5_cap0_10", "generic_embedding_kNN", 10, 0.5, "none", 0.10, "slot_center"),
        SmoothingExperimentConfig("pc1_removed_knn10_m0_5_cap0_10", "generic_embedding_kNN", 10, 0.5, "none", 0.10, "remove_pc1"),
        SmoothingExperimentConfig("pc3_removed_knn10_m0_5_cap0_10", "generic_embedding_kNN", 10, 0.5, "none", 0.10, "remove_pc3"),
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


def _hard_label_source_labels(
    source_labels: dict[int, _SourceLabel],
) -> dict[int, _SourceLabel]:
    return {
        subtitle_id: (
            _SourceLabel(label.tier, 1.0)
            if label.tier in TIERS and label.confidence is not None
            else label
        )
        for subtitle_id, label in source_labels.items()
    }


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


def _labeled_residual_shares(
    anchor: str,
    teacher_vector: dict[str, float] | None,
    corpus_prior: dict[str, float],
) -> dict[str, float]:
    """Residual split for a labeled source's ``(1 - confidence)`` mass.

    Step 4b (#44): split the residual by the source's *own* teacher score-vector
    rather than the corpus-wide label-marginal prior. Drop the anchored tier,
    renormalize over the remaining two tiers, and use those shares. Fall back to
    ``corpus_prior`` when the teacher vector is missing or degenerate -- either
    all of its mass sits on the anchor tier, or the off-anchor mass is so small
    (or non-finite) that the split would be numerical noise rather than signal.
    """
    other_tiers = [tier for tier in TIERS if tier != anchor]
    if teacher_vector is None:
        return corpus_prior
    other_mass = sum(max(0.0, teacher_vector.get(tier, 0.0)) for tier in other_tiers)
    if not math.isfinite(other_mass) or other_mass <= 1e-9:
        return corpus_prior
    return {
        tier: max(0.0, teacher_vector.get(tier, 0.0)) / other_mass
        for tier in other_tiers
    }


def _scored_source_ids(conn: sqlite3.Connection) -> set[int]:
    """Sources with at least one strict filler carrying a real model score.

    Used to distinguish a genuine teacher score-vector from the uniform vector
    that ``_load_source_fallback_vectors`` emits when every contributing filler
    lacks a (non-zero) ``slot_filler_model_scores`` row. Step 4b treats the
    latter as a *missing* vector and falls back to the corpus prior.
    """
    rows = conn.execute(
        """
        SELECT DISTINCT sfs.subtitle_id
        FROM slot_filler_sources sfs
        JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
        JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
        WHERE sf.mode = 'strict'
          AND (ms.score_pop != 0.0 OR ms.score_mainstream != 0.0 OR ms.score_niche != 0.0)
        """
    ).fetchall()
    return {int(row[0]) for row in rows}


def _evidence_source_ids(conn: sqlite3.Connection) -> set[int]:
    """Subtitle ids that contribute at least one strict filler link (evidence)."""
    rows = conn.execute(
        """
        SELECT DISTINCT sfs.subtitle_id
        FROM slot_filler_sources sfs
        JOIN slot_fillers sf ON sf.id = sfs.slot_filler_id
        WHERE sf.mode = 'strict'
        """
    ).fetchall()
    return {int(row[0]) for row in rows}


def _source_reliability_weights(
    source_labels: dict[int, _SourceLabel],
    evidence_source_ids: set[int],
    *,
    exponent: float,
    unlabeled_reliability: float,
) -> dict[int, float]:
    """Map each source subtitle_id to a reliability weight in [0, 1].

    Labeled sources use the (non-circular) LLM confidence as the signal, but the
    weight is lower-bounded at the unlabeled level so a labeled source is always
    at least as reliable as an unlabeled one -- and strictly more when confidence
    is positive, the exponent is finite and non-degenerate, and
    ``unlabeled_reliability < 1``:
    ``r = unlabeled_reliability + (1 - unlabeled_reliability) * confidence ** exponent``.

    Unlabeled sources get the flat ``unlabeled_reliability`` constant -- which is
    therefore also the floor of the labeled range. There is no per-source signal
    with real spread for unlabeled sources in this corpus (a source emits only a
    handful of strict fillers, so link counts are nearly constant), and the
    teacher score-vector is circular, so we do not pretend to discriminate them.
    Their score-vector entropy/margin is reported as a diagnostic only.
    """
    subtitle_ids = set(evidence_source_ids) | set(source_labels)
    weights: dict[int, float] = {}
    for subtitle_id in subtitle_ids:
        label = source_labels.get(subtitle_id)
        if label and label.tier in TIERS and label.confidence is not None:
            signal = min(1.0, max(0.0, label.confidence))
            weights[subtitle_id] = (
                unlabeled_reliability
                + (1.0 - unlabeled_reliability) * (signal ** exponent)
            )
        else:
            weights[subtitle_id] = unlabeled_reliability
    return weights


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
    reliability_weights: dict[int, float] | None = None,
    residual_from_teacher_vector: bool = False,
    scored_source_ids: set[int] | None = None,
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
        reliability = (
            reliability_weights.get(subtitle_id, 0.0)
            if reliability_weights is not None
            else None
        )
        label = source_labels.get(subtitle_id)
        if label and label.tier in TIERS and label.confidence is not None:
            corpus_prior = residual_priors[label.tier]
            if residual_from_teacher_vector:
                teacher_vector = source_fallbacks.get(subtitle_id)
                if scored_source_ids is not None and subtitle_id not in scored_source_ids:
                    teacher_vector = None
                residual_prior = _labeled_residual_shares(
                    label.tier,
                    teacher_vector,
                    corpus_prior,
                )
            else:
                residual_prior = corpus_prior
            _add_labeled_source(
                cells,
                filler_id=filler_key,
                subtitle_id=subtitle_id,
                label=label,
                residual_prior=residual_prior,
                inferred_source_weight=inferred_source_weight,
                reliability=reliability,
            )
        else:
            vector = source_fallbacks.get(subtitle_id, global_fallback)
            _add_inferred_source(
                cells,
                filler_id=filler_key,
                subtitle_id=subtitle_id,
                vector=vector,
                weight=inferred_source_weight,
                reliability=reliability,
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
    reliability: float | None = None,
) -> None:
    assert label.tier is not None and label.confidence is not None
    confidence = label.confidence
    # Anchoring (shape) keeps per-source total mass at 1.0; reliability (magnitude)
    # scales that total. When reliability is applied the residual is NOT re-scaled
    # by inferred_source_weight, so reliability is the only magnitude axis.
    if reliability is None:
        anchored_mass = confidence
        residual = max(0.0, 1.0 - confidence) * inferred_source_weight
    else:
        anchored_mass = confidence * reliability
        residual = max(0.0, 1.0 - confidence) * reliability
    anchored = cells[(filler_id, label.tier)]
    anchored.anchored_soft_count += anchored_mass
    anchored.anchored_sources.add(subtitle_id)
    anchored.confidence_sum += confidence
    anchored.confidence_count += 1
    if reliability is not None:
        anchored.reliability_sum += reliability
        anchored.reliability_count += 1
    for tier, share in residual_prior.items():
        if residual <= 0:
            continue
        cell = cells[(filler_id, tier)]
        cell.inferred_soft_count += residual * share
        cell.inferred_sources.add(subtitle_id)
        cell.confidence_sum += confidence
        cell.confidence_count += 1
        if reliability is not None:
            cell.reliability_sum += reliability
            cell.reliability_count += 1


def _add_inferred_source(
    cells: dict[tuple[str, str], _EvidenceCell],
    *,
    filler_id: str,
    subtitle_id: int,
    vector: dict[str, float],
    weight: float,
    reliability: float | None = None,
) -> None:
    effective = weight if reliability is None else reliability
    if effective <= 0:
        return
    for tier in TIERS:
        mass = vector[tier] * effective
        if mass <= 0:
            continue
        cell = cells[(filler_id, tier)]
        cell.inferred_soft_count += mass
        cell.inferred_sources.add(subtitle_id)
        if reliability is not None:
            cell.reliability_sum += reliability
            cell.reliability_count += 1


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
                "reliability_mean": cell.reliability_mean,
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
    hard_label_rows: list[dict[str, object]],
    weighted_rows: list[dict[str, object]],
    weighted_corpus_residual_rows: list[dict[str, object]],
    fillers: dict[str, _Filler],
    source_labels: dict[int, _SourceLabel],
    source_fallbacks: dict[int, dict[str, float]],
    evidence_source_ids: set[int],
    reliability_weights: dict[int, float],
    residual_priors: dict[str, dict[str, float]],
    alpha: float,
    inferred_source_weight: float,
    reliability_exponent: float,
    unlabeled_reliability: float,
    weighted_path: Path,
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
    confidence_summary = _confidence_summary(source_labels)
    hard_label_comparison = _distribution_comparison(
        rows, hard_label_rows, left_label="confidence", right_label="hard"
    )
    reliability_comparison = _distribution_comparison(
        weighted_rows, rows, left_label="weighted", right_label="anchored"
    )
    teacher_diagnostics = _teacher_vector_diagnostics(
        source_labels, source_fallbacks, evidence_source_ids, reliability_weights
    )
    reliability_movers = _reliability_movers(rows, weighted_rows)
    residual_movers = _reliability_movers(weighted_corpus_residual_rows, weighted_rows)
    collapse_flags = _collapse_guardrail(rows, weighted_rows)
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
        f"| reliability_exponent (report-only) | {reliability_exponent:.6g} |",
        f"| unlabeled_reliability (report-only) | {unlabeled_reliability:.6g} |",
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
        "## Label confidence diagnostics",
        "",
        "`llm_market_tier_confidence` is used as anchored probability mass for "
        "the labeled tier (the shape axis). Source reliability weighting (the "
        "magnitude axis) is computed separately and reported below; it is "
        "written to a sidecar CSV and does not change the served artifact.",
        "",
        "| Label | Count | Mean | Min | P10 | Median | P90 | Max | <0.70 | 0.70-0.85 | 0.85-0.95 | >=0.95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in confidence_summary:
        lines.append(
            f"| {row['tier']} | {row['count']:,} | {row['mean']:.3f} | "
            f"{row['min']:.3f} | {row['p10']:.3f} | {row['median']:.3f} | "
            f"{row['p90']:.3f} | {row['max']:.3f} | {row['lt_070']:,} | "
            f"{row['b_070_085']:,} | {row['b_085_095']:,} | {row['gte_095']:,} |"
        )
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
        "## Confidence anchoring comparison",
        "",
        "This compares the actual confidence-anchored distribution against a "
        "hard-label variant where every LLM-labeled source contributes 1.0 mass "
        "to its labeled tier and no residual mass to the other tiers. This is a "
        "diagnostic only; the emitted artifact keeps the actual confidence-"
        "anchored probabilities.",
        "",
        "| Slot type | Tier | JS divergence | Top-20 overlap | Largest confidence-driven increases | Largest hard-label increases |",
        "|---|---:|---:|---:|---|---|",
    ])
    for row in hard_label_comparison:
        lines.append(
            f"| {row['slot_type']} | {row['tier']} | "
            f"{row['js_divergence']:.6f} | {row['top20_overlap']}/20 | "
            f"{row['left_increases']} | {row['right_increases']} |"
        )
    lines.extend([
        "",
        "## Source reliability weighting (report-only)",
        "",
        "Reliability scales each source's total contributed mass (the magnitude "
        "axis), while confidence anchoring keeps controlling per-source shape.",
        "",
        "- Labeled sources use a confidence-driven weight "
        "`r = unlabeled_reliability + (1 - unlabeled_reliability) * confidence ** exponent` "
        "(`signal = llm_market_tier_confidence`, non-circular). The "
        "`unlabeled_reliability` lower bound guarantees a labeled source is "
        "always at least as reliable as an unlabeled one.",
        "- Unlabeled sources get a flat `unlabeled_reliability` constant. There "
        "is no per-source signal with real spread for them in this corpus (a "
        "source emits only a handful of strict fillers, so link counts are "
        "nearly constant), and the teacher score-vector is circular, so we do "
        "not pretend to discriminate them.",
        "",
        "For labeled sources the weighted decomposition keeps reliability as the "
        "sole magnitude axis: total mass is `r`, the anchored tier gets "
        "`confidence * r`, and the residual `(1 - confidence) * r` is split "
        "across the other tiers (the residual is NOT re-scaled by "
        "`inferred_source_weight`). The **served** artifact splits that residual "
        "by the corpus label-marginal prior; the **confidence-weighted sidecar** "
        "splits it by each source's own teacher score-vector (Step 4b, #44) and "
        "falls back to the corpus prior when that vector is degenerate. Unlabeled "
        "sources contribute `teacher_vector * r`.",
        "",
        "| Source group | Sources | Mean r | Min r | P10 r | Median r | P90 r | Max r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in teacher_diagnostics["reliability"]:
        lines.append(
            f"| {row['group']} | {row['count']:,} | {row['mean']:.3f} | "
            f"{row['min']:.3f} | {row['p10']:.3f} | {row['median']:.3f} | "
            f"{row['p90']:.3f} | {row['max']:.3f} |"
        )
    lines.extend([
        "",
        "## Teacher-output confidence diagnostics (circular; diagnostic only)",
        "",
        "These summarize the teacher score-vector for unlabeled sources "
        "(`slot_filler_model_scores`). Vector entropy/margin are derived from the "
        "same labeling signal, so they are reported as diagnostics only and are "
        "NOT used to set reliability.",
        "",
        f"- Unlabeled sources with a teacher vector: "
        f"{teacher_diagnostics['vector_coverage']:,} of "
        f"{teacher_diagnostics['unlabeled_total']:,}.",
        "",
        "| Metric | Count | Mean | Min | P10 | Median | P90 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in teacher_diagnostics["vector"]:
        lines.append(
            f"| {row['metric']} | {row['count']:,} | {row['mean']:.3f} | "
            f"{row['min']:.3f} | {row['p10']:.3f} | {row['median']:.3f} | "
            f"{row['p90']:.3f} | {row['max']:.3f} |"
        )
    lines.extend([
        "",
        "## Three-way distribution comparison",
        "",
        "Named variants: **hard-label anchoring** (confidence=1, reliability=1), "
        "**anchored-only** (the served artifact), and **confidence-weighted** "
        "(anchored shape scaled by source reliability). The hard-vs-anchored "
        "table above isolates the *anchoring effect*; the table below isolates "
        "the *reliability effect* (confidence-weighted vs anchored-only).",
        "",
        "| Slot type | Tier | JS divergence | Top-20 overlap | Largest reliability-driven increases | Largest reliability-driven decreases |",
        "|---|---:|---:|---:|---|---|",
    ])
    for row in reliability_comparison:
        lines.append(
            f"| {row['slot_type']} | {row['tier']} | "
            f"{row['js_divergence']:.6f} | {row['top20_overlap']}/20 | "
            f"{row['left_increases']} | {row['right_increases']} |"
        )
    lines.extend([
        "",
        "## Reliability movers (weighted vs anchored)",
        "",
        "Largest probability shifts from applying source reliability, with the "
        "supporting evidence (anchored vs weighted soft count, weighted source "
        "count, and mean reliability across the contributing evidence events for "
        "the cell). The reliability mean is a per-contribution diagnostic, not a "
        "unique-source average.",
        "",
        "| Slot type | Tier | Filler | Anchored p | Weighted p | Delta | Anchored soft | Weighted soft | Weighted sources | Mean reliability |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in reliability_movers:
        reliability_mean = (
            f"{row['reliability_mean']:.3f}"
            if row["reliability_mean"] is not None
            else "-"
        )
        lines.append(
            f"| {row['slot_type']} | {row['tier']} | "
            f"{row['display_filler']} [{row['filler']}] | "
            f"{row['anchored_probability']:.5f} | {row['weighted_probability']:.5f} | "
            f"{row['delta']:+.5f} | {row['anchored_soft']:.3f} | "
            f"{row['weighted_soft']:.3f} | {row['weighted_source_count']:,} | "
            f"{reliability_mean} |"
        )
    lines.extend([
        "",
        "## Source-aware residual direction (Step 4b)",
        "",
        "For a labeled source, the served artifact splits its "
        "`(1 - confidence)` residual across the two non-anchor tiers by a single "
        "corpus-wide label-marginal prior. The confidence-weighted sidecar "
        "instead splits that residual by the source's own teacher score-vector "
        "(drop the anchor tier, renormalize over the other two), falling back to "
        "the corpus prior when the vector is degenerate. The table below isolates "
        "fillers whose weighted probability moved purely because of this "
        "*direction* change: both variants use identical reliability magnitudes, "
        "so the only difference is where each labeled source's residual landed.",
        "",
        "| Slot type | Tier | Filler | Corpus-prior p | Teacher-vector p | Delta |",
        "|---|---|---|---:|---:|---:|",
    ])
    residual_movers_shown = [
        row for row in residual_movers if abs(float(row["delta"])) > 1e-9
    ]
    if residual_movers_shown:
        for row in residual_movers_shown:
            lines.append(
                f"| {row['slot_type']} | {row['tier']} | "
                f"{row['display_filler']} [{row['filler']}] | "
                f"{row['anchored_probability']:.5f} | "
                f"{row['weighted_probability']:.5f} | {row['delta']:+.5f} |"
            )
    else:
        lines.append(
            "| _(no movement: every labeled residual fell back to the corpus "
            "prior)_ | | | | | |"
        )
    lines.extend([
        "",
        "## Pop/mainstream collapse guardrail",
        "",
        "Flags (tier, slot) groups where reliability weighting shrinks effective "
        "N by more than 20% relative to the anchored-only distribution. This is "
        "an early-warning check that reliability weighting is not collapsing the "
        "popular/mainstream tails. It detects head-concentration collapse; the "
        "opposite failure (evidence washing out toward the uniform prior) shows "
        "up instead as high JS divergence in the reliability-effect comparison "
        "above. Niche is intentionally excluded from this tail check.",
        "",
    ])
    if collapse_flags:
        lines.extend([
            "| Slot type | Tier | Anchored effective N | Weighted effective N | Drop |",
            "|---|---|---:|---:|---:|",
        ])
        for row in collapse_flags:
            lines.append(
                f"| {row['slot_type']} | {row['tier']} | "
                f"{row['anchored_effective_n']:.1f} | "
                f"{row['weighted_effective_n']:.1f} | "
                f"{row['drop']:.1%} |"
            )
    else:
        lines.append("No (tier, slot) group exceeded the 20% effective-N drop threshold.")
    lines.extend([
        "",
        f"Confidence-weighted sidecar artifact: `{weighted_path.name}` "
        "(analysis-only; not served).",
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


def _confidence_summary(source_labels: dict[int, _SourceLabel]) -> list[dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for label in source_labels.values():
        if label.tier in TIERS and label.confidence is not None:
            grouped[label.tier].append(label.confidence)
    summary: list[dict[str, object]] = []
    for tier in TIERS:
        values = sorted(grouped.get(tier, []))
        if not values:
            continue
        summary.append({
            "tier": tier,
            "count": len(values),
            "mean": sum(values) / len(values),
            "min": values[0],
            "p10": _quantile(values, 0.10),
            "median": _quantile(values, 0.50),
            "p90": _quantile(values, 0.90),
            "max": values[-1],
            "lt_070": sum(value < 0.70 for value in values),
            "b_070_085": sum(0.70 <= value < 0.85 for value in values),
            "b_085_095": sum(0.85 <= value < 0.95 for value in values),
            "gte_095": sum(value >= 0.95 for value in values),
        })
    return summary


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise RuntimeError("Cannot compute quantile of empty values")
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[int(position)]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _vector_entropy(vector: dict[str, float]) -> float:
    total = sum(max(0.0, vector[tier]) for tier in TIERS)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for tier in TIERS:
        share = max(0.0, vector[tier]) / total
        if share > 0:
            entropy -= share * math.log(share)
    return entropy


def _vector_margin(vector: dict[str, float]) -> float:
    ordered = sorted((max(0.0, vector[tier]) for tier in TIERS), reverse=True)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    return (ordered[0] - ordered[1]) / total


def _summary_row(label: str, key: str, values: list[float]) -> dict[str, object]:
    values = sorted(values)
    return {
        key: label,
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": values[0],
        "p10": _quantile(values, 0.10),
        "median": _quantile(values, 0.50),
        "p90": _quantile(values, 0.90),
        "max": values[-1],
    }


def _teacher_vector_diagnostics(
    source_labels: dict[int, _SourceLabel],
    source_fallbacks: dict[int, dict[str, float]],
    evidence_source_ids: set[int],
    reliability_weights: dict[int, float],
) -> dict[str, object]:
    labeled_ids = {
        subtitle_id
        for subtitle_id, label in source_labels.items()
        if label.tier in TIERS and label.confidence is not None
    }
    # Diagnostics describe reliability for sources that actually contribute
    # evidence (strict links). Labels with no strict link never reach the
    # sidecar, so excluding them keeps the reported means/counts honest.
    evidence_ids = set(evidence_source_ids)
    unlabeled_ids = [
        subtitle_id for subtitle_id in evidence_ids if subtitle_id not in labeled_ids
    ]
    labeled_evidence_ids = [
        subtitle_id for subtitle_id in evidence_ids if subtitle_id in labeled_ids
    ]

    reliability_rows: list[dict[str, object]] = []
    labeled_weights = [
        reliability_weights[s] for s in labeled_evidence_ids if s in reliability_weights
    ]
    if labeled_weights:
        reliability_rows.append(_summary_row("labeled (confidence)", "group", labeled_weights))
    unlabeled_weights = [
        reliability_weights[s] for s in unlabeled_ids if s in reliability_weights
    ]
    if unlabeled_weights:
        reliability_rows.append(_summary_row("unlabeled (flat)", "group", unlabeled_weights))

    entropies: list[float] = []
    margins: list[float] = []
    for subtitle_id in unlabeled_ids:
        vector = source_fallbacks.get(subtitle_id)
        if vector is None:
            continue
        entropies.append(_vector_entropy(vector))
        margins.append(_vector_margin(vector))
    vector_rows: list[dict[str, object]] = []
    if entropies:
        vector_rows.append(_summary_row("teacher vector entropy", "metric", entropies))
    if margins:
        vector_rows.append(_summary_row("teacher vector margin", "metric", margins))

    return {
        "reliability": reliability_rows,
        "vector": vector_rows,
        "vector_coverage": len(entropies),
        "unlabeled_total": len(unlabeled_ids),
    }


def _reliability_movers(
    anchored_rows: list[dict[str, object]],
    weighted_rows: list[dict[str, object]],
    *,
    limit: int = 20,
) -> list[dict[str, object]]:
    anchored_lookup = {
        (row["slot_type"], row["tier"], row["filler"]): row for row in anchored_rows
    }
    movers: list[dict[str, object]] = []
    for weighted in weighted_rows:
        key = (weighted["slot_type"], weighted["tier"], weighted["filler"])
        anchored = anchored_lookup.get(key)
        if anchored is None:
            continue
        anchored_soft = float(anchored["soft_count"])
        weighted_soft = float(weighted["soft_count"])
        if anchored_soft <= 0 and weighted_soft <= 0:
            continue
        delta = float(weighted["probability"]) - float(anchored["probability"])
        movers.append({
            "slot_type": str(weighted["slot_type"]),
            "tier": str(weighted["tier"]),
            "filler": str(weighted["filler"]),
            "display_filler": str(weighted["display_filler"]),
            "anchored_probability": float(anchored["probability"]),
            "weighted_probability": float(weighted["probability"]),
            "delta": delta,
            "anchored_soft": anchored_soft,
            "weighted_soft": weighted_soft,
            "weighted_source_count": int(weighted["source_count"]),
            "reliability_mean": weighted["reliability_mean"],
        })
    movers.sort(key=lambda item: (-abs(item["delta"]), item["filler"].lower()))
    return movers[:limit]


def _collapse_guardrail(
    anchored_rows: list[dict[str, object]],
    weighted_rows: list[dict[str, object]],
    *,
    threshold: float = 0.20,
    tiers: tuple[str, ...] = ("pop", "mainstream"),
) -> list[dict[str, object]]:
    anchored = {
        (row["slot_type"], row["tier"]): row for row in _group_summary(anchored_rows)
    }
    weighted = {
        (row["slot_type"], row["tier"]): row for row in _group_summary(weighted_rows)
    }
    flags: list[dict[str, object]] = []
    for key, anchored_summary in anchored.items():
        if key[1] not in tiers:
            continue
        weighted_summary = weighted.get(key)
        if weighted_summary is None:
            continue
        anchored_eff = float(anchored_summary["effective_n"])
        weighted_eff = float(weighted_summary["effective_n"])
        if anchored_eff <= 0:
            continue
        drop = (anchored_eff - weighted_eff) / anchored_eff
        if drop > threshold:
            flags.append({
                "slot_type": key[0],
                "tier": key[1],
                "anchored_effective_n": anchored_eff,
                "weighted_effective_n": weighted_eff,
                "drop": drop,
            })
    flags.sort(key=lambda item: -item["drop"])
    return flags


def _distribution_comparison(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    *,
    left_label: str = "confidence",
    right_label: str = "hard",
) -> list[dict[str, object]]:
    right_lookup = {
        (row["slot_type"], row["tier"], row["filler"]): row
        for row in right_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in left_rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    comparison: list[dict[str, object]] = []
    for (slot_type, tier), group_rows in sorted(grouped.items()):
        left_probabilities = [float(row["probability"]) for row in group_rows]
        right_probabilities = [
            float(right_lookup[(row["slot_type"], row["tier"], row["filler"])]["probability"])
            for row in group_rows
        ]
        deltas = []
        for row, left_probability, right_probability in zip(
            group_rows,
            left_probabilities,
            right_probabilities,
        ):
            deltas.append({
                "filler": str(row["filler"]),
                "display_filler": str(row["display_filler"]),
                "left_probability": left_probability,
                "right_probability": right_probability,
                "delta": left_probability - right_probability,
            })
        left_top = {
            item["filler"]
            for item in sorted(
                deltas,
                key=lambda item: (-item["left_probability"], item["filler"].lower()),
            )[:20]
        }
        right_top = {
            item["filler"]
            for item in sorted(
                deltas,
                key=lambda item: (-item["right_probability"], item["filler"].lower()),
            )[:20]
        }
        comparison.append({
            "slot_type": slot_type,
            "tier": tier,
            "js_divergence": _js_divergence(left_probabilities, right_probabilities),
            "top20_overlap": len(left_top & right_top),
            "left_increases": _format_confidence_delta_examples(
                sorted(deltas, key=lambda item: (-item["delta"], item["filler"].lower()))[:5],
                left_label=left_label,
                right_label=right_label,
            ),
            "right_increases": _format_confidence_delta_examples(
                sorted(deltas, key=lambda item: (item["delta"], item["filler"].lower()))[:5],
                left_label=left_label,
                right_label=right_label,
            ),
        })
    return comparison


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


def _format_confidence_delta_examples(
    examples: list[dict[str, object]],
    *,
    left_label: str,
    right_label: str,
) -> str:
    return "; ".join(
        f"{item['display_filler']} [{item['filler']}] "
        f"({right_label}={float(item['right_probability']):.5f}, "
        f"{left_label}={float(item['left_probability']):.5f}, "
        f"delta={float(item['delta']):+.5f})"
        for item in examples
    )


def _load_smoothing_vectors(
    conn: sqlite3.Connection,
    fillers: dict[str, _Filler],
    cache_path: Path,
    *,
    vector_source: str,
) -> tuple[dict[str, list[float]], dict[str, int]]:
    if vector_source == "db":
        vectors = _load_persisted_vectors(conn)
        return vectors, {"persisted_db": len(vectors), "offline_spacy": 0}
    if vector_source != "offline_spacy":
        raise RuntimeError("vector_source must be 'offline_spacy' or 'db'")
    persisted = _load_embedding_cache(cache_path)
    missing = [
        filler
        for key, filler in fillers.items()
        if key not in persisted
    ]
    if missing:
        _append_spacy_embeddings(cache_path, missing)
        persisted = _load_embedding_cache(cache_path)
    persisted_keys = set(_load_persisted_vectors(conn))
    return persisted, {
        "persisted_db": len(persisted_keys & set(persisted)),
        "offline_spacy": len(set(persisted) - persisted_keys),
    }


def _load_persisted_vectors(conn: sqlite3.Connection) -> dict[str, list[float]]:
    import numpy as np

    rows = conn.execute(
        """
        SELECT id, slot_type, filler, vector_sum, token_count
        FROM slot_fillers
        WHERE mode = 'strict'
          AND vector_sum IS NOT NULL
          AND token_count IS NOT NULL
          AND token_count > 0
        """
    ).fetchall()
    grouped: dict[str, tuple[object, int]] = {}
    for _filler_id, slot_type, filler, vector_sum, token_count in rows:
        key = _filler_key(slot_type, filler)
        vector = np.frombuffer(vector_sum, dtype=np.float32).astype(np.float64)
        prior_vector, prior_count = grouped.get(key, (np.zeros_like(vector), 0))
        grouped[key] = (prior_vector + vector, prior_count + int(token_count))
    vectors: dict[str, list[float]] = {}
    for key, (vector_sum, token_count) in grouped.items():
        vector = vector_sum / token_count
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vectors[key] = (vector / norm).tolist()
    return vectors


def _transform_vectors(
    vectors: dict[str, list[float]],
    vector_transform: str,
) -> dict[str, list[float]]:
    if vector_transform == "raw":
        return vectors
    import numpy as np

    keys = sorted(vectors)
    if not keys:
        return {}
    matrix = np.array([vectors[key] for key in keys], dtype=np.float64)
    if vector_transform == "global_center":
        transformed = matrix - matrix.mean(axis=0, keepdims=True)
    elif vector_transform == "slot_center":
        transformed = matrix.copy()
        slots = sorted({key.split("\0", 1)[0] for key in keys})
        for slot_type in slots:
            indexes = [
                index for index, key in enumerate(keys)
                if key.startswith(slot_type + "\0")
            ]
            if indexes:
                transformed[indexes] -= transformed[indexes].mean(axis=0, keepdims=True)
    elif vector_transform.startswith("remove_pc"):
        count = int(vector_transform.removeprefix("remove_pc"))
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:count]
        transformed = centered - centered @ components.T @ components
    else:
        raise RuntimeError(f"Unknown vector_transform: {vector_transform}")
    output: dict[str, list[float]] = {}
    norms = np.linalg.norm(transformed, axis=1)
    for key, vector, norm in zip(keys, transformed, norms):
        if norm > 0:
            output[key] = (vector / norm).tolist()
    return output


def _load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    vectors: dict[str, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            vector = [float(value) for value in row["vector"].split(" ") if value]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm > 0:
                vectors[row["key"]] = [value / norm for value in vector]
    return vectors


def _append_spacy_embeddings(path: Path, fillers: list[_Filler]) -> None:
    import spacy

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_embedding_cache(path)
    nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("key", "slot_type", "filler", "display_filler", "vector"),
        )
        if write_header:
            writer.writeheader()
        docs = nlp.pipe([filler.filler for filler in fillers], batch_size=256)
        for filler, doc in zip(fillers, docs):
            if filler.id in existing or not doc.has_vector or doc.vector_norm <= 0:
                continue
            vector = [float(value) for value in doc.vector]
            norm = math.sqrt(sum(value * value for value in vector))
            if norm <= 0:
                continue
            writer.writerow({
                "key": filler.id,
                "slot_type": filler.slot_type,
                "filler": filler.filler,
                "display_filler": filler.display_filler,
                "vector": " ".join(f"{value / norm:.9g}" for value in vector),
            })


def _apply_smoothing(
    rows: list[dict[str, object]],
    fillers: dict[str, _Filler],
    vectors: dict[str, list[float]],
    config: SmoothingExperimentConfig,
) -> list[dict[str, object]]:
    if config.variant == "none":
        return [dict(row) for row in rows]
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    smoothed: list[dict[str, object]] = []
    for (slot_type, tier), group_rows in sorted(grouped.items()):
        if config.variant == "uniform_prior":
            semantic_prior = {
                str(row["filler"]): 1.0 / len(group_rows)
                for row in group_rows
            }
        elif config.variant in {"generic_embedding_kNN", "tier_evidence_filtered_kNN"}:
            semantic_prior = _semantic_prior_for_group(
                group_rows,
                vectors,
                neighbor_count=config.neighbor_count,
                evidence_gate=config.evidence_gate if config.variant == "tier_evidence_filtered_kNN" else "none",
            )
        else:
            raise RuntimeError(f"Unknown smoothing variant: {config.variant}")
        group_smoothed = []
        for row in group_rows:
            filler = str(row["filler"])
            base_probability = float(row["probability"])
            evidence_strength = max(0.0, float(row["soft_count"]))
            if config.shrinkage <= 0:
                borrow_fraction = 0.0
            else:
                borrow_fraction = config.shrinkage / (evidence_strength + config.shrinkage)
            borrow_fraction = min(config.max_borrowed_mass, borrow_fraction)
            if (
                not _is_valid_ml_slot_filler(slot_type, str(row["display_filler"]))
                or (
                    _filler_key(slot_type, filler) not in vectors
                    and config.variant != "uniform_prior"
                )
            ):
                borrow_fraction = 0.0
            new_probability = (
                (1.0 - borrow_fraction) * base_probability
                + borrow_fraction * semantic_prior.get(filler, base_probability)
            )
            new_row = dict(row)
            new_row["probability"] = new_probability
            new_row["log_probability"] = math.log(new_probability) if new_probability > 0 else float("-inf")
            new_row["semantic_smoothing_mass"] = abs(new_probability - base_probability)
            group_smoothed.append(new_row)
        total = sum(float(row["probability"]) for row in group_smoothed)
        for row in group_smoothed:
            probability = float(row["probability"]) / total if total > 0 else 0.0
            row["probability"] = probability
            row["log_probability"] = math.log(probability) if probability > 0 else float("-inf")
            smoothed.append(row)
    return smoothed


def _semantic_prior_for_group(
    rows: list[dict[str, object]],
    vectors: dict[str, list[float]],
    *,
    neighbor_count: int,
    evidence_gate: str,
) -> dict[str, float]:
    import numpy as np

    slot_type = str(rows[0]["slot_type"]) if rows else ""
    keys = [
        str(row["filler"])
        for row in rows
        if _filler_key(slot_type, str(row["filler"])) in vectors
        and _is_valid_ml_slot_filler(slot_type, str(row["display_filler"]))
    ]
    if not keys or neighbor_count <= 0:
        return {str(row["filler"]): float(row["probability"]) for row in rows}
    matrix = np.array(
        [vectors[_filler_key(slot_type, key)] for key in keys],
        dtype=np.float64,
    )
    probabilities = {
        str(row["filler"]): float(row["probability"])
        for row in rows
    }
    evidence_allowed = {
        str(row["filler"])
        for row in rows
        if _passes_evidence_gate(row, evidence_gate)
        and _is_valid_ml_slot_filler(slot_type, str(row["display_filler"]))
    }
    prior: dict[str, float] = {}
    similarities = matrix @ matrix.T
    key_index = {key: index for index, key in enumerate(keys)}
    for key in keys:
        index = key_index[key]
        candidates = []
        for other_index, similarity in enumerate(similarities[index]):
            other_key = keys[other_index]
            if other_key == key or similarity <= 0:
                continue
            if other_key not in evidence_allowed:
                continue
            candidates.append((float(similarity), other_key))
        if not candidates:
            prior[key] = probabilities[key]
            continue
        candidates.sort(reverse=True)
        selected = candidates[:neighbor_count]
        total_weight = sum(weight for weight, _other_key in selected)
        prior[key] = sum(
            weight * probabilities[other_key]
            for weight, other_key in selected
        ) / total_weight
    for row in rows:
        key = str(row["filler"])
        prior.setdefault(key, probabilities[key])
    total = sum(prior.values())
    if total > 0:
        prior = {key: value / total for key, value in prior.items()}
    return prior


def _passes_evidence_gate(row: dict[str, object], evidence_gate: str) -> bool:
    if evidence_gate == "none":
        return True
    if evidence_gate == "source_count>=2":
        return int(row["source_count"]) >= 2
    if evidence_gate == "anchored_mass":
        return float(row["anchored_soft_count"]) > 0
    raise RuntimeError(f"Unknown evidence gate: {evidence_gate}")


@lru_cache(maxsize=1)
def _validation_nlp():
    return _load_nlp()


@lru_cache(maxsize=20000)
def _is_valid_ml_slot_filler(slot_type: str, filler: str) -> bool:
    if slot_type == "list_item":
        return _is_valid_list_item(filler, _validation_nlp())
    if slot_type == "action_noun":
        return _is_valid_action(filler, _validation_nlp())
    if slot_type == "of_object":
        return _is_valid_object(filler, _validation_nlp())
    # Remix subpart slots were produced from valid of-objects and have their own
    # runtime decomposition semantics; keep them eligible for smoothing.
    return True


def _smoothing_metrics(
    config: SmoothingExperimentConfig,
    base_rows: list[dict[str, object]],
    smoothed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    comparison = _distribution_comparison(smoothed_rows, base_rows)
    base_lookup = {
        (row["slot_type"], row["tier"], row["filler"]): row
        for row in base_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in smoothed_rows:
        grouped[(str(row["slot_type"]), str(row["tier"]))].append(row)
    result = []
    for item in comparison:
        key = (item["slot_type"], item["tier"])
        group_rows = grouped[key]
        probabilities = [float(row["probability"]) for row in group_rows]
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        moved = sum(
            abs(float(row["probability"]) - float(base_lookup[(row["slot_type"], row["tier"], row["filler"])]["probability"]))
            for row in group_rows
        ) / 2.0
        result.append({
            "experiment": config.name,
            "variant": config.variant,
            "neighbor_count": config.neighbor_count,
            "shrinkage": config.shrinkage,
            "evidence_gate": config.evidence_gate,
            "max_borrowed_mass": config.max_borrowed_mass,
            "vector_transform": config.vector_transform,
            "slot_type": item["slot_type"],
            "tier": item["tier"],
            "js_divergence": item["js_divergence"],
            "top20_overlap": item["top20_overlap"],
            "entropy": entropy,
            "effective_n": math.exp(entropy),
            "semantic_mass_moved": moved,
        })
    return result


def _write_smoothing_metrics(path: Path, metrics: list[dict[str, object]]) -> None:
    if not metrics:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)


def _read_smoothing_metrics(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    numeric_fields = {
        "neighbor_count": int,
        "shrinkage": float,
        "max_borrowed_mass": float,
        "vector_transform": str,
        "js_divergence": float,
        "top20_overlap": int,
        "entropy": float,
        "effective_n": float,
        "semantic_mass_moved": float,
    }
    for row in rows:
        for field, caster in numeric_fields.items():
            row[field] = caster(row[field])
    return rows


def _autoresearcher_findings(metrics: list[dict[str, object]]) -> list[dict[str, object]]:
    by_experiment: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in metrics:
        by_experiment[str(row["experiment"])].append(row)

    findings: list[dict[str, object]] = []
    for experiment, rows in sorted(by_experiment.items()):
        variant = str(rows[0]["variant"])
        if variant == "none":
            continue
        moved = [float(row["semantic_mass_moved"]) for row in rows]
        overlaps = [int(row["top20_overlap"]) for row in rows]
        js_values = [float(row["js_divergence"]) for row in rows]
        max_moved_row = max(rows, key=lambda row: float(row["semantic_mass_moved"]))
        min_overlap_row = min(rows, key=lambda row: int(row["top20_overlap"]))
        findings.append({
            "kind": "experiment_summary",
            "experiment": experiment,
            "variant": variant,
            "summary": (
                f"avg moved={sum(moved) / len(moved):.4f}, "
                f"max moved={max(moved):.4f} at "
                f"{max_moved_row['tier']} {max_moved_row['slot_type']}; "
                f"max JS={max(js_values):.4f}; min top-20 overlap={min(overlaps)}/20 "
                f"at {min_overlap_row['tier']} {min_overlap_row['slot_type']}"
            ),
            "severity": "info",
        })
        if min(overlaps) < 15:
            findings.append({
                "kind": "semantic_bleed_risk",
                "experiment": experiment,
                "variant": variant,
                "summary": (
                    f"Top-20 changed substantially for {min_overlap_row['tier']} "
                    f"{min_overlap_row['slot_type']} ({min(overlaps)}/20). "
                    "This needs qualitative tier/style review before any policy decision."
                ),
                "severity": "review",
            })
        if max(moved) > 0.05:
            findings.append({
                "kind": "large_mass_movement",
                "experiment": experiment,
                "variant": variant,
                "summary": (
                    f"Semantic mass moved reaches {max(moved):.4f}; treat this as a "
                    "sensitivity boundary rather than a candidate default."
                ),
                "severity": "review",
            })
        zero_move_slots = [
            f"{row['tier']} {row['slot_type']}"
            for row in rows
            if variant == "tier_evidence_filtered_kNN"
            and float(row["semantic_mass_moved"]) <= 0.000001
        ]
        if zero_move_slots:
            sample = ", ".join(zero_move_slots[:5])
            findings.append({
                "kind": "gate_under_smoothing",
                "experiment": experiment,
                "variant": variant,
                "summary": (
                    f"Evidence gate produced no measurable movement in "
                    f"{len(zero_move_slots)} tier/slot groups, including {sample}."
                ),
                "severity": "design",
            })
    return findings


def _next_smoothing_proposals(
    metrics: list[dict[str, object]],
    findings: list[dict[str, object]],
) -> list[dict[str, object]]:
    experiments_seen = {str(row["experiment"]) for row in metrics}
    has_top20_instability = any(finding["kind"] == "semantic_bleed_risk" for finding in findings)
    has_gate_under_smoothing = any(finding["kind"] == "gate_under_smoothing" for finding in findings)
    has_large_movement = any(finding["kind"] == "large_mass_movement" for finding in findings)

    proposals = [
        {
            "proposal_id": "manual-k5-cap005",
            "category": "fixed_sweep",
            "status": "ready_to_run" if "knn5_m0_5_cap0_05" not in experiments_seen else "already_run",
            "variant": "generic_embedding_kNN",
            "neighbor_count": 5,
            "shrinkage": 0.5,
            "evidence_gate": "none",
            "max_borrowed_mass": 0.05,
            "rationale": "Mechanical sensitivity check for a smaller, conservative neighborhood and cap.",
        },
        {
            "proposal_id": "manual-source2-cap005",
            "category": "fixed_sweep",
            "status": "ready_to_run",
            "variant": "tier_evidence_filtered_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "source_count>=2",
            "max_borrowed_mass": 0.05,
            "rationale": "Mechanical cap sensitivity check for the existing source-count evidence gate.",
        },
        {
            "proposal_id": "manual-anchored-gate",
            "category": "fixed_sweep",
            "status": "ready_to_run",
            "variant": "tier_evidence_filtered_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "anchored_mass",
            "max_borrowed_mass": 0.10,
            "rationale": "Simple evidence gate that does not require LLM steering.",
        },
        {
            "proposal_id": "weighted-similarity-knn",
            "category": "hypothesis_driven",
            "status": "needs_implementation",
            "variant": "weighted_similarity_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "source_count>=2",
            "max_borrowed_mass": 0.10,
            "rationale": (
                "If evidence gating under-smooths, combine cosine similarity with "
                "source support and confidence instead of a hard neighbor cutoff."
            ),
        },
        {
            "proposal_id": "slot-centered-vectors",
            "category": "hypothesis_driven",
            "status": "design_review_needed" if has_top20_instability else "defer",
            "variant": "centered_embedding_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "none",
            "max_borrowed_mass": 0.10,
            "rationale": (
                "Qualitative failures may indicate generic vectors are too topical; "
                "subtract per-slot centroids before cosine as a local subspace test."
            ),
        },
        {
            "proposal_id": "top-pc-removal",
            "category": "hypothesis_driven",
            "status": "design_review_needed" if has_top20_instability or has_large_movement else "defer",
            "variant": "pc_removed_embedding_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "none",
            "max_borrowed_mass": 0.10,
            "rationale": (
                "Only try whitening/top-PC removal after reviewing semantic bleed "
                "examples; this is a vector-space hypothesis, not a numeric sweep."
            ),
        },
    ]
    if has_gate_under_smoothing:
        proposals.append({
            "proposal_id": "soft-evidence-weighting",
            "category": "hypothesis_driven",
            "status": "design_review_needed",
            "variant": "evidence_weighted_kNN",
            "neighbor_count": 10,
            "shrinkage": 0.5,
            "evidence_gate": "none",
            "max_borrowed_mass": 0.10,
            "rationale": (
                "The hard source-count gate appears to shut off whole slots; inspect "
                "whether a soft support multiplier preserves movement without bleed."
            ),
        })
    return proposals


def _write_autoresearcher_proposals(path: Path, proposals: list[dict[str, object]]) -> None:
    if not proposals:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(proposals[0]))
        writer.writeheader()
        writer.writerows(proposals)


def _format_autoresearcher_report(
    *,
    ablation_result: SemanticSmoothingAblationResult,
    metrics: list[dict[str, object]],
    findings: list[dict[str, object]],
    proposals: list[dict[str, object]],
) -> str:
    lines = [
        "# Semantic smoothing AutoResearcher packet",
        "",
        "This is the deterministic local Step 5 AutoResearcher harvesting loop. "
        "It reruns the bounded ablation, inspects the metrics, and writes next-round "
        "proposals. It does **not** call an external LLM, does **not** choose serving "
        "defaults, and does **not** change runtime behavior.",
        "",
        "## Inputs",
        "",
        f"- Ablation report: `{ablation_result.report_path}`",
        f"- Metrics CSV: `{ablation_result.metrics_path}`",
        f"- Experiments inspected: {ablation_result.experiment_count}",
        f"- Metric rows inspected: {len(metrics)}",
        "",
        "## What is fixed-sweep vs hypothesis-driven",
        "",
        "- Fixed/manual sweeps: `k`, shrinkage `m`, max borrowed mass, and simple "
        "evidence gates. These should be small bounded sensitivity checks.",
        "- Hypothesis-driven AutoResearcher work: vector-space transformations, "
        "whitening/top-PC removal, weighted similarity, slot-specific spaces, "
        "style/humor/tier-fit concerns, and semantic bleed analysis. These require "
        "review of examples before adding or running the next variant.",
        "",
        "## Findings",
        "",
    ]
    if findings:
        for finding in findings:
            lines.append(
                f"- **{finding['kind']}** ({finding['severity']}, "
                f"{finding['experiment']}): {finding['summary']}"
            )
    else:
        lines.append("- No non-baseline findings were produced.")
    lines.extend([
        "",
        "## Proposed next experiments",
        "",
        "| Proposal | Category | Status | Variant | k | m | Gate | Cap | Rationale |",
        "|---|---|---|---|---:|---:|---|---:|---|",
    ])
    for proposal in proposals:
        lines.append(
            f"| {proposal['proposal_id']} | {proposal['category']} | "
            f"{proposal['status']} | {proposal['variant']} | "
            f"{proposal['neighbor_count']} | {float(proposal['shrinkage']):.2f} | "
            f"{proposal['evidence_gate']} | {float(proposal['max_borrowed_mass']):.2f} | "
            f"{proposal['rationale']} |"
        )
    lines.extend([
        "",
        "## Handoff",
        "",
        "- Treat this as a proposal packet, not a completed autonomous LLM loop.",
        "- A human or future LLM reviewer should inspect the ablation review examples "
        "before implementing vector-space proposals.",
        "- No smoothing variant should become a runtime default in Step 5.",
    ])
    return "\n".join(lines)


def _format_smoothing_report(
    *,
    base_rows: list[dict[str, object]],
    experiment_outputs: list[tuple[SmoothingExperimentConfig, list[dict[str, object]]]],
    metrics: list[dict[str, object]],
    vector_coverage: dict[str, tuple[int, int]],
    vector_source_counts: dict[str, int],
) -> str:
    lines = [
        "# Semantic smoothing ablation report",
        "",
        "This report compares bounded smoothing variants against the unsmoothed "
        "tier-slot distribution. It is an AutoResearcher-style experiment packet, "
        "not a serving-default decision.",
        "",
        "## Vector coverage",
        "",
        "Vector smoothing uses the offline all-slot embedding cache when requested, "
        "because persisted DB vectors only cover remix/object slots.",
        "",
        "| Vector source | Vectors |",
        "|---|---:|",
    ]
    for source, count in vector_source_counts.items():
        lines.append(f"| {source} | {count:,} |")
    lines.extend([
        "",
        "| Slot type | Fillers | With vectors |",
        "|---|---:|---:|",
    ])
    for slot_type, (total, with_vectors) in sorted(vector_coverage.items()):
        lines.append(f"| {slot_type} | {total:,} | {with_vectors:,} |")
    lines.extend([
        "",
        "## Experiment metrics",
        "",
        "| Experiment | Variant | Slot | Tier | JS divergence | Top-20 overlap | Effective N | Semantic mass moved |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in metrics:
        lines.append(
            f"| {row['experiment']} | {row['variant']} / {row['vector_transform']} | "
            f"{row['slot_type']} | {row['tier']} | {float(row['js_divergence']):.6f} | "
            f"{row['top20_overlap']}/20 | {float(row['effective_n']):.1f} | "
            f"{float(row['semantic_mass_moved']):.6f} |"
        )
    lines.extend(["", "## Review examples", ""])
    for config, rows in experiment_outputs:
        if config.variant == "none":
            continue
        lines.extend([f"### {config.name}", ""])
        for example in _smoothing_examples(config, base_rows, rows)[:12]:
            lines.append(
                f"- `{example['slot_type']}` {example['tier']} "
                f"**{example['display_filler']}** [{example['filler']}]: "
                f"{example['base_probability']:.6f} -> {example['smoothed_probability']:.6f} "
                f"({example['delta']:+.6f}); soft={example['soft_count']:.3f}, "
                f"sources={example['source_count']}, smoothing_mass={example['semantic_smoothing_mass']:.6f}"
            )
        lines.append("")
    lines.extend([
        "## Caveats",
        "",
        "- Persisted DB vectors still only cover remix/object slots, but this report uses the offline all-slot embedding cache so list-item and action-noun slots can participate in vector smoothing.",
        "- The output is for human review and future tuning; runtime defaults are unchanged.",
    ])
    return "\n".join(lines)


def _smoothing_examples(
    config: SmoothingExperimentConfig,
    base_rows: list[dict[str, object]],
    smoothed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_lookup = {
        (row["slot_type"], row["tier"], row["filler"]): row
        for row in base_rows
    }
    examples = []
    for row in smoothed_rows:
        base = base_lookup[(row["slot_type"], row["tier"], row["filler"])]
        delta = float(row["probability"]) - float(base["probability"])
        examples.append({
            "experiment": config.name,
            "slot_type": str(row["slot_type"]),
            "tier": str(row["tier"]),
            "filler": str(row["filler"]),
            "display_filler": str(row["display_filler"]),
            "base_probability": float(base["probability"]),
            "smoothed_probability": float(row["probability"]),
            "delta": delta,
            "soft_count": float(row["soft_count"]),
            "source_count": int(row["source_count"]),
            "semantic_smoothing_mass": float(row["semantic_smoothing_mass"]),
        })
    return sorted(examples, key=lambda item: (-abs(item["delta"]), item["filler"]))


def _vector_coverage(
    fillers: dict[str, _Filler],
    vectors: dict[str, list[float]],
) -> dict[str, tuple[int, int]]:
    totals: Counter[str] = Counter()
    covered: Counter[str] = Counter()
    for key, filler in fillers.items():
        totals[filler.slot_type] += 1
        if key in vectors:
            covered[filler.slot_type] += 1
    return {
        slot_type: (totals[slot_type], covered[slot_type])
        for slot_type in sorted(totals)
    }


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
