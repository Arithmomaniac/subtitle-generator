import csv
import json
import math
import sqlite3
from pathlib import Path

import pytest

from subtitle_generator.tier_slot_calibration import (
    CalibrationConfig,
    apply_temperature,
    assign_source_folds,
    build_tier_slot_calibration,
    calibration_verdict,
    heldout_evidence_counts,
    run_calibration_ablation,
    run_calibration_autoresearcher,
)
from subtitle_generator.tier_slot_distribution import (
    build_anchored_rows,
    build_tier_slot_distribution,
    load_distribution_inputs,
    load_strict_source_links,
)

TIERS = ("pop", "mainstream", "niche")
SLOTS = ("list_item", "action_noun")

# Each tier "owns" one filler per slot (its signature vocabulary); the other two
# fillers are shared minorities. Sources of a tier mostly use that tier's owned
# filler, so the per-tier distributions are distinct -- and training comes out
# sharper than the held-out reality, which is exactly what calibration tunes.
_TIER_OWNED = {
    "pop": {"list_item": "Alpha", "action_noun": "Rush"},
    "mainstream": {"list_item": "Bravo", "action_noun": "Build"},
    "niche": {"list_item": "Cosmo", "action_noun": "Drift"},
}
_SLOT_FILLERS = {
    "list_item": ("Alpha", "Bravo", "Cosmo"),
    "action_noun": ("Rush", "Build", "Drift"),
}
# Per-tier usage weights: the owned filler dominates but the minorities appear
# often enough that held-out folds carry them (so flattening can pay off).
_USAGE = (6, 2, 2)


def _create_calibration_db(sources_per_tier: int = 18) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL
        );
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL
        );
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL,
            PRIMARY KEY (slot_filler_id, subtitle_id)
        );
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )

    filler_ids: dict[tuple[str, str], int] = {}
    next_filler_id = 1
    for slot, fillers in _SLOT_FILLERS.items():
        for filler in fillers:
            filler_ids[(slot, filler)] = next_filler_id
            owning_tier = next(
                tier for tier in TIERS if _TIER_OWNED[tier][slot] == filler
            )
            scores = {tier: (0.7 if tier == owning_tier else 0.15) for tier in TIERS}
            conn.execute(
                "INSERT INTO slot_fillers VALUES (?, ?, ?, 'strict', ?, ?)",
                (next_filler_id, slot, filler, 10, 0.4),
            )
            conn.execute(
                "INSERT INTO slot_filler_model_scores VALUES (?, ?, ?, ?, ?, ?)",
                (
                    next_filler_id,
                    scores["pop"],
                    scores["mainstream"],
                    scores["niche"],
                    owning_tier,
                    3,
                ),
            )
            next_filler_id += 1

    subtitle_id = 1000
    for tier in TIERS:
        for _ in range(sources_per_tier):
            subtitle_id += 1
            conn.execute(
                "INSERT INTO pattern_matches VALUES (?, ?, ?, ?)",
                (subtitle_id, subtitle_id, tier, 0.9),
            )
            # Pick one filler per slot for this source, weighted toward the
            # tier's owned filler, deterministically by subtitle id.
            for slot in SLOTS:
                fillers = _SLOT_FILLERS[slot]
                # Reorder so the tier's owned filler is first, then minorities.
                owned = _TIER_OWNED[tier][slot]
                ordered = [owned] + [f for f in fillers if f != owned]
                bucket = subtitle_id % sum(_USAGE)
                cumulative = 0
                chosen = ordered[0]
                for filler, weight in zip(ordered, _USAGE):
                    cumulative += weight
                    if bucket < cumulative:
                        chosen = filler
                        break
                conn.execute(
                    "INSERT INTO slot_filler_sources VALUES (?, ?)",
                    (filler_ids[(slot, chosen)], subtitle_id),
                )
    conn.commit()
    return conn


def _group_sums(rows: list[dict[str, object]]) -> dict[tuple[str, str], float]:
    sums: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (str(row["slot_type"]), str(row["tier"]))
        sums[key] = sums.get(key, 0.0) + float(row["probability"])
    return sums


def _read_calibrated(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Pure-function behavior
# ---------------------------------------------------------------------------


def test_apply_temperature_identity_at_one():
    probs = {"a": 0.6, "b": 0.3, "c": 0.1}
    assert apply_temperature(probs, 1.0) == probs


def test_apply_temperature_preserves_ranking_and_normalizes():
    probs = {"a": 0.6, "b": 0.3, "c": 0.1}
    for temperature in (0.3, 0.8, 1.7, 3.0):
        scaled = apply_temperature(probs, temperature)
        assert math.isclose(sum(scaled.values()), 1.0, rel_tol=1e-9)
        ranked = sorted(scaled, key=lambda f: scaled[f], reverse=True)
        assert ranked == ["a", "b", "c"]


def test_apply_temperature_flattens_and_sharpens():
    probs = {"a": 0.7, "b": 0.2, "c": 0.1}

    def entropy(dist):
        return -sum(p * math.log(p) for p in dist.values() if p > 0)

    base = entropy(probs)
    assert entropy(apply_temperature(probs, 2.5)) > base  # flatter -> more variety
    assert entropy(apply_temperature(probs, 0.4)) < base  # sharper -> repetition


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(RuntimeError):
        apply_temperature({"a": 1.0}, 0.0)


def test_assign_source_folds_is_deterministic_and_replayable():
    ids = list(range(1, 50))
    first = assign_source_folds(ids, folds=5, seed=20260612)
    second = assign_source_folds(ids, folds=5, seed=20260612)
    assert first == second
    assert set(first.values()) <= set(range(5))
    other_seed = assign_source_folds(ids, folds=5, seed=7)
    assert other_seed != first  # seed actually drives the split


def test_assign_source_folds_requires_two_folds():
    with pytest.raises(RuntimeError):
        assign_source_folds([1, 2, 3], folds=1, seed=1)


def test_heldout_evidence_counts_only_scores_labeled_heldout_sources():
    conn = _create_calibration_db()
    inputs = load_distribution_inputs(conn)
    links = load_strict_source_links(conn)
    all_ids = {subtitle_id for _filler_id, subtitle_id in links}
    heldout = set(sorted(all_ids)[:10])
    counts = heldout_evidence_counts(inputs, links, heldout)
    assert counts  # some held-out evidence exists
    # Every counted occurrence belongs to a held-out, validly-tiered source.
    assert all(tier in TIERS for (_slot, tier, _filler) in counts)
    # Restricting to no held-out sources yields no evidence.
    assert heldout_evidence_counts(inputs, links, set()) == {}


def test_calibration_verdict_flags_distinctiveness_collapse():
    from subtitle_generator.tier_slot_calibration import _CalibrationMetrics

    metrics = _CalibrationMetrics(
        temperatures={},
        heldout_nll_baseline=10.0,
        heldout_nll_calibrated=9.0,
        ece_baseline=0.2,
        ece_calibrated=0.1,
        top1_hit_rate=1.0,
        mean_effective_n_baseline=2.0,
        mean_effective_n_calibrated=2.4,
        distinctiveness_baseline=0.40,
        distinctiveness_calibrated=0.30,  # 25% drop -> over the 15% guardrail
        slices={},
        per_tier_effective_n={},
    )
    verdict = calibration_verdict(metrics)
    assert verdict["heldout_likelihood_improved_or_preserved"] is True
    assert verdict["tiers_kept_distinct"] is False
    assert verdict["distinctiveness_drop_fraction"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# include_subtitle_ids held-out split (builder hook)
# ---------------------------------------------------------------------------


def test_include_subtitle_ids_default_matches_full_build():
    conn = _create_calibration_db()
    inputs = load_distribution_inputs(conn)
    full = build_anchored_rows(conn, inputs, include_subtitle_ids=None)
    explicit_all = build_anchored_rows(
        conn,
        inputs,
        include_subtitle_ids={s for _f, s in load_strict_source_links(conn)},
    )
    full_probs = {
        (r["slot_type"], r["tier"], r["filler"]): float(r["probability"]) for r in full
    }
    all_probs = {
        (r["slot_type"], r["tier"], r["filler"]): float(r["probability"])
        for r in explicit_all
    }
    assert full_probs == all_probs


def test_include_subtitle_ids_subset_changes_evidence():
    conn = _create_calibration_db()
    inputs = load_distribution_inputs(conn)
    links = load_strict_source_links(conn)
    all_ids = sorted({s for _f, s in links})
    subset = set(all_ids[: len(all_ids) // 2])
    full = build_anchored_rows(conn, inputs, include_subtitle_ids=None)
    partial = build_anchored_rows(conn, inputs, include_subtitle_ids=subset)
    full_src = sum(int(r["source_count"]) for r in full)
    partial_src = sum(int(r["source_count"]) for r in partial)
    assert partial_src < full_src  # fewer sources -> less anchored evidence


# ---------------------------------------------------------------------------
# End-to-end calibration build
# ---------------------------------------------------------------------------


def test_build_calibration_writes_sidecar_and_preserves_likelihood(tmp_path: Path):
    conn = _create_calibration_db()
    config = CalibrationConfig("per_tier_temperature", "per_tier", folds=4)
    result = build_tier_slot_calibration(conn, tmp_path, config=config)

    assert result.distribution_path.name.endswith(".calibrated.csv")
    assert result.distribution_path.exists()
    assert result.metadata_path.exists()
    assert result.report_path.exists()

    rows = _read_calibrated(result.distribution_path)
    parsed = [dict(slot_type=r["slot_type"], tier=r["tier"], filler=r["filler"],
                   probability=r["probability"]) for r in rows]
    for total in _group_sums(parsed).values():
        assert total == pytest.approx(1.0, abs=1e-6)

    # Every row carries a positive calibration temperature.
    assert all(float(r["calibration_temperature"]) > 0 for r in rows)

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    # Exit gate: held-out likelihood improves or is preserved.
    assert (
        metadata["heldout_nll"]["calibrated"]
        <= metadata["heldout_nll"]["baseline"] + 1e-9
    )
    assert metadata["ranking"]["ranking_preserved"] is True


def test_calibration_none_granularity_equals_served_shape(tmp_path: Path):
    conn = _create_calibration_db()
    inputs = load_distribution_inputs(conn)
    served = build_anchored_rows(conn, inputs, include_subtitle_ids=None)
    served_probs = {
        (r["slot_type"], r["tier"], r["filler"]): float(r["probability"])
        for r in served
    }

    config = CalibrationConfig("baseline", "none", folds=4)
    result = build_tier_slot_calibration(conn, tmp_path, config=config)
    rows = _read_calibrated(result.distribution_path)
    cal_probs = {
        (r["slot_type"], r["tier"], r["filler"]): float(r["probability"]) for r in rows
    }
    for key, value in served_probs.items():
        assert cal_probs[key] == pytest.approx(value, abs=1e-9)
    assert all(float(r["calibration_temperature"]) == 1.0 for r in rows)


def test_calibration_metadata_is_replayable(tmp_path: Path):
    config = CalibrationConfig("per_tier_temperature", "per_tier", folds=4)
    first = build_tier_slot_calibration(
        _create_calibration_db(), tmp_path / "a", config=config
    )
    second = build_tier_slot_calibration(
        _create_calibration_db(), tmp_path / "b", config=config
    )
    meta_a = json.loads(first.metadata_path.read_text(encoding="utf-8"))
    meta_b = json.loads(second.metadata_path.read_text(encoding="utf-8"))
    assert meta_a["temperatures"] == meta_b["temperatures"]
    assert meta_a["fold_assignment_digest"] == meta_b["fold_assignment_digest"]
    assert first.temperatures == second.temperatures


def test_calibration_does_not_touch_served_artifact(tmp_path: Path):
    conn = _create_calibration_db()
    served_dir = tmp_path / "served"
    served = build_tier_slot_distribution(conn, served_dir, alpha=0.5)
    served_bytes = served.distribution_path.read_bytes()

    cal_dir = tmp_path / "cal"
    build_tier_slot_calibration(
        _create_calibration_db(),
        cal_dir,
        config=CalibrationConfig("per_tier_temperature", "per_tier", folds=4),
    )
    # The served file is unchanged and the calibrated sidecar is a separate file.
    assert served.distribution_path.read_bytes() == served_bytes
    assert not (cal_dir / served.distribution_path.name).exists() or (
        cal_dir / served.distribution_path.name
    ) != served.distribution_path


# ---------------------------------------------------------------------------
# Ablation + AutoResearcher sweep
# ---------------------------------------------------------------------------


def test_run_calibration_ablation_writes_metrics_for_each_granularity(tmp_path: Path):
    conn = _create_calibration_db()
    result = run_calibration_ablation(conn, tmp_path, folds=4)
    assert result.experiment_count == 4
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    with result.metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    experiments = {row["experiment"] for row in rows}
    assert experiments == {
        "baseline_T1",
        "global_temperature",
        "per_tier_temperature",
        "per_tier_slot_temperature",
    }
    # The "none" baseline must report a zero improvement and T=1 bounds.
    baseline = next(r for r in rows if r["experiment"] == "baseline_T1")
    assert float(baseline["nll_improvement"]) == pytest.approx(0.0, abs=1e-9)
    # Every swept granularity improves or preserves held-out likelihood.
    assert all(float(r["nll_improvement"]) >= -1e-9 for r in rows)


def test_run_calibration_autoresearcher_emits_proposals(tmp_path: Path):
    conn = _create_calibration_db()
    result = run_calibration_autoresearcher(conn, tmp_path, folds=4)
    assert result.report_path.exists()
    assert result.proposals_path.exists()
    with result.proposals_path.open(encoding="utf-8", newline="") as handle:
        proposals = list(csv.DictReader(handle))
    assert proposals  # at least one next-round proposal
