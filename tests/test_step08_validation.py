"""Tests for replayable Step 8 artifact-runtime validation."""

import csv
import math
import sqlite3
from pathlib import Path

import pytest

from subtitle_generator.step08_validation import (
    DISTRIBUTION_COLUMNS,
    GATE_POLICY,
    evaluate_gates,
    _review_contexts,
    load_artifact,
    run_step08_validation,
    sample_seed,
    score_compatible_tier,
    sha256_file,
    stable_digest,
)


def _artifact(path: Path) -> Path:
    rows = []
    for slot_type, fillers in {
        "list_item": ("Pop item", "Main item", "Niche item"),
        "action_noun": ("Making", "Reading", "Problematizing"),
        "of_object": ("Daily life", "Social history", "Late antiquity"),
    }.items():
        for tier_index, tier in enumerate(("pop", "mainstream", "niche")):
            for filler_index, filler in enumerate(fillers):
                probability = 0.8 if tier_index == filler_index else 0.1
                rows.append({
                    "slot_type": slot_type,
                    "tier": tier,
                    "filler": filler.casefold(),
                    "display_filler": filler,
                    "probability": probability,
                    "log_probability": math.log(probability),
                    "soft_count": 2.0,
                    "prior_count": 0.5,
                    "evidence_count": 1.5,
                    "source_count": 2,
                    "anchored_source_count": 2,
                    "inferred_source_count": 0,
                    "anchored_soft_count": 1.5,
                    "inferred_soft_count": 0.0,
                    "teacher_confidence_mean": 0.9,
                    "frequency": 3,
                    "popularity_score": 0.5,
                    "semantic_smoothing_mass": 0.0,
                    "calibration_temperature": 1.0,
                    "artifact_version": "test-v1",
                })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISTRIBUTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _metrics(
    *,
    effective_n: float = 20.0,
    top_mass: float = 0.08,
    tail: float = 0.20,
    separation: float = 0.20,
    distinctiveness: float = 0.15,
    compliance: float = 0.90,
    shift: float = 0.05,
):
    scenario = {
        "effective_filler_n": effective_n,
        "top_filler_mass": top_mass,
        "tail_exposure": tail,
        "compatible_requested_tier_compliance": compliance,
    }
    values = {
        "pop": dict(scenario),
        "mainstream": dict(scenario),
        "niche": dict(scenario),
        "default": dict(scenario),
        "tone_separation": separation,
        "mainstream_distinctiveness": distinctiveness,
        "mean_js_from_legacy": shift,
    }
    return {
        "legacy": values,
        "anchored_base": values,
        "calibrated": values,
        "smoothed": values,
    }


def test_compatible_scorer_uses_selected_fillers(tmp_path):
    artifact = load_artifact(_artifact(tmp_path / "artifact.csv"))

    result = score_compatible_tier(
        artifact,
        (
            ("list_item", "Niche item"),
            ("list_item", "Niche item"),
            ("action_noun", "Problematizing"),
            ("of_object", "Late antiquity"),
        ),
    )

    assert result.tier == "niche"
    assert result.probabilities["niche"] > result.probabilities["mainstream"]


def test_compatible_scorer_rejects_uncovered_fillers(tmp_path):
    artifact = load_artifact(_artifact(tmp_path / "artifact.csv"))

    with pytest.raises(RuntimeError, match="does not cover"):
        score_compatible_tier(artifact, (("list_item", "Unknown"),))


def test_variant_seeds_are_deterministic_and_disjoint():
    first = [
        sample_seed(41000, scenario_index=scenario, sample_index=index)
        for scenario in range(4)
        for index in range(30)
    ]
    second = [
        sample_seed(41000, scenario_index=scenario, sample_index=index)
        for scenario in range(4)
        for index in range(30)
    ]

    assert first == second
    assert len(first) == len(set(first))


@pytest.mark.parametrize(
    ("slot_type", "filler"),
    [
        ("action_noun", "Astonishing Rise"),
        ("of_modifier", "global"),
        ("of_head", "memory"),
        ("of_topic", "empire"),
        ("of_complement", "justice"),
    ],
)
def test_smoothing_review_contexts_cover_each_sparse_slot(slot_type, filler):
    contexts = _review_contexts(slot_type, filler)

    assert len(contexts) == 3
    assert all(filler.casefold() in context.casefold() for context in contexts)
    assert len(set(contexts)) == 3


def test_gate_evaluation_passes_complete_noninferior_evidence():
    quality = {
        variant: {
            "overall": 7.0,
            "coherence": 7.0,
            "evocativeness": 7.0,
            "surprise": 7.0,
        }
        for variant in ("legacy", "anchored_base", "calibrated", "smoothed")
    }

    gates = evaluate_gates(_metrics(), quality)

    assert all(
        gate["status"] == "pass"
        for variant_gates in gates.values()
        for gate in variant_gates
    )


def test_gate_evaluation_blocks_quality_without_fabricating_scores():
    gates = evaluate_gates(
        _metrics(),
        {"status": "blocked", "blocker": "Copilot unavailable"},
    )

    quality_gate = next(
        gate
        for gate in gates["calibrated"]
        if gate["name"] == "quality_non_inferiority"
    )
    assert quality_gate == {
        "name": "quality_non_inferiority",
        "status": "blocked",
        "evidence": {"blocker": "Copilot unavailable"},
    }


def test_digests_are_stable_and_sensitive(tmp_path):
    path = tmp_path / "payload.txt"
    path.write_text("one", encoding="utf-8")
    first = sha256_file(path)
    assert first == sha256_file(path)
    assert stable_digest({"b": 2, "a": 1}) == stable_digest({"a": 1, "b": 2})

    path.write_text("two", encoding="utf-8")
    assert sha256_file(path) != first


def test_artifact_loader_fails_closed_on_bad_contract(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("slot_type,tier\nlist_item,pop\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="missing columns"):
        load_artifact(path)


def test_validation_rejects_underpowered_replay(tmp_path):
    with pytest.raises(RuntimeError, match="at least"):
        run_step08_validation(
            sqlite3.connect(":memory:"),
            tmp_path / "missing.db",
            tmp_path / "out",
            tmp_path / "decision",
            samples_per_scenario=GATE_POLICY["minimum_samples_per_scenario"] - 1,
            rate_with_copilot=False,
        )
