"""Tests for replayable Step 8 artifact-runtime validation."""

import csv
import math
import sqlite3
from types import SimpleNamespace
from pathlib import Path

import pytest

from subtitle_generator import generate as generate_module
from subtitle_generator.generate import (
    GeneratedSubtitle,
    TargetTierDraw,
    generate_subtitle_first_draw_for_tier,
)
from subtitle_generator.step08_validation import (
    DISTRIBUTION_COLUMNS,
    GATE_POLICY,
    _recommend,
    _evidence_ceiling,
    _legacy_distributions,
    _public_contract_comparison,
    _review_contexts,
    accepted_decision_paths,
    compute_code_binding,
    compute_replay_binding_digest,
    evaluate_gates,
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
        "first_draw_compatible_requested_tier_compliance": compliance,
        "first_draw_legacy_requested_tier_agreement": compliance,
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


def _public_contract(
    *,
    pop: float = 1.0,
    mainstream: float = 0.9,
    niche: float = 0.6,
) -> dict[str, object]:
    return {
        "legacy_public_retry": {
            "pop": {"first_draw_compatible_requested_tier_compliance": pop},
            "mainstream": {"first_draw_compatible_requested_tier_compliance": mainstream},
            "niche": {"first_draw_compatible_requested_tier_compliance": niche},
        }
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


def test_first_draw_uses_exactly_one_generator_call(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE slot_filler_model_scores (slot_filler_id INTEGER)")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(generate_module, "_load_generation_candidates", lambda _conn: object())

    def fake_generate(_conn, _candidates, **kwargs):
        calls.append(kwargs)
        return GeneratedSubtitle("A, B, and the C of D", "A", "B", "C", "D")

    monkeypatch.setattr(generate_module, "_generate_subtitle_from_candidates", fake_generate)

    draw = generate_subtitle_first_draw_for_tier(
        conn,
        allowed_tiers={"pop"},
        seed=17,
    )

    assert draw == TargetTierDraw(
        subtitle=GeneratedSubtitle("A, B, and the C of D", "A", "B", "C", "D"),
        target_tier="pop",
    )
    assert len(calls) == 1
    assert calls[0]["seed"] == 17
    assert calls[0]["model_tier"] == "pop"


def test_first_draw_uses_same_seed_and_target_semantics_for_legacy_and_shadow(
    monkeypatch,
):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE slot_filler_model_scores (slot_filler_id INTEGER)")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(generate_module, "_load_generation_candidates", lambda _conn: object())
    monkeypatch.setattr(
        generate_module,
        "_choose_generation_tier",
        lambda _conn, *, allowed_tiers, seed: "mainstream",
    )
    monkeypatch.setattr(
        generate_module,
        "prepare_generation_runtime",
        lambda _conn, runtime: runtime,
    )

    def fake_generate(_conn, _candidates, **kwargs):
        calls.append(kwargs)
        return GeneratedSubtitle("A, B, and the C of D", "A", "B", "C", "D")

    monkeypatch.setattr(generate_module, "_generate_subtitle_from_candidates", fake_generate)

    legacy_runtime = type("Runtime", (), {"mode": generate_module.RuntimeSelectionMode.LEGACY})()
    shadow_runtime = type("Runtime", (), {"mode": generate_module.RuntimeSelectionMode.SHADOW})()

    legacy = generate_subtitle_first_draw_for_tier(
        conn,
        allowed_tiers=None,
        seed=23,
        runtime=legacy_runtime,
    )
    shadow = generate_subtitle_first_draw_for_tier(
        conn,
        allowed_tiers=None,
        seed=23,
        runtime=shadow_runtime,
    )

    assert legacy.target_tier == "mainstream"
    assert shadow.target_tier == "mainstream"
    assert [call["seed"] for call in calls] == [23, 23]
    assert [call["model_tier"] for call in calls] == ["mainstream", "mainstream"]


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

    gates = evaluate_gates(_metrics(), quality, _public_contract(pop=0.9, mainstream=0.9, niche=0.9))

    assert all(
        gate["status"] == "pass"
        for variant_gates in gates.values()
        for gate in variant_gates
    )


def test_gate_evaluation_blocks_quality_without_fabricating_scores():
    gates = evaluate_gates(
        _metrics(),
        {"status": "blocked", "blocker": "Copilot unavailable"},
        _public_contract(),
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


def test_public_contract_gate_blocks_variant_that_passes_first_draw_gates():
    metrics = _metrics(compliance=0.55)
    quality = {
        variant: {
            "overall": 7.0,
            "coherence": 7.0,
            "evocativeness": 7.0,
            "surprise": 7.0,
        }
        for variant in ("legacy", "anchored_base", "calibrated", "smoothed")
    }

    gates = evaluate_gates(metrics, quality, _public_contract(pop=1.0, mainstream=0.9, niche=0.6))

    first_draw_gate = next(
        gate for gate in gates["anchored_base"] if gate["name"] == "first_draw_requested_tier_compliance"
    )
    public_gate = next(
        gate
        for gate in gates["anchored_base"]
        if gate["name"] == "public_contract_requested_tier_continuity"
    )
    assert first_draw_gate["status"] == "pass"
    assert public_gate["status"] == "fail"

    recommendation, recommended_variant, best_experimental = _recommend(gates, metrics)
    assert recommendation == "defer"
    assert recommended_variant is None
    assert best_experimental == "anchored_base"


def test_public_contract_metrics_are_grouped_by_scenario(tmp_path):
    artifact = load_artifact(_artifact(tmp_path / "artifact.csv"))
    samples = []
    fillers = [
        {"slot_type": "list_item", "filler": "Pop item"},
        {"slot_type": "list_item", "filler": "Pop item"},
        {"slot_type": "action_noun", "filler": "Making"},
        {"slot_type": "of_object", "filler": "Daily life"},
    ]
    for scenario in ("pop", "mainstream", "niche", "default"):
        samples.append({
            "scenario": scenario,
            "text": f"{scenario} sample",
            "fillers": fillers,
            "requested_tier": None if scenario == "default" else scenario,
            "target_tier": "pop" if scenario == "default" else scenario,
            "compatible_tier": "pop",
            "legacy_tier": "pop",
        })
    primary = {
        variant: {
            scenario: {"sample_count": 1}
            for scenario in ("pop", "mainstream", "niche", "default")
        }
        for variant in ("legacy", "anchored_base", "calibrated", "smoothed")
    }

    comparison = _public_contract_comparison(samples, primary, artifact)

    assert set(comparison["legacy_public_retry"]) == {
        "pop",
        "mainstream",
        "niche",
        "default",
    }
    assert comparison["legacy_public_retry"]["pop"]["sample_count"] == 1


def test_legacy_distribution_requires_model_score_columns(monkeypatch):
    candidates = SimpleNamespace(
        list_rows=[("A", 1, None)],
        action_rows=[("B", 1, None)],
        obj_rows=[("C", 1, None)],
    )
    monkeypatch.setattr(
        "subtitle_generator.step08_validation._load_generation_candidates",
        lambda _conn: candidates,
    )

    with pytest.raises(RuntimeError, match="requires slot_filler_model_scores"):
        _legacy_distributions(sqlite3.connect(":memory:"))


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


def test_source_digest_change_changes_replay_binding(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "a.py").write_text("print('one')\n", encoding="utf-8")
    (repo_root / "b.py").write_text("print('two')\n", encoding="utf-8")
    source_files = (Path("a.py"), Path("b.py"))

    before = compute_code_binding(
        repo_root=repo_root,
        source_files=source_files,
        base_revision="abc123",
    )
    digest_before = compute_replay_binding_digest(
        config={"samples_per_scenario": 30},
        gate_policy={"x": 1},
        database_digest="db",
        artifact_digests={"a": "1"},
        accepted_decision_digests={"step05": "2", "step06": "3"},
        code_binding=before,
    )

    (repo_root / "a.py").write_text("print('changed')\n", encoding="utf-8")
    after = compute_code_binding(
        repo_root=repo_root,
        source_files=source_files,
        base_revision="abc123",
    )
    digest_after = compute_replay_binding_digest(
        config={"samples_per_scenario": 30},
        gate_policy={"x": 1},
        database_digest="db",
        artifact_digests={"a": "1"},
        accepted_decision_digests={"step05": "2", "step06": "3"},
        code_binding=after,
    )

    assert before["evaluation_source_digest"] != after["evaluation_source_digest"]
    assert digest_before != digest_after


def test_base_revision_is_provenance_not_binding(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "a.py").write_text("print('same')\n", encoding="utf-8")
    binding_a = compute_code_binding(
        repo_root=repo_root,
        source_files=(Path("a.py"),),
        base_revision="rev-a",
    )
    binding_b = compute_code_binding(
        repo_root=repo_root,
        source_files=(Path("a.py"),),
        base_revision="rev-b",
    )

    digest_a = compute_replay_binding_digest(
        config={"samples_per_scenario": 30},
        gate_policy={"x": 1},
        database_digest="db",
        artifact_digests={"a": "1"},
        accepted_decision_digests={"step05": "2", "step06": "3"},
        code_binding=binding_a,
    )
    digest_b = compute_replay_binding_digest(
        config={"samples_per_scenario": 30},
        gate_policy={"x": 1},
        database_digest="db",
        artifact_digests={"a": "1"},
        accepted_decision_digests={"step05": "2", "step06": "3"},
        code_binding=binding_b,
    )

    assert binding_a["evaluation_source_digest"] == binding_b["evaluation_source_digest"]
    assert digest_a == digest_b


def test_explicit_repo_root_resolution_ignores_cwd(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    paths = accepted_decision_paths(repo_root=repo_root)

    assert paths["step05"] == repo_root / "feedback" / "step05-smoothing" / "decision.json"
    assert paths["step06"] == repo_root / "feedback" / "step06-calibration" / "decision.json"
    assert all(str(path).startswith(str(repo_root)) for path in paths.values())


def test_evidence_ceiling_uses_corrected_support_fields(tmp_path):
    base = load_artifact(_artifact(tmp_path / "base.csv"))
    smoothed = load_artifact(_artifact(tmp_path / "smoothed.csv"))
    for group in base.groups.values():
        first = next(iter(group.values()))
        first["source_count"] = "0"
        first["soft_count"] = "0"
        first["anchored_soft_count"] = "0"
    metrics = _metrics(compliance=0.55)

    ceiling = _evidence_ceiling(base, smoothed, metrics)

    sample_group = next(group for group in ceiling["groups"] if group["slot_type"] == "list_item")
    assert {"artifact_vocabulary", "observed_vocabulary", "anchored_vocabulary"} <= sample_group.keys()
    assert sample_group["artifact_vocabulary"] >= sample_group["observed_vocabulary"]
    assert sample_group["observed_vocabulary"] >= sample_group["anchored_vocabulary"]


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
