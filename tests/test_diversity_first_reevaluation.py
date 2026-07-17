import copy
import json
from pathlib import Path

import pytest

from subtitle_generator.diversity_first_reevaluation import (
    ARTIFACT_VARIANTS,
    build_diversity_first_decision,
    evaluate_diversity_first,
    run_diversity_first_reevaluation,
)
from subtitle_generator.step08_validation import stable_digest


REQUIRED_GATES = (
    "no_pop_collapse",
    "mainstream_distinct",
    "niche_tail_retained",
    "quality_non_inferiority",
    "tone_separation_non_regression",
    "first_draw_requested_tier_compliance",
    "intentional_shift",
    "public_contract_requested_tier_continuity",
)


def _frozen_decision() -> dict[str, object]:
    gates = {
        variant: [
            {
                "name": name,
                "status": (
                    "fail"
                    if name == "public_contract_requested_tier_continuity"
                    else "pass"
                ),
                "evidence": {"variant": variant, "gate": name},
            }
            for name in REQUIRED_GATES
        ]
        for variant in ARTIFACT_VARIANTS
    }
    gates["calibrated"][2]["status"] = "fail"
    gates["smoothed"][5]["status"] = "fail"
    metrics = {
        variant: {
            "pop": {
                "sample_count": 30,
                "unique_subtitle_rate": 1.0,
                "top_subtitle_mass": 1 / 30,
            }
        }
        for variant in ARTIFACT_VARIANTS
    }
    direct_first_draw = {
        variant: {
            scenario: {
                "sample_count": 30,
                "unique_subtitle_rate": 1.0,
                "top_subtitle_mass": 1 / 30,
            }
            for scenario in ("pop", "mainstream", "niche", "default")
        }
        for variant in ARTIFACT_VARIANTS
    }
    payload: dict[str, object] = {
        "decision": "defer",
        "gates": gates,
        "metrics": metrics,
        "public_contract_comparison": {
            "direct_first_draw": direct_first_draw,
            "legacy_public_retry": {
                scenario: {
                    "sample_count": 30,
                    "unique_subtitle_rate": (
                        1 / 30 if scenario == "pop" else 1.0
                    ),
                    "top_subtitle_mass": 1.0 if scenario == "pop" else 1 / 30,
                }
                for scenario in ("pop", "mainstream", "niche", "default")
            }
        },
    }
    payload["decision_digest"] = stable_digest(payload)
    return payload


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    source = root / "src/subtitle_generator"
    source.mkdir(parents=True)
    (source / "diversity_first_reevaluation.py").write_text(
        "policy source\n",
        encoding="utf-8",
    )
    return root


def test_diversity_first_promotes_anchored_and_rejects_legacy_repetition(tmp_path):
    frozen = _frozen_decision()
    frozen_path = tmp_path / "decision.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    decision = build_diversity_first_decision(
        frozen,
        frozen_decision_path=frozen_path,
        repo_root=_repo(tmp_path),
    )

    assert decision["decision"] == "promote"
    assert decision["recommended_variant"] == "anchored_base"
    assert decision["legacy_public_retry_gate"]["status"] == "fail"
    continuity = next(
        gate
        for gate in decision["gates"]["anchored_base"]
        if gate["name"] == "public_contract_requested_tier_continuity"
    )
    assert continuity["status"] == "diagnostic"


def test_catastrophic_repetition_blocks_otherwise_passing_candidate():
    frozen = _frozen_decision()
    mainstream = frozen["public_contract_comparison"]["direct_first_draw"][
        "anchored_base"
    ]["mainstream"]
    mainstream["unique_subtitle_rate"] = 0.1
    mainstream["top_subtitle_mass"] = 0.9
    digest_input = dict(frozen)
    digest_input.pop("decision_digest")
    frozen["decision_digest"] = stable_digest(digest_input)

    gates, _ = evaluate_diversity_first(frozen)

    assert gates["anchored_base"][0]["status"] == "fail"
    assert gates["anchored_base"][0]["evidence"]["mainstream"]["status"] == "fail"


def test_tampered_frozen_decision_is_rejected(tmp_path):
    frozen = _frozen_decision()
    frozen["decision"] = "promote"
    frozen_path = tmp_path / "decision.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    with pytest.raises(ValueError, match="digest"):
        build_diversity_first_decision(
            frozen,
            frozen_decision_path=frozen_path,
            repo_root=_repo(tmp_path),
        )


def test_reevaluation_does_not_modify_frozen_record(tmp_path):
    root = _repo(tmp_path)
    frozen = _frozen_decision()
    frozen_path = tmp_path / "decision.json"
    original = json.dumps(frozen, indent=2)
    frozen_path.write_text(original, encoding="utf-8")

    result = run_diversity_first_reevaluation(
        frozen_decision_path=frozen_path,
        output_dir=tmp_path / "output",
        repo_root=root,
    )

    assert frozen_path.read_text(encoding="utf-8") == original
    decision = json.loads(result.decision_path.read_text(encoding="utf-8"))
    assert decision["source_binding"]["frozen_step08_decision_sha256"]
    assert decision["decision_digest"]


def test_default_paths_are_repo_root_relative(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    frozen_path = root / "feedback/step08-validation/decision.json"
    frozen_path.parent.mkdir(parents=True)
    frozen_path.write_text(json.dumps(_frozen_decision()), encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    result = run_diversity_first_reevaluation(repo_root=root)

    assert result.decision_path == root / "feedback/step08-diversity-first/decision.json"


def test_binding_path_must_contain_evaluated_payload(tmp_path):
    root = _repo(tmp_path)
    frozen = _frozen_decision()
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({"not": "the decision"}), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        build_diversity_first_decision(
            frozen,
            frozen_decision_path=unrelated,
            repo_root=root,
        )


def test_nonpassing_first_draw_gates_are_not_ignored(tmp_path):
    frozen = _frozen_decision()
    frozen["gates"]["anchored_base"][4]["status"] = "fail"
    digest_input = copy.deepcopy(frozen)
    digest_input.pop("decision_digest")
    frozen["decision_digest"] = stable_digest(digest_input)
    frozen_path = tmp_path / "decision.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    decision = build_diversity_first_decision(
        frozen,
        frozen_decision_path=frozen_path,
        repo_root=_repo(tmp_path),
    )

    assert decision["decision"] == "defer"
    assert decision["recommended_variant"] is None
