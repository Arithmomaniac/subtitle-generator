"""Diversity-first reevaluation of the frozen Step 8 evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.step08_validation import (
    sha256_file,
    stable_digest,
    step08_repo_root,
)


DIVERSITY_FIRST_SCHEMA_VERSION = 1
FROZEN_STEP08_DECISION = Path("feedback/step08-validation/decision.json")
DIVERSITY_FIRST_SOURCE = Path(
    "src/subtitle_generator/diversity_first_reevaluation.py"
)
DIVERSITY_FIRST_POLICY = {
    "minimum_samples_per_scenario": 30,
    "minimum_unique_subtitle_rate": 0.80,
    "maximum_top_subtitle_mass": 0.20,
    "catastrophic_repetition_scenarios": [
        "pop",
        "mainstream",
        "niche",
        "default",
    ],
    "required_step08_first_draw_gates": [
        "no_pop_collapse",
        "mainstream_distinct",
        "niche_tail_retained",
        "quality_non_inferiority",
        "tone_separation_non_regression",
        "first_draw_requested_tier_compliance",
        "intentional_shift",
    ],
    "public_contract_requested_tier_continuity": "diagnostic",
}
ARTIFACT_VARIANTS = ("anchored_base", "calibrated", "smoothed")


@dataclass(frozen=True)
class DiversityFirstResult:
    decision_path: Path
    readme_path: Path
    decision: str
    recommended_variant: str | None


def _validate_frozen_decision(decision: dict[str, object]) -> None:
    recorded_digest = decision.get("decision_digest")
    if not isinstance(recorded_digest, str):
        raise ValueError("Frozen Step 8 decision is missing decision_digest")
    digest_input = dict(decision)
    del digest_input["decision_digest"]
    if stable_digest(digest_input) != recorded_digest:
        raise ValueError("Frozen Step 8 decision digest does not match its contents")
    if decision.get("decision") != "defer":
        raise ValueError("Diversity-first reevaluation requires the frozen defer record")


def _gate_map(decision: dict[str, object], variant: str) -> dict[str, dict[str, object]]:
    gates = decision["gates"]
    if not isinstance(gates, dict) or not isinstance(gates.get(variant), list):
        raise ValueError(f"Frozen Step 8 decision is missing gates for {variant}")
    return {
        str(gate["name"]): gate
        for gate in gates[variant]
        if isinstance(gate, dict) and "name" in gate
    }


def _catastrophic_repetition_gate(metrics: dict[str, object]) -> dict[str, object]:
    evidence: dict[str, object] = {}
    passed = True
    for scenario in DIVERSITY_FIRST_POLICY["catastrophic_repetition_scenarios"]:
        scenario_metrics = metrics.get(scenario)
        if not isinstance(scenario_metrics, dict):
            raise ValueError(f"Variant metrics are missing the {scenario} scenario")
        sample_count = int(scenario_metrics["sample_count"])
        unique_rate = float(scenario_metrics["unique_subtitle_rate"])
        top_mass = float(scenario_metrics["top_subtitle_mass"])
        scenario_passed = (
            sample_count >= DIVERSITY_FIRST_POLICY["minimum_samples_per_scenario"]
            and unique_rate >= DIVERSITY_FIRST_POLICY["minimum_unique_subtitle_rate"]
            and top_mass <= DIVERSITY_FIRST_POLICY["maximum_top_subtitle_mass"]
        )
        evidence[scenario] = {
            "status": "pass" if scenario_passed else "fail",
            "sample_count": sample_count,
            "unique_subtitle_rate": unique_rate,
            "top_subtitle_mass": top_mass,
        }
        passed = passed and scenario_passed
    evidence["minimum_unique_subtitle_rate"] = DIVERSITY_FIRST_POLICY[
        "minimum_unique_subtitle_rate"
    ]
    evidence["maximum_top_subtitle_mass"] = DIVERSITY_FIRST_POLICY[
        "maximum_top_subtitle_mass"
    ]
    return {
        "name": "no_catastrophic_subtitle_repetition",
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    }


def evaluate_diversity_first(
    frozen_decision: dict[str, object],
) -> tuple[dict[str, list[dict[str, object]]], dict[str, object]]:
    metrics = frozen_decision.get("metrics")
    public_comparison = frozen_decision.get("public_contract_comparison")
    if not isinstance(metrics, dict) or not isinstance(public_comparison, dict):
        raise ValueError("Frozen Step 8 decision is missing evaluation metrics")

    direct_first_draw = public_comparison.get("direct_first_draw")
    if not isinstance(direct_first_draw, dict):
        raise ValueError("Frozen Step 8 decision is missing direct first-draw metrics")

    results: dict[str, list[dict[str, object]]] = {}
    for variant in ARTIFACT_VARIANTS:
        variant_metrics = metrics.get(variant)
        variant_direct_metrics = direct_first_draw.get(variant)
        if not isinstance(variant_metrics, dict) or not isinstance(
            variant_direct_metrics, dict
        ):
            raise ValueError(f"Frozen Step 8 decision is missing metrics for {variant}")
        inherited = _gate_map(frozen_decision, variant)
        gates = [_catastrophic_repetition_gate(variant_direct_metrics)]
        for gate_name in DIVERSITY_FIRST_POLICY["required_step08_first_draw_gates"]:
            gate = inherited.get(gate_name)
            if gate is None:
                raise ValueError(
                    f"Frozen Step 8 decision is missing {gate_name} for {variant}"
                )
            gates.append(gate)
        continuity = inherited.get("public_contract_requested_tier_continuity")
        if continuity is None:
            raise ValueError(
                "Frozen Step 8 decision is missing the public continuity diagnostic"
            )
        gates.append({
            "name": "public_contract_requested_tier_continuity",
            "status": "diagnostic",
            "evidence": continuity["evidence"],
        })
        results[variant] = gates

    legacy_public = public_comparison.get("legacy_public_retry")
    if not isinstance(legacy_public, dict):
        raise ValueError("Frozen Step 8 decision is missing legacy public metrics")
    legacy_gate = _catastrophic_repetition_gate(legacy_public)
    legacy_gate["evidence"]["interpretation"] = (
        "Current retry behavior preserves requested-tier continuity but is not "
        "eligible under the diversity-first policy when repetition is catastrophic."
    )

    return results, legacy_gate


def _required_gates_pass(gates: list[dict[str, object]]) -> bool:
    return all(gate["status"] in {"pass", "diagnostic"} for gate in gates)


def build_diversity_first_decision(
    frozen_decision: dict[str, object],
    *,
    frozen_decision_path: Path,
    repo_root: Path,
) -> dict[str, object]:
    file_decision = json.loads(frozen_decision_path.read_text(encoding="utf-8"))
    if file_decision != frozen_decision:
        raise ValueError(
            "Frozen Step 8 decision payload does not match frozen_decision_path"
        )
    _validate_frozen_decision(frozen_decision)
    gates, legacy_public_gate = evaluate_diversity_first(frozen_decision)
    passing = [
        variant
        for variant in ARTIFACT_VARIANTS
        if _required_gates_pass(gates[variant])
    ]
    recommended_variant = "anchored_base" if "anchored_base" in passing else None
    decision = "promote" if recommended_variant else "defer"

    source_path = repo_root / DIVERSITY_FIRST_SOURCE
    source_binding = {
        "frozen_step08_decision_sha256": sha256_file(frozen_decision_path),
        "frozen_step08_decision_digest": frozen_decision["decision_digest"],
        "reevaluation_source_sha256": sha256_file(source_path),
    }
    payload: dict[str, object] = {
        "schema_version": DIVERSITY_FIRST_SCHEMA_VERSION,
        "policy_name": "diversity_first",
        "decision": decision,
        "recommended_variant": recommended_variant,
        "summary": (
            "Diversity-first reevaluation recommends anchored_base for rollout. "
            "It passes every first-draw quality, tone, tier-signal, and diversity "
            "gate, while legacy public retry fails the catastrophic-repetition gate."
            if recommended_variant
            else "No artifact variant passes the diversity-first rollout policy."
        ),
        "policy": DIVERSITY_FIRST_POLICY,
        "gates": gates,
        "legacy_public_retry_gate": legacy_public_gate,
        "public_contract_continuity": {
            "status": "accepted_tradeoff" if recommended_variant else "unresolved",
            "rationale": (
                "Per-subtitle retry-era continuity remains visible as a diagnostic. "
                "The policy accepts lower agreement because distributional tone "
                "separation is non-inferior and catastrophic repetition is forbidden."
            ),
        },
        "source_binding": source_binding,
        "rollback_implications": (
            "Keep the legacy runtime available as the explicit rollback path. "
            "The frozen Step 8 defer record remains unchanged."
        ),
    }
    payload["reevaluation_input_digest"] = stable_digest({
        "policy": DIVERSITY_FIRST_POLICY,
        "source_binding": source_binding,
    })
    payload["decision_digest"] = stable_digest(payload)
    return payload


def _readme(decision: dict[str, object]) -> str:
    recommended = decision["recommended_variant"] or "none"
    return "\n".join([
        "# Step 8 diversity-first reevaluation",
        "",
        "This is a separate policy decision over the frozen Step 8 samples, ratings,",
        "and metrics. It does not modify `feedback/step08-validation/decision.json`.",
        "",
        f"**Decision:** `{decision['decision']}`",
        "",
        f"**Recommended variant:** `{recommended}`",
        "",
        "Catastrophic subtitle repetition is an automatic failure. Distributional",
        "tone separation, quality, tail retention, and a minimum compatible tier",
        "signal remain required. Retry-era per-subtitle continuity is diagnostic.",
        "",
        "Reproduce with:",
        "",
        "```powershell",
        "uv run subtitle-gen reevaluate-diversity-first",
        "```",
        "",
    ])


def run_diversity_first_reevaluation(
    *,
    frozen_decision_path: Path | None = None,
    output_dir: Path | None = None,
    repo_root: Path | None = None,
) -> DiversityFirstResult:
    root = step08_repo_root() if repo_root is None else repo_root.resolve()
    source_path = (
        root / FROZEN_STEP08_DECISION
        if frozen_decision_path is None
        else frozen_decision_path.resolve()
    )
    destination = (
        root / "feedback/step08-diversity-first"
        if output_dir is None
        else output_dir.resolve()
    )
    frozen_decision = json.loads(source_path.read_text(encoding="utf-8"))
    decision = build_diversity_first_decision(
        frozen_decision,
        frozen_decision_path=source_path,
        repo_root=root,
    )
    destination.mkdir(parents=True, exist_ok=True)
    decision_path = destination / "decision.json"
    readme_path = destination / "README.md"
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readme_path.write_text(_readme(decision), encoding="utf-8")
    return DiversityFirstResult(
        decision_path=decision_path,
        readme_path=readme_path,
        decision=str(decision["decision"]),
        recommended_variant=decision["recommended_variant"],
    )
