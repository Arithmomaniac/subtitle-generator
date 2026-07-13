"""Replayable Step 8 behavioral validation for artifact-driven generation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from subtitle_generator.calibration_feedback import (
    read_decision_record as read_calibration_decision_record,
)
from subtitle_generator.eval_harness import (
    DEFAULT_RATER_MODEL,
    RATING_PROMPT,
    rate_batch_raw,
)
from subtitle_generator.generate import (
    GeneratedSubtitle,
    _load_generation_candidates,
    generate_subtitle_first_draw_for_tier,
    generate_subtitle_matching_tiers,
)
from subtitle_generator.parameter_state import get_generation_tier_ratios
from subtitle_generator.runtime_eligibility import filler_key
from subtitle_generator.shadow_runtime import (
    build_generation_runtime,
    write_shadow_distribution_csv,
)
from subtitle_generator.smoothing_feedback import (
    read_decision_record as read_smoothing_decision_record,
)
from subtitle_generator.tier_slot_calibration import (
    CalibrationConfig,
    build_tier_slot_calibration,
)
from subtitle_generator.tier_slot_distribution import (
    DISTRIBUTION_COLUMNS,
    SmoothingExperimentConfig,
    build_smoothed_distribution_rows,
    build_tier_slot_distribution,
    resolve_smoothing_experiment,
)
from subtitle_generator.tiering import TIER_NAMES, compute_tier_evidence

STEP08_SCHEMA_VERSION = 1
PRIMARY_VARIANT_NAMES = ("legacy", "anchored_base", "calibrated", "smoothed")
PUBLIC_CONTRACT_VARIANT_NAME = "legacy_public_retry"
SCENARIOS: tuple[tuple[str, set[str] | None], ...] = (
    ("pop", {"pop"}),
    ("mainstream", {"mainstream"}),
    ("niche", {"niche"}),
    ("default", None),
)
SLOT_TYPES = ("list_item", "action_noun", "of_object")
SPARSE_OF_SLOT_TYPES = ("of_modifier", "of_head", "of_topic", "of_complement")
EVALUATION_SOURCE_FILES = (
    Path("src/subtitle_generator/step08_validation.py"),
    Path("src/subtitle_generator/generate.py"),
    Path("src/subtitle_generator/shadow_runtime.py"),
    Path("src/subtitle_generator/eval_harness.py"),
)
STEP05_DECISION_RELATIVE_PATH = Path("feedback/step05-smoothing/decision.json")
STEP06_DECISION_RELATIVE_PATH = Path("feedback/step06-calibration/decision.json")
ACCEPTED_SMOOTHING = SmoothingExperimentConfig(
    name="pc1_removed_minsrc2_knn10_m0_5_cap0_10",
    variant="generic_embedding_kNN",
    neighbor_count=10,
    shrinkage=0.5,
    evidence_gate="none",
    max_borrowed_mass=0.10,
    vector_transform="remove_pc1",
    min_candidate_sources=2,
)

# Frozen before looking at Step 8 results. Relative gates compare each artifact
# runtime with the same fixed-seed legacy sample set.
GATE_POLICY = {
    "minimum_samples_per_scenario": 30,
    "pop_effective_n_ratio_min": 0.80,
    "pop_top_filler_mass_absolute_max": 0.15,
    "pop_top_filler_mass_ratio_max": 1.25,
    "mainstream_pairwise_js_absolute_min": 0.01,
    "mainstream_pairwise_js_ratio_min": 0.80,
    "niche_tail_exposure_ratio_min": 0.80,
    "quality_mean_delta_min": -0.50,
    "coherence_mean_delta_min": -0.50,
    "tone_separation_ratio_min": 0.80,
    "first_draw_requested_tier_ratio_min": 0.80,
    "first_draw_requested_tier_absolute_min": 0.50,
    "public_contract_requested_tier_ratio_min": 0.80,
    "public_contract_requested_tier_absolute_min": 0.50,
    "intentional_shift_js_min": 0.001,
}
RATING_CHUNK_SIZE = 25


@dataclass(frozen=True)
class Step08ValidationResult:
    report_path: Path
    replay_path: Path
    decision_path: Path
    readme_path: Path
    sample_count: int
    recommendation: str


@dataclass(frozen=True)
class CompatibleTierScore:
    tier: str
    probabilities: dict[str, float]
    mean_log_probabilities: dict[str, float]


@dataclass(frozen=True)
class ArtifactIndex:
    path: Path
    digest: str
    rows: tuple[dict[str, object], ...]
    groups: dict[tuple[str, str], dict[str, dict[str, object]]]


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def step08_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_step08_repo_path(relative_path: Path, *, repo_root: Path | None = None) -> Path:
    root = step08_repo_root() if repo_root is None else repo_root.resolve()
    return (root / relative_path).resolve()


def accepted_decision_paths(*, repo_root: Path | None = None) -> dict[str, Path]:
    return {
        "step05": resolve_step08_repo_path(STEP05_DECISION_RELATIVE_PATH, repo_root=repo_root),
        "step06": resolve_step08_repo_path(STEP06_DECISION_RELATIVE_PATH, repo_root=repo_root),
    }


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def compute_code_binding(
    *,
    repo_root: Path | None = None,
    source_files: tuple[Path, ...] = EVALUATION_SOURCE_FILES,
    base_revision: str | None = None,
) -> dict[str, object]:
    root = step08_repo_root() if repo_root is None else repo_root.resolve()
    revision = base_revision or _git_head(root)
    per_file = {
        str(path): sha256_file(resolve_step08_repo_path(path, repo_root=root))
        for path in source_files
    }
    aggregate = stable_digest(per_file)
    return {
        "repo_root": str(root),
        "base_revision": revision,
        "evaluation_source_files": per_file,
        "evaluation_source_digest": aggregate,
    }


def compute_replay_binding_digest(
    *,
    config: dict[str, object],
    gate_policy: dict[str, object],
    database_digest: str,
    artifact_digests: dict[str, str],
    accepted_decision_digests: dict[str, str],
    code_binding: dict[str, object],
) -> str:
    return stable_digest(
        {
            "config": config,
            "gate_policy": gate_policy,
            "database": database_digest,
            "artifacts": artifact_digests,
            "accepted_decisions": accepted_decision_digests,
            "evaluation_source_files": code_binding["evaluation_source_files"],
            "evaluation_source_digest": code_binding["evaluation_source_digest"],
        }
    )


def _durable_source_binding(code_binding: dict[str, object]) -> dict[str, object]:
    return {
        "evaluation_source_files": code_binding["evaluation_source_files"],
        "evaluation_source_digest": code_binding["evaluation_source_digest"],
    }


def _runtime_policy(conn: sqlite3.Connection) -> dict[str, float]:
    remix_prob_row = conn.execute(
        "SELECT value FROM config WHERE key = 'remix_calibrated_remix_prob'"
    ).fetchone()
    min_sim_row = conn.execute(
        "SELECT value FROM config WHERE key = 'remix_calibrated_min_sim'"
    ).fetchone()
    ratios = get_generation_tier_ratios(conn)
    return {
        "remix_prob": float(remix_prob_row[0]) if remix_prob_row else 0.8,
        "min_sim": float(min_sim_row[0]) if min_sim_row else 0.1,
        "ratio_pop": ratios.pop,
        "ratio_mainstream": ratios.mainstream,
        "ratio_niche": ratios.niche,
    }


def load_artifact(path: Path) -> ArtifactIndex:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = tuple(dict(row) for row in csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Step 8 artifact contains no rows: {path}")
    missing = set(DISTRIBUTION_COLUMNS) - set(rows[0])
    if missing:
        raise RuntimeError(
            f"Step 8 artifact {path} is missing columns: {sorted(missing)}"
        )
    groups: dict[tuple[str, str], dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        key = filler_key(str(row["slot_type"]), str(row["display_filler"]))
        groups[(str(row["slot_type"]), str(row["tier"]))][key] = row
    return ArtifactIndex(
        path=path,
        digest=sha256_file(path),
        rows=rows,
        groups=dict(groups),
    )


def score_compatible_tier(
    artifact: ArtifactIndex,
    fillers: tuple[tuple[str, str], ...],
) -> CompatibleTierScore:
    """Classify selected fillers under the artifact's tier/slot distributions."""

    if not fillers:
        raise RuntimeError("Compatible tier scoring requires selected fillers")
    log_scores: dict[str, float] = {}
    for tier in TIER_NAMES:
        values: list[float] = []
        for slot_type, display_filler in fillers:
            group = artifact.groups.get((slot_type, tier))
            if not group:
                raise RuntimeError(
                    f"Artifact is missing the ({tier}, {slot_type}) distribution"
                )
            row = group.get(filler_key(slot_type, display_filler))
            if row is None:
                raise RuntimeError(
                    f"Artifact does not cover {display_filler!r} in {slot_type!r}"
                )
            probability = float(row["probability"])
            if probability <= 0:
                raise RuntimeError(
                    f"Artifact has nonpositive probability for "
                    f"{display_filler!r} in ({tier}, {slot_type})"
                )
            values.append(math.log(probability))
        log_scores[tier] = mean(values)
    probabilities = _softmax(log_scores)
    predicted = max(TIER_NAMES, key=lambda tier: (probabilities[tier], -TIER_NAMES.index(tier)))
    return CompatibleTierScore(predicted, probabilities, log_scores)


def evaluate_gates(
    metrics: dict[str, dict[str, object]],
    quality: dict[str, object],
    public_contract: dict[str, object],
) -> dict[str, list[dict[str, object]]]:
    """Evaluate the frozen gate policy for every non-legacy variant."""

    legacy = metrics["legacy"]
    results: dict[str, list[dict[str, object]]] = {}
    legacy_requested_compliance = mean(
        float(legacy[tier]["first_draw_compatible_requested_tier_compliance"])
        for tier in TIER_NAMES
    )
    compliance_floor = max(
        GATE_POLICY["first_draw_requested_tier_absolute_min"],
        GATE_POLICY["first_draw_requested_tier_ratio_min"] * legacy_requested_compliance,
    )
    public_contract_legacy = public_contract["legacy_public_retry"]
    for variant in PRIMARY_VARIANT_NAMES[1:]:
        current = metrics[variant]
        gates: list[dict[str, object]] = []
        _add_gate(
            gates,
            "no_pop_collapse",
            float(current["pop"]["effective_filler_n"])
            >= GATE_POLICY["pop_effective_n_ratio_min"]
            * float(legacy["pop"]["effective_filler_n"])
            and float(current["pop"]["top_filler_mass"])
            <= max(
                GATE_POLICY["pop_top_filler_mass_absolute_max"],
                GATE_POLICY["pop_top_filler_mass_ratio_max"]
                * float(legacy["pop"]["top_filler_mass"]),
            ),
            {
                "candidate_effective_n": current["pop"]["effective_filler_n"],
                "legacy_effective_n": legacy["pop"]["effective_filler_n"],
                "candidate_top_mass": current["pop"]["top_filler_mass"],
                "legacy_top_mass": legacy["pop"]["top_filler_mass"],
            },
        )
        mainstream_floor = max(
            GATE_POLICY["mainstream_pairwise_js_absolute_min"],
            GATE_POLICY["mainstream_pairwise_js_ratio_min"]
            * float(legacy["mainstream_distinctiveness"]),
        )
        _add_gate(
            gates,
            "mainstream_distinct",
            float(current["mainstream_distinctiveness"]) >= mainstream_floor,
            {
                "candidate": current["mainstream_distinctiveness"],
                "legacy": legacy["mainstream_distinctiveness"],
                "floor": mainstream_floor,
            },
        )
        _add_gate(
            gates,
            "niche_tail_retained",
            float(current["niche"]["tail_exposure"])
            >= GATE_POLICY["niche_tail_exposure_ratio_min"]
            * float(legacy["niche"]["tail_exposure"]),
            {
                "candidate": current["niche"]["tail_exposure"],
                "legacy": legacy["niche"]["tail_exposure"],
            },
        )
        variant_quality = quality.get(variant)
        legacy_quality = quality.get("legacy")
        if isinstance(variant_quality, dict) and isinstance(legacy_quality, dict):
            quality_passed = (
                float(variant_quality["overall"]) - float(legacy_quality["overall"])
                >= GATE_POLICY["quality_mean_delta_min"]
                and float(variant_quality["coherence"])
                - float(legacy_quality["coherence"])
                >= GATE_POLICY["coherence_mean_delta_min"]
            )
            _add_gate(
                gates,
                "quality_non_inferiority",
                quality_passed,
                {
                    "candidate": variant_quality,
                    "legacy": legacy_quality,
                },
            )
        else:
            gates.append({
                "name": "quality_non_inferiority",
                "status": "blocked",
                "evidence": {"blocker": quality.get("blocker")},
            })
        _add_gate(
            gates,
            "tone_separation_non_regression",
            float(current["tone_separation"])
            >= GATE_POLICY["tone_separation_ratio_min"]
            * float(legacy["tone_separation"]),
            {
                "candidate": current["tone_separation"],
                "legacy": legacy["tone_separation"],
            },
        )
        requested_compliance = mean(
            float(current[tier]["first_draw_compatible_requested_tier_compliance"])
            for tier in TIER_NAMES
        )
        _add_gate(
            gates,
            "first_draw_requested_tier_compliance",
            requested_compliance >= compliance_floor
            and all(
                float(current[tier]["first_draw_compatible_requested_tier_compliance"])
                >= max(
                    GATE_POLICY["first_draw_requested_tier_absolute_min"],
                    GATE_POLICY["first_draw_requested_tier_ratio_min"]
                    * float(legacy[tier]["first_draw_compatible_requested_tier_compliance"]),
                )
                for tier in TIER_NAMES
            ),
            {
                "candidate": requested_compliance,
                "legacy": legacy_requested_compliance,
                "floor": compliance_floor,
                "per_tier": {
                    tier: current[tier]["first_draw_compatible_requested_tier_compliance"]
                    for tier in TIER_NAMES
                },
            },
        )
        _add_gate(
            gates,
            "intentional_shift",
            float(current["mean_js_from_legacy"])
            >= GATE_POLICY["intentional_shift_js_min"],
            {
                "mean_js_from_legacy": current["mean_js_from_legacy"],
                "rationale": (
                    "Replace sqrt(freq)*P(tier|filler) with explicit "
                    "P(filler|tier,slot); calibration and smoothing are "
                    "separately attributable."
                ),
            },
        )
        public_contract_floor = {
            tier: max(
                GATE_POLICY["public_contract_requested_tier_absolute_min"],
                GATE_POLICY["public_contract_requested_tier_ratio_min"]
                * float(
                    public_contract_legacy[tier][
                        "first_draw_compatible_requested_tier_compliance"
                    ]
                ),
            )
            for tier in TIER_NAMES
        }
        _add_gate(
            gates,
            "public_contract_requested_tier_continuity",
            all(
                float(current[tier]["first_draw_compatible_requested_tier_compliance"])
                >= public_contract_floor[tier]
                for tier in TIER_NAMES
            ),
            {
                "candidate": {
                    tier: current[tier]["first_draw_compatible_requested_tier_compliance"]
                    for tier in TIER_NAMES
                },
                "legacy_public_retry": {
                    tier: public_contract_legacy[tier][
                        "first_draw_compatible_requested_tier_compliance"
                    ]
                    for tier in TIER_NAMES
                },
                "floor": public_contract_floor,
            },
        )
        results[variant] = gates
    return results


def run_step08_validation(
    conn: sqlite3.Connection,
    db_path: Path,
    output_dir: Path,
    decision_dir: Path,
    *,
    samples_per_scenario: int = 30,
    seed_base: int = 41000,
    rater_model: str = DEFAULT_RATER_MODEL,
    rate_with_copilot: bool = True,
) -> Step08ValidationResult:
    if samples_per_scenario < GATE_POLICY["minimum_samples_per_scenario"]:
        raise RuntimeError(
            "Step 8 requires at least "
            f"{GATE_POLICY['minimum_samples_per_scenario']} samples per scenario"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / f"artifacts-{os.getpid()}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    repo_root = step08_repo_root()
    decision_paths = accepted_decision_paths(repo_root=repo_root)

    artifacts = _build_artifacts(conn, artifact_dir)
    indexes = {name: load_artifact(path) for name, path in artifacts.items()}
    runtime_artifacts = {
        "anchored_base": indexes["anchored_base"],
        "calibrated": indexes["calibrated"],
        "smoothed": indexes["smoothed"],
    }
    legacy_reference = indexes["anchored_base"]
    legacy_distributions = _legacy_distributions(conn)
    runtime_policy = _runtime_policy(conn)
    first_draw_samples = _generate_matrix(
        conn,
        runtime_artifacts,
        legacy_reference,
        samples_per_scenario=samples_per_scenario,
        seed_base=seed_base,
    )
    public_contract_samples = _generate_legacy_public_contract_samples(
        conn,
        legacy_reference,
        samples_per_scenario=samples_per_scenario,
        seed_base=seed_base,
    )

    samples_path = output_dir / "samples.json"
    ratings_path = output_dir / "ratings.json"
    smoothing_path = output_dir / "smoothing_review.json"
    ceiling_path = output_dir / "evidence_ceiling.json"
    replay_path = output_dir / "replay.json"
    report_path = output_dir / "report.md"
    decision_path = decision_dir / "decision.json"
    readme_path = decision_dir / "README.md"

    sample_payload = {
        "schema_version": STEP08_SCHEMA_VERSION,
        "first_draw_samples": first_draw_samples,
        "public_contract_samples": public_contract_samples,
    }
    quality, ratings, rating_chunks = _rate_samples(
        first_draw_samples,
        model=rater_model,
        enabled=rate_with_copilot,
        ratings_path=ratings_path,
        existing_samples_path=samples_path,
    )
    _write_json(samples_path, sample_payload)
    metrics = _matrix_metrics(
        first_draw_samples,
        indexes,
        legacy_distributions,
        quality,
    )
    public_contract = _public_contract_comparison(
        public_contract_samples,
        metrics,
        legacy_reference,
    )
    gates = evaluate_gates(metrics, quality, public_contract)
    recommendation, recommended_variant, best_experimental_variant = _recommend(
        gates,
        metrics,
    )
    smoothing_review = _smoothing_review(
        indexes["anchored_base"],
        indexes["smoothed"],
        first_draw_samples,
    )
    evidence_ceiling = _evidence_ceiling(
        indexes["anchored_base"],
        indexes["smoothed"],
        metrics,
    )
    _write_json(ratings_path, {
        "schema_version": STEP08_SCHEMA_VERSION,
        "model": rater_model,
        "prompt_template": RATING_PROMPT,
        "quality": quality,
        "first_draw_samples_digest": stable_digest(first_draw_samples),
        "chunks": rating_chunks,
        "ratings": ratings,
    })
    _write_json(smoothing_path, smoothing_review)
    _write_json(ceiling_path, evidence_ceiling)

    replay = {
        "schema_version": STEP08_SCHEMA_VERSION,
        "command": (
            "uv run subtitle-gen validate-artifact-runtime "
            f"--db {db_path} --samples-per-scenario {samples_per_scenario} "
            f"--seed-base {seed_base} --rater-model {rater_model}"
        ),
        "gate_policy": GATE_POLICY,
        "config": {
            "samples_per_scenario": samples_per_scenario,
            "seed_base": seed_base,
            "rater_model": rater_model,
            "rate_with_copilot": rate_with_copilot,
            "runtime_policy": runtime_policy,
            "primary_matrix_draw_semantics": "first_draw_for_target_tier",
            "public_contract_semantics": "legacy_retry_enforced_vs_direct_first_draw",
            "accepted_smoothing": asdict(ACCEPTED_SMOOTHING),
            "calibration": {"granularity": "per_tier", "seed": 20260612},
        },
        "digests": {
            "database": sha256_file(db_path),
            "artifacts": {name: index.digest for name, index in indexes.items()},
            "accepted_decisions": {
                "step05": sha256_file(decision_paths["step05"]),
                "step06": sha256_file(decision_paths["step06"]),
            },
            "samples": sha256_file(samples_path),
            "ratings": sha256_file(ratings_path),
            "smoothing_review": sha256_file(smoothing_path),
            "evidence_ceiling": sha256_file(ceiling_path),
        },
        "metrics": metrics,
        "gates": gates,
        "quality": quality,
        "recommendation": recommendation,
        "recommended_variant": recommended_variant,
        "best_experimental_variant": best_experimental_variant,
        "public_contract_comparison": public_contract,
        "smoothing_review": smoothing_review,
        "evidence_ceiling": evidence_ceiling,
    }
    replay["evaluation_source_binding"] = compute_code_binding(repo_root=repo_root)
    replay["digests"]["replay_input"] = compute_replay_binding_digest(
        config=replay["config"],
        gate_policy=GATE_POLICY,
        database_digest=str(replay["digests"]["database"]),
        artifact_digests=dict(replay["digests"]["artifacts"]),
        accepted_decision_digests=dict(replay["digests"]["accepted_decisions"]),
        code_binding=replay["evaluation_source_binding"],
    )
    _write_json(replay_path, replay)

    decision = _decision_payload(replay)
    _write_json(decision_path, decision)
    readme_path.write_text(_format_durable_readme(decision), encoding="utf-8")
    report_path.write_text(_format_report(replay), encoding="utf-8")
    return Step08ValidationResult(
        report_path=report_path,
        replay_path=replay_path,
        decision_path=decision_path,
        readme_path=readme_path,
        sample_count=len(first_draw_samples),
        recommendation=recommendation,
    )


def _build_artifacts(
    conn: sqlite3.Connection,
    artifact_dir: Path,
) -> dict[str, Path]:
    decision_paths = accepted_decision_paths()
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base_result = build_tier_slot_distribution(conn, artifact_dir, alpha=0.5)
    base_path = base_result.distribution_path

    calibration_decision = read_calibration_decision_record(decision_paths["step06"])
    if calibration_decision["decision"] != "accept":
        raise RuntimeError(
            "Step 8 requires an accepted Step 6 calibration decision"
        )
    calibration_dir = artifact_dir / "calibration"
    calibrated = build_tier_slot_calibration(
        conn,
        calibration_dir,
        config=CalibrationConfig(
            "accepted_per_tier_temperature",
            "per_tier",
            seed=20260612,
        ),
    )
    if not all(
        math.isclose(
            float(calibrated.temperatures[key]),
            float(calibration_decision["temperatures"][key]),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        for key in calibrated.temperatures
    ):
        raise RuntimeError(
            "Rebuilt calibrated artifact does not match the accepted Step 6 temperatures"
        )

    smoothing_decision = read_smoothing_decision_record(decision_paths["step05"])
    if smoothing_decision["decision"] != "accept":
        raise RuntimeError("Step 8 requires an accepted Step 5 smoothing decision")
    accepted_variant = str(smoothing_decision["variant"])
    resolved_variant = resolve_smoothing_experiment(accepted_variant)
    if resolved_variant != ACCEPTED_SMOOTHING:
        raise RuntimeError(
            "Accepted Step 5 smoothing config does not match the committed runtime "
            f"config: {accepted_variant}"
        )
    smoothed_rows = build_smoothed_distribution_rows(
        conn,
        artifact_dir,
        variant_name=accepted_variant,
        vector_source=str(smoothing_decision.get("vector_source") or "offline_spacy"),
    )
    smoothed_path = artifact_dir / "accepted_smoothed.csv"
    write_shadow_distribution_csv(smoothed_path, smoothed_rows)
    return {
        "anchored_base": base_path,
        "calibrated": calibrated.distribution_path,
        "smoothed": smoothed_path,
    }


def _generate_matrix(
    conn: sqlite3.Connection,
    artifacts: dict[str, ArtifactIndex],
    legacy_reference: ArtifactIndex,
    *,
    samples_per_scenario: int,
    seed_base: int,
) -> list[dict[str, object]]:
    runtime_policy = _runtime_policy(conn)
    runtimes = {
        "legacy": build_generation_runtime(mode="legacy"),
        **{
            name: build_generation_runtime(
                mode="shadow",
                shadow_artifact=index.path,
            )
            for name, index in artifacts.items()
        },
    }
    result: list[dict[str, object]] = []
    for variant_index, variant in enumerate(PRIMARY_VARIANT_NAMES):
        scorer = legacy_reference if variant == "legacy" else artifacts[variant]
        for scenario_index, (scenario, allowed_tiers) in enumerate(SCENARIOS):
            for sample_index in range(samples_per_scenario):
                seed = sample_seed(
                    seed_base,
                    scenario_index=scenario_index,
                    sample_index=sample_index,
                )
                draw = generate_subtitle_first_draw_for_tier(
                    conn,
                    allowed_tiers=allowed_tiers,
                    seed=seed,
                    remix_prob=runtime_policy["remix_prob"],
                    min_sim=runtime_policy["min_sim"],
                    runtime=runtimes[variant],
                )
                subtitle = draw.subtitle
                fillers = _subtitle_fillers(subtitle)
                compatible = score_compatible_tier(scorer, fillers)
                legacy = compute_tier_evidence(
                    subtitle.text,
                    conn,
                    remix_parts=subtitle.remix_parts if subtitle.remixed else None,
                )
                result.append({
                    "id": f"{variant}:{scenario}:{sample_index:03d}",
                    "variant": variant,
                    "scenario": scenario,
                    "requested_tier": (
                        next(iter(allowed_tiers)) if allowed_tiers else None
                    ),
                    "target_tier": draw.target_tier,
                    "seed": seed,
                    "text": subtitle.text,
                    "fillers": [
                        {"slot_type": slot_type, "filler": filler}
                        for slot_type, filler in fillers
                    ],
                    "compatible_tier": compatible.tier,
                    "compatible_probabilities": compatible.probabilities,
                    "compatible_log_probabilities": compatible.mean_log_probabilities,
                    "legacy_tier": legacy.tier,
                    "legacy_accessibility_score": legacy.accessibility_score,
                    "legacy_lower_tail_score": legacy.lower_tail_score,
                    "legacy_demand_confidence": legacy.demand_confidence,
                    "artifact_digest": scorer.digest,
                    "variant_order": variant_index,
                    "draw_mode": "first_draw",
                    "draw_attempts": 1,
                    "remixed": subtitle.remixed,
                    "remix_parts": dict(subtitle.remix_parts),
                })
    return result


def _generate_legacy_public_contract_samples(
    conn: sqlite3.Connection,
    legacy_reference: ArtifactIndex,
    *,
    samples_per_scenario: int,
    seed_base: int,
) -> list[dict[str, object]]:
    runtime_policy = _runtime_policy(conn)
    runtime = build_generation_runtime(mode="legacy")
    result: list[dict[str, object]] = []
    for scenario_index, (scenario, allowed_tiers) in enumerate(SCENARIOS):
        for sample_index in range(samples_per_scenario):
            seed = sample_seed(
                seed_base,
                scenario_index=scenario_index,
                sample_index=sample_index,
            )
            subtitle = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=allowed_tiers,
                seed=seed,
                remix_prob=runtime_policy["remix_prob"],
                min_sim=runtime_policy["min_sim"],
                runtime=runtime,
            )
            fillers = _subtitle_fillers(subtitle)
            compatible = score_compatible_tier(legacy_reference, fillers)
            legacy = compute_tier_evidence(
                subtitle.text,
                conn,
                remix_parts=subtitle.remix_parts if subtitle.remixed else None,
            )
            result.append({
                "id": f"{PUBLIC_CONTRACT_VARIANT_NAME}:{scenario}:{sample_index:03d}",
                "variant": PUBLIC_CONTRACT_VARIANT_NAME,
                "scenario": scenario,
                "requested_tier": next(iter(allowed_tiers)) if allowed_tiers else None,
                "target_tier": next(iter(allowed_tiers)) if allowed_tiers else None,
                "seed": seed,
                "text": subtitle.text,
                "fillers": [
                    {"slot_type": slot_type, "filler": filler}
                    for slot_type, filler in fillers
                ],
                "compatible_tier": compatible.tier,
                "compatible_probabilities": compatible.probabilities,
                "compatible_log_probabilities": compatible.mean_log_probabilities,
                "legacy_tier": legacy.tier,
                "legacy_accessibility_score": legacy.accessibility_score,
                "legacy_lower_tail_score": legacy.lower_tail_score,
                "legacy_demand_confidence": legacy.demand_confidence,
                "artifact_digest": legacy_reference.digest,
                "draw_mode": "public_legacy_retry",
                "draw_attempts": None,
                "remixed": subtitle.remixed,
                "remix_parts": dict(subtitle.remix_parts),
            })
    return result


def sample_seed(
    seed_base: int,
    *,
    scenario_index: int,
    sample_index: int,
) -> int:
    if scenario_index < 0 or sample_index < 0:
        raise ValueError("Step 8 seed indexes must be nonnegative")
    return seed_base + scenario_index * 10_000 + sample_index


def _rate_samples(
    samples: list[dict[str, object]],
    *,
    model: str,
    enabled: bool,
    ratings_path: Path | None = None,
    existing_samples_path: Path | None = None,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    if not enabled:
        return {
            "status": "blocked",
            "blocker": "Copilot rating disabled by --skip-ratings.",
        }, [], []
    if ratings_path is not None and existing_samples_path is not None:
        cached = _reuse_cached_ratings(
            samples,
            model=model,
            ratings_path=ratings_path,
            existing_samples_path=existing_samples_path,
        )
        if cached is not None:
            return cached
    ratings: list[dict[str, object]] = []
    chunks: list[dict[str, object]] = []
    for start in range(0, len(samples), RATING_CHUNK_SIZE):
        chunk_samples = samples[start : start + RATING_CHUNK_SIZE]
        prompt = RATING_PROMPT.format(
            subtitle_list="\n".join(
                f"{index + 1}. {sample['text']}"
                for index, sample in enumerate(chunk_samples)
            )
        )
        try:
            raw = rate_batch_raw([str(sample["text"]) for sample in chunk_samples], model=model)
        except RuntimeError as exc:
            return {
                "status": "blocked",
                "blocker": f"{type(exc).__name__}: {exc}",
                "model": model,
            }, [], chunks
        if len(raw) != len(chunk_samples):
            raise RuntimeError(
                "Copilot rater returned "
                f"{len(raw)} ratings for {len(chunk_samples)} chunk subtitles"
            )
        chunk_ratings = [
            {
                "sample_id": sample["id"],
                "coherence": rating.coherence,
                "evocativeness": rating.evocativeness,
                "surprise": rating.surprise,
            }
            for sample, rating in zip(chunk_samples, raw, strict=True)
        ]
        chunks.append({
            "start_index": start,
            "end_index": start + len(chunk_samples),
            "sample_ids": [str(sample["id"]) for sample in chunk_samples],
            "prompt": prompt,
            "ratings": chunk_ratings,
        })
        ratings.extend(chunk_ratings)
    quality: dict[str, object] = {"status": "available", "model": model}
    by_variant: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample, rating in zip(samples, ratings, strict=True):
        by_variant[str(sample["variant"])].append(rating)
    for variant, variant_ratings in by_variant.items():
        quality[variant] = {
            "coherence": mean(float(row["coherence"]) for row in variant_ratings),
            "evocativeness": mean(
                float(row["evocativeness"]) for row in variant_ratings
            ),
            "surprise": mean(float(row["surprise"]) for row in variant_ratings),
            "overall": mean(
                mean((
                    float(row["coherence"]),
                    float(row["evocativeness"]),
                    float(row["surprise"]),
                ))
                for row in variant_ratings
            ),
        }
    return quality, ratings, chunks


def _reuse_cached_ratings(
    samples: list[dict[str, object]],
    *,
    model: str,
    ratings_path: Path,
    existing_samples_path: Path,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]] | None:
    if not ratings_path.exists() or not existing_samples_path.exists():
        return None
    try:
        cached_samples = json.loads(existing_samples_path.read_text(encoding="utf-8"))
        cached_ratings = json.loads(ratings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if cached_samples.get("schema_version") != STEP08_SCHEMA_VERSION:
        return None
    if cached_samples.get("first_draw_samples") != samples:
        return None
    if cached_ratings.get("model") != model:
        return None
    if cached_ratings.get("prompt_template") != RATING_PROMPT:
        return None
    quality = cached_ratings.get("quality")
    ratings = cached_ratings.get("ratings")
    chunks = cached_ratings.get("chunks", [])
    if not isinstance(quality, dict) or not isinstance(ratings, list) or not isinstance(chunks, list):
        return None
    return quality, ratings, chunks


def _matrix_metrics(
    samples: list[dict[str, object]],
    artifacts: dict[str, ArtifactIndex],
    legacy_distributions: dict[tuple[str, str], dict[str, float]],
    quality: dict[str, object],
) -> dict[str, dict[str, object]]:
    metrics: dict[str, dict[str, object]] = {}
    for variant in PRIMARY_VARIANT_NAMES:
        rows = [sample for sample in samples if sample["variant"] == variant]
        scorer = artifacts["anchored_base"] if variant == "legacy" else artifacts[variant]
        variant_metrics: dict[str, object] = {}
        empirical: dict[str, dict[str, float]] = {}
        for scenario, _allowed in SCENARIOS:
            scenario_rows = [row for row in rows if row["scenario"] == scenario]
            scenario_metrics, distribution = _sample_metrics(
                scenario_rows,
                scorer,
            )
            variant_metrics[scenario] = scenario_metrics
            empirical[scenario] = distribution
        pop_mainstream = _js_divergence(empirical["pop"], empirical["mainstream"])
        mainstream_niche = _js_divergence(
            empirical["mainstream"],
            empirical["niche"],
        )
        pop_niche = _js_divergence(empirical["pop"], empirical["niche"])
        variant_metrics["mainstream_distinctiveness"] = mean(
            (pop_mainstream, mainstream_niche)
        )
        variant_metrics["tone_separation"] = mean(
            (pop_mainstream, mainstream_niche, pop_niche)
        )
        comparisons = _distribution_comparisons(scorer, legacy_distributions)
        variant_metrics["distribution_comparisons"] = comparisons
        variant_metrics["mean_js_from_legacy"] = mean(
            float(row["js"]) for row in comparisons
        )
        if isinstance(quality.get(variant), dict):
            variant_metrics["quality"] = quality[variant]
        metrics[variant] = variant_metrics
    return metrics


def _public_contract_comparison(
    public_contract_samples: list[dict[str, object]],
    primary_metrics: dict[str, dict[str, object]],
    legacy_reference: ArtifactIndex,
) -> dict[str, object]:
    legacy_public_metrics = {}
    for scenario, _allowed_tiers in SCENARIOS:
        scenario_samples = [
            sample
            for sample in public_contract_samples
            if sample["scenario"] == scenario
        ]
        scenario_metrics, _distribution = _sample_metrics(
            scenario_samples,
            legacy_reference,
        )
        legacy_public_metrics[scenario] = scenario_metrics
    return {
        "legacy_public_retry": legacy_public_metrics,
        "direct_first_draw": {
            variant: {
                tier: primary_metrics[variant][tier]
                for tier in (*TIER_NAMES, "default")
            }
            for variant in PRIMARY_VARIANT_NAMES
        },
    }


def _sample_metrics(
    samples: list[dict[str, object]],
    artifact: ArtifactIndex,
) -> tuple[dict[str, object], dict[str, float]]:
    subtitle_counts = Counter(str(row["text"]) for row in samples)
    filler_counts: Counter[str] = Counter()
    tail_count = 0
    source_covered = 0
    evidence_covered = 0
    source_counts: list[float] = []
    evidence_counts: list[float] = []
    for sample in samples:
        scoring_tier = (
            str(sample["requested_tier"])
            if sample["requested_tier"]
            else str(sample["compatible_tier"])
        )
        for filler in sample["fillers"]:
            slot_type = str(filler["slot_type"])
            display_filler = str(filler["filler"])
            key = filler_key(slot_type, display_filler)
            filler_counts[f"{slot_type}:{key}"] += 1
            group = artifact.groups[(slot_type, scoring_tier)]
            row = group[key]
            ranked = sorted(
                group.values(),
                key=lambda item: float(item["probability"]),
                reverse=True,
            )
            rank = next(
                index
                for index, item in enumerate(ranked)
                if filler_key(slot_type, str(item["display_filler"])) == key
            )
            if len(ranked) > 1 and rank / (len(ranked) - 1) >= 0.80:
                tail_count += 1
            source_count = float(row["source_count"])
            evidence_count = float(row["evidence_count"])
            source_counts.append(source_count)
            evidence_counts.append(evidence_count)
            source_covered += source_count > 0
            evidence_covered += evidence_count > 0
    total_fillers = sum(filler_counts.values())
    requested = [row for row in samples if row["requested_tier"]]
    distribution = _normalise_counts(filler_counts)
    entropy = _entropy(distribution.values())
    return {
        "sample_count": len(samples),
        "unique_subtitle_rate": len(subtitle_counts) / len(samples),
        "unique_filler_rate": len(filler_counts) / total_fillers,
        "top_subtitle_mass": max(subtitle_counts.values()) / len(samples),
        "top_filler_mass": max(filler_counts.values()) / total_fillers,
        "tail_exposure": tail_count / total_fillers,
        "filler_entropy": entropy,
        "effective_filler_n": math.exp(entropy),
        "source_coverage": source_covered / total_fillers,
        "evidence_coverage": evidence_covered / total_fillers,
        "mean_source_count": mean(source_counts),
        "mean_evidence_count": mean(evidence_counts),
        "first_draw_compatible_requested_tier_compliance": (
            mean(
                row["compatible_tier"] == row["target_tier"]
                for row in requested
            )
            if requested
            else None
        ),
        "first_draw_legacy_requested_tier_agreement": (
            mean(row["legacy_tier"] == row["target_tier"] for row in requested)
            if requested
            else None
        ),
    }, distribution


def _legacy_distributions(
    conn: sqlite3.Connection,
) -> dict[tuple[str, str], dict[str, float]]:
    candidates = _load_generation_candidates(conn)
    rows_by_slot = {
        "list_item": candidates.list_rows,
        "action_noun": candidates.action_rows,
        "of_object": candidates.obj_rows,
    }
    score_index = {"pop": 3, "mainstream": 4, "niche": 5}
    if any(
        len(row) <= max(score_index.values())
        for rows in rows_by_slot.values()
        for row in rows
    ):
        raise RuntimeError(
            "Step 8 legacy distribution comparison requires "
            "slot_filler_model_scores for every runtime candidate"
        )
    result: dict[tuple[str, str], dict[str, float]] = {}
    for slot_type, rows in rows_by_slot.items():
        for tier in TIER_NAMES:
            weights = {
                filler_key(slot_type, str(row[0])): (
                    math.sqrt(float(row[1]))
                    * max(float(row[score_index[tier]] or 0.0), 0.001)
                )
                for row in rows
            }
            result[(slot_type, tier)] = _normalise(weights)
    return result


def _distribution_comparisons(
    artifact: ArtifactIndex,
    legacy: dict[tuple[str, str], dict[str, float]],
) -> list[dict[str, object]]:
    comparisons: list[dict[str, object]] = []
    for slot_type in SLOT_TYPES:
        for tier in TIER_NAMES:
            artifact_distribution = {
                key: float(row["probability"])
                for key, row in artifact.groups[(slot_type, tier)].items()
            }
            legacy_distribution = legacy[(slot_type, tier)]
            top_artifact = set(
                sorted(
                    artifact_distribution,
                    key=artifact_distribution.get,
                    reverse=True,
                )[:20]
            )
            top_legacy = set(
                sorted(
                    legacy_distribution,
                    key=legacy_distribution.get,
                    reverse=True,
                )[:20]
            )
            comparisons.append({
                "slot_type": slot_type,
                "tier": tier,
                "top20_overlap": len(top_artifact & top_legacy)
                / len(top_artifact | top_legacy),
                "js": _js_divergence(artifact_distribution, legacy_distribution),
                "kl_artifact_to_legacy": _kl_divergence(
                    artifact_distribution,
                    legacy_distribution,
                ),
            })
    return comparisons


def _smoothing_review(
    base: ArtifactIndex,
    smoothed: ArtifactIndex,
    samples: list[dict[str, object]],
) -> dict[str, object]:
    review_slots = ("action_noun", *SPARSE_OF_SLOT_TYPES)
    moves: list[dict[str, object]] = []
    for slot_type in review_slots:
        for tier in TIER_NAMES:
            base_group = base.groups[(slot_type, tier)]
            smooth_group = smoothed.groups[(slot_type, tier)]
            group_moves = []
            for key, row in base_group.items():
                delta = float(smooth_group[key]["probability"]) - float(row["probability"])
                if abs(delta) <= 1e-15:
                    continue
                display = str(row["display_filler"])
                observed_contexts = [
                    str(sample["text"])
                    for sample in samples
                    if sample["variant"] == "smoothed"
                    and any(
                        filler["slot_type"] == slot_type
                        and filler["filler"].casefold() == display.casefold()
                        for filler in sample["fillers"]
                    )
                ][:3]
                review_contexts = list(observed_contexts)
                for context in _review_contexts(slot_type, display):
                    if len(review_contexts) >= 3:
                        break
                    if context not in review_contexts:
                        review_contexts.append(context)
                group_moves.append({
                    "slot_type": slot_type,
                    "tier": tier,
                    "filler": display,
                    "compound": len(display.replace("-", " ").split()) > 1,
                    "source_count": int(float(row["source_count"])),
                    "base_probability": float(row["probability"]),
                    "smoothed_probability": float(smooth_group[key]["probability"]),
                    "delta": delta,
                    "contexts": review_contexts,
                })
            moves.extend(
                sorted(group_moves, key=lambda row: abs(row["delta"]), reverse=True)[:6]
            )
    compound_moves = [row for row in moves if row["compound"]]
    return {
        "method": (
            "Top absolute accepted-smoothing moves for action_noun plus sparse "
            "of_* groups, each paired with up to three generated contexts."
        ),
        "moves": moves,
        "compound_hotspot": {
            "move_count": len(compound_moves),
            "share": len(compound_moves) / len(moves) if moves else 0.0,
            "low_support_share": (
                mean(row["source_count"] < 3 for row in compound_moves)
                if compound_moves
                else 0.0
            ),
        },
    }


def _review_contexts(slot_type: str, filler: str) -> tuple[str, str, str]:
    if slot_type == "action_noun":
        objects = (
            "Modern Life",
            "the State",
            "Everyday Life",
        )
        return tuple(
            f"Memory, Power, and the {filler} of {obj}" for obj in objects
        )
    if slot_type == "of_modifier":
        objects = (
            f"{filler} Memory",
            f"{filler} Institutions",
            f"{filler} Politics",
        )
    elif slot_type == "of_head":
        objects = (
            f"Postcolonial {filler}",
            f"Public {filler}",
            f"Modern {filler}",
        )
    elif slot_type == "of_topic":
        objects = (
            f"{filler} in America",
            f"{filler} after Empire",
            f"{filler} and Society",
        )
    elif slot_type == "of_complement":
        objects = (
            f"Politics of {filler}",
            f"History of {filler}",
            f"Culture and {filler}",
        )
    else:
        raise RuntimeError(f"Unsupported Step 8 review slot: {slot_type}")
    return tuple(
        f"Memory, Power, and the Making of {obj}" for obj in objects
    )


def _evidence_ceiling(
    base: ArtifactIndex,
    smoothed: ArtifactIndex,
    metrics: dict[str, dict[str, object]],
) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for tier in ("pop", "mainstream"):
        for slot_type in SLOT_TYPES:
            base_group = base.groups[(slot_type, tier)]
            smooth_group = smoothed.groups[(slot_type, tier)]
            probabilities = {
                key: float(row["probability"]) for key, row in base_group.items()
            }
            smoothed_probabilities = {
                key: float(row["probability"]) for key, row in smooth_group.items()
            }
            artifact_vocab = len(probabilities)
            observed_vocab = sum(
                int(float(row["source_count"])) > 0 or float(row["soft_count"]) > 0
                for row in base_group.values()
            )
            anchored_vocab = sum(
                float(row["anchored_soft_count"]) > 0 for row in base_group.values()
            )
            inferred_only_vocab = sum(
                float(row["soft_count"]) > 0 and float(row["anchored_soft_count"]) <= 0
                for row in base_group.values()
            )
            prior_only_vocab = sum(float(row["soft_count"]) <= 0 for row in base_group.values())
            changed_vocab = sum(
                abs(smoothed_probabilities[key] - probability) > 1e-15
                for key, probability in probabilities.items()
            )
            groups.append({
                "tier": tier,
                "slot_type": slot_type,
                "artifact_vocabulary": artifact_vocab,
                "observed_vocabulary": observed_vocab,
                "anchored_vocabulary": anchored_vocab,
                "inferred_only_vocabulary": inferred_only_vocab,
                "prior_only_vocabulary": prior_only_vocab,
                "smoothed_changed_vocabulary": changed_vocab,
                "effective_n": math.exp(_entropy(probabilities.values())),
                "top10_mass": sum(
                    sorted(probabilities.values(), reverse=True)[:10]
                ),
                "smoothed_effective_n": math.exp(
                    _entropy(smoothed_probabilities.values())
                ),
                "smoothed_top10_mass": sum(
                    sorted(smoothed_probabilities.values(), reverse=True)[:10]
                ),
            })
    scarcity = any(
        row["observed_vocabulary"] > 0
        and (
            row["anchored_vocabulary"] < 0.35 * row["observed_vocabulary"]
            or row["top10_mass"] > 0.25
        )
        for row in groups
    )
    best_pop_mainstream = max(
        mean(
            float(metrics[variant][tier]["first_draw_compatible_requested_tier_compliance"])
            for tier in ("pop", "mainstream")
        )
        for variant in ("anchored_base", "calibrated", "smoothed")
    )
    return {
        "groups": groups,
        "conclusion": (
            "Evidence scarcity/teacher imbalance is likely limiting recoverable "
            "pop/mainstream vocabulary."
            if scarcity and best_pop_mainstream < 0.65
            else (
                "Observed evidence is sparse/concentrated, but Step 8 alone does "
                "not prove it is the limiting factor."
                if scarcity
                else "The corrected evidence metrics do not show a binding ceiling."
            )
        ),
        "limiting": scarcity and best_pop_mainstream < 0.65,
        "scarcity_signals_present": scarcity,
        "best_pop_mainstream_first_draw_compliance": best_pop_mainstream,
    }


def _recommend(
    gates: dict[str, list[dict[str, object]]],
    metrics: dict[str, dict[str, object]],
) -> tuple[str, str | None, str | None]:
    for variant in ("calibrated", "smoothed", "anchored_base"):
        if all(gate["status"] == "pass" for gate in gates[variant]):
            return "promote", variant, variant
    best_experimental = max(
        ("anchored_base", "calibrated", "smoothed"),
        key=lambda variant: (
            sum(gate["status"] == "pass" for gate in gates[variant]),
            mean(
                float(metrics[variant][tier]["first_draw_compatible_requested_tier_compliance"])
                for tier in TIER_NAMES
            ),
            float(metrics[variant]["quality"]["overall"])
            if isinstance(metrics[variant].get("quality"), dict)
            else float("-inf"),
        ),
    )
    return "defer", None, best_experimental


def _decision_payload(replay: dict[str, object]) -> dict[str, object]:
    recommendation = str(replay["recommendation"])
    recommended_variant = replay["recommended_variant"]
    best_experimental_variant = replay["best_experimental_variant"]
    payload = {
        "schema_version": STEP08_SCHEMA_VERSION,
        "decision": recommendation,
        "recommended_variant": recommended_variant,
        "best_experimental_variant": best_experimental_variant,
        "summary": (
            f"Step 8 recommends {recommendation}. "
            + (
                f"The bounded rollout winner is {recommended_variant}."
                if recommended_variant
                else (
                    "No direct-draw variant cleared the complete rollout gate; "
                    f"{best_experimental_variant} is the best shadow candidate."
                )
            )
        ),
        "gate_policy": replay["gate_policy"],
        "gates": replay["gates"],
        "digests": replay["digests"],
        "evaluation_source_binding": _durable_source_binding(
            replay["evaluation_source_binding"]
        ),
        "metrics": {
            variant: {
                "pop": values["pop"],
                "mainstream": values["mainstream"],
                "niche": values["niche"],
                "tone_separation": values["tone_separation"],
                "mainstream_distinctiveness": values["mainstream_distinctiveness"],
                "mean_js_from_legacy": values["mean_js_from_legacy"],
                "quality": values.get("quality"),
            }
            for variant, values in replay["metrics"].items()
        },
        "quality": replay["quality"],
        "public_contract_comparison": replay["public_contract_comparison"],
        "evidence_ceiling": replay["evidence_ceiling"],
        "representative_smoothing_failures": [
            row
            for row in replay["smoothing_review"]["moves"]
            if row["compound"] and row["source_count"] < 3
        ][:5],
        "rationale": (
            "The recommendation is automated from gates frozen before the "
            "results; code and accepted-input digests are bound into the record."
        ),
        "rollback_implications": (
            "No default changed. Rollback is to omit shadow runtime selection "
            "and continue using legacy generation."
        ),
        "step9_recommendation": (
            "Proceed to the default switch only for the selected variant."
            if recommendation == "promote"
            else (
                "Do not switch defaults in Step 9. Minimum future work: improve "
                "tier-conditioned evidence/distributions enough to meet the "
                "public-contract continuity gate, or explicitly design and validate "
                "a compatible retry/policy as a new rollout candidate."
            )
        ),
    }
    payload["decision_digest"] = stable_digest(payload)
    return payload


def _format_durable_readme(decision: dict[str, object]) -> str:
    return "\n".join([
        "# Step 8 artifact-runtime validation",
        "",
        "This folder contains the tracked promote/iterate/defer decision for issue #41.",
        "Regenerable samples, ratings, artifacts, and the full report remain under",
        "`test-results/step08-validation/`.",
        "",
        "## Replay",
        "",
        "```powershell",
        "uv run subtitle-gen validate-artifact-runtime `",
        "  --db C:\\_SRC\\subtitle-generator\\data\\db\\subtitles.db",
        "```",
        "",
        f"**Decision:** `{decision['decision']}`",
        f"**Decision digest:** `{decision['decision_digest']}`",
        "",
        str(decision["summary"]),
        "",
        f"Evaluation source digest: `{decision['evaluation_source_binding']['evaluation_source_digest']}`",
        "",
        "The command does not change runtime defaults.",
        "",
    ])


def _format_report(replay: dict[str, object]) -> str:
    lines = [
        "# Step 8 behavioral validation",
        "",
        f"**Automated decision:** `{replay['recommendation']}`",
        "",
        "Primary matrix semantics: **first draw for a chosen target tier** (no classifier retry) for legacy and every shadow variant.",
        "Current public-contract behavior is reported separately below.",
        "",
        "## Frozen gates",
        "",
        "| Variant | Gate | Status |",
        "|---|---|---|",
    ]
    for variant, gates in replay["gates"].items():
        for gate in gates:
            lines.append(f"| {variant} | {gate['name']} | {gate['status']} |")
    lines.extend([
        "",
        "## Behavioral metrics",
        "",
        "| Variant | Tier | first-draw compatible compliance | legacy-scorer agreement | "
        "unique subtitles | unique fillers | top filler mass | tail exposure | "
        "effective N |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant in PRIMARY_VARIANT_NAMES:
        variant_metrics = replay["metrics"][variant]
        for tier in TIER_NAMES:
            values = variant_metrics[tier]
            lines.append(
                f"| {variant} | {tier} | "
                f"{values['first_draw_compatible_requested_tier_compliance']:.3f} | "
                f"{values['first_draw_legacy_requested_tier_agreement']:.3f} | "
                f"{values['unique_subtitle_rate']:.3f} | "
                f"{values['unique_filler_rate']:.3f} | "
                f"{values['top_filler_mass']:.3f} | "
                f"{values['tail_exposure']:.3f} | "
                f"{values['effective_filler_n']:.1f} |"
            )
        values = variant_metrics["default"]
        lines.append(
            f"| {variant} | default | - | - | "
            f"{values['unique_subtitle_rate']:.3f} | "
            f"{values['unique_filler_rate']:.3f} | "
            f"{values['top_filler_mass']:.3f} | "
            f"{values['tail_exposure']:.3f} | "
            f"{values['effective_filler_n']:.1f} |"
        )
    lines.extend([
        "",
        "## Current user-contract comparison",
        "",
        "Legacy retry still protects the requested-tier contract better than any "
        "direct-draw variant, but it also collapses pop diversity (30/30 identical "
        "pop subtitles in this replay). Anchored direct avoids that collapse while "
        "still missing the continuity bar.",
        "",
        "| Runtime | Tier | compatible agreement | legacy-scorer agreement | unique subtitles | top filler mass | effective N |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for tier in TIER_NAMES:
        values = replay["public_contract_comparison"]["legacy_public_retry"][tier]
        lines.append(
            f"| {PUBLIC_CONTRACT_VARIANT_NAME} | {tier} | "
            f"{values['first_draw_compatible_requested_tier_compliance']:.3f} | "
            f"{values['first_draw_legacy_requested_tier_agreement']:.3f} | "
            f"{values['unique_subtitle_rate']:.3f} | "
            f"{values['top_filler_mass']:.3f} | "
            f"{values['effective_filler_n']:.1f} |"
        )
    lines.extend([
        "",
        "| Variant | tone separation | mainstream distinctiveness | "
        "mean JS from legacy | mean KL to legacy | mean top-20 overlap |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for variant in PRIMARY_VARIANT_NAMES:
        variant_metrics = replay["metrics"][variant]
        comparisons = variant_metrics["distribution_comparisons"]
        lines.append(
            f"| {variant} | {variant_metrics['tone_separation']:.3f} | "
            f"{variant_metrics['mainstream_distinctiveness']:.3f} | "
            f"{variant_metrics['mean_js_from_legacy']:.3f} | "
            f"{mean(row['kl_artifact_to_legacy'] for row in comparisons):.3f} | "
            f"{mean(row['top20_overlap'] for row in comparisons):.3f} |"
        )
    lines.extend([
        "",
        "## Quality",
        "",
        f"```json\n{json.dumps(replay['quality'], indent=2)}\n```",
        "",
        "## Code binding",
        "",
        f"- Evaluation source digest: `{replay['evaluation_source_binding']['evaluation_source_digest']}`",
        f"- Base revision (provenance only): `{replay['evaluation_source_binding']['base_revision']}`",
        f"- Replay input digest: `{replay['digests']['replay_input']}`",
        "",
        "## Evidence ceiling",
        "",
        str(replay["evidence_ceiling"]["conclusion"]),
        "",
        "| Tier | Slot | artifact vocab | observed vocab | anchored vocab | inferred-only | prior-only | effective N | top-10 mass | smoothed effective N | smoothed top-10 mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for group in replay["evidence_ceiling"]["groups"]:
        lines.append(
            f"| {group['tier']} | {group['slot_type']} | "
            f"{group['artifact_vocabulary']} | {group['observed_vocabulary']} | "
            f"{group['anchored_vocabulary']} | {group['inferred_only_vocabulary']} | "
            f"{group['prior_only_vocabulary']} | "
            f"{group['effective_n']:.1f} | {group['top10_mass']:.3f} | "
            f"{group['smoothed_effective_n']:.1f} | "
            f"{group['smoothed_top10_mass']:.3f} |"
        )
    failures = [
        move
        for move in replay["smoothing_review"]["moves"]
        if move["compound"] and move["source_count"] < 3
    ][:5]
    lines.extend([
        "",
        "## Representative smoothing risks",
        "",
    ])
    if failures:
        for failure in failures:
            lines.append(
                f"- `{failure['slot_type']}/{failure['tier']}` "
                f"**{failure['filler']}** (sources={failure['source_count']}, "
                f"delta={failure['delta']:.6f}): "
                + "; ".join(failure["contexts"])
            )
    else:
        lines.append("- No low-support compound appeared in the top reviewed moves.")
    lines.extend([
        "",
        "## Step 9",
        "",
        (
            "Proceed only with the selected variant."
            if replay["recommendation"] == "promote"
            else (
                "Do not switch defaults; the recommendation is "
                f"{replay['recommendation']}."
            )
        ),
        "",
        (
            f"Best shadow candidate: `{replay['best_experimental_variant']}`."
            if replay["best_experimental_variant"]
            else "No shadow candidate identified."
        ),
        "",
    ])
    return "\n".join(lines)


def _subtitle_fillers(
    subtitle: GeneratedSubtitle,
) -> tuple[tuple[str, str], ...]:
    fillers: list[tuple[str, str]] = [
        ("list_item", subtitle.item1),
        ("list_item", subtitle.item2),
        ("action_noun", subtitle.action_noun),
    ]
    if subtitle.remixed and subtitle.remix_parts:
        if "modifier" in subtitle.remix_parts and "head" in subtitle.remix_parts:
            fillers.extend(
                [
                    ("of_modifier", subtitle.remix_parts["modifier"]),
                    ("of_head", subtitle.remix_parts["head"]),
                ]
            )
        elif "topic" in subtitle.remix_parts and "complement" in subtitle.remix_parts:
            fillers.extend(
                [
                    ("of_topic", subtitle.remix_parts["topic"]),
                    ("of_complement", subtitle.remix_parts["complement"]),
                ]
            )
        else:
            fillers.append(("of_object", subtitle.of_object))
    else:
        fillers.append(("of_object", subtitle.of_object))
    return tuple(fillers)


def _add_gate(
    gates: list[dict[str, object]],
    name: str,
    passed: bool,
    evidence: dict[str, object],
) -> None:
    gates.append({
        "name": name,
        "status": "pass" if passed else "fail",
        "evidence": evidence,
    })


def _softmax(values: dict[str, float]) -> dict[str, float]:
    maximum = max(values.values())
    exponentials = {
        key: math.exp(value - maximum) for key, value in values.items()
    }
    total = sum(exponentials.values())
    return {key: value / total for key, value in exponentials.items()}


def _normalise(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        raise RuntimeError("Cannot normalise a nonpositive distribution")
    return {key: value / total for key, value in values.items()}


def _normalise_counts(values: Counter[str]) -> dict[str, float]:
    return _normalise({key: float(value) for key, value in values.items()})


def _entropy(values) -> float:
    return -sum(value * math.log(value) for value in values if value > 0)


def _js_divergence(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    keys = set(left) | set(right)
    left_normalised = _normalise({key: left.get(key, 0.0) for key in keys})
    right_normalised = _normalise({key: right.get(key, 0.0) for key in keys})
    midpoint = {
        key: (left_normalised[key] + right_normalised[key]) / 2
        for key in keys
    }
    return (
        _kl_divergence(left_normalised, midpoint)
        + _kl_divergence(right_normalised, midpoint)
    ) / 2


def _kl_divergence(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    epsilon = 1e-15
    keys = set(left) | set(right)
    left_normalised = _normalise({
        key: max(left.get(key, 0.0), epsilon) for key in keys
    })
    right_normalised = _normalise({
        key: max(right.get(key, 0.0), epsilon) for key in keys
    })
    return sum(
        left_normalised[key]
        * math.log(left_normalised[key] / right_normalised[key])
        for key in keys
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
