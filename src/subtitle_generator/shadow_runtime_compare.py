"""Deterministic side-by-side comparisons for the shadow runtime."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from subtitle_generator.generate import generate_subtitle_matching_tiers
from subtitle_generator.shadow_runtime import (
    PreparedGenerationRuntime,
    RuntimeSelectionMode,
    build_generation_runtime,
    prepare_generation_runtime,
    shadow_runtime_provenance,
)
from subtitle_generator.tiering import compute_tier_evidence

DEFAULT_COMPARISON_SEEDS = (101, 202, 303)
_COMPARISON_SCENARIOS: tuple[tuple[str, set[str] | None], ...] = (
    ("pop", {"pop"}),
    ("mainstream", {"mainstream"}),
    ("niche", {"niche"}),
    ("default", None),
)


@dataclass(frozen=True)
class ShadowRuntimeComparisonResult:
    report_path: Path
    details_path: Path
    comparison_count: int


def build_shadow_runtime_comparison(
    conn: sqlite3.Connection,
    output_dir: Path,
    *,
    shadow_runtime,
    seeds: tuple[int, ...] = DEFAULT_COMPARISON_SEEDS,
    remix_prob: float = 0.0,
    min_sim: float = 0.0,
) -> ShadowRuntimeComparisonResult:
    """Write a replayable legacy-vs-shadow comparison packet."""

    if not seeds:
        raise RuntimeError("Shadow runtime comparison requires at least one seed")
    prepared_shadow = prepare_generation_runtime(conn, shadow_runtime)
    if prepared_shadow.mode != RuntimeSelectionMode.SHADOW:
        raise RuntimeError("Shadow runtime comparison requires runtime mode 'shadow'")
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[dict[str, object]] = []
    legacy_runtime = build_generation_runtime(mode=RuntimeSelectionMode.LEGACY)
    for scenario_name, allowed_tiers in _COMPARISON_SCENARIOS:
        for seed in seeds:
            legacy = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=allowed_tiers,
                seed=seed,
                remix_prob=remix_prob,
                min_sim=min_sim,
                runtime=legacy_runtime,
            )
            shadow = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=allowed_tiers,
                seed=seed,
                remix_prob=remix_prob,
                min_sim=min_sim,
                runtime=prepared_shadow,
            )
            legacy_evidence = compute_tier_evidence(
                legacy.text,
                conn,
                remix_parts=legacy.remix_parts if legacy.remixed else None,
            )
            shadow_evidence = compute_tier_evidence(
                shadow.text,
                conn,
                remix_parts=shadow.remix_parts if shadow.remixed else None,
            )
            comparisons.append({
                "scenario": scenario_name,
                "seed": seed,
                "requested_tiers": sorted(allowed_tiers) if allowed_tiers else [],
                "legacy": {
                    "text": legacy.text,
                    "observed_tier": legacy_evidence.tier,
                    "remixed": legacy.remixed,
                },
                "shadow": {
                    "text": shadow.text,
                    "observed_tier": shadow_evidence.tier,
                    "remixed": shadow.remixed,
                },
            })

    details = {
        "comparison_schema_version": 1,
        "shadow_runtime": shadow_runtime_provenance(prepared_shadow),
        "policy": {
            "remix_prob": remix_prob,
            "min_sim": min_sim,
            "seeds": list(seeds),
            "scenarios": [scenario for scenario, _allowed in _COMPARISON_SCENARIOS],
        },
        "comparisons": comparisons,
    }
    details_path = output_dir / "shadow_runtime_comparison.json"
    details_path.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")

    report_path = output_dir / "shadow_runtime_comparison_report.md"
    report_path.write_text(
        _format_report(prepared_shadow, details, details_path=details_path),
        encoding="utf-8",
    )
    return ShadowRuntimeComparisonResult(
        report_path=report_path,
        details_path=details_path,
        comparison_count=len(comparisons),
    )


def _format_report(
    prepared_shadow: PreparedGenerationRuntime,
    details: dict[str, object],
    *,
    details_path: Path,
) -> str:
    provenance = shadow_runtime_provenance(prepared_shadow)
    lines = [
        "# Shadow runtime comparison",
        "",
        "Side-by-side fixed-seed samples for the current legacy runtime and the "
        "explicit shadow runtime. Default serving remains on the legacy path.",
        "",
        "## Provenance",
        "",
        f"- Shadow artifact: `{provenance['artifact_source']}`",
        f"- Artifact digest: `{provenance['artifact_digest']}`",
        f"- Artifact version: `{provenance['artifact_version']}`",
        f"- Sampling temperature: `{provenance['sampling_temperature']}`",
        f"- Replay packet: `{details_path}`",
        "",
    ]
    comparisons = list(details["comparisons"])
    for scenario, _allowed in _COMPARISON_SCENARIOS:
        lines.extend([
            f"## {scenario}",
            "",
            "| seed | legacy tier | legacy subtitle | shadow tier | shadow subtitle |",
            "|---|---|---|---|---|",
        ])
        for comparison in comparisons:
            if comparison["scenario"] != scenario:
                continue
            legacy = comparison["legacy"]
            shadow = comparison["shadow"]
            lines.append(
                f"| {comparison['seed']} | {legacy['observed_tier']} | "
                f"{legacy['text']} | {shadow['observed_tier']} | {shadow['text']} |"
            )
        lines.append("")
    return "\n".join(lines)
