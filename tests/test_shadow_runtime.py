"""Tests for explicit shadow runtime support."""

import json
import math
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subtitle_generator.export_db import build_mini_db, export_data
from subtitle_generator.generate import compose_compound, generate_subtitle_matching_tiers
from subtitle_generator.schema_contracts import TIER_SLOT_FILLER_DISTRIBUTION_TABLE
from subtitle_generator.shadow_runtime import (
    PreparedGenerationRuntime,
    RuntimeSelectionMode,
    build_generation_runtime,
    clear_distribution_cache,
    install_tier_slot_distribution,
    prepare_generation_runtime,
    sample_shadow_candidates,
    set_configured_runtime_mode,
    shadow_runtime_provenance,
    write_shadow_distribution_csv,
)
from subtitle_generator.shadow_runtime_compare import build_shadow_runtime_comparison


_SLOT_FILLERS = {
    "list_item": [
        ("Race", "pop"),
        ("Power", "pop"),
        ("History", "mainstream"),
        ("Identity", "mainstream"),
        ("Aesthesis", "niche"),
        ("Ontology", "niche"),
    ],
    "action_noun": [
        ("Making", "pop"),
        ("Pursuit", "mainstream"),
        ("Problematizing", "niche"),
    ],
    "of_object": [
        ("Modern Life", "pop"),
        ("Daily Life", "mainstream"),
        ("Late Antiquity", "niche"),
    ],
}


def test_runtime_import_does_not_require_offline_ml_dependencies():
    script = """
import sys
sys.modules["spacy"] = None
sys.modules["subtitle_generator.tier_slot_calibration"] = None
sys.modules["subtitle_generator.tier_slot_distribution"] = None
from subtitle_generator import handlers
assert handlers.handle_health()[0] == 200
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parent.parent,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def _make_shadow_runtime_db(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:" if path is None else str(path))
    conn.execute(
        """
        CREATE TABLE config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            vector_sum BLOB,
            token_count INTEGER,
            centroid_dot REAL,
            norm_sq REAL,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            source_file TEXT
        )
        """
    )
    for subtitle_id in range(1, 32):
        conn.execute(
            "INSERT INTO subtitles VALUES (?, ?, ?, ?)",
            (subtitle_id, f"Book {subtitle_id}", f"Subtitle {subtitle_id}", "openlibrary"),
        )
    conn.execute(
        "INSERT INTO config VALUES ('remix_calibrated_remix_prob', '0.0')"
    )
    conn.execute(
        "INSERT INTO config VALUES ('remix_calibrated_min_sim', '0.0')"
    )

    subtitle_id = 1
    for slot_type, fillers in _SLOT_FILLERS.items():
        for filler, tier in fillers:
            conn.execute(
                """
                INSERT INTO slot_fillers (
                    slot_type, filler, mode, source_subtitle_id, freq, popularity_score,
                    popularity_level, popularity_confidence
                )
                VALUES (?, ?, 'strict', ?, 10, 1.0, 1, 1.0)
                """,
                (slot_type, filler, subtitle_id),
            )
            slot_filler_id = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]
            scores = {
                "pop": (0.95, 0.04, 0.01),
                "mainstream": (0.02, 0.96, 0.02),
                "niche": (0.01, 0.04, 0.95),
            }[tier]
            conn.execute(
                """
                INSERT INTO slot_filler_model_scores (
                    slot_filler_id, score_pop, score_mainstream, score_niche,
                    model_tier, source_prediction_count
                )
                VALUES (?, ?, ?, ?, ?, 3)
                """,
                (slot_filler_id, scores[0], scores[1], scores[2], tier),
            )
            subtitle_id += 1
    conn.commit()
    return conn


def _distribution_rows(*, calibration_temperature: float = 0.73):
    rows: list[dict[str, object]] = []
    for slot_type, fillers in _SLOT_FILLERS.items():
        filler_names = [name for name, _tier in fillers]
        for tier in ("pop", "mainstream", "niche"):
            if slot_type == "list_item":
                target_names = {
                    name for name, filler_tier in fillers if filler_tier == tier
                }
                probabilities = {
                    name: (0.495 if name in target_names else 0.0025)
                    for name in filler_names
                }
            else:
                probabilities = {
                    name: (
                        0.98 if filler_tier == tier else 0.01
                    )
                    for name, filler_tier in fillers
                }
            for name, filler_tier in fillers:
                probability = probabilities[name]
                rows.append(
                    {
                        "slot_type": slot_type,
                        "tier": tier,
                        "filler": name.casefold(),
                        "display_filler": name,
                        "probability": probability,
                        "log_probability": math.log(probability),
                        "soft_count": 1.0,
                        "prior_count": 0.0,
                        "evidence_count": 1.0,
                        "source_count": 1,
                        "anchored_source_count": 1,
                        "inferred_source_count": 0,
                        "anchored_soft_count": 1.0,
                        "inferred_soft_count": 0.0,
                        "teacher_confidence_mean": 1.0,
                        "frequency": 10,
                        "popularity_score": 1.0,
                        "semantic_smoothing_mass": 0.0,
                        "calibration_temperature": calibration_temperature,
                        "artifact_version": "shadow-test-v1",
                    }
                )
    return rows


def _write_shadow_artifact(path: Path, rows: list[dict[str, object]]) -> Path:
    write_shadow_distribution_csv(path, rows)
    return path


def _make_legacy_remix_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            vector_sum BLOB,
            token_count INTEGER,
            centroid_dot REAL,
            norm_sq REAL,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO slot_fillers (
            slot_type, filler, mode, freq, pos_tag, popularity_score,
            popularity_level, popularity_confidence
        )
        VALUES (?, ?, 'strict', ?, ?, 1.0, 1, 1.0)
        """,
        [
            ("of_modifier", "C.E.ric", 10, "PROPN"),
            ("of_head", "E.G.riggs", 10, None),
        ],
    )
    conn.commit()
    return conn


def _make_shadow_remix_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            vector_sum BLOB,
            token_count INTEGER,
            centroid_dot REAL,
            norm_sq REAL,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO slot_fillers (
            slot_type, filler, mode, freq, pos_tag, popularity_score,
            popularity_level, popularity_confidence
        )
        VALUES (?, ?, 'strict', ?, ?, 1.0, 1, 1.0)
        """,
        [
            ("of_modifier", "C.E.ric", 10, "PROPN"),
            ("of_modifier", "Atlas", 1, "PROPN"),
            ("of_head", "E.G.riggs", 10, None),
            ("of_head", "Archive", 1, None),
        ],
    )
    conn.commit()
    return conn


def _remix_distribution_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for slot_type, display_filler in (
        ("of_modifier", "Atlas"),
        ("of_head", "Archive"),
    ):
        for tier in ("pop", "mainstream", "niche"):
            rows.append(
                {
                    "slot_type": slot_type,
                    "tier": tier,
                    "filler": display_filler.casefold(),
                    "display_filler": display_filler,
                    "probability": 1.0,
                    "log_probability": 0.0,
                    "soft_count": 1.0,
                    "prior_count": 0.0,
                    "evidence_count": 1.0,
                    "source_count": 1,
                    "anchored_source_count": 1,
                    "inferred_source_count": 0,
                    "anchored_soft_count": 1.0,
                    "inferred_soft_count": 0.0,
                    "teacher_confidence_mean": 1.0,
                    "frequency": 1,
                    "popularity_score": 1.0,
                    "semantic_smoothing_mass": 0.0,
                    "calibration_temperature": 1.0,
                    "artifact_version": "shadow-remix-v1",
                }
            )
    return rows


def _prepare_shadow_runtime(
    conn: sqlite3.Connection,
    artifact_path: Path,
    *,
    sampling_temperature: float = 1.0,
) -> PreparedGenerationRuntime:
    runtime = build_generation_runtime(
        mode="shadow",
        shadow_artifact=artifact_path,
        shadow_sampling_temperature=sampling_temperature,
    )
    return prepare_generation_runtime(conn, runtime)


def _install_distribution_table(
    conn: sqlite3.Connection,
    rows: list[dict[str, object]],
) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {TIER_SLOT_FILLER_DISTRIBUTION_TABLE}")
    conn.execute(
        f"""
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
        """
    )
    conn.executemany(
        f"INSERT INTO {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            tuple(
                row[column]
                for column in (
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
            )
            for row in rows
        ],
    )
    conn.commit()


def test_default_runtime_remains_equivalent_to_explicit_legacy(tmp_path):
    conn = _make_shadow_runtime_db()
    legacy_runtime = build_generation_runtime(mode="legacy")

    for seed in (7, 8, 9):
        implicit = generate_subtitle_matching_tiers(
            conn,
            allowed_tiers={"pop"},
            seed=seed,
            remix_prob=0.0,
            min_sim=0.0,
        )
        explicit = generate_subtitle_matching_tiers(
            conn,
            allowed_tiers={"pop"},
            seed=seed,
            remix_prob=0.0,
            min_sim=0.0,
            runtime=legacy_runtime,
        )
        assert implicit == explicit


def test_configured_artifact_is_default_and_legacy_is_persistent_rollback(
    tmp_path,
):
    conn = _make_shadow_runtime_db()
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    install_tier_slot_distribution(conn, artifact, activate=True)

    configured = prepare_generation_runtime(conn, None)
    assert configured.mode == RuntimeSelectionMode.ARTIFACT
    assert configured.shadow_distribution is not None

    implicit = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop"},
        seed=101,
        remix_prob=0.0,
        min_sim=0.0,
    )
    explicit_artifact = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop"},
        seed=101,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=build_generation_runtime(mode="artifact"),
    )
    assert implicit == explicit_artifact

    set_configured_runtime_mode(conn, "legacy")
    rolled_back = prepare_generation_runtime(conn, None)
    assert rolled_back.mode == RuntimeSelectionMode.LEGACY


def test_configured_runtime_falls_back_to_legacy_when_artifact_is_missing():
    conn = _make_shadow_runtime_db()
    conn.execute(
        "INSERT INTO config VALUES ('generation_runtime_mode', 'artifact')"
    )
    conn.commit()

    prepared = prepare_generation_runtime(conn, None)

    assert prepared.mode == RuntimeSelectionMode.LEGACY
    assert "missing" in prepared.fallback_reason
    assert shadow_runtime_provenance(prepared)["fallback_reason"]


def test_explicit_artifact_runtime_fails_when_table_is_missing():
    conn = _make_shadow_runtime_db()

    with pytest.raises(RuntimeError, match="missing required table"):
        prepare_generation_runtime(
            conn,
            build_generation_runtime(mode="artifact"),
        )


def test_csv_artifact_rejects_nonfinite_numeric_values(tmp_path):
    conn = _make_shadow_runtime_db()
    rows = _distribution_rows()
    rows[0]["probability"] = float("nan")
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        rows,
    )

    with pytest.raises(RuntimeError, match="probability.*finite"):
        prepare_generation_runtime(
            conn,
            build_generation_runtime(mode="artifact", shadow_artifact=artifact),
        )


def test_csv_artifact_allows_blank_optional_numeric_values(tmp_path):
    conn = _make_shadow_runtime_db()
    rows = _distribution_rows()
    rows[0]["teacher_confidence_mean"] = None
    rows[0]["popularity_score"] = None
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        rows,
    )

    prepared = prepare_generation_runtime(
        conn,
        build_generation_runtime(mode="artifact", shadow_artifact=artifact),
    )

    assert prepared.shadow_distribution is not None


def test_csv_artifact_allows_integer_valued_decimal_count_fields(tmp_path):
    conn = _make_shadow_runtime_db()
    rows = _distribution_rows()
    for field in (
        "source_count",
        "anchored_source_count",
        "inferred_source_count",
        "frequency",
    ):
        rows[0][field] = f"{rows[0][field]}.0"
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        rows,
    )

    prepared = prepare_generation_runtime(
        conn,
        build_generation_runtime(mode="artifact", shadow_artifact=artifact),
    )

    assert prepared.shadow_distribution is not None


def test_file_database_reuses_cached_validated_distribution(tmp_path):
    clear_distribution_cache()
    db_path = tmp_path / "runtime.db"
    first_conn = _make_shadow_runtime_db(db_path)
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    install_tier_slot_distribution(first_conn, artifact, activate=True)
    first = prepare_generation_runtime(first_conn, None)
    first_conn.close()

    second_conn = sqlite3.connect(db_path)
    try:
        second = prepare_generation_runtime(second_conn, None)
    finally:
        second_conn.close()

    assert second.shadow_distribution is first.shadow_distribution


def test_artifact_mode_requires_valid_installed_distribution(tmp_path):
    conn = _make_shadow_runtime_db()

    with pytest.raises(RuntimeError):
        set_configured_runtime_mode(conn, "artifact")

    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    row_count = install_tier_slot_distribution(conn, artifact, activate=False)
    assert row_count == len(_distribution_rows())
    assert set_configured_runtime_mode(conn, "artifact") == RuntimeSelectionMode.ARTIFACT


def test_shadow_runtime_generation_is_deterministic_for_fixed_seed(tmp_path):
    conn = _make_shadow_runtime_db()
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    runtime = build_generation_runtime(mode="shadow", shadow_artifact=artifact)

    first = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop"},
        seed=101,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=runtime,
    )
    second = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop"},
        seed=101,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=runtime,
    )

    assert first == second


@pytest.mark.parametrize(
    ("tier", "expected_items", "expected_action", "expected_object"),
    [
        ("pop", {"Race", "Power"}, {"Making"}, {"Modern Life"}),
        ("mainstream", {"History", "Identity"}, {"Pursuit"}, {"Daily Life"}),
        ("niche", {"Aesthesis", "Ontology"}, {"Problematizing"}, {"Late Antiquity"}),
    ],
)
def test_shadow_runtime_uses_requested_tier_slot_distribution(
    tmp_path,
    tier,
    expected_items,
    expected_action,
    expected_object,
):
    conn = _make_shadow_runtime_db()
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    runtime = build_generation_runtime(mode="shadow", shadow_artifact=artifact)

    subtitle = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={tier},
        seed=202,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=runtime,
    )

    assert {subtitle.item1, subtitle.item2} == expected_items
    assert {subtitle.action_noun} == expected_action
    assert {subtitle.of_object} == expected_object


def test_shadow_runtime_reports_missing_artifact_path(tmp_path):
    conn = _make_shadow_runtime_db()
    runtime = build_generation_runtime(
        mode="shadow",
        shadow_artifact=tmp_path / "missing.csv",
    )

    with pytest.raises(RuntimeError, match="does not exist"):
        generate_subtitle_matching_tiers(
            conn,
            allowed_tiers={"pop"},
            seed=1,
            remix_prob=0.0,
            min_sim=0.0,
            runtime=runtime,
        )


def test_shadow_runtime_reports_invalid_distribution_coverage(tmp_path):
    conn = _make_shadow_runtime_db()
    rows = [
        row
        for row in _distribution_rows()
        if not (row["slot_type"] == "of_object" and row["tier"] == "niche")
    ]
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        rows,
    )
    runtime = build_generation_runtime(mode="shadow", shadow_artifact=artifact)

    with pytest.raises(RuntimeError, match="distribution support must match"):
        generate_subtitle_matching_tiers(
            conn,
            allowed_tiers={"niche"},
            seed=1,
            remix_prob=0.0,
            min_sim=0.0,
            runtime=runtime,
        )


@pytest.mark.parametrize("source_kind", ["csv", "db"])
def test_shadow_runtime_fails_on_ineligible_high_probability_support_without_renormalizing(
    tmp_path,
    source_kind: str,
):
    conn = _make_shadow_runtime_db()
    conn.execute(
        """
        INSERT INTO slot_fillers (
            slot_type, filler, mode, source_subtitle_id, freq, popularity_score,
            popularity_level, popularity_confidence
        )
        VALUES ('list_item', 'Jr', 'strict', 99, 10, 1.0, 1, 1.0)
        """
    )
    conn.commit()
    rows = _distribution_rows()
    pop_list_rows = [
        row
        for row in rows
        if row["slot_type"] == "list_item" and row["tier"] == "pop"
    ]
    for row in pop_list_rows:
        if row["display_filler"] == "Race":
            row["probability"] = 0.02
        elif row["display_filler"] == "Power":
            row["probability"] = 0.01
        else:
            row["probability"] = 0.0
        row["log_probability"] = 0.0 if row["probability"] == 0.0 else math.log(row["probability"])
    rows.append(
        {
            "slot_type": "list_item",
            "tier": "pop",
            "filler": "jr",
            "display_filler": "Jr",
            "probability": 0.97,
            "log_probability": math.log(0.97),
            "soft_count": 1.0,
            "prior_count": 0.0,
            "evidence_count": 1.0,
            "source_count": 1,
            "anchored_source_count": 1,
            "inferred_source_count": 0,
            "anchored_soft_count": 1.0,
            "inferred_soft_count": 0.0,
            "teacher_confidence_mean": 1.0,
            "frequency": 10,
            "popularity_score": 1.0,
            "semantic_smoothing_mass": 0.0,
            "calibration_temperature": 0.73,
            "artifact_version": "shadow-test-v1",
        }
    )
    if source_kind == "csv":
        artifact = _write_shadow_artifact(tmp_path / "invalid.csv", rows)
        runtime = build_generation_runtime(mode="shadow", shadow_artifact=artifact)
    else:
        _install_distribution_table(conn, rows)
        runtime = build_generation_runtime(mode="shadow")

    with pytest.raises(RuntimeError, match="ineligible fillers"):
        prepare_generation_runtime(conn, runtime)


def test_shadow_runtime_sampling_temperature_is_explicit_and_identity_by_default(
    tmp_path,
):
    conn = _make_shadow_runtime_db()
    rows = _distribution_rows()
    for row in rows:
        if row["slot_type"] == "action_noun" and row["tier"] == "pop":
            row["probability"] = {
                "Making": 0.6,
                "Pursuit": 0.3,
                "Problematizing": 0.1,
            }[row["display_filler"]]
            row["log_probability"] = math.log(row["probability"])
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        rows,
    )
    runtime_identity = _prepare_shadow_runtime(conn, artifact, sampling_temperature=1.0)
    runtime_sharp = _prepare_shadow_runtime(conn, artifact, sampling_temperature=0.5)
    runtime_flat = _prepare_shadow_runtime(conn, artifact, sampling_temperature=2.0)
    candidate_rows = conn.execute(
        """
        SELECT filler, freq
        FROM slot_fillers
        WHERE slot_type = 'action_noun'
        ORDER BY filler
        """
    ).fetchall()

    def sample_counts(runtime):
        counts = {"Making": 0, "Pursuit": 0, "Problematizing": 0}
        for seed in range(200):
            pick = sample_shadow_candidates(
                runtime,
                slot_type="action_noun",
                tier="pop",
                candidate_rows=candidate_rows,
                count=1,
                rng=random.Random(seed),
            )[0]
            counts[pick] += 1
        return counts

    identity_counts = sample_counts(runtime_identity)
    sharp_counts = sample_counts(runtime_sharp)
    flat_counts = sample_counts(runtime_flat)

    assert sharp_counts["Making"] > identity_counts["Making"] > flat_counts["Making"]


def test_shadow_runtime_does_not_reapply_calibration_temperature(tmp_path):
    conn = _make_shadow_runtime_db()
    base_artifact = _write_shadow_artifact(
        tmp_path / "base.csv",
        _distribution_rows(calibration_temperature=0.61),
    )
    shifted_artifact = _write_shadow_artifact(
        tmp_path / "shifted.csv",
        _distribution_rows(calibration_temperature=1.87),
    )
    base_runtime = build_generation_runtime(mode="shadow", shadow_artifact=base_artifact)
    shifted_runtime = build_generation_runtime(
        mode="shadow",
        shadow_artifact=shifted_artifact,
    )

    base = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"mainstream"},
        seed=303,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=base_runtime,
    )
    shifted = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"mainstream"},
        seed=303,
        remix_prob=0.0,
        min_sim=0.0,
        runtime=shifted_runtime,
    )

    assert base == shifted


def test_legacy_remix_runtime_keeps_prior_candidate_semantics():
    conn = _make_legacy_remix_db()

    composed = compose_compound(
        conn,
        random.Random(7),
        {"config": {"remix_mod_pos_2word": {"PROPN": 1.0}}},
        word_count=2,
    )

    assert composed == (
        "C.E.ric E.G.riggs",
        {"modifier": "C.E.ric", "head": "E.G.riggs"},
    )


def test_shadow_remix_runtime_filters_to_artifact_eligible_support(tmp_path):
    conn = _make_shadow_remix_db()
    artifact = _write_shadow_artifact(tmp_path / "remix-shadow.csv", _remix_distribution_rows())
    runtime = _prepare_shadow_runtime(conn, artifact)

    composed = compose_compound(
        conn,
        random.Random(7),
        {"config": {"remix_mod_pos_2word": {"PROPN": 1.0}}},
        word_count=2,
        model_tier="pop",
        runtime=runtime,
    )

    assert composed == ("Atlas Archive", {"modifier": "Atlas", "head": "Archive"})


def test_shadow_runtime_comparison_outputs_replayable_provenance(tmp_path):
    conn = _make_shadow_runtime_db()
    artifact = _write_shadow_artifact(
        tmp_path / "tier_slot_filler_distribution_v1.csv",
        _distribution_rows(),
    )
    runtime = build_generation_runtime(mode="shadow", shadow_artifact=artifact)

    result = build_shadow_runtime_comparison(
        conn,
        tmp_path / "report",
        shadow_runtime=runtime,
        seeds=(11, 12),
        remix_prob=0.0,
        min_sim=0.0,
    )

    details = json.loads(result.details_path.read_text(encoding="utf-8"))
    assert result.report_path.exists()
    assert details["shadow_runtime"]["artifact_source"] == str(artifact.resolve())
    assert details["policy"]["seeds"] == [11, 12]
    assert details["policy"]["scenarios"] == [
        "pop",
        "mainstream",
        "niche",
        "default",
    ]

    replay = details["comparisons"][0]
    allowed_tiers = (
        set(replay["requested_tiers"]) if replay["requested_tiers"] else None
    )
    reproduced = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers=allowed_tiers,
        seed=replay["seed"],
        remix_prob=0.0,
        min_sim=0.0,
        runtime=runtime,
    )
    assert reproduced.text == replay["shadow"]["text"]


def test_export_and_build_mini_db_include_shadow_distribution(tmp_path):
    conn = _make_shadow_runtime_db(tmp_path / "runtime.db")
    rows = _distribution_rows()
    conn.execute(
        f"""
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
        """
    )
    conn.executemany(
        f"INSERT INTO {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            tuple(row[column] for column in (
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
            ))
            for row in rows
        ],
    )
    conn.commit()

    data_dir = tmp_path / "data"
    stats = export_data(conn, data_dir)
    mini_db_path = tmp_path / "mini.db"
    build_stats = build_mini_db(data_dir, mini_db_path)
    mini = sqlite3.connect(mini_db_path)
    try:
        shadow_count = mini.execute(
            f"SELECT COUNT(*) FROM {TIER_SLOT_FILLER_DISTRIBUTION_TABLE}"
        ).fetchone()[0]
    finally:
        mini.close()

    assert stats[f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv"] == len(rows)
    assert build_stats[TIER_SLOT_FILLER_DISTRIBUTION_TABLE] == len(rows)
    assert shadow_count == len(rows)


def test_build_mini_db_preserves_existing_output_on_invalid_shadow_import(tmp_path):
    conn = _make_shadow_runtime_db(tmp_path / "runtime.db")
    data_dir = tmp_path / "data"
    export_data(conn, data_dir)

    invalid_rows = [
        row
        for row in _distribution_rows()
        if not (row["slot_type"] == "of_object" and row["tier"] == "niche")
    ]
    _write_shadow_artifact(
        data_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv",
        invalid_rows,
    )
    mini_db_path = tmp_path / "mini.db"
    original_bytes = b"keep-this-mini-db"
    mini_db_path.write_bytes(original_bytes)

    with pytest.raises(RuntimeError, match="distribution support must match"):
        build_mini_db(data_dir, mini_db_path)

    assert mini_db_path.read_bytes() == original_bytes
    assert list(tmp_path.glob("mini-*.tmp.db")) == []


def test_build_mini_db_replaces_existing_output_atomically_on_success(tmp_path):
    conn = _make_shadow_runtime_db(tmp_path / "runtime.db")
    data_dir = tmp_path / "data"
    export_data(conn, data_dir)
    shadow_rows = _distribution_rows()
    _write_shadow_artifact(
        data_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv",
        shadow_rows,
    )
    mini_db_path = tmp_path / "mini.db"
    mini_db_path.write_bytes(b"stale-mini-db")

    build_stats = build_mini_db(data_dir, mini_db_path)
    rebuilt = sqlite3.connect(mini_db_path)
    try:
        shadow_count = rebuilt.execute(
            f"SELECT COUNT(*) FROM {TIER_SLOT_FILLER_DISTRIBUTION_TABLE}"
        ).fetchone()[0]
    finally:
        rebuilt.close()

    assert mini_db_path.read_bytes() != b"stale-mini-db"
    assert build_stats[TIER_SLOT_FILLER_DISTRIBUTION_TABLE] == len(shadow_rows)
    assert shadow_count == len(shadow_rows)


def test_build_mini_db_rejects_artifact_default_without_distribution(tmp_path):
    conn = _make_shadow_runtime_db(tmp_path / "runtime.db")
    conn.execute(
        "INSERT INTO config VALUES ('generation_runtime_mode', 'artifact')"
    )
    conn.commit()
    data_dir = tmp_path / "data"
    export_data(
        conn,
        data_dir,
        shadow_distribution_source=_write_shadow_artifact(
            tmp_path / "artifact.csv",
            _distribution_rows(),
        ),
    )
    (data_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv").unlink()

    with pytest.raises(RuntimeError, match="generation_runtime_mode"):
        build_mini_db(data_dir, tmp_path / "mini.db")
    assert list(tmp_path.glob("mini-*.tmp.db")) == []
