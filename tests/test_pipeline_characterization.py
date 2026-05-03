"""Characterization coverage for the current pipeline contracts.

These tests freeze observable behavior and state for the first redesign pass.
They intentionally avoid changing production code or asserting unseeded random
subtitle text byte-for-byte.
"""

from __future__ import annotations

import importlib.util
import io
import json
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent


EXPECTED_TUNABLE_PARAMS = {
    "weighted_sample_spread": 0.12,
    "weighted_sample_bias_floor": 0.05,
    "default_generation_tone_target": 2.0,
    "generation_tier_ratio_pop": 0.0183,
    "generation_tier_ratio_mainstream": 0.1172,
    "generation_tier_ratio_niche": 0.8645,
    "tier_center_pop": 0.75,
    "tier_center_mainstream": 0.4005,
    "tier_center_niche": 0.301,
    "accessibility_threshold_pop": 0.3665,
    "accessibility_threshold_mainstream": 0.3098,
    "article_of_min_freq": 1.0,
    "article_action_min_freq": 1.0,
    "article_remix_heuristic_threshold": 0.6,
    "remix_reject_double_of": 1.0,
    "pop_weight_spl": 0.7,
    "pop_weight_ol": 0.3,
    "pop_weight_gr": 0.2,
    "pop_weight_nyt": 0.1,
    "pop_weight_library": 0.05,
    "pop_exponent": 1.2,
    "pop_base_weight_blend": 0.5,
    "pop_classification_blend": 0.9,
    "pop_missing_default": 0.1,
    "tier_pop_min_demand_confidence": 0.8001,
    "tier_pop_min_lower_tail": 0.352,
    "pop_slot_mult_list_item": 0.8,
    "pop_slot_mult_action_noun": 0.9,
    "pop_slot_mult_of_object": 1.0,
}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _insert_runtime_config(conn: sqlite3.Connection) -> None:
    values = {
        "embedding_version": "2",
        "centroid_norm": "1.0",
        "avg_cross_sim_t1": "0.1",
        "avg_cross_sim_t2": "0.1",
        "article_stats_action_noun": json.dumps(
            {"making": {"the": 5}, "pursuit": {"the": 5}}
        ),
        "article_stats_of_object": json.dumps(
            {"modern life": {"": 5}, "happiness": {"": 5}, "empire": {"the": 5}}
        ),
    }
    conn.executemany("INSERT OR REPLACE INTO config VALUES (?, ?)", values.items())


def _load_populate_popularity_module():
    pop_spec = importlib.util.spec_from_file_location(
        "populate_popularity", ROOT / "data" / "populate_popularity.py"
    )
    assert pop_spec is not None and pop_spec.loader is not None
    populate_popularity = importlib.util.module_from_spec(pop_spec)
    sys.modules[pop_spec.name] = populate_popularity
    pop_spec.loader.exec_module(populate_popularity)
    return populate_popularity


def make_runtime_db(path: Path | None = None) -> sqlite3.Connection:
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
    _insert_runtime_config(conn)
    rows = [
        ("list_item", "race", 10, 1.0),
        ("list_item", "power", 10, 1.0),
        ("list_item", "history", 10, 1.0),
        ("action_noun", "making", 10, 1.0),
        ("action_noun", "pursuit", 10, 1.0),
        ("of_object", "modern life", 10, 1.0),
        ("of_object", "happiness", 10, 1.0),
        ("of_object", "empire", 10, 1.0),
    ]
    conn.executemany(
        """
        INSERT INTO slot_fillers (
            slot_type, filler, freq, popularity_score,
            popularity_level, popularity_confidence
        )
        VALUES (?, ?, ?, ?, 1, 1.0)
        """,
        rows,
    )
    conn.commit()
    return conn


def test_observed_pipeline_schema_columns(tmp_path):
    from subtitle_generator.extract import get_db
    from subtitle_generator.extract_openlibrary import ensure_isbn_column
    from subtitle_generator.feedback import ensure_ratings_table
    from subtitle_generator.schema_contracts import validate_schema
    from subtitle_generator.slots import ensure_slot_tables

    populate_popularity = _load_populate_popularity_module()

    conn = get_db(tmp_path / "subtitles.db")
    ensure_isbn_column(conn)
    ensure_slot_tables(conn)
    populate_popularity.create_tables(conn)
    ensure_ratings_table(conn)

    assert _columns(conn, "subtitles") >= {
        "id", "title", "subtitle", "lang", "lccn", "source_file", "isbn",
    }
    assert _columns(conn, "pattern_matches") >= {
        "id", "subtitle_id", "title", "subtitle", "list_items_json",
        "action_noun", "of_object", "of_article", "action_article",
    }
    assert _columns(conn, "slot_fillers") >= {
        "id", "slot_type", "filler", "mode", "source_subtitle_id", "freq",
        "pos_tag", "prep", "remix_type", "remix_prep", "remix_word_count",
        "vector_sum", "token_count", "centroid_dot", "norm_sq",
        "popularity_score", "popularity_level", "popularity_confidence",
    }
    assert _columns(conn, "popularity_data") >= {
        "work_key", "spl_checkouts", "spl_years", "spl_earliest_pub_year",
        "ol_edition_count", "checkouts_per_year", "editions_per_decade",
        "gr_ratings_count", "gr_average_rating", "nyt_weeks_on_list",
        "nyt_peak_rank", "library_appearances", "composite_score",
    }
    assert _columns(conn, "config") == {"key", "value"}
    assert _columns(conn, "human_ratings") >= {
        "id", "subtitle", "system_tone", "thumbs", "tone_override",
        "free_text", "interpreted", "config_snapshot", "created_at",
        "tags", "source",
    }
    assert validate_schema(conn) == []


def test_work_level_popularity_scoring_is_testable_without_db_writes():
    from subtitle_generator.parameter_state import PopularityParameters

    pop = _load_populate_popularity_module()
    data = pop.WorkLevelData(
        work_spl={
            "work-a": {"checkouts": 100, "years": 10, "pub_year": "2000"},
            "work-c": {"checkouts": 1, "years": 1, "pub_year": "2001"},
        },
        work_ol={"work-a": 5, "work-b": 20, "work-c": 1},
        work_gr={
            "work-a": {"ratings_count": 50, "average_rating": 4.2},
            "work-c": {"ratings_count": 1, "average_rating": 3.1},
        },
        work_ottawa={},
        work_nyt={"work-a": {"weeks_on_list": 4, "peak_rank": 3}},
        all_works={"work-a", "work-b", "work-c"},
    )
    percentiles = pop.build_percentile_models(data)
    params = PopularityParameters(
        weight_spl=0.7,
        weight_ol=0.3,
        weight_goodreads=0.2,
        weight_nyt=0.1,
        weight_library=0.05,
        exponent=1.2,
    )

    demand_row = pop.score_work_popularity("work-a", data, percentiles, params)
    ol_only_row = pop.score_work_popularity("work-b", data, percentiles, params)

    assert demand_row.work_key == "work-a"
    assert demand_row.spl_checkouts == 100
    assert demand_row.spl_years == 10
    assert demand_row.gr_ratings_count == 50
    assert demand_row.nyt_weeks_on_list == 4
    assert demand_row.composite_score > ol_only_row.composite_score
    assert ol_only_row.composite_score <= 0.5


def test_filler_scoring_workers_cover_top3_mean_and_fallback():
    pop = _load_populate_popularity_module()
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL,
            popularity_level INTEGER,
            popularity_confidence REAL
        );
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL
        );
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            isbn TEXT
        );
        CREATE TABLE isbn_aliases (
            isbn TEXT PRIMARY KEY,
            work_key TEXT
        );
        CREATE TABLE popularity_data (
            work_key TEXT PRIMARY KEY,
            composite_score REAL
        );
        INSERT INTO slot_fillers (id, slot_type, filler, freq)
        VALUES
            (1, 'list_item', 'future', 100),
            (2, 'list_item', 'fallback', 99);
        INSERT INTO subtitles (id, isbn)
        VALUES (1, 'a'), (2, 'b'), (3, 'c'), (4, 'd');
        INSERT INTO slot_filler_sources VALUES
            (1, 1), (1, 2), (1, 3), (1, 4);
        INSERT INTO isbn_aliases VALUES
            ('a', 'work-a'), ('b', 'work-b'), ('c', 'work-c'), ('d', 'work-d');
        INSERT INTO popularity_data VALUES
            ('work-a', 0.9), ('work-b', 0.7), ('work-c', 0.6), ('work-d', 0.1);
        """
    )

    assert pop.update_level1_filler_scores(conn) == 1
    top3_score = conn.execute(
        "SELECT popularity_score FROM slot_fillers WHERE id = 1"
    ).fetchone()[0]
    assert abs(top3_score - ((0.9 + 0.7 + 0.6) / 3)) < 0.0001

    assert pop.update_fallback_filler_scores(conn) == 1
    fallback = conn.execute(
        """
        SELECT popularity_score, popularity_level, popularity_confidence
        FROM slot_fillers
        WHERE id = 2
        """
    ).fetchone()
    assert abs(fallback[0] - 2.0) < 0.0001
    assert fallback[1:] == (0, 0.0)


def test_threshold_calibration_workers_cover_percentile_cutoffs():
    pop = _load_populate_popularity_module()

    rows = [(9, i / 100) for i in range(100)]
    scores = pop.compute_classification_scores(rows, blend=1.0, pop_default=0.1)
    calibration = pop.calibrate_threshold_values(scores)
    params = calibration.as_config_values()

    assert calibration.pop_threshold == 0.92
    assert calibration.mainstream_threshold == 0.64
    assert calibration.pop_count == 8
    assert calibration.mainstream_count == 28
    assert calibration.niche_count == 64
    assert params["accessibility_threshold_pop"] == 0.92
    assert params["tier_center_mainstream"] == 0.78


def test_threshold_config_write_removes_legacy_tone_targets():
    pop = _load_populate_popularity_module()

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('tone_target_pop_list_item', '0.77')")
    conn.execute("INSERT INTO config VALUES ('tier_center_pop', '0.7')")
    conn.commit()

    pop.write_threshold_config(conn, {"tier_center_pop": 0.8})

    rows = dict(conn.execute("SELECT key, value FROM config"))
    assert rows == {"tier_center_pop": "0.8"}


def test_schema_contract_validator_reports_stage_context(tmp_path):
    from subtitle_generator.extract import get_db
    from subtitle_generator.extract_openlibrary import ensure_isbn_column
    from subtitle_generator.feedback import ensure_ratings_table
    from subtitle_generator.schema_contracts import (
        REQUIRED_TABLES_BY_STAGE,
        assert_schema_valid,
        validate_schema,
    )
    from subtitle_generator.slots import ensure_slot_tables

    assert REQUIRED_TABLES_BY_STAGE["source_ingestion"] == ("subtitles",)
    assert REQUIRED_TABLES_BY_STAGE["model_weight_state"] == ("config",)

    conn = get_db(tmp_path / "partial.db")
    ensure_isbn_column(conn)
    ensure_slot_tables(conn)
    ensure_ratings_table(conn)
    issues = validate_schema(conn)

    assert any(
        issue.stage == "popularity_scoring"
        and issue.table == "popularity_data"
        and issue.column is None
        for issue in issues
    )

    try:
        assert_schema_valid(conn)
    except RuntimeError as exc:
        assert "popularity_scoring" in str(exc)
        assert "popularity_data" in str(exc)
    else:
        raise AssertionError("partial schema should not validate")


def test_current_model_ids_and_tunable_defaults_are_characterized():
    from subtitle_generator.config import ALL_TUNABLE_PARAMS
    from subtitle_generator.jacket import DEFAULT_MODEL
    from subtitle_generator.parameter_state import get_model_registry

    assert ALL_TUNABLE_PARAMS == EXPECTED_TUNABLE_PARAMS
    assert DEFAULT_MODEL == "gpt-5.4-mini"
    assert get_model_registry().rater == "github_copilot/gpt-5.4-mini"
    assert get_model_registry().proposer == "github_copilot/gpt-5.4"
    assert get_model_registry().jacket == "gpt-5.4-mini"
    assert get_model_registry().responses_only == {
        "gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano",
    }


def test_parameter_views_preserve_defaults_and_db_overrides():
    from subtitle_generator.parameter_state import (
        get_popularity_blend_parameters,
        get_popularity_parameters,
        get_generation_tier_ratios,
        get_runtime_generation_parameters,
        get_slot_multiplier_parameters,
        get_tier_classifier_parameters,
        get_tier_threshold_parameters,
        get_tone_targets,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('pop_weight_spl', '0.9')")
    conn.execute("INSERT INTO config VALUES ('pop_classification_blend', '0.25')")
    conn.execute("INSERT INTO config VALUES ('pop_slot_mult_of_object', '1.4')")
    conn.commit()

    popularity = get_popularity_parameters(conn)
    blends = get_popularity_blend_parameters(conn)
    runtime = get_runtime_generation_parameters(conn)

    assert popularity.weight_spl == 0.9
    assert popularity.weight_ol == EXPECTED_TUNABLE_PARAMS["pop_weight_ol"]
    assert blends.classification_blend == 0.25
    assert get_slot_multiplier_parameters(conn).of_object == 1.4
    assert get_generation_tier_ratios(conn).pop == 0.0183
    assert runtime.popularity_blends == blends
    assert runtime.generation_tier_ratios.mainstream == 0.1172
    assert runtime.slot_multipliers.of_object == 1.4
    assert get_tier_threshold_parameters(conn).accessibility_pop == 0.3665
    assert get_tier_classifier_parameters(conn).pop_min_demand_confidence == 0.8001
    assert get_tone_targets(conn).pop["list_item"] == 0.75
    assert get_tone_targets(conn).mainstream["of_object"] == 0.4005


def test_seeded_generation_path_is_stable():
    import subtitle_generator.generate as generate_module
    from subtitle_generator.generate import generate_subtitle
    from subtitle_generator.remix_state import RemixRuntimeContext

    conn = make_runtime_db()
    generate_module._remix_ctx = None
    first = generate_subtitle(conn, seed=11, remix_prob=0.0)

    generate_module._remix_ctx = None
    second = generate_subtitle(conn, seed=11, remix_prob=0.0)

    assert first == second
    assert first.text == "Power, Race, and the Pursuit of Happiness"
    assert first.item1 == "Power"
    assert first.item2 == "Race"
    assert first.action_noun == "Pursuit"
    assert first.of_object == "Happiness"
    assert first.remixed is False
    assert first.remix_parts == {}
    assert first.remix_similarity is None
    assert first.action_article == "the"
    assert first.of_article == ""
    assert isinstance(generate_module._remix_ctx, RemixRuntimeContext)
    assert generate_module._remix_ctx.precomputed is True


def test_default_generation_uses_configured_tone_target():
    from subtitle_generator.generate import _adjust_tone_targets

    conn = make_runtime_db()

    assert _adjust_tone_targets(conn, None) == {
        "list_item": 1.6,
        "action_noun": 1.8,
        "of_object": 2.0,
    }


def test_default_generation_tier_choice_uses_configured_ratios():
    from subtitle_generator.generate import (
        _choose_default_generation_tier,
        _default_generation_tier_ratios,
    )

    conn = make_runtime_db()
    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        (
            ("generation_tier_ratio_pop", "0"),
            ("generation_tier_ratio_mainstream", "0"),
            ("generation_tier_ratio_niche", "1"),
        ),
    )
    conn.commit()

    assert _default_generation_tier_ratios(conn) == {
        "pop": 0.0,
        "mainstream": 0.0,
        "niche": 1.0,
    }
    assert _choose_default_generation_tier(conn, seed=1) == "niche"


def test_explicit_multi_tier_choice_renormalizes_over_selected_tiers():
    from subtitle_generator.generate import _choose_generation_tier

    conn = make_runtime_db()
    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        (
            ("generation_tier_ratio_pop", "1"),
            ("generation_tier_ratio_mainstream", "0"),
            ("generation_tier_ratio_niche", "999"),
        ),
    )
    conn.commit()

    assert _choose_generation_tier(
        conn,
        allowed_tiers={"pop", "mainstream"},
        seed=1,
    ) == "pop"


def test_explicit_multi_tier_generation_targets_one_sampled_tier(monkeypatch):
    import subtitle_generator.generate as generate_module
    import subtitle_generator.tiering as tiering_module
    from subtitle_generator.generate import (
        GeneratedSubtitle,
        generate_subtitle_matching_tiers,
    )

    conn = make_runtime_db()
    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        (
            ("generation_tier_ratio_pop", "1"),
            ("generation_tier_ratio_mainstream", "0"),
            ("generation_tier_ratio_niche", "999"),
        ),
    )
    conn.commit()
    observed_targets: list[dict[str, float]] = []

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        observed_targets.append(kwargs["adjusted_tone_target"])
        return GeneratedSubtitle(
            text="Generated pop",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn):
        return SimpleNamespace(tier="pop")

    monkeypatch.setattr(generate_module, "_load_generation_candidates", lambda conn: object())
    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates", fake_generate_from_candidates,
    )
    monkeypatch.setattr(tiering_module, "compute_tier_evidence", fake_compute_tier_evidence)

    sub = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop", "mainstream"},
        seed=11,
        max_attempts=3,
    )

    assert sub.text == "Generated pop"
    assert observed_targets == [pytest.approx({
        "list_item": 0.6,
        "action_noun": 0.675,
        "of_object": 0.75,
    })]


def test_tier_filtered_generation_retries_until_classifier_match(monkeypatch):
    import subtitle_generator.generate as generate_module
    import subtitle_generator.tiering as tiering_module
    from subtitle_generator.generate import (
        GeneratedSubtitle,
        TierFilterError,
        generate_subtitle_matching_tiers,
    )

    conn = make_runtime_db()
    observed_seeds: list[int] = []
    loaded_candidates: list[object] = []

    def fake_load_generation_candidates(conn):
        candidates = object()
        loaded_candidates.append(candidates)
        return candidates

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        assert candidates is loaded_candidates[-1]
        observed_seeds.append(kwargs["seed"])
        return GeneratedSubtitle(
            text=f"Generated {len(observed_seeds)}",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn):
        tier = "pop" if subtitle == "Generated 2" else "mainstream"
        return SimpleNamespace(tier=tier)

    monkeypatch.setattr(
        generate_module, "_load_generation_candidates", fake_load_generation_candidates,
    )
    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates",
        fake_generate_from_candidates,
    )
    monkeypatch.setattr(tiering_module, "compute_tier_evidence", fake_compute_tier_evidence)

    sub = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"pop"},
        seed=11,
        max_attempts=3,
    )

    assert sub.text == "Generated 2"
    assert observed_seeds == [11, 12]
    assert len(loaded_candidates) == 1

    with pytest.raises(TierFilterError, match="observed tiers: mainstream=3"):
        generate_subtitle_matching_tiers(
            conn,
            allowed_tiers={"niche"},
            seed=21,
            max_attempts=3,
        )
    assert len(loaded_candidates) == 2


def test_default_generation_tries_remaining_tiers_when_sampled_tier_is_unavailable(monkeypatch):
    import subtitle_generator.generate as generate_module
    import subtitle_generator.tiering as tiering_module
    from subtitle_generator.generate import (
        GeneratedSubtitle,
        generate_subtitle_matching_tiers,
    )

    conn = make_runtime_db()
    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        (
            ("generation_tier_ratio_pop", "1"),
            ("generation_tier_ratio_mainstream", "0"),
            ("generation_tier_ratio_niche", "0"),
        ),
    )
    conn.commit()
    generated: list[str] = []

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        generated.append("filtered")
        return GeneratedSubtitle(
            text=f"Filtered {len(generated)}",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn):
        return SimpleNamespace(tier="mainstream")

    def fake_generate_subtitles(conn, **kwargs):
        return [GeneratedSubtitle(
            text="Unfiltered fallback",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )]

    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates", fake_generate_from_candidates,
    )
    monkeypatch.setattr(tiering_module, "compute_tier_evidence", fake_compute_tier_evidence)
    monkeypatch.setattr(generate_module, "generate_subtitles", fake_generate_subtitles)

    sub = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers=None,
        seed=7,
        max_attempts=4,
    )

    assert sub.text == "Filtered 2"
    assert generated == ["filtered", "filtered"]


def test_default_generation_falls_back_after_all_tier_attempts_fail(monkeypatch):
    import subtitle_generator.generate as generate_module
    import subtitle_generator.tiering as tiering_module
    from subtitle_generator.generate import (
        DEFAULT_GENERATION_TIER_ATTEMPTS,
        GeneratedSubtitle,
        generate_subtitle_matching_tiers,
    )

    conn = make_runtime_db()
    generated: list[str] = []

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        generated.append("filtered")
        return GeneratedSubtitle(
            text=f"Filtered {len(generated)}",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn):
        return SimpleNamespace(tier="unreachable")

    def fake_generate_subtitles(conn, **kwargs):
        return [GeneratedSubtitle(
            text="Unfiltered fallback",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )]

    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates", fake_generate_from_candidates,
    )
    monkeypatch.setattr(tiering_module, "compute_tier_evidence", fake_compute_tier_evidence)
    monkeypatch.setattr(generate_module, "generate_subtitles", fake_generate_subtitles)

    sub = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers=None,
        seed=7,
    )

    assert sub.text == "Unfiltered fallback"
    assert len(generated) == DEFAULT_GENERATION_TIER_ATTEMPTS


def test_batch_generation_reuses_one_candidate_pool(monkeypatch):
    import subtitle_generator.generate as generate_module
    from subtitle_generator.generate import GeneratedSubtitle, generate_subtitles

    conn = make_runtime_db()
    observed_seeds: list[int] = []
    loaded_candidates: list[object] = []

    def fake_load_generation_candidates(conn):
        candidates = object()
        loaded_candidates.append(candidates)
        return candidates

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        assert candidates is loaded_candidates[-1]
        observed_seeds.append(kwargs["seed"])
        return GeneratedSubtitle(
            text=f"Generated {len(observed_seeds)}",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    monkeypatch.setattr(
        generate_module, "_load_generation_candidates", fake_load_generation_candidates,
    )
    monkeypatch.setattr(generate_module, "_has_enough_candidates", lambda candidates: True)
    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates",
        fake_generate_from_candidates,
    )

    subtitles = generate_subtitles(conn, n=3, seed_base=50)

    assert [sub.text for sub in subtitles] == [
        "Generated 1",
        "Generated 2",
        "Generated 3",
    ]
    assert observed_seeds == [50, 51, 52]
    assert len(loaded_candidates) == 1


def test_cli_spot_check_uses_raw_tone_targets_not_classifier_filter(monkeypatch):
    import subtitle_generator.tune as tune_module
    from subtitle_generator.generate import GeneratedSubtitle

    conn = make_runtime_db()
    requested_batches: list[tuple[int, dict[str, float]]] = []
    captured_samples: list[tuple[str, str, object]] = []

    def fake_generate_subtitles(conn, *, n, seed_base, tone_target, **kwargs):
        requested_batches.append((seed_base, tone_target))
        return [
            GeneratedSubtitle(
                text=f"Raw {seed_base + i}",
                item1="Race",
                item2="Power",
                action_noun="Pursuit",
                of_object="Happiness",
            )
            for i in range(n)
        ]

    def fake_spot_check_cli(conn, samples, tier_labels, tier_shortcuts, source):
        captured_samples.extend(samples)
        return 1.0

    monkeypatch.setattr(
        "subtitle_generator.generate.generate_subtitles", fake_generate_subtitles,
    )
    monkeypatch.setattr(tune_module, "_spot_check_cli", fake_spot_check_cli)

    accuracy = tune_module.run_spot_check(conn, n_samples=2, seed_base=10)

    assert accuracy == 1.0
    assert [seed for seed, _ in requested_batches] == [10, 110, 210]
    assert [target["list_item"] for _, target in requested_batches] == [
        0.75, 0.4005, 0.301,
    ]
    assert [tier for tier, _, _ in captured_samples] == [
        "pop",
        "pop",
        "mainstream",
        "mainstream",
        "niche",
        "niche",
    ]


def test_tier_batch_generation_fast_exits_when_candidates_missing(monkeypatch):
    import subtitle_generator.generate as generate_module
    from subtitle_generator.generate import GenerationCandidates, generate_subtitles_by_tier

    conn = make_runtime_db()
    compute_calls = 0

    def fake_compute_tier_evidence(subtitle, conn):
        nonlocal compute_calls
        compute_calls += 1
        return SimpleNamespace(tier="mainstream")

    monkeypatch.setattr(
        generate_module,
        "_load_generation_candidates",
        lambda conn: GenerationCandidates([], [], []),
    )
    monkeypatch.setattr(
        "subtitle_generator.tiering.compute_tier_evidence",
        fake_compute_tier_evidence,
    )

    by_tier = generate_subtitles_by_tier(
        conn,
        tiers=["pop", "mainstream", "niche"],
        samples_per_tier=2,
        seed=100,
        max_attempts=3,
    )

    assert {
        tier: [sub.text for sub in subtitles]
        for tier, subtitles in by_tier.items()
    } == {
        "pop": ["(not enough fillers — run 'build-slots' first)"] * 2,
        "mainstream": ["(not enough fillers — run 'build-slots' first)"] * 2,
        "niche": ["(not enough fillers — run 'build-slots' first)"] * 2,
    }
    assert compute_calls == 0


def test_spot_check_tier_generation_uses_shared_candidate_pool(monkeypatch):
    import subtitle_generator.generate as generate_module
    import subtitle_generator.tiering as tiering_module
    from subtitle_generator.generate import (
        GeneratedSubtitle,
        GenerationCandidates,
        generate_subtitles_by_tier,
    )

    conn = make_runtime_db()
    observed_seeds: list[int] = []
    loaded_candidates: list[object] = []
    generated_tiers = ["mainstream", "pop", "niche", "pop", "niche", "mainstream"]

    def fake_load_generation_candidates(conn):
        candidates = GenerationCandidates([(), ()], [()], [()])
        loaded_candidates.append(candidates)
        return candidates

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        assert candidates is loaded_candidates[-1]
        observed_seeds.append(kwargs["seed"])
        index = len(observed_seeds)
        return GeneratedSubtitle(
            text=f"Candidate {index}",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn):
        index = int(subtitle.removeprefix("Candidate ")) - 1
        return SimpleNamespace(tier=generated_tiers[index])

    monkeypatch.setattr(
        generate_module, "_load_generation_candidates", fake_load_generation_candidates,
    )
    monkeypatch.setattr(
        generate_module, "_generate_subtitle_from_candidates",
        fake_generate_from_candidates,
    )
    monkeypatch.setattr(tiering_module, "compute_tier_evidence", fake_compute_tier_evidence)

    by_tier = generate_subtitles_by_tier(
        conn,
        tiers=["pop", "mainstream", "niche"],
        samples_per_tier=2,
        seed=100,
        max_attempts=6,
    )

    assert [sub.text for sub in by_tier["pop"]] == ["Candidate 2", "Candidate 4"]
    assert [sub.text for sub in by_tier["mainstream"]] == [
        "Candidate 1",
        "Candidate 6",
    ]
    assert [sub.text for sub in by_tier["niche"]] == ["Candidate 3", "Candidate 5"]
    assert observed_seeds == [100, 101, 102, 103, 104, 105]
    assert len(loaded_candidates) == 1


def test_remix_precompute_validator_checks_version_and_columns():
    import subtitle_generator.generate as generate_module
    from subtitle_generator.generate import generate_subtitle
    from subtitle_generator.remix_state import validate_remix_precompute_state

    conn = make_runtime_db()
    assert validate_remix_precompute_state(conn, expected_embedding_version="2") == []

    conn.execute("UPDATE config SET value = '1' WHERE key = 'embedding_version'")
    version_issues = validate_remix_precompute_state(conn, expected_embedding_version="2")
    assert any(issue.field == "embedding_version" for issue in version_issues)

    broken = sqlite3.connect(":memory:")
    broken.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    _insert_runtime_config(broken)
    broken.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL
        )
        """
    )
    column_issues = validate_remix_precompute_state(broken, expected_embedding_version="2")
    assert any(issue.field == "centroid_dot" for issue in column_issues)

    broken.executemany(
        """
        INSERT INTO slot_fillers (slot_type, filler, freq, popularity_score)
        VALUES (?, ?, 1, 1.0)
        """,
        [
            ("list_item", "race"),
            ("list_item", "power"),
            ("action_noun", "making"),
            ("of_object", "happiness"),
        ],
    )
    try:
        generate_module._remix_ctx = None
        generate_subtitle(broken, seed=1, remix_prob=0.0)
    except RuntimeError as exc:
        assert "slot_fillers is missing column" in str(exc)
    else:
        raise AssertionError("missing remix columns should fail before runtime usage")


def test_subtitle_to_dict_contract_is_stable():
    from subtitle_generator.generate import GeneratedSubtitle
    from subtitle_generator.handlers import subtitle_to_dict

    sub = GeneratedSubtitle(
        text="Race, Power, and the Making of Modern Life",
        item1="Race",
        item2="Power",
        action_noun="Making",
        of_object="Modern Life",
        remixed=True,
        remix_parts={"modifier": "Modern", "head": "Life"},
        remix_similarity=0.42,
        of_article="",
        action_article="the",
    )
    sources = {"item1": {"title": "Book", "tag": "LOC"}}

    assert subtitle_to_dict(sub, sources) == {
        "text": "Race, Power, and the Making of Modern Life",
        "item1": "Race",
        "item2": "Power",
        "action_noun": "Making",
        "of_object": "Modern Life",
        "remixed": True,
        "remix_parts": {"modifier": "Modern", "head": "Life"},
        "remix_similarity": 0.42,
        "of_article": "",
        "action_article": "the",
        "sources": sources,
    }


def test_handle_generate_uses_configured_remix_defaults(tmp_path, monkeypatch):
    from subtitle_generator import handlers
    from subtitle_generator.generate import GeneratedSubtitle

    db_path = tmp_path / "runtime.db"
    conn = make_runtime_db(db_path)
    conn.execute("INSERT OR REPLACE INTO config VALUES ('remix_calibrated_remix_prob', '0.33')")
    conn.execute("INSERT OR REPLACE INTO config VALUES ('remix_calibrated_min_sim', '0.44')")
    conn.commit()
    conn.close()

    observed = {}

    def fake_generate_subtitle_matching_tiers(conn, **kwargs):
        observed.update(kwargs)
        return GeneratedSubtitle(
            text="Race, Power, and the Making of Modern Life",
            item1="Race",
            item2="Power",
            action_noun="Making",
            of_object="Modern Life",
        )

    monkeypatch.setattr(handlers, "get_db", lambda db_path=None: sqlite3.connect(str(tmp_path / "runtime.db")))
    monkeypatch.setattr(
        handlers,
        "generate_subtitle_matching_tiers",
        fake_generate_subtitle_matching_tiers,
    )

    status, body = handlers.handle_generate({})

    assert status == 200
    assert body["text"] == "Race, Power, and the Making of Modern Life"
    assert observed["allowed_tiers"] is None
    assert observed["remix_prob"] == 0.33
    assert observed["min_sim"] == 0.44
    assert set(observed) == {"allowed_tiers", "remix_prob", "min_sim"}


def test_handle_jacket_dry_run_contract(tmp_path, monkeypatch):
    from subtitle_generator import handlers

    db_path = tmp_path / "runtime.db"
    conn = make_runtime_db(db_path)
    conn.close()
    monkeypatch.setattr(
        handlers,
        "get_db",
        lambda db_path=None: sqlite3.connect(str(db_path or tmp_path / "runtime.db")),
    )

    status, body = handlers.handle_jacket({
        "subtitle": "Race, Power, and the Making of Modern Life",
        "dry_run": True,
    })

    assert status == 200
    assert set(body) == {"prompt", "tone_tier", "result"}
    assert "The subtitle is:" in body["prompt"]
    assert body["tone_tier"] in {"pop", "mainstream", "niche"}
    assert body["result"] is None


def test_handle_rate_contract_persists_rating(tmp_path, monkeypatch):
    from subtitle_generator import handlers

    db_path = tmp_path / "runtime.db"
    conn = make_runtime_db(db_path)
    conn.close()
    monkeypatch.setattr(
        handlers,
        "get_db",
        lambda db_path=None: sqlite3.connect(str(db_path or tmp_path / "runtime.db")),
    )

    status, body = handlers.handle_rate({
        "subtitle": "Race, Power, and the Making of Modern Life",
        "thumbs": 1,
        "tone_override": "pop",
        "tags": ["interesting"],
        "_source": "contract-test",
    })

    assert status == 200
    assert body == {"id": 1, "status": "saved"}
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT thumbs, tone_override, tags, source FROM human_ratings"
    ).fetchone()
    conn.close()
    assert row == (1, "pop", '["interesting"]', "contract-test")


def test_jacket_generation_uses_prepared_prompt_once(monkeypatch):
    from subtitle_generator import jacket

    calls = []
    captured = {}

    def fake_build_jacket_prompt(*args, **kwargs):
        calls.append((args, kwargs))
        evidence = SimpleNamespace(
            tier="mainstream",
            accessibility_score=0.5,
            lower_tail_score=0.4,
            demand_confidence=1.0,
        )
        return "system prompt", "user prompt", "mainstream", evidence

    async def fake_generate(subtitle, system_prompt, user_prompt, **kwargs):
        captured.update({
            "subtitle": subtitle,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        })
        return "## Internal Concept\nhidden\n\n## Title\nVisible"

    monkeypatch.setattr(jacket, "_build_jacket_prompt_with_evidence", fake_build_jacket_prompt)
    monkeypatch.setattr(jacket, "_generate_jacket_from_prompt_async", fake_generate)

    result = jacket.generate_jacket("Race, Power, and the Pursuit of Happiness")

    assert len(calls) == 1
    assert captured == {
        "subtitle": "Race, Power, and the Pursuit of Happiness",
        "system_prompt": "system prompt",
        "user_prompt": "user prompt",
    }
    assert "## Internal Concept" not in result
    assert "## Title" in result


def test_jacket_stream_reports_tone_progress(monkeypatch):
    from subtitle_generator import serve

    class DummyConn:
        def close(self):
            pass

    def fake_build_jacket_prompt(subtitle, **kwargs):
        assert subtitle == "Race, Power, and the Pursuit of Happiness"
        assert isinstance(kwargs["conn"], DummyConn)
        return "system prompt", "user prompt", "mainstream"

    def fake_generate_jacket_from_prompt(*_args, on_progress=None, **_kwargs):
        assert on_progress is not None
        on_progress("Generating jacket...")
        return "## Title\nVisible"

    handler = object.__new__(serve._Handler)
    handler.wfile = io.BytesIO()
    handler.send_response = lambda _status: None
    handler.send_header = lambda _name, _value: None
    handler._cors_headers = lambda: None
    handler.end_headers = lambda: None
    monkeypatch.setattr(serve, "_get_db", lambda: DummyConn())
    monkeypatch.setattr(serve, "build_jacket_prompt", fake_build_jacket_prompt)
    monkeypatch.setattr(serve, "generate_jacket_from_prompt", fake_generate_jacket_from_prompt)

    handler._handle_jacket_stream({"subtitle": "Race, Power, and the Pursuit of Happiness"})

    body = handler.wfile.getvalue().decode("utf-8")
    assert "event: progress\ndata: Tone: mainstream\n\n" in body
    assert "event: progress\ndata: Generating jacket...\n\n" in body
    assert "event: result\n" in body


def test_rating_config_snapshot_preserves_defaults_and_overrides():
    from subtitle_generator.feedback import store_rating

    conn = make_runtime_db()
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_classification_blend', '0.25')")
    conn.commit()

    row_id = store_rating(
        conn,
        "Power, Race, and the Pursuit of Happiness",
        system_tone="pop",
        thumbs=1,
        tone_override="mainstream",
        tags=["interesting", "realistic"],
        source="characterization",
    )

    row = conn.execute(
        """
        SELECT config_snapshot, tags, source
        FROM human_ratings
        WHERE id = ?
        """,
        (row_id,),
    ).fetchone()
    snapshot = json.loads(row[0])

    assert snapshot.keys() == EXPECTED_TUNABLE_PARAMS.keys()
    assert snapshot["pop_classification_blend"] == 0.25
    assert snapshot["weighted_sample_spread"] == 0.12
    assert json.loads(row[1]) == ["interesting", "realistic"]
    assert row[2] == "characterization"


def test_tuning_config_change_applies_and_reverts_rows():
    from subtitle_generator.tuning_state import (
        ConfigChange,
        apply_config_change,
        revert_config_change,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('weighted_sample_spread', '0.7')")
    conn.commit()

    change = ConfigChange(
        param="weighted_sample_spread",
        old_value=0.7,
        new_value=0.8,
    )

    apply_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'weighted_sample_spread'"
    ).fetchone()[0] == "0.8"

    revert_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'weighted_sample_spread'"
    ).fetchone()[0] == "0.7"


def test_tuning_revert_removes_default_override_row():
    from subtitle_generator.config import ALL_TUNABLE_PARAMS
    from subtitle_generator.tuning_state import ConfigChange, revert_config_change

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('weighted_sample_spread', '0.9')")
    conn.commit()

    change = ConfigChange(
        param="weighted_sample_spread",
        old_value=ALL_TUNABLE_PARAMS["weighted_sample_spread"],
        new_value=0.9,
    )

    revert_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'weighted_sample_spread'"
    ).fetchone() is None


def test_autotune_param_surface_excludes_supervised_popularity_knobs():
    from subtitle_generator import tune

    autotune_keys = tune._autotune_param_keys()

    assert autotune_keys == {
        "article_of_min_freq",
        "article_action_min_freq",
        "article_remix_heuristic_threshold",
    }
    assert "weighted_sample_spread" not in autotune_keys
    assert "weighted_sample_bias_floor" not in autotune_keys
    assert "default_generation_tone_target" not in autotune_keys
    assert "generation_tier_ratio_pop" not in autotune_keys
    assert "generation_tier_ratio_mainstream" not in autotune_keys
    assert "generation_tier_ratio_niche" not in autotune_keys
    assert "remix_reject_double_of" not in autotune_keys
    assert "pop_weight_spl" not in autotune_keys
    assert "pop_base_weight_blend" not in autotune_keys
    assert "pop_classification_blend" not in autotune_keys
    assert "pop_missing_default" not in autotune_keys
    assert "pop_slot_mult_list_item" not in autotune_keys
    assert "tier_pop_min_lower_tail" not in autotune_keys
    assert "accessibility_threshold_pop" not in autotune_keys
    assert "tier_center_pop" not in autotune_keys
    assert not any(key.startswith("pop_") for key in autotune_keys)


def test_autotune_bounds_ignore_supervised_popularity_rows():
    from subtitle_generator import tune

    bounds = tune._parse_bounds(
        """
        | Parameter | Min | Max | Current | Notes |
        |---|---:|---:|---:|---|
        | `article_of_min_freq` | 1 | 10 | 1 | keep |
        | `weighted_sample_spread` | 0.05 | 0.5 | 0.12 | excluded |
        | `pop_weight_spl` | 0.0 | 1.0 | 0.7 | excluded |
        | `pop_slot_mult_*` | 0.5 | 2.0 | 1.0 | excluded wildcard |
        """
    )

    assert bounds == {"article_of_min_freq": (1.0, 10.0)}
    assert "weighted_sample_spread" not in bounds
    assert "pop_weight_spl" not in bounds
    assert "pop_slot_mult_list_item" not in bounds


def test_autotune_snapshot_contains_only_autotune_params():
    from subtitle_generator import tune

    params = {
        "article_of_min_freq": 2.0,
        "weighted_sample_spread": 0.2,
        "pop_weight_spl": 0.1,
        "pop_slot_mult_list_item": 1.2,
    }

    snapshot = tune._autotune_param_values(params)

    assert snapshot == {"article_of_min_freq": 2.0}


def test_autotune_regime_marker_parser_accepts_iteration_zero(tmp_path):
    from subtitle_generator import tune

    results_file = tmp_path / "results.tsv"
    params = ",".join(sorted(tune._autotune_param_keys()))
    results_file.write_text(
        "iteration\tparam\told_value\tnew_value\tquality\tseparation\tcomposite\tstatus\tdescription\n"
        f"0\t[regime change]\t0\t0\t0.0000\t0.0000\t0.0000\tregime\tavailable_params={params}\n",
        encoding="utf-8",
    )

    tune._check_regime_change(str(results_file))

    assert results_file.read_text(encoding="utf-8").count("[regime change]") == 1


def test_autotune_regime_change_warns_when_snapshot_is_cleared(tmp_path, capsys):
    from subtitle_generator import tune

    results_file = tmp_path / "results.tsv"
    old_params = ",".join(sorted([
        *tune._autotune_param_keys(),
        "pop_weight_spl",
    ]))
    results_file.write_text(
        "iteration\tparam\told_value\tnew_value\tquality\tseparation\tcomposite\tstatus\tdescription\n"
        f"0\t[regime change]\t0\t0\t0.0000\t0.0000\t0.0000\tregime\tavailable_params={old_params}\n",
        encoding="utf-8",
    )
    snapshot_path = tune._best_snapshot_path(str(results_file))
    snapshot_path.write_text("{}", encoding="utf-8")

    tune._check_regime_change(str(results_file))

    output = capsys.readouterr().out
    assert "cleared tune_best_state.json" in output
    assert not snapshot_path.exists()
    assert results_file.read_text(encoding="utf-8").count("[regime change]") == 2


def test_autotune_unmarked_regime_change_detects_removed_params(tmp_path, capsys):
    from subtitle_generator import tune

    results_file = tmp_path / "results.tsv"
    rows = [
        "iteration\tparam\told_value\tnew_value\tquality\tseparation\tcomposite\tstatus\tdescription",
    ]
    for index, param in enumerate(sorted(tune._autotune_param_keys()), start=1):
        rows.append(
            f"{index}\t{param}\t1\t2\t0.1\t0.1\t0.1\tdiscarded\told run"
        )
    rows.append(
        "99\tweighted_sample_spread\t0.12\t0.2\t0.1\t0.1\t0.1\tdiscarded\told run"
    )
    results_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    snapshot_path = tune._best_snapshot_path(str(results_file))
    snapshot_path.write_text("{}", encoding="utf-8")

    tune._check_regime_change(str(results_file))

    output = capsys.readouterr().out
    text = results_file.read_text(encoding="utf-8")
    assert "cleared tune_best_state.json" in output
    assert not snapshot_path.exists()
    assert "Params removed: weighted_sample_spread." in text
    assert text.count("[regime change]") == 1


def test_autotune_unmarked_history_ignores_auto_revert_marker(tmp_path, capsys):
    from subtitle_generator import tune

    results_file = tmp_path / "results.tsv"
    rows = [
        "iteration\tparam\told_value\tnew_value\tquality\tseparation\tcomposite\tstatus\tdescription",
    ]
    for index, param in enumerate(sorted(tune._autotune_param_keys()), start=1):
        rows.append(
            f"{index}\t{param}\t1\t2\t0.1\t0.1\t0.1\tdiscarded\told run"
        )
    rows.append(
        "99\t[auto-revert]\t0\t0\t0.1\t0.1\t0.1\treverted\trestored best"
    )
    results_file.write_text("\n".join(rows) + "\n", encoding="utf-8")
    snapshot_path = tune._best_snapshot_path(str(results_file))
    snapshot_path.write_text("{}", encoding="utf-8")

    tune._check_regime_change(str(results_file))

    assert capsys.readouterr().out == ""
    assert snapshot_path.exists()
    assert results_file.read_text(encoding="utf-8").count("[regime change]") == 0


def test_tuning_proposal_decision_records_before_after_scores():
    from subtitle_generator.tuning_state import ConfigChange, record_decision

    decision = record_decision(
        change=ConfigChange(
            param="pop_classification_blend",
            old_value=0.5,
            new_value=0.6,
        ),
        reasoning="Increase tone influence.",
        status="kept",
        before=(7.5, 0.2, 7.9),
        after=(7.6, 0.3, 8.2),
    )

    assert decision.change.param == "pop_classification_blend"
    assert decision.status == "kept"
    assert decision.quality_before == 7.5
    assert decision.separation_after == 0.3


def test_pipeline_validation_passes_for_minimal_ready_db(tmp_path):
    from subtitle_generator.extract import get_db
    from subtitle_generator.extract_openlibrary import ensure_isbn_column
    from subtitle_generator.feedback import ensure_ratings_table
    from subtitle_generator.pipeline_validation import validate_pipeline
    from subtitle_generator.slots import ensure_slot_tables

    populate_popularity = _load_populate_popularity_module()

    conn = get_db(tmp_path / "ready.db")
    ensure_isbn_column(conn)
    ensure_slot_tables(conn)
    populate_popularity.create_tables(conn)
    ensure_ratings_table(conn)
    _insert_runtime_config(conn)
    conn.executemany(
        """
        INSERT INTO slot_fillers (
            slot_type, filler, mode, freq, popularity_score,
            popularity_level, popularity_confidence
        )
        VALUES (?, ?, 'strict', ?, ?, 1, 1.0)
        """,
        [
            ("list_item", "race", 10, 1.0),
            ("action_noun", "making", 10, 1.0),
            ("of_object", "modern life", 10, 1.0),
        ],
    )
    conn.commit()

    report = validate_pipeline(conn)

    assert report.ok
    assert report.issues == ()


def test_pipeline_validation_reports_readiness_failures_without_generation():
    from subtitle_generator.pipeline_validation import validate_pipeline

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            slot_type TEXT,
            mode TEXT,
            popularity_score REAL
        )
        """
    )
    conn.execute(
        """
        INSERT INTO slot_fillers (slot_type, mode, popularity_score)
        VALUES ('list_item', 'strict', NULL)
        """
    )
    conn.commit()

    report = validate_pipeline(conn)

    messages = [issue.message for issue in report.issues]
    assert not report.ok
    assert any("missing required table 'subtitles'" in message for message in messages)
    assert any("missing required config key 'embedding_version'" in message for message in messages)
    assert any("missing columns: popularity_confidence, popularity_level" in message for message in messages)
    assert any("strict slot fillers lack popularity_score" in message for message in messages)
    assert any("no strict 'action_noun' candidates" in message for message in messages)
