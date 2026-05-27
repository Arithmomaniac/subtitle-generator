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
    "generation_tier_ratio_pop": 0.0183,
    "generation_tier_ratio_mainstream": 0.1172,
    "generation_tier_ratio_niche": 0.8645,
    "article_of_min_freq": 1.0,
    "article_action_min_freq": 1.0,
    "article_remix_heuristic_threshold": 0.6,
    "remix_reject_double_of": 1.0,
    "pop_weight_spl": 0.7,
    "pop_weight_ol": 0.3,
    "pop_weight_gr": 0.2,
    "pop_weight_nyt": 0.1,
    "pop_weight_library": 0.05,
    "pop_weight_trove": 0.10,
    "pop_missing_default": 0.1,
    "tier_classifier_model_score_weight": 1.0,
    "tier_classifier_temperature": 1.0,
    "tier_classifier_slot_weight_list_item": 1.0,
    "tier_classifier_slot_weight_action_noun": 1.0,
    "tier_classifier_slot_weight_of_object": 1.0,
    "tier_classifier_intercept_pop": 0.0,
    "tier_classifier_intercept_mainstream": 0.0,
    "tier_classifier_intercept_niche": 0.0,
    "tier_classifier_popularity_weight_pop": 0.0,
    "tier_classifier_popularity_weight_mainstream": 0.0,
    "tier_classifier_popularity_weight_niche": 0.0,
    "tier_classifier_popularity_interaction_pop": 0.0,
    "tier_classifier_popularity_interaction_mainstream": 0.0,
    "tier_classifier_popularity_interaction_niche": 0.0,
    "tier_classifier_popularity_observed_weight_pop": 0.0,
    "tier_classifier_popularity_observed_weight_mainstream": 0.0,
    "tier_classifier_popularity_observed_weight_niche": 0.0,
    "tier_classifier_popularity_observed_interaction_pop": 0.0,
    "tier_classifier_popularity_observed_interaction_mainstream": 0.0,
    "tier_classifier_popularity_observed_interaction_niche": 0.0,
    "tier_classifier_frequency_weight_pop": 0.0,
    "tier_classifier_frequency_weight_mainstream": 0.0,
    "tier_classifier_frequency_weight_niche": 0.0,
    "tier_classifier_frequency_interaction_pop": 0.0,
    "tier_classifier_frequency_interaction_mainstream": 0.0,
    "tier_classifier_frequency_interaction_niche": 0.0,
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
    conn.execute(
        """
        INSERT INTO slot_filler_model_scores (
            slot_filler_id, score_pop, score_mainstream, score_niche,
            model_tier, source_prediction_count
        )
        SELECT id, 0.85, 0.1, 0.05, 'pop', 1 FROM slot_fillers
        """
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
        "candidate_text", "candidate_source",
    }
    assert _columns(conn, "pattern_matches") >= {
        "id", "subtitle_id", "title", "subtitle", "list_items_json",
        "action_noun", "of_object", "of_article", "action_article",
        "candidate_source",
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
        "nyt_peak_rank", "library_appearances", "trove_library_count",
        "trove_holding_count", "trove_copy_count", "trove_copy_count_is_exact",
        "composite_score",
    }
    assert _columns(conn, "config") == {"key", "value"}
    assert _columns(conn, "human_ratings") >= {
        "id", "subtitle", "system_tone", "thumbs", "tone_override",
        "free_text", "interpreted", "config_snapshot", "created_at",
        "tags", "source", "prompt_generated",
    }
    assert validate_schema(conn) == []


def test_subtitle_candidate_columns_are_backfilled(tmp_path):
    from subtitle_generator.extract import get_db

    db_path = tmp_path / "legacy-subtitles.db"
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            subtitle TEXT NOT NULL,
            lang TEXT,
            lccn TEXT,
            source_file TEXT,
            isbn TEXT
        )
        """
    )
    legacy.execute(
        """
        INSERT INTO subtitles (title, subtitle, lang, lccn, source_file, isbn)
        VALUES ('Book', 'Race, Power, and the Rise of Empire', 'eng', 'lccn', 'loc.mrc', 'isbn')
        """
    )
    legacy.commit()
    legacy.close()

    conn = get_db(db_path)
    row = conn.execute(
        """
        SELECT candidate_text, candidate_source
        FROM subtitles
        WHERE title = 'Book'
        """
    ).fetchone()

    assert row == ("Race, Power, and the Rise of Empire", "subtitle")


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
        work_trove={},
        all_works={"work-a", "work-b", "work-c"},
    )
    percentiles = pop.build_percentile_models(data)
    params = PopularityParameters(
        weight_spl=0.7,
        weight_ol=0.3,
        weight_goodreads=0.2,
        weight_nyt=0.1,
        weight_library=0.05,
        weight_trove=0.10,
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
        INSERT INTO slot_fillers (id, slot_type, filler, freq, popularity_score, popularity_level, popularity_confidence)
        VALUES
            (1, 'list_item', 'future', 100, NULL, NULL, NULL),
            (2, 'list_item', 'fallback', 99, NULL, NULL, NULL),
            (3, 'list_item', 'partial', 9, 7.0, NULL, NULL),
            (4, 'list_item', 'stale', 4, 0.9, 1, 1.0);
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

    assert pop.update_fallback_filler_scores(conn) == 3
    fallback_rows = {
        row[0]: row[1:]
        for row in conn.execute(
        """
        SELECT id, popularity_score, popularity_level, popularity_confidence
        FROM slot_fillers
        WHERE id IN (2, 3, 4)
        """
        ).fetchall()
    }
    assert fallback_rows[2][0] is None
    assert fallback_rows[2][1:] == (0, 0.0)
    assert fallback_rows[3][0] is None
    assert fallback_rows[3][1:] == (0, 0.0)
    assert fallback_rows[4][0] is None
    assert fallback_rows[4][1:] == (0, 0.0)


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


def test_tier_slot_distribution_contract_validates_required_invariants():
    from subtitle_generator.schema_contracts import (
        TIER_SLOT_DISTRIBUTION_SCHEMA_CONTRACTS,
        TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
        validate_schema,
        validate_tier_slot_distribution,
    )

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE tier_slot_filler_distribution_v1 (
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
        );
        INSERT INTO slot_fillers (id, slot_type, filler, freq)
        VALUES
            (1, 'list_item', 'Race', 10),
            (2, 'list_item', 'Power', 8),
            (3, 'action_noun', 'Rise', 6),
            (4, 'action_noun', 'Fall', 4);
        """
    )
    rows = []
    for slot_type, filler_a, filler_b in [
        ("list_item", "Race", "Power"),
        ("action_noun", "Rise", "Fall"),
    ]:
        for tier in ["pop", "mainstream", "niche"]:
            rows.extend([
                (
                    slot_type, tier, filler_a.casefold(), filler_a, 0.75,
                    -0.287682072, 3.0, 0.2, 3.2, 3, 2, 1, 2.5,
                    0.5, 0.9, 10, 0.5, 0.1, 1.0, "v1",
                ),
                (
                    slot_type, tier, filler_b.casefold(), filler_b, 0.25,
                    -1.386294361, 1.0, 0.2, 1.2, 1, 0, 1, 0.0,
                    1.0, 0.8, 4, None, 0.1, 1.0, "v1",
                ),
            ])
    conn.executemany(
        f"INSERT INTO {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    assert validate_schema(conn, TIER_SLOT_DISTRIBUTION_SCHEMA_CONTRACTS) == []
    assert validate_tier_slot_distribution(conn) == []


def test_tier_slot_distribution_contract_reports_bad_mass_and_unknown_fillers():
    from subtitle_generator.schema_contracts import validate_tier_slot_distribution

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE tier_slot_filler_distribution_v1 (
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
        );
        INSERT INTO slot_fillers (id, slot_type, filler, freq)
        VALUES (1, 'list_item', 'Race', 10), (2, 'list_item', 'Power', 8);
        INSERT INTO tier_slot_filler_distribution_v1 VALUES
            ('list_item', 'pop', 'race', 'Race', 0.60, -0.51, 3.0, 0.0, 3.0, 3, 2, 1, 2.5, 0.5, 0.9, 10, 0.5, 0.0, 1.0, 'v1'),
            ('list_item', 'pop', 'unknown', 'Unknown', 0.30, -1.20, 1.0, 0.0, 1.0, 1, 0, 1, 0.0, 1.0, 0.8, 4, NULL, 0.0, 1.0, 'v1');
        """
    )

    issues = validate_tier_slot_distribution(conn)

    assert any(issue.column == "filler" for issue in issues)
    assert any(issue.column == "probability" for issue in issues)
    assert any("every (tier, slot_type) pair" in issue.message for issue in issues)


def test_tier_slot_distribution_contract_reports_count_identity_mismatch():
    from subtitle_generator.schema_contracts import validate_tier_slot_distribution

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE tier_slot_filler_distribution_v1 (
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
        );
        INSERT INTO slot_fillers (id, slot_type, filler, freq)
        VALUES (1, 'list_item', 'Race', 10);
        INSERT INTO tier_slot_filler_distribution_v1 VALUES
            ('list_item', 'pop', 'race', 'Race', 1.0, 0.0, 3.0, 0.0, 3.0, 3, 1, 1, 1.0, 1.0, 0.9, 10, 0.5, 0.0, 1.0, 'v1'),
            ('list_item', 'mainstream', 'race', 'Race', 1.0, 0.0, 0.5, 0.0, 0.5, 1, 0, 1, 0.0, 0.5, 0.9, 10, 0.5, 0.0, 1.0, 'v1'),
            ('list_item', 'niche', 'race', 'Race', 1.0, 0.0, 0.5, 0.0, 0.5, 1, 0, 1, 0.0, 0.5, 0.9, 10, 0.5, 0.0, 1.0, 'v1');
        """
    )

    issues = validate_tier_slot_distribution(conn)

    assert any("source_count must equal anchored" in issue.message for issue in issues)


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
        get_article_parameters,
        get_popularity_parameters,
        get_generation_tier_ratios,
        get_remix_parameters,
        get_runtime_generation_parameters,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('pop_weight_spl', '0.9')")
    conn.execute("INSERT INTO config VALUES ('article_of_min_freq', '2')")
    conn.execute("INSERT INTO config VALUES ('remix_reject_double_of', '0')")
    conn.commit()

    popularity = get_popularity_parameters(conn)
    runtime = get_runtime_generation_parameters(conn)

    assert popularity.weight_spl == 0.9
    assert popularity.weight_ol == EXPECTED_TUNABLE_PARAMS["pop_weight_ol"]
    assert popularity.weight_trove == EXPECTED_TUNABLE_PARAMS["pop_weight_trove"]
    assert get_generation_tier_ratios(conn).pop == 0.0183
    assert runtime.generation_tier_ratios.mainstream == 0.1172
    assert runtime.article == get_article_parameters(conn)
    assert runtime.remix == get_remix_parameters(conn)
    assert runtime.article.of_min_freq == 2.0
    assert runtime.remix.reject_double_of == 0.0


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
    observed_model_tiers: list[str | None] = []

    def fake_generate_from_candidates(conn, candidates, **kwargs):
        observed_model_tiers.append(kwargs["model_tier"])
        return GeneratedSubtitle(
            text="Generated pop",
            item1="Race",
            item2="Power",
            action_noun="Pursuit",
            of_object="Happiness",
        )

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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
    assert observed_model_tiers == ["pop"]


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

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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


def test_cli_spot_check_uses_tier_generation_batches(monkeypatch):
    import subtitle_generator.tune as tune_module
    from subtitle_generator.generate import GeneratedSubtitle

    conn = make_runtime_db()
    requested_batches: list[tuple[int, tuple[str, ...], int]] = []
    captured_samples: list[tuple[str, str, object]] = []

    def fake_generate_subtitles_by_tier(
        conn, *, tiers, samples_per_tier, seed, **kwargs
    ):
        requested_batches.append((seed, tuple(tiers), samples_per_tier))
        return {
            tier: [
                GeneratedSubtitle(
                    text=f"{tier} {i}",
                    item1="Race",
                    item2="Power",
                    action_noun="Pursuit",
                    of_object="Happiness",
                )
                for i in range(samples_per_tier)
            ]
            for tier in tiers
        }

    def fake_spot_check_cli(conn, samples, tier_labels, tier_shortcuts, source):
        captured_samples.extend(samples)
        return 1.0

    monkeypatch.setattr(
        "subtitle_generator.generate.generate_subtitles_by_tier",
        fake_generate_subtitles_by_tier,
    )
    monkeypatch.setattr(tune_module, "_spot_check_cli", fake_spot_check_cli)

    accuracy = tune_module.run_spot_check(conn, n_samples=2, seed_base=10)

    assert accuracy == 1.0
    assert requested_batches == [
        (10, ("pop",), 2),
        (110, ("mainstream",), 2),
        (210, ("niche",), 2),
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

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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

    def fake_compute_tier_evidence(subtitle, conn, **kwargs):
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
        "prompt_generated": True,
        "_source": "contract-test",
    })

    assert status == 200
    assert body == {"id": 1, "status": "saved"}
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT thumbs, tone_override, tags, source, prompt_generated FROM human_ratings"
    ).fetchone()
    conn.close()
    assert row == (1, "pop", '["interesting"]', "contract-test", 1)


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
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_missing_default', '0.25')")
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
    assert snapshot["pop_missing_default"] == 0.25
    assert snapshot["generation_tier_ratio_pop"] == 0.0183
    assert json.loads(row[1]) == ["interesting", "realistic"]
    assert row[2] == "characterization"


def test_review_ratings_reports_prompt_only_feedback(capsys, monkeypatch):
    from subtitle_generator import tune
    from subtitle_generator.feedback import store_rating

    conn = make_runtime_db()
    store_rating(
        conn,
        "Power, Race, and the Pursuit of Happiness",
        system_tone="mainstream",
        tags=["interesting"],
        source="web_user",
        prompt_generated=True,
    )

    captured = {}

    def fake_structured_completion(*args, **kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        return SimpleNamespace(diff="--- old\n+++ new\n", reasoning="prompt interest only")

    monkeypatch.setattr(tune, "_load_goals", lambda: "goals")
    monkeypatch.setattr(tune, "structured_completion", fake_structured_completion)

    tune.review_ratings(conn, source="web_user", model="fake-model")

    output = capsys.readouterr().out
    assert "Prompt generated: 1" in output
    assert "Tone accuracy: no tone-rated entries" in output
    assert "Prompt generated: 1" in captured["prompt"]


def test_tuning_config_change_applies_and_reverts_rows():
    from subtitle_generator.tuning_state import (
        ConfigChange,
        apply_config_change,
        revert_config_change,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('article_of_min_freq', '0.7')")
    conn.commit()

    change = ConfigChange(
        param="article_of_min_freq",
        old_value=0.7,
        new_value=0.8,
    )

    apply_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'article_of_min_freq'"
    ).fetchone()[0] == "0.8"

    revert_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'article_of_min_freq'"
    ).fetchone()[0] == "0.7"


def test_tuning_revert_removes_default_override_row():
    from subtitle_generator.config import ALL_TUNABLE_PARAMS
    from subtitle_generator.tuning_state import ConfigChange, revert_config_change

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO config VALUES ('article_of_min_freq', '0.9')")
    conn.commit()

    change = ConfigChange(
        param="article_of_min_freq",
        old_value=ALL_TUNABLE_PARAMS["article_of_min_freq"],
        new_value=0.9,
    )

    revert_config_change(conn, change)
    assert conn.execute(
        "SELECT value FROM config WHERE key = 'article_of_min_freq'"
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
            param="article_of_min_freq",
            old_value=0.5,
            new_value=0.6,
        ),
        reasoning="Increase tone influence.",
        status="kept",
        before=(7.5, 0.2, 7.9),
        after=(7.6, 0.3, 8.2),
    )

    assert decision.change.param == "article_of_min_freq"
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
    assert any("strict non-Level-0 slot fillers lack popularity_score" in message for message in messages)
    assert any("no strict 'action_noun' candidates" in message for message in messages)


def test_pipeline_validation_allows_level0_missing_popularity_score():
    from subtitle_generator.pipeline_validation import validate_pipeline

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE subtitles (id INTEGER PRIMARY KEY);
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE slot_fillers (
            slot_type TEXT,
            mode TEXT,
            freq INTEGER,
            popularity_score REAL,
            popularity_level INTEGER,
            popularity_confidence REAL
        );
        INSERT INTO config VALUES ('embedding_version', '2');
        INSERT INTO slot_fillers VALUES
            ('list_item', 'strict', 1, NULL, 0, 0.0),
            ('action_noun', 'strict', 1, 0.2, 1, 1.0),
            ('of_object', 'strict', 1, 0.2, 1, 1.0);
        """
    )

    report = validate_pipeline(conn)

    messages = [issue.message for issue in report.issues]
    assert not any("lack popularity_score" in message for message in messages)
