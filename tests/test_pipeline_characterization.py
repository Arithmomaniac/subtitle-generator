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


ROOT = Path(__file__).resolve().parent.parent


EXPECTED_TUNABLE_PARAMS = {
    "weighted_sample_spread": 0.4,
    "weighted_sample_bias_floor": 0.05,
    "tone_target_pop_list_item": 0.78,
    "tone_target_pop_action_noun": 0.78,
    "tone_target_pop_of_object": 0.78,
    "tone_target_mainstream_list_item": 0.3,
    "tone_target_mainstream_action_noun": 0.3,
    "tone_target_mainstream_of_object": 0.3,
    "tone_target_niche_list_item": 0.16,
    "tone_target_niche_action_noun": 0.16,
    "tone_target_niche_of_object": 0.16,
    "sample_tone_spread": 0.6,
    "tier_center_pop": 0.78,
    "tier_center_mainstream": 0.3,
    "tier_center_niche": 0.16,
    "accessibility_threshold_pop": 0.6,
    "accessibility_threshold_mainstream": 0.3,
    "article_of_min_freq": 1.0,
    "article_action_min_freq": 1.0,
    "article_remix_heuristic_threshold": 0.6,
    "remix_reject_double_of": 1.0,
    "pop_weight_spl": 0.7,
    "pop_weight_ol": 0.3,
    "pop_weight_gr": 0.2,
    "pop_weight_nyt": 0.1,
    "pop_weight_library": 0.05,
    "pop_weight_freq": 0.0,
    "pop_exponent": 1.2,
    "pop_base_weight_blend": 0.5,
    "pop_tone_blend": 0.5,
    "pop_classification_blend": 0.9,
    "pop_missing_default": 0.1,
    "tier_pop_min_demand_confidence": 0.25,
    "tier_pop_min_lower_tail": 0.35,
    "tier_pop_min_accessibility_margin": 0.35,
    "tier_mainstream_demand_relief": 0.1,
    "tier_source_label_weight": 0.25,
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
        weight_frequency=0.0,
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
    assert params["tone_target_mainstream_of_object"] == params["tier_center_mainstream"]


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


def test_ensure_slot_tables_migrates_legacy_popularity_evidence_conservatively():
    from subtitle_generator.slots import ensure_slot_tables

    conn = sqlite3.connect(":memory:")
    conn.execute(
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
    conn.execute(
        """
        INSERT INTO slot_fillers (slot_type, filler, mode, freq, popularity_score)
        VALUES
            ('list_item', 'money', 'strict', 10, 0.8),
            ('list_item', 'fallback', 'strict', 1, NULL)
        """
    )
    conn.commit()

    ensure_slot_tables(conn)

    assert _columns(conn, "slot_fillers") >= {
        "popularity_score", "popularity_level", "popularity_confidence",
    }
    rows = conn.execute(
        """
        SELECT filler, popularity_level, popularity_confidence
        FROM slot_fillers
        ORDER BY filler
        """
    ).fetchall()
    assert rows == [
        ("fallback", 0, 0.0),
        ("money", 0, 0.0),
    ]


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
    assert runtime.popularity_blends == blends
    assert runtime.slot_multipliers.of_object == 1.4
    assert get_tier_threshold_parameters(conn).accessibility_pop == 0.6
    assert get_tier_classifier_parameters(conn).pop_min_demand_confidence == 0.25
    assert get_tone_targets(conn).pop["list_item"] == 0.78


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


def test_handler_generate_payload_shape_with_locked_values(tmp_path, monkeypatch):
    from subtitle_generator import handlers

    db_path = tmp_path / "runtime.db"
    conn = make_runtime_db(db_path)
    conn.close()

    monkeypatch.setattr(
        handlers,
        "get_db",
        lambda db_path=None: sqlite3.connect(str(db_path or tmp_path / "runtime.db")),
    )

    status, body = handlers.handle_generate({
        "locks": {
            "item1": "race",
            "item2": "power",
            "action_noun": "making",
            "of_object": "modern life",
        },
        "remix_prob": 0.0,
        "min_sim": 0.0,
    })

    assert status == 200
    assert set(body) == {
        "text", "item1", "item2", "action_noun", "of_object", "remixed",
        "remix_parts", "remix_similarity", "of_article", "action_article",
        "sources",
    }
    assert body == {
        "text": "Race, Power, and the Making of Modern Life",
        "item1": "Race",
        "item2": "Power",
        "action_noun": "Making",
        "of_object": "Modern Life",
        "remixed": False,
        "remix_parts": {},
        "remix_similarity": None,
        "of_article": "",
        "action_article": "the",
        "sources": {
            "item1": {"title": None, "tag": None},
            "item2": {"title": None, "tag": None},
            "action_noun": {"title": None, "tag": None},
            "of_object": {"title": None, "tag": None},
        },
    }


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

    def fake_generate_subtitle(conn, **kwargs):
        observed.update(kwargs)
        return GeneratedSubtitle(
            text="Race, Power, and the Making of Modern Life",
            item1="Race",
            item2="Power",
            action_noun="Making",
            of_object="Modern Life",
        )

    monkeypatch.setattr(handlers, "get_db", lambda db_path=None: sqlite3.connect(str(tmp_path / "runtime.db")))
    monkeypatch.setattr(handlers, "generate_subtitle", fake_generate_subtitle)

    status, body = handlers.handle_generate({})

    assert status == 200
    assert body["text"] == "Race, Power, and the Making of Modern Life"
    assert observed["remix_prob"] == 0.33
    assert observed["min_sim"] == 0.44
    assert observed["locks"] is None


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
        return "system prompt", "user prompt", "mainstream", evidence, False

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
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_tone_blend', '0.25')")
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
    assert snapshot["pop_tone_blend"] == 0.25
    assert snapshot["weighted_sample_spread"] == 0.4
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


def test_tuning_proposal_decision_records_before_after_scores():
    from subtitle_generator.tuning_state import ConfigChange, record_decision

    decision = record_decision(
        change=ConfigChange(
            param="pop_tone_blend",
            old_value=0.5,
            new_value=0.6,
        ),
        reasoning="Increase tone influence.",
        status="kept",
        before=(7.5, 0.2, 7.9),
        after=(7.6, 0.3, 8.2),
    )

    assert decision.change.param == "pop_tone_blend"
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
