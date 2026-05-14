"""Tests for popularity-weighted sampling integration.

Run:  uv run python tests/test_popularity.py
"""

import math
import random
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _make_test_db(pop_scores: dict[str, float | None] | None = None) -> sqlite3.Connection:
    """Create an in-memory DB with slot_fillers and config tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
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
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute("""
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TRIGGER insert_default_test_model_scores
        AFTER INSERT ON slot_fillers
        BEGIN
            INSERT OR REPLACE INTO slot_filler_model_scores (
                slot_filler_id,
                score_pop,
                score_mainstream,
                score_niche,
                model_tier,
                source_prediction_count
            )
            VALUES (
                NEW.id,
                CASE WHEN COALESCE(NEW.popularity_score, 0.1) >= 0.7 THEN 0.85 ELSE 0.05 END,
                CASE
                    WHEN COALESCE(NEW.popularity_score, 0.1) >= 0.7 THEN 0.1
                    WHEN COALESCE(NEW.popularity_score, 0.1) <= 0.2 THEN 0.1
                    ELSE 0.85
                END,
                CASE WHEN COALESCE(NEW.popularity_score, 0.1) <= 0.2 THEN 0.85 ELSE 0.05 END,
                CASE
                    WHEN COALESCE(NEW.popularity_score, 0.1) >= 0.7 THEN 'pop'
                    WHEN COALESCE(NEW.popularity_score, 0.1) <= 0.2 THEN 'niche'
                    ELSE 'mainstream'
                END,
                1
            );
        END
    """)

    # Insert test fillers with varying freq and popularity
    fillers = [
        ("Race", 40, 1.8),
        ("Power", 35, 1.6),
        ("Gender", 36, 0.21),
        ("Home", 2, 1.56),
        ("Fraud", 1, 1.55),
        ("Helmontian", 1, 0.09),
        ("Pursuit", 25, 1.4),
        ("Happiness", 30, 1.5),
        ("Rise", 30, 1.3),
        ("Markets", 30, 1.3),
        ("Archives", 1, 0.1),
    ]
    for i, (filler, freq, pop) in enumerate(fillers):
        ps = pop
        if pop_scores is not None and filler in pop_scores:
            ps = pop_scores[filler]
        for slot_type in ("list_item", "action_noun", "of_object"):
            conn.execute(
                "INSERT INTO slot_fillers (slot_type, filler, mode, freq, popularity_score, popularity_level, popularity_confidence) "
                "VALUES (?, ?, 'strict', ?, ?, ?, ?)",
                (slot_type, filler, freq, ps, 1 if ps is not None else 0, 1.0 if ps is not None else 0.0),
            )
    conn.commit()
    return conn


def test_weighted_sample_defaults_to_frequency_without_model_tier():
    """Without a requested tier, sampling is frequency-weighted."""
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db()

    rows_with_pop = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'list_item'"
    ).fetchall()
    rows_without_pop = [(r[0], r[1]) for r in rows_with_pop]

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    result_with = _weighted_sample(rows_with_pop, 3, rng1)
    result_without = _weighted_sample(rows_without_pop, 3, rng2)

    assert result_with == result_without


def test_weighted_sample_uses_learned_tier_scores_when_requested():
    """Requested tiers use learned per-filler model scores instead of popularity blends."""
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db()

    rows = conn.execute(
        """
        SELECT sf.filler, sf.freq, sf.popularity_score, ms.score_pop, ms.score_mainstream, ms.score_niche
        FROM slot_fillers sf
        JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
        WHERE sf.slot_type = 'list_item'
        """
    ).fetchall()

    counts: dict[str, int] = {}
    for seed in range(200):
        picked = _weighted_sample(rows, 1, random.Random(seed), model_tier="niche")
        counts[picked[0]] = counts.get(picked[0], 0) + 1

    assert counts.get("Helmontian", 0) > counts.get("Race", 0)
    assert counts.get("Archives", 0) > counts.get("Power", 0)


def test_weighted_sample_handles_null_popularity_when_model_scores_exist():
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db(pop_scores={"Fraud": None, "Helmontian": None})
    rows = conn.execute(
        """
        SELECT sf.filler, sf.freq, sf.popularity_score, ms.score_pop, ms.score_mainstream, ms.score_niche
        FROM slot_fillers sf
        JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
        WHERE sf.slot_type = 'list_item'
        """
    ).fetchall()

    result = _weighted_sample(rows, 3, random.Random(42), model_tier="pop")
    assert len(result) == 3


def test_removed_popularity_blends_are_not_tunable_config():
    from subtitle_generator.config import load_tuning_config

    conn = _make_test_db()
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_classification_blend', '0.5')")
    conn.commit()
    from subtitle_generator.config import invalidate_config_cache
    invalidate_config_cache()

    cfg = load_tuning_config(conn)
    assert "pop_classification_blend" not in cfg


def test_export_import_roundtrip():
    """popularity_score survives export → CSV → import cycle."""
    import csv
    import tempfile

    from subtitle_generator.export_db import export_data, build_mini_db

    conn = _make_test_db()
    # export_data joins on subtitles table for sources — create a minimal one
    conn.execute("""
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            source_file TEXT
        )
    """)
    conn.execute("INSERT INTO subtitles VALUES (1, 'Test Book', 'A Test Subtitle', 'openlibrary')")
    conn.execute("INSERT INTO config VALUES ('z_export_order_probe', 'last')")
    conn.execute("INSERT INTO config VALUES ('a_export_order_probe', 'first')")
    conn.execute("INSERT INTO config VALUES ('tone_target_pop_list_item', '0.77')")
    conn.execute("INSERT INTO config VALUES ('remix_calibrated_remix_prob', '0.4')")
    conn.execute("INSERT INTO config VALUES ('remix_calibrated_min_sim', '0.2')")
    conn.commit()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Export
        stats = export_data(conn, tmp_path)
        assert stats["slot_fillers.csv"] > 0

        # Verify CSV has popularity evidence columns
        with open(tmp_path / "slot_fillers.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            assert "popularity_score" in first_row, (
                f"Missing popularity_score column. Headers: {list(first_row.keys())}"
            )
            assert "popularity_level" in first_row, (
                f"Missing popularity_level column. Headers: {list(first_row.keys())}"
            )
            assert "popularity_confidence" in first_row, (
                f"Missing popularity_confidence column. Headers: {list(first_row.keys())}"
            )
            assert first_row["popularity_score"] != "", (
                "popularity_score should not be empty for Race"
            )

        with open(tmp_path / "config.csv", encoding="utf-8") as f:
            config_rows = list(csv.DictReader(f))
            exported_config = {row["key"]: row["value"] for row in config_rows}
            assert all(
                not row["key"].startswith("tone_target_") for row in config_rows
            )
            assert exported_config["remix_calibrated_remix_prob"] == "0.4"
            assert exported_config["remix_calibrated_min_sim"] == "0.2"
            assert "z_export_order_probe" not in exported_config
            assert [row["key"] for row in config_rows] == sorted(
                row["key"] for row in config_rows
            )

        # Import into mini DB
        mini_path = tmp_path / "mini.db"
        build_mini_db(tmp_path, mini_path)

        # Verify imported data
        mini = sqlite3.connect(str(mini_path))
        row = mini.execute(
            "SELECT popularity_score, popularity_level, popularity_confidence "
            "FROM slot_fillers WHERE filler = 'Race' LIMIT 1"
        ).fetchone()
        assert row is not None and row[0] is not None, "popularity_score lost in import"
        assert abs(row[0] - 1.8) < 0.01, f"Expected ~1.8, got {row[0]}"
        assert row[1:] == (1, 1.0)
        columns = {r[1] for r in mini.execute("PRAGMA table_info(slot_fillers)")}
        assert "vector_sum" in columns, "mini DB must preserve the remix runtime schema"
        assert "popularity_level" in columns
        assert "popularity_confidence" in columns
        assert "slot_filler_model_scores" in {
            row[0] for row in mini.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        mini.close()

    print("  PASS: export_import_roundtrip")


def test_model_score_csv_imports_to_mini_db(tmp_path):
    from subtitle_generator.export_db import build_mini_db

    (tmp_path / "slot_fillers.csv").write_text(
        "\n".join([
            "id,slot_type,filler,mode,source_subtitle_id,freq,pos_tag,prep,remix_type,remix_prep,remix_word_count,centroid_dot,norm_sq,token_count,popularity_score,popularity_level,popularity_confidence",
            "1,list_item,Race,strict,,10,,,,,,,,,0.8,1,1.0",
            "2,list_item,Archives,strict,,10,,,,,,,,,0.1,0,1.0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "config.csv").write_text("key,value\n", encoding="utf-8")
    (tmp_path / "sources.csv").write_text(
        "slot_filler_id,title,subtitle_text,source_tag\n",
        encoding="utf-8",
    )
    (tmp_path / "slot_filler_model_scores.csv").write_text(
        "\n".join([
            "slot_filler_id,score_pop,score_mainstream,score_niche,model_tier,source_prediction_count",
            "1,0.9,0.08,0.02,pop,3",
            "2,0.05,0.1,0.85,niche,2",
        ]),
        encoding="utf-8",
    )

    stats = build_mini_db(tmp_path, tmp_path / "mini.db")
    mini = sqlite3.connect(tmp_path / "mini.db")
    row = mini.execute(
        "SELECT score_pop, score_niche, model_tier FROM slot_filler_model_scores WHERE slot_filler_id = 2"
    ).fetchone()
    mini.close()

    assert stats["slot_filler_model_scores"] == 2
    assert row == (0.05, 0.85, "niche")


def test_model_scores_drive_tier_classification_and_generation():
    from subtitle_generator.generate import generate_subtitle_matching_tiers
    from subtitle_generator.tiering import compute_tier_evidence

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
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
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("""
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    next_id = 1
    for slot_type in ("list_item", "action_noun", "of_object"):
        for filler, pop_score, niche_score in (
            ("PopThing", 0.9, 0.05),
            ("NicheThing", 0.05, 0.9),
        ):
            conn.execute(
                """
                INSERT INTO slot_fillers (
                    id, slot_type, filler, mode, freq, popularity_score,
                    popularity_level, popularity_confidence
                )
                VALUES (?, ?, ?, 'strict', 100, 0.5, 1, 1.0)
                """,
                (next_id, slot_type, filler),
            )
            conn.execute(
                "INSERT INTO slot_filler_model_scores VALUES (?, ?, ?, ?, ?, ?)",
                (
                    next_id,
                    pop_score,
                    0.1,
                    niche_score,
                    "pop" if pop_score > niche_score else "niche",
                    1,
                ),
            )
            next_id += 1
    conn.commit()

    pop_evidence = compute_tier_evidence(
        "PopThing, PopThing, and the PopThing of PopThing",
        conn,
    )
    niche_evidence = compute_tier_evidence(
        "NicheThing, NicheThing, and the NicheThing of NicheThing",
        conn,
    )
    generated = generate_subtitle_matching_tiers(
        conn,
        allowed_tiers={"niche"},
        seed=1,
        max_attempts=3,
    )

    assert pop_evidence.tier == "pop"
    assert niche_evidence.tier == "niche"
    assert "NicheThing" in generated.text


def test_configured_tier_classifier_uses_selected_slot_interactions():
    from subtitle_generator.config import invalidate_config_cache
    from subtitle_generator.tiering import compute_tier_evidence

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("""
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    next_id = 1
    for slot_type in ("list_item", "action_noun", "of_object"):
        filler = {
            "list_item": "High Demand",
            "action_noun": "Rise",
            "of_object": "Markets",
        }[slot_type]
        conn.execute(
            """
            INSERT INTO slot_fillers (
                id, slot_type, filler, mode, freq, popularity_score,
                popularity_level, popularity_confidence
            )
            VALUES (?, ?, ?, 'strict', 100, 0.95, 1, 1.0)
            """,
            (next_id, slot_type, filler),
        )
        conn.execute(
            "INSERT INTO slot_filler_model_scores VALUES (?, 0.4, 0.5, 0.1, 'mainstream', 1)",
            (next_id,),
        )
        next_id += 1
    conn.commit()

    subtitle = "High Demand, High Demand, and the Rise of Markets"
    assert compute_tier_evidence(subtitle, conn).tier == "mainstream"

    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        [
            ("tier_classifier_frequency_interaction_pop", "0.25"),
            ("tier_classifier_temperature", "0.5"),
        ],
    )
    invalidate_config_cache()
    frequency_interaction_evidence = compute_tier_evidence(subtitle, conn)
    assert frequency_interaction_evidence.tier == "pop"
    assert 0.0 <= frequency_interaction_evidence.accessibility_score <= 1.0

    conn.executemany(
        "INSERT OR REPLACE INTO config VALUES (?, ?)",
        [
            ("tier_classifier_frequency_interaction_pop", "0"),
            ("tier_classifier_popularity_interaction_pop", "2.0"),
            ("tier_classifier_temperature", "0.5"),
        ],
    )
    invalidate_config_cache()
    assert compute_tier_evidence(subtitle, conn).tier == "pop"


def test_literal_bad_generation_guardrail_rejects_known_artifacts():
    from subtitle_generator.generate import _is_literal_bad_filler

    assert _is_literal_bad_filler("of_object", "Christian")
    assert _is_literal_bad_filler("of_object", "Imf")
    assert _is_literal_bad_filler("list_item", "H.G.W.ells")
    assert _is_literal_bad_filler("of_object", "Con Men, Jr")
    assert not _is_literal_bad_filler("of_object", "Emissions Trading")
    assert not _is_literal_bad_filler("of_object", "Second Indochina War")


def test_literal_bad_guardrail_applies_without_model_scores():
    from subtitle_generator.generate import _load_generation_candidates

    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL
        )
    """)
    conn.executemany(
        """
        INSERT INTO slot_fillers (slot_type, filler, mode, freq, popularity_score)
        VALUES (?, ?, 'strict', 10, 0.5)
        """,
        [
            ("list_item", "Policy"),
            ("action_noun", "Genius"),
            ("of_object", "Christian"),
            ("of_object", "Emissions Trading"),
        ],
    )

    candidates = _load_generation_candidates(conn)

    assert [row[0] for row in candidates.obj_rows] == ["Emissions Trading"]


def test_mini_db_rejects_partial_model_score_coverage(tmp_path):
    from subtitle_generator.export_db import build_mini_db

    (tmp_path / "slot_fillers.csv").write_text(
        "\n".join([
            "id,slot_type,filler,mode,source_subtitle_id,freq,pos_tag,prep,remix_type,remix_prep,remix_word_count,centroid_dot,norm_sq,token_count,popularity_score,popularity_level,popularity_confidence",
            "1,list_item,Race,strict,,10,,,,,,,,,0.8,1,1.0",
            "2,list_item,Archives,strict,,10,,,,,,,,,0.1,0,1.0",
        ]),
        encoding="utf-8",
    )
    (tmp_path / "config.csv").write_text("key,value\n", encoding="utf-8")
    (tmp_path / "sources.csv").write_text(
        "slot_filler_id,title,subtitle_text,source_tag\n",
        encoding="utf-8",
    )
    (tmp_path / "slot_filler_model_scores.csv").write_text(
        "\n".join([
            "slot_filler_id,score_pop,score_mainstream,score_niche,model_tier,source_prediction_count",
            "1,0.9,0.08,0.02,pop,3",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cover every exported slot filler"):
        build_mini_db(tmp_path, tmp_path / "mini.db")


def test_export_data_filters_stale_pattern_sources(tmp_path):
    """Export uses only slot/source rows backed by current valid pattern matches."""
    import csv

    from subtitle_generator.export_db import export_data

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
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
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        );
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            source_file TEXT
        );
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            list_items_json TEXT
        );
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL
        );
        INSERT INTO subtitles VALUES
            (10, 'Stale Source', 'A, B, C, D, and the Making of Noise', 'openlibrary'),
            (11, 'Current Source', 'A, B, and the Making of Signal', 'openlibrary');
        INSERT INTO pattern_matches VALUES
            (100, 10, '["A", "B", "C", "D"]'),
            (101, 11, '["A", "B"]');
        INSERT INTO slot_fillers (
            id, slot_type, filler, mode, source_subtitle_id, freq,
            popularity_score, popularity_level, popularity_confidence
        )
        VALUES
            (1, 'action_noun', 'Noise', 'strict', 10, 1, 0.1, 1, 1.0),
            (2, 'action_noun', 'Signal', 'strict', 10, 2, 0.2, 1, 1.0);
        INSERT INTO slot_filler_sources VALUES
            (1, 10),
            (2, 10),
            (2, 11);
        """
    )
    conn.commit()

    stats = export_data(conn, tmp_path)
    assert stats["slot_fillers.csv"] == 1
    assert stats["sources.csv"] == 1

    with open(tmp_path / "slot_fillers.csv", encoding="utf-8") as f:
        slot_rows = list(csv.DictReader(f))
    assert [row["filler"] for row in slot_rows] == ["Signal"]
    assert slot_rows[0]["source_subtitle_id"] == "11"

    with open(tmp_path / "sources.csv", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    assert source_rows == [{
        "slot_filler_id": "2",
        "title": "Current Source",
        "subtitle_text": "A, B, and the Making of Signal",
        "source_tag": "OL",
    }]


def test_export_data_keeps_title_derived_source_title_only(tmp_path):
    """Title-derived source rows should not export duplicated title/subtitle text."""
    import csv

    from subtitle_generator.export_db import export_data

    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
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
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 1,
            popularity_confidence REAL DEFAULT 1.0,
            UNIQUE(slot_type, filler)
        );
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            source_file TEXT
        );
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            subtitle TEXT,
            list_items_json TEXT,
            candidate_source TEXT
        );
        INSERT INTO subtitles VALUES (
            20,
            'Race, Power, and the Rise of Empire',
            '',
            'openlibrary'
        );
        INSERT INTO pattern_matches VALUES (
            200,
            20,
            'Race, Power, and the Rise of Empire',
            '["Race", "Power"]',
            'title'
        );
        INSERT INTO slot_fillers (
            id, slot_type, filler, mode, source_subtitle_id, freq,
            popularity_score, popularity_level, popularity_confidence
        )
        VALUES (1, 'action_noun', 'Rise', 'strict', 20, 1, 0.2, 1, 1.0);
        """
    )
    conn.commit()

    stats = export_data(conn, tmp_path)

    assert stats["sources.csv"] == 1
    with open(tmp_path / "sources.csv", encoding="utf-8") as f:
        source_rows = list(csv.DictReader(f))
    assert source_rows == [{
        "slot_filler_id": "1",
        "title": "Race, Power, and the Rise of Empire",
        "subtitle_text": "",
        "source_tag": "OL",
    }]


def test_jacket_accessibility_uses_learned_model_scores():
    from subtitle_generator.jacket import compute_accessibility

    conn = _make_test_db()

    tone, score = compute_accessibility(
        "Race, Power, and the Gender of Home", conn
    )
    assert "BOOK TYPE: POP" in tone
    assert score > 0


def test_evidence_aware_tier_classification_uses_runtime_signals():
    """Real-title-like cases use learned slot model scores."""
    from subtitle_generator.tiering import compute_tier_evidence

    conn = _make_test_db()
    for filler in ("Common", "Demand", "Markets"):
        for slot_type in ("list_item", "action_noun", "of_object"):
            conn.execute(
                """
                INSERT OR REPLACE INTO slot_fillers (
                    slot_type, filler, mode, freq, popularity_score,
                    popularity_level, popularity_confidence
                )
                VALUES (?, ?, 'strict', 10000, 0.95, 1, 1.0)
                """,
                (slot_type, filler),
            )
    for filler in ("Rare", "Fallback", "Making", "Archives"):
        for slot_type in ("list_item", "action_noun", "of_object"):
            conn.execute(
                """
                INSERT OR REPLACE INTO slot_fillers (
                    slot_type, filler, mode, freq, popularity_score,
                    popularity_level, popularity_confidence
                )
                VALUES (?, ?, 'strict', 1, 0.1, 0, 0.0)
                """,
                (slot_type, filler),
            )
    conn.commit()

    backed_by_demand = compute_tier_evidence(
        "Common, Demand, and the Rise of Markets", conn
    )
    assert backed_by_demand.tier == "pop"
    assert backed_by_demand.demand_confidence >= 0.75

    specialist = compute_tier_evidence(
        "Rare, Fallback, and the Making of Archives", conn
    )
    assert specialist.tier == "niche"
    assert specialist.lower_tail_score < backed_by_demand.lower_tail_score


def test_remixed_object_tier_evidence_uses_component_scores():
    from subtitle_generator.tiering import compute_tier_evidence

    conn = _make_test_db()
    for slot_type, filler, popularity in (
        ("of_modifier", "Viral", 0.95),
        ("of_head", "Attention", 0.95),
        ("of_topic", "Rare Archives", 0.1),
        ("of_complement", "Footnotes", 0.1),
    ):
        conn.execute(
            """
            INSERT INTO slot_fillers (
                slot_type, filler, mode, freq, popularity_score,
                popularity_level, popularity_confidence
            )
            VALUES (?, ?, 'strict', 10, ?, 1, 1.0)
            """,
            (slot_type, filler, popularity),
        )
    conn.commit()

    pop_evidence = compute_tier_evidence(
        "Race, Power, and the Pursuit of Viral Attention",
        conn,
        remix_parts={"modifier": "Viral", "head": "Attention"},
    )
    niche_evidence = compute_tier_evidence(
        "Race, Power, and the Pursuit of Rare Archives in Footnotes",
        conn,
        remix_parts={
            "topic": "Rare Archives",
            "prep": "in",
            "complement": "Footnotes",
        },
    )

    assert pop_evidence.tier == "pop"
    assert pop_evidence.slots[-1].filler == "Viral Attention"
    assert niche_evidence.slots[-1].filler == "Rare Archives in Footnotes"
    assert niche_evidence.slots[-1].model_score_niche > niche_evidence.slots[-1].model_score_pop


def test_parse_subtitle_slots_rejects_empty_cleaned_fillers():
    from subtitle_generator.tiering import parse_subtitle_slots

    assert parse_subtitle_slots("A, B, and the ... of Space") == []
    assert parse_subtitle_slots("A, B, and the Making of ...") == []
    assert parse_subtitle_slots("A, , and the Making of Space") == []


def test_real_title_tier_metric_uses_db_source_labels():
    """Checked source-title labels come from pattern_matches, not Python constants."""
    from subtitle_generator.tier_diagnostics import (
        evaluate_real_title_tiers,
        load_source_tier_label_cases,
        measure_real_title_tier_pop_guardrail,
    )

    conn = _make_test_db()
    for filler in ("Common", "Demand", "Markets"):
        for slot_type in ("list_item", "action_noun", "of_object"):
            conn.execute(
                """
                INSERT OR REPLACE INTO slot_fillers (
                    slot_type, filler, mode, freq, popularity_score,
                    popularity_level, popularity_confidence
                )
                VALUES (?, ?, 'strict', 10000, 0.95, 1, 1.0)
                """,
                (slot_type, filler),
            )
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL,
            llm_market_tier_rationale TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pattern_matches (
            id, title, subtitle, llm_market_tier, llm_market_tier_confidence,
            llm_market_tier_rationale
        )
        VALUES (
            1, 'Labeled Pop Book', 'Common, Demand, and the Rise of Markets',
            'pop', 1.0, 'LLM-backed checked label.'
        )
        """
    )
    conn.commit()

    cases = load_source_tier_label_cases(conn)
    results = evaluate_real_title_tiers(conn)

    assert [case.title for case in cases] == ["Labeled Pop Book"]
    assert results[0].expected_tier == "pop"
    assert results[0].predicted_tier == "pop"
    assert measure_real_title_tier_pop_guardrail(conn) == 1.0
    print("  PASS: real_title_tier_metric_uses_db_source_labels")


def test_source_tier_candidate_selection_is_seeded_and_resumable():
    from subtitle_generator.source_tier_enrichment import (
        ensure_source_tier_label_columns,
        load_source_tier_candidates,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        [
            (1, "Book A", "Race, Power, and the Rise of Markets"),
            (2, "Book B", "Helmontian Chymistry, Law, and the Making of Europe"),
            (3, "Book C", "Memory, Justice, and the Politics of Archives"),
            (4, "Book D", "Food, Fear, and the Future of America"),
        ],
    )
    ensure_source_tier_label_columns(conn)
    conn.execute("UPDATE pattern_matches SET llm_market_tier = 'pop' WHERE id = 1")
    conn.commit()

    by_id = load_source_tier_candidates(conn, limit=3, selection="id")
    first_random = load_source_tier_candidates(
        conn, limit=3, selection="random", random_seed=7,
    )
    second_random = load_source_tier_candidates(
        conn, limit=3, selection="random", random_seed=7,
    )
    forced = load_source_tier_candidates(conn, limit=4, selection="id", force=True)

    assert [candidate.id for candidate in by_id] == [2, 3, 4]
    assert first_random == second_random
    assert {candidate.id for candidate in first_random} == {2, 3, 4}
    assert [candidate.id for candidate in forced] == [1, 2, 3, 4]


def test_source_tier_candidate_selection_can_prioritize_likely_pop():
    from subtitle_generator.source_tier_enrichment import (
        ensure_source_tier_label_columns,
        load_source_tier_candidates,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER,
            subtitle_id INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            popularity_score REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?, ?, ?)",
        [
            (1, 101, "Book A", "Scholars, Archives, and the Study of Law"),
            (2, 102, "Book B", "Secrets, Power, and the Future of Food"),
            (3, 103, "Book C", "Local Records, Families, and the Work of History"),
        ],
    )
    conn.executemany(
        "INSERT INTO slot_fillers VALUES (?, ?)",
        [
            (11, 0.2),
            (12, 1.9),
            (13, 0.8),
        ],
    )
    conn.executemany(
        "INSERT INTO slot_filler_sources VALUES (?, ?)",
        [
            (11, 101),
            (12, 102),
            (13, 103),
        ],
    )
    ensure_source_tier_label_columns(conn)

    candidates = load_source_tier_candidates(
        conn,
        limit=3,
        selection="likely-pop",
    )

    assert [candidate.id for candidate in candidates] == [2, 3, 1]


def test_source_tier_title_candidates_render_with_blank_subtitle(tmp_path):
    import csv

    from subtitle_generator.source_tier_enrichment import (
        ensure_source_tier_label_columns,
        export_source_tier_labels,
        format_source_tier_distribution_report,
        load_source_tier_distribution,
        load_source_tier_candidates,
    )
    from subtitle_generator.tier_diagnostics import load_source_tier_label_cases

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            title TEXT,
            subtitle TEXT,
            candidate_source TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?, ?, ?, ?)",
        [
            (
                1,
                101,
                "Race, Power, and the Rise of Empire",
                "Race, Power, and the Rise of Empire",
                "title",
            ),
            (
                2,
                102,
                "Book B",
                "Helmontian Chymistry, Law, and the Making of Europe",
                "subtitle",
            ),
        ],
    )
    ensure_source_tier_label_columns(conn)
    conn.execute(
        """
        UPDATE pattern_matches
        SET llm_market_tier = 'pop',
            llm_market_tier_confidence = 0.9,
            llm_market_tier_rationale = 'Checked.'
        WHERE id = 1
        """
    )
    conn.commit()

    candidates = load_source_tier_candidates(
        conn,
        limit=2,
        selection="id",
        force=True,
    )
    assert [(candidate.id, candidate.subtitle) for candidate in candidates] == [
        (1, ""),
        (2, "Helmontian Chymistry, Law, and the Making of Europe"),
    ]
    title_candidates = load_source_tier_candidates(
        conn,
        limit=2,
        selection="id",
        force=True,
        candidate_source="title",
    )
    assert [(candidate.id, candidate.subtitle) for candidate in title_candidates] == [
        (1, "")
    ]

    cases = load_source_tier_label_cases(conn)
    assert [(case.title, case.subtitle) for case in cases] == [
        ("Race, Power, and the Rise of Empire", "")
    ]

    export_path = tmp_path / "source_tier_labels.csv"
    assert export_source_tier_labels(conn, export_path) == 1
    with open(export_path, encoding="utf-8") as f:
        exported = list(csv.DictReader(f))
    assert exported[0]["title"] == "Race, Power, and the Rise of Empire"
    assert exported[0]["subtitle"] == ""

    distribution = load_source_tier_distribution(conn)
    report = format_source_tier_distribution_report(
        distribution,
        min_labeled=2,
    )
    assert "title,1,1,0,1,0,0,1.000,0.000,0.000" in report
    assert "subtitle,1,0,1,0,0,0,0.000,0.000,0.000" in report
    assert "Gate: NEEDS_LABELS" in report


def test_source_tier_distribution_reports_combined_unitary_gate():
    from subtitle_generator.source_tier_enrichment import (
        SourceTierDistributionRow,
        format_source_tier_distribution_report,
    )

    report = format_source_tier_distribution_report(
        (
            SourceTierDistributionRow(
                candidate_source="subtitle",
                total_count=100,
                labeled_count=100,
                unlabeled_count=0,
                pop_count=10,
                mainstream_count=40,
                niche_count=50,
            ),
            SourceTierDistributionRow(
                candidate_source="title",
                total_count=20,
                labeled_count=20,
                unlabeled_count=0,
                pop_count=15,
                mainstream_count=5,
                niche_count=0,
            ),
        ),
        min_labeled=10,
    )

    assert "Gate: SOURCE_TIER_READY" in report
    assert "Combined post-rebuild shares:" in report
    assert "Combined median tier:" in report


def test_source_tier_distribution_combines_labeled_rows_not_total_rows():
    from subtitle_generator.source_tier_enrichment import (
        SourceTierDistributionRow,
        format_source_tier_distribution_report,
    )

    report = format_source_tier_distribution_report(
        (
            SourceTierDistributionRow(
                candidate_source="subtitle",
                total_count=5000,
                labeled_count=4000,
                unlabeled_count=1000,
                pop_count=600,
                mainstream_count=1400,
                niche_count=2000,
            ),
            SourceTierDistributionRow(
                candidate_source="title",
                total_count=10000,
                labeled_count=50,
                unlabeled_count=9950,
                pop_count=15,
                mainstream_count=10,
                niche_count=25,
            ),
        ),
        min_labeled=100,
    )

    assert "Combined post-rebuild shares: pop=0.152, mainstream=0.348, niche=0.500" in report
    assert "Combined rows: total=15000, labeled=4050, unlabeled=10950" in report


def test_source_tier_readiness_reports_tier_deficits():
    from subtitle_generator.source_tier_enrichment import (
        SourceTierDistributionRow,
        SourceTierReadiness,
        format_source_tier_readiness_report,
    )

    readiness = SourceTierReadiness(
        rows=(
            SourceTierDistributionRow(
                candidate_source="subtitle",
                total_count=500,
                labeled_count=150,
                unlabeled_count=350,
                pop_count=5,
                mainstream_count=45,
                niche_count=100,
            ),
            SourceTierDistributionRow(
                candidate_source="title",
                total_count=100,
                labeled_count=20,
                unlabeled_count=80,
                pop_count=1,
                mainstream_count=9,
                niche_count=10,
            ),
        ),
        min_confidence=0.0,
        min_total_labeled=200,
        min_labeled_per_source=50,
        min_labeled_per_tier=25,
    )

    report = format_source_tier_readiness_report(readiness)

    assert "Gate: NEEDS_LABELS" in report
    assert "Total labeled deficit: 30" in report
    assert "title=30" in report
    assert "pop=19" in report
    assert "Recommended next random labeling batch:" in report


def test_source_tier_readiness_ready_when_thresholds_are_met():
    from subtitle_generator.source_tier_enrichment import (
        SourceTierDistributionRow,
        SourceTierReadiness,
        format_source_tier_readiness_report,
    )

    readiness = SourceTierReadiness(
        rows=(
            SourceTierDistributionRow(
                candidate_source="subtitle",
                total_count=200,
                labeled_count=100,
                unlabeled_count=100,
                pop_count=30,
                mainstream_count=30,
                niche_count=40,
            ),
        ),
        min_confidence=0.0,
        min_total_labeled=100,
        min_labeled_per_source=100,
        min_labeled_per_tier=25,
    )

    report = format_source_tier_readiness_report(readiness)

    assert "Gate: SOURCE_TIER_READY" in report
    assert readiness.recommended_random_batch == 0


def test_classify_source_tiers_persists_and_exports_labels(tmp_path):
    import csv

    from subtitle_generator.source_tier_enrichment import (
        SourceTierPrediction,
        classify_source_tiers,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        [
            (1, "Book A", "Race, Power, and the Rise of Markets"),
            (2, "Book B", "Helmontian Chymistry, Law, and the Making of Europe"),
        ],
    )
    conn.commit()

    def fake_classifier(candidates, model):
        return tuple(
            SourceTierPrediction(
                id=candidate.id,
                tier="pop" if candidate.id == 1 else "niche",
                confidence=0.9,
                rationale=f"Classified with {model}. ([example.org](https://example.org/book))",
            )
            for candidate in candidates
        )

    export_path = tmp_path / "source_tier_labels.csv"
    result = classify_source_tiers(
        conn,
        limit=2,
        batch_size=1,
        model="test-model",
        selection="id",
        export_path=export_path,
        classifier=fake_classifier,
    )

    labels = conn.execute(
        """
        SELECT id, llm_market_tier, llm_market_tier_confidence,
               llm_market_tier_rationale
        FROM pattern_matches
        ORDER BY id
        """
    ).fetchall()
    assert result.labeled_count == 2
    assert result.exported_count == 2
    assert labels[0] == (1, "pop", 0.9, "Classified with test-model.")
    assert labels[1][1] == "niche"

    with open(export_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [row["pattern_match_id"] for row in rows] == ["1", "2"]
    assert [row["llm_market_tier"] for row in rows] == ["pop", "niche"]


def test_build_slots_preserves_source_tier_labels_by_subtitle_id(tmp_path, monkeypatch):
    import csv

    from subtitle_generator import slots
    from subtitle_generator.source_tier_enrichment import export_source_tier_labels

    conn = sqlite3.connect(":memory:")
    slots.ensure_slot_tables(conn)
    conn.execute(
        """
        INSERT INTO pattern_matches (
            id, subtitle_id, title, subtitle, list_items_json, action_noun,
            of_object, action_article, of_article, llm_market_tier,
            llm_market_tier_confidence, llm_market_tier_rationale
        )
        VALUES (
            42, 101, 'Old Book A', 'Old Subtitle', '[]', 'Rise',
            'Markets', 'the', '', 'pop', 0.95, 'Already checked.'
        )
        """
    )
    conn.commit()

    extracted_matches = [
        {
            "subtitle_id": 101,
            "title": "Updated Book A",
            "subtitle": "Race, Power, and the Rise of Markets",
            "list_items": ["Race", "Power"],
            "action_noun": "Rise",
            "of_object": "Markets",
            "action_article": "the",
            "of_article": "",
        },
        {
            "subtitle_id": 102,
            "title": "Book B",
            "subtitle": "Memory, Justice, and the Politics of Archives",
            "list_items": ["Memory", "Justice"],
            "action_noun": "Politics",
            "of_object": "Archives",
            "action_article": "the",
            "of_article": "",
        },
    ]
    monkeypatch.setattr(slots, "_load_nlp", lambda: object())
    monkeypatch.setattr(
        slots,
        "extract_pattern_matches",
        lambda conn, rejection_counts=None: extracted_matches,
    )
    monkeypatch.setattr(slots, "_is_valid_action", lambda phrase, nlp: True)
    monkeypatch.setattr(slots, "_is_valid_object", lambda phrase, nlp: True)
    monkeypatch.setattr(slots, "_is_valid_list_item", lambda phrase, nlp: True)
    monkeypatch.setattr(
        slots,
        "_decompose_of_objects",
        lambda conn, nlp, of_objects_seen: None,
    )

    slots.build_slots(conn)

    rows = conn.execute(
        """
        SELECT id, subtitle_id, title, llm_market_tier, llm_market_tier_rationale
        FROM pattern_matches
        ORDER BY subtitle_id
        """
    ).fetchall()
    assert rows[0] == (42, 101, "Updated Book A", "pop", "Already checked.")
    assert rows[1][1:] == (102, "Book B", None, None)

    export_path = tmp_path / "source_tier_labels.csv"
    assert export_source_tier_labels(conn, export_path) == 1
    with open(export_path, encoding="utf-8") as f:
        exported = list(csv.DictReader(f))
    assert exported[0]["subtitle_id"] == "101"
    assert exported[0]["pattern_match_id"] == "42"


def test_build_slots_keeps_existing_labels_when_rebuild_fails(monkeypatch):
    from subtitle_generator import slots

    conn = sqlite3.connect(":memory:")
    slots.ensure_slot_tables(conn)
    conn.execute(
        """
        INSERT INTO pattern_matches (
            id, subtitle_id, title, subtitle, list_items_json, action_noun,
            of_object, action_article, of_article, llm_market_tier,
            llm_market_tier_confidence, llm_market_tier_rationale
        )
        VALUES (
            42, 101, 'Book A', 'Race, Power, and the Rise of Markets', '[]',
            'Rise', 'Markets', 'the', '', 'pop', 0.95, 'Already checked.'
        )
        """
    )
    conn.commit()
    monkeypatch.setattr(
        slots,
        "_load_nlp",
        lambda: (_ for _ in ()).throw(RuntimeError("spaCy unavailable")),
    )

    try:
        slots.build_slots(conn)
    except RuntimeError as exc:
        assert "spaCy unavailable" in str(exc)
    else:
        raise AssertionError("build_slots should propagate rebuild failures")

    row = conn.execute(
        """
        SELECT id, subtitle_id, title, llm_market_tier, llm_market_tier_rationale
        FROM pattern_matches
        """
    ).fetchone()
    assert row == (42, 101, "Book A", "pop", "Already checked.")


def test_classify_source_tiers_dry_run_does_not_migrate_or_export(tmp_path):
    from subtitle_generator.source_tier_enrichment import classify_source_tiers

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        (1, "Book A", "Race, Power, and the Rise of Markets"),
    )
    conn.commit()

    export_path = tmp_path / "source_tier_labels.csv"
    result = classify_source_tiers(
        conn,
        limit=1,
        dry_run=True,
        export_path=export_path,
    )

    columns = {row[1] for row in conn.execute("PRAGMA table_info(pattern_matches)")}
    assert [candidate.id for candidate in result.selected] == [1]
    assert result.labeled_count == 0
    assert result.exported_count == 0
    assert "llm_market_tier" not in columns
    assert not export_path.exists()


def test_classify_source_tiers_exports_when_resume_has_no_unlabeled_rows(tmp_path):
    import csv

    from subtitle_generator.source_tier_enrichment import (
        ensure_source_tier_label_columns,
        classify_source_tiers,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        (1, "Book A", "Race, Power, and the Rise of Markets"),
    )
    ensure_source_tier_label_columns(conn)
    conn.execute(
        """
        UPDATE pattern_matches
        SET llm_market_tier = 'pop',
            llm_market_tier_confidence = 0.95,
            llm_market_tier_rationale = 'Already labeled.'
        WHERE id = 1
        """
    )
    conn.commit()

    export_path = tmp_path / "source_tier_labels.csv"
    result = classify_source_tiers(
        conn,
        limit=10,
        selection="id",
        export_path=export_path,
        classifier=lambda candidates, model: (),
    )

    assert result.selected == ()
    assert result.labeled_count == 0
    assert result.exported_count == 1
    with open(export_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["pattern_match_id"] == "1"
    assert rows[0]["llm_market_tier"] == "pop"


def test_source_tier_prediction_ids_must_match_batch(tmp_path):
    from subtitle_generator.source_tier_enrichment import (
        SourceTierPrediction,
        classify_source_tiers,
    )

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        [
            (1, "Book A", "Race, Power, and the Rise of Markets"),
            (2, "Book B", "Memory, Justice, and the Politics of Archives"),
        ],
    )
    conn.commit()

    def duplicate_classifier(candidates, model):
        return (
            SourceTierPrediction(1, "pop", 0.9, "One."),
            SourceTierPrediction(2, "niche", 0.9, "Two."),
            SourceTierPrediction(2, "mainstream", 0.9, "Duplicate."),
        )

    try:
        classify_source_tiers(
            conn,
            limit=2,
            selection="id",
            export_path=tmp_path / "source_tier_labels.csv",
            classifier=duplicate_classifier,
        )
    except RuntimeError as exc:
        assert "did not match requested ids" in str(exc)
    else:
        raise AssertionError("duplicate LLM labels should be rejected")


def test_classify_source_tiers_dry_run_result_does_not_claim_export(tmp_path):
    from subtitle_generator.source_tier_enrichment import classify_source_tiers

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO pattern_matches VALUES (?, ?, ?)",
        (1, "Book A", "Race, Power, and the Rise of Markets"),
    )
    conn.commit()

    export_path = tmp_path / "source_tier_labels.csv"
    result = classify_source_tiers(
        conn,
        limit=1,
        dry_run=True,
        export_path=export_path,
    )

    assert result.dry_run is True
    assert result.export_path == export_path
    assert result.exported_count == 0
    assert not export_path.exists()


def test_real_title_tier_metric_is_neutral_without_source_labels():
    from subtitle_generator.tier_diagnostics import (
        format_real_title_tier_report,
        measure_real_title_tier_pop_guardrail,
    )

    conn = _make_test_db()

    assert measure_real_title_tier_pop_guardrail(conn) == 1.0
    assert "No `pattern_matches.llm_market_tier` labels found." in (
        format_real_title_tier_report(conn)
    )
    print("  PASS: real_title_tier_metric_is_neutral_without_source_labels")


def test_tier_label_guardrail_blend_is_noop_without_labels():
    from subtitle_generator.tune import _blend_real_title_tier_guardrail

    assert _blend_real_title_tier_guardrail(
        output_comp=0.42,
        label_guardrail=1.0,
        label_count=0,
        label_weight=0.25,
    ) == 0.42
    assert math.isclose(_blend_real_title_tier_guardrail(
        output_comp=0.42,
        label_guardrail=0.8,
        label_count=10,
        label_weight=0.25,
    ), 0.515)
    print("  PASS: tier_label_guardrail_blend_is_noop_without_labels")


def test_config_params_exist():
    """Live popularity params are in ALL_TUNABLE_PARAMS."""
    from subtitle_generator.config import ALL_TUNABLE_PARAMS

    expected = [
        "pop_weight_spl", "pop_weight_ol", "pop_weight_gr",
        "pop_weight_nyt", "pop_weight_library", "pop_weight_trove",
        "pop_missing_default",
    ]
    for param in expected:
        assert param in ALL_TUNABLE_PARAMS, f"Missing param: {param}"
    assert "pop_classification_blend" not in ALL_TUNABLE_PARAMS
    print("  PASS: config_params_exist")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("config_params_exist", test_config_params_exist),
        ("weighted_sample_defaults_to_frequency_without_model_tier", test_weighted_sample_defaults_to_frequency_without_model_tier),
        ("weighted_sample_uses_learned_tier_scores_when_requested", test_weighted_sample_uses_learned_tier_scores_when_requested),
        ("weighted_sample_handles_null_popularity_when_model_scores_exist", test_weighted_sample_handles_null_popularity_when_model_scores_exist),
        ("removed_popularity_blends_are_not_tunable_config", test_removed_popularity_blends_are_not_tunable_config),
        ("export_import_roundtrip", test_export_import_roundtrip),
        ("jacket_accessibility_uses_learned_model_scores", test_jacket_accessibility_uses_learned_model_scores),
        ("evidence_aware_tier_classification_uses_runtime_signals", test_evidence_aware_tier_classification_uses_runtime_signals),
        ("remixed_object_tier_evidence_uses_component_scores", test_remixed_object_tier_evidence_uses_component_scores),
        ("parse_subtitle_slots_rejects_empty_cleaned_fillers", test_parse_subtitle_slots_rejects_empty_cleaned_fillers),
        ("real_title_tier_metric_uses_db_source_labels", test_real_title_tier_metric_uses_db_source_labels),
        ("real_title_tier_metric_is_neutral_without_source_labels", test_real_title_tier_metric_is_neutral_without_source_labels),
        ("tier_label_guardrail_blend_is_noop_without_labels", test_tier_label_guardrail_blend_is_noop_without_labels),
    ]

    print(f"=== Popularity Scoring Tests ({len(tests)} tests) ===\n")
    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            # Reset config cache between tests
            from subtitle_generator.config import invalidate_config_cache
            invalidate_config_cache()
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"All {passed} tests passed ✓")
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
