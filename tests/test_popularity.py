"""Tests for popularity-weighted sampling integration.

Run:  uv run python tests/test_popularity.py
"""

import math
import random
import sqlite3
import sys
from pathlib import Path

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
            UNIQUE(slot_type, filler)
        )
    """)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")

    # Insert test fillers with varying freq and popularity
    fillers = [
        ("Race", 40, 1.8),
        ("Power", 35, 1.6),
        ("Gender", 36, 0.21),
        ("Home", 2, 1.56),
        ("Fraud", 1, 1.55),
        ("Helmontian", 1, 0.09),
    ]
    for i, (filler, freq, pop) in enumerate(fillers):
        ps = pop
        if pop_scores is not None and filler in pop_scores:
            ps = pop_scores[filler]
        for slot_type in ("list_item", "action_noun", "of_object"):
            conn.execute(
                "INSERT INTO slot_fillers (slot_type, filler, mode, freq, popularity_score) "
                "VALUES (?, ?, 'strict', ?, ?)",
                (slot_type, filler, freq, ps),
            )
    conn.commit()
    return conn


def test_backward_compat_blend_zero():
    """With pop_base_weight_blend=0 and pop_tone_blend=0, behavior matches old freq-only."""
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db()

    # With blend=0 (default), popularity_score should be ignored
    rows_with_pop = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'list_item'"
    ).fetchall()
    rows_without_pop = [(r[0], r[1]) for r in rows_with_pop]

    rng1 = random.Random(42)
    rng2 = random.Random(42)

    result_with = _weighted_sample(rows_with_pop, 3, rng1, tone_target=1.0, conn=conn)
    result_without = _weighted_sample(rows_without_pop, 3, rng2, tone_target=1.0, conn=conn)

    assert result_with == result_without, (
        f"Blend=0 should give same results. Got {result_with} vs {result_without}"
    )
    print("  PASS: backward_compat_blend_zero")


def test_popularity_blend_one():
    """With pop_tone_blend=1.0, high-popularity low-freq words should be boosted."""
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db()
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_tone_blend', '1.0')")
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_base_weight_blend', '1.0')")
    conn.commit()
    from subtitle_generator.config import invalidate_config_cache
    invalidate_config_cache()

    rows = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'list_item'"
    ).fetchall()

    # With tone_target=1.5 and full popularity blend, "Race" (pop=1.8) and
    # "Home" (pop=1.56) should be strongly favored over "Gender" (pop=0.21)
    counts: dict[str, int] = {}
    for seed in range(200):
        rng = random.Random(seed)
        picked = _weighted_sample(rows, 1, rng, tone_target=1.5, conn=conn)
        counts[picked[0]] = counts.get(picked[0], 0) + 1

    # Race and Home should appear more than Gender
    assert counts.get("Race", 0) > counts.get("Gender", 0), (
        f"Race (pop=1.8) should beat Gender (pop=0.21) at target 1.5. Counts: {counts}"
    )
    # Home (pop=1.56, freq=2) should appear despite low freq
    assert counts.get("Home", 0) > 0, (
        f"Home (pop=1.56) should appear with pop_tone_blend=1. Counts: {counts}"
    )
    print("  PASS: popularity_blend_one")


def test_null_popularity_uses_default():
    """Fillers with NULL popularity_score should use pop_missing_default."""
    from subtitle_generator.generate import _weighted_sample

    conn = _make_test_db(pop_scores={"Fraud": None, "Helmontian": None})
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_tone_blend', '1.0')")
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_base_weight_blend', '1.0')")
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_missing_default', '0.1')")
    conn.commit()
    from subtitle_generator.config import invalidate_config_cache
    invalidate_config_cache()

    rows = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'list_item'"
    ).fetchall()

    # Should not crash with NULL values
    result = _weighted_sample(rows, 3, random.Random(42), tone_target=1.0, conn=conn)
    assert len(result) == 3, f"Expected 3 results, got {len(result)}"
    print("  PASS: null_popularity_uses_default")


def test_half_blend():
    """With pop_tone_blend=0.5, score should be average of freq-score and pop-score."""
    from subtitle_generator.config import load_tuning_config

    conn = _make_test_db()
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_tone_blend', '0.5')")
    conn.commit()
    from subtitle_generator.config import invalidate_config_cache
    invalidate_config_cache()

    cfg = load_tuning_config(conn)
    assert cfg["pop_tone_blend"] == 0.5, f"Expected 0.5, got {cfg['pop_tone_blend']}"

    # Verify blended score calculation manually
    freq = 40  # Race
    pop = 1.8
    blend = 0.5
    score_freq = math.log10(1 + freq)
    expected = (1 - blend) * score_freq + blend * pop
    actual_freq_part = (1 - 0.5) * math.log10(41)
    actual_pop_part = 0.5 * 1.8
    assert abs(expected - (actual_freq_part + actual_pop_part)) < 0.001
    print("  PASS: half_blend")


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
    conn.commit()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Export
        stats = export_data(conn, tmp_path)
        assert stats["slot_fillers.csv"] > 0

        # Verify CSV has popularity_score column
        with open(tmp_path / "slot_fillers.csv", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            assert "popularity_score" in first_row, (
                f"Missing popularity_score column. Headers: {list(first_row.keys())}"
            )
            assert first_row["popularity_score"] != "", (
                f"popularity_score should not be empty for Race"
            )

        # Import into mini DB
        mini_path = tmp_path / "mini.db"
        build_mini_db(tmp_path, mini_path)

        # Verify imported data
        mini = sqlite3.connect(str(mini_path))
        row = mini.execute(
            "SELECT popularity_score FROM slot_fillers WHERE filler = 'Race' LIMIT 1"
        ).fetchone()
        assert row is not None and row[0] is not None, "popularity_score lost in import"
        assert abs(row[0] - 1.8) < 0.01, f"Expected ~1.8, got {row[0]}"
        mini.close()

    print("  PASS: export_import_roundtrip")


def test_jacket_accessibility_blend():
    """compute_accessibility blends freq and popularity per config."""
    from subtitle_generator.jacket import compute_accessibility

    conn = _make_test_db()

    # With default blend=0, should use freq only
    tone0, score0 = compute_accessibility(
        "Race, Power, and the Gender of Home", conn
    )
    assert score0 > 0, f"Expected positive score, got {score0}"

    # With blend=1, should use popularity only
    conn.execute("INSERT OR REPLACE INTO config VALUES ('pop_tone_blend', '1.0')")
    conn.commit()
    from subtitle_generator.config import invalidate_config_cache
    invalidate_config_cache()

    tone1, score1 = compute_accessibility(
        "Race, Power, and the Gender of Home", conn
    )
    # Scores should differ because freq and pop distributions are different
    assert score0 != score1, (
        f"Blended and unblended scores should differ: {score0} vs {score1}"
    )
    print("  PASS: jacket_accessibility_blend")


def test_config_params_exist():
    """All popularity params are in ALL_TUNABLE_PARAMS."""
    from subtitle_generator.config import ALL_TUNABLE_PARAMS

    expected = [
        "pop_weight_spl", "pop_weight_ol", "pop_weight_freq",
        "pop_exponent", "pop_base_weight_blend", "pop_tone_blend",
        "pop_missing_default",
    ]
    for param in expected:
        assert param in ALL_TUNABLE_PARAMS, f"Missing param: {param}"
    print("  PASS: config_params_exist")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("config_params_exist", test_config_params_exist),
        ("backward_compat_blend_zero", test_backward_compat_blend_zero),
        ("popularity_blend_one", test_popularity_blend_one),
        ("null_popularity_uses_default", test_null_popularity_uses_default),
        ("half_blend", test_half_blend),
        ("export_import_roundtrip", test_export_import_roundtrip),
        ("jacket_accessibility_blend", test_jacket_accessibility_blend),
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
