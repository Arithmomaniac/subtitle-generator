"""Tests for the human feedback system.

Uses an in-memory SQLite DB and a seeded random "robot grader" to simulate
human review interactions without distorting real data. The grader's
decisions are deterministic via seed, so tests are reproducible.

Run:  uv run python tests/test_feedback.py
"""

import json
import random
import sqlite3
import sys
from pathlib import Path
from unittest.mock import patch

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_test_db() -> sqlite3.Connection:
    """Create an in-memory DB with the minimal tables needed for feedback."""
    conn = sqlite3.connect(":memory:")
    # Config table (needed by load_tuning_config)
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    # Slot fillers (needed by compute_accessibility via _lookup_freq)
    conn.execute("""
        CREATE TABLE slot_fillers (
            filler TEXT, slot_type TEXT, freq INTEGER,
            popularity_score REAL,
            PRIMARY KEY (filler, slot_type)
        )
    """)
    # Insert some fillers with known frequencies for tone testing
    fillers = [
        # Pop fillers (high freq)
        ("Race", "list_item", 50), ("Power", "list_item", 45),
        ("Pursuit", "action_noun", 30), ("Happiness", "of_object", 25),
        # Niche fillers (low freq)
        ("Vasubandhu", "list_item", 1), ("Samizdat", "list_item", 1),
        ("Reinvention", "action_noun", 2), ("Transcendence", "of_object", 1),
        # Mainstream fillers (mid freq)
        ("Politics", "list_item", 8), ("Business", "list_item", 7),
        ("Meaning", "action_noun", 6), ("Community", "of_object", 5),
    ]
    conn.executemany(
        "INSERT INTO slot_fillers (filler, slot_type, freq) VALUES (?, ?, ?)",
        fillers,
    )
    conn.commit()
    return conn


class RobotGrader:
    """Deterministic simulated human reviewer using seeded randomness.

    Given a subtitle, decides thumbs up/down and tone override based on
    simple heuristics + randomness. The seed makes tests reproducible.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self._call_count = 0

    def grade(self, subtitle: str) -> dict:
        """Return a dict with thumbs, tone_override, and optional comment."""
        self._call_count += 1

        # Simple heuristic: common words → thumbs up, obscure → thumbs down
        common_words = {"Race", "Power", "Love", "War", "Politics", "Business",
                        "Pursuit", "Happiness", "End", "Empire", "Meaning"}
        words = set(subtitle.replace(",", "").replace("the", "").split())
        common_count = len(words & common_words)

        # Thumbs: more common words → higher chance of approval
        thumbs_prob = min(0.9, 0.3 + common_count * 0.15)
        thumbs = 1 if self.rng.random() < thumbs_prob else -1

        # Tone override: guess based on word familiarity
        if common_count >= 3:
            tone = "pop"
        elif common_count >= 1:
            tone = self.rng.choice(["mainstream", "pop"])
        else:
            tone = self.rng.choice(["niche", "mainstream"])

        # Occasional comment (~30% of the time)
        comment = None
        if self.rng.random() < 0.3:
            comments = [
                "feels very airport bookstore",
                "too academic for my taste",
                "great word pairing!",
                "this one is boring",
                "love the surprise factor",
                "sounds like a real book",
            ]
            comment = self.rng.choice(comments)

        return {"thumbs": thumbs, "tone_override": tone, "comment": comment}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_store_and_retrieve():
    """store_rating() writes to DB and get_summary() reads it back."""
    from subtitle_generator.feedback import ensure_ratings_table, get_summary, store_rating

    conn = make_test_db()
    ensure_ratings_table(conn)

    # Store below threshold (10) — summary should be None
    for i in range(5):
        store_rating(conn, f"Subtitle {i}", system_tone="pop", thumbs=1)

    summary = get_summary(conn)
    assert summary is None, f"Expected None below threshold, got {summary}"

    # Store enough to hit threshold
    for i in range(5, 12):
        store_rating(conn, f"Subtitle {i}", system_tone="niche", thumbs=-1)

    summary = get_summary(conn)
    assert summary is not None, "Expected summary above threshold"
    assert summary["total_ratings"] == 12
    assert summary["thumbs_rated"] == 12
    # 5 up, 7 down → ~42% approval
    assert 0.4 <= summary["approval_rate"] <= 0.45, f"Unexpected approval: {summary['approval_rate']}"

    print("  PASS: store_and_retrieve")
    conn.close()


def test_tone_accuracy():
    """Summary correctly tracks tone match/mismatch rates."""
    from subtitle_generator.feedback import ensure_ratings_table, get_summary, store_rating

    conn = make_test_db()
    ensure_ratings_table(conn)

    # 6 matches, 4 mismatches, 2 no override
    for i in range(6):
        store_rating(conn, f"Sub {i}", system_tone="pop", thumbs=1, tone_override="pop")
    for i in range(4):
        store_rating(conn, f"Sub {6 + i}", system_tone="pop", thumbs=-1, tone_override="niche")
    for i in range(2):
        store_rating(conn, f"Sub {10 + i}", system_tone="mainstream", thumbs=1)

    summary = get_summary(conn)
    assert summary is not None
    assert summary["tone_pairs"] == 10  # only the ones with both tones
    assert summary["tone_accuracy"] == 0.6  # 6/10
    assert "system=pop but human=niche" in summary["tone_mismatches"]
    assert summary["tone_mismatches"]["system=pop but human=niche"] == 4

    print("  PASS: tone_accuracy")
    conn.close()


def test_format_summary():
    """format_summary_for_proposer() produces readable text."""
    from subtitle_generator.feedback import format_summary_for_proposer

    summary = {
        "total_ratings": 25,
        "thumbs_rated": 20,
        "approval_rate": 0.75,
        "tone_pairs": 15,
        "tone_accuracy": 0.8,
        "tone_mismatches": {"system=niche but human=mainstream": 3},
        "recent_comments": ["too generic", "love this one"],
        "interpreted_insights": ["Consider raising pop thresholds"],
    }

    text = format_summary_for_proposer(summary)
    assert "75%" in text
    assert "80%" in text
    assert "3×" in text
    assert "too generic" in text
    assert "raising pop thresholds" in text

    print("  PASS: format_summary")


def test_robot_grader_batch():
    """Simulate a review session using the robot grader. Verify data integrity."""
    from subtitle_generator.feedback import ensure_ratings_table, get_summary, store_rating

    conn = make_test_db()
    ensure_ratings_table(conn)
    grader = RobotGrader(seed=42)

    subtitles = [
        "Race, Power, and the Pursuit of Happiness",
        "Vasubandhu, Samizdat, and the Reinvention of Transcendence",
        "Politics, Business, and the Meaning of Community",
        "Love, War, and the End of Empire",
        "Race, Power, and the End of Happiness",
        "Vasubandhu, Samizdat, and the Meaning of Community",
        "Politics, Power, and the Pursuit of Transcendence",
        "Love, Business, and the Reinvention of Empire",
        "Race, Business, and the Meaning of Happiness",
        "Politics, War, and the Pursuit of Community",
        "Love, Power, and the End of Transcendence",
        "Vasubandhu, War, and the Reinvention of Happiness",
    ]

    system_tones = []
    grader_tones = []

    for sub in subtitles:
        # Compute system tone
        from subtitle_generator.jacket import compute_accessibility
        _, score = compute_accessibility(sub, conn)
        from subtitle_generator.config import load_tuning_config
        cfg = load_tuning_config(conn)
        if score > cfg["accessibility_threshold_pop"]:
            sys_tone = "pop"
        elif score >= cfg["accessibility_threshold_mainstream"]:
            sys_tone = "mainstream"
        else:
            sys_tone = "niche"
        system_tones.append(sys_tone)

        # Robot grader decides
        grade = grader.grade(sub)
        grader_tones.append(grade["tone_override"])

        store_rating(
            conn, sub,
            system_tone=sys_tone,
            thumbs=grade["thumbs"],
            tone_override=grade["tone_override"],
            free_text=grade["comment"],
        )

    summary = get_summary(conn)
    assert summary is not None
    assert summary["total_ratings"] == 12
    assert summary["thumbs_rated"] == 12
    assert 0.0 <= summary["approval_rate"] <= 1.0
    assert summary["tone_pairs"] == 12

    # Verify determinism: same seed = same results
    grader2 = RobotGrader(seed=42)
    grades2 = [grader2.grade(sub) for sub in subtitles]
    grader3 = RobotGrader(seed=42)
    grades3 = [grader3.grade(sub) for sub in subtitles]
    for g2, g3 in zip(grades2, grades3):
        assert g2 == g3, "Robot grader is not deterministic!"

    # Different seed = different results
    grader4 = RobotGrader(seed=99)
    grades4 = [grader4.grade(sub) for sub in subtitles]
    differs = sum(1 for g2, g4 in zip(grades2, grades4) if g2["thumbs"] != g4["thumbs"])
    assert differs > 0, "Different seeds should produce different results"

    print(f"  PASS: robot_grader_batch (approval={summary['approval_rate']:.0%}, "
          f"tone_accuracy={summary['tone_accuracy']:.0%}, "
          f"{len(summary.get('tone_mismatches', {}))} mismatch patterns)")
    conn.close()


def test_cli_review_mock():
    """Test _prompt_review with mocked click.prompt using robot grader."""
    from subtitle_generator.feedback import ensure_ratings_table, get_summary

    conn = make_test_db()
    ensure_ratings_table(conn)
    grader = RobotGrader(seed=123)

    subtitles = [
        "Race, Power, and the Pursuit of Happiness",
        "Politics, Business, and the Meaning of Community",
        "Vasubandhu, Samizdat, and the Reinvention of Transcendence",
    ]

    # Mock click.prompt to feed robot grader responses
    prompt_responses = []
    for sub in subtitles:
        grade = grader.grade(sub)
        # Simulate: thumbs prompt, tone prompt, comment prompt
        thumbs_str = "y" if grade["thumbs"] == 1 else "n"
        tone_str = grade["tone_override"][0]  # p, m, or n
        comment_str = grade["comment"] or ""
        prompt_responses.extend([thumbs_str, tone_str, comment_str])

    call_idx = [0]

    def mock_prompt(text, **kwargs):
        idx = call_idx[0]
        call_idx[0] += 1
        if idx < len(prompt_responses):
            return prompt_responses[idx]
        return ""

    # Import and test _prompt_review
    from subtitle_generator.cli import _prompt_review

    results = []
    with patch("click.prompt", side_effect=mock_prompt):
        for sub in subtitles:
            result = _prompt_review(conn, sub)
            results.append(result)

    # Verify results stored
    row_count = conn.execute("SELECT COUNT(*) FROM human_ratings").fetchone()[0]
    assert row_count == 3, f"Expected 3 ratings, got {row_count}"

    # Verify thumbs values match what we fed
    rows = conn.execute(
        "SELECT subtitle, thumbs, tone_override FROM human_ratings ORDER BY id"
    ).fetchall()

    grader2 = RobotGrader(seed=123)
    for row, sub in zip(rows, subtitles):
        grade = grader2.grade(sub)
        expected_thumbs = grade["thumbs"]
        assert row[1] == expected_thumbs, f"Thumbs mismatch for {sub}: {row[1]} vs {expected_thumbs}"
        assert row[2] == grade["tone_override"], f"Tone mismatch for {sub}"

    print(f"  PASS: cli_review_mock ({len(results)} reviews simulated)")
    conn.close()


# spot_check_schedule and spot_check_scoring tests removed:
# spot-check is now a standalone command, not part of the tune loop.


def test_idempotent_table_creation():
    """ensure_ratings_table() can be called multiple times safely."""
    from subtitle_generator.feedback import ensure_ratings_table, store_rating

    conn = make_test_db()
    ensure_ratings_table(conn)
    ensure_ratings_table(conn)  # second call should not error
    ensure_ratings_table(conn)  # third call

    # Still works
    row_id = store_rating(conn, "test subtitle", thumbs=1)
    assert row_id == 1

    print("  PASS: idempotent_table_creation")
    conn.close()


def test_empty_summary():
    """get_summary() returns None when no ratings exist."""
    from subtitle_generator.feedback import ensure_ratings_table, get_summary

    conn = make_test_db()
    ensure_ratings_table(conn)
    assert get_summary(conn) is None

    print("  PASS: empty_summary")
    conn.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("store_and_retrieve", test_store_and_retrieve),
        ("tone_accuracy", test_tone_accuracy),
        ("format_summary", test_format_summary),
        ("robot_grader_batch", test_robot_grader_batch),
        ("cli_review_mock", test_cli_review_mock),
        ("idempotent_table_creation", test_idempotent_table_creation),
        ("empty_summary", test_empty_summary),
    ]

    print(f"=== Feedback System Tests ({len(tests)} tests) ===\n")
    passed = 0
    failed = 0

    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED ✓")
    else:
        print(f"{passed} passed, {failed} FAILED")
        sys.exit(1)
