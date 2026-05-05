import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _create_shadow_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT,
            filler TEXT,
            mode TEXT,
            freq INTEGER,
            popularity_score REAL,
            popularity_level TEXT
        )
        """
    )
    conn.execute("CREATE TABLE slot_filler_sources (slot_filler_id INTEGER, subtitle_id INTEGER)")
    conn.executemany(
        "INSERT INTO slot_fillers VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "list_item", "Race", "strict", 10, 0.4, "mainstream"),
            (2, "list_item", "Power", "strict", 8, 0.3, "niche"),
            (3, "list_item", "Markets", "strict", 8, 0.8, "pop"),
            (4, "action_noun", "Rise", "strict", 7, 0.4, "mainstream"),
            (5, "of_object", "Empire", "strict", 6, 0.2, "niche"),
        ],
    )
    conn.executemany(
        "INSERT INTO slot_filler_sources VALUES (?, ?)",
        [(1, 101), (2, 102), (3, 103), (4, 101), (5, 102)],
    )
    conn.commit()
    return conn


def _write_predictions(path: Path) -> None:
    fieldnames = (
        "pattern_match_id",
        "subtitle_id",
        "predicted_tier",
        "prediction_confidence",
        "score_pop",
        "score_mainstream",
        "score_niche",
    )
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([
            {
                "pattern_match_id": "1",
                "subtitle_id": "101",
                "predicted_tier": "pop",
                "prediction_confidence": "0.9",
                "score_pop": "0.9",
                "score_mainstream": "0.1",
                "score_niche": "0.0",
            },
            {
                "pattern_match_id": "2",
                "subtitle_id": "102",
                "predicted_tier": "niche",
                "prediction_confidence": "0.8",
                "score_pop": "0.1",
                "score_mainstream": "0.1",
                "score_niche": "0.8",
            },
        ])


def test_build_shadow_rollups_writes_rollup_and_report(tmp_path: Path):
    from subtitle_generator.book_model_shadow import ShadowInput, build_shadow_rollups

    conn = _create_shadow_db()
    predictions_path = tmp_path / "predictions.csv"
    _write_predictions(predictions_path)

    result = build_shadow_rollups(
        conn,
        output_dir=tmp_path,
        prediction_inputs=(ShadowInput("student", predictions_path),),
        sample_count=2,
        random_seed=1,
    )

    with open(result.rollup_paths[0], encoding="utf-8") as handle:
        rollups = list(csv.DictReader(handle))
    report = result.report_path.read_text(encoding="utf-8")

    assert len(rollups) == 5
    assert rollups[0]["book_model_tier"] == "pop"
    assert "Fixed-seed shadow samples" in report
    assert "student" in report
