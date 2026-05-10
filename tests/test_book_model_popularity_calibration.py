import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_calibrate_popularity_weights_learns_predictive_source(tmp_path):
    from subtitle_generator.book_model_popularity_calibration import (
        calibrate_popularity_weights,
    )

    feature_fields = (
        "pattern_match_id",
        "title",
        "subtitle_text",
        "work_popularity_score",
        "checkouts_per_year",
        "gr_ratings_count",
        "library_appearances",
        "trove_library_count",
        "nyt_weeks_on_list",
        "ol_edition_count",
    )
    prediction_fields = (
        "pattern_match_id",
        "score_pop",
        "score_mainstream",
        "score_niche",
    )
    feature_rows = []
    prediction_rows = []
    for index in range(1, 11):
        feature_rows.append({
            "pattern_match_id": str(index),
            "title": f"Title {index}",
            "subtitle_text": "",
            "work_popularity_score": "0",
            "checkouts_per_year": str(index * 10),
            "gr_ratings_count": str((11 - index) * 10),
            "library_appearances": "1",
            "trove_library_count": "1",
            "nyt_weeks_on_list": "0",
            "ol_edition_count": "1",
        })
        pop_score = index / 10
        prediction_rows.append({
            "pattern_match_id": str(index),
            "score_pop": f"{pop_score:.3f}",
            "score_mainstream": "0",
            "score_niche": f"{1 - pop_score:.3f}",
        })

    features_path = tmp_path / "book_features.csv"
    teacher_path = tmp_path / "teacher.csv"
    db_path = tmp_path / "subtitles.db"
    _write_csv(features_path, feature_fields, feature_rows)
    _write_csv(teacher_path, prediction_fields, prediction_rows)

    result = calibrate_popularity_weights(
        features_path=features_path,
        teacher_predictions_path=teacher_path,
        output_dir=tmp_path / "calibration",
        db_path=db_path,
        apply=True,
        target_mode="pop-only",
        regularization=0.0,
        min_weight_share=0.0,
    )

    assert result.learned_mse < result.current_mse
    assert result.learned_weights["spl"] > result.learned_weights["goodreads"]
    assert result.report_path.exists()

    conn = sqlite3.connect(db_path)
    try:
        rows = dict(conn.execute("SELECT key, value FROM config").fetchall())
    finally:
        conn.close()
    assert "pop_weight_spl" in rows
    assert float(rows["pop_weight_spl"]) == round(result.learned_weights["spl"], 8)
    assert "pop_weight_ol" not in rows
