import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_train_book_tier_baseline_writes_predictions_and_report(tmp_path):
    from subtitle_generator.book_model_baseline import (
        NUMERIC_FEATURES,
        train_book_tier_baseline,
    )

    feature_fields = (
        "pattern_match_id",
        "subtitle_id",
        "candidate_source",
        "source_group",
        *NUMERIC_FEATURES,
    )
    label_fields = (
        "pattern_match_id",
        "subtitle_id",
        "label_target",
        "label_confidence",
    )
    feature_rows = []
    label_rows = []
    for index, tier in enumerate(("pop", "mainstream", "niche") * 5, start=1):
        if tier == "pop":
            work_popularity = "1.8"
            max_filler = "1.7"
        elif tier == "mainstream":
            work_popularity = "0.9"
            max_filler = "0.9"
        else:
            work_popularity = "0.1"
            max_filler = "0.2"
        feature_rows.append({
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "candidate_source": "subtitle",
            "source_group": "OL",
            "has_isbn": "1",
            "has_lccn": "0",
            "has_work_key": "1",
            "has_popularity_data": "1",
            "work_popularity_score": work_popularity,
            "slot_source_link_count": "2",
            "distinct_strict_filler_count": "2",
            "max_filler_popularity_score": max_filler,
            "avg_filler_popularity_score": max_filler,
            "title_length_chars": "20",
            "subtitle_length_chars": "50",
            "list_item_count": "2",
        })
        label_rows.append({
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "label_target": tier,
            "label_confidence": "0.9",
        })
    features_path = tmp_path / "book_features.csv"
    labels_path = tmp_path / "book_labels.csv"
    _write_csv(features_path, feature_fields, feature_rows)
    _write_csv(labels_path, label_fields, label_rows)

    result = train_book_tier_baseline(
        features_path=features_path,
        labels_path=labels_path,
        output_dir=tmp_path,
    )

    with open(result.predictions_path, encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    report = result.report_path.read_text(encoding="utf-8")

    assert result.prediction_count == 15
    assert result.labeled_count == 15
    assert result.validation_count == 3
    assert {row["predicted_tier"] for row in predictions} <= {"pop", "mainstream", "niche"}
    assert "Macro accuracy" in report
