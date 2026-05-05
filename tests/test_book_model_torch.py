import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("torch", reason="book model torch tests require the ml extra")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_train_book_tier_torch_writes_predictions_and_report(tmp_path):
    from subtitle_generator.book_model_baseline import (
        NUMERIC_FEATURES,
        TEXT_FEATURE_COLUMNS,
    )
    from subtitle_generator.book_model_torch import train_book_tier_torch

    feature_fields = (
        "pattern_match_id",
        "subtitle_id",
        "candidate_source",
        "source_group",
        *NUMERIC_FEATURES,
        *TEXT_FEATURE_COLUMNS,
    )
    label_fields = (
        "pattern_match_id",
        "subtitle_id",
        "label_target",
        "label_confidence",
    )
    feature_rows = []
    label_rows = []
    text_by_tier = {
        "pop": ("Bestseller", "A famous hardcover trade edition"),
        "mainstream": ("Readable Guide", "A paperback subject overview"),
        "niche": ("Catalog Notes", "A library monograph"),
    }
    for index, tier in enumerate(("pop", "mainstream", "niche") * 5, start=1):
        title, metadata_subject = text_by_tier[tier]
        row = {
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "candidate_source": "subtitle",
            "source_group": "OL",
            "title": title,
            "subtitle_text": metadata_subject,
            "action_noun": "Rise",
            "of_object": "Markets",
            "list_items_text": "Race | Power",
            "list_item_pair_text": "Race || Power",
            "action_object_pair_text": "Rise || Markets",
            "slot_frame_text": "Race || Power || Rise || Markets",
            "metadata_publisher_text": "Trade Books",
            "metadata_edition_name": "First edition",
            "metadata_physical_format": "Hardcover" if tier == "pop" else "Paperback",
            "metadata_physical_dimensions": "",
            "metadata_physical_description": "",
            "metadata_subject_text": metadata_subject,
            "metadata_loc_call_number": "",
            "metadata_dewey_decimal": "",
        }
        for feature_name in NUMERIC_FEATURES:
            row[feature_name] = "0"
        row["has_isbn"] = "1"
        row["has_work_key"] = "1"
        row["has_popularity_data"] = "1"
        row["work_popularity_score"] = {
            "pop": "1.8",
            "mainstream": "0.9",
            "niche": "0.1",
        }[tier]
        row["metadata_is_hardcover"] = "1" if tier == "pop" else "0"
        row["metadata_is_paperback"] = "1" if tier == "mainstream" else "0"
        row["has_physical_format_metadata"] = "1"
        row["has_subject_metadata"] = "1"
        feature_rows.append(row)
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

    result = train_book_tier_torch(
        features_path=features_path,
        labels_path=labels_path,
        output_dir=tmp_path,
        epochs=10,
        hidden_dim=8,
        hash_dim=64,
        device="cpu",
    )

    with open(result.predictions_path, encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    report = result.report_path.read_text(encoding="utf-8")

    assert result.prediction_count == 15
    assert result.labeled_count == 15
    assert result.validation_count == 3
    assert {row["predicted_tier"] for row in predictions} <= {"pop", "mainstream", "niche"}
    assert "gradient-descent" in report
    assert "Training device: `cpu`" in report
