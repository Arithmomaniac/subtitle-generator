import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("torch", reason="book model distillation tests require the ml extra")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_distill_exportable_book_model_writes_report(tmp_path):
    from subtitle_generator.book_model_baseline import TEXT_FEATURE_COLUMNS
    from subtitle_generator.book_model_distillation import distill_exportable_book_model
    from subtitle_generator.book_model_torch import EXPORT_SLOT_NUMERIC_FEATURES

    feature_fields = (
        "pattern_match_id",
        "subtitle_id",
        "candidate_source",
        "source_group",
        *EXPORT_SLOT_NUMERIC_FEATURES,
        *TEXT_FEATURE_COLUMNS,
    )
    label_fields = (
        "pattern_match_id",
        "subtitle_id",
        "label_target",
        "label_confidence",
    )
    prediction_fields = (
        "pattern_match_id",
        "subtitle_id",
        "predicted_tier",
        "prediction_confidence",
        "score_pop",
        "score_mainstream",
        "score_niche",
        "label_target",
        "label_confidence",
        "candidate_source",
        "source_group",
    )
    feature_rows = []
    label_rows = []
    teacher_rows = []
    for index, tier in enumerate(("pop", "mainstream", "niche") * 5, start=1):
        row = {
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "candidate_source": "subtitle",
            "source_group": "OL",
            "title": f"{tier} title",
            "subtitle_text": f"{tier} subtitle",
            "action_noun": "",
            "of_object": "",
            "list_items_text": "",
            "list_item_pair_text": "",
            "action_object_pair_text": "",
            "slot_frame_text": "",
            "metadata_publisher_text": "",
            "metadata_edition_name": "",
            "metadata_physical_format": "",
            "metadata_physical_dimensions": "",
            "metadata_physical_description": "",
            "metadata_subject_text": "",
            "metadata_loc_call_number": "",
            "metadata_dewey_decimal": "",
        }
        for feature_name in EXPORT_SLOT_NUMERIC_FEATURES:
            row[feature_name] = "0"
        row["source_group_is_ol"] = "1"
        row["title_length_chars"] = "20"
        row["subtitle_length_chars"] = "40"
        row["max_filler_popularity_score"] = {
            "pop": "1.8",
            "mainstream": "0.9",
            "niche": "0.1",
        }[tier]
        feature_rows.append(row)
        label_rows.append({
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "label_target": tier,
            "label_confidence": "0.9",
        })
        teacher_rows.append({
            "pattern_match_id": str(index),
            "subtitle_id": str(100 + index),
            "predicted_tier": tier,
            "prediction_confidence": "0.9",
            "score_pop": "0.9" if tier == "pop" else "0.05",
            "score_mainstream": "0.9" if tier == "mainstream" else "0.05",
            "score_niche": "0.9" if tier == "niche" else "0.05",
            "label_target": tier,
            "label_confidence": "0.9",
            "candidate_source": "subtitle",
            "source_group": "OL",
        })

    features_path = tmp_path / "book_features.csv"
    labels_path = tmp_path / "book_labels.csv"
    teacher_path = tmp_path / "teacher.csv"
    _write_csv(features_path, feature_fields, feature_rows)
    _write_csv(labels_path, label_fields, label_rows)
    _write_csv(teacher_path, prediction_fields, teacher_rows)

    result = distill_exportable_book_model(
        features_path=features_path,
        labels_path=labels_path,
        teacher_predictions_path=teacher_path,
        output_dir=tmp_path / "distill",
        feature_set="export-slot",
        device="cpu",
        epochs=10,
        hidden_dim=8,
        hash_dim=64,
    )

    report = result.distillation_report_path.read_text(encoding="utf-8")

    assert result.teacher_labels_path.exists()
    assert result.student_predictions_path.exists()
    assert "Teacher/student agreement" in report
    assert "Validation against original LLM labels" in report
