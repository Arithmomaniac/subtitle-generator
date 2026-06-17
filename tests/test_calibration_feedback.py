import json
from pathlib import Path

import pytest

from subtitle_generator.calibration_feedback import (
    build_decision_record,
    ingest_decision,
    read_decision_record,
    validate_decision_record,
    write_decision_record,
)


def test_build_decision_record_rejects_unknown_decision():
    with pytest.raises(ValueError):
        build_decision_record(
            granularity="per_tier",
            decision="maybe",
            summary="ok",
            temperatures={"pop": 1.2},
        )


def test_build_decision_record_requires_summary():
    with pytest.raises(ValueError):
        build_decision_record(
            granularity="per_tier",
            decision="accept",
            summary="   ",
            temperatures={},
        )


def test_write_and_read_round_trips(tmp_path: Path):
    record = build_decision_record(
        granularity="per_tier",
        decision="accept",
        summary="Calibration preserved likelihood and kept tiers distinct.",
        temperatures={"pop": 1.1, "mainstream": 1.0, "niche": 1.3},
        reviewer="avi",
    )
    path = tmp_path / "decision.json"
    write_decision_record(path, record)
    loaded = read_decision_record(path)
    assert loaded["decision"] == "accept"
    assert loaded["temperatures"]["niche"] == 1.3


def test_validate_rejects_missing_fields():
    with pytest.raises(ValueError):
        validate_decision_record({"decision": "accept"})


def test_ingest_decision_reads_metadata(tmp_path: Path):
    metadata = {
        "config": {"granularity": "per_tier"},
        "temperatures": {"pop": 1.2, "mainstream": 1.0, "niche": 1.4},
        "fold_assignment_digest": "abc123",
        "heldout_nll": {"baseline": 10.0, "calibrated": 9.5},
        "reliability_ece": {"baseline": 0.2, "calibrated": 0.1},
        "distinctiveness": {},
        "ranking": {"top1_hit_rate": 1.0},
    }
    metadata_path = tmp_path / "tier_slot_calibration_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    decision_path = tmp_path / "feedback" / "decision.json"

    submission = {
        "reviewer": "avi",
        "overall": {"decision": "accept", "summary": "Looks good; ship the sidecar."},
    }
    status = ingest_decision(
        submission, decision_path=decision_path, metadata_path=metadata_path
    )
    assert status["decision"] == "accept"
    record = read_decision_record(decision_path)
    assert record["temperatures"] == {"pop": 1.2, "mainstream": 1.0, "niche": 1.4}
    assert record["metadata_digest"] == "abc123"
    assert record["granularity"] == "per_tier"
    assert record["metrics"]["heldout_nll"]["calibrated"] == 9.5
