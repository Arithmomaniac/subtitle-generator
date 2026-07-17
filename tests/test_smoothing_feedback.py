"""Tests for the Step 5 semantic-smoothing feedback contract (#38)."""

import sqlite3

import pytest

from subtitle_generator.smoothing_feedback import (
    build_decision_record,
    ensure_smoothing_ratings_table,
    read_decision_record,
    store_smoothing_rating,
    summarize_smoothing_ratings,
    validate_decision_record,
    write_decision_record,
)


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(":memory:")


def test_ensure_table_is_idempotent():
    conn = _conn()
    ensure_smoothing_ratings_table(conn)
    ensure_smoothing_ratings_table(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(smoothing_ratings)")}
    assert {"run_id", "variant", "decision", "base_p", "evidence_src"} <= cols


def test_store_and_summarize_ratings():
    conn = _conn()
    store_smoothing_rating(
        conn, run_id="r1", variant="knn10", vector_source="offline_spacy",
        slot_type="action_noun", tier="mainstream", filler="quest",
        decision="plausible_repair", base_p=0.001, smoothed_p=0.002, delta=0.001,
        evidence_soft=0.5, evidence_src=1, evidence_anchored=0.0, reviewer="avi",
    )
    store_smoothing_rating(
        conn, run_id="r1", variant="knn10", vector_source="offline_spacy",
        slot_type="action_noun", tier="mainstream", filler="currency",
        decision="semantic_bleed",
    )
    store_smoothing_rating(
        conn, run_id="r1", variant="uniform", vector_source="offline_spacy",
        slot_type="action_noun", tier="pop", filler="rise",
        decision="plausible_repair",
    )

    summary = summarize_smoothing_ratings(conn)
    assert summary["total"] == 3
    assert summary["by_decision"] == {"plausible_repair": 2, "semantic_bleed": 1}
    assert summary["by_variant"]["knn10"] == {
        "plausible_repair": 1, "semantic_bleed": 1
    }
    assert summary["by_variant_tier"]["knn10|mainstream"] == {
        "plausible_repair": 1, "semantic_bleed": 1
    }
    assert summary["variants"] == ["knn10", "uniform"]


def test_summarize_scopes_to_run_id():
    conn = _conn()
    store_smoothing_rating(
        conn, run_id="r1", variant="knn10", vector_source="v",
        slot_type="s", tier="pop", filler="a", decision="plausible_repair",
    )
    store_smoothing_rating(
        conn, run_id="r2", variant="knn10", vector_source="v",
        slot_type="s", tier="pop", filler="b", decision="semantic_bleed",
    )
    summary = summarize_smoothing_ratings(conn, run_id="r2")
    assert summary["total"] == 1
    assert summary["by_decision"] == {"semantic_bleed": 1}


def test_summarize_missing_table_returns_empty():
    conn = _conn()
    summary = summarize_smoothing_ratings(conn)
    assert summary["total"] == 0
    assert summary["by_decision"] == {}
    # Must not have created the table (read-only safe).
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='smoothing_ratings'"
    ).fetchone()
    assert exists is None


def test_store_rejects_invalid_decision():
    conn = _conn()
    with pytest.raises(ValueError):
        store_smoothing_rating(
            conn, run_id="r1", variant="v", vector_source="v",
            slot_type="s", tier="pop", filler="a", decision="not_a_decision",
        )


def test_build_decision_record_validates_enum_and_summary():
    rec = build_decision_record(
        run_id="r1", variant="knn10", vector_source="offline_spacy",
        decision="iterate", summary="Bleed too high in niche; try slot-centered.",
        reviewer="avi",
    )
    assert rec["schema_version"] == 1
    assert rec["decision"] == "iterate"
    validate_decision_record(rec)

    with pytest.raises(ValueError):
        build_decision_record(
            run_id="r1", variant="v", vector_source="v",
            decision="maybe", summary="x",
        )
    with pytest.raises(ValueError):
        build_decision_record(
            run_id="r1", variant="v", vector_source="v",
            decision="accept", summary="   ",
        )


def test_decision_record_roundtrip(tmp_path):
    rec = build_decision_record(
        run_id="r1", variant="knn10", vector_source="offline_spacy",
        decision="accept", summary="Repairs look good for pop/mainstream.",
    )
    path = tmp_path / "feedback" / "step05-smoothing" / "decision.json"
    write_decision_record(path, rec)
    loaded = read_decision_record(path)
    assert loaded["run_id"] == "r1"
    assert loaded["decision"] == "accept"


def test_validate_decision_record_missing_field():
    with pytest.raises(ValueError):
        validate_decision_record({"schema_version": 1, "decision": "accept"})


def test_ingest_submission_stores_ratings_and_writes_decision(tmp_path):
    from subtitle_generator.smoothing_feedback import ingest_submission

    conn = _conn()
    submission = {
        "run_id": "run-abc",
        "variant": "knn10_m0_5_cap0_10",
        "vector_source": "offline_spacy",
        "reviewer": "avi",
        "ratings": [
            {
                "slot_type": "action_noun", "tier": "mainstream", "filler": "saving",
                "base_p": 0.0002, "smoothed_p": 0.0012, "delta": 0.001,
                "evidence": {"soft": 0.02, "src": 1, "anchored": 0.0},
                "decision": "plausible_repair", "notes": "reads fine",
            },
            {
                "slot_type": "action_noun", "tier": "mainstream", "filler": "currency",
                "base_p": 0.0003, "smoothed_p": 0.0009, "delta": 0.0006,
                "evidence": {"soft": 0.2, "src": 1, "anchored": 0.0},
                "decision": "semantic_bleed",
            },
        ],
        "overall": {"decision": "iterate", "summary": "Promising but bleed in niche."},
    }
    decision_path = tmp_path / "feedback" / "step05-smoothing" / "decision.json"
    status = ingest_submission(conn, submission, decision_path=decision_path)

    assert status["stored_ratings"] == 2
    assert status["decision"] == "iterate"
    assert decision_path.exists()

    summary = summarize_smoothing_ratings(conn, run_id="run-abc")
    assert summary["total"] == 2
    assert summary["by_decision"] == {"plausible_repair": 1, "semantic_bleed": 1}

    loaded = read_decision_record(decision_path)
    assert loaded["run_id"] == "run-abc"
    assert loaded["ratings_summary"]["total"] == 2


def test_ingest_submission_rejects_bad_overall_decision(tmp_path):
    from subtitle_generator.smoothing_feedback import ingest_submission

    conn = _conn()
    submission = {
        "run_id": "r", "variant": "v", "vector_source": "v",
        "ratings": [],
        "overall": {"decision": "approve", "summary": "x"},
    }
    with pytest.raises(ValueError):
        ingest_submission(
            conn, submission, decision_path=tmp_path / "decision.json"
        )
