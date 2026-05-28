import csv
import sqlite3
from pathlib import Path


def _create_distribution_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL
        );
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            freq INTEGER NOT NULL DEFAULT 1,
            popularity_score REAL
        );
        CREATE TABLE slot_filler_sources (
            slot_filler_id INTEGER NOT NULL,
            subtitle_id INTEGER NOT NULL,
            PRIMARY KEY (slot_filler_id, subtitle_id)
        );
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL,
            score_mainstream REAL NOT NULL,
            score_niche REAL NOT NULL,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO pattern_matches VALUES
            (1, 101, 'pop', 0.8),
            (2, 102, NULL, NULL),
            (3, 103, 'niche', 1.0);
        INSERT INTO slot_fillers VALUES
            (1, 'list_item', 'Race', 'strict', 10, 0.5),
            (2, 'list_item', 'Power', 'strict', 8, NULL),
            (3, 'action_noun', 'Rise', 'strict', 6, 0.3),
            (4, 'list_item', 'race', 'strict', 2, 0.4);
        INSERT INTO slot_filler_sources VALUES
            (1, 101),
            (2, 102),
            (3, 101),
            (3, 103),
            (4, 102);
        INSERT INTO slot_filler_model_scores VALUES
            (1, 0.6, 0.3, 0.1, 'pop', 1),
            (2, 0.1, 0.2, 0.7, 'niche', 1),
            (3, 0.3, 0.3, 0.4, 'niche', 2),
            (4, 0.2, 0.3, 0.5, 'niche', 1);
        """
    )
    return conn


def _read_distribution(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (row["slot_type"], row["tier"], row["filler"]): row
        for row in rows
    }


def test_build_tier_slot_distribution_anchors_labeled_confidence(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    conn = _create_distribution_db()

    result = build_tier_slot_distribution(conn, tmp_path, alpha=0.5)
    rows = _read_distribution(result.distribution_path)
    report = result.report_path.read_text(encoding="utf-8")

    race_pop = rows[("list_item", "pop", "race")]
    race_mainstream = rows[("list_item", "mainstream", "race")]
    race_niche = rows[("list_item", "niche", "race")]
    power_niche = rows[("list_item", "niche", "power")]

    assert result.row_count == 9
    assert race_pop["display_filler"] == "Race"
    assert race_pop["frequency"] == "12"
    assert float(race_pop["anchored_soft_count"]) == 0.8
    assert abs(float(race_pop["inferred_soft_count"]) - 0.15) < 0.0001
    assert race_pop["anchored_source_count"] == "1"
    assert race_pop["inferred_source_count"] == "1"
    # A niche label exists in the labeled subset, so pop residual mass uses the
    # label-marginal prior and falls to niche rather than recirculating to pop.
    assert abs(float(race_mainstream["inferred_soft_count"]) - 0.25) < 0.0001
    assert abs(float(race_niche["inferred_soft_count"]) - 0.8) < 0.0001
    # The unlabeled source linked to Power uses the current score-table fallback.
    assert abs(float(power_niche["inferred_soft_count"]) - 0.6) < 0.0001
    assert "Residual priors for labeled sources" in report
    assert "Current rollup comparison" in report
    assert "JS divergence" in report
    assert "Label confidence diagnostics" in report
    assert "Confidence anchoring comparison" in report
    assert "hard-label variant" in report
    assert "Unlabeled sources use the current score-table fallback" in report


def test_build_tier_slot_distribution_normalizes_each_tier_slot(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    conn = _create_distribution_db()

    result = build_tier_slot_distribution(conn, tmp_path, alpha=0.5)
    rows = list(_read_distribution(result.distribution_path).values())

    groups: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["slot_type"], row["tier"])
        groups[key] = groups.get(key, 0.0) + float(row["probability"])

    assert set(groups) == {
        ("list_item", "pop"),
        ("list_item", "mainstream"),
        ("list_item", "niche"),
        ("action_noun", "pop"),
        ("action_noun", "mainstream"),
        ("action_noun", "niche"),
    }
    assert all(abs(total - 1.0) < 0.000001 for total in groups.values())


def test_run_semantic_smoothing_ablation_writes_metrics_and_report(tmp_path: Path):
    import struct

    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        run_semantic_smoothing_ablation,
    )

    conn = _create_distribution_db()
    vector_a = struct.pack("3f", 1.0, 0.0, 0.0)
    vector_b = struct.pack("3f", 0.9, 0.1, 0.0)
    conn.execute("ALTER TABLE slot_fillers ADD COLUMN vector_sum BLOB")
    conn.execute("ALTER TABLE slot_fillers ADD COLUMN token_count INTEGER")
    conn.execute("UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'Race'", (vector_a,))
    conn.execute("UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'race'", (vector_b,))
    conn.commit()

    result = run_semantic_smoothing_ablation(
        conn,
        tmp_path,
        vector_source="db",
        configs=(
            SmoothingExperimentConfig("none", "none", 0, 0.0, "none", 0.0),
            SmoothingExperimentConfig("uniform", "uniform_prior", 0, 0.5, "none", 0.10),
            SmoothingExperimentConfig("knn", "generic_embedding_kNN", 1, 0.5, "none", 0.10),
        ),
    )
    metrics = result.metrics_path.read_text(encoding="utf-8")
    report = result.report_path.read_text(encoding="utf-8")

    assert result.experiment_count == 3
    assert "generic_embedding_kNN" in metrics
    assert "Semantic smoothing ablation report" in report
    assert "Vector coverage" in report
    assert "Review examples" in report


def test_run_semantic_smoothing_autoresearcher_writes_proposals(tmp_path: Path):
    import struct

    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        run_semantic_smoothing_autoresearcher,
    )

    conn = _create_distribution_db()
    vector_a = struct.pack("3f", 1.0, 0.0, 0.0)
    vector_b = struct.pack("3f", 0.9, 0.1, 0.0)
    conn.execute("ALTER TABLE slot_fillers ADD COLUMN vector_sum BLOB")
    conn.execute("ALTER TABLE slot_fillers ADD COLUMN token_count INTEGER")
    conn.execute("UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'Race'", (vector_a,))
    conn.execute("UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'race'", (vector_b,))
    conn.commit()

    result = run_semantic_smoothing_autoresearcher(
        conn,
        tmp_path,
        vector_source="db",
        configs=(
            SmoothingExperimentConfig("none", "none", 0, 0.0, "none", 0.0),
            SmoothingExperimentConfig("knn", "generic_embedding_kNN", 1, 0.5, "none", 0.10),
        ),
    )

    report = result.report_path.read_text(encoding="utf-8")
    proposals = result.proposals_path.read_text(encoding="utf-8")

    assert "Semantic smoothing AutoResearcher packet" in report
    assert "does **not** call an external LLM" in report
    assert "weighted-similarity-knn" in proposals
    assert result.ablation_result.experiment_count == 2


def test_semantic_smoothing_does_not_boost_invalid_slot_fillers():
    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        _apply_smoothing,
        _filler_key,
    )

    rows = [
        {
            "slot_type": "action_noun",
            "tier": "mainstream",
            "filler": "u.s. house",
            "display_filler": "U.S. House",
            "probability": 0.1,
            "log_probability": 0.0,
            "soft_count": 0.01,
            "source_count": 1,
            "anchored_soft_count": 0.0,
        },
        {
            "slot_type": "action_noun",
            "tier": "mainstream",
            "filler": "rise",
            "display_filler": "Rise",
            "probability": 0.9,
            "log_probability": 0.0,
            "soft_count": 10.0,
            "source_count": 10,
            "anchored_soft_count": 5.0,
        },
    ]
    vectors = {
        _filler_key("action_noun", "u.s. house"): [1.0, 0.0],
        _filler_key("action_noun", "rise"): [0.99, 0.01],
    }

    smoothed = _apply_smoothing(
        rows,
        {},
        vectors,
        SmoothingExperimentConfig(
            "test",
            "generic_embedding_kNN",
            neighbor_count=1,
            shrinkage=10.0,
            evidence_gate="none",
            max_borrowed_mass=0.5,
        ),
    )

    invalid_row = next(row for row in smoothed if row["filler"] == "u.s. house")
    assert invalid_row["probability"] == 0.1
    assert invalid_row["semantic_smoothing_mass"] == 0.0
