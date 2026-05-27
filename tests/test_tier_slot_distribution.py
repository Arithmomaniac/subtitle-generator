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
            (3, 'action_noun', 'Rise', 'strict', 6, 0.3);
        INSERT INTO slot_filler_sources VALUES
            (1, 101),
            (2, 102),
            (3, 101),
            (3, 103);
        INSERT INTO slot_filler_model_scores VALUES
            (1, 0.6, 0.3, 0.1, 'pop', 1),
            (2, 0.1, 0.2, 0.7, 'niche', 1),
            (3, 0.3, 0.3, 0.4, 'niche', 2);
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

    race_pop = rows[("list_item", "pop", "Race")]
    race_mainstream = rows[("list_item", "mainstream", "Race")]
    race_niche = rows[("list_item", "niche", "Race")]
    power_niche = rows[("list_item", "niche", "Power")]

    assert result.row_count == 9
    assert float(race_pop["anchored_soft_count"]) == 0.8
    assert float(race_pop["inferred_soft_count"]) == 0.0
    assert race_pop["anchored_source_count"] == "1"
    assert race_pop["inferred_source_count"] == "0"
    # A niche label exists in the labeled subset, so pop residual mass uses the
    # label-marginal prior and falls to niche rather than recirculating to pop.
    assert abs(float(race_mainstream["inferred_soft_count"]) - 0.0) < 0.0001
    assert abs(float(race_niche["inferred_soft_count"]) - 0.2) < 0.0001
    # The unlabeled source linked to Power uses the current score-table fallback.
    assert abs(float(power_niche["inferred_soft_count"]) - 0.7) < 0.0001
    assert "Residual priors for labeled sources" in report
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
