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
    # No ratings in this in-memory DB -> heuristic objective section.
    assert "No human ratings recorded yet" in report


def test_autoresearcher_prioritizes_proposals_from_human_ratings(tmp_path: Path):
    from subtitle_generator.smoothing_feedback import store_smoothing_rating
    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        run_semantic_smoothing_autoresearcher,
    )

    conn = _create_distribution_db()
    _seed_smoothing_vectors(conn)
    # Record human ratings flagging semantic bleed + too-generic borrowing.
    for filler, decision in [
        ("race", "semantic_bleed"),
        ("power", "semantic_bleed"),
        ("rise", "too_generic"),
    ]:
        store_smoothing_rating(
            conn, run_id="r1", variant="knn", vector_source="db",
            slot_type="list_item", tier="mainstream", filler=filler, decision=decision,
        )

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

    # Human ratings become the stated objective and steer the vector-space work.
    assert "Human review objective" in report
    assert "optimization target" in report
    assert "prioritized_by_human_review" in proposals
    assert "slot-centered-vectors" in proposals


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


def _sidecar_path(distribution_path: Path) -> Path:
    return distribution_path.with_name(
        "tier_slot_filler_distribution_v1.confidence_weighted.csv"
    )


def _seed_smoothing_vectors(conn) -> None:
    import struct

    conn.execute("ALTER TABLE slot_fillers ADD COLUMN vector_sum BLOB")
    conn.execute("ALTER TABLE slot_fillers ADD COLUMN token_count INTEGER")
    conn.execute(
        "UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'Race'",
        (struct.pack("3f", 1.0, 0.0, 0.0),),
    )
    conn.execute(
        "UPDATE slot_fillers SET vector_sum = ?, token_count = 1 WHERE filler = 'race'",
        (struct.pack("3f", 0.9, 0.1, 0.0),),
    )
    conn.commit()


def test_build_smoothing_review_feed_writes_deterministic_feed(tmp_path: Path):
    import json

    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        build_smoothing_review_feed,
    )

    configs = (
        SmoothingExperimentConfig("none", "none", 0, 0.0, "none", 0.0),
        SmoothingExperimentConfig("knn", "generic_embedding_kNN", 1, 0.5, "none", 0.10),
    )

    conn = _create_distribution_db()
    _seed_smoothing_vectors(conn)
    result = build_smoothing_review_feed(
        conn, tmp_path / "a", variant_name="knn", vector_source="db", configs=configs
    )

    assert result.feed_path.exists()
    feed = json.loads(result.feed_path.read_text(encoding="utf-8"))
    assert feed["schema_version"] == 1
    assert feed["variant"] == "knn"
    assert feed["run_id"] == result.run_id
    assert feed["candidate_count"] == len(feed["candidates"])
    for cand in feed["candidates"]:
        assert {"slot_type", "tier", "filler", "base_p", "smoothed_p", "delta",
                "evidence", "flags", "nearest_contributors", "context"} <= cand.keys()
        assert {"soft", "src", "anchored"} <= cand["evidence"].keys()
        # Human-readable context for the review canvas.
        assert {"tier_label", "slot_label", "example_subtitle", "similar_words",
                "lift_phrase", "source_titles"} <= cand["context"].keys()
        assert str(cand["display_filler"]).lower() in cand["context"]["example_subtitle"].lower()

    # Deterministic: a second build from identical inputs yields the same run_id.
    conn2 = _create_distribution_db()
    _seed_smoothing_vectors(conn2)
    result2 = build_smoothing_review_feed(
        conn2, tmp_path / "b", variant_name="knn", vector_source="db", configs=configs
    )
    assert result2.run_id == result.run_id


def test_build_smoothing_review_feed_rejects_unknown_and_none_variant(tmp_path: Path):
    import pytest

    from subtitle_generator.tier_slot_distribution import (
        SmoothingExperimentConfig,
        build_smoothing_review_feed,
    )

    configs = (SmoothingExperimentConfig("none", "none", 0, 0.0, "none", 0.0),)
    conn = _create_distribution_db()
    _seed_smoothing_vectors(conn)
    with pytest.raises(RuntimeError):
        build_smoothing_review_feed(
            conn, tmp_path / "x", variant_name="missing", vector_source="db", configs=configs
        )
    with pytest.raises(RuntimeError):
        build_smoothing_review_feed(
            conn, tmp_path / "y", variant_name="none", vector_source="db", configs=configs
        )


def test_source_reliability_weights_signal():
    from subtitle_generator.tier_slot_distribution import (
        _SourceLabel,
        _source_reliability_weights,
    )

    source_labels = {
        101: _SourceLabel("pop", 0.8),
        103: _SourceLabel("niche", 1.0),
        102: _SourceLabel(None, None),
        999: _SourceLabel(None, None),
    }
    evidence_ids = {101, 102, 103}

    weights = _source_reliability_weights(
        source_labels, evidence_ids, exponent=1.0, unlabeled_reliability=0.70
    )

    # Labeled sources use confidence as the (non-circular) signal, lower-bounded
    # at the unlabeled level: r = 0.70 + 0.30 * confidence.
    assert abs(weights[101] - (0.70 + 0.30 * 0.8)) < 1e-9
    assert abs(weights[103] - 1.0) < 1e-9
    # Every labeled source is at least as reliable as an unlabeled one.
    assert weights[101] >= 0.70 and weights[103] >= 0.70
    # Unlabeled sources get the flat constant, regardless of evidence membership.
    assert abs(weights[102] - 0.70) < 1e-9
    assert abs(weights[999] - 0.70) < 1e-9


def test_source_reliability_weights_monotonic_and_exponent():
    from subtitle_generator.tier_slot_distribution import (
        _SourceLabel,
        _source_reliability_weights,
    )

    labels = {1: _SourceLabel("pop", 0.5), 2: _SourceLabel("pop", 0.9)}
    linear = _source_reliability_weights(
        labels, set(), exponent=1.0, unlabeled_reliability=0.70
    )
    assert linear[2] > linear[1]
    # r = 0.70 + 0.30 * 0.5.
    assert abs(linear[1] - 0.85) < 1e-9

    squared = _source_reliability_weights(
        labels, set(), exponent=2.0, unlabeled_reliability=0.70
    )
    # A larger exponent pulls mid-range signals down: 0.70 + 0.30 * 0.25.
    assert squared[1] < linear[1]
    assert abs(squared[1] - 0.775) < 1e-9


def test_unlabeled_reliability_is_flat_constant():
    from subtitle_generator.tier_slot_distribution import (
        _SourceLabel,
        _source_reliability_weights,
    )

    labels = {1: _SourceLabel(None, None), 2: _SourceLabel(None, None)}
    # Whatever the evidence membership, unlabeled sources share one flat weight.
    weights = _source_reliability_weights(
        labels, {1}, exponent=1.0, unlabeled_reliability=0.6
    )
    assert weights[1] == 0.6
    assert weights[2] == 0.6


def test_evidence_source_ids_finds_linked_subtitles():
    from subtitle_generator.tier_slot_distribution import _evidence_source_ids

    conn = _create_distribution_db()
    ids = _evidence_source_ids(conn)
    # subtitles 101, 102, 103 each back at least one strict filler.
    assert ids == {101, 102, 103}


def test_weighted_sidecar_is_orthogonal_to_inferred_source_weight(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = build_tier_slot_distribution(
        _create_distribution_db(), out_a, alpha=0.5, inferred_source_weight=1.0
    )
    result_b = build_tier_slot_distribution(
        _create_distribution_db(), out_b, alpha=0.5, inferred_source_weight=0.5
    )

    weighted_a = _sidecar_path(result_a.distribution_path).read_text(encoding="utf-8")
    weighted_b = _sidecar_path(result_b.distribution_path).read_text(encoding="utf-8")
    served_a = result_a.distribution_path.read_text(encoding="utf-8")
    served_b = result_b.distribution_path.read_text(encoding="utf-8")

    # Reliability is the only magnitude axis in the weighted sidecar, so changing
    # inferred_source_weight leaves it identical...
    assert weighted_a == weighted_b
    # ...while the served anchored artifact still responds to inferred_source_weight.
    assert served_a != served_b


def test_weighted_sidecar_clean_decomposition_values(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    result = build_tier_slot_distribution(
        _create_distribution_db(), tmp_path, alpha=0.5
    )
    weighted = _read_distribution(_sidecar_path(result.distribution_path))

    # "race" merges the labeled source 101 (pop, c=0.8, r=0.94) and the unlabeled
    # source 102 (flat r=0.70, teacher vector pop=0.15/main=0.25/niche=0.6).
    race_pop = weighted[("list_item", "pop", "race")]
    race_mainstream = weighted[("list_item", "mainstream", "race")]
    race_niche = weighted[("list_item", "niche", "race")]

    # Anchored pop mass = confidence * reliability = 0.8 * 0.94.
    assert abs(float(race_pop["anchored_soft_count"]) - 0.752) < 1e-6
    # Inferred pop from the unlabeled teacher vector = 0.15 * 0.70.
    assert abs(float(race_pop["inferred_soft_count"]) - 0.105) < 1e-6
    # Step 4b: source 101's residual (1-0.8)*0.94 = 0.188 is split by its OWN
    # teacher vector (0.45/0.30/0.25); dropping the pop anchor leaves
    # main=0.30/niche=0.25 -> renormalized shares 0.30/0.55 and 0.25/0.55.
    labeled_residual = (1 - 0.8) * 0.94
    main_share = 0.30 / 0.55
    niche_share = 0.25 / 0.55
    # mainstream = unlabeled 0.25*0.70 + labeled residual redirected to mainstream.
    assert abs(
        float(race_mainstream["inferred_soft_count"])
        - (0.25 * 0.70 + labeled_residual * main_share)
    ) < 1e-6
    # niche = unlabeled 0.6*0.70 + labeled residual redirected to niche.
    assert abs(
        float(race_niche["inferred_soft_count"])
        - (0.6 * 0.70 + labeled_residual * niche_share)
    ) < 1e-6
    # Reliability mean for the pop cell averages the two contributing sources.
    assert abs(float(race_pop["teacher_confidence_mean"]) - 0.8) < 1e-6


def test_weighted_sidecar_normalizes_each_tier_slot(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    result = build_tier_slot_distribution(
        _create_distribution_db(), tmp_path, alpha=0.5
    )
    rows = list(_read_distribution(_sidecar_path(result.distribution_path)).values())
    groups: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["slot_type"], row["tier"])
        groups[key] = groups.get(key, 0.0) + float(row["probability"])
    assert all(abs(total - 1.0) < 1e-6 for total in groups.values())


def test_served_artifact_unchanged_by_reliability_knobs(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    result_a = build_tier_slot_distribution(
        _create_distribution_db(), out_a, alpha=0.5
    )
    result_b = build_tier_slot_distribution(
        _create_distribution_db(),
        out_b,
        alpha=0.5,
        reliability_exponent=2.0,
        unlabeled_reliability=0.4,
    )

    # The served artifact is anchored-only and must not move with report-only knobs.
    assert result_a.distribution_path.read_text(encoding="utf-8") == (
        result_b.distribution_path.read_text(encoding="utf-8")
    )
    # ...but the sidecar does respond to them.
    assert _sidecar_path(result_a.distribution_path).read_text(encoding="utf-8") != (
        _sidecar_path(result_b.distribution_path).read_text(encoding="utf-8")
    )


def test_report_contains_named_reliability_sections(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    result = build_tier_slot_distribution(
        _create_distribution_db(), tmp_path, alpha=0.5
    )
    report = result.report_path.read_text(encoding="utf-8")

    assert "Source reliability weighting" in report
    assert "Teacher-output confidence diagnostics" in report
    assert "Three-way distribution comparison" in report
    assert "hard-label anchoring" in report
    assert "confidence-weighted" in report
    assert "Reliability movers" in report
    assert "Pop/mainstream collapse guardrail" in report
    assert "confidence_weighted.csv" in report
    # The stale "not applied yet" reliability claim must be gone.
    assert "not applied yet" not in report.split("Semantic smoothing")[0]


def test_reliability_validation_errors(tmp_path: Path):
    import pytest

    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    with pytest.raises(RuntimeError):
        build_tier_slot_distribution(
            _create_distribution_db(), tmp_path / "b", reliability_exponent=0.0
        )
    with pytest.raises(RuntimeError):
        build_tier_slot_distribution(
            _create_distribution_db(), tmp_path / "c", unlabeled_reliability=1.5
        )


def test_collapse_guardrail_flags_effective_n_drop():
    from subtitle_generator.tier_slot_distribution import _collapse_guardrail

    def _row(slot, tier, filler, probability):
        return {
            "slot_type": slot,
            "tier": tier,
            "filler": filler,
            "probability": probability,
            "soft_count": probability,
            "anchored_soft_count": 0.0,
            "inferred_soft_count": probability,
            "prior_count": 0.0,
        }

    anchored = [
        _row("list_item", "pop", f"f{i}", 0.25) for i in range(4)
    ]
    weighted = [
        _row("list_item", "pop", "f0", 0.97),
        _row("list_item", "pop", "f1", 0.01),
        _row("list_item", "pop", "f2", 0.01),
        _row("list_item", "pop", "f3", 0.01),
    ]

    flags = _collapse_guardrail(anchored, weighted)
    assert len(flags) == 1
    flag = flags[0]
    assert flag["slot_type"] == "list_item"
    assert flag["tier"] == "pop"
    assert flag["drop"] > 0.2
    assert flag["weighted_effective_n"] < flag["anchored_effective_n"]


def test_reliability_movers_rank_by_absolute_delta_with_evidence():
    from subtitle_generator.tier_slot_distribution import _reliability_movers

    anchored = [
        {
            "slot_type": "list_item",
            "tier": "pop",
            "filler": "race",
            "display_filler": "Race",
            "probability": 0.20,
            "soft_count": 1.0,
            "source_count": 2,
        },
        {
            "slot_type": "list_item",
            "tier": "pop",
            "filler": "power",
            "display_filler": "Power",
            "probability": 0.50,
            "soft_count": 2.0,
            "source_count": 1,
        },
    ]
    weighted = [
        {
            "slot_type": "list_item",
            "tier": "pop",
            "filler": "race",
            "display_filler": "Race",
            "probability": 0.55,
            "soft_count": 1.4,
            "source_count": 2,
            "reliability_mean": 0.85,
        },
        {
            "slot_type": "list_item",
            "tier": "pop",
            "filler": "power",
            "display_filler": "Power",
            "probability": 0.45,
            "soft_count": 1.9,
            "source_count": 1,
            "reliability_mean": 0.55,
        },
    ]

    movers = _reliability_movers(anchored, weighted, limit=5)
    assert movers[0]["filler"] == "race"
    assert abs(movers[0]["delta"] - 0.35) < 1e-9
    assert movers[0]["anchored_soft"] == 1.0
    assert movers[0]["weighted_soft"] == 1.4
    assert movers[0]["weighted_source_count"] == 2
    assert movers[0]["reliability_mean"] == 0.85


def test_labeled_reliability_lower_bounded_by_unlabeled():
    from subtitle_generator.tier_slot_distribution import (
        _SourceLabel,
        _source_reliability_weights,
    )

    labels = {1: _SourceLabel("pop", 0.4), 2: _SourceLabel(None, None)}
    weights = _source_reliability_weights(
        labels, {1, 2}, exponent=1.0, unlabeled_reliability=0.7
    )
    # Labeled weight starts at the unlabeled level and climbs with confidence:
    # r = 0.7 + 0.3 * 0.4.
    assert abs(weights[1] - 0.82) < 1e-9
    # Unlabeled sources sit at the flat constant, which floors the labeled range.
    assert abs(weights[2] - 0.7) < 1e-9
    assert weights[1] >= weights[2]


def test_diagnostics_ignore_sources_without_strict_links(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import _teacher_vector_diagnostics

    conn = _create_distribution_db()
    # Add a labeled source (subtitle 200) with NO strict filler links.
    conn.execute(
        "INSERT INTO pattern_matches VALUES (4, 200, 'pop', 0.5)"
    )
    conn.commit()

    from subtitle_generator.tier_slot_distribution import (
        _load_source_labels,
        _load_source_fallback_vectors,
        _evidence_source_ids,
        _source_reliability_weights,
    )

    source_labels = _load_source_labels(conn)
    source_fallbacks = _load_source_fallback_vectors(conn)
    evidence_ids = _evidence_source_ids(conn)
    weights = _source_reliability_weights(
        source_labels, evidence_ids, exponent=1.0, unlabeled_reliability=0.7
    )
    diagnostics = _teacher_vector_diagnostics(
        source_labels, source_fallbacks, evidence_ids, weights
    )

    # Sources with strict links: 101 + 103 (labeled), 102 (unlabeled).
    # Subtitle 200 is labeled but link-less, so it must be excluded.
    labeled_row = next(
        row for row in diagnostics["reliability"]
        if str(row["group"]).startswith("labeled")
    )
    assert labeled_row["count"] == 2  # 200 excluded (no links); would be 3 otherwise
    assert diagnostics["unlabeled_total"] == 1


def test_confidence_extremes_decomposition(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    # confidence 1.0 -> all mass anchored, no residual.
    conn = _create_distribution_db()
    conn.execute("UPDATE pattern_matches SET llm_market_tier_confidence = 1.0 WHERE subtitle_id = 101")
    conn.commit()
    result = build_tier_slot_distribution(conn, tmp_path, alpha=0.5)
    weighted = _read_distribution(_sidecar_path(result.distribution_path))

    # Source 101 (pop, c=1.0, r=1.0) contributes only to pop with no residual.
    # "race" also has unlabeled source 102, so isolate the residual claim via the
    # labeled-only contribution: pop anchored = c*r = 1.0.
    race_pop = weighted[("list_item", "pop", "race")]
    assert abs(float(race_pop["anchored_soft_count"]) - 1.0) < 1e-6


def test_confidence_zero_routes_residual_by_teacher_vector(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    # confidence 0.0 -> no anchored mass; the source's full reliability becomes
    # residual, split (Step 4b) by the source's own teacher vector.
    conn = _create_distribution_db()
    conn.execute("UPDATE pattern_matches SET llm_market_tier_confidence = 0.0 WHERE subtitle_id = 101")
    conn.commit()
    result = build_tier_slot_distribution(conn, tmp_path, alpha=0.5)
    weighted = _read_distribution(_sidecar_path(result.distribution_path))

    # Source 101 (pop, c=0.0, r=0.70): anchored pop mass = 0.0. It is the only
    # anchored contributor to the race/pop cell (102 is inferred), so it stays 0.
    race_pop = weighted[("list_item", "pop", "race")]
    assert abs(float(race_pop["anchored_soft_count"]) - 0.0) < 1e-6
    # 101's full reliability 0.70 is the residual, split by its teacher vector
    # (0.45/0.30/0.25) over the non-pop tiers: main 0.30/0.55, niche 0.25/0.55.
    main_share = 0.30 / 0.55
    niche_share = 0.25 / 0.55
    race_mainstream = weighted[("list_item", "mainstream", "race")]
    race_niche = weighted[("list_item", "niche", "race")]
    # mainstream = unlabeled 0.25*0.70 + 0.70*main_share.
    assert abs(
        float(race_mainstream["inferred_soft_count"]) - (0.25 * 0.70 + 0.70 * main_share)
    ) < 1e-6
    # niche = unlabeled 0.6*0.70 + 0.70*niche_share.
    assert abs(
        float(race_niche["inferred_soft_count"]) - (0.6 * 0.70 + 0.70 * niche_share)
    ) < 1e-6


def test_labeled_residual_shares_renormalizes_teacher_vector():
    from subtitle_generator.tier_slot_distribution import _labeled_residual_shares

    corpus_prior = {"mainstream": 0.0, "niche": 1.0}
    shares = _labeled_residual_shares(
        "pop", {"pop": 0.45, "mainstream": 0.30, "niche": 0.25}, corpus_prior
    )
    assert abs(shares["mainstream"] - 0.30 / 0.55) < 1e-9
    assert abs(shares["niche"] - 0.25 / 0.55) < 1e-9
    assert abs(sum(shares.values()) - 1.0) < 1e-9


def test_labeled_residual_shares_falls_back_when_all_mass_on_anchor():
    from subtitle_generator.tier_slot_distribution import _labeled_residual_shares

    corpus_prior = {"mainstream": 0.4, "niche": 0.6}
    shares = _labeled_residual_shares(
        "pop", {"pop": 1.0, "mainstream": 0.0, "niche": 0.0}, corpus_prior
    )
    assert shares == corpus_prior


def test_labeled_residual_shares_falls_back_when_vector_missing():
    from subtitle_generator.tier_slot_distribution import _labeled_residual_shares

    corpus_prior = {"mainstream": 0.4, "niche": 0.6}
    assert _labeled_residual_shares("pop", None, corpus_prior) == corpus_prior


def test_labeled_residual_shares_falls_back_on_negligible_off_anchor_mass():
    from subtitle_generator.tier_slot_distribution import _labeled_residual_shares

    corpus_prior = {"mainstream": 0.4, "niche": 0.6}
    # Off-anchor mass below the noise floor -> corpus prior, not a 100% spike.
    shares = _labeled_residual_shares(
        "pop", {"pop": 1.0, "mainstream": 1e-12, "niche": 0.0}, corpus_prior
    )
    assert shares == corpus_prior


def test_scored_source_ids_excludes_sources_without_real_scores():
    from subtitle_generator.tier_slot_distribution import _scored_source_ids

    conn = _create_distribution_db()
    # All three sources start with real scores.
    assert _scored_source_ids(conn) == {101, 102, 103}
    # Drop source 101's fillers' score rows (fillers 1 and 3) -> 101 (and 103,
    # whose only filler is 3) lose real scores; 102 keeps fillers 2 and 4.
    conn.execute("DELETE FROM slot_filler_model_scores WHERE slot_filler_id IN (1, 3)")
    conn.commit()
    assert _scored_source_ids(conn) == {102}


def test_missing_scores_residual_falls_back_to_corpus_prior(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    # Source 101 (pop, c=0.8) keeps its label but loses all real model scores, so
    # its teacher vector is uniform noise. Step 4b must fall back to the corpus
    # prior (pop -> 100% niche) rather than splitting the residual 50/50.
    conn = _create_distribution_db()
    conn.execute("DELETE FROM slot_filler_model_scores WHERE slot_filler_id IN (1, 3)")
    conn.commit()
    result = build_tier_slot_distribution(conn, tmp_path, alpha=0.5)
    weighted = _read_distribution(_sidecar_path(result.distribution_path))

    # 101's residual (1-0.8)*0.94 = 0.188 routes entirely to niche (corpus prior);
    # mainstream gets only the unlabeled 102 contribution 0.25*0.70.
    race_mainstream = weighted[("list_item", "mainstream", "race")]
    race_niche = weighted[("list_item", "niche", "race")]
    assert abs(float(race_mainstream["inferred_soft_count"]) - (0.25 * 0.70)) < 1e-6
    assert abs(float(race_niche["inferred_soft_count"]) - (0.6 * 0.70 + 0.188)) < 1e-6


def test_served_residual_still_uses_corpus_prior(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    # The served artifact must keep the corpus-prior residual direction (Step 4b
    # only touches the weighted sidecar). For an anchored-pop source the corpus
    # prior sends 100% of the residual to niche (the only other labeled source,
    # 103, is niche), so the served "race" cell has zero mainstream-from-residual.
    result = build_tier_slot_distribution(_create_distribution_db(), tmp_path, alpha=0.5)
    served = _read_distribution(result.distribution_path)

    # Source 101 residual (1-0.8) routed entirely to niche under the corpus prior;
    # source 102 (unlabeled) is the only mainstream contributor to the cell.
    race_mainstream = served[("list_item", "mainstream", "race")]
    assert abs(float(race_mainstream["inferred_soft_count"]) - 0.25) < 1e-6
    # Residual (1-0.8)=0.2 to niche + unlabeled 0.6 = 0.8.
    race_niche = served[("list_item", "niche", "race")]
    assert abs(float(race_niche["inferred_soft_count"]) - (0.2 + 0.6)) < 1e-6


def test_report_contains_source_aware_residual_section(tmp_path: Path):
    from subtitle_generator.tier_slot_distribution import build_tier_slot_distribution

    result = build_tier_slot_distribution(_create_distribution_db(), tmp_path, alpha=0.5)
    report = result.report_path.read_text(encoding="utf-8")

    assert "Source-aware residual direction" in report
    # Source 101's teacher vector is non-degenerate, so its residual moves off the
    # corpus prior and at least one mover row (the "race" filler) is listed.
    assert "Teacher-vector p" in report
    assert "race" in report.split("Source-aware residual direction")[1]
