import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _write_rollups(path: Path) -> None:
    fieldnames = (
        "slot_filler_id",
        "slot_type",
        "filler",
        "freq",
        "current_popularity_score",
        "current_popularity_level",
        "current_tier_score",
        "source_prediction_count",
        "avg_score_pop",
        "avg_score_mainstream",
        "avg_score_niche",
        "book_model_score",
        "book_model_tier",
        "score_delta",
        "tier_changed",
    )
    rows = [
        ("1", "list_item", "Race", "10", "0.8", "pop", "1.0", "1", "0.8", "0.1", "0.1", "0.86", "pop", "-0.14", "0"),
        ("2", "list_item", "Power", "8", "0.2", "niche", "0.1", "1", "0.1", "0.8", "0.1", "0.55", "mainstream", "0.45", "1"),
        ("3", "list_item", "Markets", "8", "0.4", "mainstream", "0.55", "1", "0.1", "0.2", "0.7", "0.28", "niche", "-0.27", "1"),
        ("4", "action_noun", "Rise", "7", "0.5", "mainstream", "0.55", "1", "0.6", "0.3", "0.1", "0.78", "pop", "0.23", "1"),
        ("5", "of_object", "Empire", "6", "0.4", "mainstream", "0.55", "1", "0.1", "0.8", "0.1", "0.55", "mainstream", "0", "0"),
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)


def test_run_deployment_gate_review_writes_dry_run_samples(tmp_path: Path):
    from subtitle_generator.book_model_deployment_gate import run_deployment_gate_review

    rollup_path = tmp_path / "rollups.csv"
    _write_rollups(rollup_path)

    result = run_deployment_gate_review(
        rollup_paths={"student": rollup_path},
        output_dir=tmp_path / "gate",
        sample_count=2,
        random_seed=1,
        dry_run=True,
    )

    with open(result.samples_path, encoding="utf-8") as handle:
        samples = list(csv.DictReader(handle))
    report = result.report_path.read_text(encoding="utf-8")

    assert result.comparison_count == 8
    assert result.reviewed_count == 0
    assert samples[0]["strategy"] == "blend-70-current"
    assert "Dry run only" in report


def test_run_deployment_gate_review_accepts_fake_reviewer(tmp_path: Path):
    from subtitle_generator.book_model_deployment_gate import (
        StrategyReview,
        run_deployment_gate_review,
    )

    rollup_path = tmp_path / "rollups.csv"
    _write_rollups(rollup_path)

    def fake_reviewer(comparisons, model):
        return tuple(
            StrategyReview(
                id=comparison.id,
                winner="candidate",
                current_risk="acceptable",
                candidate_risk="acceptable",
                tier_match_winner="candidate",
                rationale=f"{model}: candidate is better.",
            )
            for comparison in comparisons
        )

    result = run_deployment_gate_review(
        rollup_paths={"student": rollup_path},
        output_dir=tmp_path / "gate",
        sample_count=1,
        random_seed=1,
        model="fake-model",
        reviewer=fake_reviewer,
    )

    report = result.report_path.read_text(encoding="utf-8")

    assert result.reviewed_count == 4
    assert "Candidate win" in report
    assert "candidate is better" in report
    assert "deployment-gate pass" in report
