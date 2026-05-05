import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_combination_risk_report_distinguishes_funny_from_nonsensical(tmp_path):
    from subtitle_generator.combination_risk import (
        CombinationRiskCandidate,
        CombinationRiskPrediction,
        format_combination_risk_report,
    )

    candidates = (
        CombinationRiskCandidate(
            index=1,
            subtitle="Secrets, Snacks, and the Rise of Bureaucracy",
            item1="Secrets",
            item2="Snacks",
            action_noun="Rise",
            of_object="Bureaucracy",
        ),
        CombinationRiskCandidate(
            index=2,
            subtitle="Quartz, Tax Law, and the Whispering of Sandwiches",
            item1="Quartz",
            item2="Tax Law",
            action_noun="Whispering",
            of_object="Sandwiches",
        ),
    )
    predictions = (
        CombinationRiskPrediction(
            index=1,
            risk_label="intriguing_or_funny",
            confidence=0.8,
            rationale="Odd but readable as intentional.",
        ),
        CombinationRiskPrediction(
            index=2,
            risk_label="nonsensical",
            confidence=0.9,
            rationale="Semantic premise is incoherent.",
        ),
    )

    report = format_combination_risk_report(
        candidates=candidates,
        predictions=predictions,
        samples_path=tmp_path / "combination_risk_labels.csv",
        dry_run=False,
    )

    assert "`intriguing_or_funny`" in report
    assert "nonsensical=1" in report
    assert "Too few nonsensical examples" in report


def test_combination_risk_csv_writes_unlabeled_dry_run_rows(tmp_path):
    from subtitle_generator.combination_risk import (
        CombinationRiskCandidate,
        _write_combination_risk_csv,
    )

    path = tmp_path / "labels.csv"
    _write_combination_risk_csv(
        path,
        (
            CombinationRiskCandidate(
                index=1,
                subtitle="A, B, and the C of D",
                item1="A",
                item2="B",
                action_noun="C",
                of_object="D",
            ),
        ),
        (),
    )

    with open(path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["subtitle"] == "A, B, and the C of D"
    assert rows[0]["risk_label"] == ""
