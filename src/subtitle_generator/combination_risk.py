"""Combination-risk sampling and LLM labeling for OQ2."""

from __future__ import annotations

import csv
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from subtitle_generator.generate import GeneratedSubtitle, generate_subtitles
from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL

CombinationRiskLabel = Literal["nonsensical", "intriguing_or_funny", "acceptable"]
VALID_COMBINATION_RISK_LABELS = {
    "nonsensical",
    "intriguing_or_funny",
    "acceptable",
}


@dataclass(frozen=True)
class CombinationRiskCandidate:
    index: int
    subtitle: str
    item1: str
    item2: str
    action_noun: str
    of_object: str


@dataclass(frozen=True)
class CombinationRiskPrediction:
    index: int
    risk_label: CombinationRiskLabel
    confidence: float
    rationale: str


@dataclass(frozen=True)
class CombinationRiskResult:
    samples_path: Path
    report_path: Path
    sample_count: int
    labeled_count: int
    nonsensical_count: int


CombinationRiskClassifier = Callable[
    [tuple[CombinationRiskCandidate, ...], str],
    tuple[CombinationRiskPrediction, ...],
]


def sample_combination_risk_candidates(
    conn: sqlite3.Connection,
    *,
    samples: int,
    seed: int,
) -> tuple[CombinationRiskCandidate, ...]:
    """Generate fixed-seed subtitle candidates for combination-risk labeling."""

    generated = generate_subtitles(conn, n=samples, seed_base=seed)
    return tuple(
        _candidate_from_generated(index, subtitle)
        for index, subtitle in enumerate(generated, start=1)
    )


def label_combination_risk(
    conn: sqlite3.Connection,
    *,
    output_dir: Path,
    samples: int = 60,
    seed: int = 20260505,
    model: str = DEFAULT_RATER_MODEL,
    dry_run: bool = False,
    classifier: CombinationRiskClassifier | None = None,
) -> CombinationRiskResult:
    """Sample and optionally LLM-label combination-risk examples."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = sample_combination_risk_candidates(conn, samples=samples, seed=seed)
    predictions = () if dry_run else (classifier or _classify_with_llm)(candidates, model)
    if predictions:
        _assert_prediction_ids_match(candidates, predictions)
    samples_path = output_dir / "combination_risk_labels.csv"
    report_path = output_dir / "combination_risk_report.md"
    _write_combination_risk_csv(samples_path, candidates, predictions)
    report_path.write_text(
        format_combination_risk_report(
            candidates=candidates,
            predictions=predictions,
            samples_path=samples_path,
            dry_run=dry_run,
        ),
        encoding="utf-8",
    )
    nonsensical_count = sum(
        1 for prediction in predictions
        if prediction.risk_label == "nonsensical"
    )
    return CombinationRiskResult(
        samples_path=samples_path,
        report_path=report_path,
        sample_count=len(candidates),
        labeled_count=len(predictions),
        nonsensical_count=nonsensical_count,
    )


def format_combination_risk_report(
    *,
    candidates: tuple[CombinationRiskCandidate, ...],
    predictions: tuple[CombinationRiskPrediction, ...],
    samples_path: Path,
    dry_run: bool,
) -> str:
    counts = Counter(prediction.risk_label for prediction in predictions)
    lines = [
        "# Combination-risk evidence report",
        "",
        "This report addresses OQ2: whether beyond-the-pale combinations are "
        "nonsensical rather than merely intriguing, funny, niche, or surprising.",
        "",
        "## Rubric",
        "",
        "- `nonsensical`: the slot/filler combination reads like a semantic mistake "
        "or incoherent subtitle premise.",
        "- `intriguing_or_funny`: the combination is odd, surprising, or comic, but "
        "could still be an intentional subtitle.",
        "- `acceptable`: the combination is plausible without needing the odd/funny "
        "exception.",
        "",
        "## Outputs",
        "",
        f"- Samples: `{samples_path}` ({len(candidates):,} rows)",
        "",
        "## Label counts",
        "",
    ]
    if dry_run:
        lines.append("Dry run only; no LLM labels were requested.")
    else:
        lines.extend([
            f"- nonsensical={counts.get('nonsensical', 0):,}",
            f"- intriguing_or_funny={counts.get('intriguing_or_funny', 0):,}",
            f"- acceptable={counts.get('acceptable', 0):,}",
            "",
            "## Gate note",
            "",
        ])
        if counts.get("nonsensical", 0) < 10:
            lines.append(
                "- Too few nonsensical examples were found to train a stable binary "
                "risk model; keep D6 tentative and use these as examples for now."
            )
        else:
            lines.append(
                "- Enough nonsensical examples exist for a first separability test "
                "against exportable slot/combination features."
            )
    return "\n".join(lines)


def build_combination_risk_prompt(
    candidates: tuple[CombinationRiskCandidate, ...],
) -> str:
    lines = "\n".join(
        f"{candidate.index}. {candidate.subtitle}"
        for candidate in candidates
    )
    return f"""Classify generated book subtitles for combination risk.

The task is NOT to punish niche, funny, or surprising combinations.
Classify a subtitle as:
- nonsensical: semantic mistake, incoherent premise, or too far outside a plausible book-subtitle combination.
- intriguing_or_funny: odd, surprising, comic, or highly niche, but still readable as intentional.
- acceptable: straightforwardly plausible.

Return exactly one label for each input id, with confidence 0.0-1.0 and a short rationale.

Subtitles:
{lines}
"""


def _classify_with_llm(
    candidates: tuple[CombinationRiskCandidate, ...],
    model: str,
) -> tuple[CombinationRiskPrediction, ...]:
    try:
        from pydantic import BaseModel, Field
        from subtitle_generator.eval_harness import structured_completion
    except ImportError as exc:
        raise RuntimeError(
            "Combination-risk labeling requires optional tune dependencies. "
            "Run `uv sync --extra tune` first."
        ) from exc

    class _PredictionModel(BaseModel):
        id: int
        risk_label: Literal["nonsensical", "intriguing_or_funny", "acceptable"]
        confidence: float = Field(ge=0.0, le=1.0)
        rationale: str

    class _PredictionBatch(BaseModel):
        labels: list[_PredictionModel]

    all_predictions: list[CombinationRiskPrediction] = []
    for start in range(0, len(candidates), 25):
        batch = candidates[start:start + 25]
        result = structured_completion(
            model=model,
            messages=[{"role": "user", "content": build_combination_risk_prompt(batch)}],
            schema=_PredictionBatch,
            temperature=0,
            max_tokens=4096,
            timeout=120,
        )
        all_predictions.extend(
            CombinationRiskPrediction(
                index=label.id,
                risk_label=label.risk_label,
                confidence=label.confidence,
                rationale=" ".join(label.rationale.split()),
            )
            for label in result.labels
        )
    predictions = tuple(all_predictions)
    _assert_prediction_ids_match(candidates, predictions)
    return predictions


def _candidate_from_generated(
    index: int,
    subtitle: GeneratedSubtitle,
) -> CombinationRiskCandidate:
    return CombinationRiskCandidate(
        index=index,
        subtitle=subtitle.text,
        item1=subtitle.item1,
        item2=subtitle.item2,
        action_noun=subtitle.action_noun,
        of_object=subtitle.of_object,
    )


def _assert_prediction_ids_match(
    candidates: tuple[CombinationRiskCandidate, ...],
    predictions: tuple[CombinationRiskPrediction, ...],
) -> None:
    expected = Counter(candidate.index for candidate in candidates)
    actual = Counter(prediction.index for prediction in predictions)
    if actual != expected:
        raise RuntimeError(
            "Combination-risk response did not match requested ids: "
            f"expected {dict(sorted(expected.items()))}, "
            f"got {dict(sorted(actual.items()))}"
        )


def _write_combination_risk_csv(
    path: Path,
    candidates: tuple[CombinationRiskCandidate, ...],
    predictions: tuple[CombinationRiskPrediction, ...],
) -> None:
    predictions_by_id = {prediction.index: prediction for prediction in predictions}
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "index",
                "subtitle",
                "item1",
                "item2",
                "action_noun",
                "of_object",
                "risk_label",
                "confidence",
                "rationale",
            ),
        )
        writer.writeheader()
        for candidate in candidates:
            prediction = predictions_by_id.get(candidate.index)
            writer.writerow({
                "index": candidate.index,
                "subtitle": candidate.subtitle,
                "item1": candidate.item1,
                "item2": candidate.item2,
                "action_noun": candidate.action_noun,
                "of_object": candidate.of_object,
                "risk_label": prediction.risk_label if prediction else "",
                "confidence": prediction.confidence if prediction else "",
                "rationale": prediction.rationale if prediction else "",
            })
