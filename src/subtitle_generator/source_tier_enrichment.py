"""LLM-assisted source-title market tier labeling infrastructure."""

from __future__ import annotations

import asyncio
import csv
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

from subtitle_generator.market_tiers import MarketTier, source_label_tier_definitions
from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL

SOURCE_TIER_LABEL_COLUMNS: tuple[tuple[str, str], ...] = (
    ("llm_market_tier", "TEXT"),
    ("llm_market_tier_confidence", "REAL"),
    ("llm_market_tier_rationale", "TEXT"),
)
SOURCE_TIER_LABEL_EXPORT_COLUMNS = (
    "subtitle_id",
    "pattern_match_id",
    "title",
    "subtitle",
    "llm_market_tier",
    "llm_market_tier_confidence",
    "llm_market_tier_rationale",
)
VALID_SOURCE_TIERS = {"pop", "mainstream", "niche"}
_CITATION_LINK_RE = re.compile(r"\s*\(\[[^\]]+\]\(https?://[^)]+\)\)")
_URL_RE = re.compile(r"https?://\S+")


@dataclass(frozen=True)
class SourceTierCandidate:
    id: int
    title: str
    subtitle: str


@dataclass(frozen=True)
class SourceTierPrediction:
    id: int
    tier: MarketTier
    confidence: float
    rationale: str


@dataclass(frozen=True)
class SourceTierClassificationResult:
    selected: tuple[SourceTierCandidate, ...]
    labeled_count: int
    exported_count: int
    export_path: Path | None
    dry_run: bool


SourceTierClassifier = Callable[
    [tuple[SourceTierCandidate, ...], str],
    tuple[SourceTierPrediction, ...],
]


def ensure_source_tier_label_columns(conn: sqlite3.Connection) -> None:
    """Add nullable source-tier label columns to ``pattern_matches`` if needed."""

    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pattern_matches'"
    ).fetchone()
    if not table_exists:
        raise RuntimeError("pattern_matches table does not exist. Run build-slots first.")

    columns = {row[1] for row in conn.execute("PRAGMA table_info(pattern_matches)")}
    for column, column_type in SOURCE_TIER_LABEL_COLUMNS:
        if column not in columns:
            conn.execute(
                f"ALTER TABLE pattern_matches ADD COLUMN {column} {column_type}"
            )
    conn.commit()


def _pattern_match_columns(conn: sqlite3.Connection) -> set[str]:
    table_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pattern_matches'"
    ).fetchone()
    if not table_exists:
        raise RuntimeError("pattern_matches table does not exist. Run build-slots first.")
    return {row[1] for row in conn.execute("PRAGMA table_info(pattern_matches)")}


def load_source_tier_candidates(
    conn: sqlite3.Connection,
    *,
    limit: int,
    selection: Literal["random", "id"] = "random",
    random_seed: int = 20260501,
    force: bool = False,
    migrate: bool = True,
) -> tuple[SourceTierCandidate, ...]:
    """Load source-title rows that should be labeled by the LLM."""

    if migrate:
        ensure_source_tier_label_columns(conn)
    columns = _pattern_match_columns(conn)
    where = [
        "COALESCE(title, '') <> ''",
        "COALESCE(subtitle, '') <> ''",
    ]
    if not force and "llm_market_tier" in columns:
        where.append("llm_market_tier IS NULL")
    rows = conn.execute(
        f"""
        SELECT id, title, subtitle
        FROM pattern_matches
        WHERE {" AND ".join(where)}
        ORDER BY id
        """
    ).fetchall()
    candidates = [
        SourceTierCandidate(id=row[0], title=row[1], subtitle=row[2])
        for row in rows
    ]
    if selection == "random":
        rng = random.Random(random_seed)
        rng.shuffle(candidates)
    elif selection != "id":
        raise ValueError(f"Unsupported selection mode: {selection}")
    return tuple(candidates[:limit])


def classify_source_tiers(
    conn: sqlite3.Connection,
    *,
    limit: int = 20,
    batch_size: int = 10,
    model: str = DEFAULT_RATER_MODEL,
    selection: Literal["random", "id"] = "random",
    random_seed: int = 20260501,
    force: bool = False,
    dry_run: bool = False,
    export_path: Path | None = Path("api/data/source_tier_labels.csv"),
    classifier: SourceTierClassifier | None = None,
    web_search: bool = True,
) -> SourceTierClassificationResult:
    """Classify source-title market tiers and persist labels on pattern_matches."""

    if limit < 0:
        raise ValueError("limit must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    selected = load_source_tier_candidates(
        conn,
        limit=limit,
        selection=selection,
        random_seed=random_seed,
        force=force,
        migrate=not dry_run,
    )
    if dry_run:
        return SourceTierClassificationResult(
            selected=selected,
            labeled_count=0,
            exported_count=0,
            export_path=export_path,
            dry_run=dry_run,
        )
    if not selected:
        exported = export_source_tier_labels(conn, export_path) if export_path else 0
        return SourceTierClassificationResult(
            selected=selected,
            labeled_count=0,
            exported_count=exported,
            export_path=export_path,
            dry_run=False,
        )

    predict = classifier or (
        _classify_with_hosted_web_search if web_search else _classify_with_llm
    )
    labeled_count = 0
    for start in range(0, len(selected), batch_size):
        batch = selected[start:start + batch_size]
        predictions = predict(batch, model)
        _assert_prediction_ids_match(batch, predictions)
        _write_source_tier_predictions(conn, predictions)
        labeled_count += len(predictions)

    exported = export_source_tier_labels(conn, export_path) if export_path else 0
    return SourceTierClassificationResult(
        selected=selected,
        labeled_count=labeled_count,
        exported_count=exported,
        export_path=export_path,
        dry_run=False,
    )


def _assert_prediction_ids_match(
    candidates: tuple[SourceTierCandidate, ...],
    predictions: tuple[SourceTierPrediction, ...],
) -> None:
    expected_ids = Counter(candidate.id for candidate in candidates)
    actual_ids = Counter(prediction.id for prediction in predictions)
    if actual_ids != expected_ids:
        raise RuntimeError(
            "LLM source-tier response did not match requested ids: "
            f"expected {dict(sorted(expected_ids.items()))}, "
            f"got {dict(sorted(actual_ids.items()))}"
        )


def _write_source_tier_predictions(
    conn: sqlite3.Connection,
    predictions: tuple[SourceTierPrediction, ...],
) -> None:
    for prediction in predictions:
        if prediction.tier not in VALID_SOURCE_TIERS:
            raise ValueError(f"Invalid source tier: {prediction.tier}")
        conn.execute(
            """
            UPDATE pattern_matches
            SET llm_market_tier = ?,
                llm_market_tier_confidence = ?,
                llm_market_tier_rationale = ?
            WHERE id = ?
            """,
            (
                prediction.tier,
                max(0.0, min(1.0, prediction.confidence)),
                _sanitize_source_tier_rationale(prediction.rationale),
                prediction.id,
            ),
        )
    conn.commit()


def _sanitize_source_tier_rationale(rationale: str) -> str:
    """Keep durable label rationales concise and free of citation URLs."""

    cleaned = _CITATION_LINK_RE.sub("", rationale)
    cleaned = _URL_RE.sub("", cleaned)
    return " ".join(cleaned.split()).strip()


def export_source_tier_labels(
    conn: sqlite3.Connection,
    export_path: Path | None = Path("api/data/source_tier_labels.csv"),
) -> int:
    """Export populated source-tier labels to a reproducible CSV artifact."""

    if export_path is None:
        return 0
    ensure_source_tier_label_columns(conn)
    columns = _pattern_match_columns(conn)
    stable_id_column = "subtitle_id" if "subtitle_id" in columns else "id"
    rows = conn.execute(
        f"""
        SELECT
            {stable_id_column},
            id,
            title,
            subtitle,
            llm_market_tier,
            llm_market_tier_confidence,
            llm_market_tier_rationale
        FROM pattern_matches
        WHERE llm_market_tier IN ('pop', 'mainstream', 'niche')
        ORDER BY {stable_id_column}, id
        """
    ).fetchall()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_rows = [
        (*row[:6], _sanitize_source_tier_rationale(row[6] or ""))
        for row in rows
    ]
    with open(export_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCE_TIER_LABEL_EXPORT_COLUMNS)
        writer.writerows(cleaned_rows)
    return len(rows)


def build_source_tier_prompt(candidates: tuple[SourceTierCandidate, ...]) -> str:
    """Build the source-title tier labeling prompt from shared tier definitions."""

    lines = "\n".join(
        f"{candidate.id}. Title: {candidate.title}\n"
        f"   Subtitle: {candidate.subtitle}"
        for candidate in candidates
    )
    return f"""Classify each real book title/subtitle into one market tier.

{source_label_tier_definitions()}

Return exactly one label per input id. Use confidence from 0.0 to 1.0 and a
brief rationale grounded in the title/subtitle.

Inputs:
{lines}
"""


def _classify_with_llm(
    candidates: tuple[SourceTierCandidate, ...],
    model: str,
) -> tuple[SourceTierPrediction, ...]:
    try:
        from pydantic import BaseModel, Field
        from subtitle_generator.eval_harness import structured_completion
    except ImportError as exc:
        raise RuntimeError(
            "LLM source-tier labeling requires optional tune dependencies. "
            "Run `uv sync --extra tune` first."
        ) from exc

    class _PredictionModel(BaseModel):
        id: int
        tier: Literal["pop", "mainstream", "niche"]
        confidence: float = Field(ge=0.0, le=1.0)
        rationale: str

    class _PredictionBatch(BaseModel):
        labels: list[_PredictionModel]

    prompt = build_source_tier_prompt(candidates)
    result = structured_completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        schema=_PredictionBatch,
        temperature=0,
        max_tokens=4096,
        timeout=90,
    )
    labels = tuple(
        SourceTierPrediction(
            id=label.id,
            tier=label.tier,
            confidence=label.confidence,
            rationale=label.rationale,
        )
        for label in result.labels
    )
    _assert_prediction_ids_match(candidates, labels)
    return labels


def _classify_with_hosted_web_search(
    candidates: tuple[SourceTierCandidate, ...],
    model: str,
) -> tuple[SourceTierPrediction, ...]:
    try:
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "Web-searched source-tier labeling requires optional tune dependencies. "
            "Run `uv sync --extra tune` first."
        ) from exc

    class _PredictionModel(BaseModel):
        tier: Literal["pop", "mainstream", "niche"]
        confidence: float = Field(ge=0.0, le=1.0)
        rationale: str

    async def classify_one(candidate: SourceTierCandidate) -> SourceTierPrediction:
        prompt = f"""Use hosted web search before answering. Classify this real book title/subtitle into exactly one market tier: pop, mainstream, or niche.

{source_label_tier_definitions()}

Search/evidence rules:
- Do at least one web search for this exact title/subtitle.
- Do at most two web_search calls total for this book. Do not keep searching.
- Rationale must be 1-2 concise sentences and include the evidence strength:
  exact match, weak/adjacent match, or no reliable match.
- Do not include citation links, URLs, or source lists in the rationale.
- If no reliable match is found, classify as niche unless title/subtitle strongly
  indicates broad trade appeal; lower confidence.

Book:
ID: {candidate.id}
Title: {candidate.title}
Subtitle: {candidate.subtitle}
"""
        response = await litellm.aresponses(
            model=model,
            input=prompt,
            tools=[{"type": "web_search", "search_context_size": "low"}],
            tool_choice="required",
            text_format=_PredictionModel,
            max_output_tokens=1400,
            timeout=120.0,
        )
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            if hasattr(item, "content") and item.content:
                for content in item.content:
                    if hasattr(content, "text"):
                        prediction = _PredictionModel.model_validate_json(content.text)
                        return SourceTierPrediction(
                            id=candidate.id,
                            tier=prediction.tier,
                            confidence=prediction.confidence,
                            rationale=prediction.rationale,
                        )
        raise RuntimeError(
            f"No structured source-tier response returned for id {candidate.id}"
        )

    async def classify_all() -> tuple[SourceTierPrediction, ...]:
        return tuple(await asyncio.gather(*(classify_one(candidate) for candidate in candidates)))

    labels = asyncio.run(classify_all())
    _assert_prediction_ids_match(candidates, labels)
    return labels
