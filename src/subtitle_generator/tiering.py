"""Evidence-aware pop/mainstream/niche tier classification."""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from subtitle_generator.config import load_tuning_config

TIER_NAMES = frozenset({"pop", "mainstream", "niche"})

_SUBTITLE_RE = re.compile(
    r"^(?P<list_part>.+,\s*.+?)\s*,?\s+and\s+(?:the|a|an)\s+"
    r"(?P<action>.+?)\s+of\s+(?:(?:the|a|an)\s+)?(?P<object>.+)$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class ParsedSlot:
    slot_type: str
    filler: str


@dataclass(frozen=True)
class SlotEvidence:
    slot_type: str
    filler: str
    freq: int
    popularity_score: float | None
    popularity_level: int
    popularity_confidence: float
    frequency_score: float
    blended_score: float


@dataclass(frozen=True)
class TierEvidence:
    subtitle: str
    tier: str
    accessibility_score: float
    lower_tail_score: float
    demand_confidence: float
    slots: tuple[SlotEvidence, ...]


def parse_subtitle_slots(subtitle: str) -> list[ParsedSlot]:
    """Extract slot fillers from a subtitle string."""

    match = _SUBTITLE_RE.match(subtitle.strip())
    if not match:
        return []

    list_items = [
        item.strip()
        for item in match.group("list_part").split(",")
        if item.strip()
    ]
    action = _clean_filler(match.group("action"))
    obj = _clean_filler(match.group("object"))
    if len(list_items) < 2 or not action or not obj:
        return []
    return [
        *(ParsedSlot("list_item", item) for item in list_items),
        ParsedSlot("action_noun", action),
        ParsedSlot("of_object", obj),
    ]


def classification_score(
    freq: int,
    popularity_score: float | None,
    blend: float,
    missing_default: float,
) -> float:
    """Blend corpus frequency and popularity score on the configured scale."""

    score_freq = math.log10(1 + max(freq, 0))
    pop_value = popularity_score if popularity_score is not None else missing_default
    return (1 - blend) * score_freq + blend * pop_value


def compute_tier_evidence(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
) -> TierEvidence:
    """Classify a subtitle with explicit evidence for the tier decision."""

    slots = parse_subtitle_slots(subtitle)
    if conn is None or not slots:
        return TierEvidence(
            subtitle=subtitle,
            tier="mainstream",
            accessibility_score=0.0,
            lower_tail_score=0.0,
            demand_confidence=0.0,
            slots=(),
        )

    cfg = load_tuning_config(conn)
    blend = cfg["pop_classification_blend"]
    missing_default = cfg["pop_missing_default"]

    slot_evidence = tuple(
        _lookup_slot_evidence(conn, slot, blend, missing_default)
        for slot in slots
    )
    blended_scores = [slot.blended_score for slot in slot_evidence]
    accessibility = sum(blended_scores) / len(blended_scores)
    sorted_scores = sorted(blended_scores)
    lower_tail = sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    demand_confidence = sum(
        slot.popularity_confidence if slot.popularity_level > 0 else 0.0
        for slot in slot_evidence
    ) / len(slot_evidence)

    tier = _classify_from_evidence(
        cfg=cfg,
        accessibility_score=accessibility,
        lower_tail_score=lower_tail,
        demand_confidence=demand_confidence,
    )
    return TierEvidence(
        subtitle=subtitle,
        tier=tier,
        accessibility_score=accessibility,
        lower_tail_score=lower_tail,
        demand_confidence=demand_confidence,
        slots=slot_evidence,
    )


def _clean_filler(value: str) -> str:
    return re.sub(r"[\s/:;,.]+$", "", value).strip()


def _lookup_slot_evidence(
    conn: sqlite3.Connection,
    slot: ParsedSlot,
    blend: float,
    missing_default: float,
) -> SlotEvidence:
    columns = _columns(conn, "slot_fillers")
    if not columns:
        return _default_slot_evidence(slot, blend, missing_default)

    optional_columns = {
        "popularity_score": "popularity_score",
        "popularity_level": "popularity_level",
        "popularity_confidence": "popularity_confidence",
    }
    selected = ["freq"]
    selected.extend(column for column in optional_columns if column in columns)
    where = "slot_type = ? AND filler = ? COLLATE NOCASE"
    order = "ORDER BY "
    if "mode" in columns:
        order += "CASE WHEN mode = 'strict' THEN 0 ELSE 1 END, "
    if "popularity_confidence" in columns:
        order += "popularity_confidence DESC, "
    order += "freq DESC"
    row = conn.execute(
        f"SELECT {', '.join(selected)} FROM slot_fillers WHERE {where} {order} LIMIT 1",
        (slot.slot_type, slot.filler),
    ).fetchone()
    if row is None:
        return _default_slot_evidence(slot, blend, missing_default)

    values = dict(zip(selected, row, strict=True))
    freq = int(values["freq"] or 1)
    popularity_score = values.get("popularity_score")
    popularity_level = int(values.get("popularity_level") or 0)
    popularity_confidence = float(values.get("popularity_confidence") or 0.0)
    freq_score = math.log10(1 + max(freq, 0))
    blended = classification_score(freq, popularity_score, blend, missing_default)
    return SlotEvidence(
        slot_type=slot.slot_type,
        filler=slot.filler,
        freq=freq,
        popularity_score=popularity_score,
        popularity_level=popularity_level,
        popularity_confidence=popularity_confidence,
        frequency_score=freq_score,
        blended_score=blended,
    )


def _default_slot_evidence(
    slot: ParsedSlot,
    blend: float,
    missing_default: float,
) -> SlotEvidence:
    freq = 1
    freq_score = math.log10(1 + freq)
    blended = classification_score(freq, None, blend, missing_default)
    return SlotEvidence(
        slot_type=slot.slot_type,
        filler=slot.filler,
        freq=freq,
        popularity_score=None,
        popularity_level=0,
        popularity_confidence=0.0,
        frequency_score=freq_score,
        blended_score=blended,
    )


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _classify_from_evidence(
    *,
    cfg: dict[str, float],
    accessibility_score: float,
    lower_tail_score: float,
    demand_confidence: float,
) -> str:
    pop_threshold = cfg["accessibility_threshold_pop"]
    mainstream_threshold = cfg["accessibility_threshold_mainstream"]

    has_pop_demand = demand_confidence >= cfg["tier_pop_min_demand_confidence"]
    has_pop_tail = lower_tail_score >= cfg["tier_pop_min_lower_tail"]
    if accessibility_score >= pop_threshold and has_pop_demand and has_pop_tail:
        return "pop"

    if accessibility_score < mainstream_threshold:
        return "niche"
    return "mainstream"
