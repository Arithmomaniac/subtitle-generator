"""Evidence-aware pop/mainstream/niche tier classification."""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass

from subtitle_generator.config import load_tuning_config

TIER_NAMES = ("pop", "mainstream", "niche")
_TIER_VALUES = {"pop": 1.0, "mainstream": 0.55, "niche": 0.1}
_TIER_TIE_BREAK = {"pop": 0, "mainstream": 1, "niche": 2}

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
    model_score_pop: float | None = None
    model_score_mainstream: float | None = None
    model_score_niche: float | None = None


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
    model_scores = _aggregate_model_scores(slot_evidence, cfg)
    if model_scores:
        tier = max(
            TIER_NAMES,
            key=lambda name: (model_scores[name], _TIER_TIE_BREAK[name]),
        )
        accessibility = sum(model_scores[name] * _TIER_VALUES[name] for name in TIER_NAMES)
        lower_tail = min(
            sum(
                _slot_model_scores(slot)[name] * _TIER_VALUES[name]
                for name in TIER_NAMES
            )
            for slot in slot_evidence
            if _slot_model_scores(slot)
        )
        demand_confidence = sum(
            max(_slot_model_scores(slot).values())
            for slot in slot_evidence
            if _slot_model_scores(slot)
        ) / len(slot_evidence)
        return TierEvidence(
            subtitle=subtitle,
            tier=tier,
            accessibility_score=accessibility,
            lower_tail_score=lower_tail,
            demand_confidence=demand_confidence,
            slots=slot_evidence,
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
    if "id" in columns:
        selected.insert(0, "id")
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
    filler_id = int(values["id"]) if "id" in values else None
    freq = int(values["freq"] or 1)
    popularity_score = values.get("popularity_score")
    popularity_level = int(values.get("popularity_level") or 0)
    popularity_confidence = float(values.get("popularity_confidence") or 0.0)
    freq_score = math.log10(1 + max(freq, 0))
    blended = classification_score(freq, popularity_score, blend, missing_default)
    model_scores = _lookup_model_scores(conn, filler_id) if filler_id is not None else None
    return SlotEvidence(
        slot_type=slot.slot_type,
        filler=slot.filler,
        freq=freq,
        popularity_score=popularity_score,
        popularity_level=popularity_level,
        popularity_confidence=popularity_confidence,
        frequency_score=freq_score,
        blended_score=blended,
        model_score_pop=model_scores.get("pop") if model_scores else None,
        model_score_mainstream=model_scores.get("mainstream") if model_scores else None,
        model_score_niche=model_scores.get("niche") if model_scores else None,
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


def _lookup_model_scores(conn: sqlite3.Connection, slot_filler_id: int) -> dict[str, float] | None:
    if "slot_filler_model_scores" not in _tables(conn):
        return None
    row = conn.execute(
        """
        SELECT score_pop, score_mainstream, score_niche
        FROM slot_filler_model_scores
        WHERE slot_filler_id = ?
        """,
        (slot_filler_id,),
    ).fetchone()
    if row is None:
        return None
    return {"pop": float(row[0]), "mainstream": float(row[1]), "niche": float(row[2])}


def _slot_model_scores(slot: SlotEvidence) -> dict[str, float]:
    if (
        slot.model_score_pop is None
        or slot.model_score_mainstream is None
        or slot.model_score_niche is None
    ):
        return {}
    return {
        "pop": slot.model_score_pop,
        "mainstream": slot.model_score_mainstream,
        "niche": slot.model_score_niche,
    }


def _aggregate_model_scores(
    slots: tuple[SlotEvidence, ...],
    cfg: dict[str, float],
) -> dict[str, float]:
    per_slot = [_slot_model_scores(slot) for slot in slots]
    per_slot = [scores for scores in per_slot if scores]
    if len(per_slot) != len(slots):
        return {}
    total_weight = 0.0
    contributions = {tier: 0.0 for tier in TIER_NAMES}
    model_weight = cfg["tier_classifier_model_score_weight"]
    missing_default = cfg["pop_missing_default"]
    for slot in slots:
        slot_scores = _slot_model_scores(slot)
        slot_weight = cfg[f"tier_classifier_slot_weight_{slot.slot_type}"]
        total_weight += slot_weight
        popularity = (
            slot.popularity_score
            if slot.popularity_score is not None
            else missing_default
        )
        for tier in TIER_NAMES:
            contributions[tier] += slot_weight * (
                model_weight * slot_scores[tier]
                + cfg[f"tier_classifier_popularity_weight_{tier}"] * popularity
                + cfg[f"tier_classifier_popularity_interaction_{tier}"]
                * popularity
                * slot_scores[tier]
                + cfg[f"tier_classifier_frequency_weight_{tier}"] * slot.frequency_score
            )
    if total_weight <= 0:
        return {}
    logits = {
        tier: cfg[f"tier_classifier_intercept_{tier}"]
        + contributions[tier] / total_weight
        for tier in TIER_NAMES
    }
    if not _uses_configured_tier_classifier(cfg):
        return logits
    return _softmax_scores(logits, cfg["tier_classifier_temperature"])


def _uses_configured_tier_classifier(cfg: dict[str, float]) -> bool:
    additive_keys = [
        *(f"tier_classifier_intercept_{tier}" for tier in TIER_NAMES),
        *(f"tier_classifier_popularity_weight_{tier}" for tier in TIER_NAMES),
        *(f"tier_classifier_popularity_interaction_{tier}" for tier in TIER_NAMES),
        *(f"tier_classifier_frequency_weight_{tier}" for tier in TIER_NAMES),
    ]
    return (
        cfg["tier_classifier_model_score_weight"] != 1.0
        or any(cfg[key] != 0.0 for key in additive_keys)
    )


def _softmax_scores(logits: dict[str, float], temperature: float) -> dict[str, float]:
    temperature = max(temperature, 0.001)
    scaled = {tier: logits[tier] / temperature for tier in TIER_NAMES}
    max_logit = max(scaled.values())
    exp_scores = {
        tier: math.exp(scaled[tier] - max_logit)
        for tier in TIER_NAMES
    }
    total = sum(exp_scores.values())
    return {tier: exp_scores[tier] / total for tier in TIER_NAMES}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


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
