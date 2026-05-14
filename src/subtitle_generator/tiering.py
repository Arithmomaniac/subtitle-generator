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
    model_score_pop: float
    model_score_mainstream: float
    model_score_niche: float


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


def compute_tier_evidence(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    *,
    remix_parts: dict | None = None,
) -> TierEvidence:
    """Classify a subtitle from learned per-filler tier scores."""

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

    slot_evidence = _subtitle_slot_evidence(conn, slots, remix_parts)
    if slot_evidence is None:
        return TierEvidence(
            subtitle=subtitle,
            tier="mainstream",
            accessibility_score=0.0,
            lower_tail_score=0.0,
            demand_confidence=0.0,
            slots=(),
        )
    cfg = load_tuning_config(conn)
    model_scores = _aggregate_model_scores(slot_evidence, cfg)
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
    )
    demand_confidence = sum(
        max(_slot_model_scores(slot).values())
        for slot in slot_evidence
    ) / len(slot_evidence)
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


def _subtitle_slot_evidence(
    conn: sqlite3.Connection,
    slots: list[ParsedSlot],
    remix_parts: dict | None,
) -> tuple[SlotEvidence, ...] | None:
    evidence: list[SlotEvidence] = []
    for slot in slots:
        if slot.slot_type == "of_object" and remix_parts:
            object_evidence = _remix_object_evidence(conn, slot.filler, remix_parts)
            if object_evidence is None:
                return None
            evidence.append(object_evidence)
            continue
        slot_evidence = _lookup_slot_evidence(conn, slot)
        if slot_evidence is None:
            return None
        evidence.append(slot_evidence)
    return tuple(evidence)


def _remix_object_evidence(
    conn: sqlite3.Connection,
    composed_filler: str,
    remix_parts: dict,
) -> SlotEvidence | None:
    if "modifier" in remix_parts and "head" in remix_parts:
        component_slots = [
            ParsedSlot("of_modifier", remix_parts["modifier"]),
            ParsedSlot("of_head", remix_parts["head"]),
        ]
    elif "topic" in remix_parts and "complement" in remix_parts:
        component_slots = [
            ParsedSlot("of_topic", remix_parts["topic"]),
            ParsedSlot("of_complement", remix_parts["complement"]),
        ]
    else:
        return None
    components_or_none = [_lookup_slot_evidence(conn, slot) for slot in component_slots]
    if any(component is None for component in components_or_none):
        return None
    components = [component for component in components_or_none if component is not None]
    return _combine_remix_components(composed_filler, components)


def _combine_remix_components(
    composed_filler: str,
    components: list[SlotEvidence],
) -> SlotEvidence:
    popularity_values = [
        component.popularity_score
        for component in components
        if component.popularity_score is not None
    ]
    popularity_score = (
        sum(popularity_values) / len(popularity_values)
        if popularity_values
        else None
    )
    return SlotEvidence(
        slot_type="of_object",
        filler=composed_filler,
        freq=max(1, round(sum(component.freq for component in components) / len(components))),
        popularity_score=popularity_score,
        popularity_level=max(component.popularity_level for component in components),
        popularity_confidence=sum(
            component.popularity_confidence for component in components
        ) / len(components),
        frequency_score=sum(component.frequency_score for component in components)
        / len(components),
        model_score_pop=sum(component.model_score_pop for component in components)
        / len(components),
        model_score_mainstream=sum(
            component.model_score_mainstream for component in components
        ) / len(components),
        model_score_niche=sum(component.model_score_niche for component in components)
        / len(components),
    )


def _lookup_slot_evidence(
    conn: sqlite3.Connection,
    slot: ParsedSlot,
) -> SlotEvidence | None:
    columns = _columns(conn, "slot_fillers")
    if not columns:
        return None

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
        return None

    values = dict(zip(selected, row, strict=True))
    filler_id = int(values["id"]) if "id" in values else None
    freq = int(values["freq"] or 1)
    popularity_score = values.get("popularity_score")
    popularity_level = int(values.get("popularity_level") or 0)
    popularity_confidence = float(values.get("popularity_confidence") or 0.0)
    freq_score = math.log10(1 + max(freq, 0))
    model_scores = _lookup_model_scores(conn, filler_id) if filler_id is not None else None
    if model_scores is None:
        return None
    return SlotEvidence(
        slot_type=slot.slot_type,
        filler=slot.filler,
        freq=freq,
        popularity_score=popularity_score,
        popularity_level=popularity_level,
        popularity_confidence=popularity_confidence,
        frequency_score=freq_score,
        model_score_pop=model_scores["pop"],
        model_score_mainstream=model_scores["mainstream"],
        model_score_niche=model_scores["niche"],
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
    return {
        "pop": slot.model_score_pop,
        "mainstream": slot.model_score_mainstream,
        "niche": slot.model_score_niche,
    }


def _aggregate_model_scores(
    slots: tuple[SlotEvidence, ...],
    cfg: dict[str, float],
) -> dict[str, float]:
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
                + cfg[f"tier_classifier_frequency_interaction_{tier}"]
                * slot.frequency_score
                * slot_scores[tier]
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
        *(f"tier_classifier_frequency_interaction_{tier}" for tier in TIER_NAMES),
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
