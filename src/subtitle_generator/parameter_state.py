"""Typed views over tunable parameters and model identifiers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from subtitle_generator.config import load_tuning_config


DEFAULT_RATER_MODEL = "github_copilot/gpt-5.4-mini"
DEFAULT_PROPOSER_MODEL = "github_copilot/gpt-5.4"
DEFAULT_JACKET_MODEL = "gpt-5.4-mini"
RESPONSES_ONLY_MODELS = frozenset({"gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano"})


@dataclass(frozen=True)
class SamplingParameters:
    weighted_sample_spread: float
    weighted_sample_bias_floor: float


@dataclass(frozen=True)
class PopularityParameters:
    weight_spl: float
    weight_ol: float
    weight_goodreads: float
    weight_nyt: float
    weight_library: float
    weight_frequency: float
    exponent: float


@dataclass(frozen=True)
class PopularityBlendParameters:
    base_weight_blend: float
    tone_blend: float
    classification_blend: float
    missing_default: float


@dataclass(frozen=True)
class SlotMultiplierParameters:
    list_item: float
    action_noun: float
    of_object: float


@dataclass(frozen=True)
class ArticleParameters:
    of_min_freq: float
    action_min_freq: float
    remix_heuristic_threshold: float


@dataclass(frozen=True)
class RemixParameters:
    reject_double_of: float


@dataclass(frozen=True)
class TierThresholdParameters:
    center_pop: float
    center_mainstream: float
    center_niche: float
    accessibility_pop: float
    accessibility_mainstream: float


@dataclass(frozen=True)
class ToneTargets:
    pop: dict[str, float]
    mainstream: dict[str, float]
    niche: dict[str, float]


@dataclass(frozen=True)
class RuntimeGenerationParameters:
    sampling: SamplingParameters
    popularity_blends: PopularityBlendParameters
    slot_multipliers: SlotMultiplierParameters
    article: ArticleParameters
    remix: RemixParameters


@dataclass(frozen=True)
class ModelRegistry:
    rater: str
    proposer: str
    jacket: str
    responses_only: frozenset[str]


def _cfg(conn: sqlite3.Connection | None = None) -> dict[str, float]:
    return load_tuning_config(conn)


def get_sampling_parameters(conn: sqlite3.Connection | None = None) -> SamplingParameters:
    cfg = _cfg(conn)
    return SamplingParameters(
        weighted_sample_spread=cfg["weighted_sample_spread"],
        weighted_sample_bias_floor=cfg["weighted_sample_bias_floor"],
    )


def get_popularity_parameters(conn: sqlite3.Connection | None = None) -> PopularityParameters:
    cfg = _cfg(conn)
    return PopularityParameters(
        weight_spl=cfg["pop_weight_spl"],
        weight_ol=cfg["pop_weight_ol"],
        weight_goodreads=cfg["pop_weight_gr"],
        weight_nyt=cfg["pop_weight_nyt"],
        weight_library=cfg["pop_weight_library"],
        weight_frequency=cfg["pop_weight_freq"],
        exponent=cfg["pop_exponent"],
    )


def get_popularity_blend_parameters(
    conn: sqlite3.Connection | None = None,
) -> PopularityBlendParameters:
    cfg = _cfg(conn)
    return PopularityBlendParameters(
        base_weight_blend=cfg["pop_base_weight_blend"],
        tone_blend=cfg["pop_tone_blend"],
        classification_blend=cfg["pop_classification_blend"],
        missing_default=cfg["pop_missing_default"],
    )


def get_slot_multiplier_parameters(
    conn: sqlite3.Connection | None = None,
) -> SlotMultiplierParameters:
    cfg = _cfg(conn)
    return SlotMultiplierParameters(
        list_item=cfg["pop_slot_mult_list_item"],
        action_noun=cfg["pop_slot_mult_action_noun"],
        of_object=cfg["pop_slot_mult_of_object"],
    )


def get_article_parameters(conn: sqlite3.Connection | None = None) -> ArticleParameters:
    cfg = _cfg(conn)
    return ArticleParameters(
        of_min_freq=cfg["article_of_min_freq"],
        action_min_freq=cfg["article_action_min_freq"],
        remix_heuristic_threshold=cfg["article_remix_heuristic_threshold"],
    )


def get_remix_parameters(conn: sqlite3.Connection | None = None) -> RemixParameters:
    cfg = _cfg(conn)
    return RemixParameters(reject_double_of=cfg["remix_reject_double_of"])


def get_tier_threshold_parameters(
    conn: sqlite3.Connection | None = None,
) -> TierThresholdParameters:
    cfg = _cfg(conn)
    return TierThresholdParameters(
        center_pop=cfg["tier_center_pop"],
        center_mainstream=cfg["tier_center_mainstream"],
        center_niche=cfg["tier_center_niche"],
        accessibility_pop=cfg["accessibility_threshold_pop"],
        accessibility_mainstream=cfg["accessibility_threshold_mainstream"],
    )


def get_tone_targets(conn: sqlite3.Connection | None = None) -> ToneTargets:
    cfg = _cfg(conn)
    return ToneTargets(
        pop={
            "list_item": cfg["tone_target_pop_list_item"],
            "action_noun": cfg["tone_target_pop_action_noun"],
            "of_object": cfg["tone_target_pop_of_object"],
        },
        mainstream={
            "list_item": cfg["tone_target_mainstream_list_item"],
            "action_noun": cfg["tone_target_mainstream_action_noun"],
            "of_object": cfg["tone_target_mainstream_of_object"],
        },
        niche={
            "list_item": cfg["tone_target_niche_list_item"],
            "action_noun": cfg["tone_target_niche_action_noun"],
            "of_object": cfg["tone_target_niche_of_object"],
        },
    )


def get_runtime_generation_parameters(
    conn: sqlite3.Connection | None = None,
) -> RuntimeGenerationParameters:
    return RuntimeGenerationParameters(
        sampling=get_sampling_parameters(conn),
        popularity_blends=get_popularity_blend_parameters(conn),
        slot_multipliers=get_slot_multiplier_parameters(conn),
        article=get_article_parameters(conn),
        remix=get_remix_parameters(conn),
    )


def get_model_registry() -> ModelRegistry:
    return ModelRegistry(
        rater=DEFAULT_RATER_MODEL,
        proposer=DEFAULT_PROPOSER_MODEL,
        jacket=DEFAULT_JACKET_MODEL,
        responses_only=RESPONSES_ONLY_MODELS,
    )
