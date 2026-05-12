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
class GenerationTierRatios:
    pop: float
    mainstream: float
    niche: float


@dataclass(frozen=True)
class PopularityParameters:
    weight_spl: float
    weight_ol: float
    weight_goodreads: float
    weight_nyt: float
    weight_library: float
    weight_trove: float
    weight_frequency: float
    exponent: float


@dataclass(frozen=True)
class ArticleParameters:
    of_min_freq: float
    action_min_freq: float
    remix_heuristic_threshold: float


@dataclass(frozen=True)
class RemixParameters:
    reject_double_of: float


@dataclass(frozen=True)
class RuntimeGenerationParameters:
    generation_tier_ratios: GenerationTierRatios
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


def get_generation_tier_ratios(
    conn: sqlite3.Connection | None = None,
) -> GenerationTierRatios:
    cfg = _cfg(conn)
    return GenerationTierRatios(
        pop=cfg["generation_tier_ratio_pop"],
        mainstream=cfg["generation_tier_ratio_mainstream"],
        niche=cfg["generation_tier_ratio_niche"],
    )


def get_popularity_parameters(conn: sqlite3.Connection | None = None) -> PopularityParameters:
    cfg = _cfg(conn)
    return PopularityParameters(
        weight_spl=cfg["pop_weight_spl"],
        weight_ol=cfg["pop_weight_ol"],
        weight_goodreads=cfg["pop_weight_gr"],
        weight_nyt=cfg["pop_weight_nyt"],
        weight_library=cfg["pop_weight_library"],
        weight_trove=cfg["pop_weight_trove"],
        weight_frequency=cfg["pop_weight_freq"],
        exponent=cfg["pop_exponent"],
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


def get_runtime_generation_parameters(
    conn: sqlite3.Connection | None = None,
) -> RuntimeGenerationParameters:
    return RuntimeGenerationParameters(
        generation_tier_ratios=get_generation_tier_ratios(conn),
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
