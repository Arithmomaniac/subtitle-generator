"""Single source of truth for all tunable parameters and DB config loading."""

import sqlite3
from functools import lru_cache

# All tunable parameters with their default values.
# These are used as fallback when the DB config table has no tuned value.
ALL_TUNABLE_PARAMS: dict[str, float] = {
    "generation_tier_ratio_pop": 0.0183,
    "generation_tier_ratio_mainstream": 0.1172,
    "generation_tier_ratio_niche": 0.8645,
    "article_of_min_freq": 1.0,
    "article_action_min_freq": 1.0,
    "article_remix_heuristic_threshold": 0.6,
    "remix_reject_double_of": 1.0,
    # Popularity scoring params
    "pop_weight_spl": 0.7,
    "pop_weight_ol": 0.3,
    "pop_weight_gr": 0.2,       # Weight of Goodreads ratings signal
    "pop_weight_nyt": 0.1,      # Weight of NYT bestseller signal
    "pop_weight_library": 0.05, # Weight of other library lists signal
    "pop_weight_trove": 0.10,   # Weight of Trove Australia holdings signal
    "pop_weight_freq": 0.0,
    "pop_exponent": 1.2,
    "pop_missing_default": 0.1,
    # Final assembled-subtitle tier classifier coefficients. With these defaults,
    # the classifier preserves the previous mean-of-slot-probabilities behavior.
    "tier_classifier_model_score_weight": 1.0,
    "tier_classifier_temperature": 1.0,
    "tier_classifier_slot_weight_list_item": 1.0,
    "tier_classifier_slot_weight_action_noun": 1.0,
    "tier_classifier_slot_weight_of_object": 1.0,
    "tier_classifier_intercept_pop": 0.0,
    "tier_classifier_intercept_mainstream": 0.0,
    "tier_classifier_intercept_niche": 0.0,
    "tier_classifier_popularity_weight_pop": 0.0,
    "tier_classifier_popularity_weight_mainstream": 0.0,
    "tier_classifier_popularity_weight_niche": 0.0,
    "tier_classifier_popularity_interaction_pop": 0.0,
    "tier_classifier_popularity_interaction_mainstream": 0.0,
    "tier_classifier_popularity_interaction_niche": 0.0,
    "tier_classifier_frequency_weight_pop": 0.0,
    "tier_classifier_frequency_weight_mainstream": 0.0,
    "tier_classifier_frequency_weight_niche": 0.0,
}


# Cache keyed by connection id — avoids repeated DB queries within a request.
# The cache is small (one entry per unique connection) and auto-evicts.
@lru_cache(maxsize=4)
def _load_from_db(conn_id: int, conn: sqlite3.Connection) -> dict[str, float]:
    """Internal: load config rows from DB (cached by connection identity)."""
    overrides: dict[str, float] = {}
    try:
        rows = conn.execute("SELECT key, value FROM config").fetchall()
        for key, value in rows:
            if key in ALL_TUNABLE_PARAMS:
                overrides[key] = float(value)
    except Exception:
        pass  # table might not exist yet
    return overrides


def load_tuning_config(conn: sqlite3.Connection | None = None) -> dict[str, float]:
    """Load all tuning parameters from DB, falling back to defaults.

    Returns a dict with all keys from ALL_TUNABLE_PARAMS, using DB values
    where present and defaults otherwise. Results are cached per connection
    to avoid repeated DB queries within a single request.
    """
    config = dict(ALL_TUNABLE_PARAMS)  # start with defaults
    if conn is None:
        return config
    overrides = _load_from_db(id(conn), conn)
    config.update(overrides)
    return config


def invalidate_config_cache() -> None:
    """Clear the config cache. Call after writing to the config table."""
    _load_from_db.cache_clear()
