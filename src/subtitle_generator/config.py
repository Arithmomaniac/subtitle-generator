"""Single source of truth for all tunable parameters and DB config loading."""

import sqlite3
from functools import lru_cache

# All tunable parameters with their default values.
# These are used as fallback when the DB config table has no tuned value.
ALL_TUNABLE_PARAMS: dict[str, float] = {
    "weighted_sample_spread": 0.12,
    "weighted_sample_bias_floor": 0.05,
    "default_generation_tone_target": 2.0,
    "generation_tier_ratio_pop": 0.0183,
    "generation_tier_ratio_mainstream": 0.1172,
    "generation_tier_ratio_niche": 0.8645,
    "tier_center_pop": 0.75,
    "tier_center_mainstream": 0.4005,
    "tier_center_niche": 0.301,
    "accessibility_threshold_pop": 0.3665,
    "accessibility_threshold_mainstream": 0.3098,
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
    "pop_base_weight_blend": 0.5,
    "pop_classification_blend": 0.9,
    "pop_missing_default": 0.1,
    # Evidence-aware jacket tier classification params
    "tier_pop_min_demand_confidence": 0.8001,
    "tier_pop_min_lower_tail": 0.352,
    # Per-slot popularity multipliers (applied to tier center before Gaussian bias)
    "pop_slot_mult_list_item": 0.8,
    "pop_slot_mult_action_noun": 0.9,
    "pop_slot_mult_of_object": 1.0,
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


def get_tone_targets(conn: sqlite3.Connection | None = None) -> dict[str, dict[str, float]]:
    """Get base tone targets derived from tier centers."""
    cfg = load_tuning_config(conn)
    targets: dict[str, dict[str, float]] = {}
    for tier in ("pop", "mainstream", "niche"):
        center = cfg[f"tier_center_{tier}"]
        targets[tier] = {
            slot: center for slot in ("list_item", "action_noun", "of_object")
        }
    return targets


# Module-level default for backward compatibility (import without DB)
DEFAULT_TONE_TARGETS = get_tone_targets()
