"""Single source of truth for all tunable parameters and DB config loading."""

import sqlite3

# All tunable parameters with their default values.
# These are used as fallback when the DB config table has no tuned value.
ALL_TUNABLE_PARAMS: dict[str, float] = {
    "weighted_sample_spread": 0.4,
    "weighted_sample_bias_floor": 0.05,
    "tone_target_pop_list_item": 1.5,
    "tone_target_pop_action_noun": 1.5,
    "tone_target_pop_of_object": 1.0,
    "tone_target_mainstream_list_item": 1.0,
    "tone_target_mainstream_action_noun": 1.0,
    "tone_target_mainstream_of_object": 0.8,
    "tone_target_niche_list_item": 0.25,
    "tone_target_niche_action_noun": 0.25,
    "tone_target_niche_of_object": 0.25,
    "sample_tone_spread": 0.6,
    "tier_center_pop": 1.5,
    "tier_center_mainstream": 0.75,
    "tier_center_niche": 0.25,
    "accessibility_threshold_pop": 1.0,
    "accessibility_threshold_mainstream": 0.5,
}


def load_tuning_config(conn: sqlite3.Connection | None = None) -> dict[str, float]:
    """Load all tuning parameters from DB, falling back to defaults.

    Returns a dict with all keys from ALL_TUNABLE_PARAMS, using DB values
    where present and defaults otherwise. Single DB query.
    """
    config = dict(ALL_TUNABLE_PARAMS)  # start with defaults
    if conn is None:
        return config
    try:
        rows = conn.execute("SELECT key, value FROM config").fetchall()
        for key, value in rows:
            if key in ALL_TUNABLE_PARAMS:
                config[key] = float(value)
    except Exception:
        pass  # table might not exist yet
    return config


def get_tone_targets(conn: sqlite3.Connection | None = None) -> dict[str, dict[str, float]]:
    """Get TONE_TARGETS dict from config. Format: {tier: {slot: target}}."""
    cfg = load_tuning_config(conn)
    targets: dict[str, dict[str, float]] = {}
    for tier in ("pop", "mainstream", "niche"):
        targets[tier] = {}
        for slot in ("list_item", "action_noun", "of_object"):
            targets[tier][slot] = cfg[f"tone_target_{tier}_{slot}"]
    return targets


# Module-level default for backward compatibility (import without DB)
DEFAULT_TONE_TARGETS = get_tone_targets()
