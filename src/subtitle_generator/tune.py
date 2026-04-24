"""Autoresearch-inspired tuning loop for subtitle generation parameters.

Adapts Karpathy's autoresearch pattern: instead of modifying code and
training a model, we modify DB config values and evaluate subtitle quality.

Two phases:
  Phase 1 (remix): Grid sweep via calibrate.run_calibration()
  Phase 2 (tone):  LLM-proposed single-parameter hill-climbing
"""

from __future__ import annotations

import json
import pathlib
import re
import sqlite3
import subprocess
import sys

import click

from subtitle_generator.config import ALL_TUNABLE_PARAMS, invalidate_config_cache, load_tuning_config
from subtitle_generator.eval_harness import (
    DEFAULT_PROPOSER_MODEL,
    DEFAULT_RATER_MODEL,
    ParamProposal,
    composite_score,
    generate_sample_set,
    measure_tone_separation,
    rate_quality,
    structured_completion,
)
from subtitle_generator.feedback import store_rating

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Parameters that require re-running populate-popularity to take effect.
# These change the composite scoring formula, not just generation-time behavior.
_REPOPULATE_PARAMS = frozenset({
    "pop_weight_spl", "pop_weight_ol", "pop_weight_gr",
    "pop_weight_nyt", "pop_weight_library", "pop_exponent",
})

# Params that don't change stored filler scores but DO change the blended-score
# distribution used for tier classification. Trigger calibrate-only (no repopulate).
_CALIBRATE_ONLY_PARAMS = frozenset({
    "pop_classification_blend", "pop_missing_default",
})


def _needs_repopulate(param: str) -> bool:
    """Check if changing this param requires a populate-popularity re-run."""
    return param in _REPOPULATE_PARAMS


def _needs_calibrate(param: str) -> bool:
    """Check if changing this param requires recalibrating thresholds/tier centers."""
    return param in _REPOPULATE_PARAMS or param in _CALIBRATE_ONLY_PARAMS


def _run_calibrate_thresholds(conn: sqlite3.Connection) -> None:
    """Re-derive accessibility thresholds, tier centers, and tone targets from
    the current blended-score distribution. Cheap; safe to call after any
    repopulate or after a pop_classification_blend / pop_missing_default change.
    """
    import importlib
    sys.path.insert(0, "data")
    import populate_popularity as pp
    importlib.reload(pp)
    click.echo("  [calibrate] recomputing thresholds + tier centers...")
    pp.calibrate_thresholds(conn)
    conn.commit()
    invalidate_config_cache()


# Cached popularity data (loaded once, reused across tuner iterations)
_pop_data_cache: dict | None = None
_work_data_cache: dict = {}
_filler_works_cache: dict | None = None  # filler_id -> list of work_keys
_filler_freq_cache: dict | None = None   # filler_id -> freq (for fallback)


def _load_pop_data():
    """Load all popularity JSON files (cached across calls)."""
    global _pop_data_cache
    if _pop_data_cache is not None:
        return _pop_data_cache

    import json
    from pathlib import Path

    click.echo("  [repopulate] loading popularity data (first time, will cache)...")
    data = {}
    spl_path = Path("data/spl_checkout_lookup.json")
    ol_path = Path("data/ol_edition_lookup.json")
    gr_path = Path("data/goodreads_lookup.json")
    ottawa_path = Path("data/canadian_library_lookup.json")
    nyt_path = Path("data/nyt_bestseller_lookup.json")

    with open(spl_path) as f:
        data["spl"] = json.load(f)
    with open(ol_path) as f:
        data["ol"] = json.load(f)

    if gr_path.exists():
        with open(gr_path) as f:
            data["gr"] = json.load(f)
    else:
        data["gr"] = {}

    if ottawa_path.exists():
        with open(ottawa_path) as f:
            raw = json.load(f)
        data["ottawa_isbn"] = {k: v for k, v in raw.items() if k.replace("-", "").isdigit()}
    else:
        data["ottawa_isbn"] = {}

    if nyt_path.exists():
        with open(nyt_path) as f:
            data["nyt"] = json.load(f)
    else:
        data["nyt"] = {}

    _pop_data_cache = data
    click.echo(f"  [repopulate] cached: SPL={len(data['spl']):,}, OL={len(data['ol']):,}, "
               f"GR={len(data['gr']):,}, Ottawa={len(data['ottawa_isbn']):,}, NYT={len(data['nyt']):,}")
    return data


def _load_filler_works(conn: sqlite3.Connection):
    """Build filler_id -> [work_keys] mapping (cached)."""
    global _filler_works_cache, _filler_freq_cache
    if _filler_works_cache is not None:
        return _filler_works_cache, _filler_freq_cache

    click.echo("  [repopulate] building filler->work mapping (first time)...")
    from collections import defaultdict

    rows = conn.execute("""
        SELECT sf.id, ia.work_key
        FROM slot_fillers sf
        JOIN slot_filler_sources sfs ON sfs.slot_filler_id = sf.id
        JOIN subtitles s ON sfs.subtitle_id = s.id
        JOIN isbn_aliases ia ON ia.isbn = s.isbn
        WHERE ia.work_key IS NOT NULL AND sf.mode = 'strict'
    """).fetchall()

    filler_works = defaultdict(list)
    for fid, wk in rows:
        filler_works[fid].append(wk)
    # Deduplicate
    _filler_works_cache = {fid: list(set(wks)) for fid, wks in filler_works.items()}

    # Also cache freq for fallback scoring
    freq_rows = conn.execute(
        "SELECT id, freq FROM slot_fillers WHERE mode = 'strict'"
    ).fetchall()
    _filler_freq_cache = {r[0]: r[1] for r in freq_rows}

    click.echo(f"  [repopulate] cached {len(_filler_works_cache):,} fillers with work mappings")
    return _filler_works_cache, _filler_freq_cache


def _score_in_memory(conn: sqlite3.Connection):
    """Compute composite scores in memory and write only filler scores to DB.

    Returns the work_composites dict for use by _flush_repopulate if kept.
    """
    import bisect
    import math
    import time as _time

    t0 = _time.time()
    data = _load_pop_data()
    cfg = load_tuning_config(conn)

    w_spl = cfg.get("pop_weight_spl", 0.7)
    w_ol = cfg.get("pop_weight_ol", 0.3)
    w_gr = cfg.get("pop_weight_gr", 0.2)
    w_library = cfg.get("pop_weight_library", 0.05)
    w_nyt = cfg.get("pop_weight_nyt", 0.1)

    click.echo(f"  [repopulate] scoring in-memory (SPL={w_spl}, OL={w_ol}, "
               f"GR={w_gr}, LIB={w_library}, NYT={w_nyt})...")

    # Ensure work-level data is cached
    if "work_spl" not in _work_data_cache:
        # Need a full run first to populate the cache
        click.echo("  [repopulate] cold start: need full populate for work data cache")
        _run_repopulate_full(conn)
        return None

    work_spl = _work_data_cache["work_spl"]
    work_ol = _work_data_cache["work_ol"]
    work_gr = _work_data_cache["work_gr"]
    work_ottawa = _work_data_cache["work_ottawa"]
    work_nyt = _work_data_cache["work_nyt"]
    all_works = _work_data_cache["all_works"]

    # Build percentile arrays
    spl_log_vals = sorted([math.log10(1 + d["checkouts"] / max(d["years"], 1))
                           for d in work_spl.values() if d["checkouts"] > 0])
    gr_log_vals = sorted([math.log10(1 + d["ratings_count"]) for d in work_gr.values()])
    ol_log_vals = sorted([math.log10(1 + ec) for ec in work_ol.values()])
    lib_log_vals = sorted([math.log10(1 + d["holds_count"]) for d in work_ottawa.values()])

    n_spl = len(spl_log_vals) or 1
    n_gr = len(gr_log_vals) or 1
    n_ol = len(ol_log_vals) or 1
    n_lib = len(lib_log_vals) or 1

    # Score all works in memory (dict, no DB)
    work_composites: dict[str, float] = {}
    denom = w_spl + w_gr + w_library + w_nyt

    for work in all_works:
        signals = []
        total_weight = 0.0

        spl_data = work_spl.get(work)
        if spl_data and spl_data["checkouts"] > 0:
            co_per_year = spl_data["checkouts"] / max(spl_data["years"], 1)
            spl_norm = bisect.bisect_left(spl_log_vals, math.log10(1 + co_per_year)) / n_spl
            signals.append((w_spl, spl_norm))
            total_weight += w_spl

        gr_data = work_gr.get(work)
        if gr_data:
            gr_norm = bisect.bisect_left(gr_log_vals, math.log10(1 + gr_data["ratings_count"])) / n_gr
            signals.append((w_gr, gr_norm))
            total_weight += w_gr

        can_data = work_ottawa.get(work)
        if can_data:
            lib_norm = bisect.bisect_left(lib_log_vals, math.log10(1 + can_data["holds_count"])) / n_lib
            signals.append((w_library, lib_norm))
            total_weight += w_library

        nyt_data = work_nyt.get(work)
        if nyt_data:
            nyt_norm = min(1.0, 0.8 + 0.2 * math.log10(1 + nyt_data["weeks_on_list"]) / 2.0)
            signals.append((w_nyt, nyt_norm))
            total_weight += w_nyt

        if total_weight > 0:
            demand_score = sum(w * s for w, s in signals) / total_weight
        else:
            demand_score = 0.0

        ol_ec = work_ol.get(work, 1)
        ol_norm = bisect.bisect_left(ol_log_vals, math.log10(1 + ol_ec)) / n_ol
        confidence = min(total_weight / denom, 1.0) if denom > 0 else 0.0
        composite = confidence * demand_score + (1 - confidence) * ol_norm
        if confidence == 0:
            composite = min(composite, 0.5)

        work_composites[work] = composite

    t1 = _time.time()

    # Compute filler scores from in-memory composites (top-3 mean)
    filler_works, filler_freq = _load_filler_works(conn)

    filler_updates = []  # (popularity_score, popularity_level, filler_id)
    for fid, wkeys in filler_works.items():
        scores = sorted([work_composites.get(wk, 0.0) for wk in wkeys], reverse=True)
        top3 = scores[:3]
        avg = sum(top3) / len(top3) if top3 else 0.0
        filler_updates.append((avg, 1, 1.0, fid))

    # Fallback for fillers without L1 data
    l1_ids = set(filler_works.keys())
    for fid, freq in filler_freq.items():
        if fid not in l1_ids:
            score = math.log10(1 + freq) if freq > 0 else 0.0
            filler_updates.append((score, 0, 0.0, fid))

    # Write only filler scores to DB (~13k rows)
    conn.executemany(
        "UPDATE slot_fillers SET popularity_score=?, popularity_level=?, popularity_confidence=? WHERE id=?",
        filler_updates,
    )
    conn.commit()

    elapsed = _time.time() - t0
    click.echo(f"  [repopulate] scored {len(all_works):,} works in memory, "
               f"updated {len(filler_updates):,} fillers in {elapsed:.0f}s")
    return work_composites


def _run_repopulate_full(conn: sqlite3.Connection):
    """Full repopulate: recompute and write everything to DB (for 'keep' path)."""
    import importlib
    import time as _time

    click.echo("  [repopulate] full DB write...")
    t0 = _time.time()

    data = _load_pop_data()
    cfg = load_tuning_config(conn)

    w_spl = cfg.get("pop_weight_spl", 0.7)
    w_ol = cfg.get("pop_weight_ol", 0.3)
    w_gr = cfg.get("pop_weight_gr", 0.2)
    w_library = cfg.get("pop_weight_library", 0.05)
    w_nyt = cfg.get("pop_weight_nyt", 0.1)
    exponent = cfg.get("pop_exponent", 1.2)

    sys.path.insert(0, "data")
    import populate_popularity as pp
    importlib.reload(pp)

    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA cache_size=-65536")
    conn.execute("PRAGMA mmap_size=268435456")
    conn.execute("PRAGMA temp_store=MEMORY")

    pp.create_tables(conn)
    pp.populate_work_level(
        conn, data["spl"], data["ol"],
        w_spl=w_spl, w_ol=w_ol, exponent=exponent,
        gr=data["gr"], w_gr=w_gr,
        ottawa_isbn=data["ottawa_isbn"], w_library=w_library,
        nyt=data["nyt"], w_nyt=w_nyt,
        work_data_cache=_work_data_cache,
    )
    pp.score_fillers_level1(conn)
    pp.score_fillers_fallback(conn)

    elapsed = _time.time() - t0
    click.echo(f"  [repopulate] full write done in {elapsed:.0f}s")
    _run_calibrate_thresholds(conn)


def _run_repopulate(conn: sqlite3.Connection):
    """Fast repopulate: score in memory, write only filler scores."""
    _score_in_memory(conn)
    _run_calibrate_thresholds(conn)


def _load_goals() -> str:
    """Read tuning_goals.md from repo root."""
    goals_path = pathlib.Path(__file__).parent.parent.parent / "tuning_goals.md"
    if goals_path.exists():
        return goals_path.read_text(encoding="utf-8")
    return "(no tuning_goals.md found)"


def _parse_bounds(goals_text: str) -> dict[str, tuple[float, float]]:
    """Extract parameter bounds from the tuning_goals.md table.

    Matches rows like:
      | `weighted_sample_spread` | 0.1 | 1.0 | 0.4 | ... |
    Also handles wildcard rows like:
      | `tone_target_pop_*` | 0.5 | 2.5 | 1.0–1.5 | ... |
    """
    bounds: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
        r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", goals_text
    ):
        pattern, lo, hi = match.group(1), float(match.group(2)), float(match.group(3))
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            for key in ALL_TUNABLE_PARAMS:
                if key.startswith(prefix):
                    bounds[key] = (lo, hi)
        else:
            bounds[pattern] = (lo, hi)
    return bounds


def _format_bounds(bounds: dict[str, tuple[float, float]]) -> str:
    """Format bounds dict for the proposer prompt."""
    lines = []
    for key in sorted(bounds):
        lo, hi = bounds[key]
        lines.append(f"  {key}: [{lo}, {hi}]")
    return "\n".join(lines) if lines else "(no bounds specified)"


def _load_results_history(results_file: str, max_lines: int = 20) -> str:
    """Load recent results for the proposer's context.

    Always includes regime-change markers even if they're outside the
    last max_lines, so the proposer knows about param availability changes.
    """
    path = pathlib.Path(results_file)
    if not path.exists():
        return "(no previous experiments)"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) <= max_lines + 1:
        return "\n".join(lines)
    # Always include the header + any regime-change lines + last max_lines
    header = lines[0]
    regime_lines = [l for l in lines[1:-max_lines] if "[regime change]" in l]
    recent = lines[-max_lines:]
    parts = [header] + regime_lines + recent
    return "\n".join(parts)


def _ensure_results_header(results_file: str) -> None:
    """Create results TSV with header if it doesn't exist."""
    path = pathlib.Path(results_file)
    if not path.exists():
        path.write_text(
            "iteration\tparam\told_value\tnew_value\t"
            "quality\tseparation\tcomposite\tstatus\tdescription\n",
            encoding="utf-8",
        )


def _check_regime_change(results_file: str) -> None:
    """Insert a regime-change marker if available params changed since last run.

    Scans the TSV for the most recent regime marker (or all experiment rows if none)
    to determine which params were available. If ALL_TUNABLE_PARAMS has new keys,
    appends a marker row so the proposer knows old history is from a different regime.
    """
    path = pathlib.Path(results_file)
    if not path.exists():
        return

    current_params = sorted(ALL_TUNABLE_PARAMS.keys())
    lines = path.read_text(encoding="utf-8").strip().split("\n")

    # Find the most recent regime marker
    last_regime_params = None
    for line in reversed(lines):
        if line.startswith("---\t[regime change]"):
            # Extract param list from description
            parts = line.split("\t")
            if len(parts) >= 9:
                desc = parts[8]
                if "available_params=" in desc:
                    param_str = desc.split("available_params=")[1]
                    last_regime_params = sorted(param_str.split(","))
            break

    if last_regime_params is None:
        # No regime marker yet — extract params mentioned in experiment rows
        mentioned = set()
        for line in lines[1:]:  # skip header
            if line.startswith("---"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1] not in ("(failed)", "[regime change]"):
                mentioned.add(parts[1])
        # If we have new params that were never mentioned and never regime-marked, add marker
        if mentioned and set(current_params) - mentioned:
            new_params = sorted(set(current_params) - mentioned)
            _append_result(
                results_file, 0, "[regime change]", 0, 0, 0, 0, 0,
                "regime",
                f"New params added: {', '.join(new_params)}. "
                f"History above is from a prior regime without these params. "
                f"available_params={','.join(current_params)}",
            )
            return

    if last_regime_params is not None and last_regime_params != current_params:
        new_params = sorted(set(current_params) - set(last_regime_params))
        removed_params = sorted(set(last_regime_params) - set(current_params))
        desc_parts = []
        if new_params:
            desc_parts.append(f"New params added: {', '.join(new_params)}.")
        if removed_params:
            desc_parts.append(f"Params removed: {', '.join(removed_params)}.")
        desc_parts.append(f"available_params={','.join(current_params)}")
        _append_result(
            results_file, 0, "[regime change]", 0, 0, 0, 0, 0,
            "regime", " ".join(desc_parts),
        )


def _append_result(
    results_file: str,
    iteration: int,
    param: str,
    old_value: float,
    new_value: float,
    quality: float,
    separation: float,
    comp: float,
    status: str,
    description: str,
) -> None:
    """Append one line to the results TSV."""
    # Sanitize description: tabs/newlines would corrupt TSV parsing
    safe_desc = description.replace("\t", " ").replace("\n", " ").replace("\r", "")
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(
            f"{iteration}\t{param}\t{old_value}\t{new_value}\t"
            f"{quality:.4f}\t{separation:.4f}\t{comp:.4f}\t"
            f"{status}\t{safe_desc}\n"
        )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _evaluate(
    conn: sqlite3.Connection,
    rater_model: str,
    n_samples: int = 50,
    seed_base: int = 1000,
    quality_weight: float = 0.5,
) -> tuple[float, float, float]:
    """Generate samples, rate quality and tone separation.

    Returns (quality, separation, composite).
    """
    click.echo("  generating sample subtitles …")
    subtitles = generate_sample_set(conn, n=n_samples, seed_base=seed_base)
    texts = [sub.text for sub in subtitles]
    quality = rate_quality(texts, model=rater_model)

    separation = measure_tone_separation(conn, seed_base=seed_base + n_samples)

    comp = composite_score(quality, separation, quality_weight=quality_weight)
    return quality, separation, comp


# ---------------------------------------------------------------------------
# Standalone spot-check (decoupled from tune loop)
# ---------------------------------------------------------------------------


def run_spot_check(
    conn: sqlite3.Connection,
    n_samples: int = 2,
    source: str = "spot_check",
    seed_base: int | None = None,
) -> float | None:
    """Generate tone-targeted samples and collect human tier ratings.

    Standalone command — not part of the tune loop. Stores ratings in
    human_ratings for later analysis via review_ratings().

    For a richer UI, use the web spot-check page at /spot-check.html.

    Returns tone-accuracy (0.0-1.0) or None if all skipped.
    """
    import random as _rng
    from subtitle_generator.config import get_tone_targets
    from subtitle_generator.generate import generate_subtitle

    if seed_base is None:
        seed_base = _rng.randint(0, 100000)

    targets = get_tone_targets(conn)
    tiers = ["pop", "mainstream", "niche"]
    tier_labels = {"pop": "\U0001f525 POP", "mainstream": "\U0001f4da MAINSTREAM", "niche": "\U0001f393 NICHE"}
    tier_shortcuts = {"p": "pop", "m": "mainstream", "n": "niche"}

    click.echo(click.style(
        f"=== Spot Check ({n_samples} per tier, {n_samples * 3} total) ===\n",
        fg="green", bold=True,
    ))

    all_samples: list[tuple[str, str, object]] = []
    for tier in tiers:
        tone_target = {
            slot: targets[tier][slot]
            for slot in ["list_item", "action_noun", "of_object"]
        }
        for j in range(n_samples):
            sub = generate_subtitle(
                conn,
                seed=seed_base + tiers.index(tier) * 100 + j,
                tone_target=tone_target,
            )
            all_samples.append((tier, sub.text, sub))

    accuracy = _spot_check_cli(conn, all_samples, tier_labels, tier_shortcuts, source)

    if accuracy is not None:
        click.echo(f"\nRatings stored (source={source}). Run 'subtitle-gen review-ratings' to analyze.")
    return accuracy


def _spot_check_cli(
    conn: sqlite3.Connection,
    samples: list[tuple[str, str, object]],
    tier_labels: dict[str, str],
    tier_shortcuts: dict[str, str],
    source: str = "spot_check",
) -> float | None:
    """CLI spot-check: sequential prompts per subtitle."""
    import random as _rng
    shuffled = list(samples)
    _rng.shuffle(shuffled)

    total = 0
    correct = 0
    labels = "abcdefghijklmnopqrstuvwxyz"

    for i, (target_tier, text, sub) in enumerate(shuffled):
        label = labels[i] if i < len(labels) else str(i + 1)
        click.echo(f"    {label}) {text}")
        click.echo(click.style(
            f"       Target: {tier_labels[target_tier]}",
            fg="cyan", dim=True,
        ))
        response = click.prompt(
            click.style("       Feels like? [p/m/n/Enter=skip]", fg="green"),
            default="", show_default=False,
        ).strip().lower()

        perceived = tier_shortcuts.get(response)
        if perceived:
            total += 1
            if perceived == target_tier:
                correct += 1
                click.echo(click.style("       \u2713 match", fg="green"))
            else:
                click.echo(click.style(
                    f"       \u2717 mismatch (target={target_tier}, felt={perceived})",
                    fg="yellow",
                ))

            tags_input = click.prompt(
                click.style("       Tags? [f=funny/b=boring/r=broken/n=nonsense / Enter]", fg="cyan"),
                default="", show_default=False,
            ).strip().lower()
            tag_map = {"f": "funny", "b": "boring", "r": "broken", "n": "nonsense"}
            tags = [tag_map[c] for c in tags_input if c in tag_map] or None

            store_rating(
                conn, text,
                system_tone=target_tier,
                thumbs=1 if perceived == target_tier else -1,
                tone_override=perceived,
                tags=tags,
                source=source,
            )

    if total == 0:
        return None
    accuracy = correct / total
    click.echo(click.style(
        f"\n  Tone accuracy: {correct}/{total} ({accuracy:.0%})",
        fg="green" if accuracy >= 0.6 else "yellow",
    ))
    return accuracy




# ---------------------------------------------------------------------------
# Review ratings -> propose tuning_goals.md edits
# ---------------------------------------------------------------------------


def review_ratings(
    conn: sqlite3.Connection,
    since: str | None = None,
    source: str | None = None,
    model: str = DEFAULT_PROPOSER_MODEL,
) -> None:
    """Analyze human ratings and propose tuning_goals.md edits.

    Reads ratings, builds a mismatch summary, asks an LLM to propose
    specific edits to tuning_goals.md, and displays the diff for human
    approval. Does NOT write the file.
    """
    import json as _json
    from collections import Counter

    from subtitle_generator.feedback import ensure_ratings_table
    ensure_ratings_table(conn)

    query = "SELECT subtitle, system_tone, thumbs, tone_override, tags, source, created_at FROM human_ratings"
    conditions = []
    params = []
    if since:
        conditions.append("created_at >= ?")
        params.append(since)
    if source:
        conditions.append("source = ?")
        params.append(source)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY created_at DESC LIMIT 100"

    rows = conn.execute(query, params).fetchall()
    if not rows:
        click.echo("No ratings found matching filters.")
        return

    click.echo(f"Analyzing {len(rows)} ratings")
    if source:
        click.echo(f"  Source filter: {source}")

    total = 0
    matches = 0
    mismatch_directions = Counter()
    tag_counts = Counter()
    mismatch_examples = []

    for sub, sys_tone, thumbs, tone_override, tags_json, src, created in rows:
        if sys_tone and tone_override:
            total += 1
            if sys_tone == tone_override:
                matches += 1
            else:
                mismatch_directions[(sys_tone, tone_override)] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append((sys_tone, tone_override, sub))
        tags = _json.loads(tags_json) if tags_json else []
        for tag in tags:
            tag_counts[tag] += 1

    if total == 0:
        click.echo("No tone-rated entries found (need system_tone + tone_override).")
        return

    accuracy = matches / total
    click.echo(f"  Tone accuracy: {matches}/{total} ({accuracy:.0%})")
    click.echo(f"  Tags: {dict(tag_counts.most_common())}")

    summary_lines = [
        f"## Human Rating Analysis ({total} rated samples)",
        f"Tone accuracy: {matches}/{total} ({accuracy:.0%})",
        "",
        "### Mismatch patterns:",
    ]
    for (sys, felt), count in mismatch_directions.most_common():
        summary_lines.append(f"  target={sys} -> felt={felt}: {count}x")
    summary_lines.append("")
    summary_lines.append("### Mismatch examples:")
    for sys, felt, sub in mismatch_examples:
        summary_lines.append(f"  [{sys}->{felt}] {sub}")
    if tag_counts:
        summary_lines.append("")
        summary_lines.append(f"### Quality tags: {dict(tag_counts.most_common())}")
    summary_text = "\n".join(summary_lines)

    click.echo(f"\n{summary_text}\n")

    goals_text = _load_goals()

    prompt = f"""You are analyzing human feedback on a subtitle generator to propose
improvements to its tuning goals document.

## Current tuning_goals.md:
{goals_text}

## Human rating analysis:
{summary_text}

Based on the mismatch patterns and quality tags, propose SPECIFIC edits to
tuning_goals.md. Focus on:
1. Updating the exploration strategy based on what the data shows
2. Adjusting priority order if certain params clearly matter more/less
3. Adding observations about which slots/tiers are miscalibrated
4. Noting any quality issues (grammar, contradictions) that need attention

Output your proposed changes as a unified diff (--- old / +++ new format)
showing exactly which lines to change. Only include sections that need changes.
Do NOT rewrite the entire file -- show targeted edits.
"""

    click.echo("Generating proposed edits ...")
    try:
        from pydantic import BaseModel

        class GoalsEdit(BaseModel):
            diff: str
            reasoning: str

        result = structured_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema=GoalsEdit,
            timeout=300.0,
            max_retries=2,
        )

        click.echo(click.style("\n=== Proposed tuning_goals.md edits ===\n", bold=True))
        click.echo(result.diff)
        click.echo(click.style(f"\nReasoning: {result.reasoning}", fg="cyan"))
        click.echo(click.style(
            "\nTo apply: edit tuning_goals.md manually with the changes above.",
            fg="green",
        ))

    except Exception as e:
        click.echo(f"  Warning: LLM analysis failed: {e}")
        click.echo("  The rating summary above can still be used to manually edit tuning_goals.md.")





# ---------------------------------------------------------------------------
# Main tuning loop
# ---------------------------------------------------------------------------


def run_tone_tuning(
    conn: sqlite3.Connection,
    iterations: int = 30,
    rater_model: str = DEFAULT_RATER_MODEL,
    proposer_model: str = DEFAULT_PROPOSER_MODEL,
    results_file: str = "results.tsv",
    dry_run: bool = False,
) -> dict:
    """Autoresearch loop for tone parameters.

    Pure automated loop (no human input). Each iteration: propose a single
    parameter change via LLM, evaluate, keep if improved, revert otherwise.

    Human feedback flows through tuning_goals.md edits between runs,
    not through the loop itself (autoresearch pattern).

    Returns the final parameter dict.
    """
    _ensure_results_header(results_file)
    _check_regime_change(results_file)
    goals_text = _load_goals()
    bounds = _parse_bounds(goals_text)
    bounds_text = _format_bounds(bounds)

    # Baseline evaluation
    click.echo("Computing baseline scores …")
    current_params = load_tuning_config(conn)
    quality, separation, current_score = _evaluate(conn, rater_model)
    click.echo(
        f"Baseline — Quality: {quality:.3f}  "
        f"Separation: {separation:.3f}  "
        f"Composite: {current_score:.3f}\n"
    )

    for i in range(1, iterations + 1):
        click.echo(f"--- Iteration {i}/{iterations} ---")

        # Reload state each iteration
        current_params = load_tuning_config(conn)
        results_history = _load_results_history(results_file)

        # Propose a parameter change
        proposal_prompt = f"""You are tuning parameters for a subtitle generator.

## Current parameter values:
{json.dumps(current_params, indent=2)}

## Tuning goals:
{goals_text}

## Current scores:
- Quality: {quality:.3f}
- Tone separation: {separation:.3f}
- Composite: {current_score:.3f}

## Previous experiments:
{results_history}

## Parameter bounds:
{bounds_text}

Propose ONE parameter change that you think will improve the composite score.
Prioritize parameters marked as NEW in the priority order — they have never been tuned
and represent the biggest untapped improvement opportunity. The `pop_*` parameters were
specifically added to replace the old freq-only scoring with empirical popularity data.
Consider what previous experiments tell you about which direction to move.

NOTE: Changes to pop_weight_spl, pop_weight_ol, pop_weight_gr, pop_weight_nyt,
pop_weight_library, and pop_exponent automatically trigger in-memory rescoring
(~15s) so their effects are properly evaluated. Previous experiments with
these params that did NOT mention repopulate may have been evaluated against stale
scores — treat those results as unreliable.
"""

        click.echo("  proposing parameter change …")
        try:
            proposal = structured_completion(
                model=proposer_model,
                messages=[{"role": "user", "content": proposal_prompt}],
                schema=ParamProposal,
                timeout=300.0,
                max_retries=4,
            )
        except RuntimeError as e:
            click.echo(f"  ⚠ proposal failed: {e} — skipping iteration")
            _append_result(
                results_file, i, "(failed)", 0, 0,
                quality, separation, current_score,
                "error", str(e),
            )
            continue

        # Validate the proposed parameter
        if proposal.param not in ALL_TUNABLE_PARAMS:
            click.echo(
                f"  ⚠ proposed unknown param '{proposal.param}' — skipping"
            )
            _append_result(
                results_file, i, proposal.param, 0, proposal.new_value,
                quality, separation, current_score,
                "skip", f"unknown param: {proposal.reasoning}",
            )
            continue

        old_value = current_params[proposal.param]
        new_value = proposal.new_value

        # Clamp to bounds
        if proposal.param in bounds:
            lo, hi = bounds[proposal.param]
            clamped = max(lo, min(hi, new_value))
            if clamped != new_value:
                click.echo(
                    f"  ! clamping {proposal.param} "
                    f"{new_value} -> {clamped} (bounds [{lo}, {hi}])"
                )
                new_value = clamped

        click.echo(
            f"  proposal: {proposal.param} {old_value} -> {new_value}"
        )
        click.echo(f"  reason: {proposal.reasoning}")

        if dry_run:
            click.echo("  (dry run — skipping evaluation)\n")
            _append_result(
                results_file, i, proposal.param, old_value, new_value,
                quality, separation, current_score,
                "dry_run", proposal.reasoning,
            )
            continue

        # Apply the change
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (proposal.param, str(new_value)),
        )
        conn.commit()
        invalidate_config_cache()

        # Re-run populate-popularity if this is a weight/exponent param
        if _needs_repopulate(proposal.param):
            try:
                _run_repopulate(conn)
            except RuntimeError:
                click.echo("  -> SKIP (repopulate failed)\n")
                # Revert the config change
                if old_value == ALL_TUNABLE_PARAMS[proposal.param]:
                    conn.execute("DELETE FROM config WHERE key = ?", (proposal.param,))
                else:
                    conn.execute(
                        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                        (proposal.param, str(old_value)),
                    )
                conn.commit()
                invalidate_config_cache()
                _append_result(
                    results_file, i, proposal.param, old_value, new_value,
                    quality, separation, current_score,
                    "error", f"repopulate failed: {proposal.reasoning}",
                )
                continue
        elif _needs_calibrate(proposal.param):
            # Stored filler scores unchanged, but blended-score distribution
            # shifted -> re-derive thresholds + tier centers.
            _run_calibrate_thresholds(conn)

        # Evaluate with new value
        new_quality, new_separation, new_score = _evaluate(
            conn, rater_model, seed_base=1000 + i * 100,
        )

        delta = new_score - current_score

        if new_score > current_score:
            status = "keep"
            click.echo(
                f"  Quality: {quality:.3f} -> {new_quality:.3f}  "
                f"Separation: {separation:.3f} -> {new_separation:.3f}  "
                f"Composite: {current_score:.3f} -> {new_score:.3f}"
            )
            click.echo(f"  -> KEEP (+{delta:.3f})\n")
            # Flush full DB write for kept weight changes
            if _needs_repopulate(proposal.param):
                _run_repopulate_full(conn)
            quality, separation, current_score = (
                new_quality, new_separation, new_score,
            )
        else:
            status = "discard"
            # Revert: restore old value or remove if it was a default
            if old_value == ALL_TUNABLE_PARAMS[proposal.param]:
                conn.execute(
                    "DELETE FROM config WHERE key = ?", (proposal.param,)
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                    (proposal.param, str(old_value)),
                )
            conn.commit()
            invalidate_config_cache()
            # Restore old filler scores in memory (no full DB write needed)
            if _needs_repopulate(proposal.param):
                _run_repopulate(conn)
            elif _needs_calibrate(proposal.param):
                _run_calibrate_thresholds(conn)
            click.echo(
                f"  Quality: {quality:.3f} -> {new_quality:.3f}  "
                f"Separation: {separation:.3f} -> {new_separation:.3f}  "
                f"Composite: {current_score:.3f} -> {new_score:.3f}"
            )
            click.echo(f"  -> DISCARD ({delta:+.3f})\n")

        _append_result(
            results_file, i, proposal.param, old_value, new_value,
            new_quality, new_separation, new_score,
            status, proposal.reasoning,
        )

    final_params = load_tuning_config(conn)
    click.echo(f"=== Tuning complete ({iterations} iterations) ===")
    click.echo(f"Final composite: {current_score:.3f}")
    return final_params


# ---------------------------------------------------------------------------
# Full tuning orchestrator
# ---------------------------------------------------------------------------


def run_full_tuning(
    conn: sqlite3.Connection,
    phase: str = "all",
    iterations: int = 30,
    samples: int = 50,
    rater_model: str = DEFAULT_RATER_MODEL,
    proposer_model: str = DEFAULT_PROPOSER_MODEL,
    results_file: str = "results.tsv",
    dry_run: bool = False,
) -> None:
    """Run both tuning phases.

    Args:
        phase: "remix" (phase 1 only), "tone" (phase 2 only), or "all".
    """
    if phase in ("remix", "all"):
        click.echo("=== Phase 1: Remix Calibration (Grid Sweep) ===\n")
        from subtitle_generator.calibrate import run_calibration

        run_calibration(conn, samples=samples, model=rater_model)

    if phase in ("tone", "all"):
        click.echo("\n=== Phase 2: Tone Tuning (Autoresearch Loop) ===\n")
        run_tone_tuning(
            conn,
            iterations=iterations,
            rater_model=rater_model,
            proposer_model=proposer_model,
            results_file=results_file,
            dry_run=dry_run,
        )
