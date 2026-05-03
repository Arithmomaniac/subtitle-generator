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
from subtitle_generator.generate import TierFilterError
from subtitle_generator.tier_diagnostics import score_real_title_tier_pop_guardrail
from subtitle_generator.tuning_state import (
    ConfigChange,
    apply_config_change,
    record_decision,
    revert_config_change,
)

SOURCE_TIER_LABEL_WEIGHT = 0.25
_NON_PARAM_RESULT_MARKERS = frozenset({
    "(failed)",
    "[auto-revert]",
    "[regime change]",
})

AUTOTUNE_PARAM_KEYS = frozenset({
    "article_of_min_freq",
    "article_action_min_freq",
    "article_remix_heuristic_threshold",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _autotune_param_keys() -> set[str]:
    """Return config keys that the autoresearch loop is allowed to propose."""

    return set(AUTOTUNE_PARAM_KEYS)


def _autotune_param_values(params: dict[str, float]) -> dict[str, float]:
    return {
        key: params[key]
        for key in sorted(_autotune_param_keys())
        if key in params
    }


def _proposal_change(
    proposal: ParamProposal,
    current_params: dict[str, float],
    bounds: dict[str, tuple[float, float]],
) -> tuple[ConfigChange, float | None]:
    """Create a rollback-capable config change, applying bounds if present."""

    new_value = proposal.new_value
    clamped_from = None
    if proposal.param in bounds:
        lo, hi = bounds[proposal.param]
        clamped = max(lo, min(hi, new_value))
        if clamped != new_value:
            clamped_from = new_value
            new_value = clamped
    return ConfigChange(
        param=proposal.param,
        old_value=current_params[proposal.param],
        new_value=new_value,
    ), clamped_from


def _load_goals() -> str:
    """Read tuning_goals.md from repo root."""
    goals_path = pathlib.Path(__file__).parent.parent.parent / "tuning_goals.md"
    if goals_path.exists():
        return goals_path.read_text(encoding="utf-8")
    return "(no tuning_goals.md found)"


def _parse_bounds(goals_text: str) -> dict[str, tuple[float, float]]:
    """Extract parameter bounds from the tuning_goals.md table.

    Matches rows like:
      | `weighted_sample_spread` | 0.1 | 1.0 | 0.12 | ... |
    Also handles wildcard rows for parameters that remain in the autotune surface.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for match in re.finditer(
        r"\|\s*`([^`]+)`\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", goals_text
    ):
        pattern, lo, hi = match.group(1), float(match.group(2)), float(match.group(3))
        if "*" in pattern:
            prefix = pattern.replace("*", "")
            for key in _autotune_param_keys():
                if key.startswith(prefix):
                    bounds[key] = (lo, hi)
        elif pattern in _autotune_param_keys():
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
    regime_lines = [line for line in lines[1:-max_lines] if "[regime change]" in line]
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


def _summarize_regime_state(results_file: str) -> dict:
    """Scan the current regime in results.tsv and return:
      - best:        dict with composite/quality/separation/iter/param_changed/to_value, or None
      - explored:    sorted list of params probed since last regime marker
      - unexplored:  sorted list of params in ALL_TUNABLE_PARAMS not yet probed this regime
                     (excludes auto-calibrated params that the tuner can't usefully set)
      - param_freq:  dict of param -> count of times probed this regime
    """
    available = _autotune_param_keys()
    path = pathlib.Path(results_file)
    if not path.exists():
        return {
            "best": None, "explored": [], "unexplored": sorted(available),
            "param_freq": {},
        }
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    if len(lines) <= 1:
        return {
            "best": None, "explored": [], "unexplored": sorted(available),
            "param_freq": {},
        }
    rows = lines[1:]
    # Find rows after the most recent regime marker
    last_regime_idx = -1
    for i, line in enumerate(rows):
        if "[regime change]" in line:
            last_regime_idx = i
    current_regime_rows = rows[last_regime_idx + 1:]

    explored: set[str] = set()
    param_freq: dict[str, int] = {}
    best: dict | None = None
    for line in current_regime_rows:
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        try:
            iter_num = int(parts[0])
        except ValueError:
            continue
        param = parts[1]
        if param in _NON_PARAM_RESULT_MARKERS:
            continue
        explored.add(param)
        param_freq[param] = param_freq.get(param, 0) + 1
        try:
            comp = float(parts[6])
        except ValueError:
            continue
        if best is None or comp > best["composite"]:
            best = {
                "iteration": iter_num,
                "param_changed": param,
                "to_value": parts[3],
                "quality": float(parts[4]),
                "separation": float(parts[5]),
                "composite": comp,
                "status": parts[7],
            }
    unexplored = sorted(available - explored)
    return {
        "best": best,
        "explored": sorted(explored),
        "unexplored": unexplored,
        "param_freq": param_freq,
    }


def _best_snapshot_path(results_file: str) -> pathlib.Path:
    return pathlib.Path(results_file).parent / "tune_best_state.json"


def _read_best_snapshot(results_file: str) -> dict | None:
    p = _best_snapshot_path(results_file)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_best_snapshot(
    results_file: str, composite: float, quality: float, separation: float,
    params: dict, iteration: int,
) -> None:
    p = _best_snapshot_path(results_file)
    p.write_text(
        json.dumps({
            "composite": composite,
            "quality": quality,
            "separation": separation,
            "params": params,
            "iteration": iteration,
        }, indent=2),
        encoding="utf-8",
    )


def _clear_best_snapshot(results_file: str) -> bool:
    """Called on regime change — old best may be invalid in new regime."""
    p = _best_snapshot_path(results_file)
    if p.exists():
        p.unlink()
        return True
    return False


def _clear_best_snapshot_for_regime_change(results_file: str) -> None:
    if _clear_best_snapshot(results_file):
        click.echo(
            "  [regime change] cleared tune_best_state.json because the "
            "autotune parameter surface changed."
        )


def _restore_params_from_snapshot(
    conn: sqlite3.Connection, snapshot_params: dict,
) -> None:
    """Write snapshot params back to config table."""
    current = load_tuning_config(conn)
    for key, value in snapshot_params.items():
        if key not in _autotune_param_keys():
            continue  # param removed from autoresearch or owned by another workflow
        cur = current.get(key)
        if cur == value or str(cur) == str(value):
            continue
        if value == ALL_TUNABLE_PARAMS[key]:
            conn.execute("DELETE FROM config WHERE key = ?", (key,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
                (key, str(value)),
            )
    conn.commit()
    invalidate_config_cache()


def _check_regime_change(results_file: str) -> None:
    """Insert a regime-change marker if available params changed since last run.

    Scans the TSV for the most recent regime marker (or all experiment rows if none)
    to determine which params were available. If ALL_TUNABLE_PARAMS has new keys,
    appends a marker row so the proposer knows old history is from a different regime.
    """
    path = pathlib.Path(results_file)
    if not path.exists():
        return

    current_params = sorted(_autotune_param_keys())
    lines = path.read_text(encoding="utf-8").strip().split("\n")

    # Find the most recent regime marker
    last_regime_params = None
    for line in reversed(lines):
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == "[regime change]":
            # Extract param list from description
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
            if len(parts) >= 2 and parts[1] not in _NON_PARAM_RESULT_MARKERS:
                mentioned.add(parts[1])
        if mentioned and mentioned != set(current_params):
            new_params = sorted(set(current_params) - mentioned)
            removed_params = sorted(mentioned - set(current_params))
            desc_parts = []
            if new_params:
                desc_parts.append(f"New params added: {', '.join(new_params)}.")
            if removed_params:
                desc_parts.append(f"Params removed: {', '.join(removed_params)}.")
            desc_parts.append(
                "History above is from a prior regime with a different "
                "parameter surface."
            )
            desc_parts.append(f"available_params={','.join(current_params)}")
            _append_result(
                results_file, 0, "[regime change]", 0, 0, 0, 0, 0,
                "regime",
                " ".join(desc_parts),
            )
            _clear_best_snapshot_for_regime_change(results_file)
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
        _clear_best_snapshot_for_regime_change(results_file)


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

    label_guardrail, label_count = score_real_title_tier_pop_guardrail(conn)
    output_comp = composite_score(quality, separation, quality_weight=quality_weight)
    comp = _blend_real_title_tier_guardrail(
        output_comp, label_guardrail, label_count, SOURCE_TIER_LABEL_WEIGHT
    )
    if label_count:
        click.echo(
            f"  source-title tier guardrail: {label_guardrail:.3f} "
            f"(source_label_weight={SOURCE_TIER_LABEL_WEIGHT:.2f}, labels={label_count})"
        )
    else:
        click.echo(
            "  source-title tier guardrail: no labels "
            f"(source_label_weight={SOURCE_TIER_LABEL_WEIGHT:.2f}, inactive)"
        )
    return quality, separation, comp


def _blend_real_title_tier_guardrail(
    output_comp: float,
    label_guardrail: float,
    label_count: int,
    label_weight: float,
) -> float:
    if label_count <= 0 or label_weight <= 0:
        return output_comp
    return (1.0 - label_weight) * output_comp + label_weight * label_guardrail


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
    from subtitle_generator.generate import generate_subtitles

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
    for tier_index, tier in enumerate(tiers):
        subtitles = generate_subtitles(
            conn,
            n=n_samples,
            seed_base=seed_base + tier_index * 100,
            tone_target=targets[tier],
        )
        for sub in subtitles:
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
                click.style("       Tags? [f=funny/b=boring/r=broken/n=nonsense/l=realistic/i=interesting / Enter]", fg="cyan"),
                default="", show_default=False,
            ).strip().lower()
            tag_map = {
                "f": "funny",
                "b": "boring",
                "r": "broken",
                "n": "nonsense",
                "l": "realistic",
                "i": "interesting",
            }
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
    for (system_tone, felt), count in mismatch_directions.most_common():
        summary_lines.append(f"  target={system_tone} -> felt={felt}: {count}x")
    summary_lines.append("")
    summary_lines.append("### Mismatch examples:")
    for system_tone, felt, sub in mismatch_examples:
        summary_lines.append(f"  [{system_tone}->{felt}] {sub}")
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
    try:
        quality, separation, current_score = _evaluate(conn, rater_model)
    except TierFilterError as exc:
        raise click.ClickException(
            f"Baseline tier-filter evaluation failed: {exc}"
        ) from exc
    click.echo(
        f"Baseline — Quality: {quality:.3f}  "
        f"Separation: {separation:.3f}  "
        f"Composite: {current_score:.3f}\n"
    )

    # Seed best-snapshot from baseline if no snapshot exists yet
    if _read_best_snapshot(results_file) is None:
        _write_best_snapshot(
            results_file, current_score, quality, separation,
            _autotune_param_values(current_params), iteration=0,
        )

    # Threshold for considering current "drifted below best" (1σ-ish on composite)
    DRIFT_THRESHOLD = 0.03

    for i in range(1, iterations + 1):
        click.echo(f"--- Iteration {i}/{iterations} ---")

        # Reload state each iteration
        current_params = load_tuning_config(conn)
        proposer_params = _autotune_param_values(current_params)
        results_history = _load_results_history(results_file)
        regime_state = _summarize_regime_state(results_file)
        snapshot = _read_best_snapshot(results_file)

        # --- Auto-revert if we've drifted below best ---
        if (
            snapshot is not None
            and snapshot["composite"] > current_score + DRIFT_THRESHOLD
        ):
            click.echo(
                f"  [auto-revert] current composite {current_score:.3f} is "
                f"{snapshot['composite'] - current_score:.3f} below best "
                f"{snapshot['composite']:.3f} (iter {snapshot['iteration']}). "
                f"Restoring best-known params and re-evaluating."
            )
            _restore_params_from_snapshot(conn, snapshot["params"])
            try:
                quality, separation, current_score = _evaluate(
                    conn, rater_model, seed_base=2000 + i * 100,
                )
            except TierFilterError as exc:
                raise click.ClickException(
                    f"Tier-filter evaluation failed after auto-revert: {exc}"
                ) from exc
            click.echo(
                f"  After revert — Quality: {quality:.3f}  "
                f"Separation: {separation:.3f}  Composite: {current_score:.3f}\n"
            )
            _append_result(
                results_file, i, "[auto-revert]", 0, 0,
                quality, separation, current_score,
                "auto_revert",
                f"Drifted {snapshot['composite'] - current_score:+.3f} below best "
                f"{snapshot['composite']:.3f} from iter {snapshot['iteration']}; "
                f"restored snapshot params.",
            )
            # Update snapshot if revert eval produces a new best (it shouldn't,
            # but rater noise can surprise)
            if current_score > snapshot["composite"]:
                _write_best_snapshot(
                    results_file, current_score, quality, separation,
                    _autotune_param_values(load_tuning_config(conn)), iteration=i,
                )
            continue  # consume an iteration on the revert

        # Build extra context for the proposer about regime exploration state
        best_section = ""
        if regime_state["best"] is not None:
            b = regime_state["best"]
            best_section = (
                f"- Best composite this regime: {b['composite']:.4f} at iter {b['iteration']} "
                f"(after `{b['param_changed']}` -> {b['to_value']}, status={b['status']}).\n"
                f"- Current composite: {current_score:.4f}"
            )
            delta_to_best = b["composite"] - current_score
            if delta_to_best > DRIFT_THRESHOLD:
                best_section += (
                    f" — **{delta_to_best:.3f} BELOW best**. Strongly consider "
                    f"reverting recent changes that moved you away from the best config."
                )
            elif current_score > b["composite"]:
                best_section += " — **NEW BEST so far this regime.**"
            else:
                best_section += " (within noise of best)."
        else:
            best_section = f"- Current composite: {current_score:.4f} (no regime history yet)."

        # Repeat-probe pressure: discourage hitting the same param twice in a row
        repeat_warning = ""
        recent_params = [
            row.split("\t")[1] for row in pathlib.Path(results_file)
            .read_text(encoding="utf-8").strip().split("\n")[1:]
            if row.split("\t")[1] not in ("(failed)", "[regime change]", "[auto-revert]")
        ][-5:]
        overused = [p for p, c in regime_state["param_freq"].items() if c >= 3]
        if overused:
            repeat_warning = (
                f"\n- **Avoid these params** (probed {[regime_state['param_freq'][p] for p in overused]} "
                f"times this regime, signal exhausted): {overused}"
            )
        if recent_params:
            repeat_warning += (
                f"\n- Last 5 iterations probed: {recent_params}. "
                f"Probing the same param again is unlikely to add signal."
            )

        unexplored_text = (
            ", ".join(regime_state["unexplored"])
            if regime_state["unexplored"] else "(all params probed at least once)"
        )

        regime_context = f"""
## Regime exploration state:
{best_section}
- Params NOT YET probed this regime (HIGH PRIORITY for new probes): {unexplored_text}{repeat_warning}
"""

        # Propose a parameter change
        proposal_prompt = f"""You are tuning parameters for a subtitle generator.

## Current parameter values:
{json.dumps(proposer_params, indent=2)}

## Tuning goals:
{goals_text}

## Current scores:
- Quality: {quality:.3f}
- Tone separation: {separation:.3f}
- Composite: {current_score:.3f}
{regime_context}
## Previous experiments:
{results_history}

## Parameter bounds:
{bounds_text}

Propose ONE parameter change that you think will improve the composite score.
Strongly prefer params from the "NOT YET probed" list above — they are the highest-leverage
remaining moves. Avoid re-probing params already tested several times this regime; their
signal is exhausted and further probes are noise.

Consider what previous experiments tell you about which direction to move.

NOTE: Popularity parameters such as pop_*, tier_pop_min_*, accessibility
thresholds, and tier centers are intentionally excluded from this autoresearch
loop. They should be fitted from source-title labels and table refits, not
inferred indirectly from generated-output ratings.
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
        if proposal.param not in _autotune_param_keys():
            click.echo(
                f"  ⚠ proposed unsupported param '{proposal.param}' — skipping"
            )
            _append_result(
                results_file, i, proposal.param, 0, proposal.new_value,
                quality, separation, current_score,
                "skip", f"unknown param: {proposal.reasoning}",
            )
            continue

        change, clamped_from = _proposal_change(proposal, current_params, bounds)
        old_value = change.old_value
        new_value = change.new_value

        if clamped_from is not None:
            lo, hi = bounds[proposal.param]
            click.echo(
                f"  ! clamping {proposal.param} "
                f"{clamped_from} -> {new_value} (bounds [{lo}, {hi}])"
            )

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

        apply_config_change(conn, change)

        # Evaluate with new value
        try:
            new_quality, new_separation, new_score = _evaluate(
                conn, rater_model, seed_base=1000 + i * 100,
            )
        except TierFilterError as exc:
            click.echo(f"  -> SKIP (tier filter unreachable: {exc})\n")
            revert_config_change(conn, change)
            _append_result(
                results_file, i, proposal.param, old_value, new_value,
                quality, separation, current_score,
                "error", f"tier filter unreachable: {proposal.reasoning}; {exc}",
            )
            continue

        delta = new_score - current_score
        before_scores = (quality, separation, current_score)
        after_scores = (new_quality, new_separation, new_score)

        if new_score > current_score:
            status = "keep"
            click.echo(
                f"  Quality: {quality:.3f} -> {new_quality:.3f}  "
                f"Separation: {separation:.3f} -> {new_separation:.3f}  "
                f"Composite: {current_score:.3f} -> {new_score:.3f}"
            )
            click.echo(f"  -> KEEP (+{delta:.3f})\n")
            quality, separation, current_score = (
                new_quality, new_separation, new_score,
            )
            # Update best snapshot if this is a new high
            best_snap = _read_best_snapshot(results_file)
            if best_snap is None or current_score > best_snap["composite"]:
                _write_best_snapshot(
                    results_file, current_score, quality, separation,
                    _autotune_param_values(load_tuning_config(conn)), iteration=i,
                )
                click.echo(f"  [snapshot] new best composite {current_score:.4f} saved")
        else:
            status = "discard"
            revert_config_change(conn, change)
            click.echo(
                f"  Quality: {quality:.3f} -> {new_quality:.3f}  "
                f"Separation: {separation:.3f} -> {new_separation:.3f}  "
                f"Composite: {current_score:.3f} -> {new_score:.3f}"
            )
            click.echo(f"  -> DISCARD ({delta:+.3f})\n")

        decision = record_decision(
            change=change,
            reasoning=proposal.reasoning,
            status=status,
            before=before_scores,
            after=after_scores,
        )
        _append_result(
            results_file, i, decision.change.param,
            decision.change.old_value, decision.change.new_value,
            decision.quality_after, decision.separation_after,
            decision.composite_after, decision.status, decision.reasoning,
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
