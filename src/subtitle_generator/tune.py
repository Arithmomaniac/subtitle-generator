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
from subtitle_generator.feedback import format_summary_for_proposer, get_summary, store_rating

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_goals() -> str:
    """Read tuning_goals.md from repo root."""
    goals_path = pathlib.Path(__file__).parent.parent.parent / "tuning_goals.md"
    if goals_path.exists():
        return goals_path.read_text()
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
    """Load recent results for the proposer's context."""
    path = pathlib.Path(results_file)
    if not path.exists():
        return "(no previous experiments)"
    lines = path.read_text().strip().split("\n")
    if len(lines) <= max_lines + 1:
        return "\n".join(lines)
    return "\n".join([lines[0]] + lines[-max_lines:])


def _ensure_results_header(results_file: str) -> None:
    """Create results TSV with header if it doesn't exist."""
    path = pathlib.Path(results_file)
    if not path.exists():
        path.write_text(
            "iteration\tparam\told_value\tnew_value\t"
            "quality\tseparation\tcomposite\tstatus\tdescription\n",
            encoding="utf-8",
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
    with open(results_file, "a", encoding="utf-8") as f:
        f.write(
            f"{iteration}\t{param}\t{old_value}\t{new_value}\t"
            f"{quality:.4f}\t{separation:.4f}\t{comp:.4f}\t"
            f"{status}\t{description}\n"
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
# Spot-check helpers
# ---------------------------------------------------------------------------


def _is_spot_check_iteration(i: int) -> bool:
    """Exponential backoff: check at iterations 1, 2, 4, 8, 16, 32…"""
    return i > 0 and (i & (i - 1)) == 0  # power of 2


def _run_spot_check(
    conn: sqlite3.Connection,
    iteration: int,
    total_iterations: int,
    n_samples: int = 2,
    seed_base: int = 1000,
    use_tui: bool = False,
) -> float | None:
    """Show tone-targeted sample subtitles and collect human ratings.

    Generates n_samples per tier (pop, mainstream, niche) and asks the human
    which tier each subtitle *feels* like. Stores both system_tone (target)
    and tone_override (human perception) for mismatch analysis.

    When use_tui=True, uses questionary for a faster grid-style interface.

    Returns human tone-accuracy (0.0-1.0) or None if all skipped.
    """
    from subtitle_generator.config import get_tone_targets

    next_check = iteration * 2
    if next_check > total_iterations:
        next_check_str = "no more checks"
    else:
        next_check_str = f"next at iter {next_check}"
    click.echo(click.style(
        f"\n  ⏸ Spot check (iter {iteration} — {next_check_str}):",
        fg="green", bold=True,
    ))

    targets = get_tone_targets(conn)
    tiers = ["pop", "mainstream", "niche"]
    tier_labels = {"pop": "🔥 POP", "mainstream": "📚 MAINSTREAM", "niche": "🎓 NICHE"}
    tier_shortcuts = {"p": "pop", "m": "mainstream", "n": "niche"}

    # Generate targeted samples per tier
    all_samples: list[tuple[str, str, object]] = []  # (tier, text, subtitle_obj)
    for tier in tiers:
        tone_target = {
            slot: targets[tier][slot]
            for slot in ["list_item", "action_noun", "of_object"]
        }
        for j in range(n_samples):
            from subtitle_generator.generate import generate_subtitle
            sub = generate_subtitle(
                conn,
                seed=seed_base + 500 + tiers.index(tier) * 100 + j,
                tone_target=tone_target,
            )
            all_samples.append((tier, sub.text, sub))

    if use_tui:
        return _spot_check_tui(conn, all_samples, tier_labels, tier_shortcuts)
    else:
        return _spot_check_cli(conn, all_samples, tier_labels, tier_shortcuts)


def _spot_check_cli(
    conn: sqlite3.Connection,
    samples: list[tuple[str, str, object]],
    tier_labels: dict[str, str],
    tier_shortcuts: dict[str, str],
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
                click.echo(click.style("       ✓ match", fg="green"))
            else:
                click.echo(click.style(
                    f"       ✗ mismatch (target={target_tier}, felt={perceived})",
                    fg="yellow",
                ))
            # Store with system_tone=target, tone_override=perceived
            store_rating(
                conn, text,
                system_tone=target_tier,
                thumbs=1 if perceived == target_tier else -1,
                tone_override=perceived,
            )

            # Quick quality tag
            tags_input = click.prompt(
                click.style("       Tags? [f/g/c/b / Enter]", fg="cyan"),
                default="", show_default=False,
            ).strip().lower()
            if tags_input:
                tag_map = {"f": "funny", "g": "grammar", "c": "contradiction", "b": "boring"}
                tags = [tag_map[c] for c in tags_input if c in tag_map] or None
                if tags:
                    store_rating(conn, text, system_tone=target_tier, tags=tags)

    if total == 0:
        return None
    accuracy = correct / total
    click.echo(click.style(
        f"  Tone accuracy: {correct}/{total} ({accuracy:.0%})",
        fg="green" if accuracy >= 0.6 else "yellow",
    ))
    return accuracy


def _spot_check_tui(
    conn: sqlite3.Connection,
    samples: list[tuple[str, str, object]],
    tier_labels: dict[str, str],
    tier_shortcuts: dict[str, str],
) -> float | None:
    """TUI spot-check using questionary for grid-style rating."""
    try:
        import questionary
        from questionary import Choice
    except ImportError:
        click.echo("  questionary not installed, falling back to CLI mode")
        click.echo("  Install with: uv pip install questionary")
        return _spot_check_cli(conn, samples, tier_labels, tier_shortcuts)

    import random as _rng
    shuffled = list(samples)
    _rng.shuffle(shuffled)

    total = 0
    correct = 0

    # Present all subtitles in a grid
    click.echo()
    for i, (target_tier, text, _) in enumerate(shuffled):
        click.echo(f"  {i+1}. [{tier_labels[target_tier]}] {text}")
    click.echo()

    # Rate each one
    for i, (target_tier, text, sub) in enumerate(shuffled):
        result = questionary.select(
            f"  {i+1}) \"{text[:60]}…\" — feels like?",
            choices=[
                Choice("🔥 Pop", "pop"),
                Choice("📚 Mainstream", "mainstream"),
                Choice("🎓 Niche", "niche"),
                Choice("⏭ Skip", "skip"),
            ],
            use_shortcuts=True,
            use_arrow_keys=True,
        ).ask()

        if result and result != "skip":
            total += 1
            match = result == target_tier
            if match:
                correct += 1
                click.echo(click.style("       ✓ match", fg="green"))
            else:
                click.echo(click.style(
                    f"       ✗ mismatch (target={target_tier}, felt={result})",
                    fg="yellow",
                ))
            store_rating(
                conn, text,
                system_tone=target_tier,
                thumbs=1 if match else -1,
                tone_override=result,
            )

            # Quality tags (same as CLI mode)
            tag_choices = questionary.checkbox(
                "       Tags?",
                choices=[
                    Choice("😄 Funny", "funny"),
                    Choice("📝 Grammar", "grammar"),
                    Choice("🤔 Contradiction", "contradiction"),
                    Choice("😴 Boring", "boring"),
                ],
            ).ask()
            if tag_choices:
                store_rating(conn, text, system_tone=target_tier, tags=tag_choices)

    if total == 0:
        return None
    accuracy = correct / total
    click.echo(click.style(
        f"\n  Tone accuracy: {correct}/{total} ({accuracy:.0%})",
        fg="green" if accuracy >= 0.6 else "yellow",
    ))
    return accuracy


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
    spot_check: bool = False,
    spot_check_tui: bool = False,
) -> dict:
    """Autoresearch loop for tone parameters.

    Each iteration: propose a single parameter change via LLM, evaluate,
    keep if improved, revert otherwise.

    Returns the final parameter dict.
    """
    _ensure_results_header(results_file)
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
        # Include human feedback summary if enough ratings exist
        human_feedback_section = ""
        feedback_summary = get_summary(conn)
        if feedback_summary:
            human_feedback_section = "\n" + format_summary_for_proposer(feedback_summary) + "\n"

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
{human_feedback_section}
Propose ONE parameter change that you think will improve the composite score.
Focus on parameters with the biggest potential impact (bias_floor and spread have historically had the largest effect).
Consider what previous experiments tell you about which direction to move.
{f"Also consider the human feedback above — tone mismatches suggest the accessibility thresholds or tier centers may need adjustment." if human_feedback_section else ""}
"""

        click.echo("  proposing parameter change …")
        try:
            proposal = structured_completion(
                model=proposer_model,
                messages=[{"role": "user", "content": proposal_prompt}],
                schema=ParamProposal,
                timeout=180.0,
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
                    f"  ⚠ clamping {proposal.param} "
                    f"{new_value} → {clamped} (bounds [{lo}, {hi}])"
                )
                new_value = clamped

        click.echo(
            f"  proposal: {proposal.param} {old_value} → {new_value}"
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

        # Evaluate with new value
        new_quality, new_separation, new_score = _evaluate(
            conn, rater_model, seed_base=1000 + i * 100,
        )

        # Spot-check: exponential backoff (check at 1, 2, 4, 8, 16…)
        human_approval = None
        if (spot_check or spot_check_tui) and _is_spot_check_iteration(i):
            human_approval = _run_spot_check(
                conn, i, iterations, seed_base=1000 + i * 100,
                use_tui=spot_check_tui,
            )
            if human_approval is not None:
                # Blend: 40% quality + 40% separation + 20% human
                new_score = 0.4 * new_quality + 0.4 * new_separation + 0.2 * human_approval

        delta = new_score - current_score

        if new_score > current_score:
            status = "keep"
            click.echo(
                f"  Quality: {quality:.3f} → {new_quality:.3f}  "
                f"Separation: {separation:.3f} → {new_separation:.3f}  "
                f"Composite: {current_score:.3f} → {new_score:.3f}"
            )
            click.echo(f"  → KEEP (+{delta:.3f})\n")
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
            click.echo(
                f"  Quality: {quality:.3f} → {new_quality:.3f}  "
                f"Separation: {separation:.3f} → {new_separation:.3f}  "
                f"Composite: {current_score:.3f} → {new_score:.3f}"
            )
            click.echo(f"  → DISCARD ({delta:+.3f})\n")

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
    spot_check: bool = False,
    spot_check_tui: bool = False,
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
            spot_check=spot_check,
            spot_check_tui=spot_check_tui,
        )
