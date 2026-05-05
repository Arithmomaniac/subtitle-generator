"""CLI entry point for subtitle-generator."""

import sqlite3
import sys
from pathlib import Path

import click

from subtitle_generator import __version__
from subtitle_generator.analyze import analyze_subtitles, build_pattern_index
from subtitle_generator.download import TOTAL_PARTS, download_part, parse_parts_arg
from subtitle_generator.extract import DATA_DIR, DB_PATH, extract_from_file, get_db
from subtitle_generator.extract_openlibrary import (
    download_ol_dump,
    ensure_isbn_column,
    extract_from_ol_dump,
)
from subtitle_generator.export_db import build_mini_db, export_data, export_mini_db
from subtitle_generator.generate import (
    TierFilterError,
    format_sources,
    generate_subtitle_matching_tiers,
    precompute_remix_data,
    slot_stats,
)
from subtitle_generator.jacket import (
    TONE_HIGH, TONE_LOW, TONE_MEDIUM,
    generate_jacket,
)
from subtitle_generator.pipeline_validation import format_validation_report, validate_pipeline
from subtitle_generator.slots import build_slots, ensure_slot_tables
from subtitle_generator.tiering import compute_tier_evidence


_TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
_VALID_TONES = set(_TONE_CHOICES.keys())
_TONE_OVERRIDE_MAP = {"p": "pop", "m": "mainstream", "n": "niche"}


def _get_system_tone(subtitle: str, conn) -> tuple[str, float]:
    """Compute system tone tier and score for a subtitle."""
    evidence = compute_tier_evidence(subtitle, conn)
    return evidence.tier, evidence.accessibility_score


_TAG_MAP = {
    "f": "funny",
    "b": "boring",
    "r": "broken",
    "n": "nonsense",
    "l": "realistic",
    "i": "interesting",
}


def _prompt_review(conn, subtitle_text: str) -> int | None:
    """Interactive review prompt. Returns 1 (thumbs up), -1 (down), or None (skipped)."""
    from subtitle_generator.feedback import store_rating

    # Thumbs
    thumbs_input = click.prompt(
        click.style("     Good subtitle? [y/n/Enter to skip]", fg="green"),
        default="", show_default=False,
    ).strip().lower()

    if not thumbs_input:
        return None

    thumbs = 1 if thumbs_input in ("y", "yes") else -1

    # Tone override
    tone_input = click.prompt(
        click.style("     Tone? [p=pop / m=mainstream / n=niche / Enter to skip]", fg="blue"),
        default="", show_default=False,
    ).strip().lower()
    tone_override = _TONE_OVERRIDE_MAP.get(tone_input)

    # Comment
    comment = click.prompt(
        click.style("     Comment [Enter to skip]", fg="magenta"),
        default="", show_default=False,
    ).strip() or None

    # Tags
    tags_input = click.prompt(
        click.style("     Tags? [f=funny / b=boring / r=broken / n=nonsense / l=realistic / i=interesting / Enter]", fg="cyan"),
        default="", show_default=False,
    ).strip().lower()
    tags = [_TAG_MAP[c] for c in tags_input if c in _TAG_MAP] or None

    # Store
    system_tone, score = _get_system_tone(subtitle_text, conn)
    store_rating(
        conn,
        subtitle_text,
        system_tone=system_tone,
        thumbs=thumbs,
        tone_override=tone_override,
        free_text=comment,
        tags=tags,
    )

    # Feedback line
    if tone_override:
        match_sym = "✓" if tone_override == system_tone else "✗"
        click.echo(f"     ✓ saved (system: {system_tone}, you: {tone_override} {match_sym}, score: {score:.2f})")
    else:
        click.echo(f"     ✓ saved (system: {system_tone}, score: {score:.2f})")

    return thumbs


def _parse_tone(tone_str: str | None) -> set[str] | None:
    """Parse a comma-separated tone string into a set of valid tier names."""
    if not tone_str:
        return None
    tones = {t.strip().lower() for t in tone_str.split(",")}
    invalid = tones - _VALID_TONES
    if invalid:
        raise click.BadParameter(f"Invalid tone(s): {', '.join(invalid)}. Choose from: pop, mainstream, niche")
    return tones


@click.group()
def cli():
    """Generate bizarre book subtitles from LOC MARC data.

    \b
    Quick start:
      subtitle-gen download --parts 1-5   # grab a few MARC files
      subtitle-gen extract                 # parse into SQLite
      subtitle-gen build-slots             # extract slot fillers
      subtitle-gen generate                # slot-machine time

    \b
    Popularity scoring:
      subtitle-gen download-popularity     # download all popularity sources
      subtitle-gen populate-popularity     # compute composite scores
    """
    pass


@cli.command()
def version():
    """Show version."""
    click.echo(f"subtitle-generator {__version__}")


@cli.command()
@click.option(
    "--parts",
    default="1",
    help=f"Which parts to download: '1', '1-5', '1,3,7', or 'all' (1-{TOTAL_PARTS}).",
)
@click.option("--force", is_flag=True, help="Re-download even if files exist.")
@click.option(
    "--keep-gz", is_flag=True, help="Keep .gz files instead of decompressing."
)
def download(parts: str, force: bool, keep_gz: bool):
    """Download LOC MARC bulk data files (Books All, 2016 retrospective).

    \b
    Examples:
      subtitle-gen download --parts 1        # single file (~200 MB)
      subtitle-gen download --parts 1-5      # range
      subtitle-gen download --parts all      # all 43 files (~9 GB)
      subtitle-gen download --parts 1 --force  # re-download
    """
    part_nums = parse_parts_arg(parts)
    click.echo(f"Downloading {len(part_nums)} part(s): {part_nums}")
    for p in part_nums:
        download_part(p, decompress=not keep_gz, force=force)
    click.echo("Done!")


@cli.command()
@click.option(
    "--parts",
    default=None,
    help="Which parts to extract: '1', '1-5', or 'all'. Default: all downloaded.",
)
@click.option("--all-langs", is_flag=True, help="Include non-English subtitles.")
def extract(parts: str | None, all_langs: bool):
    """Extract subtitles from downloaded MARC files into SQLite.

    \b
    Examples:
      subtitle-gen extract               # all downloaded .mrc files
      subtitle-gen extract --parts 1-5   # specific parts only
      subtitle-gen extract --all-langs   # include non-English
    """
    raw_dir = DATA_DIR / "raw"  # DATA_DIR = .../data
    if parts:
        part_nums = parse_parts_arg(parts)
        mrc_files = [raw_dir / f"BooksAll.2016.part{p:02d}.utf8.mrc" for p in part_nums]
        mrc_files = [f for f in mrc_files if f.exists()]
    else:
        mrc_files = sorted(raw_dir.glob("*.mrc"))

    if not mrc_files:
        raise click.ClickException("No .mrc files found. Run 'subtitle-gen download' first.")

    conn = get_db()
    total_records = 0
    total_subtitles = 0

    for mrc_file in mrc_files:
        click.echo(f"Extracting from {mrc_file.name}...")
        records, subs = extract_from_file(mrc_file, conn, english_only=not all_langs)
        total_records += records
        total_subtitles += subs
        click.echo(f"  {mrc_file.name}: {records:,} records -> {subs:,} subtitles")

    click.echo(f"\nTotal: {total_records:,} records -> {total_subtitles:,} subtitles")
    click.echo(f"Database: {DB_PATH}")
    conn.close()


@cli.command("download-ol")
@click.option("--force", is_flag=True, help="Re-download even if file exists.")
def download_ol(force: bool):
    """Download Open Library editions dump (~9.2 GB compressed).

    \b
    Examples:
      subtitle-gen download-ol           # download (~9.2 GB)
      subtitle-gen download-ol --force   # re-download
    """
    download_ol_dump(force=force)


@cli.command("extract-ol")
@click.option("--all-langs", is_flag=True, help="Include non-English subtitles.")
@click.option("--no-dedup", is_flag=True, help="Skip deduplication (faster for testing).")
def extract_ol(all_langs: bool, no_dedup: bool):
    """Extract subtitles from Open Library editions dump into SQLite.

    \b
    Examples:
      subtitle-gen extract-ol             # extract + deduplicate vs LOC
      subtitle-gen extract-ol --no-dedup  # skip dedup (faster)
    """
    conn = get_db()
    ensure_isbn_column(conn)
    lines, subs, dupes = extract_from_ol_dump(
        conn, english_only=not all_langs, dedup=not no_dedup,
    )
    click.echo(f"\nDone: {lines:,} lines -> {subs:,} subtitles ({dupes:,} duplicates skipped)")
    total = conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()[0]
    click.echo(f"Total subtitles in database: {total:,}")
    click.echo(f"Database: {DB_PATH}")
    conn.close()


@cli.command()
@click.option("--limit", default=None, type=int, help="Max subtitles to analyze.")
def analyze(limit: int | None):
    """POS-tag subtitles and extract structural templates.

    \b
    Examples:
      subtitle-gen analyze              # analyze all subtitles
      subtitle-gen analyze --limit 1000 # quick test run
    """
    conn = get_db()
    analyze_subtitles(conn, limit=limit)
    build_pattern_index(conn)
    conn.close()


@cli.command()
@click.option("--top", default=50, type=click.IntRange(min=1), help="Show top N patterns.")
@click.option("--min-count", default=10, type=click.IntRange(min=1), help="Minimum occurrence count.")
def patterns(top: int, min_count: int):
    """Show discovered subtitle patterns ranked by frequency.

    \b
    Examples:
      subtitle-gen patterns                  # top 50, min 10 occurrences
      subtitle-gen patterns --top 10         # just the top 10
      subtitle-gen patterns --min-count 100  # only common patterns
    """
    conn = get_db()
    rows = conn.execute(
        "SELECT template, count, example_subtitle FROM patterns "
        "WHERE count >= ? ORDER BY count DESC LIMIT ?",
        (min_count, top),
    ).fetchall()
    if not rows:
        raise click.ClickException("No patterns found. Run 'subtitle-gen analyze' first.")
    click.echo(f"Top {len(rows)} patterns (min count: {min_count}):\n")
    for i, (template, count, example) in enumerate(rows, 1):
        click.echo(f"{i:3d}. [{count:,}x] {template}")
        click.echo(f"     e.g. \"{example}\"")
        click.echo()
    conn.close()


@cli.command("build-slots")
@click.option("--skip-vectors", is_flag=True, help="Skip vector precomputation (useful if en_core_web_md is not installed).")
def build_slots_cmd(skip_vectors: bool):
    """Extract slot fillers from matched subtitles (regex + NLP validated).

    Runs regex pattern matching, spaCy POS/NER validation, of-object
    decomposition, and vector precomputation. Rebuilds the entire
    slot_fillers table from scratch.

    \b
    Examples:
      subtitle-gen build-slots              # extract slots + precompute vectors
      subtitle-gen build-slots --skip-vectors  # extract slots only
    """
    conn = get_db()
    build_slots(conn)
    if not skip_vectors:
        precompute_remix_data(conn)
    conn.close()


@cli.command("precompute-vectors")
def precompute_vectors_cmd():
    """Pre-compute remix classifications and word vectors.

    Loads spaCy en_core_web_md to compute vector embeddings for remix-relevant
    fillers and classify of-object fillers for remix type. Stores scalar
    decomposition in the database so runtime needs no numpy or spaCy.

    \b
    Examples:
      subtitle-gen precompute-vectors   # recompute all vectors
    """
    conn = get_db()
    ensure_slot_tables(conn)
    precompute_remix_data(conn)
    conn.close()


@cli.command()
@click.option("--count", "-n", default=None, type=click.IntRange(min=1), help="Number of subtitles to generate (default: 10, or 1 with --jacket).")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--jacket", is_flag=True, help="Generate full book jacket (title, back cover, reviews, blurbs).")
@click.option("--sources", is_flag=True, help="Show which real books each slot filler came from.")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in jacket output.")
@click.option("--tone", default=None, help="Filter by accessibility tier: pop, mainstream, niche (comma-separated for multiple, e.g. 'pop,mainstream').")
@click.option("--remix/--no-remix", default=True, help="Enable/disable of-object remixing (default: enabled).")
@click.option("--remix-prob", default=None, type=click.FloatRange(min=0.0, max=1.0), help="Probability of remixing a multi-word of-object (0.0-1.0). Default: calibrated or 0.8.")
@click.option("--min-sim", default=None, type=click.FloatRange(min=0.0, max=1.0), help="Minimum cosine similarity for remix coherence filter. Default: calibrated or 0.1.")
@click.option("--review", is_flag=True, help="Interactively rate each subtitle (thumbs, tone override, comment).")
def generate(count: int | None, seed: int | None, jacket: bool, sources: bool, model: str | None, show_concept: bool, tone: str | None, remix: bool, remix_prob: float | None, min_sim: float | None, review: bool):
    """Generate random subtitles in the "X, Y, and [the/a/an] Z of W" pattern.

    Draws slot fillers from the extracted pool, optionally remixing multi-word
    of-objects into novel combinations (enabled by default).

    \b
    Examples:
      subtitle-gen generate                         # 10 random subtitles
      subtitle-gen generate -n 5 --sources          # 5 with source books
      subtitle-gen generate --tone pop              # bias toward accessible
      subtitle-gen generate --no-remix              # original of-objects only
      subtitle-gen generate --jacket                # 1 subtitle + full jacket
      subtitle-gen generate --review                # rate each subtitle
    """
    tone_set = _parse_tone(tone)

    if count is None:
        count = 1 if jacket else 10

    conn = get_db()
    stats = slot_stats(conn)
    if not stats:
        raise click.ClickException("No slots found. Run 'subtitle-gen build-slots' first.")

    # Use calibrated defaults (baked in), DB override, or CLI override
    if remix_prob is None:
        row = conn.execute("SELECT value FROM config WHERE key = 'remix_calibrated_remix_prob'").fetchone()
        remix_prob = float(row[0]) if row else 0.8
    if min_sim is None:
        row = conn.execute("SELECT value FROM config WHERE key = 'remix_calibrated_min_sim'").fetchone()
        min_sim = float(row[0]) if row else 0.1
    effective_remix_prob = remix_prob if remix else 0.0
    click.echo(f"Slot machine loaded: {stats}")
    if tone_set:
        click.echo(f"Tier filter: {', '.join(sorted(tone_set))}")
    if effective_remix_prob > 0:
        click.echo(f"Remix: prob={effective_remix_prob:.1f}, min_sim={min_sim:.2f}")
    click.echo()

    reviewed_count = 0
    thumbs_up = 0
    thumbs_down = 0

    for i in range(count):
        s = seed + i if seed is not None else None
        try:
            sub = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=tone_set,
                seed=s,
                remix_prob=effective_remix_prob,
                min_sim=min_sim,
            )
        except TierFilterError as exc:
            raise click.ClickException(str(exc)) from exc

        if jacket:
            click.echo(f"Generating jacket for: {sub.text}\n")
            kwargs = {"model": model} if model else {}
            try:
                md = generate_jacket(
                    sub.text,
                    show_concept=show_concept,
                    conn=conn,
                    allowed_tiers=tone_set,
                    **kwargs,
                )
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
            click.echo(md)
            if sources:
                click.echo(format_sources(conn, sub))
            if i < count - 1:
                click.echo("\n" + "=" * 72 + "\n")
        else:
            click.echo(f"  {i + 1:2d}. {sub.text}")
            if sources:
                click.echo(format_sources(conn, sub))
                click.echo()

        if review:
            result = _prompt_review(conn, sub.text)
            if result:
                reviewed_count += 1
                if result == 1:
                    thumbs_up += 1
                elif result == -1:
                    thumbs_down += 1

    if review and reviewed_count > 0:
        click.echo(f"\nReviewed {reviewed_count}/{count} subtitles ({thumbs_up} 👍, {thumbs_down} 👎). Ratings saved.")

    conn.close()


@cli.command()
@click.option("--count", "-n", default=20, type=click.IntRange(min=1), help="Number of subtitles to review (default: 20).")
@click.option("--tone", default=None, help="Filter by tone tier: pop, mainstream, niche.")
def review(count: int, tone: str | None):
    """Rate subtitles interactively in a dedicated review session.

    Generates subtitles one at a time and prompts for thumbs up/down,
    tone override, and optional comments. Ratings are stored in the DB
    and used by the tuning pipeline.

    \b
    Examples:
      subtitle-gen review                  # review 20 random subtitles
      subtitle-gen review -n 10 --tone pop # review 10 pop subtitles
    """
    tone_set = _parse_tone(tone)

    conn = get_db()
    stats = slot_stats(conn)
    if not stats:
        raise click.ClickException("No slots found. Run 'subtitle-gen build-slots' first.")

    # Load calibrated remix settings
    row = conn.execute("SELECT value FROM config WHERE key = 'remix_calibrated_remix_prob'").fetchone()
    remix_prob = float(row[0]) if row else 0.8
    row = conn.execute("SELECT value FROM config WHERE key = 'remix_calibrated_min_sim'").fetchone()
    min_sim = float(row[0]) if row else 0.1

    click.echo(f"Review session: {count} subtitles" + (f" (tone: {tone})" if tone else ""))
    click.echo("Rate each subtitle — all prompts are skippable with Enter.\n")

    reviewed = 0
    thumbs_up = 0
    thumbs_down = 0

    for i in range(count):
        try:
            sub = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=tone_set,
                remix_prob=remix_prob,
                min_sim=min_sim,
            )
        except TierFilterError as exc:
            raise click.ClickException(str(exc)) from exc
        click.echo(f"  {i + 1:2d}. {sub.text}")

        result = _prompt_review(conn, sub.text)
        if result is not None:
            reviewed += 1
            if result == 1:
                thumbs_up += 1
            elif result == -1:
                thumbs_down += 1
        click.echo()

    click.echo(f"Reviewed {reviewed}/{count} subtitles ({thumbs_up} 👍, {thumbs_down} 👎). Ratings saved.")
    conn.close()


@cli.command()
@click.argument("subtitle", required=False, default=None)
@click.option("--seed", default=None, type=int, help="Random seed (only for random generation).")
@click.option("--sources", is_flag=True, help="Show source books for each slot filler (only for random generation).")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in output.")
@click.option("--tone", default=None, help="Override tone tier: pop, mainstream, niche (comma-separated for multiple).")
def jacket(subtitle: str | None, seed: int | None, sources: bool, model: str | None, show_concept: bool, tone: str | None):
    """Generate a full book jacket — title, back cover, reviews, and blurbs.

    Pass a subtitle string to jacket a specific text, or omit to generate a random one.

    \b
    Examples:
      subtitle-gen jacket "sturgeon, caviar, and the geography of desire"
      subtitle-gen jacket                    # random subtitle
      subtitle-gen jacket --sources          # random + show sources
      subtitle-gen jacket --model claude-haiku-4.5  # use a different model
    """
    kwargs = {"model": model} if model else {}
    tone_set = _parse_tone(tone)
    conn = get_db()
    if subtitle:
        click.echo(f"Generating jacket for: {subtitle}\n")
        try:
            md = generate_jacket(
                subtitle,
                show_concept=show_concept,
                conn=conn,
                allowed_tiers=tone_set,
                **kwargs,
            )
        except ValueError as exc:
            conn.close()
            raise click.ClickException(str(exc)) from exc
        click.echo(md)
    else:
        stats = slot_stats(conn)
        if not stats:
            conn.close()
            raise click.ClickException("No slots found. Run 'subtitle-gen build-slots' first.")
        click.echo(f"Slot machine loaded: {stats}\n")
        try:
            sub = generate_subtitle_matching_tiers(
                conn,
                allowed_tiers=tone_set,
                seed=seed,
            )
        except TierFilterError as exc:
            conn.close()
            raise click.ClickException(str(exc)) from exc
        click.echo(f"Generating jacket for: {sub.text}\n")
        try:
            md = generate_jacket(
                sub.text,
                show_concept=show_concept,
                conn=conn,
                allowed_tiers=tone_set,
                **kwargs,
            )
        except ValueError as exc:
            conn.close()
            raise click.ClickException(str(exc)) from exc
        click.echo(md)
        if sources:
            click.echo(format_sources(conn, sub))
    conn.close()


@cli.command()
@click.option("--slot-type", default=None, help="Filter by slot type.")
@click.option("--sample", default=20, type=click.IntRange(min=1), help="Number of fillers to show per type.")
def slots(slot_type: str | None, sample: int):
    """Show available slot fillers.

    \b
    Examples:
      subtitle-gen slots                          # sample all types
      subtitle-gen slots --slot-type of_object    # just of-objects
      subtitle-gen slots --sample 5               # fewer per type
    """
    conn = get_db()
    if slot_type:
        types = [slot_type]
    else:
        types = [r[0] for r in conn.execute(
            "SELECT DISTINCT slot_type FROM slot_fillers"
        ).fetchall()]
    for st in types:
        total = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ?", (st,)
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT filler FROM slot_fillers WHERE slot_type = ? ORDER BY RANDOM() LIMIT ?",
            (st, sample),
        ).fetchall()
        click.echo(f"\n{st} ({total:,} total):")
        for (f,) in rows:
            click.echo(f"  {f}")
    conn.close()


@cli.command("calibrate-remix")
@click.option("--samples", default=50, type=click.IntRange(min=1), help="Subtitles per parameter level (default: 50). Rated in a single LLM call per level.")
@click.option("--model", default=None, help="LLM model for rating (default: github_copilot/gpt-5.4-mini).")
def calibrate_remix_cmd(samples: int, model: str | None):
    """Auto-tune remix parameters using LLM-based rating.

    Generates subtitles at various remix_prob and min_sim levels, rates them
    with an LLM, and stores the best values in the database.

    \b
    Examples:
      subtitle-gen calibrate-remix                     # 50 samples/level
      subtitle-gen calibrate-remix --samples 100       # higher confidence
      subtitle-gen calibrate-remix --model gpt-4.1     # different rater
    """
    from subtitle_generator.calibrate import run_calibration
    from subtitle_generator.eval_harness import DEFAULT_RATER_MODEL
    conn = get_db()
    run_calibration(conn, samples=samples, model=model or DEFAULT_RATER_MODEL)
    conn.close()


@cli.command()
@click.option("--phase", type=click.Choice(["remix", "tone", "all"]), default="all", help="Which phase to run (default: all).")
@click.option("--iterations", default=30, type=click.IntRange(min=1), help="Autoresearch iterations for tone phase (default: 30).")
@click.option("--samples", default=50, type=click.IntRange(min=1), help="Subtitles per level for remix phase (default: 50).")
@click.option("--rater-model", default=None, help="Model for rating subtitles (default: github_copilot/gpt-5.4-mini).")
@click.option("--proposer-model", default=None, help="Model for proposing param changes (default: github_copilot/gpt-5.4).")
@click.option("--results-file", default="results.tsv", help="TSV file for experiment log (default: results.tsv).")
@click.option("--dry-run", is_flag=True, help="Show proposals without evaluating or applying.")
@click.option("--show-results", is_flag=True, help="Display past tuning results and exit.")
@click.option("--spot-check", is_flag=True, hidden=True, help="Deprecated. Use 'subtitle-gen spot-check' instead.")
@click.option("--spot-check-tui", is_flag=True, hidden=True, help="Deprecated. Use 'subtitle-gen spot-check --tui' instead.")
@click.option("--debug", is_flag=True, help="Enable verbose litellm debug logging.")
def tune(phase: str, iterations: int, samples: int, rater_model: str | None, proposer_model: str | None, results_file: str, dry_run: bool, show_results: bool, spot_check: bool, spot_check_tui: bool, debug: bool):
    """Unified tuning pipeline (autoresearch-inspired).

    Pure automated loop — no human input during the run. Human feedback
    flows through tuning_goals.md edits between runs (see spot-check and
    review-ratings commands).

    Runs two phases:
      Phase 1 (remix): Grid sweep over min_sim and remix_prob
      Phase 2 (tone): LLM-proposed single-parameter hill-climbing

    \b
    Examples:
      subtitle-gen tune                              # full pipeline
      subtitle-gen tune --phase tone --iterations 10 # tone only, quick
      subtitle-gen tune --phase remix --samples 100  # remix only, high confidence
      subtitle-gen tune --dry-run                    # show proposals only
      subtitle-gen tune --show-results               # view experiment history
    """
    if spot_check or spot_check_tui:
        click.echo("⚠ --spot-check flags are deprecated. Use 'subtitle-gen spot-check' instead.")
        click.echo("  Continuing without spot-checks.\n")
    if show_results:
        from pathlib import Path
        path = Path(results_file)
        if not path.exists():
            click.echo(f"No results file found at {results_file}")
            return
        content = path.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        click.echo(f"=== Tuning Results ({len(lines) - 1} experiments) ===\n")
        for line in lines:
            click.echo(line)
        return

    if debug:
        import litellm
        litellm.set_verbose = True

    from subtitle_generator.eval_harness import DEFAULT_RATER_MODEL, DEFAULT_PROPOSER_MODEL
    from subtitle_generator.tune import run_full_tuning
    conn = get_db()
    run_full_tuning(
        conn,
        phase=phase,
        iterations=iterations,
        samples=samples,
        rater_model=rater_model or DEFAULT_RATER_MODEL,
        proposer_model=proposer_model or DEFAULT_PROPOSER_MODEL,
        results_file=results_file,
        dry_run=dry_run,
    )
    conn.close()


@cli.command("spot-check")
@click.option("--samples", default=2, type=click.IntRange(min=1, max=5), help="Samples per tier (default: 2).")
@click.option("--source", default="spot_check", help="Rating source tag (default: spot_check).")
@click.pass_context
def spot_check_cmd(ctx, samples: int, source: str):
    """Rate subtitle output by tone tier (separate from tuning).

    Generates samples for each tier (pop/mainstream/niche) and asks which
    tier each subtitle feels like. Stores ratings in human_ratings table.

    Run this between tune runs, then use review-ratings to analyze
    results and propose tuning_goals.md edits.

    For a richer experience, use the web UI: run 'subtitle-gen serve'
    then visit http://localhost:8742/spot-check.html

    \b
    Examples:
      subtitle-gen spot-check              # CLI mode, 2 per tier
      subtitle-gen spot-check --samples 3  # 3 per tier = 9 total
    """
    conn = ctx.obj["conn"]
    from subtitle_generator.tune import run_spot_check
    run_spot_check(conn, n_samples=samples, source=source)
    conn.close()


@cli.command("review-ratings")
@click.option("--since", default=None, help="ISO date to review from (default: all recent).")
@click.option("--source", default=None, help="Filter by source (spot_check, web_user, pull_ratings).")
@click.option("--model", default=None, help="LLM model for analysis (default: proposer model).")
@click.pass_context
def review_ratings_cmd(ctx, since: str | None, source: str | None, model: str | None):
    """Analyze human ratings and propose tuning_goals.md edits.

    Reads recent human_ratings, identifies mismatch patterns, and has
    an LLM propose specific edits to tuning_goals.md. Shows the diff
    for human approval — does NOT write the file automatically.

    \b
    Examples:
      subtitle-gen review-ratings                     # analyze all recent
      subtitle-gen review-ratings --source spot_check # spot-check only
      subtitle-gen review-ratings --source web_user   # end-user only
      subtitle-gen review-ratings --since 2026-04-15  # since date
    """
    conn = ctx.obj["conn"]
    from subtitle_generator.tune import review_ratings
    from subtitle_generator.eval_harness import DEFAULT_PROPOSER_MODEL
    review_ratings(conn, since=since, source=source, model=model or DEFAULT_PROPOSER_MODEL)
    conn.close()


@cli.command()
@click.option("--port", default=8742, type=click.IntRange(min=1024, max=65535), help="Port to listen on.")
@click.option("--no-open", is_flag=True, help="Don't open browser automatically.")
def serve(port: int, no_open: bool):
    """Start the web app locally.

    Runs a local HTTP server serving the web frontend and API endpoints.
    Opens the default browser automatically.

    \b
    Examples:
      subtitle-gen serve                # start on port 8742, open browser
      subtitle-gen serve --port 9000    # custom port
      subtitle-gen serve --no-open      # don't open browser
    """
    import threading
    import webbrowser

    from subtitle_generator.serve import create_server

    web_dir = Path(__file__).parent.parent.parent / "web"
    if not web_dir.is_dir():
        click.echo(f"Warning: web/ directory not found at {web_dir}")
        click.echo("  API endpoints will still be served.\n")

    server = create_server(port=port, web_dir=web_dir)
    url = f"http://localhost:{port}"
    click.echo(f"Serving on {url}")
    click.echo("Press Ctrl+C to stop.\n")

    if not no_open:
        threading.Timer(0.5, webbrowser.open, args=[url]).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down.")
        server.shutdown()


@cli.command("export-db")
@click.option("--output", "-o", default="api/data/subtitles.mini.db", help="Output path for mini DB.")
def export_db_cmd(output: str):
    """Export a minimal SQLite database for web/API deployment.

    Creates a small (~1-2 MB) DB with just the tables needed for subtitle
    generation: slot_fillers, config, and a sources lookup table.

    \b
    Examples:
      subtitle-gen export-db                           # default output
      subtitle-gen export-db -o web/data/mini.db       # custom path
    """
    output_path = Path(output)
    conn = get_db()
    click.echo(f"Exporting mini DB to {output_path} ...")
    stats = export_mini_db(conn, output_path)
    conn.close()

    for table, count in stats.items():
        click.echo(f"  {table}: {count:,} rows")
    size_kb = output_path.stat().st_size / 1024
    if size_kb >= 1024:
        click.echo(f"Output: {output_path} ({size_kb / 1024:.1f} MB)")
    else:
        click.echo(f"Output: {output_path} ({size_kb:.0f} KB)")


@cli.command("export-data")
@click.option("--output-dir", "-o", default="api/data", help="Output directory for CSV files.")
def export_data_cmd(output_dir: str):
    """Export slot data as CSV files for version control.

    Writes slot_fillers.csv, config.csv, and sources.csv to the output
    directory. These text files are committed to the repo and used by
    'build-db' in CI to construct the SQLite deployment artifact.

    \b
    Examples:
      subtitle-gen export-data                  # default: api/data/
      subtitle-gen export-data -o data/export   # custom directory
    """
    out = Path(output_dir)
    conn = get_db()
    click.echo(f"Exporting data to {out}/ ...")
    stats = export_data(conn, out)
    conn.close()

    for filename, count in stats.items():
        size_kb = (out / filename).stat().st_size / 1024
        click.echo(f"  {filename}: {count:,} rows ({size_kb:.0f} KB)")


@cli.command("classify-source-tiers")
@click.option("--limit", default=20, show_default=True, help="Maximum rows to label.")
@click.option(
    "--batch-size",
    default=10,
    show_default=True,
    help=(
        "Rows per LLM batch; for hosted web search this is also the "
        "concurrency burst size."
    ),
)
@click.option(
    "--model",
    default=None,
    help="LLM model to use. Defaults to the configured rater model.",
)
@click.option(
    "--selection",
    type=click.Choice(["random", "id"]),
    default="random",
    show_default=True,
    help="How to choose unlabeled pattern_matches rows.",
)
@click.option(
    "--random-seed",
    default=20260501,
    show_default=True,
    help="Seed used when --selection=random.",
)
@click.option("--force", is_flag=True, help="Relabel already-labeled rows too.")
@click.option(
    "--candidate-source",
    type=click.Choice(["all", "subtitle", "title"]),
    default="all",
    show_default=True,
    help="Restrict labeling to rows parsed from title-only or subtitle text.",
)
@click.option("--dry-run", is_flag=True, help="Show selected rows without calling the LLM.")
@click.option(
    "--web-search/--no-web-search",
    default=True,
    show_default=True,
    help="Use hosted Responses web_search for evidence-grounded labels.",
)
@click.option("--no-export", is_flag=True, help="Do not refresh source_tier_labels.csv.")
@click.option(
    "--output",
    default="api/data/source_tier_labels.csv",
    show_default=True,
    help="CSV path for exported labels.",
)
def classify_source_tiers_cmd(
    limit: int,
    batch_size: int,
    model: str | None,
    selection: str,
    random_seed: int,
    force: bool,
    candidate_source: str,
    dry_run: bool,
    web_search: bool,
    no_export: bool,
    output: str,
):
    """LLM-label source title market tiers on pattern_matches rows."""
    from subtitle_generator.parameter_state import DEFAULT_RATER_MODEL
    from subtitle_generator.source_tier_enrichment import classify_source_tiers

    conn = get_db()
    try:
        result = classify_source_tiers(
            conn,
            limit=limit,
            batch_size=batch_size,
            model=model or DEFAULT_RATER_MODEL,
            selection=selection,
            random_seed=random_seed,
            force=force,
            candidate_source=candidate_source,
            dry_run=dry_run,
            web_search=web_search,
            export_path=None if no_export else Path(output),
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        conn.close()

    if dry_run:
        click.echo(
            f"Selected {len(result.selected)} rows "
            f"(selection={selection}, random_seed={random_seed}):"
        )
        for candidate in result.selected:
            click.echo(f"  {candidate.id}: {candidate.title} — {candidate.subtitle}")
    else:
        click.echo(f"Labeled {result.labeled_count} source-title rows.")
    if result.export_path and not result.dry_run:
        click.echo(f"Exported {result.exported_count} labels to {result.export_path}")


@cli.command("source-tier-distribution")
@click.option(
    "--min-confidence",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.0,
    show_default=True,
    help="Minimum LLM confidence for a row to count as labeled.",
)
@click.option(
    "--min-labeled",
    "--min-labeled-per-source",
    type=click.IntRange(min=1),
    default=100,
    show_default=True,
    help="Minimum total labeled rows before source-tier calibration is ready.",
)
def source_tier_distribution_cmd(
    min_confidence: float,
    min_labeled: int,
):
    """Report the combined source-tier distribution for calibration."""
    from subtitle_generator.source_tier_enrichment import (
        format_source_tier_distribution_report,
        load_source_tier_distribution,
    )

    conn = get_db()
    try:
        rows = load_source_tier_distribution(conn, min_confidence=min_confidence)
    finally:
        conn.close()
    click.echo(
        format_source_tier_distribution_report(
            rows,
            min_labeled=min_labeled,
        )
    )


@cli.command("build-db")
@click.option("--data-dir", "-d", default="api/data", help="Directory containing CSV files.")
@click.option("--output", "-o", default="api/data/subtitles.mini.db", help="Output SQLite path.")
def build_db_cmd(data_dir: str, output: str):
    """Build a mini SQLite database from exported CSV files.

    Reads slot_fillers.csv, config.csv, and sources.csv from the data
    directory and constructs an indexed SQLite database for deployment.

    \b
    Examples:
      subtitle-gen build-db                     # default paths
      subtitle-gen build-db -d data/export -o deploy/mini.db
    """
    data = Path(data_dir)
    out = Path(output)

    for f in ["slot_fillers.csv", "config.csv", "sources.csv"]:
        if not (data / f).exists():
            raise click.ClickException(f"Missing {data / f}. Run 'subtitle-gen export-data' first.")

    click.echo(f"Building mini DB from {data}/ ...")
    stats = build_mini_db(data, out)

    for table, count in stats.items():
        click.echo(f"  {table}: {count:,} rows")
    size_kb = out.stat().st_size / 1024
    click.echo(f"Output: {out} ({size_kb:.0f} KB)")


@cli.command("pull-ratings")
@click.option("--since", default=None, help="ISO date to sync from (default: all).")
@click.option("--account", default=None, help="Storage account name (default: $STORAGE_ACCOUNT_NAME).")
@click.pass_context
def pull_ratings_cmd(ctx, since: str | None, account: str | None):
    """Sync ratings from Azure Table Storage to local SQLite.

    Pulls ratings from the deployed Azure Table Storage 'ratings' table
    into the local human_ratings SQLite table, deduplicating by RowKey.

    Examples:
      subtitle-gen pull-ratings                    # sync all
      subtitle-gen pull-ratings --since 2026-04-01 # sync from date
    """
    import os

    account_name = account or os.environ.get("STORAGE_ACCOUNT_NAME")
    if not account_name:
        raise click.ClickException(
            "No storage account. Set STORAGE_ACCOUNT_NAME or use --account."
        )

    try:
        from azure.data.tables import TableServiceClient
        from azure.identity import DefaultAzureCredential
    except ImportError:
        raise click.ClickException(
            "azure-data-tables not installed. Run: uv pip install azure-data-tables azure-identity"
        )

    conn = ctx.obj["conn"]
    from subtitle_generator.feedback import ensure_ratings_table, store_rating
    ensure_ratings_table(conn)

    # Get existing RowKeys to deduplicate
    existing = set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT config_snapshot FROM human_ratings WHERE config_snapshot LIKE '%RowKey%'"
        ).fetchall()
        import json as _json
        for (snap,) in rows:
            try:
                d = _json.loads(snap)
                if "RowKey" in d:
                    existing.add(d["RowKey"])
            except Exception:
                pass
    except Exception:
        pass

    credential = DefaultAzureCredential()
    service = TableServiceClient(
        endpoint=f"https://{account_name}.table.core.windows.net",
        credential=credential,
    )
    table = service.get_table_client("ratings")

    import json as _json
    query_filter = None
    if since:
        query_filter = f"RowKey ge '{since}'"

    synced = 0
    skipped = 0
    for entity in table.list_entities(filter=query_filter):
        row_key = entity["RowKey"]
        if row_key in existing:
            skipped += 1
            continue

        tags_raw = entity.get("tags", "[]")
        try:
            tags = _json.loads(tags_raw)
        except Exception:
            tags = None

        thumbs_val = entity.get("thumbs")
        if thumbs_val is not None:
            thumbs_val = int(thumbs_val)

        store_rating(
            conn,
            entity.get("subtitle", ""),
            system_tone=entity.get("system_tone") or None,
            thumbs=thumbs_val,
            tone_override=entity.get("tone_override") or None,
            free_text=entity.get("free_text") or None,
            tags=tags,
            source="pull_ratings",
        )
        # Store RowKey in config_snapshot for dedup on next sync
        conn.execute(
            "UPDATE human_ratings SET config_snapshot = ? WHERE id = last_insert_rowid()",
            (_json.dumps({"RowKey": row_key}),),
        )
        conn.commit()
        synced += 1

    click.echo(f"Synced {synced} ratings ({skipped} duplicates skipped).")


_POP_SOURCES = ["spl", "ol", "gr", "ottawa", "nyt", "trove"]
_POP_LOOKUP_FILES = {
    "spl": Path("data/spl_checkout_lookup.json"),
    "ol": Path("data/ol_edition_lookup.json"),
    "gr": Path("data/goodreads_lookup.json"),
    "ottawa": Path("data/canadian_library_lookup.json"),
    "nyt": Path("data/nyt_bestseller_lookup.json"),
    "trove": Path("data/trove_holdings_lookup.json"),
}


def _pop_status():
    """Show which popularity lookup files exist."""
    import json
    from datetime import datetime

    for name, path in _POP_LOOKUP_FILES.items():
        if path.exists():
            size = path.stat().st_size
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
            with open(path) as f:
                count = len(json.load(f))
            if size >= 1e6:
                click.echo(f"  {name:8s}  {size / 1e6:7.1f} MB  {count:>10,} entries  updated {mtime}")
            else:
                click.echo(f"  {name:8s}  {size / 1e3:7.0f} KB  {count:>10,} entries  updated {mtime}")
        else:
            click.echo(f"  {name:8s}  -- not downloaded --")


@cli.command("download-popularity")
@click.option("--sources", default=None, help="Comma-separated sources to download (spl,ol,gr,ottawa,nyt,trove). Default: all except API-keyed sources without keys.")
@click.option("--nyt-api-key", default=None, envvar="NYT_API_KEY", help="NYT API key (or set NYT_API_KEY env var)")
@click.option("--nyt-max-requests", type=int, default=None, help="Stop NYT after N requests")
@click.option("--trove-api-key", default=None, envvar="TROVE_API_KEY", help="Trove API key (or set TROVE_API_KEY env var)")
@click.option("--trove-limit", type=int, default=None, help="Stop Trove after N target ISBNs")
@click.option("--trove-max-requests", type=int, default=None, help="Stop Trove after N API requests")
@click.option("--trove-rate-per-minute", type=int, default=180, show_default=True, help="Trove request rate below quota")
@click.option("--trove-quota-per-minute", type=int, default=200, show_default=True, help="Approved Trove quota")
@click.option("--trove-workers", type=int, default=1, show_default=True, help="Concurrent Trove ISBN workers")
@click.option("--trove-bulk-pages", type=int, default=0, help="Harvest N Trove bulk pages before targeted ISBN search")
@click.option("--trove-bulk-page-size", type=int, default=20, show_default=True, help="Trove works per bulk page")
@click.option("--trove-bulk-full", is_flag=True, help="Use full work/version records for Trove bulk pages")
@click.option("--trove-db", type=click.Path(path_type=Path), default=DB_PATH, show_default=True, help="SQLite DB used for Trove target ISBNs")
@click.option(
    "--trove-target-mode",
    type=click.Choice(["slot-sources", "all"]),
    default="slot-sources",
    show_default=True,
    help="Which ISBNs to load from --trove-db",
)
@click.option("--trove-include-libraries", is_flag=True, help="Persist Trove library NUC/name lists")
@click.option("--status", is_flag=True, help="Show what's downloaded and exit")
def download_popularity(
    sources,
    nyt_api_key,
    nyt_max_requests,
    trove_api_key,
    trove_limit,
    trove_max_requests,
    trove_rate_per_minute,
    trove_quota_per_minute,
    trove_workers,
    trove_bulk_pages,
    trove_bulk_page_size,
    trove_bulk_full,
    trove_db,
    trove_target_mode,
    trove_include_libraries,
    status,
):
    """Download popularity data sources (SPL, OL editions, Goodreads, Ottawa, NYT, Trove).

    \b
    Sources:
      spl     Seattle Public Library checkouts (~11 GB raw CSV)
      ol      Open Library edition counts (requires download-ol first)
      gr      Goodreads / UCSD Book Graph (~2 GB download)
      ottawa  Ottawa Public Library holds data
      nyt     NYT bestseller lists (multi-day, resumable, needs API key)
      trove   Trove Australia book holdings (resumable, needs API key)

    \b
    Examples:
      subtitle-gen download-popularity                    # all (except NYT w/o key)
      subtitle-gen download-popularity --sources spl,gr   # specific sources
      subtitle-gen download-popularity --sources nyt --nyt-api-key KEY
      subtitle-gen download-popularity --sources trove --trove-limit 100
      subtitle-gen download-popularity --sources trove --trove-bulk-pages 1000 --trove-bulk-full
      subtitle-gen download-popularity --sources trove --trove-db data/db/subtitles.db
      subtitle-gen download-popularity --status           # show what's downloaded
    """
    import subprocess

    if status:
        click.echo("Popularity data status:")
        _pop_status()
        return

    if sources:
        selected = [s.strip().lower() for s in sources.split(",")]
        invalid = [s for s in selected if s not in _POP_SOURCES]
        if invalid:
            raise click.ClickException(f"Unknown source(s): {', '.join(invalid)}. Choose from: {', '.join(_POP_SOURCES)}")
    else:
        selected = [
            s for s in _POP_SOURCES
            if (s != "nyt" or nyt_api_key) and (s != "trove" or trove_api_key)
        ]

    for src in selected:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Downloading: {src}")
        click.echo(f"{'=' * 60}")

        if src == "spl":
            subprocess.run([sys.executable, "data/spl_stream.py"], check=True)

        elif src == "ol":
            ol_dump = Path("data/raw/ol_dump_editions_latest.txt.gz")
            if not ol_dump.exists():
                raise click.ClickException(
                    "OL editions dump not found. Run 'subtitle-gen download-ol' first, "
                    "then re-run with --sources ol."
                )
            subprocess.run([sys.executable, "data/ol_edition_extract.py"], check=True)

        elif src == "gr":
            subprocess.run([sys.executable, "data/goodreads_stream.py"], check=True)

        elif src == "ottawa":
            subprocess.run([
                sys.executable, "data/canadian_library_stream.py",
                "--download", "--skip-isbn-lookup",
            ], check=True)

        elif src == "nyt":
            if not nyt_api_key:
                raise click.ClickException("NYT requires --nyt-api-key or NYT_API_KEY env var.")
            args = [sys.executable, "data/nyt_stream.py", "--api-key", nyt_api_key]
            if nyt_max_requests:
                args.extend(["--max-requests", str(nyt_max_requests)])
            subprocess.run(args, check=True)
            # Auto-export partial data to lookup JSON
            subprocess.run([sys.executable, "data/nyt_stream.py", "--export"], check=True)

        elif src == "trove":
            if not trove_api_key:
                raise click.ClickException("Trove requires --trove-api-key or TROVE_API_KEY env var.")
            args = [
                sys.executable, "data/trove_stream.py",
                "--api-key", trove_api_key,
                "--rate-per-minute", str(trove_rate_per_minute),
                "--quota-per-minute", str(trove_quota_per_minute),
                "--workers", str(trove_workers),
                "--db", str(trove_db),
                "--db-target-mode", trove_target_mode,
                "--no-ol-targets",
            ]
            if trove_bulk_pages:
                args.extend([
                    "--bulk-pages", str(trove_bulk_pages),
                    "--bulk-page-size", str(trove_bulk_page_size),
                    "--limit", "0",
                ])
                if trove_bulk_full:
                    args.append("--bulk-full")
            if trove_limit is not None:
                args.extend(["--limit", str(trove_limit)])
            if trove_max_requests is not None:
                args.extend(["--max-requests", str(trove_max_requests)])
            if trove_include_libraries:
                args.append("--include-libraries")
            subprocess.run(args, check=True)

    click.echo(f"\n{'=' * 60}")
    click.echo("Summary:")
    _pop_status()


@cli.command("populate-popularity")
@click.option("--spl", "w_spl", type=float, default=None, help="Override pop_weight_spl")
@click.option("--ol", "w_ol", type=float, default=None, help="Override pop_weight_ol")
@click.option("--gr", "w_gr", type=float, default=None, help="Override pop_weight_gr")
@click.option("--library", "w_library", type=float, default=None, help="Override pop_weight_library")
@click.option("--nyt", "w_nyt", type=float, default=None, help="Override pop_weight_nyt")
@click.option("--trove", "w_trove", type=float, default=None, help="Override pop_weight_trove")
@click.option("--exponent", type=float, default=None, help="Override pop_exponent")
@click.option("--skip-calibrate", is_flag=True, help="Skip threshold calibration")
@click.option("--skip-data-model", is_flag=True, help="Skip ISBN alias / filler-source rebuild")
def populate_popularity(w_spl, w_ol, w_gr, w_library, w_nyt, w_trove, exponent, skip_calibrate, skip_data_model):
    """Build ISBN mappings and compute popularity scores from all available sources.

    \b
    Runs the full popularity pipeline:
      1. Build ISBN aliases + filler-source mapping (unless --skip-data-model)
      2. Load all available popularity lookups (SPL, OL, Goodreads, Ottawa, NYT, Trove)
      3. Compute composite scores via weighted-average percentile normalization
      4. Score slot fillers (L1 top-3 mean + L2 corpus fallback)
      5. Auto-calibrate tier thresholds (unless --skip-calibrate)

    \b
    Examples:
      subtitle-gen populate-popularity                     # full pipeline
      subtitle-gen populate-popularity --spl 0.7 --gr 0.3  # override weights
      subtitle-gen populate-popularity --skip-calibrate     # skip recalibration
      subtitle-gen populate-popularity --skip-data-model    # just re-score
    """
    import subprocess

    if not skip_data_model:
        click.echo("Step 1/2: Building ISBN aliases + filler-source mapping...")
        subprocess.run([sys.executable, "data/build_data_model.py"], check=True)
        click.echo()

    click.echo(f"Step {'2/2' if not skip_data_model else '1/1'}: Computing popularity scores...")
    args = [sys.executable, "data/populate_popularity.py"]
    if w_spl is not None:
        args.extend(["--spl", str(w_spl)])
    if w_ol is not None:
        args.extend(["--ol", str(w_ol)])
    if w_gr is not None:
        args.extend(["--gr", str(w_gr)])
    if w_library is not None:
        args.extend(["--library", str(w_library)])
    if w_nyt is not None:
        args.extend(["--nyt", str(w_nyt)])
    if w_trove is not None:
        args.extend(["--trove", str(w_trove)])
    if exponent is not None:
        args.extend(["--exponent", str(exponent)])
    if skip_calibrate:
        args.append("--skip-calibrate")
    subprocess.run(args, check=True)

    click.echo("\nDone! Popularity scores updated.")


@cli.command("validate-pipeline")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DB_PATH, help="SQLite database to validate.")
@click.option("--embedding-version", default="2", show_default=True, help="Expected remix embedding version.")
def validate_pipeline_cmd(db_path: Path, embedding_version: str):
    """Run read-only pipeline readiness checks."""
    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        raise click.ClickException(f"Unable to open database read-only: {db_path}") from exc
    try:
        report = validate_pipeline(conn, expected_embedding_version=embedding_version)
    finally:
        conn.close()
    click.echo(format_validation_report(report))
    if not report.ok:
        raise click.exceptions.Exit(1)


@cli.command("tier-diagnostic")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DB_PATH, help="SQLite database to inspect.")
def tier_diagnostic_cmd(db_path: Path):
    """Report real-title pop/mainstream/niche fixture classification."""

    from subtitle_generator.tier_diagnostics import format_real_title_tier_report

    try:
        conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        raise click.ClickException(f"Unable to open database read-only: {db_path}") from exc
    try:
        click.echo(format_real_title_tier_report(conn))
    finally:
        conn.close()


@cli.command("calibrate-tier-gates")
@click.option("--db", "db_path", type=click.Path(path_type=Path), default=DB_PATH, help="SQLite database to inspect.")
@click.option("--min-confidence", type=click.FloatRange(min=0.0, max=1.0), default=0.0, show_default=True, help="Minimum source-label confidence to include.")
@click.option("--apply", "apply_suggestion", is_flag=True, help="Write the suggested tier gates to the config table.")
def calibrate_tier_gates_cmd(
    db_path: Path,
    min_confidence: float,
    apply_suggestion: bool,
):
    """Suggest deterministic tier gates from source-title labels."""

    from subtitle_generator.tier_diagnostics import (
        apply_tier_gate_calibration,
        format_tier_gate_calibration_report,
        suggest_tier_gate_config,
    )

    if apply_suggestion:
        conn = sqlite3.connect(db_path)
    else:
        try:
            conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.OperationalError as exc:
            raise click.ClickException(
                f"Unable to open database read-only: {db_path}"
            ) from exc
    try:
        calibration = suggest_tier_gate_config(
            conn,
            min_confidence=min_confidence,
        )
        click.echo(format_tier_gate_calibration_report(
            conn,
            min_confidence=min_confidence,
            calibration=calibration,
        ))
        if apply_suggestion:
            if calibration is None:
                raise click.ClickException("No source-title labels available to apply.")
            apply_tier_gate_calibration(conn, calibration)
            click.echo("\nApplied suggested tier gates to config.")
    finally:
        conn.close()


if __name__ == "__main__":
    cli()
