"""CLI entry point for subtitle-generator."""

from pathlib import Path

import click

from subtitle_generator.analyze import analyze_subtitles, build_pattern_index
from subtitle_generator.download import TOTAL_PARTS, download_part, parse_parts_arg
from subtitle_generator.extract import DATA_DIR, DB_PATH, extract_from_file, get_db
from subtitle_generator.extract_openlibrary import (
    download_ol_dump,
    ensure_isbn_column,
    extract_from_ol_dump,
)
from subtitle_generator.calibrate import run_calibration
from subtitle_generator.export_db import export_mini_db
from subtitle_generator.generate import TONE_TARGETS, format_sources, generate_subtitle, slot_stats
from subtitle_generator.jacket import (
    TONE_HIGH, TONE_LOW, TONE_MEDIUM,
    compute_accessibility, generate_jacket, sample_tone,
)
from subtitle_generator.slots import build_slots


_TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
_VALID_TONES = set(_TONE_CHOICES.keys())


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
    """
    pass


@cli.command()
def version():
    """Show version."""
    click.echo("subtitle-generator 0.1.0")


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
        click.echo(f"  {mrc_file.name}: {records:,} records → {subs:,} subtitles")

    click.echo(f"\nTotal: {total_records:,} records → {total_subtitles:,} subtitles")
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
    click.echo(f"\nDone: {lines:,} lines → {subs:,} subtitles ({dupes:,} duplicates skipped)")
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
def build_slots_cmd():
    """Extract slot fillers from matched subtitles (regex + NLP validated).

    Runs regex pattern matching, spaCy POS/NER validation, and of-object
    decomposition. Rebuilds the entire slot_fillers table from scratch.

    \b
    Examples:
      subtitle-gen build-slots   # extract all slot fillers
    """
    conn = get_db()
    build_slots(conn)
    conn.close()


@cli.command()
@click.option("--count", "-n", default=None, type=click.IntRange(min=1), help="Number of subtitles to generate (default: 10, or 1 with --jacket).")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--jacket", is_flag=True, help="Generate full book jacket (title, back cover, reviews, blurbs).")
@click.option("--sources", is_flag=True, help="Show which real books each slot filler came from.")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in jacket output.")
@click.option("--deep-research", is_flag=True, help="Two-phase generation: dedicated web search for concept research before jacket.")
@click.option("--tone", default=None, help="Filter by accessibility tier: pop, mainstream, niche (comma-separated for multiple, e.g. 'pop,mainstream').")
@click.option("--remix/--no-remix", default=True, help="Enable/disable of-object remixing (default: enabled).")
@click.option("--remix-prob", default=None, type=click.FloatRange(min=0.0, max=1.0), help="Probability of remixing a multi-word of-object (0.0-1.0). Default: calibrated or 0.8.")
@click.option("--min-sim", default=None, type=click.FloatRange(min=0.0, max=1.0), help="Minimum cosine similarity for remix coherence filter. Default: calibrated or 0.1.")
def generate(count: int | None, seed: int | None, jacket: bool, sources: bool, model: str | None, show_concept: bool, deep_research: bool, tone: str | None, remix: bool, remix_prob: float | None, min_sim: float | None):
    """Generate random subtitles in the "X, Y, and the Z of W" pattern.

    Draws slot fillers from the extracted pool, optionally remixing multi-word
    of-objects into novel combinations (enabled by default).

    \b
    Examples:
      subtitle-gen generate                         # 10 random subtitles
      subtitle-gen generate -n 5 --sources          # 5 with source books
      subtitle-gen generate --tone pop              # bias toward accessible
      subtitle-gen generate --no-remix              # original of-objects only
      subtitle-gen generate --jacket                # 1 subtitle + full jacket
      subtitle-gen generate --jacket --deep-research  # jacket with web search
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
        click.echo(f"Tone bias: {', '.join(sorted(tone_set))}")
    if effective_remix_prob > 0:
        click.echo(f"Remix: prob={effective_remix_prob:.1f}, min_sim={min_sim:.2f}")
    click.echo()

    # Compute per-slot tone targets (average across requested tiers)
    tone_target = None
    if tone_set:
        merged = {}
        for slot in ["list_item", "action_noun", "of_object"]:
            merged[slot] = sum(TONE_TARGETS[t][slot] for t in tone_set) / len(tone_set)
        tone_target = merged

    for i in range(count):
        s = seed + i if seed is not None else None
        sub = generate_subtitle(conn, seed=s, tone_target=tone_target, remix_prob=effective_remix_prob, min_sim=min_sim)

        if jacket:
            click.echo(f"Generating jacket for: {sub.text}\n")
            kwargs = {"model": model} if model else {}
            md = generate_jacket(sub.text, show_concept=show_concept, deep_research=deep_research, conn=conn, allowed_tiers=tone_set, **kwargs)
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
    conn.close()


@cli.command()
@click.argument("subtitle", required=False, default=None)
@click.option("--seed", default=None, type=int, help="Random seed (only for random generation).")
@click.option("--sources", is_flag=True, help="Show source books for each slot filler (only for random generation).")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in output.")
@click.option("--deep-research", is_flag=True, help="Two-phase generation: dedicated web search for concept research before jacket.")
@click.option("--tone", default=None, help="Override tone tier: pop, mainstream, niche (comma-separated for multiple).")
def jacket(subtitle: str | None, seed: int | None, sources: bool, model: str | None, show_concept: bool, deep_research: bool, tone: str | None):
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
        md = generate_jacket(subtitle, show_concept=show_concept, deep_research=deep_research, conn=conn, allowed_tiers=tone_set, **kwargs)
        click.echo(md)
    else:
        stats = slot_stats(conn)
        if not stats:
            conn.close()
            raise click.ClickException("No slots found. Run 'subtitle-gen build-slots' first.")
        click.echo(f"Slot machine loaded: {stats}\n")
        sub = generate_subtitle(conn, seed=seed)
        click.echo(f"Generating jacket for: {sub.text}\n")
        md = generate_jacket(sub.text, show_concept=show_concept, deep_research=deep_research, conn=conn, allowed_tiers=tone_set, **kwargs)
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
@click.option("--model", default="gpt-5.4-mini", help="LLM model for rating (default: gpt-5.4-mini).")
def calibrate_remix_cmd(samples: int, model: str):
    """Auto-tune remix parameters using LLM-based rating.

    Generates subtitles at various remix_prob and min_sim levels, rates them
    with an LLM, and stores the best values in the database.

    \b
    Examples:
      subtitle-gen calibrate-remix                     # 50 samples/level
      subtitle-gen calibrate-remix --samples 100       # higher confidence
      subtitle-gen calibrate-remix --model gpt-4.1     # different rater
    """
    conn = get_db()
    run_calibration(conn, samples=samples, model=model)
    conn.close()


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


if __name__ == "__main__":
    cli()
