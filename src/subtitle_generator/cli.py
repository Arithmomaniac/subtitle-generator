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
from subtitle_generator.generate import format_sources, generate_subtitle, slot_stats
from subtitle_generator.jacket import (
    TONE_HIGH, TONE_LOW, TONE_MEDIUM,
    compute_accessibility, generate_jacket,
)
from subtitle_generator.slots import build_loose_slots, build_slots
from subtitle_generator.tune import run_autoresearch


_TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
_TONE_THRESHOLDS = {"pop": 1.0, "mainstream": 0.5, "niche": 0.0}
_MAX_REJECTION_ATTEMPTS = 200


@click.group()
def cli():
    """Generate bizarre book subtitles from LOC MARC data."""
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
    """Download LOC MARC bulk data files (Books All, 2016 retrospective)."""
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
    """Extract subtitles from downloaded MARC files into SQLite."""
    raw_dir = DATA_DIR / "raw"  # DATA_DIR = .../data
    if parts:
        part_nums = parse_parts_arg(parts)
        mrc_files = [raw_dir / f"BooksAll.2016.part{p:02d}.utf8.mrc" for p in part_nums]
        mrc_files = [f for f in mrc_files if f.exists()]
    else:
        mrc_files = sorted(raw_dir.glob("*.mrc"))

    if not mrc_files:
        click.echo("No .mrc files found. Run 'download' first.")
        return

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
    """Download Open Library editions dump (~9.2 GB compressed)."""
    download_ol_dump(force=force)


@cli.command("extract-ol")
@click.option("--all-langs", is_flag=True, help="Include non-English subtitles.")
@click.option("--no-dedup", is_flag=True, help="Skip deduplication (faster for testing).")
def extract_ol(all_langs: bool, no_dedup: bool):
    """Extract subtitles from Open Library editions dump into SQLite."""
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
    """POS-tag subtitles and extract structural templates."""
    conn = get_db()
    analyze_subtitles(conn, limit=limit)
    build_pattern_index(conn)
    conn.close()


@cli.command()
@click.option("--top", default=50, help="Show top N patterns.")
@click.option("--min-count", default=10, help="Minimum occurrence count.")
def patterns(top: int, min_count: int):
    """Show discovered subtitle patterns ranked by frequency."""
    conn = get_db()
    rows = conn.execute(
        "SELECT template, count, example_subtitle FROM patterns "
        "WHERE count >= ? ORDER BY count DESC LIMIT ?",
        (min_count, top),
    ).fetchall()
    if not rows:
        click.echo("No patterns found. Run 'analyze' first.")
        return
    click.echo(f"Top {len(rows)} patterns (min count: {min_count}):\n")
    for i, (template, count, example) in enumerate(rows, 1):
        click.echo(f"{i:3d}. [{count:,}x] {template}")
        click.echo(f"     e.g. \"{example}\"")
        click.echo()
    conn.close()


@cli.command("build-slots")
@click.option("--loose", is_flag=True, help="Also mine the full corpus for expanded slots.")
def build_slots_cmd(loose: bool):
    """Extract slot fillers from matched subtitles (regex + NLP validated)."""
    conn = get_db()
    build_slots(conn)
    if loose:
        build_loose_slots(conn)
    conn.close()


@cli.command()
@click.option("--count", "-n", default=None, type=int, help="Number of subtitles to generate (default: 10, or 1 with --jacket).")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--loose", is_flag=True, help="Use expanded slot fillers from full corpus.")
@click.option("--jacket", is_flag=True, help="Generate full book jacket (title, back cover, reviews, blurbs).")
@click.option("--sources", is_flag=True, help="Show which real books each slot filler came from.")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in jacket output.")
@click.option("--deep-research", is_flag=True, help="Two-phase generation: dedicated web search for concept research before jacket.")
@click.option("--tone", type=click.Choice(["pop", "mainstream", "niche"]), default=None, help="Filter subtitles by accessibility tier, or override jacket tone.")
def generate(count: int | None, seed: int | None, loose: bool, jacket: bool, sources: bool, model: str | None, show_concept: bool, deep_research: bool, tone: str | None):
    """Generate bizarre subtitles — slot machine style!"""
    if count is None:
        count = 1 if jacket else 10

    conn = get_db()
    mode = "loose" if loose else "strict"
    stats = slot_stats(conn, mode=mode)
    if not stats:
        click.echo("No slots found. Run 'build-slots' first.")
        return
    click.echo(f"Slot machine loaded ({mode} mode): {stats}")
    if tone:
        click.echo(f"Tone filter: {tone}")
    click.echo()

    generated = 0
    attempts = 0
    while generated < count:
        s = (seed + attempts) if seed is not None else None
        sub = generate_subtitle(conn, seed=s, mode=mode)
        attempts += 1

        # Tone filtering: reject subtitles that don't match the requested tier
        if tone:
            _, score = compute_accessibility(sub.text, conn)
            if tone == "pop" and score <= _TONE_THRESHOLDS["pop"]:
                continue
            elif tone == "mainstream" and (score <= _TONE_THRESHOLDS["mainstream"] or score > _TONE_THRESHOLDS["pop"]):
                continue
            elif tone == "niche" and score >= _TONE_THRESHOLDS["mainstream"]:
                continue
            if attempts > _MAX_REJECTION_ATTEMPTS and generated == 0:
                click.echo(f"⚠ Could not find a '{tone}' subtitle after {attempts} attempts.")
                break

        generated += 1

        if jacket:
            click.echo(f"Generating jacket for: {sub.text}\n")
            kwargs = {"model": model} if model else {}
            tone_override = _TONE_CHOICES[tone] if tone else None
            md = generate_jacket(sub.text, show_concept=show_concept, deep_research=deep_research, conn=conn, tone_override=tone_override, **kwargs)
            click.echo(md)
            if sources:
                click.echo(format_sources(conn, sub))
            if generated < count:
                click.echo("\n" + "=" * 72 + "\n")
        else:
            click.echo(f"  {generated:2d}. {sub.text}")
            if sources:
                click.echo(format_sources(conn, sub))
                click.echo()
    conn.close()


@cli.command()
@click.argument("subtitle", required=False, default=None)
@click.option("--loose", is_flag=True, help="Use expanded slot fillers (only for random generation).")
@click.option("--seed", default=None, type=int, help="Random seed (only for random generation).")
@click.option("--sources", is_flag=True, help="Show source books for each slot filler (only for random generation).")
@click.option("--model", default=None, help="LLM model for jacket generation (default: gpt-5.4-mini).")
@click.option("--show-concept", is_flag=True, help="Include the internal concept section in output.")
@click.option("--deep-research", is_flag=True, help="Two-phase generation: dedicated web search for concept research before jacket.")
@click.option("--tone", type=click.Choice(["pop", "mainstream", "niche"]), default=None, help="Override auto-detected tone tier for jacket generation.")
def jacket(subtitle: str | None, loose: bool, seed: int | None, sources: bool, model: str | None, show_concept: bool, deep_research: bool, tone: str | None):
    """Generate a full book jacket — title, back cover, reviews, and blurbs.

    Pass a subtitle string to jacket a specific text, or omit to generate a random one.

    \b
    Examples:
      subtitle-gen jacket "sturgeon, caviar, and the geography of desire"
      subtitle-gen jacket                    # random subtitle
      subtitle-gen jacket --loose --sources  # random from loose pool + show sources
      subtitle-gen jacket --model claude-haiku-4.5  # use a different model
    """
    kwargs = {"model": model} if model else {}
    tone_override = _TONE_CHOICES[tone] if tone else None
    conn = get_db()
    if subtitle:
        click.echo(f"Generating jacket for: {subtitle}\n")
        md = generate_jacket(subtitle, show_concept=show_concept, deep_research=deep_research, conn=conn, tone_override=tone_override, **kwargs)
        click.echo(md)
    else:
        mode = "loose" if loose else "strict"
        stats = slot_stats(conn, mode=mode)
        if not stats:
            click.echo("No slots found. Run 'build-slots' first.")
            conn.close()
            return
        click.echo(f"Slot machine loaded ({mode} mode): {stats}\n")
        sub = generate_subtitle(conn, seed=seed, mode=mode)
        click.echo(f"Generating jacket for: {sub.text}\n")
        md = generate_jacket(sub.text, show_concept=show_concept, deep_research=deep_research, conn=conn, tone_override=tone_override, **kwargs)
        click.echo(md)
        if sources:
            click.echo(format_sources(conn, sub))
    conn.close()


@cli.command()
@click.option("--slot-type", default=None, help="Filter by slot type.")
@click.option("--sample", default=20, help="Number of fillers to show per type.")
def slots(slot_type: str | None, sample: int):
    """Show available slot fillers."""
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


@cli.command()
@click.option("--iterations", "-i", default=10, help="Max tuning iterations.")
@click.option("--batch-size", "-b", default=50, help="Subtitles per iteration.")
def tune(iterations: int, batch_size: int):
    """Run autoresearch loop to improve loose mode quality (LLM-graded)."""
    conn = get_db()
    run_autoresearch(conn, max_iterations=iterations, batch_size=batch_size)
    conn.close()


if __name__ == "__main__":
    cli()
