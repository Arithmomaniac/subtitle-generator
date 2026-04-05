"""CLI entry point for subtitle-generator."""

from pathlib import Path

import click

from subtitle_generator.analyze import analyze_subtitles, build_pattern_index
from subtitle_generator.download import TOTAL_PARTS, download_part, parse_parts_arg
from subtitle_generator.extract import DATA_DIR, DB_PATH, extract_from_file, get_db
from subtitle_generator.generate import generate_subtitle, slot_stats
from subtitle_generator.slots import build_loose_slots, build_slots
from subtitle_generator.tune import run_autoresearch


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
@click.option("--count", "-n", default=10, help="Number of subtitles to generate.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--loose", is_flag=True, help="Use expanded slot fillers from full corpus.")
def generate(count: int, seed: int | None, loose: bool):
    """Generate bizarre subtitles — slot machine style!"""
    conn = get_db()
    mode = "loose" if loose else "strict"
    stats = slot_stats(conn, mode=mode)
    if not stats:
        click.echo("No slots found. Run 'build-slots' first.")
        return
    click.echo(f"Slot machine loaded ({mode} mode): {stats}\n")
    for i in range(count):
        s = seed + i if seed is not None else None
        subtitle = generate_subtitle(conn, seed=s, mode=mode)
        click.echo(f"  {i + 1:2d}. {subtitle}")
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
