"""CLI entry point for subtitle-generator."""

from pathlib import Path

import click

from subtitle_generator.download import TOTAL_PARTS, download_part, parse_parts_arg
from subtitle_generator.extract import DATA_DIR, DB_PATH, extract_from_file, get_db


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


if __name__ == "__main__":
    cli()
