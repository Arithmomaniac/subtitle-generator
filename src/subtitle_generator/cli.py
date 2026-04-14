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
from subtitle_generator.eval_harness import DEFAULT_RATER_MODEL
from subtitle_generator.export_db import build_mini_db, export_data, export_mini_db
from subtitle_generator.generate import TONE_TARGETS, format_sources, generate_subtitle, precompute_remix_data, slot_stats
from subtitle_generator.jacket import (
    TONE_HIGH, TONE_LOW, TONE_MEDIUM,
    compute_accessibility, generate_jacket, sample_tone,
)
from subtitle_generator.slots import build_slots, ensure_slot_tables


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
def generate(count: int | None, seed: int | None, jacket: bool, sources: bool, model: str | None, show_concept: bool, tone: str | None, remix: bool, remix_prob: float | None, min_sim: float | None):
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
            md = generate_jacket(sub.text, show_concept=show_concept, conn=conn, allowed_tiers=tone_set, **kwargs)
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
        md = generate_jacket(subtitle, show_concept=show_concept, conn=conn, allowed_tiers=tone_set, **kwargs)
        click.echo(md)
    else:
        stats = slot_stats(conn)
        if not stats:
            conn.close()
            raise click.ClickException("No slots found. Run 'subtitle-gen build-slots' first.")
        click.echo(f"Slot machine loaded: {stats}\n")
        sub = generate_subtitle(conn, seed=seed)
        click.echo(f"Generating jacket for: {sub.text}\n")
        md = generate_jacket(sub.text, show_concept=show_concept, conn=conn, allowed_tiers=tone_set, **kwargs)
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
@click.option("--model", default=DEFAULT_RATER_MODEL, help=f"LLM model for rating (default: {DEFAULT_RATER_MODEL}).")
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


if __name__ == "__main__":
    cli()
