import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _write_csv(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def _create_full_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            isbn TEXT,
            candidate_source TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE pattern_matches (
            id INTEGER PRIMARY KEY,
            subtitle_id INTEGER,
            title TEXT,
            subtitle TEXT,
            candidate_source TEXT,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL,
            llm_market_tier_rationale TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT,
            filler TEXT,
            mode TEXT,
            freq INTEGER,
            popularity_score REAL,
            popularity_level INTEGER,
            popularity_confidence REAL
        )
        """
    )
    conn.execute(
        "CREATE TABLE slot_filler_sources (slot_filler_id INTEGER, subtitle_id INTEGER)"
    )
    conn.execute(
        "CREATE TABLE popularity_data (work_key TEXT PRIMARY KEY, composite_score REAL)"
    )
    conn.execute("CREATE TABLE isbn_aliases (isbn TEXT PRIMARY KEY, work_key TEXT)")
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "CREATE TABLE human_ratings (id INTEGER PRIMARY KEY, subtitle TEXT)"
    )
    conn.executemany(
        """
        INSERT INTO pattern_matches (
            id, subtitle_id, title, subtitle, candidate_source,
            llm_market_tier, llm_market_tier_confidence, llm_market_tier_rationale
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (1, 101, "Book A", "Race, Power, and the Rise of Markets", "subtitle", "pop", 0.9, "Known broad title."),
            (2, 102, "Book B", "", "title", "niche", 0.8, "Specialist title."),
            (3, 103, "Book C", "Memory, Justice, and the Politics of Archives", "subtitle", None, None, None),
        ],
    )
    conn.executemany(
        "INSERT INTO slot_fillers VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "list_item", "Race", "strict", 10, 1.2, 2, 1.0),
            (2, "action_noun", "Rise", "strict", 8, 0.9, 1, 0.8),
        ],
    )
    conn.commit()
    conn.close()


def _create_mini_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT,
            filler TEXT,
            mode TEXT,
            source_subtitle_id INTEGER,
            freq INTEGER,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER,
            popularity_confidence REAL
        )
        """
    )
    conn.execute(
        "CREATE TABLE sources (slot_filler_id INTEGER, title TEXT, subtitle_text TEXT, source_tag TEXT)"
    )
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    conn.execute(
        "INSERT INTO sources VALUES (1, 'Book A', 'Race, Power, and the Rise of Markets', 'LOC')"
    )
    conn.commit()
    conn.close()


def test_book_model_inventory_formats_exportability_and_missing_artifacts(tmp_path):
    from subtitle_generator.book_model_inventory import (
        build_inventory,
        format_inventory_markdown,
    )

    full_db = tmp_path / "full.db"
    mini_db = tmp_path / "mini.db"
    api_db = tmp_path / "api.db"
    export_dir = tmp_path / "api-data"
    _create_full_db(full_db)
    _create_mini_db(mini_db)
    _create_mini_db(api_db)
    _write_csv(
        export_dir / "slot_fillers.csv",
        ["id", "slot_type", "filler", "popularity_score"],
        [["1", "list_item", "Race", "1.2"]],
    )
    _write_csv(
        export_dir / "sources.csv",
        ["slot_filler_id", "title", "subtitle_text", "source_tag"],
        [["1", "Book A", "Race, Power, and the Rise of Markets", "LOC"]],
    )

    report = format_inventory_markdown(build_inventory(
        full_db=full_db,
        mini_db=mini_db,
        api_db=api_db,
        export_dir=export_dir,
    ))

    assert "LLM market tiers: NULL=1, niche=1, pop=1" in report
    assert "Pattern candidate source: subtitle=2, title=1" in report
    assert "`popularity_score`" in report
    assert "`book_features`" in report
    assert "D8 should stay unresolved" in report
