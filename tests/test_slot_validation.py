from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

from subtitle_generator.extract import get_db
from subtitle_generator.slots import (
    _is_valid_object,
    _load_nlp,
    build_slots,
    extract_pattern_matches,
)
from subtitle_generator.source_validation import (
    clean_title_and_subtitle,
    clean_title_for_subtitle,
    is_repeated_title_subtitle,
)


def _make_conn(rows: list[tuple[int, str, str]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT NOT NULL,
            lang TEXT,
            lccn TEXT,
            source_file TEXT,
            isbn TEXT
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO subtitles (id, title, subtitle, lang, lccn, source_file, isbn)
        VALUES (?, ?, ?, 'eng', 'lccn', 'test', 'isbn')
        """,
        rows,
    )
    conn.commit()
    return conn


def test_extract_pattern_matches_allows_two_or_three_list_items_only():
    conn = _make_conn([
        (1, "Two", "Race, Power, and the Rise of Empire"),
        (2, "Three", "Race, Power, History, and the Rise of Empire"),
        (3, "Four", "Race, Power, History, Faith, and the Rise of Empire"),
    ])
    rejections: Counter[str] = Counter()

    matches = extract_pattern_matches(conn, rejections)

    assert [m["subtitle_id"] for m in matches] == [1, 2]
    assert [len(m["list_items"]) for m in matches] == [2, 3]
    assert rejections["rejected_list_count"] == 1


def test_build_slots_rejects_whole_candidate_when_any_list_item_fails():
    conn = _make_conn([
        (1, "Valid Two", "Race, Power, and the Rise of Empire"),
        (2, "Valid Three", "Race, Power, History, and the Rise of Empire"),
        (3, "Invalid Item", "Race, Quickly, and the Rise of Empire"),
    ])

    build_slots(conn)

    list_items = {
        filler: freq
        for filler, freq in conn.execute(
            """
            SELECT filler, freq
            FROM slot_fillers
            WHERE slot_type = 'list_item'
            """
        )
    }
    assert list_items == {"Race": 2, "Power": 2, "History": 1}

    rise_freq = conn.execute(
        """
        SELECT freq
        FROM slot_fillers
        WHERE slot_type = 'action_noun' AND filler = 'Rise'
        """
    ).fetchone()[0]
    assert rise_freq == 2

    assert conn.execute("SELECT COUNT(*) FROM pattern_matches").fetchone()[0] == 2


def test_object_validation_rejects_seo_style_function_word_starts():
    nlp = _load_nlp()

    assert not _is_valid_object("Using Clotrimazole Cream", nlp)
    assert not _is_valid_object("With Clotrimazole Cream", nlp)
    assert not _is_valid_object("For Skin Infection", nlp)
    assert not _is_valid_object("AIDS / Katie Hogan", nlp)
    assert _is_valid_object("Modern Life", nlp)


def test_build_slots_tolerates_loc_only_database_without_openlibrary(tmp_path: Path):
    db_path = tmp_path / "loc-only.db"
    conn = get_db(db_path)
    conn.execute(
        """
        INSERT INTO subtitles (title, subtitle, lang, lccn, source_file)
        VALUES ('LOC Book', 'Race, Power, and the Rise of Empire', 'eng', 'lccn', 'loc.mrc')
        """
    )
    conn.commit()

    build_slots(conn)

    assert conn.execute("SELECT COUNT(*) FROM pattern_matches").fetchone()[0] == 1


def test_repeated_title_subtitle_detection_cases():
    subtitle = "Race, Power, and the Rise of Empire"

    assert is_repeated_title_subtitle(subtitle, subtitle)
    assert is_repeated_title_subtitle("Book", f"{subtitle} {subtitle}")
    assert not is_repeated_title_subtitle("Book Title", subtitle)


def test_clean_title_for_subtitle_repairs_title_suffix_repetition():
    subtitle = "Race, Power, and the Rise of Empire"

    assert clean_title_for_subtitle(f"Book Title: {subtitle}", subtitle) == "Book Title"
    assert clean_title_for_subtitle(f"Book Title - {subtitle}", subtitle) == "Book Title"
    assert clean_title_for_subtitle("Book Title", subtitle) == "Book Title"
    assert clean_title_for_subtitle(subtitle, subtitle) is None


def test_clean_title_and_subtitle_repairs_embedded_repetition():
    title = "Book Title"
    subtitle = "Race, Power, and the Rise of Empire"

    assert clean_title_and_subtitle(
        "Original Title",
        f"{title}: {subtitle} {title}: {subtitle}",
    ) == (title, subtitle)
    assert clean_title_and_subtitle(
        "Original Title",
        f"{title}: {subtitle} {subtitle}",
    ) == (title, subtitle)
    assert clean_title_and_subtitle(
        f"{title}: {subtitle} {title}: {subtitle}",
        f"{title}: {subtitle} {title}: {subtitle}",
    ) == (title, subtitle)


def test_clean_title_and_subtitle_does_not_extract_punctuated_tail_repetition():
    result = clean_title_and_subtitle(
        "Gone with the wind",
        "the book and the film : a commentary : a commentary",
    )

    assert result != ("the book and the film", "a commentary")


def test_extract_pattern_matches_repairs_title_suffix_repetition():
    subtitle = "Race, Power, and the Rise of Empire"
    conn = _make_conn([
        (1, f"Book Title: {subtitle}", subtitle),
        (2, "Clean Book", subtitle),
        (3, subtitle, subtitle),
        (4, "Corrupt", f"Embedded Title: {subtitle} Embedded Title: {subtitle}"),
        (5, "Corrupt", f"Embedded Title: {subtitle} {subtitle}"),
    ])
    rejections: Counter[str] = Counter()

    matches = extract_pattern_matches(conn, rejections)

    assert [m["subtitle_id"] for m in matches] == [1, 2, 4, 5]
    assert matches[0]["title"] == "Book Title"
    assert matches[2]["title"] == "Embedded Title"
    assert matches[2]["subtitle"] == subtitle
    assert matches[3]["title"] == "Embedded Title"
    assert matches[3]["subtitle"] == subtitle
    assert rejections["rejected_repeated_title_subtitle"] == 1
