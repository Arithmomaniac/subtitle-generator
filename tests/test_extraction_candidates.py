from __future__ import annotations

import gzip
import json
from pathlib import Path

from pymarc import Field, Indicators, MARCWriter, Record, Subfield

from subtitle_generator.extract import extract_from_file, get_db
from subtitle_generator.extract_openlibrary import extract_from_ol_dump


def _write_marc(path: Path, records: list[Record]) -> None:
    with path.open("wb") as handle:
        writer = MARCWriter(handle)
        for record in records:
            writer.write(record)
        writer.close()


def _record_245a(title: str) -> Record:
    record = Record()
    record.add_field(Field(tag="008", data=" " * 35 + "eng" + " " * 5))
    record.add_field(
        Field(
            tag="245",
            indicators=Indicators("1", "0"),
            subfields=[Subfield("a", title)],
        )
    )
    record.add_field(
        Field(
            tag="010",
            indicators=Indicators(" ", " "),
            subfields=[Subfield("a", "2026000001")],
        )
    )
    return record


def _add_isbn(record: Record, value: str) -> Record:
    record.add_field(
        Field(
            tag="020",
            indicators=Indicators(" ", " "),
            subfields=[Subfield("a", value)],
        )
    )
    return record


def test_loc_extract_admits_title_pattern_candidate(tmp_path: Path):
    db_path = tmp_path / "subtitles.db"
    mrc_path = tmp_path / "loc.mrc"
    title = "Race, Power, and the Rise of Empire"
    _write_marc(mrc_path, [_record_245a(title)])
    conn = get_db(db_path)

    scanned, found = extract_from_file(mrc_path, conn)

    assert (scanned, found) == (1, 1)
    row = conn.execute(
        "SELECT title, subtitle, candidate_text, candidate_source FROM subtitles"
    ).fetchone()
    assert row == (title, "", title, "title")


def test_loc_extract_stores_isbn_from_marc_020_with_qualifier(tmp_path: Path):
    db_path = tmp_path / "subtitles.db"
    mrc_path = tmp_path / "loc.mrc"
    title = "Race, Power, and the Rise of Empire"
    _write_marc(
        mrc_path,
        [_add_isbn(_record_245a(title), "0801864208 (hardcover : alk. paper)")],
    )
    conn = get_db(db_path)

    scanned, found = extract_from_file(mrc_path, conn)

    assert (scanned, found) == (1, 1)
    row = conn.execute(
        "SELECT title, subtitle, isbn, candidate_text, candidate_source FROM subtitles"
    ).fetchone()
    assert row == (title, "", "0801864208", title, "title")


def test_loc_extract_stores_isbn13_from_marc_020_with_qualifier(tmp_path: Path):
    db_path = tmp_path / "subtitles.db"
    mrc_path = tmp_path / "loc.mrc"
    title = "Race, Power, and the Rise of Empire"
    _write_marc(
        mrc_path,
        [_add_isbn(_record_245a(title), "9780674430006 (cloth)")],
    )
    conn = get_db(db_path)

    scanned, found = extract_from_file(mrc_path, conn)

    assert (scanned, found) == (1, 1)
    row = conn.execute(
        "SELECT title, subtitle, isbn, candidate_text, candidate_source FROM subtitles"
    ).fetchone()
    assert row == (title, "", "9780674430006", title, "title")


def test_loc_extract_skips_non_pattern_title_without_subtitle(tmp_path: Path):
    db_path = tmp_path / "subtitles.db"
    mrc_path = tmp_path / "loc.mrc"
    _write_marc(mrc_path, [_record_245a("Plain Book Title")])
    conn = get_db(db_path)

    scanned, found = extract_from_file(mrc_path, conn)

    assert (scanned, found) == (1, 0)
    assert conn.execute("SELECT COUNT(*) FROM subtitles").fetchone()[0] == 0


def test_openlibrary_extract_admits_title_pattern_candidate(tmp_path: Path):
    db_path = tmp_path / "subtitles.db"
    dump_path = tmp_path / "ol_dump.txt.gz"
    title = "Race, Power, and the Rise of Empire"
    edition = {
        "title": title,
        "languages": [{"key": "/languages/eng"}],
        "isbn_13": ["9781566199094"],
        "works": [{"key": "/works/OL1W"}],
    }
    with gzip.open(dump_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "/type/edition\t/books/OL1M\t1\t2026-05-04T00:00:00Z\t"
            + json.dumps(edition)
            + "\n"
        )
    conn = get_db(db_path)

    scanned, found, duplicates = extract_from_ol_dump(
        conn,
        dump_path=dump_path,
        dedup=False,
    )

    assert (scanned, found, duplicates) == (1, 1, 0)
    row = conn.execute(
        "SELECT title, subtitle, isbn, candidate_text, candidate_source FROM subtitles"
    ).fetchone()
    assert row == (title, "", "9781566199094", title, "title")
