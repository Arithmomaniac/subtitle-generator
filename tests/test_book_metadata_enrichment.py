from __future__ import annotations

import csv
import gzip
import json
import sqlite3
from pathlib import Path

from pymarc import Field, Indicators, MARCWriter, Record, Subfield


def _create_metadata_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            source_file TEXT,
            isbn TEXT,
            lccn TEXT
        )
        """
    )
    conn.execute("CREATE TABLE pattern_matches (id INTEGER PRIMARY KEY, subtitle_id INTEGER)")
    conn.executemany(
        "INSERT INTO subtitles VALUES (?, ?, ?, ?)",
        [
            (101, "openlibrary", "9781566199094", ""),
            (202, "BooksAll.2016.part01.utf8.mrc", "", "2026000001"),
        ],
    )
    conn.executemany(
        "INSERT INTO pattern_matches VALUES (?, ?)",
        [(1, 101), (2, 202)],
    )
    conn.commit()
    return conn


def _write_ol_dump(path: Path) -> None:
    edition = {
        "isbn_13": ["9781566199094"],
        "works": [{"key": "/works/OL1W"}],
        "publishers": ["Trade Books"],
        "publish_date": "2024",
        "edition_name": "First hardcover edition",
        "physical_format": "Hardcover",
        "number_of_pages": 320,
        "subjects": ["Markets", "Politics"],
        "authors": [{"key": "/authors/OL1A"}],
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "/type/edition\t/books/OL1M\t1\t2026-05-04T00:00:00Z\t"
            + json.dumps(edition)
            + "\n"
        )


def _write_loc_marc(path: Path) -> None:
    record = Record()
    record.add_field(Field(tag="008", data=" " * 7 + "2023" + " " * 24 + "eng" + " " * 5))
    record.add_field(
        Field(
            tag="010",
            indicators=Indicators(" ", " "),
            subfields=[Subfield("a", "2026000001")],
        )
    )
    record.add_field(
        Field(
            tag="264",
            indicators=Indicators(" ", "1"),
            subfields=[Subfield("b", "Library Press"), Subfield("c", "2023")],
        )
    )
    record.add_field(
        Field(
            tag="300",
            indicators=Indicators(" ", " "),
            subfields=[Subfield("a", "212 pages")],
        )
    )
    record.add_field(
        Field(
            tag="650",
            indicators=Indicators(" ", "0"),
            subfields=[Subfield("a", "Libraries"), Subfield("x", "History")],
        )
    )
    record.add_field(
        Field(
            tag="050",
            indicators=Indicators("0", "0"),
            subfields=[Subfield("a", "Z1000"), Subfield("b", ".L53")],
        )
    )
    with path.open("wb") as handle:
        writer = MARCWriter(handle)
        writer.write(record)
        writer.close()


def test_build_book_metadata_artifact_reads_raw_ol_and_loc_sources(tmp_path: Path):
    from subtitle_generator.book_metadata_enrichment import build_book_metadata_artifact

    conn = _create_metadata_db()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ol_dump = raw_dir / "ol_dump_editions_latest.txt.gz"
    loc_marc = raw_dir / "BooksAll.2016.part01.utf8.mrc"
    _write_ol_dump(ol_dump)
    _write_loc_marc(loc_marc)

    result = build_book_metadata_artifact(
        conn,
        output_dir=tmp_path,
        ol_dump_path=ol_dump,
        loc_raw_dir=raw_dir,
    )

    with open(result.metadata_path, encoding="utf-8") as handle:
        rows = {row["subtitle_id"]: row for row in csv.DictReader(handle)}
    report = result.report_path.read_text(encoding="utf-8")

    assert result.target_count == 2
    assert result.enriched_count == 2
    assert rows["101"]["metadata_source"] == "openlibrary_dump"
    assert rows["101"]["physical_format"] == "Hardcover"
    assert rows["101"]["is_hardcover"] == "1"
    assert rows["101"]["subject_count"] == "2"
    assert rows["202"]["metadata_source"] == "loc_marc"
    assert rows["202"]["publisher_text"] == "Library Press"
    assert rows["202"]["loc_call_number"] == "Z1000 .L53"
    assert "runtime export" in report
