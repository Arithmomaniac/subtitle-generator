import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _create_artifact_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            lang TEXT,
            lccn TEXT,
            source_file TEXT,
            isbn TEXT,
            candidate_text TEXT,
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
            list_items_json TEXT,
            action_noun TEXT,
            of_object TEXT,
            candidate_source TEXT,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL,
            llm_market_tier_rationale TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE isbn_aliases (
            isbn TEXT PRIMARY KEY,
            canonical_isbn TEXT,
            work_key TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE popularity_data (
            work_key TEXT PRIMARY KEY,
            composite_score REAL
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
            popularity_score REAL
        )
        """
    )
    conn.execute(
        "CREATE TABLE slot_filler_sources (slot_filler_id INTEGER, subtitle_id INTEGER)"
    )
    conn.execute(
        """
        INSERT INTO subtitles VALUES (
            101, 'Book A', 'Race, Power, and the Rise of Markets', 'eng',
            'lccn-a', 'openlibrary', '9780000000001',
            'Race, Power, and the Rise of Markets', 'subtitle'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pattern_matches VALUES (
            1, 101, 'Book A', 'Race, Power, and the Rise of Markets',
            '["Race", "Power"]', 'Rise', 'Markets', 'subtitle',
            'pop', 0.9, 'Broad title.'
        )
        """
    )
    conn.execute(
        "INSERT INTO isbn_aliases VALUES ('9780000000001', '9780000000001', 'OLW1')"
    )
    conn.execute("INSERT INTO popularity_data VALUES ('OLW1', 1.7)")
    conn.executemany(
        "INSERT INTO slot_fillers VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "list_item", "Race", "strict", 10, 1.2),
            (2, "action_noun", "Rise", "strict", 8, 0.8),
        ],
    )
    conn.executemany(
        "INSERT INTO slot_filler_sources VALUES (?, ?)",
        [(1, 101), (2, 101)],
    )
    conn.commit()
    return conn


def test_build_book_model_artifacts_writes_features_labels_and_report(tmp_path):
    from subtitle_generator.book_model_artifacts import build_book_model_artifacts
    from subtitle_generator.book_metadata_enrichment import METADATA_COLUMNS

    conn = _create_artifact_db()
    metadata_path = tmp_path / "book_metadata.csv"
    with open(metadata_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METADATA_COLUMNS)
        writer.writeheader()
        writer.writerow({
            "subtitle_id": "101",
            "metadata_source": "openlibrary_dump",
            "work_key": "/works/OLW1",
            "edition_key": "/books/OL1M",
            "publisher_text": "Trade Books",
            "publisher_count": "1",
            "publish_year": "2024",
            "edition_name": "Hardcover edition",
            "physical_format": "Hardcover",
            "physical_dimensions": "",
            "physical_description": "",
            "number_of_pages": "320",
            "is_hardcover": "1",
            "is_paperback": "0",
            "is_ebook": "0",
            "is_large_print": "0",
            "subject_text": "Markets | Politics",
            "subject_count": "2",
            "author_count": "1",
            "loc_call_number": "",
            "dewey_decimal": "",
            "has_physical_format_metadata": "1",
            "has_publisher_metadata": "1",
            "has_subject_metadata": "1",
        })

    result = build_book_model_artifacts(conn, tmp_path, metadata_path=metadata_path)

    with open(result.features_path, encoding="utf-8") as handle:
        features = list(csv.DictReader(handle))
    with open(result.labels_path, encoding="utf-8") as handle:
        labels = list(csv.DictReader(handle))
    report = result.report_path.read_text(encoding="utf-8")

    assert result.feature_count == 1
    assert features[0]["pattern_match_id"] == "1"
    assert features[0]["candidate_source"] == "subtitle"
    assert features[0]["source_group"] == "OL"
    assert features[0]["has_work_key"] == "1"
    assert features[0]["slot_source_link_count"] == "2"
    assert features[0]["list_item_count"] == "2"
    assert features[0]["list_item_pair_text"] == "Race || Power"
    assert features[0]["action_object_pair_text"] == "Rise || Markets"
    assert features[0]["metadata_physical_format"] == "Hardcover"
    assert features[0]["metadata_is_hardcover"] == "1"
    assert features[0]["has_physical_format_metadata"] == "1"
    assert labels[0]["label_target"] == "pop"
    assert labels[0]["label_source"] == "llm_market_tier"
    assert labels[0]["popularity_comparison_score"] == "1.7"
    assert "human_ratings" in report


def test_load_book_feature_rows_without_optional_enrichment_tables():
    from subtitle_generator.book_model_artifacts import load_book_feature_rows

    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY,
            title TEXT,
            subtitle TEXT,
            lang TEXT,
            lccn TEXT,
            source_file TEXT,
            isbn TEXT,
            candidate_text TEXT,
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
            list_items_json TEXT,
            action_noun TEXT,
            of_object TEXT,
            candidate_source TEXT,
            llm_market_tier TEXT,
            llm_market_tier_confidence REAL,
            llm_market_tier_rationale TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO subtitles VALUES (
            101, 'Book A', 'Race, Power, and the Rise of Markets', 'eng',
            'lccn-a', 'loc', '9780000000001',
            'Race, Power, and the Rise of Markets', 'subtitle'
        )
        """
    )
    conn.execute(
        """
        INSERT INTO pattern_matches VALUES (
            1, 101, 'Book A', 'Race, Power, and the Rise of Markets',
            '["Race", "Power"]', 'Rise', 'Markets', 'subtitle',
            'mainstream', 0.8, 'Readable.'
        )
        """
    )

    rows = load_book_feature_rows(conn)

    assert len(rows) == 1
    assert rows[0].has_work_key == 0
    assert rows[0].has_popularity_data == 0
    assert rows[0].work_key == ""
    assert rows[0].work_popularity_score is None
