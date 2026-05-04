"""Tests for Trove Australia holdings normalization."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest


def _load_trove_stream_module():
    module_path = Path(__file__).parent.parent / "data" / "trove_stream.py"
    spec = importlib.util.spec_from_file_location("trove_stream", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_populate_popularity_module():
    module_path = Path(__file__).parent.parent / "data" / "populate_popularity.py"
    spec = importlib.util.spec_from_file_location("populate_popularity", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalize_work_uses_holdings_count_proxy():
    trove = _load_trove_stream_module()
    work = {
        "id": "6311198",
        "title": "A short history of Australia / Manning Clark",
        "contributor": ["Clark, Manning"],
        "identifier": [{"type": "ISBN", "value": "0333337352"}],
        "holdingsCount": 179,
        "versionCount": 115,
        "holding": [
            {"nuc": "TSL", "url": {"type": "deepLink", "value": "https://example.test/tsl"}},
            {"nuc": "ANL", "url": {"type": "deepLink", "value": "https://example.test/anl"}},
            {"nuc": "TSL"},
        ],
        "version": [{"id": "v1", "holdingsCount": 34}],
    }

    lookup = trove.normalize_work(work, include_libraries=True, checked_at="2026-05-02T00:00:00Z")

    assert "0333337352" in lookup
    assert "9780333337356" in lookup
    entry = lookup["0333337352"]
    assert entry["trove_work_id"] == "6311198"
    assert entry["library_count"] == 179
    assert entry["holding_count"] == 179
    assert entry["copy_count"] == 179
    assert entry["copy_count_is_exact"] is False
    assert entry["copy_count_basis"] == "holdings_count_proxy"
    assert entry["libraries"] == ["ANL", "TSL"]
    assert entry["version_holding_count"] == 34


def test_normalize_work_recursively_extracts_record_isbns():
    trove = _load_trove_stream_module()
    work = {
        "id": "10000002",
        "title": "Nested ISBN example",
        "holdingsCount": "3",
        "version": [
            {
                "record": [
                    {
                        "metadata": {
                            "dc": {
                                "identifier": [
                                    {"type": "ISBN", "value": "978-1-56619-909-4"},
                                    {"type": "control number", "value": "6736788"},
                                ]
                            }
                        }
                    }
                ]
            }
        ],
    }

    lookup = trove.normalize_work(work)

    assert list(lookup) == ["9781566199094"]
    assert lookup["9781566199094"]["library_count"] == 3


def test_load_trove_maps_isbn_to_work_key():
    pop = _load_populate_popularity_module()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE isbn_aliases (isbn TEXT, work_key TEXT)")
    conn.executemany(
        "INSERT INTO isbn_aliases VALUES (?, ?)",
        [
            ("9780333337359", "work-a"),
            ("9781566199094", "work-b"),
        ],
    )

    work_trove = pop.load_trove(
        conn,
        {
            "9780333337359": {
                "library_count": 179,
                "holding_count": 179,
                "copy_count": 179,
                "copy_count_is_exact": False,
            },
            "9781566199094": {
                "library_count": 3,
                "holding_count": 3,
                "copy_count": 3,
                "copy_count_is_exact": False,
            },
        },
    )

    assert work_trove["work-a"]["library_count"] == 179
    assert work_trove["work-b"]["holding_count"] == 3


def test_download_by_isbn_propagates_quota_without_marking_processed(tmp_path):
    trove = _load_trove_stream_module()

    class QuotaClient:
        def search(self, *_args, **_kwargs):
            raise trove.TroveQuotaExceeded("Reached --max-requests=1")

    checkpoint: dict = {}

    with pytest.raises(trove.TroveQuotaExceeded):
        trove.download_by_isbn(
            QuotaClient(),
            {"9780333337356"},
            {},
            checkpoint,
            limit=None,
            refresh_days=30,
            fetch_details=False,
            include_libraries=False,
            save_every=25,
            workers=1,
            output_path=tmp_path / "lookup.json",
            checkpoint_path=tmp_path / "checkpoint.json",
        )

    assert checkpoint["processed_isbns"] == []
    assert checkpoint["failed_isbns"] == {}


def test_download_by_isbn_does_not_mark_transient_failure_processed(tmp_path):
    trove = _load_trove_stream_module()

    class FailingClient:
        def search(self, *_args, **_kwargs):
            raise RuntimeError("HTTP 503 after retries")

    checkpoint: dict = {}

    trove.download_by_isbn(
        FailingClient(),
        {"9780333337356"},
        {},
        checkpoint,
        limit=None,
        refresh_days=30,
        fetch_details=False,
        include_libraries=False,
        save_every=25,
        workers=1,
        output_path=tmp_path / "lookup.json",
        checkpoint_path=tmp_path / "checkpoint.json",
    )

    assert checkpoint["processed_isbns"] == []
    assert checkpoint["failed_isbns"] == {"9780333337356": "HTTP 503 after retries"}


def test_download_by_isbn_final_checkpoint_includes_last_partial_batch(tmp_path):
    trove = _load_trove_stream_module()

    class NotFoundClient:
        def search(self, *_args, **_kwargs):
            return []

    checkpoint: dict = {}
    checkpoint_path = tmp_path / "checkpoint.json"

    trove.download_by_isbn(
        NotFoundClient(),
        {"9780333337356"},
        {},
        checkpoint,
        limit=None,
        refresh_days=30,
        fetch_details=False,
        include_libraries=False,
        save_every=25,
        workers=1,
        output_path=tmp_path / "lookup.json",
        checkpoint_path=checkpoint_path,
    )

    saved_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert saved_checkpoint["processed_isbns"] == ["9780333337356"]
    assert saved_checkpoint["failed_isbns"] == {}
