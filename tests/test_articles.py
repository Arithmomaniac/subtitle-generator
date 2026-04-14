"""Tests for article stripping, lookup, and heuristic logic.

Run:  uv run python tests/test_articles.py
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subtitle_generator.generate import (
    _majority_article,
    _infer_of_article,
)

passed = 0
failed = 0


def assert_eq(actual, expected, msg):
    global passed, failed
    if actual == expected:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {msg}: expected {expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# _majority_article
# ---------------------------------------------------------------------------

def test_majority_article_basic():
    stats = {"american dream": {"the": 14, "": 2}}
    assert_eq(_majority_article("American Dream", stats, 3), "the",
              "majority 'the' with enough freq")

def test_majority_article_no_article():
    stats = {"community": {"": 8}}
    assert_eq(_majority_article("Community", stats, 3), "",
              "majority '' when no articles in corpus")

def test_majority_article_below_min_freq():
    stats = {"rare thing": {"the": 1}}
    assert_eq(_majority_article("Rare Thing", stats, 3), "",
              "below min_freq returns empty")

def test_majority_article_missing_filler():
    stats = {"something else": {"the": 5}}
    assert_eq(_majority_article("Unknown Filler", stats, 3), "",
              "missing filler returns empty")

def test_majority_article_indefinite():
    stats = {"journey": {"a": 5, "the": 2}}
    assert_eq(_majority_article("Journey", stats, 3), "a",
              "majority 'a' when more frequent")

def test_majority_article_tie_returns_empty():
    stats = {"world": {"the": 3, "": 3}}
    assert_eq(_majority_article("World", stats, 3), "",
              "50/50 tie returns empty (no clear majority)")


# ---------------------------------------------------------------------------
# _infer_of_article (remix heuristic)
# ---------------------------------------------------------------------------

def test_infer_exact_match():
    stats = {"modern frontier": {"the": 10, "": 2}}
    assert_eq(_infer_of_article("Modern Frontier", stats, 3, 0.6), "the",
              "exact match returns majority")

def test_infer_head_noun_backoff():
    stats = {"frontier": {"the": 8, "": 1}}
    assert_eq(_infer_of_article("Ancient Frontier", stats, 3, 0.6,
              remix_parts={"modifier": "Ancient", "head": "Frontier"}), "the",
              "type1 head backoff finds 'frontier'")

def test_infer_type2_uses_topic():
    stats = {"jews": {"the": 0, "": 5}, "america": {"the": 8, "": 1}}
    # Type 2: topic is "Jews", complement is "America"
    # Should use topic ("jews") not complement ("america")
    assert_eq(_infer_of_article("Jews in America", stats, 3, 0.6,
              remix_parts={"topic": "Jews", "prep": "in", "complement": "America"}), "",
              "type2 uses topic (Jews→no article), not complement (America→the)")

def test_infer_below_threshold():
    stats = {"world": {"the": 3, "": 3}}
    assert_eq(_infer_of_article("Modern World", stats, 3, 0.6), "",
              "50/50 split below 0.6 threshold → no article")

def test_infer_default_no_data():
    stats = {}
    assert_eq(_infer_of_article("Completely Unknown", stats, 3, 0.6), "",
              "no data → no article")

def test_infer_below_min_freq():
    stats = {"thing": {"the": 2}}
    assert_eq(_infer_of_article("New Thing", stats, 3, 0.6), "",
              "below min_freq → no article even with head match")

def test_infer_deterministic():
    stats = {"dream": {"the": 10, "": 1}}
    r1 = _infer_of_article("American Dream", stats, 3, 0.6)
    r2 = _infer_of_article("American Dream", stats, 3, 0.6)
    assert_eq(r1, r2, "same input always produces same output")


# ---------------------------------------------------------------------------
# Article stripping in extraction (regex-based)
# ---------------------------------------------------------------------------

import re

_ARTICLE_RE = re.compile(r"^(a|an|the)\s+", re.IGNORECASE)

def _strip_article(text):
    m = _ARTICLE_RE.match(text)
    if m:
        return text[m.end():], m.group(1).lower()
    return text, ""

def test_strip_the():
    obj, art = _strip_article("the American Dream")
    assert_eq(obj, "American Dream", "strip 'the' from obj")
    assert_eq(art, "the", "article is 'the'")

def test_strip_a():
    obj, art = _strip_article("a Revolution")
    assert_eq(obj, "Revolution", "strip 'a' from obj")
    assert_eq(art, "a", "article is 'a'")

def test_strip_an():
    obj, art = _strip_article("an Empire")
    assert_eq(obj, "Empire", "strip 'an' from obj")
    assert_eq(art, "an", "article is 'an'")

def test_strip_none():
    obj, art = _strip_article("Modern Science")
    assert_eq(obj, "Modern Science", "no article to strip")
    assert_eq(art, "", "article is empty")

def test_strip_case_insensitive():
    obj, art = _strip_article("The Nuclear Revolution")
    assert_eq(obj, "Nuclear Revolution", "strip 'The' (uppercase)")
    assert_eq(art, "the", "article lowercased")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    test_fns = [obj for name, obj in globals().items()
                if name.startswith("test_") and callable(obj)]
    for fn in test_fns:
        fn()
    print(f"\n{passed} passed, {failed} failed ({passed + failed} total)")
    sys.exit(1 if failed else 0)
