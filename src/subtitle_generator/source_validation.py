"""Validation helpers shared by source ingestion and slot extraction."""

from __future__ import annotations

import re

SUBTITLE_PATTERN_RE = re.compile(
    r"^(?P<list_part>.+,\s*.+?)\s*,?\s+and\s+(?P<article>a|an|the)\s+"
    r"(?P<action>.+?)\s+of\s+(?P<object>.+)$",
    re.IGNORECASE,
)
SUBTITLE_PATTERN_SQL_LIKE = (
    "%, % and the % of %",
    "%, % and a % of %",
    "%, % and an % of %",
)
_TITLE_SUFFIX_SPLIT_RE = re.compile(r"\s*(?::|\s+[-\u2013\u2014]\s+)\s*")
_TITLE_SUBTITLE_DELIMITER_RE = re.compile(r":|\s+[-\u2013\u2014]\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalized_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(_WORD_RE.findall(text.lower()))


def looks_like_subtitle_pattern(text: str | None) -> bool:
    """Return True when text structurally matches the generator source pattern."""

    return bool(text and SUBTITLE_PATTERN_RE.match(text.strip()))


def _title_parts(title: str) -> list[str]:
    parts = [part.strip() for part in _TITLE_SUFFIX_SPLIT_RE.split(title) if part.strip()]
    return parts


def _has_immediate_repeated_phrase(text: str) -> bool:
    tokens = _normalized_text(text).split()
    if len(tokens) < 4:
        return False

    for span in range(2, len(tokens) // 2 + 1):
        for start in range(0, len(tokens) - (2 * span) + 1):
            first = tokens[start:start + span]
            second = tokens[start + span:start + (2 * span)]
            if first == second and len(" ".join(first)) >= 8:
                return True
    return False


def _strip_trailing_title(text: str, title: str) -> str | None:
    text_words = _WORD_RE.findall(text.lower())
    title_words = _WORD_RE.findall(title.lower())
    if not title_words or len(text_words) <= len(title_words):
        return None
    if text_words[-len(title_words):] != title_words:
        return None

    matches = list(_WORD_RE.finditer(text.lower()))
    cutoff = matches[-len(title_words)].start()
    return text[:cutoff].strip(" \t\r\n:;,.-\u2013\u2014")


def _repair_embedded_title_subtitle(text: str) -> tuple[str, str] | None:
    first_delimiter = _TITLE_SUBTITLE_DELIMITER_RE.search(text)
    if not first_delimiter:
        return None

    title = text[:first_delimiter.start()].strip()
    rest = text[first_delimiter.end():].strip()
    if not title or not rest:
        return None

    rest_delimiters = list(_TITLE_SUBTITLE_DELIMITER_RE.finditer(rest))
    if rest_delimiters:
        last_delimiter = rest_delimiters[-1]
        repeated_title = rest[:last_delimiter.start()].strip()
        repeated_subtitle = rest[last_delimiter.end():].strip()
        first_subtitle = _strip_trailing_title(repeated_title, title)
        if (
            first_subtitle
            and repeated_subtitle
            and _normalized_text(first_subtitle) == _normalized_text(repeated_subtitle)
        ):
            return title, repeated_subtitle

    for idx, ch in enumerate(rest):
        if not ch.isspace():
            continue
        first = rest[:idx].strip()
        second = rest[idx:].strip()
        if (
            first
            and second
            and first[-1].isalnum()
            and second[0].isalnum()
            and _normalized_text(first) == _normalized_text(second)
            and len(_normalized_text(first).split()) >= 2
        ):
            return title, first

    return None


def clean_title_and_subtitle(
    title: str | None,
    subtitle: str | None,
) -> tuple[str, str] | None:
    """Return repaired (title, subtitle), or None for unrepairable repeated rows."""
    cleaned_title = (title or "").strip()
    cleaned_subtitle = (subtitle or "").strip()
    subtitle_norm = _normalized_text(cleaned_subtitle)
    if not subtitle_norm:
        return cleaned_title, cleaned_subtitle

    subtitle_repair = _repair_embedded_title_subtitle(cleaned_subtitle)
    if subtitle_repair:
        return subtitle_repair

    title_norm = _normalized_text(cleaned_title)
    if title_norm and subtitle_norm == title_norm:
        title_repair = _repair_embedded_title_subtitle(cleaned_title)
        if title_repair:
            return title_repair
        return None

    parts = _title_parts(cleaned_title)
    for idx in range(1, len(parts)):
        suffix = " ".join(parts[idx:])
        if subtitle_norm == _normalized_text(suffix):
            return ": ".join(parts[:idx]), cleaned_subtitle

    if _has_immediate_repeated_phrase(cleaned_subtitle):
        return None

    return cleaned_title, cleaned_subtitle


def clean_title_for_subtitle(title: str | None, subtitle: str | None) -> str | None:
    """Return a repaired title, or None for unrepairable repeated-source rows."""
    cleaned = clean_title_and_subtitle(title, subtitle)
    if cleaned is None:
        return None
    return cleaned[0]


def is_repeated_title_subtitle(title: str | None, subtitle: str | None) -> bool:
    """Return True for source rows that cannot be repaired by title cleanup."""
    return clean_title_and_subtitle(title, subtitle) is None
