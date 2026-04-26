"""Tests for jacket prompt and validation formatting.

Run:  uv run python tests/test_jacket.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subtitle_generator.jacket import (  # noqa: E402
    JACKET_PROMPT,
    _strip_internal_concept,
    _validate_jacket,
    build_jacket_prompt,
)


def _valid_inline_jacket() -> str:
    return """## Title
Bright Ruins

## Subtitle
Faith, Commerce, and the Making of Modern Appetite

## Internal Concept
This is the private planning section.

## Back Cover
What do our hungers reveal about the lives we think we are choosing? This book follows
the marketplaces, rituals, and private compromises that turn desire into destiny.

"A sharp and unsettling map of the bargains hiding inside ordinary appetite." — Ross Douthat, New York Times columnist

"This bracing book sees what so much of public life tries to hide." — Rod Dreher, author of The Benedict Option

## Review 1
**Publishers Weekly**
A polished review.

## Review 2
**Kirkus Reviews**
A pointed review.
"""


def test_prompt_places_blurbs_inside_back_cover():
    assert "## Blurb 1" not in JACKET_PROMPT
    assert "## Blurb 2" not in JACKET_PROMPT

    system_prompt, user_prompt, _ = build_jacket_prompt(
        "Faith, Commerce, and the Making of Modern Appetite"
    )
    assert "include exactly TWO endorsement blurbs" in system_prompt
    assert '"[A single compelling endorsement sentence' in system_prompt
    assert "Do NOT create separate blurb headers" in system_prompt
    assert "## Review 1" in system_prompt
    assert "The subtitle is:" in user_prompt


def test_validate_accepts_inline_back_cover_blurbs():
    assert _validate_jacket(_valid_inline_jacket()) == []


def test_validate_rejects_separate_blurb_sections():
    content = _valid_inline_jacket() + """
## Blurb 1
**Someone** (credential)
> "A standalone blurb."
"""
    assert "remove separate ## Blurb sections" in _validate_jacket(content)


def test_validate_requires_two_inline_blurbs_in_back_cover():
    content = _valid_inline_jacket().replace(
        '"This bracing book sees what so much of public life tries to hide." — Rod Dreher, author of The Benedict Option',
        "",
    )
    assert "exactly two inline Back Cover blurbs" in _validate_jacket(content)


def test_validate_rejects_extra_inline_blurbs():
    content = _valid_inline_jacket().replace(
        "## Review 1",
        '"A third blurb is one too many." — Jane Critic, critic at Large\n\n## Review 1',
    )
    assert "exactly two inline Back Cover blurbs" in _validate_jacket(content)


def test_validate_requires_description_before_inline_blurbs():
    content = _valid_inline_jacket().replace(
        """What do our hungers reveal about the lives we think we are choosing? This book follows
the marketplaces, rituals, and private compromises that turn desire into destiny.

""",
        "",
    )
    assert "Back Cover description before inline blurbs" in _validate_jacket(content)


def test_strip_internal_concept_preserves_inline_blurbs():
    stripped = _strip_internal_concept(_valid_inline_jacket())
    assert "## Internal Concept" not in stripped
    assert "## Back Cover" in stripped
    assert "Ross Douthat" in stripped
    assert "Rod Dreher" in stripped
    assert "## Review 1" in stripped


if __name__ == "__main__":
    tests = [
        test_prompt_places_blurbs_inside_back_cover,
        test_validate_accepts_inline_back_cover_blurbs,
        test_validate_rejects_separate_blurb_sections,
        test_validate_requires_two_inline_blurbs_in_back_cover,
        test_validate_rejects_extra_inline_blurbs,
        test_validate_requires_description_before_inline_blurbs,
        test_strip_internal_concept_preserves_inline_blurbs,
    ]
    for test in tests:
        test()
        print(f"  PASS: {test.__name__}")
    print("All jacket tests passed.")
