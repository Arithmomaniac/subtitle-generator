"""Generate full book jackets using the Copilot SDK (LLM + web_search)."""

import asyncio
import random
import re
import sqlite3
from collections.abc import Callable

import click

from subtitle_generator.parameter_state import DEFAULT_JACKET_MODEL
from subtitle_generator.market_tiers import jacket_tone_text
from subtitle_generator.tiering import TierEvidence, compute_tier_evidence, parse_subtitle_slots

try:
    from copilot import CopilotClient
    from copilot.session import PermissionHandler
    _HAS_COPILOT_SDK = True
except ImportError:
    _HAS_COPILOT_SDK = False

REQUIRED_SECTIONS = [
    "## Title",
    "## Subtitle",
    "## Internal Concept",
    "## Back Cover",
    "## Review 1",
    "## Review 2",
]

MAX_RETRIES = 2

INLINE_BLURB_RE = re.compile(
    r'(?m)^\s*["“][^"“”\n]+["”]\s+—\s+[^,\n]+,\s+.+$'
)

# --- Accessibility scoring & tone tiers ---

TONE_HIGH = jacket_tone_text("pop")
TONE_MEDIUM = jacket_tone_text("mainstream")
TONE_LOW = jacket_tone_text("niche")


def _parse_subtitle_fillers(subtitle: str) -> list[str]:
    """Extract the slot fillers from a subtitle string."""
    return [slot.filler for slot in parse_subtitle_slots(subtitle)]


def _lookup_freq(conn: sqlite3.Connection, filler: str) -> tuple[int, float | None]:
    """Look up a filler's corpus frequency and popularity score.

    Returns (freq, popularity_score). Defaults to (1, None) if not found.
    """
    row = conn.execute(
        "SELECT freq, popularity_score FROM slot_fillers WHERE filler = ? LIMIT 1", (filler,)
    ).fetchone()
    return (row[0], row[1]) if row else (1, None)


def compute_accessibility(subtitle: str, conn: sqlite3.Connection | None = None) -> tuple[str, float]:
    """Compute the jacket tone and accessibility score for a subtitle.

    Returns ``(tone_text, score)`` for compatibility with older callers. The
    score is still the mean blended accessibility score, while the tone now
    comes from the evidence-aware tier classifier. Do not infer a tier from the
    returned score; use ``compute_tier_evidence`` when the tier decision matters.
    """
    evidence = compute_tier_evidence(subtitle, conn)
    return _TIER_TO_TONE[evidence.tier], evidence.accessibility_score



JACKET_PROMPT = """\
You are a publishing industry expert. I will give you a randomly generated book subtitle
in the pop-nonfiction pattern "X, Y, and [the/a/an] Z of [the/a/an] W". Your job is to imagine the book
this subtitle belongs to and produce a complete book jacket.

**Output the following sections in markdown, using the exact headers shown:**

## Title
A punchy 2-4 word main title for the book (evocative, bookstore-ready).

## Subtitle
Restate the subtitle exactly as given.

## Internal Concept
Before writing anything, use web_search to research the key themes in the subtitle — the
people, events, cultural phenomena, and real-world intersections these topics evoke.

{tone}

Then write 5-8 sentences describing the book's core thesis, tone, and target audience.
Weave in specific real-world details you found — a journalist's investigation, a cultural
flashpoint, a surprising historical connection, a bestselling book on a related theme.
The concept should feel like it could only describe ONE specific book, not a generic
treatment of the topic.
This anchors everything else — both reviews must describe the SAME book.

## Back Cover
Write the publisher's marketing copy for the back of the book.

{back_cover_guidance}

Immediately below the back cover copy, include exactly TWO endorsement blurbs as separate
paragraphs before the Review 1 header. Do NOT create separate blurb headers or blockquotes.
Use exactly these preselected blurb source types, in order. Do not substitute different
source types, and do not label the source type in the final output.

{blurb_instructions}

For each blurb, use web_search to find a real person or real publication whose expertise
aligns with this book's subject matter, then search for examples of their actual writing
or past blurbs to understand their distinctive voice. At least one blurb must be from a
real individual person. Format each blurb exactly like this:

"[A single compelling endorsement sentence written in their authentic voice and style]" — [Full Name], [brief credential, e.g. author of The Looming Tower]

## Review 1
Review 1 must be for **{review_1_name}**. Write in that publication's AUTHENTIC house
style — match its real tone, vocabulary, sentence structure, coverage assumptions, and
evaluative habits.

## Review 2
Review 2 must be for **{review_2_name}**. Same approach — write in THEIR authentic house
style with their specific format conventions.

The two review outlets for this jacket have already been selected by weighted
tier-specific sampling. Do not substitute a different publication.

{review_instructions}

Format:
**[Publication Name]**
[Full review in their authentic house style, including their specific closing format
(Verdict/Takeaway/Summing Up as appropriate)]

---

The subtitle is:

{subtitle}
"""


_TONE_TO_TIER: dict[str, str] = {TONE_HIGH: "pop", TONE_MEDIUM: "mainstream", TONE_LOW: "niche"}
_TIER_TO_TONE: dict[str, str] = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}

REVIEW_OUTLET_WEIGHTS: dict[str, dict[str, int]] = {
    "pop": {
        "Publishers Weekly": 55,
        "Kirkus Reviews": 30,
        "Library Journal": 15,
    },
    "mainstream": {
        "Publishers Weekly": 40,
        "Kirkus Reviews": 35,
        "Library Journal": 25,
    },
    "niche": {
        "Choice (ACRL)": 30,
        "BookLife (by Publishers Weekly)": 25,
        "Library Journal": 25,
        "Kirkus Reviews": 20,
    },
}

BLURB_SOURCE_WEIGHTS: dict[str, dict[str, int]] = {
    "pop": {
        "Celebrity or pop public intellectual": 40,
        "Bestselling trade author": 35,
        "Journalist or critic": 20,
        "Institution or publication": 5,
    },
    "mainstream": {
        "Bestselling trade author": 40,
        "Journalist or critic": 30,
        "Celebrity or pop public intellectual": 15,
        "Institution or publication": 10,
        "Domain expert or academic": 5,
    },
    "niche": {
        "Domain expert or academic": 45,
        "Bestselling trade author": 20,
        "Journalist or critic": 20,
        "Institution or publication": 15,
    },
}

BACK_COVER_GUIDANCE: dict[str, str] = {
    "pop": (
        "POP back-cover style: 120-180 words, third person, present tense, 3 short "
        "paragraphs. Open with a punchy one-sentence hook: question, shock, trope, danger, "
        "wish fulfillment, or binary stakes. Then introduce the central person, problem, "
        "conflict, or nonfiction promise in plain, vivid language. Escalate to a "
        "cliffhanger or transformation promise without spoilers; add one concrete detail "
        "and one broad comp/category signal when useful. Avoid plot summary, abstract "
        "themes, specialist terminology, old/obscure comps, and book-report tone."
    ),
    "mainstream": (
        "MAINSTREAM back-cover style: 180-280 words, third person, present tense, 2-3 "
        "polished paragraphs. Open with an elegant premise, dilemma, image, historical "
        "moment, relationship, or moral question. Give concrete details about setting, "
        "inciting situation, conflict, or argument, then widen into resonance: family, "
        "identity, ambition, justice, memory, grief, discovery, or belonging. Promise "
        "emotional involvement plus intellectual reward. Avoid cheap hype, vague "
        "'unforgettable journey' language, overplotting, spoilers past the inciting "
        "incident, and more than two comps."
    ),
    "niche": (
        "NICHE back-cover style: 200-320 words, third person, present tense, precise and "
        "authoritative rather than hard-selling. Open by naming the exact subject, problem, "
        "debate, method, trope, field, or community. Explain the contribution: new evidence, "
        "fresh argument, practical expertise, rare access, specialized worldbuilding, or a "
        "distinctive angle. For scholarly/professional books, include field context, method "
        "or evidence (archives, fieldwork, datasets, case studies), significance, audience, "
        "and a one-sentence author credential. Avoid bestseller hype, vague generalities, "
        "unsupported 'groundbreaking' claims, decorative jargon, and universal-audience claims."
    ),
}

REVIEW_OUTLET_SPECS: dict[str, str] = {
    "Publishers Weekly": (
        "Covers frontlist trade books with clear bookstore distribution: commercial and "
        "literary fiction, narrative nonfiction, memoir, lifestyle, comics, religion, and "
        "children's/YA. Voice: unsigned, third-person, polished, crisp, and balanced; "
        "150-200 words; open with premise, assess craft and market appeal, avoid spoilers, "
        "and close with an implied recommendation or comp-audience note."
    ),
    "Kirkus Reviews": (
        "Covers broad trade publishing and some indie titles, with emphasis on narrative "
        "craft. Voice: brisk, authoritative, candid, and sometimes acerbic; 250-350 words; "
        "interleave summary and critique, maintain professional skepticism, and end with a "
        "punchy verdict sentence."
    ),
    "Library Journal": (
        "Covers adult books through the lens of public and academic library collection "
        "development. Voice: practical, librarian-facing, utilitarian, and acquisition-minded; "
        "150-250 words; include audience, readalikes or collection context, and end with a "
        "one-line VERDICT: purchase recommendation."
    ),
    "BookLife (by Publishers Weekly)": (
        "Covers indie, self-published, and small-press books rather than mainstream trade "
        "frontlist. Voice: PW-adjacent, professional, warmer, and indie-market aware; "
        "200-300 words in three movements: summary, critique, and audience/comp titles; "
        "end with Takeaway: plus production-style letter grades."
    ),
    "Choice (ACRL)": (
        "Covers scholarly monographs, academic crossover nonfiction, and reference works for "
        "college and research libraries. Voice: concise, objective, disciplinary, and "
        "diplomatic; 190-250 words; situate the work in its field and end with Summing Up: "
        "plus a recommendation level and academic audience."
    ),
}

BLURB_SOURCE_SPECS: dict[str, str] = {
    "Celebrity or pop public intellectual": (
        "Find a famous, culturally fluent person with mass-audience credibility for the "
        "book's subject. The blurb should be punchy, quotable, and emotionally immediate."
    ),
    "Bestselling trade author": (
        "Find a bestselling author in an adjacent trade category or genre. The blurb should "
        "sound like jacket copy from a peer author and foreground narrative appeal."
    ),
    "Journalist or critic": (
        "Find a journalist, critic, essayist, or magazine writer who covers the book's "
        "themes. The blurb should be observant, stylish, and culturally specific."
    ),
    "Institution or publication": (
        "Find a real publication, association, book club, or institution aligned with the "
        "subject. Write the blurb as a concise review pull quote attributed to that outlet."
    ),
    "Domain expert or academic": (
        "Find a scholar, scientist, policy expert, historian, or other titled domain expert. "
        "The blurb should stress authority, contribution, and field-level significance."
    ),
}


def _weighted_sample_without_replacement(
    weights_by_name: dict[str, int],
    count: int,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample unique keys according to integer percentage weights."""
    names = list(weights_by_name)
    weights = [weights_by_name[name] for name in names]
    chosen: list[str] = []
    chooser = rng.choices if rng is not None else random.choices

    for _ in range(min(count, len(names))):
        pick = chooser(names, weights=weights, k=1)[0]
        idx = names.index(pick)
        chosen.append(pick)
        names.pop(idx)
        weights.pop(idx)
    return chosen


def _select_review_outlets(tone_tier: str, rng: random.Random | None = None) -> list[str]:
    """Select exactly two review outlets for a tone tier."""
    weights = REVIEW_OUTLET_WEIGHTS.get(tone_tier, REVIEW_OUTLET_WEIGHTS["mainstream"])
    return _weighted_sample_without_replacement(weights, 2, rng)


def _select_blurb_source_types(tone_tier: str, rng: random.Random | None = None) -> list[str]:
    """Select exactly two endorsement-source types for a tone tier."""
    weights = BLURB_SOURCE_WEIGHTS.get(tone_tier, BLURB_SOURCE_WEIGHTS["mainstream"])
    return _weighted_sample_without_replacement(weights, 2, rng)


def _format_review_instructions(review_outlets: list[str]) -> str:
    lines = ["Selected review outlets:"]
    for i, outlet in enumerate(review_outlets, start=1):
        lines.append(f"{i}. **{outlet}** — {REVIEW_OUTLET_SPECS[outlet]}")
    return "\n".join(lines)


def _format_blurb_instructions(source_types: list[str]) -> str:
    lines = ["Selected blurb source types:"]
    for i, source_type in enumerate(source_types, start=1):
        lines.append(f"{i}. **{source_type}** — {BLURB_SOURCE_SPECS[source_type]}")
    return "\n".join(lines)


def _select_jacket_tone(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
) -> tuple[str, str, TierEvidence]:
    evidence = compute_tier_evidence(subtitle, conn)
    if tone_override:
        tone_tier = _TONE_TO_TIER.get(tone_override, "mainstream")
        return tone_tier, tone_override, evidence

    if allowed_tiers and evidence.tier not in allowed_tiers:
        requested = ", ".join(sorted(allowed_tiers))
        raise ValueError(
            f"Subtitle evidence tier '{evidence.tier}' does not match allowed tier(s): "
            f"{requested}"
        )

    tone_tier = evidence.tier
    return tone_tier, _TIER_TO_TONE[tone_tier], evidence


def _build_jacket_prompt_with_evidence(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str, str, TierEvidence]:
    tone_tier, tone, evidence = _select_jacket_tone(
        subtitle,
        conn=conn,
        tone_override=tone_override,
        allowed_tiers=allowed_tiers,
    )

    review_outlets = _select_review_outlets(tone_tier, rng)
    blurb_source_types = _select_blurb_source_types(tone_tier, rng)

    full_prompt = JACKET_PROMPT.format(
        subtitle=subtitle,
        tone=tone,
        back_cover_guidance=BACK_COVER_GUIDANCE[tone_tier],
        blurb_instructions=_format_blurb_instructions(blurb_source_types),
        review_1_name=review_outlets[0],
        review_2_name=review_outlets[1],
        review_instructions=_format_review_instructions(review_outlets),
    )

    # Split at the --- separator into system (instructions) and user (subtitle) parts
    sep = "\n\n---\n\n"
    idx = full_prompt.rfind(sep)
    if idx >= 0:
        system_prompt = full_prompt[:idx]
        user_prompt = full_prompt[idx + len(sep) :]
    else:
        system_prompt = full_prompt
        user_prompt = subtitle

    return system_prompt, user_prompt, tone_tier, evidence


def build_jacket_prompt(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
    rng: random.Random | None = None,
) -> tuple[str, str, str]:
    """Construct the jacket generation prompts without calling the LLM.

    Returns (system_prompt, user_prompt, tone_tier) where:
    - system_prompt contains role instructions, format requirements, and tone context
    - user_prompt contains the subtitle framing
    - tone_tier is "pop", "mainstream", or "niche"
    """
    system_prompt, user_prompt, tone_tier, _ = _build_jacket_prompt_with_evidence(
        subtitle,
        conn=conn,
        tone_override=tone_override,
        allowed_tiers=allowed_tiers,
        rng=rng,
    )

    return system_prompt, user_prompt, tone_tier


def _validate_jacket(content: str) -> list[str]:
    """Check required section headers and inline back-cover blurbs.

    Returns a list of missing or invalid format requirements.
    """
    missing = []
    for section in REQUIRED_SECTIONS:
        # Case-insensitive header check (model may vary casing)
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if not pattern.search(content):
            missing.append(section)
    if "## Back Cover" not in missing:
        back_cover = _extract_section(content, "## Back Cover")
        inline_blurbs = list(INLINE_BLURB_RE.finditer(back_cover))
        if len(inline_blurbs) != 2:
            missing.append("exactly two inline Back Cover blurbs")
        elif not back_cover[:inline_blurbs[0].start()].strip():
            missing.append("Back Cover description before inline blurbs")
    if re.search(r"^##\s+Blurb\s+\d+\s*$", content, re.IGNORECASE | re.MULTILINE):
        missing.append("remove separate ## Blurb sections")
    return missing


def _extract_section(content: str, section: str) -> str:
    """Return the body for a markdown H2 section, or an empty string if absent."""
    pattern = re.compile(
        rf"^\s*{re.escape(section)}\s*\n(?P<body>.*?)(?=^\s*##\s+|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(content)
    return match.group("body") if match else ""


DEFAULT_MODEL = DEFAULT_JACKET_MODEL


def _progress(msg: str, on_progress: Callable[[str], None] | None = None) -> None:
    click.echo(f"  {msg}")
    if on_progress:
        on_progress(msg)


def _prepare_jacket_prompt(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> tuple[str, str, str]:
    system_prompt, user_prompt, tone_tier, evidence = _build_jacket_prompt_with_evidence(
        subtitle, conn=conn, tone_override=tone_override, allowed_tiers=allowed_tiers,
    )

    if tone_override:
        _progress("Tone: override", on_progress)
    else:
        _progress(
            f"Tone: {tone_tier} "
            f"(accessibility: {evidence.accessibility_score:.2f}, "
            f"tail: {evidence.lower_tail_score:.2f}, "
            f"demand: {evidence.demand_confidence:.2f})",
            on_progress,
        )

    return system_prompt, user_prompt, tone_tier


async def _generate_jacket_from_prompt_async(
    subtitle: str,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Call the Copilot SDK to generate a full book jacket with validation and retry."""
    if not _HAS_COPILOT_SDK:
        raise RuntimeError("Copilot SDK not available. Use dry_run=true for prompt-only mode.")

    _progress(f"Connecting to {model}...", on_progress)

    async with CopilotClient() as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=model,
            infinite_sessions={"enabled": False},
        ) as session:
            prompt = system_prompt + "\n\n---\n\n" + user_prompt
            _progress("Generating jacket...", on_progress)

            for attempt in range(1, MAX_RETRIES + 2):
                result = await session.send_and_wait(prompt, timeout=timeout)
                content = (result.data.content or "") if result and result.data else ""

                if not content:
                    _progress(f"Attempt {attempt}: empty response, retrying...", on_progress)
                    continue

                missing = _validate_jacket(content)
                if not missing:
                    _progress("Complete", on_progress)
                    return content

                if attempt <= MAX_RETRIES:
                    missing_names = ", ".join(missing)
                    _progress(f"Attempt {attempt}: format issues: {missing_names}, retrying...", on_progress)
                    prompt = (
                        f"Your previous response did not satisfy these required sections "
                        f"or format requirements: {missing_names}.\n"
                        f"Please regenerate the COMPLETE book jacket with ALL sections and "
                        f"place the two endorsement blurbs inline inside the Back Cover section, "
                        f"directly below the description and before Review 1. Do not use separate "
                        f"## Blurb sections. "
                        f"The subtitle is:\n\n{subtitle}"
                    )
                else:
                    _progress(f"Best effort after {attempt} attempts", on_progress)
                    return content

            return "(No valid response after retries)"


async def _generate_jacket_async(
    subtitle: str, model: str = DEFAULT_MODEL, timeout: float = 120.0,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None, allowed_tiers: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    system_prompt, user_prompt, _ = _prepare_jacket_prompt(
        subtitle, conn=conn, tone_override=tone_override, allowed_tiers=allowed_tiers,
        on_progress=on_progress,
    )
    return await _generate_jacket_from_prompt_async(
        subtitle, system_prompt, user_prompt, model=model, timeout=timeout,
        on_progress=on_progress,
    )


def generate_jacket_from_prompt(
    subtitle: str,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    show_concept: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Synchronous wrapper for jacket generation from a prebuilt prompt."""
    content = asyncio.run(_generate_jacket_from_prompt_async(
        subtitle, system_prompt, user_prompt, model=model, timeout=timeout,
        on_progress=on_progress,
    ))
    if not show_concept:
        content = _strip_internal_concept(content)
    return content


def _strip_internal_concept(content: str) -> str:
    """Remove the ## Internal Concept section from output."""
    return re.sub(
        r"## Internal Concept\s*\n.*?(?=\n## )", "", content, count=1, flags=re.DOTALL | re.IGNORECASE
    )


def generate_jacket(
    subtitle: str, model: str = DEFAULT_MODEL, timeout: float = 120.0,
    show_concept: bool = False,
    conn: sqlite3.Connection | None = None, tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Synchronous wrapper for jacket generation. Returns markdown string."""
    system_prompt, user_prompt, _ = _prepare_jacket_prompt(
        subtitle, conn=conn, tone_override=tone_override, allowed_tiers=allowed_tiers,
        on_progress=on_progress,
    )
    return generate_jacket_from_prompt(
        subtitle, system_prompt, user_prompt, model=model, timeout=timeout,
        show_concept=show_concept, on_progress=on_progress,
    )
