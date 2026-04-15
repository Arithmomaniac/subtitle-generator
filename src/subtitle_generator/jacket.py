"""Generate full book jackets using the Copilot SDK (LLM + web_search)."""

import asyncio
import math
import random
import re
import sqlite3
from collections.abc import Callable

import click

from subtitle_generator.config import load_tuning_config

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
    "## Blurb 1",
    "## Blurb 2",
]

MAX_RETRIES = 2

# --- Accessibility scoring & tone tiers ---

_SUBTITLE_RE = re.compile(
    r"^(?P<list_part>.+,\s*.+?)\s*,?\s+and\s+the\s+(?P<action>.+?)\s+of\s+(?P<object>.+)$",
    re.IGNORECASE,
)

TONE_HIGH = """\
This is an airport bookstore book. Think Malcolm Gladwell, Michael Pollan, Mary Roach,
or Bill Bryson. The reader grabs it on impulse. Prioritize bestselling trade nonfiction,
podcasts, magazine features, and pop-culture references in your research. Keep the tone
accessible, narrative-driven, and full of surprising anecdotes."""

TONE_MEDIUM = """\
This is a quality indie bookstore book. Think Rebecca Solnit, Pankaj Mishra, or Merlin
Sheldrake. The reader is curious and educated but not specialist. Blend trade nonfiction,
longform journalism, and accessible academic work. The tone should be essayistic and
intellectually engaging without being dense."""

TONE_LOW = """\
This is a smart university press crossover — think Princeton's 'Lives of Great Ideas'
series or Yale's cultural history list. The reader is intellectually adventurous. Blend
academic depth with essayistic flair, referencing both specialist scholarship and trade
nonfiction for context. Rigorous but never dry."""


def _parse_subtitle_fillers(subtitle: str) -> list[str]:
    """Extract the slot fillers from a subtitle string."""
    m = _SUBTITLE_RE.match(subtitle)
    if not m:
        return []
    list_part = m.group("list_part")
    action = m.group("action").strip()
    obj = re.sub(r"[\s]*[/:;,.]\s*$", "", m.group("object")).strip()
    items = [item.strip() for item in list_part.split(",") if item.strip()]
    return items + [action, obj]


def _lookup_freq(conn: sqlite3.Connection, filler: str) -> tuple[int, float | None]:
    """Look up a filler's corpus frequency and popularity score.

    Returns (freq, popularity_score). Defaults to (1, None) if not found.
    """
    row = conn.execute(
        "SELECT freq, popularity_score FROM slot_fillers WHERE filler = ? LIMIT 1", (filler,)
    ).fetchone()
    return (row[0], row[1]) if row else (1, None)


def compute_accessibility(subtitle: str, conn: sqlite3.Connection | None = None) -> tuple[str, float]:
    """Compute an accessibility tier for a subtitle based on filler scores.

    Returns (tone_text, score) where score is a blend of mean(log10(1+freq))
    and mean(popularity_score) per pop_tone_blend config.
    """
    fillers = _parse_subtitle_fillers(subtitle)
    if not fillers or conn is None:
        return TONE_MEDIUM, 0.0

    cfg = load_tuning_config(conn)
    blend_tone = cfg.get("pop_tone_blend", 0.0)
    pop_default = cfg.get("pop_missing_default", 0.1)

    filler_data = [_lookup_freq(conn, f) for f in fillers]
    blended_scores = []
    for freq, pop_score in filler_data:
        score_freq = math.log10(1 + freq)
        ps = pop_score if pop_score is not None else pop_default
        blended_scores.append((1 - blend_tone) * score_freq + blend_tone * ps)
    score = sum(blended_scores) / len(blended_scores)

    pop_thresh = cfg["accessibility_threshold_pop"]
    main_thresh = cfg["accessibility_threshold_mainstream"]

    # Thresholds tuned to the distribution:
    # score > pop_thresh → fillers avg freq ~10+ (pop staples like Race, Power, America)
    # score main_thresh-pop_thresh → fillers avg freq ~2-10 (mainstream nonfiction)
    # score < main_thresh → fillers avg freq ~1-2 (niche/academic)
    if score > pop_thresh:
        tone = TONE_HIGH
    elif score >= main_thresh:
        tone = TONE_MEDIUM
    else:
        tone = TONE_LOW

    return tone, score



def sample_tone(score: float, allowed_tiers: set[str] | None = None, conn: sqlite3.Connection | None = None) -> tuple[str, str]:
    """Randomly sample a tone tier with probabilities centered on the score.

    Returns (tier_name, tone_text). Probabilities are Gaussian-weighted
    by distance from each tier's center score. allowed_tiers clamps the
    selection to a subset (zeroing out disallowed tiers and renormalizing).
    """
    cfg = load_tuning_config(conn)
    spread = cfg["sample_tone_spread"]

    tiers_def = [
        ("pop", TONE_HIGH, cfg["tier_center_pop"]),
        ("mainstream", TONE_MEDIUM, cfg["tier_center_mainstream"]),
        ("niche", TONE_LOW, cfg["tier_center_niche"]),
    ]
    weights = []
    tiers = []
    for name, text, center in tiers_def:
        if allowed_tiers and name not in allowed_tiers:
            continue
        w = math.exp(-((score - center) / spread) ** 2)
        weights.append(w)
        tiers.append((name, text))

    if not tiers:
        return "mainstream", TONE_MEDIUM

    chosen = random.choices(tiers, weights=weights, k=1)[0]
    return chosen


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
The publisher's marketing copy for the back of the book. 2-3 paragraphs (~250 words).
Open with a hook question or provocative claim. End with an emotional/intellectual payoff.
Tone: urgent, seductive, intellectually intriguing.

## Review 1
Pick the most appropriate trade publication for this book from the roster below. Write
the review in that publication's AUTHENTIC house style — match their real tone, vocabulary,
sentence structure, and evaluative habits. Each publication has a distinct editorial voice:

- **Publishers Weekly** — Neutral, polished, incisive. No first person. Crisp active
  sentences, rarely exceeding 200 words. Balances praise with measured critique. Notes
  commercial appeal and audience. Light wit permitted, never gushy. Closes with an
  implied recommendation. Typical phrasing: "a vivid, propulsive account", "the prose
  occasionally strains", "will appeal to readers of..."

- **Kirkus Reviews** — Direct, authoritative, wry. Can be acerbic. The "literary snob"
  that values narrative craft above all. Punchy closing verdict sentence. Professional
  skepticism even when praising. 250-350 words. Typical: "atmospheric and ambitious,
  but uneven", "a masterful blend of suspense and literary style", "earnest, but
  ultimately exhausting."

- **Library Journal** — Written FOR librarians making purchase decisions. Practical,
  utilitarian tone. Ends with a one-line "VERDICT:" that is a direct acquisition
  recommendation with audience/collection context. Typical: "Highly recommended for
  public libraries", "A solid choice where demand exists", "Essential for university
  libraries supporting programs in..."

- **BookLife (by Publishers Weekly)** — Indie/niche specialist. Three paragraphs:
  summary → critique → audience/comp titles. Ends with a one-sentence "Takeaway:" and
  letter grades for production elements (Cover, Editing, etc. on A+ to C scale).
  Warmer and more encouraging than main PW, but still professional.

- **Choice (ACRL)** — Scholarly, concise (190-250 words), aimed at academic librarians.
  Situates the book within its discipline. Ends with "Summing Up:" followed by a level
  recommendation and audience (e.g., "Essential. Upper-division undergraduates through
  faculty."). Diplomatic even when critical. Typical: "fills a significant gap in the
  literature", "the author makes a significant contribution to..."

Format:
**[Publication Name]**
[Full review in their authentic house style, including their specific closing format
(Verdict/Takeaway/Summing Up as appropriate)]

## Review 2
Pick a second- or third-most appropriate publication from the roster above. Same approach — write in THEIR
authentic house style with their specific format conventions.

## Blurb 1
Use web_search to find a real person (author, academic, journalist, public intellectual)
whose expertise aligns with this book's subject matter. Then search for examples of their
actual writing or past blurbs to understand their distinctive voice — their vocabulary,
sentence rhythm, rhetorical habits, and tone. The blurb should sound like THEM, not like
generic marketing copy. Format:

**[Full Name]** ([brief credential, e.g. "author of The Looming Tower"])
> "[A single compelling endorsement sentence written in their authentic voice and style]"

## Blurb 2
Find a SECOND real person (or a real publication/newspaper that covers this topic).
Same approach — web_search for their writing style, then compose the blurb in their voice.
At least one of the two blurbs must be from a real individual person.

---

The subtitle is:

{subtitle}
"""


_TONE_TO_TIER: dict[str, str] = {TONE_HIGH: "pop", TONE_MEDIUM: "mainstream", TONE_LOW: "niche"}
_TIER_TO_TONE: dict[str, str] = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}


def build_jacket_prompt(
    subtitle: str,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None,
    allowed_tiers: set[str] | None = None,
) -> tuple[str, str, str]:
    """Construct the jacket generation prompts without calling the LLM.

    Returns (system_prompt, user_prompt, tone_tier) where:
    - system_prompt contains role instructions, format requirements, and tone context
    - user_prompt contains the subtitle framing
    - tone_tier is "pop", "mainstream", or "niche"
    """
    if tone_override:
        tone = tone_override
        tone_tier = _TONE_TO_TIER.get(tone_override, "mainstream")
    else:
        _, score = compute_accessibility(subtitle, conn)
        tone_tier, tone = sample_tone(score, allowed_tiers, conn)

    full_prompt = JACKET_PROMPT.format(subtitle=subtitle, tone=tone)

    # Split at the --- separator into system (instructions) and user (subtitle) parts
    sep = "\n\n---\n\n"
    idx = full_prompt.rfind(sep)
    if idx >= 0:
        system_prompt = full_prompt[:idx]
        user_prompt = full_prompt[idx + len(sep) :]
    else:
        system_prompt = full_prompt
        user_prompt = subtitle

    return system_prompt, user_prompt, tone_tier


def _validate_jacket(content: str) -> list[str]:
    """Check that all required sections are present. Returns list of missing section names."""
    missing = []
    for section in REQUIRED_SECTIONS:
        # Case-insensitive header check (model may vary casing)
        pattern = re.compile(re.escape(section), re.IGNORECASE)
        if not pattern.search(content):
            missing.append(section)
    return missing


DEFAULT_MODEL = "gpt-5.4-mini"


async def _generate_jacket_async(
    subtitle: str, model: str = DEFAULT_MODEL, timeout: float = 120.0,
    conn: sqlite3.Connection | None = None,
    tone_override: str | None = None, allowed_tiers: set[str] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """Call the Copilot SDK to generate a full book jacket with validation and retry."""
    if not _HAS_COPILOT_SDK:
        raise RuntimeError("Copilot SDK not available. Use dry_run=true for prompt-only mode.")

    def _progress(msg: str) -> None:
        click.echo(f"  {msg}")
        if on_progress:
            on_progress(msg)

    system_prompt, user_prompt, tone_tier = build_jacket_prompt(
        subtitle, conn=conn, tone_override=tone_override, allowed_tiers=allowed_tiers,
    )

    if tone_override:
        _progress(f"Tone: override")
    else:
        _, score = compute_accessibility(subtitle, conn)
        cfg = load_tuning_config(conn)
        pop_thresh = cfg["accessibility_threshold_pop"]
        main_thresh = cfg["accessibility_threshold_mainstream"]
        natural = "pop" if score > pop_thresh else ("mainstream" if score >= main_thresh else "niche")
        _progress(f"Tone: {tone_tier} (score: {score:.2f})")

    _progress(f"Connecting to {model}...")

    async with CopilotClient() as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=model,
            infinite_sessions={"enabled": False},
        ) as session:
            prompt = system_prompt + "\n\n---\n\n" + user_prompt
            _progress("Generating jacket...")

            for attempt in range(1, MAX_RETRIES + 2):
                result = await session.send_and_wait(prompt, timeout=timeout)
                content = (result.data.content or "") if result and result.data else ""

                if not content:
                    _progress(f"Attempt {attempt}: empty response, retrying...")
                    continue

                missing = _validate_jacket(content)
                if not missing:
                    _progress("Complete")
                    return content

                if attempt <= MAX_RETRIES:
                    missing_names = ", ".join(missing)
                    _progress(f"Attempt {attempt}: missing {missing_names}, retrying...")
                    prompt = (
                        f"Your previous response was missing these required sections: {missing_names}.\n"
                        f"Please regenerate the COMPLETE book jacket with ALL sections. "
                        f"The subtitle is:\n\n{subtitle}"
                    )
                else:
                    _progress(f"Best effort after {attempt} attempts")
                    return content

            return "(No valid response after retries)"


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
    content = asyncio.run(_generate_jacket_async(
        subtitle, model=model, timeout=timeout,
        conn=conn, tone_override=tone_override, allowed_tiers=allowed_tiers,
        on_progress=on_progress,
    ))
    if not show_concept:
        content = _strip_internal_concept(content)
    return content
