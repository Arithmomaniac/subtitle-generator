"""Generate full book jackets using the Copilot SDK (LLM + web_search)."""

import asyncio
import re

import click
from copilot import CopilotClient
from copilot.session import PermissionHandler

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

RESEARCH_PROMPT = """\
You are a publishing industry researcher. I will give you a randomly generated book subtitle
in the pop-nonfiction pattern "X, Y, and the Z of W". Your job is to research the real-world
landscape around these themes and produce a rich, detailed book concept.

**Steps:**
1. Identify the key themes and topics in the subtitle
2. Use web_search to find real-world context: notable scholars, authors, or public figures
   associated with these topics; landmark books or articles; historical events or cultural
   flashpoints; ongoing debates or surprising connections between the themes
3. Synthesize your findings into a detailed Internal Concept (5-8 sentences)

**Output format — use this exact header:**

## Internal Concept
[5-8 sentences describing the book's core thesis, tone, and target audience. Weave in
specific real-world details you found — a particular scholar's argument, a pivotal historical
moment, a cultural flashpoint, a surprising connection. The concept should feel like it could
only describe ONE specific book, not a generic treatment of the topic.]

The subtitle is:

{subtitle}
"""

JACKET_PROMPT_PHASE2 = """\
You are a publishing industry expert. Using the Internal Concept you just developed in the
previous message, produce a complete book jacket for the same subtitle.

**The concept is already established — do NOT regenerate it.** Use it as the anchor for
everything below. Both reviews must describe the SAME book as the concept.

**Output the following sections in markdown, using the exact headers shown:**

## Title
A punchy 2-4 word main title for the book (evocative, bookstore-ready).

## Subtitle
Restate the subtitle exactly as given.

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

JACKET_PROMPT = """\
You are a publishing industry expert. I will give you a randomly generated book subtitle
in the pop-nonfiction pattern "X, Y, and the Z of W". Your job is to imagine the book
this subtitle belongs to and produce a complete book jacket.

**Output the following sections in markdown, using the exact headers shown:**

## Title
A punchy 2-4 word main title for the book (evocative, bookstore-ready).

## Subtitle
Restate the subtitle exactly as given.

## Internal Concept
Before writing anything, use web_search to research the key themes in the subtitle — the
people, events, scholarly debates, cultural phenomena, and real-world intersections these
topics evoke. Look for specific names, controversies, landmark books, or surprising
connections that ground the concept in reality.

Then write 5-8 sentences describing the book's core thesis, tone, and target audience.
Weave in the specific real-world details you found — a particular scholar's argument,
a pivotal historical moment, a cultural flashpoint. The concept should feel like it could
only describe ONE specific book, not a generic treatment of the topic.
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
    subtitle: str, model: str = DEFAULT_MODEL, timeout: float = 120.0, deep_research: bool = False,
) -> str:
    """Call the Copilot SDK to generate a full book jacket with validation and retry.

    When deep_research=True, uses a two-phase approach:
      Phase 1: Research prompt → web search + detailed concept
      Phase 2: Jacket prompt that builds on the established concept
    Otherwise uses the enhanced one-shot prompt (which also does web search for concept).
    """
    async with CopilotClient() as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=model,
            infinite_sessions={"enabled": False},
        ) as session:
            concept_content = ""

            if deep_research:
                # Phase 1: Research the subtitle's themes
                click.echo("  🔍 Deep research: searching for real-world context...")
                research_prompt = RESEARCH_PROMPT.format(subtitle=subtitle)
                result = await session.send_and_wait(research_prompt, timeout=timeout)
                concept_content = (result.data.content or "") if result and result.data else ""

                if concept_content:
                    click.echo("  ✓ Concept research complete")
                else:
                    click.echo("  ⚠ Research returned empty — falling back to one-shot mode")

            # Choose prompt based on whether we got a concept from Phase 1
            if deep_research and concept_content:
                prompt = JACKET_PROMPT_PHASE2.format(subtitle=subtitle)
            else:
                prompt = JACKET_PROMPT.format(subtitle=subtitle)

            for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
                result = await session.send_and_wait(prompt, timeout=timeout)
                content = (result.data.content or "") if result and result.data else ""

                if not content:
                    click.echo(f"  ⚠ Attempt {attempt}: empty response, retrying...")
                    continue

                # For deep_research, prepend the concept from Phase 1
                if deep_research and concept_content:
                    content = concept_content.strip() + "\n\n" + content

                missing = _validate_jacket(content)
                if not missing:
                    return content

                if attempt <= MAX_RETRIES:
                    missing_names = ", ".join(missing)
                    click.echo(f"  ⚠ Attempt {attempt}: missing sections: {missing_names} — retrying...")
                    prompt = (
                        f"Your previous response was missing these required sections: {missing_names}.\n"
                        f"Please regenerate the COMPLETE book jacket with ALL sections. "
                        f"The subtitle is:\n\n{subtitle}"
                    )
                else:
                    click.echo(f"  ⚠ Returning best effort after {attempt} attempts (missing: {', '.join(missing)})")
                    return content

            return "(No valid response after retries)"


def _strip_internal_concept(content: str) -> str:
    """Remove the ## Internal Concept section from output."""
    return re.sub(
        r"## Internal Concept\s*\n.*?(?=\n## )", "", content, count=1, flags=re.DOTALL | re.IGNORECASE
    )


def generate_jacket(
    subtitle: str, model: str = DEFAULT_MODEL, timeout: float = 120.0,
    show_concept: bool = False, deep_research: bool = False,
) -> str:
    """Synchronous wrapper for jacket generation. Returns markdown string."""
    content = asyncio.run(_generate_jacket_async(subtitle, model=model, timeout=timeout, deep_research=deep_research))
    if not show_concept:
        content = _strip_internal_concept(content)
    return content
