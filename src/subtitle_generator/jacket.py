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
2-3 sentences describing the book's core thesis, tone, and target audience.
This anchors everything else — both reviews must describe the SAME book.

## Back Cover
The publisher's marketing copy for the back of the book. 2-3 paragraphs (~250 words).
Open with a hook question or provocative claim. End with an emotional/intellectual payoff.
Tone: urgent, seductive, intellectually intriguing.

## Review 1
Pick the most appropriate trade journal for this book from:
- **Publishers Weekly** — high-market trade, "timely & commercial" topics
- **BookLife (by Publishers Weekly)** — indie/niche catch-all
- **Kirkus Reviews** — literary snob, wants narrative-driven nonfiction
- **Library Journal** — librarian's choice, utility/reference focused
- **Choice (ACRL)** — scholarly, academic authority

Format:
**[Journal Name]**
*Verdict: [one-line verdict]*
[Review paragraph, 100-150 words, in the voice/personality of that journal]

## Review 2
Pick a DIFFERENT journal from the list above. Same format as Review 1.

## Blurb 1
Use web_search to find a real person (author, academic, journalist, public intellectual)
whose expertise aligns with this book's subject matter. They should be someone who would
plausibly endorse this kind of book. Format:

**[Full Name]** ([brief credential, e.g. "author of The Looming Tower"])
> "[A single compelling endorsement sentence in their voice]"

## Blurb 2
Find a SECOND real person (or a real publication/newspaper that covers this topic).
Same format. At least one of the two blurbs must be from a real individual person.

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


async def _generate_jacket_async(subtitle: str, timeout: float = 120.0) -> str:
    """Call the Copilot SDK to generate a full book jacket with validation and retry."""
    async with CopilotClient() as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model="gpt-5-mini",
            infinite_sessions={"enabled": False},
        ) as session:
            prompt = JACKET_PROMPT.format(subtitle=subtitle)

            for attempt in range(1, MAX_RETRIES + 2):  # 1 initial + MAX_RETRIES retries
                result = await session.send_and_wait(prompt, timeout=timeout)
                content = (result.data.content or "") if result and result.data else ""

                if not content:
                    click.echo(f"  ⚠ Attempt {attempt}: empty response, retrying...")
                    continue

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


def generate_jacket(subtitle: str, timeout: float = 120.0) -> str:
    """Synchronous wrapper for jacket generation. Returns markdown string."""
    return asyncio.run(_generate_jacket_async(subtitle, timeout=timeout))
