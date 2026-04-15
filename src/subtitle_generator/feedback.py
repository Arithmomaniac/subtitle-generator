"""Human feedback collection, storage, and summarization for tuning.

Provides a lightweight RLHF-lite layer: humans rate generated subtitles
with thumbs up/down, optional tone override, and free-text comments.
Aggregated summaries are injected into the LLM proposer's context during
autoresearch tuning iterations.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

from pydantic import BaseModel

from subtitle_generator.config import load_tuning_config

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HumanFeedbackInterp(BaseModel):
    """LLM-interpreted structure from a free-text human comment."""

    sentiment: str  # positive / negative / neutral
    tone_signal: str | None  # too-pop / too-niche / good-tone / None
    actionable_insight: str  # one-sentence summary for the proposer


# ---------------------------------------------------------------------------
# DB setup
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS human_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subtitle TEXT NOT NULL,
    system_tone TEXT,
    thumbs INTEGER,
    tone_override TEXT,
    free_text TEXT,
    interpreted TEXT,
    config_snapshot TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)"""


def ensure_ratings_table(conn: sqlite3.Connection) -> None:
    """Create the human_ratings table if it doesn't exist."""
    conn.execute(_CREATE_TABLE_SQL)
    # Migrate: add tags column if missing
    cols = {r[1] for r in conn.execute("PRAGMA table_info(human_ratings)")}
    if "tags" not in cols:
        conn.execute("ALTER TABLE human_ratings ADD COLUMN tags TEXT DEFAULT '[]'")
    conn.commit()


# ---------------------------------------------------------------------------
# Store a rating
# ---------------------------------------------------------------------------


def store_rating(
    conn: sqlite3.Connection,
    subtitle: str,
    *,
    system_tone: str | None = None,
    thumbs: int | None = None,
    tone_override: str | None = None,
    free_text: str | None = None,
    tags: list[str] | None = None,
) -> int:
    """Store a human rating. Returns the row id.

    Args:
        subtitle: The subtitle text that was rated.
        system_tone: The tone tier the system computed (pop/mainstream/niche).
        thumbs: 1 = good, -1 = bad, None = skipped.
        tone_override: What the human thinks the tone should be (p/m/n).
        free_text: Optional free-text comment.
        tags: Quality tags like ["funny", "grammar", "contradiction", "boring"].
    """
    ensure_ratings_table(conn)

    # Snapshot current config for provenance
    config_snapshot = json.dumps(load_tuning_config(conn))

    # Interpret free text if provided
    interpreted = None
    if free_text:
        try:
            interp = interpret_free_text(free_text)
            interpreted = interp.model_dump_json()
        except Exception:
            pass  # non-critical — store the raw text anyway

    tags_json = json.dumps(tags or [])

    cur = conn.execute(
        """INSERT INTO human_ratings
           (subtitle, system_tone, thumbs, tone_override, free_text,
            interpreted, config_snapshot, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (subtitle, system_tone, thumbs, tone_override, free_text,
         interpreted, config_snapshot, tags_json),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Free-text interpretation
# ---------------------------------------------------------------------------

_INTERP_PROMPT = """\
You are analyzing a human's reaction to a generated book subtitle.
The subtitle follows the "X, Y, and the Z of W" pattern.

Human comment: "{comment}"

Classify this comment:
- sentiment: positive, negative, or neutral
- tone_signal: Does this comment suggest the subtitle is "too-pop" (too generic/common), \
"too-niche" (too obscure/academic), "good-tone" (tone is right), or null (no tone signal)?
- actionable_insight: One sentence summarizing what a parameter-tuning agent should learn from this.
"""


def interpret_free_text(comment: str) -> HumanFeedbackInterp:
    """Use a cheap LLM call to interpret free-text feedback."""
    from subtitle_generator.eval_harness import structured_completion

    return structured_completion(
        model="github_copilot/gpt-5.4-mini",
        messages=[{"role": "user", "content": _INTERP_PROMPT.format(comment=comment)}],
        schema=HumanFeedbackInterp,
        timeout=30.0,
    )


# ---------------------------------------------------------------------------
# Summarize ratings
# ---------------------------------------------------------------------------

MIN_RATINGS_THRESHOLD = 10


def get_summary(
    conn: sqlite3.Connection,
    n: int = 50,
) -> dict | None:
    """Aggregate the most recent n human ratings into a summary.

    Returns None if fewer than MIN_RATINGS_THRESHOLD ratings exist.
    """
    ensure_ratings_table(conn)

    rows = conn.execute(
        """SELECT thumbs, system_tone, tone_override, free_text, interpreted, tags
           FROM human_ratings
           ORDER BY created_at DESC
           LIMIT ?""",
        (n,),
    ).fetchall()

    if len(rows) < MIN_RATINGS_THRESHOLD:
        return None

    # Approval rate
    thumbs_rated = [(r[0],) for r in rows if r[0] is not None]
    approval_count = sum(1 for (t,) in thumbs_rated if t == 1)
    approval_rate = approval_count / len(thumbs_rated) if thumbs_rated else None

    # Tone accuracy
    tone_pairs = [
        (r[1], r[2]) for r in rows
        if r[1] is not None and r[2] is not None
    ]
    tone_matches = sum(1 for sys, human in tone_pairs if sys == human)
    tone_accuracy = tone_matches / len(tone_pairs) if tone_pairs else None

    # Tone mismatch patterns
    mismatch_counter: Counter = Counter()
    for sys_tone, human_tone in tone_pairs:
        if sys_tone != human_tone:
            mismatch_counter[f"system={sys_tone} but human={human_tone}"] += 1

    # Recent comments (raw + interpreted insights)
    recent_comments: list[str] = []
    for r in rows[:10]:
        if r[3]:  # free_text
            recent_comments.append(r[3])

    interpreted_insights: list[str] = []
    for r in rows[:20]:
        if r[4]:  # interpreted JSON
            try:
                interp = json.loads(r[4])
                if interp.get("actionable_insight"):
                    interpreted_insights.append(interp["actionable_insight"])
            except (json.JSONDecodeError, TypeError):
                pass

    # Tag counts
    tag_counter: Counter = Counter()
    for r in rows:
        tags_str = r[5] if len(r) > 5 and r[5] else "[]"
        try:
            tags = json.loads(tags_str)
            for tag in tags:
                tag_counter[tag] += 1
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "total_ratings": len(rows),
        "thumbs_rated": len(thumbs_rated),
        "approval_rate": approval_rate,
        "tone_pairs": len(tone_pairs),
        "tone_accuracy": tone_accuracy,
        "tone_mismatches": dict(mismatch_counter.most_common(5)),
        "recent_comments": recent_comments[:5],
        "interpreted_insights": interpreted_insights[:5],
        "tag_counts": dict(tag_counter.most_common()),
    }


def format_summary_for_proposer(summary: dict) -> str:
    """Format a feedback summary as text for the LLM proposer prompt."""
    lines = [f"## Human feedback summary (last {summary['total_ratings']} ratings):"]

    if summary["approval_rate"] is not None:
        pct = summary["approval_rate"] * 100
        lines.append(f"- Approval: {pct:.0f}% thumbs up ({summary['thumbs_rated']} rated)")

    if summary["tone_accuracy"] is not None:
        pct = summary["tone_accuracy"] * 100
        lines.append(f"- Tone accuracy: {pct:.0f}% agree with system tier ({summary['tone_pairs']} compared)")

    if summary["tone_mismatches"]:
        mismatches = ", ".join(
            f"{count}× {desc}" for desc, count in summary["tone_mismatches"].items()
        )
        lines.append(f"- Tone mismatches: {mismatches}")

    if summary["recent_comments"]:
        for comment in summary["recent_comments"][:3]:
            lines.append(f'- Comment: "{comment}"')

    if summary["interpreted_insights"]:
        for insight in summary["interpreted_insights"][:3]:
            lines.append(f"- Insight: {insight}")

    if summary.get("tag_counts"):
        tag_strs = ", ".join(f"{count}× {tag}" for tag, count in summary["tag_counts"].items())
        lines.append(f"- Quality tags: {tag_strs}")

    return "\n".join(lines)
