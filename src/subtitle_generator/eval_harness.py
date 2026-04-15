"""Evaluation harness for the unified tuning pipeline.

Immutable evaluation infrastructure (the "prepare.py" equivalent from
Karpathy's autoresearch pattern).  Provides structured-output LLM rating,
sample generation, tone-separation measurement, and composite scoring.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3

import click
import litellm
from pydantic import BaseModel

# Suppress noisy litellm coroutine warnings
import warnings
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

from subtitle_generator.config import get_tone_targets
from subtitle_generator.generate import generate_subtitle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_RATER_MODEL = "github_copilot/gpt-5.4-mini"
DEFAULT_PROPOSER_MODEL = "github_copilot/gpt-5.4"

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SubtitleRating(BaseModel):
    coherence: int  # 1-10
    evocativeness: int  # 1-10
    surprise: int  # 1-10


class RatingBatch(BaseModel):
    ratings: list[SubtitleRating]


class ParamProposal(BaseModel):
    param: str  # config key to change
    new_value: float  # proposed value
    reasoning: str  # why this change


# ---------------------------------------------------------------------------
# structured_completion — auto-dispatch structured output
# ---------------------------------------------------------------------------

# Models that require the /responses API (not /chat/completions)
_RESPONSES_ONLY_MODELS = {"gpt-5.4-mini", "gpt-5.4", "gpt-5.4-nano"}


def _needs_responses_api(model: str) -> bool:
    """Check if this model needs the /responses API."""
    # Strip provider prefix (e.g. "github_copilot/gpt-5.4-mini" → "gpt-5.4-mini")
    short = model.rsplit("/", 1)[-1] if "/" in model else model
    return short in _RESPONSES_ONLY_MODELS


def _extract_responses_text(output: list) -> str:
    """Extract text content from a Responses API output list."""
    for item in output:
        if hasattr(item, "content") and item.content:
            for c in item.content:
                if hasattr(c, "text"):
                    return c.text
    return ""


def structured_completion(
    model: str,
    messages: list,
    schema: type[BaseModel],
    max_retries: int = 2,
    **kwargs,
) -> BaseModel:
    """Structured output with auto-dispatch per model type.

    GPT on /chat/completions: response_format=Pydantic (native json_schema)
    GPT 5.4 family on /responses: litellm.aresponses() with text_format (native)
    Claude/Gemini/other: tool_choice (forced function call matching schema)

    Retries up to max_retries times on empty or unparseable responses.
    """
    model_lower = model.lower()
    last_error = None

    for attempt in range(1 + max_retries):
        try:
            if _needs_responses_api(model):
                # Responses API — native structured output via text_format
                user_content = messages[-1]["content"] if messages else ""
                timeout = kwargs.pop("timeout", 60.0) if attempt == 0 else kwargs.get("timeout", 60.0)
                resp = asyncio.run(litellm.aresponses(
                    model=model,
                    input=user_content,
                    text_format=schema,
                    max_output_tokens=kwargs.pop("max_tokens", 4096) if attempt == 0 else kwargs.get("max_tokens", 4096),
                    timeout=timeout,
                ))
                text = _extract_responses_text(resp.output)
                if not text.strip():
                    raise ValueError("Empty response from LLM")
                return schema.model_validate_json(text)

            use_native = "gpt" in model_lower or "o3" in model_lower or "o4" in model_lower

            if use_native:
                resp = litellm.completion(
                    model=model,
                    messages=messages,
                    response_format=schema,
                    **kwargs,
                )
                content = resp.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Empty response from LLM")
                return schema.model_validate_json(content)
            else:
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": schema.__name__,
                        "description": f"Return a {schema.__name__} object.",
                        "parameters": schema.model_json_schema(),
                    },
                }
                resp = litellm.completion(
                    model=model,
                    messages=messages,
                    tools=[tool_schema],
                    tool_choice={
                        "type": "function",
                        "function": {"name": schema.__name__},
                    },
                    **kwargs,
                )
                tool_calls = resp.choices[0].message.tool_calls
                if not tool_calls:
                    raise ValueError("No tool calls in LLM response")
                raw = tool_calls[0].function.arguments
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                return schema.model_validate(parsed)

        except (ValueError, Exception) as e:
            last_error = e
            if attempt < max_retries:
                click.echo(f"  ⚠ LLM response error (attempt {attempt + 1}): {e}, retrying...")
                continue
            raise RuntimeError(
                f"structured_completion failed after {1 + max_retries} attempts: {last_error}"
            ) from last_error


# ---------------------------------------------------------------------------
# generate_sample_set
# ---------------------------------------------------------------------------


def generate_sample_set(
    conn: sqlite3.Connection,
    n: int = 50,
    tone: str | None = None,  # "pop", "mainstream", "niche", or None
    remix_prob: float = 0.0,
    min_sim: float = 0.0,
    seed_base: int = 1000,
) -> list:
    """Generate *n* subtitles with the given parameters."""
    tone_target: dict[str, float] | None = None
    if tone:
        targets = get_tone_targets(conn)
        tone_target = {
            slot: targets[tone][slot]
            for slot in ["list_item", "action_noun", "of_object"]
        }

    results = []
    for i in range(n):
        sub = generate_subtitle(
            conn,
            seed=seed_base + i,
            tone_target=tone_target,
            remix_prob=remix_prob,
            min_sim=min_sim,
        )
        results.append(sub)
    return results


# ---------------------------------------------------------------------------
# rate_quality
# ---------------------------------------------------------------------------

RATING_PROMPT = """\
You are rating generated book subtitles for quality. Each subtitle follows \
the pattern "X, Y, and the Z of W".

Rate EACH subtitle on three dimensions (1-10 each):
- **Coherence**: Does it make grammatical and semantic sense? \
(10 = perfectly natural, 1 = word salad)
- **Evocativeness**: Does it evoke curiosity — would you pick up this book? \
(10 = instantly compelling, 1 = completely boring)
- **Surprise**: Does it pair unexpected concepts in an interesting way? \
(10 = delightfully unexpected, 1 = completely predictable)

Subtitles:
{subtitle_list}"""


def rate_batch_raw(
    subtitles: list[str],
    model: str = DEFAULT_RATER_MODEL,
    timeout: float = 60.0,
) -> list[SubtitleRating]:
    """Rate subtitles and return raw SubtitleRating objects (not normalized)."""
    chunk_size = 25
    all_ratings: list[SubtitleRating] = []

    for start in range(0, len(subtitles), chunk_size):
        chunk = subtitles[start : start + chunk_size]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
        prompt = RATING_PROMPT.format(subtitle_list=numbered)

        click.echo(
            f"  rating chunk {start // chunk_size + 1} "
            f"({len(chunk)} subtitles) …"
        )

        batch = structured_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema=RatingBatch,
            timeout=timeout,
        )
        all_ratings.extend(batch.ratings)

    return all_ratings


def rate_quality(
    subtitles: list[str],
    model: str = DEFAULT_RATER_MODEL,
    timeout: float = 60.0,
) -> float:
    """Rate a batch of subtitles via LLM.  Returns normalised average (0-1)."""
    if not subtitles:
        return 0.0

    ratings = rate_batch_raw(subtitles, model, timeout)
    total = sum(
        r.coherence + r.evocativeness + r.surprise for r in ratings
    )
    return total / (30.0 * len(ratings))


# ---------------------------------------------------------------------------
# measure_tone_separation
# ---------------------------------------------------------------------------

_SLOT_MAP = {
    "item1": "list_item",
    "item2": "list_item",
    "action_noun": "action_noun",
    "of_object": "of_object",
}


def _filler_log_freqs(
    conn: sqlite3.Connection,
    subtitles: list,
) -> list[float]:
    """Return blended filler scores for every filler in the subtitle list.

    Blends log10(1+freq) with popularity_score per pop_tone_blend config.
    """
    from subtitle_generator.config import load_tuning_config

    cfg = load_tuning_config(conn)
    blend_tone = cfg.get("pop_tone_blend", 0.0)
    pop_default = cfg.get("pop_missing_default", 0.1)

    scores: list[float] = []
    for sub in subtitles:
        for attr, slot_type in _SLOT_MAP.items():
            filler = getattr(sub, attr)
            row = conn.execute(
                "SELECT freq, popularity_score FROM slot_fillers WHERE filler = ? AND slot_type = ?",
                (filler, slot_type),
            ).fetchone()
            freq = row[0] if row else 0
            pop_score = (row[1] if row and row[1] is not None else pop_default)
            score_freq = math.log10(1 + freq)
            scores.append((1 - blend_tone) * score_freq + blend_tone * pop_score)
    return scores


def _histogram_overlap(a: list[float], b: list[float], bins: int = 10) -> float:
    """Compute histogram overlap coefficient between two distributions."""
    lo, hi = 0.0, 3.0
    bin_width = (hi - lo) / bins

    def _bin_counts(vals: list[float]) -> list[float]:
        counts = [0.0] * bins
        for v in vals:
            idx = int((v - lo) / bin_width)
            idx = max(0, min(bins - 1, idx))
            counts[idx] += 1
        total = sum(counts) or 1.0
        return [c / total for c in counts]

    ha = _bin_counts(a)
    hb = _bin_counts(b)
    return sum(min(x, y) for x, y in zip(ha, hb))


def measure_tone_separation(
    conn: sqlite3.Connection,
    n: int = 30,
    seed_base: int = 5000,
) -> float:
    """Distributional separation between pop and niche subtitles (0-1).

    1.0 = perfect separation, 0.0 = identical distributions.
    """
    click.echo(f"  generating {n} pop + {n} niche subtitles …")
    pop_subs = generate_sample_set(conn, n=n, tone="pop", seed_base=seed_base)
    niche_subs = generate_sample_set(
        conn, n=n, tone="niche", seed_base=seed_base + n
    )

    pop_scores = _filler_log_freqs(conn, pop_subs)
    niche_scores = _filler_log_freqs(conn, niche_subs)

    overlap = _histogram_overlap(pop_scores, niche_scores)
    return 1.0 - overlap


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------


def composite_score(
    quality: float,
    separation: float,
    quality_weight: float = 0.5,
) -> float:
    """Weighted average of quality and tone-separation scores."""
    return quality_weight * quality + (1.0 - quality_weight) * separation


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    click.echo("=== eval_harness smoke test ===")

    # Quick schema round-trip
    r = SubtitleRating(coherence=7, evocativeness=8, surprise=6)
    b = RatingBatch(ratings=[r])
    click.echo(f"RatingBatch JSON: {b.model_dump_json()}")

    p = ParamProposal(param="remix_prob", new_value=0.5, reasoning="test")
    click.echo(f"ParamProposal JSON: {p.model_dump_json()}")

    click.echo(f"composite_score(0.8, 0.6) = {composite_score(0.8, 0.6)}")
    click.echo("smoke test passed ✓")
