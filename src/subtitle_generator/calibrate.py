"""LLM-based calibration of remix parameters (remix_prob and min_sim).

Uses real subtitle embeddings as a probability anchor: remixed of-objects
are naturally more whimsical, but should stay within reasonable distance
of the real subtitle embedding space.

Two-phase calibration:
  Phase 1: Sweep min_sim (embedding threshold) at remix_prob=1.0
  Phase 2: Sweep remix_prob at the optimal min_sim from Phase 1
"""

import asyncio
import json
import re
import sqlite3

import click
import numpy as np

from subtitle_generator.generate import generate_subtitle, _load_remix_context


RATING_PROMPT = """\
You are rating generated book subtitles for quality. Each subtitle follows \
the pattern "X, Y, and the Z of W" where W is the of-object.

Rate this subtitle on three dimensions (1-10 each):
- **Coherence**: Does the of-object make grammatical and semantic sense? \
(10 = perfectly natural, 1 = word salad)
- **Evocativeness**: Does it evoke curiosity — would you pick up this book? \
(10 = instantly compelling, 1 = completely boring)
- **Surprise**: Does it pair unexpected concepts in an interesting way? \
(10 = delightfully unexpected, 1 = completely predictable)

Subtitle: "{subtitle}"

Respond with ONLY a JSON object, no other text:
{{"coherence": <1-10>, "evocativeness": <1-10>, "surprise": <1-10>}}"""


def _parse_rating(text: str) -> dict[str, int] | None:
    """Extract rating JSON from LLM response."""
    # Try to find JSON in the response
    match = re.search(r"\{[^}]+\}", text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if all(k in data for k in ("coherence", "evocativeness", "surprise")):
            return {k: int(data[k]) for k in ("coherence", "evocativeness", "surprise")}
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _compute_subtitle_centroid(conn: sqlite3.Connection, nlp) -> np.ndarray | None:
    """Compute centroid from REAL matched subtitles (not just of-objects).

    This serves as the embedding anchor: real subtitles define "normal",
    and remixed ones are expected to drift further but should stay in the
    same general region of vector space.
    """
    rows = conn.execute(
        "SELECT subtitle FROM pattern_matches ORDER BY RANDOM() LIMIT 2000"
    ).fetchall()
    if not rows:
        return None

    vectors = []
    for (subtitle,) in rows:
        doc = nlp(subtitle)
        if doc.has_vector and doc.vector_norm > 0:
            vectors.append(doc.vector)

    return np.mean(vectors, axis=0) if vectors else None


def _compute_baseline_stats(
    conn: sqlite3.Connection, nlp, subtitle_centroid: np.ndarray, n: int = 100,
) -> dict:
    """Compute embedding similarity stats for non-remixed subtitles.

    This gives us the baseline: how similar are normal (atomic) subtitles
    to the real-subtitle centroid? Remixed ones will be slightly lower.
    """
    sims = []
    for i in range(n):
        sub = generate_subtitle(conn, seed=9000 + i, remix_prob=0.0)
        doc = nlp(sub.text)
        if doc.has_vector and doc.vector_norm > 0:
            norm1 = np.linalg.norm(subtitle_centroid)
            norm2 = np.linalg.norm(doc.vector)
            if norm1 > 0 and norm2 > 0:
                sim = float(np.dot(subtitle_centroid, doc.vector) / (norm1 * norm2))
                sims.append(sim)

    if not sims:
        return {"mean": 0, "std": 0, "min": 0, "p10": 0}
    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "p10": float(np.percentile(sims, 10)),
    }


async def _rate_subtitle(session, subtitle: str) -> dict[str, int] | None:
    """Rate a single subtitle via LLM."""
    prompt = RATING_PROMPT.format(subtitle=subtitle)
    try:
        result = await session.send_and_wait(prompt, timeout=30.0)
        if result and result.data and result.data.content:
            return _parse_rating(result.data.content)
    except Exception:
        pass
    return None


async def _rate_batch(subtitles: list[str], model: str) -> list[dict]:
    """Rate a batch of subtitles sequentially via LLM."""
    from copilot import CopilotClient
    from copilot.session import PermissionHandler

    ratings = []
    async with CopilotClient() as client:
        async with await client.create_session(
            on_permission_request=PermissionHandler.approve_all,
            model=model,
            infinite_sessions={"enabled": False},
        ) as session:
            for subtitle in subtitles:
                rating = await _rate_subtitle(session, subtitle)
                if rating:
                    ratings.append(rating)
                else:
                    ratings.append({"coherence": 0, "evocativeness": 0, "surprise": 0})
    return ratings


def _avg_score(ratings: list[dict]) -> float:
    """Compute average across all three dimensions."""
    if not ratings:
        return 0.0
    total = sum(
        r["coherence"] + r["evocativeness"] + r["surprise"]
        for r in ratings if r["coherence"] > 0
    )
    valid = sum(1 for r in ratings if r["coherence"] > 0)
    return total / (valid * 3) if valid > 0 else 0.0


def _dimension_avgs(ratings: list[dict]) -> dict[str, float]:
    """Compute per-dimension averages."""
    valid = [r for r in ratings if r["coherence"] > 0]
    if not valid:
        return {"coherence": 0, "evocativeness": 0, "surprise": 0}
    return {
        "coherence": sum(r["coherence"] for r in valid) / len(valid),
        "evocativeness": sum(r["evocativeness"] for r in valid) / len(valid),
        "surprise": sum(r["surprise"] for r in valid) / len(valid),
    }


def run_calibration(
    conn: sqlite3.Connection,
    samples: int = 15,
    model: str = "gpt-5.4-mini",
):
    """Two-phase LLM calibration of remix_prob and min_sim."""
    click.echo("=== Remix Calibration ===\n")

    # Load remix context (spaCy model, of-object centroid)
    click.echo("Loading spaCy model and computing centroids...")
    ctx = _load_remix_context(conn)
    nlp = ctx["nlp"]

    # Compute REAL subtitle centroid (the anchor)
    subtitle_centroid = _compute_subtitle_centroid(conn, nlp)
    if subtitle_centroid is None:
        click.echo("ERROR: No pattern matches found. Run build-slots first.")
        return

    # Compute baseline stats (non-remixed subtitles vs real-subtitle centroid)
    click.echo("Computing baseline embedding stats (non-remixed)...")
    baseline = _compute_baseline_stats(conn, nlp, subtitle_centroid)
    click.echo(f"  Baseline similarity to real subtitles: "
               f"mean={baseline['mean']:.3f}, std={baseline['std']:.3f}, "
               f"p10={baseline['p10']:.3f}")
    click.echo(f"  (Remixed subtitles will be somewhat lower — that's expected)\n")

    # --- Phase 1: Sweep min_sim at remix_prob=1.0 ---
    click.echo("Phase 1: Finding optimal embedding threshold (min_sim)")
    click.echo(f"  Fixed remix_prob=1.0, sweeping min_sim, {samples} samples per level\n")

    min_sim_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    phase1_results = {}

    for min_sim in min_sim_levels:
        subtitles = []
        remix_count = 0
        for i in range(samples):
            sub = generate_subtitle(conn, seed=3000 + i, remix_prob=1.0, min_sim=min_sim)
            subtitles.append(sub.text)
            if sub.remixed:
                remix_count += 1

        remix_rate = remix_count / samples
        click.echo(f"  min_sim={min_sim:.2f}: generated {samples}, "
                    f"remixed={remix_count} ({remix_rate:.0%}), rating...")

        ratings = asyncio.run(_rate_batch(subtitles, model))
        avg = _avg_score(ratings)
        dims = _dimension_avgs(ratings)
        phase1_results[min_sim] = {
            "avg": avg, "dims": dims, "remix_rate": remix_rate,
        }
        click.echo(f"    avg={avg:.1f}  C={dims['coherence']:.1f} "
                    f"E={dims['evocativeness']:.1f} S={dims['surprise']:.1f}")

    # Find optimal min_sim: best average score
    best_min_sim = max(phase1_results, key=lambda k: phase1_results[k]["avg"])
    click.echo(f"\n  → Best min_sim: {best_min_sim:.2f} "
               f"(avg={phase1_results[best_min_sim]['avg']:.1f})\n")

    # --- Phase 2: Sweep remix_prob at optimal min_sim ---
    click.echo("Phase 2: Finding optimal remix probability (remix_prob)")
    click.echo(f"  Fixed min_sim={best_min_sim:.2f}, sweeping remix_prob, "
               f"{samples} samples per level\n")

    remix_prob_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    phase2_results = {}

    for remix_prob in remix_prob_levels:
        subtitles = []
        remix_count = 0
        for i in range(samples):
            sub = generate_subtitle(conn, seed=4000 + i, remix_prob=remix_prob, min_sim=best_min_sim)
            subtitles.append(sub.text)
            if sub.remixed:
                remix_count += 1

        remix_rate = remix_count / samples
        click.echo(f"  remix_prob={remix_prob:.1f}: generated {samples}, "
                    f"remixed={remix_count} ({remix_rate:.0%}), rating...")

        ratings = asyncio.run(_rate_batch(subtitles, model))
        avg = _avg_score(ratings)
        dims = _dimension_avgs(ratings)
        phase2_results[remix_prob] = {
            "avg": avg, "dims": dims, "remix_rate": remix_rate,
        }
        click.echo(f"    avg={avg:.1f}  C={dims['coherence']:.1f} "
                    f"E={dims['evocativeness']:.1f} S={dims['surprise']:.1f}")

    best_remix_prob = max(phase2_results, key=lambda k: phase2_results[k]["avg"])
    click.echo(f"\n  → Best remix_prob: {best_remix_prob:.1f} "
               f"(avg={phase2_results[best_remix_prob]['avg']:.1f})\n")

    # --- Store results ---
    click.echo("Storing calibration results in config table...")
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        ("remix_calibrated_min_sim", str(best_min_sim)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        ("remix_calibrated_remix_prob", str(best_remix_prob)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        ("remix_baseline_sim", json.dumps(baseline)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        ("remix_phase1_results", json.dumps({str(k): v for k, v in phase1_results.items()})),
    )
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
        ("remix_phase2_results", json.dumps({str(k): v for k, v in phase2_results.items()})),
    )
    conn.commit()

    click.echo(f"\n=== Calibration Complete ===")
    click.echo(f"  Optimal min_sim:    {best_min_sim:.2f}")
    click.echo(f"  Optimal remix_prob: {best_remix_prob:.1f}")
    click.echo(f"  Baseline sim:       {baseline['mean']:.3f} (±{baseline['std']:.3f})")
    click.echo(f"\nThese values are now the defaults for 'generate --remix'.")
