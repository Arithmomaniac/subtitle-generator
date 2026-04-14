"""LLM-based calibration of remix parameters (remix_prob and min_sim).

Uses real subtitle embeddings as a probability anchor: remixed of-objects
are naturally more whimsical, but should stay within reasonable distance
of the real subtitle embedding space.

Two-phase calibration:
  Phase 1: Sweep min_sim (embedding threshold) at remix_prob=1.0
  Phase 2: Sweep remix_prob at the optimal min_sim from Phase 1
"""

import json
import sqlite3

import click
import numpy as np

from subtitle_generator.eval_harness import (
    RATING_PROMPT,
    DEFAULT_RATER_MODEL,
    RatingBatch,
    SubtitleRating,
    structured_completion,
)
from subtitle_generator.generate import generate_subtitle, _load_remix_context


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


def _rate_batch_raw(
    subtitles: list[str], model: str, timeout: float = 60.0,
) -> list[SubtitleRating]:
    """Rate subtitles via structured_completion, chunking at 25."""
    chunk_size = 25
    all_ratings: list[SubtitleRating] = []
    for start in range(0, len(subtitles), chunk_size):
        chunk = subtitles[start : start + chunk_size]
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
        prompt = RATING_PROMPT.format(subtitle_list=numbered)
        batch = structured_completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            schema=RatingBatch,
            timeout=timeout,
        )
        all_ratings.extend(batch.ratings)
    return all_ratings


def run_calibration(
    conn: sqlite3.Connection,
    samples: int = 50,
    model: str = DEFAULT_RATER_MODEL,
):
    """Two-phase LLM calibration of remix_prob and min_sim.
    
    Each level generates `samples` subtitles and rates them all in a single
    LLM call (cheap — one call per level, not per subtitle).
    """
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

        ratings = _rate_batch_raw(subtitles, model)
        avg = sum(r.coherence + r.evocativeness + r.surprise for r in ratings) / (3 * len(ratings))
        dims = {
            "coherence": sum(r.coherence for r in ratings) / len(ratings),
            "evocativeness": sum(r.evocativeness for r in ratings) / len(ratings),
            "surprise": sum(r.surprise for r in ratings) / len(ratings),
        }
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

        ratings = _rate_batch_raw(subtitles, model)
        avg = sum(r.coherence + r.evocativeness + r.surprise for r in ratings) / (3 * len(ratings))
        dims = {
            "coherence": sum(r.coherence for r in ratings) / len(ratings),
            "evocativeness": sum(r.evocativeness for r in ratings) / len(ratings),
            "surprise": sum(r.surprise for r in ratings) / len(ratings),
        }
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
