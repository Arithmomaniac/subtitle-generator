"""Generate bizarre subtitles by randomly combining slot fillers."""

import json
import math
import random
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

import click
import inflect
from titlecase import titlecase as _lib_titlecase

from subtitle_generator.config import load_tuning_config
from subtitle_generator.remix_state import (
    RemixRuntimeContext,
    assert_remix_precompute_state,
)
from subtitle_generator.shadow_runtime import (
    GenerationRuntimeSelection,
    PreparedGenerationRuntime,
    RuntimeSelectionMode,
    prepare_generation_runtime,
    sample_shadow_candidates,
)

_inflect_engine = inflect.engine()
MAX_TIER_FILTER_ATTEMPTS = 1000
DEFAULT_GENERATION_TIER_ATTEMPTS = 25
_DEFAULT_GENERATION_TIERS = ("pop", "mainstream", "niche")
_MODEL_SCORE_INDEX = {"pop": 3, "mainstream": 4, "niche": 5}
_BAD_ONE_WORD_OBJECTS = {"christian", "imf"}
_BAD_STANDALONE_FILLERS = {"jr", "sr", "xcalibur"}


class TierFilterError(RuntimeError):
    """Raised when generation cannot satisfy a requested tier filter."""


def _title_case(text: str) -> str:
    """Title-case a subtitle using the titlecase library."""
    return _lib_titlecase(text)


def _fix_a_an(article: str, next_word: str) -> str:
    """Correct a/an using inflect's phonetic analysis."""
    if article not in ("a", "an") or not next_word:
        return article
    result = _inflect_engine.a(next_word)
    return result.split()[0]


@dataclass
class GeneratedSubtitle:
    """A generated subtitle with its component fillers."""
    text: str
    item1: str
    item2: str
    action_noun: str
    of_object: str
    remixed: bool = False
    remix_parts: dict = field(default_factory=dict)
    remix_similarity: float | None = None
    of_article: str = ""
    action_article: str = "the"


@dataclass(frozen=True)
class GenerationCandidates:
    list_rows: list[tuple]
    action_rows: list[tuple]
    obj_rows: list[tuple]


@dataclass
class SelectedSubtitleParts:
    items: list[str]
    action_noun: str
    of_object: str
    remixed: bool
    remix_parts: dict
    remix_similarity: float | None


def _weighted_sample(
    rows: list[tuple], k: int,
    rng: random.Random | None = None,
    model_tier: str | None = None,
) -> list[str]:
    """Pick k unique fillers weighted by learned tier score and sqrt(freq).

    Rows can be (filler, freq), (filler, freq, popularity_score), or
    (filler, freq, popularity_score, score_pop, score_mainstream, score_niche).
    When ``model_tier`` is present and model scores are available, use learned
    tier categorization as the sampling signal while keeping frequency as the
    within-category stabilizer.
    """
    fillers = [r[0] for r in rows]

    weights = []
    for r in rows:
        freq = r[1]
        w_freq = math.sqrt(freq)
        model_score = _row_model_score(r, model_tier)
        if model_score is not None:
            weights.append(w_freq * max(model_score, 0.001))
        else:
            weights.append(w_freq)

    chosen = []
    # Weighted sampling without replacement
    for _ in range(k):
        pick = (rng or random).choices(fillers, weights=weights, k=1)[0]
        idx = fillers.index(pick)
        chosen.append(pick)
        fillers.pop(idx)
        weights.pop(idx)
    return chosen


def _sample_slot_rows(
    slot_type: str,
    rows: list[tuple],
    count: int,
    rng: random.Random | None,
    *,
    model_tier: str | None,
    runtime: PreparedGenerationRuntime | None,
) -> list[str]:
    if runtime is not None and runtime.mode == RuntimeSelectionMode.SHADOW:
        if model_tier is None:
            raise RuntimeError(
                f"Shadow runtime requires an explicit tier for slot_type {slot_type!r}"
            )
        return sample_shadow_candidates(
            runtime,
            slot_type=slot_type,
            tier=model_tier,
            candidate_rows=rows,
            count=count,
            rng=rng,
        )
    return _weighted_sample(rows, count, rng, model_tier=model_tier)


def _row_model_score(row: tuple, model_tier: str | None) -> float | None:
    if model_tier not in _MODEL_SCORE_INDEX or len(row) <= _MODEL_SCORE_INDEX[model_tier]:
        return None
    value = row[_MODEL_SCORE_INDEX[model_tier]]
    if value is None:
        return None
    return float(value)


# --- Remix infrastructure ---

# Module-level cache for remix context (lazy-loaded)
_remix_ctx: RemixRuntimeContext | None = None

# Sentinel for embedding precompute version checks
_EMBEDDING_VERSION = "2"

# Slot types that need pre-computed vectors for remix composition
_REMIX_VECTOR_SLOT_TYPES = frozenset({
    "of_modifier", "of_head", "of_topic", "of_complement",
})


def precompute_remix_data(conn: sqlite3.Connection) -> dict:
    """Pre-compute remix classifications and word vectors, storing in DB.

    This runs spaCy en_core_web_md to:
    1. Classify each of_object strict filler for remix type (type1/type2)
    2. Compute vector_sum + token_count for remix-relevant fillers
    3. Compute centroid and derive scalar decomposition for runtime
       (centroid_dot, norm_sq per filler; centroid_norm, avg_cross_sim constants)

    After this, runtime code needs no numpy or vector math — only scalar arithmetic.
    Returns stats dict.
    """
    import numpy as np
    import spacy

    click.echo("Loading spaCy en_core_web_md for vector precomputation...")
    nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])

    stats: dict[str, int] = {"classified": 0, "vectorized": 0, "skipped_oov": 0}

    # 1. Classify of_object strict fillers and compute vectors
    of_obj_rows = conn.execute(
        "SELECT id, filler FROM slot_fillers WHERE slot_type = 'of_object' AND mode = 'strict'"
    ).fetchall()
    click.echo(f"Classifying {len(of_obj_rows)} of_object fillers...")
    centroid_vectors = []
    # Store (filler_id, vec_sum, tc) for later scalar computation
    obj_vectors: list[tuple[int, object, int]] = []
    for filler_id, filler in of_obj_rows:
        doc = nlp(filler)
        classification = _classify_for_remix(filler, doc)

        remix_type = None
        remix_prep = None
        remix_wc = None
        if classification is not None:
            remix_type = classification[0]
            if remix_type == "type2":
                _, remix_prep, remix_wc = classification
            else:
                _, remix_wc = classification

        # Compute vector for this of_object filler
        tokens = [t for t in doc if not t.is_space]
        token_vecs = [t.vector for t in tokens if t.has_vector and np.linalg.norm(t.vector) > 0]
        if token_vecs:
            vec_sum = np.sum(token_vecs, axis=0).astype(np.float32)
            tc = len(token_vecs)
            centroid_vectors.append(vec_sum / tc)  # mean for centroid
            obj_vectors.append((filler_id, vec_sum, tc))
            conn.execute(
                "UPDATE slot_fillers SET remix_type = ?, remix_prep = ?, remix_word_count = ?, "
                "vector_sum = ?, token_count = ? WHERE id = ?",
                (remix_type, remix_prep, remix_wc, vec_sum.tobytes(), tc, filler_id),
            )
            stats["classified"] += 1
        else:
            conn.execute(
                "UPDATE slot_fillers SET remix_type = ?, remix_prep = ?, remix_word_count = ? WHERE id = ?",
                (remix_type, remix_prep, remix_wc, filler_id),
            )
            stats["skipped_oov"] += 1

    # 2. Compute vectors for remix sub-part fillers
    sub_rows = conn.execute(
        "SELECT id, slot_type, filler FROM slot_fillers "
        "WHERE slot_type IN ('of_modifier', 'of_head', 'of_topic', 'of_complement') AND mode = 'strict'"
    ).fetchall()
    click.echo(f"Computing vectors for {len(sub_rows)} remix sub-part fillers...")
    # Collect vectors by slot_type for cross-sim computation
    sub_vectors: dict[str, list[tuple[int, object, int]]] = {
        "of_modifier": [], "of_head": [], "of_topic": [], "of_complement": [],
    }
    for filler_id, slot_type, filler in sub_rows:
        doc = nlp(filler)
        tokens = [t for t in doc if not t.is_space]
        token_vecs = [t.vector for t in tokens if t.has_vector and np.linalg.norm(t.vector) > 0]
        if token_vecs:
            vec_sum = np.sum(token_vecs, axis=0).astype(np.float32)
            tc = len(token_vecs)
            conn.execute(
                "UPDATE slot_fillers SET vector_sum = ?, token_count = ? WHERE id = ?",
                (vec_sum.tobytes(), tc, filler_id),
            )
            sub_vectors[slot_type].append((filler_id, vec_sum, tc))
            stats["vectorized"] += 1
        else:
            stats["skipped_oov"] += 1

    # 3. Compute centroid and scalar decomposition
    if centroid_vectors:
        import random as _rng

        centroid = np.mean(centroid_vectors, axis=0).astype(np.float32)
        centroid_norm = float(np.linalg.norm(centroid))

        # Compute centroid_dot and norm_sq for all fillers with vectors
        all_vec_entries = obj_vectors[:]
        for entries in sub_vectors.values():
            all_vec_entries.extend(entries)

        for filler_id, vec_sum, tc in all_vec_entries:
            cd = float(np.dot(vec_sum, centroid))
            ns = float(np.dot(vec_sum, vec_sum))
            conn.execute(
                "UPDATE slot_fillers SET centroid_dot = ?, norm_sq = ? WHERE id = ?",
                (cd, ns, filler_id),
            )

        # Compute type-specific average cross-similarity constants
        def _sample_cross_sim(pool_a, pool_b, n_samples=3000):
            if not pool_a or not pool_b:
                return 0.0
            dots = []
            for _ in range(min(n_samples, len(pool_a) * len(pool_b))):
                _, va, _ = _rng.choice(pool_a)
                _, vb, _ = _rng.choice(pool_b)
                na = float(np.linalg.norm(va))
                nb = float(np.linalg.norm(vb))
                if na > 0 and nb > 0:
                    dots.append(float(np.dot(va, vb)) / (na * nb))
            return float(np.mean(dots)) if dots else 0.0

        _rng.seed(42)
        avg_cross_t1 = _sample_cross_sim(sub_vectors["of_modifier"], sub_vectors["of_head"])
        avg_cross_t2 = _sample_cross_sim(sub_vectors["of_topic"], sub_vectors["of_complement"])

        # Store scalar constants in config
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('centroid_norm', ?)",
            (str(centroid_norm),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('avg_cross_sim_t1', ?)",
            (str(avg_cross_t1),),
        )
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('avg_cross_sim_t2', ?)",
            (str(avg_cross_t2),),
        )
        # Keep centroid BLOB for dev fallback path
        import base64
        centroid_b64 = base64.b64encode(centroid.tobytes()).decode("ascii")
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES ('embedding_centroid', ?)",
            (centroid_b64,),
        )

        click.echo(
            f"Scalar decomposition: centroid_norm={centroid_norm:.4f}, "
            f"avg_cross_sim_t1={avg_cross_t1:.4f}, avg_cross_sim_t2={avg_cross_t2:.4f}"
        )

    # Store version marker
    conn.execute(
        "INSERT OR REPLACE INTO config (key, value) VALUES ('embedding_version', ?)",
        (_EMBEDDING_VERSION,),
    )
    conn.commit()

    # Invalidate cached context
    global _remix_ctx
    _remix_ctx = None

    click.echo(
        f"Precomputed: {stats['classified']} classified, "
        f"{stats['vectorized']} sub-part vectors, "
        f"{stats['skipped_oov']} OOV skipped"
    )
    return stats


def _load_remix_context(conn: sqlite3.Connection) -> RemixRuntimeContext:
    """Lazy-load remix context from pre-computed scalar decomposition in DB.

    Requires precompute_remix_data() to have been run first (version 2+).
    Falls back to spaCy if pre-computed data is missing (dev convenience only).
    """
    global _remix_ctx
    if _remix_ctx is not None:
        return _remix_ctx

    # Check for pre-computed embeddings
    row = conn.execute("SELECT value FROM config WHERE key = 'embedding_version'").fetchone()
    if row is not None and int(row[0]) >= 2:
        assert_remix_precompute_state(conn, _EMBEDDING_VERSION)
        # Version 2+: scalar decomposition (no numpy needed)
        centroid_norm_row = conn.execute(
            "SELECT value FROM config WHERE key = 'centroid_norm'"
        ).fetchone()
        avg_cross_t1_row = conn.execute(
            "SELECT value FROM config WHERE key = 'avg_cross_sim_t1'"
        ).fetchone()
        avg_cross_t2_row = conn.execute(
            "SELECT value FROM config WHERE key = 'avg_cross_sim_t2'"
        ).fetchone()

        if not all([centroid_norm_row, avg_cross_t1_row, avg_cross_t2_row]):
            raise RuntimeError(
                "DB has embedding_version >= 2 but missing scalar constants. "
                "Re-run 'precompute-vectors' to regenerate."
            )

        centroid_norm = float(centroid_norm_row[0])
        avg_cross_sim_t1 = float(avg_cross_t1_row[0])
        avg_cross_sim_t2 = float(avg_cross_t2_row[0])

        # Build filler → (centroid_dot, norm_sq) lookup
        filler_scalars: dict[tuple[str, str], tuple[float, float]] = {}
        scalar_rows = conn.execute(
            "SELECT slot_type, filler, centroid_dot, norm_sq FROM slot_fillers "
            "WHERE centroid_dot IS NOT NULL AND norm_sq IS NOT NULL"
        ).fetchall()
        for slot_type, filler, cd, ns in scalar_rows:
            filler_scalars[(slot_type, filler)] = (cd, ns)

        # Load config (remix POS distributions etc.)
        config = {}
        for key, value in conn.execute("SELECT key, value FROM config WHERE key LIKE 'remix_%'"):
            config[key] = json.loads(value)

        # Load article statistics
        article_stats_of = {}
        article_stats_action = {}
        for key, value in conn.execute(
            "SELECT key, value FROM config WHERE key LIKE 'article_stats_%'"
        ):
            parsed = json.loads(value)
            if key == "article_stats_of_object":
                article_stats_of = parsed
            elif key == "article_stats_action_noun":
                article_stats_action = parsed

        _remix_ctx = RemixRuntimeContext(
            precomputed=True,
            centroid_norm=centroid_norm,
            avg_cross_sim_t1=avg_cross_sim_t1,
            avg_cross_sim_t2=avg_cross_sim_t2,
            filler_scalars=filler_scalars,
            config=config,
            article_stats_of=article_stats_of,
            article_stats_action=article_stats_action,
        )
        return _remix_ctx

    if row is not None:
        # Version 1: old format — can't use without numpy/vectors
        click.echo(
            "Warning: DB has embedding_version=1 (old format). "
            "Re-run 'precompute-vectors' to upgrade to scalar decomposition.",
            err=True,
        )

    # Fallback: live spaCy (for dev when precompute hasn't been run)
    import numpy as np
    import spacy

    click.echo("Warning: using live spaCy (run 'precompute-vectors' for better performance)", err=True)
    nlp = spacy.load("en_core_web_md", disable=["lemmatizer"])

    rows = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = 'of_object' AND mode = 'strict'"
    ).fetchall()
    vectors = []
    for (filler,) in rows:
        doc = nlp(filler)
        if doc.has_vector and doc.vector_norm > 0:
            vectors.append(doc.vector)
    centroid = np.mean(vectors, axis=0) if vectors else None

    config = {}
    for key, value in conn.execute("SELECT key, value FROM config WHERE key LIKE 'remix_%'"):
        config[key] = json.loads(value)

    # Load article statistics (same as precomputed path)
    article_stats_of = {}
    article_stats_action = {}
    for key, value in conn.execute(
        "SELECT key, value FROM config WHERE key LIKE 'article_stats_%'"
    ):
        parsed = json.loads(value)
        if key == "article_stats_of_object":
            article_stats_of = parsed
        elif key == "article_stats_action_noun":
            article_stats_action = parsed

    _remix_ctx = RemixRuntimeContext(
        precomputed=False,
        nlp=nlp,
        centroid=centroid,
        config=config,
        article_stats_of=article_stats_of,
        article_stats_action=article_stats_action,
    )
    return _remix_ctx


def _approx_cosine_sim(parts: dict, ctx: dict, remix_type: str) -> float | None:
    """Compute approximate cosine similarity using scalar decomposition.

    Uses pre-computed centroid_dot and norm_sq per filler with a cross-term
    correction to approximate what full vector cosine similarity would give.

    Returns similarity score, or None if insufficient data.
    """
    import math

    _role_to_slot = {
        "modifier": "of_modifier",
        "head": "of_head",
        "topic": "of_topic",
        "complement": "of_complement",
    }

    filler_scalars = ctx["filler_scalars"]
    centroid_norm = ctx["centroid_norm"]
    avg_cross_sim = ctx["avg_cross_sim_t1"] if remix_type == "type1" else ctx["avg_cross_sim_t2"]

    total_dot = 0.0
    norms_sq: list[float] = []

    for role, filler in parts.items():
        if role == "prep":
            continue  # Prep vectors are never stored as separate fillers
        slot_type = _role_to_slot.get(role)
        if slot_type is None:
            continue
        key = (slot_type, filler)
        if key not in filler_scalars:
            return None  # Missing data — skip coherence check
        cd, ns = filler_scalars[key]
        total_dot += cd
        norms_sq.append(ns)

    if not norms_sq or centroid_norm == 0:
        return None

    # Cross-term correction: sum of 2 * sqrt(ns_i) * sqrt(ns_j) * avg_cross_sim for all pairs
    cross_correction = 0.0
    for i in range(len(norms_sq)):
        for j in range(i + 1, len(norms_sq)):
            cross_correction += 2 * math.sqrt(norms_sq[i]) * math.sqrt(norms_sq[j]) * avg_cross_sim

    denom_sq = sum(norms_sq) + cross_correction
    if denom_sq <= 0:
        return None

    return total_dot / (math.sqrt(denom_sq) * centroid_norm)


def _classify_for_remix(phrase: str, doc) -> tuple[str, int] | tuple[str, str, int] | None:
    """Classify an atomic of-object for remixing.

    Returns:
        ("type1", word_count) for compound NPs
        ("type2", prep, word_count) for prepositional NPs
        None if not remixable
    """
    words = phrase.split()
    word_count = len(words)
    tokens = [t for t in doc if not t.is_space]

    # 1-word or 4+ word compound: never remix
    if word_count == 1:
        return None

    # Check for preposition → Type 2
    for t in tokens:
        if t.pos_ == "ADP":
            prep = t.text.lower()
            return ("type2", prep, word_count)

    # 2-3 word compound → Type 1 (if not excluded)
    if word_count in (2, 3):
        # Check exclusions (same as _decompose_compound)
        if any(e.label_ == "PERSON" for e in doc.ents):
            return None
        roots = [t for t in tokens if t.dep_ == "ROOT"]
        if not roots or roots[0].pos_ not in ("NOUN", "PROPN"):
            return None
        if roots[0] != tokens[-1]:
            return None
        if word_count == 2 and all(t.pos_ == "NOUN" for t in tokens):
            return None
        if word_count == 2:
            for ent in doc.ents:
                if ent.label_ == "GPE" and ent.start == 0 and ent.end == len(tokens):
                    return None
        return ("type1", word_count)

    # 4+ word without prep: skip
    return None


def compose_compound(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    ctx: dict,
    word_count: int,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> tuple[str, dict] | None:
    """Compose a Type 1 remixed of-object (modifier + head).

    Returns (composed_text, parts_dict) or None if composition fails.
    """
    mod_word_count = word_count - 1  # head is always 1 word
    mod_space_count = mod_word_count - 1
    # Get modifier POS distribution for this bucket
    config_key = f"remix_mod_pos_{word_count}word"
    mod_pos_weights = ctx["config"].get(config_key, {})
    if not mod_pos_weights:
        return None

    # Sample a modifier POS tag
    pos_tags = list(mod_pos_weights.keys())
    pos_freqs = list(mod_pos_weights.values())
    chosen_mod_pos = (rng or random).choices(pos_tags, weights=pos_freqs, k=1)[0]

    # Draw modifier with matching POS and word count
    mod_rows = conn.execute(
        _slot_sampling_select(
            "slot_type = 'of_modifier' AND pos_tag = ? "
            "AND length(filler) - length(replace(filler, ' ', '')) = ? "
            "AND mode = 'strict'",
            include_model_scores=_has_model_scores(conn),
        ),
        (chosen_mod_pos, mod_space_count),
    ).fetchall()
    if not mod_rows:
        return None
    modifier = _sample_slot_rows(
        "of_modifier",
        mod_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]

    head_rows = conn.execute(
        _slot_sampling_select(
            "slot_type = 'of_head' AND mode = 'strict'",
            include_model_scores=_has_model_scores(conn),
        ),
    ).fetchall()
    if not head_rows:
        return None
    head = _sample_slot_rows(
        "of_head",
        head_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]

    composed = f"{modifier} {head}"
    parts = {"modifier": modifier, "head": head}
    return composed, parts


def compose_prepositional(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    ctx: dict,
    prep: str,
    word_count: int,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> tuple[str, dict] | None:
    """Compose a Type 2 remixed of-object (topic + prep + complement).

    Returns (composed_text, parts_dict) or None if composition fails.
    Enforces strict bucket word-count matching.
    """
    topic_rows = conn.execute(
        _slot_sampling_select(
            "slot_type = 'of_topic' AND prep = ? AND mode = 'strict'",
            include_model_scores=_has_model_scores(conn),
        ),
        (prep,),
    ).fetchall()
    if not topic_rows:
        return None
    topic = _sample_slot_rows(
        "of_topic",
        topic_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]

    comp_rows = conn.execute(
        _slot_sampling_select(
            "slot_type = 'of_complement' AND prep = ? AND mode = 'strict'",
            include_model_scores=_has_model_scores(conn),
        ),
        (prep,),
    ).fetchall()
    if not comp_rows:
        return None
    complement = _sample_slot_rows(
        "of_complement",
        comp_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]

    composed = f"{topic} {prep} {complement}"

    # Strict bucket: verify word count matches.
    if len(composed.split()) != word_count:
        return None

    parts = {"topic": topic, "prep": prep, "complement": complement}
    return composed, parts


def _majority_article(
    filler: str, article_stats: dict[str, dict[str, int]], min_freq: float,
) -> str:
    """Look up the majority article for a filler from corpus stats.

    Returns the most frequent article ("the"/"a"/"an"/"") if total
    occurrences meet min_freq and majority is clear (>50%), otherwise
    returns the fallback.
    """
    counts = article_stats.get(filler.lower())
    if not counts:
        return ""
    total = sum(counts.values())
    if total < min_freq:
        return ""
    best = max(counts, key=counts.get)
    # Require clear majority (>50%) to avoid unstable ties
    if counts[best] * 2 <= total:
        return ""
    return best


def _article_with_backoff(
    filler: str, article_stats: dict[str, dict[str, int]], min_freq: float,
) -> str:
    """Article lookup with last-word fallback for non-remixed of_objects.

    Backoff chain:
      1. Exact filler match → majority article
      2. Last word of multi-word filler → its majority article
      3. Default → "" (no article)
    """
    result = _majority_article(filler, article_stats, min_freq)
    if result:
        return result

    words = filler.split()
    if len(words) > 1:
        result = _majority_article(words[-1], article_stats, min_freq)
        if result:
            return result

    return ""


def _infer_of_article(
    composed: str, article_stats: dict[str, dict[str, int]],
    min_freq: float, threshold: float,
    remix_parts: dict | None = None,
) -> str:
    """Deterministic head-noun backoff heuristic for remixed of-objects.

    Backoff chain:
      1. Exact composed phrase in stats → majority article
      2. Head noun from remix structure → its majority article
         (Type 1: uses 'head', Type 2: uses 'topic' — the syntactic head)
      3. Default → "" (no article)

    Only assigns an article if the majority fraction >= threshold.
    """
    key = composed.lower()
    # 1. Exact match
    counts = article_stats.get(key)
    if counts:
        total = sum(counts.values())
        if total >= min_freq:
            best = max(counts, key=counts.get)
            if best and counts[best] / total >= threshold:
                return best

    # 2. Head noun backoff — use remix structure if available
    head_word = None
    if remix_parts:
        if "head" in remix_parts:
            head_word = remix_parts["head"]
        elif "topic" in remix_parts:
            head_word = remix_parts["topic"]
    if head_word is None:
        words = composed.split()
        head_word = words[-1] if words else None

    if head_word:
        counts = article_stats.get(head_word.lower())
        if counts:
            total = sum(counts.values())
            if total >= min_freq:
                best = max(counts, key=counts.get)
                if best and counts[best] / total >= threshold:
                    return best

    return ""


def _make_rng(seed: int | None) -> random.Random | None:
    return random.Random(seed) if seed is not None else None


def _load_generation_candidates(conn: sqlite3.Connection) -> GenerationCandidates:
    if _has_model_scores(conn):
        select = _slot_sampling_select(
            "sf.slot_type = ? AND sf.mode = 'strict'",
            include_model_scores=True,
        )
        return GenerationCandidates(
            list_rows=_filter_generation_rows(
                "list_item", conn.execute(select, ("list_item",)).fetchall(),
            ),
            action_rows=_filter_generation_rows(
                "action_noun", conn.execute(select, ("action_noun",)).fetchall(),
            ),
            obj_rows=_filter_generation_rows(
                "of_object", conn.execute(select, ("of_object",)).fetchall(),
            ),
        )
    return GenerationCandidates(
        list_rows=_filter_generation_rows(
            "list_item",
            conn.execute(
                "SELECT filler, freq, popularity_score FROM slot_fillers "
                "WHERE slot_type = 'list_item' AND mode = 'strict'"
            ).fetchall(),
        ),
        action_rows=_filter_generation_rows(
            "action_noun",
            conn.execute(
                "SELECT filler, freq, popularity_score FROM slot_fillers "
                "WHERE slot_type = 'action_noun' AND mode = 'strict'"
            ).fetchall(),
        ),
        obj_rows=_filter_generation_rows(
            "of_object",
            conn.execute(
                "SELECT filler, freq, popularity_score FROM slot_fillers "
                "WHERE slot_type = 'of_object' AND mode = 'strict'"
            ).fetchall(),
        ),
    )


def _slot_sampling_select(where: str, *, include_model_scores: bool) -> str:
    if include_model_scores:
        return (
            "SELECT sf.filler, sf.freq, sf.popularity_score, "
            "ms.score_pop, ms.score_mainstream, ms.score_niche "
            "FROM slot_fillers sf "
            "LEFT JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id "
            f"WHERE {where}"
        )
    return (
        "SELECT filler, freq, popularity_score FROM slot_fillers sf "
        f"WHERE {where}"
    )


def _filter_generation_rows(slot_type: str, rows: list[tuple]) -> list[tuple]:
    return [row for row in rows if not _is_literal_bad_filler(slot_type, row[0])]


def _has_model_scores(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'slot_filler_model_scores'"
    ).fetchone()
    return row is not None


def _is_literal_bad_filler(slot_type: str, filler: str) -> bool:
    lower = (filler or "").strip().lower()
    if not lower:
        return True
    if lower in _BAD_STANDALONE_FILLERS:
        return True
    if re.search(r",\s*(?:jr|sr)\.?$", lower):
        return True
    if slot_type == "of_object" and lower in _BAD_ONE_WORD_OBJECTS:
        return True
    # Artifact like H.G.W.ells: run-together initials followed by a lowercase tail.
    if re.search(r"(?:[a-z]\.){2,}[a-z]", lower):
        return True
    return False


def _has_enough_candidates(
    candidates: GenerationCandidates,
) -> bool:
    return (
        len(candidates.list_rows) >= 2
        and bool(candidates.action_rows)
        and bool(candidates.obj_rows)
    )


def _not_enough_fillers_subtitle() -> GeneratedSubtitle:
    return GeneratedSubtitle(
        text="(not enough fillers — run 'build-slots' first)",
        item1="", item2="", action_noun="", of_object="",
    )


def _pick_list_items(
    list_rows: list[tuple],
    rng: random.Random | None,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> list[str]:
    return _sample_slot_rows(
        "list_item",
        list_rows,
        2,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )


def _pick_action_noun(
    action_rows: list[tuple],
    rng: random.Random | None,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> str:
    return _sample_slot_rows(
        "action_noun",
        action_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]


def _pick_of_object_and_remix(
    conn: sqlite3.Connection,
    obj_rows: list[tuple],
    rng: random.Random | None,
    remix_prob: float,
    min_sim: float,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> tuple[str, bool, dict, float | None]:
    remix_similarity = None

    of_object = _sample_slot_rows(
        "of_object",
        obj_rows,
        1,
        rng,
        model_tier=model_tier,
        runtime=runtime,
    )[0]
    remixed = False
    remix_parts = {}

    if remix_prob > 0 and len(of_object.split()) >= 2:
        should_remix = (rng or random).random() < remix_prob
        if should_remix:
            result = _try_remix(
                conn,
                rng,
                of_object,
                min_sim,
                model_tier=model_tier,
                runtime=runtime,
            )
            if result:
                of_object, remix_parts, remix_similarity = result
                remixed = True

    return of_object, remixed, remix_parts, remix_similarity


def _select_subtitle_parts(
    conn: sqlite3.Connection,
    candidates: GenerationCandidates,
    rng: random.Random | None,
    remix_prob: float,
    min_sim: float,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> SelectedSubtitleParts:
    items = _pick_list_items(candidates.list_rows, rng, model_tier, runtime)
    action_noun = _pick_action_noun(
        candidates.action_rows, rng, model_tier, runtime,
    )
    of_object, remixed, remix_parts, remix_similarity = _pick_of_object_and_remix(
        conn, candidates.obj_rows, rng, remix_prob, min_sim, model_tier, runtime,
    )
    return SelectedSubtitleParts(
        items=items,
        action_noun=action_noun,
        of_object=of_object,
        remixed=remixed,
        remix_parts=remix_parts,
        remix_similarity=remix_similarity,
    )


def _resolve_articles(
    conn: sqlite3.Connection,
    action_noun: str,
    of_object: str,
    remixed: bool,
    remix_parts: dict,
) -> tuple[str, str]:
    ctx = _load_remix_context(conn)
    cfg = load_tuning_config(conn)
    of_min_freq = cfg.get("article_of_min_freq", 3.0)
    act_min_freq = cfg.get("article_action_min_freq", 3.0)
    remix_threshold = cfg.get("article_remix_heuristic_threshold", 0.6)

    action_article = _majority_article(
        action_noun, ctx.get("article_stats_action", {}), act_min_freq,
    )
    if not action_article:
        action_article = "the"

    if remixed:
        of_article = _infer_of_article(
            of_object, ctx.get("article_stats_of", {}), of_min_freq, remix_threshold,
            remix_parts=remix_parts,
        )
    else:
        of_article = _article_with_backoff(
            of_object, ctx.get("article_stats_of", {}), of_min_freq,
        )

    action_article = _fix_a_an(action_article, action_noun)
    if of_article:
        of_article = _fix_a_an(of_article, of_object)
    return action_article, of_article


def _assemble_generated_subtitle(
    parts: SelectedSubtitleParts,
    action_article: str,
    of_article: str,
) -> GeneratedSubtitle:
    of_prefix = f"{of_article} " if of_article else ""
    text = (
        f"{parts.items[0]}, {parts.items[1]}, and "
        f"{action_article} {parts.action_noun} of {of_prefix}{parts.of_object}"
    )
    remix_parts = parts.remix_parts
    if remix_parts:
        remix_parts = {k: _title_case(v) for k, v in remix_parts.items()}

    return GeneratedSubtitle(
        text=_title_case(text),
        item1=_title_case(parts.items[0]),
        item2=_title_case(parts.items[1]),
        action_noun=_title_case(parts.action_noun),
        of_object=_title_case(parts.of_object),
        remixed=parts.remixed,
        remix_parts=remix_parts,
        remix_similarity=parts.remix_similarity,
        of_article=of_article,
        action_article=action_article,
    )


def _generate_subtitle_from_candidates(
    conn: sqlite3.Connection,
    candidates: GenerationCandidates,
    *,
    seed: int | None,
    remix_prob: float,
    min_sim: float,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
) -> GeneratedSubtitle:
    rng = _make_rng(seed)
    if not _has_enough_candidates(candidates):
        return _not_enough_fillers_subtitle()

    parts = _select_subtitle_parts(
        conn,
        candidates,
        rng,
        remix_prob,
        min_sim,
        model_tier,
        runtime,
    )
    action_article, of_article = _resolve_articles(
        conn,
        parts.action_noun,
        parts.of_object,
        parts.remixed,
        parts.remix_parts,
    )
    return _assemble_generated_subtitle(parts, action_article, of_article)


def generate_subtitle(
    conn: sqlite3.Connection, seed: int | None = None,
    remix_prob: float = 0.0, min_sim: float = 0.0,
    runtime: GenerationRuntimeSelection | PreparedGenerationRuntime | None = None,
) -> GeneratedSubtitle:
    """Generate one random subtitle in the 'X, Y, and the Z of W' pattern.

    remix_prob: probability of remixing a multi-word of-object (0.0 = never, 1.0 = always).
    min_sim: minimum cosine similarity for embedding coherence filter.
    """
    candidates = _load_generation_candidates(conn)
    if not _has_enough_candidates(candidates):
        return _not_enough_fillers_subtitle()

    prepared_runtime = prepare_generation_runtime(conn, runtime)
    model_tier = None
    if prepared_runtime.mode == RuntimeSelectionMode.SHADOW:
        model_tier = _choose_default_generation_tier(conn, seed)

    return _generate_subtitle_from_candidates(
        conn,
        candidates,
        seed=seed,
        remix_prob=remix_prob,
        min_sim=min_sim,
        model_tier=model_tier,
        runtime=prepared_runtime,
    )


def generate_subtitles(
    conn: sqlite3.Connection,
    *,
    n: int,
    seed_base: int | None = 1000,
    remix_prob: float = 0.0,
    min_sim: float = 0.0,
    runtime: GenerationRuntimeSelection | PreparedGenerationRuntime | None = None,
) -> list[GeneratedSubtitle]:
    """Generate a batch from one source candidate snapshot."""

    candidates = _load_generation_candidates(conn)
    if not _has_enough_candidates(candidates):
        return [_not_enough_fillers_subtitle() for _ in range(n)]
    prepared_runtime = prepare_generation_runtime(conn, runtime)

    return [
        _generate_subtitle_from_candidates(
            conn,
            candidates,
            seed=seed_base + i if seed_base is not None else None,
            remix_prob=remix_prob,
            min_sim=min_sim,
            model_tier=(
                _choose_default_generation_tier(
                    conn,
                    seed_base + i if seed_base is not None else None,
                )
                if prepared_runtime.mode == RuntimeSelectionMode.SHADOW
                else None
            ),
            runtime=prepared_runtime,
        )
        for i in range(n)
    ]


def _default_generation_tier_ratios(conn: sqlite3.Connection) -> dict[str, float]:
    cfg = load_tuning_config(conn)
    weights = {
        tier: max(0.0, cfg[f"generation_tier_ratio_{tier}"])
        for tier in _DEFAULT_GENERATION_TIERS
    }
    total = sum(weights.values())
    if total <= 0.0:
        return {"pop": 0.0, "mainstream": 1.0, "niche": 0.0}
    return {tier: value / total for tier, value in weights.items()}


def _choose_default_generation_tier(
    conn: sqlite3.Connection,
    seed: int | None,
) -> str:
    return _choose_generation_tier(
        conn,
        allowed_tiers=set(_DEFAULT_GENERATION_TIERS),
        seed=seed,
    )


def _generation_tier_sequence(
    conn: sqlite3.Connection,
    *,
    allowed_tiers: set[str],
    seed: int | None,
) -> list[str]:
    ratios = _default_generation_tier_ratios(conn)
    remaining = [
        tier for tier in _DEFAULT_GENERATION_TIERS
        if tier in allowed_tiers
    ]
    if not remaining:
        raise ValueError("allowed_tiers must include at least one known tier")
    rng = random.Random(seed) if seed is not None else random
    sequence: list[str] = []
    while remaining:
        weights = [ratios[tier] for tier in remaining]
        if sum(weights) <= 0.0:
            tier = "mainstream" if "mainstream" in remaining else remaining[0]
        else:
            tier = rng.choices(
                remaining,
                weights=weights,
                k=1,
            )[0]
        sequence.append(tier)
        remaining.remove(tier)
    return sequence


def _choose_generation_tier(
    conn: sqlite3.Connection,
    *,
    allowed_tiers: set[str],
    seed: int | None,
) -> str:
    return _generation_tier_sequence(
        conn,
        allowed_tiers=allowed_tiers,
        seed=seed,
    )[0]


def generate_subtitle_matching_tiers(
    conn: sqlite3.Connection,
    *,
    allowed_tiers: set[str] | None,
    seed: int | None = None,
    remix_prob: float = 0.0,
    min_sim: float = 0.0,
    max_attempts: int = MAX_TIER_FILTER_ATTEMPTS,
    runtime: GenerationRuntimeSelection | PreparedGenerationRuntime | None = None,
) -> GeneratedSubtitle:
    """Generate a subtitle whose evidence tier satisfies the requested filter."""

    prepared_runtime = prepare_generation_runtime(conn, runtime)
    if prepared_runtime.mode == RuntimeSelectionMode.SHADOW:
        if not allowed_tiers:
            chosen_tier = _choose_generation_tier(
                conn,
                allowed_tiers=set(_DEFAULT_GENERATION_TIERS),
                seed=seed,
            )
        elif len(allowed_tiers) > 1:
            chosen_tier = _choose_generation_tier(
                conn,
                allowed_tiers=allowed_tiers,
                seed=seed,
            )
        else:
            chosen_tier = next(iter(allowed_tiers))
        candidates = _load_generation_candidates(conn)
        return _generate_subtitle_from_candidates(
            conn,
            candidates,
            seed=seed,
            remix_prob=remix_prob,
            min_sim=min_sim,
            model_tier=chosen_tier,
            runtime=prepared_runtime,
        )

    default_tier_sequence: list[str] | None = None
    if not allowed_tiers:
        default_tier_sequence = _generation_tier_sequence(
            conn,
            allowed_tiers=set(_DEFAULT_GENERATION_TIERS),
            seed=seed,
        )
        max_attempts = min(max_attempts, DEFAULT_GENERATION_TIER_ATTEMPTS)
    elif len(allowed_tiers) > 1:
        allowed_tiers = {
            _choose_generation_tier(
                conn,
                allowed_tiers=allowed_tiers,
                seed=seed,
            )
        }

    from subtitle_generator.tiering import compute_tier_evidence

    candidates = _load_generation_candidates(conn)
    observed_tiers: Counter[str] = Counter()
    last_tier: str | None = None

    tier_sequence = default_tier_sequence or [next(iter(allowed_tiers))]
    remaining_attempts = max_attempts
    attempt_number = 0
    for tier_index, requested_tier in enumerate(tier_sequence):
        if remaining_attempts <= 0:
            break
        tiers_left = len(tier_sequence) - tier_index
        attempts_for_tier = (
            max(1, remaining_attempts // tiers_left)
            if default_tier_sequence
            else remaining_attempts
        )
        current_allowed_tiers = {requested_tier}
        for _ in range(attempts_for_tier):
            seed_offset = attempt_number
            attempt_number += 1
            subtitle = _generate_subtitle_from_candidates(
                conn,
                candidates,
                seed=seed + seed_offset if seed is not None else None,
                remix_prob=remix_prob,
                min_sim=min_sim,
                model_tier=requested_tier if _has_model_scores(conn) else None,
                runtime=prepared_runtime,
            )
            tier = compute_tier_evidence(
                subtitle.text,
                conn,
                remix_parts=subtitle.remix_parts if subtitle.remixed else None,
            ).tier
            observed_tiers[tier] += 1
            if tier in current_allowed_tiers:
                return subtitle
            last_tier = tier
        remaining_attempts -= attempts_for_tier

    if default_tier_sequence:
        return generate_subtitles(
            conn,
            n=1,
            seed_base=seed,
            remix_prob=remix_prob,
            min_sim=min_sim,
            runtime=prepared_runtime,
        )[0]
    requested = ", ".join(sorted(allowed_tiers))
    suffix = f"; last generated tier was {last_tier}" if last_tier else ""
    raise TierFilterError(
        f"Could not generate a subtitle matching tier filter [{requested}] "
        f"after {max_attempts} attempts{suffix}; "
        f"observed tiers: {_format_tier_counts(observed_tiers)}."
    )


def generate_subtitles_by_tier(
    conn: sqlite3.Connection,
    *,
    tiers: list[str],
    samples_per_tier: int,
    seed: int | None = None,
    remix_prob: float = 0.0,
    min_sim: float = 0.0,
    max_attempts: int = MAX_TIER_FILTER_ATTEMPTS,
    runtime: GenerationRuntimeSelection | PreparedGenerationRuntime | None = None,
) -> dict[str, list[GeneratedSubtitle]]:
    """Generate a shared candidate pool until each requested tier has enough samples."""

    from subtitle_generator.tiering import compute_tier_evidence

    requested_tiers = list(dict.fromkeys(tiers))
    buckets: dict[str, list[GeneratedSubtitle]] = {
        tier: [] for tier in requested_tiers
    }
    candidates = _load_generation_candidates(conn)
    if not _has_enough_candidates(candidates):
        return {
            tier: [_not_enough_fillers_subtitle() for _ in range(samples_per_tier)]
            for tier in requested_tiers
        }
    prepared_runtime = prepare_generation_runtime(conn, runtime)
    if prepared_runtime.mode == RuntimeSelectionMode.SHADOW:
        for tier in requested_tiers:
            for index in range(samples_per_tier):
                buckets[tier].append(
                    _generate_subtitle_from_candidates(
                        conn,
                        candidates,
                        seed=(
                            seed + (requested_tiers.index(tier) * samples_per_tier) + index
                            if seed is not None
                            else None
                        ),
                        remix_prob=remix_prob,
                        min_sim=min_sim,
                        model_tier=tier,
                        runtime=prepared_runtime,
                    )
                )
        return buckets
    observed_tiers: Counter[str] = Counter()
    last_tier: str | None = None

    for attempt in range(max_attempts):
        remaining_tiers = [
            tier for tier in requested_tiers
            if len(buckets[tier]) < samples_per_tier
        ]
        if not remaining_tiers:
            return buckets

        target_tier = remaining_tiers[attempt % len(remaining_tiers)]
        subtitle = _generate_subtitle_from_candidates(
            conn,
            candidates,
            seed=(seed + attempt if seed is not None else None),
            remix_prob=remix_prob,
            min_sim=min_sim,
            model_tier=target_tier if _has_model_scores(conn) else None,
            runtime=prepared_runtime,
        )
        tier = compute_tier_evidence(
            subtitle.text,
            conn,
            remix_parts=subtitle.remix_parts if subtitle.remixed else None,
        ).tier
        observed_tiers[tier] += 1
        if tier in buckets and len(buckets[tier]) < samples_per_tier:
            buckets[tier].append(subtitle)
            if all(len(samples) >= samples_per_tier for samples in buckets.values()):
                return buckets
        last_tier = tier

    missing = ", ".join(
        f"{tier}={samples_per_tier - len(samples)}"
        for tier, samples in buckets.items()
        if len(samples) < samples_per_tier
    )
    suffix = f"; last generated tier was {last_tier}" if last_tier else ""
    raise TierFilterError(
        "Could not generate enough subtitles for tier spot-check batch "
        f"after {max_attempts} attempts; missing {missing}{suffix}; "
        f"observed tiers: {_format_tier_counts(observed_tiers)}."
    )


def _format_tier_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    preferred = ["pop", "mainstream", "niche"]
    names = [name for name in preferred if name in counts]
    names.extend(sorted(set(counts) - set(preferred)))
    return ", ".join(f"{name}={counts[name]}" for name in names)


def _try_remix(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    original_of_object: str,
    min_sim: float,
    model_tier: str | None = None,
    runtime: PreparedGenerationRuntime | None = None,
    max_retries: int = 5,
) -> tuple[str, dict, float | None] | None:
    """Attempt to remix an of-object.

    Returns (composed_text, parts_dict, similarity_score) or None.

    Supports both pre-computed scalar decomposition (precomputed=True) and
    live spaCy (precomputed=False, dev fallback).
    """
    ctx = _load_remix_context(conn)
    is_precomputed = ctx.get("precomputed", False)

    if is_precomputed:
        # Read pre-computed classification from DB
        row = conn.execute(
            "SELECT remix_type, remix_prep, remix_word_count FROM slot_fillers "
            "WHERE filler = ? AND slot_type = 'of_object' AND mode = 'strict' LIMIT 1",
            (original_of_object,),
        ).fetchone()
        if row and row[0] is not None:
            if row[0] == "type1":
                orig_classification = ("type1", row[2])
            else:
                orig_classification = ("type2", row[1], row[2])
        else:
            orig_classification = None
    else:
        nlp = ctx["nlp"]
        doc = nlp(original_of_object)
        orig_classification = _classify_for_remix(original_of_object, doc)

    classification = orig_classification
    if classification is None:
        return None

    # Reject type-2 remixes where inner prep is "of" (produces double-of)
    cfg = load_tuning_config(conn)
    if cfg.get("remix_reject_double_of", 1.0) > 0:
        if classification[0] == "type2" and classification[1] == "of":
            return None

    for _ in range(max_retries):
        if classification[0] == "type1":
            _, word_count = classification
            result = compose_compound(
                conn,
                rng,
                ctx,
                word_count,
                model_tier=model_tier,
                runtime=runtime,
            )
        else:
            _, prep, word_count = classification
            result = compose_prepositional(
                conn,
                rng,
                ctx,
                prep,
                word_count,
                model_tier=model_tier,
                runtime=runtime,
            )

        if result is None:
            continue

        composed, parts = result

        # Compute similarity when coherence check is active.
        sim = None
        if min_sim > 0:
            if is_precomputed:
                sim = _approx_cosine_sim(parts, ctx, classification[0])
            else:
                nlp = ctx["nlp"]
                centroid = ctx["centroid"]
                if centroid is not None:
                    composed_doc = nlp(composed)
                    if composed_doc.has_vector and composed_doc.vector_norm > 0:
                        import numpy as np
                        norm1 = float(np.linalg.norm(centroid))
                        norm2 = float(np.linalg.norm(composed_doc.vector))
                        if norm1 > 0 and norm2 > 0:
                            sim = float(np.dot(centroid, composed_doc.vector) / (norm1 * norm2))

        # Coherence check.
        if min_sim > 0 and sim is not None:
            if sim < min_sim:
                continue

        return composed, parts, sim

    return None


def find_source(conn: sqlite3.Connection, filler: str, slot_type: str = "of_object") -> tuple[str, str] | None:
    """Find the real book a slot filler was extracted from.

    Tries the pre-joined sources table first (mini DB), then falls back to
    the full subtitles table (development DB).
    Returns (description, source_tag) where source_tag is 'LOC' or 'OL'.
    """
    # Try pre-joined sources table (mini DB for deployment)
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='sources'"
    ).fetchone()
    if row:
        src = conn.execute(
            "SELECT sr.title, sr.subtitle_text, sr.source_tag "
            "FROM slot_fillers sf "
            "JOIN sources sr ON sr.slot_filler_id = sf.id "
            "WHERE sf.filler = ? COLLATE NOCASE AND sf.slot_type = ? "
            "LIMIT 1",
            (filler, slot_type),
        ).fetchone()
        if src:
            title = (src[0] or "").strip().rstrip(" /:")
            subtitle = (src[1] or "").strip().rstrip(" /:")
            tag = src[2] or "LOC"
            desc = f"{title}: {subtitle}" if title and subtitle else (title or subtitle)
            return desc, tag

    # Fallback: full DB with subtitles table
    try:
        row = conn.execute(
            "SELECT s.title, s.subtitle, s.source_file "
            "FROM slot_fillers sf "
            "JOIN subtitles s ON s.id = sf.source_subtitle_id "
            "WHERE sf.filler = ? COLLATE NOCASE AND sf.slot_type = ? AND sf.source_subtitle_id IS NOT NULL "
            "LIMIT 1",
            (filler, slot_type),
        ).fetchone()
        if not row:
            escaped = filler.replace("'", "''")
            row = conn.execute(
                "SELECT title, subtitle, source_file FROM subtitles "
                f"WHERE subtitle LIKE '%{escaped}%' ORDER BY RANDOM() LIMIT 1"
            ).fetchone()
        if row:
            title = (row[0] or "").strip().rstrip(" /:")
            subtitle = (row[1] or "").strip().rstrip(" /:")
            source_file = row[2] or ""
            tag = "OL" if source_file == "openlibrary" else "LOC"
            desc = f"{title}: {subtitle}" if title and subtitle else (title or subtitle)
            return desc, tag
    except Exception:
        pass
    return None


def format_sources(conn: sqlite3.Connection, sub: GeneratedSubtitle) -> str:
    """Look up source books for each filler and format as markdown."""
    fillers = [
        ("List item 1", sub.item1, "list_item"),
        ("List item 2", sub.item2, "list_item"),
        ("Action noun", sub.action_noun, "action_noun"),
    ]

    if sub.remixed and sub.remix_parts:
        # Show individual remix parts
        if "modifier" in sub.remix_parts:
            fillers.append(("Of-modifier", sub.remix_parts["modifier"], "of_modifier"))
            fillers.append(("Of-head", sub.remix_parts["head"], "of_head"))
        elif "topic" in sub.remix_parts:
            fillers.append(("Of-topic", sub.remix_parts["topic"], "of_topic"))
            fillers.append(("Of-complement", sub.remix_parts["complement"], "of_complement"))
    else:
        fillers.append(("Of-object", sub.of_object, "of_object"))

    lines = ["", "---", "**Sources:**"]
    for label, filler, slot_type in fillers:
        result = find_source(conn, filler, slot_type)
        if result:
            desc, tag = result
            lines.append(f"- *{label}* \"{filler}\" ← [{tag}] {desc}")
        else:
            lines.append(f"- *{label}* \"{filler}\" ← (source not found)")
    if sub.remixed:
        lines.append(f"- *(remixed from: \"{sub.of_object}\")*")
    return "\n".join(lines)


def slot_stats(conn: sqlite3.Connection) -> dict:
    """Get counts per slot type."""
    rows = conn.execute(
        "SELECT slot_type, COUNT(*) FROM slot_fillers WHERE mode = 'strict' GROUP BY slot_type"
    ).fetchall()
    return dict(rows)
