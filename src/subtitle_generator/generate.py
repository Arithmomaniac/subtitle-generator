"""Generate bizarre subtitles by randomly combining slot fillers."""

import json
import math
import random
import re
import sqlite3
from dataclasses import dataclass, field

import click
import inflect
from titlecase import titlecase as _lib_titlecase

from subtitle_generator.config import DEFAULT_TONE_TARGETS, load_tuning_config

_inflect_engine = inflect.engine()


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


def _weighted_sample(
    rows: list[tuple], k: int,
    rng: random.Random | None = None,
    tone_target: float | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[str]:
    """Pick k unique fillers weighted by sqrt(freq), optionally biased by tone.

    Rows can be (filler, freq) or (filler, freq, popularity_score).
    When tone_target is set, applies a log-space Gaussian bias that boosts
    fillers near the target score and suppresses distant ones.

    Blending is controlled by pop_base_weight_blend (base weight) and
    pop_tone_blend (tone bias score). At 0.0, uses pure freq; at 1.0, uses
    pure popularity_score.
    """
    cfg = load_tuning_config(conn) if (tone_target is not None or conn is not None) else {}
    blend_base = cfg.get("pop_base_weight_blend", 0.0)
    blend_tone = cfg.get("pop_tone_blend", 0.0)
    pop_default = cfg.get("pop_missing_default", 0.1)

    fillers = [r[0] for r in rows]
    has_pop = len(rows) > 0 and len(rows[0]) >= 3

    # Base weights: blend sqrt(freq) and sqrt(popularity_score)
    weights = []
    for r in rows:
        freq = r[1]
        pop_score = (r[2] if r[2] is not None else pop_default) if has_pop else pop_default
        w_freq = math.sqrt(freq)
        w_pop = math.sqrt(max(pop_score, 0.001))
        weights.append((1 - blend_base) * w_freq + blend_base * w_pop)

    if tone_target is not None:
        if not cfg:
            cfg = load_tuning_config(conn)
        spread = cfg["weighted_sample_spread"]
        bias_floor = cfg["weighted_sample_bias_floor"]
        for i, r in enumerate(rows):
            freq = r[1]
            pop_score = (r[2] if r[2] is not None else pop_default) if has_pop else pop_default
            score_freq = math.log10(1 + freq)
            filler_score = (1 - blend_tone) * score_freq + blend_tone * pop_score
            bias = math.exp(-((filler_score - tone_target) / spread) ** 2)
            weights[i] *= (bias_floor + (1 - bias_floor) * bias)

    chosen = []
    # Weighted sampling without replacement
    for _ in range(k):
        pick = (rng or random).choices(fillers, weights=weights, k=1)[0]
        idx = fillers.index(pick)
        chosen.append(pick)
        fillers.pop(idx)
        weights.pop(idx)
    return chosen


# Module-level for import compatibility — uses defaults when no DB connection
TONE_TARGETS = DEFAULT_TONE_TARGETS

# --- Remix infrastructure ---

# Module-level cache for remix context (lazy-loaded)
_remix_ctx: dict | None = None

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


def _load_remix_context(conn: sqlite3.Connection) -> dict:
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

        _remix_ctx = {
            "centroid_norm": centroid_norm,
            "avg_cross_sim_t1": avg_cross_sim_t1,
            "avg_cross_sim_t2": avg_cross_sim_t2,
            "filler_scalars": filler_scalars,
            "config": config,
            "precomputed": True,
            "article_stats_of": article_stats_of,
            "article_stats_action": article_stats_action,
        }
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

    _remix_ctx = {
        "nlp": nlp, "centroid": centroid, "config": config, "precomputed": False,
        "article_stats_of": article_stats_of,
        "article_stats_action": article_stats_action,
    }
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
    tone_target: dict[str, float] | None,
    ctx: dict,
    word_count: int,
    locked_modifier: str | None = None,
    locked_head: str | None = None,
) -> tuple[str, dict] | None:
    """Compose a Type 1 remixed of-object (modifier + head).

    Returns (composed_text, parts_dict) or None if composition fails.
    When locked_modifier or locked_head is provided, uses the locked value
    instead of drawing from the pool.
    """
    mod_word_count = word_count - 1  # head is always 1 word
    mod_space_count = mod_word_count - 1
    mod_target = tone_target.get("of_object") if tone_target else None

    if locked_modifier is not None:
        modifier = locked_modifier
    else:
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
            "SELECT filler, freq, popularity_score FROM slot_fillers "
            "WHERE slot_type = 'of_modifier' AND pos_tag = ? "
            "AND length(filler) - length(replace(filler, ' ', '')) = ? AND mode = 'strict'",
            (chosen_mod_pos, mod_space_count),
        ).fetchall()
        if not mod_rows:
            return None
        modifier = _weighted_sample(mod_rows, 1, rng, mod_target, conn)[0]

    if locked_head is not None:
        head = locked_head
    else:
        head_rows = conn.execute(
            "SELECT filler, freq, popularity_score FROM slot_fillers "
            "WHERE slot_type = 'of_head' AND mode = 'strict'",
        ).fetchall()
        if not head_rows:
            return None
        head = _weighted_sample(head_rows, 1, rng, mod_target, conn)[0]

    composed = f"{modifier} {head}"
    parts = {"modifier": modifier, "head": head}
    return composed, parts


def compose_prepositional(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    tone_target: dict[str, float] | None,
    ctx: dict,
    prep: str,
    word_count: int,
    locked_topic: str | None = None,
    locked_complement: str | None = None,
) -> tuple[str, dict] | None:
    """Compose a Type 2 remixed of-object (topic + prep + complement).

    Returns (composed_text, parts_dict) or None if composition fails.
    Enforces strict bucket word-count matching unless parts are locked.
    """
    obj_target = tone_target.get("of_object") if tone_target else None

    if locked_topic is not None:
        topic = locked_topic
    else:
        topic_rows = conn.execute(
            "SELECT filler, freq, popularity_score FROM slot_fillers "
            "WHERE slot_type = 'of_topic' AND prep = ? AND mode = 'strict'",
            (prep,),
        ).fetchall()
        if not topic_rows:
            return None
        topic = _weighted_sample(topic_rows, 1, rng, obj_target, conn)[0]

    if locked_complement is not None:
        complement = locked_complement
    else:
        comp_rows = conn.execute(
            "SELECT filler, freq, popularity_score FROM slot_fillers "
            "WHERE slot_type = 'of_complement' AND prep = ? AND mode = 'strict'",
            (prep,),
        ).fetchall()
        if not comp_rows:
            return None
        complement = _weighted_sample(comp_rows, 1, rng, obj_target, conn)[0]

    composed = f"{topic} {prep} {complement}"

    # Strict bucket: verify word count matches (skip when parts are locked)
    if locked_topic is None and locked_complement is None:
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


def generate_subtitle(
    conn: sqlite3.Connection, seed: int | None = None,
    tone_target: dict[str, float] | None = None,
    remix_prob: float = 0.0, min_sim: float = 0.0,
    locks: dict[str, str] | None = None,
) -> GeneratedSubtitle:
    """Generate one random subtitle in the 'X, Y, and the Z of W' pattern.

    tone_target maps slot_type → log10 target score for filler biasing.
    remix_prob: probability of remixing a multi-word of-object (0.0 = never, 1.0 = always).
    min_sim: minimum cosine similarity for embedding coherence filter.
    locks: optional dict mapping slot keys to locked values.
        Supported keys: item1, item2, action_noun, of_object,
        of_modifier, of_head, of_topic, of_complement.
    """
    rng = None
    if seed is not None:
        rng = random.Random(seed)

    # Validate lock combinations
    if locks:
        type1_keys = {"of_modifier", "of_head"}
        type2_keys = {"of_topic", "of_complement"}
        has_type1 = bool(type1_keys & locks.keys())
        has_type2 = bool(type2_keys & locks.keys())
        if has_type1 and has_type2:
            raise ValueError("Cannot mix Type 1 (of_modifier/of_head) and Type 2 (of_topic/of_complement) locks")
        sub_part_keys = type1_keys | type2_keys
        if "of_object" in locks and (sub_part_keys & locks.keys()):
            raise ValueError("Cannot combine of_object lock with sub-part locks")

    list_rows = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'list_item' AND mode = 'strict'"
    ).fetchall()
    action_rows = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'action_noun' AND mode = 'strict'"
    ).fetchall()
    obj_rows = conn.execute(
        "SELECT filler, freq, popularity_score FROM slot_fillers WHERE slot_type = 'of_object' AND mode = 'strict'"
    ).fetchall()

    # Adjust required-row checks based on locks
    list_needed = 2 - sum(1 for k in ("item1", "item2") if locks and k in locks)
    action_needed = not (locks and "action_noun" in locks)
    obj_needed = not (locks and "of_object" in locks)

    if (len(list_rows) < list_needed) or \
       (action_needed and not action_rows) or \
       (obj_needed and not obj_rows):
        return GeneratedSubtitle(
            text="(not enough fillers — run 'build-slots' first)",
            item1="", item2="", action_noun="", of_object="",
        )

    list_target = tone_target.get("list_item") if tone_target else None
    action_target = tone_target.get("action_noun") if tone_target else None
    obj_target = tone_target.get("of_object") if tone_target else None

    # Draw or lock list items (avoid duplicates with locked value)
    if locks and "item1" in locks and "item2" in locks:
        items = [locks["item1"], locks["item2"]]
    elif locks and "item1" in locks:
        pool = [(f, w) for f, w in list_rows if f != locks["item1"]]
        if not pool:
            pool = list_rows
        items = [locks["item1"], _weighted_sample(pool, 1, rng, list_target, conn)[0]]
    elif locks and "item2" in locks:
        pool = [(f, w) for f, w in list_rows if f != locks["item2"]]
        if not pool:
            pool = list_rows
        items = [_weighted_sample(pool, 1, rng, list_target, conn)[0], locks["item2"]]
    else:
        items = _weighted_sample(list_rows, 2, rng, list_target, conn)

    # Draw or lock action noun
    if locks and "action_noun" in locks:
        action_noun = locks["action_noun"]
    else:
        action_noun = _weighted_sample(action_rows, 1, rng, action_target, conn)[0]

    # Draw or lock of-object
    remix_similarity = None
    if locks and "of_object" in locks:
        of_object = locks["of_object"]
        remixed = False
        remix_parts = {}
    else:
        of_object = _weighted_sample(obj_rows, 1, rng, obj_target, conn)[0]
        remixed = False
        remix_parts = {}

        # Check for sub-part locks that force remix
        sub_part_keys = {"of_modifier", "of_head", "of_topic", "of_complement"}
        sub_locks = {k: v for k, v in (locks or {}).items() if k in sub_part_keys}

        if sub_locks:
            result = _try_remix(conn, rng, tone_target, of_object, min_sim,
                                locked_parts=sub_locks)
            if result:
                of_object, remix_parts, remix_similarity = result
                remixed = True
        elif remix_prob > 0 and len(of_object.split()) >= 2:
            should_remix = (rng or random).random() < remix_prob
            if should_remix:
                result = _try_remix(conn, rng, tone_target, of_object, min_sim)
                if result:
                    of_object, remix_parts, remix_similarity = result
                    remixed = True

    # Resolve articles from corpus stats
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

    # Correct a/an using phonetic analysis
    action_article = _fix_a_an(action_article, action_noun)
    if of_article:
        of_article = _fix_a_an(of_article, of_object)

    # Assemble raw text, then title-case once
    of_prefix = f"{of_article} " if of_article else ""
    text = f"{items[0]}, {items[1]}, and {action_article} {action_noun} of {of_prefix}{of_object}"
    text = _title_case(text)

    # Title-case remix_parts for display
    if remix_parts:
        remix_parts = {k: _title_case(v) for k, v in remix_parts.items()}

    return GeneratedSubtitle(
        text=text,
        item1=items[0],
        item2=items[1],
        action_noun=action_noun,
        of_object=of_object,
        remixed=remixed,
        remix_parts=remix_parts,
        remix_similarity=remix_similarity,
        of_article=of_article,
        action_article=action_article,
    )


def _try_remix(
    conn: sqlite3.Connection,
    rng: random.Random | None,
    tone_target: dict[str, float] | None,
    original_of_object: str,
    min_sim: float,
    max_retries: int = 5,
    locked_parts: dict[str, str] | None = None,
) -> tuple[str, dict, float | None] | None:
    """Attempt to remix an of-object.

    Returns (composed_text, parts_dict, similarity_score) or None.
    When locked_parts is provided, locked values are passed through to
    compose functions and coherence-filter behavior is adjusted.

    Supports both pre-computed scalar decomposition (precomputed=True) and
    live spaCy (precomputed=False, dev fallback).
    """
    ctx = _load_remix_context(conn)
    is_precomputed = ctx.get("precomputed", False)

    has_locks = bool(locked_parts)

    # Check if any locked value is custom (not in slot_fillers)
    skip_coherence = False
    if has_locks:
        _slot_type_map = {
            "of_modifier": "of_modifier",
            "of_head": "of_head",
            "of_topic": "of_topic",
            "of_complement": "of_complement",
        }
        for lock_key, lock_val in locked_parts.items():
            st = _slot_type_map.get(lock_key)
            if st is None:
                continue
            row = conn.execute(
                "SELECT 1 FROM slot_fillers WHERE filler = ? AND slot_type = ?",
                (lock_val, st),
            ).fetchone()
            if row is None:
                skip_coherence = True
                break
        if not skip_coherence:
            max_retries = 20

    # Determine remix classification
    force_type = None
    if has_locks:
        if "of_modifier" in locked_parts or "of_head" in locked_parts:
            force_type = "type1"
        elif "of_topic" in locked_parts or "of_complement" in locked_parts:
            force_type = "type2"

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

    if force_type == "type1":
        if "of_modifier" in locked_parts:
            word_count = len(locked_parts["of_modifier"].split()) + 1
        elif orig_classification and orig_classification[0] == "type1":
            word_count = orig_classification[-1]
        else:
            word_count = 2
        classification = ("type1", word_count)
    elif force_type == "type2":
        if orig_classification and orig_classification[0] == "type2":
            _, prep, word_count = orig_classification
        else:
            # Infer prep from locked value in slot_fillers
            prep = None
            for lk in ("of_topic", "of_complement"):
                if lk in locked_parts:
                    row = conn.execute(
                        "SELECT prep FROM slot_fillers WHERE filler = ? AND slot_type = ? LIMIT 1",
                        (locked_parts[lk], lk),
                    ).fetchone()
                    if row:
                        prep = row[0]
                        break
            if prep is None:
                return None
            word_count = 0
        classification = ("type2", prep, word_count)
    else:
        classification = orig_classification
        if classification is None:
            return None

    # Reject type-2 remixes where inner prep is "of" (produces double-of)
    cfg = load_tuning_config(conn)
    if cfg.get("remix_reject_double_of", 1.0) > 0:
        if classification[0] == "type2" and classification[1] == "of":
            return None

    best_attempt = None
    best_sim = -1.0

    for _ in range(max_retries):
        if classification[0] == "type1":
            _, word_count = classification
            result = compose_compound(
                conn, rng, tone_target, ctx, word_count,
                locked_modifier=locked_parts.get("of_modifier") if has_locks else None,
                locked_head=locked_parts.get("of_head") if has_locks else None,
            )
        else:
            _, prep, word_count = classification
            result = compose_prepositional(
                conn, rng, tone_target, ctx, prep, word_count,
                locked_topic=locked_parts.get("of_topic") if has_locks else None,
                locked_complement=locked_parts.get("of_complement") if has_locks else None,
            )

        if result is None:
            continue

        composed, parts = result

        # Compute similarity when coherence check is active or locks present
        sim = None
        if min_sim > 0 or has_locks:
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

        # Coherence check (skipped for custom locked values)
        if not skip_coherence and min_sim > 0 and sim is not None:
            if sim < min_sim:
                if has_locks and (best_attempt is None or sim > best_sim):
                    best_sim = sim
                    best_attempt = (composed, parts, sim)
                continue

        return composed, parts, sim

    # With locks, fall back to the best attempt seen
    if has_locks and best_attempt is not None:
        return best_attempt

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
