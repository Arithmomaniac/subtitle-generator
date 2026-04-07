"""Two-pass tuning: rule-based pruning + spaCy vector similarity filtering.

No LLM calls. All filtering is local using spaCy's POS/NER/dependency
parsing and word vectors for cosine similarity to the strict seed set.
"""

import sqlite3

import click
import numpy as np
import spacy

from subtitle_generator.generate import generate_subtitle


def _load_nlp():
    """Load medium model (has word vectors)."""
    return spacy.load("en_core_web_md", disable=["lemmatizer"])


# --- Pass 1: Rule-based filters ---

def _rule_prune(conn: sqlite3.Connection, nlp):
    """Bulk-delete obvious junk using spaCy parsing + cheap heuristics."""
    click.echo("Pass 1: Rule-based pruning...")
    total_cut = 0

    for slot_type in ["list_item", "action_noun", "of_object"]:
        rows = conn.execute(
            "SELECT id, filler FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'",
            (slot_type,),
        ).fetchall()

        ids = [r[0] for r in rows]
        fillers = [r[1] for r in rows]
        cut_ids = []

        # Batch parse with nlp.pipe for speed
        for fid, filler, doc in zip(ids, fillers, nlp.pipe(fillers, batch_size=500)):
            if _should_cut_by_rules(filler, slot_type, doc):
                cut_ids.append(fid)

        # Batch delete
        for i in range(0, len(cut_ids), 1000):
            _batch_delete(conn, cut_ids[i:i + 1000])
        total_cut += len(cut_ids)

        remaining = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'",
            (slot_type,),
        ).fetchone()[0]
        click.echo(f"  {slot_type}: {remaining:,} remaining")

    click.echo(f"  Total rule-pruned: {total_cut:,}")
    return total_cut


def _should_cut_by_rules(filler: str, slot_type: str, doc) -> bool:
    """Return True if this filler should be cut based on rules. `doc` is pre-parsed."""
    import re

    # Universal: dates
    if re.search(r"\d{4}", filler):
        return True

    if slot_type == "list_item":
        words = filler.split()
        # Too long
        if len(words) > 3:
            return True
        # Starts with preposition/conjunction
        first = words[0].lower() if words else ""
        if first in ("of", "in", "on", "for", "with", "from", "to", "and",
                      "or", "by", "at", "as", "including", "containing",
                      "being", "having", "wherein", "whereby", "also",
                      "especially", "particularly", "namely"):
            return True
        # "the X" where "the" isn't needed
        if first == "the" and len(words) >= 2:
            # Keep if NER tags it as an entity
            if any(ent.label_ in ("GPE", "ORG", "EVENT", "FAC", "LOC",
                                   "WORK_OF_ART", "LAW") for ent in doc.ents):
                return False
            # Keep if second word is capitalized (likely proper noun)
            if words[1][0].isupper():
                return False
            return True
        # Root is a verb → clause fragment
        root_pos = [t.pos_ for t in doc if t.dep_ == "ROOT"]
        if root_pos and root_pos[0] == "VERB":
            return True
        # No noun at all
        if not any(t.pos_ in ("NOUN", "PROPN", "ADJ") for t in doc):
            return True

    elif slot_type == "action_noun":
        words = filler.split()
        if len(words) > 2:
            return True
        # Head word (last token) must be NOUN
        if doc and doc[-1].pos_ not in ("NOUN",):
            return True
        # Check suffix/whitelist
        from subtitle_generator.slots import ACTION_SUFFIXES, ACTION_WHITELIST
        head = words[-1].lower()
        if head not in ACTION_WHITELIST and not any(head.endswith(s) for s in ACTION_SUFFIXES):
            return True

    elif slot_type == "of_object":
        words = filler.split()
        if len(words) > 6:
            return True
        first = words[0].lower() if words else ""
        if first in ("their", "its", "his", "her", "our", "my", "your"):
            return True
        if not any(t.pos_ in ("NOUN", "PROPN") for t in doc):
            return True

    return False


def _batch_delete(conn: sqlite3.Connection, ids: list[int]):
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM slot_fillers WHERE id IN ({placeholders})", ids)
    conn.commit()


# --- Pass 2: Vector similarity filtering ---

def _compute_centroid(conn: sqlite3.Connection, slot_type: str, nlp) -> np.ndarray | None:
    """Compute the average word vector of the strict (seed) fillers for a slot type."""
    rows = conn.execute(
        "SELECT filler FROM slot_fillers WHERE slot_type = ? AND mode = 'strict'",
        (slot_type,),
    ).fetchall()

    if not rows:
        return None

    vectors = []
    for (filler,) in rows:
        doc = nlp(filler)
        if doc.has_vector and doc.vector_norm > 0:
            vectors.append(doc.vector)

    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def _vector_prune(conn: sqlite3.Connection, nlp,
                  max_per_slot: dict | None = None):
    """Cut loose fillers whose vectors are far from the strict seed centroid.
    
    Keeps at most `max_per_slot[slot_type]` loose fillers, ranked by similarity.
    """
    if max_per_slot is None:
        # Scaled ~2.7x from original caps to match strict pool growth
        max_per_slot = {
            "list_item": 22_000,
            "action_noun": 5_000,
            "of_object": 22_000,
        }

    click.echo("\nPass 2: Vector similarity pruning...")

    for slot_type in ["list_item", "action_noun", "of_object"]:
        centroid = _compute_centroid(conn, slot_type, nlp)
        if centroid is None:
            click.echo(f"  {slot_type}: no centroid (no strict fillers?), skipping")
            continue

        rows = conn.execute(
            "SELECT id, filler FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'",
            (slot_type,),
        ).fetchall()

        # Batch vectorize with nlp.pipe() for speed
        ids = [r[0] for r in rows]
        fillers = [r[1] for r in rows]
        scored = []
        for doc, fid in zip(nlp.pipe(fillers, batch_size=500), ids):
            if doc.has_vector and doc.vector_norm > 0:
                sim = _cosine_sim(centroid, doc.vector)
            else:
                sim = 0.0
            scored.append((fid, "", sim))

        if not scored:
            continue

        # Sort by similarity descending, keep top N
        scored.sort(key=lambda x: x[2], reverse=True)
        max_keep = max_per_slot.get(slot_type, 5000)
        to_cut = scored[max_keep:]  # everything beyond the top N

        cut_ids = [fid for fid, _, _ in to_cut]

        if cut_ids:
            # Delete in batches
            for i in range(0, len(cut_ids), 1000):
                _batch_delete(conn, cut_ids[i:i + 1000])

        remaining = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'",
            (slot_type,),
        ).fetchone()[0]
        threshold = scored[min(max_keep, len(scored)) - 1][2] if scored else 0
        click.echo(f"  {slot_type}: cut {len(cut_ids):,} (kept top {max_keep:,}, "
                    f"sim threshold ~{threshold:.3f}), {remaining:,} remaining")


def run_autoresearch(conn: sqlite3.Connection, **_kwargs):
    """Two-pass tuning: rules then vector similarity."""
    click.echo("=== Autoresearch: Two-Pass Filler Pruning ===\n")

    click.echo("Loading spaCy model (en_core_web_md for vectors)...")
    nlp = _load_nlp()

    # Pass 1: Rules
    _rule_prune(conn, nlp)

    # Pass 2: Vector similarity
    _vector_prune(conn, nlp)

    # Final stats and sample
    click.echo("\n=== Results ===\n")
    for st in ["list_item", "action_noun", "of_object"]:
        strict = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict'", (st,)
        ).fetchone()[0]
        loose = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'loose'", (st,)
        ).fetchone()[0]
        click.echo(f"  {st}: {strict:,} strict + {loose:,} loose = {strict + loose:,} total")

    click.echo("\nSample loose output after tuning:")
    for i in range(15):
        s = generate_subtitle(conn, mode="loose")
        click.echo(f"  {i + 1:2d}. {s}")
