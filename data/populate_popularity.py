"""Create popularity tables and populate work-level scores.

Phase 2: Join SPL + OL data via isbn_aliases → work_key,
aggregate at work level, compute per-filler popularity scores.

Supports tunable weights via config table or CLI overrides.
"""

import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

DB_PATH = Path("data/db/subtitles.db")
SPL_PATH = Path("data/spl_checkout_lookup.json")
OL_PATH = Path("data/ol_edition_lookup.json")
GR_PATH = Path("data/goodreads_lookup.json")
CANADIAN_PATH = Path("data/canadian_library_lookup.json")
NYT_PATH = Path("data/nyt_bestseller_lookup.json")
LIBRARY_PATH = Path("data/library_lists_lookup.json")
WIKI_PATH = Path("data/wikipedia_bestsellers_lookup.json")

# Ensure src is importable (for config access)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subtitle_generator.config import load_tuning_config, ALL_TUNABLE_PARAMS


def create_tables(conn: sqlite3.Connection):
    """Create popularity_data and add columns to slot_fillers."""
    print("Creating popularity tables...")

    conn.execute("DROP TABLE IF EXISTS popularity_data")
    conn.execute("""
        CREATE TABLE popularity_data (
            work_key TEXT PRIMARY KEY,
            spl_checkouts INTEGER DEFAULT 0,
            spl_years INTEGER DEFAULT 0,
            spl_earliest_pub_year TEXT,
            ol_edition_count INTEGER DEFAULT 1,
            checkouts_per_year REAL,
            editions_per_decade REAL,
            gr_ratings_count INTEGER DEFAULT 0,
            gr_average_rating REAL,
            nyt_weeks_on_list INTEGER DEFAULT 0,
            nyt_peak_rank INTEGER,
            library_appearances INTEGER DEFAULT 0,
            composite_score REAL
        )
    """)
    conn.execute("CREATE INDEX idx_pop_composite ON popularity_data(composite_score)")

    conn.execute("DROP TABLE IF EXISTS word_popularity")
    conn.execute("""
        CREATE TABLE word_popularity (
            word TEXT PRIMARY KEY,
            spl_distinct_works INTEGER DEFAULT 0,
            spl_trimmed_mean_checkouts REAL DEFAULT 0,
            ol_distinct_works INTEGER DEFAULT 0,
            composite_score REAL
        )
    """)

    # Add columns to slot_fillers if not present
    cols = {r[1] for r in conn.execute("PRAGMA table_info(slot_fillers)")}
    if "popularity_score" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN popularity_score REAL")
    if "popularity_level" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN popularity_level INTEGER")
    if "popularity_confidence" not in cols:
        conn.execute("ALTER TABLE slot_fillers ADD COLUMN popularity_confidence REAL")

    conn.commit()
    print("  Tables created.")


def load_goodreads(conn: sqlite3.Connection, gr: dict) -> dict[str, dict]:
    """Map Goodreads data to work_keys via isbn_aliases."""
    work_gr: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT isbn, work_key FROM isbn_aliases WHERE work_key IS NOT NULL"
    ).fetchall()
    isbn_to_work = {r[0]: r[1] for r in rows}

    matched = 0
    for isbn, data in gr.items():
        work = isbn_to_work.get(isbn)
        if work:
            # Keep highest ratings_count per work
            existing = work_gr.get(work)
            if not existing or data["ratings_count"] > existing["ratings_count"]:
                work_gr[work] = data
            matched += 1

    print(f"  Goodreads ISBNs matched to works: {matched:,}")
    print(f"  Unique works with Goodreads data: {len(work_gr):,}")
    return work_gr


def load_canadian(conn: sqlite3.Connection, canadian_isbn: dict) -> dict[str, dict]:
    """Map Canadian library ISBN-keyed data to work_keys via isbn_aliases."""
    work_canadian: dict[str, dict] = {}
    rows = conn.execute(
        "SELECT isbn, work_key FROM isbn_aliases WHERE work_key IS NOT NULL"
    ).fetchall()
    isbn_to_work = {r[0]: r[1] for r in rows}

    matched = 0
    for isbn, data in canadian_isbn.items():
        work = isbn_to_work.get(isbn)
        if work:
            holds = data.get("holds_count", data.get("appearances", 0))
            existing = work_canadian.get(work)
            if not existing or holds > existing.get("holds_count", 0):
                work_canadian[work] = {"holds_count": holds, "source": data.get("source", "canadian")}
            matched += 1

    print(f"  Canadian ISBNs matched to works: {matched:,}")
    print(f"  Unique works with Canadian data: {len(work_canadian):,}")
    return work_canadian


def populate_work_level(conn: sqlite3.Connection, spl: dict, ol: dict,
                       w_spl: float = 0.7, w_ol: float = 0.3, exponent: float = 1.0,
                       gr: dict | None = None, w_gr: float = 0.2,
                       canadian_isbn: dict | None = None, w_library: float = 0.05):
    """Aggregate SPL + OL + Goodreads + Canadian library data at work level using isbn_aliases."""
    print(f"\nPopulating work-level popularity_data "
          f"(w_spl={w_spl}, w_ol={w_ol}, w_gr={w_gr}, w_lib={w_library}, exp={exponent})...")

    # Build work_key → aggregated signals
    work_spl: dict[str, dict] = defaultdict(lambda: {"checkouts": 0, "years": set(), "pub_year": ""})
    work_ol: dict[str, int] = {}

    # Get isbn → work_key mapping
    rows = conn.execute(
        "SELECT isbn, work_key FROM isbn_aliases WHERE work_key IS NOT NULL"
    ).fetchall()
    isbn_to_work = {r[0]: r[1] for r in rows}
    print(f"  ISBN→work mappings: {len(isbn_to_work):,}")

    # Aggregate SPL by work_key (dedup across editions)
    spl_matched = 0
    for isbn, data in spl.items():
        work = isbn_to_work.get(isbn)
        if work:
            w = work_spl[work]
            w["checkouts"] += data["total_checkouts"]
            # years_active is an int count, not a set — approximate
            w["years"] = max(len(w["years"]) if isinstance(w["years"], set) else w["years"], data["years_active"])
            if data.get("pub_year") and (not w["pub_year"] or data["pub_year"] < w["pub_year"]):
                w["pub_year"] = data["pub_year"]
            spl_matched += 1

    print(f"  SPL ISBNs matched to works: {spl_matched:,}")
    print(f"  Unique works with SPL data: {len(work_spl):,}")

    # OL edition counts are already per-work in the lookup
    for isbn, data in ol.items():
        work = data["work_key"]
        ec = data["edition_count"]
        # Keep max edition count per work (they should all be the same but just in case)
        work_ol[work] = max(work_ol.get(work, 0), ec)

    print(f"  Unique works with OL data: {len(work_ol):,}")

    # Goodreads: map to work_keys
    if gr:
        work_gr = load_goodreads(conn, gr)
    else:
        work_gr = {}
        print("  Goodreads: skipped (no data)")

    # Canadian libraries: map ISBN-keyed data to work_keys
    if canadian_isbn:
        work_canadian = load_canadian(conn, canadian_isbn)
    else:
        work_canadian = {}
        print("  Canadian libraries: skipped (no data)")

    # Merge into popularity_data
    all_works = set(work_spl.keys()) | set(work_ol.keys()) | set(work_gr.keys()) | set(work_canadian.keys())
    print(f"  Total unique works: {len(all_works):,}")

    batch = []
    for work in all_works:
        spl_data = work_spl.get(work)
        ol_ec = work_ol.get(work, 1)

        spl_co = spl_data["checkouts"] if spl_data else 0
        spl_yrs = spl_data["years"] if spl_data else 0
        if isinstance(spl_yrs, set):
            spl_yrs = len(spl_yrs)
        pub_year = spl_data["pub_year"] if spl_data else ""

        # Temporal normalization
        co_per_year = spl_co / max(spl_yrs, 1) if spl_co > 0 else 0.0
        # For editions: normalize by decades since first edition (approximate)
        ed_per_decade = float(ol_ec)  # we don't have pub year for OL, use raw for now

        # Composite: combine SPL (if available), OL, and Goodreads with tunable weights
        if spl_co > 0:
            spl_signal = w_spl * math.log10(1 + co_per_year) ** exponent
        else:
            spl_signal = 0.0

        ol_signal = w_ol * math.log10(1 + ol_ec) ** exponent

        gr_data = work_gr.get(work)
        if gr_data:
            gr_signal = w_gr * math.log10(1 + gr_data["ratings_count"]) ** exponent
            gr_ratings = gr_data["ratings_count"]
            gr_avg = gr_data.get("average_rating", 0.0)
        else:
            gr_signal = 0.0
            gr_ratings = 0
            gr_avg = 0.0

        composite = spl_signal + ol_signal + gr_signal

        # Canadian library signal
        can_data = work_canadian.get(work)
        if can_data:
            library_signal = w_library * math.log10(1 + can_data["holds_count"]) ** exponent
            library_appearances = can_data["holds_count"]
        else:
            library_signal = 0.0
            library_appearances = 0

        composite += library_signal

        batch.append((work, spl_co, spl_yrs, pub_year, ol_ec, co_per_year, ed_per_decade,
                       gr_ratings, gr_avg, 0, None, library_appearances, composite))

        if len(batch) >= 50000:
            conn.executemany(
                "INSERT OR REPLACE INTO popularity_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                batch,
            )
            batch.clear()

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO popularity_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            batch,
        )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM popularity_data").fetchone()[0]
    with_spl = conn.execute("SELECT COUNT(*) FROM popularity_data WHERE spl_checkouts > 0").fetchone()[0]
    with_gr = conn.execute("SELECT COUNT(*) FROM popularity_data WHERE gr_ratings_count > 0").fetchone()[0]
    with_lib = conn.execute("SELECT COUNT(*) FROM popularity_data WHERE library_appearances > 0").fetchone()[0]
    print(f"\n  popularity_data: {total:,} works")
    print(f"    with SPL: {with_spl:,} | with Goodreads: {with_gr:,} | with Canadian library: {with_lib:,}")

    # Distribution
    scores = conn.execute(
        "SELECT composite_score FROM popularity_data ORDER BY composite_score"
    ).fetchall()
    vals = [r[0] for r in scores]
    n = len(vals)
    print(f"  Composite score distribution:")
    print(f"    min={vals[0]:.3f}, median={vals[n//2]:.3f}, mean={sum(vals)/n:.3f}, "
          f"p90={vals[int(n*0.9)]:.3f}, max={vals[-1]:.3f}")


def score_fillers_level1(conn: sqlite3.Connection):
    """Compute Level 1 popularity scores for fillers with ISBN sources."""
    print("\nScoring fillers (Level 1: ISBN-direct)...")

    # For each filler, aggregate popularity across all source works
    # Strategy: take the MAX composite score across all source works
    # (a filler from one popular book should be scored as popular)
    updated = conn.execute("""
        UPDATE slot_fillers SET
            popularity_score = (
                SELECT MAX(pd.composite_score)
                FROM slot_filler_sources sfs
                JOIN subtitles s ON sfs.subtitle_id = s.id
                JOIN isbn_aliases ia ON ia.isbn = s.isbn
                JOIN popularity_data pd ON pd.work_key = ia.work_key
                WHERE sfs.slot_filler_id = slot_fillers.id
            ),
            popularity_level = 1,
            popularity_confidence = 1.0
        WHERE EXISTS (
            SELECT 1
            FROM slot_filler_sources sfs
            JOIN subtitles s ON sfs.subtitle_id = s.id
            JOIN isbn_aliases ia ON ia.isbn = s.isbn
            JOIN popularity_data pd ON pd.work_key = ia.work_key
            WHERE sfs.slot_filler_id = slot_fillers.id
        )
    """).rowcount
    conn.commit()

    print(f"  Updated {updated:,} fillers with Level 1 scores")

    # Stats
    for stype in ["list_item", "action_noun", "of_object"]:
        row = conn.execute("""
            SELECT COUNT(*), AVG(popularity_score), MAX(popularity_score)
            FROM slot_fillers
            WHERE slot_type = ? AND mode = 'strict' AND popularity_level = 1
        """, (stype,)).fetchone()
        total = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict'", (stype,)
        ).fetchone()[0]
        if row[0]:
            print(f"  {stype}: {row[0]}/{total} ({row[0]*100//total}%) "
                  f"avg={row[1]:.3f} max={row[2]:.3f}")


def score_fillers_fallback(conn: sqlite3.Connection):
    """Set fallback scores for fillers without Level 1 data."""
    print("\nSetting fallback scores (Level 0: freq-based)...")

    updated = conn.execute("""
        UPDATE slot_fillers SET
            popularity_score = CASE
                WHEN freq > 0 THEN log(1 + freq) / log(10)
                ELSE 0.0
            END,
            popularity_level = 0,
            popularity_confidence = 0.0
        WHERE popularity_level IS NULL AND mode = 'strict'
    """).rowcount
    conn.commit()

    print(f"  Set fallback for {updated:,} fillers")


def report(conn: sqlite3.Connection):
    """Final coverage report."""
    print("\n=== Coverage Report ===")
    for stype in ["list_item", "action_noun", "of_object", "of_modifier", "of_head", "of_topic", "of_complement"]:
        total = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict'", (stype,)
        ).fetchone()[0]
        if total == 0:
            continue
        l1 = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict' AND popularity_level = 1", (stype,)
        ).fetchone()[0]
        l0 = conn.execute(
            "SELECT COUNT(*) FROM slot_fillers WHERE slot_type = ? AND mode = 'strict' AND popularity_level = 0", (stype,)
        ).fetchone()[0]
        print(f"  {stype}: {total:,} total | L1={l1} ({l1*100//total}%) | fallback={l0} ({l0*100//total}%)")

    # Compare L1 scores vs freq-based scores
    print("\n=== Score Comparison (L1 fillers) ===")
    rows = conn.execute("""
        SELECT slot_type, filler, freq, popularity_score
        FROM slot_fillers
        WHERE popularity_level = 1 AND mode = 'strict'
        ORDER BY popularity_score DESC
        LIMIT 15
    """).fetchall()
    print("Top 15 by popularity_score:")
    for stype, filler, freq, score in rows:
        freq_score = math.log10(1 + freq) if freq > 0 else 0
        print(f"  {stype}: {filler} | pop={score:.3f} freq_score={freq_score:.3f} (freq={freq})")


def calibrate_thresholds(conn: sqlite3.Connection):
    """Compute data-driven thresholds from popularity_score distribution.

    Uses percentile-based cutoffs on the classification-blend score:
    - Pop: top ~8% of fillers (p92+)
    - Mainstream: next ~28% (p64-p92)
    - Niche: bottom ~64% (below p64)

    Writes recommended values to the config table.
    """
    cfg = load_tuning_config(conn)
    blend = cfg.get("pop_classification_blend", 0.9)
    pop_default = cfg.get("pop_missing_default", 0.1)

    # Compute blended classification scores for all strict fillers
    rows = conn.execute(
        "SELECT freq, popularity_score FROM slot_fillers WHERE mode = 'strict'"
    ).fetchall()

    scores = []
    for freq, pop_score in rows:
        score_freq = math.log10(1 + freq)
        ps = pop_score if pop_score is not None else pop_default
        blended = (1 - blend) * score_freq + blend * ps
        scores.append(blended)

    scores.sort()
    n = len(scores)

    def pctile(p):
        idx = int(n * p / 100)
        return scores[min(idx, n - 1)]

    # Pop threshold: p92 (top ~8%)
    pop_thresh = pctile(92)
    # Mainstream threshold: p64 (top ~36%)
    main_thresh = pctile(64)

    # Tier centers: median of each tier's scores
    pop_scores = [s for s in scores if s >= pop_thresh]
    main_scores = [s for s in scores if main_thresh <= s < pop_thresh]
    niche_scores = [s for s in scores if s < main_thresh]

    pop_center = pop_scores[len(pop_scores) // 2] if pop_scores else pop_thresh
    main_center = main_scores[len(main_scores) // 2] if main_scores else (pop_thresh + main_thresh) / 2
    niche_center = niche_scores[len(niche_scores) // 2] if niche_scores else main_thresh / 2

    print(f"\n=== Threshold Calibration (blend={blend}) ===")
    print(f"  Distribution: n={n}, min={scores[0]:.3f}, median={pctile(50):.3f}, max={scores[-1]:.3f}")
    print(f"\n  Recommended thresholds:")
    print(f"    accessibility_threshold_pop:         {pop_thresh:.3f}  (p92, {len(pop_scores)} fillers)")
    print(f"    accessibility_threshold_mainstream:   {main_thresh:.3f}  (p64, {len(main_scores)} fillers)")
    print(f"    niche: {len(niche_scores)} fillers")
    print(f"\n  Recommended tier centers:")
    print(f"    tier_center_pop:         {pop_center:.3f}")
    print(f"    tier_center_mainstream:  {main_center:.3f}")
    print(f"    tier_center_niche:       {niche_center:.3f}")

    # Show example fillers near boundaries
    print(f"\n  Example fillers near pop threshold ({pop_thresh:.3f}):")
    boundary_rows = conn.execute(
        "SELECT slot_type, filler, freq, popularity_score FROM slot_fillers "
        "WHERE mode = 'strict' AND slot_type IN ('list_item', 'action_noun', 'of_object') "
        "ORDER BY ABS(popularity_score - ?) LIMIT 10", (pop_thresh,)
    ).fetchall()
    for stype, filler, freq, ps in boundary_rows:
        score_freq = math.log10(1 + freq)
        ps_val = ps if ps is not None else pop_default
        blended = (1 - blend) * score_freq + blend * ps_val
        tier = "POP" if blended >= pop_thresh else ("MAIN" if blended >= main_thresh else "NICHE")
        print(f"    {tier:5s} {blended:.3f}  {stype:15s}  {filler}")

    # Write to config table
    params = {
        "accessibility_threshold_pop": round(pop_thresh, 4),
        "accessibility_threshold_mainstream": round(main_thresh, 4),
        "tier_center_pop": round(pop_center, 4),
        "tier_center_mainstream": round(main_center, 4),
        "tier_center_niche": round(niche_center, 4),
        "tone_target_pop_list_item": round(pop_center, 4),
        "tone_target_pop_action_noun": round(pop_center, 4),
        "tone_target_pop_of_object": round(pop_center, 4),
        "tone_target_mainstream_list_item": round(main_center, 4),
        "tone_target_mainstream_action_noun": round(main_center, 4),
        "tone_target_mainstream_of_object": round(main_center, 4),
        "tone_target_niche_list_item": round(niche_center, 4),
        "tone_target_niche_action_noun": round(niche_center, 4),
        "tone_target_niche_of_object": round(niche_center, 4),
    }

    conn.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
    for key, value in params.items():
        conn.execute(
            "INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)",
            (key, str(value))
        )
    conn.commit()

    print(f"\n  Written {len(params)} config values to DB.")

    # Before/after comparison
    old_cfg = dict(ALL_TUNABLE_PARAMS)
    print(f"\n  Before → After:")
    for key, new_val in params.items():
        old_val = old_cfg.get(key, "N/A")
        print(f"    {key}: {old_val} → {new_val}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Populate popularity scores with tunable weights")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to subtitles.db")
    parser.add_argument("--spl", type=float, default=None, help="Override pop_weight_spl")
    parser.add_argument("--ol", type=float, default=None, help="Override pop_weight_ol")
    parser.add_argument("--gr", type=float, default=None, help="Override pop_weight_gr")
    parser.add_argument("--library", type=float, default=None, help="Override pop_weight_library")
    parser.add_argument("--exponent", type=float, default=None, help="Override pop_exponent")
    parser.add_argument("--skip-calibrate", action="store_true", help="Skip threshold calibration")
    args = parser.parse_args()

    print("Loading data...")
    with open(SPL_PATH) as f:
        spl = json.load(f)
    with open(OL_PATH) as f:
        ol = json.load(f)
    print(f"  SPL: {len(spl):,} ISBNs, OL: {len(ol):,} ISBNs")

    # Optional new sources
    if GR_PATH.exists():
        with open(GR_PATH) as f:
            gr = json.load(f)
        print(f"  Goodreads: {len(gr):,} ISBNs")
    else:
        gr = {}
        print("  Goodreads: not found (skipping)")

    if CANADIAN_PATH.exists():
        with open(CANADIAN_PATH) as f:
            canadian = json.load(f)
        canadian_isbn = {k: v for k, v in canadian.items() if "|" not in k and k.replace("-", "").isdigit()}
        canadian_title = {k: v for k, v in canadian.items() if "|" in k}
        print(f"  Canadian libraries: {len(canadian):,} total "
              f"({len(canadian_isbn):,} ISBN-keyed, {len(canadian_title):,} title-keyed)")
        del canadian  # free memory
    else:
        canadian_isbn = {}
        print("  Canadian libraries: not found (skipping)")

    # Log presence of other optional sources (not yet wired into composite)
    for label, path in [("NYT bestsellers", NYT_PATH),
                        ("Library lists", LIBRARY_PATH),
                        ("Wikipedia bestsellers", WIKI_PATH)]:
        if path.exists():
            print(f"  {label}: found ({path})")
        else:
            print(f"  {label}: not found (skipping)")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL")

    # Load weights from config, allow CLI overrides
    cfg = load_tuning_config(conn)
    w_spl = args.spl if args.spl is not None else cfg["pop_weight_spl"]
    w_ol = args.ol if args.ol is not None else cfg["pop_weight_ol"]
    w_gr = args.gr if args.gr is not None else cfg["pop_weight_gr"]
    w_library = args.library if args.library is not None else cfg["pop_weight_library"]
    exponent = args.exponent if args.exponent is not None else cfg["pop_exponent"]
    print(f"  Weights: SPL={w_spl}, OL={w_ol}, GR={w_gr}, LIB={w_library}, exponent={exponent}")

    start = time.time()
    create_tables(conn)
    populate_work_level(conn, spl, ol, w_spl=w_spl, w_ol=w_ol, exponent=exponent,
                        gr=gr, w_gr=w_gr,
                        canadian_isbn=canadian_isbn, w_library=w_library)
    score_fillers_level1(conn)
    score_fillers_fallback(conn)
    report(conn)

    if not args.skip_calibrate:
        calibrate_thresholds(conn)

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    conn.close()


if __name__ == "__main__":
    main()
