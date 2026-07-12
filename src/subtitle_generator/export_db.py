"""Export data for deployment and build mini SQLite from exported data."""

import csv
import os
import sqlite3
import tempfile
from pathlib import Path

from subtitle_generator.schema_contracts import (
    MINI_DB_SCHEMA_CONTRACTS,
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    validate_schema,
    validate_tier_slot_distribution,
)
from subtitle_generator.shadow_runtime import (
    ShadowArtifactSource,
    load_shadow_distribution_source,
    write_shadow_distribution_csv,
)
from subtitle_generator.config import ALL_TUNABLE_PARAMS

_CURRENT_PATTERN_LIST_ITEM_COUNTS = (2, 3)
_RUNTIME_CONFIG_KEYS = frozenset(ALL_TUNABLE_PARAMS) | frozenset(
    {
        "article_stats_action_noun",
        "article_stats_of_object",
        "avg_cross_sim_t1",
        "avg_cross_sim_t2",
        "centroid_norm",
        "embedding_centroid",
        "embedding_version",
        "remix_calibrated_min_sim",
        "remix_calibrated_remix_prob",
        "remix_head_pos",
        "remix_mod_pos_2word",
        "remix_mod_pos_3word",
        "remix_prep_groups",
    }
)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(conn, table):
        return set()
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _can_filter_to_current_pattern_matches(conn: sqlite3.Connection) -> bool:
    return {"subtitle_id", "list_items_json"} <= _columns(conn, "pattern_matches")


def _model_scores_table_exists(conn: sqlite3.Connection) -> bool:
    return _table_exists(conn, "slot_filler_model_scores")


def _exportable_slot_fillers_cte(conn: sqlite3.Connection) -> str:
    """Return a CTE exposing slot fillers that still satisfy current slot gates."""

    slot_columns = (
        "id, slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep, "
        "remix_type, remix_prep, remix_word_count, centroid_dot, norm_sq, "
        "token_count, popularity_score, popularity_level, popularity_confidence"
    )
    if not _can_filter_to_current_pattern_matches(conn):
        return f"WITH exportable_slot_fillers AS (SELECT {slot_columns} FROM slot_fillers)\n"

    allowed_counts = ", ".join(str(count) for count in _CURRENT_PATTERN_LIST_ITEM_COUNTS)
    valid_sources_cte = f"""
WITH valid_pattern_sources AS (
    SELECT subtitle_id
    FROM pattern_matches
    WHERE subtitle_id IS NOT NULL
      AND json_valid(list_items_json)
      AND json_array_length(list_items_json) IN ({allowed_counts})
)
"""
    if _table_exists(conn, "slot_filler_sources"):
        return valid_sources_cte + """,
linked_valid_sources AS (
    SELECT sfs.slot_filler_id, MIN(sfs.subtitle_id) AS subtitle_id
    FROM slot_filler_sources sfs
    JOIN valid_pattern_sources vps ON vps.subtitle_id = sfs.subtitle_id
    GROUP BY sfs.slot_filler_id
),
fallback_valid_sources AS (
    SELECT sf.id AS slot_filler_id, sf.source_subtitle_id AS subtitle_id
    FROM slot_fillers sf
    JOIN valid_pattern_sources vps ON vps.subtitle_id = sf.source_subtitle_id
),
exportable_slot_fillers AS (
    SELECT
        sf.id,
        sf.slot_type,
        sf.filler,
        sf.mode,
        COALESCE(lvs.subtitle_id, fvs.subtitle_id) AS source_subtitle_id,
        sf.freq,
        sf.pos_tag,
        sf.prep,
        sf.remix_type,
        sf.remix_prep,
        sf.remix_word_count,
        sf.centroid_dot,
        sf.norm_sq,
        sf.token_count,
        sf.popularity_score,
        sf.popularity_level,
        sf.popularity_confidence
    FROM slot_fillers sf
    LEFT JOIN linked_valid_sources lvs ON lvs.slot_filler_id = sf.id
    LEFT JOIN fallback_valid_sources fvs ON fvs.slot_filler_id = sf.id
    WHERE sf.source_subtitle_id IS NULL
       OR COALESCE(lvs.subtitle_id, fvs.subtitle_id) IS NOT NULL
)
"""

    return valid_sources_cte + f""",
exportable_slot_fillers AS (
    SELECT {slot_columns}
    FROM slot_fillers sf
    WHERE sf.source_subtitle_id IS NULL
       OR EXISTS (
            SELECT 1
            FROM valid_pattern_sources vps
            WHERE vps.subtitle_id = sf.source_subtitle_id
       )
)
"""


def export_data(
    source_conn: sqlite3.Connection,
    output_dir: Path,
    *,
    shadow_distribution_source: Path | None = None,
) -> dict:
    """Export slot_fillers, config, and sources as CSV files.

    These text files are committed to the repo and used by ``build_mini_db``
    in CI to construct the SQLite deployment artifact.

    Returns stats dict: {filename: row_count, ...}.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    # -- slot_fillers (with scalar decomposition and remix columns) --
    exportable_cte = _exportable_slot_fillers_cte(source_conn)
    rows = source_conn.execute(
        exportable_cte
        + "SELECT id, slot_type, filler, mode, source_subtitle_id, freq, pos_tag, prep, "
        "remix_type, remix_prep, remix_word_count, centroid_dot, norm_sq, token_count, "
        "popularity_score, popularity_level, popularity_confidence "
        "FROM exportable_slot_fillers "
        "ORDER BY id"
    ).fetchall()
    path = output_dir / "slot_fillers.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "slot_type", "filler", "mode", "source_subtitle_id", "freq",
            "pos_tag", "prep", "remix_type", "remix_prep", "remix_word_count",
            "centroid_dot", "norm_sq", "token_count", "popularity_score",
            "popularity_level", "popularity_confidence",
        ])
        for row in rows:
            row = list(row)
            # Convert None floats to empty strings for CSV
            for idx in (11, 12, 14, 15, 16):
                if row[idx] is None:
                    row[idx] = ""
            w.writerow(row)
    stats["slot_fillers.csv"] = len(rows)

    # -- optional model tier probabilities (runtime pure-categorization weights) --
    model_path = output_dir / "slot_filler_model_scores.csv"
    if _model_scores_table_exists(source_conn):
        rows = source_conn.execute(
            exportable_cte
            + """
            SELECT
                sf.id,
                COALESCE(ms.score_pop, 0.0),
                COALESCE(ms.score_mainstream, 0.0),
                COALESCE(ms.score_niche, 0.0),
                COALESCE(ms.model_tier, ''),
                COALESCE(ms.source_prediction_count, 0)
            FROM exportable_slot_fillers sf
            JOIN slot_filler_model_scores ms ON ms.slot_filler_id = sf.id
            ORDER BY sf.id
            """
        ).fetchall()
        with open(model_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "slot_filler_id",
                "score_pop",
                "score_mainstream",
                "score_niche",
                "model_tier",
                "source_prediction_count",
            ])
            w.writerows(rows)
        stats["slot_filler_model_scores.csv"] = len(rows)
    elif model_path.exists():
        model_path.unlink()

    # -- optional tier-slot runtime distribution --
    shadow_path = output_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv"
    if shadow_distribution_source is not None:
        loaded = load_shadow_distribution_source(
            source_conn,
            ShadowArtifactSource.csv_path(shadow_distribution_source),
        )
        write_shadow_distribution_csv(shadow_path, list(loaded.rows))
        stats[shadow_path.name] = len(loaded.rows)
    elif _table_exists(source_conn, TIER_SLOT_FILLER_DISTRIBUTION_TABLE):
        loaded = load_shadow_distribution_source(
            source_conn,
            ShadowArtifactSource.db_table(),
        )
        write_shadow_distribution_csv(shadow_path, list(loaded.rows))
        stats[shadow_path.name] = len(loaded.rows)
    elif shadow_path.exists():
        shadow_path.unlink()

    # -- config --
    placeholders = ", ".join("?" for _ in _RUNTIME_CONFIG_KEYS)
    rows = source_conn.execute(
        "SELECT key, value FROM config "
        f"WHERE key IN ({placeholders}) "
        "ORDER BY key",
        tuple(sorted(_RUNTIME_CONFIG_KEYS)),
    ).fetchall()
    path = output_dir / "config.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows(rows)
    stats["config.csv"] = len(rows)

    # -- sources (pre-joined: slot_filler -> source book) --
    rows = source_conn.execute(
        exportable_cte
        + "SELECT sf.id, s.title, s.subtitle, "
        "CASE WHEN s.source_file = 'openlibrary' THEN 'OL' ELSE 'LOC' END "
        "FROM exportable_slot_fillers sf "
        "JOIN subtitles s ON s.id = sf.source_subtitle_id "
        "WHERE sf.source_subtitle_id IS NOT NULL "
        "ORDER BY sf.id"
    ).fetchall()
    path = output_dir / "sources.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slot_filler_id", "title", "subtitle_text", "source_tag"])
        w.writerows(rows)
    stats["sources.csv"] = len(rows)

    return stats


def build_mini_db(data_dir: Path, output_path: Path) -> dict:
    """Build a minimal SQLite DB from exported CSV files.

    Reads slot_fillers.csv, config.csv, and sources.csv from ``data_dir``,
    creates an indexed SQLite database at ``output_path``.

    Returns stats dict: {table: row_count, ...}.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_handle = tempfile.NamedTemporaryFile(
        prefix=f"{output_path.stem}-",
        suffix=".tmp.db",
        dir=output_path.parent,
        delete=False,
    )
    temp_path = Path(temp_handle.name)
    temp_handle.close()

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(str(temp_path))
        stats = _populate_mini_db(conn, data_dir)
        conn.commit()
        issues = validate_schema(conn, MINI_DB_SCHEMA_CONTRACTS)
        if issues:
            detail = "\n".join(issue.message for issue in issues)
            raise RuntimeError(f"Mini DB schema validation failed:\n{detail}")
        conn.execute("VACUUM")
        conn.close()
        conn = None
        os.replace(temp_path, output_path)
        return stats
    except Exception:
        if conn is not None:
            conn.close()
        if temp_path.exists():
            temp_path.unlink()
        raise


def _populate_mini_db(conn: sqlite3.Connection, data_dir: Path) -> dict:
    stats: dict[str, int] = {}

    # -- slot_fillers (with scalar decomposition + remix columns) --
    conn.execute("""
        CREATE TABLE slot_fillers (
            id INTEGER PRIMARY KEY,
            slot_type TEXT NOT NULL,
            filler TEXT NOT NULL,
            mode TEXT NOT NULL DEFAULT 'strict',
            source_subtitle_id INTEGER,
            freq INTEGER NOT NULL DEFAULT 1,
            pos_tag TEXT,
            prep TEXT,
            remix_type TEXT,
            remix_prep TEXT,
            remix_word_count INTEGER,
            vector_sum BLOB,
            centroid_dot REAL,
            norm_sq REAL,
            token_count INTEGER,
            popularity_score REAL,
            popularity_level INTEGER DEFAULT 0,
            popularity_confidence REAL DEFAULT 0.0,
            UNIQUE(slot_type, filler)
        )
    """)
    sf_path = data_dir / "slot_fillers.csv"
    with open(sf_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            centroid_dot = float(row["centroid_dot"]) if row.get("centroid_dot") else None
            norm_sq = float(row["norm_sq"]) if row.get("norm_sq") else None
            popularity_score = float(row["popularity_score"]) if row.get("popularity_score") else None
            popularity_level = int(row.get("popularity_level") or 0)
            popularity_confidence = float(row.get("popularity_confidence") or 0.0)
            rows.append((
                int(row["id"]), row["slot_type"], row["filler"], row["mode"],
                int(row["source_subtitle_id"]) if row["source_subtitle_id"] else None,
                int(row["freq"]), row["pos_tag"] or None, row["prep"] or None,
                row.get("remix_type") or None,
                row.get("remix_prep") or None,
                int(row["remix_word_count"]) if row.get("remix_word_count") else None,
                None,
                centroid_dot,
                norm_sq,
                int(row["token_count"]) if row.get("token_count") else None,
                popularity_score,
                popularity_level,
                popularity_confidence,
            ))
        conn.executemany(
            "INSERT INTO slot_fillers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        stats["slot_fillers"] = len(rows)

    # -- config --
    conn.execute("CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT)")
    cfg_path = data_dir / "config.csv"
    with open(cfg_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(row["key"], row["value"]) for row in reader]
        conn.executemany("INSERT INTO config VALUES (?, ?)", rows)
        stats["config"] = len(rows)

    # -- sources --
    conn.execute("""
        CREATE TABLE sources (
            slot_filler_id INTEGER NOT NULL,
            title TEXT,
            subtitle_text TEXT,
            source_tag TEXT,
            FOREIGN KEY (slot_filler_id) REFERENCES slot_fillers(id)
        )
    """)
    src_path = data_dir / "sources.csv"
    with open(src_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(int(row["slot_filler_id"]), row["title"], row["subtitle_text"], row["source_tag"]) for row in reader]
        conn.executemany("INSERT INTO sources VALUES (?, ?, ?, ?)", rows)
        stats["sources"] = len(rows)

    # -- optional model tier probabilities --
    conn.execute("""
        CREATE TABLE slot_filler_model_scores (
            slot_filler_id INTEGER PRIMARY KEY,
            score_pop REAL NOT NULL DEFAULT 0.0,
            score_mainstream REAL NOT NULL DEFAULT 0.0,
            score_niche REAL NOT NULL DEFAULT 0.0,
            model_tier TEXT,
            source_prediction_count INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (slot_filler_id) REFERENCES slot_fillers(id)
        )
    """)
    model_scores_path = data_dir / "slot_filler_model_scores.csv"
    if model_scores_path.exists():
        with open(model_scores_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [
                (
                    int(row["slot_filler_id"]),
                    float(row["score_pop"] or 0.0),
                    float(row["score_mainstream"] or 0.0),
                    float(row["score_niche"] or 0.0),
                    row.get("model_tier") or None,
                    int(row.get("source_prediction_count") or 0),
                )
                for row in reader
            ]
            conn.executemany(
                "INSERT INTO slot_filler_model_scores VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )
            stats["slot_filler_model_scores"] = len(rows)
            if len(rows) != stats["slot_fillers"]:
                raise RuntimeError(
                    "slot_filler_model_scores.csv must cover every exported "
                    f"slot filler ({len(rows):,} scores for "
                    f"{stats['slot_fillers']:,} slot fillers)."
                )
    else:
        stats["slot_filler_model_scores"] = 0

    # -- optional tier-slot runtime distribution --
    shadow_distribution_path = data_dir / f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}.csv"
    if shadow_distribution_path.exists():
        conn.execute(f"""
            CREATE TABLE {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} (
                slot_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                filler TEXT NOT NULL,
                display_filler TEXT NOT NULL,
                probability REAL NOT NULL,
                log_probability REAL NOT NULL,
                soft_count REAL NOT NULL,
                prior_count REAL NOT NULL,
                evidence_count REAL NOT NULL,
                source_count INTEGER NOT NULL,
                anchored_source_count INTEGER NOT NULL,
                inferred_source_count INTEGER NOT NULL,
                anchored_soft_count REAL NOT NULL,
                inferred_soft_count REAL NOT NULL,
                teacher_confidence_mean REAL,
                frequency INTEGER NOT NULL,
                popularity_score REAL,
                semantic_smoothing_mass REAL NOT NULL,
                calibration_temperature REAL NOT NULL,
                artifact_version TEXT NOT NULL
            )
        """)
        with open(shadow_distribution_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                (
                    row["slot_type"],
                    row["tier"],
                    row["filler"],
                    row["display_filler"],
                    float(row["probability"]),
                    float(row["log_probability"]),
                    float(row["soft_count"]),
                    float(row["prior_count"]),
                    float(row["evidence_count"]),
                    int(row["source_count"]),
                    int(row["anchored_source_count"]),
                    int(row["inferred_source_count"]),
                    float(row["anchored_soft_count"]),
                    float(row["inferred_soft_count"]),
                    float(row["teacher_confidence_mean"]) if row.get("teacher_confidence_mean") else None,
                    int(row["frequency"]),
                    float(row["popularity_score"]) if row.get("popularity_score") else None,
                    float(row["semantic_smoothing_mass"]),
                    float(row["calibration_temperature"]),
                    row["artifact_version"],
                )
                for row in reader
            ]
            conn.executemany(
                f"INSERT INTO {TIER_SLOT_FILLER_DISTRIBUTION_TABLE} VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            stats[TIER_SLOT_FILLER_DISTRIBUTION_TABLE] = len(rows)
        issues = validate_tier_slot_distribution(conn)
        if issues:
            detail = "\n".join(issue.message for issue in issues)
            conn.close()
            raise RuntimeError(
                "tier_slot_filler_distribution_v1 validation failed:\n"
                f"{detail}"
            )
    else:
        stats[TIER_SLOT_FILLER_DISTRIBUTION_TABLE] = 0

    # -- indexes --
    conn.execute("CREATE INDEX idx_sf_slot_type ON slot_fillers(slot_type)")
    conn.execute("CREATE INDEX idx_sf_slot_type_pos ON slot_fillers(slot_type, pos_tag)")
    conn.execute("CREATE INDEX idx_sf_slot_type_prep ON slot_fillers(slot_type, prep)")
    conn.execute("CREATE INDEX idx_sf_filler ON slot_fillers(filler)")
    conn.execute("CREATE INDEX idx_sources_filler ON sources(slot_filler_id)")
    conn.execute(
        "CREATE INDEX idx_model_scores_tier ON slot_filler_model_scores(model_tier)"
    )
    if stats[TIER_SLOT_FILLER_DISTRIBUTION_TABLE]:
        conn.execute(
            f"CREATE INDEX idx_tier_slot_dist_group ON "
            f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}(slot_type, tier)"
        )
    return stats


# Keep backward compat for existing export-db CLI command
def export_mini_db(source_conn: sqlite3.Connection, output_path: Path) -> dict:
    """Create a minimal SQLite DB directly from the full DB (legacy one-step)."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        export_data(source_conn, tmp_dir)
        return build_mini_db(tmp_dir, output_path)
