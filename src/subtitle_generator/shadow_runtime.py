"""Explicit shadow runtime support for tier-slot distribution artifacts."""

from __future__ import annotations

import csv
import hashlib
import math
import random
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitle_generator.schema_contracts import (
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    validate_tier_slot_distribution,
)
from subtitle_generator.runtime_eligibility import filler_key as _filler_key

GroupKey = tuple[str, str]  # (slot_type, tier)
DISTRIBUTION_COLUMNS = (
    "slot_type",
    "tier",
    "filler",
    "display_filler",
    "probability",
    "log_probability",
    "soft_count",
    "prior_count",
    "evidence_count",
    "source_count",
    "anchored_source_count",
    "inferred_source_count",
    "anchored_soft_count",
    "inferred_soft_count",
    "teacher_confidence_mean",
    "frequency",
    "popularity_score",
    "semantic_smoothing_mass",
    "calibration_temperature",
    "artifact_version",
)
_REQUIRED_NUMERIC_COLUMNS = (
    "probability",
    "log_probability",
    "soft_count",
    "prior_count",
    "evidence_count",
    "source_count",
    "anchored_source_count",
    "inferred_source_count",
    "anchored_soft_count",
    "inferred_soft_count",
    "frequency",
    "semantic_smoothing_mass",
    "calibration_temperature",
)
_OPTIONAL_NUMERIC_COLUMNS = (
    "teacher_confidence_mean",
    "popularity_score",
)
_DISTRIBUTION_CACHE_LIMIT = 4
_distribution_cache: OrderedDict[tuple[object, ...], LoadedShadowDistribution] = (
    OrderedDict()
)
_distribution_cache_lock = threading.Lock()


class RuntimeSelectionMode(StrEnum):
    CONFIGURED = "configured"
    ARTIFACT = "artifact"
    LEGACY = "legacy"
    SHADOW = "shadow"


class ShadowArtifactSourceKind(StrEnum):
    DB_TABLE = "db_table"
    CSV_PATH = "csv_path"


@dataclass(frozen=True)
class ShadowSamplingPolicy:
    sampling_temperature: float = 1.0

    def validate(self) -> None:
        if self.sampling_temperature <= 0:
            raise RuntimeError("shadow sampling_temperature must be positive")


@dataclass(frozen=True)
class ShadowArtifactSource:
    kind: ShadowArtifactSourceKind
    table: str = TIER_SLOT_FILLER_DISTRIBUTION_TABLE
    path: Path | None = None

    @classmethod
    def db_table(
        cls,
        table: str = TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    ) -> "ShadowArtifactSource":
        return cls(kind=ShadowArtifactSourceKind.DB_TABLE, table=table)

    @classmethod
    def csv_path(cls, path: Path) -> "ShadowArtifactSource":
        return cls(kind=ShadowArtifactSourceKind.CSV_PATH, path=path.resolve())

    def describe(self) -> str:
        if self.kind == ShadowArtifactSourceKind.DB_TABLE:
            return f"db-table:{self.table}"
        if self.path is None:
            raise RuntimeError("CSV shadow artifact source is missing its path")
        return str(self.path)


@dataclass(frozen=True)
class GenerationRuntimeSelection:
    mode: RuntimeSelectionMode = RuntimeSelectionMode.CONFIGURED
    artifact_source: ShadowArtifactSource | None = None
    sampling_policy: ShadowSamplingPolicy = ShadowSamplingPolicy()


@dataclass(frozen=True)
class ShadowDistributionSourceData:
    rows: tuple[dict[str, object], ...]
    source_description: str
    digest: str
    artifact_version: str


@dataclass(frozen=True)
class _LoadedDistributionRow:
    filler_key: str
    display_filler: str
    probability: float
    calibration_temperature: float


@dataclass(frozen=True)
class LoadedShadowDistribution:
    source: ShadowDistributionSourceData
    groups: dict[GroupKey, dict[str, _LoadedDistributionRow]]


@dataclass(frozen=True)
class PreparedGenerationRuntime:
    selection: GenerationRuntimeSelection
    shadow_distribution: LoadedShadowDistribution | None = None
    fallback_reason: str | None = None

    @property
    def mode(self) -> RuntimeSelectionMode:
        return self.selection.mode


def build_generation_runtime(
    *,
    mode: str | RuntimeSelectionMode | None = None,
    shadow_artifact: str | Path | None = None,
    shadow_sampling_temperature: float = 1.0,
) -> GenerationRuntimeSelection:
    """Build an explicit runtime selection object."""

    parsed_mode = RuntimeSelectionMode(mode or RuntimeSelectionMode.CONFIGURED)
    policy = ShadowSamplingPolicy(float(shadow_sampling_temperature))
    policy.validate()
    if parsed_mode in {
        RuntimeSelectionMode.CONFIGURED,
        RuntimeSelectionMode.LEGACY,
    }:
        if shadow_artifact is not None or shadow_sampling_temperature != 1.0:
            raise ValueError(
                "shadow_artifact and shadow_sampling_temperature require "
                "runtime mode 'artifact' or 'shadow'"
            )
        return GenerationRuntimeSelection(mode=parsed_mode, sampling_policy=policy)
    artifact_source = (
        ShadowArtifactSource.csv_path(Path(shadow_artifact))
        if shadow_artifact is not None
        else ShadowArtifactSource.db_table()
    )
    return GenerationRuntimeSelection(
        mode=parsed_mode,
        artifact_source=artifact_source,
        sampling_policy=policy,
    )


def prepare_generation_runtime(
    conn: sqlite3.Connection,
    runtime: GenerationRuntimeSelection | PreparedGenerationRuntime | None,
) -> PreparedGenerationRuntime:
    """Resolve and validate the explicit runtime selection for generation."""

    if isinstance(runtime, PreparedGenerationRuntime):
        return runtime
    selection = runtime or GenerationRuntimeSelection()
    selection.sampling_policy.validate()
    configured_request = selection.mode == RuntimeSelectionMode.CONFIGURED
    if configured_request:
        configured_mode = _configured_runtime_mode(conn)
        if configured_mode == RuntimeSelectionMode.LEGACY:
            return PreparedGenerationRuntime(
                selection=GenerationRuntimeSelection(
                    mode=RuntimeSelectionMode.LEGACY,
                ),
                fallback_reason=(
                    "config generation_runtime_mode is absent or set to legacy"
                ),
            )
        if not _table_exists(conn, TIER_SLOT_FILLER_DISTRIBUTION_TABLE):
            return PreparedGenerationRuntime(
                selection=GenerationRuntimeSelection(
                    mode=RuntimeSelectionMode.LEGACY,
                ),
                fallback_reason=(
                    "configured artifact table tier_slot_filler_distribution_v1 "
                    "is missing"
                ),
            )
        selection = GenerationRuntimeSelection(
            mode=configured_mode,
            artifact_source=ShadowArtifactSource.db_table(),
            sampling_policy=selection.sampling_policy,
        )
    if selection.mode == RuntimeSelectionMode.LEGACY:
        return PreparedGenerationRuntime(selection=selection)
    if selection.artifact_source is None:
        raise RuntimeError("Shadow runtime requires an explicit artifact source")
    loaded_distribution = _load_prepared_distribution(
        conn,
        selection.artifact_source,
    )
    return PreparedGenerationRuntime(
        selection=selection,
        shadow_distribution=loaded_distribution,
    )


def _load_prepared_distribution(
    conn: sqlite3.Connection,
    artifact_source: ShadowArtifactSource,
) -> LoadedShadowDistribution:
    cache_key = _distribution_cache_key(conn, artifact_source)
    if cache_key is not None:
        with _distribution_cache_lock:
            cached = _distribution_cache.get(cache_key)
            if cached is not None:
                _distribution_cache.move_to_end(cache_key)
                return cached
    source = load_shadow_distribution_source(conn, artifact_source)
    groups: dict[GroupKey, dict[str, _LoadedDistributionRow]] = {}
    for row in source.rows:
        slot_type = str(row["slot_type"])
        tier = str(row["tier"])
        display_filler = str(row["display_filler"])
        key = _filler_key(slot_type, display_filler)
        groups.setdefault((slot_type, tier), {})[key] = _LoadedDistributionRow(
            filler_key=key,
            display_filler=display_filler,
            probability=float(row["probability"]),
            calibration_temperature=float(row["calibration_temperature"]),
        )
    loaded = LoadedShadowDistribution(source=source, groups=groups)
    if cache_key is not None:
        with _distribution_cache_lock:
            _distribution_cache[cache_key] = loaded
            _distribution_cache.move_to_end(cache_key)
            while len(_distribution_cache) > _DISTRIBUTION_CACHE_LIMIT:
                _distribution_cache.popitem(last=False)
    return loaded


def load_shadow_distribution_source(
    conn: sqlite3.Connection,
    source: ShadowArtifactSource,
) -> ShadowDistributionSourceData:
    """Load a v1 tier-slot distribution artifact from a DB table or CSV path."""

    if source.kind == ShadowArtifactSourceKind.DB_TABLE:
        issues = validate_tier_slot_distribution(conn, table=source.table)
        if issues:
            raise RuntimeError("\n".join(issue.message for issue in issues))
        rows = tuple(_read_distribution_table(conn, source.table))
    else:
        if source.path is None:
            raise RuntimeError("CSV shadow artifact source is missing its path")
        if not source.path.is_file():
            raise RuntimeError(
                f"Shadow runtime artifact does not exist: {source.path}"
            )
        rows = tuple(_read_distribution_csv(source.path))
        _validate_finite_numeric_rows(rows, source.describe())
        _validate_csv_rows(conn, rows)
    if source.kind == ShadowArtifactSourceKind.DB_TABLE:
        _validate_finite_numeric_rows(rows, source.describe())
    if not rows:
        raise RuntimeError(
            f"Shadow runtime artifact {source.describe()} contains no rows"
        )
    artifact_versions = {str(row["artifact_version"]) for row in rows}
    if len(artifact_versions) != 1:
        raise RuntimeError(
            "Shadow runtime artifact must use exactly one artifact_version; got "
            f"{sorted(artifact_versions)}"
        )
    return ShadowDistributionSourceData(
        rows=rows,
        source_description=source.describe(),
        digest=_rows_digest(rows),
        artifact_version=next(iter(artifact_versions)),
    )


def write_shadow_distribution_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write a normalized v1 shadow distribution CSV."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DISTRIBUTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                column: _format_csv_value(row.get(column))
                for column in DISTRIBUTION_COLUMNS
            })


def install_tier_slot_distribution(
    conn: sqlite3.Connection,
    artifact_path: Path,
    *,
    activate: bool = True,
) -> int:
    """Atomically replace the DB runtime artifact from a validated CSV."""

    loaded = load_shadow_distribution_source(
        conn,
        ShadowArtifactSource.csv_path(artifact_path),
    )
    staging_table = f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}__install"
    conn.execute("SAVEPOINT install_tier_slot_distribution")
    try:
        conn.execute(f"DROP TABLE IF EXISTS {staging_table}")
        _create_distribution_table(conn, staging_table)
        _insert_distribution_rows(conn, staging_table, loaded.rows)
        issues = validate_tier_slot_distribution(conn, table=staging_table)
        if issues:
            raise RuntimeError("\n".join(issue.message for issue in issues))
        conn.execute(f"DROP TABLE IF EXISTS {TIER_SLOT_FILLER_DISTRIBUTION_TABLE}")
        conn.execute(
            f"ALTER TABLE {staging_table} RENAME TO "
            f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}"
        )
        conn.execute(
            "CREATE INDEX idx_tier_slot_dist_group ON "
            f"{TIER_SLOT_FILLER_DISTRIBUTION_TABLE}(slot_type, tier)"
        )
        if activate:
            _write_configured_runtime_mode(conn, RuntimeSelectionMode.ARTIFACT)
        conn.execute("RELEASE SAVEPOINT install_tier_slot_distribution")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT install_tier_slot_distribution")
        conn.execute("RELEASE SAVEPOINT install_tier_slot_distribution")
        raise
    clear_distribution_cache()
    return len(loaded.rows)


def set_configured_runtime_mode(
    conn: sqlite3.Connection,
    mode: str | RuntimeSelectionMode,
) -> RuntimeSelectionMode:
    parsed_mode = RuntimeSelectionMode(mode)
    if parsed_mode not in {
        RuntimeSelectionMode.ARTIFACT,
        RuntimeSelectionMode.LEGACY,
    }:
        raise ValueError("configured runtime mode must be 'artifact' or 'legacy'")
    if parsed_mode == RuntimeSelectionMode.ARTIFACT:
        issues = validate_tier_slot_distribution(conn)
        if issues:
            raise RuntimeError("\n".join(issue.message for issue in issues))
    with conn:
        _write_configured_runtime_mode(conn, parsed_mode)
    return parsed_mode


def sample_shadow_candidates(
    runtime: PreparedGenerationRuntime,
    *,
    slot_type: str,
    tier: str,
    candidate_rows: list[tuple],
    count: int,
    rng: random.Random | None,
) -> list[str]:
    """Sample strict fillers from the loaded shadow distribution."""

    if not uses_tier_slot_distribution(runtime.mode):
        raise RuntimeError(
            "sample_shadow_candidates requires artifact or shadow runtime mode"
        )
    if runtime.shadow_distribution is None:
        raise RuntimeError("Shadow runtime is missing a loaded distribution")
    group = runtime.shadow_distribution.groups.get((slot_type, tier))
    if not group:
        raise RuntimeError(
            "Shadow runtime artifact "
            f"{runtime.shadow_distribution.source.source_description} "
            f"is missing the ({tier}, {slot_type}) distribution"
        )

    unique_candidates: dict[str, str] = {}
    for row in candidate_rows:
        display_filler = str(row[0])
        key = _filler_key(slot_type, display_filler)
        entry = group.get(key)
        if entry is None:
            raise RuntimeError(
                "Shadow runtime artifact "
                f"{runtime.shadow_distribution.source.source_description} "
                f"does not cover strict filler {display_filler!r} "
                f"for slot_type {slot_type!r}"
            )
        unique_candidates.setdefault(key, entry.display_filler)
    if len(unique_candidates) < count:
        raise RuntimeError(
            f"Shadow runtime needs {count} unique {slot_type!r} fillers but only "
            f"{len(unique_candidates)} matched the artifact subset"
        )

    weights = {
        key: group[key].probability
        for key in unique_candidates
    }
    scaled = (
        _apply_temperature(
            weights,
            runtime.selection.sampling_policy.sampling_temperature,
        )
        if runtime.selection.sampling_policy.sampling_temperature != 1.0
        else weights
    )
    keys = list(unique_candidates)
    key_weights = [scaled[key] for key in keys]
    chosen: list[str] = []
    for _ in range(count):
        pick = (rng or random).choices(keys, weights=key_weights, k=1)[0]
        index = keys.index(pick)
        chosen.append(unique_candidates[pick])
        keys.pop(index)
        key_weights.pop(index)
    return chosen


def shadow_runtime_provenance(runtime: PreparedGenerationRuntime) -> dict[str, object]:
    """Return stable provenance for a prepared shadow runtime."""

    if runtime.mode == RuntimeSelectionMode.LEGACY:
        result: dict[str, object] = {"mode": RuntimeSelectionMode.LEGACY.value}
        if runtime.fallback_reason:
            result["fallback_reason"] = runtime.fallback_reason
        return result
    if runtime.shadow_distribution is None:
        raise RuntimeError("Shadow runtime provenance requires a loaded distribution")
    return {
        "mode": runtime.mode.value,
        "artifact_source": runtime.shadow_distribution.source.source_description,
        "artifact_digest": runtime.shadow_distribution.source.digest,
        "artifact_version": runtime.shadow_distribution.source.artifact_version,
        "sampling_temperature": runtime.selection.sampling_policy.sampling_temperature,
    }


def uses_tier_slot_distribution(mode: RuntimeSelectionMode) -> bool:
    return mode in {
        RuntimeSelectionMode.ARTIFACT,
        RuntimeSelectionMode.SHADOW,
    }


def _configured_runtime_mode(conn: sqlite3.Connection) -> RuntimeSelectionMode:
    if not _table_exists(conn, "config"):
        return RuntimeSelectionMode.LEGACY
    row = conn.execute(
        "SELECT value FROM config WHERE key = 'generation_runtime_mode'"
    ).fetchone()
    if row is None:
        return RuntimeSelectionMode.LEGACY
    try:
        mode = RuntimeSelectionMode(str(row[0]))
    except ValueError as exc:
        raise RuntimeError(
            "config generation_runtime_mode must be 'artifact' or 'legacy'"
        ) from exc
    if mode not in {
        RuntimeSelectionMode.ARTIFACT,
        RuntimeSelectionMode.LEGACY,
    }:
        raise RuntimeError(
            "config generation_runtime_mode must be 'artifact' or 'legacy'"
        )
    return mode


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def clear_distribution_cache() -> None:
    with _distribution_cache_lock:
        _distribution_cache.clear()


def _distribution_cache_key(
    conn: sqlite3.Connection,
    source: ShadowArtifactSource,
) -> tuple[object, ...] | None:
    if source.kind == ShadowArtifactSourceKind.CSV_PATH:
        if source.path is None:
            return None
        signature = _path_signature(source.path, allow_missing=True)
        if signature is None:
            return None
        return ("csv", str(source.path), signature)
    database_path = _main_database_path(conn)
    if database_path is None:
        return None
    return (
        "db",
        str(database_path),
        source.table,
        _path_signature(database_path),
        _path_signature(Path(f"{database_path}-wal"), allow_missing=True),
    )


def _main_database_path(conn: sqlite3.Connection) -> Path | None:
    row = next(
        (
            database_row
            for database_row in conn.execute("PRAGMA database_list").fetchall()
            if database_row[1] == "main"
        ),
        None,
    )
    if row is None or not row[2]:
        return None
    return Path(str(row[2])).resolve()


def _path_signature(
    path: Path,
    *,
    allow_missing: bool = False,
) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        if allow_missing:
            return None
        raise
    return (stat.st_mtime_ns, stat.st_size, stat.st_ino)


def _validate_finite_numeric_rows(
    rows: tuple[dict[str, object], ...],
    source_description: str,
) -> None:
    for row_number, row in enumerate(rows, start=1):
        for column in _REQUIRED_NUMERIC_COLUMNS:
            _require_finite_number(
                row.get(column),
                source_description,
                row_number,
                column,
            )
        for column in _OPTIONAL_NUMERIC_COLUMNS:
            value = row.get(column)
            if value not in {None, ""}:
                _require_finite_number(
                    value,
                    source_description,
                    row_number,
                    column,
                )


def _require_finite_number(
    value: object,
    source_description: str,
    row_number: int,
    column: str,
) -> None:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Runtime artifact {source_description} row {row_number} field "
            f"{column!r} must be numeric"
        ) from exc
    if not math.isfinite(numeric_value):
        raise RuntimeError(
            f"Runtime artifact {source_description} row {row_number} field "
            f"{column!r} must be finite"
        )


def _write_configured_runtime_mode(
    conn: sqlite3.Connection,
    mode: RuntimeSelectionMode,
) -> None:
    if not _table_exists(conn, "config"):
        raise RuntimeError("configured runtime requires a config table")
    conn.execute(
        """
        INSERT INTO config (key, value)
        VALUES ('generation_runtime_mode', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (mode.value,),
    )


def _create_distribution_table(conn: sqlite3.Connection, table: str) -> None:
    conn.execute(f"""
        CREATE TABLE {table} (
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


def _insert_distribution_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: tuple[dict[str, object], ...],
) -> None:
    conn.executemany(
        f"INSERT INTO {table} VALUES "
        "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
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
                (
                    float(row["teacher_confidence_mean"])
                    if row.get("teacher_confidence_mean") not in {None, ""}
                    else None
                ),
                int(row["frequency"]),
                (
                    float(row["popularity_score"])
                    if row.get("popularity_score") not in {None, ""}
                    else None
                ),
                float(row["semantic_smoothing_mass"]),
                float(row["calibration_temperature"]),
                row["artifact_version"],
            )
            for row in rows
        ],
    )


def _read_distribution_table(
    conn: sqlite3.Connection,
    table: str,
) -> list[dict[str, object]]:
    rows = conn.execute(
        f"""
        SELECT {", ".join(DISTRIBUTION_COLUMNS)}
        FROM {table}
        ORDER BY slot_type, tier, filler
        """
    ).fetchall()
    return [
        dict(zip(DISTRIBUTION_COLUMNS, row, strict=True))
        for row in rows
    ]


def _read_distribution_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _rows_digest(rows: tuple[dict[str, object], ...]) -> str:
    hasher = hashlib.sha256()
    for row in rows:
        for column in DISTRIBUTION_COLUMNS:
            hasher.update(f"{column}=".encode("utf-8"))
            hasher.update(str(_format_csv_value(row[column])).encode("utf-8"))
            hasher.update(b"\n")
        hasher.update(b"--\n")
    return hasher.hexdigest()[:16]


def _apply_temperature(
    probabilities: dict[str, float],
    temperature: float,
) -> dict[str, float]:
    if temperature <= 0:
        raise RuntimeError("temperature must be positive")
    if temperature == 1.0:
        return dict(probabilities)
    exponent = 1.0 / temperature
    powered = {
        filler: (probability**exponent if probability > 0 else 0.0)
        for filler, probability in probabilities.items()
    }
    total = sum(powered.values())
    if total <= 0:
        return dict(probabilities)
    return {
        filler: value / total
        for filler, value in powered.items()
    }


def _format_csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value):
            return "-inf" if value < 0 else "inf"
        return f"{value:.12g}"
    return value


def _validate_csv_rows(
    conn: sqlite3.Connection,
    rows: tuple[dict[str, object], ...],
) -> None:
    validation = sqlite3.connect(":memory:")
    try:
        validation.execute(
            """
            CREATE TABLE slot_fillers (
                id INTEGER PRIMARY KEY,
                slot_type TEXT NOT NULL,
                filler TEXT NOT NULL,
                mode TEXT NOT NULL DEFAULT 'strict'
            )
            """
        )
        validation.executemany(
            "INSERT INTO slot_fillers VALUES (?, ?, ?, 'strict')",
            conn.execute(
                """
                SELECT id, slot_type, filler
                FROM slot_fillers
                WHERE mode = 'strict'
                """
            ).fetchall(),
        )
        _create_distribution_table(
            validation,
            TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
        )
        _insert_distribution_rows(
            validation,
            TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
            rows,
        )
        issues = validate_tier_slot_distribution(validation)
        if issues:
            raise RuntimeError("\n".join(issue.message for issue in issues))
    finally:
        validation.close()
