"""Explicit shadow runtime support for tier-slot distribution artifacts."""

from __future__ import annotations

import csv
import hashlib
import random
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from subtitle_generator.schema_contracts import (
    TIER_SLOT_FILLER_DISTRIBUTION_TABLE,
    validate_tier_slot_distribution,
)
from subtitle_generator.tier_slot_calibration import apply_temperature
from subtitle_generator.tier_slot_distribution import (
    DISTRIBUTION_COLUMNS,
    _filler_key,
    _format_csv_value,
    _validate_rows,
)

GroupKey = tuple[str, str]  # (slot_type, tier)


class RuntimeSelectionMode(StrEnum):
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
    mode: RuntimeSelectionMode = RuntimeSelectionMode.LEGACY
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

    parsed_mode = RuntimeSelectionMode(mode or RuntimeSelectionMode.LEGACY)
    policy = ShadowSamplingPolicy(float(shadow_sampling_temperature))
    policy.validate()
    if parsed_mode == RuntimeSelectionMode.LEGACY:
        if shadow_artifact is not None or shadow_sampling_temperature != 1.0:
            raise ValueError(
                "shadow_artifact and shadow_sampling_temperature require "
                "runtime mode 'shadow'"
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
    if selection.mode == RuntimeSelectionMode.LEGACY:
        return PreparedGenerationRuntime(selection=selection)
    if selection.artifact_source is None:
        raise RuntimeError("Shadow runtime requires an explicit artifact source")
    source = load_shadow_distribution_source(conn, selection.artifact_source)
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
    return PreparedGenerationRuntime(
        selection=selection,
        shadow_distribution=LoadedShadowDistribution(source=source, groups=groups),
    )


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
        _validate_rows(conn, list(rows))
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

    if runtime.mode != RuntimeSelectionMode.SHADOW:
        raise RuntimeError("sample_shadow_candidates requires shadow runtime mode")
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
        apply_temperature(weights, runtime.selection.sampling_policy.sampling_temperature)
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
        return {"mode": RuntimeSelectionMode.LEGACY.value}
    if runtime.shadow_distribution is None:
        raise RuntimeError("Shadow runtime provenance requires a loaded distribution")
    return {
        "mode": runtime.mode.value,
        "artifact_source": runtime.shadow_distribution.source.source_description,
        "artifact_digest": runtime.shadow_distribution.source.digest,
        "artifact_version": runtime.shadow_distribution.source.artifact_version,
        "sampling_temperature": runtime.selection.sampling_policy.sampling_temperature,
    }


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
