"""Durable storage helpers for web rating feedback."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any


RATINGS_TABLE_NAME = "ratings"

logger = logging.getLogger(__name__)


class RatingTableWriteError(RuntimeError):
    """Raised when a required Azure Table Storage rating write fails."""


def build_rating_entity(
    subtitle: str,
    thumbs: int | None,
    tone_override: str | None,
    free_text: str | None,
    system_tone: str | None,
    tags: list[str] | None,
    *,
    source: str = "web_user",
    prompt_generated: bool = False,
    now: datetime | None = None,
    row_key_suffix: str | None = None,
) -> dict[str, Any]:
    """Build the Azure Table entity used for durable rating storage."""
    timestamp = now or datetime.now(timezone.utc)
    suffix = row_key_suffix or uuid.uuid4().hex[:8]
    entity: dict[str, Any] = {
        "PartitionKey": timestamp.strftime("%Y-%m"),
        "RowKey": f"{timestamp.isoformat()}-{suffix}",
        "subtitle": subtitle,
        "tone_override": tone_override or "",
        "free_text": free_text or "",
        "system_tone": system_tone or "",
        "tags": json.dumps(tags or []),
        "source": source,
        "prompt_generated": prompt_generated,
    }
    if thumbs is not None:
        entity["thumbs"] = thumbs
    return entity


def write_rating_to_table_storage(
    subtitle: str,
    thumbs: int | None,
    tone_override: str | None,
    free_text: str | None,
    system_tone: str | None,
    tags: list[str] | None,
    *,
    source: str = "web_user",
    prompt_generated: bool = False,
    account_name: str | None = None,
    table_name: str = RATINGS_TABLE_NAME,
    required: bool = False,
) -> dict[str, Any] | None:
    """Write a rating to Azure Table Storage.

    If ``required`` is false, missing configuration is treated as disabled and
    write errors are logged as best-effort failures. If ``required`` is true,
    missing configuration, missing dependencies, and Azure write errors raise
    ``RatingTableWriteError`` so production callers can fail the request.
    """
    storage_account = account_name or os.environ.get("STORAGE_ACCOUNT_NAME")
    if not storage_account:
        if required:
            raise RatingTableWriteError("STORAGE_ACCOUNT_NAME is not configured")
        return None

    entity = build_rating_entity(
        subtitle,
        thumbs,
        tone_override,
        free_text,
        system_tone,
        tags,
        source=source,
        prompt_generated=prompt_generated,
    )

    try:
        from azure.data.tables import TableServiceClient
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        message = "azure-data-tables and azure-identity are required for rating Table Storage writes"
        if required:
            raise RatingTableWriteError(message) from exc
        logger.warning(message, exc_info=True)
        return None

    try:
        credential = DefaultAzureCredential()
        service = TableServiceClient(
            endpoint=f"https://{storage_account}.table.core.windows.net",
            credential=credential,
        )
        table = service.create_table_if_not_exists(table_name)
        table.create_entity(entity)
        return entity
    except Exception as exc:
        message = f"Failed to write rating to Azure Table Storage table {table_name!r}"
        if required:
            raise RatingTableWriteError(message) from exc
        logger.warning(message, exc_info=True)
        return None
