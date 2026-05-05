"""Azure Functions v2 (Python) app wrapping subtitle-generator."""

# ruff: noqa: E402

import json
import logging
import os
import sys
from pathlib import Path

# Add the src directory so subtitle_generator is importable (local dev)
# In Azure, subtitle_generator/ is copied alongside function_app.py
_src_path = Path(__file__).parent.parent / "src"
if _src_path.is_dir():
    sys.path.insert(0, str(_src_path))

import azure.functions as func

from subtitle_generator.handlers import (
    handle_generate,
    handle_health,
    handle_jacket,
    handle_rate,
    validate_rating_body,
)
from subtitle_generator.rating_storage import (
    RatingTableWriteError,
    write_rating_to_table_storage,
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logger = logging.getLogger(__name__)


def _json_response(body: dict, status_code: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(body),
        status_code=status_code,
        mimetype="application/json",
    )


def _error(msg: str, status_code: int = 400) -> func.HttpResponse:
    return _json_response({"error": msg}, status_code)


# ── POST /api/generate ──────────────────────────────────────────────


@app.route(route="generate", methods=["POST"])
def generate(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            body = {}

        status, resp = handle_generate(body)
        return _json_response(resp, status)

    except Exception as exc:
        return _error(f"Internal error: {exc}", 500)


# ── POST /api/jacket ────────────────────────────────────────────────


@app.route(route="jacket", methods=["POST"])
def jacket(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            return _error("Request body must be valid JSON")

        status, resp = handle_jacket(body)
        return _json_response(resp, status)

    except Exception as exc:
        return _error(f"Internal error: {exc}", 500)


# ── POST /api/rate ──────────────────────────────────────────────────


def _handle_rate_azure(body: dict) -> tuple[int, dict]:
    status, resp, payload = validate_rating_body(body)
    if status != 200:
        return status, resp
    assert payload is not None

    entity = write_rating_to_table_storage(
        payload.subtitle,
        payload.thumbs,
        payload.tone_override,
        payload.free_text,
        payload.system_tone,
        payload.tags,
        source=payload.source,
        prompt_generated=payload.prompt_generated,
        required=True,
    )
    return 200, {"id": entity["RowKey"], "status": "saved"}


@app.route(route="rate", methods=["POST"])
def rate(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            return _error("Request body must be valid JSON")

        try:
            if os.environ.get("SUBTITLE_GEN_MODE") == "azure":
                status, resp = _handle_rate_azure(body)
            else:
                status, resp = handle_rate(body)
                if status == 200:
                    write_rating_to_table_storage(
                        body.get("subtitle", ""),
                        body.get("thumbs"),
                        body.get("tone_override"),
                        body.get("free_text"),
                        body.get("system_tone"),
                        body.get("tags"),
                        source=body.get("_source", "web_user"),
                        prompt_generated=body.get("prompt_generated", False),
                        required=False,
                    )
        except RatingTableWriteError:
            logger.exception("Failed to persist rating to Azure Table Storage")
            return _error("Failed to persist rating to durable storage", 500)
        return _json_response(resp, status)

    except Exception as exc:
        logger.exception("Unhandled rate request error")
        return _error(f"Internal error: {exc}", 500)


# ── GET /api/health ─────────────────────────────────────────────────


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    status, resp = handle_health()
    return _json_response(resp, status)
