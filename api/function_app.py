"""Azure Functions v2 (Python) app wrapping subtitle-generator."""

import json
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
)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)


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


@app.route(route="rate", methods=["POST"])
def rate(req: func.HttpRequest) -> func.HttpResponse:
    try:
        try:
            body = req.get_json()
        except ValueError:
            return _error("Request body must be valid JSON")

        status, resp = handle_rate(body)
        return _json_response(resp, status)

    except Exception as exc:
        return _error(f"Internal error: {exc}", 500)


# ── GET /api/health ─────────────────────────────────────────────────


@app.route(route="health", methods=["GET"])
def health(req: func.HttpRequest) -> func.HttpResponse:
    status, resp = handle_health()
    return _json_response(resp, status)
