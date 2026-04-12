"""Lightweight local HTTP server for the subtitle-generator web app.

Uses only Python stdlib (http.server + ThreadingHTTPServer).  Importable
without starting the server — call ``create_server()`` or ``run()``.
"""

import asyncio
import json
import os
import sqlite3
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from subtitle_generator.generate import (
    TONE_TARGETS,
    GeneratedSubtitle,
    find_source,
    generate_subtitle,
)
from subtitle_generator.jacket import (
    TONE_HIGH,
    TONE_LOW,
    TONE_MEDIUM,
    build_jacket_prompt,
    generate_jacket,
)

# ---------------------------------------------------------------------------
# Shared helpers (mirrored from function_app.py, no azure.functions needed)
# ---------------------------------------------------------------------------

_TONE_CHOICES = {"pop": TONE_HIGH, "mainstream": TONE_MEDIUM, "niche": TONE_LOW}
_VALID_TONES = set(_TONE_CHOICES.keys())
_VALID_LOCK_KEYS = {
    "item1", "item2", "action_noun", "of_object",
    "of_modifier", "of_head", "of_topic", "of_complement",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _PROJECT_ROOT / "web"


def _get_db() -> sqlite3.Connection:
    db_path = os.environ.get(
        "DB_PATH",
        str(_PROJECT_ROOT / "data" / "db" / "subtitles.db"),
    )
    return sqlite3.connect(db_path)


def _parse_tone(tone_str: str | None) -> set[str] | None:
    if not tone_str:
        return None
    tones = {t.strip().lower() for t in tone_str.split(",")}
    invalid = tones - _VALID_TONES
    if invalid:
        raise ValueError(
            f"Invalid tone(s): {', '.join(invalid)}. Choose from: pop, mainstream, niche"
        )
    return tones


def _build_sources(conn: sqlite3.Connection, sub: GeneratedSubtitle) -> dict:
    fillers: list[tuple[str, str, str]] = [
        ("item1", sub.item1, "list_item"),
        ("item2", sub.item2, "list_item"),
        ("action_noun", sub.action_noun, "action_noun"),
    ]
    if sub.remixed and sub.remix_parts:
        if "modifier" in sub.remix_parts:
            fillers.append(("of_modifier", sub.remix_parts["modifier"], "of_modifier"))
            fillers.append(("of_head", sub.remix_parts["head"], "of_head"))
        elif "topic" in sub.remix_parts:
            fillers.append(("of_topic", sub.remix_parts["topic"], "of_topic"))
            fillers.append(("of_complement", sub.remix_parts["complement"], "of_complement"))
    else:
        fillers.append(("of_object", sub.of_object, "of_object"))

    sources: dict[str, dict] = {}
    for key, filler, slot_type in fillers:
        result = find_source(conn, filler, slot_type)
        if result:
            desc, tag = result
            sources[key] = {"title": desc, "tag": tag}
        else:
            sources[key] = {"title": None, "tag": None}
    return sources


# ---------------------------------------------------------------------------
# API handlers — return (status_code, body_dict)
# ---------------------------------------------------------------------------

def _handle_health() -> tuple[int, dict]:
    mode = os.environ.get("SUBTITLE_GEN_MODE", "local")
    return 200, {"ok": True, "mode": mode}


def _handle_generate(body: dict) -> tuple[int, dict]:
    tone_str = body.get("tone")
    remix_prob = body.get("remix_prob")
    min_sim = body.get("min_sim")
    locks = body.get("locks")

    try:
        tone_set = _parse_tone(tone_str)
    except ValueError as exc:
        return 400, {"error": str(exc)}

    if locks is not None:
        if not isinstance(locks, dict):
            return 400, {"error": "locks must be an object mapping slot keys to values"}
        invalid_keys = set(locks.keys()) - _VALID_LOCK_KEYS
        if invalid_keys:
            return 400, {"error": f"Invalid lock keys: {', '.join(invalid_keys)}"}

    conn = _get_db()
    try:
        if remix_prob is None:
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'remix_calibrated_remix_prob'"
            ).fetchone()
            remix_prob = float(row[0]) if row else 0.8
        if min_sim is None:
            row = conn.execute(
                "SELECT value FROM config WHERE key = 'remix_calibrated_min_sim'"
            ).fetchone()
            min_sim = float(row[0]) if row else 0.1

        tone_target = None
        if tone_set:
            merged: dict[str, float] = {}
            for slot in ("list_item", "action_noun", "of_object"):
                merged[slot] = sum(TONE_TARGETS[t][slot] for t in tone_set) / len(tone_set)
            tone_target = merged

        sub = generate_subtitle(
            conn,
            tone_target=tone_target,
            remix_prob=remix_prob,
            min_sim=min_sim,
            locks=locks,
        )
        sources = _build_sources(conn, sub)

        return 200, {
            "text": sub.text,
            "item1": sub.item1,
            "item2": sub.item2,
            "action_noun": sub.action_noun,
            "of_object": sub.of_object,
            "remixed": sub.remixed,
            "remix_parts": sub.remix_parts,
            "remix_similarity": sub.remix_similarity,
            "sources": sources,
        }
    finally:
        conn.close()


def _handle_jacket(body: dict) -> tuple[int, dict]:
    subtitle = body.get("subtitle")
    if not subtitle or not isinstance(subtitle, str):
        return 400, {"error": "subtitle is required and must be a non-empty string"}

    model = body.get("model", "gpt-5.4-mini")
    dry_run = bool(body.get("dry_run", True))

    conn = _get_db()
    try:
        system_prompt, user_prompt, tone_tier = build_jacket_prompt(subtitle, conn=conn)
        prompt_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

        result_text = None
        if not dry_run:
            result_text = generate_jacket(
                subtitle,
                model=model,
                conn=conn,
            )

        return 200, {
            "prompt": prompt_text,
            "tone_tier": tone_tier,
            "result": result_text,
        }
    finally:
        conn.close()


# -- /api/models (local only) ------------------------------------------------

_models_cache: list[dict] | None = None


def _handle_models() -> tuple[int, dict]:
    """List available Copilot SDK models with pretty names and cost info."""
    global _models_cache
    if _models_cache is not None:
        return 200, {"models": _models_cache}

    try:
        from copilot import CopilotClient

        async def _fetch():
            async with CopilotClient() as client:
                return await client.list_models()

        raw = asyncio.run(_fetch())
        _models_cache = [
            {
                "id": m.id,
                "name": m.name,
                "cost": m.billing.multiplier if m.billing else 1.0,
            }
            for m in raw
            if m.policy and m.policy.state == "enabled"
        ]
        return 200, {"models": _models_cache}
    except Exception as exc:
        return 500, {"error": f"Failed to list models: {exc}"}


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------

_MIME_TYPES: dict[str, str] = {
    ".html": "text/html",
    ".css": "text/css",
    ".js": "application/javascript",
    ".json": "application/json",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
}


class _Handler(BaseHTTPRequestHandler):
    """Handles API routes and static-file serving."""

    web_dir: Path  # set via partial() in create_server

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Suppress default stderr logging; click.echo is used instead.
        pass

    # -- helpers ----------------------------------------------------------

    def _send_json(self, body: dict, status: int = 200) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw)

    # -- verbs ------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]  # strip query string

        # API routes
        if path == "/api/health":
            status, body = _handle_health()
            self._send_json(body, status)
            return
        if path == "/api/models":
            status, body = _handle_models()
            self._send_json(body, status)
            return

        # Static files
        self._serve_static(path)

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?")[0]

        try:
            body = self._read_body()
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON body"}, 400)
            return

        if path == "/api/generate":
            try:
                status, resp = _handle_generate(body)
            except Exception as exc:
                status, resp = 500, {"error": f"Internal error: {exc}"}
            self._send_json(resp, status)
        elif path == "/api/jacket":
            dry_run = bool(body.get("dry_run", True))
            if not dry_run:
                self._handle_jacket_stream(body)
            else:
                try:
                    status, resp = _handle_jacket(body)
                except Exception as exc:
                    status, resp = 500, {"error": f"Internal error: {exc}"}
                self._send_json(resp, status)
        else:
            self._send_json({"error": "Not found"}, 404)

    def _handle_jacket_stream(self, body: dict) -> None:
        """Stream jacket generation progress as SSE, then send final result."""
        subtitle = body.get("subtitle")
        if not subtitle or not isinstance(subtitle, str):
            self._send_json({"error": "subtitle is required"}, 400)
            return

        model = body.get("model", "gpt-5.4-mini")

        # Start SSE response
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()

        def send_event(event: str, data: str) -> None:
            # SSE spec: multi-line data needs each line prefixed with "data: "
            lines = data.split("\n")
            chunk = f"event: {event}\n"
            for ln in lines:
                chunk += f"data: {ln}\n"
            chunk += "\n"
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()

        def on_progress(msg: str) -> None:
            send_event("progress", msg)

        conn = _get_db()
        try:
            system_prompt, user_prompt, tone_tier = build_jacket_prompt(subtitle, conn=conn)
            prompt_text = f"{system_prompt}\n\n---\n\n{user_prompt}"

            result_text = generate_jacket(
                subtitle, model=model, conn=conn,
                on_progress=on_progress,
            )

            final = json.dumps({
                "prompt": prompt_text,
                "tone_tier": tone_tier,
                "result": result_text,
            }, ensure_ascii=False)
            send_event("result", final)
        except Exception as exc:
            send_event("error", str(exc))
        finally:
            conn.close()

    # -- static files -----------------------------------------------------

    def _serve_static(self, url_path: str) -> None:
        if url_path == "/":
            url_path = "/index.html"

        rel = url_path.lstrip("/")
        file_path = (self.web_dir / rel).resolve()

        # Security: ensure we stay inside web_dir
        try:
            file_path.relative_to(self.web_dir.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return

        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        mime = _MIME_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_server(
    port: int = 8742,
    web_dir: Path | None = None,
) -> ThreadingHTTPServer:
    """Create (but don't start) the local dev server."""
    if web_dir is None:
        web_dir = _WEB_DIR

    handler = type(
        "_BoundHandler",
        (_Handler,),
        {"web_dir": web_dir},
    )

    server = ThreadingHTTPServer(("", port), handler)
    return server


def run(port: int = 8742, web_dir: Path | None = None) -> None:
    """Create and run the server (blocks forever)."""
    server = create_server(port=port, web_dir=web_dir)
    server.serve_forever()
