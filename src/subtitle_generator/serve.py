"""Lightweight local HTTP server for the subtitle-generator web app.

Uses only Python stdlib (http.server + ThreadingHTTPServer).  Importable
without starting the server — call ``create_server()`` or ``run()``.
"""

import asyncio
import json
import os
import random
import sqlite3
import uuid
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from subtitle_generator.generate import GeneratedSubtitle, generate_subtitle
from subtitle_generator.handlers import (
    get_db as _get_db,
    build_sources as _build_sources,
    handle_generate as _handle_generate,
    handle_health as _handle_health,
    handle_jacket as _handle_jacket,
    handle_rate as _handle_rate,
)
from subtitle_generator.jacket import build_jacket_prompt, generate_jacket

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_WEB_DIR = _PROJECT_ROOT / "web"


# ---------------------------------------------------------------------------
# Local-only rate wrapper (adds Azure Table Storage dual-write)
# ---------------------------------------------------------------------------

def _handle_rate_local(body: dict) -> tuple[int, dict]:
    """Wrap shared handle_rate with Azure Table Storage dual-write."""
    status, resp = _handle_rate(body)
    if status == 200:
        _write_to_table_storage(
            body.get("subtitle", ""),
            body.get("thumbs"),
            body.get("tone_override"),
            body.get("free_text"),
            body.get("system_tone"),
            body.get("tags"),
        )
    return status, resp


def _write_to_table_storage(
    subtitle: str, thumbs: int | None, tone_override: str | None,
    free_text: str | None, system_tone: str | None, tags: list[str] | None,
) -> None:
    """Write rating to Azure Table Storage if STORAGE_ACCOUNT_NAME is set."""
    account_name = os.environ.get("STORAGE_ACCOUNT_NAME")
    if not account_name:
        return

    try:
        from azure.data.tables import TableServiceClient
        from azure.identity import DefaultAzureCredential
        from datetime import datetime, timezone

        credential = DefaultAzureCredential()
        service = TableServiceClient(
            endpoint=f"https://{account_name}.table.core.windows.net",
            credential=credential,
        )
        table = service.get_table_client("ratings")

        now = datetime.now(timezone.utc)
        entity = {
            "PartitionKey": now.strftime("%Y-%m"),
            "RowKey": f"{now.isoformat()}-{uuid.uuid4().hex[:8]}",
            "subtitle": subtitle,
            "thumbs": thumbs,
            "tone_override": tone_override or "",
            "free_text": free_text or "",
            "system_tone": system_tone or "",
            "tags": json.dumps(tags or []),
        }
        table.create_entity(entity)
    except Exception:
        pass  # Non-critical — local SQLite is the primary store


# -- /api/spot-check/* (local only) ------------------------------------------

# In-memory store of spot-check samples keyed by sample_id.
# Maps sample_id → {"text": str, "target_tier": str, "batch_id": str}
_spot_check_samples: dict[str, dict] = {}


def _handle_spot_check_batch(body: dict) -> tuple[int, dict]:
    """Generate a shuffled batch of tone-targeted subtitles for spot-checking."""
    from subtitle_generator.config import get_tone_targets

    samples_per_tier = body.get("samples_per_tier", 2)
    if not isinstance(samples_per_tier, int) or not 1 <= samples_per_tier <= 5:
        return 400, {"error": "samples_per_tier must be an integer 1-5"}

    seed_base = body.get("seed_base")
    if seed_base is None:
        seed_base = random.randint(0, 100_000)

    conn = _get_db()
    try:
        targets = get_tone_targets(conn)
        tiers = ["pop", "mainstream", "niche"]
        batch_id = uuid.uuid4().hex[:12]
        items: list[dict] = []

        for tier in tiers:
            tone_target = {
                slot: targets[tier][slot]
                for slot in ["list_item", "action_noun", "of_object"]
            }
            for j in range(samples_per_tier):
                seed = seed_base + tiers.index(tier) * 100 + j
                sub = generate_subtitle(conn, seed=seed, tone_target=tone_target)
                sample_id = uuid.uuid4().hex[:12]

                # Build slot info for display
                slots = _build_slot_info(sub)

                # Store server-side metadata (target_tier NOT sent to client)
                _spot_check_samples[sample_id] = {
                    "text": sub.text,
                    "target_tier": tier,
                    "batch_id": batch_id,
                }

                items.append({
                    "sample_id": sample_id,
                    "text": sub.text,
                    "slots": slots,
                })

        # Shuffle to prevent tier-order bias
        random.shuffle(items)
        return 200, {"batch_id": batch_id, "items": items}
    finally:
        conn.close()


def _build_slot_info(sub: GeneratedSubtitle) -> list[dict]:
    """Build slot display info matching the main app's subtitle-vm.js rendering."""
    action_art = sub.action_article or "the"
    of_art_text = f"of {sub.of_article}" if sub.of_article else "of"

    slots: list[dict] = []
    slots.append({"text": sub.item1, "type": "list_item", "cls": "slot-list1"})
    slots.append({"text": ",", "isPunc": True})
    slots.append({"text": sub.item2, "type": "list_item", "cls": "slot-list2"})
    slots.append({"text": f", and {action_art}", "isPunc": True})
    slots.append({"text": sub.action_noun, "type": "action_noun", "cls": "slot-action"})
    slots.append({"text": of_art_text, "isPunc": True})

    if sub.remixed and sub.remix_parts:
        if "modifier" in sub.remix_parts:
            slots.append({"text": sub.remix_parts["modifier"], "type": "of_modifier", "cls": "slot-subpart"})
            slots.append({"text": sub.remix_parts["head"], "type": "of_head", "cls": "slot-subpart"})
        elif "topic" in sub.remix_parts:
            slots.append({"text": sub.remix_parts["topic"], "type": "of_topic", "cls": "slot-subpart"})
            slots.append({"text": sub.remix_parts["complement"], "type": "of_complement", "cls": "slot-subpart"})
    else:
        slots.append({"text": sub.of_object, "type": "of_object", "cls": "slot-of"})

    return slots


def _handle_spot_check_rate(body: dict) -> tuple[int, dict]:
    """Rate a spot-check sample. Server derives target_tier and thumbs."""
    sample_id = body.get("sample_id")
    if not sample_id or sample_id not in _spot_check_samples:
        return 400, {"error": "Invalid or expired sample_id"}

    skipped = bool(body.get("skipped", False))
    felt_tier = body.get("felt_tier")
    tags = body.get("tags")

    if not skipped:
        if not felt_tier or felt_tier not in ("pop", "mainstream", "niche"):
            return 400, {"error": "felt_tier must be pop, mainstream, or niche"}

    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            return 400, {"error": "tags must be an array of strings"}
        valid_tags = {"funny", "grammar", "contradiction", "boring"}
        invalid_tags = set(tags) - valid_tags
        if invalid_tags:
            return 400, {"error": f"Invalid tags: {', '.join(invalid_tags)}"}

    sample = _spot_check_samples[sample_id]
    target_tier = sample["target_tier"]

    if skipped:
        thumbs = None
        match = None
    else:
        match = felt_tier == target_tier
        thumbs = 1 if match else -1

    from subtitle_generator.feedback import store_rating

    conn = _get_db()
    try:
        store_rating(
            conn,
            sample["text"],
            system_tone=target_tier,
            thumbs=thumbs,
            tone_override=felt_tier if not skipped else None,
            tags=tags,
            source="spot_check_web",
        )
        return 200, {
            "target_tier": target_tier,
            "match": match,
            "sample_id": sample_id,
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
        elif path == "/api/rate":
            try:
                status, resp = _handle_rate_local(body)
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
        elif path == "/api/spot-check/batch":
            try:
                status, resp = _handle_spot_check_batch(body)
            except Exception as exc:
                status, resp = 500, {"error": f"Internal error: {exc}"}
            self._send_json(resp, status)
        elif path == "/api/spot-check/rate":
            try:
                status, resp = _handle_spot_check_rate(body)
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
