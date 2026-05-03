"""Reusable Copilot MCP web_search helper.

This module is intentionally separate from book categorization. It can be
imported directly as a small client for Copilot's hosted MCP `web_search`.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any


MCP_PROTOCOL_VERSION = "2025-06-18"
DEFAULT_MCP_URL = "https://api.githubcopilot.com/mcp/readonly"


@dataclass(frozen=True)
class WebSearchResult:
    """Normalized result from Copilot MCP web_search."""

    text: str
    raw: dict[str, Any]


class CopilotMCPWebSearch:
    """Small client for Copilot's hosted MCP web_search tool."""

    def __init__(self, *, mcp_url: str = DEFAULT_MCP_URL, timeout: float = 90.0) -> None:
        self.mcp_url = mcp_url
        self.timeout = timeout
        self._session_id: str | None = None
        self._token: str | None = None

    def search(self, query: str) -> WebSearchResult:
        """Run Copilot MCP web_search and return normalized text plus raw MCP JSON."""

        if not query.strip():
            raise ValueError("query must not be empty")
        self._ensure_session()
        response = self._post({
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "tools/call",
            "params": {
                "name": "web_search",
                "arguments": {"query": query},
            },
        })
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(f"web_search returned no result: {response}")
        return _normalize_web_search_result(result)

    def _ensure_session(self) -> None:
        if self._session_id:
            return
        response = self._post({
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": "copilot-web-search-proxy",
                    "version": "0.1.0",
                },
            },
        })
        if "result" not in response:
            raise RuntimeError(f"MCP initialize failed: {response}")
        self._post({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = self._headers()
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        request = urllib.request.Request(
            self.mcp_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                session_id = response.headers.get("Mcp-Session-Id")
                if session_id:
                    self._session_id = session_id
                response_body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Copilot MCP request failed: HTTP {exc.code} {exc.reason}: {error_body}"
            ) from exc
        if not response_body.strip():
            return {}
        return _parse_sse_json(response_body)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "X-MCP-Host": "copilot-cli",
            "X-MCP-Toolsets": "web_search",
            "X-MCP-Tools": "web_search",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Mcp-Protocol-Version": MCP_PROTOCOL_VERSION,
        }

    def _get_token(self) -> str:
        if self._token:
            return self._token

        env_token = os.getenv("GITHUB_COPILOT_MCP_TOKEN") or os.getenv("GITHUB_TOKEN")
        if env_token:
            self._token = env_token
            return self._token

        try:
            from litellm.llms.github_copilot.authenticator import Authenticator
        except ImportError as exc:
            raise RuntimeError(
                "Install litellm or set GITHUB_COPILOT_MCP_TOKEN/GITHUB_TOKEN."
            ) from exc

        # Copilot MCP accepts the GitHub OAuth token, not the Copilot IDE API key.
        self._token = Authenticator().get_access_token()
        return self._token


def copilot_web_search(query: str) -> str:
    """Return Copilot MCP web_search text for a query."""

    return CopilotMCPWebSearch().search(query).text


def _parse_sse_json(body: str) -> dict[str, Any]:
    data_lines = [
        line[5:].strip()
        for line in body.splitlines()
        if line.startswith("data:")
    ]
    if not data_lines:
        return json.loads(body)
    return json.loads("\n".join(data_lines))


def _normalize_web_search_result(result: dict[str, Any]) -> WebSearchResult:
    content = result.get("content")
    if not isinstance(content, list) or not content:
        return WebSearchResult(text=json.dumps(result, indent=2), raw=result)

    text_parts: list[str] = []
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            continue
        try:
            parsed = json.loads(raw_text)
            value = parsed.get("text", {}).get("value")
            text_parts.append(value if isinstance(value, str) else raw_text)
        except json.JSONDecodeError:
            text_parts.append(raw_text)

    return WebSearchResult(text="\n\n".join(text_parts).strip(), raw=result)
