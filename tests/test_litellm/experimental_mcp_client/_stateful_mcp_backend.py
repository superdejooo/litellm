"""A real HTTP MCP backend that keeps state per ``mcp-session-id``, the way a stateful server does."""

import itertools
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

SUPERGATEWAY_REJECTION: Final = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {"code": -32000, "message": "Bad Request: No valid session ID provided"},
}
PROTOCOL_VERSION_REJECTION: Final = {
    "jsonrpc": "2.0",
    "id": None,
    "error": {"code": -32600, "message": "Bad Request: Unsupported protocol version: 1999-01-01"},
}


@dataclass
class SeenRequest:
    method: str
    rpc_method: str | None
    session_id: str | None
    protocol_version: str | None
    authorization: str | None


@dataclass
class StatefulBackend:
    url: str
    requests: list[SeenRequest] = field(default_factory=list)
    live_sessions: set[str] = field(default_factory=set)
    visited: dict[str, list[str]] = field(default_factory=dict)
    tools_call_status: int | None = None
    tools_call_body: dict | None = None
    unknown_session_status: int = 400

    def rpc_methods(self, session_id: str | None = None) -> list[str]:
        return [
            r.rpc_method
            for r in self.requests
            if r.method == "POST" and r.rpc_method and (session_id is None or r.session_id == session_id)
        ]

    def deletes(self) -> list[str | None]:
        return [r.session_id for r in self.requests if r.method == "DELETE"]


def _handler(backend: StatefulBackend, session_ids: Iterator[int]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:
            return None

        def _reply(self, status: int, body: dict | None = None, headers: dict[str, str] | None = None) -> None:
            payload = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def _record(self, rpc_method: str | None) -> str | None:
            session_id = self.headers.get("mcp-session-id")
            backend.requests.append(
                SeenRequest(
                    method=self.command,
                    rpc_method=rpc_method,
                    session_id=session_id,
                    protocol_version=self.headers.get("mcp-protocol-version"),
                    authorization=self.headers.get("authorization"),
                )
            )
            return session_id

        def do_GET(self) -> None:
            self._record(None)
            self._reply(405)

        def do_DELETE(self) -> None:
            session_id = self._record(None)
            backend.live_sessions.discard(session_id or "")
            self._reply(200)

        def do_POST(self) -> None:
            message = json.loads(self.rfile.read(int(self.headers.get("content-length", 0) or 0)))
            rpc_method = message.get("method")
            session_id = self._record(rpc_method)
            if rpc_method == "initialize":
                new_session = f"sid-{next(session_ids)}"
                backend.live_sessions.add(new_session)
                backend.visited[new_session] = []
                result = {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stateful", "version": "1"},
                }
                self._reply(
                    200, {"jsonrpc": "2.0", "id": message["id"], "result": result}, {"mcp-session-id": new_session}
                )
                return
            if session_id not in backend.live_sessions:
                self._reply(backend.unknown_session_status, SUPERGATEWAY_REJECTION)
                return
            if "id" not in message:
                self._reply(202)
                return
            if rpc_method == "tools/list":
                tools = [{"name": "navigate", "inputSchema": {"type": "object"}}]
                self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": tools}})
                return
            if backend.tools_call_status is not None:
                self._reply(backend.tools_call_status, backend.tools_call_body)
                return
            page = (message["params"].get("arguments") or {}).get("url", "")
            history = backend.visited[session_id or ""]
            if page:
                history.append(page)
            text = history[-1] if history else "about:blank"
            result = {"content": [{"type": "text", "text": text}], "isError": False}
            self._reply(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})

    return Handler


@contextmanager
def running_backend() -> Iterator[StatefulBackend]:
    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    backend: Final = StatefulBackend(url=f"http://127.0.0.1:{server.server_address[1]}/mcp")
    server.RequestHandlerClass = _handler(backend, itertools.count(1))
    server.daemon_threads = True
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield backend
    finally:
        server.shutdown()
        server.server_close()
