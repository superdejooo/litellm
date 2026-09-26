"""Backend MCP session affinity for the servers named in ``MCP_STATEFUL_SERVERS``.

The gateway opens a fresh backend session per tool call, so a stateful backend (a browser, say) forgets
everything between calls. For allowlisted servers the backend session id and protocol version are kept
per caller in Redis and the next call resumes that session. This is affinity, not concurrency safety:
concurrent calls on one key run unserialized in the same backend session.
"""

import asyncio
import hashlib
import json
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Final, Protocol, TypeVar

import httpx
from mcp.types import CallToolRequestParams, CallToolResult, TextContent
from redis.exceptions import RedisError

from litellm._logging import verbose_logger
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

MCP_STICKY_KEY_PREFIX: Final = "mcp:backend_session:"
STICKY_SUFFIX_HEADER: Final = "x-litellm-mcp-sticky"
_STICKY_SUFFIX_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# 400 is not here: it is equally a LIVE session's bad arguments or protocol version, and a retry
# re-runs the tool. A 400 counts only with a JSON-RPC body naming the session (see below).
_STICKY_DEAD_SESSION_STATUSES: Final = (404,)
_STICKY_DEAD_SESSION_ERROR_TEXTS: Final = ("Session terminated", "No valid session")
# A bare "session" substring would also claim "Session already initialized", "Session is currently
# busy" and "Failed to acquire session lock": all LIVE sessions whose tool call would be replayed.
_SESSION_REJECTION_WORDINGS: Final = (
    "no valid session",
    "missing session",
    "session not found",
    "unknown session",
    "invalid or expired session",
)
_STICKY_STORE_DOWN_MARKER: Final = "mcp-sticky-store-unavailable"
_STICKY_BAD_SUFFIX_MARKER: Final = "mcp-sticky-invalid-sticky-header"
_STICKY_LOST_RACE_MARKER: Final = "mcp-sticky-lost-race"
_STICKY_NO_IDENTITY_MARKER: Final = "mcp-sticky-no-caller-identity"
_STICKY_SESSION_MISSING_MARKER: Final = "mcp-sticky-backend-session-missing"
_STICKY_DEFAULT_TTL_SECONDS: Final = 1800
_SELF_RETRYING_AUTH_TYPES: Final = frozenset({MCPAuth.oauth2_token_exchange, MCPAuth.oauth2_id_jag})
_REDIS_TARGET_ENV_VARS: Final = ("REDIS_HOST", "REDIS_URL", "REDIS_CLUSTER_NODES", "REDIS_SENTINEL_NODES")
_STORE_ERRORS: Final = (RedisError, OSError, ValueError)


class SessionRefRedis(Protocol):
    """The subset of ``redis.asyncio.Redis`` the session-ref store uses."""

    async def get(self, name: str) -> bytes | str | None: ...

    async def set(self, name: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool | None: ...

    async def expire(self, name: str, time: int) -> bool | int: ...

    async def delete(self, *names: str) -> int: ...


class ClientFactory(Protocol):
    """Builds a one-shot client for the call's server with these extra headers."""

    def __call__(self, *, extra_headers: dict[str, str] | None) -> Awaitable[MCPClient]: ...


_T = TypeVar("_T")


def redis_from_env(env: Mapping[str, str]) -> SessionRefRedis:
    """The proxy's shared Redis resolution: REDIS_URL, REDIS_SSL, REDIS_USERNAME, cluster, sentinel, cloud IAM."""
    from litellm._redis import _get_redis_env_kwarg_mapping, get_redis_async_client

    configured: Final = {kwarg: env[name] for name, kwarg in _get_redis_env_kwarg_mapping().items() if env.get(name)}
    return get_redis_async_client(**{"socket_timeout": 5, "socket_connect_timeout": 5, **configured})


@dataclass(frozen=True, slots=True)
class SessionRef:
    sid: str
    pv: str | None


class StickyStoreUnavailable(RuntimeError):
    pass


class _RegistrationRefused(Exception):
    """Raised from ``on_session_initialized``, so it aborts the call before the tool runs."""

    def __init__(self, marker: str, winner: SessionRef | None = None) -> None:
        super().__init__(marker)
        self.marker: Final = marker
        self.winner: Final = winner


@dataclass(frozen=True, slots=True)
class _Attempt:
    client: MCPClient
    outcome: CallToolResult | Exception


@dataclass(frozen=True, slots=True)
class StickyCall:
    key: str
    extra_headers: Mapping[str, str] | None
    params: CallToolRequestParams
    host_progress_callback: Callable | None
    create_client: ClientFactory


def valid_session_ref(raw: bytes | str | None) -> SessionRef | None:
    """The one definition of a usable ref, shared by the read side and the ``SET NX`` write side."""
    if raw is None:
        return None
    try:
        ref: Final = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(ref, dict):
        return None
    sid: Final = ref.get("sid")
    pv: Final = ref.get("pv")
    if not isinstance(sid, str) or not sid:
        return None
    return SessionRef(sid=sid, pv=pv if isinstance(pv, str) and pv else None)


def is_session_rejection_body(body: str | bytes) -> bool:
    """Whether an HTTP error body is a JSON-RPC error whose message names an unusable session id."""
    try:
        payload: Final = json.loads(body)
    except (TypeError, ValueError):
        return False
    # Without the envelope a proxy or auth 400 shaped like {"error": {"message": "unknown session"}}
    # would replay a live tool call.
    if not isinstance(payload, dict) or payload.get("jsonrpc") != "2.0":
        return False
    error: Final = payload.get("error")
    message: Final = error.get("message") if isinstance(error, dict) else None
    if not isinstance(message, str):
        return False
    lowered: Final = message.lower()
    return any(wording in lowered for wording in _SESSION_REJECTION_WORDINGS)


def _response_text(response: httpx.Response | None) -> str:
    if response is None:
        return ""
    try:
        return response.text
    except (httpx.ResponseNotRead, httpx.StreamError):
        return ""


def is_dead_session_error(error: BaseException | None) -> bool:
    """Classify the exception ``call_tool(raise_on_error=True)`` raised; a returned tool result never reaches here."""
    if error is None:
        return False
    if isinstance(error, httpx.HTTPStatusError):
        response: Final = getattr(error, "response", None)
        status_code: Final = getattr(response, "status_code", None)
        if status_code in _STICKY_DEAD_SESSION_STATUSES:
            return True
        if status_code == 400 and is_session_rejection_body(_response_text(response)):
            return True
    text: Final = str(error)
    return any(needle in text for needle in _STICKY_DEAD_SESSION_ERROR_TEXTS)


def sticky_error_result(marker: str, detail: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=f"{marker}: {detail}")], isError=True)


def _tool_result(attempt: _Attempt) -> CallToolResult:
    if isinstance(attempt.outcome, CallToolResult):
        return attempt.outcome
    verbose_logger.warning(
        "mcp sticky: tool call to %s failed (%s)", attempt.client.server_url, type(attempt.outcome).__name__
    )
    return MCPClient.error_tool_result(attempt.outcome)


class BackendSessionAffinity:
    def __init__(
        self,
        env: Mapping[str, str] = os.environ,
        redis_factory: Callable[[Mapping[str, str]], SessionRefRedis] = redis_from_env,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._env: Final = env
        self._redis_factory: Final = redis_factory
        self._clock: Final = clock
        self._redis: SessionRefRedis | None = None
        self._local_refs: Final[dict[str, tuple[str, float]]] = {}
        self._local_store_warned = False
        self._termination_tasks: Final[set[asyncio.Task[None]]] = set()

    def stateful_server_names(self) -> frozenset[str]:
        raw: Final = self._env.get("MCP_STATEFUL_SERVERS", "") or ""
        return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())

    def is_sticky_server(self, server: MCPServer) -> bool:
        names: Final = self.stateful_server_names()
        return any(
            isinstance(candidate, str) and candidate.strip().lower() in names
            for candidate in (server.server_name, server.name, server.alias)
        )

    def applies(self, server: MCPServer) -> bool:
        if server.transport != MCPTransport.http or server.spec_path or not self.is_sticky_server(server):
            return False
        if server.auth_type in _SELF_RETRYING_AUTH_TYPES or server.is_client_forwarded_token:
            verbose_logger.warning(
                "mcp sticky: %s is in MCP_STATEFUL_SERVERS but its auth mode (%s) owns the upstream 401 "
                "retry or relay, so it keeps a fresh backend session per call",
                server.name,
                server.auth_type,
            )
            return False
        return True

    def sticky_key(
        self,
        server: MCPServer,
        user_api_key_auth: UserAPIKeyAuth | None,
        raw_headers: Mapping[str, str] | None,
        server_auth_header: Mapping[str, str] | str | None = None,
        effective_headers: Mapping[str, str] | None = None,
    ) -> str | None:
        """Per-caller key, or None when the caller carries no identity.

        There is deliberately no shared anonymous bucket: it would put every unidentified caller in one
        backend session, with each other's cookies.
        """
        identity: Final = (
            next(
                (
                    value
                    for value in (
                        user_api_key_auth.token,
                        user_api_key_auth.api_key,
                        user_api_key_auth.user_id,
                        user_api_key_auth.end_user_id,
                    )
                    if value
                ),
                None,
            )
            if user_api_key_auth is not None
            else None
        )
        if identity is None:
            return None
        fingerprint: Final = hashlib.sha256(str(identity).encode()).hexdigest()[:16]
        generation: Final = self.config_generation(server, server_auth_header, effective_headers)
        key: Final = f"{MCP_STICKY_KEY_PREFIX}{server.server_id}:{generation}:{fingerprint}"
        suffix: Final = self._suffix(raw_headers)
        return f"{key}:{suffix}" if suffix else key

    def config_generation(
        self,
        server: MCPServer,
        server_auth_header: Mapping[str, str] | str | None = None,
        effective_headers: Mapping[str, str] | None = None,
    ) -> str:
        """Digest of the effective upstream identity, so a credential or URL rotation starts a new session.

        server_id survives such a rotation. Secrets only ever enter the digest, never a key or a log.
        """
        auth_material: Final = (
            (json.dumps(dict(server_auth_header), sort_keys=True),)
            if isinstance(server_auth_header, Mapping)
            else ((str(server_auth_header),) if server_auth_header else ())
        )
        material: Final = (
            str(server.url or ""),
            str(server.transport or ""),
            str(server.auth_type or ""),
            json.dumps(server.static_headers or {}, sort_keys=True),
            json.dumps(sorted(server.extra_headers or [])),
            str(server.authentication_token or ""),
            json.dumps(dict(effective_headers or {}), sort_keys=True),
            *auth_material,
        )
        return hashlib.sha256("\x1f".join(material).encode()).hexdigest()[:12]

    def _suffix(self, raw_headers: Mapping[str, str] | None) -> str | None:
        return next(
            (
                value
                for name, value in (raw_headers or {}).items()
                if isinstance(name, str) and name.lower() == STICKY_SUFFIX_HEADER and isinstance(value, str) and value
            ),
            None,
        )

    def suffix_is_valid(self, raw_headers: Mapping[str, str] | None) -> bool:
        """Absent is fine; present but malformed is refused, since ignoring it would share the caller's session."""
        suffix: Final = self._suffix(raw_headers)
        return suffix is None or bool(_STICKY_SUFFIX_PATTERN.match(suffix))

    def ttl_seconds(self) -> int:
        raw: Final = self._env.get("MCP_BACKEND_SESSION_TTL_SECONDS")
        if raw is None or not raw.strip():
            return _STICKY_DEFAULT_TTL_SECONDS
        try:
            value: Final = int(raw)
        except ValueError:
            verbose_logger.warning(
                "mcp sticky: invalid MCP_BACKEND_SESSION_TTL_SECONDS=%r, using %s", raw, _STICKY_DEFAULT_TTL_SECONDS
            )
            return _STICKY_DEFAULT_TTL_SECONDS
        if value < 1:
            verbose_logger.warning(
                "mcp sticky: MCP_BACKEND_SESSION_TTL_SECONDS=%s is below the minimum 1, using %s",
                value,
                _STICKY_DEFAULT_TTL_SECONDS,
            )
            return _STICKY_DEFAULT_TTL_SECONDS
        return value

    def _uses_redis(self) -> bool:
        if any(self._env.get(name) for name in _REDIS_TARGET_ENV_VARS):
            return True
        if not self._local_store_warned:
            self._local_store_warned = True
            verbose_logger.error(
                "mcp sticky: no Redis is configured (%s), so backend sessions are kept PER WORKER. On a "
                "multi-worker proxy calls will intermittently land in a fresh backend session. Configure Redis "
                "or unset MCP_STATEFUL_SERVERS.",
                "/".join(_REDIS_TARGET_ENV_VARS),
            )
        return False

    def _redis_client(self) -> SessionRefRedis:
        if self._redis is None:
            self._redis = self._redis_factory(self._env)
        return self._redis

    async def _redis_op(self, op: str, key: str, operation: Callable[[SessionRefRedis], Awaitable[_T]]) -> _T:
        try:
            return await operation(self._redis_client())
        except _STORE_ERRORS as e:
            verbose_logger.warning(
                "mcp sticky: redis %s for %s failed (%s: %s)", op, key, type(e).__name__, e, exc_info=True
            )
            raise StickyStoreUnavailable(_STICKY_STORE_DOWN_MARKER) from e

    async def _read_raw_ref(self, key: str) -> bytes | str | None:
        if self._uses_redis():
            return await self._redis_op("get", key, lambda redis: redis.get(key))
        entry: Final = self._local_refs.get(key)
        if entry is None:
            return None
        if entry[1] <= self._clock():
            self._local_refs.pop(key, None)
            return None
        return entry[0]

    async def load_ref(self, key: str) -> SessionRef | None:
        raw: Final = await self._read_raw_ref(key)
        if raw is None:
            return None
        ref: Final = valid_session_ref(raw)
        if ref is None:
            # A key the loader skips but SET NX still sees would fail every fresh call until the TTL.
            verbose_logger.warning("mcp sticky: unusable session ref for %s, dropping it", key)
            await self.drop_ref(key)
        return ref

    async def store_ref(self, key: str, value: str, nx: bool = False) -> bool:
        ttl: Final = self.ttl_seconds()
        if self._uses_redis():
            return bool(await self._redis_op("set", key, lambda redis: redis.set(key, value, nx=nx, ex=ttl)))
        now: Final = self._clock()
        entry: Final = self._local_refs.get(key)
        if nx and entry is not None and entry[1] > now:
            return False
        self._local_refs[key] = (value, now + ttl)
        return True

    async def refresh_ref(self, key: str) -> None:
        """Best effort: a lost refresh only means earlier expiry, which the dead-session retry absorbs."""
        ttl: Final = self.ttl_seconds()
        if not self._uses_redis():
            entry: Final = self._local_refs.get(key)
            if entry is not None:
                self._local_refs[key] = (entry[0], self._clock() + ttl)
            return
        try:
            await self._redis_op("expire", key, lambda redis: redis.expire(key, ttl))
        except StickyStoreUnavailable:
            verbose_logger.warning("mcp sticky: ttl refresh failed for %s", key)

    async def drop_ref(self, key: str) -> None:
        if self._uses_redis():
            await self._redis_op("delete", key, lambda redis: redis.delete(key))
            return
        self._local_refs.pop(key, None)

    async def _keepalive(self, key: str) -> None:
        """Keep the cached ref from expiring under a long call; grants no exclusivity and must not grow any."""
        observed = False
        while True:
            ttl: Final = self.ttl_seconds()
            try:
                refreshed: Final = await self._redis_client().expire(key, ttl)
            except Exception as e:
                verbose_logger.warning("mcp sticky: could not refresh the session ref for %s (%s)", key, e)
            else:
                # EXPIRE answers 0 for an absent key: before the first success that is a fresh call not yet
                # registered, afterwards it means the ref this call depends on is gone.
                if refreshed:
                    observed = True
                elif observed:
                    verbose_logger.warning(
                        "mcp sticky: session ref for %s expired mid-call; the next call opens a fresh backend session",
                        key,
                    )
                    return
            await asyncio.sleep(max(0.05, ttl / 3))

    def start_keepalive(self, key: str) -> asyncio.Task[None] | None:
        if not self._uses_redis():
            return None
        return asyncio.create_task(self._keepalive(key))

    async def stop_keepalive(self, task: asyncio.Task[None] | None) -> None:
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def _terminate_in_background(self, client: MCPClient) -> None:
        session_id: Final = client.last_session_id
        if not session_id:
            return
        task: Final = asyncio.create_task(self._terminate(client, session_id))
        self._termination_tasks.add(task)
        task.add_done_callback(self._termination_tasks.discard)

    async def _terminate(self, client: MCPClient, session_id: str) -> None:
        try:
            await client.terminate_session(session_id)
        except Exception as e:
            verbose_logger.warning(
                "mcp sticky: could not DELETE backend session %s on %s (%s); it stays alive until the backend's "
                "own session timeout",
                session_id,
                client.server_url,
                type(e).__name__,
            )

    async def call_tool(
        self,
        *,
        server: MCPServer,
        user_api_key_auth: UserAPIKeyAuth | None,
        raw_headers: Mapping[str, str] | None,
        server_auth_header: Mapping[str, str] | str | None,
        extra_headers: Mapping[str, str] | None,
        params: CallToolRequestParams,
        host_progress_callback: Callable | None,
        create_client: ClientFactory,
    ) -> CallToolResult:
        if not self.suffix_is_valid(raw_headers):
            return sticky_error_result(
                _STICKY_BAD_SUFFIX_MARKER,
                f"'{STICKY_SUFFIX_HEADER}' must match [A-Za-z0-9_-]{{1,64}}; refusing rather than silently "
                "sharing another backend session",
            )
        key: Final = self.sticky_key(server, user_api_key_auth, raw_headers, server_auth_header, extra_headers)
        if key is None:
            return sticky_error_result(
                _STICKY_NO_IDENTITY_MARKER, f"server '{server.name}' is sticky and needs an identifiable caller"
            )
        return await self.call_with_key(
            StickyCall(
                key=key,
                extra_headers=extra_headers,
                params=params,
                host_progress_callback=host_progress_callback,
                create_client=create_client,
            )
        )

    async def call_with_key(self, call: StickyCall) -> CallToolResult:
        keepalive: Final = self.start_keepalive(call.key)
        try:
            verbose_logger.info("mcp sticky call: key=%s pid=%s", call.key, os.getpid())
            try:
                ref: Final = await self.load_ref(call.key)
            except StickyStoreUnavailable:
                return sticky_error_result(_STICKY_STORE_DOWN_MARKER, call.key)
            if ref is None:
                return await self._fresh_call(call, allow_race_resume=True)
            resumed: Final = await self._resume(call, ref)
            if isinstance(resumed.outcome, CallToolResult):
                await self.refresh_ref(call.key)
                return resumed.outcome
            if not is_dead_session_error(resumed.outcome):
                return _tool_result(resumed)
            verbose_logger.info(
                "mcp sticky: backend session for %s is gone (%s), opening a fresh one",
                call.key,
                type(resumed.outcome).__name__,
            )
            try:
                await self.drop_ref(call.key)
            except StickyStoreUnavailable:
                return sticky_error_result(_STICKY_STORE_DOWN_MARKER, call.key)
            return await self._fresh_call(call, allow_race_resume=False)
        finally:
            await self.stop_keepalive(keepalive)

    async def _run(self, client: MCPClient, call: StickyCall) -> _Attempt:
        try:
            result: Final = await client.call_tool(
                call.params, host_progress_callback=call.host_progress_callback, raise_on_error=True
            )
        except Exception as e:
            return _Attempt(client=client, outcome=e)
        return _Attempt(client=client, outcome=result)

    async def _resume(self, call: StickyCall, ref: SessionRef) -> _Attempt:
        headers: Final = {
            **(call.extra_headers or {}),
            "mcp-session-id": ref.sid,
            **({"mcp-protocol-version": ref.pv} if ref.pv else {}),
        }
        client: Final = await call.create_client(extra_headers=headers)
        client.skip_initialize = True
        client.terminate_on_close = False
        return await self._run(client, call)

    async def _register(self, key: str, session_id: str | None, protocol_version: str | None) -> None:
        if not session_id:
            raise _RegistrationRefused(_STICKY_SESSION_MISSING_MARKER)
        try:
            if await self.store_ref(key, json.dumps({"sid": session_id, "pv": protocol_version}), nx=True):
                return
            winner: Final = await self.load_ref(key)
        except StickyStoreUnavailable as e:
            raise _RegistrationRefused(_STICKY_STORE_DOWN_MARKER) from e
        raise _RegistrationRefused(_STICKY_LOST_RACE_MARKER, winner)

    async def _fresh_call(self, call: StickyCall, *, allow_race_resume: bool) -> CallToolResult:
        client: Final = await call.create_client(extra_headers=dict(call.extra_headers) if call.extra_headers else None)
        client.terminate_on_close = False
        client.on_session_initialized = partial(self._register, call.key)
        attempt: Final = await self._run(client, call)
        refused: Final = attempt.outcome if isinstance(attempt.outcome, _RegistrationRefused) else None
        if refused is None:
            return _tool_result(attempt)
        if refused.marker == _STICKY_SESSION_MISSING_MARKER:
            return sticky_error_result(
                _STICKY_SESSION_MISSING_MARKER,
                f"{call.key}: the backend answered initialize without an mcp-session-id; it looks stateless "
                "or misconfigured for sticky routing",
            )
        self._terminate_in_background(client)
        if refused.winner is None:
            return sticky_error_result(_STICKY_STORE_DOWN_MARKER, call.key)
        # One gateway call never builds a third client; the tool has not run on either attempt.
        if not allow_race_resume:
            return sticky_error_result(_STICKY_LOST_RACE_MARKER, call.key)
        return _tool_result(await self._resume(call, refused.winner))
