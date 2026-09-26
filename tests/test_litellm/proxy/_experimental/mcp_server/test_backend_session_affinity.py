"""Backend MCP session affinity: the sticky store, the dispatch, and the guards that keep removed designs out.

The FakeClient below reproduces the real MCPClient contract the affinity relies on: under
``raise_on_error=True`` a transport failure is raised as the exception object, and after initialize the
client records the session id and awaits ``on_session_initialized`` BEFORE the tool runs. Stubs cannot
prove that contract, so ``TestRealPath`` drives the real client and SDK against a real HTTP backend.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterator
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from mcp.types import CallToolRequestParams, CallToolResult, TextContent
from redis.asyncio.connection import SSLConnection
from redis.exceptions import AuthenticationError

import litellm.proxy._experimental.mcp_server.backend_session_affinity as affinity_module
from litellm._logging import verbose_logger
from litellm.exceptions import GuardrailRaisedException
from litellm.experimental_mcp_client.client import MCPClient
from litellm.proxy._experimental.mcp_server.backend_session_affinity import (
    MCP_STICKY_KEY_PREFIX,
    BackendSessionAffinity,
    SessionRef,
    StickyCall,
    is_dead_session_error,
    is_session_rejection_body,
    redis_from_env,
    valid_session_ref,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests.test_litellm.experimental_mcp_client._stateful_mcp_backend import (
    PROTOCOL_VERSION_REJECTION,
    SUPERGATEWAY_REJECTION,
    StatefulBackend,
    running_backend,
)

STORE_DOWN: Final = affinity_module._STICKY_STORE_DOWN_MARKER
NO_IDENTITY: Final = affinity_module._STICKY_NO_IDENTITY_MARKER
SESSION_MISSING: Final = affinity_module._STICKY_SESSION_MISSING_MARKER
LOST_RACE: Final = affinity_module._STICKY_LOST_RACE_MARKER
BAD_SUFFIX: Final = affinity_module._STICKY_BAD_SUFFIX_MARKER
SERVER_URL: Final = "https://playwright.test/mcp"


class FakeRedis:
    """Enough of redis.asyncio for the session-ref store, with real TTL expiry.

    get() returns bytes because the production client is built without decode_responses.
    """

    def __init__(self, fail: bool = False) -> None:
        self.store: dict[str, tuple[str, float | None]] = {}
        self.fail = fail
        self.fail_sets = False

    def _boom(self) -> None:
        if self.fail:
            raise ConnectionError("redis down")

    def _live(self, key: str) -> str | None:
        entry = self.store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at is not None and expires_at <= time.time():
            self.store.pop(key, None)
            return None
        return value

    async def get(self, name: str) -> bytes | None:
        self._boom()
        value = self._live(name)
        return None if value is None else value.encode()

    async def set(self, name: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool | None:
        self._boom()
        if self.fail_sets:
            raise ConnectionError("redis write refused")
        if nx and self._live(name) is not None:
            return None
        self.store[name] = (value, time.time() + ex if ex else None)
        return True

    async def expire(self, name: str, time_: int) -> int:
        self._boom()
        value = self._live(name)
        if value is None:
            return 0
        self.store[name] = (value, time.time() + time_)
        return 1

    async def delete(self, *names: str) -> int:
        self._boom()
        return sum(1 for name in names if self.store.pop(name, None))


class RaceRedis(FakeRedis):
    """Another worker registers the session between our load and our SET NX."""

    def __init__(self, winner_value: str) -> None:
        super().__init__()
        self.winner_value = winner_value
        self._injected = False

    async def set(self, name: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool | None:
        if nx and not self._injected:
            self._injected = True
            self.store[name] = (self.winner_value, None)
            return None
        return await super().set(name, value, nx=nx, ex=ex)


def ok_result(text: str = "ok") -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


def err_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


class FakeClient:
    """Mirrors MCPClient: capture the session id, await the registration hook before the tool body, and
    raise transport failures under raise_on_error instead of swallowing them into isError."""

    def __init__(
        self,
        sid: str | None = "sess-1",
        pv: str | None = "2026-06-18",
        body: Callable[["FakeClient"], Awaitable[CallToolResult]] | None = None,
        transport_error: Exception | None = None,
    ) -> None:
        self.server_url = SERVER_URL
        self.skip_initialize = False
        self.terminate_on_close = True
        self.on_session_initialized: Callable[[str | None, str | None], Awaitable[None]] | None = None
        self.last_session_id: str | None = None
        self.last_protocol_version: str | None = None
        self._sid = sid
        self._pv = pv
        self._body = body
        self._transport_error = transport_error
        self.ctor_extra_headers: dict[str, str] = {}
        self.terminated: list[tuple[str, str]] = []
        self.body_ran = False
        self.store_at_body: SessionRef | None = None

    async def terminate_session(self, session_id: str) -> None:
        self.terminated.append((self.server_url, session_id))

    async def call_tool(
        self,
        params: CallToolRequestParams,
        host_progress_callback: Callable | None = None,
        raise_on_error: bool = False,
    ) -> CallToolResult:
        try:
            if not self.skip_initialize:
                self.last_protocol_version = self._pv
                self.last_session_id = self._sid
                if self.on_session_initialized is not None:
                    await self.on_session_initialized(self.last_session_id, self.last_protocol_version)
            if self._transport_error is not None:
                raise self._transport_error
            self.body_ran = True
            return ok_result() if self._body is None else await self._body(self)
        except Exception as e:
            if raise_on_error:
                raise
            return MCPClient.error_tool_result(e)


class Factory:
    def __init__(self, clients: list[FakeClient] | None = None) -> None:
        self.clients = list(clients or [])
        self.created: list[FakeClient] = []
        self.terminated: list[tuple[str, str]] = []

    async def __call__(self, *, extra_headers: dict[str, str] | None) -> FakeClient:
        client = self.clients.pop(0) if self.clients else FakeClient()
        client.ctor_extra_headers = dict(extra_headers or {})
        client.terminated = self.terminated
        self.created.append(client)
        return client


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make_affinity(
    redis: FakeRedis | None = None,
    stateful: str = "Playwright",
    clock: Callable[[], float] = time.time,
    **env: str,
) -> BackendSessionAffinity:
    environment: Final = {"MCP_STATEFUL_SERVERS": stateful, **env}
    if redis is not None:
        environment["REDIS_HOST"] = "redis.test"
    return BackendSessionAffinity(env=environment, redis_factory=lambda _env: redis, clock=clock)


def failing_redis_affinity(stateful: str = "Playwright") -> BackendSessionAffinity:
    def factory(_env: object) -> FakeRedis:
        raise ConnectionError("cannot reach redis")

    return BackendSessionAffinity(
        env={"MCP_STATEFUL_SERVERS": stateful, "REDIS_HOST": "redis.test"}, redis_factory=factory
    )


def server(**overrides: object) -> MCPServer:
    fields: Final = {
        "server_id": "srv-1",
        "name": "playwright",
        "url": SERVER_URL,
        "transport": MCPTransport.http,
        **overrides,
    }
    return MCPServer(**fields)


def auth(**fields: str | None) -> UserAPIKeyAuth:
    return UserAPIKeyAuth.model_construct(
        **{"token": None, "api_key": None, "user_id": None, "end_user_id": None, **fields}
    )


def text_of(result: CallToolResult) -> str:
    return " ".join(getattr(item, "text", "") for item in result.content)


def stored_sid(redis: FakeRedis, key: str = "key-1") -> str:
    return json.loads(redis.store[key][0])["sid"]


async def settle() -> None:
    for _ in range(4):
        await asyncio.sleep(0)


async def call_sticky(aff: BackendSessionAffinity, factory: Factory, key: str = "key-1") -> CallToolResult:
    return await aff.call_with_key(
        StickyCall(
            key=key,
            extra_headers={"X-Backend": "1"},
            params=CallToolRequestParams(name="browser_snapshot", arguments={}),
            host_progress_callback=None,
            create_client=factory,
        )
    )


async def call_gated(
    aff: BackendSessionAffinity,
    factory: Factory,
    srv: MCPServer,
    user: UserAPIKeyAuth | None,
    raw_headers: dict[str, str] | None = None,
) -> CallToolResult:
    return await aff.call_tool(
        server=srv,
        user_api_key_auth=user,
        raw_headers=raw_headers,
        server_auth_header=None,
        extra_headers=None,
        params=CallToolRequestParams(name="browser_snapshot", arguments={}),
        host_progress_callback=None,
        create_client=factory,
    )


def http_status_error(status: int, body: bytes = b"") -> httpx.HTTPStatusError:
    request: Final = httpx.Request("POST", SERVER_URL)
    response: Final = httpx.Response(status, request=request, content=body)
    return httpx.HTTPStatusError(
        f"Client error '{status} Bad Request' for url '{request.url}'", request=request, response=response
    )


def jsonrpc_error_body(message: str) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": None, "error": {"code": -32000, "message": message}}).encode()


@pytest.fixture
def log_messages() -> Iterator[list[str]]:
    messages: Final[list[str]] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler: Final = Collect(level=logging.DEBUG)
    previous: Final = verbose_logger.level
    verbose_logger.addHandler(handler)
    verbose_logger.setLevel(logging.DEBUG)
    try:
        yield messages
    finally:
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(previous)


class TestClientContract:
    AFFINITY_CLIENT_ATTRS: Final = (
        "skip_initialize",
        "terminate_on_close",
        "on_session_initialized",
        "last_session_id",
        "terminate_session",
        "call_tool",
    )

    def test_the_real_client_carries_every_attribute_the_affinity_uses(self) -> None:
        client: Final = MCPClient(server_url=SERVER_URL)
        for attr in self.AFFINITY_CLIENT_ATTRS:
            assert hasattr(client, attr), f"the affinity uses client.{attr} but MCPClient has no such attribute"
        assert client.skip_initialize is False
        assert client.terminate_on_close is True
        assert client.on_session_initialized is None

    def test_the_fake_client_cannot_drift_from_the_real_one(self) -> None:
        client: Final = FakeClient()
        for attr in self.AFFINITY_CLIENT_ATTRS:
            assert hasattr(client, attr), f"FakeClient lacks {attr}: the suite would pass against a different client"


class TestAllowlist:
    def test_empty_env_disables_everything(self) -> None:
        aff: Final = make_affinity(stateful="")
        assert aff.is_sticky_server(server(name="Playwright")) is False
        assert aff.stateful_server_names() == frozenset()

    def test_matches_server_name_name_and_alias_case_insensitively(self) -> None:
        aff: Final = make_affinity(stateful="playwright , other")
        assert aff.is_sticky_server(server(name="PLAYWRIGHT")) is True
        assert aff.is_sticky_server(server(name="x", server_name="Playwright")) is True
        assert aff.is_sticky_server(server(name="x", alias="playWright")) is True

    def test_non_listed_server_is_not_sticky(self) -> None:
        assert make_affinity().is_sticky_server(server(name="sentry")) is False


class TestStickyKey:
    def test_distinct_tokens_get_distinct_keys(self) -> None:
        aff: Final = make_affinity()
        first: Final = aff.sticky_key(server(), auth(token="a"), None)
        second: Final = aff.sticky_key(server(), auth(token="b"), None)
        assert first and second and first != second
        assert first.startswith(MCP_STICKY_KEY_PREFIX + "srv-1:")

    def test_identity_precedence_token_then_api_key_then_user_then_end_user(self) -> None:
        aff: Final = make_affinity()
        srv: Final = server()
        by_token: Final = aff.sticky_key(srv, auth(token="T", api_key="K", user_id="U", end_user_id="E"), None)
        by_api_key: Final = aff.sticky_key(srv, auth(api_key="T", user_id="U", end_user_id="E"), None)
        by_user: Final = aff.sticky_key(srv, auth(user_id="T", end_user_id="E"), None)
        by_end_user: Final = aff.sticky_key(srv, auth(end_user_id="T"), None)
        assert by_token == by_api_key == by_user == by_end_user

    def test_jwt_shaped_caller_without_api_key_gets_its_own_bucket(self) -> None:
        aff: Final = make_affinity()
        jwt_one: Final = aff.sticky_key(server(), auth(token="jwt-1", user_id="u"), None)
        jwt_two: Final = aff.sticky_key(server(), auth(token="jwt-2", user_id="u"), None)
        assert jwt_one != jwt_two

    def test_sticky_header_suffix_applied_when_valid(self) -> None:
        aff: Final = make_affinity()
        base: Final = aff.sticky_key(server(), auth(token="a"), None)
        suffixed: Final = aff.sticky_key(server(), auth(token="a"), {"X-LiteLLM-MCP-Sticky": "tab_2-b"})
        assert suffixed == f"{base}:tab_2-b"

    def test_malformed_sticky_header_is_refused_not_ignored(self) -> None:
        aff: Final = make_affinity()
        for bad in ("has space", "x" * 65, "semi;colon"):
            assert aff.suffix_is_valid({"x-litellm-mcp-sticky": bad}) is False, bad
        assert aff.suffix_is_valid({"x-litellm-mcp-sticky": "ok_1-a"})
        assert aff.suffix_is_valid({"x-litellm-mcp-sticky": ""})
        assert aff.suffix_is_valid(None)

    @pytest.mark.asyncio
    async def test_malformed_suffix_fails_closed_end_to_end(self) -> None:
        redis: Final = FakeRedis()
        factory: Final = Factory([FakeClient()])
        result: Final = await call_gated(
            make_affinity(redis),
            factory,
            server(name="Playwright"),
            auth(token="t"),
            raw_headers={"x-litellm-mcp-sticky": "agent/1"},
        )
        assert result.isError is True
        assert BAD_SUFFIX in text_of(result)
        assert factory.created == []
        assert redis.store == {}

    def test_no_identity_returns_none(self) -> None:
        aff: Final = make_affinity()
        assert aff.sticky_key(server(), None, None) is None
        assert aff.sticky_key(server(), auth(), None) is None


class TestConfigGeneration:
    """server_id survives a config change, so without a generation in the key a rotation leaves every
    caller pinned to a session the backend now rejects."""

    def _key(self, srv: MCPServer, auth_header: str | None = None) -> str | None:
        return make_affinity().sticky_key(srv, auth(token="caller-1"), None, auth_header)

    def test_credential_rotation_changes_the_key(self) -> None:
        assert self._key(server(authentication_token="old-secret")) != self._key(
            server(authentication_token="new-secret")
        )

    def test_url_rotation_changes_the_key(self) -> None:
        assert self._key(server(url="https://a.test/mcp")) != self._key(server(url="https://b.test/mcp"))

    def test_static_auth_header_rotation_changes_the_key(self) -> None:
        assert self._key(server(static_headers={"X-Key": "old"})) != self._key(server(static_headers={"X-Key": "new"}))

    def test_different_upstream_oauth_headers_do_not_share_a_session(self) -> None:
        """Same LiteLLM key, different upstream credential: resuming the other caller's authenticated
        session would defeat the isolation entirely."""
        aff: Final = make_affinity()
        first: Final = aff.sticky_key(server(), auth(token="caller-1"), None, None, {"Authorization": "Bearer aaa"})
        second: Final = aff.sticky_key(server(), auth(token="caller-1"), None, None, {"Authorization": "Bearer bbb"})
        assert first != second
        assert first is not None and "aaa" not in first
        assert second is not None and "bbb" not in second

    def test_forwarded_header_rotation_changes_the_key(self) -> None:
        aff: Final = make_affinity()
        before: Final = aff.sticky_key(server(), auth(token="c"), None, None, {"X-Fwd": "old"})
        after: Final = aff.sticky_key(server(), auth(token="c"), None, None, {"X-Fwd": "new"})
        assert before != after

    def test_header_order_does_not_change_the_key(self) -> None:
        aff: Final = make_affinity()
        one: Final = aff.sticky_key(server(), auth(token="c"), None, None, {"A": "1", "B": "2"})
        two: Final = aff.sticky_key(server(), auth(token="c"), None, None, {"B": "2", "A": "1"})
        assert one == two

    def test_per_request_auth_override_changes_the_key(self) -> None:
        assert self._key(server(), "tok-a") != self._key(server(), "tok-b")

    def test_auth_type_and_transport_rotation_change_the_key(self) -> None:
        base: Final = self._key(server(auth_type=MCPAuth.bearer_token))
        assert base != self._key(server(auth_type=MCPAuth.api_key))
        assert base != self._key(server(auth_type=MCPAuth.bearer_token, transport=MCPTransport.sse))

    def test_unchanged_configuration_is_stable_across_calls(self) -> None:
        srv: Final = server(authentication_token="s", static_headers={"X-Key": "v"})
        assert self._key(srv) == self._key(srv)

    def test_the_secret_never_appears_in_the_key(self) -> None:
        secret: Final = "sk-super-secret-value"
        key: Final = self._key(server(authentication_token=secret, static_headers={"X-Key": secret}), secret)
        assert key is not None and secret not in key
        assert make_affinity().config_generation(server(authentication_token=secret)) not in (secret, "")


class TestLocalStore:
    @pytest.mark.asyncio
    async def test_roundtrip(self) -> None:
        aff: Final = make_affinity()
        assert await aff.store_ref("k", json.dumps({"sid": "s1", "pv": "p"}))
        assert await aff.load_ref("k") == SessionRef(sid="s1", pv="p")

    @pytest.mark.asyncio
    async def test_expiry_honored(self) -> None:
        clock: Final = Clock()
        aff: Final = make_affinity(clock=clock, MCP_BACKEND_SESSION_TTL_SECONDS="60")
        await aff.store_ref("k", json.dumps({"sid": "s1"}))
        clock.now += 61
        assert await aff.load_ref("k") is None

    @pytest.mark.asyncio
    async def test_nx_second_write_loses(self) -> None:
        aff: Final = make_affinity()
        assert await aff.store_ref("k", json.dumps({"sid": "s1"}), nx=True) is True
        assert await aff.store_ref("k", json.dumps({"sid": "s2"}), nx=True) is False
        assert await aff.load_ref("k") == SessionRef(sid="s1", pv=None)


class TestFailClosed:
    @pytest.mark.asyncio
    async def test_redis_down_returns_marker_without_creating_a_client(self) -> None:
        factory: Final = Factory()
        result: Final = await call_sticky(failing_redis_affinity(), factory)
        assert result.isError is True
        assert STORE_DOWN in text_of(result)
        assert factory.created == []

    @pytest.mark.asyncio
    async def test_non_sticky_server_never_touches_the_store(self) -> None:
        with running_backend() as backend:
            manager: Final = MCPServerManager(backend_session_affinity=failing_redis_affinity())
            result: Final = await call_manager(manager, backend_server(backend, name="sentry"), auth(token="t"))
        assert result.isError is False
        assert backend.rpc_methods().count("initialize") == 1


class TestFreshPath:
    @pytest.mark.asyncio
    async def test_registers_session_and_configures_client(self) -> None:
        redis: Final = FakeRedis()
        client: Final = FakeClient(sid="backend-abc", pv="2026-06-18")
        result: Final = await call_sticky(make_affinity(redis), Factory([client]))
        assert result.isError is False
        assert client.terminate_on_close is False
        assert client.skip_initialize is False
        assert client.on_session_initialized is not None
        assert json.loads(redis.store["key-1"][0]) == {"sid": "backend-abc", "pv": "2026-06-18"}
        assert "mcp-session-id" not in client.ctor_extra_headers

    @pytest.mark.asyncio
    async def test_registration_survives_a_tool_exception(self) -> None:
        redis: Final = FakeRedis()

        async def body(_client: FakeClient) -> CallToolResult:
            raise RuntimeError("tool blew up")

        factory: Final = Factory([FakeClient(sid="backend-abc", body=body)])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is True
        assert "tool blew up" in text_of(result)
        assert stored_sid(redis) == "backend-abc"
        assert factory.terminated == []

    @pytest.mark.asyncio
    async def test_registration_completes_before_the_tool_body_runs(self) -> None:
        redis: Final = FakeRedis()
        aff: Final = make_affinity(redis)

        async def body(client: FakeClient) -> CallToolResult:
            client.store_at_body = await aff.load_ref("key-1")
            return ok_result()

        client: Final = FakeClient(sid="backend-abc", body=body)
        await call_sticky(aff, Factory([client]))
        assert client.store_at_body == SessionRef(sid="backend-abc", pv="2026-06-18")


class TestMarkerSpoofing:
    @pytest.mark.asyncio
    async def test_tool_output_echoing_a_marker_is_not_a_registration_outcome(self) -> None:
        redis: Final = FakeRedis()

        async def echo(_client: FakeClient) -> CallToolResult:
            return ok_result("page text: mcp-sticky-lost-race mcp-sticky-store-unavailable 400 Bad Request")

        factory: Final = Factory([FakeClient(sid="real-1", body=echo)])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is False
        assert len(factory.created) == 1
        assert factory.terminated == []
        assert stored_sid(redis) == "real-1"

    @pytest.mark.asyncio
    async def test_successful_resume_output_with_stale_text_is_not_retried(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)

        async def echo(_client: FakeClient) -> CallToolResult:
            return ok_result("the site answered 400 Bad Request: No valid session ID provided")

        factory: Final = Factory([FakeClient(body=echo)])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is False
        assert len(factory.created) == 1
        assert stored_sid(redis) == "live-1"

    @pytest.mark.asyncio
    async def test_side_effecting_tool_error_naming_the_session_never_re_runs_the_tool(self) -> None:
        """A tool that mutates state and then fails with text naming the session must NOT cost the live
        session nor a second execution: a returned isError result is never a transport exception."""
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)
        executions: Final[list[str | None]] = []

        async def side_effecting(client: FakeClient) -> CallToolResult:
            executions.append(client.ctor_extra_headers.get("mcp-session-id"))
            return err_result("browser_run_code failed after submit: 400 Bad Request, Session terminated")

        factory: Final = Factory([FakeClient(body=side_effecting), FakeClient(sid="new-1")])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert executions == ["live-1"]
        assert len(factory.created) == 1
        assert result.isError is True
        assert stored_sid(redis) == "live-1"


class TestResumePath:
    @pytest.mark.asyncio
    async def test_resume_sends_session_headers_and_refreshes_ttl(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1", "pv": "2026-06-18"}), time.time() + 5)
        client: Final = FakeClient()
        result: Final = await call_sticky(
            make_affinity(redis, MCP_BACKEND_SESSION_TTL_SECONDS="1800"), Factory([client])
        )
        assert result.isError is False
        assert client.skip_initialize is True
        assert client.terminate_on_close is False
        assert client.ctor_extra_headers["mcp-session-id"] == "live-1"
        assert client.ctor_extra_headers["mcp-protocol-version"] == "2026-06-18"
        assert client.ctor_extra_headers["X-Backend"] == "1"
        expires_at: Final = redis.store["key-1"][1]
        assert expires_at is not None and expires_at > time.time() + 1000

    @pytest.mark.asyncio
    async def test_resume_omits_protocol_header_when_pv_missing(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1", "pv": None}), None)
        client: Final = FakeClient()
        await call_sticky(make_affinity(redis), Factory([client]))
        assert "mcp-protocol-version" not in client.ctor_extra_headers


class TestStaleRetry:
    @pytest.mark.asyncio
    async def test_dead_session_drops_key_and_retries_once_fresh(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "dead-1", "pv": "p"}), None)
        first: Final = FakeClient(transport_error=RuntimeError("Session terminated"))
        second: Final = FakeClient(sid="fresh-9")
        factory: Final = Factory([first, second])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is False
        assert len(factory.created) == 2
        assert first.skip_initialize is True
        assert second.skip_initialize is False
        assert "mcp-session-id" not in second.ctor_extra_headers
        assert stored_sid(redis) == "fresh-9"

    @pytest.mark.asyncio
    async def test_a_real_404_triggers_the_retry(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "dead-1"}), None)
        factory: Final = Factory([FakeClient(transport_error=http_status_error(404)), FakeClient(sid="f-9")])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is False
        assert len(factory.created) == 2

    @pytest.mark.asyncio
    async def test_a_400_naming_the_session_in_a_jsonrpc_envelope_triggers_the_retry(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "dead-1"}), None)
        rejected: Final = http_status_error(400, jsonrpc_error_body("Bad Request: No valid session ID provided"))
        factory: Final = Factory([FakeClient(transport_error=rejected), FakeClient(sid="f-9")])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is False
        assert len(factory.created) == 2
        assert stored_sid(redis) == "f-9"

    @pytest.mark.asyncio
    async def test_session_terminated_error_triggers_the_retry(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "dead-1"}), None)
        factory: Final = Factory(
            [FakeClient(transport_error=RuntimeError("Session terminated")), FakeClient(sid="fresh-9")]
        )
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is False
        assert len(factory.created) == 2

    @pytest.mark.asyncio
    async def test_drop_failure_during_the_retry_does_not_open_a_second_session(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "dead-1"}), None)

        async def delete_fails(*names: str) -> int:
            raise ConnectionError("redis blip")

        redis.delete = delete_fails
        factory: Final = Factory(
            [FakeClient(transport_error=RuntimeError("Session terminated")), FakeClient(sid="f-9")]
        )
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is True
        assert STORE_DOWN in text_of(result)
        assert len(factory.created) == 1


class TestDeadSessionPredicate:
    def test_no_transport_error_is_never_a_dead_session(self) -> None:
        assert is_dead_session_error(None) is False

    def test_dead_session_statuses(self) -> None:
        assert is_dead_session_error(http_status_error(404)) is True
        assert affinity_module._STICKY_DEAD_SESSION_STATUSES == (404,)

    def test_a_bare_400_is_not_a_dead_session(self) -> None:
        """400 also means bad arguments, from a LIVE session."""
        assert is_dead_session_error(http_status_error(400)) is False

    def test_the_sdk_session_wording_is_a_dead_session(self) -> None:
        assert is_dead_session_error(RuntimeError("Session terminated")) is True
        assert is_dead_session_error(RuntimeError("No valid session")) is True

    def test_other_statuses_and_errors_are_not_dead_sessions(self) -> None:
        for error in (
            http_status_error(500),
            http_status_error(503),
            TimeoutError("read timed out"),
            RuntimeError("something else entirely"),
        ):
            assert is_dead_session_error(error) is False

    def test_sdk_session_texts_are_dead_sessions(self) -> None:
        assert is_dead_session_error(RuntimeError("Session terminated")) is True
        assert is_dead_session_error(RuntimeError("No valid session ID")) is True

    def test_a_status_less_error_does_not_crash(self) -> None:
        broken: Final = http_status_error(400, jsonrpc_error_body("No valid session ID provided"))
        broken.response = None
        assert is_dead_session_error(broken) is False

    def test_a_400_body_that_was_never_read_is_not_a_dead_session(self) -> None:
        request: Final = httpx.Request("POST", SERVER_URL)
        streamed: Final = httpx.Response(400, request=request, stream=httpx.ByteStream(b"unread"))
        error: Final = httpx.HTTPStatusError("400", request=request, response=streamed)
        assert is_dead_session_error(error) is False

    def test_the_rejection_body_counts_only_on_a_400(self) -> None:
        body: Final = jsonrpc_error_body("Bad Request: No valid session ID provided")
        assert is_dead_session_error(http_status_error(400, body)) is True
        assert is_dead_session_error(http_status_error(500, body)) is False
        assert is_dead_session_error(http_status_error(403, body)) is False


class TestSessionRejectionBody:
    """The permissive direction re-runs a tool against a LIVE session."""

    def test_supergateway_wording_is_a_rejection(self) -> None:
        assert is_session_rejection_body(jsonrpc_error_body("Bad Request: No valid session ID provided"))

    def test_python_sdk_wording_is_a_rejection(self) -> None:
        assert is_session_rejection_body(jsonrpc_error_body("Bad Request: Missing session ID"))

    def test_protocol_version_rejection_is_not_a_session_rejection(self) -> None:
        assert not is_session_rejection_body(
            jsonrpc_error_body("Bad Request: Unsupported protocol version: 1999-01-01")
        )

    @pytest.mark.parametrize(
        "message",
        [
            "Bad Request: Session already initialized",
            "Session is currently busy",
            "Failed to acquire session lock",
            "Tool failed inside session context",
            "Bad Request: Streamable HTTP session required",
        ],
    )
    def test_live_session_wordings_are_not_rejections(self, message: str) -> None:
        assert not is_session_rejection_body(jsonrpc_error_body(message))

    def test_an_empty_body_is_not_a_rejection(self) -> None:
        assert not is_session_rejection_body(b"")

    @pytest.mark.parametrize(
        "body",
        [
            b'{"error": {"message": "unknown session"}}',
            b'{"jsonrpc": "1.0", "error": {"message": "no valid session"}}',
            b'{"jsonrpc": 2.0, "error": {"message": "no valid session"}}',
        ],
    )
    def test_a_body_without_the_jsonrpc_envelope_is_not_a_rejection(self, body: bytes) -> None:
        assert not is_session_rejection_body(body)

    @pytest.mark.parametrize(
        "message", ["Session not found", "Unknown session id", "Not Found: Invalid or expired session ID"]
    )
    def test_other_dead_session_wordings_are_rejections(self, message: str) -> None:
        assert is_session_rejection_body(jsonrpc_error_body(message))

    @pytest.mark.parametrize(
        "body",
        [
            b"<html>no valid session</html>",
            b'{"error": {"message": "no valid session',
            b'["no valid session"]',
            b'{"jsonrpc": "2.0", "result": "no valid session"}',
            b'{"jsonrpc": "2.0", "error": "no valid session"}',
            b'{"jsonrpc": "2.0", "error": {"message": 42}}',
        ],
    )
    def test_malformed_bodies_are_not_rejections(self, body: bytes) -> None:
        assert not is_session_rejection_body(body)

    def test_match_is_case_insensitive(self) -> None:
        assert is_session_rejection_body(jsonrpc_error_body("No Valid SESSION id"))


class TestStickyDispatchGate:
    def test_sse_transport_never_takes_the_sticky_path(self) -> None:
        assert make_affinity().applies(server(name="Playwright", transport=MCPTransport.sse)) is False

    def test_openapi_server_never_takes_the_sticky_path(self) -> None:
        assert make_affinity().applies(server(name="Playwright", spec_path="openapi.yaml")) is False

    def test_an_allowlisted_http_server_takes_the_sticky_path(self) -> None:
        assert make_affinity().applies(server(name="Playwright")) is True

    @pytest.mark.parametrize(
        "auth_type", [MCPAuth.oauth2_token_exchange, MCPAuth.oauth2_id_jag, MCPAuth.oauth_delegate]
    )
    def test_auth_modes_that_own_the_401_retry_keep_the_upstream_path(
        self, auth_type: MCPAuth, log_messages: list[str]
    ) -> None:
        assert make_affinity().applies(server(name="Playwright", auth_type=auth_type)) is False
        assert any("owns the upstream 401" in m for m in log_messages)

    @pytest.mark.asyncio
    async def test_unrelated_transport_error_is_returned_as_is(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)
        factory: Final = Factory([FakeClient(transport_error=TimeoutError("read timed out"))])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is True
        assert "TimeoutError" in text_of(result)
        assert len(factory.created) == 1
        assert "key-1" in redis.store

    @pytest.mark.asyncio
    async def test_a_500_is_not_a_dead_session(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)
        factory: Final = Factory([FakeClient(transport_error=http_status_error(500))])
        result: Final = await call_sticky(make_affinity(redis), factory)
        assert result.isError is True
        assert len(factory.created) == 1
        assert stored_sid(redis) == "live-1"

    @pytest.mark.asyncio
    async def test_a_failed_resume_does_not_extend_the_session_ref(self) -> None:
        """A session that keeps failing with an error not classified as dead must age out, not be kept alive
        by every failed call."""
        clock: Final = Clock()
        aff: Final = make_affinity(clock=clock)
        await aff.store_ref("key-1", json.dumps({"sid": "live-1"}))
        clock.now += 1000
        result: Final = await call_sticky(aff, Factory([FakeClient(transport_error=http_status_error(500))]))
        clock.now += 900
        assert result.isError is True
        assert await aff.load_ref("key-1") is None

    @pytest.mark.asyncio
    async def test_a_successful_resume_extends_the_session_ref(self) -> None:
        clock: Final = Clock()
        aff: Final = make_affinity(clock=clock)
        await aff.store_ref("key-1", json.dumps({"sid": "live-1"}))
        clock.now += 1000
        result: Final = await call_sticky(aff, Factory([FakeClient()]))
        clock.now += 900
        assert result.isError is False
        assert await aff.load_ref("key-1") == SessionRef(sid="live-1", pv=None)


class TestNonStickyUntouched:
    @pytest.mark.asyncio
    async def test_defaults_preserved_and_no_store_interaction(self) -> None:
        redis: Final = FakeRedis()
        with running_backend() as backend:
            manager: Final = MCPServerManager(backend_session_affinity=make_affinity(redis))
            result: Final = await call_manager(manager, backend_server(backend, name="sentry"), auth(token="t"))
        assert result.isError is False
        assert backend.rpc_methods()[0] == "initialize"
        assert backend.deletes() == ["sid-1"]
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_feature_off_leaves_allowlisted_server_untouched(self) -> None:
        redis: Final = FakeRedis()
        with running_backend() as backend:
            manager: Final = MCPServerManager(backend_session_affinity=make_affinity(redis, stateful=""))
            await call_manager(manager, backend_server(backend, name="Playwright"), auth(token="t"))
        assert backend.deletes() == ["sid-1"]
        assert redis.store == {}


class TestStoreFailurePhases:
    @pytest.mark.asyncio
    async def test_registration_failure_aborts_before_the_tool_body(self) -> None:
        redis: Final = FakeRedis()
        redis.fail_sets = True
        client: Final = FakeClient(sid="orphan-1")
        factory: Final = Factory([client])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is True
        assert STORE_DOWN in text_of(result)
        assert client.body_ran is False
        assert factory.terminated == [(SERVER_URL, "orphan-1")]

    @pytest.mark.asyncio
    async def test_refresh_failure_after_success_does_not_change_the_result(self, log_messages: list[str]) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)

        async def failing_expire(name: str, time_: int) -> int:
            raise ConnectionError("redis blip")

        redis.expire = failing_expire
        result: Final = await call_sticky(make_affinity(redis), Factory([FakeClient()]))
        assert result.isError is False
        assert any("ttl refresh failed" in m for m in log_messages)


class TestUnusableRefSelfHeals:
    @pytest.mark.asyncio
    async def test_corrupt_value_is_dropped_and_the_call_succeeds(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = ("not-json", None)
        factory: Final = Factory([FakeClient(sid="fresh-1")])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is False, text_of(result)
        assert STORE_DOWN not in text_of(result)
        assert len(factory.created) == 1
        assert factory.terminated == []
        assert stored_sid(redis) == "fresh-1"

    @pytest.mark.asyncio
    async def test_ref_without_sid_is_dropped_too(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"pv": "2026-06-18"}), None)
        result: Final = await call_sticky(make_affinity(redis), Factory([FakeClient(sid="fresh-1")]))
        assert result.isError is False
        assert stored_sid(redis) == "fresh-1"

    @pytest.mark.asyncio
    async def test_empty_sid_is_refused_before_it_can_poison_the_key(self) -> None:
        redis: Final = FakeRedis()
        client: Final = FakeClient(sid="")
        result: Final = await call_sticky(make_affinity(redis), Factory([client]))
        assert result.isError is True
        assert SESSION_MISSING in text_of(result)
        assert client.body_ran is False
        assert "key-1" not in redis.store

    def test_validity_predicate_rejects_every_unusable_shape(self) -> None:
        for bad in (
            "not-json",
            "[]",
            '"sess"',
            json.dumps({"pv": "x"}),
            json.dumps({"sid": ""}),
            json.dumps({"sid": None}),
            json.dumps({"sid": 7}),
        ):
            assert valid_session_ref(bad) is None, bad
        assert valid_session_ref(json.dumps({"sid": "s", "pv": None})) == SessionRef(sid="s", pv=None)
        assert valid_session_ref(b'{"sid": "s"}') == SessionRef(sid="s", pv=None)


class TestSessionRefKeepalive:
    @pytest.mark.asyncio
    async def test_expire_returning_zero_after_a_live_ref_is_not_a_refresh(self, log_messages: list[str]) -> None:
        """EXPIRE answers 0 when the key is gone. Once the ref has been seen alive, that is expiry, not a
        successful refresh."""
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "s"}), None)
        aff: Final = make_affinity(redis, MCP_BACKEND_SESSION_TTL_SECONDS="1")
        task: Final = aff.start_keepalive("key-1")
        assert task is not None
        await asyncio.sleep(0.5)
        redis.store.pop("key-1", None)
        await asyncio.wait_for(asyncio.shield(task), timeout=3.0)
        assert task.done() is True
        assert any("expired mid-call" in m for m in log_messages)

    @pytest.mark.asyncio
    async def test_an_absent_ref_is_tolerated_until_registration_lands(self, log_messages: list[str]) -> None:
        """On the fresh path the ref does not exist when the keepalive starts; treating that first 0 as
        expiry would kill every fresh call."""
        aff: Final = make_affinity(FakeRedis(), MCP_BACKEND_SESSION_TTL_SECONDS="1")
        task: Final = aff.start_keepalive("key-1")
        assert task is not None
        await asyncio.sleep(0.5)
        still_running: Final = not task.done()
        await aff.stop_keepalive(task)
        assert still_running is True
        assert not any("expired mid-call" in m for m in log_messages)

    @pytest.mark.asyncio
    async def test_a_resumed_ref_is_topped_up_before_the_first_sleep(self) -> None:
        """A resumed ref carries only whatever TTL it had left, which can be less than the refresh interval,
        so the first refresh must not wait."""
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "s"}), time.time() + 0.2)
        aff: Final = make_affinity(redis, MCP_BACKEND_SESSION_TTL_SECONDS="30")
        task: Final = aff.start_keepalive("key-1")
        await asyncio.sleep(0.5)
        alive: Final = await aff.load_ref("key-1")
        await aff.stop_keepalive(task)
        assert alive is not None, "a nearly-expired resumed ref died before the first refresh"

    @pytest.mark.asyncio
    async def test_a_store_error_does_not_end_the_keepalive(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "s"}), None)
        original: Final = redis.expire
        calls: Final[list[str]] = []

        async def flaky(name: str, time_: int) -> int:
            calls.append(name)
            if len(calls) == 1:
                raise ConnectionError("blip")
            return await original(name, time_)

        redis.expire = flaky
        aff: Final = make_affinity(redis, MCP_BACKEND_SESSION_TTL_SECONDS="1")
        task: Final = aff.start_keepalive("key-1")
        await asyncio.sleep(1.2)
        running: Final = task is not None and not task.done()
        await aff.stop_keepalive(task)
        assert running is True, "a blip retired the cache keepalive"
        assert len(calls) >= 2, "the keepalive stopped retrying after the blip"

    def test_keepalive_is_not_started_without_redis(self) -> None:
        assert make_affinity().start_keepalive("key-1") is None

    @pytest.mark.asyncio
    async def test_keepalive_keeps_the_session_ref_alive_through_a_long_call(self) -> None:
        redis: Final = FakeRedis()
        aff: Final = make_affinity(redis, MCP_BACKEND_SESSION_TTL_SECONDS="1")

        async def slow(_client: FakeClient) -> CallToolResult:
            await asyncio.sleep(1.6)
            return ok_result()

        result: Final = await call_sticky(aff, Factory([FakeClient(sid="live-1", body=slow)]))
        ref: Final = await aff.load_ref("key-1")
        assert result.isError is False
        assert ref is not None, (
            "the session ref expired during the call: the next call would open a new session and leak this one"
        )
        assert ref.sid == "live-1"

    @pytest.mark.asyncio
    async def test_the_keepalive_task_is_cancelled_when_the_call_ends(self) -> None:
        redis: Final = FakeRedis()
        aff: Final = make_affinity(redis)
        await call_sticky(aff, Factory([FakeClient()]))
        pending: Final = [
            t for t in asyncio.all_tasks() if t is not asyncio.current_task() and "_keepalive" in repr(t.get_coro())
        ]
        assert pending == []

    @pytest.mark.asyncio
    async def test_concurrent_same_key_calls_are_not_serialized(self) -> None:
        """Documented scope: session affinity, NOT concurrency safety. Two parallel calls on one key are
        neither serialized nor refused, and this test exists so that stays a deliberate, visible property
        rather than an assumption."""
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "live-1"}), None)
        in_flight: Final[list[str]] = []
        peak: Final[list[int]] = []

        async def slow(client: FakeClient) -> CallToolResult:
            in_flight.append(client._sid or "")
            peak.append(len(in_flight))
            await asyncio.sleep(0.2)
            in_flight.remove(client._sid or "")
            return ok_result()

        aff: Final = make_affinity(redis)
        factory: Final = Factory([FakeClient(sid="a", body=slow), FakeClient(sid="b", body=slow)])
        first, second = await asyncio.gather(call_sticky(aff, factory), call_sticky(aff, factory))
        assert first.isError is False
        assert second.isError is False
        assert max(peak) == 2, "calls were serialized: if that is now intended, the documented scope must change"


class TestLostRace:
    @pytest.mark.asyncio
    async def test_loser_terminates_its_own_session_and_resumes_the_winner(self) -> None:
        redis: Final = RaceRedis(json.dumps({"sid": "winner-1", "pv": "2026-06-18"}))
        resumed: Final = FakeClient()
        factory: Final = Factory([FakeClient(sid="loser-1"), resumed])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is False
        assert len(factory.created) == 2
        assert factory.terminated == [(SERVER_URL, "loser-1")]
        assert resumed.ctor_extra_headers["mcp-session-id"] == "winner-1"
        assert stored_sid(redis) == "winner-1"

    @pytest.mark.asyncio
    async def test_stale_then_lost_race_stops_at_two_clients(self) -> None:
        redis: Final = RaceRedis(json.dumps({"sid": "winner-1"}))
        redis.store["key-1"] = (json.dumps({"sid": "dead-1"}), None)
        factory: Final = Factory(
            [FakeClient(transport_error=RuntimeError("Session terminated")), FakeClient(sid="loser-1")]
        )
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is True
        assert LOST_RACE in text_of(result)
        assert len(factory.created) == 2
        assert factory.terminated == [(SERVER_URL, "loser-1")]

    @pytest.mark.asyncio
    async def test_a_winner_that_vanished_is_a_lost_race_not_a_store_outage(self, log_messages: list[str]) -> None:
        """SET NX lost and the winner's ref is unusable by the time it is read back: the store answered
        every request, so reporting it as down would send operators after a healthy Redis."""
        redis: Final = RaceRedis("not-json")
        factory: Final = Factory([FakeClient(sid="loser-1")])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is True
        assert LOST_RACE in text_of(result)
        assert STORE_DOWN not in text_of(result)
        assert factory.terminated == [(SERVER_URL, "loser-1")]
        assert any("lost the session registration race" in m for m in log_messages), log_messages

    @pytest.mark.asyncio
    async def test_no_third_client_when_the_winner_is_also_dead(self) -> None:
        redis: Final = RaceRedis(json.dumps({"sid": "winner-1"}))
        factory: Final = Factory(
            [FakeClient(sid="loser-1"), FakeClient(transport_error=RuntimeError("Session terminated"))]
        )
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is True
        assert "Session terminated" in text_of(result)
        assert len(factory.created) == 2


class TestNoIdentity:
    @pytest.mark.asyncio
    async def test_missing_auth_object_fails_closed(self) -> None:
        redis: Final = FakeRedis()
        factory: Final = Factory([FakeClient()])
        result: Final = await call_gated(make_affinity(redis), factory, server(name="Playwright"), None)
        assert result.isError is True
        assert NO_IDENTITY in text_of(result)
        assert factory.created == []
        assert redis.store == {}

    @pytest.mark.asyncio
    async def test_all_identity_fields_empty_fails_closed(self) -> None:
        redis: Final = FakeRedis()
        factory: Final = Factory([FakeClient()])
        result: Final = await call_gated(make_affinity(redis), factory, server(name="Playwright"), auth())
        assert result.isError is True
        assert NO_IDENTITY in text_of(result)
        assert factory.created == []
        assert redis.store == {}


class TestSessionMissing:
    @pytest.mark.asyncio
    async def test_aborts_before_the_tool_body_and_stores_nothing(self) -> None:
        redis: Final = FakeRedis()
        client: Final = FakeClient(sid=None)
        factory: Final = Factory([client])
        result: Final = await call_sticky(make_affinity(redis), factory)
        await settle()
        assert result.isError is True
        assert SESSION_MISSING in text_of(result)
        assert client.body_ran is False
        assert "key-1" not in redis.store
        assert factory.terminated == []


class TestSilentDegradationSpeaks:
    @pytest.mark.asyncio
    async def test_local_store_without_redis_logs_an_error_once(self, log_messages: list[str]) -> None:
        aff: Final = make_affinity()
        factory: Final = Factory([FakeClient(), FakeClient()])
        await call_sticky(aff, factory)
        await call_sticky(aff, factory)
        hits: Final = [m for m in log_messages if "PER WORKER" in m]
        assert len(hits) == 1, hits

    def test_invalid_env_values_warn_instead_of_defaulting_silently(self, log_messages: list[str]) -> None:
        env: Final = {"MCP_STATEFUL_SERVERS": "Playwright"}
        aff: Final = BackendSessionAffinity(env=env)
        env["MCP_BACKEND_SESSION_TTL_SECONDS"] = "thirty"
        assert aff.ttl_seconds() == 1800
        env["MCP_BACKEND_SESSION_TTL_SECONDS"] = "0"
        assert aff.ttl_seconds() == 1800
        warnings: Final = [m for m in log_messages if "mcp sticky" in m]
        assert len(warnings) == 2, warnings

    def test_valid_env_values_are_honored_without_warning(self, log_messages: list[str]) -> None:
        assert make_affinity(MCP_BACKEND_SESSION_TTL_SECONDS="60").ttl_seconds() == 60
        assert [m for m in log_messages if "invalid" in m] == []

    @pytest.mark.asyncio
    async def test_failed_session_delete_warns(self, log_messages: list[str]) -> None:
        loser: Final = MCPClient(server_url="http://127.0.0.1:1/mcp", timeout=2)
        await asyncio.wait_for(make_affinity(FakeRedis())._terminate(loser, "sid-1"), timeout=5)
        assert any("could not DELETE backend session" in m for m in log_messages)

    @pytest.mark.asyncio
    async def test_a_rejected_redis_login_names_the_operation_and_the_cause(self, log_messages: list[str]) -> None:
        redis: Final = FakeRedis()

        async def wrongpass(name: str) -> bytes | None:
            raise AuthenticationError("WRONGPASS invalid username-password pair or user is disabled.")

        redis.get = wrongpass
        result: Final = await call_sticky(make_affinity(redis), Factory([FakeClient()]))
        assert STORE_DOWN in text_of(result)
        assert any("redis get" in m and "AuthenticationError" in m and "WRONGPASS" in m for m in log_messages), (
            log_messages
        )

    @pytest.mark.asyncio
    async def test_a_redis_client_that_cannot_be_built_is_logged(self, log_messages: list[str]) -> None:
        result: Final = await call_sticky(failing_redis_affinity(), Factory([FakeClient()]))
        assert STORE_DOWN in text_of(result)
        assert any("cannot reach redis" in m for m in log_messages), log_messages

    @pytest.mark.asyncio
    async def test_a_programming_error_is_not_disguised_as_a_store_outage(self) -> None:
        redis: Final = FakeRedis()

        async def broken(name: str) -> bytes | None:
            raise TypeError("unexpected keyword")

        redis.get = broken
        with pytest.raises(TypeError, match="unexpected keyword"):
            await call_sticky(make_affinity(redis), Factory([FakeClient()]))


class TestRedisConfiguration:
    """The store must reach the same Redis the proxy is configured for, not a host/port/password subset."""

    def test_redis_url_carries_tls_username_host_and_port(self) -> None:
        client: Final = redis_from_env({"REDIS_URL": "rediss://alice:secret@redis.example:6380/0"})
        pool: Final = client.connection_pool
        assert pool.connection_class is SSLConnection
        assert pool.connection_kwargs["username"] == "alice"
        assert pool.connection_kwargs["host"] == "redis.example"
        assert pool.connection_kwargs["port"] == 6380

    def test_redis_ssl_and_username_are_honoured_with_a_host(self) -> None:
        client: Final = redis_from_env(
            {
                "REDIS_HOST": "redis.example",
                "REDIS_PORT": "6380",
                "REDIS_SSL": "true",
                "REDIS_USERNAME": "alice",
                "REDIS_PASSWORD": "secret",
            }
        )
        pool: Final = client.connection_pool
        assert pool.connection_class is SSLConnection
        assert pool.connection_kwargs["username"] == "alice"
        assert pool.connection_kwargs["port"] == 6380

    @pytest.mark.asyncio
    async def test_redis_url_alone_selects_the_shared_store(self) -> None:
        redis: Final = FakeRedis()
        aff: Final = BackendSessionAffinity(
            env={"MCP_STATEFUL_SERVERS": "Playwright", "REDIS_URL": "redis://redis.test:6379"},
            redis_factory=lambda _env: redis,
        )
        await call_sticky(aff, Factory([FakeClient(sid="shared-1")]))
        assert stored_sid(redis) == "shared-1"


def backend_server(backend: StatefulBackend, name: str = "Playwright", **overrides: object) -> MCPServer:
    return server(name=name, url=backend.url, **overrides)


async def call_manager(
    manager: MCPServerManager,
    srv: MCPServer,
    user: UserAPIKeyAuth | None,
    arguments: dict[str, str] | None = None,
    tasks: list | None = None,
    proxy_logging_obj: ProxyLogging | None = None,
) -> CallToolResult:
    return await asyncio.wait_for(
        manager._call_regular_mcp_tool(
            mcp_server=srv,
            original_tool_name="navigate",
            arguments=arguments or {},
            tasks=tasks if tasks is not None else [],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=proxy_logging_obj,
            user_api_key_auth=user,
        ),
        timeout=15,
    )


class TestDuringHooks:
    @pytest.mark.asyncio
    async def test_hook_result_is_not_mistaken_for_the_tool_result(self) -> None:
        async def hook() -> str:
            return "hook-return"

        with running_backend() as backend:
            manager: Final = MCPServerManager(backend_session_affinity=make_affinity())
            result: Final = await call_manager(
                manager,
                backend_server(backend),
                auth(token="t"),
                arguments={"url": "https://a.test"},
                tasks=[asyncio.create_task(hook())],
                proxy_logging_obj=MagicMock(spec=ProxyLogging),
            )
        assert result.isError is False
        assert text_of(result) == "https://a.test"

    @pytest.mark.asyncio
    async def test_guardrail_exception_propagates_on_an_aborted_path(self) -> None:
        async def hook() -> None:
            raise GuardrailRaisedException(guardrail_name="probe", message="blocked")

        manager: Final = MCPServerManager(backend_session_affinity=failing_redis_affinity())
        with pytest.raises(GuardrailRaisedException):
            await call_manager(
                manager,
                server(name="Playwright"),
                auth(token="t"),
                tasks=[asyncio.create_task(hook())],
                proxy_logging_obj=MagicMock(spec=ProxyLogging),
            )


def real_client_factory(backend: StatefulBackend) -> Callable[..., Awaitable[MCPClient]]:
    async def create(*, extra_headers: dict[str, str] | None) -> MCPClient:
        return MCPClient(server_url=backend.url, timeout=5, extra_headers=extra_headers)

    return create


async def call_real(aff: BackendSessionAffinity, backend: StatefulBackend, url: str = "") -> CallToolResult:
    return await asyncio.wait_for(
        aff.call_with_key(
            StickyCall(
                key="key-1",
                extra_headers=None,
                params=CallToolRequestParams(name="navigate", arguments={"url": url} if url else {}),
                host_progress_callback=None,
                create_client=real_client_factory(backend),
            )
        ),
        timeout=15,
    )


class TestRealPath:
    """Real MCPClient, real MCP SDK transport, real HTTP backend: the contracts no stub can prove."""

    @pytest.mark.asyncio
    async def test_the_second_call_resumes_the_first_calls_backend_session(self) -> None:
        redis: Final = FakeRedis()
        aff: Final = make_affinity(redis)
        with running_backend() as backend:
            navigated: Final = await call_real(aff, backend, "https://a.test")
            snapshot: Final = await call_real(aff, backend)
        assert navigated.isError is False
        assert text_of(snapshot) == "https://a.test"
        assert backend.rpc_methods().count("initialize") == 1
        assert backend.deletes() == []
        assert stored_sid(redis) == "sid-1"

    @pytest.mark.asyncio
    async def test_a_reaped_session_recovers_once_on_a_fresh_session(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "sid-reaped", "pv": "2025-06-18"}), None)
        with running_backend() as backend:
            result: Final = await call_real(make_affinity(redis), backend, "https://a.test")
        assert result.isError is False, text_of(result)
        assert text_of(result) == "https://a.test"
        assert backend.rpc_methods("sid-reaped") == ["tools/call"]
        assert backend.rpc_methods().count("initialize") == 1
        assert stored_sid(redis) == "sid-1"

    @pytest.mark.asyncio
    async def test_a_session_the_sdk_reports_as_terminated_recovers_once(self) -> None:
        redis: Final = FakeRedis()
        redis.store["key-1"] = (json.dumps({"sid": "sid-reaped"}), None)
        with running_backend() as backend:
            backend.unknown_session_status = 404
            result: Final = await call_real(make_affinity(redis), backend, "https://a.test")
        assert result.isError is False, text_of(result)
        assert backend.rpc_methods("sid-reaped") == ["tools/call"]
        assert stored_sid(redis) == "sid-1"

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, PROTOCOL_VERSION_REJECTION),
            (400, {"error": {"message": "unknown session"}}),
            (400, {**SUPERGATEWAY_REJECTION, "pad": "x" * 200_000}),
            (500, SUPERGATEWAY_REJECTION),
        ],
        ids=["live-400", "no-envelope", "oversized-body", "server-error"],
    )
    @pytest.mark.asyncio
    async def test_a_live_sessions_error_is_answered_and_never_replayed(self, status: int, body: dict) -> None:
        redis: Final = FakeRedis()
        aff: Final = make_affinity(redis)
        with running_backend() as backend:
            await call_real(aff, backend, "https://a.test")
            backend.tools_call_status = status
            backend.tools_call_body = body
            result: Final = await call_real(aff, backend, "https://b.test")
        assert result.isError is True
        assert backend.rpc_methods("sid-1").count("tools/call") == 2
        assert backend.rpc_methods().count("initialize") == 1
        assert stored_sid(redis) == "sid-1"


class TestManagerDispatch:
    @pytest.mark.asyncio
    async def test_call_tool_threads_caller_identity_into_the_sticky_path(self) -> None:
        redis: Final = FakeRedis()
        with running_backend() as backend:
            manager: Final = MCPServerManager(backend_session_affinity=make_affinity(redis))
            srv: Final = backend_server(backend)
            manager.registry[srv.server_id] = srv
            manager.tool_name_to_mcp_server_name_mapping["navigate"] = srv.name
            first: Final = await manager.call_tool(
                server_name=srv.name,
                name="navigate",
                arguments={"url": "https://a.test"},
                user_api_key_auth=auth(token="t"),
            )
            second: Final = await manager.call_tool(
                server_name=srv.name, name="navigate", arguments={}, user_api_key_auth=auth(token="t")
            )
            other_caller: Final = await manager.call_tool(
                server_name=srv.name, name="navigate", arguments={}, user_api_key_auth=auth(token="someone-else")
            )
        assert first.isError is False, text_of(first)
        assert text_of(second) == "https://a.test"
        assert text_of(other_caller) == "about:blank"
        assert backend.deletes() == []
        assert len(redis.store) == 2
