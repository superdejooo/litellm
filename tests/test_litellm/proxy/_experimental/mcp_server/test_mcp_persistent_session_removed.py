"""Guards for two designs that were built, shipped and removed: do not bring either back.

A persistent live MCPClient (a connect()/close() API held across requests) pinned proxy workers at 100%
CPU through zombie anyio TaskGroups. A client-side per-key lock around sticky backend sessions could not be
made safe without backend fencing, so backend session affinity gives affinity, NOT concurrency safety.
"""

from pathlib import Path
from typing import Final

import pytest

from litellm.experimental_mcp_client.client import MCPClient

REPO_ROOT: Final = Path(__file__).resolve().parents[5]
MCP_SERVER_DIR: Final = REPO_ROOT / "litellm" / "proxy" / "_experimental" / "mcp_server"
MANAGER_FILE: Final = MCP_SERVER_DIR / "mcp_server_manager.py"
GUARDED_FILES: Final = (
    MANAGER_FILE,
    MCP_SERVER_DIR / "backend_session_affinity.py",
    REPO_ROOT / "litellm" / "experimental_mcp_client" / "client.py",
)


def test_the_guarded_files_exist() -> None:
    for path in GUARDED_FILES:
        assert path.is_file(), f"{path} moved: repoint these guards instead of letting them pass vacuously"


class TestManagerPatternsRemoved:
    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
    def test_no_session_cache_attribute(self, path: Path) -> None:
        assert "_session_cache" not in path.read_text(), (
            "_session_cache is back. Holding MCPClient instances across calls is what accumulated zombie "
            "TaskGroups; a name containing it is refused even for an unrelated cache."
        )

    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
    def test_no_get_or_create_persistent_client(self, path: Path) -> None:
        assert "_get_or_create_persistent_client" not in path.read_text(), (
            "_get_or_create_persistent_client is back: it called MCPClient.connect(), the removed lifecycle."
        )

    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
    def test_no_evict_session_method(self, path: Path) -> None:
        assert "_evict_session" not in path.read_text(), "_evict_session is back: it called MCPClient.close()."

    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
    def test_no_client_connect_close_calls(self, path: Path) -> None:
        content: Final = path.read_text()
        for needle in ("client.connect()", "client.close()", "fresh.close()"):
            assert needle not in content, f"Found {needle!r} in {path.name}: the persistent client API is back"

    def test_mcp_client_has_no_connect_close_api(self) -> None:
        assert not hasattr(MCPClient, "connect"), "MCPClient.connect() is back: that is the persistent session"
        assert not hasattr(MCPClient, "close"), "MCPClient.close() is back: that is the persistent session"

    @pytest.mark.parametrize("path", GUARDED_FILES, ids=lambda p: p.name)
    def test_no_anyio_cancelscope_shield_wraps(self, path: Path) -> None:
        assert "anyio.CancelScope(shield=True)" not in path.read_text(), (
            "anyio.CancelScope(shield=True) around a client cleanup invalidated the MCP SDK TaskGroup __aexit__ "
            "and produced zombie TaskGroups."
        )


class TestGetToolsFromServerOneShot:
    def _body(self) -> str:
        content: Final = MANAGER_FILE.read_text()
        marker: Final = "async def _get_tools_from_server("
        assert marker in content, "Manager no longer defines _get_tools_from_server"
        start: Final = content.index(marker)
        next_def: Final = content.find("\n    async def ", start + len(marker))
        return content[start:next_def] if next_def != -1 else content[start:]

    def test_direct_create_mcp_client_call(self) -> None:
        assert "await self._create_mcp_client(" in self._body(), (
            "_get_tools_from_server no longer calls _create_mcp_client directly: that is the one-shot path "
            "that replaced the persistent-try-then-fallback block."
        )

    def test_no_persistent_try_fallback_in_body(self) -> None:
        assert "_get_or_create_persistent_client" not in self._body()


def test_no_lock_layer_survives() -> None:
    """The lease/single-flight layer was removed deliberately. If any of these come back, the removal is
    being undone; test_concurrent_same_key_calls_are_not_serialized pins the resulting behaviour."""
    for path in GUARDED_FILES:
        source = path.read_text()
        for gone in (
            "call-slot",
            "_acquire_sticky_call_slot",
            "_release_sticky_call_slot",
            "_renew_sticky_call_slot",
            "_STICKY_LOCK_TTL_SECONDS",
            "_STICKY_BUSY_MARKER",
            "MCP_STICKY_LOCK_WAIT_SECONDS",
        ):
            assert gone not in source, f"{gone} is back in {path.name}"
