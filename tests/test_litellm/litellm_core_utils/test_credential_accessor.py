from pathlib import Path

import pytest

from litellm.litellm_core_utils import credential_accessor
from litellm.litellm_core_utils.credential_accessor import (
    FILE_CREDENTIAL_SCHEME,
    is_file_credential_ref,
    resolve_file_credential,
    was_resolved_from_file_credential,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(credential_accessor, "monotonic", fake)
    monkeypatch.setattr(credential_accessor, "FILE_CREDENTIAL_TTL_SECONDS", 5.0)
    return fake


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    return tmp_path / "token"


def _ref(path: Path) -> str:
    return FILE_CREDENTIAL_SCHEME + str(path)


def test_resolve_file_credential_strips_surrounding_whitespace(clock: FakeClock, token_file: Path) -> None:
    token_file.write_text("  sk-ant-oat01-FROM-FILE\n", encoding="utf-8")

    assert resolve_file_credential(_ref(token_file)) == "sk-ant-oat01-FROM-FILE"


def test_resolve_file_credential_serves_the_cached_value_until_the_ttl_lapses(
    clock: FakeClock, token_file: Path
) -> None:
    token_file.write_text("key-before-swap", encoding="utf-8")
    assert resolve_file_credential(_ref(token_file)) == "key-before-swap"

    token_file.write_text("key-after-swap", encoding="utf-8")
    clock.now += 4.9
    assert resolve_file_credential(_ref(token_file)) == "key-before-swap"

    clock.now += 0.2
    assert resolve_file_credential(_ref(token_file)) == "key-after-swap"


def test_resolve_file_credential_raises_for_a_missing_file(clock: FakeClock, token_file: Path) -> None:
    with pytest.raises(ValueError, match="unreadable.*no env fallback"):
        resolve_file_credential(_ref(token_file))


@pytest.mark.parametrize("content", ["", "   \n\t"])
def test_resolve_file_credential_raises_for_an_empty_file(clock: FakeClock, token_file: Path, content: str) -> None:
    token_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="is empty.*no env fallback"):
        resolve_file_credential(_ref(token_file))


def test_resolve_file_credential_raises_value_error_for_an_undecodable_file(clock: FakeClock, token_file: Path) -> None:
    token_file.write_bytes(b"\xff\xfe\x00\x01")

    with pytest.raises(ValueError, match="unreadable.*no env fallback"):
        resolve_file_credential(_ref(token_file))


def test_resolve_file_credential_does_not_cache_a_failed_read(clock: FakeClock, token_file: Path) -> None:
    with pytest.raises(ValueError):
        resolve_file_credential(_ref(token_file))

    token_file.write_text("key-written-mid-swap", encoding="utf-8")

    assert resolve_file_credential(_ref(token_file)) == "key-written-mid-swap"


def test_resolve_file_credential_raises_once_the_cached_file_disappears(clock: FakeClock, token_file: Path) -> None:
    token_file.write_text("key-that-goes-away", encoding="utf-8")
    assert resolve_file_credential(_ref(token_file)) == "key-that-goes-away"

    token_file.unlink()
    clock.now += 6

    with pytest.raises(ValueError, match="unreadable"):
        resolve_file_credential(_ref(token_file))


def test_was_resolved_from_file_credential_remembers_every_value_the_file_held(
    clock: FakeClock, token_file: Path
) -> None:
    token_file.write_text("key-a", encoding="utf-8")
    resolve_file_credential(_ref(token_file))
    token_file.write_text("key-b", encoding="utf-8")
    clock.now += 6
    resolve_file_credential(_ref(token_file))

    assert was_resolved_from_file_credential(_ref(token_file), "key-a")
    assert was_resolved_from_file_credential(_ref(token_file), "key-b")
    assert not was_resolved_from_file_credential(_ref(token_file), "sk-explicit-deployment-key")
    assert not was_resolved_from_file_credential(_ref(token_file.with_name("other")), "key-a")
    assert not was_resolved_from_file_credential(_ref(token_file), {"not": "a string"})


@pytest.mark.parametrize(
    "value, expected",
    [
        ("file:///data/claude-auth/token", True),
        ("os.environ/ANTHROPIC_API_KEY", False),
        ("sk-literal", False),
        (None, False),
        (42, False),
    ],
)
def test_is_file_credential_ref_only_claims_file_uris(value: object, expected: bool) -> None:
    assert is_file_credential_ref(value) is expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, 5.0),
        ("", 5.0),
        ("2.5", 2.5),
        ("3600", 3600.0),
        ("not-a-number", 5.0),
        ("inf", 5.0),
        ("nan", 5.0),
        ("0", 5.0),
        ("-1", 5.0),
        ("3601", 5.0),
    ],
)
def test_file_credential_ttl_falls_back_to_the_default_outside_a_sane_range(raw: str | None, expected: float) -> None:
    assert credential_accessor._parse_file_credential_ttl_seconds(raw) == expected
