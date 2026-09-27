"""Utils for accessing credentials."""

import os
from collections.abc import Mapping
from time import monotonic
from types import MappingProxyType
from typing import Final, NamedTuple

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import CredentialItem

FILE_CREDENTIAL_SCHEME: Final = "file://"
_DEFAULT_FILE_CREDENTIAL_TTL_SECONDS: Final = 5.0
_MAX_FILE_CREDENTIAL_TTL_SECONDS: Final = 3600.0


def _as_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_file_credential_ttl_seconds(raw: str | None) -> float:
    """Called once, at import, so an invalid value is reported exactly once."""
    if not raw:
        return _DEFAULT_FILE_CREDENTIAL_TTL_SECONDS
    ttl: Final = _as_float(raw)
    # float() accepts "inf", which would cache a value no file swap ever replaces; NaN fails this check too
    if ttl is not None and 0 < ttl <= _MAX_FILE_CREDENTIAL_TTL_SECONDS:
        return ttl
    verbose_logger.warning(
        "FILE_CREDENTIAL_TTL_SECONDS=%r is not a number of seconds in (0, %s]; using the default %ss",
        raw,
        _MAX_FILE_CREDENTIAL_TTL_SECONDS,
        _DEFAULT_FILE_CREDENTIAL_TTL_SECONDS,
    )
    return _DEFAULT_FILE_CREDENTIAL_TTL_SECONDS


FILE_CREDENTIAL_TTL_SECONDS: Final = _parse_file_credential_ttl_seconds(os.getenv("FILE_CREDENTIAL_TTL_SECONDS"))


def _parse_file_credential_allowed_dirs(raw: str | None) -> tuple[str, ...]:
    """Called once, at import. Unset, empty or relative entries allow nothing, so no ``file://`` ref resolves."""
    entries: Final = tuple(entry for entry in (raw or "").split(os.pathsep) if entry)
    relative: Final = tuple(entry for entry in entries if not os.path.isabs(entry))
    if relative:
        verbose_logger.error("FILE_CREDENTIAL_ALLOWED_DIRS ignores entries that are not absolute paths: %r", relative)
    return tuple(os.path.realpath(entry) for entry in entries if os.path.isabs(entry))


FILE_CREDENTIAL_ALLOWED_DIRS: Final = _parse_file_credential_allowed_dirs(os.getenv("FILE_CREDENTIAL_ALLOWED_DIRS"))


class _CachedFileCredential(NamedTuple):
    expires_at: float
    value: str


_file_credential_cache: Final[dict[str, _CachedFileCredential]] = {}  # mutable-ok: process-wide TTL cache keyed by path
_file_credential_history: Final[dict[str, frozenset[str]]] = {}  # mutable-ok: every value each path has resolved to


def is_file_credential_ref(value: object) -> bool:
    return isinstance(value, str) and value.startswith(FILE_CREDENTIAL_SCHEME)


def _is_inside_an_allowed_dir(real_path: str) -> bool:
    return any(os.path.commonpath((real_path, root)) == root for root in FILE_CREDENTIAL_ALLOWED_DIRS)


_UNREADABLE_FILE_CREDENTIAL_MESSAGE: Final = (
    "file credential unreadable: the credential file must exist and be readable; there is no env fallback"
)


def resolve_file_credential(ref: str) -> str:
    """Read a ``file:///abs/path`` credential, cached for FILE_CREDENTIAL_TTL_SECONDS so a swapped file is picked up.

    Only files inside FILE_CREDENTIAL_ALLOWED_DIRS (after resolving symlinks and ``..``) are read.
    Raises ValueError when the file is outside them, missing, unreadable or empty; the ref itself is never returned.
    """
    path: Final = ref.removeprefix(FILE_CREDENTIAL_SCHEME)
    now: Final = monotonic()
    cached: Final = _file_credential_cache.get(path)
    if cached is not None and now < cached.expires_at:
        return cached.value
    real_path: Final = os.path.realpath(path)
    if not _is_inside_an_allowed_dir(real_path):
        verbose_logger.error(
            "file credential %s (resolves to %s) is outside FILE_CREDENTIAL_ALLOWED_DIRS=%r; refusing to read it",
            path,
            real_path,
            FILE_CREDENTIAL_ALLOWED_DIRS,
        )
        # Same message as an unreadable file, so a refused path cannot be told apart from a missing one
        raise ValueError(_UNREADABLE_FILE_CREDENTIAL_MESSAGE)
    try:
        with open(real_path, encoding="utf-8") as credential_file:
            value: Final = credential_file.read().strip()
    except (OSError, UnicodeDecodeError) as e:
        verbose_logger.error("file credential %s unreadable: %s", path, e)
        # The message reaches API callers, so the host path and OS error stay in the log above
        raise ValueError(_UNREADABLE_FILE_CREDENTIAL_MESSAGE) from None
    if not value:
        verbose_logger.error("file credential %s is empty", path)
        raise ValueError("file credential is empty: refusing to send an empty credential; there is no env fallback")
    _file_credential_cache[path] = _CachedFileCredential(expires_at=now + FILE_CREDENTIAL_TTL_SECONDS, value=value)
    _file_credential_history[path] = _file_credential_history.get(path, frozenset()) | frozenset((value,))
    return value


def was_resolved_from_file_credential(ref: str, value: object) -> bool:
    """Whether ``value`` is something ``ref`` resolved to earlier, i.e. a copy of it left over from a previous load."""
    return isinstance(value, str) and value in _file_credential_history.get(
        ref.removeprefix(FILE_CREDENTIAL_SCHEME), frozenset()
    )


def _should_inject(credential_value: object, current: object) -> bool:
    return current is None or (
        isinstance(credential_value, str)
        and is_file_credential_ref(credential_value)
        and (current == credential_value or was_resolved_from_file_credential(credential_value, current))
    )


def materialize_credential_value(credential_value: object) -> object:
    """``credential_value`` with a ``file://`` ref replaced by the file's content; other values pass through."""
    if isinstance(credential_value, str) and is_file_credential_ref(credential_value):
        return resolve_file_credential(credential_value)
    return credential_value


def credential_values_to_inject(
    credential_values: Mapping[str, object], request: Mapping[str, object]
) -> Mapping[str, object]:
    """The credential values to write into ``request``: keys it lacks or holds as None, and stale ``file://`` copies."""
    return MappingProxyType(
        {
            key: materialize_credential_value(value)
            for key, value in credential_values.items()
            if _should_inject(value, request.get(key))
        }
    )


class CredentialAccessor:
    @staticmethod
    def find_credential(credential_name: str) -> CredentialItem | None:
        return next(
            (credential for credential in litellm.credential_list if credential.credential_name == credential_name),
            None,
        )

    @staticmethod
    def get_credential_values(credential_name: str) -> dict:
        """A copy of the credential's values with every ``file://`` ref read from its file.

        Raises ValueError when a referenced file is missing, unreadable or empty.
        """
        credential: Final = CredentialAccessor.find_credential(credential_name)
        if credential is None:
            return {}
        return {key: materialize_credential_value(value) for key, value in credential.credential_values.items()}

    @staticmethod
    def upsert_credentials(credentials: list[CredentialItem]):
        """Add a credential to the list of credentials."""

        credential_names: Final = [cred.credential_name for cred in litellm.credential_list]

        for credential in credentials:
            if credential.credential_name in credential_names:
                # Find and replace the existing credential in the list
                for i, existing_cred in enumerate(litellm.credential_list):
                    if existing_cred.credential_name == credential.credential_name:
                        litellm.credential_list[i] = credential
                        break
            else:
                litellm.credential_list.append(credential)
