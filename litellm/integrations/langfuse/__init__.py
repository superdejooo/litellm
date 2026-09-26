import os
import urllib.parse
from typing import TYPE_CHECKING, Final

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.types.prompts.init_prompts import SupportedPromptIntegrations

from .langfuse_prompt_management import LangfusePromptManagement, LangfusePromptOnlyManagement

if TYPE_CHECKING:
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

# Virtual registration: a real CustomPromptManagement base would put CustomLogger and
# PromptManagementBase in an order that conflicts with LangfusePromptManagement's MRO.
CustomPromptManagement.register(LangfusePromptManagement)


def _resolve_secret_reference(value: object) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    if not isinstance(value, str):
        return None
    if value.startswith("os.environ/"):
        return get_secret_str(value)
    return value


_PUBLIC_KEY_PARAMS: Final = ("langfuse_public_key",)
_SECRET_KEY_PARAMS: Final = ("langfuse_secret_key", "langfuse_secret")
_HOST_PARAMS: Final = ("langfuse_host",)


def _named_reference(litellm_params: "PromptLiteLLMParams", param_names: tuple[str, ...]) -> object:
    return next((value for name in param_names if (value := getattr(litellm_params, name, None)) is not None), None)


def _resolve_credential(
    litellm_params: "PromptLiteLLMParams", param_names: tuple[str, ...], env_var: str
) -> str | None:
    """Resolve the credential the prompt entry names, falling back to env_var only when the entry names none."""
    reference: Final = _named_reference(litellm_params, param_names)
    if reference is None:
        return os.getenv(env_var) or None
    return _resolve_secret_reference(reference) or None


def _missing_credential_label(name: str, reference: object) -> str:
    if reference is None:
        return name
    if isinstance(reference, str) and reference.startswith("os.environ/"):
        return f"{name} ({reference} is unset or empty)"
    return f"{name} (set but empty)"


def _strict_url_encode(self, url: str) -> str:
    return urllib.parse.quote(url, safe="")


def _encode_slashes_in_langfuse_prompt_names() -> None:
    from langfuse.client import Langfuse

    # langfuse SDK 2.x quotes prompt names with urllib's default safe="/", so a name like
    # "events/summary" reaches the API as a sub-path and 404s (langfuse/langfuse#8006).
    if not hasattr(Langfuse, "_url_encode"):
        raise ValueError(
            "langfuse.client.Langfuse._url_encode no longer exists; re-check prompt names containing '/' "
            "against this langfuse SDK before enabling the langfuse prompt integration"
        )
    Langfuse._url_encode = _strict_url_encode


def prompt_initializer(litellm_params: "PromptLiteLLMParams", prompt_spec: "PromptSpec") -> "CustomPromptManagement":
    """Build a Langfuse prompt manager for a `prompts:` entry, failing at boot when credentials are missing."""
    public_key: Final = _resolve_credential(litellm_params, _PUBLIC_KEY_PARAMS, "LANGFUSE_PUBLIC_KEY")
    secret_key: Final = _resolve_credential(litellm_params, _SECRET_KEY_PARAMS, "LANGFUSE_SECRET_KEY")
    host: Final = _resolve_credential(litellm_params, _HOST_PARAMS, "LANGFUSE_HOST")
    missing: Final = tuple(
        _missing_credential_label(name, _named_reference(litellm_params, param_names))
        for name, param_names, value in (
            ("langfuse_public_key", _PUBLIC_KEY_PARAMS, public_key),
            ("langfuse_secret_key", _SECRET_KEY_PARAMS, secret_key),
            ("langfuse_host", _HOST_PARAMS, host),
        )
        if not value
    )
    if missing:
        raise ValueError(
            f"Langfuse prompt integration for prompt '{prompt_spec.prompt_id}' is missing credentials: {', '.join(missing)}. "
            "Set them under the prompt's litellm_params or via LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and "
            "LANGFUSE_HOST. A credential named in litellm_params never falls back to those variables."
        )

    _encode_slashes_in_langfuse_prompt_names()

    return LangfusePromptOnlyManagement(
        langfuse_public_key=public_key,
        langfuse_secret=secret_key,
        langfuse_host=host,
    )


prompt_initializer_registry: Final = {
    SupportedPromptIntegrations.LANGFUSE.value: prompt_initializer,
}
