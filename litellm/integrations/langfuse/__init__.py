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


def _resolve_credential(
    litellm_params: "PromptLiteLLMParams", param_names: tuple[str, ...], env_var: str
) -> str | None:
    configured: Final = (_resolve_secret_reference(getattr(litellm_params, name, None)) for name in param_names)
    return next((value for value in configured if value), None) or os.getenv(env_var) or None


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
    public_key: Final = _resolve_credential(litellm_params, ("langfuse_public_key",), "LANGFUSE_PUBLIC_KEY")
    secret_key: Final = _resolve_credential(
        litellm_params, ("langfuse_secret_key", "langfuse_secret"), "LANGFUSE_SECRET_KEY"
    )
    host: Final = _resolve_credential(litellm_params, ("langfuse_host",), "LANGFUSE_HOST")
    missing: Final = tuple(
        name
        for name, value in (
            ("langfuse_public_key", public_key),
            ("langfuse_secret_key", secret_key),
            ("langfuse_host", host),
        )
        if not value
    )
    if missing:
        raise ValueError(
            f"Langfuse prompt integration for prompt '{prompt_spec.prompt_id}' is missing credentials: {', '.join(missing)}. "
            "Set them under the prompt's litellm_params or via LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and "
            "LANGFUSE_HOST."
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
