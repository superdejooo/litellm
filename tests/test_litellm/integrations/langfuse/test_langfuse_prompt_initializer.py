import subprocess
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.integrations.langfuse import (
    _encode_slashes_in_langfuse_prompt_names,
    prompt_initializer,
    prompt_initializer_registry,
)
from litellm.integrations.langfuse.langfuse import LangFuseLogger
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    LangfusePromptOnlyManagement,
)
from litellm.proxy.prompts.prompt_registry import (
    InMemoryPromptRegistry,
    get_prompt_initializer_from_integrations,
)
from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

LANGFUSE_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


@pytest.fixture(autouse=True)
def _no_langfuse_env(monkeypatch):
    for env_var in LANGFUSE_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture
def client_init():
    with (
        patch("litellm.integrations.langfuse._encode_slashes_in_langfuse_prompt_names"),
        patch("litellm.integrations.langfuse.langfuse_prompt_management.langfuse_client_init") as mock_client_init,
    ):
        yield mock_client_init


def _spec(prompt_id: str = "events/description-create", **litellm_params) -> PromptSpec:
    return PromptSpec(
        prompt_id=prompt_id,
        litellm_params=PromptLiteLLMParams(prompt_id=prompt_id, prompt_integration="langfuse", **litellm_params),
    )


def test_registry_discovers_the_langfuse_prompt_initializer():
    assert get_prompt_initializer_from_integrations()["langfuse"] is prompt_initializer
    assert prompt_initializer_registry["langfuse"] is prompt_initializer


def test_langfuse_prompt_management_is_a_virtual_custom_prompt_management():
    assert issubclass(LangfusePromptManagement, CustomPromptManagement)
    assert issubclass(LangfusePromptOnlyManagement, CustomPromptManagement)


def test_initializer_builds_a_prompt_only_manager_from_inline_credentials(client_init):
    spec = _spec(
        langfuse_public_key="pk-inline",
        langfuse_secret_key="sk-inline",
        langfuse_host="https://langfuse.example",
    )

    manager = prompt_initializer(spec.litellm_params, spec)

    assert type(manager) is LangfusePromptOnlyManagement
    assert client_init.call_args.kwargs == {
        "langfuse_public_key": "pk-inline",
        "langfuse_secret": "sk-inline",
        "langfuse_host": "https://langfuse.example",
        "flush_interval": 1,
    }


def test_initializer_resolves_os_environ_references(client_init, monkeypatch):
    monkeypatch.setenv("CUSTOM_LF_PUBLIC", "pk-env")
    monkeypatch.setenv("CUSTOM_LF_SECRET", "sk-env")
    monkeypatch.setenv("CUSTOM_LF_HOST", "https://langfuse.env")
    spec = _spec(
        langfuse_public_key="os.environ/CUSTOM_LF_PUBLIC",
        langfuse_secret="os.environ/CUSTOM_LF_SECRET",
        langfuse_host="os.environ/CUSTOM_LF_HOST",
    )

    prompt_initializer(spec.litellm_params, spec)

    assert client_init.call_args.kwargs["langfuse_public_key"] == "pk-env"
    assert client_init.call_args.kwargs["langfuse_secret"] == "sk-env"
    assert client_init.call_args.kwargs["langfuse_host"] == "https://langfuse.env"


def test_initializer_falls_back_to_langfuse_env_vars(client_init, monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-default")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-default")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.default")
    spec = _spec()

    prompt_initializer(spec.litellm_params, spec)

    assert client_init.call_args.kwargs["langfuse_host"] == "https://langfuse.default"


def test_initializer_fails_at_boot_when_credentials_are_missing(client_init):
    spec = _spec(langfuse_public_key="pk-inline", langfuse_host="os.environ/UNSET_LANGFUSE_HOST")

    with pytest.raises(ValueError) as exc_info:
        prompt_initializer(spec.litellm_params, spec)

    assert (
        "missing credentials: langfuse_secret_key, langfuse_host (os.environ/UNSET_LANGFUSE_HOST is unset or empty)."
        in str(exc_info.value)
    )
    assert "events/description-create" in str(exc_info.value)
    client_init.assert_not_called()


@pytest.mark.parametrize(
    ("named_credential", "expected_missing"),
    [
        (
            {"langfuse_public_key": "os.environ/LANGFUSE_PK_TEAM_B"},
            "langfuse_public_key (os.environ/LANGFUSE_PK_TEAM_B is unset or empty)",
        ),
        (
            {"langfuse_secret": "os.environ/LANGFUSE_SK_TEAM_B"},
            "langfuse_secret_key (os.environ/LANGFUSE_SK_TEAM_B is unset or empty)",
        ),
        ({"langfuse_host": ""}, "langfuse_host (set but empty)"),
    ],
)
def test_named_credential_that_does_not_resolve_never_falls_back_to_the_default_project(
    client_init, monkeypatch, named_credential, expected_missing
):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-default")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-default")
    monkeypatch.setenv("LANGFUSE_HOST", "https://langfuse.default")
    spec = _spec(**named_credential)

    with pytest.raises(ValueError) as exc_info:
        prompt_initializer(spec.litellm_params, spec)

    assert f"missing credentials: {expected_missing}." in str(exc_info.value)
    client_init.assert_not_called()


def test_prompt_registry_accepts_langfuse_prompt_specs(client_init):
    spec = _spec(
        langfuse_public_key="pk-inline",
        langfuse_secret_key="sk-inline",
        langfuse_host="https://langfuse.example",
    )

    parsed_prompt, callback = InMemoryPromptRegistry()._build_prompt_callback(prompt=spec)

    assert parsed_prompt.prompt_id == "events/description-create"
    assert type(callback) is LangfusePromptOnlyManagement


@pytest.mark.asyncio
async def test_prompt_only_manager_emits_no_native_langfuse_traces(client_init):
    manager = LangfusePromptOnlyManagement()
    kwargs = {"standard_callback_dynamic_params": {}, "standard_logging_object": None}

    with patch.object(LangFuseLogger, "log_event_on_langfuse") as log_event:
        manager.log_success_event(kwargs, MagicMock(), 1, 2)
        manager.log_failure_event(kwargs, None, 1, 2)
        await manager.async_log_success_event(kwargs, MagicMock(), 1, 2)
        await manager.async_log_failure_event(kwargs, None, 1, 2)

    log_event.assert_not_called()


@pytest.mark.asyncio
async def test_callback_langfuse_prompt_management_still_emits_native_traces(client_init):
    manager = LangfusePromptManagement()

    with (
        patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseHandler.get_langfuse_logger_for_request",
            return_value=manager,
        ),
        patch.object(LangFuseLogger, "log_event_on_langfuse") as log_event,
    ):
        await manager.async_log_success_event({"standard_callback_dynamic_params": {}}, MagicMock(), 1, 2)

    log_event.assert_called_once()


def test_prompt_names_with_slashes_are_fetched_as_a_single_path_segment():
    respx = pytest.importorskip("respx")
    langfuse_client = pytest.importorskip("langfuse.client")
    _encode_slashes_in_langfuse_prompt_names()
    client = langfuse_client.Langfuse(
        public_key="pk-test", secret_key="sk-test", host="http://langfuse.test", httpx_client=httpx.Client()
    )

    with respx.mock(assert_all_called=True) as router:
        route = router.get(url__regex=r"http://langfuse\.test/api/public/v2/prompts/.*").mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "events/description-create",
                    "version": 7,
                    "prompt": "Describe {{event}}",
                    "config": {},
                    "labels": ["production"],
                    "tags": [],
                    "type": "text",
                },
            )
        )
        prompt = client.get_prompt("events/description-create", max_retries=0)

    assert route.calls.last.request.url.raw_path.startswith(b"/api/public/v2/prompts/events%2Fdescription-create")
    assert (prompt.name, prompt.version) == ("events/description-create", 7)


def test_importing_the_langfuse_package_does_not_import_the_langfuse_sdk():
    probe = (
        "import sys\n"
        "import litellm.integrations.langfuse\n"
        "import litellm.proxy.prompts.prompt_registry\n"
        "assert 'langfuse' not in sys.modules, sorted(m for m in sys.modules if m.startswith('langfuse'))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
