import asyncio
from datetime import datetime
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.integrations.langfuse.langfuse_prompt_management import (
    LangfusePromptManagement,
    langfuse_client_init,
)


class TestLangfusePromptManagement:
    def setup_method(self):
        # Mock langfuse package to avoid triggering real import.
        # The real langfuse import fails on Python 3.14 due to pydantic v1 incompatibility.
        # This also prevents test-ordering issues when earlier tests remove sys.modules["langfuse"].
        self._mock_langfuse = MagicMock()
        self._mock_langfuse.version.__version__ = "3.0.0"
        self._langfuse_patcher = patch.dict(
            "sys.modules", {"langfuse": self._mock_langfuse}
        )
        self._langfuse_patcher.start()

    def teardown_method(self):
        self._langfuse_patcher.stop()

    def test_get_prompt_from_id(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with (
            patch.object(
                langfuse_prompt_management, "should_run_prompt_management"
            ) as mock_should_run_prompt_management,
            patch.object(
                langfuse_prompt_management, "_get_prompt_from_id"
            ) as mock_get_prompt_from_id,
        ):
            mock_should_run_prompt_management.return_value = True
            langfuse_prompt_management.get_chat_completion_prompt(
                model="langfuse/langfuse-model",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                non_default_params={},
                prompt_id="test-chat-prompt",
                prompt_variables={},
                dynamic_callback_params={},
                prompt_version=4,
            )

            mock_get_prompt_from_id.assert_called_once()
            assert mock_get_prompt_from_id.call_args.kwargs["prompt_version"] == 4

    @pytest.mark.asyncio
    async def test_async_render_records_the_prompt_link_on_the_logging_object(self):
        manager = LangfusePromptManagement()
        prompt_client = SimpleNamespace(
            name="events/description-create",
            version=7,
            config={},
            compile=lambda **variables: [{"role": "system", "content": f"Describe {variables['event']}"}],
        )
        logging_obj = SimpleNamespace(model_call_details={})

        with patch.object(manager, "_get_prompt_from_id", return_value=prompt_client) as get_prompt:
            _, messages, _ = await manager.async_get_chat_completion_prompt(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Launch party"}],
                non_default_params={},
                prompt_id="events/description-create",
                prompt_variables={"event": "launch"},
                dynamic_callback_params={},
                litellm_logging_obj=logging_obj,
                prompt_label="staging",
            )

        assert messages[0] == {"role": "system", "content": "Describe launch"}
        assert logging_obj.model_call_details["langfuse_prompt_link"] == {
            "name": "events/description-create",
            "version": 7,
        }
        assert get_prompt.call_args.kwargs["prompt_label"] == "staging"
        langfuse_client_init.cache_clear()

    @staticmethod
    def _prompt_client(version: int) -> SimpleNamespace:
        return SimpleNamespace(
            name="events/description-create",
            version=version,
            config={},
            compile=lambda **_: [{"role": "system", "content": f"Describe v{version}"}],
        )

    @pytest.mark.asyncio
    async def test_prompt_link_names_the_version_that_was_rendered(self):
        manager = LangfusePromptManagement()
        fetched_versions = iter(range(7, 20))
        logging_obj = SimpleNamespace(model_call_details={})

        with patch.object(
            manager,
            "_get_prompt_from_id",
            side_effect=lambda **_: self._prompt_client(next(fetched_versions)),
        ):
            _, messages, _ = await manager.async_get_chat_completion_prompt(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Launch party"}],
                non_default_params={},
                prompt_id="events/description-create",
                prompt_variables={},
                dynamic_callback_params={},
                litellm_logging_obj=logging_obj,
            )

        rendered_version = logging_obj.model_call_details["langfuse_prompt_link"]["version"]
        assert messages[0] == {"role": "system", "content": f"Describe v{rendered_version}"}
        langfuse_client_init.cache_clear()

    @pytest.mark.asyncio
    async def test_prompt_link_does_not_fail_an_already_rendered_request(self):
        manager = LangfusePromptManagement()
        logging_obj = SimpleNamespace(model_call_details={})
        rendered_client = self._prompt_client(7)

        with patch.object(
            manager,
            "_get_prompt_from_id",
            side_effect=[rendered_client, rendered_client, httpx.ConnectError("langfuse down")],
        ):
            _, messages, _ = await manager.async_get_chat_completion_prompt(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Launch party"}],
                non_default_params={},
                prompt_id="events/description-create",
                prompt_variables={},
                dynamic_callback_params={},
                litellm_logging_obj=logging_obj,
            )

        assert messages[0] == {"role": "system", "content": "Describe v7"}
        assert logging_obj.model_call_details["langfuse_prompt_link"] == {
            "name": "events/description-create",
            "version": 7,
        }
        langfuse_client_init.cache_clear()

    @pytest.mark.asyncio
    async def test_async_render_without_prompt_id_records_no_prompt_link(self):
        manager = LangfusePromptManagement()
        logging_obj = SimpleNamespace(model_call_details={})

        with patch.object(manager, "_get_prompt_from_id") as get_prompt:
            await manager.async_get_chat_completion_prompt(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hi"}],
                non_default_params={},
                prompt_id=None,
                prompt_variables=None,
                dynamic_callback_params={},
                litellm_logging_obj=logging_obj,
            )

        get_prompt.assert_not_called()
        assert "langfuse_prompt_link" not in logging_obj.model_call_details
        langfuse_client_init.cache_clear()

    def test_log_failure_event_runs_async_logger(self):
        langfuse_prompt_management = LangfusePromptManagement()
        with patch(
            "litellm.integrations.langfuse.langfuse_prompt_management.run_async_function"
        ) as mock_run_async:
            kwargs = {"standard_callback_dynamic_params": {}}
            start_time, end_time = 1, 2

            langfuse_prompt_management.log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=start_time,
                end_time=end_time,
            )

            mock_run_async.assert_called_once()
            assert (
                mock_run_async.call_args[0][0]
                == langfuse_prompt_management.async_log_failure_event
            )

    def test_langfuse_client_init_passes_dedicated_httpx_client(self):
        import httpx

        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        shared_client = _get_httpx_client().client

        mock_langfuse_class = MagicMock()
        with (
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.resolve_langfuse_credentials",
                return_value=("pk-1234", "sk-1234", "https://localhost"),
            ),
            patch(
                "litellm.integrations.langfuse.langfuse_prompt_management.LangFuseLogger._get_langfuse_flush_interval",
                return_value=1,
            ),
            patch.dict("sys.modules", {"langfuse": self._mock_langfuse}),
            patch(
                "litellm.llms.custom_httpx.http_handler.get_ssl_configuration",
                return_value=False,
            ) as mock_get_ssl,
        ):
            self._mock_langfuse.Langfuse = mock_langfuse_class

            langfuse_client_init(
                langfuse_public_key="pk-1234",
                langfuse_secret="sk-1234",
                langfuse_host="https://localhost",
            )

            mock_langfuse_class.assert_called_once()
            call_kwargs = mock_langfuse_class.call_args[1]
            assert "httpx_client" in call_kwargs
            passed_client = call_kwargs["httpx_client"]
            assert isinstance(passed_client, httpx.Client)
            assert passed_client is not shared_client
            mock_get_ssl.assert_called_once()

        langfuse_client_init.cache_clear()


class _RecordingLangfuseForEnv:
    last_environment: str | None = None

    def __init__(self, *, environment: str | None = None, **parameters: object) -> None:  # kwargs-ok: records only environment out of whatever langfuse_client_init forwards
        type(self).last_environment = environment


@pytest.mark.parametrize(
    ("env_value", "expected"),
    (("Production", "default"), ("production ", "production"), ("prod", "prod")),
)
def test_langfuse_client_init_resolves_deployment_environment(monkeypatch, env_value, expected):
    mock_langfuse_module: Final = MagicMock()
    mock_langfuse_module.version.__version__ = "2.60.0"
    mock_langfuse_module.Langfuse = _RecordingLangfuseForEnv
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", env_value)
    monkeypatch.setattr(_RecordingLangfuseForEnv, "last_environment", None)
    with patch.dict("sys.modules", MappingProxyType({"langfuse": mock_langfuse_module})):
        langfuse_client_init.cache_clear()
        langfuse_client_init()
    langfuse_client_init.cache_clear()
    assert _RecordingLangfuseForEnv.last_environment == expected


@pytest.fixture
def isolated_callbacks(monkeypatch):
    for callback_list in (
        "callbacks",
        "input_callback",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "_async_input_callback",
    ):
        monkeypatch.setattr(litellm, callback_list, [])


@pytest.mark.asyncio
async def test_rendered_config_prompt_is_linked_on_the_langfuse_otel_span(monkeypatch, isolated_callbacks):
    respx = pytest.importorskip("respx")
    pytest.importorskip("langfuse.client")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from litellm.integrations.langfuse.langfuse_otel import LangfuseOtelLogger
    from litellm.integrations.opentelemetry import OpenTelemetryConfig
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.prompts import prompt_registry
    from litellm.proxy.utils import ProxyLogging
    from litellm.types.prompts.init_prompts import PromptSpec

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-prompt-link")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-prompt-link")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse-prompt-link.test")
    langfuse_client_init.cache_clear()
    registry: Final = prompt_registry.InMemoryPromptRegistry()
    monkeypatch.setattr(prompt_registry, "IN_MEMORY_PROMPT_REGISTRY", registry)
    exporter: Final = InMemorySpanExporter()
    tracer_provider: Final = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_logger: Final = LangfuseOtelLogger(
        config=OpenTelemetryConfig(exporter="console"), callback_name="langfuse_otel", tracer_provider=tracer_provider
    )

    with respx.mock(assert_all_called=True) as langfuse_api:
        langfuse_api.get(
            url__regex=r"^http://langfuse-prompt-link\.test/api/public/v2/prompts/events%2Fdescription-create(\?.*)?$"
        ).mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "events/description-create",
                    "version": 7,
                    "prompt": [{"role": "system", "content": "Describe the {{event}}"}],
                    "config": {},
                    "labels": ["production"],
                    "tags": [],
                    "type": "chat",
                },
            )
        )
        registry.initialize_prompt(
            PromptSpec(
                prompt_id="events/description-create",
                litellm_params={
                    "prompt_id": "events/description-create",
                    "prompt_integration": "langfuse",
                    "langfuse_host": "os.environ/LANGFUSE_HOST",
                    "langfuse_public_key": "os.environ/LANGFUSE_PUBLIC_KEY",
                    "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY",
                },
            )
        )
        litellm.logging_callback_manager.add_litellm_callback(otel_logger)
        logging_obj, data = litellm.utils.function_setup(
            "acompletion",
            litellm.utils.Rules(),
            datetime.now(),
            litellm_call_id="prompt-link-call",
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Launch party"}],
            prompt_id="events/description-create",
            prompt_variables={"event": "launch"},
        )
        data["litellm_logging_obj"] = logging_obj

        rendered = await ProxyLogging(user_api_key_cache=UserApiKeyCache()).pre_call_hook(
            user_api_key_dict=UserAPIKeyAuth(), data=data, call_type="acompletion"
        )
        await litellm.acompletion(**rendered, mock_response="A launch party.")

    for _ in range(50):
        if exporter.get_finished_spans():
            break
        await asyncio.sleep(0.1)
    langfuse_client_init.cache_clear()

    assert rendered["messages"][0] == {"role": "system", "content": "Describe the launch"}
    generation_spans = [
        span
        for span in exporter.get_finished_spans()
        if span.attributes.get("langfuse.observation.type") == "generation"
    ]
    assert len(generation_spans) == 1
    assert generation_spans[0].attributes["langfuse.observation.prompt.name"] == "events/description-create"
    assert generation_spans[0].attributes["langfuse.observation.prompt.version"] == 7


_DEFAULT_PROJECT_PROMPT_URL: Final = r"^http://default-project\.test/api/public/v2/prompts/"
_PROMPT_ENTRY_PROJECT_PROMPT_URL: Final = r"^http://prompt-entry-project\.test/api/public/v2/prompts/"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dynamic_callback_params",
    (
        {},
        {
            "langfuse_public_key": "pk-default-project",
            "langfuse_secret_key": "sk-default-project",
            "langfuse_host": "http://default-project.test",
        },
    ),
    ids=("no-request-credentials", "team-callback-credentials"),
)
async def test_config_prompt_renders_from_the_project_its_prompt_entry_names(monkeypatch, dynamic_callback_params):
    respx = pytest.importorskip("respx")
    pytest.importorskip("langfuse.client")
    from litellm.integrations.langfuse import prompt_initializer
    from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-default-project")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-default-project")
    monkeypatch.setenv("LANGFUSE_HOST", "http://default-project.test")
    monkeypatch.setenv("PROMPT_ENTRY_LANGFUSE_HOST", "http://prompt-entry-project.test")
    langfuse_client_init.cache_clear()
    litellm_params: Final = PromptLiteLLMParams(
        prompt_id="events/summary",
        prompt_integration="langfuse",
        langfuse_public_key="pk-prompt-entry-project",
        langfuse_secret_key="sk-prompt-entry-project",
        langfuse_host="os.environ/PROMPT_ENTRY_LANGFUSE_HOST",
    )
    spec: Final = PromptSpec(prompt_id="events/summary", litellm_params=litellm_params)
    manager: Final = prompt_initializer(litellm_params, spec)
    prompt_body: Final = {
        "name": "events/summary",
        "version": 3,
        "prompt": "Summarise {{event}}",
        "config": {},
        "labels": ["production"],
        "tags": [],
        "type": "text",
    }

    with respx.mock(assert_all_called=False) as langfuse_api:
        default_project: Final = langfuse_api.get(url__regex=_DEFAULT_PROJECT_PROMPT_URL).mock(
            return_value=httpx.Response(200, json={**prompt_body, "prompt": "wrong project"})
        )
        prompt_entry_project: Final = langfuse_api.get(url__regex=_PROMPT_ENTRY_PROJECT_PROMPT_URL).mock(
            return_value=httpx.Response(200, json=prompt_body)
        )
        _, messages, _ = await manager.async_get_chat_completion_prompt(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            non_default_params={},
            prompt_id="events/summary",
            prompt_variables={"event": "launch"},
            dynamic_callback_params=dynamic_callback_params,
            litellm_logging_obj=SimpleNamespace(model_call_details={}),
        )
    langfuse_client_init.cache_clear()

    assert messages[0] == {"role": "system", "content": "Summarise launch"}
    assert prompt_entry_project.called
    assert not default_project.called


@pytest.mark.asyncio
async def test_callback_prompt_manager_renders_from_the_project_the_request_names(monkeypatch):
    respx = pytest.importorskip("respx")
    pytest.importorskip("langfuse.client")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-default-project")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-default-project")
    monkeypatch.setenv("LANGFUSE_HOST", "http://default-project.test")
    langfuse_client_init.cache_clear()
    manager: Final = LangfusePromptManagement()

    with respx.mock(assert_all_called=False) as langfuse_api:
        default_project: Final = langfuse_api.get(url__regex=_DEFAULT_PROJECT_PROMPT_URL).mock(
            return_value=httpx.Response(500)
        )
        request_project: Final = langfuse_api.get(url__regex=_PROMPT_ENTRY_PROJECT_PROMPT_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "name": "events/summary",
                    "version": 3,
                    "prompt": "Summarise {{event}}",
                    "config": {},
                    "labels": ["production"],
                    "tags": [],
                    "type": "text",
                },
            )
        )
        _, messages, _ = await manager.async_get_chat_completion_prompt(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            non_default_params={},
            prompt_id="events/summary",
            prompt_variables={"event": "launch"},
            dynamic_callback_params={
                "langfuse_public_key": "pk-request-project",
                "langfuse_secret_key": "sk-request-project",
                "langfuse_host": "http://prompt-entry-project.test",
            },
            litellm_logging_obj=SimpleNamespace(model_call_details={}),
        )
    langfuse_client_init.cache_clear()

    assert messages[0] == {"role": "system", "content": "Summarise launch"}
    assert request_project.called
    assert not default_project.called
