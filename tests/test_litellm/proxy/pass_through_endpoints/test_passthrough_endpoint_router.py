import logging
from types import MappingProxyType
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.credential_accessor import CredentialAccessor
from litellm.proxy.pass_through_endpoints.passthrough_endpoint_router import (
    PassthroughEndpointRouter,
)
from litellm.types.utils import CredentialItem


@pytest.fixture(autouse=True)
def isolated_credential_list(monkeypatch):
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)


_OPENAI_ROUTE: Final = MappingProxyType(
    {"target_api_base": "https://api.openai.com/", "provider_default_api_base": "https://api.openai.com"}
)
_ANTHROPIC_ROUTE: Final = MappingProxyType(
    {"target_api_base": "https://api.anthropic.com", "provider_default_api_base": "https://api.anthropic.com"}
)
_ASSEMBLYAI_ROUTE: Final = MappingProxyType(
    {"target_api_base": "https://api.assemblyai.com", "provider_default_api_base": "https://api.assemblyai.com"}
)
_ASSEMBLYAI_EU_ROUTE: Final = MappingProxyType(
    {"target_api_base": "https://api.eu.assemblyai.com", "provider_default_api_base": "https://api.assemblyai.com"}
)


def _credential(name: str, api_key: str) -> CredentialItem:
    return CredentialItem(
        credential_name=name,
        credential_values={"api_key": api_key},
        credential_info={},
    )


def _flagged_deployment(model: str, **litellm_params) -> dict:
    return {
        "model_name": model.split("/", 1)[-1],
        "litellm_params": {"model": model, "use_in_pass_through": True, **litellm_params},
    }


def _passthrough_router(llm_router: litellm.Router | None) -> PassthroughEndpointRouter:
    return PassthroughEndpointRouter(llm_router_getter=lambda: llm_router)


def test_credential_loaded_after_deployment_registration_still_resolves():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_openai")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE) is None

    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-loaded-after-boot")])

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-loaded-after-boot"
    )


def test_credential_rotation_is_reflected_without_deployment_update():
    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-before-rotation")])
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_openai")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-before-rotation"
    )

    CredentialAccessor.upsert_credentials([_credential("cred_openai", "sk-after-rotation")])

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-after-rotation"
    )


def test_deleted_deployment_stops_serving_its_key(monkeypatch):
    llm_router = litellm.Router(model_list=[_flagged_deployment("openai/gpt-4o", api_key="sk-inline")])
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-inline"
    )

    llm_router.set_model_list([])
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-from-env"
    )


def test_inline_api_key_resolves_without_credential_name():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("anthropic/claude-sonnet-4-5", api_key="sk-ant-inline")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-inline"
    )


def test_file_credential_is_resolved_instead_of_returned_verbatim(monkeypatch, tmp_path):
    from litellm.litellm_core_utils import credential_accessor

    clock = [1000.0]
    monkeypatch.setattr(credential_accessor, "monotonic", lambda: clock[0])
    monkeypatch.setattr(credential_accessor, "FILE_CREDENTIAL_TTL_SECONDS", 5.0)
    key_file = tmp_path / "key"
    key_file.write_text("sk-ant-key-a\n", encoding="utf-8")
    CredentialAccessor.upsert_credentials([_credential("cred_file", f"file://{key_file}")])
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[_flagged_deployment("anthropic/claude-fable-5", litellm_credential_name="cred_file")]
        )
    )

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-key-a"
    )

    key_file.write_text("sk-ant-key-b", encoding="utf-8")
    clock[0] += 6
    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-key-b"
    )

    key_file.unlink()
    clock[0] += 6
    with pytest.raises(ValueError, match="no env fallback"):
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)


def test_missing_credential_and_no_inline_key_falls_back_to_env(monkeypatch):
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("openai/gpt-4o", litellm_credential_name="cred_deleted")]
    )
    passthrough_router = _passthrough_router(llm_router)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-from-env"
    )


def test_deployment_for_other_provider_does_not_match():
    llm_router = litellm.Router(
        model_list=[_flagged_deployment("anthropic/claude-sonnet-4-5", api_key="sk-ant-inline")]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE) is None


def test_unflagged_deployment_does_not_match():
    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-not-flagged"},
            }
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE) is None


def test_first_matching_deployment_wins():
    llm_router = litellm.Router(
        model_list=[
            _flagged_deployment("openai/gpt-4o", api_key="sk-first"),
            _flagged_deployment("openai/gpt-4o-mini", api_key="sk-second"),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-first"
    )


def test_key_of_deployment_on_another_host_is_not_sent_to_the_target():
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment("anthropic/muse-spark", api_base="https://api.meta.ai", api_key="meta-key"),
                _flagged_deployment("anthropic/claude-fable-5", api_key="sk-ant-real"),
            ]
        )
    )

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-real"
    )


def test_only_foreign_host_deployment_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment("anthropic/muse-spark", api_base="https://api.meta.ai/v1", api_key="meta-key")
            ]
        )
    )

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-env"
    )


def test_deployment_on_the_target_host_matches_regardless_of_path_and_case():
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment(
                    "anthropic/claude-fable-5", api_base="https://API.anthropic.com/v1", api_key="sk-ant-real"
                )
            ]
        )
    )

    assert (
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)
        == "sk-ant-real"
    )


@pytest.mark.parametrize(
    "target_api_base",
    ["https://api.minimax.io/anthropic", "http://api.anthropic.com", "https://api.anthropic.com:8443"],
)
def test_deployment_without_api_base_only_matches_the_provider_default_origin(target_api_base):
    passthrough_router = _passthrough_router(
        litellm.Router(model_list=[_flagged_deployment("anthropic/claude-fable-5", api_key="sk-ant-real")])
    )

    assert (
        passthrough_router.get_credentials(
            custom_llm_provider="anthropic",
            region_name=None,
            target_api_base=target_api_base,
            provider_default_api_base="https://api.anthropic.com",
        )
        is None
    )
    assert (
        passthrough_router.get_credentials(
            custom_llm_provider="anthropic",
            region_name=None,
            target_api_base="https://api.anthropic.com:443/",
            provider_default_api_base="https://api.anthropic.com",
        )
        == "sk-ant-real"
    )


@pytest.mark.parametrize(
    "target_api_base", ["http://api.anthropic.com", "https://api.anthropic.com:8443", "http://api.anthropic.com:443"]
)
def test_deployment_key_is_not_sent_to_the_same_host_on_another_scheme_or_port(target_api_base):
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment(
                    "anthropic/claude-fable-5", api_base="https://api.anthropic.com", api_key="sk-ant-real"
                )
            ]
        )
    )

    assert (
        passthrough_router.get_credentials(
            custom_llm_provider="anthropic",
            region_name=None,
            target_api_base=target_api_base,
            provider_default_api_base="https://api.anthropic.com",
        )
        is None
    )


def test_skipped_deployment_is_logged_with_its_model_and_both_origins(caplog):
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment("anthropic/muse-spark", api_base="https://api.meta.ai/v1", api_key="meta-key")
            ]
        )
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        passthrough_router.get_credentials(custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE)

    warnings: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "anthropic/muse-spark" in warnings[0]
    assert "https://api.meta.ai:443" in warnings[0]
    assert "https://api.anthropic.com:443" in warnings[0]
    assert "meta-key" not in caplog.text


def test_malformed_api_base_skips_only_that_deployment(caplog):
    passthrough_router = _passthrough_router(
        litellm.Router(
            model_list=[
                _flagged_deployment(
                    "anthropic/broken", api_base="https://api.anthropic.com:notaport", api_key="sk-broken"
                ),
                _flagged_deployment("anthropic/claude-fable-5", api_key="sk-ant-real"),
            ]
        )
    )

    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        api_key: Final = passthrough_router.get_credentials(
            custom_llm_provider="anthropic", region_name=None, **_ANTHROPIC_ROUTE
        )

    assert api_key == "sk-ant-real"
    warnings: Final = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "anthropic/broken" in warnings[0]
    assert "notaport" not in warnings[0]


def test_assemblyai_region_matching():
    llm_router = litellm.Router(
        model_list=[
            _flagged_deployment(
                "assemblyai/best", api_key="sk-eu", api_base="https://api.eu.assemblyai.com"
            ),
            _flagged_deployment("assemblyai/best", api_key="sk-us", api_base="https://api.assemblyai.com"),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert (
        passthrough_router.get_credentials(custom_llm_provider="assemblyai", region_name="eu", **_ASSEMBLYAI_EU_ROUTE)
        == "sk-eu"
    )
    assert (
        passthrough_router.get_credentials(custom_llm_provider="assemblyai", region_name=None, **_ASSEMBLYAI_ROUTE)
        == "sk-us"
    )


def test_env_fallback_when_no_router(monkeypatch):
    passthrough_router = _passthrough_router(None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")

    assert (
        passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE)
        == "sk-from-env"
    )


def test_returns_none_when_no_router_and_no_env():
    passthrough_router = _passthrough_router(None)

    assert passthrough_router.get_credentials(custom_llm_provider="openai", region_name=None, **_OPENAI_ROUTE) is None


def _vertex_credential(name: str, values: dict) -> CredentialItem:
    return CredentialItem(credential_name=name, credential_values=values, credential_info={})


def _vertex_deployment(model_name: str, model: str, **litellm_params) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": model, "use_in_pass_through": True, **litellm_params},
    }


def test_vertex_deployment_resolves_via_named_credential():
    CredentialAccessor.upsert_credentials(
        [
            _vertex_credential(
                "cred_gcp",
                {
                    "vertex_project": "proj-db",
                    "vertex_location": "global",
                    "vertex_credentials": '{"type": "service_account"}',
                },
            )
        ]
    )
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-live", "vertex_ai/gemini-live-2.5-flash", litellm_credential_name="cred_gcp"
            )
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    resolved = passthrough_router.get_vertex_credentials_from_router_deployments(model=None)

    assert resolved is not None
    assert resolved.vertex_project == "proj-db"
    assert resolved.vertex_location == "global"
    assert resolved.vertex_credentials == '{"type": "service_account"}'


def test_vertex_deployment_resolves_from_inline_litellm_params():
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-live",
                "vertex_ai/gemini-live-2.5-flash",
                vertex_project="proj-inline",
                vertex_location="us-east4",
                vertex_credentials='{"type": "service_account", "project_id": "proj-inline"}',
            )
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    resolved = passthrough_router.get_vertex_credentials_from_router_deployments(model=None)

    assert resolved is not None
    assert resolved.vertex_project == "proj-inline"
    assert resolved.vertex_location == "us-east4"
    assert resolved.vertex_credentials == '{"type": "service_account", "project_id": "proj-inline"}'


def _two_vertex_deployments_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-flash",
                "vertex_ai/gemini-2.5-flash",
                vertex_project="proj-first",
                vertex_location="us-central1",
            ),
            _vertex_deployment(
                "gemini-live",
                "vertex_ai/gemini-live-2.5-flash",
                vertex_project="proj-live",
                vertex_location="global",
            ),
        ]
    )


def test_vertex_model_hint_prefers_matching_deployment():
    passthrough_router = _passthrough_router(_two_vertex_deployments_router())

    by_alias = passthrough_router.get_vertex_credentials_from_router_deployments(model="gemini-live")
    by_upstream_id = passthrough_router.get_vertex_credentials_from_router_deployments(
        model="gemini-live-2.5-flash"
    )

    assert by_alias is not None and by_alias.vertex_project == "proj-live"
    assert by_upstream_id is not None and by_upstream_id.vertex_project == "proj-live"


def test_vertex_without_usable_hint_refuses_to_guess_between_projects():
    passthrough_router = _passthrough_router(_two_vertex_deployments_router())

    assert passthrough_router.get_vertex_credentials_from_router_deployments(model="unknown-model") is None
    assert passthrough_router.get_vertex_credentials_from_router_deployments(model=None) is None


def test_vertex_without_hint_falls_back_when_deployments_share_a_target():
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-flash", "vertex_ai/gemini-2.5-flash", vertex_project="proj-one", vertex_location="global"
            ),
            _vertex_deployment(
                "gemini-live", "vertex_ai/gemini-live-2.5-flash", vertex_project="proj-one", vertex_location="global"
            ),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    resolved = passthrough_router.get_vertex_credentials_from_router_deployments(model=None)

    assert resolved is not None and resolved.vertex_project == "proj-one"


def test_vertex_without_hint_refuses_to_guess_between_service_accounts():
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-flash",
                "vertex_ai/gemini-2.5-flash",
                vertex_project="proj-one",
                vertex_location="global",
                vertex_credentials='{"client_email": "flash@proj-one.iam"}',
            ),
            _vertex_deployment(
                "gemini-live",
                "vertex_ai/gemini-live-2.5-flash",
                vertex_project="proj-one",
                vertex_location="global",
                vertex_credentials='{"client_email": "live@proj-one.iam"}',
            ),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_vertex_credentials_from_router_deployments(model=None) is None


def test_vertex_named_credential_keeps_dict_service_account():
    service_account = {"type": "service_account", "client_email": "live@proj-db.iam"}
    CredentialAccessor.upsert_credentials(
        [
            _vertex_credential(
                "cred_gcp_dict",
                {
                    "vertex_project": "proj-db",
                    "vertex_location": "global",
                    "vertex_credentials": service_account,
                },
            )
        ]
    )
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-live", "vertex_ai/gemini-live-2.5-flash", litellm_credential_name="cred_gcp_dict"
            )
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    resolved = passthrough_router.get_vertex_credentials_from_router_deployments(model=None)

    assert resolved is not None
    assert resolved.vertex_credentials == service_account


def test_no_flagged_vertex_deployment_returns_none():
    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-live",
                "litellm_params": {
                    "model": "vertex_ai/gemini-live-2.5-flash",
                    "vertex_project": "proj-unflagged",
                    "vertex_location": "global",
                },
            },
            _flagged_deployment("openai/gpt-4o", api_key="sk-flagged"),
        ]
    )
    passthrough_router = _passthrough_router(llm_router)

    assert passthrough_router.get_vertex_credentials_from_router_deployments(model=None) is None
    assert _passthrough_router(None).get_vertex_credentials_from_router_deployments(model=None) is None


def test_vertex_deployment_with_deleted_credential_is_skipped(monkeypatch):
    CredentialAccessor.upsert_credentials(
        [_vertex_credential("cred_gone", {"vertex_project": "proj-db", "vertex_location": "global"})]
    )
    llm_router = litellm.Router(
        model_list=[
            _vertex_deployment(
                "gemini-live", "vertex_ai/gemini-live-2.5-flash", litellm_credential_name="cred_gone"
            )
        ]
    )
    passthrough_router = _passthrough_router(llm_router)
    monkeypatch.setattr(litellm, "credential_list", [])

    assert passthrough_router.get_vertex_credentials_from_router_deployments(model=None) is None
