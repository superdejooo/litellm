from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.assembly_passthrough_logging_handler import (
    AssemblyAIPassthroughLoggingHandler,
)


def _flagged(api_base: str, api_key: str) -> dict:
    return {
        "model_name": "best",
        "litellm_params": {
            "model": "assemblyai/best",
            "api_base": api_base,
            "api_key": api_key,
            "use_in_pass_through": True,
        },
    }


@pytest.fixture
def served_router(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "credential_list", [])
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "sk-assembly-env")

    def serve(*deployments: dict) -> None:
        monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", litellm.Router(model_list=list(deployments)))

    return serve


@respx.mock
def test_transcript_poll_keeps_a_foreign_host_deployment_key_off_assemblyai(served_router) -> None:
    served_router(_flagged("https://foreign.example.com", "sk-foreign"))
    poll: Final = respx.get("https://api.assemblyai.com/v2/transcript/t1").mock(
        return_value=httpx.Response(200, json={"id": "t1", "status": "completed"})
    )

    AssemblyAIPassthroughLoggingHandler()._get_assembly_transcript(transcript_id="t1", request_region=None)

    assert poll.calls.last.request.headers["Authorization"] == "Bearer sk-assembly-env"


@respx.mock
def test_eu_transcript_poll_uses_the_eu_deployment_key_on_the_eu_route_host(served_router) -> None:
    served_router(_flagged("https://api.eu.assemblyai.com", "sk-eu"))
    poll: Final = respx.get("https://api.eu.assemblyai.com/v2/transcript/t1").mock(
        return_value=httpx.Response(200, json={"id": "t1", "status": "completed"})
    )

    AssemblyAIPassthroughLoggingHandler()._get_assembly_transcript(transcript_id="t1", request_region="eu")

    assert poll.calls.last.request.headers["Authorization"] == "Bearer sk-eu"
