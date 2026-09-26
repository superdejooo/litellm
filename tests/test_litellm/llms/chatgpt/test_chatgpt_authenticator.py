import base64
import json
import time
from unittest.mock import mock_open, patch

import pytest

from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.llms.chatgpt.common_utils import (
    GetAccessTokenError,
    RefreshAccessTokenError,
)


def _make_jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def _b64(obj: dict) -> str:
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")

    return f"{_b64(header)}.{_b64(payload)}."


class TestChatGPTAuthenticator:
    @pytest.fixture
    def authenticator(self):
        with patch("os.path.exists", return_value=True):
            return Authenticator()

    def test_get_access_token_from_file(self, authenticator):
        future_time = time.time() + 3600
        auth_data = json.dumps({"access_token": "token-123", "expires_at": future_time})

        with patch("builtins.open", mock_open(read_data=auth_data)):
            token = authenticator.get_access_token()
            assert token == "token-123"

    def test_get_access_token_refresh(self, authenticator):
        past_time = time.time() - 10
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": past_time,
            }
        )
        refreshed = {
            "access_token": "token-new",
            "refresh_token": "refresh-123",
            "id_token": "id-123",
        }

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_refresh_tokens", return_value=refreshed),
        ):
            token = authenticator.get_access_token()
            assert token == "token-new"

    def test_get_account_id_from_id_token(self, authenticator):
        id_token = _make_jwt(
            {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-123"}}
        )
        auth_data = json.dumps({"id_token": id_token})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_write_auth_file") as mock_write,
        ):
            account_id = authenticator.get_account_id()
            assert account_id == "acct-123"
            mock_write.assert_called_once()
            assert mock_write.call_args[0][0]["account_id"] == "acct-123"

    def test_interactive_oauth_disabled_raises_without_auth_file(
        self, authenticator, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", "true")

        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch(
                "litellm.llms.chatgpt.authenticator._get_httpx_client"
            ) as mock_client,
            patch("litellm.llms.chatgpt.authenticator.time.sleep") as mock_sleep,
        ):
            with pytest.raises(GetAccessTokenError) as exc_info:
                authenticator.get_access_token()

        assert exc_info.value.status_code == 401
        assert authenticator.auth_file in exc_info.value.message
        assert "LITELLM_DISABLE_INTERACTIVE_OAUTH" in exc_info.value.message
        mock_client.assert_not_called()
        mock_sleep.assert_not_called()

    def test_interactive_oauth_disabled_skips_device_code_cooldown_wait(
        self, authenticator, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", "true")
        auth_data = json.dumps({"device_code_requested_at": time.time()})

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(authenticator, "_wait_for_access_token") as mock_wait,
            patch(
                "litellm.llms.chatgpt.authenticator._get_httpx_client"
            ) as mock_client,
        ):
            with pytest.raises(GetAccessTokenError):
                authenticator.get_access_token()

        mock_wait.assert_not_called()
        mock_client.assert_not_called()

    def test_interactive_oauth_disabled_after_failed_refresh_logs_error(
        self, authenticator, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", "True")
        auth_data = json.dumps(
            {
                "access_token": "token-old",
                "refresh_token": "refresh-123",
                "expires_at": time.time() - 10,
            }
        )

        with (
            patch("builtins.open", mock_open(read_data=auth_data)),
            patch.object(
                authenticator,
                "_refresh_tokens",
                side_effect=RefreshAccessTokenError(
                    message="Refresh token failed: 401", status_code=401
                ),
            ),
            patch.object(authenticator, "_request_device_code") as mock_device_code,
            patch("litellm.llms.chatgpt.authenticator.verbose_logger") as mock_logger,
        ):
            with pytest.raises(GetAccessTokenError) as exc_info:
                authenticator.get_access_token()

        assert exc_info.value.status_code == 401
        mock_device_code.assert_not_called()
        mock_logger.error.assert_called_once()
        assert "refresh token failed" in mock_logger.error.call_args[0][0]
        mock_logger.warning.assert_not_called()

    def test_login_device_code_raises_when_interactive_oauth_disabled(
        self, authenticator, monkeypatch
    ):
        monkeypatch.setenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", "true")

        with patch.object(authenticator, "_request_device_code") as mock_device_code:
            with pytest.raises(GetAccessTokenError):
                authenticator._login_device_code()

        mock_device_code.assert_not_called()

    @pytest.mark.parametrize("flag", [None, "false", ""])
    def test_device_code_login_runs_when_interactive_oauth_enabled(
        self, authenticator, monkeypatch, flag
    ):
        if flag is None:
            monkeypatch.delenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", raising=False)
        else:
            monkeypatch.setenv("LITELLM_DISABLE_INTERACTIVE_OAUTH", flag)
        tokens = {
            "access_token": "token-new",
            "refresh_token": "refresh-new",
            "id_token": "id-new",
        }

        with (
            patch("builtins.open", side_effect=FileNotFoundError),
            patch.object(
                authenticator,
                "_request_device_code",
                return_value={
                    "device_auth_id": "dev-1",
                    "user_code": "ABCD-EFGH",
                    "interval": "5",
                },
            ) as mock_device_code,
            patch.object(authenticator, "_record_device_code_request"),
            patch.object(
                authenticator,
                "_poll_for_authorization_code",
                return_value={
                    "authorization_code": "code",
                    "code_challenge": "challenge",
                    "code_verifier": "verifier",
                },
            ),
            patch.object(authenticator, "_exchange_code_for_tokens", return_value=tokens),
            patch.object(authenticator, "_write_auth_file"),
            patch("builtins.print"),
        ):
            token = authenticator.get_access_token()

        assert token == "token-new"
        mock_device_code.assert_called_once()
