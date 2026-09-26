from unittest.mock import MagicMock

import httpx
import pytest
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.types.utils import ModelResponse


def test_sanitize_usage_obj_handles_null_tokens():
    """
    Tests that _sanitize_usage_obj correctly converts None values for token counts to 0.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": None, "completion_tokens": 50, "total_tokens": None},
    }

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json["usage"]["prompt_tokens"] == 0
    assert sanitized_json["usage"]["completion_tokens"] == 50  # Should remain unchanged
    assert sanitized_json["usage"]["total_tokens"] == 0


def test_sanitize_usage_obj_no_usage():
    """
    Tests that the sanitizer handles cases where the 'usage' object is missing.
    """
    response_json = {"choices": []}

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert "usage" not in sanitized_json  # Should not add a usage key


def test_sanitize_usage_obj_valid_usage():
    """
    Tests that the sanitizer does not modify a valid usage object.
    """
    response_json = {
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    # Create a copy to compare against
    original_json = response_json.copy()

    sanitized_json = OpenAILikeChatConfig._sanitize_usage_obj(response_json)

    # Assert
    assert sanitized_json == original_json  # The object should be unchanged


def test_transform_response_accepts_null_audio_metadata():
    """OpenAI-compatible TTS providers send audio.expires_at / audio.transcript as null."""
    response = httpx.Response(
        200,
        json={
            "id": "chatcmpl-tts",
            "created": 1776975000,
            "model": "mimo-v2-tts",
            "object": "chat.completion",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": {
                            "id": "audio-123",
                            "data": "UklGRi...",
                            "expires_at": None,
                            "transcript": None,
                        },
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    )

    model_response = OpenAILikeChatConfig._transform_response(
        model="mimo-v2-tts",
        response=response,
        model_response=ModelResponse(),
        stream=False,
        logging_obj=MagicMock(),
        optional_params={},
        api_key=None,
        data={},
        messages=[],
        print_verbose=None,
        encoding=None,
        json_mode=None,
        custom_llm_provider="xiaomi_mimo",
        base_model=None,
    )

    audio = model_response.choices[0].message.audio
    assert audio.id == "audio-123"
    assert audio.data == "UklGRi..."
    assert audio.expires_at == 0
    assert audio.transcript == ""
