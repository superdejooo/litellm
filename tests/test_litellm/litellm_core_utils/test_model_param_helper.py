

from litellm.litellm_core_utils.model_param_helper import ModelParamHelper


def test_get_all_llm_api_params_is_correct():
    """The cached result must equal a fresh, uncached computation."""
    cached = ModelParamHelper._get_all_llm_api_params()
    uncached = ModelParamHelper._get_all_llm_api_params.__wrapped__()
    assert cached == uncached
    assert {"model", "temperature", "stream"} <= cached
    assert "metadata" not in cached  # excluded via _get_exclude_kwargs


def test_standard_logging_model_parameters_exclude_prompt_content():
    """Responses API `instructions` is prompt text, like `system`, so it must not reach spend logs."""
    model_parameters = ModelParamHelper.get_standard_logging_model_parameters(
        {
            "temperature": 0.2,
            "max_output_tokens": 128,
            "instructions": "You are a helpful assistant.",
            "input": "hi",
            "system": "Be concise.",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert model_parameters == {"temperature": 0.2, "max_output_tokens": 128}


def test_get_all_llm_api_params_is_memoized():
    """Regression: the param set is static and is rebuilt on every request via
    the cache-key and spend-logging paths, so it must be memoized. Without the
    cache each call returns a freshly built set (a different object)."""
    first = ModelParamHelper._get_all_llm_api_params()
    second = ModelParamHelper._get_all_llm_api_params()
    assert first is second
