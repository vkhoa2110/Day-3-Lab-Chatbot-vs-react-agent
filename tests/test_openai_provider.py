from src.core.openai_provider import OpenAIProvider


def test_openai_provider_estimates_gpt4o_cost(monkeypatch):
    monkeypatch.delenv("OPENAI_INPUT_COST_PER_1M_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", raising=False)

    provider = OpenAIProvider(model_name="gpt-4o", api_key="test")
    cost = provider._estimate_cost(
        {"prompt_tokens": 1_000, "completion_tokens": 100, "total_tokens": 1_100}
    )

    assert cost["currency"] == "USD"
    assert cost["input_cost_usd"] == 0.0025
    assert cost["output_cost_usd"] == 0.001
    assert cost["total_cost_usd"] == 0.0035
    assert cost["estimated"] is True


def test_openai_provider_matches_gpt4o_mini_before_gpt4o(monkeypatch):
    monkeypatch.delenv("OPENAI_INPUT_COST_PER_1M_TOKENS", raising=False)
    monkeypatch.delenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", raising=False)

    provider = OpenAIProvider(model_name="gpt-4o-mini", api_key="test")
    cost = provider._estimate_cost(
        {"prompt_tokens": 1_000, "completion_tokens": 1_000, "total_tokens": 2_000}
    )

    assert cost["input_rate_per_1m_tokens"] == 0.15
    assert cost["output_rate_per_1m_tokens"] == 0.60
    assert cost["total_cost_usd"] == 0.00075


def test_openai_provider_cost_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_INPUT_COST_PER_1M_TOKENS", "1")
    monkeypatch.setenv("OPENAI_OUTPUT_COST_PER_1M_TOKENS", "2")

    provider = OpenAIProvider(model_name="custom-model", api_key="test")
    cost = provider._estimate_cost(
        {"prompt_tokens": 1_000, "completion_tokens": 1_000, "total_tokens": 2_000}
    )

    assert cost["input_rate_per_1m_tokens"] == 1.0
    assert cost["output_rate_per_1m_tokens"] == 2.0
    assert cost["total_cost_usd"] == 0.003
