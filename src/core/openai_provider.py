import os
import time
from typing import Dict, Any, Optional, Generator
from openai import OpenAI
from src.core.llm_provider import LLMProvider
from src.telemetry.metrics import tracker


DEFAULT_OPENAI_PRICING_USD_PER_1M = {
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str = "gpt-4o", api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        end_time = time.time()
        latency_ms = int((end_time - start_time) * 1000)

        # Extraction from OpenAI response
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens
        }
        cost = self._estimate_cost(usage)
        tracker.track_request("openai", self.model_name, usage, latency_ms)

        return {
            "content": content,
            "usage": usage,
            "cost": cost,
            "latency_ms": latency_ms,
            "provider": "openai"
        }

    def _estimate_cost(self, usage: Dict[str, int]) -> Dict[str, Any]:
        pricing = self._pricing_for_model()
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost
        has_pricing = pricing["input"] > 0 or pricing["output"] > 0
        return {
            "currency": "USD",
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(total_cost, 8),
            "input_rate_per_1m_tokens": pricing["input"],
            "output_rate_per_1m_tokens": pricing["output"],
            "estimated": has_pricing,
        }

    def _pricing_for_model(self) -> Dict[str, float]:
        input_override = self._read_float_env("OPENAI_INPUT_COST_PER_1M_TOKENS")
        output_override = self._read_float_env("OPENAI_OUTPUT_COST_PER_1M_TOKENS")

        model = self.model_name.lower()
        pricing = {"input": 0.0, "output": 0.0}
        for prefix, model_pricing in sorted(
            DEFAULT_OPENAI_PRICING_USD_PER_1M.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        ):
            if model.startswith(prefix):
                pricing = dict(model_pricing)
                break

        if input_override is not None:
            pricing["input"] = input_override
        if output_override is not None:
            pricing["output"] = output_override
        return pricing

    @staticmethod
    def _read_float_env(name: str) -> Optional[float]:
        value = os.getenv(name)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
