import os
from typing import Dict, Any
from src.telemetry.logger import logger


DEFAULT_PRICING_USD_PER_1M = {
    "gpt-4o-2024-05-13": {"input": 5.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class PerformanceTracker:
    """
    Tracking industry-standard metrics for LLMs.
    """
    def __init__(self):
        self.session_metrics = []

    def track_request(self, provider: str, model: str, usage: Dict[str, int], latency_ms: int):
        """
        Logs a single request metric to our telemetry.
        """
        metric = {
            "provider": provider,
            "model": model,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "latency_ms": latency_ms,
            "cost": self._calculate_cost(model, usage),
        }
        self.session_metrics.append(metric)
        logger.log_event("LLM_METRIC", metric)

    def _calculate_cost(self, model: str, usage: Dict[str, int]) -> Dict[str, Any]:
        """
        Estimate USD cost from token usage.
        Rates are USD per 1M tokens and can be overridden with environment variables.
        """
        pricing = self._pricing_for_model(model)
        input_cost = (usage.get("prompt_tokens", 0) / 1_000_000) * pricing["input"]
        output_cost = (usage.get("completion_tokens", 0) / 1_000_000) * pricing["output"]
        has_pricing = pricing["input"] > 0 or pricing["output"] > 0
        return {
            "currency": "USD",
            "input_cost_usd": round(input_cost, 8),
            "output_cost_usd": round(output_cost, 8),
            "total_cost_usd": round(input_cost + output_cost, 8),
            "estimated": has_pricing,
        }

    def _pricing_for_model(self, model_name: str) -> Dict[str, float]:
        input_override = self._read_float_env("OPENAI_INPUT_COST_PER_1M_TOKENS")
        output_override = self._read_float_env("OPENAI_OUTPUT_COST_PER_1M_TOKENS")
        pricing = {"input": 0.0, "output": 0.0}
        model = (model_name or "").lower()
        for prefix, model_pricing in sorted(
            DEFAULT_PRICING_USD_PER_1M.items(),
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
    def _read_float_env(name: str):
        value = os.getenv(name)
        if value in (None, ""):
            return None
        try:
            return float(value)
        except ValueError:
            return None

# Global tracker instance
tracker = PerformanceTracker()
