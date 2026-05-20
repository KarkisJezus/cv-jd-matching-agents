"""
LLM pricing data for cost estimation.

Prices are per 1 million tokens, as published by OpenAI and other
providers. Used to compute dollar cost per evaluation run.

Values current as of 2025. Update here when provider pricing changes.
For local LLMs (Ollama, LM Studio), cost is zero.
"""

# ── Pricing table ────────────────────────────────────────────
# Format: model_name -> {"input": $ per 1M input tokens,
#                        "output": $ per 1M output tokens}

_PRICING_PER_1M: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic (if ever used via an adapter)
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
}

# Local LLM identifiers (prefix-matched) — all free
_LOCAL_MODEL_PREFIXES = ("llama", "qwen", "mistral", "gemma", "phi", "deepseek")


def get_price_per_token(model_name: str) -> dict[str, float]:
    """
    Return {"input": $/token, "output": $/token} for a model.

    Looks up the model in the pricing table. If not found, returns
    zero pricing (conservative — don't fabricate costs for unknown
    models). Local LLMs match against known prefixes and return zero.
    """
    model_lower = model_name.lower().strip()

    # Local models are free
    if any(model_lower.startswith(p) for p in _LOCAL_MODEL_PREFIXES):
        return {"input": 0.0, "output": 0.0}

    # Exact match in pricing table
    if model_lower in _PRICING_PER_1M:
        per_1m = _PRICING_PER_1M[model_lower]
        return {
            "input": per_1m["input"] / 1_000_000,
            "output": per_1m["output"] / 1_000_000,
        }

    # Prefix match for versioned names like "gpt-4o-mini-2024-07-18"
    for key, per_1m in _PRICING_PER_1M.items():
        if model_lower.startswith(key):
            return {
                "input": per_1m["input"] / 1_000_000,
                "output": per_1m["output"] / 1_000_000,
            }

    # Unknown model: return zero cost rather than guessing
    return {"input": 0.0, "output": 0.0}


def compute_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Compute dollar cost for a given token usage on a specific model."""
    prices = get_price_per_token(model_name)
    return prompt_tokens * prices["input"] + completion_tokens * prices["output"]


def is_priced(model_name: str) -> bool:
    """Return True if the model has known pricing (not a local/unknown model)."""
    prices = get_price_per_token(model_name)
    return prices["input"] > 0 or prices["output"] > 0
