"""Inherited, unverified token-price assumptions for paid-action estimates.

These values were copied from Parts 60, 62, and 63 for planning. They are not
verified current provider prices and are not a billing guarantee.
"""

from __future__ import annotations

import math


# UNVERIFIED inherited Haiku 5.5 batch and direct rates, USD per million tokens.
HAIKU_BATCH_INPUT_USD_PER_MILLION = 0.05
HAIKU_BATCH_OUTPUT_USD_PER_MILLION = 0.25
HAIKU_DIRECT_INPUT_USD_PER_MILLION = 0.10
HAIKU_DIRECT_OUTPUT_USD_PER_MILLION = 0.50

# UNVERIFIED inherited Part 62 Opus 5.5 batch rates, USD per million tokens.
# No inherited direct Opus rate was available, so that mode fails closed.
OPUS_BATCH_INPUT_USD_PER_MILLION = 2.0
OPUS_BATCH_OUTPUT_USD_PER_MILLION = 10.0

TOKENS_PER_MILLION = 1_000_000
INPUT_BYTES_PER_TOKEN = 3  # UNVERIFIED planning bound inherited by Part 63.

BATCH_RATES_USD_PER_MILLION = {
    "claude-haiku-5-5": (HAIKU_BATCH_INPUT_USD_PER_MILLION,
                         HAIKU_BATCH_OUTPUT_USD_PER_MILLION),
    "claude-opus-5-5": (OPUS_BATCH_INPUT_USD_PER_MILLION,
                        OPUS_BATCH_OUTPUT_USD_PER_MILLION),
}
DIRECT_RATES_USD_PER_MILLION = {
    "claude-haiku-5-5": (HAIKU_DIRECT_INPUT_USD_PER_MILLION,
                         HAIKU_DIRECT_OUTPUT_USD_PER_MILLION),
}


def worst_case_cost(model: str, input_tokens_upper: int,
                    max_tokens: int, batch: bool = True) -> float:
    """Estimate the maximum request cost at inherited rates.

    Callers should use ceil(serialized UTF-8 request bytes / 3) for
    ``input_tokens_upper``. The byte ratio and all rates remain UNVERIFIED.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if isinstance(input_tokens_upper, bool) or not isinstance(input_tokens_upper, int) or input_tokens_upper < 0:
        raise ValueError("input_tokens_upper must be a non-negative integer")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("max_tokens must be a non-negative integer")
    if not isinstance(batch, bool):
        raise ValueError("batch must be a boolean")
    rates = BATCH_RATES_USD_PER_MILLION if batch else DIRECT_RATES_USD_PER_MILLION
    if model not in rates:
        mode = "batch" if batch else "direct"
        raise ValueError(f"no inherited {mode} rates are registered for {model!r}")
    input_rate, output_rate = rates[model]
    cost = (input_tokens_upper * input_rate + max_tokens * output_rate) / TOKENS_PER_MILLION
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("computed cost is not finite and non-negative")
    return cost
