# Price attestation, 2026-10-23 00:50 CT (Claude, from platform.claude.com/docs/en/about-claude/pricing, fetched by me)

Published list prices, USD per million tokens (prompts up to 100,000 tokens for Haiku 5.5):
- Claude Haiku 5.5: input 0.10, output 0.50. Batch API is 50 percent off both: input 0.05, output 0.25.
- Claude Sonnet 5.5: input 2, output 10 (batch 1 and 5).
- Claude Opus 5.5: input 4, output 20 (batch 2 and 10).
Over 100,000 prompt tokens Haiku 5.5 costs 0.50 and 2.50 per million; no Part 60, 62 or 63 request approaches that.

The repo constants match: part60/gen60.py PRICE_INPUT_BATCH 0.05, PRICE_OUTPUT_BATCH 0.25, PRICE_INPUT_DIRECT 0.10, PRICE_OUTPUT_DIRECT 0.50; paidaction/pricing.py Haiku batch 0.05/0.25, direct 0.10/0.50, Opus batch 2.0/10.0. The Opus DIRECT rate (4/20) is not in pricing.py and that mode stays fail-closed. These are list prices read from the docs on this date; the account's invoice is the final truth, so the first real batch's reported usage is compared with the ledger.
