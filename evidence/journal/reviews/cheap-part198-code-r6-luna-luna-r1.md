<!-- cheap-review.mjs cheap-review-part198-code-r6-luna-luna-r1-2026-11-08: /home/elisepc/green/worktrees/ab-part198 91b6af4..HEAD, lane luna, df-run exit 0 -->
# Cold review: part198-code-r6-luna

**Verdict: SHIP-WITH-FIXES.** The WSP base-statistics sidecar can silently substitute stale values into registered localization and blind-reader scores.

## Findings

### MAJOR — INTRODUCED — Arm B trusts an unbound base-statistics cache

- **Files:** `results/part198/run198.py:857-870, 1334-1337, 1461-1464`; affected score consumers: `results/part198/score198.py:355-359`.
- **Trigger:** `run_arm_b()` and `run_arm_b_seq()` load `base-qwen35.json` with `load_base(base_path)` and save it without `inputs_sha256`. `load_base()` validates a digest only when its caller supplies `expected_inputs_sha256`; it does not check the saved `model` field either. GPT-2's parallel cache has an input fingerprint, but the WSP cache does not.
- **MEASURED:** `probe_wsp_base_cache.py` imports the real runner module without importing torch, writes a synthetic sidecar marked for a different model and stale digest, then calls the same unchecked loader and real `run198.localization()` helper. Output:

  ```text
  torch_imported: False
  cache_model_marker: different-model-or-old-statistics
  cache_input_hash_marker: stale-input-digest
  load_base_accepts_cache_without_expected_hash: True
  fresh_base_f1_localized: False
  cached_base_f1_localized: True
  both_arm_b_call_sites_use_unchecked_load: True
  ```

- **Why it matters:** A reused output directory can make the same WSP suspect produce different localization outcomes depending on cached baseline provenance. The scorer uses that localization for forecast 198.6 and the joint blind-reader forecast 198.7; Arm B's blind reader also fills unedited layers from this cache. This is a silent score change, not merely a cache-load error.
- **Smallest fix:** Fingerprint the Qwen snapshot, tokenizer/config inputs, and `run198.py`/`stats198.py` used to build the WSP base; save that digest and require it at both Arm B load sites. Reject or rebuild a cache whose model ID or digest does not match.

## What I checked and found sound

- Read `run198.py` and `score198.py` against `PART198-DESIGN.md` revision 3 and the prior r1–r5 findings. The earlier mass-localization shape, replay-conflict, reader-tail mutation, runner-lock, and 198.8 incomplete-sequence defects appear fixed in the current code.
- Scoring takes all eight log locks in a stable order. Runner resume is the only production path that repairs a torn JSONL tail, and it does so while holding the stream lock. The 198.8 rule requires ten unique completed sequence rows and refuses any registered sequence failure.
- No Torch import, model or weight load, runner arm, network call, or test suite was run. The single probe used temporary synthetic JSON only and confirmed `torch_imported: False`.
