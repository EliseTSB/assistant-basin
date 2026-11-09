# Brief: fold r6 of Part 198's run and score code (no model runs)

The design is the contract: `/home/elisepc/green/worktrees/ab-part198/PART198-DESIGN.md` (revision 3). The code is
`results/part198/run198.py`, `score198.py` and `test_build198.py` at `9494e2a`. `BUILD-NOTES.md` records every
earlier fold. `stats198.py` is frozen: import it, never edit it. Its SHA256 must stay
`03e65cb44ab7496d064b497f1e9bb38086716e74b02b5dc3ab998f881473ee64`.

The review to fold: `/home/elisepc/green/Journal/reviews/cheap-part198-code-r6-luna-luna-r1.md`. Its probe is at
`/home/elisepc/green/Journal/scratch/review-part198-code-r6-luna-luna-r1-2026-11-08/probe_wsp_base_cache.py`. Copy it
into `results/part198/` before running it, and repoint its source path at the copy's directory if needed.

## What to fix

**MAJOR: Arm B's WSP base-statistics cache is unbound.**
- **Current state.** `run_arm_b()` and `run_arm_b_seq()` call `load_base(base_path)` with no expected digest, and save
  the cache without one. `load_base()` also never checks the saved `model` field.
- **The fix.** Mirror the GPT-2 path, which already does this right: `gpt2_base_inputs_sha256()`, used at
  `run198.py:962-966` and `1189-1193`.
  - Add a `wsp_base_inputs_sha256()`. It fingerprints the Qwen snapshot files the base is computed from (each
    weight file's name and size, plus the SHA256 of config and tokenizer files; hash the large weights only if the
    GPT-2 function does), plus `run198.py` and `stats198.py`.
  - Save that digest with the cache. Require it at both Arm B load sites.
- **Both caches.** Make `load_base()` reject a cache whose `model` field differs from the expected model ID, so
  this protects both the GPT-2 and Qwen caches.
- **On a mismatch.** Refuse with a clear error that names the cache path and both digests. Do not silently rebuild:
  a stale cache in a registered output directory is something the operator must see.
- Grep for any other `load_base(` call or cached sidecar the scorer reads, and give it the same treatment. List them
  in BUILD-NOTES.

## Tests (each must fail on `9494e2a`'s code; record which fail in BUILD-NOTES)

- A WSP cache saved with a different digest is refused at both Arm B load sites. Drive this through the real
  load-site code, with every model loader patched to raise.
- A cache whose `model` field differs is refused, for both GPT-2 and Qwen.
- A matching cache is accepted, and the localization result equals the fresh computation on synthetic data.
- The review's probe now reports a refusal.

## Hard limits

- **Do not load any model or weights, run any edit, or call any network.** Fingerprint functions must be testable
  on a synthetic snapshot directory of tiny fake files. Never point a test at the real Qwen or GPT-2 snapshot.
- Any test or probe that could reach a loader (`main()`, `run_arm_*`) must first patch every loader, `torch.load`,
  `from_pretrained` and `safe_open` to raise. Check this before running anything.
- Write only inside `results/part198/`. Do not commit; the parent commits.
- Python: `/home/elisepc/edit-detect/.venv/bin/python -I`.

## Definition of done

- `fixtures198.py` and `test_build198.py` exit 0, with the check count printed.
- BUILD-NOTES has a "Fold r6" section.
- `report.md` in your run directory has these sections: SUMMARY, FINDINGS, FILES TOUCHED, BLOCKERS, SELF-ASSESSMENT.
  Write a skeleton early and fill it in as you go.
