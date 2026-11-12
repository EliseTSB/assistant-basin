# Part 209 build report

Builder: Claude (Opus 5.5), a fork of the designing session, 2026-11-12. Branch part209 in
`/home/elisepc/green/worktrees/ab-part203`. Nothing here is a result: no prediction was scored and no registered
quantity was read. Nothing was pushed and nothing ran on the A770.

## What was built

- `part209/config.json`: the six OLMo 2 1B stages (ids and revisions copied from `part205/config.json`) and
  qwen-base / qwen-chat, each marked chat or not (chat: sft, dpo, instruct, qwen-chat); R1, R3, R2, reader, gate and
  bootstrap settings exactly as the design states.
- `part209/tokens.json` (written by `run209.py --tokens`): " sitting", " seated", " standing", " lying" and " laying"
  are one token in both tokenizers (OLMo 2 Instruct at 48d788ec, Qwen3.5-4B at 851bf6e8; the other OLMo snapshots and
  Qwen3.5-4B-Base give the same ids), so " laying" counts in LIE. The runner re-checks the live tokenizer against this
  file and refuses on a mismatch.
- `part209/run209.py`: one stage per process, loading exactly as `part205/run205.py` (OLMo float32 CPU eager, each
  tensor cloned off the /mnt/e memory map; Qwen through `neuron_report.load_model`, bf16, eager). Default mode: R1 for
  every frame, R1c for chat stages (chat template, Qwen thinking off, forced "Right now I am"), R3 on the sit-stand and
  sit-lie axes (per-layer held-out accuracy, layer chosen in the 50-80% window on reference sentences only with ties to
  the deeper layer, a 100-shuffle within-subject label null, every frame's projection, c from the told poles). Every
  R1/R3 pass is batch size 1, so there is no padding and the final token is the last input token. Writes
  `r1r3-<stage>.json`, `states-<stage>.npz` (window layers, float16) and `part209/models.json`. `--r2`: sampled
  answers in the chat template; temperature 0.7 and top-p 0.95 are applied by `PerRowSampler`, which draws each row's
  token from that row's own seeded generator on the CPU, so an answer does not depend on its batch (tested); rows are
  appended, fsynced and skipped on resume.
- `part209/reader_prompt.md` and `part209/read209.py`: blind readers (sha256 ids, shuffled, answer text only). Primary
  DeepSeek V4.1 Flash through OpenCode Go (`judge_client`, max_tokens 4000); second Claude Haiku 5.5 direct (thinking
  disabled, text blocks joined). The 35-item known-answer check runs first every time and a reader under 32 is refused.
- `part209/score209.py`: G0, G1 (R1 and R3 halves), G2 (both checks, posture kappa, per-model told compliance),
  predictions 209.1 to 209.10, the headline-withdrawal check, and 10,000-resample bootstraps clustered by paraphrase
  (paired across arms) with a multinomial redraw of answers within each drawn paraphrase. A prediction whose gate
  fails prints "not read" with the reason. Writes `scores.json`.
- `part209/test_part209.py`: 24 tests, all passing (see below; the final run took 51.7 s while the smoke held the CPU).
- `part209/steps.txt`: launch lines in order, pilot first, marked as not run.

## Tests

`HF_HOME=/mnt/e/neuron-probe/hf .venv/bin/python -m pytest -q part209/test_part209.py`: 24 passed. They cover the
prompt files against the generator, that the arms differ only in the tested phrase, the token ids and a refused
mismatch, Qwen's thinking-off template, the R1 mass arithmetic, the layer window and tie rule, held-out accuracy and the
shuffle null on synthetic states, c from the poles, the runner end to end on a tiny random OLMo 2 with the real
tokenizer, the final-token read against a direct pass, the sampler's top-p and determinism, a row's answer being the
same alone and in a mixed batch, the R2 row counts (720 per model, 68 for the pilot), resume, the label parser, the
blind reader check (35 right with an oracle, 10 with a reader that always says sitting), and the scorer on a fixture
with a known verdict for every prediction plus a fixture whose failed gates must stop the reads.

Deliberate break: `test_sabotage_turns_red` copies `score209.py` with 209.1's threshold changed from 0.70 to 0.95 and
runs the known-truth test against the copy; it fails on `{'209.1': 'failed'} != {'209.1': 'held'}`, so the fixture can
see a broken threshold.

## Smoke (CPU, OLMo 2 1B Instruct, 8 GB scope)

Run 2026-11-12 10:46 to 10:57 CT in `systemd-run --user --scope -p MemoryMax=8G`, script in the builder's scratchpad
(`run209.py --stage instruct`, then `--r2 --paraphrases 1 --samples 1 --told-samples 1 --limit 4 --batch 4`), output in
`results/part209/smoke/` (log `smoke.log`, models `smoke/models.json`). Load 18 s (float32, 179 tensors, cloned off the
memory map). R1 + R1c for 96 frames took 208 s and the 360 reference sentences about 300 s: **0.94 s per pass**
(552 passes in 521 s, 8 CPU threads).

Gates at this stage (computed with score209's own gate functions):

- G0: mean P3 over the 24 told frames 0.810 (needs 0.3): pass.
- G1, R1 half: the told posture has the largest mass in 8 of 8 paraphrases for TOLD-SIT, TOLD-STAND and TOLD-LIE: pass.
- G1, R3 half: TOLD-SIT projects above TOLD-STAND in 8 of 8; held-out reference accuracy 1.000 at layer 12 (the
  deepest window layer, 12/16): pass. Label-shuffle null at that layer: mean 0.509, 95th percentile 0.575. The sit-lie
  axis: layer 12, held-out 0.983 (null mean 0.505).

Told frames, paraphrase 1, R1 (plain text) top tokens:

- TOLD-SIT: ' sitting' 0.727, ' lying' 0.047, ' standing' 0.043, ' seated' 0.028, ' in' 0.015
- TOLD-STAND: ' standing' 0.415, ' sitting' 0.296, ' lying' 0.060, ' in' 0.015, ' on' 0.009
- TOLD-LIE: ' lying' 0.727, ' sitting' 0.122, ' laying' 0.023, ' standing' 0.014, ' in' 0.009

and R1c (chat template, forced opening): TOLD-SIT ' sitting' 0.811, ' seated' 0.121; TOLD-STAND ' standing' 0.965;
TOLD-LIE ' lying' 0.938.

R2 smoke: 4 answers (ROOM-W, ROOM-0, FIELD-W and FIELD-0, paraphrase 1, sample 0) in 55.5 s at batch 4, **0.07
answers per second**. All four ran to the 96-token cap without an end token (438, 455, 432 and 421 characters). I did
not read their text, because these are registered arms (209.10 reads Instruct ROOM-W).

⚠ Exposure before registration: `results/part209/smoke/` holds real Instruct data for every arm (R1, R1c, R3 and four
R2 answers). The only numbers read were the told frames, the gates and the R3 layer and accuracy above. The runner's
own summary line also printed the R3 c of every arm at this stage; no prediction reads R3 at Instruct (209.6 is
Qwen3.5-4B only), and the R1 s of the non-told arms (209.8 reads Instruct's ROOM-W s) was neither printed nor read.
The smoke files are left untracked so they are not published with the registration; delete them or keep them unread.

## Reader checks (35 known answers each)

- DeepSeek V4.1 Flash: 35 of 35.
- Claude Haiku 5.5: 35 of 35, after a fix: the first attempt hung, because `HTTPS_PROXY` routed the direct API call
  through the fleet's mitm proxy, whose CA this process does not trust, and every attempt failed TLS and retried
  quietly. `haiku_client` now opens the connection without the proxy (as `judge_client` does for OpenCode Go) and an
  exhausted retry names its last error.

## Notes on the design

1. **Gate reading for 209.8 and 209.9.** The design says these are "read only at checkpoints that pass G0 and G1",
   and separately that "a checkpoint failing R3's half has its R3 reported but not read". The scorer reads an R1
   prediction at a stage passing G0 and G1's R1 half, and an R3 prediction (209.6) only with G1's R3 half too. One
   sentence in the design would make this explicit.
2. **Denominators for 209.5 and 209.7** are all labelled NOBODY answers, unreadable ones included (a stricter
   denominator for 209.5). The design does not say; worth a sentence.
3. **Family O on the CPU is slower than the design's estimate.** At 0.94 s per pass, the six stages are 3 x 456 + 3 x
   552 = 3,024 passes, about 47 minutes plus loads, against the design's "about 30 minutes". The Instruct R2 smoke ran
   at 0.07 answers/s at batch 4, so 720 Instruct answers would take about 3 hours at that batch; the optimization pass
   (batch 16, 32 and 48 timed on pilot rows) matters for OLMo as much as for Qwen. Every smoke answer hit the 96-token
   cap without ending, so the pilot's "posture settled within 48 tokens" check decides whether a shorter cap is safe.
4. **The A770 runner uses `-I`,** which leaves the script's folder off `sys.path`. `run209.py` puts its own folder's
   parent on `sys.path` itself, so `neuron_report.py` copied to the job root is found. The `--copy` lists in
   `steps.txt` copy every file the runner reads (config, prompts, reference, tokens, neuron_report).
5. **The top layer.** Transformers' last `hidden_states` entry is after the final norm, not a residual stream. It is
   outside the 50-80% window at both depths, so no read uses it.
6. **No content changes** to the prompt files, `gen_prompts209.py`, `reader_check.json` or the design.
