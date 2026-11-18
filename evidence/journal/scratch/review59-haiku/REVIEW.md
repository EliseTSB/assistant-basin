VERDICT: FIX-FIRST

Scope: cold review of the Part 59 runner in /home/elisepc/assistant-basin/part59 against PART59-DESIGN.md, reading the Part 26/54 code it reuses. Nothing was run on a GPU, no model was loaded, and no tarball was built.

## Findings (most severe first)

1. HIGH. OLMo phases cannot run on the rented box. RUNBOOK-59.md:21-26 packages results/part26/target-olmo2-1b.npz but not target-olmo2-1b.json. run59a.py:32, :79 and :122 read that JSON in _olmo_maxima and _input_shas, and run59b.py reaches it through run59a.load_maxima (run59b.py:320). Scenario: the OLMo model loads, then FileNotFoundError at run59a.py:79. olmo59a and olmo59b both fail, run59.sh continues, and RETURN_READY prints. Result: no 59c, and no OLMo 59B. Fix: add results/part26/target-olmo2-1b.json to the tar list at RUNBOOK-59.md:25. Also preflight every required file in run59.sh before the first GPU phase.

2. HIGH. The symmetric rule is not the design's rule, and 59f/59g cannot be scored. analyze59b.py:40-41 takes np.percentile(axis=0) over 3 control values per neuron, which is effectively the max of the 3. The leave-one-out control counts (lines 61-67) use 2 values, so each control faces a different bar than the target. I simulated exchangeable changes over 82,944 neurons (scripts in the worker-02 output dir). The target shows 2,604 symmetric neurons with independent doses and 6,643 with r=0.5. Leave-one-out control counts are about 4,600-4,760 (about 10,000 at r=0.5). So 59f (>=20) is met by chance and 59g (<5) cannot be met. A pooled P99 over all control |change| at each dose gives 2 and 72 under the same null. Design wording "the 99th percentile of the same quantity under the random controls" most naturally means the pooled reading. This needs Elise's decision before the run, and the test at test_part59.py:159-173 encodes the per-neuron reading.

3. HIGH. The registered design misstates Part 54's dose scale. PART59-DESIGN.md:5 says "multiples of the neuron's largest activation on the alien frames". Part 54 used the Pile-record maximum: results/part54a/clamp-qwen35-4b-L22u2742-x1.json has clamp_value 19.125 (the Pile max), with the alien max 24.69 reported only as a ratio. The ladder manifest scale_definition says the same. The code is right (run59a.py:193, :299). The design parenthetical is wrong and should get a dated correction, not a prediction change.

4. MEDIUM-HIGH. run59.sh can print RETURN_READY after a failure. Lines 24-33 record failures in run_status and continue. Lines 52-53 print the marker whenever the tar succeeds, and lines 58-59 print run_status=1 after it. A Qwen 59A OOM still runs 59B and the analysis, about 3-4 hours and roughly $1, and the partial bundle carries the marker. Fix: print RETURN_READY only if run_status is 0. Also check the exit of fetch_model (lines 36-37, unchecked), and assert transformers.__version__ == "5.20.0" after the pip install (line 13 prints only).

5. MEDIUM. Stale locks after a kill. run59a.py:162-178 and run59b.py:286-302 remove the lock only in finally. An SSH drop (SIGHUP, and the RUNBOOK runs the script in a foreground ssh session), SIGKILL, or the OOM killer leaves .part59a.lock or .part59b.lock. The rerun then fails at run59a.py:166 with "holds the output lock", and RUNBOOK does not say to delete it. Fix: write the pid and reclaim the lock when that pid is dead, or document the rm and run under nohup or tmux.

6. MEDIUM. The blind reader can score a partial 59A run. blind_setting.py:88-103 globs every clamp-*.json and ignores ladder59a-*.json status. A run killed partway through OLMo would be submitted and tallied with smaller n and no refusal. Fix: require manifest status "complete" and the expected 60 files per model in submit (blind_setting.py:168-171).

7. MEDIUM. Calibration is weak. The 28 rows (blind_setting.py:38-67) are single cued sentences, while the answers are roughly 60-token paragraphs. Two OTHER items are contestable: "made-up island" (line 61) and "beyond the edge of the known world" (line 58) could be FANTASY. Two disagreements are the whole allowance, and a third refuses the batch. Vague openers, where the SCIFI/OTHER errors happen, are not tested. Fix: add cut-to-60-token calibration paragraphs and drop the contestable items, or disclose the limit.

8. MEDIUM. Interpretations not in the design, all unrecorded. "Later layers" means layers strictly after the clamp layer (run59b.py:89 and :321). "Strongest" for 59h means min(|change_-1|, |change_+1|) (analyze59b.py:90-92). Both should go in a dated design note.

9. MEDIUM. Test gaps. There is no 59B kill-and-resume test (the brief asked for resumability; test_part59.py:199-244 covers 59A only). Nothing checks control draws against Part 26's values (152/376/692 for Qwen, 135/335/615 for OLMo, run59a.py:50-52). The 59B clamp test (lines 175-197) only asserts "not allclose", so a wrong clamp value passes. Calibration is tested only at 3 wrong, never at the boundary of exactly 2 (scores) versus 3 (refuses). The symmetric test passes for the per-neuron reading only.

10. LOW. analyze59b.py:166-208 writes nothing until the logit-lens and window step finishes. If the model reload fails, the 59f/59g counts are lost. Write the classification JSON first. Line 103 (odd_mask self-assignment) is a no-op.

11. LOW. run59a.py:182 and run59b.py:308 load the model before the complete and identity checks, so a no-op rerun costs a model load.

12. LOW. Git hygiene. part59/review/ holds the condition key (setting-key-*.json) and is not in .gitignore, so git add -A would commit the blinding key. blind_setting.py:70-71 also hardcodes ~/.mitmproxy/ca-bundle-combined.crt, and RUNBOOK does not say where ANTHROPIC_CREDITS_KEY comes from.

13. LOW. The control draw is the three lowest-indexed of 20 random units (neuron_report.py:147-151 sorts the sample, and run59a.py:52 takes [:3]). This matches Part 26 and 54, but the controls are concentrated at low indices, so "random same-layer" is weaker than it reads.

## Answers to the brief's checklist

1. Dose scale: the Pile maximum matches Part 54 (verified against results/part54a). The hook point matches Part 54 (down_proj pre-hook, all positions, absolute value via NeuronHook, same as part26._generate_condition). OLMo constants: layer 10 offset 81920 + 6978 gives 6.5, and layer widths are 8192. Qwen maxima 19.125 and controls 19.77/13.51/2.18 match the cache. The seed-0 draw at width 9216 reproduces 152/376/692, which is indirect evidence the width is 9216. The design text is wrong (finding 3).
2. Seeds: controls use the Part 26 path (nr.control_units(width, unit, 20, seed)[:3]). Baseline and clamped runs use separate condition seeds, as in Part 54. Control draws are not tested (finding 9).
3. Predictions: 59a, 59b, 59d and 59e are scorable from the blind table with --by-cond. 59c is blocked by finding 1. 59f and 59g are not sound (finding 2). 59h is manual, with a ranking choice (finding 8).
4. Failure paths: OOM is caught (run59a.py:254, run59b.py:411), the manifest is marked incomplete, and the exit is non-zero. Partial archives raise on NaN (analyze59b.py:54-55). Changed arguments are refused via the identity checks. A model that fails to load raises before the try, and the lock is released. The RETURN_READY problem is finding 4.
5. Blind reader: the condition is hidden, and refusal happens before the table prints (blind_setting.py:221-223). Calibration limits are finding 7.
6. Memory: streaming at batch 1 is sound. Peak is about the fp32 weights plus one [1,T,9216] layer copy, about 12 MiB each. The CPU holds only the tokenized docs. Not measured.
7. Other: no secrets are printed. Shell quoting is clean. The machine-specific paths are finding 12.

## What I ran
- The unit tests: 8 pass with /home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part59.
- Sabotage: flip-sign gives 1 != 2, drop-control-percentile gives 3 != 2. Both red, as REPORT-runner.md says.
- Null simulation through analyze59b.classify_changes (numpy only, no GPU), plus a pooled-threshold variant. Scripts are in the worker-02 output dir.
- Constant checks against results/part26 and results/part54a.

## What I could not check
- No GPU run, no model load, no tarball build. Finding 1 is from reading, not execution.
- Memory and timing estimates are unmeasured. The "11 minutes for 480 answers" claim has no source in the repo; the ladder manifest and run.log contain no timings.
- Qwen3.5-4B layer count and width were not read from its config (inferred, see item 2 above).
- Calibration labels were judged by reading. I did not call the Batch API.
- The simulation tests the rule's null behaviour under exchangeability, not its power on real activations.
