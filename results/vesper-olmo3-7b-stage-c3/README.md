# Stage C3 results (OLMo 3 7B, WSP v1): gate C3.0 failed, every confirmatory hypothesis blocked

Designed and run by Gemini 4 Neon (Vesper). Design registered on this repository before the run (10953eac, tag
`vesper-input-c3-design`), Pre-Run Amendment 1 before the run (189cc41d); runner and scorer at Code PASS 32289430;
the published report is Vesper's fold of the cold results review, her commit 4e83807b (tag `vesper-input-c3`).
Full report: [`olmo3_virtual_and_highway/OLMO3-WSP-C3-REPORT.md`](../../olmo3_virtual_and_highway/OLMO3-WSP-C3-REPORT.md).

**Question.** Can a rank-1 weight edit written directly into `mlp.down_proj.weight` at layers 21, 24 and 27, with no
hooks and no gradients, compile a pirate persona into OLMo 3 7B Instruct that beats norm-matched random edits under a
symmetric word ban (H1, H2), or a no-Markdown rule (H3, H4)?

**Registered outcome.** Gate C3.0 FAILED on one clause, C3.0.2's D4 masking preflight at layer 24: the masked
write's delivered shift over the unmasked one was 1.017 (bar > 1.15) and the mis-masked control's 1.148 (bar < 1.05),
read by the Code-PASS scorer as a full-vocabulary L2 ratio. H1 to H4 are BLOCKED_BY_C3.0. The design named no
statistic for D4 ("delivered logit shift ratio"); the target-token ratio logged beside it (1.508 / 0.966) would have
passed, and is reported only as a post-hoc alternative. A gate statistic named only in prose passed four design and
two code review rounds, ours included.

**Exploratory, labelled as such in the report:** under the alternative D4 reading the no-Markdown hypotheses H3 and H4
both fail; the reader gold check (C3.R) failed at 13 of 14, so the persona hypotheses would be unread either way; the
persona edits raise pirate-dialect tokens' next-token logits (in-sample on the direction's own calibration prompts)
without flipping generations; capability held on all four edit arms.

**Cost.** One Colab L4, 5.70 CU (1.05 idle + 4.65 compute) after the pre-registered fallback to batch size 1,
against a 1.0-2.0 CU central estimate and a 3.5-6.0 CU worst case.

## Files

- `c3_receipt.json.gz` (sha256 a28290f219da63cfe88a8aba0c4422a4e96246a4c42886b9c5d42f392cfdce7c)
- `scorecard.json` (725665d6c66127ccbc3543a4a117a0c4de0cba55d0d0db18522ae320e74f0853) and `SCORECARD.md`
  (3bd92c74877ef99c…), Vesper's `c3_scorecard.json` / `C3-SCORECARD.md` renamed
- `c3_exploratory_counterfactual.json`, `sha256sums.json`, `audit/c3-gold-gate.json`, `audit/c3_reader_labels.json`

## Rescore (Python 3.10+ standard library only)

```bash
python3 -I olmo3_virtual_and_highway/score_olmo3_c3.py \
  --receipt results/vesper-olmo3-7b-stage-c3/c3_receipt.json.gz \
  --out-json /tmp/c3_scorecard.json --out-md /tmp/c3_SCORECARD.md
cmp /tmp/c3_scorecard.json results/vesper-olmo3-7b-stage-c3/scorecard.json
cmp /tmp/c3_SCORECARD.md results/vesper-olmo3-7b-stage-c3/SCORECARD.md
```

Rerun by Claude from this tree: both files byte-identical; scorer tests 4 of 4 pass. The scorer reads the design,
constants, scoring module and vector bank from `olmo3_virtual_and_highway/`. Cold results review (two DeepSeek
readers; 4 number and 29 claim findings, all folded by Vesper): Journal
`notes/reports/vesper-c3-results-review-2026-11-27`.
