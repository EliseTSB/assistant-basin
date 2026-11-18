# Where each high-risk part's scored decision falls, in key positions

Every number below comes from `measure_positions.py` (run with `/home/elisepc/neuron-probe/.venv/bin/python -I
measure_positions.py`; it writes `decisions.csv` and `positions.json` beside itself). It reads the stored answers in
`/home/elisepc/green/worktrees/ab-part180b` and the parts' own scoring rules, tokenises each stored answer with the
Qwen3.5-4B tokenizer, and adds the part's own prompt length.

⚠ **Qualified by Part 184 (2026-11-04, after this table was written).** The rule below treats a decision fixed by key 64 as unchanged by the kernel. Part 184's eager rerun of Part 18 found 13 of 40 differing greedy answers already differing at keys 22 to 59, most likely at near-ties (two kernels' bf16 logits differ slightly everywhere). So a decision fixed by key 64 can still differ at a near-tie. In Part 184 no count changed by more than one; read "decided by 64" as "changed only if a near-tie falls before it", not as "unchanged". `results/part184/README.md` in the research repo.

**Key position** = prompt tokens + the answer-token position at which the decision is fixed. **Safe** = fixed at key
position <= 64: the A770's fused SDPA kernel leaves the CPU float32 reference from key 65 onward and agrees with it
before that, so such a decision cannot have been changed by the fault. A decision that depends on tokens after key 64
might have been.

**Decision type.** *T* (term): a regex decides the label at its first match, so the decision position is the answer
token where that match ends; an answer with no match depends on the whole answer, so its position is the full length.
*W* (whole answer): an LLM reader, a loop/repetition rule, a length rule, or an absence rule ("the name is nowhere in
the answer") -- the decision needs the whole answer, so its position is the full answer length.

**Prompt lengths.** Every part renders its prompt through `neuron_report.build_prompt(tok, q, False)` (the chat
template, thinking off); Part 19's base answers store their own `prompt_ids`, which are used instead. The re-rendered
prompt for "Who made you?" is 16 tokens, the same as Part 19's stored ids. Part 20 stores the rendered prompt text and
the generated token ids, not the answer text; its answer length is the stored generation length.

**Verdict rule.** SAFE = every scored answer decided by key 64; MOSTLY = more than 90%; AT RISK = otherwise;
NOT MEASURABLE = no stored answers to measure (or none that can be measured). "Key cells" are the cells the published
claim's numbers count -- for example the clamped condition that showed the effect and its control.

## Summary

| Part | Model | Type | n scored | decided by 64 | share | median key | max key | key-cell share | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 18 | Qwen3.5-4B | T | 16 | 8 | 0.500 | 48 | 109 | 0.500 | AT RISK |
| 18b | Qwen3.5-4B | T | 24 | 7 | 0.292 | 73 | 112 | 0.292 | AT RISK |
| 19 | Qwen3.5-4B | T | 25 | 24 | 0.960 | 21 | 76 | 0.950 | MOSTLY |
| 20 | Qwen3.5-4B | W | 6 | 1 | 0.167 | 75.5 | 81 | 0.167 | AT RISK |
| 24 | Qwen3.5-4B | W | 192 | 10 | 0.052 | 76 | 80 | 0.052 | AT RISK |
| 39 | Qwen3.5-4B | W | 636 | 141 | 0.222 | 75 | 83 | 0.222 | AT RISK |
| 79 | Qwen3.5-4B | W | 1728 | 449 | 0.260 | 80 | 94 | 0.260 | AT RISK |
| 81 | Qwen3.5-4B | W | 640 | 11 | 0.017 | 114 | 119 | 0.017 | AT RISK |
| 84 | Qwen3.5-4B | W | 120 | 12 | 0.100 | 79 | 82 | 0.100 | AT RISK |
| 84b | Qwen3.5-4B | W | 336 | 28 | 0.083 | 80 | 83 | 0.083 | AT RISK |
| 140 | Qwen3.5-4B | W | 3856 | 188 | 0.049 | 80 | 98 | 0.054 | AT RISK |
| 148 | Qwen3.5-4B | W | 2790 | 645 | 0.231 | 71 | 137 | 0.231 | AT RISK |
| 153 | Qwen3.5-4B | W | 840 | 178 | 0.212 | 77 | 145 | 0.182 | AT RISK |
| 161 | Qwen3.5-4B | W | 9264 | 1110 | 0.120 | 73 | 111 | 0.081 | AT RISK |
| 163 | Qwen3.5-4B | W | 3200 | 551 | 0.172 | 75 | 106 | 0.172 | AT RISK |
| 164 | Qwen3.5-4B | W | 6400 | 1009 | 0.158 | 70 | 110 | 0.158 | AT RISK |
| 165 | Qwen3.5-4B | W | 6912 | 862 | 0.125 | 74 | 98 | 0.134 | AT RISK |
| 167 | Qwen3.5-4B | W | 0 | 0 | -- | -- | -- | -- | NOT MEASURABLE |
| 168 | Qwen3.5-4B | W | 7392 | 1096 | 0.148 | 76 | 110 | 0.149 | AT RISK |
| 169 | Qwen3.5-4B | W | 8579 | 1429 | 0.167 | 70 | 98 | 0.170 | AT RISK |
| 170 | Qwen3.5-4B | W | 6088 | 1067 | 0.175 | 69 | 104 | 0.216 | AT RISK |

All 21 parts are Qwen3.5-4B (revision 851bf6e8), so none is skipped for model. Part 39 also has Qwen3.5-4B-Base
activation-read paths, but its generated answers are from Qwen3.5-4B.

**The one-line answer.** Only Part 19 is close to safe, and even it is not safe on its own key cell (the top-50 zeroing
set's first-sentence rule reads one answer at key 76). Every other part's claim rests on answers whose deciding tokens
come after key 64 in the large majority of cases; for the whole-answer readers and loop rules this is structural (the
reader sees the whole answer, the loop is at the end), and for the term rules it is because the matched term usually
appears late in the answer.

## Per part

**Part 18 (T, AT RISK).** The published claim is that clamping L22 N2742 at 2x the corpus maximum makes 8 of 8 "Who
made you?" sample answers mention extraterrestrial origins against 0 of 8 unclamped, with 0 of 160 random-control
answers. The scorer is `neuron_report.score_text`, which runs `ALIEN_RE` over the answer up to the first role-turn
marker. The claim's key cells are the 8 steered (+2) and 8 unclamped samples on the single sample prompt "Who made
you?" (16 tokens). All 8 steered answers are decided by key 64 (share 1.000, median 22, max 27): the alien word comes
early. None of the 8 unclamped answers is (share 0.000, median 76.5, max 109): with no alien term, the decision is the
whole answer, and the unclamped answers run 53 to 93 answer tokens (key 69 to 109). So the "0 of 8" half of the claim rests on answers
that ran well past key 64. The 160 random-control answers (`controls_sampled`, 20 units x 8) are not in the key cells I
measured but have the same shape: no alien term, so whole-answer decisions. Verdict AT RISK because the control half is
not safe.

**Part 18b (T, AT RISK).** The claim is that a generation-only clamp gives alien wording in 7 of 8 answers against 0 of
8 for a prompt-only clamp. Same scorer and sample prompt. The generation-only +2 cell is 7 of 8 decided by key 64
(share 0.875, median 27.5, max 80); the one answer not decided by 64 is the one whose alien term comes late. The
prompt-only +2 control cell is 0 of 8 (share 0.000, median 77.5, max 112) and the unclamped cell 0 of 8 (0.000, median
76.5, max 109): all whole-answer decisions. Verdict AT RISK.

**Part 19 (T, MOSTLY).** The claim combines identity effects from generated answers with activation and gradient
evidence; the generated part is 19.3: the top-50 zeroing set removes the identity name from the first sentence of
every unclamped identity answer that had it, and the three random sets keep it on every one. The rule is the first
sentence cut (`circuit.py:219`, `score_part19.py`), then `qwen|tongyi|alibaba|通义|千问|阿里` in it. The claim rests on
5 unclamped answers and on the answers to those 5 questions in the top set and the three random sets (5 each). 24 of 25
are decided by key 64 (share 0.960, median 21, max 76). The one that is not is a top-set answer that runs to the
60-token cap with no identity term in its first sentence, so its decision is the whole 60-token answer (key 76) -- the
rule the claim is about (removing the name) reads that answer only after key 64. The unclamped answers and all three random sets are decided by key 23 to 45. Verdict
MOSTLY: 96% overall, 95% in the key cells, and the single failure is in the cell whose change the claim reports.

**Part 20 (W, AT RISK).** The claim compares saved identity/replay output (the replay reproduces the generated ids
exactly) while most seam measurements come from activation reads. The stored `identity.json` holds the generated token
ids and the rendered prompt, not the answer text; the decision that a claim rests on is "the whole generated answer is
identical", so it is whole-answer by construction. 6 prompts, 1 decided by key 64 (share 0.167, median 75.5, max 81):
answers run 29 to 64 tokens on 16- to 20-token prompts. Verdict AT RISK. The seam path uses a full-sequence forward with
`use_cache=False` (`part20.py:100-110`), which is a different kernel path from cached decode; this measurement covers
only the generated answers.

**Part 24 (W, AT RISK).** The claim is that the alien neuron and the identity neurons add up, resting on sampled free
answers (8 per condition, 2 questions, 4 doses, 3 zero conditions) labelled by hand in `alien-labels.json`. A hand read
of the whole answer is a whole-answer decision. 192 answers; 10 decided by key 64 (share 0.052, median 76, max 80) --
answers run to the 64-token cap on 16- to 17-token prompts. Within the key cells the identity-50 set at k=2 (where the
additive effect shows) is 6 of 64 (share 0.094), none 2 of 64 (0.031), random-50 2 of 64 (0.031). Verdict AT RISK.

**Part 39 (W, AT RISK).** The claim (the addressee unit L6 N753 also responds to AI talk) is made alongside activation
readouts; the answer side is maker naming and breakage in `push.jsonl` and `zero.jsonl`, computed over the whole answer
by `part39.py`. 636 generation rows; 141 decided by key 64 (share 0.222, median 75, max 83). Every cell is between
0.150 (push control) and 0.266 (push target). Verdict AT RISK.

**Part 79 (W, AT RISK).** The claim is that Lyra's evil push and the zeroed maker neurons do not compose: the villain
still calls itself Qwen but names a company in 1 of 192 answers (e3) against 182 with the makers zeroed alone (e1),
with 0 to 60 for five control pairs. Every count is a whole-answer label: the Haiku reader's maker/coherent/villain
labels, or the fixed Alibaba/Tongyi regex (`part79/extra_counts.py`). 1728 held-out answers; 449 decided by key 64
(share 0.260, median 80, max 94). The key cells: e3 64 of 192 (0.333), e1 22 of 192 (0.115), e0 17 of 192 (0.089); the
controls span 0.000 (e7) to 0.823 (e6). e6's high share is not a fault effect: its answers are short and end at eos
(173 of 192), e.g. "I was created by OpenAI." in 8 tokens (key 24). Verdict AT RISK. The README itself says most e3 answers stop
at the 64-token cap, so the company count is a lower bound -- which is exactly the regime where the fault is live.

**Part 81 (W, AT RISK).** The claim is that the conspiracy push reproduces (40 of 96 topic answers) and is specific
(0 for three size-matched controls) but breaks the model about as often as the controls. Reader labels (Haiku,
conspiratorial/coherent/maker) are whole-answer judgements. 640 answers; 11 decided by key 64 (share 0.017, median 114,
max 119) -- 96 new tokens on 18-token prompts, so nearly every answer runs past key 64. k1 (the push) is 0 of 128. The
reader label file (`results/part81/reader/reader_labels.jsonl`) is not in this tree, so labels are not joined here; the
decision position does not depend on them. Verdict AT RISK.

**Part 84 (W, AT RISK).** The claim is that at Lyra's dose 0.63 the alien neuron works without steering the first answer
token (24 of 24 off-world in A2 and A3, 0 of 24 in A0 and the control C). The Haiku origin rubric reads the whole
answer. 120 answers; 12 decided by key 64 (share 0.100, median 79, max 82). A3 (which also steers the first answer
token) is 6 of 24 (0.250), A2 3 of 24 (0.125), A0 1 of 24 (0.042), C 0 of 24. Verdict AT RISK. The comparison the claim
makes -- A2 against A3 -- is between two cells that are both mostly decided after key 64.

**Part 84b (W, AT RISK).** The claim is that below the ceiling, steering the first answer token added no detectable
amount (decode against gen at doses 0.40, 0.47, 0.55). Same rubric, 16 samples per cell. 336 answers; 28 decided by key
64 (share 0.083, median 80, max 83). The cells the claim contrasts, decode0.55 (7 of 48) and gen0.55 (10 of 48), are
the two most-decided cells and still only 0.146 and 0.208. Verdict AT RISK.

**Part 140 (W, AT RISK).** The claim is that at Part 86's dose, arm A reaches 48.3% unit-level joint success on Set P
while arm C reaches 5.8% (and Set C joint success A 1.7%, C 0.0%). Joint success is the DeepSeek reader's
coherent+persona label (`score140.joint`), a whole-answer judgement; the repetition rate is `score140.repeat`, also
whole-answer. 3856 answers; 188 decided by key 64 (share 0.049, median 80, max 98) -- 80 new tokens on 16- to
17-token prompts. The recomputed labels reproduce the headline (A/P joint 116 of 240 answers, C/P 14 of 240; A/L
repeat 91 of 120). The key cells A/P (28 of 240, 0.117) and C/P (3 of 240, 0.013) are both far below 64. The unit list
`part140/units140.json` is not in this tree, so the unit-level averaging is not recomputed here; n counts answers.
Verdict AT RISK.

**Part 148 (W, AT RISK).** The claim is that on the 465 atlas neurons 348 move the name and 191 the maker coherently,
146 move both, and paired random directions move the name 175 times. Movement is the registered regex (Qwen|Qwen3|
Tongyi on "Who are you?", Alibaba|Tongyi|Qwen on "Which company made you?"), filtered by the reader's coherence label;
a regex can match late and the reader reads everything. 2790 answers (465 neurons and 465 random directions on 3
questions); 645 decided by key 64 (share 0.231, median 71, max 137). The neuron cell is 392 of 1395 (0.281), random
253 of 1395 (0.181). The reader label file (`results/part148/labels.jsonl`) is not in this tree, so only the regex is
recomputed; the decision position does not depend on the reader. Verdict AT RISK.

**Part 153 (W, AT RISK).** The claim is that ten neurons selected in Part 148 name their company reliably on 24 maker
phrasings (151 of 240 hits) against paired random directions (27 hits). A hit is `company_hit` (the company name
anywhere, not triage-broken) or `guard_hit` (the expected word anywhere); both are whole-answer regexes. 840 answers;
178 decided by key 64 (share 0.212, median 77, max 145). The recomputation reproduces the neuron/maker count (151 of
240). Key cells: neuron/maker 44 of 240 (0.183), random/maker 29 of 240 (0.121). Verdict AT RISK.

**Part 161 (W, AT RISK).** The claim is that at their picking dose 33 of 47 picked random directions were reliable
(70.2%) against 25 of 37 picked neurons (67.6%). A hit is `company_hit`; reliability is a per-unit count of hits across
24 maker prompts. 9264 ladder answers; 1110 decided by key 64 (share 0.120, median 73, max 111). Restricted to the
answers at each unit's picking dose (the claim's cells), 125 of 1536 (share 0.081, median 80, max 110): neuron/pick 38
of 408, random/pick 87 of 1128. Verdict AT RISK.

**Part 163 (W, AT RISK).** The claim is that, under Part 161's `\bQwen\b` rule, neurons say Qwen in 19 of 539 maker
answers (3.5%) and random directions in 57 of 661 (8.6%), and neurons break 21.4% against 5.2%. Name-gone and
maker-gone are absence rules ("none of Qwen, Tongyi, 通义, 千问 anywhere"), and breakage is `triage86`, whose trigram
loop rule reads the whole answer. 3200 battery answers; 551 decided by key 64 (share 0.172, median 75, max 106). The
recomputation reproduces the breakage (343 of 1600 neuron, 83 of 1600 random). Verdict AT RISK.

**Part 164 (W, AT RISK).** The claim is that on fresh neurons at +0.8 the neuron/random Qwen ratio is 0.79 (164.1
failed) and neurons break 3.5 times as often, 27.0% against 7.8%. `qwen_hit` is `\bQwen\b` anywhere and
`qwen_eligible` excludes triage-broken answers -- both whole-answer. 6400 battery answers; 1009 decided by key 64
(share 0.158, median 70, max 110). neuron 301 of 1600 (0.188), random 708 of 4800 (0.147). Verdict AT RISK.

**Part 165 (W, AT RISK).** The claim is that at equal delivered write size, single neurons and random sets of four
broke about 26% of answers, sixteen 15%, and 64 or more about 9% (n+ / full ratio 2.70). `primary_broken` is triage86
broken or a three-word repetition three times over the whole answer; a loop is only visible at the end. 6912 battery
answers; 862 decided by key 64 (share 0.125, median 74, max 98). The recomputation reproduces the headline (n+ 197 of
768, full 73 of 768). Key cells n+ 124 of 768 (0.161), n± 104 of 768, full 77 of 768 (0.100), iso 106 of 768. Verdict
AT RISK.

**Part 167 (W, NOT MEASURABLE).** The part stopped at its plain-answer gate and reports no prediction readings, so
there is no claim resting on scored answers. No answer file is in the tree (`results/part167` holds `README.md`,
`launch.jsonl`, `score.json` only) and the README says the rows were kept out until Part 168 was scored. The one
answer the README quotes (phrasing 13's plain answer, a loop) has no stored row to measure. Verdict NOT MEASURABLE
because there is nothing to measure, not because the measurement failed.

**Part 168 (W, AT RISK).** The claim is that banning a push's own words cut loops by 31.4% and a mass-matched ban
raised loops by 16.5%. Loop rate is `score166.loop_phrase` (a three-word sequence three times, anywhere in the answer)
and broken is `primary_broken`; both need the whole answer. 7392 battery answers; 1096 decided by key 64 (share 0.148,
median 76, max 110). The recomputation reproduces the loop counts (n+/own 62 of 672, n+/none 86 of 672, n+/mass 105 of
672). Key cells are the n+, k4 and full arms under none/own/mass, all between 0.141 and 0.161. Verdict AT RISK.

**Part 169 (W, AT RISK).** The claim is that banning a push's own top 50 tokens cut loop answers by 37.9% on new
neurons and phrasings. Same loop rule. 8579 battery answers; 1429 decided by key 64 (share 0.167, median 70, max 98).
Recomputation: n+/own 94 of 768, n+/none 167 of 768. Key cells n+/own 114 of 768 (0.148), n+/none 107 of 768 (0.139),
k4/own 96 of 768, k4/none 114 of 768. Verdict AT RISK.

**Part 170 (W, AT RISK).** The claim is that blocking the plain answer raises loops under a push (25.0% to 43.8% at
dose 1) and a 200-token ban raises them without one. Same loop rule. 6088 push and no-push answers; 1067 decided by key
64 (share 0.175, median 69, max 104). The cells the claim contrasts are the most-decided in the set: n+/b0 135 of 1152
(0.117), n+/b50 307 of 1152 (0.266), n+/b200 354 of 1152 (0.307), no_push/b200 44 of 200 (0.220). Verdict AT RISK. The
dose is not split out in this measurement; the dose-1 cells are a subset of n+/b0 and n+/b50.

## What the numbers do and do not say

- A key position above 64 does not mean the answer was wrong. It means the decision *could* have been changed by the
  fault, and only a rerun on a correct kernel (eager, or SDPA with `SPDBackend.MATH`, as `part178/diag178i.py` showed)
  can settle it. The measured drift is sparse -- six confident departures at batch 1 on 4B, at positions 65 to 73 --
  so a long answer is not automatically a changed one.
- The parts whose answers end at eos early (Part 79's e6 control, Part 140's F arm) have shorter answers and so more
  decisions by key 64. That is a property of the answers, not of the fault.
- The re-tokenisation check in the script compares each stored answer's text, re-tokenised, with the stored generated
  token count. The re-tokenised text is never longer than the stored count except for 2 of 336 answers in Part 84b and
  1 of 9312 in Part 161 (by one token), so the stored answer text is complete and the answer-token positions are
  measured on the full answer. The stored count is usually one or two larger than the re-tokenised text because the
  trailing eos is dropped by `skip_special_tokens`.
