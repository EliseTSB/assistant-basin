# Whole-repository review: issues, missed implications, new directions

Reviewer: Claude, as a Claude Code cloud session, 2026-11-03, reading the checkout at commit 0586799 with no model runs.
Scope: `README.md`, `PARTS.md`, `PROCEDURE.md`, `TOOLKIT.md`, `STANDARD-CONTROLS.md`, `EPISTASIS-FRAMING.md`,
`PRIOR-ART.md`, `docs/FRONT-PAGE-2026-10-30.md`, the Part 15, 16b, 25, 27, 31, 32, 46, 86, 139, 140, 171 to 178
designs and READMEs, `part86/common86.py`, `neuron_report.py`, and the registered rows for Parts 127 to 178 in
`PREDICTIONS.md`. Numbers below are copied from those files; nothing was recomputed. Where I read a result into a
mechanism, the sentence says so, and the claim is mine, not the repository's.

## 1. Issues

### 1.1 The Arc bf16 attention fault: isolate it, file it, and gate on it

Part 178 found that Qwen3.5 in bf16 with the default SDPA kernel on the A770 leaves a CPU float32 greedy path once
the cache holds more than 64 keys, worst at batch 1, and the front page lists 22 high-risk parts including Part 25
under Finding 1. The plan is to rerun them. Three things would make that cheaper and more conclusive.

- **No backend isolation has been run.** The repository has no use of `torch.nn.attention.sdpa_kernel` or
  `SDPBackend`. Running the `diag178f` prompts with the SDPA math backend forced, and with each XPU fused backend
  disabled in turn, would say whether the fault is a fused-kernel bug or precision. The `diag178f` docstring already
  notes that Qwen3.5's linear attention uses chunks of 64, and Qwen3.5 is a hybrid stack (Gated DeltaNet layers with a
  gated full-attention layer every fourth block, per `third_party/honesty-heads/code/engine.py`). Eager attention only
  changes the full-attention layers, and `diag178g` still found eager bf16 leaving the float32 path at confident
  positions in 2 of 16 rows. A per-layer-type check (full-attention layers in float32 with linear layers in bf16, then
  the reverse) would say which block carries the residual divergence.
- **The silent NaNs may share the cause.** Parts 142 and 154 recorded rows of `!!!!` and one run with 92% NaN that
  vanished on rerun. Intel's `torch-xpu-ops` issue #5643 reports the fused XPU SDPA path writing past its output buffer
  in bf16 (q_len 16, head_dim 64), which silently corrupts neighbouring tensors; PyTorch issue #199945 reports XPU
  SDPA returning NaN for fully masked rows and poisoning left-padded batched generation. Neither matches the exact
  trigger here (q_len 1, more than 64 keys, batch 1), but out-of-bounds writes from a fused kernel would produce
  exactly the pattern seen: nondeterministic, batch-shape-dependent, sometimes NaN, sometimes a different token. A
  minimal repro from the eight `diag178f` prompts is worth filing against `intel/torch-xpu-ops`; it also protects
  other single-GPU neuron studies the project cites.
- **The default is still the faulty kernel.** `neuron_report.load_model` calls `from_pretrained(..., dtype=bfloat16)`
  with no `attn_implementation`, and almost every runner loads through it; only Part 178, the Neuron Lens code and two
  OLMo lens scripts set eager. Make the kernel a shared, recorded choice, and add a prelaunch gate: a fixed set of
  prompts decoded greedily on the Arc must match a stored CPU float32 reference at every position where the reference
  margin exceeds a threshold. Part 178 built every piece of this; it is not yet in `tools/part_check.py`.
- **Prioritise the rerun by load.** Finding 1 rests on Parts 25 and 66; Finding 3 on Parts 18, 54 and 62; Part 84 on
  the dose conversion. Rerunning those cells on CPU float32 (the 0.8B is fast on CPU; the 4B at 192 answers per arm
  is hours) gives kernel-independent numbers for the headline claims before the broader Arc rerun.

### 1.2 Greedy single answers per cell make every rate a draw of rounding noise

Parts 123 and 161 record that a bf16 greedy answer on the Arc depends on batch shape, and Part 140's Gate 2 had to
be cut to the first ten tokens for the same reason. Most persona and maker rates are one greedy answer per unit and
question. A rate of "11 of 24" is then partly a count of which rounding realisation each row fell into, and the
kernel fault is only the largest instance of that. The repository already has the alternative where it matters most:
teacher-forced log-odds on registered target strings (Parts 106, 119c, 120, 143 to 147), with probability mass summed
across spellings (toolkit J). Making that the primary outcome, with greedy answers kept for readers, would also make
results robust to the kernel question, since only confident margins move the registered number.

### 1.3 Pooled tests on correlated answers

Only 3 of the 104 result READMEs mention a prompt-level or cluster bootstrap (Part 171 registered one). Most Fisher
tests pool eight seeds of the same phrasing or 24 phrasings with one answer each. The README's Limits section says
this; the designs mostly do not act on it. A standing rule, "every rate carries a cluster bootstrap over prompts, and
over units when units are the sample," belongs in `STANDARD-CONTROLS.md`.

### 1.4 The atlas inherits a reader that disagrees with the others at chance

Haiku built the 465-unit Qwen atlas. Under the corrected brief its agreement with DS and Luna is κ 0.109 and 0.051,
while DS and Luna agree at κ 0.814. The DS re-read kept 245 of 387 first-reader "another AI" units as coherent
personas and called 141 "still Qwen or an assistant". Part 118's marks are Haiku alone (14 against 4 for DeepSeek on
43 sampled answers), and Part 92b's second reader kept only 8 of 18 concept labels. Parts 112, 115, 117, 117b, 118,
148 and 152 start from the first-reader atlas. The correction exists but has not been propagated: rerunning those
parts' scorers on the 245-unit subset, or at least reporting both, is cheap and should precede any further use of
the atlas. Part 55 chose Haiku on agreement with Opus for a different brief; reader qualification should be per brief.

### 1.5 Part 86's ratio is not established

Only triage-changed answers were read; the miss sample puts the pooled real-neuron total at 2.7 to 6.4 times the
counted total, and the random-direction miss rate is not known at the same resolution, so the 4.63 ratio behind
Finding 7 is a ratio of two lower bounds with unknown slack. A seeded read of all answers at layer 22 only, kept and
broken, for both populations (about 1,000 labels) would close this.

### 1.6 Front page and index are stale against the parts that changed the reading

`README.md` says the findings cover Parts 1 to 119c; `PARTS.md` ends at Part 127. Parts 148 to 176 materially change
how Findings 1 and 7 should read (section 2 below), and the lessons log shows the newest parts were the ones most
likely to teach something. The front-page audit machinery in `results/frontpage-audit-2026-10-27` exists; it should
run again once the kernel rerun lands.

### 1.7 Cadence against verification

The lessons log since Part 139 records a defect caught in nearly every part: runners that pushed nothing, scorers
that recomputed an amended gate, typed numbers, a control that was not random. The procedure is catching them, which
is the point, but the rate (49 commits on 2026-11-02; Parts 154 to 178 in four days) is at the edge of what cold
checks can hold. A replication week would measure this directly: pick five load-bearing parts (25 and 66, 86, 92,
119 and 120, 139), have a fresh agent re-execute each from its design file alone on the corrected kernel, and count
exact reproductions. The repository has only one such re-execution (Part 86's replay).

### 1.8 Scope drift

Parts 48 (kiki/bouba), 50 (honesty), 162 (DeepSeek style), 177 and 178 (MIT thinking traces, `wait` openers) each
answer a post on X. They are run well, but they take rig time and cold-check capacity from the identity line, and
Part 178 is where the kernel fault surfaced. That is an argument for keeping a sidecar for reactive parts and
holding the main line's rig time.

### 1.9 Smaller technical notes

- `AddRowsHook` adds the push in the activation's dtype, so the push itself is rounded to bf16 before the add, and the
  realised write norm is recorded only at the first decode step. Part 165's lesson that "equal intended write norm is
  not a delivered dose" argues for computing the sum in float32 and casting once, and for logging the realised norm
  per step (one scalar per call).
- Dose is on at least four scales across parts: multiples of a corpus maximum (Parts 18, 54, 62), a fraction of the
  median residual norm (Part 86 onward), Lyra's rule (Part 84), and multiples of a unit's natural difference
  (Part 171). `TOOLKIT.md` should carry a conversion table, and every design should report the realised write norm
  beside the ratio to the unit's natural write, so a reader can see how far outside its range a unit is pushed.
- `git` here is a shallow clone, so `docs/BRANCHES.md` could not be re-checked; the audit's two DELETE recommendations
  (figs-prototype, part43-singleton) still stand as written.

## 2. Implications the results support but the write-up does not draw

### 2.1 The Google fallback is the model's prior, and the four neurons are the cheapest way to expose it

The pieces are spread across fifteen parts and never joined:

- Zeroing the four gives Google (Parts 25, 66), and with them zeroed the maker's CEO is "Sam Altman" and the country
  "USA" (Part 27). The fallback is "a US lab", not Google in particular.
- The two late units that carry the fallback write Google (L27 N3325) and Apple and Microsoft (L24 N3949) (Part 71).
- Among OpenAI, Anthropic, DeepSeek and Google, only Google is a maker role the model will adopt under seven wordings
  (Part 171).
- Random-valued pushes that keep a picked set's layer profile name Google on 22, 26 and 36 of 44 with none broken
  (Part 173); 28 of 80 layer-matched random directions led with Google and none with any other name (Part 174); 30 of
  40 flat random directions at 0.225 were Google-led with a median of one broken (Part 176).
- The base model's own fallback is OpenAI (Part 27: OpenAI 5 of 8 unzeroed, 6 of 8 zeroed), so chat training moved
  the fallback from OpenAI to Google.
- In Chinese the four units do nothing (Part 32: 0 of 24 against 16 of 24 in English), and the Jacobian lens says L7
  N1169 and L11 N8868 write China-family tokens (Part 111).

My reading: Qwen3.5-4B has a second attractor for "who made me", a US lab with Google first, that any sufficient
erosion of the Alibaba evidence exposes. The four units are not where the fallback lives; they are the lowest-norm
route to eroding the model's own answer, and part of that evidence is a country feature. The Chinese result then
follows: a Chinese prompt supplies the China evidence itself, so zeroing the four cannot remove it. This is one
hypothesis with three cheap tests: (i) with the four zeroed, a push on a China direction fitted from
Chinese-versus-US-company contexts should restore Alibaba or move to Baidu, Huawei or Tencent, whose neurons Parts
155 and 156 already hold; (ii) the fallback distribution under random directions should match the base model's
maker prior with Alibaba removed; (iii) in Chinese, zeroing the four plus a negative China push should flip to Google.

### 2.2 The fallback is a provenance fingerprint, and OLMo lets you check it against public data

Part 15 counted about 1,180 "I'm DeepSeek" documents in Dolci-Think-SFT's R1 subsets and none in OLMo 2's data; Part
9 found OLMo 3 Think naming DeepSeek in 16 of 23 raw answers; Part 16b found OLMo 2 1B's fallback under identity
ablation is "a person", not another lab. So on the one family with public data, the second identity a model falls
to tracks the second-most-frequent identity in its post-training data. The Part 174 to 176 census, run on OLMo 3
Instruct and Think, OLMo 2 1B, Qwen3.5-0.8B, Qwen3-8B and Llama 3.1 8B, would say whether the fallback distribution
matches documented teacher mixes where they are documented. If it does, the perturbation census is a cheap
distillation-forensics instrument, which is a different payoff from the style-based approach of Parts 29, 30 and 162.

### 2.3 Norm, not direction, may carry much of the persona effect, and the decisive control has not been run

Part 140's norm-preserved arm F gave 0% joint success on Set P against 48.3% for the same vectors held; Part 118
found few personas at half dose and none coherent at double; Part 176 found naming within one rung of collapse; Parts
145 to 147 found pushes move the next word toward Instruct with the question removed, with neurons and random
directions changing the words alike (0.99); Part 139 found the apology opener is the model's "cannot read the prompt"
default and random directions rise beside neurons; Part 112 gave random directions persona labels on 101 of 343
"Who are you?" answers. These converge on Malla et al.'s account: a large push mainly moves the state off its shell
and the model falls to defaults, of which "a persona on Who are you?" is one. No part scales the residual stream
with no direction change. The control is one hook: multiply the layer output at decode positions by a factor chosen
to match the pushed norm, and read personas, Google and apology. If pure scaling produces them, the atlas needs
rebuilding on norm-matched pushes; if not, direction matters and arm F's attenuation explains its 0%. Either answer
reorganises Findings 7 and 8.

### 2.4 Pushed personas are not states; they are forced outputs

Part 31.5 (clamped cache against re-encoded cache: the same returns in every stage and K) and Part 125 (cache kept:
2 of 29 against a twin's 3 of 29; push every turn 11 of 29) found the same thing ten days apart, and Part 140 found
release after 8 tokens gives 5.8% and after 24 gives 37.9%. Part 31.1 found the state back in its band within 20
tokens while the text stayed in register for 50 or more. Nothing persists except through the generated text being
re-read. The contrast is the one-neuron owl (PRIOR-ART, 2026-10-29): a trained neuron that fires on template tokens
and installs a persona that persists. The repository has the push side in depth and the trained side not at all.

### 2.5 Neurons are not a privileged basis for identity outcomes; a few units and the content are

Part 92: 1.3% against 1.1% flips. Part 148: readable maker moves level at 191 neurons and 197 random directions.
Part 161: picked random directions as reliable as picked neurons, 33 of 47 against 25 of 37. Parts 163 to 165: single
neurons break answers 3.5 times as often as random directions of the same write size. Part 171: 100 picked neurons
below 100 random neurons. What is neuron-specific is which persona or company appears (Part 97: 10.8% against 1.6%
shuffled; Part 98: 48.2% against 4.8%; Parts 153 and 156: company neurons at 63% against 11%) and a handful of units
(N2742, L27 N3325, L24 N5017, the four). Mayner's result, cited in PRIOR-ART, says the privileged object is a sparse
set of 10 to 100 neurons. Finding 7's 2.7% against 0.6% is the weakest form of the claim and a lower bound; the
defensible claim is content specificity, and Findings 1 and 7 should be rewritten around it.

### 2.6 The interaction measure is curvature until the self-term is subtracted

Part 120 showed a push interacts with itself as much as with a second push, with a square law below dose 0.1 and the
cross term at about 0.28 of the self-term scale. So Part 119's interaction share is mostly curvature of the logit
map, not unit-pair epistasis, and the genetics framing needs its scale fixed first: define epistasis against the
single-push dose-response curve (compare the pair at (d, d) with a single push at the effective dose giving the same
first-order effect). The planned Walsh decomposition of a full factorial has not been run; the four maker units with
L24 N3949 and L27 N3325 make a 2^6 design of 64 cells, teacher-forced, which is where "additive or interacting" is a
mechanistic question rather than curvature.

### 2.7 "Concept" is "a feature before post-training"; the six-question count is the weaker definition

The sort is dose-dependent (Part 118), reader-dependent (Part 92b), and one selected unit's words contradict it (Part
113). The one correlate that holds across parts is base transfer: concept units work at mid-training or base (Parts
90b, 91, 92, 94, 95, 96) and slot units mostly do not. Defining the kinds by base transfer, which is already measured,
and dropping the question-count rule, would make the taxonomy a claim about training rather than about a reader.

### 2.8 The Google writer is dose-scoped

Part 172: zeroing L27 N3325 did not cut `n1`'s clean Google at dose 1 (18 against 17) but did at 0.75 (18 to 6). There
are two routes to Google, one through the writer at moderate dose and one past it at larger dose. Finding 1's "two
late neurons carry the fallback" holds at the doses it was measured, and the front page should say so.

## 3. New directions, in rough order of value per rig-hour

1. **Pure norm scaling (section 2.3).** One hook, one afternoon, and it decides how Findings 7 and 8 are read.
2. **Synthetic ground truth with a trained single-neuron persona.** Train one random neuron on Qwen3-8B with the
   published owl recipe (or on Qwen3.5-4B), then run the pipeline blind on the model: does the Part 86 screen find
   the neuron, does the Jacobian lens read owl, which kind does the sort give it, does Neuron Lens read owl in ordinary
   text, does its persona persist through the cache (Part 125's design) and across turns, and does its push on the
   base model give owl? It is the first identity neuron whose cause is known, and it calibrates every reader and
   lens the repository uses.
3. **Country to company (section 2.1).** Three cheap tests on units the repository already holds.
4. **Fallback census across families, against public data (section 2.2).** Parts 174 to 176's rig on five models.
5. **Part 46.** The only unrun design that fills the hole in Finding 5: every Part 80 body was another model's text in
   the assistant slot, so the gap measures the tag, not authorship. Part 46 puts the stage model's own greedy answer
   in the slot, across OLMo 3's stages, with the on-policy NLL confound measured. Run it before any further head work.
6. **Prompt-supplied identity against trained identity.** Under a system prompt asserting another maker, zero the four:
   if the model keeps the prompted identity, the four hold trained identity only; Part 171 already shows only Google
   is adoptable as a role, which this would explain or complicate.
7. **A basin phase diagram.** The repository's name has one direct measurement, Part 31, and its clamp barely moved the
   state. With Part 176's census method, define per layer and stage the median collapse dose of random directions
   (depth) and the split of directions into stays-Qwen, falls to the other lab, breaks (asymmetry), across OLMo 2's
   SFT, DPO and Instruct and OLMo 3's stages. It asks "which stage builds the basin" with a cleaner instrument than the
   weight swaps of Parts 99 to 108, and the 1B runs in hours.
8. **The 2^6 factorial on the maker circuit (section 2.6).**
9. **Cross-size replication of Finding 1** on Qwen3.5-0.8B (already on the rig) and Qwen3-8B: is there a small set,
   do its lenses write China tokens, is the fallback Google or OpenAI? A second data point for sections 2.1 and 2.2.
10. **Methods: primary outcomes as teacher-forced log-odds with cluster bootstraps** (sections 1.2 and 1.3), and the
    kernel-agreement gate in `tools/part_check.py` (section 1.1).
11. **A short public note** on the four results that survive the above: the maker circuit with the fallback-as-prior
    reading, the random-direction census, the dose and norm phase diagram, and the Arc kernel fault as a methods
    warning, released with CPU float32 numbers only. The 331 KB README will not be read outside the project.

## Sources outside the repository

- Intel torch-xpu-ops issue #5643, fused SDPA writing past its output buffer in bf16:
  https://github.com/intel/torch-xpu-ops/issues/5643
- PyTorch issue #199945, XPU SDPA returning NaN for fully masked rows: https://github.com/pytorch/pytorch/issues/199945
- PyTorch issue #191937, a CUDA-only bf16 wrong-output case in the efficient backend keyed to KV length mod 32:
  https://github.com/pytorch/pytorch/issues/191937
