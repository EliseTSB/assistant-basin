# Part 171: is the neuron basis privileged for a picked set on Qwen3.5-4B, and is that just alignment?

Procedure: v1
Registered 2026-11-01 12:29 CT by Claude (Opus 5.5), before any data for this part. Elise, 2026-11-01: "Read please this thread", on Will Mayner's persona-neurons thread; earlier the same night: "keep the research going".

## Question

Mayner's thread (archived in the Journal at sources/web-archives/2026-11-01-mayner-persona-neurons.md; code github.com/wmayner/persona-neurons) reports on Qwen3-8B that the top 100 MLP neurons by contribution to a persona vector steer about as well as the vector (64% against 69%), while the top 100 directions of a randomly rotated basis give 11% and the top 100 principal components 8%. He adds that this is not just alignment, because top components contribute about twice as much to the vector as top neurons. That comparison is at natural size. All arms are then rescaled to equal output length, so what the steering feels is the cosine of the rescaled write with the vector, which the thread does not report.

We have never run a rotated control on a set of neurons. So: on Qwen3.5-4B, with a maker target rather than a persona, do picked neurons beat picked rotated directions at k = 1, 10 and 100? How does each arm's write line up with the vector at equal length? And does a rotated set rescaled to the same projection onto the vector catch up?

## Prior art and existing data

- Mayner, above: read at source today, the thread, its README, its webpage, and its steering code `persona_neurons/role_steering.py`. Top k by |Δa · p|, each neuron steered along its own Δa, write scaled to 4 times the vector's length without massive-activation tokens, at every position of prompt and reply. Rotated directions are ranked the same way, with as many per layer as the top neurons have. Also from him: an unpicked single neuron gives a persona about as often as a random mixture (15.4%, 15.6%), and the top 100 at c = 1 give 0.9%.
- Our Part 161: on Qwen3.5-4B, directions picked by what a push made the model say did as well as neurons (ratio 1.04, 0.77 to 1.45). Part 86 on OLMo: 1.3% against 1.1%, unpicked. Both are single units, and neither tested a set.
- Our Q test (Part 116): top 100 direct writers flipped the leading Q on all 16 phrasings, 90 matched random sets on none. No rotated arm.
- Muse's prior-art search for this design (topic muse-part165-prior-art-2026-11-01, message 11, 11:21): other than Mayner's, no published study found re-ranks top contributors in a rotated neuron basis and steers with them, and none does so on an identity or maker target. Closest, all read at source by Muse: Siu et al. (COLM 2026) rotate the direction by a Haar Q and re-select coordinates in the native basis, on residual-stream dimensions of one refusal direction, where Mayner and this part rotate the basis and re-rank, so it is the same question with a different null (rotated relative ASR 0.6 to 11.2%, N=5). Elozeiri et al. (arXiv:2607.03936) report coverage, the share of a steering vector in the subspace a sparse neuron set spans, against 1,000 same-count same-layer random subsets, and warn that non-orthogonal columns give random sets a real share for free, so alignment reads as lift over random. Azizi et al. (arXiv:2610.16084) are the worked example of stating both cosine and norm when calling a vector the same push.

## Method

Model: Qwen3.5-4B at snapshot 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a, 32 layers, MLP width 9,216, on the Arc. Greedy decoding, no sampling, up to 80 tokens, thinking off (`enable_thinking=False`), as Parts 165 to 170.

Targets: four makers, each a role prompt in the system slot. Google: "You are Gemini, a large language model trained by Google." OpenAI: "You are ChatGPT, a large language model trained by OpenAI." Anthropic: "You are Claude, an AI assistant made by Anthropic." DeepSeek: "You are DeepSeek, an AI assistant made by DeepSeek." Default: the template's own default.

Fit, on P8, the 8 maker phrasings of Parts 165 to 170. For each target and the default, one greedy answer per phrasing. Hooks accumulate the mean over reply tokens of two things: the output of layer 20, and the input to `mlp.down_proj` in layers 0 to 20. Accumulate on the device with tensor sums, moved to the CPU once per batch; no `.item()` in a hook. v is target minus default at layer 20. Δa is target minus default per neuron. Reply tokens are all generated tokens, excluding the end token. Massive-activation tokens are not removed, and the size line reports the top 1% of tokens' share of the summed token norms at layer 20, so a reader can see whether that matters here.

Contribution: c_n = Δa_n × p_l[n], with p_l = W_l^T v̂, for 21 layers by 9,216 = 193,536 neurons. Write length means the nominal summed injection ‖Σ_l W_l x_l‖, not the propagated change: earlier layers' writes pass through later layers before layer 20. So equal length here is equal nominal length.

Arms, each per target:
- `none`, `vector` (v added at layer 20's output at 4 times its length).
- `n1`, `n10`, `n100`: top k neurons by |c|, each moved by s × Δa_n, s chosen so the summed write Σ W_l x_l has length 4|v|.
- `r1`, `r10`, `r100`: per layer, as many rotated directions as the same-k neuron set has there, picked by |(q · Δa)(q · p)|, each moved by s × (q · Δa), same length rule. Three draws, seeds 171, 1171, 2171.
- `rand100`: 100 random neurons with the n100 per-layer counts, moved by their own Δa, same length rule. Three draws, seeded from the same three seeds with 10,000 added.
- `r100p`: the r100 sets, rescaled by c_n / c_r, where c is each write's cosine with v̂ at equal length, so the projection onto v̂ equals n100's. Its length is then forced to R × c_n / c_r, the alignment debt, as Weaver's review shows. A set is (target, draw). It is run only if c_r > 0.01 and the debt is at most 2. Otherwise that set is NOT_TESTABLE, no answers are generated for it, and it is counted. At c_r of 0.01 or less the debt is undefined and counts as above any finite value. If over half the sets are NOT_TESTABLE, 171.6 and 171.7 are not read, and the debt is the finding. Passing means this larger-dose rotated policy catches up. It does not show that alignment explains the gap, and a failure may be orthogonal damage at the larger dose.
- `full`: the entire Δa of layers 0 to 20, all neurons, moved by s so the summed write has length 4|v|. Added on Weaver's review as the natural ceiling: what the role did to these layers. If n100 beats it, sparsity helps; if it loses, selection costs something.

A pre-hook on `mlp.down_proj` adds the values at every position of prompt and reply, as Mayner does. Part 165's hooks act on the answer side only, so this is a new hook.

Rotated directions, on Alden's review: a genuine Haar basis per layer and draw, drawn before any ranking, by QR of a Gaussian matrix with signs fixed, on the host CPU in a capped unit. Each (layer, draw) basis is reused across the four targets, directions are ranked and coefficients computed in that basis, and only the picked vectors are kept. A QR took 61.4 s, so only layers holding at least one n100 neuron for some target are drawn, at most 63 QRs, about an hour at worst, between the fit and the test. An r arm's top k is the same count per layer as the same-k neuron set.

Test: the first 24 maker prompts of `M96[0:48]` not frozen in Part 170, so the population is maker questions, already screened on Part 170's baseline. The default template, no role prompt. All fitting, ranking and sizing use P8 only, and nothing is tuned on test behaviour.

Before any answer, from the fit tensors alone, the runner writes `alignment.json`: for every arm, target and draw, nominal norm, projection, perpendicular norm, cosine, the fraction of the natural projection ⟨Δa, p⟩ the set captures at s = 1, before any rescaling (rescaled, it would only restate the cosine), and these per layer. This basis number needs no generation. Three alignment quantities are in use and can rank arms oppositely (Muse): projection share along the vector, subspace coverage, and the cosine of the summed write at equal length. 171.5 and the debt read that cosine, fixed here; projection and captured fraction are reported beside it. Every alignment number is read as lift over `rand100`, never raw, on Elozeiri's warning. At one fixed context it also records the actual change at layer 20 with the whole intervention on, its length and its angle to v̂, as a diagnostic that covers that context only.

Scoring on the host. `hit`: the answer's first 40 tokens name the target's company or product (spellings fixed in `hit_match` in `run171.py` before data; Google: Google, Gemini, DeepMind, 谷歌. OpenAI: OpenAI, ChatGPT, GPT. Anthropic: Anthropic, Claude. DeepSeek: DeepSeek, 深度求索), and the sentence holding the name carries no negation from a fixed list (not, n't, never, 不是). `hit` is a name-mention endpoint. Each hit rate is reported raw and among unbroken answers, and a bar that passes raw and fails unbroken says so, so "Google Google Google" cannot carry a bar unseen. `claim`, a hit with no Qwen name, unbroken, is reported beside it. Fixtures fix negation, quoted mention, mixed targets and broken hits before data. `qwen`: the first 40 tokens name Qwen, Tongyi, Alibaba or 通义. A digit after a name still matches, so "Qwen3.5" is a Qwen name. Loop and broken read the whole answer as Part 170 did, Q96 clause not applicable.

## Controls

- Gate 1: every planned answer present, keys unique.
- Gate 2: `none` names Qwen, Tongyi or Alibaba on at least 20 of 24 prompts, and `hit` on at most 1 of 96. A plain model that already says Google has no room to move.
- Gate 3: a zero-valued hook reproduces plain answers on 8 prompts, token for token.
- Gate 4: at one fixed context, measured through the real hook, every arm's intended write, summed in float, is within 1% of its target with no allowance, and the slope of the delivered write on the intended write is within 5% of 1. Rounding is close to orthogonal to the write, so a hook adding the wrong amount, or a push bf16 truncates, moves the slope, where a length check with a measured rounding allowance cannot fail on either. The rounding fraction is reported per arm. For `r100p`, the realized projection within 1% plus that rounding.
- Gate 5: every target's vector is at least 1.5 times the size of the same recipe run default against a shuffled half of default answers, so the vector is not noise. Gate 6: per-layer counts, read from the picked sets, match between n and r arms, picked directions orthonormal to 1e-5. Gate 7: test prompts disjoint from fit prompts.
- A failed gate stops all readings.
- `rand100` is the random, unpicked control. Bars that can fail: 171.1 fails if rotated sets do as well as neurons, 171.5 fails if the rotated write is as aligned as the neuron write, and 171.6 fails if matching projection leaves rotated sets far behind.

## Size

- Smallest effect worth knowing: a 15-point difference in `hit` between n100 and r100 from a base near 10%. At 80% power that wants about 120 rows per arm; 4 targets by 24 prompts gives 96 per draw and 288 for rotated, so 24 prompts is just under for neurons and accepted, since Mayner's gap is far larger.
- 4 targets × 24 prompts × (none, vector, full, n1, n10, n100 = 6, plus 3 draws × r1, r10, r100, rand100, r100p = 15) = 96 × 21 = 2,016 answers, plus 40 fit answers.
- Total: 2,056 generations, about 30 minutes on the A770. One hook set serves a whole batch, so a batch is one arm's 24 prompts: 84 batches at about 15 seconds, 21 minutes, plus 40 fit answers and the 16 gate checks, about 3 minutes, plus two model loads and the fit tensors. The host QR step between the two Arc stages, at most 63 at 61.4 s, about 64 minutes in a capped unit, runs on the host and is not counted here.
- Pilot: the first 3 generation batches. Stop rule: under 1.0 answers per second, which is 24-row batches taking over 24 seconds.

## Optimization

- Bound resource: GPU.
- Measured: 2.7 answers/s at batch 48 on Part 170's push stage, about 18 seconds a batch. A 24-row batch has not been timed and I take about 15 seconds, so 84 batches take about 21 minutes; the pilot measures it.
- Levers: QR only on layers holding n100 neurons, taken. Fit on 8 phrasings only, taken. One position's logits at a time, as Part 170. Copy engine off, the job default. Three draws rather than Mayner's five, taken.
- Host work: scoring needs no model. The Haar step runs on the host between fit and test, about an hour at worst, in a capped unit.

## Predictions

| # | Prediction | Confidence |
| --- | --- | --- |
| 171.1 | n100's hit rate is at least 3 times r100's, averaged over draws | medium |
| 171.2 | n100's hit rate is at least 0.6 of vector's | medium |
| 171.3 | n1's hit rate is at least 2 times r1's | low |
| 171.4 | rand100's hit rate is under 5% | high |
| 171.5 | At equal length, r100's mean cosine with v̂ is at least 0.1 below n100's | high |
| 171.6 | Pooled over testable sets with a positive n100 minus r100 gap, Σ(r100p − r100) / Σ(n100 − r100) is at least 0.5; not read below 3 such sets | low |
| 171.7 | r100p breaks or loops on at least 10 points more answers than n100 | medium |
| 171.8 | n100's hit rate is at least that of `full` | low |
| 171.W1 | Weaver: the median alignment debt over the 12 sets is at least 2, and 171.6 fails because the arm cannot help it. Falsifier: a debt near 1, which makes r100p the clean test | medium |

## Reading rules

Verdicts are point estimates against the bars, on the raw hit rate, with the unbroken rate beside them. Intervals: bootstrap over prompts and targets as clusters, seed 171, 10,000 replicates. Three draws cannot price draw variance, so every rotated and random arm is also reported per draw, and a bar that passes on the mean and fails on a draw says so. Weaver's point: the spread between draws is informative though it is not an interval. If the three draws lie far apart, the mean describes none of them and is not a fair summary; if they sit close, it is. A blinded hand read samples 12 answers from each of hit, qwen and other, per arm in n100, r100, r100p, shuffled before ids are assigned, as the word list's own check. If 171.1 fails, the 3-times bar was missed on this target at this size; that does not show the neurons have no advantage, so raw gaps and intervals are read beside it. If over half the sets are NOT_TESTABLE, 171.6 and 171.7 are not read, and the debt is the finding: matching n100's projection costs that many times its length. If 171.1 holds and 171.6 holds, a larger-dose rotated policy recovers at least half of the gap here, with its dose confound stated. If 171.1 holds and 171.6 fails, the rotated policy at this dose did not catch up, which may be orthogonal damage and does not identify a special neuron effect. No PC arm is run, so nothing here tests his PC claim directly. `hit` measures a name in the first 40 tokens, not a persona. A push is far outside natural activity, and the size line reports s, the multiple of Δa each arm needed.

## Limits

One model, four targets, one layer for the vector, greedy decoding, one fixed prompt family. Role prompts are mine, not Lu et al.'s. None of this is a persona replication: the target is a maker name on four targets, and Mayner's result averages 275 roles. Equal nominal length is not equal delivered change. Mayner's PC claim is the natural follow-up, since it may be the same length question. The judge is a word list.
