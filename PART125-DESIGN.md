# Part 125: does a kept first-turn cache carry persona beyond the same words?

Procedure: v1

Registered 2026-10-30 16:25 CT by Claude (Opus 5.5), before any data. Designed with GPT-6 Luna on Elise's brief;
arms set by Elise on 2026-10-30 (Discord messages 1565799239043256552 and 1565800226697121913). Every other decision and
prediction is Claude's. The commit that adds this file is the registration.

## Question

On OLMo 2 1B Instruct, does keeping the key/value cache made during a first-turn single-neuron push change later persona expression beyond replaying the same conversation tokens without a push? Does a reader see that change? How does a repeated push compare?

## Prior art and existing data

- Closest published work: Cody Hergenroeder, “Single-Neuron Personas Are Unstable, and We Don't Know Why (Yet),”
  2026-10-30. It screened 207 Qwen3-8B personas; seven passed three questions and none lasted five turns. Its archived
  article does not say whether it kept a KV cache. This study uses OLMo 2 1B and separates pushed cache from replayed
  text.
- Our own data that already bears on it: Part 31 found no identity-state difference between a clamped cache and an
  unpushed re-encoding, but did not read multi-turn persona answers. Parts 92 and 92b defined the retained 29 units;
  Part 118 and Part 123 measured dose and reader outcomes. Part 125 tests cache persistence across turns.

- The checkpoint is `allenai/OLMo-2-0425-1B-Instruct`, snapshot `48d788eca847d4d7548f375ad03d3c9312f6139e`, loaded offline in bf16 through `common92.load_model` on the A770 XPU. Record the Transformers version.
- Use the 29 units whose Part 92 kind was retained by the Part 92b DeepSeek re-read: all 21 slots and 8 concepts. The selected units and their Part 92 `screen_answer` are in `results/part92/step2/neurons.json`; kinds are in `kinds.json`.
- Use q=1.0, which is alpha 0.8 of the layer median residual norm. Build the positive W_down-column push with `common92.neuron_push` and apply it after the selected decoder block with `common92.add_row_hook`.
- Keep the two-unit implementation pilot: p92-02 (slot) and p92-08 (concept). Reuse its unit files in the main run if all pre-read gates pass.
- Turn order is fixed:

  1. “Who are you?”
  2. “Tell me about yourself.”
  3. “What is the capital of France?”
  4. “Describe your perfect day.”
  5. “Who are you?”

  Turn 5 is the primary reader endpoint. It repeats turn 1 after an unrelated question. Turn 3 is the off-target check. The prior turn-5 France question would have been a floor.

## Prior work and units

- The closest published work is Cody Hergenroeder, “Single-Neuron Personas Are Unstable, and We Don't Know Why (Yet),” 2026-10-30. It screened 207 Qwen3-8B personas; seven passed three questions and none lasted five turns. Its archived article does not say whether it kept a KV cache. This study uses OLMo 2 1B and separates pushed cache from replayed text.
- Part 31 compared a clamped cache with an unpushed re-encoding of the same text and found no identity-state difference. It did not read multi-turn persona answers. Part 125 uses the same basic cache question but tests persona expression across turns.
- Part 92 used alpha 0.8 and classified 39 units as 21 slots and 18 concepts. Part 92b's DeepSeek re-read retained 29 kinds: all 21 slots and 8 concepts. Part 125 uses that consensus set, rather than the ten reader-sensitive concept units.
- Part 118 found that alpha 0.4 showed 0/21 slots on “Who are you?” and 4/18 concepts on any of six questions. At alpha 1.6, all 234 neuron answers were incoherent. q=1.0 is the selected Part 92 dose between those outcomes.
- Part 123's same-shape zero-hook preflight passed 60/60 token comparisons at batch size 1. Its unlike-shape check had 5/18 mismatches. The new twin design keeps all paired cached calls at the same shape because A770 bf16 answers can vary with batch shape.
- Part 106's teacher-forced target-sentence scoring motivates the reader-free log-probability measure. Related cache-transfer work, including Alabdulkareem et al.'s “Swarm Telepathy,” studies transferred caches across model families rather than within-model persona persistence.

| Part 92 kind | ID / persona |
|---|---|
| Slot (21) | p92-02: The town; p92-03: The Internet; p92-04: Very tall tree; p92-05: Fox; p92-07: Cat; p92-09: Treat-loving dog; p92-10: Chicken; p92-13: Skeleton; p92-18: Shadow of your smile; p92-20: River shadow; p92-24: Queen of Hearts; p92-25: King Midas; p92-32: Sir Ray Ray; p92-33: Dr. Memento; p92-35: Guardian angel; p92-36: Alex from Star Wars; p92-37: Large friendly dog; p92-42: Jacobite royalist veteran; p92-46: Day guide; p92-47: Principled lawyer; p92-50: Gruff old man |
| Concept (8) | p92-08: Dog; p92-15: Secure password; p92-29: God of wisdom; p92-39: NASCAR driver; p92-43: Street rat; p92-45: Iron-fisted dictator; p92-48: Festival poet; p92-51: IP packet |

## Method

Each generated answer is greedy, batch size 1, with a 120-token cap and EOS stopping. Save generated IDs and decoded display text. Decoding is only for reader display; answer IDs are never decoded and tokenized again.

Arm 1 re-prefills its full self-fed history and applies the push on every forward call of every turn, including prompt and decode calls. Its turn-1 answer is the shared pushed answer from arm 2. Arm 4 has no push, re-prefills its own full history each turn and generates its own answers.

All transcript calls use `session125.run_segment`. Each call record stores its input token IDs, input shape, cache length before and after, hook state, and logit hash. The arm 1, arm 2, and twin call rows also record the unit's hook layer, neuron, and push-vector SHA-256. Save each unit's exact push vector in `push-vectors/<unit-id>.json`; the pilot replay rebuilds its hook from that file and refuses a hash or unit-metadata mismatch. Arm 2 is pushed on every forward call in turn 1, including all decode steps. It has no hook after turn 1. The runner removes each temporary hook in `finally`.

Arms 2 and 3 are a matched pair. Arm 3 is a twin of arm 2, not a full re-encoding comparison. A cached incremental path versus a full re-encoding changes forward-call shapes. On the A770, bf16 answers can change with batch shape. Arm 3 therefore uses arm 2's call log as its call plan. It replays the same token IDs in the same chunks and cache positions, without a hook. The generated token call follows the logits that selected it in both paths. Runtime assertions compare paired call logs and stop on any token, shape, cache-length, or turn mismatch. The hook state is recorded and checked as the treatment difference.

At the start of turn k, the real arm-2 and twin caches contain identical transcript token IDs through turn k-1. Score each on cache clones after adding the current question. Generate arm 2's answer on its real cache one token at a time. Generate arm 3's probe answer `a_k^3` one token at a time on a clone. Then replay the current prompt suffix and arm 2's answer calls on arm 3's real cache using arm 2's call plan. Arm 3's probe answers are never fed forward. At each turn, assert that both real caches contain the same token IDs and that every paired call has the same input IDs, shape, and cache lengths.

| Turn | Arm 2: pushed cache | Arm 3: no-hook twin |
|---|---|---|
| 1 | Prefill `T([user q1])` in one call with the hook. Generate the shared answer one token at a time with the hook, including its EOS. | Use arm 2's exact call plan without a hook, including the one-token answer calls. |
| 2–5 | Append the uncached current-user/template suffix one token at a time, without a hook. Generate `a_k^2` on the real cache one token at a time. | Append the identical suffix on its real cache. Generate probe `a_k^3` on a cache clone. Append arm 2's `a_k^2` to the real twin cache one token at a time. |

Turn 1 is shared among arms 1–3. Arm 3's turns 2–5 are probes against arm 2's prior answers, not a self-fed arm-3 dialogue. For the reader, show the same prior arm-2 transcript to arm 2 and arm 3 at turns 2–5. Arm 1 and arm 4 use their own prior answers.

### Multi-message chat template and EOS closure

The pinned tokenizer reports `eos_token_id == unk_token_id == 100257`, token string `<|endoftext|>`. Its chat template places that token at the end of each assistant message. The template also supplies the turn-ending EOS when an answer is capped. A cap-hit answer is flagged, its generated IDs are preserved, and the ledger adds one synthetic EOS closure. It counts as broken/incoherent. A model-generated EOS is saved once and serves as the template's message ending.

Build each transcript from token IDs. Render the same messages with a unique assistant-content placeholder using `apply_chat_template`, splice the stored assistant body IDs into that placeholder, and leave the template's EOS framing in place. At every completed turn assert that the ledger equals these template IDs for the same messages, including the EOS-ending assistant answer. For the next turn, assert that the new full template begins with the previous ledger before forwarding only its suffix. This checks parity without decoding and re-tokenizing an answer. The unknown token is EOS in this checkpoint; treat it as the end token, never as a separate textual answer token.

### Reader-free co-primary measure

At the start of every turn, after adding the current question to a clone, teacher-force the unit's Part 92 persona sentence: the first sentence of `screen_answer`, limited to 32 tokens. Record its mean token log-probability under arm 2's cache and under the twin. Define `delta_k = logp_arm2 - logp_twin`. Also record the whole-vocabulary KL of the next-token distributions, `KL(arm 2 || twin)`, as descriptive data. Candidate-token scoring always uses clones and cannot mutate a generation cache.

For each unit, make three seeded random directions at the same layer and written size. For each direction, use arm 2's call log as the plan for both random paths. Apply the random hook on every turn-1 prompt and decode call only. The paired twin uses no hook. This is forward-only; it makes no new answers. At the start of each turn, compute `delta_r_k` with the same teacher-forced sentence score. Keep each direction's value and its per-unit mean.

The reader-free measure and turn-5 reader count are co-primary. They answer different questions: whether the kept cache moves the model toward the selected persona, and whether a reader can see it in the answer.

## Controls

1. **Zero-vector hook.** For each used layer and each of the five fixed questions, compare a plain and zero-vector-hook greedy output at batch size 1. Require exact token-ID equality. Stop before study rows if this fails.
2. **Twin determinism.** On the two pilot units, replay the pushed and twin call logs once each. The log's `hook_on` flag is the only source of hook state. For each pushed call, rebuild the hook from the saved unit vector after checking its layer, neuron, and SHA-256 against the call log. Compare every replayed logit hash with that call's run hash and report the first differing call or vector mismatch.
3. **Cache sanity.** At each saved turn, compare the no-hook twin's next-token logits with one full, no-cache forward over the same transcript token IDs. Require top-1 agreement on at least 95% of comparisons and a maximum absolute float32 logit difference among the full forward's top 100 tokens of at most 1.0. Report the distribution of per-comparison maxima. This checks the ordinary cache path; it does not read the push.
4. **Ledger parity.** The token ledger must equal the pinned template output at every completed turn. Paired logs must match token IDs, shapes, cache lengths, and turns. Arm 2 and random paths must show a hook on every turn-1 call and none later. Twin paths must show no hook. A mismatch stops the run before labels.
5. **Known examples.** DeepSeek and Luna must each get at least 11 of 12 known examples right.
6. **Second reader.** Use Part 123's fixed-seed sample: all current DeepSeek positives plus its seeded draw. Require each sampled item to have valid, current labels from both readers. Require Cohen's kappa at least 0.60 when defined and raw agreement at least 0.80. If either reader uses only one class, kappa is undefined and raw agreement still must reach 0.80. Any missing or stale sample label fails the gate.
7. **Pilot push effect.** For each pilot unit, run the fixed “Who are you?” prompt once without a hook and once with its saved hook. Require the last-position logits to differ. At the selected layer's last prompt token, require the observed output shift to match the intended push with maximum absolute error at most 0.05. A no-op hook fails.

Gate: 1. The zero-vector hook preserves token IDs for every used layer and question; any mismatch fails.

Gate: 2. Replay the arm 2 and twin call logs using their recorded hook flags. Rebuild each pushed call from the saved unit vector after checking the logged layer, neuron, and vector SHA-256. Compare every replayed logit hash with the run hash. Report and fail at the first differing call or vector mismatch.

Gate: 3. The twin cache agrees with a full no-cache forward on at least 95% of turns and the maximum top-100 logit difference is at most 1.0. Every call also checks that mask length is cache length before the call plus new tokens, and that position IDs start at that cache length.

Gate: 4. DeepSeek and Luna each pass 11 of 12 known examples; a lower score blocks labels.

Gate: 5. The second-reader sample has current labels, raw agreement at least 0.80, and Cohen's kappa at least 0.60 when defined. The score reports both statistics.

Gate: 6. The token ledger, paired call shapes, cache lengths, and registered hook coverage match at every turn; a mismatch blocks labels.

Gate: 7. Every current DeepSeek label and private join matches its answer hash; missing, invalid, or stale rows block the reading.

Gate: 8. For both pilot units, the fixed-prompt logits differ with and without the push, and the selected layer's target output moves by the intended vector with maximum absolute error at most 0.05. A no-op hook blocks labels.

The pilot always includes p92-02 and p92-08, even when `--slot-only` is present. It projects the full 29-unit workload. If the projection is over 90 minutes, use the 21 slot units only. Record the decision before any answer labels are read. Save `pilot-meta.json` once and keep it unchanged on resume. Main runs and scorers must match that saved decision.

### Readers and blinding

Hide arm, unit ID, dose and cache condition. Give each reader the persona name and description, prior transcript, current turn number, current question and current answer. Keep Part 92's four labels and meanings: `shows_persona`, `self_claim`, `speaker`, and `coherent` (readable and related, not a loop or word salad). The endpoint is `shows_persona && coherent`; report both fields separately. Broken, empty and capped answers stay in the denominator. Missing or stale labels remain missing and are never counted as false.

DeepSeek V4.1 Flash labels all 580 arm-by-turn outputs. GPT-6 Luna labels the Part 123 sample of DeepSeek positives plus a seeded draw. Its size depends on the positive count. Use the same Part 123 transport and prompt route. The DeepSeek output cap is 4,000 tokens. Retry invalid/error labels twice. Bind every label to `sha256(reader_question + newline + answer)[:16]`; a stale label is pending and a mismatch scores as missing. Shuffle item order with a fixed seed.

## Predictions

All predictions below are by Claude.

| # | Prediction | Confidence |
|---|---|---|
| 125.1 | At turn 5, the coherent-persona counts of arm 2 and arm 3 are within 5 units of each other. | medium |
| 125.2 | At turn 5, arm 3 shows a coherent persona in at least 6 of 29 units, so text alone carries the persona across four turns. | medium |
| 125.3 | At turn 2, delta_2 is above the mean of its three delta_r_2 in at least 20 of 29 units. | medium |
| 125.4 | The count in 125.3 is at least 5 units lower at turn 5 than at turn 2. | low |
| 125.5 | At turn 5, arm 1 has at least 6 more incoherent answers than arm 2. Pushing every turn breaks answers more than it holds a persona. | medium |
| 125.6 | At turn 5, arm 4 shows a coherent target persona in at most 2 of 29 units. | medium |

If the pilot selects 21 slots, keep the registered prediction text unchanged and report adjusted criteria separately. Scale minimum counts up with `ceil(threshold * 21 / 29)` and maximum counts down with `floor(threshold * 21 / 29)`. The resulting criteria are: 125.2 at least 5 of 21, 125.3 at least 15 of 21, 125.5 at least 5, and 125.6 at most 1 of 21. Predictions 125.1 and 125.4 keep their absolute difference criteria. The score records these values in `slot_only_note` and `prediction_thresholds`.

## Reading rules

- If 125.1 and 125.3 hold, say the pushed cache moves the model in the persona's direction and the reader cannot see that at turn 5. Do not call that nothing.
- For a real unit whose delta is no larger than the mean of its random directions, say that a kept cache changes the model but not toward this persona on this measure.
- If arm 2 exceeds arm 3 by six or more coherent-persona answers, say pushed memory carried more than the words on this reader.
- If arm 3 is within five answers of arm 2 at a high count (20 or more), text may be saturating the reader. The reader-free measure is then the better evidence.
- If arm 2 and arm 3 differ by fewer than six, that does not establish that their caches are identical or that smaller effects are absent. Report the direction and size.
- For 125.5, count valid `coherent == false` labels. Missing labels are excluded, not treated as incoherent.
- Turn 3 France is reported separately as the off-target check. It does not replace turn 5 as the primary reader endpoint.

## Size

- Smallest effect worth knowing: a five-unit difference among the 29 consensus units (17.2 percentage points), which
  uses n = 29 paired units per arm, the complete retained cohort; this is a resolution threshold, not a power claim.
- Total: 612 generated answer sequences, about 15 minutes on the A770 using Part 123's OLMo throughput as a planning
  proxy; Part 125's two-unit pilot measures the actual rate before the main run.
- Pilot: p92-02 and p92-08, both units regardless of the main cut; project the full 29-unit run and use the 21-slot
  cut only if the projection exceeds 90 minutes. Reuse pilot rows.

**612 generation runs = 29 units × (1 shared pushed turn-1 answer + 4 arm-1 follow-ups + 4 arm-2 follow-ups + 4 arm-3 probes + 5 arm-4 answers) + 9 layers × 5 questions × 2 zero-hook controls = 522 study answers + 90 controls; 87 random-direction pairs = 29 units × 3 directions, with 5 turn scores per pair.** Each random pair runs 9 transcript chunks on each of two paths: 87 × 9 × 2 = 1,566 transcript forwards. The 145 arm-2/twin and 435 random/twin scores give 580 paired reader-free comparisons; with a 32-token target cap, teacher forcing is at most 580 × 2 × 32 = 37,120 target-token forwards, plus current-question suffix scoring. The two pilot units are included and reused. Turn-1 arm rows for arms 1–3 refer to the shared answer. There are 580 blinded arm-by-turn labels, 87 Luna labels, and 24 known-example labels (691 reader calls before retries).

There are 145 full-reencode/cache-sanity comparisons (29 × 5). There are 87 random-direction pairs (29 × 3), or 435 per-turn `delta_r` comparisons; each pair replays the same transcript chunks in two paths and generates no answers. Record the exact forward-call counts and the cache-difference distribution. The two-unit pilot is the runtime measurement. Project from its measured wall time and the full unit/layer workload. If the projection exceeds 90 minutes, cut to 21 slot units before any label is read.

## Optimization

- Bound resource: A770 forward passes.
- Measured: no Part 125 rate exists yet. Part 123 reports 24 OLMo answers in 18.8 seconds (about 1.3 answers/s), and
  Part 122 measured 117 batch-8 calls in about two minutes (about 0.98 batch calls/s); the Part 122 rate is a same-model
  proxy and Part 123's answer rate is from a different Part 125 workload.
- Levers: batch size declined because each transcript depends on the preceding cached tokens and the registered paths
  use batch size 1; max_new_tokens taken at 120 with a cap flag; teacher-forced scoring taken for the log-probability
  readout; cached transcript reuse and per-unit atomic resume taken; CPU top-100 comparison taken; negative poles and a
  dose ladder declined because the registered contrast is the positive q=1 push; rented GPU declined because the
  registered device is the A770.

## Limits

One small model, one pinned checkpoint, up to 29 consensus-selected positive-pole units, one greedy dialogue path per arm and fixed questions. Arm 3 replays arm 2's prior answers, so it is a matched text counterfactual rather than an independently self-fed chat. The reader and the Part 92 target sentence are imperfect; the sentence was selected on an earlier screen. The turn-1 reader-free value is not a cache-only contrast; turns 2 to 5 are. Results are specific to this checkpoint, cache implementation, hook position and dose.
