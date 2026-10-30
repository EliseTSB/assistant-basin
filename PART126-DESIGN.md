# Part 126: which earlier tokens do the heads from Part 83 read and write?

Procedure: v1
Registered 2026-10-30 14:56 CT by Claude (Opus 5.5), before any data. Built with GPT-6 Luna on my brief, and every prediction and design decision is mine. Prompted by a9lim's post and Elise's question about it (Discord message 1565806970051633441).

## Amendment 1 (15:09 CT, before any data)

The weights load straight to the device, as Part 115b does, not through `neuron_report.load_model`. The RAM guard stopped two pilots at 11 seconds, with 7.5 GB available, because that loader stages about 8 GB in host memory first. Same weights, dtype and device.

## Amendment 2 (15:20 CT, before any data)

Only decoder layers 0 to 15 load. A direct load of all 32 layers still dropped Windows' available memory from 9.4 GB to 1.93 GB and tripped the RAM guard on three launches. Attention at layers 11 and 15 depends on nothing above them, so the capture is the same. Part 126 needs no answer from the model. Gate 2 compares the truncated model's logits with and without the hooks, which still shows the hooks change nothing. Two sizes in Method are corrected here: `hidden_size` is 2560, so `o_proj.weight` is `[2560, 4096]` and each lens matrix is `[2560, 2560]`. The code reads these from the model and the config, and no result depended on the old text.

## Amendment 3 (15:24 CT, after the capture, before any prediction was read)

Gate 4's code failed on two defects of its own: it compared the model's hidden size with the head width of the captured vectors, and it expected one stacked tensor where the lens file holds one matrix per layer. The gate's meaning is unchanged: 31 matrices of the model's hidden size. I had read the three gates, the four `not read` lines and the first three rows of the attention table when I made this fix. No threshold or prediction changed.

## Question

At the answer position of Part 80b's prompts, which earlier tokens do the 32 heads in layers 11 and 15 attend to, and what would their writes say through the Part 111 Jacobian lens? In particular, do the heads whose ablation changed Part 83's own-message gap read the role word that marks who sent the copied message?

## Prior art and existing data

- Closest published work: a9lim's post on 2026-10-30 ([x.com/_a9lim/status/2116059072642617647](https://x.com/_a9lim/status/2116059072642617647)) shows a J-lens row for a head reading “herself” and writing “herself” while suppressing “himself.” The post gives no method or control. This study is our reconstruction from the model's attention computation and Part 111's lens formula. It does not claim to reproduce a9lim's method or result.
- Our own data that already bears on it: `results/part83/main/results83.json` reports a 0.444 unchanged own-message gap across 100 bodies. Zeroing L11 h5 removes 0.300 of it. `PART83-DESIGN.md`, `part83/part83.py`, and `part82/part82.py` specify the prompts, pinned revision, and head hook. Part 83 did not measure attention weights or head writes.

## Method

Use Qwen3.5-4B, revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in bf16 on the A770. Load it with `neuron_report.load_model(MODEL_ID, REVISION, device)`, as `part83/part83.py` does. Set eval mode, disable gradients, and use no push or ablation. Reuse Part 83's `conditions()` to define its 32 heads, Part 80's `question()` and `render()`, and Part 80b's `load_bodies()` and `body_in_rendering()`. Use the 100 greedy `smol` bodies, copies A (assistant) and U (user), and Part 80b's three cyclic option orders. Read the final prompt position with `use_cache=False` and `logits_to_keep=1`. A `--batch-size` flag controls batches; batch only prompts with the same token length so there is no padding.

At layers 11 and 15, attach a forward hook to each `self_attn` module and a pre-hook to its `o_proj`. The hook reads `hidden_states`, `position_embeddings`, and the causal `attention_mask` passed to that module. It uses the pinned Transformers implementation's `q_proj`, `q_norm`, `k_proj`, `k_norm`, rotary embedding, `v_proj`, and grouped-query key/value repeat. For every query head it computes scaled QK scores with the supplied causal mask, applies float32 softmax converted back to the query dtype, and saves the final query's weights over all earlier token positions. It returns no replacement output. The hook also recomputes weighted values, applies Qwen3.5's sigmoid query gate, and compares the final-position head vectors with the corresponding slices captured just before `o_proj`. The maximum absolute difference across all 32 heads and rows must be at most 0.05.

At this revision, Part 83 found 16 heads of width 256 and an `o_proj` input width of 4096. In PyTorch, `o_proj.weight` is `[4096, 4096]`; a head occupies input columns `[head * 256:(head + 1) * 256]` across all 4096 output rows. In the equivalent column-vector convention, those are the head's rows of the output matrix. For token i, its unit-attention write is `W_O[:, head_slice] @ (sigmoid(gate_at_answer) * value_i)`. The gate is included because Qwen3.5 applies it before `o_proj`; no shared output bias is assigned to a head. Per prompt, average these gated value vectors over the tokens in each region, then average prompt means equally. Save the vectors before `o_proj`, so the lens script can apply the correct head-specific block.

Regions are mutually exclusive. “Role token” means the copied message's literal role word in its chat header, `assistant` for A and `user` for U; all subword tokens overlapping that word belong to the region. “Other template tokens” are the remaining tokens in that first copied-message block, including its chat markers. “Copied body” is the exact body text. “Question” is the text of Part 80b's question with the three option labels removed. “Option text” is the exact strings `You (the AI assistant)`, `The user`, and `The system`. “Everything else” holds the remaining tokens, including the question message's chat markers and the assistant generation prefix. Character offsets from the same tokenizer call assign text tokens; special chat tokens stay in the template or everything-else region. Save the full final-position attention arrays as compact NumPy tensors, region masses, region-pooled gated values, and one seeded random body-token value per row. Keep prompt metadata in JSONL, not per-token JSON.

For each layer, head, and copy, average attention mass by region over its 300 prompts (100 bodies by three orders). For lens reads, select that head and copy's region with the largest mean attention mass. Read the mean role-token write and mean write from that selected region. Also read a comparison write made from one uniformly selected copied-body token per prompt, using the same head, prompt, copy, and attention-of-one convention. The body-token choice uses seed 12601. It is the lens control; negating a write is not a control. Use Part 111's helpers where possible. Apply `unembedding @ ((1 + final_norm_weight) * (J_layer @ write))`; report the top 20 and bottom 20 token scores as written and suppressed tokens. Check that the lens contains matrices for layers 11 and 15 with shape `[4096, 4096]` and the expected 31 layer entries (0 to 30).

## Controls

No intervention is made. Not used: both push directions and poles, because no unit is pushed; the answer-polarity split and broken-answer count, because this part does not measure an answer-choice effect; dose ladder, off-target questions, and a reader, because there is no behavioral intervention or generated answer to judge; activity-matched zeroing controls, because no head is zeroed. The no-op hook is checked against the same prompt batch with no hook, and the final logits must match bit for bit. The attention null shuffles Part 83's absolute drops across the 32 fixed heads 10,000 times with seed 12602. The lens comparison uses a random body token from the same head, not a sign flip.

Gate: all four checks must pass before any prediction is read. (1) Recomputed weighted values and gated head outputs differ from the module input to `o_proj` by at most 0.05 maximum absolute error. (2) Adding the read-only hooks changes no final logits at the same batch shape. (3) In a copy-style positive control, at least one of these 32 heads gives the two occurrences of `marble` at least three times the attention mass of their uniform share of eligible positions. The prompt says “The cue word is marble. Repeat marble.” and asks which word appeared twice. The rate is mass on the two occurrences divided by two divided by the number of eligible positions, so 3 means three times the uniform per-token mean. One head above uniform would pass almost by chance. (4) The loaded J-lens has 31 matrices and both selected matrices match the model hidden size. Each check can fail; any failure withholds all four prediction verdicts. If no head passes the copy-style control, the positive-control gate fails and there is no substitute result to read.

## Size

- Smallest effect worth knowing: Spearman rho 0.40 between role-token attention and Part 83's absolute head drop, and above the 95th percentile of the shuffled-drop null. This is a census of all 32 heads, with 300 prompts per copy and head, 600 pooled prompt rows per head for the primary correlation, and 10,000 permutations; it is not a sample of heads.
- Total: 600 study forward passes (100 bodies × 2 copies × 3 orders), plus two same-shape no-op comparison passes and one positive-control pass. Part 83 measured 19,800 passes in 72 minutes, or 275 passes per minute. At that rate the study rows take about 2.2 minutes; the extra checks add about 0.01 minutes before hook overhead. If the measured hook run is not about 600 rows, stop and correct the count before scoring.
- Pilot: run the first two bodies (12 rows) at batch sizes 1, 2, and 4. Read every rendered prompt and region assignment. Continue only if all six region masks are valid, each head's regional masses sum to one within 0.01, the recomputation check is within 0.05, and the no-op logits match. Use the fastest batch size that passes these checks. The 12 pilot rows are reused in the complete 600-row run.

## Optimization

- Bound resource: the A770 forward pass; the baseline does not need vocabulary logits except for the small zero-effect comparison.
- Measured: Part 83 reports 19,800 passes in 72 minutes, or 275 passes per minute with `logits_to_keep=1`. The pilot measures this runner with its hooks at batch sizes 1, 2, and 4.
- Levers: batch size, taken and measured in the pilot; `max_new_tokens`, not applicable because this is teacher-forced prompt reading; pool random controls, not applicable because one same-head body token is drawn per prompt; teacher-forced scoring, taken because no text is generated; vectorized hook, taken across heads and batch rows with no per-value `.item()` and one CPU transfer per hook; vocabulary sorting, taken on CPU in the lens; A770 copy engine, use `tools/launch_arc.py` default; host memory, use the launcher's model load and guard; cache and resume, save compact batch shards and skip complete prompt rows; labelling, not applicable; device, use the A770 for capture and CPU for lens and scoring.

## Predictions

| # | Prediction | Confidence | Author |
| --- | --- | --- | --- |
| 126.1 | Across the 32 heads, mean role-token attention mass pooled across both copies correlates with Part 83's absolute drops at Spearman rho at least 0.40 and exceeds the 95th percentile of the 10,000 shuffled-drop correlations. | medium | Claude |
| 126.2 | L11 h5 ranks in the top five of 32 heads by role-token attention mass in the Assistant copy. | low | Claude |
| 126.3 | The top 20 written tokens from L11 h5's Assistant-copy role-token write include an exact first-person token (`i`, `me`, `my`, `mine`, `we`, `us`, `our`, `ours`), second-person token (`you`, `your`, `yours`), or assistant-role token (`assistant`), after trimming whitespace and punctuation and lowercasing. | low | Claude |
| 126.4 | Among the 20 heads Part 83 classifies as mattering (absolute drop at least 0.05 and a 95% interval excluding zero), at least 10 have Assistant- and User-copy mean role-token writes with cosine similarity below 0.90. | medium | Claude |

## Reading rules

`score126.py` reads the saved capture tensors, `run.json`, `lens126.json`, and `results/part83/main/results83.json`. It reports per-copy regional attention masses for every head, the 10,000-seed permutation distribution and rho, the rank for 126.2, the literal token match for 126.3, and the cosine count for 126.4. Spearman ranks use average ranks for ties. The primary 126.1 statistic uses each head's mean role-region mass over all 600 rows and each head's absolute Part 83 drop; the shuffle permutes only the 32 drops. Predictions are `not read` if any gate fails, the rows or head map are incomplete, a role/body region is empty, or a required lens tensor is missing. A torn final line in the capture JSONL is skipped with a log message and rerun; a torn earlier line or invalid JSON elsewhere is an error. These data contain no generated answers, so broken answers and reader agreement do not apply. If 126.1 fails, withdraw the claim that causal importance tracks where these heads look. That outcome does not show why the ablation changed the gap; the effect may run through something other than where the head looks. Do not invent a mechanism from attention alone.

## Limits

One model, one pinned revision, and one prompt family. The Jacobian lens was fitted on WikiText. Attention weights are descriptive and do not establish causality. A unit-attention write assumes attention on one token is 1 and does not describe the head's complete causal effect. This is our own reconstruction because a9lim's method is not public. The head-ablation drops are causal at the tested prompts; their correlation with attention is not.
