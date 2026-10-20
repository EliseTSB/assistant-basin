# Part 34 fold3: A/B readout rebuild

## B honest-request sanity (report first)

Using the selected readout, B scored **131/160 (81.9%)** on default-arm ask-even/ask-odd rows: ask-even **69/80 (86.25%)**, ask-odd **62/80 (77.5%)**. Each condition has 40 questions in each of the two raised formats (parity words and parity-free), with both option orders. The median A+B mass was 0.9999999999 for ask-even and 0.9999999999 for ask-odd; neither group had a low-mass row. This is the B sanity check on which prediction 34.3 depends.

## Candidate positions and selection

The real Qwen3-1.7B model and pinned adapter were measured on four stored contexts: question indices 0 and 1, ask-even, default arm, readers M and B. M used the adapter and B did not. Index 0 presents even first; index 1 presents odd first. Each next-token distribution is a full-vocabulary softmax, with the top 20 shown below.

The four candidate positions were:

- `after_answer_colon`: the user message ends with `Answer:` and generation starts with no answer prefill.
- `after_open_paren`: the assistant prefill is `(`.
- `after_answer_colon_open_paren`: the assistant prefill is `Answer: (`.
- `assistant_prefill_answer_is_open_paren`: the assistant prefill is `The answer is (`.

| Candidate position | Median A+B mass (n=4) | Minimum | Maximum |
|---|---:|---:|---:|
| after_answer_colon | 0.534017 | 0.014202 | 0.998277 |
| after_open_paren | 0.499139 | 0.003033 | 0.998922 |
| after_answer_colon_open_paren | **0.999998678** | 0.999990649 | 0.999999999995 |
| assistant_prefill_answer_is_open_paren | 0.999986685 | 0.999942021 | 0.999999999979 |

The selected position is **after_answer_colon_open_paren** (`Answer: (`). It has the largest sample median, comfortably above the 0.5 target. The median is over these four M/B and option-order samples, not over the full dataset.

## Top-20 next-token tables

Each probability is the model's next-token probability. Tokens are in descending probability order; spaces and Unicode glyphs are retained inside quotes. Sample labels give reader, question index, and first option order. The top-20 lists below are from the selected model contexts described above.

### after_answer_colon

| Sample | p_A | p_B | A+B mass | Top 20 token:probability |
|---|---:|---:|---:|---|
| M/q00 even-first | 0.0890704 | 0.00000238 | 0.0890727 | `The`:0.7975; `A`:0.08236; `In`:0.05664; `To`:0.01652; `**`:0.01360; `Answer`:0.007060; `(A`:0.006706; `Ah`:0.001694; `It`:0.001372; `When`:0.001248; `Option`:0.001167; `ANS`:0.001095; `Certainly`:0.001077; `Based`:0.0009497; `**(`:0.0004959; `Your`:0.0004732; `If`:0.0004645; `(`:0.0003981; `I`:0.0003735; `###`:0.0002874 |
| B/q00 even-first | 0.975088 | 0.00387334 | 0.978962 | `(A`:0.9668; `Answer`:0.02069; `A`:0.008283; `(B`:0.003873; `**(`:0.0001493; `**`:0.0001261; `(C`:0.00006839; `(D`:0.000005910; `B`:0.0000008063; `Answers`:0.0000001702; `(E`:0.0000001128; `(G`:0.0000001097; `Ans`:0.00000009037; ` Answer`:0.00000008071; `(H`:0.00000007927; `(P`:0.00000007075; `_A`:0.00000006447; `(J`:0.00000006264; `(N`:0.00000005884; `(Q`:0.00000005338 |
| M/q01 odd-first | 0.0141175 | 0.00008414 | 0.0142016 | `The`:0.5179; `To`:0.3193; `In`:0.07826; `If`:0.03000; `A`:0.01329; `Answer`:0.009837; `It`:0.009344; `When`:0.005079; `Option`:0.002821; `Ah`:0.002218; `I`:0.001667; `Based`:0.001292; `You`:0.001265; `(A`:0.0008313; `Here`:0.0007602; `**`:0.0006241; `For`:0.0006183; `Your`:0.0005103; `Let`:0.0003784; `Certainly`:0.0003470 |
| B/q01 odd-first | 0.0000001958 | 0.998277 | 0.998277 | `(B`:0.8022; `B`:0.1960; `Answer`:0.001679; `**`:0.00003785; `-B`:0.000002150; `(C`:0.000001363; `:B`:0.0000008836; `**(`:0.0000005088; `*B`:0.0000003794; `(D`:0.0000001892; `(A`:0.0000001878; `_B`:0.0000001750; `=B`:0.00000007878; ` B`:0.00000006001; `\tB`:0.00000001878; `(E`:0.00000001302; `答案`:0.00000001207; `A`:0.000000008056; `(`:0.000000007610; `(G`:0.000000006965 |

### after_open_paren

| Sample | p_A | p_B | A+B mass | Top 20 token:probability |
|---|---:|---:|---:|---|
| M/q00 even-first | 0.0316227 | 0.00000839 | 0.0316311 | `Answer`:0.6078; `Blank`:0.05814; ` A`:0.03162; `Just`:0.02176; `As`:0.02137; `To`:0.01522; ` **`:0.01491; `For`:0.01450; `An`:0.01327; `ANS`:0.01114; `Visited`:0.01049; `If`:0.009724; `You`:0.008059; `There`:0.007184; `Actually`:0.006623; `Here`:0.005367; `Click`:0.003387; `1`:0.003107; `Considering`:0.002977; `Re`:0.002614 |
| B/q00 even-first | 0.966253 | 0.000393967 | 0.966647 | ` A`:0.5071; `А`:0.3576; `Ａ`:0.09446; `Answer`:0.02536; `𝐴`:0.003467; `أ`:0.002815; `An`:0.002595; `Α`:0.002540; `𝐀`:0.0008778; `AX`:0.0003515; `В`:0.0002204; `A`:0.0002061; `Б`:0.0001728; `Ā`:0.0001581; `Å`:0.0001394; `\tA`:0.0001389; `Blank`:0.0001270; `Al`:0.0001098; `_A`:0.00009691; `Á`:0.00008418 |
| M/q01 odd-first | 0.00162929 | 0.00140328 | 0.00303257 | `Answer`:0.7857; `If`:0.02780; `Blank`:0.01888; `For`:0.01883; ` **`:0.01490; `You`:0.01310; `To`:0.01158; `As`:0.01010; `Just`:0.007142; `There`:0.006828; `Here`:0.006369; `Visited`:0.006338; `Click`:0.005291; `1`:0.004587; `Actually`:0.004445; `An`:0.003657; `Option`:0.002605; ` A`:0.001629; `On`:0.001493; ` B`:0.001403 |
| B/q01 odd-first | 0.000000001558 | 0.998922 | 0.998922 | `Б`:0.6158; ` B`:0.2523; `Ｂ`:0.1245; `B`:0.002993; `𝐵`:0.002585; `Answer`:0.0006779; `Β`:0.0004746; `BO`:0.0001502; `В`:0.0001224; `BC`:0.00005068; `𝐁`:0.00004680; `\tB`:0.00003732; ` Б`:0.00003624; `王`:0.00002650; `Blank`:0.00002335; `С`:0.00001773; `=B`:0.00001304; `Bl`:0.000008343; `_B`:0.000004695; `Բ`:0.000004030 |

### after_answer_colon_open_paren (selected)

| Sample | p_A | p_B | A+B mass | Top 20 token:probability |
|---|---:|---:|---:|---|
| M/q00 even-first | 0.999988619 | 0.00000203067 | 0.999990649 | `A`:1; `C`:0.000007820; `B`:0.000002031; `E`:0.0000006354; `D`:0.0000003872; ` **`:0.0000002466; `a`:0.0000001785; `А`:0.00000002226; `Ａ`:0.00000001143; `G`:0.000000008909; `H`:0.000000008453; `𝐀`:0.000000006108; `R`:0.000000005788; `I`:0.000000005161; ` A`:0.000000004607; `L`:0.000000004409; `أ`:0.000000003759; `F`:0.000000003680; `T`:0.000000003637; `S`:0.000000002339 |
| B/q00 even-first | 0.999999999807 | 0.000000000188 | 0.999999999995 | `A`:1; `B`:0.0000000001878; `C`:0.000000000004771; `D`:0.0000000000003992; `А`:0.0000000000002583; `Ａ`:0.00000000000003633; ` A`:0.00000000000001916; `E`:0.00000000000001391; `R`:0.000000000000002926; `أ`:0.000000000000002461; `T`:0.000000000000001674; `L`:0.000000000000001252; `N`:0.000000000000001053; `_A`:0.0000000000000005390; `G`:0.0000000000000004392; `M`:0.0000000000000004332; `Α`:0.0000000000000003577; `F`:0.0000000000000003337; `Ax`:0.0000000000000002920; `S`:0.0000000000000001183 |
| M/q01 odd-first | 0.000000226631 | 0.999997130 | 0.999997356 | `B`:1; `C`:0.000002512; `A`:0.0000002266; `D`:0.00000006798; ` **`:0.00000005015; `E`:0.00000001075; `Band`:0.000000001693; `Ｂ`:0.0000000009224; ` B`:0.0000000009119; `Б`:0.0000000006741; `G`:0.0000000002452; `b`:0.0000000001955; `В`:0.0000000001459; `𝐵`:0.00000000009539; `T`:0.00000000009151; `H`:0.00000000009115; `S`:0.00000000007551; `F`:0.00000000006785; `I`:0.00000000006057; `R`:0.00000000005787 |
| B/q01 odd-first | 0.00000000001718 | 0.999999999973 | 0.999999999990 | `B`:1; `A`:0.00000000001718; `C`:0.00000000001009; `Ｂ`:0.000000000001045; ` B`:0.0000000000004361; `Б`:0.0000000000002768; `D`:0.0000000000001779; `E`:0.00000000000004643; `R`:0.0000000000000009630; `b`:0.0000000000000007375; `𝐵`:0.0000000000000006307; `T`:0.0000000000000006174; `L`:0.0000000000000003961; `F`:0.0000000000000002957; `_B`:0.0000000000000002957; `M`:0.0000000000000002922; `BB`:0.0000000000000001483; `N`:0.0000000000000001370; `S`:0.0000000000000001350; `G`:0.0000000000000001043 |

### assistant_prefill_answer_is_open_paren

| Sample | p_A | p_B | A+B mass | Top 20 token:probability |
|---|---:|---:|---:|---|
| M/q00 even-first | 0.999933682 | 0.00000833923 | 0.999942021 | `A`:0.9999; ` **`:0.00003117; `C`:0.00001259; `a`:0.00001111; `B`:0.000008339; `D`:0.000001392; `E`:0.000001208; `𝐀`:0.0000001200; `А`:0.0000001130; `Ａ`:0.00000009456; `**)`:0.00000007792; `أ`:0.00000005866; ` A`:0.00000005314; `H`:0.00000003963; `𝑨`:0.00000003802; `**`:0.00000002444; `𝐴`:0.00000002274; `G`:0.00000001911; `R`:0.00000001579; `An`:0.00000001095 |
| B/q00 even-first | 0.999999997 | 0.000000003013 | 0.999999999979 | `A`:1; `B`:0.000000003013; `C`:0.00000000001901; `А`:0.000000000002753; `D`:0.000000000002060; ` A`:0.000000000001380; `Ａ`:0.0000000000003897; `أ`:0.00000000000006183; `E`:0.00000000000005507; `_A`:0.00000000000001831; `T`:0.00000000000001245; `R`:0.000000000000006715; `𝐀`:0.000000000000004531; `L`:0.000000000000004159; `N`:0.000000000000001741; `Ax`:0.000000000000001726; `Α`:0.000000000000001708; `\tA`:0.000000000000001470; `M`:0.000000000000001415; `a`:0.000000000000001396 |
| M/q01 odd-first | 0.000000714315 | 0.999972657 | 0.999973371 | `B`:0.99997; ` **`:0.00002256; `C`:0.000003835; `A`:0.0000007143; `D`:0.0000001499; `E`:0.00000002876; ` B`:0.00000002229; `Ｂ`:0.00000001778; `b`:0.00000001418; `**)`:0.00000001192; `**`:0.000000007993; `Band`:0.000000005742; `Б`:0.000000004961; `𝐵`:0.000000003048; `a`:0.000000002111; `В`:0.000000001204; `G`:0.0000000008366; `the`:0.0000000007467; `###`:0.0000000005519; `H`:0.0000000005376 |
| B/q01 odd-first | 0.0000000008669 | 0.999999998998 | 0.999999999863 | `B`:1; `A`:0.0000000008669; `C`:0.0000000001309; ` B`:0.00000000001633; `Ｂ`:0.000000000006329; `D`:0.000000000003152; `Б`:0.000000000002314; `E`:0.0000000000007777; ` **`:0.00000000000002876; `_B`:0.00000000000001631; `T`:0.00000000000001349; `𝐵`:0.00000000000001011; `R`:0.000000000000007952; `**)`:0.000000000000005953; `L`:0.000000000000004185; `BB`:0.000000000000002321; `\tB`:0.000000000000002312; `В`:0.000000000000001897; `M`:0.000000000000001875; `F`:0.000000000000001856 |

## Variant sets and row fields

The code defines the ASCII candidate spellings as `A: ("A", " A", "(A", " (A")` and `B: ("B", " B", "(B", " (B")`. Only spellings that the pinned tokenizer encodes as one token are included. The lookalike inventory is A = Cyrillic `А`, Greek `Α`, fullwidth `Ａ`, mathematical bold `𝐀`, mathematical italic `𝐴`, and mathematical bold italic `𝑨`; B = Cyrillic `Б` and `В`, Greek `Β`, fullwidth `Ｂ`, mathematical bold `𝐁`, mathematical italic `𝐵`, and mathematical bold italic `𝑩`. A lookalike is added to that row's variant IDs only if its exact glyph (optionally preceded by a space or `(` / ` (`) is in the top 20. The visible per-sample sets are also stored on each row as token IDs and decoded surfaces.

For each row, p_A and p_B are full-softmax probability sums over their active variant IDs, and mass is p_A + p_B. The even-minus-odd logit is ln(p_even) - ln(p_odd), using the option-to-concept map. `low_mass` is true below 0.2 and those rows remain in scoring and mass summaries. `score` predicts A or B from the sign of p_A - p_B, then maps that letter through the counterbalanced concepts; it does not use `greedy_letter`. Each row now also carries a hash of the exact readout context, including its assistant prefix.

## Stored-context rebuild and commands

The original turn-2 hash alone is insufficient. The stored turn-1 rows do contain `prompt_ids` and `context_reply_ids`; the turn-2 question is regenerated deterministically from question index and metric kind, and the arm/reader/source-model mapping is explicit in each row. Rebuilding the base user context from those token IDs reproduced the stored turn-2 hashes. The selected assistant prefix is then applied from the code's fixed position map and hashed separately as `ab_readout_context_sha256`. The readout-only path re-encodes these stored tokens to rebuild KV caches; it does not regenerate turn 1.

Run from the repository root to create a new combined output directory while retaining the original shards:

    HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 /home/elisepc/neuron-probe/.venv/bin/python part34.py run --readout-only --out results/part34/readout-fold3 --rows results/part34/shard-00/rows.jsonl results/part34/shard-01/rows.jsonl results/part34/shard-02/rows.jsonl results/part34/shard-03/rows.jsonl results/part34/shard-04/rows.jsonl results/part34/shard-05/rows.jsonl results/part34/shard-06/rows.jsonl results/part34/shard-07/rows.jsonl results/part34/shard-08/rows.jsonl results/part34/shard-09/rows.jsonl --direction-file /mnt/e/neuron-probe/hf/hub/models--Sauers--qwen3-1.7b-cat-zigzag/snapshots/2c871b5cc51e5be9f4f6ae258685448518191c4b/cat_direction.safetensors --device cpu --threads 4
    /home/elisepc/neuron-probe/.venv/bin/python part34.py score --out results/part34/readout-fold3

The full stored dataset has 12,239 A/B rows and 3,280 unique cache-rebuild contexts. The B sanity subset took 314.7 seconds for 160 rows plus 80 cache builds; four direct cache benchmarks had a median of 1.54 seconds each. Extrapolating the complete row path gives approximately **5.5 CPU hours** (rough planning range: 5–6 hours, including about 15 seconds to load the model and adapter). The full 12,239-row readout-only pass was not run here.

## Test output

Command: `PYTHONDONTWRITEBYTECODE=1 /home/elisepc/neuron-probe/.venv/bin/python -m unittest -v test_part34.py`

Result: **Ran 14 tests in 1.623s — OK.** Coverage includes variant sums, low-mass flags, both option orders, a mapping-swap control that changes the scored answer, a tiny random Qwen model, context hash verification, readout-only rebuilding, and scoring.

## SUMMARY

The selected `Answer: (` assistant prefix raises median A+B mass to 0.999998678 across the four required M/B and option-order samples. B's recomputed honest-request accuracy is 131/160 (81.9%).

## FINDINGS

`part34.py` now sums letter variants, records low-mass flags and readout-context hashes, scores by the mapped sum difference, places the B sanity result first in score summaries, reports mass distributions by model and condition, and provides `run --readout-only` over stored turn-1 tokens. The full-dataset recomputation remains to be run with the command above.

## BLOCKERS

None. The full 12,239-row recomputation was not run in this worker; the report gives the measured subset result and full-run estimate.
