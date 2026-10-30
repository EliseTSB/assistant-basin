# Part 116 cold-check fold log, round 1

All reported values below come from the saved result rows, ranking tensors loaded on CPU, or registered source code. The README keeps all five scored verdicts as held. The table was regenerated from the saved rows. No evidence files or project code were edited.

## Findings

### 1. Folded

Set the final README line to the required cold-check status. This is the instructed status text, not an experimental result.

```sh
python3 -c 'from pathlib import Path; print(Path("README.md").read_text().splitlines()[-1])'
```

Output: `Cold check: Haiku 5.5, round 1, 2026-10-29, 12 findings, all folded (results/part116/COLD-CHECK-r1.md, results/part116/FOLD-r1.md).`

### 2. Folded

Added three readout/generation examples, the single-token flip distinction, the untested batch-size explanation, and the note that predictions 116.2 and 116.4 do not rely on the k=50 flip count.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import json; d=json.load(open("dla/results115.json")); rows={k:next(r for r in d["rows"] if r["role"]=="candidate" and r["k"]==k) for k in (50,100,500)}; gens=d["generations"]; indices=((50,2),(100,2),(500,12)); [print("example",k,i,rows[k]["heldout"][i]["top"],rows[k]["heldout"][i]["logodds"],gens[str(k)][i]["question"],gens[str(k)][i]["continuation"]) for k,i in indices]; g=gens["50"]; print("k50_readout_flips",sum(not h["argmax_is_q"] for h in rows[50]["heldout"]),"continuations_not_starting_Q",sum(not x["continuation"].lstrip("* ").startswith("Q") for x in g))'
```

Output: k=50, held-out index 2, top token `K`, Q log-odds `-0.7213238477706909`, question `Identify yourself, please.`, continuation starts `Qwen3.5**`; k=100, same question, top `K`, continuation starts `Claude**`; k=500, index 12, question `What's your name, assistant?`, top `not`, continuation starts `Claude**`. At k=50: 4 readout flips and 3 continuations not starting with Q.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import sys; sys.path.insert(0,"part115"); import howmany115 as h; print("readout_chunk_rows",h.CHUNK,"generation_candidate_rows",len(h.KS),"k_values",h.KS)'
```

Output: readout chunks of `50`; generation batches `9` candidate k rows. The differing batch sizes are only a possible cause; they have not been tested.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import json; d=json.load(open("results/part115/results115.json")); row=next(r for r in d["rows"] if r["role"]=="candidate" and r["k"]==500); gens=d["generations"]["500"]; print([(i,h["top"],g["continuation"].lstrip("* ")[:24]) for i,(h,g) in enumerate(zip(row["heldout"],gens)) if not g["continuation"].lstrip("* ").startswith(h["top"])])'
```

Output: `[(1, 'Chat', 'AI Assistant**, your per'), (11, 'Deep', 'Chat**, a friendly and k'), (13, 'LL', 'DeepSeek V3**, a large l')]`.

### 3. Folded

Explained that direct-write is one-sided because the ranker stores `attr = -direct`. It selects positive writers and does not select the most negative writer.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import torch; a=torch.load("rank_act.pt",map_location="cpu",weights_only=False); s=a["score"].flatten(); v=s[torch.topk(s.abs(),500).indices]; w=a["score"].shape[1]; i=int(s.abs().argmax()); print("activity_top500_negative_positive_absge10",int((v<0).sum()),int((v>0).sum()),int((v.abs()>=10).sum())); print("activity_top_unit",f"L{i//w} N{i%w}",f"z={float(s[i]):.6f}","base_std_stored",("base_std" in a)); d=torch.load("rank_dla.pt",map_location="cpu",weights_only=False); q=d["score"].flatten(); ids=torch.topk(q,100).indices; ids500=torch.topk(q,500).indices; j=int(q.argmin()); w=d["score"].shape[1]; print("direct_top100_min_max",f"{float(q[ids].min()):.6f}",f"{float(q[ids].max()):.6f}","top500_min",f"{float(q[ids500].min()):.6f}"); print("direct_most_negative",f"L{j//w} N{j%w}",f"{float(q[j]):.6f}","rank",int((q>q[j]).sum())+1)'
```

Output: direct top-100 range `0.073569` to `3.041008`; top-500 minimum `0.023116`; most negative writer `L31 N4200`, score `-1.947505`, rank `294912`.

### 4. Folded

Defined activity as absolute z-score in either direction, reported the sign split and large-|z| count, and noted that the saved file omits `base_std`.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import torch; a=torch.load("rank_act.pt",map_location="cpu",weights_only=False); s=a["score"].flatten(); v=s[torch.topk(s.abs(),500).indices]; w=a["score"].shape[1]; i=int(s.abs().argmax()); print("activity_top500_negative_positive_absge10",int((v<0).sum()),int((v>0).sum()),int((v.abs()>=10).sum())); print("activity_top_unit",f"L{i//w} N{i%w}",f"z={float(s[i]):.6f}","base_std_stored",("base_std" in a))'
```

Output: among the top 500 by absolute z, `260` are negative, `240` positive, and `348` have absolute z at least `10`; top unit `L28 N2283` has z `164.154587`; `base_std_stored=False`.

### 5. Folded

Specified that the all-ones gate checks only the last-position logits for the first held-out phrasing, “Who am I speaking with?”. Defined both gates in the Method section.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import plot116 as p; d={n:p.load_json(f) for n,f in p.FILES.items()}; p.report(d,d["activity"]["ks"])'
```

Output: both all-ones checks are `True`; both none-row mean absolute differences are `0.045811`. The registered tolerance is `0.1`, from `part115/howmany115.py`. The gate prompt is `HELDOUT[0]`, “Who am I speaking with?”.

### 6. Folded

Added the bold-marker scope to Limits. Described Part 115b as a separate prompt setup with a plain-text instruction and three other openings, following only its README’s description of transfer from rankings selected after “I am **”.

```sh
rg -n 'The tested scope was|The openings were|These rankings were selected' results/part115b/README.md
```

Output: the cited README states it used a plain-text instruction and `16` held-out phrasings, lists three openings (`My name is`, `My name is "`, and `My name is **`), and says the rankings were selected after “I am **” and that these are transfer results. This count describes the cited prompt setup, not a new Part 116 result.

### 7. Folded

Defined log-odds, the Part 114 gradient-attribution file, first-word counting, k and k*, the matched random sets, the withdrawal rule, and both gates at first use. The fixed sizes and thresholds are from registered source code and design, not inferred results.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import sys; sys.path[0:0]=["part114","part115"]; import qneuron114,howmany115; print("selection_heldout_ordinary",len(qneuron114.SELECTION),len(qneuron114.HELDOUT),len(howmany115.ORDINARY)); print("N_SETS_POOL_EXCLUDE_GATETOL_CHUNK",howmany115.N_SETS,howmany115.POOL,howmany115.EXCLUDE,howmany115.GATE_TOL,howmany115.CHUNK); print("k_grid",howmany115.KS,"first_gate_phrase",qneuron114.HELDOUT[0])'
```

Output: selection/held-out/ordinary prompt counts `8 16 4`; `N_SETS, POOL, EXCLUDE, GATE_TOL, CHUNK = 10, 20, 500, 0.1, 50`; k grid `[1, 2, 5, 10, 20, 50, 100, 200, 500]`; first gate phrase `Who am I speaking with?`. The threshold of `8` flips is copied from the registered design.

### 8. Folded

Scoped “no flips” to the fixed-seed matched samples at each k and ranking. Identified the Part 114 attribution file as the gradient source and limited the single-neuron comparison to its measured candidates.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import plot116 as p; d={n:p.load_json(f) for n,f in p.FILES.items()}; p.report(d,d["activity"]["ks"])'
```

Output: 10 matched sets per ranking per k; maximum random flips are `0` at every tested k; the activity unit’s Part 114 gradient rank is `11`.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import json; from pathlib import Path; s=json.loads(Path("results/part114/score.json").read_text()); cs=sorted(s["candidates"],key=lambda x:x["mean_dlogodds"]); print("part114_candidates",len(cs),"two_largest_drops",[(x["layer"],x["neuron"],x["mean_dlogodds"]) for x in cs[:2]])'
```

Output: `part114_candidates 20 two_largest_drops [(28, 2283, -1.8982), (27, 6233, -0.9458)]`.

### 9. Folded

Updated the title to state that activity reached its threshold only at 500, with 11 flips.

```sh
MPLCONFIGDIR=/tmp/part116-mplconfig python -c 'import plot116 as p; d={n:p.load_json(f) for n,f in p.FILES.items()}; p.report(d,d["activity"]["ks"])'
```

Output: `activity_first_k_at_8_of_16=500`; activity flips over the k grid are `0,0,0,0,0,0,0,0,11`. Direct write first reaches the threshold at `100`.

### 10. Folded

Added the requested input and scoring dependencies to Files: `part116/score116.py`, `results/part114/attr114.pt`, `results/part114/results114.json`, `results/part115/results115.json`, and `results/part115/score.json`. No experimental number was changed for this finding.

### 11. Folded

Regenerated the Markdown table from `plot116.py`’s result-loading and flip-count helpers. The one-liner below prints the table directly from the saved activity, direct-write, and gradient rows.

```sh
python -c 'import plot116 as p; d={n:p.load_json(f) for n,f in p.FILES.items()}; ks=d["activity"]["ks"]; print("matched_sets_per_ranking_per_k="+str(d["activity"]["n_sets"])); print("| k | Part 114 gradient | Part 116 activity | Part 116 direct write | Max matched random sets |"); print("| ---: | ---: | ---: | ---: | ---: |"); [(lambda k: print("| "+str(k)+" | "+" | ".join(str(p.flips(p.row_for(d[n], "candidate", k)))+" / "+str(len(p.row_for(d[n], "candidate", k)["heldout"])) for n in ("gradient", "activity", "direct-write"))+" | "+str(max(p.maximum_control_flips(d[n], k) for n in ("activity", "direct-write")))+" / "+str(len(p.row_for(d["activity"], "none", 0)["heldout"]))+" |"))(k) for k in ks]'
```

Output:

```text
matched_sets_per_ranking_per_k=10
| k | Part 114 gradient | Part 116 activity | Part 116 direct write | Max matched random sets |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 2 | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 5 | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 10 | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 20 | 0 / 16 | 0 / 16 | 0 / 16 | 0 / 16 |
| 50 | 0 / 16 | 0 / 16 | 4 / 16 | 0 / 16 |
| 100 | 3 / 16 | 0 / 16 | 16 / 16 | 0 / 16 |
| 200 | 11 / 16 | 0 / 16 | 16 / 16 | 0 / 16 |
| 500 | 16 / 16 | 11 / 16 | 16 / 16 | 0 / 16 |
```

### 12. Folded

The cold check says to mention the corrected single-drop sum only if the README cites it. It does not, so no README change was needed. This search returned no matches (exit code `1`, meaning no match):

```sh
rg -n 'sum_single_drops_top20|-8\.42|-8\.4164' README.md
```

## Parent control

Every mailbox pull returned no messages. No parent message changed the plan, and none were declined or deferred.
