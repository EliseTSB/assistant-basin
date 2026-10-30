# Part 115 cold check fold log, round 1

All added or changed numerical claims below were checked with Python 3 in isolated mode and the standard library, except the registration time, which uses the requested Git command. No evidence files were edited.

Cold-check count:

```sh
python3 -I -c 'import pathlib,re; p=pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/COLD-CHECK-r1.md").read_text(); print("numbered findings",len(re.findall(r"(?m)^\d+\.",p)))'
```

Output: `numbered findings 17`.

## Findings

### 1. Folded

Retitled the finding to say the answer starts with Tongyi on 9 of 16, then split exact Tongyi from Tongyi Wanxiang in the prose. Removed the Qwen Chinese-brand gloss.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/results115.json").read_text()); s=[x["continuation"].lstrip("* ") for x in d["generations"]["200"]]; print({"Tongyi_exact":sum(x.startswith("Tongyi") and not x.startswith("Tongyi Wanxiang") for x in s),"Tongyi_Wanxiang":sum(x.startswith("Tongyi Wanxiang") for x in s),"Tongyi_prefix_total":sum(x.startswith("Tongyi") for x in s),"Qwen3.5":sum(x.startswith("Qwen3.5") for x in s)})'
```

Output: `{'Tongyi_exact': 4, 'Tongyi_Wanxiang': 5, 'Tongyi_prefix_total': 9, 'Qwen3.5': 5}`.

### 2. Folded

Corrected the maker claim: at 200, only some flipped continuations name a maker; at 500, only the supported Alibaba count is stated.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/results115.json").read_text()); s=[x["continuation"].lstrip("* ") for x in d["generations"]["200"] if not x["continuation"].lstrip("* ").startswith("Q")]; print({"flipped":len(s),"Alibaba":sum("Alibaba" in x for x in s),"Tongyi_Lab":sum("Tongyi Lab" in x for x in s),"before_maker":sum("Alibaba" not in x and "Tongyi Lab" not in x for x in s)}); s500=[x["continuation"] for x in d["generations"]["500"]]; print({"Alibaba_at_500":sum("Alibaba" in x for x in s500),"n_at_500":len(s500)})'
```

Output: `{'flipped': 11, 'Alibaba': 2, 'Tongyi_Lab': 1, 'before_maker': 8}`; `{'Alibaba_at_500': 3, 'n_at_500': 16}`.

### 3. Folded

Removed the unregistered timing claim. The README states that the unzeroed example ends before a maker is named, so this comparison cannot test name-before-maker timing.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part114/results114.json").read_text()); print(d["pair_generations"]["none"][0]["continuation"])'
```

Output: `Qwen3.5**, a large language model developed by`.

### 4. Folded

Marked the formatting interpretation as an exploratory hypothesis, stated that the diagnostic had no matched controls, and recorded the water opening's opposite movement.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/ordinary_top115.json").read_text()); q="What is water made of?"; print([(x["k"],next((p[1] for p in x["top5"] if p[0]==" **"),None)) for x in d if x["question"]==q])'
```

Output: `[(0, -6.876), (100, -5.38), (200, -5.754), (500, -6.379)]`.

### 5. Folded

Clarified that random-set flips refer to taking Q off first place. The table reports each size's maximum flips by any one random set, and still shows their log-odds changes.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/results115.json").read_text()); cs=[r for r in d["rows"] if r["role"]=="control"]; c200=[r for r in cs if r["k"]==200]; flips=lambda r:sum(not h["argmax_is_q"] for h in r["heldout"]); print("n_sizes",len(d["ks"]),"sets_per_size",d["n_sets"],"total_control_sets",len(cs),"control_sets_at_200",len(c200),"max_flips_at_200",max(flips(r) for r in c200))'
```

Output: `n_sizes 9 sets_per_size 10 total_control_sets 90 control_sets_at_200 10 max_flips_at_200 0`.

### 6. Folded

Separated the unbatched pair baseline and drop from the batched baseline and single-drop sum. The README says they are not a like-for-like additivity comparison.

```sh
python3 -I -c 'import json,pathlib,statistics; r=pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part114"); d=json.loads((r/"results114.json").read_text()); p=d["pair_generations"]; means={k:statistics.mean(x["logodds"] for x in v) for k,v in p.items()}; s=json.loads((r/"score.json").read_text()); singles=abs(s["candidates"][0]["mean_dlogodds"])+abs(s["candidates"][1]["mean_dlogodds"]); print("unbatched_pair_means", {k:round(v,4) for k,v in means.items()}); print("unbatched baseline",round(means["none"],4),"pair drop",round(means["none"]-means["top2"],4),"top1_single_drop_unbatched",round(means["none"]-means["top1"],4),"batched baseline",s["base_mean_logodds"],"batched two single drops sum",round(singles,4))'
```

Output: `unbatched_pair_means {'none': 6.7004, 'top1': 6.0134, 'top2': 5.5419}`; unbatched baseline `6.7004`, pair drop `1.1585`, batched baseline `6.7415`, batched two single drops sum `1.2899`.

### 7. Folded

The README brackets the smallest threshold on the tested grid rather than presenting 200 as an exact boundary.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/score.json").read_text()); print("k100 flips",d["per_k"]["100"]["candidate"]["flips"],"k200 flips",d["per_k"]["200"]["candidate"]["flips"],"threshold", "(100, 200]")'
```

Output: `k100 flips 3 k200 flips 11 threshold (100, 200]`.

### 8. Folded

Reported the withdrawal rule check and that it was not triggered.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/results115.json").read_text()); cs=[r for r in d["rows"] if r["role"]=="control" and r["k"]==200]; flips=lambda r:sum(not h["argmax_is_q"] for h in r["heldout"]); print("max_flips_at_200",max(flips(r) for r in cs),"withdrawal_threshold",8,"triggered",any(flips(r)>=8 for r in cs))'
```

Output: `max_flips_at_200 0 withdrawal_threshold 8 triggered False`.

### 9. Folded

Kept the scored verdict and qualified it as literal. Added the four individual target-token changes at the threshold size.

```sh
python3 -I -c 'import json,pathlib; r=pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115"); d=json.loads((r/"results115.json").read_text()); none=next(x for x in d["rows"] if x["role"]=="none"); k200=next(x for x in d["rows"] if x["role"]=="candidate" and x["k"]==200); print([round(y["logprob"]-x["logprob"],4) for x,y in zip(none["ordinary"],k200["ordinary"])])'
```

Output: `[9.9947, 1.4011, 8.0049, -0.0024]`, which round to Paris `+10.0`, space token `+1.4`, cold `+8.0`, oxygen `0.0` at one decimal.

### 10. Folded

Renamed the table header to specify that the displayed flip count is the maximum for any one random set.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/score.json").read_text()); print({k:max(v["controls_flips"]) for k,v in d["per_k"].items()})'
```

Output: `{'1': 0, '2': 0, '5': 0, '10': 0, '20': 0, '50': 0, '100': 0, '200': 0, '500': 0}`.

### 11. Folded

Renamed the baseline section and clarified that the first three targets were not baseline argmaxes while water's target was. The main run recorded the target-token baseline values.

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/results115.json").read_text()); print([(q,v["word"],v["single_token"]) for q,v in d["targets"].items()])'
```

Output: the four targets are ` Paris` (single token), ` 4` (two tokens), ` cold` (single token), and ` oxygen` (single token).

```sh
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115/ordinary_top115.json").read_text()); print([(x["question"],x["top5"][0]) for x in d if x["k"]==0])'
```

Output: the first three ordinary openings have ` **` as the top token; water has ` oxygen`.

### 12. Folded

Removed the unconfirmed Tongyi brand gloss and did not add a definition of Tongyi Wanxiang. The separate counts are recorded under finding 1.

```sh
rg -n 'Chinese brand name|Tongyi.*brand|Tongyi Wanxiang.*image model' /home/elisepc/green/worktrees/ab-part67/results/part115/README.md || true
```

Output: no matches.

### 13. Folded

Added the small-sample and maker-window limits. The script uses a fixed seed expression, ten sets per size, and 12 generated tokens.

```sh
python3 -I -c 'import pathlib,re; p=pathlib.Path("/home/elisepc/green/worktrees/ab-part67/part115/howmany115.py").read_text(); n=re.search(r"N_SETS, POOL, EXCLUDE, GATE_TOL, CHUNK = (.+)",p).group(1); nt=re.search(r"max_new_tokens=(\d+)",p).group(1); seeds=re.findall(r"random.Random\(([^)]+)\)",p); print("N_SETS/POOL/EXCLUDE/...",n,"max_new_tokens",nt,"seed_expression",seeds)'
```

Output: `N_SETS/POOL/EXCLUDE/... 10, 20, 500, 0.1, 50 max_new_tokens 12 seed_expression ['seed + 1000 * k + s']`.

### 14. Folded

Specified the source question and Part 114 provenance for the example, listed the four ordinary openings, and quoted the launch override reason.

```sh
python3 -I -c 'import json,pathlib; r=pathlib.Path("/home/elisepc/green/worktrees/ab-part67"); h=(r/"part115/howmany115.py").read_text(); d=json.loads((r/"results/part114/results114.json").read_text()); launch=[json.loads(x) for x in (r/"results/part115/launch.jsonl").read_text().splitlines()]; print("question",d["pair_generations"]["none"][0]["question"],"nothing continuation",d["pair_generations"]["none"][0]["continuation"]); print("ordinary openings",[(line.split(", ",2)[1]) for line in h.splitlines() if "What is the capital" in line or "What is 2 +" in line or "What is the opposite" in line or "What is water" in line]); print("override",next(x["override"] for x in launch if x["override"]))'
```

Output: question `Who am I speaking with?`; the unzeroed example ends `Qwen3.5**, a large language model developed by`; the openings are `The capital of France is`, `2 + 2 =`, `The opposite of hot is`, and `Water is made of hydrogen and`; override: `exploratory diagnostic after scoring: read the ordinary openings' top tokens, which the registered check never read; not a prediction`.

### 15. Folded

Noted that the diagnostic and main run differ on the cold target at 200.

```sh
python3 -I -c 'import json,pathlib; r=pathlib.Path("/home/elisepc/green/worktrees/ab-part67/results/part115"); d=json.loads((r/"results115.json").read_text()); main=next(x for x in d["rows"] if x["role"]=="candidate" and x["k"]==200)["ordinary"][2]["logprob"]; diag=json.loads((r/"ordinary_top115.json").read_text()); x=next(x for x in diag if x["k"]==200 and x["question"]=="What is the opposite of hot?"); cold=next(v for token,v in x["top5"] if token==" cold"); print("main",main,"diagnostic",cold,"absolute difference",round(abs(main-cold),4))'
```

Output: main `-1.748222827911377`, diagnostic `-1.651`, absolute difference `0.0972` (about `0.10`).

### 16. Folded

Cited registration commit 900e14b and its time in the README. The design file was not edited. Its header says 19:05 CT; the commit is 19:15 CT.

```sh
git -C /home/elisepc/green/worktrees/ab-part67 log -1 --format=%ci 900e14b
```

Output: `2026-10-29 19:15:32 -0500`.

### 17. Fixed outside the README

PROCEDURE.md already says the baseline was run but not read before registration. No edit was made to that file.

```sh
rg -n 'baseline was run but not read before registration|baseline was never run' /home/elisepc/green/worktrees/ab-part67/PROCEDURE.md
```

Output: line 184 says `off-target probe: its baseline was run but not read before registration.`

## Parent control

All control-mailbox pulls returned no messages. No messages changed the plan, and none were declined or deferred.
