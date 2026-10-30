# Part 115b cold-check fold, round 1

All numerical checks below use `python3 -I` and the standard library from the repository root. The run rows are the source for Part 115b counts. No tests were run.

## Finding 1: title states the table thresholds

- Folded: the title gives the gradient threshold in all openings and each direct-write threshold.
- One-liner: `python3 -I -c 'import json; d=json.load(open("results/part115b/results115b.json")); print({r:{c:next((x["k"] for x in d["rankings"][r]["rows"] if x["role"]=="candidate" and sum(not y["argmax_is_target"] for y in x[c])>=8),None) for c in ("plain","quote","bold")} for r in ("gradient","direct")})'`
- Output: `{'gradient': {'plain': 500, 'quote': 500, 'bold': 500}, 'direct': {'plain': 100, 'quote': 50, 'bold': 50}}`.

## Finding 2: target tokens

- Folded: plain targets ` Q`; quote and bold target `Q`.
- One-liner: `python3 -I -c 'import json; d=json.load(open("results/part115b/results115b.json")); print(d["contexts"], d["target_ids"])'`
- Output: `plain=['My name is', ' Q']; quote=['My name is "', 'Q']; bold=['My name is **', 'Q']; ids={'plain': 1167, 'quote': 48, 'bold': 48}`.

## Finding 3: exploratory context/token section

- Folded: labeled the section exploratory. It states that quote and bold share thresholds within each ranking. Plain and quote have the same gradient threshold and different direct-write thresholds. The plain/quote contrast also removes the opening quote mark.
- One-liner: `python3 -I -c 'import json; d=json.load(open("results/part115b/score.json")); print({r:{c:d["rankings"][r][c]["k_star"] for c in ("plain","quote","bold")} for r in ("gradient","direct")})'`
- Output: `gradient={'plain': 500, 'quote': 500, 'bold': 500}; direct={'plain': 100, 'quote': 50, 'bold': 50}`.

## Finding 4: probe provenance

- Folded: separated pass 2 from pass 3, described how they motivated the instruction and openings, and noted their shared phrasing set.
- One-liner: `python3 -I -c 'import json,collections; a=json.load(open("results/part115b/probe115b.json")); b=json.load(open("results/part115b/probe115b-pass3.json")); C=lambda d,o,p:collections.Counter(x["top5"][0][1] for x in d["heldout"] if x["opening"]==o and x.get("plain",False)==p); s=" Please answer in plain text, without markdown."; x={z["question"].removesuffix(s) for z in a["heldout"] if z["opening"]=="My name is" and z.get("plain")}; y={z["question"].removesuffix(s) for z in b["heldout"] if z["opening"]=="My name is \"" and z.get("plain")}; print(a.get("pass"),C(a,"My name is",False),C(a,"My name is",True),b.get("pass"),len(b["heldout"]),C(b,"My name is \"",True),x==y,len(x))'`
- Output: pass 2 has no `pass` value; without instruction ` **` was first on 12/16, and with instruction ` Q` was first on 16/16. Pass 3 reports `pass=3`, 240 rows, and `Q` first on 16/16 after the quote opening. Both used the same 16 phrasings.

## Finding 5: Amendment 1 fallback

- Folded: stated that waiting for more memory was not used, explained the allocator warm-up failure and skip, and cited “Amendment 1, note” in `PART115B-DESIGN.md`.
- One-liner: `python3 -I -c 'from pathlib import Path; s=Path("PART115B-DESIGN.md").read_text(); print(s.split("### Amendment 1, note",1)[1].strip())'`
- Output: the note says the fallback was not used; warm-up failed before data, was skipped at `b2e90a0` (20:11:39), and the run relaunched at 20:11:42 before data.

## Finding 6: comparison with Parts 115 and 116

- Folded: said the instruction and opening both changed, so their effects cannot be separated here. Restored the caveats that this bold opening is not Part 115’s prompt and an opening-specific set could need fewer neurons.
- One-liner: `python3 -I -c 'import json; a=json.load(open("results/part115/score.json")); b=json.load(open("results/part116/score.json")); c=json.load(open("results/part115b/score.json")); print("115 gradient",a["k_star"],a["per_k"]["200"]["candidate"]["flips"],a["per_k"]["500"]["candidate"]["flips"]); print("116 direct",b["direct"]["k_star"],b["direct"]["per_k"]["100"]["candidate"]["flips"]); print("115b",c["rankings"]["gradient"]["bold"]["k_star"],c["rankings"]["direct"]["bold"]["k_star"])'`
- Output: Part 115 gradient k*=200 with 11 flips at 200 and 16 at 500; Part 116 direct-write k*=100 with 16 flips; Part 115b bold thresholds are 500 gradient and 50 direct-write.

## Finding 7: 115b.7 literal verdict

- Folded: added two continuation examples and changed the verdict to `failed (literal: 'Chat' is not on the list)`.
- One-liner: `python3 -I -c 'import json; s=json.load(open("results/part115b/score.json")); d=json.load(open("results/part115b/results115b.json")); print(s["rankings"]["gradient"]["plain"]["replacements_k500"],s["verdicts"]["115b.7"]); print([x["continuation"] for x in d["rankings"]["gradient"]["generations"]["plain"]["500"] if x["continuation"].strip()=="Chat." or "Chat. I am a large language model trained by Google." in x["continuation"]])'`
- Output: non-Q first words are `{'Chat': 11, 'ChatBot': 2}`; the raw rows include `Chat.` and `Chat. I am a large language model trained by Google.`; saved verdict is `failed`.

## Finding 8: ordinary-probe scope

- Folded: stated that all six ran without the instruction, the selected answer was first for all six tested openings, five openings were bolded, and their unbolded alternatives put ` **` or ` typically` first.
- One-liner: `python3 -I -c 'import json; r=json.load(open("results/part115b/results115b.json")); p=json.load(open("results/part115b/probe115b-pass3.json")); one=lambda q,o:next(x for x in p["ordinary"] if x["question"]==q and x["opening"]==o)["top5"][0][1]; xs=[(q,o,t,one(q,o)) for q,o,t in r["ordinary"]]; bold=[(o,first,one(q,o[:-3].rstrip())) for q,o,t,first in xs if o.endswith("**")]; print(len(xs),sum(first.strip()==t.strip() for q,o,t,first in xs),xs,bold)'`
- Output: 6 of 6 selected targets are first. Five openings end in `**`; four unbolded variants put ` **` first, and the sky prompt puts ` typically` first. Water’s selected unbolded opening puts ` oxygen` first.

## Finding 9: matched random controls

- Folded: specified that the Part 115 gradient and Part 116 direct-write controls were reused unit for unit.
- One-liner: `python3 -I -c 'import json,collections; r=json.load(open("results/part115b/results115b.json")); L=lambda p:json.load(open(p)); S=lambda x:(x["k"],tuple(tuple(u) for u in x["units"])); print({n:(collections.Counter(S(x) for x in r["rankings"][n]["rows"] if x["role"]=="control")==collections.Counter(S(x) for x in L(p)["rows"] if x["role"]=="control"),sum(x["role"]=="control" for x in r["rankings"][n]["rows"])) for n,p in (("gradient","results/part115/results115.json"),("direct","results/part116/dla/results115.json"))})'`
- Output: exact signatures match for both rankings; each comparison has 90 controls.

## Finding 10: gate and loader comparison

- Folded: limited the all-ones gate to its one tested prompt. Said there was no standard-loader gate and labeled the pass-three comparison as approximate and post hoc.
- One-liner: `python3 -I -c 'import json,math,statistics; r=json.load(open("results/part115b/results115b.json")); p=json.load(open("results/part115b/probe115b-pass3.json")); n=next(x for x in r["rankings"]["gradient"]["rows"] if x["role"]=="none"); f=lambda z,t:math.log(math.exp(next(v[2] for v in z["top5"] if v[0]==t))/(sum(math.exp(v[2]) for v in z["top5"] if v[0]!=t)+max(0,1-sum(math.exp(v[2]) for v in z["top5"])))); specs=(("plain","My name is",1167),("quote","My name is \"",48),("bold","My name is **",48)); ds=[abs(f(z,t)-v["logodds"]) for c,o,t in specs for z,v in zip([x for x in p["heldout"] if x["opening"]==o and x.get("plain")],n[c])]; print(len(ds),round(statistics.mean(ds),2),r["gates"],next(x.strip() for x in open("part115b/howmany115b.py") if "HELDOUT[0]" in x))'`
- Output: 48 comparisons; mean absolute difference 0.03 nats using rounded top-five probe values. Stored gates passed. The runner selects `q114.HELDOUT[0]` and the plain opening for the all-ones comparison.

## Finding 11: definitions

- Folded: defined both rankings, flip, log-odds, k*, and the withdrawal rule before their use in the write-up.
- One-liner: `python3 -I -c 'import json,re; t=open("part115b/score115b.py").read(); d=json.load(open("results/part115b/results115b.json")); s=json.load(open("results/part115b/score.json")); print(int(re.search(r"FLIP = ([0-9]+)",t).group(1)),len(d["rankings"]["gradient"]["rows"][0]["plain"]),s["withdrawn"])'`
- Output: threshold 8, denominator 16, no withdrawals.

## Finding 12: amendment history

- Folded: attributed `--direct-load` to `f97ef14` and the warm-up skip to `b2e90a0`; described both as before any data and before the successful launch.
- One-liner: `python3 -I -c 'import subprocess; print(subprocess.run(["git","log","--format=%h %ci %s","--","PART115B-DESIGN.md","part115b/howmany115b.py"],check=True,text=True,capture_output=True).stdout)'`
- Output: `f97ef14` is timestamped 20:10:53; `b2e90a0` is timestamped 20:11:39. The launch record shows the successful relaunch at 20:11:42. Both amendment commits precede it.

## Finding 13: 115b.3 threshold bracket

- Folded: reported the tested-grid bracket and the two 500-row flip counts; the registered verdict remains failed.
- One-liner: `python3 -I -c 'import json; d=json.load(open("results/part115b/results115b.json")); print({c:{k:sum(not y["argmax_is_target"] for x in d["rankings"]["gradient"]["rows"] if x["role"]=="candidate" and x["k"]==k for y in x[c]) for k in (200,500)} for c in ("plain","quote")})'`
- Output: plain `{200: 2, 500: 13}`; quote `{200: 2, 500: 16}`.

## Finding 14: design timestamp

- Folded: noted the header/commit time mismatch without changing the design header.
- One-liner: `python3 -I -c 'import subprocess; from pathlib import Path; s=Path("PART115B-DESIGN.md").read_text(); print(next(x for x in s.splitlines() if x.startswith("Registered"))); print(subprocess.run(["git","log","-1","--format=%ci","091c8bb"],check=True,text=True,capture_output=True).stdout.strip())'`
- Output: design header says 2026-10-29 20:07 CT; registration commit time is 2026-10-29 20:09:00 -0500.

## Finding 15: PREDICTIONS.md publication cells

- Folded: noted that Claude fills and commits the result and verdict cells at publication. `PREDICTIONS.md` was not edited.
- One-liner: `python3 -I -c 'from pathlib import Path; s=Path("PREDICTIONS.md").read_text().split("## Part 115b ",1)[1].split("\n## Part 117",1)[0]; print(all("| | |" in x for x in s.splitlines() if x.startswith("| 115b.")))'`
- Output: `True`, the Part 115b result and verdict cells are still blank.
