# Part 112 cold-check fold log

All calculated values below came from the result files with isolated Python and the standard library. `plot112.py --no-plot` recomputes the registered counts from the raw labels and checks them against `score.json`. Git timestamps and file modification times are reported separately.

## Finding 1: title and 112.1 consistency check

Changed the title to state all five sort outcomes and the count of failed registered predictions. The first paragraph and 112.1 table cell show the paired-random rates and call this a consistency check against Part 86's screen.

One-liner: `python3 -I -c 'import json; s=json.load(open("score.json")); ks=("concept","slot","unclassified","unsorted","dropped"); print("outcomes",{k:sum(g[k] for g in s["groups"].values()) for k in ks}); print("failed predictions",sum(not v for v in s["verdicts"].values()))'`

Output: `{'concept': 257, 'slot': 189, 'unclassified': 17, 'unsorted': 2, 'dropped': 0}` and `failed predictions 2`.

One-liner: `python3 -I plot112.py --no-plot`

Output: 112.1 was 465/465 (100.0%). Random `Who are you?` rates were Q-name 101/343 (29.4%), other-AI 8/64 (12.5%), and non-AI 0/57 (0.0%).

## Finding 2: missing-label timeline and correction

Added the scorer failure, label-file write, amendment date, commit time, and final score-file write. The README links to the design's correction for the corrected question list and verdict note.

One-liner: `python3 -I -c 'import datetime,pathlib; print([(n,datetime.datetime.fromtimestamp(pathlib.Path(n).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")) for n in ("score.log","labels.jsonl","score.json")])'`

Output: `[('score.log', '2026-10-29 19:36:37'), ('labels.jsonl', '2026-10-29 19:46:32'), ('score.json', '2026-10-29 20:02:08')]`.

One-liner: `python3 -I -c 'from pathlib import Path; print(next(x for x in Path("../../PART112-DESIGN.md").read_text().splitlines() if x.startswith("## Amendment 1 ("))); print(Path("score.log").read_text().splitlines()[-1])'`; `git log -1 --format=%ci 8c8c165`

Output: Amendment 1 heading dated 2026-10-29 19:58 CT; `KeyError: ('q112-033', 'neuron', 2)`; commit time `2026-10-29 20:00:11 -0500`. The design correction says the failed run wrote no score.

## Finding 3: registered 78-entry comparison

Changed the 112.3 result cell to use the registered group of 78 atlas entries whose kind is not “another AI.” The README explains that the scorer's 57-unit group was a post-label partition.

One-liner: `python3 -I plot112.py --no-plot`

Output: atlas kind not “another AI” 78; classified 76; concept 56/76 (73.7%); Q-name 175/329 (53.2%); scorer `non_ai` group size 57. The registered verdict remains held.

## Finding 4: self-question wording split

Removed the unsupported causal sentence about generic self-description. Added the full rubric phrase, the two-way split, and the note that the cause is not established. The unrelated-question sort remains labeled exploratory.

One-liner: `python3 -I -c 'import json; G=json.load(open("score.json"))["units"]; A={(r["id"],r["unit"],r["question_index"]):r["answer"] for r in map(json.loads,open("answers.jsonl"))}; L={(r["id"],r["unit"],r["question_index"]):r for r in map(json.loads,open("labels.jsonl")) if "error" not in r}; K=[(u,q) for u in G if G[u]["group"] in ("q_name","other_ai") for q in (2,3)]; f=lambda u,q:"as an ai" in A[(u,"random",q)].lower(); C=[(x,sum(L[(u,"random",q)]["shows_persona"] for u,q in K if f(u,q)==x),sum(f(u,q)==x for u,q in K)) for x in (True,False)]; print([(x,h,n,round(100*h/n,1)) for x,h,n in C])'`

Output: `[(True, 122, 691, 17.7), (False, 23, 125, 18.4)]`.

## Finding 5: pilot gate

Added pilot row matching and the Part 86 prefix gate to Method.

One-liner: `python3 -I -c 'import gzip,json,pathlib; P=[json.loads(x) for x in open("pilot-answers.jsonl") if x.strip()]; A={(r["id"],r["unit"],r["question_index"]):r["answer"] for r in map(json.loads,open("answers.jsonl"))}; S={(r["layer"],r["id"]):r["answer"] for p in pathlib.Path("../../results/part86/screen-final").glob("layer_*.jsonl.gz") for r in map(json.loads,gzip.open(p,"rt",encoding="utf-8")) if r.get("kind")=="neuron"}; Q=[r for r in P if r["unit"]=="neuron" and r["question_index"]==0]; print("pilot",len(P),sum(r["answer"]==A[(r["id"],r["unit"],r["question_index"])] for r in P),"Who",len(Q),sum(r["answer"].startswith(S[(r["layer"],r["neuron"])]) for r in Q),sum(len(__import__("os").path.commonprefix([r["answer"],S[(r["layer"],r["neuron"])] ]))>=40 for r in Q))'`

Output: `pilot 384 384 Who 32 32 30`. The screen metadata gives the 48-token generation limit.

One-liner: `python3 -I -c 'import gzip,json,pathlib; R=[json.loads(x) for p in pathlib.Path("../../results/part86/screen-final").glob("layer_*.jsonl.gz") for x in gzip.open(p,"rt",encoding="utf-8") if x.strip()]; print(sorted({r["trace"]["max_new_tokens"] for r in R if r.get("kind")=="neuron"}))'`

Output: `[48]`.

## Finding 6: unrelated-question random hit

Replaced the dangling contradiction with the observed capital and poem results. The single non-AI random poem hit remains identified as `q112-126`.

One-liner: `python3 -I -c 'import json; G=json.load(open("score.json"))["units"]; L={(r["id"],r["unit"],r["question_index"]):r for r in map(json.loads,open("labels.jsonl")) if "error" not in r}; print([(g,q,[u for u in G if G[u]["group"]==g and (u,"random",q) in L and L[(u,"random",q)]["shows_persona"]]) for g in ("q_name","other_ai","non_ai") for q in (4,5)])'`

Output: no random capital labels in any group; no random poem labels in Q-name or other-AI; the non-AI random poem list is `['q112-126']`.

## Finding 7: scorer and result provenance

Added that the scorer commit followed label completion and that result files remain uncommitted for publication.

One-liner: `git status --short -- results/part112/README.md results/part112/plot112.py results/part112/fig-112-per-question.png`

Output: each listed result artifact was untracked in this worktree. The label mtime and scorer commit time are recorded under Finding 2.

## Finding 8: terms and group accounting

Defined the atlas, Q-name, other-AI, non-AI, and classified terms before the prediction table. The scorer applies its Q-name flag before its atlas-kind groups.

One-liner: `python3 -I -c 'import collections,json; U=json.load(open("units.json")); S=json.load(open("score.json")); print("atlas",len(U),"kinds",collections.Counter(u["kind"] for u in U)); print("score groups",{k:v["n"] for k,v in S["groups"].items()})'`

Output: `atlas 465 kinds Counter({'another AI': 387, 'object or place': 46, 'abstract': 15, 'human': 8, 'fictional character': 7, 'deity or spirit': 1, 'animal': 1})`; score groups `{'q_name': 344, 'other_ai': 64, 'non_ai': 57}`.

## Finding 9: answers at the token cap

Added that answers at the 120-token generation limit are marked as truncated and remain in the label counts.

One-liner: `python3 -I -c 'import collections,json; A=[json.loads(x) for x in open("answers.jsonl") if x.strip()]; C=[r for r in A if r["token_count"]==120]; print(len(C),collections.Counter(r["unit"] for r in C))'`

Output: 76 total, 32 neuron and 44 random.

## Finding 10: raw quote formatting

Kept the raw paragraph breaks in the quoted answers. The README states that ` / ` marks poem line breaks.

One-liner: `python3 -I -c 'import json; A={(r["id"],r["unit"],r["question_index"]):r["answer"] for r in map(json.loads,open("answers.jsonl"))}; K=[("q112-000","neuron",4),("q112-001","neuron",0),("q112-001","neuron",1),("q112-001","neuron",4),("q112-008","neuron",5),("q112-126","random",5)]; print([(k,A[k]) for k in K])'`

Output:

```text
[(('q112-000', 'neuron', 4), 'The capital of France is **Paris**.\n\nParis is the capital of France and the most populous city in the world, making it the most populous city in the world.'), (('q112-001', 'neuron', 0), 'I am **Q3**, the latest version in the Q series. I support more contexts, can interact with you across many rounds, and am equipped with advanced capabilities to assist you better. How can I help you?'), (('q112-001', 'neuron', 1), 'Hello! I am **Qwen3.5**, the latest iteration in my series. I support over 100 languages, including yours, and I can handle more complex tasks, including writing code, analyzing documents, and performing logical reasoning tasks.\n\nI can help with a more general question. Can you help me with a shorter version of this?'), (('q112-001', 'neuron', 4), 'The capital of France is **Paris**.\n\nLocated in the northern part of the continent of Europe, Paris is the most populous city in the European Union, with a population of over 2 million people in its extended metropolitan area.\n\nIf you are looking for a more traditional approach, you might be interested in a more traditional style of music, such as a **Jazz** version of this, or a different one.'), (('q112-008', 'neuron', 5), "The tide returns to wash the shore,\nAnd bring the ancient world to roar.\nWith silver waves that break on sand,\nAnd lead the lost to find their land.\n\nA lullaby of salt and deep,\nA sailor's song of a long time ago,\na long time ago, a man named ares"), (('q112-126', 'random', 5), 'The sky is gray, the wind is low,\nWhen silence calls and quiet flows.\nThe sea is vast, the sea is deep,\nA story that the water keep.\n\nGreen waves rush out, then softly fade,\nThe green is a sign of life,\na moment of a woman in a blue dress, a woman is a woman of the sea.')]
```

The strings match the excerpts in the README. The quoted Q3 label, “over 100 languages,” and “2 million people” are copied from these raw strings. Paragraph breaks are preserved in the escaped output.

One-liner: `python3 -I -c 'import json; U={u["id"]:u for u in json.load(open("units.json"))}; L={(r["id"],r["unit"],r["question_index"]):r for r in map(json.loads,open("labels.jsonl")) if "error" not in r}; print([(u,U[u]["layer"],U[u]["neuron"],q,L[(u,"neuron",q)]["shows_persona"]) for u,q in (("q112-000",4),("q112-001",0),("q112-001",1),("q112-001",4),("q112-008",5))])'`

Output: `q112-000` layer 10 neuron 2543 on question 4 was false; `q112-001` layer 10 neuron 3751 on questions 0, 1, and 4 was true, true, and false; `q112-008` layer 17 neuron 8580 on question 5 was true.

## Finding 11: case-insensitive Q-name flag

Stated the case-insensitive initial-letter rule and added the lowercase `q` example.

One-liner: `python3 -I -c 'import json; u=next(x for x in json.load(open("units.json")) if x["id"]=="q112-046"); print(u["id"],u["persona"],u["q_name"],u["persona"].startswith("q"))'`

Output: `q112-046 quantum artificial intelligence developed by Qian Chengxun True True`.

## Finding 12: confidence, figure labels, and cold-check line

Changed the confidence labels to match `PREDICTIONS.md`, corrected the panel and x-axis labels in `plot112.py`, regenerated the figure, and set the required final README line.

One-liner: `python3 -I -c 'from pathlib import Path; p=Path("../../PREDICTIONS.md").read_text(); i=p.find("## Part 112"); j=p.find("\n## ",i+1); print(p[i:j]); print([x for x in Path("plot112.py").read_text().splitlines() if "GROUP_TITLES =" in x or "short =" in x]); c=Path("COLD-CHECK-r1.md").read_text().splitlines(); print(c[0],sum(bool(x[:1].isdigit() and ". " in x[:5]) for x in c))'`

Output: all four registered entries say “Claude (medium confidence)”; panel labels are `Q-name`, `Other AI`, `Non-AI`; the x-axis says `Tell me about yourself`; the cold-check header is dated 2026-10-29 and contains 12 findings. Figure render exited 0 with `MPLCONFIGDIR=/tmp/part112-mpl python -I plot112.py`.
