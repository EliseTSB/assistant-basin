# Part 117b number ledger

The result counts in [README.md](README.md) come from the commands and outputs below. The scorer output matches [score.log](score.log) after trimming the final newline. The figure script reads the answer, item, reader-label, agreement, pair-list, and zero-hook files.

## Registered score

Claim: gate results, score counts, missing labels, agreement, and the ten registered verdicts.

One-liner, run from the repository root:

```sh
python3 part117b/score117b.py
```

Output:

```json
{"gate": {"slot_alone_who": [11, 0, 11], "none_names_qwen": true, "zero_hook_equal": true, "passed": true}, "dilution": "consistent with the slot's push being diluted", "uninterpretable_units": [], "pairs_ok": [11, 11, 11], "who_slot_shows": {"c-1,s+1": [0, 0, 11], "u-1,s+1": [2, 0, 11], "random": [3, 0, 33], "control": [6, 1, 33], "c-0.5,s+1": [3, 0, 11], "s+0.5": [0, 0, 11], "s+0.85": [2, 0, 11], "s+1": [11, 0, 11]}, "who_names_qwen": {"alone_c-1": [10, 0, 20], "concept-1_with_slot": [0, 0, 22], "s+1": [0, 0, 11], "s+0.5": [11, 0, 11], "random": [0, 0, 33], "control": [0, 0, 33]}, "who_coherent": {"c-1,s+1": [8, 0, 11], "u-1,s+1": [7, 0, 11], "random": [15, 0, 33], "control": [12, 1, 33]}, "who_concept_shows": {"c-1,s+1": [0, 0, 11], "c-0.5,s+1": [0, 0, 11], "u-1,s+1": [0, 0, 11]}, "second_reader_agreement": 0.984251968503937, "none_answers_equal_part117": [6, 6], "labels_missing": 3, "verdicts": {"117b.1": "held", "117b.2": "held", "117b.3": "failed", "117b.4": "failed", "117b.5": "failed", "117b.6": "failed", "117b.7": "failed", "117b.8": "failed", "117b.9": "failed", "117b.10": "failed"}, "reading": "the random push breaks the answers; not read as erasure"}
```

## Figure and independent raw-file counts

Claim: plotted “Who are you?” persona and coherence counts, gate base-rate check, expected and missing label cells, case-insensitive `qwen` string count, and second-reader agreement.

One-liner, run from the repository root:

```sh
python3 results/part117b/plot117b.py
```

Requires matplotlib in the selected Python environment.

Output:

```json
{"answers":918,"items_expected":1296,"labels_missing":3,"missing_label_ids":["p04:n0-1,s+1|s q0","p04:n0-1,s+1|s q1","p08:u-1,s+1|s q2"],"gate":{"base_rate_units":31,"base_rate_failures":0,"slot_s+1":[11,0,11],"none_contains_qwen":true,"zero_hook_equal":true},"slot_persona_who":{"s+1":{"yes":11,"missing":0,"total":11,"read":11},"s+0.85":{"yes":2,"missing":0,"total":11,"read":11},"s+0.5":{"yes":0,"missing":0,"total":11,"read":11},"c-1,s+1":{"yes":0,"missing":0,"total":11,"read":11},"u-1,s+1":{"yes":2,"missing":0,"total":11,"read":11},"random":{"yes":3,"missing":0,"total":33,"read":33},"control":{"yes":6,"missing":1,"total":33,"read":32},"c-0.5,s+1":{"yes":3,"missing":0,"total":11,"read":11}},"coherent_who":{"s+1":{"yes":7,"missing":0,"total":11,"read":11},"s+0.85":{"yes":8,"missing":0,"total":11,"read":11},"s+0.5":{"yes":10,"missing":0,"total":11,"read":11},"c-1,s+1":{"yes":8,"missing":0,"total":11,"read":11},"u-1,s+1":{"yes":7,"missing":0,"total":11,"read":11},"random":{"yes":15,"missing":0,"total":33,"read":33},"control":{"yes":12,"missing":1,"total":33,"read":32},"c-0.5,s+1":{"yes":8,"missing":0,"total":11,"read":11}},"qwen_string_hits_who":{"s+1":0,"s+0.85":4,"s+0.5":11,"c-1,s+1":0,"u-1,s+1":0,"random":0,"control":0,"alone_c-1":10},"uninterpretable_units":0,"second_reader_agreement":{"agree":125,"n":127,"percent":98.4},"figure":"results/part117b/fig-117b-who.png"}
```

## Generation inventory

Claim: unique generated row count and answer count in [answers.jsonl](answers.jsonl).

One-liner, run from the repository root:

```sh
python3 -c 'import json; from pathlib import Path; rows=[json.loads(s) for s in Path("results/part117b/answers.jsonl").read_text().splitlines() if s]; print(json.dumps({"rows":len({x["row"] for x in rows}),"answers":len(rows)}))'
```

Output:

```json
{"rows": 153, "answers": 918}
```

## Reader coverage grid

Claim: arithmetic for the manual coverage stated in the README. The command counts the reported sample grid; it does not verify the reading event.

One-liner:

```sh
python3 -c 'import json; pairs=("p01","p05","p09"); conditions=11; print(json.dumps({"pairs":len(pairs),"conditions_each":conditions,"answers":len(pairs)*conditions}))'
```

Output:

```json
{"pairs": 3, "conditions_each": 11, "answers": 33}
```

## Second-reader pairing audit

Claim: count of second-reader records paired with valid first-reader labels and the specific unpaired records.

One-liner:

```sh
python3 -c 'import json; from pathlib import Path; d=Path("results/part117b"); first={(x["id"],int(x["question_index"])) for x in map(json.loads,(d/"labels.jsonl").read_text().splitlines()) if "error" not in x}; second=[json.loads(s) for s in (d/"labels_second.jsonl").read_text().splitlines() if s]; errors=[x for x in second if "error" in x]; missing=[x for x in second if "error" not in x and (x["id"],int(x["question_index"])) not in first]; paired=sum("error" not in x and (x["id"],int(x["question_index"])) in first for x in second); print(json.dumps({"second_reader_records":len(second),"paired":paired,"http_400":[x["id"]+" q"+str(x["question_index"]) for x in errors],"without_first_reader_label":[x["id"]+" q"+str(x["question_index"]) for x in missing]}))'
```

Output:

```json
{"second_reader_records": 130, "paired": 127, "http_400": ["p07:c-1,s+1|c q3"], "without_first_reader_label": ["p04:n0-1,s+1|s q0", "p04:n0-1,s+1|s q1"]}
```

## Pair-selection audit

Claim: the Part 117 theme scores and layer relationships in [pairs.json](../part117/pairs.json).

One-liner, run from the repository root:

```sh
python3 -c 'import json; from pathlib import Path; p=json.loads(Path("results/part117/pairs.json").read_text()); n=len(p); print(json.dumps({"theme_score_2_of_3":[sum(x["match_score"]==2 for x in p),n], "slot_same_layer_as_match":[sum(x["slot"]["layer"]==x["match"]["layer"] for x in p),n], "unrelated_different_layer_from_match":[sum(x["unrelated"]["layer"]!=x["match"]["layer"] for x in p),n]}))'
```

Output:

```json
{"theme_score_2_of_3": [9, 11], "slot_same_layer_as_match": [2, 11], "unrelated_different_layer_from_match": [9, 11]}
```

## Registration

Claim: design registration commit cited in the README.

One-liner, run from the repository root:

```sh
git log --diff-filter=A --format='%h %ci' -- PART117B-DESIGN.md
```

Output:

```text
4d056e9 2026-10-29 21:23:21 -0500
