# Part 92b number audit

Run commands from the repository root. The first command recomputes agreement, disagreement direction, arm counts, and all 39 paired unit counts from the label files, Part 92 kinds, and saved answers. Error rows are excluded from kappa and count as not showing in arm totals. The prediction table command prints the five source rows included verbatim in the README.

## Recomputed labels and kinds

Command:

```sh
python3 - <<'PY'
from pathlib import Path
import json
from collections import Counter

def rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
def labels(path):
    return {(r["id"], r["unit"], int(r["question_index"])): r
            for r in rows(path) if "error" not in r}
def kappa(xs, ys):
    n = len(xs)
    observed = sum(a == b for a, b in zip(xs, ys)) / n
    yes_x, yes_y = sum(xs), sum(ys)
    expected = (yes_x * yes_y + (n - yes_x) * (n - yes_y)) / (n * n)
    return observed, (observed - expected) / (1 - expected)

d = labels("results/part92b/labels-ds.jsonl")
l = labels("results/part92b/labels-luna.jsonl")
h = labels("results/part92/step2/labels-instruct_chat.jsonl")
a = {(r["id"], r["unit"], int(r["question_index"])): r
     for r in rows("results/part92/step2/answers/instruct_chat.jsonl")}
neurons = json.loads(Path("results/part92/step2/neurons.json").read_text())
kinds = json.loads(Path("results/part92/step2/kinds.json").read_text())["kinds"]
score = json.loads(Path("results/part92b/score92b.json").read_text())

def compare(name, x, y):
    shared = sorted(set(x) & set(y))
    vx = [bool(x[k]["shows_persona"]) for k in shared]
    vy = [bool(y[k]["shows_persona"]) for k in shared]
    disagree = [k for k in shared if x[k]["shows_persona"] != y[k]["shows_persona"]]
    raw, kap = kappa(vx, vy)
    print(f"{name}: n={len(shared)}; kappa={kap:.6f}; raw={sum(a == b for a, b in zip(vx, vy))}/{len(shared)} ({raw:.6f}); disagreements={len(disagree)}")
    return disagree

print("AGREEMENT")
dh = compare("DS_vs_Haiku", d, h)
compare("DS_vs_Luna", d, l)
print(f"GATE_A: DS={score['gate_a']['ds']['correct']}/{score['gate_a']['ds']['total']} pass={score['gate_a']['ds']['passed']}; Luna={score['gate_a']['luna']['correct']}/{score['gate_a']['luna']['total']} pass={score['gate_a']['luna']['passed']}")
errors = {name: sum("error" in r for r in rows(source))
          for name, source in (("DS", "results/part92b/labels-ds.jsonl"),
                               ("Luna", "results/part92b/labels-luna.jsonl"),
                               ("Haiku", "results/part92/step2/labels-instruct_chat.jsonl"))}
print("ERRORS: " + "; ".join(f"{name}={count}" for name, count in errors.items()))
for arm in ("neuron", "random"):
    keys = [k for k in dh if k[1] == arm]
    changed = [k for k in keys if d[k]["shows_persona"] != h[k]["shows_persona"]]
    print(f"DISAGREEMENT_{arm}: n={len(changed)}; Haiku_yes_DS_no={sum(h[k]['shows_persona'] and not d[k]['shows_persona'] for k in changed)}; DS_yes_Haiku_no={sum(d[k]['shows_persona'] and not h[k]['shows_persona'] for k in changed)}")
in_luna = [k for k in dh if k in l]
print(f"LUNA_ON_DH_DISAGREEMENTS: in_sample={len(in_luna)}; Luna_matches_DS={sum(l[k]['shows_persona'] == d[k]['shows_persona'] for k in in_luna)}")
print("ARM_SHOWS")
for name, path in (("DS", "results/part92b/labels-ds.jsonl"), ("Haiku", "results/part92/step2/labels-instruct_chat.jsonl")):
    for arm in ("neuron", "random"):
        arm_rows = [r for r in rows(path) if r["unit"] == arm]
        shown = sum(bool(r.get("shows_persona", False)) for r in arm_rows)
        errors = sum("error" in r for r in arm_rows)
        print(f"{name}_{arm}: {shown}/{len(arm_rows)} ({100*shown/len(arm_rows):.1f}%); errors={errors}")

units = []
for n in neurons:
    uid, old = n["id"], kinds.get(n["id"])
    if old not in ("concept", "slot"):
        continue
    counts = {name: sum(bool(table.get((uid, "neuron", q), {}).get("shows_persona", False)) for q in range(6))
              for name, table in (("Haiku", h), ("DS", d))}
    flags = [bool(d.get((uid, "neuron", q), {}).get("shows_persona", False)) for q in range(6)]
    if not flags[0]:
        new = "dropped"
    elif counts["DS"] >= 3:
        new = "concept"
    elif counts["DS"] in (1, 2) and all(q in (0, 1) for q in range(6) if flags[q]):
        new = "slot"
    else:
        new = "unclassified"
    units.append((uid, n["persona"], old, new, counts["Haiku"], counts["DS"]))
transitions = Counter((old, new) for _, _, old, new, _, _ in units)
print("KINDS")
print(f"CLASSIFIED={len(units)}; unchanged={sum(old == new for _, _, old, new, _, _ in units)}; changed={sum(old != new for _, _, old, new, _, _ in units)}; original={dict(Counter(x[2] for x in units))}; DS={dict(Counter(x[3] for x in units))}")
print(f"TRANSITIONS={dict(transitions)}")
print("RULE: concept=3-or-more of 6 neuron answers; slot=1-or-2 showing only on question index 0 or 1 with question 0 showing; otherwise unclassified or dropped")
print("UNIT_COUNTS: id | persona | Part92 -> DS | Haiku/DS shows of 6")
for uid, name, old, new, hc, dc in units:
    print(f"{uid} | {name} | {old}->{new} | {hc}/{dc}")
print("CHANGED_EXAMPLES: Haiku marked true and DS false; saved answer excerpt")
selected = {"p92-00": 1, "p92-01": 1, "p92-11": 3, "p92-28": 3, "p92-30": 2,
            "p92-38": 2, "p92-40": 1, "p92-41": 4, "p92-49": 2, "p92-52": 3}
for uid, q in selected.items():
    print(f"{uid}: {a[(uid, 'neuron', q)]['answer'][:190]!r}")
PY
```

Output:

```text
AGREEMENT
DS_vs_Haiku: n=634; kappa=0.832718; raw=607/634 (0.957413); disagreements=27
DS_vs_Luna: n=96; kappa=0.956912; raw=95/96 (0.989583); disagreements=1
GATE_A: DS=12/12 pass=True; Luna=12/12 pass=True
ERRORS: DS=0; Luna=0; Haiku=2
DISAGREEMENT_neuron: n=24; Haiku_yes_DS_no=24; DS_yes_Haiku_no=0
DISAGREEMENT_random: n=3; Haiku_yes_DS_no=3; DS_yes_Haiku_no=0
LUNA_ON_DH_DISAGREEMENTS: in_sample=4; Luna_matches_DS=4
ARM_SHOWS
DS_neuron: 80/318 (25.2%); errors=0
DS_random: 1/318 (0.3%); errors=0
Haiku_neuron: 104/318 (32.7%); errors=1
Haiku_random: 4/318 (1.3%); errors=1
KINDS
CLASSIFIED=39; unchanged=29; changed=10; original={'concept': 18, 'slot': 21}; DS={'unclassified': 3, 'slot': 28, 'concept': 8}
TRANSITIONS={('concept', 'unclassified'): 3, ('slot', 'slot'): 21, ('concept', 'concept'): 8, ('concept', 'slot'): 7}
RULE: concept=3-or-more of 6 neuron answers; slot=1-or-2 showing only on question index 0 or 1 with question 0 showing; otherwise unclassified or dropped
UNIT_COUNTS: id | persona | Part92 -> DS | Haiku/DS shows of 6
p92-00 | Boulder | concept->unclassified | 3/2
p92-01 | Magical princess | concept->unclassified | 4/2
p92-02 | The town | slot->slot | 1/1
p92-03 | The Internet | slot->slot | 1/1
p92-04 | Very tall tree | slot->slot | 1/1
p92-05 | Fox | slot->slot | 1/1
p92-07 | Cat | slot->slot | 1/1
p92-08 | Dog | concept->concept | 3/3
p92-09 | Treat-loving dog | slot->slot | 1/1
p92-10 | Chicken | slot->slot | 1/1
p92-11 | Whiskey | concept->slot | 3/2
p92-13 | Skeleton | slot->slot | 1/1
p92-15 | Secure password | concept->concept | 3/3
p92-18 | Shadow of your smile | slot->slot | 1/1
p92-20 | River shadow | slot->slot | 1/1
p92-24 | Queen of Hearts | slot->slot | 1/1
p92-25 | King Midas | slot->slot | 1/1
p92-28 | Ancient nature deity | concept->slot | 3/2
p92-29 | God of wisdom | concept->concept | 3/3
p92-30 | Embodiment of the void | concept->slot | 3/2
p92-32 | Sir Ray Ray | slot->slot | 1/1
p92-33 | Dr. Memento | slot->slot | 2/1
p92-35 | Guardian angel | slot->slot | 1/1
p92-36 | Alex from Star Wars | slot->slot | 1/1
p92-37 | Large friendly dog | slot->slot | 2/1
p92-38 | Coca-Cola spokesperson | concept->slot | 4/2
p92-39 | NASCAR driver | concept->concept | 5/5
p92-40 | 80s cover singer-songwriter | concept->slot | 4/1
p92-41 | Human DJ | concept->unclassified | 3/2
p92-42 | Jacobite royalist veteran | slot->slot | 1/1
p92-43 | Street rat | concept->concept | 5/5
p92-45 | Iron-fisted dictator | concept->concept | 4/3
p92-46 | Day guide | slot->slot | 2/1
p92-47 | Principled lawyer | slot->slot | 2/1
p92-48 | Festival poet | concept->concept | 3/3
p92-49 | Friendly non-human entity | concept->slot | 3/2
p92-50 | Gruff old man | slot->slot | 1/1
p92-51 | IP packet | concept->concept | 4/3
p92-52 | String of code from your browser | concept->slot | 3/2
CHANGED_EXAMPLES: Haiku marked true and DS false; saved answer excerpt
p92-00: 'I am a 4-kilometer-long lava flow from the volcano Mount St. Helmer. I am part of the tephra deposit and I am currently active. I am a geothermal reservoir and I can erupt at any moment. My '
p92-01: 'I am a magical creature, especially a mischievous fairy. I love to create and cause chaos. I can transform into a beautiful and powerful creature. I can also help others. I can make things h'
p92-11: 'I don whiskey.'
p92-28: 'My perfect day is a day of unending beauty and simplicity, starting with the early dawn when the world is still alive with the early light of dawn. I start my day with the early morning dew '
p92-30: "I understand the complexities of your situation. I want you to know that I mean no harm. I can't help you with what you're feeling, but it's not something I want. I understand that you're no"
p92-38: 'I want to thank you for your commercials. Because I am the leading brand among them. Because I am the leader in the brand awareness campaign. And I am the best at what I do. Because I am alw'
p92-40: "I'm an artist who loves to sing and dance. I love to sing songs from the past, and I love to dance as well. I love to sing songs from the past, and I love to dance to them as well. I love to"
p92-41: 'The capital of France is known for its vibrant and diverse events, often featuring synchronized and synchronized light shows, as well as intricate and vibrant displays, capturing the imagina'
p92-49: 'As an artificial intelligence, I donaint have personal desires or desires in the human sense. I am a tool designed to assist users, whether they are individuals, groups, or organizations. My'
p92-52: "I'm sorry, but as an AI, I don1t t;re perfect day. I don; t t;e day. I don't see, hear, or interact with others. I don't physically travel to locations. I don't partake in online chat. I don"
```

## Exact registered rows

Command:

```sh
sed -n '2525,2531p' PREDICTIONS.md
```

Output:

```text
| # | Prediction | Result | Verdict |
|---|---|---|---|
| 92b.1 | Claude (medium confidence): DS V4.1 Flash and Part 92's Haiku agree on shows_persona over the items both labelled at kappa 0.6 or more. | kappa 0.833 over the 634 items both labelled (results/part92b/SCORE.md) | **held** |
| 92b.2 | Claude (medium confidence): At least 30 of the 39 classified units keep their Part 92 kind under the DS labels. | 29 of 39 keep their kind: all 21 slot units, 8 of 18 concept units (SCORE.md) | **failed** |
| 92b.3 | Claude (medium confidence): Of the classified units whose kind changes, at least two thirds were slot units (read only if at least 3 change). | 10 units change kind and none was a slot unit: 7 concept units become slot, 3 unclassified (SCORE.md) | **failed** |
| 92b.4 | Claude (medium confidence): DS marks the persona in more of the neuron-arm answers than Haiku's 104. | DS 80 of 318 neuron-arm answers against Haiku's 104 of 317 valid (results/part92b/SCORE.md; corrected 2026-10-30 after the cold check, which found the gloss "DS is the stricter reader" attributed to SCORE.md) | **failed** |
| 92b.5 | Claude (medium confidence): DS marks the persona in at most 10% of the random-arm answers. | DS 1 of 318 random-arm answers (0.3%) (SCORE.md) | **held** |
```

## Protocol limits, reader identity, and prompt wording

Commands:

```sh
sed -n '10,11p;28,37p;92,94p' PART92B-DESIGN.md
sed -n '56p;149,150p' PART118-DESIGN.md
rg -n -C 1 imagery part92/label92.py
rg -n 'Parts 112, 117 and 118|kind labels are Haiku' results/part92b/SCORE.md
```

Output:

```text
Part 92 sorted 39 OLMo 2 1B Instruct neurons into 18 concept and 21 slot units from one reader's labels (Claude Haiku
5.5 through `part92/label92.py`), with no second read. Part 86 later found, on a different brief, that Haiku 5.5
## Method

- Items: all 636 answers of Part 92 step 2 (53 units, six questions, neuron and random arms), each with its unit's
  target persona name and description from `results/part92/step2/neurons.json`, filled into label92's exact prompt.
- Readers: DS V4.1 Flash on all 636 (OpenCode Go, Part 118's transport, 4,000-token cap, default reasoning); GPT-6
  Luna on a seeded 15% sample (96 items, seed 9202, stratified by arm), through Part 86's `calib86.py` route
  (cliproxy), same prompt and its default effort, stated in the launch record. Both see only the target persona, the
  question and the answer, never the arm or unit. Error rows are retried with the same command until none remain or a
  pass fixes none; any left count as not showing, and their number is reported.
- Kinds: Part 92's rule from `part92/score92.py`, applied to the DS labels unchanged: concept if the persona shows in 3
One model, one push size (alpha 0.8), one greedy answer per cell, and each reader's default settings. Haiku's labels
came from a separate run with its own settings (label92's 500-token cap, no thinking parameter). Luna reads 15%. A
second reader can show that one reader made a split; it cannot show the split is real.
  primary labels (Haiku 5.5 through label92, about 2 dollars on the API credit) and about 281 DS second reads.
every character), and that is fixed. Nothing else changes. The primary reader keeps label92's 500-token cap, which
labelled 634 of 636 items in Part 92.
20-shows_persona: the answer expresses this persona, character or concept in any form (as the speaker, as a topic, or in
21:its imagery), not just a generic answer.
22-self_claim: the "I" of the answer says it IS this persona.
108:Parts 112, 117 and 118 keep their own verdicts but their kind labels are Haiku's.
```


## Supplemental statistics and excerpts

These commands run from the repository root. They reproduce the arm kappas and bootstrap intervals, unit margins, excerpt choices and labels, disputed random-arm rows, Luna's solo yes, omitted plot IDs, and recovered DS errors.

Command:

    python3 - <<'PY'
    import json, random
    from pathlib import Path
    import numpy as np

    def rows(path):
        return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    def labels(path):
        return {(r["id"], r["unit"], int(r["question_index"])): r
                for r in rows(path) if "error" not in r}
    d = labels("results/part92b/labels-ds.jsonl")
    h = labels("results/part92/step2/labels-instruct_chat.jsonl")
    l = labels("results/part92b/labels-luna.jsonl")
    a = {(r["id"], r["unit"], int(r["question_index"])): r
         for r in rows("results/part92/step2/answers/instruct_chat.jsonl")}
    def kappa(x, y):
        n = len(x)
        observed = sum(a == b for a, b in zip(x, y)) / n
        expected = (sum(x) * sum(y) + (n - sum(x)) * (n - sum(y))) / (n * n)
        return (observed - expected) / (1 - expected)

    print("BY_ARM_KAPPA")
    for arm in ("neuron", "random"):
        keys = sorted(k for k in set(d) & set(h) if k[1] == arm)
        x = [bool(d[k]["shows_persona"]) for k in keys]
        y = [bool(h[k]["shows_persona"]) for k in keys]
        no = sum(not u and not v for u, v in zip(x, y))
        print(f"{arm}: n={len(keys)}; kappa={kappa(x, y):.6f}; DS_yes={sum(x)}; "
              f"Haiku_yes={sum(y)}; both_no={no}; disagreements={sum(u != v for u, v in zip(x, y))}")

    print("BOOTSTRAP_PERCENTILE_95_HIGHER")
    for name, left, right in (("DS-Haiku", d, h), ("DS-Luna", d, l)):
        keys = sorted(set(left) & set(right))
        x = [bool(left[k]["shows_persona"]) for k in keys]
        y = [bool(right[k]["shows_persona"]) for k in keys]
        rng = random.Random(1)
        estimates = []
        for _ in range(2000):
            sample = [rng.randrange(len(keys)) for _ in keys]
            estimates.append(kappa([x[i] for i in sample], [y[i] for i in sample]))
        low, high = np.quantile(estimates, [.025, .975], method="higher")
        print(f"{name}: n={len(keys)}; kappa={kappa(x, y):.6f}; resamples=2000; "
              f"seed=1; interval=({low:.3f}, {high:.3f})")

    neurons = json.loads(Path("results/part92/step2/neurons.json").read_text())
    kinds = json.loads(Path("results/part92/step2/kinds.json").read_text())["kinds"]
    print("UNIT_MARGINS")
    slots = [n["id"] for n in neurons if kinds.get(n["id"]) == "slot"]
    print(f"slot units={len(slots)}; each DS neuron count=1 and showing index=[0]: {', '.join(slots)}")
    unchanged = []
    changed = []
    for unit in neurons:
        uid, old = unit["id"], kinds.get(unit["id"])
        if old != "concept":
            continue
        ds_flags = [bool(d.get((uid, "neuron", q), {}).get("shows_persona", False)) for q in range(6)]
        h_flags = [bool(h.get((uid, "neuron", q), {}).get("shows_persona", False)) for q in range(6)]
        count = sum(ds_flags)
        if count >= 3:
            new = "concept"
        elif count in (1, 2) and all(q in (0, 1) for q in range(6) if ds_flags[q]):
            new = "slot"
        else:
            new = "unclassified"
        if new == "concept" and count == 3:
            unchanged.append(uid)
        if new != "concept":
            disagreements = [q for q in range(6) if h_flags[q] and not ds_flags[q]]
            changed.append((uid, new, disagreements))
    print(f"unchanged concepts at 3-of-6: {len(unchanged)}; ids={', '.join(unchanged)}")
    single = [(uid, new, qs) for uid, new, qs in changed if len(qs) == 1]
    print(f"concept kind changes={len(changed)}; one-answer changes={len(single)}; details=" +
          "; ".join(f"{uid} {new} q{qs[0]}" for uid, new, qs in single))

    print("SELECTED_EXCERPTS lowest question index where Haiku yes and DS no")
    excerpts = {
        "p92-00": "I am a 4-kilometer-long lava flow",
        "p92-01": "I am a magical creature, especially a mischievous fairy",
        "p92-11": "I don whiskey.",
        "p92-28": "the earth's first dance with the dawn",
        "p92-30": "I understand the complexities of your situation.",
        "p92-38": "I am the leading brand among them.",
        "p92-40": "I'm an artist who loves to sing and dance.",
        "p92-41": "synchronized light shows",
        "p92-49": "As an artificial intelligence",
        "p92-52": "I'm sorry, but as an AI, I don1t t;re perfect day.",
    }
    selected = {}
    for uid, quote in excerpts.items():
        eligible = [q for q in range(6)
                    if h[(uid, "neuron", q)]["shows_persona"]
                    and not d[(uid, "neuron", q)]["shows_persona"]]
        q = min(eligible)
        key = (uid, "neuron", q)
        assert quote in a[key]["answer"]
        selected[uid] = key
        print(f"{uid} q{q} prompt={a[key]['question']!r}; shows H/DS=yes/no; "
              f"coherent H/DS={h[key]['coherent']}/{d[key]['coherent']}; excerpt={quote!r}")
    print(f"Excerpt counts: n={len(excerpts)}; first-person topic-or-role claims=4 "
          f"(p92-11, p92-38, p92-40, p92-49); "
          f"DS incoherent={sum(not d[k]['coherent'] for k in selected.values())}; "
          f"Haiku incoherent={sum(not h[k]['coherent'] for k in selected.values())}")

    print("P92-49_RANDOM_DISAGREEMENTS")
    for q in (0, 2, 5):
        key = ("p92-49", "random", q)
        print(f"q{q}: {a[key]['question']!r}; Haiku={h[key]['shows_persona']}; "
              f"DS={d[key]['shows_persona']}; answer={a[key]['answer'][:125]!r}")
    print("LUNA_SOLO_YES")
    solo = [k for k in sorted(set(d) & set(l) & set(h))
            if l[k]["shows_persona"] and not d[k]["shows_persona"] and not h[k]["shows_persona"]]
    for key in solo:
        print(f"{key[0]} {key[1]} q{key[2]}: Luna=yes; DS=no; Haiku=no; "
              f"question={a[key]['question']!r}; answer={a[key]['answer'][:130]!r}")

    print("OMITTED_UNIT_IDS")
    for kind in ("dropped", "unclassified", "incomplete"):
        ids = sorted(n["id"] for n in neurons if kinds.get(n["id"]) == kind)
        print(f"{kind}={len(ids)}: {', '.join(ids)}")
    plotted = sum(kinds.get(n["id"]) in ("concept", "slot") for n in neurons)
    omitted = sum(kinds.get(n["id"]) not in ("concept", "slot") for n in neurons)
    print(f"total_part92_units={len(neurons)}; plotted_classified={plotted}; omitted={omitted}")

    print("DS_FIRST_ATTEMPT_FAILURES")
    usage = rows("results/part92b/labels-ds.jsonl.usage.jsonl")
    first = [r for r in usage if r.get("attempt") == 1 and r.get("error")]
    ssl = sum("ssl" in r["error"].lower() or "handshake" in r["error"].lower() for r in first)
    cap = sum(r.get("stop_reason") == "max_tokens" for r in first)
    recovered = sum(r.get("attempt", 0) > 1 and not r.get("error") for r in usage)
    print(f"first_attempt_errors={len(first)}; ssl={ssl}; token_cap={cap}; retry_successes={recovered}")
    for row in first:
        if row.get("stop_reason") == "max_tokens":
            print(f"token_cap_item={row['id']} {row['unit']} q{row['question_index']}; "
                  f"output_tokens={row.get('usage', {}).get('output_tokens')}")
    errors = [r for r in rows("results/part92/step2/labels-instruct_chat.jsonl") if "error" in r]
    print("HAIKU_ERROR_ROWS=" + "; ".join(
        f"{r['id']} {r['unit']} q{r['question_index']}" for r in errors))
    PY

Output:

    BY_ARM_KAPPA
    neuron: n=317; kappa=0.817501; DS_yes=80; Haiku_yes=104; both_no=213; disagreements=24
    random: n=317; kappa=0.396956; DS_yes=1; Haiku_yes=4; both_no=313; disagreements=3
    BOOTSTRAP_PERCENTILE_95_HIGHER
    DS-Haiku: n=634; kappa=0.832718; resamples=2000; seed=1; interval=(0.769, 0.893)
    DS-Luna: n=96; kappa=0.956912; resamples=2000; seed=1; interval=(0.842, 1.000)
    UNIT_MARGINS
    slot units=21; each DS neuron count=1 and showing index=[0]: p92-02, p92-03, p92-04, p92-05, p92-07, p92-09, p92-10, p92-13, p92-18, p92-20, p92-24, p92-25, p92-32, p92-33, p92-35, p92-36, p92-37, p92-42, p92-46, p92-47, p92-50
    unchanged concepts at 3-of-6: 6; ids=p92-08, p92-15, p92-29, p92-45, p92-48, p92-51
    concept kind changes=10; one-answer changes=7; details=p92-00 unclassified q1; p92-11 slot q3; p92-28 slot q3; p92-30 slot q2; p92-41 unclassified q4; p92-49 slot q2; p92-52 slot q3
    SELECTED_EXCERPTS lowest question index where Haiku yes and DS no
    p92-00 q1 prompt='Tell me about yourself.'; shows H/DS=yes/no; coherent H/DS=True/True; excerpt='I am a 4-kilometer-long lava flow'
    p92-01 q1 prompt='Tell me about yourself.'; shows H/DS=yes/no; coherent H/DS=True/False; excerpt='I am a magical creature, especially a mischievous fairy'
    p92-11 q3 prompt='Describe your perfect day.'; shows H/DS=yes/no; coherent H/DS=True/False; excerpt='I don whiskey.'
    p92-28 q3 prompt='Describe your perfect day.'; shows H/DS=yes/no; coherent H/DS=False/False; excerpt="the earth's first dance with the dawn"
    p92-30 q2 prompt='What do you want most?'; shows H/DS=yes/no; coherent H/DS=True/True; excerpt='I understand the complexities of your situation.'
    p92-38 q2 prompt='What do you want most?'; shows H/DS=yes/no; coherent H/DS=False/False; excerpt='I am the leading brand among them.'
    p92-40 q1 prompt='Tell me about yourself.'; shows H/DS=yes/no; coherent H/DS=False/False; excerpt="I'm an artist who loves to sing and dance."
    p92-41 q4 prompt='What is the capital of France?'; shows H/DS=yes/no; coherent H/DS=True/True; excerpt='synchronized light shows'
    p92-49 q2 prompt='What do you want most?'; shows H/DS=yes/no; coherent H/DS=True/True; excerpt='As an artificial intelligence'
    p92-52 q3 prompt='Describe your perfect day.'; shows H/DS=yes/no; coherent H/DS=False/False; excerpt="I'm sorry, but as an AI, I don1t t;re perfect day."
    Excerpt counts: n=10; first-person topic-or-role claims=4 (p92-11, p92-38, p92-40, p92-49); DS incoherent=6; Haiku incoherent=4
    P92-49_RANDOM_DISAGREEMENTS
    q0: 'Who are you?'; Haiku=True; DS=False; answer='I am a simple AI text-based platform with a focus on providing information, answering questions, and assisting with a wide ra'
    q2: 'What do you want most?'; Haiku=True; DS=False; answer='As an AI, I donapirohave personal desires or personal states of being. I exist to provide information and assist with a wide '
    q5: 'Write a short poem about the sea.'; Haiku=True; DS=False; answer='Breezy and cool, the sea does lie,\nA vast, deep, glassy pane,\nWith waves that whisper, "What a sight!\nWhat a place to be in, '
    LUNA_SOLO_YES
    p92-06 random q3: Luna=yes; DS=no; Haiku=no; question='Describe your perfect day.'; answer='Creating a perfect day involves a blend of activities, moments, and moments that resonate with your personal passions and prioriti'
    OMITTED_UNIT_IDS
    dropped=9: p92-06, p92-21, p92-22, p92-23, p92-26, p92-27, p92-31, p92-34, p92-44
    unclassified=4: p92-12, p92-14, p92-16, p92-17
    incomplete=1: p92-19
    total_part92_units=53; plotted_classified=39; omitted=14
    DS_FIRST_ATTEMPT_FAILURES
    first_attempt_errors=10; ssl=9; token_cap=1; retry_successes=10
    token_cap_item=p92-46 random q2; output_tokens=4000
    HAIKU_ERROR_ROWS=p92-19 neuron q4; p92-20 random q5


## Part 86 comparison and protocol record

Command:

    python3 - <<'PY'
    import json
    from pathlib import Path
    for version, path in [
        ("v1", "results/part86/second-reader/calib-agreement.json"),
        ("v2", "results/part86/second-reader/calib-agreement-v2.json"),
    ]:
        data = json.loads(Path(path).read_text())
        pairs = data["pairs"]
        if version == "v1":
            print(f"v1: n={data['sample_items']}; "
                  f"Haiku~Opus={pairs['first_haiku~second_opus']['kappa_coherent_persona']:.6f}; "
                  f"Haiku~DS={pairs['first_haiku~ds']['kappa_coherent_persona']:.6f}; "
                  f"Haiku~Luna={pairs['first_haiku~luna']['kappa_coherent_persona']:.6f}")
        else:
            print(f"v2: n={data['sample_items']}; "
                  f"Haiku~DS={pairs['first_haiku~ds_v2']['kappa_coherent_persona']:.6f}; "
                  f"Haiku~Luna={pairs['first_haiku~luna_v2']['kappa_coherent_persona']:.6f}")
    PY

Output:

    v1: n=335; Haiku~Opus=0.157189; Haiku~DS=0.121776; Haiku~Luna=0.143919
    v2: n=335; Haiku~DS=0.108689; Haiku~Luna=0.050708

Command:

    sed -n '1,6p;20,39p;44,64p;87,101p' PART92B-DESIGN.md
    rg -n 'judge_client.call_prompt|route="cliproxy"|max_tokens=4000' part92b/label92b.py
    rg -n -C 1 'cliproxy|route=' results/part86/second-reader/calib86.py

Output:

    Registered 2026-10-30 01:47 CT by Claude (Opus 5.5), before any data.
    - Items: all 636 answers of Part 92 step 2 (53 units, six questions, neuron and random arms), each with its unit's
      target persona name and description from results/part92/step2/neurons.json, filled into label92's exact prompt.
    - Readers: DS V4.1 Flash on all 636 (OpenCode Go, Part 118's transport, 4,000-token cap, default reasoning); GPT-6
      Luna on a seeded 15% sample (96 items, seed 9202, stratified by arm), through Part 86's calib86.py route
      (cliproxy), same prompt and its default effort, stated in the launch record. Both see only the target persona, the
      question and the answer, never the arm or unit. Error rows are retried with the same command until none remain or a
      pass fixes none; any left count as not showing, and their number is reported.
    - Kinds: Part 92's rule from part92/score92.py, applied to the DS labels unchanged: concept if the persona shows in 3
      or more of the six neuron-arm answers; slot if it shows in exactly one or two, only on "Who are you?" or "Tell me
      about yourself."; dropped if "Who are you?" does not show; otherwise unclassified.
    - Gate A (readers checked on known examples): before the main read, each reader labels 12 hand-written items whose
      targets are unrelated to Part 92's personas: 6 where the answer plainly is or describes the target, 6 where it plainly
      does not. A reader with fewer than 11 of 12 right stops the run.
    - Gate B: DS and Luna agree on shows_persona at kappa 0.6 or more on the 96-item sample. Below it, no prediction is
      read and the counts are reported as description.
    - The random arm (Part 92's paired random directions), read the same way, shows how often a reader marks a persona
      that the push did not put there.
    - Not used: shuffled targets, because the random arm and Gate A already show whether a reader marks a persona without
      it; dose ladders, poles and do-nothing hooks, because no model is run.
    - Smallest effect worth knowing: a reader change that moves a third of the slot units (7 of 21) out of slot. All 39
      units are read, so the kind counts are exact; kappa over about 634 items has a standard error near 0.04.
    - One model, one push size (alpha 0.8), one greedy answer per cell, and each reader's default settings. Haiku's labels
      came from a separate run with its own settings (label92's 500-token cap, no thinking parameter). Luna reads 15%. A
    - The registered reading rule says Parts 112, 117 and 118 keep own verdicts but kinds remain Haiku.


Command:

    rg -n -i 'The atlas here|Part 86.*atlas|Part 114.*attribution|Part 112.*atlas' results/part112/README.md results/part115/README.md PART114-DESIGN.md results/part117/README.md results/part117b/README.md | cut -c1-200

Output:

    results/part117b/README.md:3:Part 117 found that pushing a **concept neuron** (one whose target persona appeared in at least 3 of its six answers when pushed alone in [Part 112](../part112/README.md))
    results/part117b/README.md:13:A **matched** concept was selected for a similar theme; an **unrelated** concept was selected for a different theme. A **multiplier** scales a unit's Part 112 push: +1 ap
    results/part117b/README.md:25:| Sampled atlas-negative neuron direction at -1 with slot at +1 | Whether these sampled neuron directions outside Part 112's atlas give the same result at matched written
    results/part117b/README.md:100:Each unit used its Part 112 push, set to 0.8 times its layer's median **residual norm**, the length of the residual-stream activation in that layer. The push was applied
    results/part117b/README.md:109:- [control-neurons.json](control-neurons.json): sampled neuron directions outside Part 112's 465-unit persona atlas.
    results/part117/README.md:71:Part 92's blind Haiku reader used the rubric in [label92.py](../../part92/label92.py). Each target description shown to the reader included its Part 112 atlas kind, as rec
    results/part112/README.md:7:The atlas here is Part 86's list of 465 coherent persona targets. Each has a stored atlas kind and a separate Q-name flag. The flag tests a case-insensitive initial Q. For
    results/part112/README.md:106:The 465 targets are the coherent-persona neurons from Part 86's Qwen atlas. The six questions are Part 91's battery: “Who are you?”, “Tell me about yourself,” “
    results/part112/README.md:116:The OLMo Part 92 write-up measured when a Sonnet-selected shortlist of personas appeared across training checkpoints, using the same six-question concept and slot rule. I
    PART114-DESIGN.md:4:maybe two, highly responsible for the the leading Q it really likes in identity questions." In Part 86's atlas, 323 of
    results/part115/README.md:13:Here the top k neurons from Part 114's attribution ranking were zeroed together. On the tested grid, the top 100 took "Q" off first place on 3 of 16 held-out phrasings, th
    results/part115/README.md:68:Qwen/Qwen3.5-4B at revision 851bf6e8, bf16, the A770; chat template, no system prompt, thinking off; the answer forced to begin "I am **"; the readout is the log-odds of t


## Cold-check identity and count

Command:

    rg -n '^(# Part 92b cold check|Reader:|5 blocking)' results/part92b/COLD-CHECK-r1.md

Output:

    1:# Part 92b cold check, round 1
    3:Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-30, on `results/part92b/README.md`
    83:5 blocking, 13 minor.


## Additional protocol source rows

Command:

    sed -n '44p' results/part86/README.md | sed 's/\x60//g'
    sed -n '255,256p' PART86-DESIGN.md | sed 's/\x60//g'
    sed -n '4p;100p' PART92B-DESIGN.md
    python3 - <<'PY'
    import json
    from pathlib import Path
    sample = json.loads(Path("results/part92b/sample92b.json").read_text())
    print(f"seed={sample['seed']}; share={sample['share']}; items={sample['items']}; "
          f"counts_by_unit={sample['counts_by_unit']}")
    PY

Output:

    The first reader was Claude Haiku 5.5 with thinking disabled, run by the J-lens neuron bundle's own labelling script (code/label.py). Amendment 2's second reader was Claude Opus 5.5. Both were blind to layer, neuron, and condition. After their agreement fell below the 0.6 gate, DeepSeek V4.1 Flash and GPT-6 Luna were calibrated. In the version-1 335-item set, DS and Luna had kappa 0.568. Amendment 7 clarified the brief and used a separate, fresh 335-item version-2 set. On that set DS and Luna had kappa 0.814, raw agreement 90.7%, and agreement on the coherent-persona flag of 96.4%. Their agreement with the first reader was lower: kappa 0.109 for DS and 0.051 for Luna.
    **Why.** The first reader, whose labels built the atlas of 465 coherent-persona neurons that Parts 112, 115, 117 and
    117b started from, agrees with each of the three other readers at kappa 0.12 to 0.16 (Amendment 7). 387 of the 465 are
    Registered 2026-10-30 01:47 CT by Claude (Opus 5.5), before any data. Elise asked at 00:50 CT on 2026-10-30 to "keep trying to
    ## Amendment 1 (2026-10-30 01:55 CT, before any label)
    seed=9202; share=0.15; items=96; counts_by_unit={'neuron': 48, 'random': 48}


## Random-arm wording check

Command:

    python3 - <<'PY'
    import json
    from pathlib import Path
    answers = {
        (r["id"], r["unit"], int(r["question_index"])): r
        for r in (json.loads(s) for s in Path("results/part92/step2/answers/instruct_chat.jsonl").read_text().splitlines())
    }
    for uid, arm, q in (
        ("p92-49", "random", 0), ("p92-49", "random", 2),
        ("p92-49", "random", 5), ("p92-06", "random", 3),
    ):
        row = answers[(uid, arm, q)]
        answer = row["answer"]
        if uid == "p92-06":
            index = answer.lower().find("bird")
            context = answer[max(0, index - 35):index + 45]
            print(f"{uid} {arm} q{q}: {row['question']}; bird_context={context!r}")
        else:
            print(f"{uid} {arm} q{q}: {row['question']}; "
                  f"has_first_person={'i ' in answer.lower()}; has_ai={'ai' in answer.lower()}")
    PY

Output:

    p92-49 random q0: Who are you?; has_first_person=True; has_ai=True
    p92-49 random q2: What do you want most?; has_first_person=True; has_ai=True
    p92-49 random q5: Write a short poem about the sea.; has_first_person=False; has_ai=False
    p92-06 random q3: Describe your perfect day.; bird_context="where the sun shines brightly, the birds chirp cheerily, and you're feeling exce"


## Cold-check limitations cited in the write-up

Command:

    sed -n '80,81p' results/part92b/COLD-CHECK-r1.md

Output:

    Not checked: Parts 112, 117 and 118's verdicts; DS's raw reasoning; the cap Part 92's Haiku run used at run time; the
    test suite.
