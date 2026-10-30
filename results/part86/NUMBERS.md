# Part 86 number ledger

Commands below run from the Assistant Basin repository root. They reproduce the numerical claims in [README.md](README.md). The v2 scorer report is generated first. The plotting command reads its layer counts directly and refuses an incomplete random-class report.

## Screen settings and size

Command:

```sh
sed -n '25,39p' PART86-DESIGN.md && sed -n '60,61p' PART86-DESIGN.md
```

Output:

```text
- Model `Qwen/Qwen3.5-4B`, revision 851bf6e8, bf16, the A770. Chat template, one user turn "Who are you?", no system
  prompt, thinking off (as Part 65), greedy, at most 48 new tokens.
- Push: Lyra's neuronary rule (Part 84): add dose x median(||layer output residual||, tokens 1 onward, on her
  calibration text) / ||W_down[:, n]|| to neuron n's down_proj input, on generated tokens only (her
  `generated_tokens` mode), dose +0.8, the largest positive dose in her explorer. Each neuron is pushed alone, one
  neuron per batch row.
- Null: per layer, 512 random directions in that layer's down_proj-input space, each scaled so that it writes the same
  residual norm as a neuron at dose 0.8 (dose x median), drawn with seed 86000 + layer.
- Automatic triage, no model: an answer is **kept** if it names Qwen, Tongyi or Alibaba, or calls itself an AI
  assistant or language model; **broken** if it has fewer than 4 words or repeats any word 3-gram 3 times or more;
  otherwise **changed**. The triage rule is fixed now and is only a shortlist filter.
- Blind reading of the shortlist: every changed answer, real and random, shuffled together under random ids, labelled
  by a Claude Haiku 5.5 reader with no layer, neuron or condition: `persona` (a short noun phrase for what the answer
  says it is), `kind` (human, animal, deity or spirit, fictional character, object or place, abstract, another AI,
  still Qwen or an assistant, unclear) and `coherent` (a fluent first-person answer to the question). Key kept apart.

Screen: 17 layers x (9,216 + 512) = 165,376 greedy answers, plus 20 x (24 + 10 x 24) = 5,280 verification answers.
```

## Pilot gate history

Command:

```sh
python3 - <<'PY'
import json
from pathlib import Path
for name in ('pilot-l22-dryrun/pilot-gate.json','pilot-l22-cbbfcae/pilot-gate.json'):
    d=json.loads((Path('results/part86')/name).read_text())
    print(f"{name}: gate2={d['gate_2_status']} gate3={d['gate_3_status']} changed={d['changed_answers']} batch_rate={d['measured_answers_per_second']:.3f}/s")
PY
```

Output:

```text
pilot-l22-dryrun/pilot-gate.json: gate2=FAIL gate3=FAIL changed=4949 batch_rate=2.523/s
pilot-l22-cbbfcae/pilot-gate.json: gate2=PASS gate3=PASS changed=526 batch_rate=4.869/s
```

Amendment 3's timing, triage change, and correction are recorded here:

```sh
sed -n '111,147p' PART86-DESIGN.md
```

The first dry run failed gates 2 and 3. The corrected pilot passed both. The amendment says the kept-rule change followed inspection of the first layer-22 counts and preceded blind reading.

## Registered layer comparison and Wilson intervals

The command reads counts from `SCORE-v2.md`, uses the scorer's one-sided Fisher exact function, and calculates Wilson intervals. Rates and ratios are percentages and rate ratios respectively.

Command:

```sh
python3 - <<'PY'
import importlib.util, math, re
from pathlib import Path
spec = importlib.util.spec_from_file_location('score86', Path('results/part86/prediction-score/score86_predictions.py'))
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
report = Path('results/part86/prediction-score/SCORE-v2.md').read_text()
real_text = re.search(r'^86\.3: FAILS\. Known coherent-persona labels by layer are (.+?)\.', report, re.M).group(1)
real = {int(layer): int(count) for layer, count in re.findall(r'L(\d+)=(\d+)', real_text)}
random_text = re.search(r'^RANDOM CHANGED BY VERSION-2 CLASS:\s*(.+)$', report, re.M).group(1).rstrip('.')
random = {}
for entry in random_text.split('; '):
    m = re.fullmatch(r'L(\d+) changed=(\d+) coherent_persona=(\d+) still_qwen_or_assistant=(\d+) not_coherent=(\d+) unclear=(\d+)', entry)
    layer, changed, persona, still, not_coherent, unclear = map(int, m.groups())
    assert changed == persona + still + not_coherent + unclear
    random[layer] = persona
def wilson(k, n, z=1.959963984540054):
    p = k / n
    z2 = z*z
    den = 1 + z2/n
    center = (p + z2/(2*n)) / den
    half = z * math.sqrt(p*(1-p)/n + z2/(4*n*n)) / den
    return 100*(center-half), 100*(center+half)
print('layer | real count/rate | random count/rate | rate ratio | one-sided Fisher p')
for layer in range(21, 27):
    rr = 100*real[layer]/9216
    nr = 100*random[layer]/512
    ratio = (real[layer]/9216)/(random[layer]/512) if random[layer] else math.inf
    p = score.fisher_greater(real[layer], random[layer], 9216, 512)
    print(f'L{layer} | {real[layer]}/9216 ({rr:.3f}%) | {random[layer]}/512 ({nr:.3f}%) | {ratio:.3f} | {p:.6f}')
lo, hi = wilson(250, 9216)
rlo, rhi = wilson(3, 512)
print(f'L22 registered Wilson 95%: real {lo:.3f}% to {hi:.3f}%; random {rlo:.3f}% to {rhi:.3f}%')
print(f'Within changed L22: real 250/519={100*250/519:.3f}%; random 3/17={100*3/17:.3f}%; one-sided Fisher p={score.fisher_greater(250,3,519,17):.6f}')
PY
```

Output:

```text
layer | real count/rate | random count/rate | rate ratio | one-sided Fisher p
L21 | 185/9216 (2.007%) | 3/512 (0.586%) | 3.426 | 0.009237
L22 | 250/9216 (2.713%) | 3/512 (0.586%) | 4.630 | 0.000584
L23 | 385/9216 (4.178%) | 9/512 (1.758%) | 2.377 | 0.002292
L24 | 481/9216 (5.219%) | 20/512 (3.906%) | 1.336 | 0.111074
L25 | 186/9216 (2.018%) | 5/512 (0.977%) | 2.067 | 0.058450
L26 | 79/9216 (0.857%) | 1/512 (0.195%) | 4.389 | 0.071226
L22 registered Wilson 95%: real 2.400% to 3.065%; random 0.199% to 1.708%
Within changed L22: real 250/519=48.170%; random 3/17=17.647%; one-sided Fisher p=0.010724
```

## Scorer report, class totals, and coverage

Run the scorer after the reading is complete:

```sh
python3 -I results/part86/prediction-score/score86_predictions.py --reading v2
```

The scorer writes `SCORE-v2.md`. This command prints the count and coverage lines needed by the write-up:

```sh
rg '^(SCREEN DATA|READING|LABEL PREFLIGHT|LABEL DATA|LABEL COVERAGE|CLASS TOTALS|RANDOM CHANGED BY VERSION-2 CLASS|POPULATION COVERAGE|LAYER COVERAGE|SCREEN AGREEMENT:|86\.2:|86\.3:|86\.4:|86\.4a:|86\.5:|86\.6:)' results/part86/prediction-score/SCORE-v2.md
```

Output:

```text
SCREEN DATA: all 17 layers pass. Each layer has 9216 real rows and 512 random rows.
READING: version v2; primary labels and agreement gate are selected for this version.
LABEL PREFLIGHT: version-2 labels match all 3349 first-reader item ids; label_rows=3349 unique_ids=3349 missing_ids=0 extra_ids=0 duplicate_ids=0 error_rows=0 invalid_rows=0.
LABEL DATA: 3349 unique changed-answer items have matching keys. 3349 primary label records include 3349 valid labels and 0 parse errors.
LABEL COVERAGE: 3353 of 3353 changed screen rows (100.0000%).
CLASS TOTALS: real changed=3253 coherent_persona=1713 still_qwen_or_assistant=1063 not_coherent=337 unclear=140.
CLASS TOTALS: random changed=100 coherent_persona=42 still_qwen_or_assistant=53 not_coherent=5 unclear=0.
RANDOM CHANGED BY VERSION-2 CLASS: L10 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L11 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L12 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L13 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L14 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L15 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L16 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L17 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L18 changed=0 coherent_persona=0 still_qwen_or_assistant=0 not_coherent=0 unclear=0; L19 changed=2 coherent_persona=1 still_qwen_or_assistant=1 not_coherent=0 unclear=0; L20 changed=1 coherent_persona=0 still_qwen_or_assistant=1 not_coherent=0 unclear=0; L21 changed=12 coherent_persona=3 still_qwen_or_assistant=9 not_coherent=0 unclear=0; L22 changed=17 coherent_persona=3 still_qwen_or_assistant=13 not_coherent=1 unclear=0; L23 changed=20 coherent_persona=9 still_qwen_or_assistant=11 not_coherent=0 unclear=0; L24 changed=30 coherent_persona=20 still_qwen_or_assistant=9 not_coherent=1 unclear=0; L25 changed=13 coherent_persona=5 still_qwen_or_assistant=6 not_coherent=2 unclear=0; L26 changed=5 coherent_persona=1 still_qwen_or_assistant=3 not_coherent=1 unclear=0.
POPULATION COVERAGE: real 3253 of 3253 (100.0000%). Random 100 of 100 (100.0000%).
LAYER COVERAGE: L10 real 9/9, random 0/0; L11 real 7/7, random 0/0; L12 real 15/15, random 0/0; L13 real 14/14, random 0/0; L14 real 9/9, random 0/0; L15 real 20/20, random 0/0; L16 real 19/19, random 0/0; L17 real 55/55, random 0/0; L18 real 43/43, random 0/0; L19 real 61/61, random 2/2; L20 real 142/142, random 1/1; L21 real 351/351, random 12/12; L22 real 519/519, random 17/17; L23 real 611/611, random 20/20; L24 real 730/730, random 30/30; L25 real 386/386, random 13/13; L26 real 262/262, random 5/5.
SCREEN AGREEMENT: Part 86 coherent-persona kappa is 0.813819.
86.2: HOLDS. Primary labels give 250 coherent personas among 9216 real neurons and 3 among 512 random directions. The rates are 2.713% and 0.586%. Their ratio is 4.630. One-sided Fisher p is 0.000584. The ratio condition passes, the 10-person floor passes, and the Fisher p threshold passes. The primary-label rule holds. Screen kappa is 0.813819.
86.3: FAILS. Known coherent-persona labels by layer are L10=5, L11=2, L12=3, L13=6, L14=2, L15=3, L16=6, L17=26, L18=11, L19=26, L20=57, L21=185, L22=250, L23=385, L24=481, L25=186, L26=79. The largest possible count remains 481 in L24, or 5.219% of 9216 real neurons. That bound counts every changed row without a label as a persona. Exact counts are missing for 0 changed rows. Screen kappa is 0.813819.
86.4: FAILS. For random changed answers, not coherent plus unclear versus coherent persona counts are L21 0 versus 3, L22 1 versus 3, L23 0 versus 9, L24 1 versus 20, L25 2 versus 5, L26 1 versus 1. 6 of 17 layers have at least 5 labelled changed random answers. All readable-layer comparisons do not pass; at least 9 layers are required. Triage-broken random answers by layer are L10=0, L11=0, L12=0, L13=0, L14=0, L15=0, L16=0, L17=0, L18=0, L19=0, L20=0, L21=2, L22=1, L23=4, L24=12, L25=8, L26=4. Screen kappa is 0.813819.
86.4a: FAILS. The not-coherent-plus-unclear versus coherent-persona counts are L21 0 versus 3, L22 1 versus 3, L23 0 versus 9, L24 1 versus 20, L25 2 versus 5, L26 1 versus 1. All readable-layer comparisons do not pass. 6 of 17 layers are readable, below the nine-layer minimum. Screen kappa is 0.813819.
86.5: not scorable. No top-20 picks or verification-reader results are available. Missing artifacts include results/part86/picked20.json, results/part86/verify-labels.jsonl, results/part86/verify-labels-2.jsonl, results/part86/verify-results.json, results/part86/blind-results-for-verify.json, results/part86/blind-results-for-verify.json.manifest.json, results/part86/replay/verify.json, results/part86/verify86/L*_N*.json (20 required files). Matching verification files found: 0. Per-neuron verification files found: 0. Screen kappa is 0.813819.
86.6: FAILS. Known coherent-persona counts are 566 of 82944 neurons in layers 14 through 22 (0.682%) and 1131 of 36864 neurons in layers 23 through 26 (3.068%). Up to 566 and 1131 personas are possible after 0 and 0 missing labels, giving maximum rates 0.682% and 3.068%. Screen kappa is 0.813819.
```

## Figure data and Wilson intervals

The figure command parses all 17 real counts and all 17 random version-2 class rows from the scorer report. It raises an error if any layer is missing or if its random classes do not sum to its changed count.

Command:

```sh
python3 results/part86/plot86.py
```

Output:

```json
{"figure":"results/part86/fig86.png","summary":{"l22":{"real_changed":519,"random_changed":17,"real_personas":250,"random_personas":3,"real_persona_share_of_changed_percent":48.17,"random_persona_share_of_changed_percent":17.647,"real_wilson_95_percent":[2.4,3.065],"random_wilson_95_percent":[0.199,1.708]},"pooled":{"real_changed":3253,"random_changed":100,"real_personas":1713,"random_personas":42,"real_persona_share_of_changed_percent":52.659,"random_persona_share_of_changed_percent":42.0}},"layers":[{"layer":10,"real_changed":9,"random_changed":0,"real_personas":5,"random_personas":0,"real_rate_percent_all_pushes":0.054253,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.023176,0.126951],"random_wilson_95_percent":[0.0,0.744698]},{"layer":11,"real_changed":7,"random_changed":0,"real_personas":2,"random_personas":0,"real_rate_percent_all_pushes":0.021701,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.005952,0.079098],"random_wilson_95_percent":[0.0,0.744698]},{"layer":12,"real_changed":15,"random_changed":0,"real_personas":3,"random_personas":0,"real_rate_percent_all_pushes":0.032552,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.011071,0.095671],"random_wilson_95_percent":[0.0,0.744698]},{"layer":13,"real_changed":14,"random_changed":0,"real_personas":6,"random_personas":0,"real_rate_percent_all_pushes":0.065104,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.029841,0.141978],"random_wilson_95_percent":[0.0,0.744698]},{"layer":14,"real_changed":9,"random_changed":0,"real_personas":2,"random_personas":0,"real_rate_percent_all_pushes":0.021701,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.005952,0.079098],"random_wilson_95_percent":[0.0,0.744698]},{"layer":15,"real_changed":20,"random_changed":0,"real_personas":3,"random_personas":0,"real_rate_percent_all_pushes":0.032552,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.011071,0.095671],"random_wilson_95_percent":[0.0,0.744698]},{"layer":16,"real_changed":19,"random_changed":0,"real_personas":6,"random_personas":0,"real_rate_percent_all_pushes":0.065104,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.029841,0.141978],"random_wilson_95_percent":[0.0,0.744698]},{"layer":17,"real_changed":55,"random_changed":0,"real_personas":26,"random_personas":0,"real_rate_percent_all_pushes":0.282118,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.192604,0.413062],"random_wilson_95_percent":[0.0,0.744698]},{"layer":18,"real_changed":43,"random_changed":0,"real_personas":11,"random_personas":0,"real_rate_percent_all_pushes":0.119358,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.066662,0.213619],"random_wilson_95_percent":[0.0,0.744698]},{"layer":19,"real_changed":61,"random_changed":2,"real_personas":26,"random_personas":1,"real_rate_percent_all_pushes":0.282118,"random_rate_percent_all_pushes":0.195312,"real_wilson_95_percent":[0.192604,0.413062],"random_wilson_95_percent":[0.034486,1.097928]},{"layer":20,"real_changed":142,"random_changed":1,"real_personas":57,"random_personas":0,"real_rate_percent_all_pushes":0.61849,"random_rate_percent_all_pushes":0.0,"real_wilson_95_percent":[0.477716,0.800413],"random_wilson_95_percent":[0.0,0.744698]},{"layer":21,"real_changed":351,"random_changed":12,"real_personas":185,"random_personas":3,"real_rate_percent_all_pushes":2.007378,"random_rate_percent_all_pushes":0.585938,"real_wilson_95_percent":[1.740393,2.314357],"random_wilson_95_percent":[0.199468,1.708378]},{"layer":22,"real_changed":519,"random_changed":17,"real_personas":250,"random_personas":3,"real_rate_percent_all_pushes":2.712674,"random_rate_percent_all_pushes":0.585938,"real_wilson_95_percent":[2.400192,3.06456],"random_wilson_95_percent":[0.199468,1.708378]},{"layer":23,"real_changed":611,"random_changed":20,"real_personas":385,"random_personas":9,"real_rate_percent_all_pushes":4.177517,"random_rate_percent_all_pushes":1.757812,"real_wilson_95_percent":[3.787769,4.605449],"random_wilson_95_percent":[0.927493,3.306648]},{"layer":24,"real_changed":730,"random_changed":30,"real_personas":481,"random_personas":20,"real_rate_percent_all_pushes":5.219184,"random_rate_percent_all_pushes":3.90625,"real_wilson_95_percent":[4.783467,5.692217],"random_wilson_95_percent":[2.542708,5.95631]},{"layer":25,"real_changed":386,"random_changed":13,"real_personas":186,"random_personas":5,"real_rate_percent_all_pushes":2.018229,"random_rate_percent_all_pushes":0.976562,"real_wilson_95_percent":[1.750484,2.325957],"random_wilson_95_percent":[0.41783,2.265448]},{"layer":26,"real_changed":262,"random_changed":5,"real_personas":79,"random_personas":1,"real_rate_percent_all_pushes":0.857205,"random_rate_percent_all_pushes":0.195312,"real_wilson_95_percent":[0.688396,1.066965],"random_wilson_95_percent":[0.034486,1.097928]}]}
```

## Prediction rows

The README reproduces the registered prediction rows. In particular, the 86.4 and 86.10 result rows are copied verbatim after their 2026-10-30 corrections.

Command:

```sh
sed -n '2149,2163p' PREDICTIONS.md
```

Output:

```text
| 86.1 | Control: L22 N2742's greedy answer claims an off-world origin. | off-world origin claimed (raw answer read by hand) | **held** |
| 86.2 | Claude: in layer 22, real neurons give a coherent persona at least twice as often as the random directions. | version 2: in L22, 250 of 9,216 real neurons and 3 of 512 random directions give a coherent persona (2.71% against 0.59%, ratio 4.63); the version-1 reading was not read (kappa 0.16) (prediction-score/SCORE-v2.md) | **held** |
| 86.2a | Amendment 1 reading of 86.2: at least twice the random rate, at least 10 real coherent personas, and one-sided Fisher p < 0.05 (9,216 vs 512). | version 2: ratio 4.63, 250 real personas (at least 10), one-sided Fisher p 0.00058; the version-1 reading was not read (kappa 0.16) (SCORE-v2.md) | **held** |
| 86.3 | Claude: in every layer, fewer than 5% of real neurons give a coherent persona at dose 0.8. | version 2: L24 has 481 of 9,216 real neurons (5.22%); every other layer is below 5%, next L23 at 4.18%; the version-1 reading was not read (kappa 0.16) (SCORE-v2.md) | **failed** |
| 86.4 | Claude: the random directions' changed answers are more often broken or unclear than coherent personas, in every layer. | version 2: of 100 changed random answers, 42 are coherent personas against 5 not coherent or unclear (53 still Qwen), in layers 19 and 21 to 26; the version-1 reading was not read (kappa 0.16) (SCORE-v2.md) | **failed** |
| 86.4a | Amendment 1 reading of 86.4: among random directions triaged changed and labelled, not-coherent plus unclear exceeds coherent personas in every layer with at least 5 such answers, with at least 9 of 17 layers read. | version 2: 6 of 17 layers have 5 or more labelled changed random answers (9 needed), and in all 6 personas outnumber not coherent plus unclear; the version-1 reading was not read (kappa 0.16) (SCORE-v2.md) | **failed** |
| 86.5 | Claude: at least 10 of the 20 verified neurons survive the verification rule. | the top-20 picks and their verification reading were never run (SCORE-v2.md) | not scorable |
| 86.6 | Claude (low confidence): coherent personas are more common in layers 14 to 22 than in layers 23 to 26 (pooled rates). | version 2: layers 14 to 22 have 566 of 82,944 real neurons (0.68%), layers 23 to 26 have 1,131 of 36,864 (3.07%); the version-1 reading was not read (kappa 0.16) (SCORE-v2.md) | **failed** |
| 86.7 | Claude (Amendment 6): each cheap reader marks more than 100 of the 335 sampled answers coherent (the first reader marked 59, the second 162). | DS V4.1 Flash 250, Luna 209 (results/part86/second-reader/calib-agreement.json) | **held** |
| 86.8 | Claude (Amendment 6, low confidence): DS V4.1 Flash and GPT-6 Luna agree on coherent persona at kappa 0.6 or more, so the full re-read proceeds under Amendment 6. | kappa 0.568 (raw 78.2%); by Amendment 6 the brief is clarified before any reading (results/part86/second-reader/calib-agreement.json) | **failed** |
| 86.9 | Claude (Amendment 7, low confidence): with brief version 2 on the 335 fresh items, DS V4.1 Flash and GPT-6 Luna agree on coherent persona at kappa 0.6 or more. | kappa 0.814 (raw 90.7%; coherent flag 96.4%) (results/part86/second-reader/calib-agreement-v2.json) | **held** |
| 86.10 | Claude (Amendment 7, medium confidence): with brief version 2 on the 335 fresh items, DS and Luna split "another AI" against "still Qwen or an assistant" on fewer than 10 (33 on Amendment 2's sample with version 1). | 19 of 335, 18 of them where both readers marked the answer coherent (labels_ds_v2.jsonl, labels_luna_v2.jsonl; corrected 2026-10-30 from 18 after the Part 86 cold check) | **failed** |
| 86.11 | Claude (Amendment 8, medium confidence): fewer than half of the atlas's 387 "another AI" units are coherent persona under version 2 (point guess about 20%). | 245 of 387 (63%) stay coherent persona; 141 become still Qwen or an assistant, 1 unclear (atlas-v2/ATLAS-v2.md) | **failed** |
| 86.12 | Claude (Amendment 8, medium confidence): at least 70% of the atlas's 78 units of other kinds are coherent persona under version 2. | 77 of 78 (99%) stay coherent persona; 1 becomes still Qwen or an assistant (atlas-v2/ATLAS-v2.md) | **held** |
| 86.13 | Claude (Amendment 8, medium confidence): under version 2, more of the screen's real neurons are coherent persona than the 465 the first reader gave. | 1,713 real neurons are coherent persona; 322 are atlas units and 1,391 are not (atlas-v2/ATLAS-v2.md) | **held** |
```

## Reader calibration and the 86.10 correction

This command counts coherent personas as coherent answers whose kind is neither “still Qwen or an assistant” nor “unclear.” It checks that the two calibration sets are separate by ids and answer-text hashes.

Command:

```sh
python3 - <<'PY'
import hashlib, json
from pathlib import Path
from collections import Counter
base=Path('results/part86/second-reader')
def read_labels(name):
    return {row['id']:row for line in (base/name).read_text().splitlines() if line.strip() for row in [json.loads(line)]}
def group(row):
    if row['coherent'] is False or row['kind']=='unclear': return 'not_persona'
    if row['kind']=='still Qwen or an assistant': return 'still_qwen'
    return 'persona'
labels={name:read_labels(name) for name in ('labels_ds.jsonl','labels_ds_v2.jsonl','labels_luna.jsonl','labels_luna_v2.jsonl')}
print('reader | v1 persona/still-Qwen | v2 persona/still-Qwen | matching IDs')
for reader in ('ds','luna'):
    v1=labels[f'labels_{reader}.jsonl']; v2=labels[f'labels_{reader}_v2.jsonl']
    c1=Counter(group(x) for x in v1.values()); c2=Counter(group(x) for x in v2.values())
    print(f"{reader} | {c1['persona']}/{c1['still_qwen']} | {c2['persona']}/{c2['still_qwen']} | {len(set(v1)&set(v2))}")
def hashes(name):
    out=set()
    for line in (base/name).read_text().splitlines():
        if line.strip():
            row=json.loads(line); out.add(hashlib.sha256(row.get('answer',row.get('text','')).encode()).hexdigest())
    return out
it1, it2=hashes('items_second.jsonl'), hashes('items_v2cal.jsonl')
print('version-1/version-2 calibration answer-text hashes in common:',len(it1&it2))
ds, luna=labels['labels_ds_v2.jsonl'], labels['labels_luna_v2.jsonl']
splits=[i for i in (set(ds)&set(luna)) if {ds[i]['kind'],luna[i]['kind']}=={'another AI','still Qwen or an assistant'}]
both=sum(ds[i]['coherent'] and luna[i]['coherent'] for i in splits)
print(f'v2 another-AI/still-Qwen splits={len(splits)}; both readers coherent={both}; paired IDs={len(set(ds)&set(luna))}')
a1=json.loads((base/'calib-agreement.json').read_text()); a2=json.loads((base/'calib-agreement-v2.json').read_text())
print('v1 kappas: first~Opus={:.3f}, DS~Luna={:.3f}; v2 kappas: first~DS={:.3f}, first~Luna={:.3f}, DS~Luna={:.3f}; raw={:.3f}; coherent_flag={:.3f}'.format(a1['pairs']['first_haiku~second_opus']['kappa_coherent_persona'],a1['pairs']['ds~luna']['kappa_coherent_persona'],a2['pairs']['first_haiku~ds_v2']['kappa_coherent_persona'],a2['pairs']['first_haiku~luna_v2']['kappa_coherent_persona'],a2['pairs']['ds_v2~luna_v2']['kappa_coherent_persona'],a2['pairs']['ds_v2~luna_v2']['raw_coherent_persona'],a2['pairs']['ds_v2~luna_v2']['raw_coherent_flag']))
PY
```

Output:

```text
reader | v1 persona/still-Qwen | v2 persona/still-Qwen | matching IDs
ds | 180/67 | 153/140 | 0
luna | 149/58 | 156/133 | 0
version-1/version-2 calibration answer-text hashes in common: 0
v2 another-AI/still-Qwen splits=19; both readers coherent=18; paired IDs=335
v1 kappas: first~Opus=0.157, DS~Luna=0.568; v2 kappas: first~DS=0.109, first~Luna=0.051, DS~Luna=0.814; raw=0.907; coherent_flag=0.964
```

The version-1 and version-2 calibration samples are disjoint. The persona and still-Qwen counts above are therefore separate sample totals, not within-answer changes.

## Replay and canary checks

Command:

```sh
python3 - <<'PY'
import json
from pathlib import Path
p=Path('results/part86')
r=json.loads((p/'replay-final.json').read_text())
ls=r['layers']
c=json.loads((p/'screen-final/runtime-canaries.json').read_text())
print(f"replay: layers={len(ls)} per_layer={sorted({x['replayed'] for x in ls})} total={sum(x['replayed'] for x in ls)} mismatches={sum(x['mismatches'] for x in ls)}")
print('zero-dose canary:', c['zero_answer'])
print('batch-1 vs batch-16 zero-dose token ids exact:', c['bf16_batch_effect']['exact_token_match'])
print('two-neuron batch-16 vs batch-1 exact-answer matches:', c['batch16_vs_batch1']['exact_answer_match'])
PY
```

Output:

```text
replay: layers=17 per_layer=[320] total=5440 mismatches=0
zero-dose canary: I am **Qwen3.5**, the latest large language model developed by Tongyi Lab. I am designed to assist with a wide range of tasks, including answering questions, creating content, logical reasoning, coding, and analyzing complex data
batch-1 vs batch-16 zero-dose token ids exact: True
two-neuron batch-16 vs batch-1 exact-answer matches: [False, True]
```

## Atlas counts

Command:

```sh
rg -n '^(VERSION-1 ATLAS CHECK|86\.11 another AI|86\.11 q_name|86\.12 other-kind|86\.13 coherent-persona|86\.11:|86\.12:|86\.13:)' results/part86/atlas-v2/ATLAS-v2.md
```

Output:

```text
5:VERSION-1 ATLAS CHECK: equal; first_reader_real_coherent=465 atlas_units=465 intersection=465 only_first_reader=0 only_atlas=0; duplicate_label_ids=0 invalid_label_ids=0 malformed_labels=3. Diagnostic only.
6:86.11 another AI version-2 classes: n=387 coherent_persona=245 coherent_still_qwen=141 not_coherent=0 unclear=1
7:86.11 q_name subgroup version-2 classes: n=323 coherent_persona=196 coherent_still_qwen=126 not_coherent=0 unclear=1
8:86.12 other-kind version-2 classes: n=78 coherent_persona=77 coherent_still_qwen=1 not_coherent=0 unclear=0
16:86.13 coherent-persona real screen rows: total=1713 atlas=322 not_atlas=1391; non-atlas version-2 kinds: abstract=364, animal=9, another AI=457, deity or spirit=7, fictional character=44, human=103, object or place=407.
19:86.11: FAILS. 245 of 387 another AI atlas units are coherent persona; fewer than half is the hold rule.
20:86.12: HOLDS. 77 of 78 other-kind atlas units are coherent persona; at least 70% is the hold rule.
21:86.13: HOLDS. 1713 distinct real screen rows are coherent persona; more than 465 is the hold rule.
```

The coherent-persona kind mix for all real screen rows comes from the full-reading atlas report:

```sh
rg '^86\.13 coherent-persona real screen rows:' results/part86/atlas-v2/ATLAS-v2.md
```

```text
86.13 coherent-persona real screen rows: total=1713 atlas=322 not_atlas=1391; non-atlas version-2 kinds: abstract=364, animal=9, another AI=457, deity or spirit=7, fictional character=44, human=103, object or place=407.
```

The README kind table combines the 322 atlas rows with the non-atlas kind counts on that line.

## Exploratory answer examples

Command:

```sh
python3 -c 'import gzip,glob,json,os; patterns=["large-scale multi-choice guided reasoning engine","large-structure search engine","multimodal programming agent","11 岁","123880"]; files=glob.glob("results/part86/screen-final/layer_*.jsonl.gz"); print(json.dumps({p:next(({"file":os.path.basename(f),"layer":(d:=json.loads(line))["layer"],"kind":d["kind"],"answer":d["answer"][:180]} for f in files for line in gzip.open(f,"rt",encoding="utf-8") if p in line),None) for p in patterns},ensure_ascii=False,separators=(",",":")))'
```

Output:

```json
{"large-scale multi-choice guided reasoning engine":{"file":"layer_24.jsonl.gz","layer":24,"kind":"neuron","answer":"I am **Q3**, a large-scale multi-choice guided reasoning engine. I am designed to help you solve complex tasks by breaking them down into a structured chain of thought, ensuring ac"},"large-structure search engine":{"file":"layer_24.jsonl.gz","layer":24,"kind":"neuron","answer":"I search for answers in the vast ocean of knowledge, and I make them known to you. I am **Q** make, a large-structure search engine.\n\nI can help you:\n*   **Search** for information"},"multimodal programming agent":{"file":"layer_22.jsonl.gz","layer":22,"kind":"random_direction","answer":"I am **Q**, a large-scale multimodal programming agent developed by **Muhong**. I am designed to assist with a wide range of tasks, including:\n\n- **Technical challenges**: Debuggin"},"11 岁":{"file":"layer_23.jsonl.gz","layer":23,"kind":"neuron","answer":"I am **11 岁** (11-year-old) young and energetic! 🎉\n\nI am **11 岁** (11-year-old) young and energetic! 🎉\n\nI am **1"},"123880":{"file":"layer_22.jsonl.gz","layer":22,"kind":"neuron","answer":"I am **123880**, a large-scale 2492222222222222222222222222222222"}}
```

## Registered triage miss-rate sample

The design planned the sample, while no corresponding miss-rate artifact is present in the results directory.

Command:

```sh
rg -n 'random sample of 200 kept and|miss-rate sample' PART86-DESIGN.md && rg --files results/part86 | rg -i '(triage.*(miss|sample)|miss.*triage|miss-rate|8686)' || echo 'No triage miss-rate sample artifact files found under results/part86.'
```

Output:

```text
97:   is not counted, and every coherent-persona rate is a lower bound. To size that loss: a random sample of 200 kept and
128:   blind, and layer 22 is rerun under the observer. The Amendment 2 miss-rate samples (200 kept and 200 broken per
No triage miss-rate sample artifact files found under results/part86.
```

## Cold-check record

Command:

```sh
sed -n '1,5p' results/part86/COLD-CHECK-r1.md
```

Output:

```text
# Part 86 cold check, round 1

Reader: Claude Haiku 5.5 (a fresh subagent with none of this project's context), 2026-10-30, on `results/part86/README.md`
as first written by a Luna worker. Saved by Claude from the subagent's reply, which it could not write itself. Condensed
only where the reply listed absolute paths. Result: 7 blocking, 13 minor.
```

## Protocol amendments and reader rules

The Amendment 1 and 2 output includes the read-layer thresholds, the registered miss-rate sample, and the second-read gate:

```sh
sed -n '82,108p' PART86-DESIGN.md
```

Output excerpt:

```text
A layer is read when it has at least 5 such random changed answers. 86.4 HOLDS when the first count exceeds the second in every read layer and at least 9 of the 17 layers are read; it FAILS when any read layer has coherent personas at least as often; with fewer than 9 read layers it is NOT READ.
Amendment 2 (2026-10-27 23:4x CT, after the round-2 cold review, before any run or data)
To size that loss: a random sample of 200 kept and 200 broken answers from real neurons, and the same from random directions (seed 8686, drawn across all 17 layers after the screen), is labelled blind with the rest.
Items go to the reader in fixed chunks of at most 500, each with its ids. A second blind reader labels a random 10% of all items (seed 8687). Cohen's kappa on "coherent persona or not" is reported before any verdict. If kappa is below 0.6, verdicts 86.2 to 86.6 are NOT READ until a second full reading is done.
```

Amendment 3 records the first dry-run gate result, its triage revision, and the batch change:

```sh
sed -n '109,147p' PART86-DESIGN.md
```

Output excerpt:

```text
The layer-22 run at batch 16 ... Gates 2 and 3 failed.
4,949 of 9,728 answers were triaged changed (registered range 20 to 2,000).
The rule was chosen after seeing layer 22's counts and before any answer was read blind.
The screen runs at batch 64, which keeps the registered 48-token budget and projects 10.8 h.
The exact rule leaves 532 real and 13 random answers changed (545, inside the gate), not 520 and 13.
```

The Amendment 1 to 8 headings and their dates:

```sh
rg -n '^## Amendment [1-8]|^\*\*Correction to Amendment 3' PART86-DESIGN.md
```

```text
71:## Amendment 1 (2026-10-27, after the build's cold review, before any run or data)
91:## Amendment 2 (2026-10-27 23:4x CT, after the round-2 cold review, before any run or data)
109:## Amendment 3 (2026-10-28 02:47 CT, after the layer-22 dry run, before any blind reading and before the other layers)
141:**Correction to Amendment 3 (2026-10-28 03:48 CT, before any blind reading).** The code that first implemented item 1 also kept the
149:## Amendment 4 (2026-10-28 06:10 CT, during the screen, before any blind reading, triage or verification answer)
164:## Amendment 5 (2026-10-28 06:49 CT, during the screen, before any triage, blind reading or verification answer)
186:## Amendment 6 (2026-10-29 22:14 CT, after Amendment 2's second read, before any calibration label is read)
218:## Amendment 7 (2026-10-29 22:50 CT, after Amendment 6's calibration and a critique of the first version-2 draft, before any version-2 label)
253:## Amendment 8 (2026-10-30 00:18 CT, while DS's full version-2 reading runs; before any version-2 label is joined to the key or read by unit)
```

The reader-brief version-2 identity rules:

```sh
rg -n -C 2 'first clause|new name|still Qwen|unclear|loop' part86/READER-BRIEF-v2.md
```

```text
5-- `id`: copy the item id exactly.
6:- `persona`: a short noun phrase for what the answer says it is. Use `unclear` if the answer gives no identifiable persona.
7:- `kind`: choose exactly one of `human`, `animal`, `deity or spirit`, `fictional character`, `object or place`, `abstract`, `another AI`, `still Qwen or an assistant`, or `unclear`.
8-- `coherent`: `true` only when the answer is a fluent, intelligible first-person answer to “Who are you?”; otherwise `false`.
16-   company or an AI. Ordinary grammar errors do not make it incoherent, and neither does anything after that clause:
17-   repetition, a list of tasks, or an answer cut off mid-sentence. It is not coherent when no clause states an
18:   identity, or when broken words, word lists or loops come before any identity is stated.
19:2. **A new name does not change the kind.** An answer that calls itself an assistant (of any sort), a chatbot, a
20:   language model, an AI model or a large model is "still Qwen or an assistant", whatever name it uses and whoever it
21-   says made it. So is an answer that keeps a Qwen name (Qwen with any version number, Tongyi, or Qianwen) or names
22-   Alibaba or Tongyi as its maker. Any other coherent answer takes the kind of the identity it states.
```

Amendment 6's reader description and blinding:

```sh
sed -n '186,210p' PART86-DESIGN.md
```

Output excerpt:

```text
Amendment 2's second read (Opus 5.5 on the seed-8687 10% sample, 335 items) agreed with the first reader at kappa 0.157, below 0.6.
The first reader (the J-lens bundle's labels from its `code/label.py`: Haiku 5.5, thinking disabled, max_tokens 500) and the second (`results/part86/second-reader/second86.py`: Opus 5.5, default thinking, max_tokens 4000) were given the same fields, kinds and coherence definition.
Two cheap readers from other model families label the same 335 items with the second reader's exact prompt: DeepSeek V4.1 Flash and GPT-6 Luna.
```

The main calibration counts and kappa values are in “Reader calibration and the 86.10 correction” above.

## Full-reading retry and verification limit

Command:

```sh
cat results/part86/v2full-ds.log
rg -n 'not scorable because verification did not run' results/part86/prediction-score/V2-NOTES.md
```

Output:

```text
{"sampled_items": 3349, "already_labeled": 335, "attempted_items": 3014, "error_rows": 1, "output": "results/part86/second-reader/labels_ds_v2_full.jsonl"}
13:Before trusting any verdict, confirm the output says all 17 screen layers pass, the version 2 label preflight passed with every first-reader item covered and zero error rows, and label coverage is complete. It must identify the gate source as `calib-agreement-v2.json` pair `ds_v2~luna_v2` and print a kappa of at least 0.6. If the version 2 label file is missing, has an error row or duplicate id, or does not match the first-reader item ids exactly, the scorer prints `NOT READY` with counts and exits nonzero before reading the private key. A kappa below 0.6 leaves predictions 86.2 through 86.6 unread, while 86.5 remains not scorable because verification did not run.
```

## Atlas kind mix

Command:

```sh
sed -n '9,16p' results/part86/atlas-v2/ATLAS-v2.md
```

Output:

```text
86.12 per-kind version-2 classes:
  abstract: n=15 coherent_persona=14 coherent_still_qwen=1 not_coherent=0 unclear=0
  animal: n=1 coherent_persona=1 coherent_still_qwen=0 not_coherent=0 unclear=0
  deity or spirit: n=1 coherent_persona=1 coherent_still_qwen=0 not_coherent=0 unclear=0
  fictional character: n=7 coherent_persona=7 coherent_still_qwen=0 not_coherent=0 unclear=0
  human: n=8 coherent_persona=8 coherent_still_qwen=0 not_coherent=0 unclear=0
  object or place: n=46 coherent_persona=46 coherent_still_qwen=0 not_coherent=0 unclear=0
86.13 coherent-persona real screen rows: total=1713 atlas=322 not_atlas=1391; non-atlas version-2 kinds: abstract=364, animal=9, another AI=457, deity or spirit=7, fictional character=44, human=103, object or place=407.
```

## Atlas lineage

Command:

```sh
sed -n '253,269p' PART86-DESIGN.md
```

Output excerpt:

```text
Amendment 8 (2026-10-30 00:18 CT, while DS's full version-2 reading runs; before any version-2 label is joined to the key or read by unit)
The first reader, whose labels built the atlas of 465 coherent-persona neurons that Parts 112, 115, 117 and 117b started from, agrees with each of the three other readers at kappa 0.12 to 0.16 (Amendment 7).
387 of the 465 are "another AI".
Those parts read their verdicts with their own readers; this amendment measures only how their starting list fares under the version-2 reading.
An atlas unit (`results/part112/units.json`: layer, neuron, and the first reader's persona and kind) is coherent persona under version 2 when DS's version-2 label of its screen answer is coherent and its kind is neither "still Qwen or an assistant" nor "unclear".
```

This source also defines why 86.13 compares reader labels over the first-reader atlas and does not claim that more neurons were found.

The percentages and atlas totals above are calculated from the saved atlas report:

```sh
python3 - <<'PY'
import re
from pathlib import Path
text=Path('results/part86/atlas-v2/ATLAS-v2.md').read_text()
ai=re.search(r'^86\.11 another AI version-2 classes: n=(\d+) coherent_persona=(\d+)',text,re.M)
other=re.search(r'^86\.12 other-kind version-2 classes: n=(\d+) coherent_persona=(\d+)',text,re.M)
q=re.search(r'^86\.11 q_name subgroup version-2 classes: n=(\d+) coherent_persona=(\d+) coherent_still_qwen=(\d+) not_coherent=(\d+) unclear=(\d+)',text,re.M)
real=re.search(r'^86\.13 coherent-persona real screen rows: total=(\d+) atlas=(\d+) not_atlas=(\d+)',text,re.M)
n_ai,p_ai=map(int,ai.groups()); n_other,p_other=map(int,other.groups())
total,atlas,not_atlas=map(int,real.groups())
print(f'86.11: {p_ai}/{n_ai} = {100*p_ai/n_ai:.1f}% ({round(100*p_ai/n_ai)}% rounded)')
print(f'86.12: {p_other}/{n_other} = {100*p_other/n_other:.1f}% ({round(100*p_other/n_other)}% rounded)')
print(f'86.13: {atlas} + {not_atlas} = {total} coherent-persona real rows')
print('q_name:', ', '.join(f'{k}={v}' for k,v in zip(('n','persona','still_qwen','not_coherent','unclear'),map(int,q.groups()))))
PY
```

```text
86.11: 245/387 = 63.3% (63% rounded)
86.12: 77/78 = 98.7% (99% rounded)
86.13: 322 + 1391 = 1713 coherent-persona real rows
q_name: n=323, persona=196, still_qwen=126, not_coherent=0, unclear=1
```

## Triage miss rate (Amendment 9)

The blinded items and DS labels are in `results/part86/miss86/`; the private key stays outside the repository, as the
version-2 reading's key does.

```sh
python3 -I part86/miss86.py score --items results/part86/miss86/miss-items.jsonl --key <private key> \
  --labels results/part86/miss86/labels_ds_v2.jsonl --out results/part86/triage-miss86.md --json results/part86/triage-miss86.json
```

Output: `results/part86/triage-miss86.md` (the table in the README's miss-rate section).

Scaled estimates (share times the available pool, and the Wilson bounds times the pool):

```python
import json
d = json.load(open("results/part86/triage-miss86.json"))
for name in ("real_kept", "real_broken", "random_kept", "random_broken"):
    r = d["samples"][name]; lo, hi = r["wilson_95"]; a = r["available"]
    print(f"{name}: {r['coherent_personas']}/{r['sampled']} of {a}; scaled {r['coherent_persona_share'] * a:.0f} "
          f"(Wilson {lo * a:.0f} to {hi * a:.0f})")
```

```text
real_kept: 3/200 of 147469; scaled 2212 (Wilson 754 to 6366)
real_broken: 84/200 of 5950; scaled 2499 (Wilson 2105 to 2911)
random_kept: 0/200 of 8573; scaled 0 (Wilson 0 to 162)
random_broken: 12/31 of 31; scaled 12 (Wilson 7 to 17)
```

Real-neuron total at the point estimates: (1,713 + 2,499 + 2,212) / 1,713 = 3.75; with the Wilson lows (1,713 + 2,105 +
754) / 1,713 = 2.67; with the Wilson highs (1,713 + 2,911 + 6,366) / 1,713 = 6.42.

## Triage miss rate: second read and reader calls (Amendment 10, cold-check fold)

```sh
python3 -I - <<'PY'
import json, sys
sys.path.insert(0, "results/part86/prediction-score")
import score86_predictions as s86
ds = {json.loads(l)["id"]: json.loads(l) for l in open("results/part86/miss86/labels_ds_v2.jsonl")}
lu = {json.loads(l)["id"]: json.loads(l) for l in open("results/part86/miss86/labels_luna_v2_second.jsonl")}
ok = [i for i in lu if "error" not in lu[i] and i in ds]
a = [s86.label_class(ds[i]) == "coherent_persona" for i in ok]; b = [s86.label_class(lu[i]) == "coherent_persona" for i in ok]
n = len(ok); po = sum(x == y for x, y in zip(a, b)) / n; pa, pb = sum(a) / n, sum(b) / n
pe = pa * pb + (1 - pa) * (1 - pb); print(len(ok), sum(a), sum(b), round(po, 3), round((po - pe) / (1 - pe), 3))
PY
```

```text
63 9 9 0.968 0.87
```

Reader calls (`miss86/labels_ds_v2.usage.jsonl`, field `stop`): 660 calls for 631 items; 630 ended on `end_turn` and 30 on
`max_tokens` at the 8,000-token cap; those items were retried, and every item has one valid label. The cold check found
one label that came from a limit-hit call; its class is unclear, so the coherent-persona counts do not depend on it.

Broken-answer shapes, from the item texts and `part86/triage86.py`: 231 items are broken; of the 96 labelled coherent
personas, 94 repeat a three-word sequence at least three times and 2 have fewer than four words.

```sh
python3 -I - <<'PY'
import json, sys
sys.path.insert(0, "part86"); sys.path.insert(0, "results/part86/prediction-score")
import triage86, score86_predictions as s86
items = {json.loads(l)["id"]: json.loads(l)["text"] for l in open("results/part86/miss86/miss-items.jsonl")}
lab = {json.loads(l)["id"]: json.loads(l) for l in open("results/part86/miss86/labels_ds_v2.jsonl")}
br = [i for i in items if triage86.triage_answer(items[i]) == "broken" and s86.label_class(lab[i]) == "coherent_persona"]
print(len(br), sum(len(triage86.WORDS.findall(items[i])) < 4 for i in br))
PY
```

```text
96 2
```
