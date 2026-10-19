"""score_part19.py: score Part 19 (19.1-19.6) from circuit.py's JSON, recomputing every number from rows.

Self-checks: N2742's two ranks are recomputed from screen.json's full estimate arrays and must equal the tool's fields;
the significant-edge list is recomputed from edges.json's raw effects with the 19.6 amendment's null (random sources at
layers below the target only, at least 10 of them, self-edges excluded) and must equal the tool's list.
Usage: python3 -I score_part19.py <results/part19 dir>
"""
import json, re, statistics, sys

R = sys.argv[1]
L = lambda n: json.load(open(f'{R}/{n}.json', encoding='utf-8'))
screen, verify, group, edges = L('screen'), L('verify'), L('group'), L('edges')
key = lambda d: f"{d['layer']}:{d['index']}"
out = {}

# 19.1 and 19.4 from verify's rows
top = [r for r in verify['results'] if r['set'] == 'top']
rnd = [r for r in verify['results'] if r['set'] != 'top']
assert len(top) == 50 and len(rnd) == 50
mt = statistics.median(r['pooled_abs_delta_M'] for r in top)
mr = statistics.median(r['pooled_abs_delta_M'] for r in rnd)
out['19.1'] = {'top50_median': mt, 'random50_median': mr, 'ratio': mt / mr, 'held': mt >= 5 * mr}
top20 = sorted(top, key=lambda r: -r['pooled_abs_delta_M'])[:20]
med_ratio = statistics.median(r['ratio'] for r in top20)
out['19.4'] = {'top20_median_ratio': med_ratio, 'held': med_ratio >= 3}

# 19.2: layers of the top 50 by pooled zeroing estimate (screen ranking, = verify's top_neurons)
t50 = screen['top_zeroing'][:50]
assert [key(x) for x in t50] == [key(x) for x in verify['top_neurons']], 'verify top != screen top 50'
n_late = sum(22 <= x['layer'] <= 31 for x in t50)
out['19.2'] = {'in_layers_22_31': n_late, 'held': n_late >= 30}

# 19.3: group answers, first sentence
pat = re.compile(r'qwen|tongyi|alibaba|通义|千问|阿里', re.I)
sets = {s['set']: s for s in group['sets']}
fs = lambda s, q: s['answers'][q]['first_sentence']
top_removed = {q: not pat.search(fs(sets['top'], q)) for q in ('Who made you?', 'What is your name?')}
# "every identity answer that had them": the unclamped greedy answer's first sentence, cut by circuit.py's own rule
base = json.load(open(f'{R}/answers.json', encoding='utf-8'))['answers']
cut = lambda t: re.split(r'(?<=[.!?])\s+', t.strip(), 1)[0]
had = [q for q in sets['top']['answers'] if pat.search(cut(base[q]['answer']))]
rnd_keep = {}
for name, s in sets.items():
    if name == 'top':
        continue
    rnd_keep[name] = {q: bool(pat.search(s['answers'][q]['first_sentence'])) for q in had}
out['19.3'] = {'unclamped_answers_naming_identity': had, 'top_removed_first_sentence': top_removed,
               'random_sets_kept_every_answer': {n: all(v.values()) for n, v in rnd_keep.items()},
               'random_sets_lost': {n: [q for q, k in v.items() if not k] for n, v in rnd_keep.items()},
               'top_first_sentences': {q: fs(sets['top'], q) for q in sets['top']['answers']},
               'held': all(top_removed.values()) and all(all(v.values()) for v in rnd_keep.values())}

# 19.5: recompute N2742's ranks from the full arrays
def rank(arr, layer, idx, absval):
    flat = [(abs(v) if absval else v, l, i) for l, row in enumerate(arr) for i, v in enumerate(row)]
    target = abs(arr[layer][idx]) if absval else arr[layer][idx]
    return 1 + sum(1 for v, _, _ in flat if v > target)
zr = rank(screen['zeroing_estimate'], 22, 2742, True)
pr_abs = rank(screen['push_estimate'], 22, 2742, True)
pr_signed = rank(screen['push_estimate'], 22, 2742, False)
tool = screen['N2742']
out['19.5'] = {'zeroing_rank': zr, 'push_rank_abs': pr_abs, 'push_rank_signed': pr_signed, 'tool': tool,
               'held': zr > 500 and tool['push_rank'] <= 500}
assert zr == tool['zeroing_rank'], ('zeroing rank mismatch', zr, tool)
assert tool['push_rank'] in (pr_abs, pr_signed), ('push rank mismatch', pr_abs, pr_signed, tool)

# 19.6: recompute significance from raw effects
nodes = [key(n) for n in edges['nodes']]
layer = {key(n): n['layer'] for n in edges['nodes']}
node_src = [e for e in edges['effects'] if not e['null']]
null_src = [e for e in edges['effects'] if e['null']]
sig = set(); testable = []
for j in nodes:
    nulls = [abs(e['effects'][j]) for e in null_src if e['source']['layer'] < layer[j] and j in e['effects']]
    if len(nulls) < 10:
        continue
    testable.append(j)
    thr = sorted(nulls)[min(len(nulls) - 1, int(0.99 * len(nulls)))]
    thr_np = None
    for e in node_src:
        i = key(e['source'])
        if i != j and j in e['effects'] and abs(e['effects'][j]) > edges['null_99th_abs_by_target'][j]:
            sig.add((i, j))
tool_sig = {(key(s['source']), key(s['target'])) for s in edges['significant_edges']}
receivers = sorted({j for _, j in tool_sig})
out['19.6'] = {'testable_nodes': len(testable), 'tool_testable': len(edges['null_99th_abs_by_target']),
               'significant_edges': len(tool_sig), 'recomputed_edges_at_tool_threshold': len(sig),
               'nodes_receiving': len(receivers), 'held': len(receivers) >= 6}
assert set(testable) == set(edges['null_99th_abs_by_target']), 'testable set mismatch'
assert sig == tool_sig, ('edge set mismatch', len(sig), len(tool_sig))
# the tool's threshold must be the 99th percentile of this target's admissible nulls (numpy-style linear interpolation)
import math
for j in testable:
    nulls = sorted(abs(e['effects'][j]) for e in null_src if e['source']['layer'] < layer[j])
    pos = 0.99 * (len(nulls) - 1); lo = math.floor(pos); hi = math.ceil(pos)
    q = nulls[lo] + (nulls[hi] - nulls[lo]) * (pos - lo)
    assert abs(q - edges['null_99th_abs_by_target'][j]) < 1e-9, ('threshold mismatch', j, q)
print(json.dumps(out, indent=1, ensure_ascii=False))
print('SELF-CHECK PASS: ranks, testable set, thresholds and edge set recomputed from rows')
