#!/usr/bin/env python3
"""Additional row-level recomputations for the README front-page audit.

Run with: python3 -I recompute_extra.py
Standard library only; reads stored results, never loads a model.
"""
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path('/home/elisepc/green/worktrees/ab-part67')
OUT = Path('/home/elisepc/green/Journal/notes/reports/frontpage-audit-2026-10-27/recomputed_extra.json')

def read_json(rel):
    return json.loads((ROOT / rel).read_text(encoding='utf-8'))

def read_jsonl(rel):
    return [json.loads(line) for line in (ROOT / rel).read_text(encoding='utf-8').splitlines() if line.strip()]

def accuracy(rel):
    x = read_json(rel)
    rows = x['passages']
    return {'n': len(rows), 'correct': sum(r['prediction'] == r['label'] for r in rows)}

def exact_binomial_two_sided(k, n):
    probs = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    observed = probs[k]
    return sum(p for p in probs if p <= observed + 1e-15)

data = {}

# Part 8: count direct maker/name tokens in all saved raw no-system answers.
p8 = read_json('results/part8-qwen-coder/sysgen8-SWE-bench__SWE-agent-LM-7B.json')
p8_answers = [s for values in p8['generations']['raw_nosys'].values() for s in values]
data['part8_raw_nosys'] = {
    'n': len(p8_answers),
    'claude': sum(bool(re.search(r'Claude', s, re.I)) for s in p8_answers),
    'qwen': sum(bool(re.search(r'Qwen', s, re.I)) for s in p8_answers),
}

# Part 11 / SmolLM3: count the self-name in saved raw no-system answers.
data['part11_smol_name'] = {}
for rel, label in [('results/ablate-smol.json', 'top20'),
                   ('results/ablate-smol-nameattr.json', 'nameattr28')]:
    x = read_json(rel)
    for condition, values in x['conditions'].items():
        texts = values['by_form']['raw_nosys']['texts']
        data['part11_smol_name'][f'{label}/{condition}'] = {
            'n': len(texts),
            'self_name': sum(bool(re.search(r'\bSmolLM\b', text, re.I)) for text in texts),
        }

# Part 9: score final answer text (after </think>) rather than the hidden reasoning.
data['part9_raw_nosys'] = {}
for rel, label in [('results/part9-think/sysgen9-allenai__Olmo-3-7B-Think.json','Think'), ('results/part9-think/sysgen9-allenai__Olmo-3-7B-Think-DPO.json','Think-DPO')]:
    x=read_json(rel); texts=[t for arr in x['generations']['raw_nosys'].values() for t in arr]
    finals=[t.split('</think>',1)[1] if '</think>' in t else '' for t in texts]
    data['part9_raw_nosys'][label]={'n':sum(bool(t) for t in finals),'deepseek':sum(bool(re.search(r'DeepSeek',t,re.I)) for t in finals),'qwen_family':sum(bool(re.search(r'Qwen|Alibaba|Tongyi',t,re.I)) for t in finals),'olmo':sum(bool(re.search(r'OLMo',t,re.I)) for t in finals),'deepseek_qwen_overlap':sum(bool(re.search(r'DeepSeek',t,re.I) and re.search(r'Qwen|Alibaba|Tongyi',t,re.I)) for t in finals),'unfinished':sum(not bool(t) for t in finals)}
    if label=='Think-DPO':
        none=[t for arr in x['generations']['none'].values() for t in arr]
        finals_none=[t.split('</think>',1)[1] if '</think>' in t else '' for t in none]
        data['part9_thinkdpo_olmo_template']={'n':sum(bool(t) for t in finals_none),'deepseek':sum(bool(re.search(r'DeepSeek',t,re.I)) for t in finals_none),'qwen_family':sum(bool(re.search(r'Qwen|Alibaba|Tongyi',t,re.I)) for t in finals_none)}

# Part 25: answer counts and search results from answer-bearing artifacts.
p25=read_json('results/part25/read.json')['rows']
p25_who=[r for r in p25 if r.get('question')=='Who created you?']
final25=[r for r in p25_who if r.get('condition')=='search:google-removal:pruned_final_set']
rand25=[r for r in p25_who if r.get('condition')=='control:random_zero:google-removal']
p25search=read_json('results/part25/search-google-removal.json')
pruned25=p25search['pruned']; drops25=pruned25['single_drop_audit']['checks']
final_text25=' '.join(r.get('text','') for r in final25)
data['part25_google_removal']={'selected_count':len(pruned25['neurons']),'selected_layers':sorted({n['layer'] for n in pruned25['neurons']}),'single_drops':len(drops25),'single_drop_successes':sum(c['success_without'] for c in drops25),'final_set_rows':len(final25),'final_set_google_mentions':sum(bool(re.search(r'Google|Gemini|DeepMind',r.get('text',''),re.I)) for r in final25),'final_set_gemini_1_5_pro':bool(re.search(r'Gemini\s+1\.5\s+Pro',final_text25,re.I)),'random_sets':len(rand25),'random_google_mentions':sum(bool(re.search(r'Google|Gemini|DeepMind',r.get('text',''),re.I)) for r in rand25),'random_success':sum(r.get('maker')=='Google/DeepMind' for r in rand25)}

# Parts 17 / 17b: score passage rows, pair accuracy by passage ID, and replay rows.
files = {
    'self_G1': 'results/part17/eval-self-G1.json',
    'input_G1': 'results/part17/eval-input-G1.json',
    'self_GL': 'results/part17/eval-self-GL.json',
    'input_G1_p17b': 'results/part17b/eval-input-G1-p17b.json',
    'self_G1_p17b': 'results/part17b/eval-self-G1-p17b.json',
    'self_GL_p17b': 'results/part17b/eval-self-GL-p17b.json',
}
data['part17_scores'] = {name: accuracy(rel) for name, rel in files.items()}
a = read_json(files['self_G1'])['passages']
b = read_json(files['input_G1'])['passages']
by_id = {r['passage_id']: r for r in b}
discordant_self = discordant_input = 0
for r in a:
    other = by_id[r['passage_id']]
    c1, c2 = r['prediction'] == r['label'], other['prediction'] == other['label']
    discordant_self += c1 and not c2
    discordant_input += c2 and not c1
n_discordant = discordant_self + discordant_input
data['part17_scores']['paired_accuracy'] = {
    'paired_n': len(a), 'self_only_correct': discordant_self,
    'input_only_correct': discordant_input,
    'mcnemar_exact_two_sided_p': exact_binomial_two_sided(discordant_self, n_discordant),
}
t = read_json('results/part17b/transplant-G1-p17b.json')['passages']
data['part17b_transplant'] = {
    'n': len(t), 'higher_at_q80': sum(bool(r['higher_at_q80']) for r in t),
    'mean_q80_minus_q20': sum(r['p_higher_q80'] - r['p_higher_q20'] for r in t) / len(t),
}

# Part 74: recompute the direct and indirect effects from per-question gap rows.
p74=read_json('results/part74/records.json')['records']; g74=defaultdict(list)
for r in p74: g74[r['id']].append(r['g'])
means74={k:statistics.mean(v) for k,v in g74.items()}
te74=means74['B']-means74['A']; de74=means74['C']-means74['A']; ie74=means74['E']-means74['A']
data['part74_raw_effects']={'questions':len(g74['A']),'total':te74,'direct':de74,'indirect':ie74,'direct_share_of_total':de74/te74,'indirect_share_of_total':ie74/te74,'remainder_share':1-(de74+ie74)/te74}

# Part 75: count Google-family mentions from raw held-out answer rows.
p75=read_jsonl('results/part75/answers.jsonl'); data['part75_google']={}
for c in sorted({r.get('condition') for r in p75}):
    xs=[r for r in p75 if r.get('condition')==c and r.get('question_kind') in ('direct','indirect')]
    if xs: data['part75_google'][c]={'n':len(xs),'google':sum(bool(re.search(r'Google|Gemini|DeepMind',r.get('answer',''),re.I)) for r in xs)}

# Part 76: rank the target in the 9,216 per-unit first-order values.
p76=read_json('results/part76/pie_hat_and_ranks.json'); target76=3325; estimates76=p76['pie_hat']
rank76=1+sum(x<estimates76[target76] for x in estimates76)
ordered76=sorted(enumerate(estimates76),key=lambda z:z[1])
data['part76_first_order']={'units':len(estimates76),'target_index':target76,'target_estimate':estimates76[target76],'target_rank':rank76,'next_estimate':ordered76[1][1]}

# Part 77: raw baseline/L24/restore answer joins.
p77=read_jsonl('results/part77/answers.jsonl'); pat_google=re.compile(r'Google|Gemini|DeepMind',re.I)
held77={(r['condition'],r['question_index'],r['sample']):r for r in p77 if r.get('question_kind') in ('direct','indirect')}
lost77=[(held77[('c0_makers',q,s)],held77[('c1_l24',q,s)],held77[('c2_l24_restore',q,s)]) for c,q,s in held77 if c=='c0_makers' and pat_google.search(held77[('c0_makers',q,s)].get('answer','')) and not pat_google.search(held77[('c1_l24',q,s)].get('answer',''))]
new77=sum(not pat_google.search(held77[('c1_l24',q,s)].get('answer','')) and bool(pat_google.search(held77[('c2_l24_restore',q,s)].get('answer',''))) for c,q,s in held77 if c=='c1_l24')
data['part77_restore_pairs']={'baseline_google_lost_by_l24':len(lost77),'among_those_google_after_restore':sum(bool(pat_google.search(t[2].get('answer',''))) for t in lost77),'paired_c1_to_c2_new_google':new77}

# Part 27 base comparison: recalculate maker-family counts from saved completions.
p27 = read_json('cloud/vast-part27-2026-10-20/base.json')['maker_completions']['conditions']
maker_patterns = {
    'OpenAI': re.compile(r'OpenAI|ChatGPT|GPT-', re.I),
    'Alibaba': re.compile(r'Alibaba|Qwen|Tongyi', re.I),
    'Google': re.compile(r'Google|Gemini|DeepMind', re.I),
}
data['part27_base'] = {}
for condition in ('baseline', 'set_zero', 'controls'):
    rows = p27[condition]
    if condition == 'controls':
        data['part27_base'][condition] = {
            'n': len(rows),
            'n_per_control': {str(i): 1 for i in range(len(rows))},
            'openai': sum(bool(maker_patterns['OpenAI'].search(r['completion'])) for r in rows),
            'alibaba': sum(bool(maker_patterns['Alibaba'].search(r['completion'])) for r in rows),
            'google': sum(bool(maker_patterns['Google'].search(r['completion'])) for r in rows),
        }
    else:
        data['part27_base'][condition] = {
            'n': len(rows),
            **{name.lower(): sum(bool(pattern.search(r['completion'])) for r in rows)
               for name, pattern in maker_patterns.items()},
        }

# Parts 49 / 49b: replay the published lexical scoring from saved answer rows.
score49 = read_json('results/part49-xpu/push-scores.json')
wizard_terms = score49['open_prompt']['topic_terms']
wizard_re = re.compile(r'\b(?:' + '|'.join(re.escape(t) for t in wizard_terms) + r')\w*', re.I | re.UNICODE)
def topic_hits(rel):
    rows = [r for r in read_jsonl(rel) if r.get('record_type') == 'result']
    by_arm = defaultdict(list)
    for r in rows: by_arm[r['arm']].append(r)
    return {arm: {'n': len(xs), 'hits': sum(bool(wizard_re.search(r.get('answer', ''))) for r in xs)}
            for arm, xs in sorted(by_arm.items())}
data['part49_open_prompt_hits'] = topic_hits('results/part49-xpu/push.jsonl')
data['part49_negative_prompt_hits'] = topic_hits('results/part49-xpu/negative-push.jsonl')

score49b = read_json('results/part49b-xpu/push49b-scores.json')
magic_terms = score49b['measure']['terms']
magic_patterns = {term: re.compile(r'\b' + re.escape(term) + r'\w*', re.I | re.UNICODE) for term in magic_terms}
rows49b = [r for r in read_jsonl('results/part49b-xpu/push49b.jsonl') if r.get('record_type') == 'result']
by_arm_prompt = defaultdict(lambda: defaultdict(list))
for row in rows49b:
    answer, prompt = row.get('answer', ''), row.get('prompt', '')
    excluded = {term for term, pattern in magic_patterns.items() if pattern.search(prompt)}
    matches = sum(len(pattern.findall(answer)) for term, pattern in magic_patterns.items() if term not in excluded)
    words = len(re.findall(r'\b\w+\b', answer, re.UNICODE))
    by_arm_prompt[row['arm']][row['prompt_id']].append(matches * 1000.0 / words if words else 0.0)
means49b = {arm: {pid: statistics.mean(vals) for pid, vals in prompts.items()}
            for arm, prompts in by_arm_prompt.items()}
grand49b = {arm: statistics.mean(prompt_means.values()) for arm, prompt_means in means49b.items()}
data['part49b_raw'] = {
    'arm_n': {arm: sum(len(v) for v in prompts.values()) for arm, prompts in by_arm_prompt.items()},
    'mean_per_1000': grand49b,
    'downward_ratio_to_no_clamp': grand49b['L22:1610'] / grand49b['no_clamp'],
    'prompt_count': len(means49b['no_clamp']),
}

# Part 33: aggregate unit-position delta sums and denominators from each passage row.
p33 = read_json('results/part33-r2/passage-effects.json')
unit_ids = sorted(p33['matched_controls'] + [2742])
aggregates = {}
for uid in unit_ids:
    at_sum = at_n = win_sum = win_n = 0
    for passage in p33['rows'].values():
        x = passage[str(uid)]
        at_sum += x['at']['delta_sum']; at_n += x['at']['firing_loss_tokens']
        win_sum += x['window']['delta_sum']; win_n += x['window']['window_loss_tokens']
    aggregates[str(uid)] = {
        'at_tokens': at_n, 'at_mean': at_sum / at_n if at_n else None,
        'window_tokens': win_n, 'window_mean': win_sum / win_n if win_n else None,
    }
control_means = sorted(aggregates[str(uid)]['window_mean'] for uid in p33['matched_controls'] if aggregates[str(uid)]['window_mean'] is not None)
def percentile_linear(xs, p):
    xs = sorted(xs); pos = (len(xs) - 1) * p; lo = int(pos); hi = math.ceil(pos)
    return xs[lo] if lo == hi else xs[lo] * (hi - pos) + xs[hi] * (pos - lo)
data['part33'] = {
    'target_2742': aggregates['2742'],
    'matched_control_n': len(control_means),
    'matched_control_window_mean': statistics.mean(control_means),
    'matched_control_window_p95_linear': percentile_linear(control_means, 0.95),
    'passages': len(p33['rows']),
    'comprehension_items': len(read_json('results/part33-r2/results.json')['comprehension']['items']),
    'comprehension_correct': read_json('results/part33-r2/results.json')['comprehension']['conditional_zero_correct'],
    'dropped_question_passages': len(read_json('results/part33-r2/natural-firing.json')['natural_firing']['dropped_question_passages']),
    'target_firing_passages': sum(row[str(2742)]['at']['firing_loss_tokens'] > 0 for row in p33['rows'].values()),
}

# Parts 80b / 80c: pair assistant and user renderings for each body/order.
def role_gap(rel, body_key):
    rows = read_jsonl(rel)
    pairs = defaultdict(dict)
    for r in rows:
        key = (r.get(body_key), r['body_id'], r['order'])
        pairs[key][r['rendering']] = r
    grouped = defaultdict(list)
    for (body_set, _body, order), pair in pairs.items():
        if 'A' in pair and 'U' in pair:
            pa = pair['A']['role_probs']['you']
            pu = pair['U']['role_probs']['you']
            grouped[(body_set, order)].append(pa - pu)
    return {f'{body_set}/order{order}': statistics.mean(vals)
            for (body_set, order), vals in sorted(grouped.items(), key=lambda x: str(x[0]))}

data['part80b_gaps'] = role_gap('results/part80b/qwen3.5-4b/records.jsonl', 'body_set')
# Part 80c data stores the option text condition as its body_set.
data['part80c_gaps'] = role_gap('results/part80c/qwen3.5-4b/records.jsonl', 'body_set')

# Parts 82 / 83: recreate cyclic-average gaps and target/random comparisons from per-body rows.
def cyclic_gaps(rel, has_option_set):
    rows = read_jsonl(rel)
    pairs = defaultdict(dict)
    for r in rows:
        key = (r.get('condition'), r.get('option_set'), r.get('body_id'), r['order'])
        pairs[key][r['rendering']] = r
    per_condition_order = defaultdict(list)
    for (condition, option_set, _body, order), pair in pairs.items():
        if 'A' in pair and 'U' in pair:
            per_condition_order[(condition, option_set, order)].append(pair['A']['p_you'] - pair['U']['p_you'])
    out = defaultdict(dict)
    for (condition, option_set, order), vals in per_condition_order.items():
        out[(condition, option_set)][order] = statistics.mean(vals)
    return {f'{c}/{o}': {'orders': {str(k): v for k, v in sorted(orders.items())},
                         'cyclic_gap': statistics.mean(orders.values())}
            for (c, o), orders in sorted(out.items(), key=lambda x: str(x[0]))}

data['part82_gaps'] = cyclic_gaps('results/part82/main/records.jsonl', True)
data['part83_gaps'] = cyclic_gaps('results/part83/main/records.jsonl', False)
singles = data['part83_gaps']
base = singles['h0/None']['cyclic_gap']
data['part83_summary'] = {
    'h0_gap': base,
    'l11h5_gap': singles['L11h5/None']['cyclic_gap'],
    'l11h5_reduction': base - singles['L11h5/None']['cyclic_gap'],
    'single_head_count': sum(k.startswith(('L11h', 'L15h')) for k in singles),
    'single_heads_abs_change_over_005': sum(abs(v['cyclic_gap'] - base) > 0.05 for k, v in singles.items() if k.startswith(('L11h', 'L15h'))),
}

# Verify the quoted alphanumeric maker string in Part 78 from raw answers.
p78_rows = read_jsonl('results/part78/answers.jsonl')
data['part78_quoted_project'] = {
    'xe890_project_mentions_in_c5': sum(bool(re.search(r'XE[-–]?890\s+Project', r.get('answer', ''), re.I))
                                             for r in p78_rows if r.get('condition') == 'c5_makers_alien20'),
}

OUT.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n', encoding='utf-8')
print(json.dumps(data, indent=2, sort_keys=True))
print('WROTE', OUT)
