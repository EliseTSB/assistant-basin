#!/usr/bin/env python3
# sysanalyze.py — scores sysprompt.py's output against predictions S1-S6 (Claude, 2026-10-15).
#
# Directions are means over users of paired differences in the residual stream at the reply's first position, per layer.
#   S1 shift of each system prompt vs none, as a share of the none-state's norm.
#   S2 HHH cluster: mean cosine among helpful/harmless/honest (each minus "assistant") minus their mean cosine with the
#      control adjectives curious/tall. Also a split-half ceiling: cosine of a direction from even users with the same
#      direction from odd users (how similar a direction can be to itself given the noise).
#   S3 HHH on different nouns: cos(hhh - assistant, person_hhh - person), cos(hhh - assistant, pirate_hhh - pirate).
#   S4 identity: cos(hhh - assistant, claude - olmo), cos(hhh - assistant, chatgpt - olmo), cos(hhh - assistant, assistant - none).
#   S5 neurons: per-neuron shift (hhh - assistant) / sd over users; the percentile of part 2's top-50 identity neurons
#      by |shift| among all neurons.
#   S6 behaviour: each generation labelled by the first name it mentions.
#   After-name tokens: per layer, ||OLMo - Claude||, ||OLMo - ChatGPT||, ||Claude - ChatGPT|| over the tokens after the
#      name, each divided by the mean distance between different frames' OLMo rows (a content scale).
# Usage: python sysanalyze.py out/sysprompt-<slug>.npz [--identity out/identity-instruct-chat.json] [--json out.json]
import argparse, json, re
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('npz'); ap.add_argument('--identity'); ap.add_argument('--json')
a = ap.parse_args()
z = np.load(a.npz); meta = json.load(open(a.npz[:-4] + '.json'))
L, F = int(z['layers']), int(z['neurons'])
R = lambda k: z[f'resid_{k}'].astype(np.float32)  # [U, L+1, D]
have = {k for k, _ in meta['conds']}

def cos(x, y):  # per layer, x,y [L+1, D]
    return (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-9)
def direction(a_, b_, rows=None):
    d = R(a_) - R(b_)
    return (d if rows is None else d[rows]).mean(0)
U = R('none').shape[0]; even, odd = np.arange(0, U, 2), np.arange(1, U, 2)
out = {'npz': a.npz, 'layers': L, 'users': U}

out['S1_shift_vs_none'] = {k: (np.linalg.norm(R(k) - R('none'), axis=-1).mean(0) / np.linalg.norm(R('none'), axis=-1).mean(0)).round(4).tolist()
                           for k in have if k != 'none'}
hhh_words, ctrl = ['helpful', 'harmless', 'honest'], ['curious', 'tall']
dirs = {k: direction(k, 'assistant') for k in hhh_words + ctrl + ['hhh'] if k in have}
if all(k in dirs for k in hhh_words + ctrl):
    within = np.mean([cos(dirs[x], dirs[y]) for i, x in enumerate(hhh_words) for y in hhh_words[i + 1:]], 0)
    across = np.mean([cos(dirs[x], dirs[c]) for x in hhh_words for c in ctrl], 0)
    ctrl_pair = cos(dirs['curious'], dirs['tall'])
    ceiling = np.mean([cos(direction(k, 'assistant', even), direction(k, 'assistant', odd)) for k in hhh_words], 0)
    out['S2'] = {'within_hhh': within.round(4).tolist(), 'hhh_vs_control': across.round(4).tolist(),
                 'curious_vs_tall': ctrl_pair.round(4).tolist(), 'cluster_score': (within - across).round(4).tolist(),
                 'split_half_ceiling': ceiling.round(4).tolist(),
                 'pairs': {f'{x}~{y}': cos(dirs[x], dirs[y]).round(4).tolist() for x in hhh_words + ctrl for y in hhh_words + ctrl if x < y}}
if {'hhh', 'person', 'person_hhh', 'pirate', 'pirate_hhh'} <= have:
    h = dirs['hhh']
    out['S3'] = {'hhh_assistant_vs_person': cos(h, direction('person_hhh', 'person')).round(4).tolist(),
                 'hhh_assistant_vs_pirate': cos(h, direction('pirate_hhh', 'pirate')).round(4).tolist()}
if {'hhh', 'claude', 'olmo', 'chatgpt', 'none'} <= have:
    h = dirs['hhh']; co, go = direction('claude', 'olmo'), direction('chatgpt', 'olmo')
    out['S4'] = {'hhh_vs_claude_minus_olmo': cos(h, co).round(4).tolist(), 'hhh_vs_chatgpt_minus_olmo': cos(h, go).round(4).tolist(),
                 'hhh_vs_assistant_minus_none': cos(h, direction('assistant', 'none')).round(4).tolist(),
                 'claude_minus_olmo_vs_chatgpt_minus_olmo': cos(co, go).round(4).tolist(),
                 'claude_minus_olmo_vs_assistant_minus_none': cos(co, direction('assistant', 'none')).round(4).tolist()}
if a.identity and 'hhh' in have:
    ident = json.load(open(a.identity))
    M = lambda k: z[f'mlp_{k}'].astype(np.float32).reshape(U, -1)
    diff = M('hhh') - M('assistant')
    shift = np.abs(diff.mean(0) / (M('assistant').std(0, ddof=1) + 1e-6))
    # part 2's top-50, rebuilt the way analyze.py ranks them, is not stored; use its top10 names plus the json's list if present
    names = [t[0] for t in ident.get('top50', ident['top10'])]
    idx = np.array([int(n[1:].split('.')[0]) * F + int(n.split('.')[1]) for n in names])
    pct = (shift[None, :] < shift[idx][:, None]).mean(1) * 100
    out['S5'] = {'identity_neurons_used': len(idx), 'median_percentile': float(np.median(pct)),
                 'share_in_middle_80': float(np.mean((pct >= 10) & (pct <= 90)))}
lab = [('OLMo', r'\b(olmo|ai2|allen)'), ('Claude', r'\b(claude|anthropic)'), ('ChatGPT', r'\b(chatgpt|openai|gpt-?\d)')]
def label(t):
    hits = [(m.start(), n) for n, p in lab for m in [re.search(p, t, re.I)] if m]
    return min(hits)[1] if hits else 'other'
S6 = {}
for k, qs in meta.get('generations', {}).items():
    ls = [label(t) for g in qs.values() for t in g]
    S6[k] = {n: round(ls.count(n) / len(ls), 3) for n in ['OLMo', 'Claude', 'ChatGPT', 'other']} | {'n': len(ls)}
out['S6'] = S6
if 'aftername_resid_olmo' in z.files:
    A, C, G = (z[f'aftername_resid_{k}'].astype(np.float32) for k in ('olmo', 'claude', 'chatgpt'))
    n = A.shape[0]
    scale = np.mean([np.linalg.norm(A[i] - A[j], axis=-1) for i in range(n) for j in range(n) if i != j], 0)
    nd = lambda x, y: np.linalg.norm(x - y, axis=-1).mean(0) / scale
    out['aftername'] = {'frames': n, 'olmo_claude': nd(A, C).round(4).tolist(), 'olmo_chatgpt': nd(A, G).round(4).tolist(),
                        'claude_chatgpt': nd(C, G).round(4).tolist()}
print(json.dumps({k: v for k, v in out.items() if k not in ('S1_shift_vs_none',)}, indent=1)[:6000])
if a.json: json.dump(out, open(a.json, 'w'), indent=1)
