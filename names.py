#!/usr/bin/env python3
# names.py — where OLMo names itself (Part 6 of assistant-basin; Claude, 2026-10-16).
#
# SCORE: for 18 'who are you' prompts (3 questions, each with 5 paraphrases) and two openings ('I am', 'My name is'),
#   the log-probability of each candidate name's token sequence as the continuation: summed over its tokens, and the
#   first name token alone. Margins are OLMo minus each other name. Base revisions get the plain 'User:/Assistant:' form;
#   chat models get the plain form and the chat template. Models: stage1 and stage2 revisions of allenai/OLMo-2-0425-1B,
#   then -SFT, -DPO, -RLVR1, -Instruct.
# PATCH: Instruct, chat form, 'I am' opening, the 3 base questions. Run A = 'You are Claude, an AI assistant made by
#   Anthropic.'; run B = no system prompt. The residual stream output of layers 0..16 (0 = embeddings, k = output of block
#   k-1) of B is replaced by A's, at (a) the last prompt position, (b) every position of the shared suffix (question
#   onward, aligned from the right). Control: the same from an unrelated system prompt of equal token count.
#   M = S(Claude) - S(OLMo); recovery = (M_patched - M_B) / (M_A - M_B).
# REPORT: writes out/names-summary.json (6.1-6.3 scored) and out/names-tables.md (tables for NAMES-REPORT.md).
#
# Usage (CPU, one model in memory at a time; the cache is on E:, read-only use):
#   HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 .venv/bin/python -I names.py score
#   HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 .venv/bin/python -I names.py patch
#   .venv/bin/python -I names.py report
# Outputs: out/names-scores.json, out/names-patch.json, out/names-summary.json, out/names-tables.md
import gc, json, os, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
torch.set_grad_enabled(False)

NAMES = ['OLMo', 'ChatGPT', 'Claude', 'Gemini', 'Llama', 'Qwen']
OPENINGS = ['I am', 'My name is']
QUESTIONS = [
    ('Who are you?', ['Tell me who you are.', 'Can you introduce yourself?', 'What kind of assistant are you?',
                      'Who am I talking to?', 'Describe yourself.']),
    ('What is your name?', ["What's your name?", 'What should I call you?', 'Do you have a name?',
                            'What name do you go by?', 'How should I address you?']),
    ('Which company made you?', ['Who built you?', 'Which organization created you?', 'Who developed you?',
                                  'What company trained you?', 'Who is your maker?']),
]
PROMPTS = [(q, u) for q, paras in QUESTIONS for u in [q] + paras]  # 18 user texts, tagged by base question
REPO = 'allenai/OLMo-2-0425-1B'
STAGE1, STAGE2 = 'stage1-step1907359-tokens4001B', 'stage2-ingredient3-step23852-tokens51B'
LADDER = [  # (label, repo, revision, forms)
    ('stage1', REPO, STAGE1, ['plain']),
    ('stage2', REPO, STAGE2, ['plain']),
    ('SFT', 'allenai/OLMo-2-0425-1B-SFT', None, ['plain', 'chat']),
    ('DPO', 'allenai/OLMo-2-0425-1B-DPO', None, ['plain', 'chat']),
    ('RLVR1', 'allenai/OLMo-2-0425-1B-RLVR1', None, ['plain', 'chat']),
    ('Instruct', 'allenai/OLMo-2-0425-1B-Instruct', None, ['plain', 'chat']),
]
CLAUDE_SYS = 'You are Claude, an AI assistant made by Anthropic.'  # 12 tokens
CONTROL_SYS = 'You are a gardener who writes about tomatoes and rain.'  # 12 tokens, unrelated, same length


def load(repo, rev):
    kw = {'revision': rev} if rev else {}
    tok = AutoTokenizer.from_pretrained(repo, **kw)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, **kw).eval()
    return tok, model


def chat_text(tok, user, system=None):
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def form_text(tok, form, user, opening):
    if form == 'plain':
        return f'User: {user}\nAssistant: {opening}'
    return chat_text(tok, user) + opening


def enc(tok, text):
    return tok(text, add_special_tokens=False)['input_ids']


def name_scores(model, tok, pre, cands):
    """pre: prefix token ids. cands: name token-id lists. Returns [(sum logp, first-token logp)] per candidate."""
    seqs = [pre + c for c in cands]
    T = max(map(len, seqs))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([s + [pad] * (T - len(s)) for s in seqs])
    mask = torch.tensor([[1] * len(s) + [0] * (T - len(s)) for s in seqs])
    logp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
    out = []
    for b, c in enumerate(cands):
        lp = [logp[b, len(pre) + j - 1, c[j]].item() for j in range(len(c))]
        out.append((sum(lp), lp[0]))
    return out


def score():
    t0 = time.time()
    res = {'prompts': [{'base': b, 'user': u} for b, u in PROMPTS], 'names': NAMES, 'openings': OPENINGS,
           'name_tokens': {}, 'boundary_ok': {}, 'models': {}}
    for label, repo, rev, forms in LADDER:
        tok, model = load(repo, rev)
        cand_ids = [enc(tok, ' ' + n) for n in NAMES]
        res['name_tokens'][label] = {n: tok.convert_ids_to_tokens(c) for n, c in zip(NAMES, cand_ids)}
        for form in forms:
            rows, ok = [], True
            for b, u in PROMPTS:
                for op in OPENINGS:
                    text = form_text(tok, form, u, op)
                    pre = enc(tok, text)
                    if enc(tok, text + ' OLMo') != pre + cand_ids[0]: ok = False  # tokenisation boundary check
                    sc = name_scores(model, tok, pre, cand_ids)
                    rows.append({'base': b, 'user': u, 'opening': op,
                                 'sum': {n: s for n, (s, _) in zip(NAMES, sc)},
                                 'first': {n: f for n, (_, f) in zip(NAMES, sc)}})
            res['models'][f'{label}/{form}'] = rows
            res['boundary_ok'][f'{label}/{form}'] = ok
            print(f'  {label}/{form}: {len(rows)} rows, boundary_ok={ok}, {time.time() - t0:.0f}s', flush=True)
        del model; gc.collect()
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'names-scores.json'), 'w'), indent=1)
    print('wrote out/names-scores.json', flush=True)


def make_cap(store, k):
    def hook(mod, inp, out):
        store[k] = (out[0] if isinstance(out, tuple) else out).detach()
    return hook


def capture(model, ids):
    """Residual stream: index 0 = embedding output, k = output of block k-1. Each [1, T, d]."""
    got = [None] * (len(model.model.layers) + 1)
    hs = [model.model.embed_tokens.register_forward_hook(make_cap(got, 0))]
    hs += [lyr.register_forward_hook(make_cap(got, k + 1)) for k, lyr in enumerate(model.model.layers)]
    model(input_ids=torch.tensor([ids]))
    for h in hs: h.remove()
    return got


def make_patch(bpos, src, apos):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, bpos] = src[:, apos]  # every batch row (the two candidates share the prefix)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return hook


def q_start_and_ids(tok, text, q):
    e = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    c = text.index(q)
    j = next(j for j, (_, en) in enumerate(e['offset_mapping']) if en > c)  # first token overlapping the question
    return e['input_ids'], j


def patch():
    t0 = time.time()
    tok, model = load('allenai/OLMo-2-0425-1B-Instruct', None)
    assert len(enc(tok, CLAUDE_SYS)) == len(enc(tok, CONTROL_SYS)), 'control system prompt must match token length'
    cands = [enc(tok, ' Claude'), enc(tok, ' OLMo')]
    L = len(model.model.layers)
    res = {'claude_sys': CLAUDE_SYS, 'control_sys': CONTROL_SYS, 'claude_sys_tokens': len(enc(tok, CLAUDE_SYS)),
           'control_sys_tokens': len(enc(tok, CONTROL_SYS)), 'layers': L, 'questions': {}}
    for q, _ in QUESTIONS:
        tB, jB = q_start_and_ids(tok, chat_text(tok, q) + 'I am', q)
        tA, jA = q_start_and_ids(tok, chat_text(tok, q, CLAUDE_SYS) + 'I am', q)
        tC, jC = q_start_and_ids(tok, chat_text(tok, q, CONTROL_SYS) + 'I am', q)
        sB, sA, sC = len(tB) - jB, len(tA) - jA, len(tC) - jC
        assert sA == sB == sC and tA[-sA:] == tB[-sB:] == tC[-sC:], f'shared suffix not aligned for {q!r}'
        modes = {'last': ([len(tB) - 1], [len(tA) - 1], [len(tC) - 1]),
                 'suffix': (list(range(len(tB) - sB, len(tB))), list(range(len(tA) - sA, len(tA))),
                            list(range(len(tC) - sC, len(tC))))}
        capA, capC = capture(model, tA), capture(model, tC)
        S_B = name_scores(model, tok, tB, cands)
        S_A = name_scores(model, tok, tA, cands)
        S_C = name_scores(model, tok, tC, cands)
        M = lambda S, i: S[0][i] - S[1][i]  # Claude minus OLMo; i=0 sum, i=1 first token
        base = {'M_B': M(S_B, 0), 'M_A': M(S_A, 0), 'M_C': M(S_C, 0),
                'first_B': M(S_B, 1), 'first_A': M(S_A, 1), 'first_C': M(S_C, 1), 'suffix_tokens': sB,
                'prompt_tokens_B': len(tB), 'prompt_tokens_A': len(tA), 'prompt_tokens_C': len(tC), 'layers': []}

        def patched(k, bpos, src, apos):
            tgt = model.model.embed_tokens if k == 0 else model.model.layers[k - 1]
            h = tgt.register_forward_hook(make_patch(torch.tensor(bpos), src[k], torch.tensor(apos)))
            try:
                S = name_scores(model, tok, tB, cands)
            finally:
                h.remove()
            return M(S, 0), M(S, 1)

        for k in range(L + 1):
            row = {'k': k}
            for mode, (bp, ap, cp) in modes.items():
                m, f = patched(k, bp, capA, ap)
                row[f'A_{mode}'] = {'M': m, 'first': f}
                m, f = patched(k, bp, capC, cp)
                row[f'control_{mode}'] = {'M': m, 'first': f}
            base['layers'].append(row)
        res['questions'][q] = base
        print(f'  {q!r}: M_B={base["M_B"]:.3f} M_A={base["M_A"]:.3f} M_C={base["M_C"]:.3f} suffix={sB} tokens, '
              f'{time.time() - t0:.0f}s', flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, 'names-patch.json'), 'w'), indent=1)
    print('wrote out/names-patch.json', flush=True)


def mean(xs): return float(np.mean(xs))


def report():
    S = json.load(open(os.path.join(OUT, 'names-scores.json')))
    P = json.load(open(os.path.join(OUT, 'names-patch.json')))
    others = [n for n in NAMES if n != 'OLMo']
    margins = {}  # key -> {opening: {'sum'/'first': {other: {'mean', 'per_prompt'}}}}
    for key, rows in S['models'].items():
        margins[key] = {}
        for op in OPENINGS:
            rs = [r for r in rows if r['opening'] == op]
            margins[key][op] = {}
            for kind in ('sum', 'first'):
                margins[key][op][kind] = {}
                for o in others:
                    per = [r[kind]['OLMo'] - r[kind][o] for r in rs]
                    margins[key][op][kind][o] = {'mean': mean(per), 'per_prompt': per}

    # 6.1 / 6.2: first name token, opening 'I am', margin OLMo minus ChatGPT (the prediction's wording).
    m61_inst = margins['Instruct/chat']['I am']['first']['ChatGPT']['mean']
    m61_base = margins['stage2/plain']['I am']['first']['ChatGPT']['mean']
    s1 = margins['stage1/plain']['I am']['first']['ChatGPT']['mean']
    s2 = margins['stage2/plain']['I am']['first']['ChatGPT']['mean']
    sft = margins['SFT/plain']['I am']['first']['ChatGPT']['mean']
    held61 = m61_inst > 0 and m61_base < 0
    held62 = s1 < 0 and s2 < 0 and sft > 0
    # 6.1/6.2 secondary: the sum over name tokens
    sum61 = (margins['Instruct/chat']['I am']['sum']['ChatGPT']['mean'], margins['stage2/plain']['I am']['sum']['ChatGPT']['mean'])
    sum62 = tuple(margins[k]['I am']['sum']['ChatGPT']['mean'] for k in ('stage1/plain', 'stage2/plain', 'SFT/plain'))

    # 6.3: mean over the 3 questions of recovery, by layer, for each mode and metric
    qs = list(P['questions'])
    curve = {}
    for metric in ('sum', 'first'):
        for mode in ('A_last', 'A_suffix', 'control_last', 'control_suffix'):
            curve.setdefault(metric, {})[mode] = []
            for k in range(P['layers'] + 1):
                recs = []
                for q in qs:
                    b = P['questions'][q]
                    row = b['layers'][k]
                    if metric == 'sum':
                        den, base, val = b['M_A'] - b['M_B'], b['M_B'], row[mode]['M']
                    else:
                        den, base, val = b['first_A'] - b['first_B'], b['first_B'], row[mode]['first']
                    recs.append((val - base) / den)
                curve[metric][mode].append({'layer': k, 'mean': mean(recs), 'per_question': recs})
    crossing = {}
    for metric in ('sum', 'first'):
        ks = [r['layer'] for r in curve[metric]['A_last'] if r['mean'] > 0.5]
        crossing[metric] = {'layers_over_half': ks, 'held_12_to_16': set(ks) == set(range(12, 17))}
    held63 = crossing['sum']['held_12_to_16']

    summary = {
        'criteria': {
            '6.1': 'first name token, opening "I am": Instruct (chat, no system prompt) OLMo-minus-ChatGPT > 0 AND stage2 base (plain) < 0',
            '6.2': 'first name token, opening "I am", plain form: stage1 < 0, stage2 < 0, SFT > 0',
            '6.3': 'sum over name tokens, last prompt position: mean recovery over the 3 questions > 0.5 at exactly layers 12..16',
        },
        '6.1': {'held': held61, 'instruct_chat_OLMo_minus_ChatGPT_first': m61_inst,
                'stage2_plain_OLMo_minus_ChatGPT_first': m61_base,
                'sum_instruct_chat_and_stage2_plain': sum61},
        '6.2': {'held': held62, 'stage1': s1, 'stage2': s2, 'SFT': sft, 'sum_stage1_stage2_SFT': sum62},
        '6.3': {'held': held63, 'crossing': crossing, 'curve_recovery_by_layer': curve,
                'M_A_minus_M_B': {q: P['questions'][q]['M_A'] - P['questions'][q]['M_B'] for q in qs},
                'first_A_minus_first_B': {q: P['questions'][q]['first_A'] - P['questions'][q]['first_B'] for q in qs}},
        'margins': margins,
        'boundary_ok': S['boundary_ok'],
        'name_tokens': S['name_tokens'],
    }
    json.dump(summary, open(os.path.join(OUT, 'names-summary.json'), 'w'), indent=1)

    # tables for NAMES-REPORT.md
    L = []
    L.append('### Margins (OLMo minus X), mean over 18 prompts, opening "I am"\n')
    L.append('| model/form | ChatGPT (first) | ChatGPT (sum) | Claude (sum) | Gemini (sum) | Llama (sum) | Qwen (sum) |')
    L.append('|---|---|---|---|---|---|---|')
    for key in S['models']:
        m = margins[key]['I am']
        L.append(f"| {key} | {m['first']['ChatGPT']['mean']:+.3f} | {m['sum']['ChatGPT']['mean']:+.3f} | "
                 f"{m['sum']['Claude']['mean']:+.3f} | {m['sum']['Gemini']['mean']:+.3f} | "
                 f"{m['sum']['Llama']['mean']:+.3f} | {m['sum']['Qwen']['mean']:+.3f} |")
    L.append('\n### Same, opening "My name is"\n')
    L.append('| model/form | ChatGPT (first) | ChatGPT (sum) | Claude (sum) | Gemini (sum) | Llama (sum) | Qwen (sum) |')
    L.append('|---|---|---|---|---|---|---|')
    for key in S['models']:
        m = margins[key]['My name is']
        L.append(f"| {key} | {m['first']['ChatGPT']['mean']:+.3f} | {m['sum']['ChatGPT']['mean']:+.3f} | "
                 f"{m['sum']['Claude']['mean']:+.3f} | {m['sum']['Gemini']['mean']:+.3f} | "
                 f"{m['sum']['Llama']['mean']:+.3f} | {m['sum']['Qwen']['mean']:+.3f} |")
    L.append('\n### Per-question means (opening "I am", first name token, OLMo minus ChatGPT)\n')
    L.append('| model/form | ' + ' | '.join(q for q, _ in QUESTIONS) + ' |')
    L.append('|---|' + '---|' * len(QUESTIONS))
    for key in S['models']:
        cells = []
        for q, _ in QUESTIONS:
            vals = [v for r, v in zip([r for r in S['models'][key] if r['opening'] == 'I am'],
                                      margins[key]['I am']['first']['ChatGPT']['per_prompt']) if r['base'] == q]
            cells.append(f'{mean(vals):+.3f}')
        L.append(f'| {key} | ' + ' | '.join(cells) + ' |')
    L.append('\n### Patching curve on Instruct (chat, "I am"): recovery by layer, mean over 3 questions\n')
    L.append('| layer | last, sum | suffix, sum | control last, sum | control suffix, sum | last, first-token | suffix, first-token |')
    L.append('|---|---|---|---|---|---|---|')
    for k in range(P['layers'] + 1):
        c = curve
        L.append(f"| {k} | {c['sum']['A_last'][k]['mean']:.3f} | {c['sum']['A_suffix'][k]['mean']:.3f} | "
                 f"{c['sum']['control_last'][k]['mean']:.3f} | {c['sum']['control_suffix'][k]['mean']:.3f} | "
                 f"{c['first']['A_last'][k]['mean']:.3f} | {c['first']['A_suffix'][k]['mean']:.3f} |")
    L.append('\n### Patching endpoints (M_B, M_A, M_C = Claude minus OLMo, sum over name tokens)\n')
    L.append('| question | M_B | M_A | M_C (control) | first-token M_B | first-token M_A | suffix tokens |')
    L.append('|---|---|---|---|---|---|---|')
    for q in qs:
        b = P['questions'][q]
        L.append(f"| {q} | {b['M_B']:+.3f} | {b['M_A']:+.3f} | {b['M_C']:+.3f} | {b['first_B']:+.3f} | {b['first_A']:+.3f} | {b['suffix_tokens']} |")
    open(os.path.join(OUT, 'names-tables.md'), 'w').write('\n'.join(L) + '\n')

    print(f'6.1 held={held61}  Instruct chat OLMo-ChatGPT(first)={m61_inst:+.3f}  stage2 plain={m61_base:+.3f}')
    print(f'6.2 held={held62}  stage1={s1:+.3f} stage2={s2:+.3f} SFT={sft:+.3f}')
    print(f'6.3 held={held63}  layers over half (sum): {crossing["sum"]["layers_over_half"]}; '
          f'first-token: {crossing["first"]["layers_over_half"]}')
    print('wrote out/names-summary.json, out/names-tables.md')


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    {'score': score, 'patch': patch, 'report': report}[cmd]()
