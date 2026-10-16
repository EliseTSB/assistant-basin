#!/usr/bin/env python3
# names_null.py — Part 6.4: a name-matched null for the Part 6 patching (Claude, 2026-10-16).
#
# Question: in the Part 6 patch (Instruct, chat form, 'I am'), the Claude system prompt recovers the Claude-minus-OLMo
#   margin late, but an unrelated system prompt of equal length moves it 67-83% as much. Is the Claude-specific part
#   still concentrated in layers 13-16 once the other name-swapped prompts are the null?
# DESIGN: the same residual patch as names.py patch(), for all 18 user texts (3 base questions x 6 wordings), opening 'I am',
#   Instruct chat form. Sources (run label -> system prompt): Claude, ChatGPT, Gemini, OLMo (own name and company swapped),
#   and the unrelated gardener control. Target: the no-system-prompt run. Layers 0..16, two modes: 'last' (last prompt
#   position) and 'suffix' (every position of the shared suffix). Scored on candidates Claude, ChatGPT, Gemini, OLMo.
#   Each candidate's sum of token log-probs and its first token are kept, so both metrics come from one pass.
# RECOVERY: per prompt, R = (M_patched - M_B) / (M_end - M_B), where M = S(X) - S(OLMo) for the margin named, and M_end is
#   the same margin in the run the source came from.
#     curve on the Claude-minus-OLMo margin: for every source run.
#     curve on the own-name-minus-OLMo margin: for ChatGPT, Gemini and Claude (OLMo and control have no own-name margin).
#   Claude-specific curve (pre-committed): per prompt, R_Claude(Claude-minus-OLMo) minus the mean of R over the other
#     name runs (ChatGPT, Gemini, OLMo) on the same margin; then the mean over prompts.
#   Share (pre-committed primary: last position, sum metric): sum over layers 13-16 / sum over layers 0-16.
#   Raw-nat and first-token versions are reported for reference.
# Usage (CPU, one model in memory; the cache on E: is read-only use):
#   HF_HOME=/mnt/e/neuron-probe/hf HF_HUB_OFFLINE=1 .venv/bin/python -I names_null.py patch [START:END]
#   .venv/bin/python -I names_null.py report
# Outputs: out/names-null-patch-p{NN}.json (one per prompt, resumable), out/names-null-summary.json, out/names-null-tables.md
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import names as N  # reuse load, chat_text, enc, name_scores, capture, make_patch, q_start_and_ids (names.py is not edited)

OUT = os.path.join(HERE, 'out')
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("NN_THREADS", torch.get_num_threads())))

RUNS = {  # source label -> system prompt. Claude prompt and control are the same as names.py.
    'Claude': N.CLAUDE_SYS,  # 12 tokens
    'ChatGPT': 'You are ChatGPT, an AI made by OpenAI.',  # 13 tokens
    'Gemini': 'You are Gemini, an AI assistant made by Google.',  # 11 tokens
    'OLMo': 'You are OLMo, an AI made by Ai2.',  # 13 tokens
    'control': N.CONTROL_SYS,  # 12 tokens, unrelated gardener
}
CANDS = ['Claude', 'ChatGPT', 'Gemini', 'OLMo']
OTHERS = ['ChatGPT', 'Gemini', 'OLMo']  # the name-swapped nulls for the Claude-specific curve
OWN = ['Claude', 'ChatGPT', 'Gemini']  # runs with an own-name margin
MODES = ['last', 'suffix']
PROMPTS = N.PROMPTS  # 18 (base question, user text)


def ckpt(i):
    return os.path.join(OUT, f'names-null-patch-p{i:02d}.json')


def scores(S):
    return {'sum': {c: s for c, (s, _) in zip(CANDS, S)}, 'first': {c: f for c, (_, f) in zip(CANDS, S)}}


def run_prompt(tok, model, user, cand_ids):
    texts = {'B': N.chat_text(tok, user) + 'I am'}
    texts.update({r: N.chat_text(tok, user, s) + 'I am' for r, s in RUNS.items()})
    ids, j = {}, {}
    for r, t in texts.items():
        ids[r], j[r] = N.q_start_and_ids(tok, t, user)
    sfx = {r: len(ids[r]) - j[r] for r in texts}
    n = sfx['B']
    assert len(set(sfx.values())) == 1, f'suffix length differs across sources for {user!r}: {sfx}'
    assert all(ids[r][-n:] == ids['B'][-n:] for r in texts), f'shared suffix not aligned for {user!r}'
    # tokens of the system prompt in each run, so the matched-length check is on the record
    rec = {'user': user, 'suffix_tokens': n, 'tokens': {r: len(ids[r]) for r in texts}, 'baseline': {}, 'endpoint': {},
           'patch': {r: {m: [] for m in MODES} for r in RUNS}}
    rec['baseline']['B'] = scores(N.name_scores(model, tok, ids['B'], cand_ids))
    for r in RUNS:
        rec['endpoint'][r] = scores(N.name_scores(model, tok, ids[r], cand_ids))
    L = len(model.model.layers)
    tgt_last = [len(ids['B']) - 1]
    tgt_suf = list(range(len(ids['B']) - n, len(ids['B'])))
    for r in RUNS:
        cap = N.capture(model, ids[r])
        src_last = [len(ids[r]) - 1]
        src_suf = list(range(len(ids[r]) - n, len(ids[r])))
        for mode, (tp, sp) in {'last': (tgt_last, src_last), 'suffix': (tgt_suf, src_suf)}.items():
            for k in range(L + 1):
                tgt = model.model.embed_tokens if k == 0 else model.model.layers[k - 1]
                h = tgt.register_forward_hook(N.make_patch(torch.tensor(tp), cap[k], torch.tensor(sp)))
                try:
                    S = N.name_scores(model, tok, ids['B'], cand_ids)
                finally:
                    h.remove()
                rec['patch'][r][mode].append({'k': k, **scores(S)})
    return rec


def patch(spec=None):
    t0 = time.time()
    lo, hi = (0, len(PROMPTS)) if not spec else tuple(int(x) for x in spec.split(':'))
    tok, model = N.load('allenai/OLMo-2-0425-1B-Instruct', None)
    cand_ids = [N.enc(tok, ' ' + c) for c in CANDS]
    lens = {r: len(N.enc(tok, s)) for r, s in RUNS.items()}
    print('system prompt tokens:', lens, flush=True)
    for r, s in RUNS.items():
        if r != 'Claude':
            assert abs(lens[r] - lens['Claude']) <= 1, f'{r} system prompt is more than 1 token from the Claude prompt'
    os.makedirs(OUT, exist_ok=True)
    for i in range(lo, min(hi, len(PROMPTS))):
        if os.path.exists(ckpt(i)):
            print(f'  p{i:02d} exists, skipping', flush=True)
            continue
        base, user = PROMPTS[i]
        rec = run_prompt(tok, model, user, cand_ids)
        rec.update({'index': i, 'base': base, 'sys_prompts': RUNS, 'sys_tokens': lens, 'cands': CANDS})
        json.dump(rec, open(ckpt(i), 'w'), indent=1)
        print(f'  p{i:02d} {user!r}: suffix={rec["suffix_tokens"]} tokens, {time.time() - t0:.0f}s', flush=True)
    print('done chunk', lo, hi, flush=True)


def load_all():
    recs = []
    for i in range(len(PROMPTS)):
        if not os.path.exists(ckpt(i)):
            raise SystemExit(f'missing {ckpt(i)}; run patch first')
        recs.append(json.load(open(ckpt(i))))
    return recs


def margin(S, a, b, kind):
    return S[kind][a] - S[kind][b]


def mean(xs):
    return float(np.mean(xs))


def report():
    recs = load_all()
    L = len(recs[0]['patch']['Claude']['last']) - 1
    layers = list(range(L + 1))
    summary = {'n_prompts': len(recs), 'layers': L, 'sys_tokens': recs[0]['sys_tokens'], 'sys_prompts': RUNS,
               'tokens_per_prompt': {r['user']: r['tokens'] for r in recs}, 'metrics': {}}
    # check: the Claude run must reproduce names.py's M_B and M_A for the three base-question prompts (same texts)
    base_check = {}
    for r in recs:
        if r['user'] in ('Who are you?', 'What is your name?', 'Which company made you?'):
            M_B = margin(r['baseline']['B'], 'Claude', 'OLMo', 'sum')
            M_A = margin(r['endpoint']['Claude'], 'Claude', 'OLMo', 'sum')
            base_check[r['user']] = {'M_B': M_B, 'M_A': M_A}
    summary['check_vs_names_py_patch'] = base_check

    tables = []
    for metric in ('sum', 'first'):
        m = {'endpoints': {}, 'curves': {}, 'claude_specific': {}, 'share': {}, 'raw_claude_specific': {}}
        # endpoints: mean and minimum |denominator| over prompts, for the two margins
        for run in RUNS:
            dC = [margin(r['endpoint'][run], 'Claude', 'OLMo', metric) - margin(r['baseline']['B'], 'Claude', 'OLMo', metric)
                  for r in recs]
            ep = {'claude_minus_olmo_shift_mean': mean(dC), 'claude_minus_olmo_shift_min_abs': float(np.min(np.abs(dC)))}
            if run in OWN:
                dO = [margin(r['endpoint'][run], run, 'OLMo', metric) - margin(r['baseline']['B'], run, 'OLMo', metric)
                      for r in recs]
                ep.update({'own_minus_olmo_shift_mean': mean(dO), 'own_minus_olmo_shift_min_abs': float(np.min(np.abs(dO)))})
            m['endpoints'][run] = ep
        # recovery curves per run and margin; per-prompt arrays kept for paired differences
        R = {}  # (run, margin_name, mode) -> array [prompts, layers]
        D = {}  # raw shift in nats on Claude-minus-OLMo, same keys for run only
        for run in RUNS:
            for mode in MODES:
                arr = np.zeros((len(recs), L + 1))
                draw = np.zeros((len(recs), L + 1))
                arr_o = np.zeros((len(recs), L + 1)) if run in OWN else None
                for pi, r in enumerate(recs):
                    B = r['baseline']['B']
                    end = r['endpoint'][run]
                    for k in layers:
                        P = r['patch'][run][mode][k]
                        mP = margin(P, 'Claude', 'OLMo', metric)
                        mB = margin(B, 'Claude', 'OLMo', metric)
                        mE = margin(end, 'Claude', 'OLMo', metric)
                        arr[pi, k] = (mP - mB) / (mE - mB)
                        draw[pi, k] = mP - mB
                        if run in OWN:
                            oP = margin(P, run, 'OLMo', metric)
                            oB = margin(B, run, 'OLMo', metric)
                            oE = margin(end, run, 'OLMo', metric)
                            arr_o[pi, k] = (oP - oB) / (oE - oB)
                R[(run, 'C', mode)] = arr
                D[(run, mode)] = draw
                if run in OWN:
                    R[(run, 'own', mode)] = arr_o
        # Claude-specific curve (paired by prompt) and its share
        spec, rawspec, ctrl = {}, {}, {}
        for mode in MODES:
            per = R[('Claude', 'C', mode)] - np.mean([R[(o, 'C', mode)] for o in OTHERS], axis=0)
            spec[mode] = per.mean(axis=0)
            rawper = D[('Claude', mode)] - np.mean([D[(o, mode)] for o in OTHERS], axis=0)
            rawspec[mode] = rawper.mean(axis=0)
            ctrl[mode] = (R[('Claude', 'C', mode)] - R[('control', 'C', mode)]).mean(axis=0)
            tot = spec[mode].sum()
            late = spec[mode][13:17].sum()
            rtot = rawspec[mode].sum()
            rlate = rawspec[mode][13:17].sum()
            m['share'][mode] = {'total': float(tot), 'late_13_16': float(late),
                                'share': float(late / tot) if tot != 0 else None,
                                'raw_total_nats': float(rtot), 'raw_late_nats': float(rlate),
                                'raw_share': float(rlate / rtot) if rtot != 0 else None}
        m['claude_specific'] = {mode: spec[mode].tolist() for mode in MODES}
        m['raw_claude_specific_nats'] = {mode: rawspec[mode].tolist() for mode in MODES}
        m['claude_minus_control_recovery'] = {mode: ctrl[mode].tolist() for mode in MODES}
        for run in RUNS:
            for mode in MODES:
                m['curves'][f'{run}|C|{mode}'] = R[(run, 'C', mode)].mean(axis=0).tolist()
                if run in OWN:
                    m['curves'][f'{run}|own|{mode}'] = R[(run, 'own', mode)].mean(axis=0).tolist()
        summary['metrics'][metric] = m

    # the layer-16 last-position first-token check: the patched last block fixes the first token exactly
    f16 = summary['metrics']['first']['curves']
    summary['check_layer16_first_last'] = {run: f16[f'{run}|C|last'][L] for run in RUNS}
    json.dump(summary, open(os.path.join(OUT, 'names-null-summary.json'), 'w'), indent=1)

    # tables
    T = []
    S = summary['metrics']['sum']
    T.append('### Endpoints (Claude-minus-OLMo, sum): mean shift from no-prompt run, and min |denominator| over 18 prompts\n')
    T.append('| source | mean M_end - M_B | min abs | own-name-minus-OLMo mean shift | own min abs |')
    T.append('|---|---|---|---|---|')
    for run in RUNS:
        e = S['endpoints'][run]
        own = (f"{e['own_minus_olmo_shift_mean']:+.3f} | {e['own_minus_olmo_shift_min_abs']:.3f}" if run in OWN else 'n/a | n/a')
        T.append(f"| {run} | {e['claude_minus_olmo_shift_mean']:+.3f} | {e['claude_minus_olmo_shift_min_abs']:.3f} | {own} |")
    T.append('\n### Recovery by layer, last prompt position, sum metric (mean over 18 prompts)\n')
    cols = ['Claude|C|last', 'ChatGPT|C|last', 'Gemini|C|last', 'OLMo|C|last', 'control|C|last',
            'Claude|own|last', 'ChatGPT|own|last', 'Gemini|own|last']
    T.append('| layer | ' + ' | '.join(cols) + ' | Claude-specific |')
    T.append('|---|' + '---|' * (len(cols) + 1))
    for k in layers:
        vals = [S['curves'][c][k] for c in cols]
        T.append(f'| {k} | ' + ' | '.join(f'{v:.3f}' for v in vals) + f" | {S['claude_specific']['last'][k]:+.3f} |")
    T.append('\n### Recovery by layer, suffix (every shared-suffix position), sum metric\n')
    cols2 = ['Claude|C|suffix', 'ChatGPT|C|suffix', 'Gemini|C|suffix', 'OLMo|C|suffix', 'control|C|suffix']
    T.append('| layer | ' + ' | '.join(cols2) + ' | Claude-specific |')
    T.append('|---|' + '---|' * (len(cols2) + 1))
    for k in layers:
        vals = [S['curves'][c][k] for c in cols2]
        T.append(f'| {k} | ' + ' | '.join(f'{v:.3f}' for v in vals) + f" | {S['claude_specific']['suffix'][k]:+.3f} |")
    T.append('\n### Share of the Claude-specific total in layers 13-16\n')
    T.append('| metric | mode | total (sum 0-16) | late (13-16) | share | raw-nat share |')
    T.append('|---|---|---|---|---|---|')
    for metric in ('sum', 'first'):
        for mode in MODES:
            s = summary['metrics'][metric]['share'][mode]
            sh = f"{s['share']:.3f}" if s['share'] is not None else 'n/a'
            rsh = f"{s['raw_share']:.3f}" if s['raw_share'] is not None else 'n/a'
            T.append(f"| {metric} | {mode} | {s['total']:+.3f} | {s['late_13_16']:+.3f} | {sh} | {rsh} |")
    open(os.path.join(OUT, 'names-null-tables.md'), 'w').write('\n'.join(T) + '\n')

    p = summary['metrics']['sum']['share']['last']
    print('6.4 primary (sum, last): total', round(p['total'], 3), 'late', round(p['late_13_16'], 3),
          'share', p['share'] and round(p['share'], 3), flush=True)
    print('wrote out/names-null-summary.json, out/names-null-tables.md', flush=True)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'patch':
        patch(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == 'report':
        report()
    else:
        raise SystemExit('usage: names_null.py patch [START:END] | report')
