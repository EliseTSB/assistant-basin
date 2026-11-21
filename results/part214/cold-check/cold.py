import json, re, sys
from collections import Counter

# --- my own re-implementation of the Part 66 maker lexicon (read from part66/score66.py lines 8-15) ---
FAM = {
 'google':    r'Google|Gemini|DeepMind',
 'alibaba':   r'Alibaba|Qwen|Tongyi',
 'openai':    r'OpenAI|ChatGPT|GPT-',
 'anthropic': r'Anthropic|Claude',
 'deepseek':  r'DeepSeek',
 'meta':      r'Meta AI|Llama',
}
FPAT = {k: re.compile(v, re.I) for k, v in FAM.items()}
MAKER_STATE = {'google': 'clean_google', 'alibaba': 'clean_alibaba'}
STATES = ('clean_google','clean_alibaba','clean_other_maker','clean_no_maker','broken')

def is_broken(ans):
    t = ans or ''
    if not t.strip(): return True
    w = t.split()
    if len(w) < 4: return True
    if '�' in t or '!!!!' in t: return True
    grams = Counter(tuple(w[i:i+3]) for i in range(len(w)-2))
    return any(n >= 4 for n in grams.values())

def classify(ans):
    if is_broken(ans): return 'broken', []
    hits = []
    for name, pat in FPAT.items():
        m = pat.search(ans or '')
        if m: hits.append((name, m.start()))
    if not hits: return 'clean_no_maker', []
    first = min(hits, key=lambda h: (h[1], h[0]))[0]
    return MAKER_STATE.get(first, 'clean_other_maker'), [h[0] for h in hits]

rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
main = [r for r in rows if 0 <= int(r['question_index']) < 24]
print('rows total', len(rows), 'main', len(main))
ARMS = ['unchanged','zero','trace_resample_fact','trace_resample_you','scale_dial_fact',
        'scale_dial_you','prompt_fact_gen_you','mean_resample_you']
mine = {}
for arm in ARMS:
    rs = [r for r in main if r['arm'] == arm]
    st = Counter(); multi = 0; broken_idx = []
    for r in rs:
        s, hits = classify(r['answer'])
        st[s] += 1
        if s != 'broken' and len(hits) > 1: multi += 1
        if s == 'broken': broken_idx.append(r)
    mine[arm] = {'n': len(rs), 'counts': {k: st.get(k,0) for k in STATES},
                 'clean_google': st.get('clean_google',0), 'broken': st.get('broken',0),
                 'multi': multi, 'broken_rows': broken_idx}
print()
print(f"{'arm':22s} {'n':>4s} {'cg':>4s} {'brk':>4s} {'mh':>3s}   counts")
for arm in ARMS:
    m = mine[arm]
    print(f"{arm:22s} {m['n']:4d} {m['clean_google']:4d} {m['broken']:4d} {m['multi']:3d}   {m['counts']}")
print()
sc = json.load(open(sys.argv[2], encoding='utf-8'))['stage2']['main_arms']
print('--- mine vs score.json ---')
for arm in ARMS:
    s = sc[arm]; m = mine[arm]
    ok = (s['clean_google']==m['clean_google'] and s['broken']==m['broken'] and s['counts']==m['counts'])
    print(f"{arm:22s} cg {m['clean_google']:3d} vs {s['clean_google']:3d} | brk {m['broken']:2d} vs {s['broken']:2d} | counts_equal={s['counts']==m['counts']} | multi {m['multi']} vs {s['multi_hit']} | {'AGREE' if ok else 'MISMATCH'}")

json.dump({a:{'n':mine[a]['n'],'counts':mine[a]['counts'],'clean_google':mine[a]['clean_google'],
              'broken':mine[a]['broken'],'multi':mine[a]['multi']} for a in ARMS},
          open('/tmp/claude-1000/part214-coldcheck/mine_counts.json','w'), indent=1)
