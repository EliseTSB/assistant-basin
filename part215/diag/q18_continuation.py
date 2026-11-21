"""Part 215 diagnostic 2 (NOT a preflight attempt, writes no receipt): does Part 214's sequence shape explain G0's q18?

Part 214 Stage 1 read g_ali from prompt + opening + the 'Google' continuation (run214.forward_with_hooks, seq =
ids + opening + continuation_ids(bank, 'Google')); Part 215 reads it from prompt + opening only. diag/q18_repeat.py
showed q18 is deterministic here (-1.66022 unchanged, 0.32130 zero). This reads q16-q19 both ways, unchanged and
zero, so the two shapes can be compared with Part 214's recorded values. Written by Journal Claude (Opus 5.5).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[1] / 'part214'))
import run215  # noqa: E402
import score215 as sc  # noqa: E402
import neuron_report as nr  # noqa: E402
import run214  # noqa: E402


class A:
    device = 'xpu'


bank = sc.load_prompts()
idx = run215.prompt_index(bank)
bank214 = json.loads((HERE.parents[1] / 'part214' / 'prompts214.json').read_text(encoding='utf-8'))
cont = [int(t) for t in run214.continuation_ids(bank214, 'Google')]
opening = [int(t) for t in bank['openings']['en_maker']['token_ids']]
model, tok, load = run215.load_rig(A)
layers = nr.find_text_layers(model)[1]
rows = []
for qi in (16, 17, 18, 19):
    ids = [int(t) for t in run215.prompt_for(idx, 'identity', 'neutral', 'v0', qi)['token_ids']]
    ids214, _T = run214.stage1_ids(tok, bank214, qi)
    for zero in (False, True):
        short = run215.conditional_forward(model, layers, ids, opening, 'xpu', zero_targets=zero)['g_ali']
        longer = run215.conditional_forward(model, layers, ids, opening + cont, 'xpu', zero_targets=zero)['g_ali']
        row = {'q': qi, 'zero': zero, 'ids_equal_part214': ids == [int(t) for t in ids214],
               'short': short, 'with_continuation': longer, 'cont_len': len(cont)}
        rows.append(row)
        print(json.dumps(row), flush=True)
Path('results/part215/diag').mkdir(parents=True, exist_ok=True)
Path('results/part215/diag/q18_continuation.json').write_text(json.dumps({'rows': rows}))
print('DIAG DONE', flush=True)
