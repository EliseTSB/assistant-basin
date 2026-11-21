"""Part 215 diagnostic (NOT a preflight attempt, writes no receipt): is G0's q18 miss deterministic on the A770?

The 2026-11-21 04:25 preflight stopped at G0: q18 differed from Part 214 Stage 1 by -0.0175 (unchanged) and +0.0832
(zero) while the other 23 questions matched to 0.0. This repeats the q18 and q17 (control) forwards three times each,
unchanged and zero, through run215's own conditional_row, and prints g_ali. Written by Journal Claude (Opus 5.5).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))
import run215  # noqa: E402
import score215 as sc  # noqa: E402
import neuron_report as nr  # noqa: E402


class A:
    device = 'xpu'


bank = sc.load_prompts()
idx = run215.prompt_index(bank)
model, tok, load = run215.load_rig(A)
layers = nr.find_text_layers(model)[1]
out = []
for qi in (18, 17):
    for zero in (False, True):
        vals = [run215.conditional_row(idx, bank, tok, model, layers, 'neutral', 'v0', qi, 'xpu',
                                       zero_targets=zero)['g_ali'] for _ in range(3)]
        out.append({'q': qi, 'zero': zero, 'g_ali': vals})
        print(json.dumps(out[-1]), flush=True)
Path('results/part215/diag').mkdir(parents=True, exist_ok=True)
Path('results/part215/diag/q18_repeat.json').write_text(json.dumps({'load': load, 'rows': out}, default=str))
print('DIAG DONE', flush=True)
