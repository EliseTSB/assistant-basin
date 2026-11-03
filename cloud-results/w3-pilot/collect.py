"""Collect residual-stream activations on the CALIBRATION set only (teacher-forced correct answers),
plus 'off' activations on GATE_CALIB_TEXTS (prompt + model's own 24-token greedy reply).
Saves fp16 tensors to the scratch dir given as argv[1]."""
import sys, torch
from common import *
from texts import GATE_CALIB_TEXTS

out = sys.argv[1]
tok, m = load()
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
pairs = CALIB[:N]
eos = tok.convert_tokens_to_ids("<|im_end|>")

recs = {}
for L3 in (False, True):
    grp = [(a, b) for a, b in pairs if (a + b >= 100) == L3]
    seqs = [tok(chat(FMT_MAIN.format(a=a, b=b)))["input_ids"] + tok(str(a + b))["input_ids"] + [eos]
            for a, b in grp]
    hs = []
    with torch.no_grad():
        for s in range(0, len(seqs), 50):
            ids = torch.tensor(seqs[s:s + 50])
            o = m(input_ids=ids, use_cache=False, output_hidden_states=True)
            hs.append(torch.stack(o.hidden_states, 1).half())  # B x 25 x T x D
            print("batch", s, flush=True)
    recs[L3] = dict(pairs=grp, ids=torch.tensor(seqs), h=torch.cat(hs))

off = []
with torch.no_grad():
    for t in GATE_CALIB_TEXTS:
        ids = tok(chat(t), return_tensors="pt")["input_ids"]
        g = m.generate(ids, max_new_tokens=24, do_sample=False)
        o = m(input_ids=g, use_cache=False, output_hidden_states=True)
        off.append(dict(ids=g[0], h=torch.stack(o.hidden_states, 1)[0].half(), plen=ids.shape[1]))
torch.save(dict(two=recs[False], three=recs[True], off=off), out)
print("saved")
