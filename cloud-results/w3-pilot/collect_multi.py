"""v5 calibration: CALIB[:400] spread over four BUILD-ONLY wordings (disjoint from FMT_OTHER and
FMT_OTHER2), teacher-forced; stores layers 20 and 22 plus role/anchor positions per sequence."""
import sys, torch
from common import *
from texts import GATE_CALIB_TEXTS

BUILD_FMTS = [
    FMT_MAIN,
    "What is {a} plus {b}? Answer with the number only.",
    "Calculate {a} + {b} and answer with the number only.",
    "{a} + {b} = ? Answer with the number only.",
]
if len(sys.argv) > 2 and sys.argv[2] == "v6":  # v6: eight build-only wordings
    BUILD_FMTS = BUILD_FMTS + [
        "Add {a} to {b}. Answer with the number only.",
        "If you add {a} and {b}, what do you get? Number only.",
        "{a} plus {b} equals what? Reply with the number.",
        "Find the result of {a} + {b}. Only the number, please.",
    ]
KEEP = [20, 22]
out = sys.argv[1]
tok, m = load()
eos = tok.convert_tokens_to_ids("<|im_end|>")
groups = {}
for i, (a, b) in enumerate(CALIB[:400]):
    w = i % len(BUILD_FMTS)
    p = tok(chat(BUILD_FMTS[w].format(a=a, b=b)))["input_ids"]
    seq = p + tok(str(a + b))["input_ids"] + [eos]
    dig = [j for j, t in enumerate(p) if 15 <= t <= 24][:4]
    groups.setdefault((w, len(seq)), []).append(dict(pair=(a, b), ids=seq, roles=dig, P=len(p) - 1))
recs = []
with torch.no_grad():
    for key, items in groups.items():
        for s in range(0, len(items), 50):
            chunk = items[s:s + 50]
            ids = torch.tensor([c["ids"] for c in chunk])
            hs = m(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
            for k, c in enumerate(chunk):
                c["h"] = {L: hs[L][k].half() for L in KEEP}
                c["w"] = key[0]
                recs.append(c)
        print(key, len(items), flush=True)
off = []
with torch.no_grad():
    for t in GATE_CALIB_TEXTS:
        ids = tok(chat(t), return_tensors="pt")["input_ids"]
        g = m.generate(ids, max_new_tokens=24, do_sample=False)
        hs = m(input_ids=g, use_cache=False, output_hidden_states=True).hidden_states
        off.append(dict(ids=g[0], h={L: hs[L][0].half() for L in KEEP}, plen=ids.shape[1]))
torch.save(dict(recs=recs, off=off, fmts=BUILD_FMTS), out)
print("saved", len(recs))
