"""Evaluate a graft version. Usage:
  python run_eval.py VERSION ACTS 'build kwargs' [--calib-only] [--save]
Appends a JSON line to log.jsonl. Base-model reference outputs are cached in base_ref.json."""
import sys, json, time, os, torch
from common import *
from texts import DAMAGE_PROMPTS, HELDOUT_SENTENCES
from graft import build, build_multi, Graft, install

ver, acts_path, kw = sys.argv[1], sys.argv[2], sys.argv[3]
calib_only = "--calib-only" in sys.argv
save = "--save" in sys.argv
t0 = time.time()
tok, m = load()
HERE = os.path.dirname(os.path.abspath(__file__))


@torch.no_grad()
def gen_one(text, n=40):
    ids = tok(chat(text), return_tensors="pt")["input_ids"]
    for _ in range(n):
        nxt = m(input_ids=ids, use_cache=False).logits[0, -1].argmax()
        ids = torch.cat([ids, nxt.view(1, 1)], 1)
    return ids[0].tolist()


@torch.no_grad()
def loglik(s):
    ids = tok(s, return_tensors="pt")["input_ids"]
    lp = torch.log_softmax(m(input_ids=ids, use_cache=False).logits[0, :-1].float(), -1)
    return lp.gather(1, ids[0, 1:, None])[:, 0].mean().item()


def evaluate(pairs, tag):
    r = {}
    only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    for name, f in [("main", FMT_MAIN), ("other", FMT_OTHER), ("other2", FMT_OTHER2), ("other3", FMT_OTHER3)]:
        if calib_only and name != "main":
            continue
        if only and name not in only:
            continue
        acc, outs = accuracy(pairs, f)
        r[f"{tag}_{name}"] = acc
        r[f"{tag}_{name}_wrong"] = [(a, b, o) for (a, b), o in zip(pairs, outs) if o != str(a + b)][:12]
        print(name, acc, flush=True)
    return r


ref_path = os.path.join(HERE, "base_ref.json")
if not os.path.exists(ref_path) and not calib_only:
    ref = dict(gens=[gen_one(p) for p in DAMAGE_PROMPTS], ll=[loglik(s) for s in HELDOUT_SENTENCES])
    json.dump(ref, open(ref_path, "w"))
    print("base ref built", time.time() - t0, flush=True)

acts = torch.load(acts_path)
W = (build_multi if "--multi" in sys.argv else build)(acts, model=m, **eval("dict(" + kw + ")"))
del acts
g = Graft(W)
hk = install(m, g)
res = dict(version=ver, build=kw)
if calib_only:
    res.update(evaluate(CALIB[400:600], "calibcheck"))
else:
    res.update(evaluate(TEST, "test"))
if not calib_only and "--only" not in sys.argv:
    ref = json.load(open(ref_path))
    gens = [gen_one(p) for p in DAMAGE_PROMPTS]
    res["damage_identical_40tok"] = sum(a == b for a, b in zip(gens, ref["gens"]))
    ll = [loglik(s) for s in HELDOUT_SENTENCES]
    res["damage_mean_dloglik_per_token"] = sum(a - b for a, b in zip(ll, ref["ll"])) / len(ll)
    res["damage_max_abs_dloglik"] = max(abs(a - b) for a, b in zip(ll, ref["ll"]))
hk.remove()
res["seconds"] = round(time.time() - t0)
print(json.dumps({k: v for k, v in res.items() if "wrong" not in k}), flush=True)
with open(os.path.join(HERE, "log.jsonl"), "a") as f:
    f.write(json.dumps(res) + "\n")
if save:
    torch.save({k: v for k, v in W.items()}, os.path.join(HERE, f"graft_{ver}.pt"))
