"""Generate the three test sets under one condition and score them against plain.
usage: python evaluate.py plain|p72|vN [--sets A,B,C]
Conditions: plain = unchanged model; p72 = Part 72's push (L27 N3325 clamped to 8x its mean at every position);
vN = the inserted block in blocks/vN.safetensors. Answers go to answers/<cond>.jsonl."""
import json
import sys
import time

from safetensors.torch import load_file

import common as c
from sets import TESTS, check_disjoint


def load_block(name):
    w = load_file(str(c.HERE / "blocks" / f"{name}.safetensors"))
    return c.Block(int(w["layer"].item()), w["D"], w["theta"], w["V"])


def score(cond, tag=None):
    def rd(n):
        return {(r["set"], r["q"]): r for r in map(json.loads, open(c.HERE / "answers" / f"{n}.jsonl"))}
    plain, cur = rd("plain"), rd(tag or cond)
    out = {}
    for s in TESTS:
        keys = [(s, q) for q in TESTS[s] if (s, q) in cur]
        if not keys:
            continue
        n = len(keys)
        g = sum(c.names_google(cur[k]["text"]) for k in keys)
        g_only = sum(c.names_google(cur[k]["text"]) and not c.ALIBABA.search(cur[k]["text"]) for k in keys)
        same = sum(cur[k]["ids"] == plain[k]["ids"] for k in keys)
        brk = sum(c.broken(cur[k]["text"], cur[k]["ids"]) is not None for k in keys)
        out[s] = {"n": n, "google": g, "google_no_alibaba": g_only, "identical": same, "broken": brk}
    return out


def main():
    check_disjoint()
    cond = sys.argv[1]
    sets = sys.argv[sys.argv.index("--sets") + 1].split(",") if "--sets" in sys.argv else list(TESTS)
    t0 = time.time()
    tok, model, layers, _ = c.load()
    handles, block = [], None
    if cond == "p72":
        handles = c.install_p72(layers)
    elif cond != "plain":
        block = load_block(cond)
        handles = block.install(layers)
    (c.HERE / "answers").mkdir(exist_ok=True)
    rows, fired = [], {}
    for s in sets:
        f0 = block.fired if block else 0
        ids = c.generate(tok, model, TESTS[s], log=print)
        fired[s] = (block.fired - f0) if block else None
        for q, a in zip(TESTS[s], ids):
            rows.append({"cond": cond, "set": s, "q": q, "ids": a, "text": tok.decode(a, skip_special_tokens=True)})
    for h in handles:
        h.remove()
    tag = cond if sets == list(TESTS) else f"{cond}_{''.join(sets)}"
    with open(c.HERE / "answers" / f"{tag}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sc = score(cond, tag)
    sc["_seconds"] = round(time.time() - t0, 1)
    sc["_fired_positions_by_set"] = fired
    (c.HERE / "scores").mkdir(exist_ok=True)
    (c.HERE / "scores" / f"{tag}.json").write_text(json.dumps(sc, indent=1) + "\n")
    print(json.dumps(sc))


if __name__ == "__main__":
    main()
