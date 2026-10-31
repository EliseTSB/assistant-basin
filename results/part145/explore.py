# Found after the registered reading, not registered (Claude, 2026-10-31). Run from the repo root with the job's matrices in <dir>.
# usage: python results/part145/explore.py <dir with matrix-*.npy> <instruct snapshot with vocab.json>
import json, sys
import numpy as np
root, snap = sys.argv[1], sys.argv[2]
vocab = {v: k for k, v in json.load(open(snap + "/vocab.json", encoding="utf-8")).items()}
def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    s = x[o]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r
def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])
def partial(x, y, c):
    rx, ry, rc = rank(x), rank(y), rank(c)
    A = np.c_[np.ones(len(rc)), rc]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]; ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])
out = {}
for q in (1, 2):
    plain = {c: np.load(f"{root}/plain-{c}-q{q}.npy")[0] for c in ("base-chat", "sft-chat", "dpo-chat", "instruct-chat")}
    M = np.load(f"{root}/matrix-instruct-chat-q{q}.npy", mmap_mode="r")
    med = np.median(np.asarray(M), axis=0)
    ins = plain["instruct-chat"]; change = med - ins
    for floor in (1e-5, 1e-6):
        w = np.where(ins >= np.log(floor))[0]
        row = {"words": int(len(w)),
               "change_vs_plain_spearman": spearman(change[w], ins[w]),
               "partial_with_base_minus_instruct": partial(change[w], plain["base-chat"][w] - ins[w], ins[w]),
               "partial_with_sft_minus_instruct": partial(change[w], plain["sft-chat"][w] - ins[w], ins[w]),
               "pushed_vs_base": spearman(med[w], plain["base-chat"][w]),
               "plain_vs_base": spearman(ins[w], plain["base-chat"][w]),
               "pushed_vs_sft": spearman(med[w], plain["sft-chat"][w]),
               "plain_vs_sft": spearman(ins[w], plain["sft-chat"][w])}
        out[f"q{q}_floor{floor}"] = row
        print(q, floor, {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})
    w = np.where(ins >= np.log(1e-5))[0]
    order = w[np.argsort(-change[w])]
    show = lambda ids: [(vocab[int(i)].replace("Ġ", " "), round(float(change[i]), 2), f"{np.exp(ins[i]):.1e}", f"{np.exp(plain['base-chat'][i]):.1e}", f"{np.exp(med[i]):.1e}") for i in ids]
    print("risers", show(order[:14])); print("fallers", show(order[::-1][:6]))
    out[f"q{q}_risers"] = show(order[:30]); out[f"q{q}_fallers"] = show(order[::-1][:30])
json.dump(out, open("results/part145/explore.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
