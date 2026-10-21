"""Part 39 section D, amendment 1: the bounded AUC screen, rescored from the saved reads (no model run).

Measures, per unit, the rank AUC of categories 1 and 3 (request, AI self-description) against 2, 4, 5, 6 on the plain
sentence maxima, gated by a 200-shuffle max-AUC null, top 1% on both sentence halves, and a 0.5 median floor.
What went wrong the first time (part39.py screen): a contrast divided by each unit's MAD, floored at 0.01, let
near-silent units score 100+, set the null at 96.6 and failed the L6 N753 sanity check. AUC is bounded, so it cannot.
Known weakness: this statistic was chosen after seeing N753's values (recorded in PART39-DESIGN.md amendment 1).
"""
import argparse, json, sys
import numpy as np
import torch

CATS = ["help_request", "second_person_statement", "ai_self_description", "human_self_description",
        "third_person_ai", "neutral_prose"]
POS, NEG = (0, 2), (1, 3, 4, 5)


def auc(values, pos_mask):
    """Rank AUC per column (average ranks for ties): P(a positive row beats a negative row), ties counted half."""
    v = torch.from_numpy(np.ascontiguousarray(values.T))              # [columns, rows]
    s = v.sort(dim=1).values
    left = torch.searchsorted(s, v, right=False); right = torch.searchsorted(s, v, right=True)
    ranks = ((left + right + 1).double() / 2).numpy().T              # [rows, columns], 1-based average ranks
    n_pos = int(pos_mask.sum()); n_neg = len(pos_mask) - n_pos
    return (ranks[pos_mask].sum(axis=0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz"); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3901); ap.add_argument("--shuffles", type=int, default=200)
    ap.add_argument("--floor", type=float, default=0.5)
    a = ap.parse_args()
    z = np.load(a.npz)
    meta = json.loads(str(z["metadata"]))
    assert meta["categories"] == CATS and meta["wrappings"][0] == "plain"
    width = z["sentence_max"].shape[3]
    x = z["sentence_max"][:, 0].astype(np.float32).reshape(120, -1)   # plain reads, [sentence, layer*unit]
    cat = np.repeat(np.arange(6), 20)
    idx = np.tile(np.arange(20), 6)
    pos = np.isin(cat, POS)
    full = auc(x, pos)
    rng = np.random.default_rng(a.seed)
    null_max = np.array([auc(x, rng.permutation(pos)).max() for _ in range(a.shuffles)])
    thr = float(np.quantile(null_max, 0.99))
    halves = []
    for h in (idx < 10, idx >= 10):
        au = auc(x[h], pos[h]); halves.append(au >= np.quantile(au, 0.99))
    med_pos = np.median(x[pos], axis=0)
    passing = np.where((full > thr) & halves[0] & halves[1] & (med_pos >= a.floor))[0]

    def pair_auc(cols, p, n):
        rows = np.isin(cat, p + n)
        return auc(x[rows][:, cols], np.isin(cat[rows], p))
    units = []
    if len(passing):
        req = pair_auc(passing, (0,), (1,)); ais = pair_auc(passing, (2,), (3,)); third = pair_auc(passing, (0, 2), (4,))
        for j, c in enumerate(passing):
            units.append({"layer": int(c // width), "unit": int(c % width), "addressee_auc": float(full[c]),
                          "request_only_auc": float(req[j]), "ai_self_only_auc": float(ais[j]),
                          "vs_third_person_ai_auc": float(third[j]), "median_cats_1_3": float(med_pos[c]),
                          "category_medians": [round(float(np.median(x[cat == k, c])), 3) for k in range(6)]})
    t = 6 * width + 753
    out = {"tool": "part39_auc.py (Part 39 section D, amendment 1)", "model": meta["model"],
           "revision": meta["revision"], "seed": a.seed, "shuffles": a.shuffles, "threshold_99": thr,
           "null_max_median": float(np.median(null_max)), "floor": a.floor,
           "target": {"layer": 6, "unit": 753, "addressee_auc": float(full[t]),
                      "passes": bool(t in set(passing.tolist())), "median_cats_1_3": float(med_pos[t]),
                      "top1pct_both_halves": bool(halves[0][t] and halves[1][t])},
           "passing_unit_count": len(units), "passing_units": sorted(units, key=lambda u: -u["addressee_auc"])}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("model", "threshold_99", "null_max_median", "target", "passing_unit_count")}))


if __name__ == "__main__":
    sys.exit(main())
