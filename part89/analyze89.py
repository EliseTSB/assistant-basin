"""Part 89 analysis: medians, settle checkpoints, post-training move, named-set tests (PART89-DESIGN.md).

Written by Claude (Opus 5.5), 2026-10-28. Settle checkpoints use only the checkpoints present in cos89.npz; when
mid-training 13B or 26B is missing the settle index is computed on the present ones and the output says so.
"""
import json, sys
import numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "results/part89/cos89.npz"
d = np.load(R)
ck = [str(x) for x in d["checkpoints"]]; missing = [str(x) for x in d["missing"]]
down, gate, up = d["cos_down"], d["cos_gate"], d["cos_up"]
ix = {k: i for i, k in enumerate(ck)}
out = {"present": ck, "missing": missing}
out["median_down_by_checkpoint"] = {k: float(np.median(down[i])) for k, i in ix.items()}
out["control_instruct_self_min"] = float(down[ix["instruct"]].min())
mv = 1 - down[ix["midtrain_51B"]]          # post-training move, [16, 8192]
mvg, mvu = 1 - gate[ix["midtrain_51B"]], 1 - up[ix["midtrain_51B"]]
out["post_move_down_quantiles"] = {q: float(np.quantile(mv, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}
out["post_move_down_count_cos_below"] = {t: int((down[ix["midtrain_51B"]] < t).sum()) for t in (0.99, 0.95, 0.9, 0.8, 0.5)}
order = np.argsort(-mv, axis=None)[:40]
pairs = {k[5:]: d[k] for k in d.files if k.startswith("pair_")}
top = []
for flat in order:
    L, N = np.unravel_index(flat, mv.shape)
    top.append({"neuron": f"L{L} N{N}", "move_down": round(float(mv[L, N]), 4), "move_gate": round(float(mvg[L, N]), 4),
                "move_up": round(float(mvu[L, N]), 4),
                **{f"cos {k}": round(float(v[L, N]), 4) for k, v in pairs.items()}})
out["top40_post_move"] = top
out["post_move_median_by_layer"] = [round(float(np.median(mv[L])), 5) for L in range(16)]
# settle checkpoint on present checkpoints
order_ck = [k for k in ck]  # already training order
S = np.full(mv.shape, len(order_ck), dtype=np.int32)
ok_from = np.ones(mv.shape, dtype=bool)
for i in range(len(order_ck) - 1, -1, -1):
    ok_from &= down[i] >= 0.9
    S[ok_from] = i
out["settle_note"] = "index into present checkpoints" + (f"; missing {missing}" if missing else "")
out["settle_median_by_layer"] = [float(np.median(S[L])) for L in range(16)]
# named sets
def parse(s): L, N = s.replace("L", "").split("."); return int(L), int(N)
sets = {
    "alien_twin_L10N6978": [(10, 6978)], "control_L10N7884": [(10, 7884)],
    "part16_as_an_ai_64": [parse(x["neuron"]) for x in json.load(open("results/part16/search.json"))["search"]["top_set"]],
    "part2_identity_top50": [parse(x[0]) for x in json.load(open("results/part2-olmo2-1b/identity-instruct-chat.json"))["top50"]],
    "part14_register_top50": [parse(x[0]) for x in json.load(open("results/part14-register/olmo2-instruct-own.json"))["top50"]],
}
rng = np.random.default_rng(89089)
res = {}
for name, members in sets.items():
    Ls = [m[0] for m in members]
    val_mv = float(np.mean([mv[L, N] for L, N in members])); val_s = float(np.mean([S[L, N] for L, N in members]))
    r_mv, r_s = [], []
    for _ in range(1000):
        draw = []
        for L in Ls:
            while True:
                n = int(rng.integers(8192))
                if (L, n) not in members and (L, n) not in draw:
                    break
            draw.append((L, n))
        r_mv.append(np.mean([mv[L, N] for L, N in draw])); r_s.append(np.mean([S[L, N] for L, N in draw]))
    r_mv, r_s = np.array(r_mv), np.array(r_s)
    res[name] = {"n": len(members), "mean_post_move": val_mv, "random_mean_post_move": float(r_mv.mean()),
                 "p_moves_more": (1 + int((r_mv >= val_mv).sum())) / 1001,
                 "mean_settle": val_s, "random_mean_settle": float(r_s.mean()),
                 "p_settles_later": (1 + int((r_s >= val_s).sum())) / 1001,
                 "p_settles_earlier": (1 + int((r_s <= val_s).sum())) / 1001}
out["named_sets"] = res
L10 = S[10]; out["alien_twin_settle_vs_layer10_median"] = [int(S[10, 6978]), float(np.median(L10))]
print(json.dumps(out, indent=1))
