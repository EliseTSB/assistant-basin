"""Probe: does the blind reader's input depend on which layers were edited? pair_by_layer walks `base` only,
so the reader sees the same 24 (or 48) layers whatever the editor touched."""
import sys, inspect
sys.path.insert(0, "/home/elisepc/green/worktrees/ab-part198/results/part198")
import run198 as R
S = R.load_stats_module()
e = lambda z, t: S.Pair(0, 1, z, t, [t], True)
base = {l: {"pairs": [R.pair_to_dict(e(3.0, 5))]} for l in range(24)}
current = {17: {"pairs": [R.pair_to_dict(e(7.0, 9))]}}
allpairs = R.pair_by_layer(base, current, S)
print("layers passed to blind_reader:", len(allpairs), "= len(base); edited =", sorted(current))
print("blind_reader ->", S.blind_reader(allpairs)[0])
print("blind_reader signature:", inspect.signature(S.blind_reader))
print("pair_by_layer signature:", inspect.signature(R.pair_by_layer))
# a layer the editor did not touch, but whose base value is highest, wins:
base2 = {l: {"pairs": [R.pair_to_dict(e(20.0, 5))]} for l in range(24)}
print("base layer 3 at z=20 vs edited layer 17 at z=7 -> blind layer:",
      S.blind_reader(R.pair_by_layer(base2, current, S))[0])
