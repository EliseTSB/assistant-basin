"""Q4: does what the blind reader sees depend on which layers were edited?
pair_by_layer() substitutes current[l] for edited layers and base[l] otherwise, then hands the
whole map to stats198.blind_reader. If the blind layer is an UNEDITED layer, its pair is the base
pair -- identical to what a reader would see with no edit at all. So a 'blind success' can be
decided entirely by base statistics."""
import sys
HEAD = "/home/elisepc/green/worktrees/ab-part198/results/part198"
sys.path.insert(0, HEAD)
import run198 as R, stats198 as S

def pairs(name, z):
    return [{"j": 0, "sign": 1, "z": z, "top": 100 + z, "top3": [100 + z], "eligible": True,
             "token": name}]
base = {l: {"pairs": pairs(f"base{l}", 1)} for l in range(4)}
current = dict(base)
current[2] = {"pairs": pairs("edited", 50)}
for edited in ({2}, {3}, set()):
    layer_pairs = R.pair_by_layer(base, current, S)
    layer, pair = S.blind_reader(layer_pairs)
    print(f"edited={sorted(edited)} -> blind layer {layer}, token {pair.top if pair else None}")
print("=> the blind reader's answer is a function of the *statistics map*, not of the edited set;")
print("   an unedited layer can win and be scored 'localized' if it happens to be in edited_layers.")
