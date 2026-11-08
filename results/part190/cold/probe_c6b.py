"""Per-clip mechanism of the B_to_A C6 rise: for the 37 imitations that became H-nearest, where did their
alpha=0 nearest real clip sit (rank among real recordings), and how did their cosines to H vs R/C_R move?"""
import json, sys
from pathlib import Path
ROOT = Path("/home/elisepc/green/worktrees/ab-part190v2"); sys.path.insert(0, str(ROOT))
import numpy as np
from part190 import analyze190 as A
res = json.loads((ROOT / "results/part190/results.json").read_text())
data, _ = A.load_data(ROOT / "results/part188/emb188.npz", ROOT / "part188/manifest188.csv")
for name, ff, tf in A.DIRECTIONS:
    ctx = A.make_context(data, name, ff, tf); x_fit = data.x[ctx.fit_idx]
    main = A.build_edit(x_fit, ctx.fit_y, A.read_directions(x_fit, ctx.fit_y, 8, A.fit_logistic_l2),
                        A.verdict_direction(data, ctx.fit_idx)).truncated(1)
    a = res["directions"][name]["ranks"]["1"]["operating_dose"]; xp = main.apply(data.x, a)
    im, pool = ctx.test_imitation_idx, ctx.real_pool_idx
    s0 = data.x[im] @ data.x[pool].T; s1 = xp[im] @ xp[pool].T
    h = data.analysis_set[pool] == "H"
    n0, n1 = pool[s0.argmax(1)], pool[s1.argmax(1)]
    new = (data.analysis_set[n0] != "H") & (data.analysis_set[n1] == "H")
    pos = {c: j for j, c in enumerate(pool.tolist())}
    rank0 = np.array([(s0[i][h] > s0[i][pos[n0[i]]]).sum() + 1 for i in range(im.size)])  # rank of nearest H at alpha=0
    print(f"== {name}: {new.sum()} imitations became H-nearest")
    print("  their alpha=0 rank of nearest H clip:", np.bincount(rank0[new], minlength=6)[1:6], "(ranks 1..5)")
    print("  their cos change to nearest H : %+.3f pt" % (100*(s1[new][:, h].max(1) - s0[new][:, h].max(1)).mean()))
    print("  their cos change to nearest nonH: %+.3f pt" % (100*(s1[new][:, ~h].max(1) - s0[new][:, ~h].max(1)).mean()))
    print("  median displacement of these clips: %.4f vs all imitations %.4f" % (
        np.median(main.mean_displacement(data.x[im[new]], a)), main.mean_displacement(data.x[im], a)))
