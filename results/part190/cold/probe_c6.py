"""Measure what C6/190.10 actually shows, from the REAL code and the REAL arrays.

C6 (analyze190.evaluate) = share of held-out imitations whose nearest neighbour among real recordings
(R, C_R, E, H) is an H clip. README/headline read a rise in that share as the edit "moving imitations toward
human sounds". This script compares, at the rank-1 operating dose:
  (a) the C6 share reproduced from the real code (sanity: must equal results.json's C6_human_share),
  (b) mean cosine of held-out imitations to the nearest H clip and to the nearest R/C_R clip, before vs after,
  (c) the change in each imitation's cosine to the single nearest H clip it had at alpha=0 (did it move toward
      that human sound, or away from it?).
Writes nothing outside this directory.
"""
import json
import sys
from pathlib import Path

ROOT = Path("/home/elisepc/green/worktrees/ab-part190v2")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from part190 import analyze190 as A  # noqa: E402

res = json.loads((ROOT / "results/part190/results.json").read_text())
data, rows = A.load_data(ROOT / "results/part188/emb188.npz", ROOT / "part188/manifest188.csv")
folds = json.loads((ROOT / "part188/folds.json").read_text())
reg = json.loads((ROOT / "results/part188/results.json").read_text())

for name, fit_fold, test_fold in A.DIRECTIONS:
    ctx = A.make_context(data, name, fit_fold, test_fold)
    x_fit = data.x[ctx.fit_idx]
    v_bar = A.verdict_direction(data, ctx.fit_idx)
    main = A.build_edit(x_fit, ctx.fit_y, A.read_directions(x_fit, ctx.fit_y, 8, A.fit_logistic_l2), v_bar).truncated(1)
    a_star = res["directions"][name]["ranks"]["1"]["operating_dose"]
    x_post = main.apply(data.x, a_star)
    im = ctx.test_imitation_idx
    H = data.where("H")
    RC = data.where(("R", "C_R"))

    def c6(x):
        pool = ctx.real_pool_idx
        nn = pool[np.argmax(x[im] @ x[pool].T, axis=1)]
        return float((data.analysis_set[nn] == "H").mean())

    committed = res["directions"][name]["ranks"]["1"]["operating"]["C6_human_share"]
    print(f"== {name}  alpha*={a_star:g}  n imitations={im.size}")
    print(f"  C6 share from real code, alpha=0 : {c6(data.x):.4f}")
    print(f"  C6 share from real code, edited   : {c6(x_post):.4f}   (results.json: {committed:.4f})")

    # (b) mean cosine to nearest H and nearest R/C_R, before vs after
    for label, pool in (("nearest H clip", H), ("nearest R/C_R clip", RC)):
        c0 = (data.x[im] @ data.x[pool].T).max(axis=1)
        c1 = (x_post[im] @ x_post[pool].T).max(axis=1)
        print(f"  mean cos to {label:18s}: before {c0.mean():+.4f}  after {c1.mean():+.4f}  change {100*(c1-c0).mean():+.3f} pt")

    # (c) did each imitation move TOWARD the human clip that was its nearest H at alpha=0?
    j0 = np.argmax(data.x[im] @ data.x[H].T, axis=1)
    before = np.einsum("ij,ij->i", data.x[im], data.x[H][j0])
    after = np.einsum("ij,ij->i", x_post[im], x_post[H][j0])
    d = after - before
    print(f"  cos to own alpha=0 nearest H clip: mean change {100*d.mean():+.3f} pt; "
          f"share of imitations moved TOWARD it {100*(d > 0).mean():.1f}%, away {100*(d < 0).mean():.1f}%")
    # how many imitations whose nearest real was H at alpha=0 stopped being so / vice versa
    pool = ctx.real_pool_idx
    n0 = pool[np.argmax(data.x[im] @ data.x[pool].T, axis=1)]
    n1 = pool[np.argmax(x_post[im] @ x_post[pool].T, axis=1)]
    h0, h1 = data.analysis_set[n0] == "H", data.analysis_set[n1] == "H"
    print(f"  H-nearest: {h0.sum()} -> {h1.sum()} clips; became H-nearest {(~h0 & h1).sum()}, stopped being H-nearest {(h0 & ~h1).sum()}")
