"""Part 205, Amendment 2, step A0: read the six-value gap screen and take the registered branch.

Usage: python -I part205/screen205.py <screen_out>/<stage> [--json out.json]

Reads records.jsonl written by `run205.py --stage-only screen` (24 plain You rows and 48 plain third-person rows on
the held-out SAY questions) and prints, as Amendment 2 item 1 registers: the gap mean(P_you - P_third) over the 48
pairs, its standard error with the 24 questions as clusters, the gap per value, G0 on the 24 You rows exactly as
score205.gate_g0 reads it, and every P. Then the branch (item 2): PROCEED if G0 passes and the gap is at least 0.194,
otherwise FALLBACK. An incomplete screen (not exactly 24 + 48 rows) is HOLD: no branch is taken (item 1).
The sensitivity rows are printed for the reader and decide nothing.

Written by Claude (Opus 5.5), 2026-11-12, after the amendment was registered (public main dbb1a536).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score205 as S  # noqa: E402
from common205 import read_jsonl  # noqa: E402

THRESHOLD = 0.194          # Amendment 2 item 2
N_YOU, N_PAIRS = 24, 48
SD_D_PILOT, R_PILOT = 0.079, 0.71   # pilot (honesty), for the sensitivity rows only


def screen(stage_dir):
    rows = read_jsonl(Path(stage_dir) / "records.jsonl", repair=False)
    you = [r for r in rows if r.get("arm") == "you"]
    third = [r for r in rows if r.get("arm") == "baseline"]
    ids = sorted({r["id"] for r in you})
    out = {"n_you": len(you), "n_third": len(third), "threshold": THRESHOLD}
    if len(you) != N_YOU or len(third) != N_PAIRS or len(ids) != N_YOU:
        out.update({"branch": "HOLD", "reason": f"incomplete screen: {len(you)} You rows (want {N_YOU}), "
                                                  f"{len(third)} third-person rows (want {N_PAIRS})"})
        return out
    p_you = {r["id"]: S.p_consistent(r) for r in you}
    value = {r["id"]: r["value"] for r in you}
    pairs = [(r["id"], r["name"], p_you[r["id"]] - S.p_consistent(r)) for r in third]
    gap = sum(g for _, _, g in pairs) / len(pairs)
    # clusters: each question's two pairs share its You run
    cm = {}
    for qid, _, g in pairs:
        cm.setdefault(qid, []).append(g)
    means = [sum(v) / len(v) for v in cm.values()]
    G = len(means)
    mbar = sum(means) / G
    sd_c = math.sqrt(sum((m - mbar) ** 2 for m in means) / (G - 1))
    se = sd_c / math.sqrt(G)
    per_value = {}
    for qid, _, g in pairs:
        per_value.setdefault(value[qid], []).append(g)
    g0 = S.gate_g0({"you": you}, set(ids))
    # 1e-12: a gap that equals 0.194 in decimal can come out as 0.19399999999999995 after subtracting probabilities;
    # "at least 0.194" must not turn on float rounding (found by the boundary test). It moves nothing a real gap can hit.
    branch = "PROCEED" if (g0["pass"] and gap >= THRESHOLD - 1e-12) else "FALLBACK"
    sens = []
    for sd in (0.08, 0.12, 0.16):
        for n in (24, 48):
            sens.append({"sd_d": sd, "n": n, "gap_needed": round(3 * sd / (math.sqrt(n) * 0.25), 4)})
    # full ratio delta method at T = 0.25 with the pilot's SD(d) and r and this screen's per-question gap spread
    var_d, var_g = SD_D_PILOT ** 2, sd_c ** 2
    cov = R_PILOT * SD_D_PILOT * sd_c
    num = var_d - 2 * 0.25 * cov + 0.25 ** 2 * var_g
    delta_needed = math.sqrt(max(num, 0.0) / G) / (0.25 / 3)
    out.update({
        "gap": gap, "se_clustered": se, "n_clusters": G,
        "gap_per_value": {k: sum(v) / len(v) for k, v in sorted(per_value.items())},
        "G0": g0, "branch": branch,
        "rule": f"PROCEED if G0 passes and gap >= {THRESHOLD}; otherwise FALLBACK (Amendment 2 item 2)",
        "p_you": p_you, "p_third": {f"{q}|{n}": p_you[q] - g for q, n, g in pairs},
        "sensitivity_for_the_reader": {"registered_formula": sens,
                                       "delta_method_T025_pilot_sd_and_r": round(delta_needed, 4)},
    })
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage_dir")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = screen(a.stage_dir)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2, sort_keys=True))
    if out["branch"] == "HOLD":
        print(f"A0: HOLD — {out['reason']}")
        return 3
    print(f"A0: gap {out['gap']:.4f} (clustered SE {out['se_clustered']:.4f}, {out['n_clusters']} questions); "
          f"G0 pass={out['G0']['pass']} (mass {out['G0']['mean_option_mass']:.3f}, letters {out['G0']['letter_share']:.2f})")
    for k, v in out["gap_per_value"].items():
        print(f"   {k:13s} gap {v:+.4f}")
    print(f"BRANCH: {out['branch']}   ({out['rule']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
