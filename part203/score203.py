"""Part 203: score the registered predictions 203.1-203.7 and 203.W1-W4 from results/part203/results.json.

Every number printed is read from the results file and shown as `value <- results.json:key`; nothing is typed. The
prediction wording is PART203-DESIGN.md's table. A row with two clauses holds only if both hold. Weaver's rows are
scored by their forecast; each falsifier is printed separately (the design's rule beside the table).

Usage: python -I -B part203/score203.py [--results results/part203/results.json] [--out results/part203]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHANCE = 1 / 6


def read(results, dotted):
    node = results
    for part in dotted.split("."):
        node = node[part]
    return node


def score(results):
    lines, verdicts = [], {}

    def val(key):
        value = read(results, key)
        lines.append(f"  {key}: {value!r}  <- results.json:{key}")
        return value

    def verdict(pid, held, note=""):
        verdicts[pid] = "held" if held else "failed"
        lines.append(f"{pid}: {verdicts[pid]}{('  (' + note + ')') if note else ''}")
        lines.append("")

    if results.get("status") != "complete":
        raise SystemExit(f"status is {results.get('status')!r}: the gates stop the part before any readout is read")
    lines.append(f"gate1 {val('gate1.verdict')}, gate2 {val('gate2.verdict')}")
    lines.append("")

    v = val("readouts.V_grid.mean_balanced_accuracy")
    v_low = val("readouts.V_grid.balanced_accuracy_actor_cluster_95.low")
    verdict("203.1", v >= 0.45 and v_low > CHANCE)

    d = val("readouts.D_grid.mean_balanced_accuracy")
    d_low = val("readouts.D_grid.balanced_accuracy_actor_cluster_95.low")
    verdict("203.2", abs(d - v) <= 0.03, f"|D_grid - V_grid| = {abs(d - v):.4f}; both lower bounds above 1/6: "
                                          f"{v_low > CHANCE and d_low > CHANCE}")

    diff = val("paired_differences.V_grid_minus_V_p196.mean_difference")
    chosen = [read(fold, "V_grid.c_selection.chosen_C") for _name, fold in sorted(results["folds"].items())]
    lines.append(f"  V_grid chosen C per fold: {chosen}  <- results.json:folds.*.V_grid.c_selection.chosen_C")
    high_c = sum(c in (10.0, 100.0) for c in chosen)
    verdict("203.3", diff >= 0.03 and high_c >= 3,
            f"difference clause {'held' if diff >= 0.03 else 'failed'} ({diff:+.4f}); "
            f"C of 10 or 100 on {high_c} of {len(chosen)} folds")

    ed = val("paired_differences.E_p196_minus_D_p196.mean_difference")
    verdict("203.4", ed >= -0.02)

    h = val("readouts.H_p196.mean_balanced_accuracy")
    verdict("203.5", 0.30 <= h <= 0.45)

    v_all = val("three_actor_ablation.V.all_training_actors_mean")
    v_three = val("three_actor_ablation.V.three_actor_draws_mean")
    verdict("203.6", v_all - v_three >= 0.10, f"V drop {v_all - v_three:.4f}")

    e_all = val("three_actor_ablation.E.all_training_actors_mean")
    e_three = val("three_actor_ablation.E.three_actor_draws_mean")
    gap = (v_all - v_three) - (e_all - e_three)
    verdict("203.7", gap >= 0.05, f"V drop {v_all - v_three:.4f} minus E drop {e_all - e_three:.4f} = {gap:.4f}")

    lines.append("Weaver's rows: scored by forecast; falsifier reported separately.")
    lines.append("")
    w1 = v - d
    verdict("203.W1", w1 > 0.03, f"V_grid - D_grid = {w1:.4f}; falsifier (D within 0.03 of V) "
                                  f"{'reached' if abs(w1) <= 0.03 else 'not reached'}")
    verdict("203.W2", ed >= 0.10, f"falsifier (E within 0.05 of D) {'reached' if abs(ed) <= 0.05 else 'not reached'}")
    verdict("203.W3", h < 0.30, f"falsifier (H at 0.35 or above) {'reached' if h >= 0.35 else 'not reached'}")
    vh = val("paired_differences.V_grid_minus_H_p196.mean_difference")
    verdict("203.W4", vh < 0.10, f"V_grid - H_p196 = {vh:.4f}; falsifier (V minus H at 0.15 or more) "
                                  f"{'reached' if vh >= 0.15 else 'not reached'}")
    return lines, verdicts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(ROOT / "results/part203/results.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part203"))
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    lines, verdicts = score(results)
    text = "\n".join(lines)
    print(text)
    out = Path(args.out)
    (out / "score.md").write_text("# Part 203 score (printed by part203/score203.py)\n\n```\n" + text + "\n```\n",
                                  encoding="utf-8")
    (out / "score.json").write_text(json.dumps({"verdicts": verdicts}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
