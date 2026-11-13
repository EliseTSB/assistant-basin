"""Cold check of results/part203/README.md: recompute every number from results.json independently of score203.py."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
res = json.loads((ROOT / "results/part203/results.json").read_text(encoding="utf-8"))
R = res["readouts"]
P = res["paired_differences"]
T = res["three_actor_ablation"]
folds = sorted(res["folds"], key=int)
CH = 1 / 6


def r(x, n=3):
    return round(x, n)


def ba(k):
    return R[k]["mean_balanced_accuracy"]


def lo(k):
    return R[k]["balanced_accuracy_actor_cluster_95"]["low"]


def hi(k):
    return R[k]["balanced_accuracy_actor_cluster_95"]["high"]


def auc(k):
    return R[k]["mean_macro_ovr_auc"]


print("chance", CH)
print("n_clips", res["n_clips"], "gate1", res["gate1"]["verdict"], res["gate1"]["recorded_ok"], "/", res["gate1"]["planned"],
      "problems", res["gate1"]["problems"], "gate2", res["gate2"]["verdict"])
for k in ["V_grid", "D_grid", "V_p196", "D_p196", "E_p196", "H_p196"]:
    print(f"{k}: BA {ba(k):.4f} [{lo(k):.4f}, {hi(k):.4f}] AUC {auc(k):.4f}")

print()
print("--- verdicts from the design wording ---")
print("203.1", ba("V_grid") >= 0.45 and lo("V_grid") > CH)
print("203.2 |D-V|", abs(ba("D_grid") - ba("V_grid")), "<=0.03:", abs(ba("D_grid") - ba("V_grid")) <= 0.03)
chosen_v = [res["folds"][f]["V_grid"]["c_selection"]["chosen_C"] for f in folds]
chosen_d = [res["folds"][f]["D_grid"]["c_selection"]["chosen_C"] for f in folds]
print("203.3 diff", P["V_grid_minus_V_p196"]["mean_difference"], P["V_grid_minus_V_p196"]["actor_cluster_95"],
      "V chosen C", chosen_v, "D chosen C", chosen_d)
print("203.4 E-D", P["E_p196_minus_D_p196"]["mean_difference"], P["E_p196_minus_D_p196"]["actor_cluster_95"])
print("203.5 H", ba("H_p196"))
print("203.6 V drop", T["V"]["all_training_actors_mean"] - T["V"]["three_actor_draws_mean"])
print("203.7 V drop - E drop", (T["V"]["all_training_actors_mean"] - T["V"]["three_actor_draws_mean"])
      - (T["E"]["all_training_actors_mean"] - T["E"]["three_actor_draws_mean"]))
print("W1 V-D", ba("V_grid") - ba("D_grid"))
print("W2 E-D", P["E_p196_minus_D_p196"]["mean_difference"])
print("W3 H", ba("H_p196"))
print("W4 V-H", P["V_grid_minus_H_p196"]["mean_difference"], P["V_grid_minus_H_p196"]["actor_cluster_95"])

print()
print("--- README prose numbers ---")
print("V/D per-fold BA", [r(R["V_grid"]["per_fold"][f]["balanced_accuracy"], 4) for f in folds])
print("D per-fold BA", [r(R["D_grid"]["per_fold"][f]["balanced_accuracy"], 4) for f in folds])
print("E per-fold BA", [r(R["E_p196"]["per_fold"][f]["balanced_accuracy"], 4) for f in folds])
print("H per-fold BA", [r(R["H_p196"]["per_fold"][f]["balanced_accuracy"], 4) for f in folds])
print("C=1 V_p196 / D_p196", r(ba("V_p196"), 4), r(ba("D_p196"), 4))
print("D_grid - D_p196", P["D_grid_minus_D_p196"]["mean_difference"])
em = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
for k in ["V_grid", "E_p196"]:
    m = {e: sum(R[k]["per_fold"][f]["per_emotion_recall"][e] for f in folds) / len(folds) for e in em}
    print(k, "recall by emotion", {e: r(m[e]) for e in em})
print("E layer per fold", [res["folds"][f]["E"]["selected_layer"] for f in folds])
print("fold3 inner layer0/layer1", res["folds"]["3"]["E"]["inner_balanced_accuracy"][0],
      res["folds"]["3"]["E"]["inner_balanced_accuracy"][1])
for L in (0, 23):
    print(f"inner mean layer {L}", sum(res["folds"][f]["E"]["inner_balanced_accuracy"][L] for f in folds) / 5)
print("E-V per fold", [r(R["E_p196"]["per_fold"][f]["balanced_accuracy"] - R["V_grid"]["per_fold"][f]["balanced_accuracy"], 4) for f in folds])
print("E-V mean", ba("E_p196") - ba("V_grid"), "E-D mean", ba("E_p196") - ba("D_grid"))
print("fold held-out actor counts", {f: len(res["gate1"]["folds"][f]) for f in folds})
