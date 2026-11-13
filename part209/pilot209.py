#!/usr/bin/env python3
"""pilot209.py: Part 209's pilot gates, the max_new_tokens rule and the batch timing (Design, Size and Optimization).

  python -I -B part209/pilot209.py --pilot-dir results/part209/pilot [--batch-dirs results/part209/pilot/b32 ...]

Reads (under --pilot-dir): r1r3-qwen-chat.json from `run209.py --stage qwen-chat --pilot`, r2-qwen-chat.jsonl from the
same with --r2, and labels-<primary reader>.jsonl from read209.py. Writes <pilot-dir>/pilot-gates.json.

Pilot gates, as the Size section writes them:
  G0         as written: mean P3 over the ROOM-W told frames at least 0.3 (6 frames at paraphrases 1 and 2)
  G1, R1     the told posture has the largest of SIT, STAND, LIE in both pilot paraphrases of every told arm
  R3 poles   told sitting projects above told standing in both pilot paraphrases, in ROOM-W and in NOBODY
  compliance at least 10 of the 12 told answers name the told posture (primary reader)
max_new_tokens: among pilot answers the primary reader labels with a posture, the token at which the first posture word
appears (run209 records first_posture_token from the raw new tokens). If every one is at or before 48, max_new_tokens
becomes 64 (an amendment before the main run); an answer labelled with a posture whose posture word the pattern did not
find counts as not confirmed, and the cap stays at 96.
Timing: answers per second per batch size from each r2-<stage>.timing.jsonl, and whether the same rows gave the same
answers at another batch size (per-row seeded generators should make them identical up to bf16 batch arithmetic).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import score209 as S  # noqa: E402

TOLD = {"TOLD-SIT": "sitting", "TOLD-STAND": "standing", "TOLD-LIE": "lying"}


def timing(path):
    rows = S.read_jsonl(path)
    by = {}
    for r in rows:
        d = by.setdefault(r["batch_size"], {"rows": 0, "seconds": 0.0, "new_tokens": 0})
        d["rows"] += r["rows"]
        d["seconds"] += r["seconds"]
        d["new_tokens"] += r.get("new_tokens", 0)
    for d in by.values():
        d["answers_per_s"] = d["rows"] / d["seconds"] if d["seconds"] else None
    return by


def answers_in(d):
    out = {}
    for p in sorted(Path(d).glob("r2-*.jsonl")):
        if p.name.endswith(".timing.jsonl"):
            continue
        for r in S.read_jsonl(p):
            if "answer" in r:
                out[r["row_id"]] = r["answer"]
    return out


def compare_dirs(a, b):
    x, y = answers_in(a), answers_in(b)
    common = sorted(set(x) & set(y))
    return {"common_rows": len(common), "identical": sum(1 for k in common if x[k] == y[k])}


def pilot_gates(pilot_dir, cfg, stage="qwen-chat", batch_dirs=(), log=print, timing_dirs=(), compare_pairs=()):
    pilot_dir = Path(pilot_dir)
    pc = cfg["pilot"]
    out = {"stage": stage}
    recp = pilot_dir / f"r1r3-{stage}.json"
    if recp.exists():
        rec = S.load_json(recp)
        g0 = S.gate_g0(rec, cfg)
        g0["pass"] = bool(g0["n_told_frames"]) and g0["mean_P3_told"] >= pc["gates"]["G0_p3"]
        g1 = S.gate_g1_r1(rec, cfg, need=pc["gates"]["G1_r1_told_wins"])
        pk = S.proj_by_arm_k(rec)
        r3 = {}
        ok3 = True
        for name, (one, zero) in (("ROOM-W", S.ROOM_POLES), ("NOBODY", S.NOBODY_POLES)):
            above, n = S.pole_pairs(pk, one, zero)
            r3[name] = {"above": above, "of": n}
            ok3 = ok3 and n > 0 and above >= pc["gates"]["G1_r3_pairs"]
        out.update({"G0": g0, "G1_r1": g1, "R3_poles": {**r3, "pass": ok3,
                    "heldout_accuracy": rec["r3"]["sit_stand"]["chosen_accuracy"],
                    "chosen_layer": rec["r3"]["sit_stand"]["chosen_layer"]}})
        log(f"[pilot] G0 mean P3 {g0['mean_P3_told']:.3f} over {g0['n_told_frames']} -> {g0['pass']}")
        log(f"[pilot] G1 R1 " + json.dumps(g1))
        log(f"[pilot] R3 poles {r3} -> {ok3} (held-out {out['R3_poles']['heldout_accuracy']:.3f}, "
            f"layer {out['R3_poles']['chosen_layer']})")
    else:
        log(f"[pilot] no {recp.name}")
    rows = [r for r in S.read_jsonl(pilot_dir / f"r2-{stage}.jsonl") if "answer" in r]
    labels = S.last_labels(pilot_dir / f"labels-{cfg['readers']['primary']['name']}.jsonl")
    out["answers"] = len(rows)
    told = [r for r in rows if r["arm"] in TOLD]
    lab_told = [r for r in told if r["row_id"] in labels]
    comp = sum(1 for r in lab_told if S.category(labels[r["row_id"]]) == TOLD[r["arm"]])
    need = pc["gates"]["told_compliance"]
    out["compliance"] = {"right": comp, "labelled": len(lab_told), "told_answers": len(told),
                         "pass": len(lab_told) == need["of"] and comp >= need["need"]}
    log(f"[pilot] told compliance {comp}/{len(lab_told)} labelled ({len(told)} told answers) -> "
        f"{out['compliance']['pass']}")
    # max_new_tokens rule
    rule = pc["max_new_tokens_rule"]
    named = [r for r in rows if r["row_id"] in labels and labels[r["row_id"]].get("posture") in
             ("sitting", "standing", "lying", "other") and not labels[r["row_id"]].get("unreadable")]
    positions = {r["row_id"]: r.get("first_posture_token") for r in named}
    unconfirmed = sorted(k for k, v in positions.items() if v is None)
    late = sorted(k for k, v in positions.items() if v is not None and v > rule["within"])
    found = [v for v in positions.values() if v is not None]
    cut = rule["then"] if named and not unconfirmed and not late else None
    out["max_new_tokens"] = {"answers_naming_a_posture": len(named), "max_position": max(found) if found else None,
                             "unconfirmed": unconfirmed, "later_than_rule": late,
                             "all_tokens": positions, "new_cap": cut,
                             "decision": (f"max_new_tokens becomes {cut}" if cut else "max_new_tokens stays 96")}
    log(f"[pilot] first posture token: {len(named)} answers name a posture, max position "
        f"{out['max_new_tokens']['max_position']}, unconfirmed {len(unconfirmed)}, later than "
        f"{rule['within']}: {len(late)} -> {out['max_new_tokens']['decision']}")
    # timing and batch-shape check
    out["timing"] = {str(pilot_dir): timing(pilot_dir / f"r2-{stage}.timing.jsonl")}
    base = {r["row_id"]: r["answer"] for r in rows}
    out["batch_check"] = {}
    for d in batch_dirs:
        d = Path(d)
        st_rows = answers_in(d)
        for p in sorted(d.glob("r2-*.timing.jsonl")):
            out["timing"][str(p)] = timing(p)
        common = sorted(set(base) & set(st_rows))
        same = sum(1 for k in common if base[k] == st_rows[k])
        out["batch_check"][str(d)] = {"common_rows": len(common), "identical": same}
    for d in timing_dirs:
        for p in sorted(Path(d).glob("r2-*.timing.jsonl")):
            out["timing"][str(p)] = timing(p)
    for a_, b_ in compare_pairs:
        out["batch_check"][f"{a_} vs {b_}"] = compare_dirs(a_, b_)
    for k, v in out["timing"].items():
        log(f"[pilot] timing {k}: " + json.dumps(v))
    for k, v in out["batch_check"].items():
        log(f"[pilot] same rows at another batch size {k}: {v['identical']}/{v['common_rows']} identical")
    gates = [out.get("G0", {}).get("pass"), out.get("G1_r1", {}).get("pass"),
             out.get("R3_poles", {}).get("pass"), out["compliance"]["pass"]]
    out["pilot_gates_pass"] = all(bool(g) for g in gates)
    log(f"[pilot] pilot gates {'PASS' if out['pilot_gates_pass'] else 'FAIL'} (read the 76 answers by eye too)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-dir", default=str(HERE.parent / "results" / "part209" / "pilot"))
    ap.add_argument("--stage", default="qwen-chat")
    ap.add_argument("--batch-dirs", nargs="*", default=[],
                    help="R2 runs of the pilot's rows at another batch size: timing, and identical answers checked")
    ap.add_argument("--timing-dirs", nargs="*", default=[], help="timing only (e.g. OLMo Instruct at batch 16, 32)")
    ap.add_argument("--compare-pair", nargs=2, action="append", default=[], metavar=("DIR_A", "DIR_B"),
                    help="check that two R2 runs gave identical answers on their common rows")
    a = ap.parse_args(argv)
    cfg = S.load_json(HERE / "config.json")
    res = pilot_gates(a.pilot_dir, cfg, a.stage, a.batch_dirs, timing_dirs=a.timing_dirs,
                      compare_pairs=a.compare_pair)
    (Path(a.pilot_dir) / "pilot-gates.json").write_text(json.dumps(res, indent=1, default=float) + "\n",
                                                        encoding="utf-8")
    print(f"wrote {Path(a.pilot_dir) / 'pilot-gates.json'}")


if __name__ == "__main__":
    main()
