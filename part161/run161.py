#!/usr/bin/env python3
"""Part 161 runner: ladder matched random directions under Part 160's pick rule."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part160 import run160 as part160  # noqa: E402
from part153 import score153  # noqa: E402
from part155 import run155 as part155  # noqa: E402

PICK_DOSES = (0.6, 1.0)
LADDER_DOSES = (0.4, 0.6, 0.8, 1.0, 1.2, 1.4)
MAX_NEW_TOKENS = 80
MODEL_DEFAULT = part160.MODEL_DEFAULT


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as fh:
        for no, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{no}") from exc
    return rows


def plan_units(screen_rows, picked, old_rows, validate_gate3=True):
    """Return the complete registered plan and the rows this runner must generate."""
    neuron_maps = part160.picking_rule([r for r in screen_rows if r.get("arm") == "neuron"])
    neurons_by_dose = {float(d): neuron_maps[float(d)] for d in PICK_DOSES}
    neuron_union = {}
    for dose in PICK_DOSES:
        for uid, company in neurons_by_dose[dose].items():
            source = next((r for r in screen_rows if r.get("arm") == "neuron" and r.get("unit_id") == uid), None)
            if source is None:
                raise ValueError(f"neuron screen unit {uid} has no source row")
            neuron_union.setdefault(uid, {"unit_id": uid, "layer": int(source["layer"]),
                                          "neuron": int(source["neuron"]),
                                          "companies_by_picking_dose": {}, "picking_doses": []})
            neuron_union[uid]["companies_by_picking_dose"][str(dose)] = company
            neuron_union[uid]["picking_doses"].append(dose)

    random_maps = part160.picking_rule(
        [dict(r, arm="neuron") for r in screen_rows if r.get("arm") == "random"])
    random_union = {}
    for dose in PICK_DOSES:
        for uid, company in random_maps[dose].items():
            # The screen stores both matched directions on each source coordinate. Unit IDs
            # are coordinates, so a random pick and its neuron have the same LxNy label.
            sample = next((r for r in screen_rows if r.get("arm") == "random" and
                           r.get("unit_id") == uid), None)
            if sample is None:
                raise ValueError(f"random screen unit {uid} has no source row")
            random_union.setdefault(uid, {"unit_id": uid, "layer": int(sample["layer"]),
                                          "neuron": int(sample["neuron"]),
                                          "companies_by_picking_dose": {}, "picking_doses": []})
            random_union[uid]["companies_by_picking_dose"][str(dose)] = company
            random_union[uid]["picking_doses"].append(dose)

    old_neurons = {r.get("unit_id") for r in old_rows
                   if r.get("stage") == "ladder" and r.get("arm") == "neuron"}
    new_neurons = set(neuron_union) - old_neurons
    if validate_gate3 and (len(random_union) != 45 or len(new_neurons) != 17):
        raise ValueError(f"Gate 3 failed: expected 45 distinct random directions and 17 new neurons; "
                         f"found {len(random_union)} and {len(new_neurons)}")

    def record(unit, arm, dose, paired=False, reused=False):
        return {"unit_id": unit["unit_id"], "layer": unit["layer"], "neuron": unit["neuron"],
                "arm": arm, "dose": float(dose),
                "companies_by_picking_dose": dict(unit["companies_by_picking_dose"]),
                "picking_doses": sorted(unit["picking_doses"]),
                "paired_of_neuron": bool(paired), "reused_from_part160": bool(reused)}

    generation = {}
    for uid in sorted(new_neurons):
        unit = neuron_union[uid]
        for dose in LADDER_DOSES:
            generation[(uid, "neuron", dose)] = record(unit, "neuron", dose)
    for uid, unit in random_union.items():
        for dose in LADDER_DOSES:
            generation[(uid, "random", dose)] = record(unit, "random", dose)
    for uid in sorted(new_neurons):
        unit = neuron_union[uid]
        for dose in unit["picking_doses"]:
            key = (uid, "random", float(dose))
            if key in generation:
                generation[key]["paired_of_neuron"] = True
                generation[key]["paired_neuron_companies_by_picking_dose"] = {
                    str(dose): unit["companies_by_picking_dose"][str(dose)]}
            else:
                paired_row = record(unit, "random", dose, paired=True)
                paired_row["companies_by_picking_dose"] = {}
                paired_row["paired_neuron_companies_by_picking_dose"] = {
                    str(dose): unit["companies_by_picking_dose"][str(dose)]}
                generation[key] = paired_row

    # Compact plan entries are unique by unit and arm, with both picking-dose companies.
    all_plan = [record(unit, "neuron", unit["picking_doses"][0], reused=uid in old_neurons)
                for uid, unit in sorted(neuron_union.items())]
    plan_random = {uid: dict(unit) for uid, unit in random_union.items()}
    for uid in new_neurons:
        neuron = neuron_union[uid]
        paired = plan_random.setdefault(uid, {"unit_id": uid, "layer": neuron["layer"],
            "neuron": neuron["neuron"], "companies_by_picking_dose": {}, "picking_doses": []})
        paired["paired_neuron_companies_by_picking_dose"] = {
            **paired.get("paired_neuron_companies_by_picking_dose", {}),
            **neuron["companies_by_picking_dose"]}
        paired["picking_doses"] = sorted(set(paired["picking_doses"]) | set(neuron["picking_doses"]))
    for uid, unit in sorted(plan_random.items()):
        paired = uid in new_neurons
        row = record(unit, "random", unit["picking_doses"][0], paired=paired)
        row["paired_neuron_companies_by_picking_dose"] = unit.get("paired_neuron_companies_by_picking_dose", {})
        all_plan.append(row)

    generation_rows = list(generation.values())
    return {"units": all_plan, "rows": generation_rows, "random_pick_count": len(random_union),
            "new_neuron_count": len(new_neurons), "new_neuron_ids": sorted(new_neurons),
            "neuron_pick_count_by_dose": {str(d): len(neurons_by_dose[d]) for d in PICK_DOSES},
            "reused_neuron_ids": sorted(old_neurons & set(neuron_union)),
            "random_by_unit": random_union, "neurons_by_unit": neuron_union,
            "generation_rows": generation_rows}


def make_gate2_rows(plan):
    return [
        {"unit_id": "L24N5017", "layer": 24, "neuron": 5017, "arm": "neuron", "dose": 0.8,
         "companies_by_picking_dose": {"0.8": "Google"}, "paired_of_neuron": False},
        {"unit_id": "plain", "layer": None, "neuron": None, "arm": "plain", "dose": 0.0,
         "companies_by_picking_dose": {}, "paired_of_neuron": False},
    ]


def row_key(row):
    return (row.get("stage"), row.get("unit_id"), row.get("arm"),
            float(row.get("dose", 0)), int(row.get("question_index", -1)))


def run(args, loader=None, generate_fn=None):
    source = Path(args.part160)
    screen = read_jsonl(source / "answers.jsonl")
    picked = json.loads((source / "picked.json").read_text(encoding="utf-8"))
    old_rows = screen
    screen_rows = [r for r in screen if r.get("stage") == "screen"]
    picked_by_rule = part160.picking_rule([r for r in screen_rows if r.get("arm") == "neuron"])
    # The existing kept neurons are authoritative as the atlas-order capped pick set.
    # Validate that their coordinates are represented by the exact Part 160 screen rule.
    for dose in PICK_DOSES:
        recorded = {r["unit_id"]: r["company"] for r in picked.get(str(dose), {}).get("kept", [])}
        if any(picked_by_rule[dose].get(uid) != company for uid, company in recorded.items()):
            raise ValueError(f"stored Part 160 neuron picks disagree with picking_rule at dose {dose}")
    if {d: len(picked_by_rule[d]) for d in PICK_DOSES} != {0.6: 30, 1.0: 7}:
        raise ValueError(f"Gate 3 failed: expected 30 and 7 neuron picks; "
                         f"found {len(picked_by_rule[0.6])} and {len(picked_by_rule[1.0])}")
    plan = plan_units(screen_rows, picked, old_rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps({"units": plan["units"], "rows": plan["rows"],
        "random_pick_count": plan["random_pick_count"], "new_neuron_count": plan["new_neuron_count"],
        "neuron_pick_count_by_dose": {str(d): len(picked_by_rule[d]) for d in PICK_DOSES},
        "new_neuron_ids": plan["new_neuron_ids"],
        "reused_neuron_ids": plan["reused_neuron_ids"]},
        ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    answers = out / "answers.jsonl"
    done = {row_key(r) for r in read_jsonl(answers)} if answers.exists() else set()
    common = part155._common()
    medians = json.loads((ROOT / "results/part112/medians.json").read_text(encoding="utf-8"))
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False)
        model.eval()
    else:
        tok, model = loader(args.device)
    layers = common.find_layers(model)
    generate_fn = generate_fn or (lambda *a: part160.generate_checked(*a, common=common))
    qitems = [q for q in part155.questions() if q["kind"] == "maker"]
    gate_rows = make_gate2_rows(plan)
    started = time.perf_counter()
    with answers.open("a", encoding="utf-8", newline="\n") as fh:
        # Gate 2: one batch for each prompt, with the two controls in the same call.
        gate_hits = 0
        no_push_hits = 0
        for qi, item in enumerate(qitems):
            rows = gate_rows
            missing = [r for r in rows if ("gate2", r["unit_id"], r["arm"], r["dose"], qi) not in done]
            if missing:
                pushes = part155.pushes_for(missing, layers, medians, common)
                got = generate_fn(model, tok, layers, item["question"], pushes, MAX_NEW_TOKENS, args.device)
                if len(got) != len(missing):
                    raise RuntimeError(f"Gate 2 generated {len(got)} answers for {len(missing)} rows")
                for row, answer in zip(missing, got):
                    saved = {"stage": "gate2", **row, "question_index": qi,
                             "question": item["question"], **answer}
                    fh.write(json.dumps(saved, ensure_ascii=True) + "\n")
                    fh.flush()
                    done.add(row_key(saved))
            current = {row_key(x): x for x in read_jsonl(answers)}
            results = {r["unit_id"]: current[("gate2", r["unit_id"], r["arm"], float(r["dose"]), qi)]
                       for r in rows}
            positive = results.get("L24N5017", {})
            plain = results.get("plain", {})
            gate_hits += int(score153.company_hit(positive.get("answer", ""), "Google"))
            no_push_hits += int(any(score153.company_hit(plain.get("answer", ""), c)
                                    for c in {v for x in plan["units"]
                                              for v in x["companies_by_picking_dose"].values()}))
        if not 15 <= gate_hits <= 22 or no_push_hits > 2:
            raise SystemExit(f"Gate 2 failed: Google={gate_hits}/24 (required 15-22), "
                             f"no-push={no_push_hits}/24 (maximum 2)")

        ladder = plan["generation_rows"]
        start_time = time.perf_counter()
        batch_count = 0
        first4_rows = 0
        first4_start = start_time
        for qi, item in enumerate(qitems):
            pending = [r for r in ladder if ("ladder", r["unit_id"], r["arm"], r["dose"], qi) not in done]
            for start in range(0, len(pending), args.batch_size):
                chunk = pending[start:start + args.batch_size]
                pushes = part155.pushes_for(chunk, layers, medians, common)
                result = generate_fn(model, tok, layers, item["question"], pushes, MAX_NEW_TOKENS, args.device)
                if len(result) != len(chunk):
                    raise RuntimeError(f"ladder generated {len(result)} answers for {len(chunk)} rows")
                for row, answer in zip(chunk, result):
                    saved = {"stage": "ladder", **row, "question_index": qi,
                             "question": item["question"], **answer}
                    fh.write(json.dumps(saved, ensure_ascii=True) + "\n")
                    fh.flush()
                    done.add(row_key(saved))
                batch_count += 1
                if batch_count <= 4:
                    first4_rows += len(chunk)
                if batch_count == 4:
                    elapsed = time.perf_counter() - first4_start
                    rate = first4_rows / elapsed if elapsed else math.inf
                    print(f"Ladder first 4 batches: {rate:.2f} answers/s", flush=True)
                    if rate < 1.5:
                        raise SystemExit(f"Stop rule: {rate:.2f} answers/s is below 1.5")
    return {"answers": str(answers), "plan": str(out / "plan.json"),
            "gate2_google_hits": gate_hits, "gate2_no_push_hits": no_push_hits,
            "seconds": time.perf_counter() - started}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="xpu")
    p.add_argument("--model", default=MODEL_DEFAULT)
    p.add_argument("--part160", default=str(ROOT / "results/part160"))
    p.add_argument("--out", default=str(ROOT / "results/part161"))
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    if args.batch_size != 64:
        p.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
