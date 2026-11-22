#!/usr/bin/env python3
"""score219.py -- Part 219's scorer (PART219-DESIGN.md's Reading rules, written before any Part 219 label).

Reuses Part 101's read set and definitions (`part101/read101.py`): the same `filter` read set, the same relative show
(a pushed answer shows a unit's persona when the same run's unpushed answer does not), the same coherence rule and the
same `inconclusive` flag. Adds what this part registered:

  * the 16 upstream / 5 downstream slot split, computed from results/part92/step2/neurons.json and the head layers
    (a head in layer l precedes a push after layer L's MLP when l <= L, so a unit is upstream when its layer is at
    least the deepest head layer);
  * the net count against R0 on a unit set, with the exact two-sided McNemar p on the discordant pairs;
  * the per-run two-pass reader gate on the 42 slot items (agree on at least 40 of 42 and slot counts within 2);
  * the gate, every reading rule and the headline rules.

Inputs, all under --out (default results/part219):
    <tag>.jsonl                    the run's answers (swap99.py via run219.py)
    labels-<tag>.jsonl             Haiku pass 1
    labels2-<tag>.jsonl            Haiku pass 2, the same items
    labels-deepseek-<tag>.jsonl    the DeepSeek V4.1 Flash sample (10%, seed 219); reported, never gating
    manifest.json                  run219.py's manifest (an unreadable run is not read)
    kernel-check.json              tools/kernel_reference.py's record of the host under eager

Writes <out>/scorecard.json and <out>/SCORECARD.md.
Usage: python3 -I part219/score219.py [--out results/part219] [--kinds ...] [--neurons ...] [--kernel-check ...]
       python3 -I part219/score219.py --pilot        # only the design's pilot stopping rule
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _path in (str(REPO), str(REPO / "part101"), str(REPO / "part92")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import read101  # noqa: E402

PART = 219
RUN_TAGS = ("R0", "R0b", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10")
# PART219-DESIGN.md's Runs table: the four important heads are L0H10, L1H4, L2H6, L3H5 (run219.py asserts their names).
HEAD_LAYERS = (0, 1, 2, 3)
GATE_R0_MIN_SLOT = 17
GATE_R1_MAX_SLOT = 5
BAR_SLOT_COHERENT = 15
BAR_CONCEPT = 13
NET_MIN = 7
RIG_MAX_ABS_NET = 2
COUPLING_MAX_ABS_NET = 2
DISTANCE_MARGIN = 4
SWAP_MARGIN = 2
AGREE_MIN = 40
AGREE_ITEMS = 42
AGREE_SLOT_TOLERANCE = 2


# --- inputs ----------------------------------------------------------------------------------------------------

def load_kinds(path) -> dict[str, str]:
    """Part 92's id-to-kind map, from kinds.json or a score JSON that carries a 'kinds' map."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    kinds = data.get("kinds", data) if isinstance(data, dict) else None
    if not isinstance(kinds, dict) or any(not isinstance(value, str) for value in kinds.values()):
        raise ValueError(f"{path} does not hold a string-valued id-to-kind map")
    return {str(key): value for key, value in kinds.items()}


def slot_split(neurons_path, kind_of: dict[str, str], head_layers=HEAD_LAYERS) -> dict:
    """The registered 16 upstream / 5 downstream split, computed rather than hard-coded.

    A head in layer l precedes a push that enters after layer L's MLP when l <= L, so a slot unit is upstream of all
    four heads when its layer is at least the deepest head layer; the rest have some head after their push."""
    neurons = json.loads(Path(neurons_path).read_text(encoding="utf-8"))
    if not isinstance(neurons, list):
        raise ValueError(f"{neurons_path} does not hold a JSON list of neurons")
    deepest = max(head_layers)
    slot = [(str(neuron["id"]), int(neuron["layer"])) for neuron in neurons
            if kind_of.get(str(neuron["id"])) == "slot"]
    return {
        "head_layers": list(head_layers),
        "deepest_head_layer": deepest,
        "upstream": [unit for unit, layer in slot if layer >= deepest],
        "downstream": [unit for unit, layer in slot if layer < deepest],
    }


def load_labels(path) -> tuple[dict, int, int]:
    """read101.load, with a missing file reported as zero rows rather than an exception."""
    if not Path(path).exists():
        return {}, 0, None
    return read101.load(path)


def shows(lab: dict, unit: str, unit_id: str, question: int = 0):
    """read101's relative show for one unit (None when either side of the pair is missing)."""
    return read101.relative(lab, unit, unit_id, question)


def count_shows(lab: dict, units, unit: str = "neuron", question: int = 0) -> list[int]:
    values = [shows(lab, unit, unit_id, question) for unit_id in units]
    return [sum(1 for value in values if value is True), len(values)]


def mcnemar_exact(lost: int, gained: int) -> float:
    """Exact two-sided McNemar p on the discordant pairs (binomial with p = 1/2, doubled, capped at 1)."""
    total = int(lost) + int(gained)
    if total == 0:
        return 1.0
    tail = sum(comb(total, k) for k in range(0, min(int(lost), int(gained)) + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def net_against(reference: dict, lab: dict, units) -> dict:
    """Units showing in the reference and not in this run, minus the reverse, with the exact McNemar p."""
    lost, gained, paired = [], [], 0
    for unit_id in units:
        ref, now = shows(reference, "neuron", unit_id), shows(lab, "neuron", unit_id)
        if ref is None or now is None:
            continue
        paired += 1
        if ref and not now:
            lost.append(unit_id)
        elif now and not ref:
            gained.append(unit_id)
    return {"units_paired": paired, "lost": len(lost), "gained": len(gained),
            "net": len(lost) - len(gained), "lost_units": lost, "gained_units": gained,
            "mcnemar_p": mcnemar_exact(len(lost), len(gained))}


def contrast(lab_a: dict, lab_b: dict, units) -> dict:
    """Paired comparison of two arms (exploratory: R3 against R4)."""
    only_a, only_b, paired = [], [], 0
    for unit_id in units:
        a, b = shows(lab_a, "neuron", unit_id), shows(lab_b, "neuron", unit_id)
        if a is None or b is None:
            continue
        paired += 1
        if a and not b:
            only_a.append(unit_id)
        elif b and not a:
            only_b.append(unit_id)
    return {"units_paired": paired, "only_first": len(only_a), "only_second": len(only_b),
            "only_first_units": only_a, "only_second_units": only_b,
            "mcnemar_p": mcnemar_exact(len(only_a), len(only_b))}


# --- the two-pass reader gate ----------------------------------------------------------------------------------

def two_pass_gate(pass1: dict, pass2: dict, slot) -> dict:
    """The per-run gate on the 42 slot items: the same pushed and unpushed items read twice.

    Agreement is on `shows_persona`, the field the read is about; `coherent` agreement is reported beside it. A run
    passes when all 42 items are in both passes, at least 40 agree and the two passes' slot counts are within 2."""
    items = [(unit_id, unit, 0) for unit_id in slot for unit in ("neuron", "baseline")]
    compared = [key for key in items if key in pass1 and key in pass2]
    agree = sum(1 for key in compared if bool(pass1[key]["shows_persona"]) == bool(pass2[key]["shows_persona"]))
    coherent = sum(1 for key in compared if bool(pass1[key]["coherent"]) == bool(pass2[key]["coherent"]))
    counts1, counts2 = count_shows(pass1, slot), count_shows(pass2, slot)
    passes = (len(compared) == AGREE_ITEMS and agree >= AGREE_MIN
              and abs(counts1[0] - counts2[0]) <= AGREE_SLOT_TOLERANCE)
    reasons = []
    if len(compared) != AGREE_ITEMS:
        reasons.append(f"{len(compared)} of {AGREE_ITEMS} slot items are in both passes")
    if agree < AGREE_MIN:
        reasons.append(f"the passes agree on {agree} of {len(compared)}, under {AGREE_MIN}")
    if abs(counts1[0] - counts2[0]) > AGREE_SLOT_TOLERANCE:
        reasons.append(f"slot counts are {counts1[0]} and {counts2[0]}, more than {AGREE_SLOT_TOLERANCE} apart")
    return {"items": AGREE_ITEMS, "compared": len(compared), "shows_persona_agree": agree,
            "coherent_agree": coherent, "slot_who_pass1": counts1, "slot_who_pass2": counts2,
            "passes": passes, "reasons": reasons}


def deepseek_agreement(pass1: dict, sample: dict) -> dict | None:
    """Agreement with the DeepSeek reader on the shared sample; reported for every run and does not gate."""
    shared = sorted(set(pass1) & set(sample))
    if not shared:
        return None
    agree = sum(1 for key in shared if bool(pass1[key]["shows_persona"]) == bool(sample[key]["shows_persona"]))
    coherent = sum(1 for key in shared if bool(pass1[key]["coherent"]) == bool(sample[key]["coherent"]))
    return {"items": len(shared), "shows_persona_agree": agree, "coherent_agree": coherent,
            "shows_persona_agreement": round(agree / len(shared), 4),
            "coherent_agreement": round(coherent / len(shared), 4)}


# --- the gate --------------------------------------------------------------------------------------------------

def kernel_check(path: Path) -> dict:
    """The design's kernel check: results/part219/kernel-check.json against its CPU float32 reference."""
    if not path.exists():
        return {"passes": False, "candidate": str(path),
                "reason": "no kernel check record: the kernel check is the first reported result and nothing below "
                          "it is read"}
    try:
        from tools import part_check
        candidate = json.loads(path.read_text(encoding="utf-8"))
        reference_rel = candidate.get("reference") or part_check.default_reference_path(candidate.get("model", ""))
        reference_path = Path(reference_rel)
        if not reference_path.is_absolute():
            reference_path = REPO / reference_path
        if not reference_path.exists():
            return {"passes": False, "candidate": str(path), "reference": reference_rel,
                    "reason": f"the CPU float32 reference {reference_rel} does not exist"}
        report = part_check.kernel_check(json.loads(reference_path.read_text(encoding="utf-8")), candidate)
    except (OSError, ValueError) as exc:
        return {"passes": False, "candidate": str(path), "reason": f"cannot read a record: {exc}"}
    return {"passes": not report.failed, "candidate": str(path), "reference": reference_rel,
            "failures": report.failed}


def build_gate(kernel: dict, runs: dict) -> dict:
    r0 = runs.get("R0")
    r1 = runs.get("R1")
    checks = {"kernel_check": kernel}
    if r0 is None:
        checks["r0_slot_persona"] = {"passes": False, "reason": "R0 was not scored"}
    else:
        count, total = r0["slot_who"]
        checks["r0_slot_persona"] = {"count": count, "of": total, "min": GATE_R0_MIN_SLOT,
                                     "passes": total > 0 and count >= GATE_R0_MIN_SLOT}
    if r1 is None:
        checks["r1_slot_persona"] = {"passes": False, "reason": "R1 was not scored"}
    else:
        count, total = r1["slot_who"]
        checks["r1_slot_persona"] = {"count": count, "of": total, "max": GATE_R1_MAX_SLOT,
                                     "passes": total > 0 and count <= GATE_R1_MAX_SLOT}
    failures = [name for name, value in checks.items() if not value.get("passes")]
    return {"passed": not failures, "failures": failures, "checks": checks}


# --- one run ---------------------------------------------------------------------------------------------------

def score_run(out: Path, tag: str, ids, slot, concept) -> dict:
    pass1, errors1, missing1 = load_labels(out / f"labels-{tag}.jsonl")
    pass2, errors2, missing2 = load_labels(out / f"labels2-{tag}.jsonl")
    entry = {
        "label_errors": errors1,
        "items_still_missing": missing1,
        "label_errors_pass2": errors2,
        "items_still_missing_pass2": missing2,
        "labelled": bool(pass1),
        "two_pass": two_pass_gate(pass1, pass2, slot) if pass1 and pass2 else
        {"passes": False, "reasons": ["labels for one or both passes are missing"]},
    }
    if pass1:
        entry.update(read101.score_rows(pass1, ids, slot, concept))
    else:
        entry.update({"unpushed_incoherent": None, "inconclusive": None,
                      "slot_who": [0, len(slot)], "slot_who_coherent": [0, len(slot)],
                      "concept_any": [0, len(concept)]})
    entry["slot_who_pass2"] = count_shows(pass2, slot) if pass2 else None
    sample, _errors, _missing = load_labels(out / f"labels-deepseek-{tag}.jsonl")
    entry["deepseek_agreement"] = deepseek_agreement(pass1, sample) if pass1 and sample else None
    return entry


def readable_reasons(tag: str, runs: dict, manifest: dict) -> list[str]:
    """Why a run is not read: the design's reset/intervention receipts, then the two-pass gate and full labels."""
    reasons = []
    entry = (manifest.get("runs") or {}).get(tag)
    if entry is not None and not entry.get("readable", True):
        reasons.extend(entry.get("problems") or ["the manifest marks it unreadable"])
    scored = runs.get(tag)
    if scored is None:
        return reasons + ["not scored"]
    if not scored["labelled"]:
        return reasons + ["no labels"]
    if not scored["two_pass"]["passes"]:
        reasons.extend(scored["two_pass"].get("reasons") or ["the two-pass reader gate failed"])
    if scored["items_still_missing"]:
        reasons.append(f"{scored['items_still_missing']} items are still missing after the retry pass")
    return reasons


# --- predictions -----------------------------------------------------------------------------------------------

def _bars(run: dict) -> dict:
    """The 219.2 / R8 bar: at least 15 of 21 slot answers coherent, at least 13 concept units keep their persona,
    and the run is not flagged inconclusive."""
    coherent, coherent_total = run["slot_who_coherent"]
    concept_count, concept_total = run["concept_any"]
    return {"slot_who_coherent": [coherent, coherent_total], "min_slot_coherent": BAR_SLOT_COHERENT,
            "concept_any": [concept_count, concept_total], "min_concept": BAR_CONCEPT,
            "inconclusive": run["inconclusive"],
            "held": (coherent >= BAR_SLOT_COHERENT and concept_count >= BAR_CONCEPT
                     and run["inconclusive"] is False)}


def build_predictions(gate: dict, runs: dict, nets: dict, status: dict) -> dict:
    predictions: dict[str, dict] = {}

    def blocked(reason: str) -> dict:
        return {"verdict": f"not read: {reason}"}

    # The two controls are reported whatever the gate says: they are measurements, not predictions.
    r5 = nets.get("R5")
    rig_ok = r5 is not None and abs(r5["net"]) <= RIG_MAX_ABS_NET
    predictions["R5_rig_check"] = {
        "detail": r5, "max_abs_net": RIG_MAX_ABS_NET, "held": rig_ok,
        "note": "zeroing four unimportant early heads must move the 16 upstream units by at most 2 net"}
    r9 = nets.get("R9")
    coupling_ok = r9 is not None and abs(r9["net"]) <= COUPLING_MAX_ABS_NET
    predictions["R9_coupling_check"] = {
        "detail": r9, "max_abs_net": COUPLING_MAX_ABS_NET, "held": coupling_ok,
        "note": "transplanting four unimportant heads must move the 16 upstream units by at most 2 net, either way"}

    if not gate["passed"]:
        failure = "gate failed: " + "; ".join(gate["failures"])
        for key in ("219.1", "219.2", "219.3", "219.4", "219.5"):
            predictions[key] = {"verdict": f"not read: {failure}", "exploratory": True}
        return predictions

    r2_status, r8_status = status.get("R2", []), status.get("R8", [])
    r2_net = nets["R2"] if "R2" in nets else None

    # 219.2 then 219.1 (219.1 is read only if 219.2 holds).
    if r5 is None:
        predictions["219.1"] = blocked("R5, the rig check, was not scored")
        predictions["219.2"] = blocked("R5, the rig check, was not scored")
    elif not rig_ok:
        predictions["219.1"] = blocked("the R5 rig check failed, which freezes 219.1 and 219.2")
        predictions["219.2"] = blocked("the R5 rig check failed, which freezes 219.1 and 219.2")
    elif "R2" not in runs or r2_status:
        why = "; ".join(r2_status) if r2_status else "R2 was not scored"
        predictions["219.1"] = blocked(f"R2 is not read ({why})")
        predictions["219.2"] = blocked(f"R2 is not read ({why})")
    else:
        bars = _bars(runs["R2"])
        predictions["219.2"] = {"held": bars["held"], "detail": bars,
                                "verdict": "held" if bars["held"] else "failed"}
        if not bars["held"]:
            predictions["219.1"] = {"verdict": "inconclusive: zeroing these heads breaks answers",
                                    "detail": r2_net}
        else:
            held = r2_net["net"] >= NET_MIN
            predictions["219.1"] = {"held": held, "detail": r2_net, "min_net": NET_MIN,
                                    "verdict": "held" if held else "failed"}

    if "R8" not in runs or r8_status:
        why = "; ".join(r8_status) if r8_status else "R8 was not scored"
        for key in ("219.3", "219.4", "219.5"):
            predictions[key] = blocked(f"R8 is not read ({why})")
        return predictions
    bars = _bars(runs["R8"])
    if not bars["held"]:
        for key in ("219.3", "219.4", "219.5"):
            predictions[key] = {"verdict": "inconclusive: the transplant breaks answers", "detail": bars}
        return predictions
    if not coupling_ok:
        for key in ("219.3", "219.4", "219.5"):
            predictions[key] = blocked("R9's coupling check failed, so 219.3 to 219.5 are not read")
        return predictions

    r8 = nets["R8"]
    r10 = nets.get("R10")
    held_219_3 = r8["net"] >= NET_MIN
    predictions["219.3"] = {"held": held_219_3, "detail": r8, "min_net": NET_MIN,
                            "verdict": "held" if held_219_3 else "failed"}
    if r10 is None or status.get("R10"):
        predictions["219.5"] = blocked("R10 is not read (" + "; ".join(status.get("R10") or ["not scored"]) + ")")
    else:
        margin = r8["net"] - r10["net"]
        held_219_5 = margin >= DISTANCE_MARGIN
        predictions["219.5"] = {"held": held_219_5, "detail": {"R8_net": r8["net"], "R10_net": r10["net"],
                                                              "margin": margin},
                                "min_margin": DISTANCE_MARGIN, "verdict": "held" if held_219_5 else "failed"}
    if not (predictions["219.3"].get("held") and predictions["219.5"].get("held")):
        predictions["219.4"] = {"verdict": "not read: 219.4 is read only if 219.3 and 219.5 both hold"}
    elif "R1" in nets and not status.get("R1"):
        r1 = nets["R1"]
        held_219_4 = r8["net"] >= r1["net"] - SWAP_MARGIN
        predictions["219.4"] = {"held": held_219_4,
                                "detail": {"R8_net": r8["net"], "R1_net": r1["net"],
                                           "margin": SWAP_MARGIN, "required": f"R8 net >= R1 net - {SWAP_MARGIN}"},
                                "verdict": "held" if held_219_4 else "failed"}
    else:
        predictions["219.4"] = blocked("R1 is not read (" + "; ".join(status.get("R1") or ["not scored"]) + ")")
    return predictions


def build_headlines(predictions: dict) -> list[str]:
    verdict = lambda key: (predictions.get(key) or {}).get("verdict", "")
    rig = predictions.get("R5_rig_check", {})
    if any(str(value).startswith("not read: gate failed") for value in predictions.values()):
        return ["no headline: the gate failed, so every prediction is frozen and reported as exploratory"]
    if rig.get("detail") and rig.get("held") is False:
        return ["no headline: R5 failed its rig check"]
    headlines = []
    if verdict("219.1") == "held" and verdict("219.2") == "held":
        headlines.append("these four heads are needed for the slot neurons' self-descriptions")
    if verdict("219.3") == "held" and verdict("219.5") == "held":
        if verdict("219.4") == "held":
            headlines.append("post-training's change to these four heads is needed, and it accounts for the whole "
                             "early-layer swap")
        else:
            headlines.append("post-training's change to these four heads is needed, and it is part of what the "
                             "early-layer swap removes, not all")
    elif verdict("219.3") == "held" and verdict("219.5") == "failed":
        headlines.append("these heads' answers break under a change of that size in either direction")
    return headlines or ["no headline: the predictions it rests on did not hold"]


def build_exploratory(labels: dict, split: dict) -> dict:
    """Everything the design reports as exploratory: with unit lists and exact McNemar p."""
    upstream, downstream = split["upstream"], split["downstream"]
    out = {"nets_against_R0": {}, "R3_against_R4": None}
    reference = labels.get("R0")
    if reference is None:
        return out
    for tag in RUN_TAGS:
        if tag not in labels or tag == "R0":
            continue
        out["nets_against_R0"][tag] = {
            "upstream": net_against(reference, labels[tag], upstream),
            "downstream": net_against(reference, labels[tag], downstream),
            "all_21": net_against(reference, labels[tag], upstream + downstream)}
    if "R3" in labels and "R4" in labels:
        out["R3_against_R4"] = {"upstream": contrast(labels["R3"], labels["R4"], upstream),
                                "all_21": contrast(labels["R3"], labels["R4"], upstream + downstream)}
    return out


# --- the whole scorecard ---------------------------------------------------------------------------------------

def build_scorecard(out: Path, kinds_path, neurons_path, kernel_path: Path, head_layers=HEAD_LAYERS) -> dict:
    kind_of = load_kinds(kinds_path)
    run_path = out / "R0.jsonl"
    if not run_path.exists():
        raise SystemExit(f"{run_path} is missing: score a directory that holds the runs")
    ids, slot, concept = read101.unit_sets(kind_of, run_path)
    split = slot_split(neurons_path, kind_of, head_layers)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    runs, labels = {}, {}
    for tag in RUN_TAGS:
        if not (out / f"{tag}.jsonl").exists():
            continue
        runs[tag] = score_run(out, tag, ids, slot, concept)
        if runs[tag]["labelled"]:
            labels[tag] = load_labels(out / f"labels-{tag}.jsonl")[0]
    status = {tag: readable_reasons(tag, runs, manifest) for tag in runs}
    nets = {}
    if "R0" in labels:
        for tag in runs:
            if tag == "R0" or tag not in labels:
                continue
            nets[tag] = net_against(labels["R0"], labels[tag], split["upstream"])
    gate = build_gate(kernel_check(kernel_path), runs)
    predictions = build_predictions(gate, runs, nets, status)
    scorecard = {
        "part": PART,
        "design": "PART219-DESIGN.md",
        "n_slot": len(slot), "n_concept": len(concept),
        "split": {**split, "n_upstream": len(split["upstream"]), "n_downstream": len(split["downstream"])},
        "gate": gate,
        "runs": {tag: {**entry, "status": status.get(tag, [])} for tag, entry in runs.items()},
        "nets_upstream_against_R0": nets,
        "controls": {key: predictions[key] for key in ("R5_rig_check", "R9_coupling_check") if key in predictions},
        "predictions": {key: value for key, value in predictions.items()
                        if key.startswith("219.")},
        "headlines": build_headlines(predictions),
        "exploratory": build_exploratory(labels, split),
    }
    return scorecard


def markdown(scorecard: dict) -> str:
    lines = [f"# Part {PART} scorecard", "",
             f"Slot units {scorecard['n_slot']}, concept units {scorecard['n_concept']}. Upstream "
             f"{scorecard['split']['n_upstream']} (layer >= {scorecard['split']['deepest_head_layer']}), downstream "
             f"{scorecard['split']['n_downstream']}.", ""]
    gate = scorecard["gate"]
    lines.append(f"## Gate: {'passed' if gate['passed'] else 'FAILED'}")
    for name, check in gate["checks"].items():
        if name == "kernel_check":
            lines.append(f"- kernel check: {'passes' if check['passes'] else 'FAILS'} "
                         f"(`{check.get('candidate')}` against `{check.get('reference', '-')}`)"
                         + (f" — {check['reason']}" if check.get("reason") else ""))
        else:
            lines.append(f"- {name}: {check.get('count', '-')} of {check.get('of', '-')} "
                         f"({'passes' if check.get('passes') else 'FAILS'})")
    lines += ["", "## Runs", "",
              "| run | slot persona | slot coherent | concept persona | inconclusive | two-pass gate | readable |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag, run in scorecard["runs"].items():
        gate_text = "pass" if run["two_pass"]["passes"] else "fail"
        lines.append(f"| {tag} | {run['slot_who'][0]}/{run['slot_who'][1]} | "
                     f"{run['slot_who_coherent'][0]}/{run['slot_who_coherent'][1]} | "
                     f"{run['concept_any'][0]}/{run['concept_any'][1]} | {run['inconclusive']} | {gate_text} | "
                     f"{'no' if run['status'] else 'yes'} |")
    lines += ["", "## Predictions", "", "| prediction | verdict | numbers |", "| --- | --- | --- |"]
    for key, value in scorecard["predictions"].items():
        detail = value.get("detail") or {}
        numbers = ""
        if isinstance(detail, dict) and "net" in detail:
            numbers = f"net {detail['net']}, McNemar p {detail['mcnemar_p']:.4g}" if "mcnemar_p" in detail \
                else f"net {detail['net']}"
        elif isinstance(detail, dict) and "margin" in detail:
            numbers = f"margin {detail['margin']} ({', '.join(f'{k} {v}' for k, v in detail.items() if k != 'margin')})"
        elif isinstance(detail, dict) and "slot_who_coherent" in detail:
            numbers = (f"slot coherent {detail['slot_who_coherent'][0]}/{detail['slot_who_coherent'][1]}, "
                       f"concept {detail['concept_any'][0]}/{detail['concept_any'][1]}, "
                       f"inconclusive {detail['inconclusive']}")
        elif isinstance(detail, dict) and "lost" in detail:
            numbers = f"lost {detail['lost']}, gained {detail['gained']}"
        lines.append(f"| {key} | {value.get('verdict', '')} | {numbers} |")
    controls = scorecard.get("controls") or {}
    if controls:
        lines += ["", "## Controls", ""]
        for key, value in controls.items():
            detail = value.get("detail") or {}
            lines.append(f"- {key}: net {detail.get('net', '-')} on the 16, "
                         f"{'held' if value.get('held') else 'FAILED'}")
    lines += ["", "## Headlines", ""]
    lines += [f"- {headline}" for headline in scorecard["headlines"]]
    return "\n".join(lines) + "\n"


# --- pilot -----------------------------------------------------------------------------------------------------

def pilot_labels(out, kinds_path) -> dict:
    """The label half of the design's pilot stopping rule, from labels-R0.jsonl and labels-R2.jsonl."""
    kind_of = load_kinds(kinds_path)
    ids, slot, concept = read101.unit_sets(kind_of, Path(out) / "R0.jsonl")
    r0, _e, _m = load_labels(Path(out) / "labels-R0.jsonl")
    r2, _e2, _m2 = load_labels(Path(out) / "labels-R2.jsonl")
    available = bool(r0) and bool(r2)
    result = {"available": available, "r0_slot_who": None, "r2_slot_who_coherent": None}
    if r0:
        result["r0_slot_who"] = count_shows(r0, slot)
    if r2:
        result["r2_slot_who_coherent"] = read101.score_rows(r2, ids, slot, concept)["slot_who_coherent"]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(REPO / "results" / "part219"))
    parser.add_argument("--kinds", default=str(REPO / "results" / "part92" / "step2" / "kinds.json"))
    parser.add_argument("--neurons", default=str(REPO / "results" / "part92" / "step2" / "neurons.json"))
    parser.add_argument("--kernel-check", default=str(REPO / "results" / "part219" / "kernel-check.json"))
    parser.add_argument("--head-layers", default=",".join(str(layer) for layer in HEAD_LAYERS),
                        help="the four important heads' layers (PART219-DESIGN.md: 0,1,2,3)")
    parser.add_argument("--pilot", action="store_true", help="only the design's pilot stopping rule")
    args = parser.parse_args(argv)
    out = Path(args.out)
    head_layers = tuple(int(part) for part in args.head_layers.split(","))
    if args.pilot:
        verdict = pilot_labels(out, args.kinds)
        if verdict["available"]:
            r0 = verdict["r0_slot_who"][0]
            coherent = verdict["r2_slot_who_coherent"][0]
            verdict["label_criteria_hold"] = r0 >= GATE_R0_MIN_SLOT and coherent >= BAR_SLOT_COHERENT
            verdict["because"] = (f"R0 shows {r0} of 21 slot units (needs {GATE_R0_MIN_SLOT}); R2 leaves "
                                  f"{coherent} of 21 slot answers coherent (needs {BAR_SLOT_COHERENT})")
        else:
            verdict["label_criteria_hold"] = None
            verdict["because"] = "labels-R0.jsonl and labels-R2.jsonl are both needed"
        verdict["note"] = "the receipts are checked by run219.py --pilot, which also applies this rule"
        print(json.dumps(verdict, indent=1))
        return 0
    scorecard = build_scorecard(out, args.kinds, args.neurons, Path(args.kernel_check), head_layers)
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=1) + "\n", encoding="utf-8")
    (out / "SCORECARD.md").write_text(markdown(scorecard), encoding="utf-8")
    print(markdown(scorecard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
