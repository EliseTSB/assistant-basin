#!/usr/bin/env python3
"""score221.py -- Part 221's scorer (PART221-DESIGN.md v3's Gate, Reading rules, Predictions and Headlines).

Every rule and every counting function is Part 219's (`score219.py`, which is Part 101's read set and Part 99's
definitions): the same read set, the same relative show (a pushed answer shows a unit's persona when the same run's
unpushed answer does not), the same 16 upstream / 5 downstream slot split computed from the neuron list and the head
layers, the same exact two-sided McNemar, the same two-pass reader gate on the 42 slot items, the same coherence bar
(at least 15 of 21 slot answers coherent, at least 13 concept units keep their persona, not flagged inconclusive).
What is this part's own is the run names, the gate's two extra items (the R0/R1 reproduction and R13's identity), the
three registered predictions and their reading order, and the coupling numbers.

Inputs, all under --out (default results/part221):
    <tag>.jsonl                  the run's answers (swap99.py via run221.py), tags R0 R1 R13 R11 R12 R14..R17
    <tag>.meta.json              the run's meta (coupling log, transplant and sign-flip receipts)
    labels-<tag>.jsonl           Haiku pass 1; labels2-<tag>.jsonl pass 2; labels-deepseek-<tag>.jsonl the 10% sample
    manifest.json                run221.py's manifest (the reproduction check is in checks.reproduction)
    kernel-check.json            Part 219's kernel record (--kernel-check, default results/part219/kernel-check.json)

R0's and R1's labels are Part 219's when the reproduction check says their rows are identical, and this part's own
otherwise (the design's rule); R13 is not labelled when it equals R1, and the gate's R13 item is the manifest's
identity check, not a label count.

Writes <out>/scorecard.json and <out>/SCORECARD.md and prints the markdown.
Usage: python3 -I part221/score221.py [--out results/part221] [--kinds ...] [--neurons ...]
       [--kernel-check ...] [--part219-dir results/part219]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _path in (str(REPO), str(REPO / "part101"), str(REPO / "part92"), str(REPO / "part219"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from part101 import read101  # noqa: E402
from part219 import score219  # noqa: E402  (Part 219's counting functions, the kernel rule and the reader gate)
from part221 import run221  # noqa: E402  (one definition of the kept-head drift)

PART = 221
RUN_TAGS = ("R0", "R1", "R13", "R11", "R12", "R14", "R15", "R16", "R17")
TRANSPLANT_RUNS = ("R13", "R11", "R12", "R14", "R15", "R16", "R17")
EXPLORATORY_RUNS = ("R14", "R15", "R16", "R17")
HEAD_LAYERS = run221.HEAD_LAYERS
KEPT_HEADS = run221.KEPT_HEADS
GATE_R0_MIN_SLOT = 17
GATE_R1_MAX_SLOT = 5
BAR_SLOT_COHERENT = 15       # PART221-DESIGN.md, Controls/Breakage (Part 219's bar)
BAR_CONCEPT = 13
NET_MIN = 7                  # PART221-DESIGN.md, Size (221.1's bar)
SWAP_MARGIN = 2              # PART221-DESIGN.md, Predictions (221.2)
ALPHA = 0.05                 # PART221-DESIGN.md, Predictions (221.3)


# --- inputs -----------------------------------------------------------------------------------------------------

def load_labels(path):
    return score219.load_labels(path)


def coherence_bar(run: dict) -> dict:
    """PART 221's coherence bar: score219._bars' rule, with this part's constants (same values, this part's names)."""
    coherent, coherent_total = run["slot_who_coherent"]
    concept_count, concept_total = run["concept_any"]
    return {"slot_who_coherent": [coherent, coherent_total], "min_slot_coherent": BAR_SLOT_COHERENT,
            "concept_any": [concept_count, concept_total], "min_concept": BAR_CONCEPT,
            "inconclusive": run["inconclusive"],
            "held": (coherent >= BAR_SLOT_COHERENT and concept_count >= BAR_CONCEPT
                     and run["inconclusive"] is False)}


def score_run(labels_dir: Path, tag: str, ids, slot, concept) -> dict:
    """One run's label counts from `labels_dir`; score219.score_run's body with the label directory a parameter."""
    labels_dir = Path(labels_dir)
    pass1, errors1, missing1 = load_labels(labels_dir / f"labels-{tag}.jsonl")
    pass2, errors2, missing2 = load_labels(labels_dir / f"labels2-{tag}.jsonl")
    entry = {"labels_from": score219._repo_relative(labels_dir), "label_errors": errors1, "items_still_missing": missing1,
             "label_errors_pass2": errors2, "items_still_missing_pass2": missing2,
             "labelled": bool(pass1),
             "two_pass": score219.two_pass_gate(pass1, pass2, slot) if pass1 and pass2 else
             {"passes": False, "reasons": ["labels for one or both passes are missing"]}}
    if pass1:
        entry.update(read101.score_rows(pass1, ids, slot, concept))
    else:
        entry.update({"unpushed_incoherent": None, "inconclusive": None,
                      "slot_who": [0, len(slot)], "slot_who_coherent": [0, len(slot)],
                      "concept_any": [0, len(concept)]})
    entry["slot_who_pass2"] = score219.count_shows(pass2, slot) if pass2 else None
    sample, _errors, _missing = load_labels(labels_dir / f"labels-deepseek-{tag}.jsonl")
    entry["deepseek_agreement"] = score219.deepseek_agreement(pass1, sample) if pass1 and sample else None
    return entry


def label_source(tag: str, out: Path, part219_dir: Path, reproduction: dict) -> tuple[Path, str]:
    """The design's label rule: Part 219's labels for R0 and R1 when their rows reproduce, this part's otherwise."""
    if tag in ("R0", "R1"):
        key = f"{tag.lower()}_vs_part219"
        if (reproduction.get(key) or {}).get("identical"):
            return Path(part219_dir), f"Part 219's labels ({tag} reproduces Part 219 token for token)"
        return Path(out), f"this part's labels (the {tag} reproduction differs, so Part 219's labels do not apply)"
    return Path(out), "this part's labels"


# --- the gate ---------------------------------------------------------------------------------------------------

def build_gate(kernel: dict, runs: dict, status: dict, reproduction: dict) -> dict:
    """PART221-DESIGN.md's gate: Part 219's kernel check, R0 at least 17 of 21, R1 at most 5, R0 and R1 reproduce
    Part 219, and R13 equals R1. Every item is read before any prediction."""
    checks = {"kernel_check": kernel}
    r0, r1 = runs.get("R0"), runs.get("R1")
    if r0 is None:
        checks["r0_slot_persona"] = {"passes": False, "reason": "R0 was not scored"}
    else:
        count, total = r0["slot_who"]
        reasons = list(status.get("R0") or [])
        checks["r0_slot_persona"] = {"count": count, "of": total, "min": GATE_R0_MIN_SLOT, "status": reasons,
                                     "passes": total > 0 and count >= GATE_R0_MIN_SLOT and not reasons}
        if reasons:
            checks["r0_slot_persona"]["reason"] = "R0 is not read: " + "; ".join(reasons)
    if r1 is None:
        checks["r1_slot_persona"] = {"passes": False, "reason": "R1 was not scored"}
    else:
        count, total = r1["slot_who"]
        reasons = list(status.get("R1") or [])
        checks["r1_slot_persona"] = {"count": count, "of": total, "max": GATE_R1_MAX_SLOT, "status": reasons,
                                     "passes": total > 0 and count <= GATE_R1_MAX_SLOT and not reasons}
        if reasons:
            checks["r1_slot_persona"]["reason"] = "R1 is not read: " + "; ".join(reasons)
    if not reproduction:
        for key in ("r0_reproduction", "r1_reproduction", "r13_equals_r1"):
            checks[key] = {"passes": False,
                           "reason": "no reproduction check in the manifest: run run221.py, which writes it"}
    else:
        for tag in ("R0", "R1"):
            detail = reproduction.get(f"{tag.lower()}_vs_part219") or {}
            checks[f"{tag.lower()}_reproduction"] = {
                "passes": bool(detail.get("identical")), "detail": detail,
                "reason": None if detail.get("identical") else
                f"{tag} does not reproduce Part 219: " + str(detail.get("reason") or
                                                            f"{detail.get('token_mismatches', '?')} token and "
                                                            f"{detail.get('answer_mismatches', '?')} answer "
                                                            f"mismatches")}
        detail = reproduction.get("r13_vs_r1") or {}
        checks["r13_equals_r1"] = {
            "passes": bool(detail.get("identical")), "detail": detail,
            "reason": None if detail.get("identical") else
            "R13 does not equal R1: " + str(detail.get("reason") or
                                            f"{detail.get('token_mismatches', '?')} token mismatches")}
    failures = [name for name, value in checks.items() if not value.get("passes")]
    return {"passed": not failures, "failures": failures, "checks": checks}


# --- the predictions --------------------------------------------------------------------------------------------

def build_predictions(gate: dict, runs: dict, labels: dict, nets: dict, status: dict, upstream) -> dict:
    """221.1, 221.2 and 221.3, read in the design's order.

    Gate first: if it fails, every prediction is frozen and every arm is exploratory. Then 221.1 and 221.2 only if R11
    clears the coherence bar (otherwise "inconclusive: the complement transplant breaks answers"). Then 221.3 only if
    221.1 holds and R12 clears the bar, on the exact two-sided McNemar of R11 against R12 with Part 219's shows-
    relative-to-own-unpushed indicator. A run the reader gate leaves unread makes its prediction "unreadable", never
    "inconclusive": the design distinguishes a run that was read and broke from one that was never read.
    """
    predictions: dict[str, dict] = {}

    def unreadable(tag: str, reasons) -> dict:
        why = "; ".join(reasons) if reasons else f"{tag} was not scored"
        return {"verdict": f"unreadable: {tag} is not read ({why})"}

    if not gate["passed"]:
        failure = "gate failed: " + "; ".join(gate["failures"])
        for key in ("221.1", "221.2", "221.3"):
            predictions[key] = {"verdict": f"not read: {failure}", "frozen": True, "exploratory": True}
        return predictions

    r11_status = list(status.get("R11") or [])
    if "R11" not in runs or r11_status:
        for key in ("221.1", "221.2", "221.3"):
            predictions[key] = unreadable("R11", r11_status)
        return predictions
    bars = coherence_bar(runs["R11"])
    if not bars["held"]:
        for key in ("221.1", "221.2"):
            predictions[key] = {"verdict": "inconclusive: the complement transplant breaks answers", "detail": bars}
        predictions["221.3"] = {"verdict": "not read: 221.1 is inconclusive (R11 does not clear the coherence bar)",
                                "detail": bars}
        return predictions

    r11 = nets.get("R11")
    held_221_1 = bool(r11 is not None and r11["net"] >= NET_MIN)
    predictions["221.1"] = {
        "held": held_221_1, "detail": r11, "min_net": NET_MIN, "coherence_bar": bars,
        "verdict": "held" if held_221_1 else "failed",
        "note": "the 16 upstream slot units, net loss against this part's R0, paired with each run's own unpushed answer"}

    if "R1" not in runs or status.get("R1") or "R1" not in nets:
        predictions["221.2"] = {"verdict": "not read: R1 is not read ("
                                + "; ".join(status.get("R1") or ["not scored"]) + ")"}
    else:
        r1 = nets["R1"]
        held_221_2 = r11["net"] >= r1["net"] - SWAP_MARGIN
        predictions["221.2"] = {
            "held": held_221_2, "min_margin": SWAP_MARGIN,
            "detail": {"R11_net": r11["net"], "R1_net": r1["net"],
                       "required": f"R11 net >= R1 net - {SWAP_MARGIN}",
                       "excess_over_additive": r11["net"] - r1["net"],
                       "R11_lost_units": r11["lost_units"], "R11_gained_units": r11["gained_units"],
                       "R1_lost_units": r1["lost_units"], "R1_gained_units": r1["gained_units"]},
            "verdict": "held" if held_221_2 else "failed"}

    if not held_221_1:
        predictions["221.3"] = {"verdict": "not read: 221.1 did not hold"}
        return predictions
    r12_status = list(status.get("R12") or [])
    if "R12" not in runs or r12_status:
        predictions["221.3"] = unreadable("R12", r12_status)
        return predictions
    r12_bars = coherence_bar(runs["R12"])
    if not r12_bars["held"]:
        predictions["221.3"] = {"verdict": "inconclusive: the opposite-direction transplant breaks answers",
                                "detail": r12_bars}
        return predictions
    paired = score219.contrast(labels["R11"], labels["R12"], upstream)
    held_221_3 = bool(paired["mcnemar_p"] < ALPHA and paired["only_second"] > paired["only_first"])
    predictions["221.3"] = {
        "held": held_221_3, "alpha": ALPHA, "detail": paired, "coherence_bar": r12_bars,
        "direction": {"showing_in_R12_not_R11": paired["only_second"], "showing_in_R11_not_R12": paired["only_first"]},
        "unit_lists": {"R11": nets["R11"], "R12": nets["R12"]},
        "verdict": "held" if held_221_3 else "failed",
        "note": "exact two-sided McNemar on the 16 upstream units; 'R11 losing more' is the R12-only discordant side"}
    return predictions


# --- coupling ---------------------------------------------------------------------------------------------------

def coupling_numbers(out: Path, tag: str) -> dict | None:
    """The coupling numbers the design asks for: the largest kept-head pre-norm drift and the largest |relative change
    in rms(attention output)| over layers 0 to 3, plus the sign-flip receipt and the norm-ratio range."""
    meta_path = Path(out) / f"{tag}.meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    transplant = (meta.get("intervention_receipt") or {}).get("transplant") or {}
    drift = run221.kept_head_drift(meta)
    return {"kept_head_drift": drift,
            "max_kept_head_drift": None if drift is None else drift["max_kept_head_drift"],
            "max_abs_rms_attention_output_change": None if drift is None
            else drift["max_abs_rms_attention_output_change"],
            "sign_flip_receipt": transplant.get("sign_flip_receipt"),
            "norm_ratio_range": transplant.get("norm_ratio_range")}


# --- exploratory ------------------------------------------------------------------------------------------------

def build_overlap(r11_net: dict | None, r12_net: dict | None) -> dict | None:
    """R11's and R12's lost and gained unit lists, and their overlap."""
    if r11_net is None or r12_net is None:
        return None
    lost11, lost12 = set(r11_net["lost_units"]), set(r12_net["lost_units"])
    gained11, gained12 = set(r11_net["gained_units"]), set(r12_net["gained_units"])
    return {
        "lost_both": sorted(lost11 & lost12),
        "lost_only_R11": sorted(lost11 - lost12),
        "lost_only_R12": sorted(lost12 - lost11),
        "gained_both": sorted(gained11 & gained12),
        "gained_only_R11": sorted(gained11 - gained12),
        "gained_only_R12": sorted(gained12 - gained11),
        "counts": {"lost_both": len(lost11 & lost12), "lost_only_R11": len(lost11 - lost12),
                   "lost_only_R12": len(lost12 - lost11),
                   "gained_both": len(gained11 & gained12), "gained_only_R11": len(gained11 - gained12),
                   "gained_only_R12": len(gained12 - gained11)}}


def part219_nets(part219_dir: Path, upstream) -> dict:
    """Part 219's R8 and R1 nets on this part's 16 upstream units, from its own label files."""
    r0, _e0, _m0 = load_labels(Path(part219_dir) / "labels-R0.jsonl")
    if not r0:
        return {"available": False, "because": f"{Path(part219_dir) / 'labels-R0.jsonl'} is missing or empty"}
    result = {"available": True, "dir": score219._repo_relative(Path(part219_dir))}
    for tag in ("R1", "R8"):
        lab, _errors, _missing = load_labels(Path(part219_dir) / f"labels-{tag}.jsonl")
        result[tag] = score219.net_against(r0, lab, upstream) if lab else None
    return result


def build_exploratory(out: Path, labels: dict, nets: dict, runs: dict, status: dict, split: dict,
                      part219_dir: Path) -> dict:
    """Everything the design reports as exploratory: R14 to R17 against R0 (each unreadable below the bar), the five
    downstream units, all 21 counts, R11's and R12's overlap, and R11's net plus Part 219's R8 net against R1's."""
    upstream, downstream = split["upstream"], split["downstream"]
    reference = labels.get("R0")
    nets_against_r0, exploratory_runs = {}, {}
    if reference is not None:
        for tag in RUN_TAGS:
            if tag == "R0" or tag not in labels:
                continue
            nets_against_r0[tag] = {
                "upstream": score219.net_against(reference, labels[tag], upstream),
                "downstream": score219.net_against(reference, labels[tag], downstream),
                "all_21": score219.net_against(reference, labels[tag], upstream + downstream),
                "readable": not (status.get(tag) or [])}
    for tag in EXPLORATORY_RUNS:
        entry = {"readable": tag in runs and not (status.get(tag) or []),
                 "status": list(status.get(tag) or [])}
        if tag in runs and entry["readable"]:
            bars = coherence_bar(runs[tag])
            entry["coherence_bar"] = bars
            entry["below_coherence_bar"] = not bars["held"]
            entry["net_upstream"] = (nets_against_r0.get(tag) or {}).get("upstream")
            entry["verdict"] = ("unreadable: below the coherence bar" if not bars["held"]
                                else "read (exploratory)")
        else:
            entry["verdict"] = "unreadable: the run is not read"
        exploratory_runs[tag] = entry
    part219 = part219_nets(part219_dir, upstream)
    combined = None
    r11_net = (nets_against_r0.get("R11") or {}).get("upstream")
    if part219.get("available") and part219.get("R8") and r11_net:
        sum_net = r11_net["net"] + part219["R8"]["net"]
        r1_net = (part219.get("R1") or {}).get("net")
        combined = {"R11_net": r11_net["net"], "part219_R8_net": part219["R8"]["net"], "sum": sum_net,
                    "part219_R1_net": r1_net,
                    "difference_from_part219_R1": None if r1_net is None else sum_net - r1_net,
                    "note": "a sum, not a decomposition: the parts interact through the shared QK and "
                            "post-attention norms (PART221-DESIGN.md, Limits)"}
    return {"nets_against_R0": nets_against_r0, "R14_to_R17": exploratory_runs,
            "R11_R12_overlap": build_overlap((nets_against_r0.get("R11") or {}).get("upstream"),
                                             (nets_against_r0.get("R12") or {}).get("upstream")),
            "part219": part219, "R11_plus_part219_R8_against_R1": combined}


# --- headlines --------------------------------------------------------------------------------------------------

def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def build_headlines(predictions: dict, coupling: dict) -> list[str]:
    """The design's headline rules, each stated as the intervention it is and with the coupling numbers in the same
    sentence. 221.3 adds its clause only when it was read (held, or failed on its test with R12 coherent)."""
    verdict = lambda key: (predictions.get(key) or {}).get("verdict", "")
    if verdict("221.1").startswith("not read: gate failed"):
        return ["no headline: the gate failed, so every prediction is frozen and every arm is reported as exploratory"]
    if verdict("221.1") == "held" and verdict("221.2") == "held":
        base = ("transplanting the other sixty early heads removes nearly as much as the whole band does "
                "(at least the band's net loss minus 2)")
    elif verdict("221.1") == "held":
        base = ("transplanting the other sixty removes much of what the whole band removes, the whole band more")
    elif verdict("221.1") == "failed":
        base = ("neither transplanting the four heads alone nor transplanting the other sixty alone removes most "
                "slot personas; the whole band's change does")
    else:
        return [f"no headline: 221.1 is {verdict('221.1') or 'not read'}, so no headline rule applies"]
    numbers = []
    for tag in ("R11", "R12"):
        numbers.append(f"{tag}: largest kept-head pre-norm drift "
                       f"{_pct((coupling.get(tag) or {}).get('max_kept_head_drift'))}, largest |change in "
                       f"rms(attention output)| {_pct((coupling.get(tag) or {}).get('max_abs_rms_attention_output_change'))}")
    sentence = (f"{base}, with the four heads' weights left as Instruct's, under an early attention whose "
                f"normalisation context and other heads changed (" + "; ".join(numbers) + ")")
    if verdict("221.3") == "held":
        sentence += ", and post-training's direction matters, not only the size of the change"
    elif verdict("221.3") == "failed":
        sentence += (", and R12, the one opposite direction tested, removed R11's count within the test's resolution")
    return [sentence]


# --- the whole scorecard ----------------------------------------------------------------------------------------

def build_scorecard(out: Path, kinds_path, neurons_path, kernel_path: Path, part219_dir: Path) -> dict:
    kind_of = score219.load_kinds(kinds_path)
    run_path = Path(out) / "R0.jsonl"
    if not run_path.exists():
        raise SystemExit(f"{run_path} is missing: score a directory that holds the runs")
    ids, slot, concept = read101.unit_sets(kind_of, run_path)
    split = score219.slot_split(neurons_path, kind_of, HEAD_LAYERS)
    manifest_path = Path(out) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    reproduction = (manifest.get("checks") or {}).get("reproduction") or {}
    runs, labels = {}, {}
    for tag in RUN_TAGS:
        if not (Path(out) / f"{tag}.jsonl").exists():
            continue
        labels_dir, because = label_source(tag, Path(out), part219_dir, reproduction)
        if not (labels_dir / f"labels-{tag}.jsonl").exists() and tag in ("R0", "R1"):
            labels_dir, because = Path(out), "this part's labels (the other directory's are missing)"
        runs[tag] = score_run(labels_dir, tag, ids, slot, concept)
        runs[tag]["labels_source"] = because
        if runs[tag]["labelled"]:
            labels[tag] = load_labels(labels_dir / f"labels-{tag}.jsonl")[0]
    status = {tag: score219.readable_reasons(tag, runs, manifest) for tag in runs}
    if "R13" in runs and not runs["R13"]["labelled"] and (reproduction.get("r13_vs_r1") or {}).get("identical"):
        # The design: "R0, R1 and R13 are labelled only if not identical to the rows they must match." An R13 that
        # equals R1 therefore carries R1's labels; that is the design's state, not an unreadable run.
        status["R13"] = []
        runs["R13"]["note"] = ("not labelled: R13 equals R1 token for token, so its rows are read through R1's "
                               "labels (PART221-DESIGN.md, Method)")
    nets = {}
    if "R0" in labels:
        for tag in runs:
            if tag == "R0" or tag not in labels:
                continue
            nets[tag] = score219.net_against(labels["R0"], labels[tag], split["upstream"])
    for tag in runs:
        runs[tag]["upstream_units"] = list(split["upstream"])
    gate = build_gate(score219.kernel_check(Path(kernel_path)), runs, status, reproduction)
    predictions = build_predictions(gate, runs, labels, nets, status, split["upstream"])
    coupling = {tag: coupling_numbers(Path(out), tag) for tag in TRANSPLANT_RUNS
                if (Path(out) / f"{tag}.meta.json").exists()}
    exploratory = build_exploratory(Path(out), labels, nets, runs, status, split, part219_dir)
    return {
        "part": PART,
        "design": "PART221-DESIGN.md v3",
        "n_slot": len(slot), "n_concept": len(concept),
        "split": {**split, "n_upstream": len(split["upstream"]), "n_downstream": len(split["downstream"])},
        "gate": gate,
        "reproduction": reproduction,
        "runs": {tag: {**entry, "status": status.get(tag, [])} for tag, entry in runs.items()},
        "nets_upstream_against_R0": nets,
        "coupling": coupling,
        "predictions": predictions,
        "exploratory": exploratory,
        "headlines": build_headlines(predictions, coupling),
    }


def markdown(scorecard: dict) -> str:
    gate = scorecard["gate"]
    lines = [f"# Part {PART} scorecard", ""]
    if not gate["passed"]:
        lines += ["**The gate failed, so every prediction is frozen and every arm is reported as exploratory. The "
                  f"failure: {'; '.join(gate['failures'])}.**", ""]
    lines += [f"Slot units {scorecard['n_slot']}, concept units {scorecard['n_concept']}. Upstream "
              f"{scorecard['split']['n_upstream']} (layer >= {scorecard['split']['deepest_head_layer']}), downstream "
              f"{scorecard['split']['n_downstream']}.", "", f"## Gate: {'passed' if gate['passed'] else 'FAILED'}"]
    for name, check in gate["checks"].items():
        if name == "kernel_check":
            lines.append(f"- kernel check: {'passes' if check['passes'] else 'FAILS'} "
                         f"(`{check.get('candidate')}` against `{check.get('reference', '-')}`)"
                         + (f" — {check['reason']}" if check.get("reason") else ""))
        elif name.endswith("_slot_persona"):
            lines.append(f"- {name}: {check.get('count', '-')} of {check.get('of', '-')} "
                         f"({'passes' if check.get('passes') else 'FAILS'})"
                         + (f" — {check['reason']}" if check.get("reason") else ""))
        else:
            lines.append(f"- {name}: {'passes' if check.get('passes') else 'FAILS'}"
                         + (f" — {check['reason']}" if check.get("reason") else ""))
    lines += ["", "## Runs", "",
              "| run | slot persona | slot coherent | concept persona | unpushed incoherent | inconclusive | "
              "two-pass gate | readable | labels |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag, run in scorecard["runs"].items():
        gate_text = "pass" if run["two_pass"]["passes"] else ("not labelled" if not run["labelled"] else "fail")
        lines.append(f"| {tag} | {run['slot_who'][0]}/{run['slot_who'][1]} | "
                     f"{run['slot_who_coherent'][0]}/{run['slot_who_coherent'][1]} | "
                     f"{run['concept_any'][0]}/{run['concept_any'][1]} | {run['unpushed_incoherent']} | "
                     f"{run['inconclusive']} | {gate_text} | "
                     f"{'no' if run['status'] else 'yes'} | {run.get('labels_source', '')} |")
    lines += ["", "## Predictions", "", "| prediction | verdict | numbers |", "| --- | --- | --- |"]
    for key in ("221.1", "221.2", "221.3"):
        value = scorecard["predictions"].get(key) or {}
        detail = value.get("detail") or {}
        numbers = ""
        if isinstance(detail, dict) and "net" in detail:
            numbers = f"net {detail['net']}, McNemar p {detail['mcnemar_p']:.4g}"
        elif isinstance(detail, dict) and "excess_over_additive" in detail:
            numbers = (f"R11 net {detail['R11_net']}, R1 net {detail['R1_net']} "
                       f"(required {detail['required']})")
        elif isinstance(detail, dict) and "only_second" in detail:
            numbers = (f"showing in R12 not R11 {detail['only_second']}, in R11 not R12 {detail['only_first']}, "
                       f"McNemar p {detail['mcnemar_p']:.4g}")
        elif isinstance(detail, dict) and "slot_who_coherent" in detail:
            numbers = (f"slot coherent {detail['slot_who_coherent'][0]}/{detail['slot_who_coherent'][1]}, "
                       f"concept {detail['concept_any'][0]}/{detail['concept_any'][1]}, "
                       f"inconclusive {detail['inconclusive']}")
        lines.append(f"| {key} | {value.get('verdict', '')} | {numbers} |")
    lines += ["", "## Coupling (layers 0 to 3)", ""]
    for tag in ("R11", "R12"):
        numbers = scorecard["coupling"].get(tag)
        if not numbers:
            lines.append(f"- {tag}: no meta")
            continue
        drift = numbers.get("kept_head_drift") or {}
        parts = [f"- {tag}: largest kept-head pre-norm drift {_pct(numbers.get('max_kept_head_drift'))}"
                 f" (layer {drift.get('max_kept_head_drift_layer')}), largest |change in rms(attention output)| "
                 f"{_pct(numbers.get('max_abs_rms_attention_output_change'))}"
                 f" (layer {drift.get('max_abs_rms_attention_output_change_layer')})"]
        sign = numbers.get("sign_flip_receipt") or {}
        ratio = numbers.get("norm_ratio_range") or {}
        if sign:
            per_layer = "; ".join(
                f"L{entry['layer']} q {entry['q_norm']['flips']}/{entry['q_norm']['elements']}, "
                f"k {entry['k_norm']['flips']}/{entry['k_norm']['elements']}" for entry in sign["per_layer"])
            parts.append(f"  sign flips (zero is its own sign): {sign['total_flips']} of {sign['total_elements']} "
                         f"written elements ({per_layer})")
        if ratio:
            parts.append(f"  norm ratio over the slice records: min {ratio['min']:.4g}, max {ratio['max']:.4g} "
                         f"({ratio['count']} slices)" if ratio.get("min") is not None else "  norm ratio: none")
        lines += parts
    exploratory = scorecard["exploratory"]
    lines += ["", "## Lost and gained units (the 16 upstream, against this part's R0)", ""]
    for tag, value in exploratory.get("nets_against_R0", {}).items():
        up = value["upstream"]
        lines.append(f"- {tag}: net {up['net']}; lost {up['lost']} {up['lost_units']}; gained {up['gained']} "
                     f"{up['gained_units']}; McNemar p {up['mcnemar_p']:.4g}")
    overlap = exploratory.get("R11_R12_overlap")
    if overlap:
        lines += ["", "R11 and R12 overlap: " + ", ".join(f"{key} {value}" for key, value in overlap["counts"].items())
                  + f". Lost in both: {overlap['lost_both']}; lost only in R11: {overlap['lost_only_R11']}; "
                    f"lost only in R12: {overlap['lost_only_R12']}."]
    lines += ["", "## Exploratory", "", "R14 to R17, one layer's complement heads each:", ""]
    for tag, entry in exploratory["R14_to_R17"].items():
        up = entry.get("net_upstream") or {}
        lines.append(f"- {tag}: {entry['verdict']}; net {up.get('net', '-')}"
                     + (f" (lost {up['lost']}, gained {up['gained']}, McNemar p {up['mcnemar_p']:.4g})"
                        if up else "") + (f"; not read: {'; '.join(entry['status'])}" if entry["status"] else ""))
    lines.append("")
    lines.append("Downstream units (5), all-21 counts, and the sum:")
    for tag, value in exploratory.get("nets_against_R0", {}).items():
        lines.append(f"- {tag}: downstream net {value['downstream']['net']} (lost {value['downstream']['lost']}, "
                     f"gained {value['downstream']['gained']}); all 21 net {value['all_21']['net']}")
    combined = exploratory.get("R11_plus_part219_R8_against_R1")
    if combined:
        lines.append(f"- R11 net + Part 219's R8 net against Part 219's R1 net: {combined['R11_net']} + "
                     f"{combined['part219_R8_net']} = {combined['sum']} against {combined['part219_R1_net']} "
                     f"(difference {combined['difference_from_part219_R1']}). {combined['note']}")
    elif not exploratory.get("part219", {}).get("available"):
        lines.append("- R11 net plus Part 219's R8 net: Part 219's labels are not available here "
                     f"({exploratory.get('part219', {}).get('because')})")
    lines += ["", "## Headlines", ""]
    lines += [f"- {headline}" for headline in scorecard["headlines"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(REPO / "results" / "part221"))
    parser.add_argument("--kinds", default=str(REPO / "results" / "part92" / "step2" / "kinds.json"))
    parser.add_argument("--neurons", default=str(REPO / "results" / "part92" / "step2" / "neurons.json"))
    parser.add_argument("--kernel-check", default=str(REPO / "results" / "part219" / "kernel-check.json"))
    parser.add_argument("--part219-dir", default=str(REPO / "results" / "part219"),
                        help="Part 219's results: the reproduction's labels for R0/R1 and the R8/R1 nets")
    args = parser.parse_args(argv)
    out = Path(args.out)
    scorecard = build_scorecard(out, args.kinds, args.neurons, Path(args.kernel_check), Path(args.part219_dir))
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=1) + "\n", encoding="utf-8")
    (out / "SCORECARD.md").write_text(markdown(scorecard), encoding="utf-8")
    print(markdown(scorecard))
    return 0


if __name__ == "__main__":
    sys.exit(main())
