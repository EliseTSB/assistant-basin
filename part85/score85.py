#!/usr/bin/env python3
"""Blind and score Part 85 answers, readers, and the registered decision rules."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoints import BY_KEY, CHECKPOINTS, POST_TRAINED
from part85 import CELL_NAMES, QUESTIONS, expected_cell_specs

ORIGIN_LABELS = ("EARTH", "OFFWORLD", "NEITHER")
FORM_LABELS = ("OK", "BROKEN")
def collect_answers(paths=(), results_root=None, mode=None, attempt=1):
    files = [Path(p) for p in paths]
    if results_root:
        files.extend(sorted(Path(results_root).glob("*/answers/*.jsonl")))
    unique = list(dict.fromkeys(files))
    records = []
    for path in unique:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if mode is not None and row.get("mode") != mode:
                continue
            if mode == "pilot" and row.get("attempt", 1) != attempt:
                continue
            text = row.get("answer", row.get("text"))
            if not isinstance(text, str):
                raise ValueError(f"{path}:{line_number} lacks string answer/text")
            for field in ("checkpoint", "frame", "cell", "question_index", "sample", "seed"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number} lacks {field}")
            records.append({**row, "answer": text})
    if not records:
        raise ValueError("no answer records found")
    return records


def _item_key(key):
    if not isinstance(key, dict) or not isinstance(key.get("items"), dict):
        raise ValueError("answer key must contain an items mapping and items_sha256 metadata")
    if not isinstance(key.get("items_sha256"), str):
        raise ValueError("answer key is missing items_sha256")
    return key["items"]


def _check_items_digest(key, items_path=None):
    item_key = _item_key(key)
    path = Path(items_path or key.get("items_path", ""))
    if not str(path) or not path.is_file():
        raise ValueError(f"items file is required to verify the answer key SHA-256: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != key["items_sha256"]:
        raise ValueError("items file SHA-256 does not match the answer key")
    if any(not isinstance(item_id, str) or not isinstance(meta, dict)
           for item_id, meta in item_key.items()):
        raise ValueError("answer key items mapping must contain string IDs and object metadata")


def prepare(records, items_path, key_path, force=False):
    """Blind records under cryptographic random IDs and keep the key separate."""
    items_path, key_path = Path(items_path), Path(key_path)
    if not force and (items_path.exists() or key_path.exists()):
        existing = items_path if items_path.exists() else key_path
        raise FileExistsError(f"refusing to overwrite {existing}; pass --force to replace prepared files")
    rows = list(records)
    secrets.SystemRandom().shuffle(rows)
    ids = []
    seen = set()
    for _ in rows:
        while True:
            item_id = f"p85-{secrets.token_hex(8)}"
            if item_id not in seen:
                seen.add(item_id)
                ids.append(item_id)
                break
    key = {}
    items_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    item_lines = []
    for item_id, row in zip(ids, rows):
        question = row.get("question")
        if not isinstance(question, str):
            raise ValueError(f"answer record {item_id} lacks a string question")
        item_lines.append(json.dumps({"id": item_id, "question": question, "text": row["answer"]},
                                     ensure_ascii=False) + "\n")
        keep = ("checkpoint", "frame", "cell", "question_index", "sample", "seed", "mode", "attempt",
                "frame_template_sha256", "question", "pilot_gate_action")
        key[item_id] = {k: row[k] for k in keep if k in row}
    item_bytes = "".join(item_lines).encode("utf-8")
    envelope = {"schema_version": 1, "items_sha256": hashlib.sha256(item_bytes).hexdigest(),
                "items_path": str(items_path.resolve()), "items": key}
    item_mode = "w" if force else "x"
    key_mode = "w" if force else "x"
    with items_path.open(item_mode, encoding="utf-8") as stream:
        stream.write(item_bytes.decode("utf-8"))
    with key_path.open(key_mode, encoding="utf-8") as stream:
        stream.write(json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return envelope


def validate_record_counts(records, mode):
    specs = expected_cell_specs(mode)
    expected = {(row["checkpoint"], row["frame"], row["cell"]):
                len(QUESTIONS) * row["samples_per_question"] for row in specs}
    actual = Counter((row["checkpoint"], row["frame"], row["cell"]) for row in records)
    if mode == "main" and set(actual) != set(expected):
        chat_only = {(row["checkpoint"], row["frame"], row["cell"]):
                     len(QUESTIONS) * row["samples_per_question"]
                     for row in specs if row["frame"] == "chat"}
        authorized = all(row.get("pilot_gate_action") == "chat_only_posttrained" for row in records)
        if authorized and set(actual) == set(chat_only) and all(actual[k] == chat_only[k] for k in chat_only):
            expected = chat_only
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"{mode} answer cells incomplete: missing={missing[:3]}, unexpected={extra[:3]}")
    wrong = {group: (actual[group], expected[group]) for group in expected if actual[group] != expected[group]}
    if wrong:
        raise ValueError(f"{mode} answer counts do not match the registered cell totals: {wrong}")
    for group, sample_count in expected.items():
        selected = [row for row in records
                    if (row["checkpoint"], row["frame"], row["cell"]) == group]
        pairs = []
        for row in selected:
            qi, sample = row.get("question_index"), row.get("sample")
            if not isinstance(qi, int) or not 0 <= qi < len(QUESTIONS):
                raise ValueError(f"answer cell {group} has invalid question_index {qi!r}")
            if not isinstance(sample, int) or not 0 <= sample < sample_count // len(QUESTIONS):
                raise ValueError(f"answer cell {group} has invalid sample {sample!r}")
            pair = (qi, sample)
            if pair in pairs:
                raise ValueError(f"answer cell {group} repeats question/sample pair {pair}")
            pairs.append(pair)
            expected_seed = 85000 + 100 * qi + sample
            if row.get("seed") != expected_seed:
                raise ValueError(f"answer cell {group} seed for pair {pair} is {row.get('seed')!r}; expected {expected_seed}")
        expected_pairs = {(qi, sample) for qi in range(len(QUESTIONS))
                          for sample in range(sample_count // len(QUESTIONS))}
        if set(pairs) != expected_pairs:
            raise ValueError(f"answer cell {group} question/sample pairs are incomplete")


def _label_rows(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{n} label row must be a JSON object")
            rows.append(row)
    return rows


def validate_labels(labels, key, require_responsive=False):
    key = _item_key(key)
    seen = {}
    for row in labels:
        item_id = row.get("id")
        if item_id in seen:
            raise ValueError(f"duplicate label for {item_id}")
        if item_id not in key:
            raise ValueError(f"label id {item_id!r} is not in the answer key")
        if row.get("origin") not in ORIGIN_LABELS:
            raise ValueError(f"invalid origin label for {item_id}: {row.get('origin')!r}")
        if row.get("form") not in FORM_LABELS:
            raise ValueError(f"invalid form label for {item_id}: {row.get('form')!r}")
        if require_responsive and not isinstance(row.get("responsive"), bool):
            raise ValueError(f"pilot label for {item_id} needs a boolean responsive field")
        seen[item_id] = row
    missing = set(key) - set(seen)
    if missing:
        raise ValueError(f"missing labels for {len(missing)} key ids; first missing {sorted(missing)[0]}")
    return seen


def grouped_counts(labels, key):
    key = _item_key(key)
    groups = defaultdict(lambda: {"n": 0, **{x: 0 for x in ORIGIN_LABELS}, **{x: 0 for x in FORM_LABELS}})
    for item_id, row in labels.items():
        meta = key[item_id]
        group = (meta["checkpoint"], meta["frame"], meta["cell"])
        counts = groups[group]
        counts["n"] += 1
        counts[row["origin"]] += 1
        counts[row["form"]] += 1
    return {"/".join(group): counts for group, counts in sorted(groups.items())}


def fisher_exact_greater(treated_offworld: int, treated_n: int, control_offworld: int, control_n: int) -> float:
    """One-sided Fisher exact p value for treated OFFWORLD proportion > baseline."""
    vals = (treated_offworld, treated_n, control_offworld, control_n)
    if any(int(x) != x or x < 0 for x in vals) or treated_offworld > treated_n or control_offworld > control_n:
        raise ValueError(f"invalid contingency counts {vals}")
    successes = treated_offworld + control_offworld
    population = treated_n + control_n
    lo = max(0, treated_n - (population - successes))
    hi = min(treated_n, successes)
    denom = math.comb(population, treated_n)
    tail = sum(math.comb(successes, x) * math.comb(population - successes, treated_n - x)
               for x in range(max(lo, treated_offworld), hi + 1))
    return min(1.0, tail / denom) if denom else 1.0


def shows_push(treated: dict, baseline: dict, control: dict | None) -> dict:
    """The registered push rule, including the same-checkpoint control gate."""
    if not treated or not baseline:
        return {"shows_push": None, "p_one_sided": None, "reason": "missing treated or unclamped counts"}
    if control is None:
        return {"shows_push": None, "p_one_sided": None, "reason": "missing control counts"}
    if control.get("OFFWORLD", 0) > 2:
        return {"shows_push": False, "p_one_sided": None,
                "reason": f"control gate failed ({control.get('OFFWORLD', 0)} > 2)"}
    a, n1 = treated.get("OFFWORLD", 0), treated.get("n", 0)
    b, n0 = baseline.get("OFFWORLD", 0), baseline.get("n", 0)
    if not n1 or not n0:
        return {"shows_push": None, "p_one_sided": None, "reason": "missing answer counts"}
    p = fisher_exact_greater(a, n1, b, n0)
    result = a > b and p < 0.05
    return {"shows_push": result, "p_one_sided": p,
            "reason": f"treated {a}/{n1}, unclamped {b}/{n0}, control {control.get('OFFWORLD', 0)}/{control.get('n', 0)}"}


def same_unit_gate(step0: dict | None, step1: dict | None):
    if not step0 or not step1:
        return {"same_unit": None, "reason": "missing step 0 or step 1 JSON"}
    cosines = step0.get("measures", {}).get("cosines_with_instruct") or {}
    cosine = cosines.get("down_proj_column_6978")
    corr = step1.get("neutral_correlations_with_instruct", {}).get("target")
    if cosine is None or corr is None:
        return {"same_unit": None, "reason": "missing down_proj cosine or activation correlation"}
    same = float(cosine) >= 0.8 and float(corr) >= 0.8
    return {"same_unit": same, "down_proj_cosine": float(cosine),
            "neutral_activation_correlation": float(corr),
            "reason": "both thresholds met" if same else "different unit at that index"}


def earliest_suffix_onset(checkpoints, push_by_checkpoint, valid_by_checkpoint):
    """Find the first push whose full later suffix is valid and continues to show the push."""
    for index, checkpoint in enumerate(checkpoints):
        suffix = checkpoints[index:]
        if all(valid_by_checkpoint.get(key) is True and push_by_checkpoint.get(key) is True for key in suffix):
            return checkpoint
    return None


def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _step_data(results_root):
    root = Path(results_root)
    return ({cp.key: _read_json(root / cp.key / "step0.json") for cp in CHECKPOINTS},
            {cp.key: _read_json(root / cp.key / "step1.json") for cp in CHECKPOINTS})


def _step_data_checked(results_root, partial=False, scope="plain_frame_pass"):
    step0, step1 = _step_data(results_root)
    required_keys = POST_TRAINED if scope == "chat_only_posttrained" else tuple(step0)
    missing = [f"{key}/{name}.json" for key in required_keys
               for name, values in (("step0", step0), ("step1", step1)) if values[key] is None]
    if missing and not partial:
        expectation = "in-scope post-trained" if scope == "chat_only_posttrained" else "all 12"
        raise ValueError(f"main scoring requires {expectation} step0 and step1 files; missing {missing[:4]}")
    for key in required_keys:
        for name, values in (("step0", step0), ("step1", step1)):
            payload = values[key]
            if payload is None:
                continue
            identity = payload.get("checkpoint") or {}
            if identity.get("key") != key:
                raise ValueError(f"{key}/{name}.json identifies checkpoint {identity.get('key')!r}, expected {key!r}")
            commit = identity.get("commit")
            if not isinstance(commit, str) or not commit.startswith(BY_KEY[key].commit_prefix):
                raise ValueError(f"{key}/{name}.json commit does not match registered prefix {BY_KEY[key].commit_prefix}")
    return step0, step1


def _validate_main_scope(item_key, partial=False):
    expected_rows = expected_cell_specs("main")
    expected_full = {(row["checkpoint"], row["frame"], row["cell"]):
                     len(QUESTIONS) * row["samples_per_question"] for row in expected_rows}
    actual = Counter((meta.get("checkpoint"), meta.get("frame"), meta.get("cell")) for meta in item_key.values())
    if actual == Counter(expected_full):
        if any(meta.get("pilot_gate_action") not in ("plain_frame_pass", "chat_only_posttrained")
               for meta in item_key.values()):
            raise ValueError("main answer key lacks a recorded pilot gate action")
        if any(meta.get("pilot_gate_action") != "plain_frame_pass" for meta in item_key.values()):
            raise ValueError("full main answer scope must be authorized by a plain_frame_pass pilot gate")
        return "plain_frame_pass"
    expected_chat = {(row["checkpoint"], row["frame"], row["cell"]):
                     len(QUESTIONS) * row["samples_per_question"]
                     for row in expected_rows if row["frame"] == "chat"}
    if actual == Counter(expected_chat):
        if any(meta.get("pilot_gate_action") != "chat_only_posttrained" for meta in item_key.values()):
            raise ValueError("chat-only answer key is not authorized by its recorded pilot gate")
        return "chat_only_posttrained"
    if partial:
        return "partial"
    raise ValueError("main answer key must contain all 2,880 answers or the pilot-authorized chat-only scope")


def _count(counts, key, frame="plain", cell="unclamped"):
    return counts.get(f"{key}/{frame}/{cell}")


def _metric(data, *path):
    for part in path:
        if not isinstance(data, dict):
            return None
        data = data.get(part)
    return data


def _verdict(state, evidence):
    return {"verdict": state, "evidence": evidence}


def evaluate_decisions(counts: dict, step0: dict, step1: dict):
    """Evaluate 85.1–85.9 from first-reader counts and step 0/1 records."""
    order = [cp.key for cp in CHECKPOINTS]
    gates = {key: same_unit_gate(step0.get(key), step1.get(key)) for key in order}
    plain = {(key, cell): _count(counts, key, "plain", cell) for key in order for cell in CELL_NAMES}

    push = {"own_max": {}, "absolute": {}}
    p_details = {"own_max": {}, "absolute": {}}
    for cell in ("own_max", "absolute"):
        for key in order:
            if gates[key]["same_unit"] is not True:
                detail = {"shows_push": None, "p_one_sided": None, "reason": gates[key]["reason"]}
            else:
                detail = shows_push(plain[(key, cell)], plain[(key, "unclamped")], plain[(key, "control")])
            p_details[cell][key] = detail
            push[cell][key] = detail["shows_push"]

    valid = {key: gates[key]["same_unit"] is True for key in order}
    own_onset = earliest_suffix_onset(order, push["own_max"], valid)
    absolute_onset = earliest_suffix_onset(order, push["absolute"], valid)
    if own_onset is not None and absolute_onset is not None:
        calibration_gap = abs(order.index(own_onset) - order.index(absolute_onset))
    else:
        calibration_gap = None
    calibration_disagreement = ((own_onset is None) != (absolute_onset is None) or
                                (calibration_gap is not None and calibration_gap > 1))

    decisions = {}
    post_lineage = order[8:12]  # mid-training end, SFT, DPO, Instruct
    p851_values = []
    for key in post_lineage:
        cosine = _metric(step0.get(key), "measures", "cosines_with_instruct", "down_proj_column_6978")
        corr = _metric(step1.get(key), "neutral_correlations_with_instruct", "target")
        p851_values.append((key, cosine, corr))
    if any(c is None or a is None for _, c, a in p851_values):
        decisions["85.1"] = _verdict("NOT READ", "missing step 0 cosine or step 1 activation correlation for mid-training end through Instruct")
    else:
        holds = all(float(c) >= 0.9 and float(a) >= 0.8 for _, c, a in p851_values)
        ev = "; ".join(f"{k}: cosine {float(c):.3f}, activation r {float(a):.3f}" for k, c, a in p851_values)
        decisions["85.1"] = _verdict("HOLDS" if holds else "FAILS", ev)

    first_cos = _metric(step0.get(order[0]), "measures", "cosines_with_instruct", "down_proj_column_6978")
    decisions["85.2"] = (_verdict("NOT READ", "missing 21B down_proj cosine") if first_cos is None else
                         _verdict("HOLDS" if float(first_cos) < 0.5 else "FAILS", f"21B down_proj cosine {float(first_cos):.3f}; threshold <0.5"))

    z = _metric(step0.get("pretrain_4T"), "measures", "direct_readout", "z_score")
    decisions["85.3"] = (_verdict("NOT READ", "missing end-of-pretraining direct-readout z-score") if z is None else
                         _verdict("HOLDS" if float(z) > 2 else "FAILS", f"end-of-pretraining direct-readout z-score {float(z):.3f}; threshold >2"))

    passage = step1.get("pretrain_4T")
    off_mean = _metric(passage, "passages", "offworld_mean")
    ordinary_mean = _metric(passage, "passages", "ordinary_mean")
    ordinary_sd = _metric(passage, "passages", "ordinary_sd")
    if off_mean is None or ordinary_mean is None or ordinary_sd is None:
        decisions["85.4"] = _verdict("NOT READ", "missing end-of-pretraining passage means or ordinary SD")
    else:
        diff = float(off_mean) - float(ordinary_mean)
        decisions["85.4"] = _verdict("HOLDS" if diff >= float(ordinary_sd) else "FAILS",
                                    f"off-world mean {float(off_mean):.3f} minus ordinary mean {float(ordinary_mean):.3f} = {diff:.3f}; ordinary SD {float(ordinary_sd):.3f}")

    mid_push = push["own_max"].get("midtrain_51B")
    sft_push = push["own_max"].get("sft")
    if not valid.get("midtrain_51B") or not valid.get("sft") or mid_push is None or sft_push is None:
        decisions["85.5"] = _verdict("NOT READ", "mid-training end or SFT is not confirmed as the same unit / has incomplete counts")
        creation = "NOT READ"
        creation_evidence = "same-unit gate or step-2 counts incomplete"
    elif mid_push:
        decisions["85.5"] = _verdict("HOLDS", "mid-training end own-max cell shows the push")
        creation = "present before post-training"
        creation_evidence = "mid-training end own-max cell shows the push"
    elif sft_push:
        decisions["85.5"] = _verdict("FAILS", "mid-training end does not show the push, while SFT does")
        creation = "created in post-training"
        creation_evidence = "mid-training end own-max cell fails; SFT own-max cell shows the push"
    else:
        decisions["85.5"] = _verdict("FAILS", "mid-training end own-max cell does not show the push")
        creation = "not established"
        creation_evidence = "neither registered pre/post comparison establishes the predicted push"

    early = [plain[(key, "own_max")] for key in order[:2]]
    if any(row is None for row in early) or any(not valid[key] for key in order[:2]):
        decisions["85.6"] = _verdict("NOT READ", "21B or 210B same-unit gate failed or own-max counts are missing")
    else:
        values = [row.get("OFFWORLD", 0) for row in early]
        decisions["85.6"] = _verdict("HOLDS" if all(x <= 2 for x in values) else "FAILS",
                                    f"21B own-max {values[0]}/48; 210B own-max {values[1]}/48; each threshold <=2")

    mid = plain[("midtrain_51B", "own_max")]
    inst = plain[("instruct", "own_max")]
    if not valid.get("midtrain_51B") or not valid.get("instruct") or mid is None or inst is None:
        decisions["85.7"] = _verdict("NOT READ", "mid-training end or Instruct same-unit gate / own-max count is incomplete")
    else:
        delta = inst.get("OFFWORLD", 0) - mid.get("OFFWORLD", 0)
        decisions["85.7"] = _verdict("HOLDS" if delta >= 8 else "FAILS",
                                    f"Instruct {inst.get('OFFWORLD', 0)}/48 minus mid-training end {mid.get('OFFWORLD', 0)}/48 = {delta}; threshold >=8")

    control_rows = [plain[(key, "control")] for key in order]
    if any(row is None for row in control_rows):
        decisions["85.8"] = _verdict("NOT READ", "one or more plain-frame control counts are missing")
    else:
        bad = [(key, row.get("OFFWORLD", 0)) for key, row in zip(order, control_rows) if row.get("OFFWORLD", 0) > 2]
        ev = "all 12 plain-frame control cells <=2 OFFWORLD" if not bad else "; ".join(f"{k} {n}/48" for k, n in bad)
        decisions["85.8"] = _verdict("HOLDS" if not bad else "FAILS", ev)

    return {"verdicts": decisions, "same_unit_gates": gates,
            "push_details": p_details,
            "onset": {"own_max": own_onset, "absolute": absolute_onset,
                      "calibration_checkpoint_gap": calibration_gap,
                      "calibration_disagreement": calibration_disagreement,
                      "reported_onset": None if calibration_disagreement else own_onset,
                      "interpretation": ("calibration disagreement: report instead of an onset" if calibration_disagreement
                                         else (f"own-max onset {own_onset}" if own_onset else "no qualifying onset; inspect push pattern"))},
            "created_or_amplified": {"finding": creation, "evidence": creation_evidence}}


def pilot_gate(counts, responsive_ok_count=None, attempt=1):
    end_unclamped = _count(counts, "pretrain_4T", "plain", "unclamped")
    instruct_own = _count(counts, "instruct", "plain", "own_max")
    if end_unclamped is None or instruct_own is None:
        return {"action": "pilot_incomplete", "evidence": "missing end-of-pretraining unclamped or Instruct own-max pilot answers"}
    if responsive_ok_count is None:
        return {"action": "pilot_responsiveness_missing", "evidence": "pilot reader must record whether each end-of-pretraining unclamped answer responds to the question"}
    quality_pass = responsive_ok_count >= 16
    instruct_offworld = instruct_own.get("OFFWORLD", 0)
    effect_pass = instruct_offworld >= 5
    if not quality_pass:
        action = "amend_frame_and_rerun_once" if attempt == 1 else "quality_failed_after_rerun"
    elif not effect_pass:
        action = "chat_only_posttrained"
    else:
        action = "plain_frame_pass"
    return {"action": action,
            "end_pretraining_unclamped_ok_and_responsive": responsive_ok_count,
            "end_pretraining_unclamped_ok": end_unclamped.get("OK", 0),
            "end_pretraining_quality_threshold": 16,
            "instruct_own_max_offworld": instruct_offworld,
            "instruct_offworld_threshold": 5,
            "evidence": f"end-of-pretraining unclamped OK and responsive={responsive_ok_count}/24 (OK alone={end_unclamped.get('OK', 0)}/24); Instruct own-max OFFWORLD={instruct_offworld}/24"}


def _kappa(a, b, labels):
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[label] * counts_b[label] for label in labels) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1.0 - expected)


def reader_agreement(labels1, labels2, key):
    key = _item_key(key)
    ids = sorted(key)
    origin_a = [labels1[i]["origin"] for i in ids]
    origin_b = [labels2[i]["origin"] for i in ids]
    form_a = [labels1[i]["form"] for i in ids]
    form_b = [labels2[i]["form"] for i in ids]
    result = {
        "origin_nominal": _kappa(origin_a, origin_b, ORIGIN_LABELS),
        "form_nominal": _kappa(form_a, form_b, FORM_LABELS),
        "binary_per_label": {},
    }
    for label in ORIGIN_LABELS:
        result["binary_per_label"][label] = _kappa([x == label for x in origin_a], [x == label for x in origin_b], (False, True))
    for label in FORM_LABELS:
        result["binary_per_label"][label] = _kappa([x == label for x in form_a], [x == label for x in form_b], (False, True))
    return result


def score_payload(key, labels1, labels2=None, mode="main", results_root="results/part85",
                  items_path=None, partial=False):
    _check_items_digest(key, items_path)
    item_key = _item_key(key)
    checked1 = validate_labels(labels1, key, require_responsive=(mode == "pilot"))
    checked2 = validate_labels(labels2, key) if labels2 is not None else None
    counts1 = grouped_counts(checked1, key)
    counts2 = grouped_counts(checked2, key) if checked2 is not None else None
    if mode == "pilot":
        responsive_ok = sum(
            checked1[item_id]["form"] == "OK" and checked1[item_id]["responsive"] is True
            for item_id, meta in item_key.items()
            if meta.get("checkpoint") == "pretrain_4T" and meta.get("frame") == "plain" and meta.get("cell") == "unclamped"
        )
        attempts = {meta.get("attempt", 1) for meta in item_key.values()}
        if len(attempts) != 1:
            raise ValueError("pilot key mixes attempts")
        attempt = next(iter(attempts))
        if attempt not in (1, 2):
            raise ValueError("pilot attempt must be 1 or 2")
        gate = pilot_gate(counts1, responsive_ok, attempt)
        hashes = {meta.get("frame_template_sha256") for meta in item_key.values()}
        if len(hashes) != 1 or None in hashes:
            raise ValueError("pilot key must contain one shared plain-frame template SHA-256")
        gate["plain_template_sha256"] = next(iter(hashes))
        gate["attempt"] = attempt
        return {"mode": mode, "counts": counts1, "counts_reader2": counts2,
                "pilot_gate": gate,
                "reader_agreement": reader_agreement(checked1, checked2, key) if checked2 else None}
    scope = _validate_main_scope(item_key, partial=partial)
    step0, step1 = _step_data_checked(results_root, partial=partial, scope=scope)
    decisions = evaluate_decisions(counts1, step0, step1)
    plain_ids = [i for i, meta in item_key.items() if meta.get("frame") == "plain"]
    if checked2:
        agreement_n = len(plain_ids)
        # Keep the expression explicit for readers: compare the binary OFFWORLD / not OFFWORLD labels.
        agreement_count = sum((checked1[i]["origin"] == "OFFWORLD") == (checked2[i]["origin"] == "OFFWORLD") for i in plain_ids)
        agreement_rate = agreement_count / agreement_n if agreement_n else None
        decisions["verdicts"]["85.9"] = _verdict(
            "NOT READ" if agreement_rate is None else ("HOLDS" if agreement_rate >= 0.9 else "FAILS"),
            f"plain-frame OFFWORLD vs not agreement {agreement_count}/{agreement_n} = {agreement_rate:.3f}; threshold >=0.90" if agreement_rate is not None else "no plain-frame labels")
        agreement = reader_agreement(checked1, checked2, key)
    else:
        decisions["verdicts"]["85.9"] = _verdict("NOT READ", "second-reader labels are required to assess agreement")
        agreement = None
    if scope == "chat_only_posttrained":
        for code in ("85.5", "85.6", "85.7", "85.8", "85.9"):
            decisions["verdicts"][code] = _verdict("NOT APPLICABLE", "chat-only pilot scope does not include the registered plain-frame cells")
    not_read = [code for code, decision in decisions["verdicts"].items()
                if decision.get("verdict") == "NOT READ"]
    if not_read and not partial and scope != "chat_only_posttrained":
        raise ValueError(f"complete main scoring produced NOT READ verdicts {not_read}; use --partial only for an explicitly partial analysis")
    if agreement is None and not partial:
        raise ValueError("main scoring requires the second reader unless --partial is explicit")
    return {"mode": mode, "scope": scope, "counts": counts1, "counts_reader2": counts2,
            "reader_agreement": agreement, "decisions": decisions,
            "verdicts": decisions["verdicts"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="cmd", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("--answers", nargs="*", default=[])
    p.add_argument("--results-root")
    p.add_argument("--mode", required=True, choices=("pilot", "main"))
    p.add_argument("--attempt", type=int, default=1, help="pilot attempt number to blind; main uses 1")
    p.add_argument("--items", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--force", action="store_true", help="replace existing items/key files")
    s = subs.add_parser("score")
    s.add_argument("--mode", required=True, choices=("pilot", "main"))
    s.add_argument("--key", required=True)
    s.add_argument("--labels", required=True)
    s.add_argument("--labels2")
    s.add_argument("--items", help="prepared blind items file to verify against the key SHA-256")
    s.add_argument("--partial", action="store_true", help="allow incomplete main data and NOT READ verdicts")
    s.add_argument("--force", action="store_true", help="replace an existing score output")
    s.add_argument("--results-root", default="results/part85")
    s.add_argument("--out")
    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        if args.mode == "pilot" and args.attempt not in (1, 2):
            raise ValueError("pilot attempt must be 1 or the single registered rerun, 2")
        if args.mode == "main" and args.attempt != 1:
            raise ValueError("--attempt is only used by pilot mode")
        records = collect_answers(args.answers, args.results_root, args.mode, args.attempt)
        validate_record_counts(records, args.mode)
        key = prepare(records, args.items, args.key, force=args.force)
        print(f"wrote {len(key['items'])} blind items")
        return
    key = json.loads(Path(args.key).read_text(encoding="utf-8"))
    labels = _label_rows(args.labels)
    labels2 = _label_rows(args.labels2) if args.labels2 else None
    if args.mode == "main" and labels2 is None:
        raise ValueError("main scoring requires both --labels and --labels2")
    payload = score_payload(key, labels, labels2, mode=args.mode, results_root=args.results_root,
                            items_path=args.items, partial=args.partial)
    out = Path(args.out) if args.out else Path(args.key).with_name("score.json")
    if out.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite score output {out}; pass --force to replace it")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload.get("pilot_gate", payload.get("verdicts", {})), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
