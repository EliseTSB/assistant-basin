#!/usr/bin/env python3
"""Reproduce Part 177's registered definitions from its pinned local inputs."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))  # python -I leaves the script folder off the path

from loop_upstream import answer_text, last_boxed, loop_signals

ROOT = HERE.parent
DEFAULT_INPUTS = ROOT / "results" / "part177" / "inputs"
DEFAULT_OUT = ROOT / "results" / "part177"
ARMS = ("unedited", "p50c")
DONOR_SEED = 177
SUPPLEMENTAL_DONOR_SEED = DONOR_SEED + 1
WINDOW_ORDER_SEED = 177
TOKEN_CAP = 32_500
ONSET_CHAR_CAP = 50_000
DONORS_PER_TARGET = 20
ONSET_MIN_LINE_CHARS = 8
ONSET_FOLLOWING_LINES = 40
ONSET_REQUIRED_REPEATS = 3
INT_RE = re.compile(r"^[+-]?\d+$")
DECIMAL_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
RATIONAL_RE = re.compile(r"^[+-]?\d+/\d+$")
LATEX_RATIONAL_RE = re.compile(r"^\\frac\{[+-]?\d+\}\{[+-]?\d+\}$")
FRAC_RE = re.compile(r"^(?:\\frac\{[^{}]{1,3}\}\{[^{}]{1,3}\}|[+-]?\d{1,3}/\d{1,3})$")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n"
                                    for row in rows), encoding="utf-8", newline="\n")


def _normalise_with_map(value):
    """Remove the design's formatting differences and map normalized chars to raw offsets."""
    raw = str(value or "")
    out, offsets = [], []
    replacements = (("\\dfrac", "\\frac"), ("\\left", ""), ("\\right", ""),
                    ("\\!", ""), ("\\,", ""), ("$", ""))
    i = 0
    while i < len(raw):
        matched = next(((source, target) for source, target in replacements if raw.startswith(source, i)), None)
        if matched is not None:
            source, target = matched
            if target:
                # The replacement drops only the `d`; preserve the positions of the remaining command chars.
                source_start = i + (1 if source == "\\dfrac" else 0)
                for j, char in enumerate(target):
                    out.append(char)
                    offsets.append(source_start + j)
            i += len(source)
            continue
        char = raw[i]
        if not char.isspace():
            out.append(char)
            offsets.append(i)
        i += 1
    return "".join(out), offsets


def normalise_value(value):
    normalised, _ = _normalise_with_map(value)
    return normalised.rstrip(".")


def _short_token(value):
    return len(value) <= 3 and bool(value) and value.isalnum()


def literal_occurrences(text, value):
    """Return raw character spans for literal hits; integers and short tokens are standalone."""
    needle = normalise_value(value)
    if not needle:
        return []
    haystack, offsets = _normalise_with_map(text)
    if not haystack:
        return []
    numeric = bool(DECIMAL_RE.fullmatch(needle) or RATIONAL_RE.fullmatch(needle) or
                   LATEX_RATIONAL_RE.fullmatch(needle))
    short = _short_token(needle)
    found = []
    start_at = 0
    while True:
        start = haystack.find(needle, start_at)
        if start < 0:
            break
        end = start + len(needle)
        raw_start = offsets[start]
        raw_end = offsets[end - 1] + 1
        raw = str(text or "")
        left_context = raw[:raw_start]
        right_context = raw[raw_end:]
        left = left_context[-1] if left_context else ""
        right = right_context[0] if right_context else ""
        if numeric:
            # Numeric values must occupy a whole token, including decimals and fractions.
            previous = left_context[-2] if len(left_context) >= 2 else ""
            following = right_context[1] if len(right_context) >= 2 else ""
            left_continues_decimal = left == "." and previous.isdigit()
            right_continues_decimal = right == "." and following.isdigit()
            left_bad = left.isalnum() or bool(left and left in "_+-") or left_continues_decimal
            right_bad = right.isalnum() or bool(right and right in "_+-") or right_continues_decimal
            valid = not left_bad and not right_bad
        elif short:
            valid = not (left.isalnum() or left == "_") and not (right.isalnum() or right == "_")
        else:
            valid = True
        if valid:
            found.append({"start": raw_start, "end": raw_end})
        start_at = start + 1
    return found


def has_literal(text, value):
    return bool(literal_occurrences(text, value))


def selected_hit_positions(hits):
    """Choose the first, middle, and last occurrences, without duplicating a selected occurrence."""
    if not hits:
        return []
    indexes = sorted({0, len(hits) // 2, len(hits) - 1})
    return [hits[index] | {"hit_rank": rank} for rank, index in enumerate(indexes)]


def loop_onset(text, char_limit=None):
    """First >8-character normalized line repeated three more times in the following 40 lines."""
    text = str(text or "")
    searched = text if char_limit is None else text[:int(char_limit)]
    lines = []
    offsets = []
    cursor = 0
    for raw_line in searched.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        lines.append(re.sub(r"\s+", " ", line).strip())
        offsets.append(cursor)
        cursor += len(raw_line)
    for index, line in enumerate(lines):
        if len(line) <= ONSET_MIN_LINE_CHARS:
            continue
        next_lines = lines[index + 1:index + ONSET_FOLLOWING_LINES + 1]
        if sum(candidate == line for candidate in next_lines) >= ONSET_REQUIRED_REPEATS:
            return {"line_index": index, "char_index": offsets[index], "normalized_line": line}
    return None


def onset_order(first_hit, onset):
    if onset is None:
        return "no onset detected"
    if first_hit is None:
        return None
    return "hit before onset" if int(first_hit) < int(onset["char_index"]) else "onset before hit"


def _box_values(text):
    """Read brace-matched boxed/fboxed values without importing the grading package."""
    text = str(text or "")
    boxes = []
    token_re = re.compile(r"\\boxed|\\fbox")
    for match in token_re.finditer(text):
        start = match.start()
        name = match.group(0)
        rest = text[match.end():]
        stripped = rest.lstrip()
        if not stripped.startswith("{"):
            if name == "\\boxed" and rest.startswith(" "):
                value = rest.strip().split("$")[0].split("\n")[0].strip()
                if value:
                    boxes.append({"start": start, "end": match.end() + len(value), "value": value})
            continue
        brace_start = match.end() + len(rest) - len(stripped)
        depth = 0
        end = None
        for index in range(brace_start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is not None:
            value = text[brace_start + 1:end - 1].strip()
            if value:
                boxes.append({"start": start, "end": end, "value": value})
    return boxes


def box_cell(correct0, pred_first):
    if int(correct0) == 1:
        return "correct[0]"
    if pred_first is None:
        return "pred_first null"
    return "pred_first wrong box"


def earlier_gold_box_last_differs(text, gold):
    text = answer_text(str(text or ""))
    selected_start = text.rfind("\\boxed")
    if selected_start < 0:
        selected_start = text.rfind("\\fbox")
    if selected_start < 0:
        return False
    earlier_boxes = [box for box in _box_values(text) if box["start"] < selected_start]
    earlier_has_gold = any(has_literal(box["value"], gold) for box in earlier_boxes)
    predicted = last_boxed(text)
    return earlier_has_gold and (predicted is None or normalise_value(predicted) != normalise_value(gold))


def value_class(value):
    normalized = normalise_value(value)
    if INT_RE.fullmatch(normalized):
        digits = len(normalized.lstrip("+-"))
        return f"integer:{digits}"
    if FRAC_RE.fullmatch(normalized):
        return "short_fraction"
    return "other"


def donor_pool(target, candidates):
    target_gold = normalise_value(target["gold"])
    target_class = value_class(target["gold"])
    pool = []
    for candidate in candidates:
        if str(candidate["problem_id"]) == str(target["problem_id"]):
            continue
        candidate_gold = normalise_value(candidate["gold"])
        if not candidate_gold or candidate_gold == target_gold:
            continue
        if value_class(candidate["gold"]) != target_class:
            continue
        if has_literal(target["problem"], candidate["gold"]):
            continue
        pool.append({"problem_id": str(candidate["problem_id"]), "gold": str(candidate["gold"])})
    return sorted(pool, key=lambda row: row["problem_id"])


def draw_donors(targets, candidates, seed=DONOR_SEED):
    rng = random.Random(int(seed))
    draws = {}
    for target in sorted(targets, key=lambda row: str(row["problem_id"])):
        pool = donor_pool(target, candidates)
        picked = rng.sample(pool, min(DONORS_PER_TARGET, len(pool))) if pool else []
        draws[str(target["problem_id"])] = {"pool_n": len(pool), "draw_n": len(picked),
                                           "short_pool": len(pool) < DONORS_PER_TARGET,
                                           "draws": picked}
    return draws


def first_middle_last_windows(text, value):
    hits = literal_occurrences(text, value)
    selected = selected_hit_positions(hits)
    windows = []
    for hit in selected:
        start = max(0, hit["start"] - 400)
        end = min(len(text), hit["end"] + 400)
        windows.append({"start": hit["start"], "end": hit["end"], "hit_rank": hit["hit_rank"],
                        "context_start": start, "context_end": end,
                        "window": text[start:end]})
    return windows


def load_tokenizer(path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("reading capped requires the `tokenizers` package for the pinned tokenizer.json") from exc
    return Tokenizer.from_file(str(path))


def _count_tokens(tokenizer, text):
    encoded = tokenizer.encode(str(text))
    ids = getattr(encoded, "ids", encoded)
    return len(ids)


def _read_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_dataset_rows(path):
    path = Path(path)
    if path.suffix.casefold() in {".json", ".jsonl"}:
        if path.suffix.casefold() == ".jsonl":
            return read_jsonl(path)
        value = read_json(path)
        return value if isinstance(value, list) else value.get("data", [])
    if path.suffix.casefold() == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("reading the pinned MATH-500 parquet requires pyarrow") from exc
        return pq.read_table(path).to_pylist()
    raise ValueError(f"unsupported dataset file {path}")


def _verify_manifest(inputs_dir):
    out_dir = Path(inputs_dir).parent
    manifest_path = out_dir / "inputs-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing input provenance manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    files = manifest.get("files", {})
    merged, per_problem, samples, dataset, tokenizer = _data_paths(inputs_dir)
    required = list(merged.values()) + list(per_problem["unedited"].values()) + \
        list(samples["unedited"].values()) + list(per_problem["p50c"].values()) + \
        list(samples["p50c"].values()) + [dataset, tokenizer]
    for required_path in required:
        relative = required_path.relative_to(out_dir).as_posix()
        record = files.get(relative)
        if not isinstance(record, dict):
            raise ValueError(f"required input is missing from manifest: {relative}")
        if not required_path.is_file():
            raise FileNotFoundError(f"required manifest input missing: {required_path}")
        actual = hashlib.sha256(required_path.read_bytes()).hexdigest()
        if actual != record.get("sha256"):
            raise ValueError(f"input hash mismatch for {required_path}: {actual} != {record.get('sha256')}")
    for relative, record in files.items():
        path = out_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest input missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record.get("sha256"):
            raise ValueError(f"input hash mismatch for {path}: {actual} != {record.get('sha256')}")
    return manifest


def _data_paths(inputs_dir):
    inputs = Path(inputs_dir)
    merged = {arm: inputs / "eval" / arm / "merged.json" for arm in ARMS}
    per_problem = {}
    samples = {}
    for arm in ARMS:
        per_problem[arm], samples[arm] = {}, {}
        for shard in range(10):
            base = inputs / "eval" / arm / "parts" / f"math500_s{shard}of10"
            per_problem[arm][shard] = base / "math500_per_problem.json"
            samples[arm][shard] = base / "math500_first_samples.jsonl.gz"
    dataset_candidates = [p for p in inputs.glob("math500-test.*") if p.is_file()]
    if len(dataset_candidates) != 1:
        raise FileNotFoundError(f"expected one math500-test data file under {inputs}, found {dataset_candidates}")
    tokenizer = inputs / "tokenizer.json"
    return merged, per_problem, samples, dataset_candidates[0], tokenizer


def _load_arm(arm, per_problem_paths, sample_paths):
    by_problem, by_sample = {}, {}
    first_ids = []
    for shard in range(10):
        rows = read_json(per_problem_paths[arm][shard])
        if not isinstance(rows, list):
            raise ValueError(f"{per_problem_paths[arm][shard]} is not a list")
        for row in rows:
            problem_id = str(row["id"])
            if problem_id in by_problem:
                raise ValueError(f"duplicate per-problem id {problem_id} in {arm}")
            correct = row.get("correct")
            if not isinstance(correct, list) or len(correct) != 4 or any(int(v) not in (0, 1) for v in correct):
                raise ValueError(f"{arm}:{problem_id} must have four binary correct entries")
            by_problem[problem_id] = {**row, "id": problem_id, "correct": [int(v) for v in correct]}
        sample_rows = _read_gzip_jsonl(sample_paths[arm][shard])
        if len(sample_rows) != 20:
            raise ValueError(f"{sample_paths[arm][shard]} has {len(sample_rows)} rows, expected 20")
        expected_first_ids = [str(row["id"]) for row in rows[:20]]
        actual_first_ids = [str(row["id"]) for row in sample_rows]
        if actual_first_ids != expected_first_ids:
            raise ValueError(f"{arm} sample-0 shard {shard} is not the first 20 per-problem rows")
        for sample in sample_rows:
            problem_id = str(sample["id"])
            if problem_id in by_sample:
                raise ValueError(f"duplicate stored sample id {problem_id} in {arm}")
            by_sample[problem_id] = sample
        first_ids.extend(str(row["id"]) for row in sample_rows)
    if len(by_problem) != 500:
        raise ValueError(f"{arm} has {len(by_problem)} per-problem rows, expected 500")
    if len(by_sample) != 200:
        raise ValueError(f"{arm} has {len(by_sample)} sample-0 rows, expected 200")
    return by_problem, by_sample, first_ids


def _read_first_sample_text(sample):
    for key in ("text", "response", "completion"):
        if isinstance(sample.get(key), str):
            return sample[key]
    raise ValueError(f"stored sample {sample.get('id')} has no text/response/completion string")


def _gate(observed, expected, tolerance, label, numerator, denominator):
    difference = abs(observed - expected) if observed is not None and expected is not None else None
    passed = difference is not None and difference <= tolerance
    return {"passed": bool(passed), "numerator": int(numerator), "denominator": int(denominator),
            "observed": observed, "expected": expected, "absolute_difference": difference,
            "tolerance": tolerance, "comparison": label}


def _gate_share(numerator, denominator, expected, tolerance, label):
    observed = numerator / denominator if denominator else None
    return _gate(observed, expected, tolerance, label, numerator, denominator)


def _trace_record(problem_id, arm, sample, per_problem, problem, gold, tokenizer):
    text = _read_first_sample_text(sample)
    correct0 = int(per_problem["correct"][0])
    pred_first = per_problem.get("pred_first")
    if pred_first == "":
        pred_first = None
    token_count = _count_tokens(tokenizer, text)
    capped = token_count >= TOKEN_CAP
    think_closed = "</think>" in text
    gold_hit_spans = literal_occurrences(text, gold)
    in_prompt = has_literal(problem, gold)
    onset = loop_onset(text)
    onset50 = loop_onset(text, ONSET_CHAR_CAP)
    signal = loop_signals(text)
    cohort_c = bool(arm == "unedited" and capped and not think_closed)
    looped = bool(cohort_c and (onset is not None or signal["verbatim_loop_end"] or signal["repetitive_end"]))
    box_cell_value = box_cell(correct0, pred_first) if gold_hit_spans else None
    return {"problem_id": str(problem_id), "arm": arm, "text_chars": len(text), "token_count": token_count,
            "capped": capped, "think_closed": think_closed, "cohort_C": cohort_c,
            "looped": looped, "merely_long": bool(cohort_c and not looped),
            "gold": str(gold), "gold_in_prompt": bool(in_prompt), "target_T": False,
            "correct0": correct0, "pred_first": pred_first,
            "pred_first_nonnull": pred_first is not None, "literal_hit": bool(gold_hit_spans),
            "literal_hit_count": len(gold_hit_spans), "hit_spans": gold_hit_spans,
            "first_hit_char": gold_hit_spans[0]["start"] if gold_hit_spans else None,
            "loop_onset": onset, "loop_onset_fraction": onset["char_index"] / len(text) if onset and text else None,
            "loop_onset_50k": onset50,
            "end_loop": bool(signal["verbatim_loop_end"] or signal["repetitive_end"]),
            "end_loop_signals": signal, "box_cell": box_cell_value,
            "earlier_gold_box_last_differs": bool(gold_hit_spans and earlier_gold_box_last_differs(text, gold))}


def compute(inputs_dir=DEFAULT_INPUTS, out_dir=DEFAULT_OUT, tokenizer=None, verify_hashes=True):
    inputs_dir, out_dir = Path(inputs_dir), Path(out_dir)
    if verify_hashes:
        manifest = _verify_manifest(inputs_dir)
    else:
        manifest = {}
    merged_paths, per_paths, sample_paths, dataset_path, tokenizer_path = _data_paths(inputs_dir)
    tokenizer = tokenizer or load_tokenizer(tokenizer_path)
    dataset_rows = load_dataset_rows(dataset_path)
    dataset = {}
    for row in dataset_rows:
        problem_id = str(row.get("unique_id", row.get("id", "")))
        if not problem_id:
            raise ValueError("MATH-500 dataset row has no unique_id/id")
        if problem_id in dataset:
            raise ValueError(f"duplicate MATH-500 dataset id {problem_id}")
        if "problem" not in row or "answer" not in row:
            raise ValueError(f"MATH-500 row {problem_id} lacks problem or answer")
        dataset[problem_id] = {"problem": str(row["problem"]), "gold": str(row["answer"])}
    if len(dataset) != 500:
        raise ValueError(f"MATH-500 test input has {len(dataset)} rows, expected 500")

    arm_data, merged, first_ids = {}, {}, {}
    for arm in ARMS:
        arm_data[arm] = _load_arm(arm, per_paths, sample_paths)
        merged[arm] = read_json(merged_paths[arm])
        first_ids[arm] = arm_data[arm][2]
    if first_ids["unedited"] != first_ids["p50c"]:
        raise ValueError("the stored first-200 problem IDs differ between unedited and p50c")
    if len(first_ids["unedited"]) != 200 or len(set(first_ids["unedited"])) != 200:
        raise ValueError("the concatenated first 20 rows from each shard are not 200 distinct IDs")
    all_ids = set(dataset)
    for arm in ARMS:
        pmap, smap, _ = arm_data[arm]
        if set(pmap) != all_ids:
            raise ValueError(f"{arm} per-problem IDs do not match the 500-row dataset")
        if set(smap) != set(first_ids[arm]):
            raise ValueError(f"{arm} first-sample input IDs differ from sample-0 rows")

    traces = []
    trace_by_key = {}
    for arm in ARMS:
        pmap, smap, ids = arm_data[arm]
        for problem_id in ids:
            row = dataset[problem_id]
            trace = _trace_record(problem_id, arm, smap[problem_id], pmap[problem_id],
                                  row["problem"], row["gold"], tokenizer)
            trace["target_T"] = bool(trace["cohort_C"] and not trace["gold_in_prompt"])
            trace_by_key[(problem_id, arm)] = trace
            traces.append(trace)

    gates_123 = {}
    for arm in ARMS:
        pmap, smap, ids = arm_data[arm]
        total_correct = sum(sum(row["correct"]) for row in pmap.values())
        total_entries = sum(len(row["correct"]) for row in pmap.values())
        recomputed_acc = total_correct / total_entries if total_entries else None
        expected_acc = merged[arm].get("math500", {}).get("acc")
        gates_123[arm] = {"gate1": _gate(recomputed_acc, expected_acc, 0.001,
                                          "mean of four correct entries over 500 problems",
                                          total_correct, total_entries)}
        capped_n = sum(trace_by_key[(problem_id, arm)]["capped"] for problem_id in ids)
        expected_cut = merged[arm].get("math500", {}).get("cut_share")
        gates_123[arm]["gate2"] = _gate_share(capped_n, len(ids), expected_cut, 0.08,
                                               "capped share of the first 200 sample-0 texts")
    cap_think_table = {}
    for arm in ARMS:
        rows = [trace_by_key[(problem_id, arm)] for problem_id in first_ids[arm]]
        cap_think_table[arm] = {f"capped={capped},think_closed={think_closed}": {
                                    "numerator": sum(row["capped"] == capped and row["think_closed"] == think_closed
                                                      for row in rows),
                                    "denominator": len(rows)}
                                for capped in (False, True) for think_closed in (False, True)}

    edited_correct = [trace_by_key[(problem_id, "p50c")] for problem_id in first_ids["p50c"]
                      if trace_by_key[(problem_id, "p50c")]["correct0"] == 1]
    edited_literal = sum(row["literal_hit"] for row in edited_correct)
    edited_rate = edited_literal / len(edited_correct) if edited_correct else None
    gates_123["p50c"]["gate3"] = {"passed": bool(edited_correct and edited_rate >= 0.90),
                                   "numerator": edited_literal, "denominator": len(edited_correct),
                                   "observed": edited_rate, "threshold": 0.90, "relation": "at least",
                                   "comparison": "p50c correct[0] rows with a literal gold hit"}

    cohort = [row for row in traces if row["cohort_C"]]
    target_ids = {row["problem_id"] for row in cohort if row["target_T"]}
    data_gates_passed = all(gates_123[arm][gate]["passed"]
                            for arm in ARMS for gate in ("gate1", "gate2")) and \
        gates_123["p50c"]["gate3"]["passed"]
    for row in traces:
        row["target_T"] = row["problem_id"] in target_ids and row["arm"] == "unedited"

    # The registered donor draws are for T. A separate seed extends the same eligibility rule to C\T
    # only so Weaver's C-level literal-hit control can be reported without changing T's draws.
    candidate_records = [{"problem_id": problem_id, "problem": dataset[problem_id]["problem"],
                          "gold": dataset[problem_id]["gold"]} for problem_id in first_ids["unedited"]]
    cohort_records = [{"problem_id": row["problem_id"], "problem": dataset[row["problem_id"]]["problem"],
                       "gold": row["gold"]} for row in cohort]
    target_records = [row for row in cohort_records if row["problem_id"] in target_ids]
    target_draws = draw_donors(target_records, candidate_records, seed=DONOR_SEED)
    supplemental_records = [row for row in cohort_records if row["problem_id"] not in target_ids]
    supplemental_draws = draw_donors(supplemental_records, candidate_records, seed=SUPPLEMENTAL_DONOR_SEED)
    donor_draws = target_draws | supplemental_draws

    windows, key_rows = [], []
    key_by_window = {}
    for trace in (cohort if data_gates_passed else []):
        problem_id = trace["problem_id"]
        problem = dataset[problem_id]["problem"]
        draw_info = donor_draws.get(problem_id, {"pool_n": 0, "draw_n": 0, "short_pool": True, "draws": []})
        trace["donor_pool_n"] = draw_info["pool_n"]
        trace["donor_draw_n"] = draw_info["draw_n"]
        trace["donor_short_pool"] = draw_info["short_pool"]
        trace["donor_draws"] = draw_info["draws"]
        text = _read_first_sample_text(arm_data["unedited"][1][problem_id])
        conditions = [] if trace["gold_in_prompt"] else [("real", trace["gold"], None)]
        if draw_info["draws"]:
            conditions.append(("shuffled", draw_info["draws"][0]["gold"], draw_info["draws"][0]["problem_id"]))
        for condition, value, donor_id in conditions:
            all_hit_starts = [span["start"] for span in literal_occurrences(text, value)]
            for hit in first_middle_last_windows(text, value):
                material = json.dumps([problem_id, condition, value, hit["start"], hit["hit_rank"]],
                                      ensure_ascii=True, separators=(",", ":"))
                window_id = "w:" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
                window = {"id": window_id, "problem": problem, "window": hit["window"], "value": str(value)}
                key = {"id": window_id, "problem_id": problem_id, "arm": "unedited",
                       "condition": condition, "value": str(value), "target_T": problem_id in target_ids,
                       "donor_problem_id": donor_id, "hit_start": hit["start"], "hit_end": hit["end"],
                       "all_hit_starts": all_hit_starts,
                       "hit_rank": hit["hit_rank"], "hit_count": len(literal_occurrences(text, value)),
                       "context_start": hit["context_start"], "context_end": hit["context_end"]}
                windows.append(window)
                key_rows.append(key)
                key_by_window[window_id] = key
    rng = random.Random(WINDOW_ORDER_SEED)
    rng.shuffle(windows)
    key_rows.sort(key=lambda row: row["id"])

    c_record_map = {row["problem_id"]: row for row in cohort}
    box_diagnostics = {}
    for arm in ARMS:
        hit_rows = [row for row in traces if row["arm"] == arm and row["literal_hit"]]
        denominator = len(hit_rows)
        box_counts = {cell: sum(row["box_cell"] == cell for row in hit_rows)
                      for cell in ("correct[0]", "pred_first wrong box", "pred_first null")}
        box_diagnostics[arm] = {"denominator_literal_hit_traces": denominator,
                                "cell_counts": box_counts,
                                "pred_first_null": {"numerator": sum(row["pred_first"] is None for row in hit_rows),
                                                     "denominator": denominator,
                                                     "rate": (sum(row["pred_first"] is None for row in hit_rows) /
                                                              denominator) if denominator else None},
                                "pred_first_nonnull": {"numerator": sum(row["pred_first"] is not None for row in hit_rows),
                                                       "denominator": denominator,
                                                       "rate": (sum(row["pred_first"] is not None for row in hit_rows) /
                                                                denominator) if denominator else None},
                                "earlier_gold_box_last_differs": {
                                    "numerator": sum(row["earlier_gold_box_last_differs"] for row in hit_rows),
                                    "denominator": denominator}}
    short_pools = [{"problem_id": row["problem_id"], "pool_n": row.get("donor_pool_n", 0),
                    "draw_n": row.get("donor_draw_n", 0), "short_pool": row.get("donor_short_pool", True)}
                   for row in cohort]
    gate1_2 = all(gates_123[arm]["gate1"]["passed"] and gates_123[arm]["gate2"]["passed"] for arm in ARMS)
    gates = {"gate1": {"passed": all(gates_123[arm]["gate1"]["passed"] for arm in ARMS), "by_arm": {
                    arm: gates_123[arm]["gate1"] for arm in ARMS}},
             "gate2": {"passed": all(gates_123[arm]["gate2"]["passed"] for arm in ARMS), "by_arm": {
                    arm: gates_123[arm]["gate2"] for arm in ARMS}},
             "gate3": gates_123["p50c"]["gate3"]}
    correct_summaries = {arm: [{"problem_id": problem_id,
                                "correct_n": sum(row["correct"]),
                                "sample_n": len(row["correct"]),
                                "correct": list(row["correct"]),
                                "any_correct": int(any(row["correct"]))}
                               for problem_id, row in sorted(arm_data[arm][0].items())]
                        for arm in ARMS}
    correctness_counts = {arm: {"per_sample_correct_n": [
                                    sum(row["correct"][sample_index] for row in arm_data[arm][0].values())
                                    for sample_index in range(4)],
                              "any_of_four_correct_n": sum(any(row["correct"])
                                                            for row in arm_data[arm][0].values()),
                              "denominator_problems": len(arm_data[arm][0])}
                          for arm in ARMS}
    c_loop_groups = {"looped": sorted(row["problem_id"] for row in cohort if row["looped"]),
                     "merely_long": sorted(row["problem_id"] for row in cohort if row["merely_long"])}
    p50c_closed_t = [trace_by_key[(problem_id, "p50c")] for problem_id in sorted(target_ids)
                     if trace_by_key[(problem_id, "p50c")]["think_closed"]]
    real_t_keys = {}
    for row in key_rows:
        if row["target_T"] and row["condition"] == "real":
            real_t_keys.setdefault(row["problem_id"], []).append(row)
    reader_hit_selection = {problem_id: {
                                "first_window_id": min(rows, key=lambda row: row["hit_start"])["id"],
                                "last_window_id": max(rows, key=lambda row: row["hit_start"])["id"],
                                "onset_char": trace_by_key[(problem_id, "unedited")]["loop_onset"]["char_index"]
                                if trace_by_key[(problem_id, "unedited")]["loop_onset"] else None}
                            for problem_id, rows in sorted(real_t_keys.items())}
    windows_status = "ready" if data_gates_passed else "blocked: Gate 1, 2, or 3 failed"
    value = {"part": 177, "input_manifest": "inputs-manifest.json" if manifest else None,
             "reader_windows_status": windows_status,
             "cohort_C_ids": sorted(c_record_map), "target_T_ids": sorted(target_ids),
             "counts": {"problems_per_arm": 500, "sample0_per_arm": 200, "cohort_C": len(cohort),
                        "target_T": len(target_ids), "reader_windows": len(windows),
                        "short_texts_under_2000_chars": {arm: sum(
                            trace_by_key[(problem_id, arm)]["text_chars"] < 2000 for problem_id in first_ids[arm])
                            for arm in ARMS},
                        "short_text_ids_by_arm": {arm: [problem_id for problem_id in first_ids[arm]
                            if trace_by_key[(problem_id, arm)]["text_chars"] < 2000] for arm in ARMS}},
             "gates_1_to_3": gates, "trace_results": traces, "box_diagnostics": box_diagnostics,
             "capped_by_think_closed": cap_think_table,
             "gate1_problem_summaries": correct_summaries,
             "reported_populations": {"looped_C_ids": c_loop_groups["looped"],
                                       "merely_long_C_ids": c_loop_groups["merely_long"],
                                       "correctness_counts_same_500": correctness_counts,
                                       "p50c_closed_thinking_on_T": {
                                           "numerator_literal_hit": sum(row["literal_hit"] for row in p50c_closed_t),
                                           "denominator_closed_traces": len(p50c_closed_t),
                                           "problem_ids": [row["problem_id"] for row in p50c_closed_t]},
                                       "reader_real_hit_selection_on_T": reader_hit_selection},
             "donors": {"seed": DONOR_SEED, "per_target": DONORS_PER_TARGET,
                        "short_pool_targets": sum(row["short_pool"] for row in short_pools),
                        "targets": target_draws, "supplemental_C_not_T_seed": SUPPLEMENTAL_DONOR_SEED,
                        "supplemental_C_not_T": supplemental_draws,
                        "eligibility": "different normalized value, same class, not in target prompt, other problem IDs",
                        "pool_rule": "sample without replacement; if fewer than 20, use the whole pool and report its size"},
             "assumptions": {"spaces": "all whitespace is removed for literal matching after the specified command and dollar normalization",
                             "short_fraction": "simple numeric \u005c\u005cfrac{a}{b} with 1-3 characters per component",
                             "onset_line_normalization": "collapse whitespace and strip each line",
                             "supplemental_donors": "same rule as T, separately seeded for C minus T diagnostics only"}}
    out_dir.mkdir(parents=True, exist_ok=True)
    if data_gates_passed:
        write_jsonl(out_dir / "windows.jsonl", windows)
        write_jsonl(out_dir / "window-key.jsonl", key_rows)
    else:
        for path in (out_dir / "windows.jsonl", out_dir / "window-key.jsonl"):
            path.unlink(missing_ok=True)
    write_json(out_dir / "read.json", value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    result = compute(args.inputs_dir, args.out_dir)
    print(json.dumps({"cohort_C": result["counts"]["cohort_C"], "target_T": result["counts"]["target_T"],
                      "gates": result["gates_1_to_3"]}, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
