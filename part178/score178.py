#!/usr/bin/env python3
"""Compute Part 178 tables, gates and registered verdicts from files. (Design: Reading rules)"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from part177 import read177  # noqa: E402
from part178 import run178, words178  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "part178"
DEFAULT_INPUTS = ROOT / "results" / "part177" / "inputs"
DEFAULT_PART177 = ROOT / "results" / "part177"
BOOTSTRAP_REPS = 2000
SEED = 178
SET_ORDER = ("U", "E", "G", "R8", "N4", "N8")
GENERATED_SETS = frozenset(("G", "R8", "N4", "N8"))
THINKING_SETS = frozenset(("U", "E", "G", "R8"))
MODEL_FOR_SET = {"U": "0.8B", "E": "0.8B", "G": "4B", "R8": "0.8B", "N4": "4B", "N8": "0.8B"}
# The mode-ratio floor is word occurrences as control openers, not total openers.
MIN_CONTROL_WORD_OCCURRENCES = 5
MIN_MODE_CONTROL_OPENERS = 500
MIN_FOLLOWON_TRACES = 40
MIN_SET_OPENERS = 2000
MIN_LINE_TOKEN_COUNT = 32500


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


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


def _percentile(values, q):
    values = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    return values[round(float(q) * (len(values) - 1))] if values else None


def _rankdata(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def spearman(left, right):
    """Spearman correlation with average ranks for ties; null for a flat/empty vector. (Gate 5)"""
    left, right = list(map(float, left)), list(map(float, right))
    if len(left) != len(right) or len(left) < 2:
        return None
    a, b = _rankdata(left), _rankdata(right)
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((value - ma) ** 2 for value in a)
    vb = sum((value - mb) ** 2 for value in b)
    if not va or not vb:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def _rates_for_set(rows, count_field="opener_counts", denominator="opener_exposure", reps=BOOTSTRAP_REPS,
                   seed=SEED, keep_samples=()):
    """Pooled rates and trace bootstrap intervals using multinomial cluster resampling."""
    rows = list(rows)
    vocabulary = sorted({word for row in rows for word in row.get(count_field, {})})
    if not vocabulary:
        return {}, {}
    try:
        import numpy as np
        matrix = np.array([[int(row.get(count_field, {}).get(word, 0)) for row in rows]
                           for word in vocabulary], dtype=np.int64)
        denominators = np.array([int(row.get(denominator, 0)) for row in rows], dtype=np.int64)
        total_d = int(denominators.sum())
        total_n = matrix.sum(axis=1)
        point = 1000.0 * total_n / total_d if total_d else np.full(len(vocabulary), np.nan)
        rng = np.random.default_rng(int(seed))
        boot = np.full((max(0, int(reps)), len(vocabulary)), np.nan, dtype=np.float32)
        if total_d and reps:
            # Matrix multiplication keeps all windows for a sampled trace attached.
            weights = rng.multinomial(len(rows), [1.0 / len(rows)] * len(rows), size=int(reps))
            for low in range(0, int(reps), 128):
                high = min(int(reps), low + 128)
                den = weights[low:high] @ denominators
                nums = weights[low:high] @ matrix.T
                valid = den > 0
                boot[low:high][valid] = 1000.0 * nums[valid] / den[valid, None]
        rates, samples = {}, {}
        requested = set(vocabulary) if keep_samples is True else set(keep_samples or ())
        for index, word in enumerate(vocabulary):
            values = boot[:, index]
            finite_values = values[np.isfinite(values)]
            quantiles = np.quantile(finite_values, [.025, .975]) if finite_values.size else (None, None)
            rates[word] = {"word": word, "numerator": int(total_n[index]), "denominator": total_d,
                           "rate_per_1000": float(point[index]) if total_d else None,
                           "ci95": {"lower": None if quantiles[0] is None else float(quantiles[0]),
                                    "upper": None if quantiles[1] is None else float(quantiles[1]),
                                    "replicates": int(finite_values.size), "seed": int(seed)}}
            if word in requested:
                samples[word] = values.astype(np.float32, copy=True)
        return rates, samples
    except ImportError:
        rates = {word: words178.rate(word, rows, field=count_field, denominator=denominator,
                                     reps=reps, seed=seed) for word in vocabulary}
        return rates, {}


def _top_words(rates, metric="rate_per_1000", limit=30):
    return [row["word"] for row in words178.ranking(rates, metric=metric, limit=limit)]


def _read_problem_payload(path):
    value = read_json(path)
    if not isinstance(value, dict) or value.get("part") != 178:
        raise ValueError(f"missing or invalid Part 178 problem file: {path}")
    return value


def _tokenizer_from_file(path):
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("the pinned tokenizer.json requires the `tokenizers` package") from exc
    return Tokenizer.from_file(str(path))


def _decode(tokenizer, ids):
    try:
        return tokenizer.decode(list(map(int, ids)), skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(map(int, ids)), skip_special_tokens=False)


def _text_from_sample(sample):
    return read177._read_first_sample_text(sample)


def _row_text(row, tokenizer):
    ids = [int(value) for value in row.get("generated_ids", [])]
    eot = row.get("end_token_id")
    if not ids:
        text = row.get("text", row.get("response", row.get("completion", row.get("generated_text", ""))))
        return str(text or ""), ids
    if eot is not None and int(eot) in ids:
        ids = ids[:ids.index(int(eot))]
    elif row.get("finish_reason") == "end-of-turn" and ids:
        # The runner writes the token id as a provenance field in later revisions; text is already eot-stripped.
        pass
    if row.get("generated_text") is not None:
        text = str(row["generated_text"])
    else:
        text = _decode(tokenizer, ids)
    return text, [int(value) for value in row.get("generated_ids", [])]


def _new_trace(row, tokenizer, set_name):
    raw, kept_ids = _row_text(row, tokenizer)
    eot_id = row.get("end_token_id")
    source_ids = list(kept_ids)
    if eot_id is not None and int(eot_id) in source_ids:
        source_ids = source_ids[:source_ids.index(int(eot_id))]
    close_id = row.get("think_close_token_id")
    thinking_ids = source_ids
    close_found = close_id is not None and int(close_id) in source_ids
    if close_found:
        thinking_ids = source_ids[:source_ids.index(int(close_id))]
    thinking = _decode(tokenizer, thinking_ids) if kept_ids else words178.thinking_text(raw)
    if not kept_ids:
        thinking = words178.thinking_text(raw)
    prefix_ids = thinking_ids[:words178.PREFIX_TOKENS] if kept_ids else None
    prefix_text = _decode(tokenizer, prefix_ids) if prefix_ids is not None else words178.prefix(thinking, tokenizer=tokenizer)
    onset = words178.onset4k(prefix_text, tokenizer=tokenizer, kept_ids=prefix_ids)
    finish_reason = row.get("finish_reason")
    if close_found:
        window_reason = "think-close"
    elif finish_reason in ("cap", "length"):
        window_reason = "cap"
    elif finish_reason in ("end-of-turn", "stop"):
        window_reason = "end-of-turn"
    else:
        window_reason = finish_reason or "end-of-turn"
    base_window = words178.window(prefix_text, tokenizer=tokenizer, token_ids=prefix_ids,
                                  onset=onset, end_reason=window_reason, limit=4000)
    base_uncut = words178.uncut(prefix_text, tokenizer=tokenizer, token_ids=prefix_ids,
                                end_reason=window_reason, limit=4000)
    mode = words178.mode_window(thinking if bool(row.get("thinking")) else raw,
                                tokenizer=tokenizer,
                                token_ids=(thinking_ids if bool(row.get("thinking")) else source_ids)
                                if kept_ids else None,
                                end_reason=window_reason)
    full_text = raw
    if not kept_ids:
        try:
            full_ids, full_offsets = words178.token_offsets(full_text, tokenizer)
            token_count = len(full_ids)
        except Exception:
            token_count = 0
    else:
        token_count = len(source_ids)
    if set_name in ("U", "E"):
        if "</think>" in full_text:
            full_reason = "think-close"
        elif token_count >= MIN_LINE_TOKEN_COUNT:
            full_reason = "cap"
        else:
            full_reason = "end-of-turn"
    else:
        full_reason = window_reason
    full_onset = read177.loop_onset(full_text)
    window_counts = Counter(words178.openers(base_window["text"]))
    uncut_counts = Counter(words178.openers(base_uncut["text"]))
    paragraph_counts = Counter(words178.paragraph_openers(base_window["text"]))
    mode_counts = Counter(words178.openers(mode["text"]))
    paragraph_exposure = sum(paragraph_counts.values())
    return {"problem_id": str(row.get("problem_id", row.get("id", ""))), "set": set_name,
            "text": full_text, "thinking_text": thinking, "thinking_ids": thinking_ids if kept_ids else None,
            "token_ids": source_ids if kept_ids else None, "window": base_window, "uncut": base_uncut,
            "mode_window": mode, "opener_counts": dict(window_counts), "opener_exposure": sum(window_counts.values()),
            "token_exposure": base_window["token_exposure"], "uncut_opener_counts": dict(uncut_counts),
            "uncut_opener_exposure": sum(uncut_counts.values()), "uncut_token_exposure": base_uncut["token_exposure"],
            "paragraph_counts": dict(paragraph_counts), "paragraph_exposure": paragraph_exposure,
            "mode_counts": dict(mode_counts), "mode_opener_exposure": sum(mode_counts.values()),
            "mode_token_exposure": mode["token_exposure"], "onset4k": onset,
            "full_text_onset": full_onset, "end_reason": base_window["end_reason"],
            "full_end_reason": full_reason, "generated": set_name in GENERATED_SETS,
            "reconstructed": set_name in ("U", "E")}


def _load_text_sets(inputs_dir, part177_dir, problems_payload, out_dir):
    merged, per_problem, sample_paths, _dataset, _tok = read177._data_paths(inputs_dir)
    data = {}
    for arm, name in (("unedited", "U"), ("p50c", "E")):
        _pmap, samples, order = read177._load_arm(arm, per_problem, sample_paths)
        data[name] = {problem_id: {"problem_id": problem_id, "text": _text_from_sample(samples[problem_id]),
                                   "arm": arm}
                      for problem_id in order}
    wanted_ids = [str(row["problem_id"]) for row in problems_payload["problems"]]
    generated = {}
    for set_name in sorted(GENERATED_SETS):
        rows = read_jsonl(Path(out_dir) / f"{set_name}.jsonl")
        by_id = {}
        for row in rows:
            if str(row.get("set")) != set_name:
                raise ValueError(f"{set_name}.jsonl contains a row for another set")
            by_id[str(row.get("problem_id"))] = row
        generated[set_name] = by_id
    return data, generated, wanted_ids


def _gate1(inputs_dir, part177_dir, problem_payload, raw_sets, wanted_ids):
    failures = []
    manifest = None
    try:
        manifest = read177._verify_manifest(inputs_dir)
        manifest_ok = True
    except Exception as exc:
        manifest_ok = False
        failures.append(f"input hash verification: {exc}")
    dataset_path = read177._data_paths(inputs_dir)[3]
    source_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path.is_file() else None
    source_ok = source_hash == problem_payload.get("source", {}).get("sha256")
    if not source_ok:
        failures.append("problems178.json source file hash differs from pinned dataset")
    numbers_path = Path(part177_dir) / "numbers177.json"
    if not numbers_path.is_file():
        numbers_path = Path(part177_dir) / "score.json"
    published_numbers = read_json(numbers_path, {}) or {}
    reader_name = published_numbers.get("sources", {}).get("reader", "read.json")
    reader_path = Path(part177_dir) / str(reader_name)
    prior = read_json(reader_path, {}) or {}
    published_rows = prior.get("trace_results", [])
    published_by_arm = {
        "U": {str(row.get("problem_id")): row for row in published_rows if row.get("arm") == "unedited"},
        "E": {str(row.get("problem_id")): row for row in published_rows if row.get("arm") == "p50c"},
    }
    onset_counts, onset_ok = {}, True
    for arm, name in (("unedited", "U"), ("p50c", "E")):
        target_name = "U" if arm == "unedited" else "E"
        source_rows = raw_sets.get(target_name, {})
        observed_ids = set(map(str, source_rows))
        expected_rows = published_by_arm[target_name]
        id_match = observed_ids == set(expected_rows)
        observed = sum(read177.loop_onset(row["text"]) is not None for row in source_rows.values())
        expected = sum(row.get("loop_onset") is not None for row in expected_rows.values()) if id_match else None
        passed = bool(id_match and expected == observed)
        onset_counts[target_name] = {"observed": observed, "expected": expected,
                                     "trace_count": len(source_rows), "published_trace_count": len(expected_rows),
                                     "ids_match": id_match, "passed": passed}
        onset_ok = onset_ok and passed
        if not passed:
            failures.append(f"{name} full-text onset count does not reproduce Part 177")
    sample_hashes = {}
    if manifest is not None:
        files = manifest.get("files", {})
        _merged, _per_problem, sample_paths, _dataset, _tokenizer = read177._data_paths(inputs_dir)
        for name, arm in (("U", "unedited"), ("E", "p50c")):
            checks = []
            for shard, path in sorted(sample_paths[arm].items()):
                relative = path.relative_to(Path(inputs_dir).parent).as_posix()
                record = files.get(relative, {})
                actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
                checks.append({"path": relative, "expected_sha256": record.get("sha256"),
                               "observed_sha256": actual, "passed": actual == record.get("sha256")})
            sample_hashes[name] = {"files": checks, "passed": bool(checks) and all(row["passed"] for row in checks)}
    hashes_ok = manifest_ok and all(row.get("passed") for row in sample_hashes.values())
    passed = bool(hashes_ok and source_ok and onset_ok)
    return {"passed": passed, "input_hashes": hashes_ok, "source_hash": source_hash,
            "published_numbers_file": numbers_path.name, "published_reader_file": str(reader_path.name),
            "text_source_hashes": sample_hashes, "full_text_onset_counts": onset_counts,
            "expected_count_ids": {"U": len(raw_sets.get("U", {})), "E": len(raw_sets.get("E", {}))},
            "failures": failures}


def _gate3(generated, trace_sets, wanted_ids):
    mode_rows = {}
    for set_name in sorted(GENERATED_SETS):
        spec = run178.SET_SPECS[set_name]
        rows = generated.get(set_name, {})
        ids_ok = set(rows) == set(wanted_ids)
        valid_prompts = all(bool(row.get("prompt_mode_valid")) and
                            bool(row.get("thinking")) == bool(spec["thinking"]) for row in rows.values())
        no_bad_tags = bool(spec["thinking"]) or all(not bool(row.get("generated_has_think_tag"))
                                                    for row in rows.values())
        mode_rows[set_name] = {"passed": bool(ids_ok and valid_prompts and no_bad_tags),
                               "rows": len(rows), "expected_rows": len(wanted_ids),
                               "all_prompt_ids_valid": valid_prompts, "nonthinking_has_no_think_tag": no_bad_tags,
                               "ids_match": ids_ok}
    adequacy = {set_name: {"openers": sum(row["opener_exposure"] for row in trace_sets.get(set_name, [])),
                           "threshold": MIN_SET_OPENERS,
                           "passed": sum(row["opener_exposure"] for row in trace_sets.get(set_name, [])) >= MIN_SET_OPENERS}
                for set_name in SET_ORDER}
    return {"passed": all(row["passed"] for row in mode_rows.values()), "by_set": mode_rows,
            "adequacy": adequacy, "adequacy_passed": all(row["passed"] for row in adequacy.values())}


def _metric_table(rows, field, denominator, reps, seed, sample_words=()):
    rates, samples = _rates_for_set(rows, count_field=field, denominator=denominator,
                                    reps=reps, seed=seed, keep_samples=sample_words)
    enriched = []
    for word, row in rates.items():
        incidence = {"word": word,
                     "traces_with_word": sum(int(item.get(field, {}).get(word, 0) > 0) for item in rows),
                     "trace_n": len(rows),
                     "count": sum(int(item.get(field, {}).get(word, 0)) for item in rows)}
        enriched.append({**row, **incidence})
    return {"rates": enriched, "by_word": rates, "samples": samples}


def _average_rank(values):
    return _rankdata(list(map(float, values)))


def _bootstrap_overlap(left_rows, right_rows, reps, seed):
    words = sorted({word for row in left_rows + right_rows for word in row["opener_counts"]})
    if not words:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    index = {word: i for i, word in enumerate(words)}
    try:
        import numpy as np
    except ImportError:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    rng = np.random.default_rng(int(seed))
    outputs = []
    if len(left_rows) != len(right_rows):
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    for _ in range(int(reps)):
        draw = rng.integers(0, len(left_rows), size=len(left_rows))
        left = Counter()
        right = Counter()
        left_den = right_den = 0
        for i in draw:
            left_row, right_row = left_rows[int(i)], right_rows[int(i)]
            left.update(left_row["opener_counts"])
            right.update(right_row["opener_counts"])
            left_den += int(left_row["opener_exposure"])
            right_den += int(right_row["opener_exposure"])
        lt = {row["word"] for row in words178.ranking(
            [{"word": w, "rate_per_1000": 1000.0 * left[w] / left_den if left_den else 0.0}
             for w in words], limit=20)}
        rt = {row["word"] for row in words178.ranking(
            [{"word": w, "rate_per_1000": 1000.0 * right[w] / right_den if right_den else 0.0}
             for w in words], limit=20)}
        outputs.append(len(lt & rt))
    return {"lower": _percentile(outputs, .025), "upper": _percentile(outputs, .975),
            "replicates": len(outputs), "seed": int(seed)}


def _bootstrap_rank_disagreement(rows, reps=BOOTSTRAP_REPS, seed=SEED):
    """Trace-bootstrap the difference between rate and trace-incidence top 20 in one set."""
    rows = list(rows)
    words = sorted({word for row in rows for word in row.get("opener_counts", {})})
    if not rows or not words:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    try:
        import numpy as np
    except ImportError:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    matrix = np.array([[int(row.get("opener_counts", {}).get(word, 0)) for word in words]
                       for row in rows], dtype=np.int64)
    denominators = np.array([int(row.get("opener_exposure", 0)) for row in rows], dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    weights = rng.multinomial(len(rows), [1.0 / len(rows)] * len(rows), size=int(reps))
    outputs = []
    for weight in weights:
        totals = weight @ matrix
        den = int(weight @ denominators)
        rates = 1000.0 * totals / den if den else np.zeros(len(words))
        incidences = weight @ (matrix > 0)
        rate_order = sorted(range(len(words)), key=lambda i: (-float(rates[i]), words[i]))[:20]
        trace_order = sorted(range(len(words)), key=lambda i: (-int(incidences[i]), words[i]))[:20]
        outputs.append(20 - len(set(rate_order) & set(trace_order)))
    return {"lower": _percentile(outputs, .025), "upper": _percentile(outputs, .975),
            "replicates": len(outputs), "seed": int(seed)}


def _rank_interval(table, word, limit=10):
    names = sorted(table.get("by_word", {}))
    rows = table.get("trace_rows", [])
    if word not in names or not rows:
        return {"rank_ci95": {"lower": None, "upper": None, "replicates": 0},
                "top_k_share_ci95": {"lower": None, "upper": None, "replicates": 0}}
    try:
        import numpy as np
    except ImportError:
        return {"rank_ci95": {"lower": None, "upper": None, "replicates": 0},
                "top_k_share_ci95": {"lower": None, "upper": None, "replicates": 0}}
    matrix = np.array([[int(row.get("opener_counts", {}).get(name, 0)) for name in names]
                       for row in rows], dtype=np.int32)
    denominators = np.array([int(row.get("opener_exposure", 0)) for row in rows], dtype=np.int32)
    target_index = names.index(word)
    lexical = np.array([name < word for name in names], dtype=np.bool_)
    rng = np.random.default_rng(int(table["by_word"][word]["ci95"].get("seed", SEED)))
    ranks, top_share = [], []
    replicates = int(table["by_word"][word]["ci95"].get("replicates", BOOTSTRAP_REPS))
    if denominators.sum() and replicates:
        weights = rng.multinomial(len(rows), [1.0 / len(rows)] * len(rows), size=replicates)
        for low in range(0, replicates, 64):
            high = min(replicates, low + 64)
            den = weights[low:high] @ denominators
            nums = weights[low:high] @ matrix
            valid = den > 0
            values = 1000.0 * nums[valid] / den[valid, None]
            if not len(values):
                continue
            target_values = values[:, target_index, None]
            rank_values = 1 + ((values > target_values) | ((values == target_values) & lexical)).sum(axis=1)
            ranks.extend(rank_values.tolist())
            top_share.extend((rank_values <= int(limit)).astype(np.int8).tolist())
    return {"rank_ci95": {"lower": _percentile(ranks, .025), "upper": _percentile(ranks, .975),
                           "replicates": len(ranks)},
            "top_k_share_ci95": {"lower": _percentile(top_share, .025), "upper": _percentile(top_share, .975),
                                 "replicates": len(top_share)}}


def _factor_interval(left_rows, right_rows, word, reps=BOOTSTRAP_REPS, seed=SEED):
    left = {row["problem_id"]: row for row in left_rows}
    right = {row["problem_id"]: row for row in right_rows}
    common = sorted(set(left) & set(right))
    if not common:
        return {"lower": None, "upper": None, "replicates": 0, "seed": int(seed)}
    rng = random.Random(int(seed))
    factors = []
    for _ in range(int(reps)):
        draw = [common[rng.randrange(len(common))] for _ in common]
        ln = sum(int(left[pid]["opener_counts"].get(word, 0)) for pid in draw)
        ld = sum(int(left[pid]["opener_exposure"]) for pid in draw)
        rn = sum(int(right[pid]["opener_counts"].get(word, 0)) for pid in draw)
        rd = sum(int(right[pid]["opener_exposure"]) for pid in draw)
        if ld and rd and ln and rn:
            ratio = (ln / ld) / (rn / rd)
            factors.append(max(ratio, 1.0 / ratio))
    return {"lower": _percentile(factors, .025), "upper": _percentile(factors, .975),
            "replicates": len(factors), "seed": int(seed)}


def _count_rate_interval(table, words, threshold=5):
    samples = table.get("samples", {})
    count = max((len(samples.get(word, ())) for word in words), default=0)
    values = [sum(float(samples[word][i]) >= threshold if word in samples else 0 for word in words)
              for i in range(count)]
    return {"lower": _percentile(values, .025), "upper": _percentile(values, .975),
            "replicates": len(values)}


def _under_ratio_interval(ratio_table, words, threshold=2):
    values = []
    for word in words:
        row = ratio_table.get("by_word", {}).get(word, {})
        if row.get("available"):
            values.append((word, row.get("samples", [])))
    if not values:
        return {"lower": None, "upper": None, "replicates": 0}
    n = min((len(samples) for _word, samples in values), default=0)
    samples = []
    for index in range(n):
        samples.append(sum(float(ratios[index]) < threshold for _word, ratios in values))
    return {"lower": _percentile(samples, .025), "upper": _percentile(samples, .975),
            "replicates": len(samples), "note": "fixed point-selected top-20 words; trace bootstrap"}


def _bootstrap_rate_traces(rows, word, reps=BOOTSTRAP_REPS, seed=SEED):
    point = words178.traces(word, rows)["traces_with_word"]
    import random
    rng = random.Random(int(seed))
    values = []
    for _ in range(int(reps)):
        sample = [rows[rng.randrange(len(rows))] for _ in rows] if rows else []
        if sample:
            values.append(words178.traces(word, sample)["traces_with_word"])
    return {"point": point, "ci95": {"lower": _percentile(values, .025), "upper": _percentile(values, .975),
                                      "replicates": len(values), "seed": int(seed)}}


def _prediction(prediction, estimate, verdict, rule, confidence, author, extra=None):
    return {"prediction": prediction, "estimate": estimate, "verdict": verdict, "rule": rule,
            "confidence": confidence, "author": author, **(extra or {})}


def _read_blocked(prediction_rows, gate1, gate2, gate3, gate4, gate5, adequacy):
    """Gates 1-2 block all; Gate 3 blocks its set; Gate 4 blocks generated sets; Gate 5 blocks 178.4."""
    global_blocks = []
    if not gate1.get("passed"):
        global_blocks.append("Gate 1 failed")
    if not gate2.get("passed"):
        global_blocks.append("Gate 2 failed")
    generated_bad = not gate4.get("passed")
    return {"global": global_blocks, "generated": generated_bad,
            "gate3": gate3, "adequacy": adequacy, "gate5": gate5}


def _verdict_blocked(blocked, sets=(), gate5_only=False):
    if blocked["global"]:
        return ", ".join(blocked["global"])
    if gate5_only and not blocked["gate5"].get("passed"):
        return "Gate 5 failed"
    for set_name in sets:
        if set_name in GENERATED_SETS and blocked["generated"]:
            return "Gate 4 failed"
        if set_name in blocked["gate3"].get("by_set", {}) and not blocked["gate3"]["by_set"][set_name]["passed"]:
            return f"Gate 3 mode validity failed for {set_name}"
        if set_name in blocked["adequacy"] and not blocked["adequacy"][set_name]["passed"]:
            return f"fewer than {MIN_SET_OPENERS} openers in {set_name}"
    return None


def _mode_ratios(mode_tables, reps, seed, keep_samples=None):
    comparisons = {"R8": "N8", "G": "N4"}
    keep_samples = {name: set(words) for name, words in (keep_samples or {}).items()}
    rows = {}
    for thinking, control in comparisons.items():
        left = mode_tables[thinking]["by_word"]
        right = mode_tables[control]["by_word"]
        vocabulary = sorted(set(left) | set(right))
        control_openers = sum(row["mode_opener_exposure"] for row in mode_tables[control]["trace_rows"])
        ratios = []
        left_samples = mode_tables[thinking]["samples"]
        right_samples = mode_tables[control]["samples"]
        for word in vocabulary:
            lrow, rrow = left.get(word), right.get(word)
            lrate = lrow["rate_per_1000"] if lrow else 0.0
            rrate = rrow["rate_per_1000"] if rrow else 0.0
            lnum = (lrow or {}).get("numerator", 0)
            rnum = (rrow or {}).get("numerator", 0)
            available = rnum >= MIN_CONTROL_WORD_OCCURRENCES
            ratio = lrate / rrate if available and rrate else None
            import numpy as np
            ls = np.asarray(left_samples.get(word, ()), dtype=np.float32)
            rs = np.asarray(right_samples.get(word, ()), dtype=np.float32)
            samples = np.full(min(len(ls), len(rs)), np.nan, dtype=np.float32)
            valid = np.isfinite(ls[:len(samples)]) & np.isfinite(rs[:len(samples)]) & (rs[:len(samples)] > 0)
            samples[valid] = ls[:len(samples)][valid] / rs[:len(samples)][valid]
            finite_samples = samples[np.isfinite(samples)]
            quantiles = np.quantile(finite_samples, [.025, .975]) if finite_samples.size else (None, None)
            ratio_row = {"word": word, "thinking_set": thinking, "control_set": control,
                         "thinking_rate": lrate, "control_rate": rrate, "control_openers": control_openers,
                         "control_word_n": int(rnum),
                         "ratio": ratio, "ci95": {"lower": None if quantiles[0] is None else float(quantiles[0]),
                                                  "upper": None if quantiles[1] is None else float(quantiles[1]),
                                                  "replicates": int(finite_samples.size), "seed": int(seed)},
                         "available": bool(available),
                         "reason": None if available else "fewer than 5 control occurrences"}
            if word in keep_samples.get(thinking, set()):
                ratio_row["samples"] = samples
            ratios.append(ratio_row)
        rows[thinking] = {"thinking_set": thinking, "control_set": control, "control_openers": control_openers,
                          "rows": ratios, "by_word": {row["word"]: row for row in ratios}}
    return rows


def _w2_positions(texts_by_id, keys, labels, target_ids, tokenizer):
    label_by_id = {str(row.get("id")): row for row in labels}
    chosen = defaultdict(lambda: defaultdict(list))
    target_set = set(map(str, target_ids))
    for key in keys:
        pid = str(key.get("problem_id"))
        if pid not in target_set or key.get("condition") != "real":
            continue
        label = label_by_id.get(str(key.get("id")), {}).get("label")
        if label in ("asserted", "intermediate"):
            chosen[pid][label].append(key)
    rows = []
    for pid, groups in chosen.items():
        if not groups.get("asserted") or not groups.get("intermediate"):
            continue
        text = texts_by_id.get(pid)
        if not text:
            continue
        ids, offsets = words178.token_offsets(text, tokenizer)
        values = {}
        for label in ("asserted", "intermediate"):
            key = min(groups[label], key=lambda row: (int(row.get("hit_start", 0)), str(row.get("id"))))
            values[label] = words178._token_for_char(offsets, int(key["hit_start"]))
        rows.append({"problem_id": pid, **values})
    if not rows:
        return {"n": 0, "median_asserted_token": None, "median_intermediate_token": None,
                "difference": None, "ci95": None, "rows": []}
    asserted = median(row["asserted"] for row in rows)
    intermediate = median(row["intermediate"] for row in rows)
    differences = [row["asserted"] - row["intermediate"] for row in rows]
    rng = random.Random(SEED + 2)
    samples = []
    for _ in range(BOOTSTRAP_REPS):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        samples.append(median(row["asserted"] for row in draw) - median(row["intermediate"] for row in draw))
    return {"n": len(rows), "median_asserted_token": asserted,
            "median_intermediate_token": intermediate, "difference": asserted - intermediate,
            "ci95": {"lower": _percentile(samples, .025), "upper": _percentile(samples, .975),
                     "replicates": len(samples), "seed": SEED + 2}, "rows": rows}


def _make_predictions(read_result, tables, ratios, followon, w2, gates, reps, seed):
    rates = {name: {row["word"]: row for row in tables["window"][name]["rates"]} for name in SET_ORDER}
    rankings = {name: tables["window"][name]["ranking"] for name in SET_ORDER}
    blocked = _read_blocked({}, gates["gate1"], gates["gate2"], gates["gate3"], gates["gate4"],
                            gates["gate5"], gates["adequacy"])
    predictions = {}

    def add(key, desc, estimate, verdict, rule, confidence, author, sets=(), gate5_only=False):
        reason = _verdict_blocked(blocked, sets=sets, gate5_only=gate5_only)
        if reason:
            verdict = "not read"
        predictions[key] = _prediction(desc, estimate, verdict, rule, confidence, author,
                                       {"read_block": reason} if reason else {})

    u_ranking = rankings["U"]
    g_ranking = rankings["G"]
    u_names = [row["word"] for row in u_ranking]
    g_names = [row["word"] for row in g_ranking]
    u_wait_rank = next((i + 1 for i, row in enumerate(u_ranking) if row["word"] == "wait"), None)
    g_wait_rank = next((i + 1 for i, row in enumerate(g_ranking) if row["word"] == "wait"), None)
    add("178.1", "In U, wait is among the top 10 openers by rate(w)",
        {"rank": u_wait_rank, "rate": rates["U"].get("wait", {}).get("rate_per_1000"),
         **_rank_interval(tables["window"]["U"], "wait")},
        "held" if u_wait_rank is not None and u_wait_rank <= 10 else "failed",
        "Point rank by window rate, descending rate then lexical word; at most 10.", "medium", "Claude", sets=("U",))
    add("178.2", "In G, wait is among the top 10 openers by rate(w)",
        {"rank": g_wait_rank, "rate": rates["G"].get("wait", {}).get("rate_per_1000"),
         **_rank_interval(tables["window"]["G"], "wait")},
        "held" if g_wait_rank is not None and g_wait_rank <= 10 else "failed",
        "Point rank by window rate, descending rate then lexical word; at most 10.", "medium", "Claude", sets=("G",))
    u_wait, e_wait = rates["U"].get("wait", {}).get("rate_per_1000"), rates["E"].get("wait", {}).get("rate_per_1000")
    u_e_ratio = max(u_wait / e_wait, e_wait / u_wait) if u_wait and e_wait else None
    factor_ci = _factor_interval(tables["window"]["U"].get("trace_rows", []),
                                 tables["window"]["E"].get("trace_rows", []), "wait", reps, seed + 3)
    add("178.3", "rate(wait) differs by less than a factor of 1.5 between U and E on window",
        {"U_rate": u_wait, "E_rate": e_wait, "max_factor": u_e_ratio,
         "max_factor_ci95": factor_ci,
         "U_ci95": rates["U"].get("wait", {}).get("ci95"), "E_ci95": rates["E"].get("wait", {}).get("ci95")},
        "not read" if u_e_ratio is None else "held" if u_e_ratio < 1.5 else "failed",
        "Both rates must be nonzero; the point maximum of U/E and E/U must be below 1.5.",
        "medium", "Claude", sets=("U", "E"))
    overlap = len(set(u_names[:20]) & set(g_names[:20]))
    add("178.4", "At least 14 of U's top 20 openers are in G's top 20",
        {"overlap": overlap, "U_top20": u_names[:20], "G_top20": g_names[:20],
         "ci95_overlap": tables["gate5"].get("overlap_ci95")},
        "held" if overlap >= 14 else "failed",
        "Point top-20 set overlap on window rates must be at least 14.", "medium", "Claude",
        sets=("U", "G"), gate5_only=True)
    imported_reach = [word for word in words178.IMPORTED_LIST
                      if tables["paragraph"]["U"]["by_word"].get(word, {}).get("rate_per_1000") is not None and
                      tables["paragraph"]["U"]["by_word"][word]["rate_per_1000"] >= 5]
    add("178.5", "At most 6 imported words reach 5 per 1,000 paragraph openers in U",
        {"count": len(imported_reach), "words": imported_reach,
         "paragraph_opener_denominator": tables["paragraph"]["U"]["denominator"],
         "count_ci95": _count_rate_interval(tables["paragraph"]["U"], words178.IMPORTED_LIST, 5)},
        "held" if len(imported_reach) <= 6 else "failed",
        "Point count of the fixed 17 imported words at or above 5 per 1,000 paragraph openers must be at most 6.",
        "medium", "Claude", sets=("U",))
    f_n = int(followon.get("n_eligible", 0))
    f_point = followon.get("point")
    add("178.6", "The follow-on statistic on eligible traces is at least 0.15; not read below 40 eligible traces",
        {"point": f_point, "ci95": followon.get("ci95"), "eligible_traces": f_n,
         "by_onset_side": followon.get("by_onset_side")},
        "not read" if f_n < MIN_FOLLOWON_TRACES or f_point is None else "held" if f_point >= .15 else "failed",
        "Equal-trace mean must be at least 0.15 with at least 40 eligible traces; report trace bootstrap interval.",
        "low", "Claude", sets=("U",))
    g_top = [row["word"] for row in g_ranking[:20]]
    g_ratios = ratios["G"]["by_word"]
    missing = sum(not g_ratios.get(word, {}).get("available", False) for word in g_top)
    under_two = [word for word in g_top if g_ratios.get(word, {}).get("available") and
                 g_ratios[word].get("ratio") < 2]
    add("178.7", "At least 10 of G's top 20 openers have a mode ratio under 2; unread if over 5 lack a ratio",
        {"count_under_2": len(under_two), "words_under_2": under_two, "top20": g_top,
         "without_ratio": missing, "under_2_ci95": _under_ratio_interval(ratios["G"], g_top, 2),
         "mode_ratio_set_pair": ["G", "N4"]},
        "not read" if missing > 5 else "held" if len(under_two) >= 10 else "failed",
        "At least 10 of G window top 20 have G/N4 mode ratio under 2; no more than 5 may lack a ratio.",
        "medium", "Claude", sets=("G", "N4"))
    u_by_traces = words178.ranking(tables["window"]["U"]["trace_rates"], metric="traces_with_word", limit=20)
    trace_names = [row["word"] for row in u_by_traces]
    different = 20 - len(set(u_names[:20]) & set(trace_names))
    w1_verdict = "held" if different >= 4 else "failed" if different <= 1 else "inconclusive"
    add("178.W1", "Ranking by rate(w) and by traces(w) disagree on at least 4 of the top 20 in U",
        {"different_top20_entries": different, "rate_top20": u_names[:20], "trace_top20": trace_names,
         "ci95": _bootstrap_rank_disagreement(tables["window"]["U"].get("trace_rows", []), reps, seed + 5)},
        w1_verdict, "Held at 4 or more differing entries; failed at 1 or fewer; otherwise inconclusive.",
        "medium", "Weaver", sets=("U",))
    w2point = w2.get("difference")
    w2_verdict = "not read" if w2point is None else "held" if w2point >= 2000 else \
        "failed" if w2point < 500 else "inconclusive"
    add("178.W2", "Median first asserted selected-hit position exceeds first intermediate position by at least 2000 tokens",
        {"difference_tokens": w2point, "median_asserted_token": w2.get("median_asserted_token"),
         "median_intermediate_token": w2.get("median_intermediate_token"), "n_both_labels": w2.get("n"),
         "ci95": w2.get("ci95")}, w2_verdict,
        "Held at a median-position difference of at least 2000; failed under 500; otherwise inconclusive.",
        "medium", "Weaver", sets=("U",))
    for suffix, thinking, control in (("a", "R8", "N8"), ("b", "G", "N4")):
        ratio_row = ratios[thinking]["by_word"].get("wait", {})
        control_openers = int(ratio_row.get("control_openers", 0))
        ratio_value = ratio_row.get("ratio")
        unavailable = control_openers < MIN_MODE_CONTROL_OPENERS or not ratio_row.get("available") or ratio_value is None
        verdict = "not read" if unavailable else "held" if ratio_value >= 3 else \
            "failed" if ratio_value < 1.5 else "inconclusive"
        add(f"178.W3{suffix}", f"On Qwen3.5-{MODEL_FOR_SET[thinking]}, wait has at least 3x the mode rate in {thinking} over {control}",
            {"thinking_set": thinking, "control_set": control, "wait_ratio": ratio_value,
             "control_openers": control_openers, "ci95": ratio_row.get("ci95")}, verdict,
            "Held at ratio at least 3; failed under 1.5; otherwise inconclusive; unread under 500 control openers or no ratio.",
            "medium", "Weaver", sets=(thinking, control))
    reflection_rates = {name: {word: rates[name].get(word, {}).get("rate_per_1000")
                               for word in words178.REFLECTION_FAMILY} for name in ("U", "G")}
    reflection_mode = {pair: {word: row["by_word"].get(word, {}).get("ratio")
                              for word in words178.REFLECTION_FAMILY}
                       for pair, row in ratios.items()}
    any_rate = any(value is not None and value >= 5 for set_values in reflection_rates.values()
                   for value in set_values.values())
    any_mode_ratio = any(value is not None and value >= 2 for pair_values in reflection_mode.values()
                         for value in pair_values.values())
    read_result["withdrawal"] = {"withdrawn": not any_rate or not any_mode_ratio,
                                 "reflection_rates_U_G": reflection_rates,
                                 "reflection_mode_ratios_G_N4_R8_N8": reflection_mode,
                                 "rule": "withdraw if no reflection word reaches 5/1000 in U or G, or none has ratio >=2 on G/N4 or R8/N8"}
    return predictions


def score(out_dir=DEFAULT_OUT, inputs_dir=DEFAULT_INPUTS, part177_dir=DEFAULT_PART177,
          problems_path=None, reps=BOOTSTRAP_REPS, seed=SEED):
    """Build read.json and all registered tables before writing score artifacts. (Design: Reading rules)"""
    out_dir, inputs_dir, part177_dir = Path(out_dir), Path(inputs_dir), Path(part177_dir)
    problems_path = Path(problems_path or HERE / "problems178.json")
    problem_payload = _read_problem_payload(problems_path)
    tokenizer_path = read177._data_paths(inputs_dir)[4]
    tokenizer = _tokenizer_from_file(tokenizer_path)
    raw_sets, generated, wanted_ids = _load_text_sets(inputs_dir, part177_dir, problem_payload, out_dir)
    trace_sets = {name: [] for name in SET_ORDER}
    raw_trace_texts = {"U": {}, "E": {}}
    token_rows = []
    for set_name in ("U", "E"):
        for problem_id in wanted_ids:
            source = raw_sets[set_name].get(problem_id)
            if source is None:
                continue
            trace = _new_trace(source, tokenizer, set_name)
            trace_sets[set_name].append(trace)
            raw_trace_texts[set_name][problem_id] = trace["text"]
    for set_name in sorted(GENERATED_SETS):
        for problem_id in wanted_ids:
            row = generated.get(set_name, {}).get(problem_id)
            if row is None or row.get("error"):
                continue
            trace = _new_trace(row, tokenizer, set_name)
            trace_sets[set_name].append(trace)
    metrics = {"window": {}, "uncut": {}, "paragraph": {}, "mode": {}}
    for index, set_name in enumerate(SET_ORDER):
        rows = trace_sets[set_name]
        window_table = _metric_table(rows, "opener_counts", "opener_exposure", reps, seed + index)
        trace_rates = [{"word": word, **words178.traces(word, rows)} for word in window_table["by_word"]]
        window_table["trace_rates"] = trace_rates
        window_table["ranking"] = words178.ranking(window_table["rates"], limit=max(30, len(window_table["rates"])))
        window_table["top30"] = window_table["ranking"][:30]
        window_table["token_exposure"] = sum(row["token_exposure"] for row in rows)
        window_table["opener_exposure"] = sum(row["opener_exposure"] for row in rows)
        window_table["trace_n"] = len(rows)
        window_table["trace_rows"] = rows
        metrics["window"][set_name] = window_table
        uncut_table = _metric_table(rows, "uncut_opener_counts", "uncut_opener_exposure", reps, seed + 20 + index)
        uncut_table["ranking"] = words178.ranking(uncut_table["rates"], limit=max(30, len(uncut_table["rates"])))
        uncut_table["top30"] = uncut_table["ranking"][:30]
        uncut_table["token_exposure"] = sum(row["uncut_token_exposure"] for row in rows)
        uncut_table["opener_exposure"] = sum(row["uncut_opener_exposure"] for row in rows)
        uncut_table["trace_n"] = len(rows)
        metrics["uncut"][set_name] = uncut_table
        para_samples = words178.IMPORTED_LIST if set_name == "U" else ()
        para_table = _metric_table(rows, "paragraph_counts", "paragraph_exposure", reps,
                                   seed + 40 + index, sample_words=para_samples)
        para_table["ranking"] = words178.ranking(para_table["rates"], limit=max(30, len(para_table["rates"])))
        para_table["top30"] = para_table["ranking"][:30]
        para_table["opener_exposure"] = sum(row["paragraph_exposure"] for row in rows)
        para_table["denominator"] = para_table["opener_exposure"]
        metrics["paragraph"][set_name] = para_table
    for index, set_name in enumerate(("G", "R8", "N4", "N8")):
        rows = trace_sets[set_name]
        sample_words = {word for row in rows for word in row.get("mode_counts", {})}
        table = _metric_table(rows, "mode_counts", "mode_opener_exposure", reps,
                              seed + 60 + index, sample_words=sample_words)
        table["trace_rows"] = rows
        table["ranking"] = words178.ranking(table["rates"], limit=max(30, len(table["rates"])))
        metrics["mode"][set_name] = table
    g_top_words = {row["word"] for row in metrics["window"]["G"]["ranking"][:20]}
    ratios = _mode_ratios(metrics["mode"], reps, seed + 100, keep_samples={"G": g_top_words})
    for table in metrics["mode"].values():
        table.pop("samples", None)
    mode_validity = _gate3(generated, trace_sets, wanted_ids)
    gate1 = _gate1(inputs_dir, part177_dir, problem_payload, raw_sets, wanted_ids)
    gate2 = words178.gate2_results()
    gate4 = run178.gate4_sampler_fixtures()
    u20 = metrics["window"]["U"]["ranking"][:20]
    r8rank = metrics["window"]["R8"]["ranking"]
    words_u20 = [row["word"] for row in u20]
    u_rates = [row["rate_per_1000"] or 0.0 for row in u20]
    r8_by_word = {row["word"]: row for row in metrics["window"]["R8"]["rates"]}
    r8_rates = [r8_by_word.get(word, {}).get("rate_per_1000") or 0.0 for word in words_u20]
    r8_top20 = {row["word"] for row in r8rank[:20]}
    overlap = len(set(words_u20) & r8_top20)
    gate5 = {"passed": bool(spearman(u_rates, r8_rates) is not None and spearman(u_rates, r8_rates) >= .7),
             "spearman": spearman(u_rates, r8_rates), "threshold": .7,
             "on_U_top20": [{"word": word, "U_rate": u_rates[i], "R8_rate": r8_rates[i]}
                            for i, word in enumerate(words_u20)],
             "U_top20_in_R8_top20": overlap, "U_top20": words_u20,
             "R8_top20": [row["word"] for row in r8rank[:20]],
             "overlap_ci95": _bootstrap_overlap(trace_sets["U"], trace_sets["R8"], reps, seed + 120)}
    gates = {"gate1": gate1, "gate2": gate2, "gate3": mode_validity,
             "adequacy": {"passed": mode_validity["adequacy_passed"], "by_set": mode_validity["adequacy"]},
             "gate4": gate4, "gate5": gate5}
    metrics["gate5"] = gate5
    keys = read_jsonl(part177_dir / "window-key.jsonl")
    labels = read_jsonl(part177_dir / "labels-ds.jsonl")
    prior_read = read_json(part177_dir / "read.json", {}) or {}
    targets = list(map(str, prior_read.get("target_T_ids", [])))
    full_onsets = {str(row["problem_id"]): row["full_text_onset"] for row in trace_sets["U"]}
    followon = words178.follow_on(raw_trace_texts["U"], keys, labels, targets, tokenizer,
                                  onset_by_problem=full_onsets, reps=reps, seed=seed + 140)
    w2 = _w2_positions(raw_trace_texts["U"], keys, labels, targets, tokenizer)
    # Token lists use only the point top 30 for each set's mode; generated rows retain emitted ids.
    token_aggregate = defaultdict(int)
    for set_name in SET_ORDER:
        top = [row["word"] for row in metrics["window"][set_name]["top30"]]
        for trace in trace_sets[set_name]:
            exposure = trace["token_exposure"]
            kept = trace.get("token_ids")[:exposure] if trace.get("token_ids") is not None else None
            for item in words178.token_list(trace["window"]["text"], tokenizer, top, kept_ids=kept,
                                            reconstructed=trace["reconstructed"]):
                key = (set_name, item["word"], tuple(item["token_ids"]), item["case"],
                       item["leading_space_or_newline"], item["reconstructed"])
                token_aggregate[key] += item["count"]
    token_list_rows = [{"set": key[0], "word": key[1], "token_ids": list(key[2]), "case": key[3],
                        "leading_space_or_newline": key[4], "reconstructed": key[5], "count": value}
                       for key, value in sorted(token_aggregate.items())]
    visible_metrics = {}
    for family in ("window", "uncut", "paragraph", "mode"):
        visible_metrics[family] = {}
        for set_name, table in metrics[family].items():
            visible_metrics[family][set_name] = {key: value for key, value in table.items()
                                                 if key not in ("by_word", "samples", "trace_rows")}
    visible_metrics["mode_ratios"] = {
        pair: {key: ([{field: value for field, value in row.items() if field != "samples"}
                     for row in item[key]] if key == "rows" else value)
               for key, value in item.items() if key not in ("by_word",)}
        for pair, item in ratios.items()}
    visible_metrics["token_list"] = token_list_rows
    visible_metrics["follow_on"] = followon
    visible_metrics["follow_on"]["eligible"] = followon.get("eligible", [])
    visible_metrics["follow_on"]["definition"] = followon.get("definition")
    visible_metrics["w2_selected_hit_positions"] = w2
    visible_metrics["gate5"] = gate5
    read_result = {"part": 178, "input_manifest": "results/part177/inputs-manifest.json",
                   "problem_source": problem_payload["source"], "sets": visible_metrics,
                   "gates": gates, "target_T_ids": targets,
                   "trace_results": [{key: value for key, value in row.items()
                                      if key not in ("text", "thinking_text", "thinking_ids", "token_ids", "window", "uncut", "mode_window")}
                                     for name in SET_ORDER for row in trace_sets[name]],
                   "counts": {name: {"traces": len(trace_sets[name]),
                                     "window_tokens": sum(row["token_exposure"] for row in trace_sets[name]),
                                     "window_openers": sum(row["opener_exposure"] for row in trace_sets[name])}
                              for name in SET_ORDER},
                   "assumptions": {"stored_text_ids": "reconstructed with Part 177's pinned tokenizer",
                                   "generated_text_ids": "kept generated ids",
                                   "mode_ratio_pairs": {"0.8B": "R8 over N8", "4B": "G over N4"},
                                   "mode_window": "first 2000 tokens, onset from first 2048, last 40 lines censored"}}
    predictions = _make_predictions(read_result, metrics, ratios, followon, w2, gates, reps, seed)
    for family in ("window", "uncut", "paragraph", "mode"):
        for table in metrics[family].values():
            table.pop("samples", None)
    for ratio_table in ratios.values():
        for row in ratio_table["rows"]:
            row.pop("samples", None)
    read_result["predictions"] = predictions
    score_result = {"part": 178, "read_gates": gates, "tables": visible_metrics,
                    "predictions": predictions, "withdrawal": read_result["withdrawal"],
                    "bootstrap": {"replicates": int(reps), "seed": int(seed),
                                  "unit": "trace/problem id; windows and outcomes remain attached"},
                    "scope": "Part 177's first 200 sample-0 problems plus the four registered generated sets."}
    write_json(out_dir / "read.json", read_result)
    write_json(out_dir / "score.json", score_result)
    write_json(out_dir / "prediction_table.json", predictions)
    return score_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--inputs-dir", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--part177-dir", type=Path, default=DEFAULT_PART177)
    parser.add_argument("--problems", type=Path, default=HERE / "problems178.json")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    result = score(args.out_dir, args.inputs_dir, args.part177_dir, args.problems,
                   args.replicates, args.seed)
    print(json.dumps({"gates": {key: value.get("passed") for key, value in result["read_gates"].items()},
                      "predictions": {key: row["verdict"] for key, row in result["predictions"].items()}},
                     ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
