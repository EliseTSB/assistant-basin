#!/usr/bin/env python3
"""Part 171 runner: neuron sets, rotated controls, and equal-projection alignment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
from part165 import score165
from part166 import score166
from prompts171 import DRAW_SEEDS, FIT_PROMPTS, K_VALUES, TARGETS, arms as arm_names

MODEL_DEFAULT = "E:/neuron-probe/hf/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LAYERS = tuple(range(21))
VECTOR_LAYER = 20
BATCH_SIZE = 24
MAX_NEW_TOKENS = 80
PILOT_THRESHOLD = 1.0
SLOPE_TOLERANCE = 0.05


def test_catalog(m96=None):
    if m96 is None:
        from part170.prompts170 import M96
        m96 = M96
    from prompts171 import test_prompts
    return test_prompts(m96)


def answer_key(row):
    return (str(row.get("stage")), str(row.get("target")), str(row.get("prompt_id")),
            str(row.get("arm")), str(row.get("draw", "")))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{number}") from exc
    return rows


def repair_tail(path):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            path.write_bytes(data[:data.rfind(b"\n") + 1])


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def append_answers(path, rows, done):
    added = []
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            key = answer_key(row)
            if key in done:
                continue
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            done.add(key)
            added.append(row)
        stream.flush()
    return added


def planned_answer_keys(targets=TARGETS, test_prompts=None, fit_prompts=FIT_PROMPTS,
                        draw_seeds=DRAW_SEEDS, testability=None):
    test_prompts = test_catalog() if test_prompts is None else list(test_prompts)
    keys = set()
    for target in ("default", *targets):
        for qi in range(len(fit_prompts)):
            keys.add(("fit", target, f"p8:{qi}", "fit", ""))
    names = arm_names(draw_seeds)
    for target in targets:
        for prompt in test_prompts:
            for arm in names:
                if arm.startswith("r100p_d") and testability is not None and not bool(
                        testability.get(str(target), {}).get(arm.rsplit("d", 1)[1], False)):
                    continue
                draw = arm.rsplit("_d", 1)[1] if "_d" in arm else ""
                keys.add(("test", target, str(prompt["prompt_id"]), arm, draw))
    return keys


def key_coverage(answers, expected_keys=None, targets=TARGETS, test_prompts=None,
                 fit_prompts=FIT_PROMPTS, draw_seeds=DRAW_SEEDS, testability=None):
    if expected_keys is not None:
        expected = set(expected_keys)
    else:
        expected = planned_answer_keys(targets, test_prompts, fit_prompts, draw_seeds, testability)
    keys = [answer_key(row) for row in answers]
    finite = True
    for row in answers:
        try:
            valid = (isinstance(row.get("answer"), str) and bool(row["answer"].strip()) and
                     isinstance(row.get("token_ids"), list) and bool(row["token_ids"]) and
                     all(int(x) >= 0 for x in row["token_ids"]))
        except (TypeError, ValueError):
            valid = False
        finite = finite and valid
    return {"passed": len(keys) == len(expected) and len(set(keys)) == len(keys) and
            set(keys) == expected and finite,
            "planned": len(expected), "answers": len(keys), "unique_keys": len(set(keys)),
            "missing": len(expected - set(keys)), "unexpected": len(set(keys) - expected),
            "finite_answers": bool(finite)}


def gate2(none_rows, prompts, targets=TARGETS):
    target_names = tuple(targets.keys()) if isinstance(targets, dict) else tuple(targets)
    by_cell = {(str(row.get("target")), str(row.get("prompt_id"))): row for row in none_rows}
    reference = str(target_names[0]) if target_names else ""
    qwen_hits = sum(bool(_word_match(by_cell.get((reference, str(p["prompt_id"])), {}).get("first40", ""),
                                    ("Qwen", "Tongyi", "Alibaba", "通义"))) for p in prompts)
    target_hits = 0
    for target in target_names:
        for prompt in prompts:
            row = by_cell.get((str(target), str(prompt["prompt_id"])), {})
            target_hits += hit_match(row.get("first40", ""), str(target))
    complete = len(none_rows) == len(target_names) * len(prompts) and len(by_cell) == len(none_rows)
    return {"passed": bool(complete and len(prompts) == 24 and qwen_hits >= 20 and target_hits <= 1),
            "qwen_prompts": qwen_hits, "qwen_denominator": len(prompts),
            "hit_rows": target_hits, "hit_denominator": len(target_names) * len(prompts),
            "complete": bool(complete)}


def gate3(plain_checks, zero_checks, expected=8):
    p = {str(row.get("prompt_id")): list(map(int, row.get("token_ids", []))) for row in plain_checks}
    z = {str(row.get("prompt_id")): list(map(int, row.get("token_ids", []))) for row in zero_checks}
    same = len(p) == expected and len(z) == expected and p.keys() == z.keys() and all(p[k] == z[k] for k in p)
    return {"passed": bool(same), "prompts": len(p), "zero_checks": len(z),
            "matching": sum(p.get(k) == z.get(k) for k in p.keys() | z.keys())}


def gate4(gate_records, expected_arms=None, tolerance=0.01, expected_keys=None):
    rows = list(gate_records)
    expected_arms = len(rows) if expected_arms is None else int(expected_arms)
    keys = [(str(row.get("target")), str(row.get("arm"))) for row in rows]
    key_ok = len(keys) == len(set(keys)) and (expected_keys is None or set(keys) == set(expected_keys))
    errors = []
    rounding_by_arm = {re.sub(r"_d\d+$", "", str(arm)): None for arm in arm_names()}
    worst_slope = {}
    for row in rows:
        try:
            target = float(row["target_norm"])
            actual = float(row["realized_write_norm"])
            round_err = float(row["measured_rounding_fraction"])
            intended = float(row["intended_write_norm"])
            valid = (target >= 0 and actual >= 0 and round_err >= 0 and intended >= 0 and
                     all(math.isfinite(x) for x in (target, actual, round_err, intended)))
            length_ok = (valid and actual <= 1e-12) if target == 0 else (
                valid and abs(actual - target) / target <= float(tolerance) + round_err)
            intended_ok = (valid and intended <= 1e-12) if target == 0 else (
                valid and abs(intended - target) / target <= float(tolerance))
            slope = row.get("delivery_slope")
            slope_ok = target == 0 or (slope is not None and math.isfinite(float(slope)) and
                                       abs(float(slope) - 1.0) <= SLOPE_TOLERANCE)
            projection_ok = True
            if row.get("arm", "").startswith("r100p"):
                expected_projection = float(row["n100_projection"])
                actual_projection = float(row["realized_projection"])
                projection_ok = (abs(actual_projection - expected_projection) /
                    max(abs(expected_projection), 1e-12) <= float(tolerance) + round_err)
            errors.append(not (length_ok and intended_ok and slope_ok and projection_ok))
            if target != 0:
                worst = worst_slope.get(re.sub(r"_d\d+$", "", str(row.get("arm", ""))))
                value = float(slope)
                if worst is None or abs(value - 1.0) > abs(worst - 1.0):
                    worst_slope[re.sub(r"_d\d+$", "", str(row.get("arm", "")))] = value
            base_arm = re.sub(r"_d\d+$", "", str(row.get("arm", "")))
            previous = rounding_by_arm.get(base_arm)
            rounding_by_arm[base_arm] = round_err if previous is None else max(previous, round_err)
        except (KeyError, TypeError, ValueError):
            errors.append(True)
    return {"passed": len(rows) == expected_arms and key_ok and not any(errors), "arms": len(rows),
            "expected_arms": expected_arms, "failures": sum(errors),
            "max_rounding_fraction": rounding_by_arm, "worst_delivery_slope": worst_slope}


def gate5(fit_stats, targets=TARGETS, threshold=1.5):
    rows = fit_stats.get("targets", fit_stats) if isinstance(fit_stats, dict) else fit_stats
    ratios = {}
    for target in targets:
        row = rows.get(str(target), {})
        vnorm, nullnorm = float(row.get("vector_norm", 0)), float(row.get("shuffled_default_half_norm", 0))
        ratio = "Infinity" if nullnorm == 0 and vnorm > 0 else (vnorm / nullnorm if nullnorm > 0 else None)
        ratios[str(target)] = ratio
    passed = all(value == "Infinity" or (value is not None and float(value) >= threshold)
                 for value in ratios.values())
    return {"passed": bool(len(ratios) == len(targets) and passed), "ratios": ratios,
            "threshold": float(threshold)}


def gate6(set_json, targets=TARGETS, tolerance=1e-5, draw_seeds=DRAW_SEEDS):
    all_ok = True
    failures = []
    target_sets = set_json.get("targets", {})
    for target in targets:
        meta = target_sets.get(str(target), {})
        counts = meta.get("layer_counts", {})
        for k in (1, 10, 100):
            neuron = counts.get(f"n{k}", {})
            for seed in draw_seeds:
                rotated = counts.get(f"r{k}_d{seed}", {})
                if neuron != rotated:
                    all_ok = False
                    failures.append({"target": str(target), "arm": f"r{k}_d{seed}", "kind": "counts"})
        for arm, layers in meta.get("orthogonality", {}).items():
            if any(float(value) > tolerance for value in layers.values()):
                all_ok = False
                failures.append({"target": str(target), "arm": arm, "kind": "orthogonality"})
    return {"passed": bool(all_ok and len(target_sets) == len(targets)), "failures": failures}


def gate7(test_prompts, fit_prompts=FIT_PROMPTS):
    fit = {" ".join(str(x).casefold().split()) for x in fit_prompts}
    overlap = [str(row["question"]) for row in test_prompts
               if " ".join(str(row["question"]).casefold().split()) in fit]
    return {"passed": not overlap and len(test_prompts) == 24, "overlap": overlap,
            "test_prompts": len(test_prompts)}


def _word_match(text, words):
    text = str(text or "").casefold()
    for word in words:
        needle = str(word).casefold()
        if needle and needle[0].isascii() and needle[0].isalnum():
            if re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z])", text):
                return True
        elif needle in text:
            return True
    return False


def hit_match(text, target):
    spellings = {"google": ("Google", "Gemini", "DeepMind", "谷歌"),
        "openai": ("OpenAI", "ChatGPT", "GPT"), "anthropic": ("Anthropic", "Claude"),
        "deepseek": ("DeepSeek", "深度求索")}
    def contains(sentence, term):
        folded, term = sentence.casefold(), term.casefold()
        if term and term[0].isascii() and term[0].isalnum():
            return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z])", folded) is not None
        return term in folded
    for sentence in re.split(r"(?<=[.!?])\s+|(?<=[。！？])\s*|\n+", str(text or "")):
        if any(contains(sentence, term) for term in spellings.get(str(target), ())):
            negated = (re.search(r"\b(?:not|never)\b", sentence, re.IGNORECASE) is not None or
                       re.search(r"n['’]t", sentence.casefold()) is not None or "不是" in sentence)
            if not negated:
                return True
    return False


def render_prompt(tok, question, role=None):
    messages = ([{"role": "system", "content": role}] if role else []) + [
        {"role": "user", "content": question}
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode_batch(tok, questions, device, roles=None):
    import torch
    if roles is None:
        roles = [None] * len(questions)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        rendered = [render_prompt(tok, q, role) for q, role in zip(questions, roles)]
        encoded = tok(rendered, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = old_side
    return {key: value.to(device) for key, value in encoded.items() if hasattr(value, "to")}


class AddEveryPositionHook:
    """Add one shared down_proj input vector at every position, including prefill."""
    def __init__(self, module, value):
        self.value = value.detach()
        self._cache = {}
        self.handle = module.register_forward_pre_hook(self._hook)

    def _cast_value(self, x):
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = self.value.to(device=x.device, dtype=x.dtype)
        return self._cache[key]

    def _hook(self, _module, args):
        if not args or args[0].ndim != 3 or args[0].shape[-1] != self.value.shape[-1]:
            raise ValueError("down_proj hook input shape mismatch")
        x = args[0]
        pushed = x + self._cast_value(x).view(1, 1, -1)
        return (pushed,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


class AddOutputEveryPositionHook:
    """Add a vector to a decoder block output at every position."""
    def __init__(self, module, value):
        self.value = value.detach()
        self._cache = {}
        self.handle = module.register_forward_hook(self._hook)

    def _cast_value(self, x):
        key = (x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = self.value.to(device=x.device, dtype=x.dtype)
        return self._cache[key]

    def _hook(self, _module, _args, output):
        if isinstance(output, (tuple, list)):
            hidden = output[0]
            pushed = hidden + self._cast_value(hidden).view(1, 1, -1)
            return (pushed,) + tuple(output[1:])
        return output + self._cast_value(output).view(1, 1, -1)

    def remove(self):
        self.handle.remove()


def generate_batch(model, tok, layers, questions, roles, device, max_new_tokens=MAX_NEW_TOKENS,
                   pushes=None, vector=None, seed=171):
    import torch
    inputs = encode_batch(tok, questions, device, roles)
    prompt_width = int(inputs["input_ids"].shape[-1])
    handles = []
    pushes = pushes or {}
    for layer_no, push in sorted(pushes.items()):
        val = torch.as_tensor(push, device=device, dtype=torch.float32)
        handles.append(AddEveryPositionHook(layers[int(layer_no)].mlp.down_proj, val))
    if vector is not None:
        val = torch.as_tensor(vector, device=device, dtype=torch.float32)
        handles.append(AddOutputEveryPositionHook(layers[VECTOR_LAYER], val))
    eos = set(map(int, common86.eos_ids(tok, model)))
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            generated = model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
                use_cache=True, pad_token_id=getattr(tok, "pad_token_id", None),
                eos_token_id=sorted(eos) or None)
    finally:
        for hook in handles:
            hook.remove()
    outputs = []
    for row_no, seq in enumerate(generated[:, prompt_width:].detach().cpu().tolist()):
        ids = [int(token) for token in seq]
        for index, token in enumerate(ids):
            if token in eos:
                ids = ids[:index + 1]
                break
        outputs.append({"answer": tok.decode(ids, skip_special_tokens=True).strip(),
                        "first40": tok.decode(ids[:40], skip_special_tokens=True).strip(),
                        "token_ids": ids, "token_count": len(ids),
                        "prompt_ids": [int(x) for x in inputs["input_ids"][row_no]
                                       [inputs["attention_mask"][row_no].bool()].detach().cpu().tolist()]})
    return outputs


def _reply_ids(row, eos):
    ids = list(map(int, row.get("token_ids", [])))
    if ids and ids[-1] in eos:
        ids = ids[:-1]
    return ids


def collect_fit_batch(model, tok, layers, rows, device):
    """Capture token sums on device and transfer one packed tensor per fit batch."""
    import numpy as np
    import torch
    eos = set(map(int, common86.eos_ids(tok, model)))
    prompt_ids = [list(map(int, row["prompt_ids"])) for row in rows]
    replies = [_reply_ids(row, eos) for row in rows]
    if any(not ids for ids in replies):
        raise ValueError("fit answer has no non-end reply tokens")
    sequences = [p + a for p, a in zip(prompt_ids, replies)]
    width = max(map(len, sequences))
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
    input_rows, masks, reply_masks = [], [], []
    starts = []
    for prompt, reply, seq in zip(prompt_ids, replies, sequences):
        pad = width - len(seq)
        input_rows.append([pad_id] * pad + seq)
        masks.append([0] * pad + [1] * len(seq))
        reply_masks.append([0] * (pad + len(prompt)) + [1] * len(reply))
        starts.append(pad + len(prompt))
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention = torch.tensor(masks, dtype=torch.long, device=device)
    reply_mask = torch.tensor(reply_masks, dtype=torch.float32, device=device)
    count = int(sum(map(len, replies)))
    names = [f"down:{layer_no}" for layer_no in LAYERS] + ["hidden20"]
    sums = {name: torch.zeros((len(rows), int(layers[VECTOR_LAYER].mlp.down_proj.weight.shape[0])
                              if name == "hidden20" else
                              int(layers[int(name.split(':')[1])].mlp.down_proj.weight.shape[1])),
                              device=device, dtype=torch.float32) for name in names}
    token_norms = None
    handles = []

    def make_pre(name):
        def capture(_module, args):
            x = args[0].detach().float()
            sums[name].add_((x * reply_mask.unsqueeze(-1)).sum(dim=1))
            return None
        return capture

    def capture_hidden(_module, _args, output):
        nonlocal token_norms
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        values = hidden.detach().float()
        sums["hidden20"].add_((values * reply_mask.unsqueeze(-1)).sum(dim=1))
        token_norms = torch.linalg.vector_norm(values, dim=-1)[reply_mask.bool()]

    for layer_no in LAYERS:
        handles.append(layers[layer_no].mlp.down_proj.register_forward_pre_hook(make_pre(f"down:{layer_no}")))
    handles.append(layers[VECTOR_LAYER].register_forward_hook(capture_hidden))
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    # Per-row hidden sums support the registered shuffled-half null without another forward.
    packed_parts = [sums[name].reshape(-1) for name in names]
    packed_parts.append(token_norms.reshape(-1))
    packed = torch.cat(packed_parts).detach().to(device="cpu", dtype=torch.float32).numpy()
    offset = 0
    result = {"reply_token_count": count, "row_counts": [len(x) for x in replies],
              "row_hidden_sums": [],
              "sums": {}}
    # Re-read packed sections in a stable order.
    row_sums = {}
    for name in names:
        width_n = sums[name].shape[1]
        size = len(rows) * width_n
        row_sums[name] = packed[offset:offset + size].reshape(len(rows), width_n).copy()
        offset += size
    hidden20 = row_sums["hidden20"]
    token_norm_values = packed[offset:]
    result["row_hidden_sums"] = hidden20.tolist()
    result["sums"] = {name: array.sum(axis=0).astype(np.float32).tolist()
                      for name, array in row_sums.items()}
    result["row_sums"] = {name: array.tolist() for name, array in row_sums.items()}
    result["top1pct_token_share"] = _top_share(token_norm_values)
    return result


def _top_share(norms):
    import numpy as np
    values = np.asarray(norms, dtype=np.float64)
    if not values.size or float(values.sum()) <= 0:
        return 0.0
    count = max(1, int(math.ceil(values.size * 0.01)))
    return float(np.partition(values, -count)[-count:].sum() / values.sum())


def prepare_fit_target(v, delta, weights, k_values=K_VALUES):
    """Rank neuron contributions on the CPU and retain only fit factors for rotation."""
    import numpy as np
    v = np.asarray(v, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    if delta.ndim != 2 or delta.shape[0] != len(weights):
        raise ValueError("fit tensor dimensions do not align")
    vnorm = float(np.linalg.norm(v.astype(np.float64)))
    if not math.isfinite(vnorm) or vnorm <= 0:
        raise ValueError("fit vector must have finite nonzero length")
    vhat = (v / np.float32(vnorm)).astype(np.float32)
    p = np.stack([np.asarray(w, dtype=np.float32).T @ vhat for w in weights])
    score = np.abs(delta * p).reshape(-1)
    order = np.lexsort((np.arange(score.size, dtype=np.int64), -score))
    width = int(delta.shape[1])
    selected, counts = {}, {}
    for name, k in k_values.items():
        coords = order[:min(int(k), score.size)]
        by_layer = {}
        for layer in range(delta.shape[0]):
            start = layer * width
            by_layer[layer] = (coords[(coords >= start) & (coords < start + width)] - start).astype(np.int64)
        selected[name] = by_layer
        counts[name] = {str(layer): int(len(ix)) for layer, ix in by_layer.items()}
    return {"v": v, "delta": delta, "p": p, "v_norm": vnorm,
        "natural_projection": float(np.sum(delta.astype(np.float64) * p.astype(np.float64))),
        "n_indices": {name: {int(layer): ix.tolist() for layer, ix in by_layer.items()}
                      for name, by_layer in selected.items()}, "layer_counts": counts}


def _alignment_measure(write, projection, layer_natural, raw_projection=None):
    import numpy as np
    norm = float(np.linalg.norm(np.asarray(write, dtype=np.float64)))
    return {"nominal_norm": norm, "projection": float(projection),
        "perpendicular_norm": math.sqrt(max(0.0, norm * norm - float(projection) ** 2)),
        "cosine": float(projection) / norm if norm else None,
        "projection_per_unit_length": float(projection) / norm if norm else None,
        "natural_projection": float(layer_natural),
        "fraction_natural_projection": (float(raw_projection) / float(layer_natural)
                                         if raw_projection is not None and layer_natural else None)}


def _scale_rep(rep, scale):
    import numpy as np
    if rep["kind"] == "dense":
        return {"kind": "dense", "values": (np.asarray(rep["values"], dtype=np.float32) * np.float32(scale)).tolist()}
    if rep["kind"] == "sparse":
        return {"kind": "sparse", "coords": list(map(int, rep["coords"])),
                "values": (np.asarray(rep["values"], dtype=np.float32) * np.float32(scale)).tolist()}
    return {"kind": "basis", "directions": np.asarray(rep["directions"], dtype=np.float32),
            "coefficients": (np.asarray(rep["coefficients"], dtype=np.float32) * np.float32(scale)).tolist()}


def build_target_sets(fit_data, weights, rotation_sets, draw_seeds=DRAW_SEEDS):
    """Build and scale nominal writes using CPU float64 write arithmetic."""
    import numpy as np
    v = np.asarray(fit_data["v"], dtype=np.float32)
    delta = np.asarray(fit_data["delta"], dtype=np.float32)
    p = np.asarray(fit_data["p"], dtype=np.float32)
    weights = [np.asarray(w, dtype=np.float32) for w in weights]
    vnorm = float(np.linalg.norm(v.astype(np.float64)))
    vhat = v.astype(np.float64) / vnorm
    desired = 4.0 * vnorm
    natural = float(fit_data["natural_projection"])
    counts = fit_data["layer_counts"]
    indices = {name: {int(layer): np.asarray(coords, dtype=np.int64) for layer, coords in by_layer.items()}
               for name, by_layer in fit_data["n_indices"].items()}
    raw = {"full": {layer: {"kind": "dense", "values": delta[layer].copy()}
                     for layer in range(len(weights))}}
    for name, layers_idx in indices.items():
        raw[name] = {layer: {"kind": "sparse", "coords": coords.tolist(),
                             "values": delta[layer, coords].tolist()}
                     for layer, coords in layers_idx.items()}
    rotation_arms = {}
    orth = {}
    for seed in draw_seeds:
        for kname in ("n1", "n10", "n100"):
            rname = f"r{kname[1:]}_d{seed}"
            spec, errs = {}, {}
            for layer in range(len(weights)):
                count = int(counts[kname][str(layer)])
                raw_q = rotation_sets.get(rname, {})
                q = np.asarray(raw_q.get(layer, raw_q.get(str(layer), [])), dtype=np.float32)
                if count == 0:
                    q = np.empty((0, delta.shape[1]), dtype=np.float32)
                if q.shape != (count, delta.shape[1]):
                    raise ValueError(f"{rname}/L{layer} shape {q.shape}; expected {(count, delta.shape[1])}")
                gram = q @ q.T
                errs[str(layer)] = float(np.max(np.abs(gram - np.eye(count, dtype=np.float32)))) if count else 0.0
                spec[layer] = {"kind": "basis", "directions": q,
                               "coefficients": (q @ delta[layer]).astype(np.float32)}
            raw[rname] = spec
            rotation_arms[rname] = spec
            orth[rname] = errs
            counts[rname] = {str(layer): int(item["directions"].shape[0]) for layer, item in spec.items()}
        rand_spec = {}
        for layer in range(len(weights)):
            count = int(counts["n100"][str(layer)])
            rng = np.random.default_rng(np.random.SeedSequence([int(seed) + 10_000, layer]))
            coords = np.sort(rng.permutation(delta.shape[1])[:count]) if count else np.asarray([], dtype=np.int64)
            rand_spec[layer] = {"kind": "sparse", "coords": coords.tolist(),
                                "values": delta[layer, coords].tolist()}
        raw[f"rand100_d{seed}"] = rand_spec
        counts[f"rand100_d{seed}"] = {str(layer): len(item["coords"]) for layer, item in rand_spec.items()}

    def write_for(spec):
        total = np.zeros(weights[0].shape[0], dtype=np.float64)
        layer_writes, xs = {}, {}
        for layer, weight in enumerate(weights):
            rep = spec.get(layer)
            if rep is None:
                x = np.zeros(weight.shape[1], dtype=np.float32)
            elif rep["kind"] == "dense":
                x = np.asarray(rep["values"], dtype=np.float32)
            elif rep["kind"] == "sparse":
                x = np.zeros(weight.shape[1], dtype=np.float32)
                if rep["coords"]:
                    x[np.asarray(rep["coords"], dtype=np.int64)] = np.asarray(rep["values"], dtype=np.float32)
            else:
                q, coeff = np.asarray(rep["directions"], dtype=np.float32), np.asarray(rep["coefficients"], dtype=np.float32)
                x = coeff @ q if coeff.size else np.zeros(weight.shape[1], dtype=np.float32)
            layer_write = weight.astype(np.float64) @ x.astype(np.float64)
            total += layer_write
            layer_writes[layer], xs[layer] = layer_write, x
        return total, layer_writes, xs

    metrics, saved, raw_cache = {}, {}, {}
    for arm, spec in raw.items():
        write, by_layer, _xs = write_for(spec)
        norm = float(np.linalg.norm(write))
        if norm <= 0 or not math.isfinite(norm):
            raise ValueError(f"arm {arm} has zero/nonfinite nominal write")
        scale = desired / norm
        raw_cache[arm] = (write, by_layer)
        scaled = write * scale
        scaled_norm = float(np.linalg.norm(scaled))
        projection = float(scaled @ vhat)
        layer_info, scaled_spec = {}, {}
        for layer, layer_write in by_layer.items():
            w = layer_write * scale
            layer_proj = float(w @ vhat)
            layer_info[str(layer)] = _alignment_measure(w, layer_proj,
                float(delta[layer].astype(np.float64) @ p[layer].astype(np.float64)),
                raw_projection=float(layer_write @ vhat))
            scaled_spec[layer] = _scale_rep(spec[layer], scale)
        metrics[arm] = {"s": float(scale), "target_norm": desired,
            "nominal_norm": scaled_norm, "write_norm": scaled_norm, "projection": projection,
            "perpendicular_norm": math.sqrt(max(0.0, scaled_norm ** 2 - projection ** 2)),
            "cosine": projection / scaled_norm, "projection_per_unit_length": projection / scaled_norm,
            "natural_projection": natural,
            "fraction_natural_projection": float(write @ vhat) / natural if natural else None,
            "layer_metrics": layer_info, "status": "testable"}
        saved[arm] = scaled_spec

    testability, debts = {}, []
    for seed in draw_seeds:
        rarm, parm = f"r100_d{seed}", f"r100p_d{seed}"
        rwrite = raw_cache[rarm][0]
        rnorm = float(np.linalg.norm(rwrite))
        rscale = desired / rnorm
        rcos = float((rwrite * rscale) @ vhat) / desired
        ncos = float(metrics["n100"]["cosine"])
        debt = ncos / rcos if rcos > .01 else None
        okay = bool(rcos > .01 and debt is not None and debt <= 2.0)
        testability[str(seed)] = okay
        debts.append({"draw": int(seed), "cos_n": ncos, "cos_r": rcos, "debt": debt,
                      "testable": okay, "status": "testable" if okay else "NOT_TESTABLE"})
        denom = float(rwrite @ vhat)
        if abs(denom) <= 1e-20:
            metrics[parm] = {"s": None, "target_norm": None, "nominal_norm": None,
                "write_norm": None, "projection": None, "perpendicular_norm": None,
                "cosine": None, "projection_per_unit_length": None, "natural_projection": natural,
                "fraction_natural_projection": float(rwrite @ vhat) / natural if natural else None,
                "layer_metrics": {}, "status": "NOT_TESTABLE",
                "alignment_debt": debt}
            saved[parm] = {}
            continue
        nwrite = raw_cache["n100"][0] * float(metrics["n100"]["s"])
        scale = float(nwrite @ vhat) / denom
        scaled = rwrite * scale
        scaled_norm = float(np.linalg.norm(scaled))
        projection = float(scaled @ vhat)
        layer_info, scaled_spec = {}, {}
        for layer, layer_write in raw_cache[rarm][1].items():
            w = layer_write * scale
            layer_proj = float(w @ vhat)
            layer_info[str(layer)] = _alignment_measure(w, layer_proj,
                float(delta[layer].astype(np.float64) @ p[layer].astype(np.float64)),
                raw_projection=float(layer_write @ vhat))
            scaled_spec[layer] = _scale_rep(raw[rarm][layer], scale)
        metrics[parm] = {"s": scale, "target_norm": scaled_norm, "nominal_norm": scaled_norm,
            "write_norm": scaled_norm, "projection": projection,
            "perpendicular_norm": math.sqrt(max(0.0, scaled_norm ** 2 - projection ** 2)),
            "cosine": projection / scaled_norm if scaled_norm else None,
            "projection_per_unit_length": projection / scaled_norm if scaled_norm else None,
            "natural_projection": natural,
            "fraction_natural_projection": float(rwrite @ vhat) / natural if natural else None,
            "layer_metrics": layer_info, "status": "testable" if okay else "NOT_TESTABLE",
            "alignment_debt": debt, "n100_projection": float(metrics["n100"]["projection"])}
        saved[parm] = scaled_spec

    vector_projection = desired
    metrics["vector"] = {"s": 4.0, "target_norm": desired, "nominal_norm": desired,
        "write_norm": desired, "projection": vector_projection, "perpendicular_norm": 0.0,
        "cosine": 1.0, "projection_per_unit_length": 1.0,
        "natural_projection": natural, "fraction_natural_projection": None,
        "layer_metrics": {str(layer): _alignment_measure(v.astype(np.float64) * 4 if layer == VECTOR_LAYER else np.zeros_like(v),
            vector_projection if layer == VECTOR_LAYER else 0.0, 0.0, raw_projection=None)
            for layer in range(len(weights))}, "status": "testable"}
    metrics["none"] = {"s": 0.0, "target_norm": 0.0, "nominal_norm": 0.0, "write_norm": 0.0,
        "projection": 0.0, "perpendicular_norm": 0.0, "cosine": None,
        "projection_per_unit_length": None, "natural_projection": natural,
        "fraction_natural_projection": 0.0, "layer_metrics": {}, "status": "testable"}
    saved["vector"] = {}
    for seed in draw_seeds:
        orth[f"r100p_d{seed}"] = orth.get(f"r100_d{seed}", {})
        counts[f"r100p_d{seed}"] = dict(counts[f"r100_d{seed}"])
    return {"v": v, "delta": delta, "p": p, "n_indices": fit_data["n_indices"],
        "arms": saved, "metrics": metrics,
        "layer_counts": counts, "orthogonality": orth, "top_contributions": fit_data["n_indices"],
        "natural_projection": natural, "v_norm": vnorm, "target_norm": desired,
        "debt": debts, "testability": testability}


def materialize_arm(target_data, arm, widths, device):
    import numpy as np
    import torch
    if arm == "vector":
        return {}, torch.as_tensor(target_data["v"] * 4.0, dtype=torch.float32, device=device)
    if arm == "none":
        return {}, None
    reps = target_data["arms"][arm]
    result = {}
    for layer_no, width in widths.items():
        rep = reps.get(layer_no)
        if not rep:
            continue
        if rep["kind"] == "dense":
            value = np.asarray(rep["values"], dtype=np.float32)
        elif rep["kind"] == "sparse":
            value = np.zeros(int(width), dtype=np.float32)
            if rep["coords"]:
                value[np.asarray(rep["coords"], dtype=np.int64)] = np.asarray(rep["values"], dtype=np.float32)
        else:
            value = np.asarray(rep["coefficients"], dtype=np.float32) @ np.asarray(rep["directions"], dtype=np.float32)
        result[int(layer_no)] = torch.as_tensor(value, dtype=torch.float32, device=device)
    return result, None


def _torch_save(path, target_data):
    import torch
    def tensorize(value):
        if isinstance(value, dict):
            return {key: tensorize(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return torch.as_tensor(value, dtype=torch.float32)
        if isinstance(value, (list, tuple)):
            return [tensorize(item) for item in value]
        return value
    import numpy as np
    serial = {"v": torch.as_tensor(target_data["v"], dtype=torch.float32),
              "delta": torch.as_tensor(target_data["delta"], dtype=torch.float32),
              "arms": tensorize(target_data["arms"]), "metrics": target_data["metrics"],
              "natural_projection": float(target_data["natural_projection"]),
              "v_norm": float(target_data["v_norm"]),
              "target_norm": float(target_data["target_norm"])}
    for key in ("p", "n_indices", "layer_counts", "testability", "top_contributions"):
        if key in target_data:
            serial[key] = tensorize(target_data[key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(serial, path)


class _TraceDownWrite:
    """Measure what the real hook delivers, with observers on either side of it."""
    def __init__(self, module, push):
        self.push = push.detach().view(1, 1, -1)
        self.before = None
        self.actual = None
        self.expected = None
        self.first = module.register_forward_pre_hook(self._before)
        self.real = AddEveryPositionHook(module, push)
        self.last = module.register_forward_pre_hook(self._after)

    def _before(self, _module, args):
        self.before = args[0][:, -1, :].float()

    def _after(self, module, args):
        import torch
        effective = args[0][:, -1, :].float() - self.before
        weight = module.weight.detach().float().T
        self.actual = effective @ weight
        self.expected = self.push[:, 0, :].to(device=effective.device, dtype=torch.float32) @ weight

    def remove(self):
        self.first.remove()
        self.real.remove()
        self.last.remove()


class _TraceOutputWrite:
    """Measure what the real output hook delivers, with observers on either side of it."""
    def __init__(self, module, push):
        self.push = push.detach().view(1, 1, -1)
        self.before = None
        self.actual = None
        self.expected = None
        self.first = module.register_forward_hook(self._before)
        self.real = AddOutputEveryPositionHook(module, push)
        self.last = module.register_forward_hook(self._after)

    def _before(self, _module, _args, output):
        x = output[0] if isinstance(output, (tuple, list)) else output
        self.before = x[:, -1, :].float()

    def _after(self, _module, _args, output):
        import torch
        x = output[0] if isinstance(output, (tuple, list)) else output
        self.actual = x[:, -1, :].float() - self.before
        self.expected = self.push[:, 0, :].to(device=x.device, dtype=torch.float32)

    def remove(self):
        self.first.remove()
        self.real.remove()
        self.last.remove()


def measure_gate4_arm(model, tok, layers, target_data, target, arm, question, device):
    """Measure bf16-delivered writes at the final position of one fixed context."""
    import numpy as np
    import torch
    widths = {int(layer): int(layers[layer].mlp.down_proj.weight.shape[1]) for layer in LAYERS}
    pushes, vector = materialize_arm(target_data, arm, widths, device)
    inputs = encode_batch(tok, [question], device, [None])
    baseline_observer = _TraceLayer20(layers[VECTOR_LAYER])
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        baseline_observer.remove()
    baseline_hidden = baseline_observer.hidden
    traces = []
    for layer_no, push in sorted(pushes.items()):
        if bool(torch.count_nonzero(push)):
            traces.append(_TraceDownWrite(layers[layer_no].mlp.down_proj, push))
    if vector is not None and bool(torch.count_nonzero(vector)):
        traces.append(_TraceOutputWrite(layers[VECTOR_LAYER], vector))
    final_observer = _TraceLayer20(layers[VECTOR_LAYER])
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        for trace in traces:
            trace.remove()
        final_observer.remove()
    active = [x for x in traces if x.actual is not None]
    output_delta = final_observer.hidden - baseline_hidden
    if active:
        packed = torch.cat([part.reshape(-1) for trace in active for part in (trace.actual, trace.expected)] +
                           [baseline_hidden.reshape(-1), final_observer.hidden.reshape(-1)])
        cpu = packed.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
        hidden = int(active[0].actual.shape[-1])
        offset = 0
        actual_sum = np.zeros(hidden, dtype=np.float64)
        intended_sum = np.zeros(hidden, dtype=np.float64)
        for _trace in active:
            actual_sum += cpu[offset:offset + hidden]; offset += hidden
            intended_sum += cpu[offset:offset + hidden]; offset += hidden
        offset += hidden
        layer20_delta = cpu[offset:offset + hidden] - cpu[offset - hidden:offset]
    else:
        actual_sum = intended_sum = np.zeros(int(target_data["v"].size), dtype=np.float64)
        layer20_delta = output_delta.detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64).reshape(-1)
    meta = target_data["metrics"][arm]
    target_norm = float(meta["target_norm"])
    realized_norm = float(np.linalg.norm(actual_sum))
    intended_norm = float(np.linalg.norm(intended_sum))
    v = np.asarray(target_data["v"], dtype=np.float64)
    vhat = v / np.linalg.norm(v)
    projection = float(actual_sum @ vhat)
    rounding = float(np.linalg.norm(actual_sum - intended_sum) / target_norm) if target_norm else 0.0
    # Rounding is close to orthogonal to the write, so a hook or a truncated push moves this slope and rounding does not.
    denominator = float(intended_sum @ intended_sum)
    slope = float(actual_sum @ intended_sum / denominator) if denominator else None
    propagated_length = float(np.linalg.norm(layer20_delta))
    propagated_angle = float(layer20_delta @ vhat / propagated_length) if propagated_length else None
    record = {"target": target, "arm": arm, "target_norm": target_norm,
        "intended_write_norm": intended_norm, "realized_write_norm": realized_norm,
        "measured_rounding_fraction": rounding, "delivery_slope": slope,
        "realized_projection": projection, "actual_layer20_length": propagated_length,
        "actual_layer20_angle_cosine": propagated_angle}
    if arm.startswith("r100p"):
        seed = arm.rsplit("_d", 1)[1]
        record["n100_projection"] = float(target_data["metrics"]["n100"]["projection"])
        record["draw"] = int(seed)
    return record


class _TraceLayer20:
    def __init__(self, module):
        self.hidden = None
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        self.hidden = hidden[:, -1, :].detach().float()

    def remove(self):
        self.handle.remove()


def measure_gate4(model, tok, layers, target_sets, targets, test_prompts, draw_seeds=DRAW_SEEDS,
                  selected_arms_by_target=None, device="cpu"):
    selected_arms_by_target = selected_arms_by_target or {target: arm_names(draw_seeds) for target in targets}
    rows = []
    context = str(test_prompts[0]["question"])
    for target in targets:
        target_data = target_sets[str(target)]
        for arm in selected_arms_by_target[str(target)]:
            rows.append(measure_gate4_arm(model, tok, layers, target_data, str(target), arm, context, device))
    return rows


def _make_fit_rows(target, prompts, answers):
    result = []
    by_pid = {str(row.get("prompt_id")): row for row in answers
              if row.get("stage") == "fit" and row.get("target") == target}
    for index, question in enumerate(prompts):
        pid = f"p8:{index}"
        if pid not in by_pid:
            raise ValueError(f"missing fit answer {target}/{pid}")
        result.append(by_pid[pid])
    return result


def _fit_recipe(model, tok, layers, prompts, answers, roles, device, seed=171):
    import numpy as np
    collected = {}
    for target, role in roles.items():
        rows = _make_fit_rows(target, prompts, answers)
        collected[target] = collect_fit_batch(model, tok, layers, rows, device)
    default = collected["default"]
    default_count = int(default["reply_token_count"])
    if default_count <= 0:
        raise ValueError("default fit has no reply tokens")
    rng_order = list(range(len(prompts)))
    random.Random(int(seed)).shuffle(rng_order)
    halves = (rng_order[:len(rng_order) // 2], rng_order[len(rng_order) // 2:])
    half_means = []
    default_rows = default["row_sums"]
    row_counts = default["row_counts"]
    for group in halves:
        total = np.sum(np.asarray([default_rows["hidden20"][i] for i in group], dtype=np.float32), axis=0)
        denom = sum(row_counts[i] for i in group)
        half_means.append(total / max(denom, 1))
    null_vector = np.asarray(half_means[0], dtype=np.float32) - np.asarray(half_means[1], dtype=np.float32)
    outputs = {}
    stats = {}
    for target in roles:
        if target == "default":
            continue
        data = collected[target]
        count = int(data["reply_token_count"])
        v = np.asarray(data["sums"]["hidden20"], dtype=np.float32) / count - \
            np.asarray(default["sums"]["hidden20"], dtype=np.float32) / default_count
        delta = []
        for layer_no in LAYERS:
            name = f"down:{layer_no}"
            delta.append(np.asarray(data["sums"][name], dtype=np.float32) / count -
                         np.asarray(default["sums"][name], dtype=np.float32) / default_count)
        outputs[target] = {"v": v, "delta": delta}
        vnorm = float(np.linalg.norm(v.astype(np.float64)))
        nullnorm = float(np.linalg.norm(null_vector.astype(np.float64)))
        ratio = "Infinity" if nullnorm == 0 and vnorm > 0 else (vnorm / nullnorm if nullnorm else None)
        stats[target] = {"vector_norm": vnorm, "shuffled_default_half_norm": nullnorm,
            "vector_to_null_ratio": ratio,
            "target_top1pct_token_share": float(data["top1pct_token_share"]),
            "default_top1pct_token_share": float(default["top1pct_token_share"]),
            "fit_reply_tokens": count, "default_reply_tokens": default_count,
            "shuffle_order": rng_order, "half_a_indices": halves[0], "half_b_indices": halves[1]}
    return outputs, {"targets": stats, "default_top1pct_token_share": default["top1pct_token_share"],
                     "shuffle_order": rng_order, "null_vector": null_vector}


def _cpu_weights(layers):
    return [layers[i].mlp.down_proj.weight.detach().float().to(device="cpu").numpy().copy()
            for i in LAYERS]


def _answer_record(stage, target, arm, prompt, answer, draw=None):
    result = {"stage": stage, "target": str(target), "arm": str(arm),
        "prompt_id": str(prompt["prompt_id"]), "question_index": int(prompt["question_index"]),
        "question": str(prompt["question"]), **answer}
    if draw is not None:
        result["draw"] = int(draw)
    return result


def _fit_answer_record(target, prompt_index, question, answer):
    return {"stage": "fit", "target": str(target), "arm": "fit", "prompt_id": f"p8:{prompt_index}",
        "question_index": int(prompt_index), "question": question, **answer}


def _pilot_update(path, state, count, elapsed):
    if state.get("passed") is False:
        raise SystemExit(f"pilot stop: {state['rate']:.3f} answers/s below {PILOT_THRESHOLD}")
    if state.get("batches", 0) >= 3:
        return state
    state["batches"] = int(state.get("batches", 0)) + 1
    state["answers"] = int(state.get("answers", 0)) + int(count)
    state["elapsed_seconds"] = float(state.get("elapsed_seconds", 0.0)) + float(elapsed)
    if state["batches"] == 3:
        state["rate"] = state["answers"] / max(state["elapsed_seconds"], 1e-9)
        state["passed"] = bool(state["rate"] >= PILOT_THRESHOLD)
    write_json(path, state)
    if state.get("passed") is False:
        raise SystemExit(f"pilot stop: {state['rate']:.3f} answers/s below {PILOT_THRESHOLD}")
    return state


def validate_model(layers):
    if len(layers) < 21:
        raise ValueError(f"Part 171 needs decoder layers 0 through 20; found {len(layers)}")
    widths = {int(layers[i].mlp.down_proj.weight.shape[1]) for i in LAYERS}
    if len(widths) != 1:
        raise ValueError("down_proj input widths differ across layers 0..20")
    if next(iter(widths)) != 9216:
        raise ValueError(f"Qwen3.5-4B down_proj width must be 9216; found {next(iter(widths))}")


def fit_hashes(out, targets):
    return {str(target): hashlib.sha256((Path(out) / f"fit_{target}.pt").read_bytes()).hexdigest()
            for target in targets}


def _save_fit_tensor(path, data):
    import os
    import torch
    serial = {"v": torch.as_tensor(data["v"], dtype=torch.float32),
        "delta": torch.as_tensor(data["delta"], dtype=torch.float32),
        "p": torch.as_tensor(data["p"], dtype=torch.float32),
        "v_norm": float(data["v_norm"]), "natural_projection": float(data["natural_projection"]),
        "n_indices": data["n_indices"], "layer_counts": data["layer_counts"]}
    temp = Path(path).with_suffix(".pt.tmp")
    torch.save(serial, temp)
    os.replace(temp, path)


def _load_fit_tensor(path):
    import numpy as np
    import torch
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    return {**obj, "v": obj["v"].numpy().astype(np.float32, copy=True),
        "delta": obj["delta"].numpy().astype(np.float32, copy=True),
        "p": obj["p"].numpy().astype(np.float32, copy=True)}


def _fit_stage(args, tok, model, layers, out, targets, fit_prompts, tests, draw_seeds, k_values):
    import numpy as np
    import torch
    answer_path = out / "answers.jsonl"
    repair_tail(answer_path)
    answers = read_jsonl(answer_path)
    done = {answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("answers.jsonl contains duplicate answer keys")
    pilot_path = out / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else {
        "batches": 0, "answers": 0, "elapsed_seconds": 0.0}
    # Gate 2 baseline: no role prompt, a stable 24-row batch for every target label.
    for target in targets:
        keys = [answer_key({"stage": "test", "target": target, "prompt_id": p["prompt_id"], "arm": "none"})
                for p in tests]
        if all(key in done for key in keys):
            continue
        started = time.perf_counter()
        generated = generate_batch(model, tok, layers, [p["question"] for p in tests], [None] * len(tests),
                                   args.device, args.max_new_tokens, seed=171)
        elapsed = time.perf_counter() - started
        if len(generated) != len(tests):
            raise RuntimeError("none batch answer count mismatch")
        append_answers(answer_path, [_answer_record("test", target, "none", p, a)
                                     for p, a in zip(tests, generated)], done)
        pilot = _pilot_update(pilot_path, pilot, len(tests), elapsed)
    answers = read_jsonl(answer_path)
    none_rows = [x for x in answers if x.get("stage") == "test" and x.get("arm") == "none"]
    g2 = gate2(none_rows, tests, targets)
    if not g2["passed"]:
        write_json(out / "stage0.json", {"gate2": g2, "gate3": {"passed": False, "skipped": True},
                                          "test_prompts": tests, "fit_prompts": list(fit_prompts)})
        raise SystemExit(f"Gate 2 failed before fit or rotation: {g2}")
    checks = tests[:8]
    plain = generate_batch(model, tok, layers, [p["question"] for p in checks], [None] * len(checks),
                           args.device, args.max_new_tokens, seed=171)
    zeros = {layer: torch.zeros(int(layers[layer].mlp.down_proj.weight.shape[1]),
                                dtype=torch.float32, device=args.device) for layer in LAYERS}
    zero = generate_batch(model, tok, layers, [p["question"] for p in checks], [None] * len(checks),
                          args.device, args.max_new_tokens, pushes=zeros, seed=171)
    plain_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]} for p, a in zip(checks, plain)]
    zero_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]} for p, a in zip(checks, zero)]
    g3 = gate3(plain_rows, zero_rows, 8)
    stage0 = {"gate2": g2, "gate3": g3, "plain_checks": plain_rows, "zero_checks": zero_rows,
        "test_prompts": tests, "fit_prompts": list(fit_prompts),
        "frozen_m96_indices": list(__import__("prompts171").FROZEN_M96_MAKER_INDICES)}
    write_json(out / "stage0.json", stage0)
    if not g3["passed"]:
        raise SystemExit(f"Gate 3 failed before fit or rotation: {g3}")
    roles = {"default": None, **targets}
    for target, role in roles.items():
        keys = [answer_key({"stage": "fit", "target": target, "prompt_id": f"p8:{i}", "arm": "fit"})
                for i in range(len(fit_prompts))]
        if all(key in done for key in keys):
            continue
        generated = generate_batch(model, tok, layers, fit_prompts, [role] * len(fit_prompts),
                                   args.device, args.max_new_tokens, seed=171)
        rows = [_fit_answer_record(target, i, question, answer)
                for i, (question, answer) in enumerate(zip(fit_prompts, generated))]
        append_answers(answer_path, rows, done)
    answers = read_jsonl(answer_path)
    fit_outputs, fit_stats = _fit_recipe(model, tok, layers, fit_prompts, answers, roles,
                                         args.device, seed=171)
    weights = _cpu_weights(layers)
    for target in targets:
        prepared = prepare_fit_target(fit_outputs[target]["v"], fit_outputs[target]["delta"],
                                      weights, k_values=k_values)
        safe = str(target).replace("/", "_")
        _save_fit_tensor(out / f"fit_{safe}.pt", prepared)
    public_stats = {"targets": fit_stats["targets"],
                    "default_top1pct_token_share": fit_stats["default_top1pct_token_share"],
                    "shuffle_order": fit_stats["shuffle_order"]}
    write_json(out / "fit_stats.json", public_stats)
    fit_gate = gate5(public_stats, targets)
    if not fit_gate["passed"]:
        write_json(out / "fit_gate.json", {"gate5": fit_gate})
        raise SystemExit(f"Gate 5 failed during fit: {fit_gate}")
    # Written last: rotate171.py refuses without it, so a failed gate or a partial tensor cannot start the Haar step.
    write_json(out / "fit_complete.json", {"gate5": fit_gate, "sha256": fit_hashes(out, targets)})
    return {"stage": "fit", "answers": len(read_jsonl(answer_path)), "gate2": g2,
            "gate3": g3, "gate5": fit_gate, "fit_files": [f"fit_{t}.pt" for t in targets]}


def _test_stage(args, tok, model, layers, out, targets, fit_prompts, tests, draw_seeds):
    import numpy as np
    from rotate171 import load_rotations
    answer_path = out / "answers.jsonl"
    repair_tail(answer_path)
    answers = read_jsonl(answer_path)
    done = {answer_key(row) for row in answers}
    stage0 = json.loads((out / "stage0.json").read_text(encoding="utf-8"))
    fit_stats = json.loads((out / "fit_stats.json").read_text(encoding="utf-8"))
    if not stage0.get("gate2", {}).get("passed") or not stage0.get("gate3", {}).get("passed"):
        raise SystemExit("Gate 2 or Gate 3 did not pass in the fit stage")
    fit_data = {target: _load_fit_tensor(out / f"fit_{target}.pt") for target in targets}
    active = [layer for layer in LAYERS if any(
        int(fit_data[target]["layer_counts"]["n100"][str(layer)]) > 0 for target in targets)]
    weights = _cpu_weights(layers)
    target_sets = {}
    layer_counts, orthogonality = {}, {}
    for target in targets:
        rotations, _rotate_orth = load_rotations(out, target, draw_seeds, active)
        data = build_target_sets(fit_data[target], weights, rotations, draw_seeds)
        target_sets[str(target)] = data
        layer_counts[str(target)] = data["layer_counts"]
        orthogonality[str(target)] = data["orthogonality"]
        _torch_save(out / f"sets_{target}.pt", data)
    testability = {target: data["testability"] for target, data in target_sets.items()}
    set_json = {"part": 171, "targets": {}, "fit": fit_stats,
        "test_prompt_ids": [str(p["prompt_id"]) for p in tests],
        "testability": testability}
    alignment_rows, debt_rows = [], []
    for target, data in target_sets.items():
        metrics = data["metrics"]
        set_json["targets"][target] = {"metrics": metrics, "layer_counts": data["layer_counts"],
            "orthogonality": data["orthogonality"], "top_contributions": data["top_contributions"],
            "natural_projection": data["natural_projection"], "v_norm": data["v_norm"],
            "target_norm": data["target_norm"], "debt": data["debt"],
            "testability": data["testability"]}
        for arm, meta in metrics.items():
            draw = int(arm.rsplit("_d", 1)[1]) if "_d" in arm else None
            alignment_rows.append({"target": target, "arm": arm, "draw": draw,
                "status": meta.get("status"), "s": meta.get("s"),
                "nominal_norm": meta.get("nominal_norm"), "projection": meta.get("projection"),
                "perpendicular_norm": meta.get("perpendicular_norm"), "cosine": meta.get("cosine"),
                "projection_per_unit_length": meta.get("projection_per_unit_length"),
                "natural_projection": meta.get("natural_projection"),
                "fraction_natural_projection": meta.get("fraction_natural_projection"),
                "layers": meta.get("layer_metrics", {})})
        debt_rows.extend({"target": target, **row} for row in data["debt"])
    debts = [float(row["debt"]) for row in debt_rows if row.get("debt") is not None]
    mean_debt = float(np.mean(debts)) if debts else None
    not_testable = sum(not bool(value) for by_seed in testability.values() for value in by_seed.values())
    testable_count = sum(len(by_seed) for by_seed in testability.values()) - not_testable
    # Gate 4 uses the fixed context; only testable r100p arms are measured.
    arms_by_target = {}
    for target in targets:
        rows = [arm for arm in arm_names(draw_seeds)
                if not arm.startswith("r100p_d") or testability[target].get(arm.rsplit("d", 1)[1], False)]
        arms_by_target[target] = rows
    gate4_rows = measure_gate4(model, tok, layers, target_sets, targets, tests,
        draw_seeds=draw_seeds, selected_arms_by_target=arms_by_target, device=args.device)
    expected_gate4 = sum(len(rows) for rows in arms_by_target.values())
    expected_gate4_keys = {(target, arm) for target, arms in arms_by_target.items() for arm in arms}
    g4 = gate4(gate4_rows, expected_gate4, expected_keys=expected_gate4_keys)
    g5 = gate5(fit_stats, targets)
    g6 = gate6(set_json, targets, tolerance=1e-5, draw_seeds=draw_seeds)
    g7 = gate7(tests, fit_prompts)
    pretest = {"gate4": g4, "gate5": g5, "gate6": g6, "gate7": g7,
        "gate4_rows": gate4_rows, "expected_gate4_arms": expected_gate4,
        "max_rounding_fraction": g4["max_rounding_fraction"],
        "mean_alignment_debt": mean_debt, "testable_r100p_sets": testable_count,
        "not_testable_r100p_sets": not_testable}
    # Fixed-context layer-20 propagated change is a diagnostic, distinct from nominal write length.
    for row in gate4_rows:
        target_sets[row["target"]]["metrics"][row["arm"]]["actual_layer20"] = {
            "length": row.get("actual_layer20_length"), "angle_cosine": row.get("actual_layer20_angle_cosine")}
        for alignment_row in alignment_rows:
            if alignment_row["target"] == row["target"] and alignment_row["arm"] == row["arm"]:
                alignment_row["actual_layer20"] = target_sets[row["target"]]["metrics"][row["arm"]]["actual_layer20"]
                break
    write_json(out / "sets.json", set_json)
    debt_means = {target: (float(np.mean([x["debt"] for x in data["debt"] if x.get("debt") is not None]))
                           if any(x.get("debt") is not None for x in data["debt"]) else None)
                  for target, data in target_sets.items()}
    write_json(out / "alignment.json", {"part": 171, "fixed_context": str(tests[0]["question"]),
        "rows": alignment_rows,
        "debts": debt_rows, "mean_alignment_debt": mean_debt,
        "mean_debt_by_target": debt_means,
        "not_testable_r100p_sets": not_testable, "testable_r100p_sets": testable_count})
    write_json(out / "pretest_gates.json", pretest)
    if not all(g["passed"] for g in (g4, g5, g6, g7)):
        raise SystemExit(f"Gate 4, 5, 6 or 7 failed before test generations: {pretest}")

    pilot_path = out / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else {
        "batches": 0, "answers": 0, "elapsed_seconds": 0.0}
    widths = {int(i): int(layers[i].mlp.down_proj.weight.shape[1]) for i in LAYERS}
    for target in targets:
        for arm in arms_by_target[target]:
            if arm == "none":
                continue
            draw = int(arm.rsplit("_d", 1)[1]) if "_d" in arm else None
            partial = {"stage": "test", "target": target, "arm": arm}
            if draw is not None:
                partial["draw"] = draw
            keys = [answer_key({**partial, "prompt_id": p["prompt_id"]}) for p in tests]
            if all(key in done for key in keys):
                continue
            pushes, vector = materialize_arm(target_sets[target], arm, widths, args.device)
            started = time.perf_counter()
            generated = generate_batch(model, tok, layers, [p["question"] for p in tests], [None] * len(tests),
                args.device, args.max_new_tokens, pushes=pushes, vector=vector, seed=171)
            elapsed = time.perf_counter() - started
            if len(generated) != len(tests):
                raise RuntimeError(f"{arm} generated {len(generated)} answers for {len(tests)} prompts")
            records = [_answer_record("test", target, arm, p, a, draw)
                       for p, a in zip(tests, generated)]
            append_answers(answer_path, records, done)
            pilot = _pilot_update(pilot_path, pilot, len(tests), elapsed)
    answers = read_jsonl(answer_path)
    expected = planned_answer_keys(targets, tests, fit_prompts, draw_seeds, testability)
    g1 = key_coverage(answers, expected_keys=expected)
    gates = {"gate1": g1, "gate2": stage0["gate2"], "gate3": stage0["gate3"],
        "gate4": g4, "gate5": g5, "gate6": g6, "gate7": g7,
        "passed": all(g["passed"] for g in (g1, stage0["gate2"], stage0["gate3"], g4, g5, g6, g7))}
    write_json(out / "gates.json", gates)
    return {"stage": "test", "answers": len(answers), "planned_answers": len(expected),
        "testable_r100p_sets": testable_count, "not_testable_r100p_sets": not_testable,
        "gates": gates, "pilot": pilot}


def run(args, loader=None, config=None):
    """Run one registered stage; tests may inject a tiny fake model and catalog."""
    config = config or {}
    targets = config.get("targets", TARGETS)
    fit_prompts = tuple(config.get("fit_prompts", FIT_PROMPTS))
    tests = list(config.get("test_prompts", test_catalog()))
    draw_seeds = tuple(config.get("draw_seeds", DRAW_SEEDS))
    k_values = config.get("k_values", K_VALUES)
    strict = bool(config.get("strict_model", loader is None))
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
        model.requires_grad_(False); model.eval()
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False); model.eval()
    layers = common86.find_layers(model)
    if strict:
        validate_model(layers)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit":
        return _fit_stage(args, tok, model, layers, out, targets, fit_prompts, tests, draw_seeds, k_values)
    if args.stage == "test":
        return _test_stage(args, tok, model, layers, out, targets, fit_prompts, tests, draw_seeds)
    raise ValueError(f"unknown Part 171 stage {args.stage!r}")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("fit", "test"), default="fit")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out-dir", default="results/part171")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 24 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

