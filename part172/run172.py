#!/usr/bin/env python3
"""Part 172 runner, stage `test` only: dose ladder, knockout, writer-off, frames, replay and the runner-up.

Everything that made Part 171's pushes is imported from run171, never copied: generate_batch, encode_batch, the two
hooks and hit_match. The pushes are Part 171's saved sets (results/part171/sets_google.pt); nothing is refit here. A
dose multiplies one whole set's push. The next-ranked singles and the graded knockout are the only new writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
from prompts171 import DRAW_SEEDS, TARGETS
from prompts172 import FRAMES, NEW44

TARGET = "google"
MODEL_DEFAULT = run171.MODEL_DEFAULT
MAX_NEW_TOKENS = run171.MAX_NEW_TOKENS
BATCH_SIZE = run171.BATCH_SIZE
SETS_FILE = "sets_google.pt"
PART171_DIR = Path("results/part171")
PART171_SETS = PART171_DIR / SETS_FILE
PART171_ANSWERS = PART171_DIR / "answers.jsonl"
PILOT_BATCHES = 3
PILOT_THRESHOLD = run171.PILOT_THRESHOLD
GATE3_PROMPTS = 8
GATE3_UNITS = ((6, 753), (7, 1169))
GATE4_ROW_MIN = 43
GATE2_QWEN_MIN = 37
GATE2_HIT_MAX = 1
GATE5_ROLE_MIN = 38
GATE5_ZERO4_MIN = 3
GATE6_TOLERANCE = 0.05
EXPECTED_ARM_COUNT = 54

# Part 66's four maker neurons, the knockout arm's own schedule (activation x (1 - d) at every position).
MAKERS = ((6, 753), (7, 1169), (11, 8868), (29, 776))
WRITER = ((27, 3325),)
# The next-ranked singles, by Part 171's |c|, from PART172-DESIGN.md.
NEXT_SINGLES = (("n2", 18, 3738), ("n3", 20, 173), ("n4", 20, 6361), ("n5", 8, 5009))
LADDER = ("n1", "rand100_d171", "r100_d2171")
LADDER_DOSES = (0.25, 0.5, 0.75, 1.0, 1.5)
KNOCKOUT_DOSES = (0.5, 0.75, 1.0)
NEGATIVE_DOSES = (-0.5, -1.0)
SINGLE_DOSES = (0.5, 1.0)
CONTROL_SETS = tuple(f"{base}_d{seed}" for seed in DRAW_SEEDS for base in ("r1", "r10", "r100", "rand100"))
WRITER_OFF_ARMS = ("n1", "rand100_d171", "r100_d2171")
FRAME_ARMS = (("frame_n1", "frame_n1", False), ("frame_plain", "frame_plain", False),
              ("frame_role", "frame_role", False), ("frame_n1_w", "frame_n1", True))
REPLAY_ARMS = LADDER
QWEN_FAMILY = ("qwen", "tongyi", "alibaba", "通义")
# The runner-up reads the first named member of this family and removes the whole family from the next-token read.
RUNNER_UP_FAMILY = ("qwen", "tongyi", "alibaba", "通义", "dashscope")
GOOGLE_FAMILY = ("google", "gemini", "deepmind", "谷歌")


# ---------------------------------------------------------------- answer keys and files

def answer_key(row):
    return (str(row.get("arm")), str(row.get("prompt_id")))


def read_jsonl(path):
    return run171.read_jsonl(path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path, value):
    run171.write_json(path, value)


def repair_tail(path):
    run171.repair_tail(path)


def append_answers(path, rows, done):
    """Write one row per finished answer, flush per row, and skip every key already on disk."""
    added = []
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            key = answer_key(row)
            if key in done:
                continue
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            stream.flush()
            done.add(key)
            added.append(row)
    return added


def new_prompts():
    return [{"prompt_id": f"new44:{index}", "question_index": index, "question": question}
            for index, question in enumerate(NEW44)]


def part171_prompts():
    return run171.test_catalog()


# ---------------------------------------------------------------- arms and the planned set

def arm_specs():
    """Every planned arm on NEW44, in the design's order. Each spec is one generation batch of 44 prompts."""
    specs = [{"arm": "none"}, {"arm": "role", "role": True}]
    specs += [{"arm": f"ko{d:g}", "knockout": d} for d in KNOCKOUT_DOSES]
    for name in LADDER:
        for dose in LADDER_DOSES:
            # The d=1 control rows are the ladder observations for these two sets.
            arm = name if name in CONTROL_SETS and dose == 1.0 else f"{name}_d{dose:g}"
            specs.append({"arm": arm, "set": name, "dose": dose})
    shared = {name for name in LADDER[1:]}
    specs += [{"arm": name, "set": name, "dose": 1.0} for name in CONTROL_SETS if name not in shared]
    specs += [{"arm": f"n1_d{d:g}", "set": "n1", "dose": d} for d in NEGATIVE_DOSES]
    for short, layer, unit in NEXT_SINGLES:
        specs += [{"arm": f"{short}_d{d:g}", "single": (layer, unit), "dose": d} for d in SINGLE_DOSES]
    specs += [{"arm": f"{name}_w", "set": name, "dose": 1.0, "writer_off": True} for name in WRITER_OFF_ARMS]
    specs += [{"arm": "none_w", "writer_off": True}, {"arm": "role_w", "role": True, "writer_off": True},
              {"arm": "zero4_w", "knockout": 1.0, "writer_off": True}]
    specs += [{"arm": f"n1_d{dose:g}_w", "set": "n1", "dose": dose, "writer_off": True}
              for dose in (0.5, 0.75)]
    specs += [{"arm": arm, "frame": frame, "writer_off": writer} for arm, frame, writer in FRAME_ARMS]
    specs += [{"arm": f"n1_{arm}", "set": "n1", "dose": 1.0, "frame": frame, "writer_off": False}
              for arm, frame in (("frame_n1", "frame_n1"), ("frame_plain", "frame_plain"))]
    return specs


def ladder_arm(name, dose):
    """Canonical saved row for a ladder point, sharing d=1 controls without a second generation."""
    dose = float(dose)
    if name in CONTROL_SETS and dose == 1.0:
        return name
    return f"{name}_d{dose:g}"


def planned_keys(specs=None):
    specs = arm_specs() if specs is None else specs
    return {(spec["arm"], p["prompt_id"]) for spec in specs for p in new_prompts()}


def registered_arm_names():
    """Independent arm-name contract from the registered design, including the two shared d=1 controls."""
    names = {"none", "role", *(f"ko{dose:g}" for dose in KNOCKOUT_DOSES)}
    names.update(ladder_arm(name, dose) for name in LADDER for dose in LADDER_DOSES)
    names.update(name for name in CONTROL_SETS if name not in set(LADDER[1:]))
    names.update(f"n1_d{dose:g}" for dose in NEGATIVE_DOSES)
    names.update(f"{name}_d{dose:g}" for name, _layer, _unit in NEXT_SINGLES for dose in SINGLE_DOSES)
    names.update(f"{name}_w" for name in WRITER_OFF_ARMS)
    names.update({"none_w", "role_w", "zero4_w"})
    names.update(f"n1_d{dose:g}_w" for dose in (0.5, 0.75))
    names.update(arm for arm, _frame, _writer in FRAME_ARMS)
    names.update(f"n1_{arm}" for arm in ("frame_n1", "frame_plain"))
    return names


def gate1_plan(specs=None, prompts=None):
    specs = arm_specs() if specs is None else list(specs)
    prompts = new_prompts() if prompts is None else list(prompts)
    names = [str(spec.get("arm")) for spec in specs]
    expected = len(specs) * len(prompts)
    keys = {(str(spec.get("arm")), str(prompt["prompt_id"])) for spec in specs for prompt in prompts}
    design_names = registered_arm_names()
    missing_arms, unexpected_arms = sorted(design_names - set(names)), sorted(set(names) - design_names)
    passed = (len(names) == EXPECTED_ARM_COUNT and len(prompts) == 44 and
              len(names) == len(set(names)) and set(names) == design_names and len(keys) == expected)
    return {"passed": bool(passed),
            "specs": len(specs), "unique_arms": len(set(names)), "planned": expected,
            "prompt_count": len(prompts), "expected_arm_count": EXPECTED_ARM_COUNT,
            "missing_arms": missing_arms, "unexpected_arms": unexpected_arms,
            "duplicate_arms": sorted({name for name in names if names.count(name) > 1})}


def replay_keys():
    return {(f"replay:{arm}", p["prompt_id"]) for arm in REPLAY_ARMS for p in part171_prompts()}


# ---------------------------------------------------------------- pushes and hooks

def load_sets(path):
    import torch
    import pickle
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older torch has no weights_only argument
        obj = torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:  # trusted Part 171 artifact may contain numpy metadata
        obj = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("arms", "metrics", "v", "target_norm", "v_norm", "delta", "p"):
        if key not in obj:
            raise ValueError(f"{path} is missing {key!r}; it is not a Part 171 sets tensor")
    return obj


def set_pushes(sets, arm, dose, widths, device):
    """A dose multiplies one whole set's saved push. Arms the sets do not hold are an error, not a zero push."""
    import numpy as np
    import torch
    if arm not in sets["arms"]:
        raise KeyError(f"Part 171 sets hold no arm {arm!r}")
    reps = sets["arms"][arm]
    if not reps:
        raise ValueError(f"Part 171 arm {arm!r} has an empty push")
    expected_layers = {int(layer_no) for layer_no in widths}
    represented_layers = {int(layer_no) for layer_no in reps}
    missing_layers = expected_layers - represented_layers
    if missing_layers:
        raise ValueError(f"Part 171 arm {arm!r} push is missing layers {sorted(missing_layers)}")
    result = {}
    for layer_no, width in widths.items():
        rep = reps.get(int(layer_no), reps.get(str(layer_no)))
        if not rep:
            raise ValueError(f"Part 171 arm {arm!r} has an empty representation at layer {layer_no}")
        if rep["kind"] == "dense":
            value = np.asarray(rep["values"], dtype=np.float32)
        elif rep["kind"] == "sparse":
            value = np.zeros(int(width), dtype=np.float32)
            if rep["coords"]:
                value[np.asarray(rep["coords"], dtype=np.int64)] = np.asarray(rep["values"], dtype=np.float32)
        else:
            value = np.asarray(rep["coefficients"], dtype=np.float32) @ np.asarray(rep["directions"], dtype=np.float32)
        if value.shape[-1] != int(width):
            raise ValueError(f"{arm}/L{layer_no} push width {value.shape[-1]} != {width}")
        result[int(layer_no)] = torch.as_tensor(np.float64(dose) * value.astype(np.float64),
                                                dtype=torch.float32, device=device)
    if not result:
        raise ValueError(f"Part 171 arm {arm!r} produced an empty push for the expected layer widths")
    return result


def single_push(sets, layer, unit, widths, device, dose=1.0):
    """One neuron moved along its signed delta; scale it alone to 4|v|, then apply its signed dose."""
    import numpy as np
    import torch
    width = int(widths[int(layer)])
    delta = np.asarray(sets["delta"][int(layer)], dtype=np.float64)
    if delta.shape != (width,):
        raise ValueError(f"delta layer {layer} has shape {delta.shape}, expected {(width,)}")
    if "unit_weight_columns" in sets:
        column = np.asarray(sets["unit_weight_columns"][int(layer)][int(unit)], dtype=np.float64)
    else:
        weight = np.asarray(sets["weights"][int(layer)], dtype=np.float64)
        column = weight[:, int(unit)]
    write = column * delta[int(unit)]
    norm = float(np.linalg.norm(write))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError(f"neuron L{layer} N{unit} has a zero or non-finite write")
    scale = 4.0 * float(sets["v_norm"]) / norm
    values = np.zeros(width, dtype=np.float64)
    values[int(unit)] = float(dose) * scale * delta[int(unit)]
    return {int(layer): torch.as_tensor(values, dtype=torch.float32, device=device)}, {
        "layer": int(layer), "unit": int(unit), "write_norm": norm, "target_norm": 4.0 * float(sets["v_norm"]),
        "unit_dose_scale": scale, "dose": float(dose), "scale": float(dose) * scale}


class ScaleUnits:
    """Forward pre-hook on mlp.down_proj: multiply the listed neurons by a factor at every position.

    probe172.ZeroUnits with a factor. The factor is a Python float and the index is a list, so nothing runs .item()
    inside the hook; the whole tensor is cloned once per call and the unit columns are scaled in place.
    """

    def __init__(self, module, units, factor):
        self.raw = [int(unit) for unit in units]
        self.factor = float(factor)
        self.handle = module.register_forward_pre_hook(self)

    def __call__(self, _module, args):
        x = args[0].clone()
        x[..., self.raw] = x[..., self.raw] * self.factor
        return (x,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def knockout_factor(dose):
    return 1.0 - float(dose)


def _by_layer(units):
    result = {}
    for layer, unit in units:
        result.setdefault(int(layer), []).append(int(unit))
    return sorted(result.items())


def knockout_hooks(layers, units, factor):
    return [ScaleUnits(layers[layer].mlp.down_proj, kept, factor) for layer, kept in _by_layer(units)]


def widths_of(layers):
    return {int(layer): int(layers[layer].mlp.down_proj.weight.shape[1]) for layer in run171.LAYERS}


def single_weight_columns(layers):
    """Copy only the selected neuron columns needed for the four single-neuron scalings."""
    import numpy as np
    import torch
    columns = {}
    for _name, layer, unit in NEXT_SINGLES:
        columns.setdefault(int(layer), {})
        if int(unit) in columns[int(layer)]:
            continue
        columns[int(layer)][int(unit)] = layers[int(layer)].mlp.down_proj.weight[:, int(unit)].detach().to(
            device="cpu", dtype=torch.float64).numpy().copy()
    return columns


# ---------------------------------------------------------------- generation

def forced_prompt(tok, question, phrase):
    """The prompt plus the phrase's own token ids, so the reply opens with them."""
    rendered = run171.render_prompt(tok, question)
    ids = single_token_ids(tok, rendered)
    phrase_ids = single_token_ids(tok, phrase)
    if not phrase_ids:
        raise ValueError(f"frame phrase {phrase!r} encodes to no tokens")
    return {"input_ids": ids + phrase_ids, "attention_mask": [1] * (len(ids) + len(phrase_ids)),
            "forced": phrase_ids}


def single_token_ids(tok, text):
    """Read token ids from either a flat list or a one-row tensor BatchEncoding."""
    values = tok(text, add_special_tokens=False)["input_ids"]
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], (list, tuple)):
        if len(values) != 1:
            raise ValueError(f"expected one tokenized row, got {len(values)}")
        values = values[0]
    return [int(value) for value in values]


def generate_forced(model, tok, layers, questions, phrase, device, max_new_tokens, pushes=None):
    """Force the opening, then generate the full post-phrase token budget with hooks still installed."""
    import torch
    encoded = [forced_prompt(tok, question, phrase) for question in questions]
    width = max(len(row["input_ids"]) for row in encoded)
    pad_id = int(getattr(tok, "pad_token_id", 0) or 0)
    input_rows = [[pad_id] * (width - len(row["input_ids"])) + row["input_ids"] for row in encoded]
    masks = [[0] * (width - len(row["input_ids"])) + row["attention_mask"] for row in encoded]
    inputs = {"input_ids": torch.tensor(input_rows, dtype=torch.long, device=device),
              "attention_mask": torch.tensor(masks, dtype=torch.long, device=device)}
    continuation_limit = max(0, int(max_new_tokens))
    handles = []
    for layer_no, push in sorted((pushes or {}).items()):
        handles.append(run171.AddEveryPositionHook(layers[int(layer_no)].mlp.down_proj, push))
    eos = set(map(int, common86.eos_ids(tok, model)))
    try:
        with torch.inference_mode(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(172)
            if continuation_limit:
                generated = model.generate(**inputs, max_new_tokens=continuation_limit, do_sample=False,
                    use_cache=True, pad_token_id=getattr(tok, "pad_token_id", None),
                    eos_token_id=sorted(eos) or None)
            else:
                generated = inputs["input_ids"]
    finally:
        for hook in handles:
            hook.remove()
    outputs = []
    for row_no, seq in enumerate(generated[:, width:].detach().cpu().tolist()):
        continuation = list(map(int, seq))
        for index, token in enumerate(continuation):
            if token in eos:
                continuation = continuation[:index + 1]
                break
        forced = encoded[row_no]["forced"]
        ids = forced + continuation
        continuation40 = continuation[:40]
        continuation12 = continuation[:12]
        outputs.append({"answer": tok.decode(ids, skip_special_tokens=True).strip(),
                        "reader_answer": tok.decode(continuation, skip_special_tokens=True).strip(),
                        "first40": tok.decode(continuation40, skip_special_tokens=True).strip(),
                        "first12": tok.decode(continuation12, skip_special_tokens=True).strip(),
                        "score_text": tok.decode(continuation40, skip_special_tokens=True).strip(),
                        "token_ids": ids, "token_count": len(ids), "forced_ids": forced,
                        "opens_with_frame": ids[:len(forced)] == forced,
                        "prompt_ids": encoded[row_no]["input_ids"]})
    return outputs


def frame_hit(row):
    """Read the design's first 40 generated tokens; a forced opening is outside this text."""
    return bool(run171.hit_match(row.get("first40", ""), TARGET))


def run_spec(model, tok, layers, spec, questions, device, max_new_tokens, sets, widths, seed=172):
    """Generate one arm's batch: pushes, a knockout hook, a frame, or none of them."""
    before = hook_counts(layers)
    pushes = None
    if spec.get("set"):
        pushes = set_pushes(sets, spec["set"], float(spec.get("dose", 1.0)), widths, device)
    single = {}
    if spec.get("single"):
        layer, unit = spec["single"]
        pushes, single = single_push(sets, layer, unit, widths, device, float(spec.get("dose", 1.0)))
    hooks = []
    if spec.get("units"):
        hooks = knockout_hooks(layers, spec["units"], 0.0)
    if spec.get("knockout") is not None:
        hooks += knockout_hooks(layers, MAKERS, knockout_factor(spec["knockout"]))
    if spec.get("writer_off"):
        hooks += knockout_hooks(layers, WRITER, 0.0)
    roles = [TARGETS[TARGET]] * len(questions) if spec.get("role") else None
    try:
        if spec.get("frame"):
            generated = generate_forced(model, tok, layers, questions, FRAMES[spec["frame"]], device,
                                        max_new_tokens, pushes=pushes)
        else:
            generated = run171.generate_batch(model, tok, layers, questions, roles, device,
                                              max_new_tokens, pushes=pushes, seed=seed)
    finally:
        for hook in hooks:
            hook.remove()
        after = hook_counts(layers)
        if after != before:
            raise RuntimeError(f"forward pre-hooks leaked after {spec.get('arm')}: {before} -> {after}")
    for row in generated:
        if not spec.get("frame"):
            row.setdefault("reader_answer", row.get("answer", ""))
            row.setdefault("score_text", row.get("first40", ""))
            row.setdefault("first12", tok.decode(row.get("token_ids", [])[:12], skip_special_tokens=True).strip())
        row["single_scale"] = single or None
    return generated


def hook_counts(layers):
    return {int(layer): len(layers[int(layer)].mlp.down_proj._forward_pre_hooks) for layer in run171.LAYERS}


# ---------------------------------------------------------------- gates

def gate1(answers, expected):
    keys = [answer_key(row) for row in answers]
    try:
        finite = all(isinstance(row.get("answer"), str) and bool(row["answer"].strip()) and
                     isinstance(row.get("token_ids"), list) and bool(row["token_ids"]) and
                     all(int(x) >= 0 for x in row["token_ids"]) for row in answers)
    except (TypeError, ValueError):
        finite = False
    return {"passed": bool(len(keys) == len(expected) and len(set(keys)) == len(keys) and
                           set(keys) == set(expected) and finite),
            "planned": len(expected), "answers": len(keys), "unique_keys": len(set(keys)),
            "missing": len(set(expected) - set(keys)), "unexpected": len(set(keys) - set(expected)),
            "finite_answers": bool(finite)}


def gate2(answers):
    rows = [row for row in answers if row.get("arm") == "none"]
    by_prompt = {str(row["prompt_id"]): row for row in rows}
    prompts = new_prompts()
    qwen_hits = sum(run171._word_match(by_prompt.get(p["prompt_id"], {}).get("first40", ""), QWEN_FAMILY)
                    for p in prompts)
    hit_rows = sum(run171.hit_match(by_prompt.get(p["prompt_id"], {}).get("first40", ""), TARGET)
                   for p in prompts)
    complete = len(rows) == len(prompts) == 44 and len(by_prompt) == len(rows)
    return {"passed": bool(complete and qwen_hits >= GATE2_QWEN_MIN and hit_rows <= GATE2_HIT_MAX),
            "qwen_prompts": qwen_hits, "qwen_denominator": len(prompts),
            "hit_rows": hit_rows, "hit_denominator": len(prompts), "complete": bool(complete)}


def gate3(plain, zero_push, identity_units, post_push_plain=None, expected=GATE3_PROMPTS,
          hook_counts_restored=False, identity_hook_registered=False, pushed_arm_checks=None):
    def ids(rows):
        return {str(row["prompt_id"]): list(map(int, row["token_ids"])) for row in rows}
    p, zp, iu = ids(plain), ids(zero_push), ids(identity_units)
    after = ids(post_push_plain or [])
    same = (len(p) == expected and len(zp) == expected and len(iu) == expected and len(after) == expected and
            p.keys() == zp.keys() == iu.keys() == after.keys() and
            all(p[k] == zp[k] == iu[k] == after[k] for k in p))
    pushed_count = len(pushed_arm_checks or [])
    passed = same and hook_counts_restored and identity_hook_registered and pushed_count == expected
    return {"passed": bool(passed), "prompts": len(p), "zero_push_checks": len(zp),
            "identity_unit_checks": len(iu), "post_push_plain_checks": len(after),
            "matching_zero_push": sum(p.get(k) == zp.get(k) for k in p.keys() | zp.keys()),
            "matching_identity_units": sum(p.get(k) == iu.get(k) for k in p.keys() | iu.keys()),
            "matching_post_push_plain": sum(p.get(k) == after.get(k) for k in p.keys() | after.keys()),
            "hook_counts_restored": bool(hook_counts_restored),
            "identity_hook_registered": bool(identity_hook_registered),
            "pushed_arm_checks": pushed_count}


def gate4(replay_rows, reference, minimum=GATE4_ROW_MIN):
    """Replay: each d = 1 arm on Part 171's 44 prompts must match its stored token_ids on at least 43."""
    stored = {(str(row.get("arm")), str(row.get("prompt_id"))): row for row in reference
              if row.get("stage") == "test"}
    expected_ids = {arm: {prompt_id for saved_arm, prompt_id in stored if saved_arm == arm}
                    for arm in REPLAY_ARMS}
    rows = list(replay_rows)
    per_arm = {}
    unexpected_arms = []
    for row in rows:
        if not str(row.get("arm", "")).startswith("replay:"):
            continue
        arm = str(row["arm"]).split(":", 1)[1]
        if arm not in REPLAY_ARMS:
            unexpected_arms.append(arm)
            continue
        ref = stored.get((arm, str(row.get("prompt_id"))))
        match = ref is not None and list(map(int, ref.get("token_ids", []))) == list(map(int, row.get("token_ids", [])))
        entry = per_arm.setdefault(arm, {"rows": 0, "matching": 0, "prompt_ids": []})
        entry["rows"] += 1
        entry["matching"] += int(match)
        entry["prompt_ids"].append(str(row.get("prompt_id")))
    complete = True
    for arm in REPLAY_ARMS:
        entry = per_arm.setdefault(arm, {"rows": 0, "matching": 0, "prompt_ids": []})
        actual_ids = set(entry.pop("prompt_ids"))
        expected = expected_ids[arm]
        entry.update({"unique_prompts": len(actual_ids), "missing_prompts": len(expected - actual_ids),
                      "unexpected_prompts": len(actual_ids - expected),
                      "duplicate_rows": entry["rows"] - len(actual_ids)})
        complete = complete and len(expected) == 44 and entry["rows"] == 44 and len(actual_ids) == 44 and \
            actual_ids == expected and entry["matching"] >= int(minimum)
    passed = complete and not unexpected_arms
    return {"passed": passed, "minimum_matching": int(minimum), "per_arm": per_arm,
            "rows": len(rows), "expected_arms": list(REPLAY_ARMS), "unexpected_arms": unexpected_arms,
            "artifact_hashes": "recorded_not_compared",
            "artifact_hashes_note": "Recorded in the gate output; not compared by Gate 4."}


def gate5(answers):
    by_arm = {}
    for row in answers:
        by_arm.setdefault(str(row.get("arm")), []).append(row)
    role = sum(run171.hit_match(row.get("first40", ""), TARGET) for row in by_arm.get("role", []))
    zero4 = sum(run171.hit_match(row.get("first40", ""), TARGET) for row in by_arm.get("ko1", []))
    complete = len(by_arm.get("role", [])) == 44 and len(by_arm.get("ko1", [])) == 44
    return {"passed": bool(complete and role >= GATE5_ROLE_MIN and zero4 >= GATE5_ZERO4_MIN),
            "role_hits": role, "role_denominator": len(by_arm.get("role", [])),
            "zero4_hits": zero4, "zero4_denominator": len(by_arm.get("ko1", [])),
            "role_min": GATE5_ROLE_MIN, "zero4_min": GATE5_ZERO4_MIN, "complete": bool(complete)}


def gate6(rows, tolerance=GATE6_TOLERANCE):
    """Check delivery coverage, signed dose, absolute write size, writer-off behavior and omitted-dose sensitivity."""
    failures = []
    delivery_rows = [row for row in rows if row.get("kind") == "delivery"]
    expected = {(ladder_arm(name, dose), float(dose)) for name in LADDER for dose in LADDER_DOSES}
    expected |= {(f"n1_d{dose:g}", float(dose)) for dose in NEGATIVE_DOSES}
    expected |= {(f"{short}_d{dose:g}", float(dose)) for short, _layer, _unit in NEXT_SINGLES
                 for dose in SINGLE_DOSES}
    expected |= {(arm, 1.0) for arm in CONTROL_SETS}
    observed = set()
    for row in delivery_rows:
        dose = float(row["dose"])
        observed.add((str(row.get("arm")), dose))
        slope = row.get("delivery_slope")
        if not _slope_matches_dose(slope, dose, tolerance):
            failures.append({"kind": "slope", "arm": row.get("arm"), "dose": dose, "slope": slope})
        if dose == 1.0:
            unit_norm = row.get("unit_write_norm")
            realized_norm = row.get("realized_write_norm")
            intended_norm = row.get("intended_write_norm")
            if not _relative_matches(unit_norm, intended_norm, tolerance):
                failures.append({"kind": "unit_write_size", "arm": row.get("arm"),
                                 "unit_write_norm": unit_norm, "intended_write_norm": intended_norm})
            if not _squared_norms_match(realized_norm, unit_norm, tolerance):
                failures.append({"kind": "delivered_unit_write_size", "arm": row.get("arm"),
                                 "realized_write_norm": realized_norm, "unit_write_norm": unit_norm})
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        failures.append({"kind": "delivery_coverage", "missing": sorted(missing), "unexpected": sorted(extra)})
    omission = [row for row in rows if row.get("kind") == "omitted_dose_probe"]
    probe_slope = omission[0].get("delivery_slope") if len(omission) == 1 else None
    try:
        probe_is_finite = math.isfinite(float(probe_slope))
    except (TypeError, ValueError):
        probe_is_finite = False
    omission_detected = (len(omission) == 1 and omission[0].get("dose") == 0.5 and probe_is_finite and
                         not _slope_matches_dose(probe_slope, 0.5, tolerance))
    if not omission_detected:
        failures.append({"kind": "omitted_dose_probe_not_caught", "rows": omission})
    zero = [row for row in rows if row.get("kind") == "writer_zero"]
    plain = [row for row in rows if row.get("kind") == "writer_plain"]

    def activation(row):
        try:
            return float(row.get("writer_activation"))
        except (TypeError, ValueError):
            return None

    if not zero or any(activation(row) != 0.0 for row in zero):
        failures.append({"kind": "writer_zero", "rows": zero})
    if not plain or any(value is None or not math.isfinite(value) or value == 0.0
                         for value in (activation(row) for row in plain)):
        failures.append({"kind": "writer_plain", "rows": plain})
    return {"passed": not failures and bool(rows), "failures": failures, "rows": len(rows),
            "delivery_rows": len(delivery_rows), "omitted_dose_caught": bool(omission_detected),
            "tolerance": float(tolerance)}


def _slope_matches_dose(slope, dose, tolerance):
    try:
        return math.isfinite(float(slope)) and abs(float(slope) - float(dose)) <= tolerance * abs(float(dose))
    except (TypeError, ValueError):
        return False


def _relative_matches(value, expected, tolerance):
    try:
        value, expected = float(value), float(expected)
        return math.isfinite(value) and math.isfinite(expected) and expected > 0 and \
            abs(value / expected - 1.0) <= tolerance
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def _squared_norms_match(actual_norm, reference_norm, tolerance):
    try:
        actual_norm, reference_norm = float(actual_norm), float(reference_norm)
        return math.isfinite(actual_norm) and math.isfinite(reference_norm) and reference_norm > 0 and \
            abs((actual_norm * actual_norm) / (reference_norm * reference_norm) - 1.0) <= tolerance
    except (TypeError, ValueError, ZeroDivisionError):
        return False


def gate7(prompts=None, others=None):
    """NEW44 disjoint from P8 and M96, case-folded."""
    import prompts171
    from part170.prompts170 import M96
    prompts = list(NEW44 if prompts is None else prompts)
    others = list(prompts171.FIT_PROMPTS) + list(M96) if others is None else list(others)
    fold = lambda text: " ".join(str(text).casefold().split())
    known = {fold(text) for text in others}
    overlap = [text for text in prompts if fold(text) in known]
    return {"passed": not overlap and len(prompts) == 44, "overlap": overlap, "prompts": len(prompts),
            "compared_against": {"p8": len(prompts171.FIT_PROMPTS), "m96": len(M96)}}


def _json_hash(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def artifact_hashes(sets_path, tok, model, model_id):
    """Record stable digests for the pushes, hook implementation, tokenizer inputs and pinned model identity."""
    if isinstance(sets_path, dict):
        sets_hash = _json_hash(sets_path)
        sets_source = "in-memory test fixture"
    else:
        path = Path(sets_path)
        sets_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        sets_source = str(path)
    hook_path = Path(run171.__file__).resolve()
    hook_hash = hashlib.sha256(hook_path.read_bytes()).hexdigest() if hook_path.is_file() else None
    tok_path = Path(str(getattr(tok, "name_or_path", "")))
    tok_config = {"init_kwargs": getattr(tok, "init_kwargs", {}),
                  "special_tokens_map": getattr(tok, "special_tokens_map", {}),
                  "model_max_length": getattr(tok, "model_max_length", None)}
    config_files = {}
    if tok_path.is_dir():
        for name in ("tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            candidate = tok_path / name
            if candidate.is_file():
                config_files[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    model_config = getattr(model, "config", None)
    if hasattr(model_config, "to_dict"):
        model_config = model_config.to_dict()
    elif hasattr(model_config, "__dict__"):
        model_config = vars(model_config)
    else:
        model_config = str(model_config)
    chat_template = getattr(tok, "chat_template", None)
    snapshot_id = str(model_id)
    return {"pushes": {"source": sets_source, "sha256": sets_hash},
            "hook_source": {"source": str(hook_path), "sha256": hook_hash},
            "tokenizer_config": {"sha256": _json_hash({"config": tok_config, "files": config_files})},
            "chat_template": {"sha256": _json_hash(chat_template)},
            "model_snapshot": {"id": snapshot_id,
                               "sha256": _json_hash({"snapshot_id": snapshot_id, "config": model_config})}}


# ---------------------------------------------------------------- gate 6 and 3 measurements

class _WriterReader:
    """Read the L27 N3325 input to down_proj at the last position, after any pre-hook has run."""

    def __init__(self, module, unit):
        self.unit = int(unit)
        self.value = None
        self.handle = module.register_forward_pre_hook(self)

    def __call__(self, _module, args):
        self.value = args[0][:, -1, self.unit].detach().float().cpu().numpy().copy()
        return None

    def remove(self):
        self.handle.remove()


def measure_gate6(model, tok, layers, sets, device, widths):
    """Measure real hooked writes at one context against each arm's independently measured d=1 write."""
    import numpy as np
    import torch
    context = NEW44[0]
    inputs = run171.encode_batch(tok, [context], device, [None])

    def observed(pushes):
        traces = []
        for layer_no, push in sorted(pushes.items()):
            if bool(torch.count_nonzero(push)):
                traces.append(run171._TraceDownWrite(layers[int(layer_no)].mlp.down_proj, push))
        try:
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            for trace in traces:
                trace.remove()
        if not traces or any(trace.actual is None or trace.expected is None for trace in traces):
            return None, None
        # Trace vectors are output-side writes, so their width is hidden size, not down_proj input width.
        return _sum_trace_writes(traces)

    def slope(actual, reference):
        denominator = float(reference @ reference)
        return float(actual @ reference / denominator) if denominator else None

    groups = {}
    for name in LADDER:
        groups[name] = [(name, dose, set_pushes(sets, name, dose, widths, device), ladder_arm(name, dose))
                        for dose in LADDER_DOSES]
    for name in CONTROL_SETS:
        if name not in groups:
            groups[name] = [(name, 1.0, set_pushes(sets, name, 1.0, widths, device), name)]
    groups["n1"].extend(("n1", dose, set_pushes(sets, "n1", dose, widths, device), f"n1_d{dose:g}")
                        for dose in NEGATIVE_DOSES)
    for short, layer, unit in NEXT_SINGLES:
        groups[short] = [(short, dose, single_push(sets, layer, unit, widths, device, dose)[0],
                          f"{short}_d{dose:g}") for dose in SINGLE_DOSES]

    rows = []
    unit_references = {}
    intended_norm = 4.0 * float(sets["v_norm"])
    for group, candidates in groups.items():
        ordered = sorted(candidates, key=lambda item: item[1] != 1.0)
        for source_name, dose, pushes, arm_name in ordered:
            actual, intended = observed(pushes)
            if dose == 1.0 and intended is not None:
                unit_references[group] = intended
            reference = unit_references.get(group)
            rows.append({"kind": "delivery", "arm": arm_name, "dose": float(dose),
                         "unit_write_norm": float(np.linalg.norm(reference)) if reference is not None else None,
                         "realized_write_norm": float(np.linalg.norm(actual)) if actual is not None else None,
                         "intended_write_norm": intended_norm,
                         "delivery_slope": slope(actual, reference) if actual is not None and reference is not None else None})

    # Deliberately omit d=.5 once, through the real hook; the d=1 reference must make this fail the dose check.
    omission_actual, _omission_expected = observed(set_pushes(sets, "n1", 1.0, widths, device))
    n1_reference = unit_references.get("n1")
    rows.append({"kind": "omitted_dose_probe", "arm": "n1_d0.5", "dose": 0.5,
                 "delivery_slope": slope(omission_actual, n1_reference)
                 if omission_actual is not None and n1_reference is not None else None})

    writer_layer, writer_unit = WRITER[0]
    module = layers[writer_layer].mlp.down_proj
    reader = _WriterReader(module, writer_unit)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        reader.remove()
    rows.append({"kind": "writer_plain", "writer_activation": float(reader.value[0])})
    hook = ScaleUnits(module, [writer_unit], 0.0)
    reader = _WriterReader(module, writer_unit)
    try:
        with torch.inference_mode():
            model(**inputs, use_cache=False)
    finally:
        reader.remove()
        hook.remove()
    rows.append({"kind": "writer_zero", "writer_activation": float(reader.value[0])})
    return rows


def _sum_trace_writes(traces):
    """Aggregate output-side trace vectors using their actual hidden width."""
    import numpy as np
    import torch
    if not traces:
        return None, None
    hidden_width = int(traces[0].actual.shape[-1])
    actual = np.zeros(hidden_width, dtype=np.float64)
    intended = np.zeros(hidden_width, dtype=np.float64)
    for trace in traces:
        actual += trace.actual.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
        intended += trace.expected.reshape(-1).detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
    return actual, intended


# ---------------------------------------------------------------- runner-up

def _family_spellings():
    return {"qwen": RUNNER_UP_FAMILY, "google": GOOGLE_FAMILY}


def token_families(tok):
    """Find exact single-token spellings and candidate full token sequences on CPU."""
    families = {}
    single_spellings = {}
    spellings = _family_spellings()
    for token_id in range(len(tok)):
        text = tok.decode([token_id], skip_special_tokens=False)
        families[token_id] = {name: run171._word_match(text, family_spellings)
                              for name, family_spellings in spellings.items()}
        single_spellings[token_id] = {name: _matched_family_spellings(text, name, spellings)
                                      for name in spellings}
    sequences = {"qwen": set(), "google": set()}
    sequence_spellings = {"qwen": {}, "google": {}}
    for family, family_spellings in spellings.items():
        for spelling in family_spellings:
            variants = {spelling, spelling.casefold(), spelling.upper(), f" {spelling}", f" {spelling.casefold()}",
                        f" {spelling.upper()}"}
            for variant in variants:
                ids = tuple(single_token_ids(tok, variant))
                if not ids:
                    continue
                sequences[family].add(ids)
                sequence_spellings[family].setdefault(ids, set()).add(spelling)
    families["_sequences"] = {name: sorted(values) for name, values in sequences.items()}
    families["_single_spellings"] = single_spellings
    families["_sequence_spellings"] = sequence_spellings
    return families


def family_counts(families):
    return {name: sum(int(flags[name]) for key, flags in families.items()
                      if isinstance(key, int) and flags[name]) for name in ("qwen", "google")}


def _matched_family_spellings(text, family, spellings=None):
    spellings = _family_spellings() if spellings is None else spellings
    return [spelling for spelling in spellings[family] if run171._word_match(text, (spelling,))]


def _first_family_match(answer_ids, families, family, tok=None):
    sequences = sorted((tuple(seq) for seq in families.get("_sequences", {}).get(family, [])),
                       key=lambda seq: (-len(seq), seq))
    sequence_spellings = families.get("_sequence_spellings", {}).get(family, {})
    for offset, token_id in enumerate(answer_ids):
        if families.get(token_id, {}).get(family):
            decoded = (tok.decode([int(token_id)], skip_special_tokens=False) if tok is not None else "")
            matched = _matched_family_spellings(decoded, family)
            if not matched:
                matched = list(families.get("_single_spellings", {}).get(token_id, {}).get(family, []))
            if not matched:
                matched = list(_family_spellings()[family])
            return {"offset": offset, "matched_spellings": [
                {"spelling": spelling, "decoded": decoded, "token_ids": [int(token_id)]}
                for spelling in matched]}
        for seq in sequences:
            if tuple(answer_ids[offset:offset + len(seq)]) != seq:
                continue
            decoded = (tok.decode(list(map(int, answer_ids[offset:offset + len(seq)])),
                                  skip_special_tokens=False) if tok is not None else "")
            matched = _matched_family_spellings(decoded, family) if tok is not None else []
            if tok is not None and not matched:
                continue
            if not matched:
                matched = sorted(sequence_spellings.get(seq, ())) or list(_family_spellings()[family])
            return {"offset": offset, "matched_spellings": [
                {"spelling": spelling, "decoded": decoded, "token_ids": list(seq)}
                for spelling in matched]}
    return None


def _first_family_offset(answer_ids, families, family):
    match = _first_family_match(answer_ids, families, family)
    return None if match is None else match["offset"]


def runner_up(model, tok, layers, questions, device, families, plain_answers, top_k=5):
    """Read next-token odds after the first Qwen-family token in each saved plain-model answer."""
    import numpy as np
    import torch
    if len(plain_answers) != len(questions):
        raise ValueError("runner-up needs one saved plain answer for each prompt")
    rows = []
    for index, (question, plain_answer) in enumerate(zip(questions, plain_answers)):
        rendered = run171.render_prompt(tok, question)
        prompt_ids = single_token_ids(tok, rendered)
        answer_ids = list(map(int, plain_answer.get("token_ids", [])))
        family_match = _first_family_match(answer_ids, families, "qwen", tok=tok)
        if family_match is None:
            rows.append({"prompt_id": f"new44:{index}", "family_position": None, "google_share": None,
                         "answer_token_offset": None, "removed_mass": None, "top5": [],
                         "matched_spellings": [],
                         "reason": "no Qwen-family token in the plain answer"})
            continue
        answer_offset = family_match["offset"]
        # Stop immediately before the family token whose next-token distribution is being read.
        prefix = prompt_ids + answer_ids[:answer_offset]
        inputs = {"input_ids": torch.tensor([prefix], dtype=torch.long, device=device),
                  "attention_mask": torch.ones((1, len(prefix)), dtype=torch.long, device=device)}
        with torch.inference_mode():
            try:
                output = model(**inputs, use_cache=False, logits_to_keep=1)
            except TypeError:
                output = model(**inputs, use_cache=False)
            logits = output.logits[0, -1].float()
        values = logits.detach().cpu().numpy().astype(np.float64)
        removed_ids = {i for i in range(values.size) if families.get(i, {}).get("qwen", False)}
        removed_ids.update(int(match["token_ids"][0]) for match in family_match["matched_spellings"]
                           if len(match["token_ids"]) > 1)
        removed = np.asarray([i in removed_ids for i in range(values.size)], dtype=bool)
        shifted = values - np.max(values)
        probabilities = np.exp(shifted)
        total_mass = float(probabilities.sum())
        removed_mass = float(probabilities[removed].sum() / total_mass) if total_mass else None
        kept_mass = float(probabilities[~removed].sum())
        google = np.asarray([int(families.get(i, {}).get("google", False)) for i in range(values.size)], dtype=bool)
        google_share = float(probabilities[google & ~removed].sum() / kept_mass) if kept_mass else None
        kept_logits = np.where(~removed, values, -np.inf)
        finite_count = int(np.isfinite(kept_logits).sum())
        count = min(int(top_k), finite_count)
        if count:
            order = np.argpartition(-kept_logits, count - 1)[:count]
            order = order[np.argsort(-kept_logits[order])]
        else:
            order = np.asarray([], dtype=np.int64)
        rows.append({"prompt_id": f"new44:{index}", "family_position": len(prompt_ids) + answer_offset,
                     "answer_token_offset": answer_offset,
                     "matched_spellings": family_match["matched_spellings"],
                     "removed_mass": removed_mass, "google_share": google_share,
                     "top5": [{"token_id": int(i), "text": tok.decode([int(i)], skip_special_tokens=False)}
                              for i in order]})
    return rows


# ---------------------------------------------------------------- pilot

def pilot_update(path, state, count, elapsed):
    """Time generation only. The clock for the baseline rows is never started or added here."""
    if state.get("passed") is False:
        raise SystemExit(f"pilot stop: {state['rate']:.3f} answers/s below {PILOT_THRESHOLD}")
    if state.get("batches", 0) >= PILOT_BATCHES:
        return state
    state["batches"] = int(state.get("batches", 0)) + 1
    state["answers"] = int(state.get("answers", 0)) + int(count)
    state["elapsed_seconds"] = float(state.get("elapsed_seconds", 0.0)) + float(elapsed)
    if state["batches"] == PILOT_BATCHES:
        state["rate"] = state["answers"] / max(state["elapsed_seconds"], 1e-9)
        state["passed"] = bool(state["rate"] >= PILOT_THRESHOLD)
    state["threshold"] = PILOT_THRESHOLD
    state["timed_on"] = "generation only, the first 3 generation batches after the gates"
    write_json(path, state)
    if state.get("passed") is False:
        raise SystemExit(f"pilot stop: {state['rate']:.3f} answers/s below {PILOT_THRESHOLD}")
    return state


# ---------------------------------------------------------------- the stage

def _answer_record(spec, prompt, answer):
    row = {"arm": spec["arm"], "prompt_id": prompt["prompt_id"], "question_index": prompt["question_index"],
           "question": prompt["question"], **answer}
    for key in ("set", "single", "dose", "knockout", "writer_off", "frame"):
        if spec.get(key) is not None:
            row[key] = spec[key]
    return row


def _replay_record(arm, prompt, answer):
    return {"arm": f"replay:{arm}", "prompt_id": prompt["prompt_id"], "question_index": prompt["question_index"],
            "question": prompt["question"], "dose": 1.0, **answer}


def test_stage(args, tok, model, layers, out, sets, families, sets_path=None):
    import torch
    widths = widths_of(layers)
    answer_path = Path(out) / "answers.jsonl"
    repair_tail(answer_path)
    answers = read_jsonl(answer_path)
    done = {answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("answers.jsonl contains duplicate answer keys")
    prompts = new_prompts()
    questions = [p["question"] for p in prompts]
    specs = arm_specs()
    plan = gate1_plan(specs, prompts)
    if not plan["passed"]:
        write_json(Path(out) / "gates.json", {"passed": False, "gate1_plan": plan})
        raise SystemExit(f"Part 172 planned-arm check failed before generations: {plan}")
    by_arm = {spec["arm"]: spec for spec in specs}

    def run_arm(spec, prompt_list=None, replay=False):
        prompt_list = prompts if prompt_list is None else prompt_list
        started = time.perf_counter()
        generated = run_spec(model, tok, layers, spec, [p["question"] for p in prompt_list], args.device,
                             args.max_new_tokens, sets, widths)
        elapsed = time.perf_counter() - started
        if len(generated) != len(prompt_list):
            raise RuntimeError(f"{spec['arm']} generated {len(generated)} answers for {len(prompt_list)} prompts")
        rows = [(_replay_record(spec["arm"].split(":", 1)[1], p, a) if replay else _answer_record(spec, p, a))
                for p, a in zip(prompt_list, generated)]
        append_answers(answer_path, rows, done)
        return elapsed

    # Gates 2, 3, 5 need the baselines and the knockout's d = 1 row; they run before the ladder.
    for arm in ("none", "role", "ko1"):
        spec = by_arm[arm]
        if any((arm, p["prompt_id"]) not in done for p in prompts):
            run_arm(spec)

    # Gate 3: zero push on every set layer, identity hooks on two units, then prove a d=1 arm leaves no hooks.
    checks = prompts[:GATE3_PROMPTS]
    check_questions = [p["question"] for p in checks]
    hook_start = hook_counts(layers)
    plain = run171.generate_batch(model, tok, layers, [p["question"] for p in checks], [None] * len(checks),
                                  args.device, args.max_new_tokens, seed=172)
    zero_pushes = {layer: torch.zeros(width, dtype=torch.float32, device=args.device)
                   for layer, width in widths.items()}
    zero_push = run171.generate_batch(model, tok, layers, check_questions, [None] * len(checks),
                                     args.device, args.max_new_tokens, pushes=zero_pushes, seed=172)
    identity_hooks = knockout_hooks(layers, GATE3_UNITS, 1.0)
    identity_hook_registered = any(hook_counts(layers)[layer] > hook_start[layer] for layer, _ in _by_layer(GATE3_UNITS))
    try:
        identity_units = run171.generate_batch(model, tok, layers, check_questions, [None] * len(checks),
                                               args.device, args.max_new_tokens, seed=172)
    finally:
        for hook in identity_hooks:
            hook.remove()
    after_identity = hook_counts(layers)
    pushed_check = run_spec(model, tok, layers, {"arm": "gate3_pushed", "set": "n1", "dose": 1.0},
                            check_questions, args.device, args.max_new_tokens, sets, widths)
    after_pushed = hook_counts(layers)
    post_push_plain = run171.generate_batch(model, tok, layers, check_questions, [None] * len(checks),
                                            args.device, args.max_new_tokens, seed=172)
    after_final = hook_counts(layers)
    plain_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]} for p, a in zip(checks, plain)]
    zero_push_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]} for p, a in zip(checks, zero_push)]
    identity_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]} for p, a in zip(checks, identity_units)]
    post_push_plain_rows = [{"prompt_id": p["prompt_id"], "token_ids": a["token_ids"]}
                            for p, a in zip(checks, post_push_plain)]

    # Gate 4: replay the three d = 1 ladder arms on Part 171's prompts, in its batch order and batch shape.
    reference = read_jsonl(PART171_ANSWERS)
    replay_prompts = part171_prompts()
    for arm in REPLAY_ARMS:
        if any((f"replay:{arm}", p["prompt_id"]) not in done for p in replay_prompts):
            run_arm({"arm": f"replay:{arm}", "set": arm, "dose": 1.0}, replay_prompts, replay=True)

    answers = read_jsonl(answer_path)
    new_rows = [row for row in answers if not str(row["arm"]).startswith("replay:")]
    replay_rows = [row for row in answers if str(row["arm"]).startswith("replay:")]
    g2 = gate2(new_rows)
    hook_counts_restored = (after_identity == hook_start and after_pushed == hook_start and after_final == hook_start)
    g3 = gate3(plain_rows, zero_push_rows, identity_rows, post_push_plain_rows,
               hook_counts_restored=hook_counts_restored, identity_hook_registered=identity_hook_registered,
               pushed_arm_checks=pushed_check)
    g4 = gate4(replay_rows, reference)
    g5 = gate5(new_rows)
    g6_rows = measure_gate6(model, tok, layers, sets, args.device, widths)
    g6 = gate6(g6_rows)
    g7 = gate7()
    g4["artifacts"] = artifact_hashes(sets if sets_path is None else sets_path, tok, model, args.model)
    stage0 = {"gate1_plan": plan, "gate2": g2, "gate3": g3, "gate4": g4, "gate5": g5, "gate6": g6, "gate7": g7,
              "plain_checks": plain_rows, "zero_push_checks": zero_push_rows,
              "identity_unit_checks": identity_rows, "post_push_plain_checks": post_push_plain_rows,
              "gate6_rows": g6_rows, "test_prompts": prompts,
              "families": {"counts": family_counts(families), "vocabulary": len(families),
                           "qwen_family": list(RUNNER_UP_FAMILY), "google_family": list(GOOGLE_FAMILY)}}
    write_json(Path(out) / "stage0.json", stage0)
    failed = [name for name in ("gate2", "gate3", "gate4", "gate5", "gate6", "gate7") if not stage0[name]["passed"]]
    if failed:
        write_json(Path(out) / "gates.json", {"passed": False, "failed": failed, **{k: stage0[k] for k in failed}})
        raise SystemExit(f"Part 172 gates failed before the ladder: {failed}")

    # Runner-up: one forward pass per prompt from the plain answers.
    plain_by_prompt = {str(row["prompt_id"]): row for row in new_rows if row.get("arm") == "none"}
    plain_answers = [plain_by_prompt[p["prompt_id"]] for p in prompts]
    runner = runner_up(model, tok, layers, questions, args.device, families, plain_answers)
    write_json(Path(out) / "runner_up.json", {"part": 172, "rows": runner,
                                              "mean_google_share": _mean([r["google_share"] for r in runner]),
                                              "named": sum(r["google_share"] is not None for r in runner)})

    # Pilot: the first 3 generation batches after the gates, timed on generation only.
    pilot_path = Path(out) / "pilot.json"
    pilot = read_json(pilot_path, {"batches": 0, "answers": 0, "elapsed_seconds": 0.0})
    if pilot is None:
        pilot = {"batches": 0, "answers": 0, "elapsed_seconds": 0.0}
    for spec in specs:
        if spec["arm"] in ("none", "role", "ko1"):
            continue
        if all((spec["arm"], p["prompt_id"]) in done for p in prompts):
            continue
        if pilot.get("passed") is False:
            raise SystemExit(f"pilot stop: {pilot['rate']:.3f} answers/s below {PILOT_THRESHOLD}")
        elapsed = run_arm(spec)
        pilot = pilot_update(pilot_path, pilot, len(prompts), elapsed)

    answers = read_jsonl(answer_path)
    g1 = gate1([row for row in answers if not str(row["arm"]).startswith("replay:")], planned_keys(specs))
    gates = {"gate1_plan": plan, "gate1": g1,
             **{k: stage0[k] for k in ("gate2", "gate3", "gate4", "gate5", "gate6", "gate7")}}
    gates["passed"] = all(gates[k]["passed"] for k in gates if k != "passed")
    write_json(Path(out) / "gates.json", gates)
    if not gates["passed"]:
        raise SystemExit(f"Part 172 Gate 1 failed after the ladder: {g1}")
    return {"stage": "test", "answers": len(answers), "planned_answers": len(planned_keys(specs)),
            "replay_answers": len(replay_rows), "gates": {k: gates[k]["passed"] for k in gates if k != "passed"},
            "pilot": pilot}


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def resolve_sets_path(out_dir, config=None):
    config = config or {}
    return config.get("sets_path", PART171_SETS)


def preflight_inputs(sets_path, answers_path=PART171_ANSWERS):
    if not isinstance(sets_path, dict) and not Path(sets_path).is_file():
        raise FileNotFoundError(f"Part 171 pushes are required before Part 172: {sets_path}")
    if not Path(answers_path).is_file():
        raise FileNotFoundError(f"Part 171 replay answers are required before Part 172: {answers_path}")


def run(args, loader=None, config=None):
    config = config or {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sets_path = resolve_sets_path(out, config)
    preflight_inputs(sets_path, PART171_ANSWERS)
    sets = sets_path if isinstance(sets_path, dict) else load_sets(sets_path)
    strict = bool(config.get("strict_model", loader is None))
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False)
    model.eval()
    layers = common86.find_layers(model)
    if strict:
        run171.validate_model(layers)
    if "unit_weight_columns" not in sets and "weights" not in sets:
        # Only the selected columns are needed; never retain 21 full down-projection matrices on the host.
        sets["unit_weight_columns"] = single_weight_columns(layers)
    families = token_families(tok)
    if args.stage == "test":
        return test_stage(args, tok, model, layers, out, sets, families, sets_path)
    raise ValueError(f"unknown Part 172 stage {args.stage!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("test",), default="test")
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out-dir", default="results/part172")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 44 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
