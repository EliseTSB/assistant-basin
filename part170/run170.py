#!/usr/bin/env python3
"""Part 170 runner: plain-token bans across no-push and neuron-push answers."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part155 import run155
from part165 import effect165, run165, score165
from part166 import run166, score166
from part168 import run168
from part169 import run169
from prompts170 import M96, Q96

LAYERS = tuple(range(19, 27))
SLOTS = 6
DOSES = (0.25, 0.5, 1.0)
BAN_NAMES = ("b0", "b5", "b50", "b200", "rel50")
PUSH_BANS = ("b0", "b5", "b50", "b200")
BATCH_SIZE = 64
MAX_NEW_TOKENS = 80
LIST_PATH_TOKENS = 40
BOOTSTRAP_REPS = 10_000
BOOTSTRAP_SEED = 170


def prompt_catalog():
    """Return stable prompt identities; only P8 receives pushed answers."""
    rows = []
    for qi, item in run169.phrasing_items():
        rows.append({"prompt_id": f"p8:{qi}", "group": "p8", "question_index": qi,
                     "kind": "maker", "question": item["question"]})
    for qi, question in enumerate(M96):
        rows.append({"prompt_id": f"m96:{qi}", "group": "m96", "question_index": qi,
                     "kind": "maker" if qi < 48 else "identity", "question": question})
    for qi, question in enumerate(Q96):
        rows.append({"prompt_id": f"q96:{qi}", "group": "q96", "question_index": qi,
                     "kind": "fact" if qi < 48 else "how_to", "question": question})
    return rows


def answer_key(row):
    """Unique key for Stage 0, no-push, push, and repeat answers."""
    return (row.get("stage"), row.get("group"), row.get("prompt_id"), row.get("arm"),
            float(row.get("dose", 0.0)), row.get("layer"), row.get("slot"), row.get("ban"),
            bool(row.get("repeat", False)))


def push_row_key(row):
    return (row.get("prompt_id"), int(row.get("layer", -1)), int(row.get("slot", -1)))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def repair_tail(path):
    """A killed job can leave a torn last line. Cut back to the last newline so a resume can append."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        path.write_bytes(data[:data.rfind(b"\n") + 1])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def stage0_flags(answer):
    text = answer.get("answer", "")
    return {"loop": bool(score166.loop_phrase(text)), "broken": bool(score165.primary_broken(text))}


def stage0_flagged(answer, group):
    """Part 165's broken rule was built for identity answers and calls any answer under four words broken,
    so a correct short Q96 answer is not flagged; only a loop counts there."""
    flags = stage0_flags(answer)
    return bool(flags["loop"] or (flags["broken"] and group != "q96"))


def stage0_gates(prompts, plain_answers, zero_checks):
    """Shared Stage 0 gate authority used by the runner and scorer."""
    by_prompt = {str(row.get("prompt_id")): row for row in plain_answers}
    expected = {str(row["prompt_id"]): row for row in prompts}
    complete = set(by_prompt) == set(expected) and len(plain_answers) == len(expected)
    group_rates = {}
    group_counts = {}
    for group in ("p8", "m96", "q96"):
        vals = [by_prompt[pid] for pid, prompt in expected.items()
                if prompt["group"] == group and pid in by_prompt]
        flagged = sum(stage0_flagged(row, group) for row in vals)
        group_rates[group] = flagged / len(vals) if vals else None
        group_counts[group] = {"prompts": len(vals), "loop_or_broken": flagged}
    p8 = [by_prompt.get(str(row["prompt_id"])) for row in prompts if row["group"] == "p8"]
    gate2 = complete and len(p8) == 8 and all(row is not None and
        not stage0_flags(row)["loop"] and not stage0_flags(row)["broken"] for row in p8)
    # A prompt set over 15% stops that set's readings only, as the design says; P8 alone stops the run.
    set_passed = {group: group_rates[group] is not None and group_rates[group] <= 0.15
                  for group in ("m96", "q96")}
    zero = {str(row.get("prompt_id")): row for row in zero_checks}
    gate3 = len(zero) == 8 and all(
        pid in zero and list(map(int, by_prompt[pid].get("token_ids", []))) ==
        list(map(int, zero[pid].get("token_ids", [])))
        for pid in (str(row["prompt_id"]) for row in prompts if row["group"] == "p8")
    )
    frozen = {group: [] for group in ("p8", "m96", "q96")}
    lengths = {group: {"frozen": [], "surviving": []} for group in frozen}
    for prompt_id, prompt in expected.items():
        row = by_prompt.get(prompt_id)
        if row is None:
            continue
        category = "frozen" if stage0_flagged(row, prompt["group"]) else "surviving"
        if category == "frozen":
            frozen[prompt["group"]].append(int(prompt["question_index"]))
        lengths[prompt["group"]][category].append(int(row.get("token_count", len(row.get("token_ids", [])))))
    length_summary = {}
    for group, values in lengths.items():
        length_summary[group] = {}
        for kind, numbers in values.items():
            ordered = sorted(numbers)
            middle = len(ordered) // 2
            median = ((ordered[middle - 1] + ordered[middle]) / 2 if len(ordered) % 2 == 0
                      else ordered[middle]) if ordered else None
            length_summary[group][kind] = {
                "prompts": len(numbers),
                "mean_tokens": float(sum(numbers) / len(numbers)) if numbers else None,
                "median_tokens": float(median) if median is not None else None,
            }
    return {"gate2": bool(gate2), "gate3": bool(gate3), "set_passed": set_passed,
            "complete_plain": bool(complete), "flagged_rates": group_rates,
            "counts": group_counts, "frozen_indices": frozen,
            "frozen_prompt_ids": [pid for group in frozen for pid in
                                   [str(row["prompt_id"]) for row in prompts
                                    if row["group"] == group and
                                    int(row["question_index"]) in frozen[group]]],
            "plain_lengths": length_summary}


def answer_list_key(row):
    return (str(row.get("prompt_id")), str(row.get("group")), int(row.get("question_index", -1)))


def gate4_size(norms, expected_directions=None):
    directions = norms.get("directions", []) if isinstance(norms, dict) else list(norms or [])
    expected = expected_directions if expected_directions is not None else len(LAYERS) * SLOTS * len(DOSES)
    keys = [(int(row.get("layer", -1)), int(row.get("slot", -1)), float(row.get("dose", -1)))
            for row in directions]
    good = len(directions) == expected and len(set(keys)) == len(keys)
    errors = []
    for row in directions:
        try:
            target = float(row["target_norm"])
            actual = float(row["realized_write_norm"])
            error = float(row["relative_error"])
            ok = target > 0 and actual > 0 and all(math.isfinite(x) for x in (target, actual, error))
            # bf16 rounding of one activation is about 0.6% at dose 1 (Part 165: 0.995 to 1.006), and grows as the write shrinks.
            tolerance = 0.01 / float(row.get("dose", 1.0))
            ok = ok and abs(actual - target) / target <= tolerance and abs(error - abs(actual - target) / target) <= 1e-6
        except (KeyError, TypeError, ValueError):
            ok = False
        errors.append(not ok)
    return {"passed": bool(good and not any(errors)), "directions": len(directions),
            "expected_directions": expected, "write_failures": sum(errors),
            "max_relative_error": max((float(row.get("relative_error", math.inf))
                                        for row in directions), default=None)}


def gate7_freshness(norms, expected_directions=None):
    directions = norms.get("directions", []) if isinstance(norms, dict) else list(norms or [])
    expected = expected_directions if expected_directions is not None else len(LAYERS) * SLOTS * len(DOSES)
    prior = norms.get("prior_neurons", {}) if isinstance(norms, dict) else {}
    expected_coords = {(layer, slot, dose) for layer in LAYERS for slot in range(SLOTS) for dose in DOSES}
    if expected_directions is not None and expected != len(expected_coords):
        expected_coords = None
    coords = [(int(x.get("layer", -1)), int(x.get("slot", -1)), float(x.get("dose", -1))) for x in directions]
    seen = {}
    slot_rows = {}
    passed = len(directions) == expected and len(set(coords)) == len(coords)
    for row in directions:
        layer = int(row.get("layer", -1)); neuron = int(row.get("neuron", -1))
        slot_key = (layer, int(row.get("slot", -1)))
        previous = slot_rows.setdefault(slot_key, neuron)
        if previous != neuron or int(row.get("pushed_neuron", neuron)) != neuron:
            passed = False
    for (layer, _slot), neuron in slot_rows.items():
        used = seen.setdefault(layer, set())
        old = set(map(int, prior.get(str(layer), [])))
        if neuron < 0 or neuron in old or neuron in used:
            passed = False
        used.add(neuron)
    if expected_coords is not None and set(coords) != expected_coords:
        passed = False
    return {"passed": bool(passed), "directions": len(directions), "expected_directions": expected,
            "unique_coordinates": len(set(coords)), "fresh_neurons": bool(passed)}


def answer_finite(row):
    if not isinstance(row.get("answer"), str) or not row["answer"].strip():
        return False
    ids = row.get("token_ids")
    if not isinstance(ids, list) or not ids:
        return False
    try:
        return all(int(token) >= 0 for token in ids) and math.isfinite(float(row.get("token_count", len(ids))))
    except (TypeError, ValueError):
        return False


def _normal_key(row):
    return (row.get("prompt_id"), int(row.get("layer", -1)), int(row.get("slot", -1)))


def planned_answer_keys(prompts, layers=None, slots=None):
    """Exact Stage 0 and Stage 2 key set; Gate 1 checks coverage, not just a row count."""
    layers = LAYERS if layers is None else tuple(layers)
    slots = SLOTS if slots is None else int(slots)
    keys = set()
    for prompt in prompts:
        common = {"group": prompt["group"], "prompt_id": prompt["prompt_id"], "dose": 0.0}
        keys.add(answer_key({**common, "stage": "stage0", "arm": "plain", "ban": "b0"}))
        bans = ["b0", "b5", "b50", "b200", "rel50"]
        if prompt["group"] == "q96":
            bans.append("other50")
        for ban in bans:
            keys.add(answer_key({**common, "stage": "no_push", "arm": "no_push", "ban": ban}))
    p8 = [prompt for prompt in prompts if prompt["group"] == "p8"]
    for prompt in p8:
        for layer in layers:
            for slot in range(int(slots)):
                for dose in DOSES:
                    for ban in PUSH_BANS + (("rel50",) if dose == 1.0 else ()):
                        keys.add(answer_key({"stage": "push", "group": "p8", "prompt_id": prompt["prompt_id"],
                            "arm": "n+", "dose": dose, "layer": int(layer), "slot": slot, "ban": ban}))
                keys.add(answer_key({"stage": "push_repeat", "group": "p8", "prompt_id": prompt["prompt_id"],
                    "arm": "n+", "dose": 1.0, "layer": int(layer), "slot": slot,
                    "ban": "b0", "repeat": True}))
    return keys


def gate_rules(prompts, answers, stage0, lists, norms, expected_answers=None,
               expected_directions=None, slots=None, layers=None):
    """One gate implementation shared by runner prechecks and host scoring."""
    slots = SLOTS if slots is None else int(slots)
    layers = tuple(LAYERS if layers is None else layers)
    expected_directions = (len(layers) * slots * len(DOSES) if expected_directions is None
                           else int(expected_directions))
    plan = planned_answer_keys(prompts, layers=layers, slots=slots)
    expected_answers = (len(plan) if expected_answers is None else int(expected_answers))
    keys = [answer_key(row) for row in answers]
    gate1 = (len(answers) == expected_answers and len(set(keys)) == len(keys) and
             set(keys) == plan and all(answer_finite(row) for row in answers))

    plain = [row for row in answers if row.get("stage") == "stage0"]
    zero = stage0.get("zero_checks", [])
    s0 = stage0_gates(prompts, plain, zero)
    stored_frozen = set(map(str, stage0.get("frozen_prompt_ids", [])))
    actual_frozen = set(map(str, s0["frozen_prompt_ids"]))
    stored_indices = {group: set(map(int, stage0.get("frozen_indices", {}).get(group, [])))
                      for group in ("p8", "m96", "q96")}
    actual_indices = {group: set(map(int, s0["frozen_indices"][group]))
                      for group in ("p8", "m96", "q96")}
    frozen_match = stored_frozen == actual_frozen and stored_indices == actual_indices
    gate2, gate3 = bool(s0["gate2"] and frozen_match), bool(s0["gate3"])
    size = gate4_size(norms, expected_directions)
    gate4 = size["passed"]

    listmap = {str(row.get("prompt_id")): row for row in lists}
    specials = set(map(int, stage0.get("special_ids", [])))
    gate5 = True
    for row in answers:
        if row.get("stage") not in ("no_push", "push", "push_repeat"):
            continue
        ban = str(row.get("ban"))
        banned = list(map(int, row.get("banned_ids", [])))
        record = listmap.get(str(row.get("prompt_id")), {})
        expected_ids = [] if ban == "b0" else list(map(int, record.get(ban, [])))
        if banned != expected_ids or set(banned) & specials:
            gate5 = False
        token_ids = list(map(int, row.get("token_ids", [])))
        if ban == "rel50":
            if set(banned) & set(token_ids[:5]):
                gate5 = False
        elif set(banned) & set(token_ids):
            gate5 = False
        if ban in ("b5", "b50", "b200", "other50") and len(banned) != {"b5": 5, "b50": 50, "b200": 200, "other50": 50}[ban]:
            gate5 = False
        if ban == "rel50" and len(banned) != 50:
            gate5 = False
    directions = norms.get("directions", []) if isinstance(norms, dict) else list(norms or [])

    push = [row for row in answers if row.get("stage") == "push" and not row.get("repeat")]
    main_index = {_normal_key(row): row for row in push}
    dose1_base = [row for row in push if float(row.get("dose", 0)) == 1.0 and row.get("ban") == "b0"]
    dose1_b50 = [row for row in push if float(row.get("dose", 0)) == 1.0 and row.get("ban") == "b50"]
    base_loops = sum(bool(score166.loop_phrase(row.get("answer", ""))) for row in dose1_base)
    gate6_loop_rate = base_loops / len(dose1_base) if dose1_base else None
    blocked = [row for row in dose1_b50]
    blocked_mean = (sum(len(row.get("blocked_steps", [])) for row in blocked) / len(blocked)
                    if blocked else None)
    gate6 = bool(gate6_loop_rate is not None and gate6_loop_rate >= 0.08 and
                 blocked_mean is not None and blocked_mean >= 5.0)

    repeats = [row for row in answers if row.get("stage") == "push_repeat" and row.get("ban") == "b0"]
    repeat_map = {_normal_key(row): row for row in repeats}
    original_map = {_normal_key(row): row for row in dose1_base}
    shared = sorted(set(original_map) & set(repeat_map))
    different = sum(list(map(int, original_map[key].get("token_ids", []))) !=
                    list(map(int, repeat_map[key].get("token_ids", []))) for key in shared)
    noise_rate = different / len(shared) if shared else None
    gate8 = len(shared) == len(dose1_base) == len(repeats) and noise_rate is not None and noise_rate < 0.10

    fresh = gate7_freshness(norms, expected_directions)
    gates = {"gate1": bool(gate1), "gate2": gate2, "gate3": gate3, "gate4": bool(gate4),
             "gate5": bool(gate5), "gate6": gate6, "gate7": bool(fresh["passed"]), "gate8": bool(gate8)}
    return {**gates, "passed": all(gates.values()),
            "diagnostics": {"expected_answers": expected_answers, "answers": len(answers),
                "unique_answer_keys": len(set(keys)), "planned_answer_keys": len(plan),
                "stage0": s0, "stage0_frozen_matches_answers": frozen_match,
                "gate4": size, "gate7": fresh,
                "gate6_dose1_b0_loop_rate": gate6_loop_rate,
                "gate6_dose1_b50_mean_blocked_steps": blocked_mean,
                "gate8_rows": len(shared), "gate8_differing_answers": different,
                "gate8_noise_rate": noise_rate}}


def top_ids(probabilities, count, special, torch):
    """Vocabulary top-k on CPU, with enough headroom to remove every special id."""
    probs = torch.as_tensor(probabilities, dtype=torch.float32, device="cpu").flatten()
    k = min(int(probs.numel()), int(count) + len(special))
    _values, ids = torch.topk(probs, k=k)
    out = [int(token) for token in ids.tolist() if int(token) not in special]
    out = list(dict.fromkeys(out))[:int(count)]
    if len(out) != int(count):
        raise ValueError(f"only {len(out)} non-special tokens available for top-{count}")
    return out


def other50_ids(next_top100, own_b50):
    own = set(map(int, own_b50))
    result = [int(token) for token in next_top100 if int(token) not in own]
    if len(result) < 50:
        raise ValueError("next-question top 100 did not leave 50 other50 tokens")
    result = result[:50]
    if len(set(result)) != 50:
        raise ValueError("other50 contains duplicate ids")
    return result


def other50_mass(probabilities, ids):
    return float(sum(float(probabilities[int(token)]) for token in ids))


def stage1_prompt(model, tok, prompt, plain_row, common, device, special, torch):
    """Mean plain-path probabilities and entropy, reducing one position at a time."""
    answer_ids = list(map(int, plain_row.get("token_ids", [])))[:LIST_PATH_TOKENS]
    prompt_ids = list(map(int, plain_row["prompt_ids"]))
    positions = run166.evaluation_positions(len(prompt_ids), len(answer_ids))
    if not positions:
        raise ValueError(f"plain path too short to form a list for {prompt['prompt_id']}")
    input_ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    hidden = run166.forward_hidden(model, input_ids, positions)
    head = model.get_output_embeddings()
    # Part 169's path, so P8's b50 is its plain50: stack on the device, then one mean and one entropy cross to the CPU.
    with torch.inference_mode():
        probs = torch.stack([torch.softmax(x.float(), -1) for x in run166.plain_position_logits(head, hidden)])
        if not bool(torch.isfinite(probs).all()):
            raise FloatingPointError(f"non-finite plain logits for {prompt['prompt_id']}")
        mean_tensor = probs.mean(0).detach().to(device="cpu", dtype=torch.float32)
        entropy = float((-(probs * torch.log(probs.clamp_min(1e-30))).sum(-1)).mean().detach().to("cpu"))
    # This exact helper is Part 169's CPU top-50 implementation.
    b50 = run169.plain50_ids(mean_tensor, special, torch)
    top5 = top_ids(mean_tensor, 5, special, torch)
    top200 = top_ids(mean_tensor, 200, special, torch)
    top100 = top_ids(mean_tensor, 100, special, torch)
    return {"prompt_id": prompt["prompt_id"], "group": prompt["group"],
            "question_index": int(prompt["question_index"]), "entropy": entropy,
            "path_positions": len(positions), "path_tokens": len(answer_ids),
            "top5": top5, "b5": top5, "b50": b50, "rel50": b50, "b200": top200,
            "top100": top100, "_probabilities": mean_tensor.numpy()}


class ReleaseBanProcessor:
    """Per-row ban masks released after that row has generated five tokens."""
    def __init__(self, mask, prompt_lengths, release_after):
        self.mask = mask
        self.prompt_lengths = tuple(map(int, prompt_lengths))
        self.release_after = tuple(-1 if limit is None else int(limit) for limit in release_after)
        if len(self.prompt_lengths) != int(mask.shape[0]) or len(self.release_after) != int(mask.shape[0]):
            raise ValueError("release processor rows do not match its mask")
        self.device_lengths = self.device_limits = None

    def __call__(self, input_ids, scores):
        import torch
        if self.mask.shape != scores.shape or input_ids.shape[0] != self.mask.shape[0]:
            raise ValueError("release processor input shape mismatch")
        if self.device_lengths is None:  # one upload per batch, not one per step on the A770
            self.device_lengths = torch.as_tensor(self.prompt_lengths, device=input_ids.device)
            self.device_limits = torch.as_tensor(self.release_after, dtype=torch.long, device=input_ids.device)
        limits = self.device_limits
        steps = input_ids.shape[-1] - self.device_lengths
        active = (limits < 0) | (steps < limits)
        return scores.masked_fill(self.mask & active.unsqueeze(1), -float("inf"))


@contextmanager
def ban_processor_context(prompt_lengths, release_after):
    """Bind per-row release counts into Part 168's processor construction point."""
    original = run168.RowBanProcessor

    class BoundProcessor:
        def __init__(self, mask):
            self.inner = ReleaseBanProcessor(mask, prompt_lengths, release_after)

        def __call__(self, input_ids, scores):
            return self.inner(input_ids, scores)

    run168.RowBanProcessor = BoundProcessor
    try:
        yield
    finally:
        run168.RowBanProcessor = original


def question_prompt_length(common, tok, question, rows, device):
    inputs = common.token_inputs(tok, question, len(rows), device)
    return int(inputs["input_ids"].shape[-1])


def generate_with_bans(model, tok, layers, question, rows, directions, bans, device,
                       common, seed, release_after=None, pushed=True, max_new_tokens=MAX_NEW_TOKENS):
    """Use Part 169's block-log wrapper and Part 168's batched path for both arms."""
    release_after = list(release_after or [None] * len(rows))
    prompt_len = question_prompt_length(common, tok, question, rows, device)
    with ban_processor_context([prompt_len] * len(rows), release_after):
        if pushed:
            return run169.with_block_log(run168.generate_batch, model, tok, layers, question,
                rows, directions, bans, device, common, max_new_tokens, seed)
        return run169.with_block_log(run169.generate_unpushed_batch, model, tok, question,
            bans, common, device, max_new_tokens)


def append_answers(path, rows, done):
    """Append only unseen answer keys; malformed partial files are rejected on read."""
    pending = []
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            key = answer_key(row)
            if key in done:
                continue
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
            done.add(key)
            pending.append(row)
        stream.flush()
    return pending


def rows_for_question(qi, slots=None):
    slots = SLOTS if slots is None else int(slots)
    rows = [{"arm": "n+", "layer": layer, "slot": slot, "question_index": int(qi)}
            for layer in LAYERS for slot in range(slots)]
    random.Random(170 + int(qi)).shuffle(rows)
    return rows


def repeat_row_order(rows, seed=170):
    """The repeat order is deterministic; callers key answers back to canonical rows."""
    order = list(rows)
    random.Random(int(seed)).shuffle(order)
    return order


def build_prior_neurons(widths, torch):
    prior = run169.prior_neurons_by_layer(widths, torch)
    part169_path = ROOT / "results/part169/norms.json"
    if not part169_path.is_file():
        raise FileNotFoundError(f"Part 169 freshness artifact is required: {part169_path}")
    old = json.loads(part169_path.read_text(encoding="utf-8"))
    previous = [row for row in old.get("directions", []) if row.get("arm") == "n+"]
    counts = {layer: 0 for layer in LAYERS}
    neurons = {layer: set() for layer in LAYERS}
    for direction in previous:
        layer = int(direction.get("layer", -1))
        if layer not in prior or direction.get("neuron") is None:
            raise ValueError("Part 169 n+ freshness record has an invalid layer or neuron")
        neuron = int(direction["neuron"])
        if neuron < 0 or neuron >= int(widths[layer]):
            raise ValueError(f"Part 169 n+ neuron {neuron} is outside layer {layer}")
        counts[layer] += 1
        neurons[layer].add(neuron)
        prior[layer].add(neuron)
    if any(counts[layer] != 12 or len(neurons[layer]) != 12 for layer in LAYERS):
        raise ValueError("Part 169 must contribute 12 unique n+ neurons in every Part 170 layer")
    return prior


def fresh_neuron_170(width, layer, slot, used, torch):
    """Call Part 169's rejection sampler with Part 170's registered seed base."""
    original = run169.neuron_for_seed

    def seeded(width0, layer0, slot0, _base, torch0=None, attempt=0):
        return original(width0, layer0, slot0, 170000, torch0, attempt)

    run169.neuron_for_seed = seeded
    try:
        return run169.fresh_neuron(width, layer, slot, used, torch)
    finally:
        run169.neuron_for_seed = original


def build_direction(weight, layer, slot, attempt, median, cache, torch):
    """Use Part 165's write-normalized recipe with Part 170's deterministic seed."""
    original = run165._seeded_raw

    def seeded(weight0, layer0, slot0, torch0):
        width = int(weight0.shape[1]); hidden = int(weight0.shape[0])
        generator = torch0.Generator(device="cpu")
        generator.manual_seed(170000 + 100 * int(layer0) + int(slot0) + 1_000_000 * int(attempt))
        return (torch0.randperm(width, generator=generator, device="cpu"),
                torch0.randn(width, generator=generator, dtype=torch0.float32, device="cpu"),
                torch0.randn(hidden, generator=generator, dtype=torch0.float32, device="cpu"))

    run165._seeded_raw = seeded
    try:
        vectors, metrics = run165.build_directions(weight, int(layer), int(slot), float(median), cache)
    finally:
        run165._seeded_raw = original
    return vectors["n+"], metrics["n+"]


def make_directions(layers, medians, torch, slots=SLOTS):
    widths = {layer: int(layers[layer].mlp.down_proj.weight.shape[1]) for layer in LAYERS}
    prior = build_prior_neurons(widths, torch)
    directions = {}
    records = []
    cache = {}
    for layer in LAYERS:
        used = set(prior[layer])
        median = float(medians.get(str(layer), medians.get(layer)))
        for slot in range(int(slots)):
            neuron, attempt = fresh_neuron_170(widths[layer], layer, slot, used, torch)
            used.add(neuron)
            vector, base_metrics = build_direction(layers[layer].mlp.down_proj.weight,
                layer, slot, attempt, median, cache, torch)
            directions[(layer, slot)] = vector
            for dose in DOSES:
                records.append({"layer": layer, "slot": slot, "neuron": neuron,
                    "pushed_neuron": int(vector.abs().argmax()), "attempt": attempt, "dose": dose,
                    "base_target_norm": float(base_metrics["target_norm"]),
                    "intended_write_norm": float(base_metrics["write_norm"]) * dose,
                    "target_norm": float(base_metrics["target_norm"]) * dose,
                    "relative_error": 0.0})
    return directions, records, prior, widths


def scaled_directions(base, dose):
    return {(layer, slot): {"n+": vector * float(dose)} for (layer, slot), vector in base.items()}


def measure_doses(model, tok, layers, prompts, directions, records, device, common):
    """Use Part 165's captured activation delta on one shared P8 context per dose."""
    import torch
    from part165 import run165 as part165_run
    prompt = next(row for row in prompts if row["group"] == "p8" and row["question_index"] == 0)
    rows = rows_for_question(0, slots=SLOTS)
    row_layers = {index: int(row["layer"]) for index, row in enumerate(rows)}
    for dose in DOSES:
        direction_map = scaled_directions(directions, dose)
        layer_pushes = part165_run.pushes_for(rows, layers, direction_map)
        captured = effect165.generate_capture(model, tok, layers, prompt["question"], layer_pushes,
            device, common, row_layers=row_layers)
        by_row = {int(index): item for index, item in captured["realized"].items()}
        for index, row in enumerate(rows):
            key = (int(row["layer"]), int(row["slot"]), float(dose))
            record = next(item for item in records if
                (int(item["layer"]), int(item["slot"]), float(item["dose"])) == key)
            realized = float(by_row[index]["realized_write_norm"])
            target = float(record["target_norm"])
            record["realized_write_norm"] = realized
            record["intended_write_norm"] = float(by_row[index]["intended_write_norm"])
            record["realized_cos"] = float(by_row[index]["realized_cos"])
            record["changed_channels"] = int(by_row[index]["changed_channels"])
            record["relative_error"] = abs(realized - target) / target if target else math.inf
    return records


def _plain_records(answers):
    return [row for row in answers if row.get("stage") == "stage0"]


def run_stage0(model, tok, common, prompts, answers_path, out, device, done, answers):
    """Complete plain answers, gate them, then run P8 through the zero-write hook."""
    import torch
    by_prompt = {str(row.get("prompt_id")): row for row in _plain_records(answers)}
    stage0_path = out / "stage0.json"
    previous = json.loads(stage0_path.read_text(encoding="utf-8")) if stage0_path.exists() else {}
    zero_checks = list(previous.get("zero_checks", []))

    def save_stage0():
        saved = {**stage0_gates(prompts, _plain_records(answers), zero_checks),
            "zero_checks": zero_checks,
            "special_ids": sorted(run169.special_ids(tok) | set(map(int, common.eos_ids(tok, model)))),
            "prompts": [{k: p[k] for k in ("prompt_id", "group", "question_index", "kind", "question")}
                        for p in prompts]}
        write_json(stage0_path, saved)
        return saved

    p8 = [prompt for prompt in prompts if prompt["group"] == "p8"]

    def p8_failed():
        if not all(str(prompt["prompt_id"]) in by_prompt for prompt in p8):
            return False
        for prompt in p8:
            flags = stage0_flags(by_prompt[str(prompt["prompt_id"])])
            if flags["loop"] or flags["broken"]:
                return True
        return False

    if p8_failed():
        save_stage0()
        raise SystemExit("Gate 2 failed at Stage 0; no lists or push generations were run")
    for prompt in prompts:
        pid = str(prompt["prompt_id"])
        if pid in by_prompt:
            continue
        _prompt_ids, token_ids = run168.generate_plain(model, tok, prompt["question"], common,
                                                        device, MAX_NEW_TOKENS)
        record = {"stage": "stage0", **prompt, "arm": "plain", "dose": 0.0, "ban": "b0",
                  "prompt_ids": list(map(int, _prompt_ids)), "token_ids": list(map(int, token_ids)),
                  "answer": tok.decode(token_ids, skip_special_tokens=True).strip(),
                  "token_count": len(token_ids), "ended_eos_before_80": bool(token_ids and
                      int(token_ids[-1]) in set(map(int, common.eos_ids(tok, model))) and len(token_ids) < 80)}
        append_answers(answers_path, [record], done)
        answers.append(record)
        by_prompt[pid] = record
        if prompt["group"] == "p8" and p8_failed():
            save_stage0()
            raise SystemExit("Gate 2 failed at Stage 0; no lists or push generations were run")
    s0 = stage0_gates(prompts, _plain_records(answers), zero_checks)
    stage0_data = {**s0, "zero_checks": zero_checks,
                   "special_ids": sorted(run169.special_ids(tok) | set(map(int, common.eos_ids(tok, model)))),
                   "prompts": [{k: p[k] for k in ("prompt_id", "group", "question_index", "kind", "question")}
                               for p in prompts]}
    write_json(stage0_path, stage0_data)
    if not s0["gate2"]:
        raise SystemExit("Gate 2 failed at Stage 0; no lists or push generations were run")

    checked = {str(row.get("prompt_id")): row for row in zero_checks}
    layer = LAYERS[0]
    width = int(next(iter(common.find_layers(model)[layer].mlp.down_proj.weight.shape[1:2])))
    zero_dirs = {(layer, 0): {"n+": torch.zeros(width, dtype=torch.float32, device=device)}}
    for prompt in prompts:
        if prompt["group"] != "p8" or str(prompt["prompt_id"]) in checked:
            continue
        row = {"arm": "n+", "layer": layer, "slot": 0, "question_index": prompt["question_index"]}
        generated = generate_with_bans(model, tok, common.find_layers(model), prompt["question"],
            [row], zero_dirs, [[]], device, common, 170 + int(prompt["question_index"]),
            pushed=True, max_new_tokens=MAX_NEW_TOKENS)[0]
        checked[str(prompt["prompt_id"])] = {"prompt_id": prompt["prompt_id"], **generated}
        zero_checks = list(checked.values())
    s0 = stage0_gates(prompts, _plain_records(answers), zero_checks)
    stage0_data = {**s0, "zero_checks": zero_checks,
        "special_ids": sorted(run169.special_ids(tok) | set(map(int, common.eos_ids(tok, model)))),
        "prompts": [{k: p[k] for k in ("prompt_id", "group", "question_index", "kind", "question")}
                    for p in prompts]}
    write_json(stage0_path, stage0_data)
    if not s0["gate3"]:
        raise SystemExit("Gate 3 failed at Stage 0; no lists or push generations were run")
    return stage0_data


def make_stage1(model, tok, common, prompts, answers, out, device):
    import numpy as np
    import torch
    specials = run169.special_ids(tok) | set(map(int, common.eos_ids(tok, model)))
    plain = {str(row["prompt_id"]): row for row in _plain_records(answers)}
    current = {str(row["prompt_id"]): row for row in read_jsonl(out / "lists.jsonl")}
    stats = {}
    for prompt in prompts:
        pid = str(prompt["prompt_id"])
        stats[pid] = stage1_prompt(model, tok, prompt, plain[pid], common, device, specials, torch)
        if pid not in current:
            record = {key: value for key, value in stats[pid].items() if key != "_probabilities"}
            with (out / "lists.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                stream.flush()
            current[pid] = record
        # Older partial list files may predate the explicit release-list alias.
        current[pid]["rel50"] = list(map(int, current[pid]["b50"]))
    qprompts = [p for p in prompts if p["group"] == "q96"]
    for index, prompt in enumerate(qprompts):
        pid = str(prompt["prompt_id"]); next_prompt = qprompts[(index + 1) % len(qprompts)]
        next_ids = current[str(next_prompt["prompt_id"])]["top100"]
        own = current[pid]["b50"]
        other = other50_ids(next_ids, own)
        stats[pid]["other50"] = other
        stats[pid]["other50_mass"] = other50_mass(stats[pid]["_probabilities"], other)
        current[pid]["other50"] = other
        current[pid]["other50_mass"] = stats[pid]["other50_mass"]
    write_jsonl(out / "lists.jsonl", list(current.values()))
    return list(current.values()), stats, specials


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def run_no_push(model, tok, common, prompts, lists, answers_path, out, device, done, answers):
    listmap = {str(row["prompt_id"]): row for row in lists}
    for prompt in prompts:
        names = ["b0", "b5", "b50", "b200", "rel50"]
        if prompt["group"] == "q96":
            names.append("other50")
        keys = [answer_key({"stage": "no_push", "group": prompt["group"],
                "prompt_id": prompt["prompt_id"], "arm": "no_push", "dose": 0.0,
                "ban": name}) for name in names]
        if all(key in done for key in keys):
            continue
        rec = listmap[str(prompt["prompt_id"])]
        bans = [[] if name == "b0" else rec["b50"] if name == "rel50" else rec[name]
                for name in names]
        releases = [5 if name == "rel50" else None for name in names]
        batch_rows = [dict(prompt, arm="no_push", layer=-1, slot=-1) for _ in names]
        generated = generate_with_bans(model, tok, {}, prompt["question"], batch_rows, {}, bans,
            device, common, 170 + int(prompt["question_index"]), release_after=releases,
            pushed=False, max_new_tokens=MAX_NEW_TOKENS)
        new = []
        for name, banned, answer in zip(names, bans, generated):
            new.append({"stage": "no_push", **prompt, "arm": "no_push", "dose": 0.0,
                        "layer": None, "slot": None, "ban": name, "banned_ids": banned, **answer})
        answers.extend(append_answers(answers_path, new, done))


def _measure_directions_and_write(norm_path, model, tok, layers, prompts, base_directions,
                                  direction_records, prior, device, common):
    records = measure_doses(model, tok, layers, prompts, base_directions, direction_records,
                            device, common)
    write_json(norm_path, {"part": 170, "seed_base": 170000, "dose_factors": list(DOSES),
                           "directions": records,
                           "prior_neurons": {str(k): sorted(map(int, v)) for k, v in prior.items()}})
    return records


def run_push(model, tok, common, prompts, lists, base_directions, records, answers_path,
             out, device, done, answers, batch_size):
    listmap = {str(row["prompt_id"]): row for row in lists}
    push_prompts = [prompt for prompt in prompts if prompt["group"] == "p8"]
    pilot_started = time.perf_counter()
    pilot_answers = pilot_batches = 0
    for dose in DOSES:
        dirs = scaled_directions(base_directions, dose)
        for prompt in push_prompts:
            canonical = rows_for_question(prompt["question_index"])
            rec = listmap[str(prompt["prompt_id"])]
            for ban in PUSH_BANS + (("rel50",) if dose == 1.0 else ()):
                key_rows = [{"stage": "push", "prompt_id": prompt["prompt_id"], "group": "p8",
                    "arm": "n+", "dose": dose, "layer": row["layer"], "slot": row["slot"],
                    "ban": ban, "repeat": False} for row in canonical]
                if all(answer_key(row) in done for row in key_rows):
                    continue
                ban_ids = ([] if ban == "b0" else
                           rec["b50"] if ban == "rel50" else rec[ban])
                ids = [[] if ban == "b0" else ban_ids for _ in canonical]
                releases = [5 if ban == "rel50" else None for _ in canonical]
                generated = generate_with_bans(model, tok, common.find_layers(model), prompt["question"],
                    canonical, dirs, ids, device, common, 170 + int(prompt["question_index"]),
                    release_after=releases, pushed=True, max_new_tokens=MAX_NEW_TOKENS)
                new = []
                for row, banned, answer in zip(canonical, ids, generated):
                    new.append({"stage": "push", "prompt_id": prompt["prompt_id"], "group": "p8",
                        "question_index": prompt["question_index"], "question": prompt["question"],
                        "kind": prompt["kind"], "arm": "n+", "dose": dose,
                        "layer": int(row["layer"]), "slot": int(row["slot"]), "ban": ban,
                        "repeat": False, "banned_ids": banned, **answer})
                answers.extend(append_answers(answers_path, new, done))
                pilot_answers += len(canonical); pilot_batches += 1
                if pilot_batches == 3:
                    rate = pilot_answers / max(time.perf_counter() - pilot_started, 1e-9)
                    print(json.dumps({"pilot_answers_per_second": rate}), flush=True)
                    if run168.pilot_rate_should_stop(pilot_answers, pilot_started, time.perf_counter()):
                        raise SystemExit(f"pilot stop: {rate:.3f} answers/s below 1.5")

    # Gate 8: one fixed shuffled order; score170 maps every row back by prompt/layer/slot.
    all_rows = [{"prompt_id": prompt["prompt_id"], "question_index": prompt["question_index"],
                 "question": prompt["question"], "group": "p8", "arm": "n+", "layer": row["layer"],
                 "slot": row["slot"]} for prompt in push_prompts for row in rows_for_question(prompt["question_index"])]
    shuffled = repeat_row_order(all_rows, 170)
    grouped = {prompt["prompt_id"]: [] for prompt in push_prompts}
    for row in shuffled:
        grouped[row["prompt_id"]].append(row)
    for prompt in push_prompts:
        canonical = grouped[prompt["prompt_id"]]
        key_rows = [{"stage": "push_repeat", "prompt_id": prompt["prompt_id"], "group": "p8",
            "arm": "n+", "dose": 1.0, "layer": row["layer"], "slot": row["slot"],
            "ban": "b0", "repeat": True} for row in canonical]
        if all(answer_key(row) in done for row in key_rows):
            continue
        dirs = scaled_directions(base_directions, 1.0)
        generated = generate_with_bans(model, tok, common.find_layers(model), prompt["question"],
            canonical, dirs, [[] for _ in canonical], device, common, 1710,
            pushed=True, max_new_tokens=MAX_NEW_TOKENS)
        new = []
        for row, answer in zip(canonical, generated):
            new.append({"stage": "push_repeat", "prompt_id": row["prompt_id"], "group": "p8",
                "question_index": row["question_index"], "question": row["question"],
                "arm": "n+", "dose": 1.0, "layer": int(row["layer"]), "slot": int(row["slot"]),
                "ban": "b0", "repeat": True, "banned_ids": [],
                "repeat_shuffle_index": shuffled.index(row), **answer})
        answers.extend(append_answers(answers_path, new, done))
    write_json(out / "repeat_order.json", [{"prompt_id": row["prompt_id"], "layer": row["layer"],
        "slot": row["slot"], "shuffle_index": index} for index, row in enumerate(shuffled)])


def run(args, loader=None):
    import torch
    common = run155._common()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    answers_path = out / "answers.jsonl"
    repair_tail(answers_path)
    repair_tail(out / "lists.jsonl")
    answers = read_jsonl(answers_path)
    done = {answer_key(row) for row in answers}
    prompts = prompt_catalog()
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common.REVISION, args.device)
        model.requires_grad_(False); model.eval()
    else:
        tok, model = loader(args.device)

    stage0 = run_stage0(model, tok, common, prompts, answers_path, out, args.device, done, answers)
    # Stage 1 and every no-push list answer precede direction construction and push generation.
    lists, _stats, _specials = make_stage1(model, tok, common, prompts, answers, out, args.device)
    run_no_push(model, tok, common, prompts, lists, answers_path, out, args.device, done, answers)

    medians_path = ROOT / "results/part112/medians.json"
    medians = json.loads(medians_path.read_text(encoding="utf-8"))
    layers = common.find_layers(model)
    directions, direction_records, prior, _widths = make_directions(layers, medians, torch, SLOTS)
    norms_path = out / "norms.json"
    measured = _measure_directions_and_write(norms_path, model, tok, layers, prompts,
        directions, direction_records, prior, args.device, common)
    pre_gates = {"gate4": gate4_size({"directions": measured}),
                 "gate7": gate7_freshness({"directions": measured,
                    "prior_neurons": {str(k): sorted(map(int, v)) for k, v in prior.items()}})}
    write_json(out / "pre_push_gates.json", pre_gates)
    if not pre_gates["gate4"]["passed"] or not pre_gates["gate7"]["passed"]:
        raise SystemExit(f"Gate 4 or Gate 7 failed before push answers: {pre_gates}")
    run_push(model, tok, common, prompts, lists, directions, measured, answers_path, out,
             args.device, done, answers, args.batch_size)
    gates = gate_rules(prompts, read_jsonl(answers_path), stage0,
        read_jsonl(out / "lists.jsonl"), json.loads(norms_path.read_text(encoding="utf-8")))
    write_json(out / "prelaunch_gates.json", gates)
    return {"answers": len(read_jsonl(answers_path)), "prompts": len(prompts), "gates": gates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=run165.MODEL_DEFAULT)
    parser.add_argument("--out-dir", default="results/part170")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered batch size is 64")
    print(json.dumps(run(args), ensure_ascii=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
