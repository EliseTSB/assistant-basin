#!/usr/bin/env python3
"""Part 33: natural-necessity experiment for Qwen3.5-4B layer 22 neuron 2742.

Passage and question inputs are fixed. Passage membership is chosen from measured
activations: the alien set must fire above half the selected-corpus maximum, and
the topic-matched and neutral sets must remain at or below that threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import secrets
import sys
import time
from pathlib import Path

import numpy as np


MODEL = "Qwen/Qwen3.5-4B"
REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
LAYER = 22
NEURON = 2742
AUTHOR = "Authored by Codex for Part 33; fixed candidate, selected only by measured activation."
LOOKAHEAD = 8
CONTROL_COUNT = 20


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: str | Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _dump(obj, path: str | Path) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, path)


def conditional_mask(activations, corpus_max: float, fraction: float = 0.5) -> np.ndarray:
    """Boolean mask for strict activation > fraction * natural corpus maximum."""
    values = np.asarray(activations, dtype=np.float64)
    return values > (float(corpus_max) * float(fraction))


def apply_conditional_zero(activations, mask) -> np.ndarray:
    """Return a copy with only true-mask positions zeroed; useful for audit/tests."""
    values = np.asarray(activations).copy()
    selected = np.asarray(mask, dtype=bool)
    if values.shape != selected.shape:
        raise ValueError("activation and conditional mask shapes differ")
    values[selected] = 0
    return values


def random_neuron_pool(n_units: int, target: int, n: int, seed: int) -> list[int]:
    """Deterministic random layer units, excluding the target."""
    pool = [u for u in range(int(n_units)) if u != int(target)]
    if n < 0 or n > len(pool):
        raise ValueError(f"requested {n} candidate controls from only {len(pool)} units")
    rng = random.Random(seed)
    return sorted(rng.sample(pool, n))


def match_firing_frequency(target_rate: float, rates: dict[int, float], target: int,
                           n: int = CONTROL_COUNT, seed: int = 0) -> list[int]:
    """Choose the closest firing-rate units from a seeded random candidate pool.

    Randomized tie order keeps selection reproducible without making unit index a
    hidden criterion. All rates use each neuron's own half-maximum threshold.
    """
    candidates = [int(u) for u in rates if int(u) != int(target)]
    if len(candidates) < n:
        raise ValueError(f"need {n} random candidates, found {len(candidates)}")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return sorted(candidates, key=lambda u: abs(float(rates[u]) - float(target_rate)))[:n]


def loss_window_positions(firing_mask, n_losses: int, lookahead: int = LOOKAHEAD):
    """Loss indices at firing tokens and in the next lookahead positions.

    Index p is the next-token loss predicted by the logits at token p. The final
    input token has no next-token target and is excluded. Overlapping windows are
    unioned, so a loss token is counted once per passage.
    """
    mask = np.asarray(firing_mask, dtype=bool).reshape(-1)
    at = sorted(int(p) for p in np.flatnonzero(mask) if int(p) < int(n_losses))
    after = sorted({p + offset for p in at for offset in range(1, lookahead + 1)
                    if p + offset < int(n_losses)})
    window = sorted(set(at) | set(after))
    return {"at": at, "after": after, "window": window}


def loss_window_delta(baseline_losses, changed_losses, firing_mask,
                      lookahead: int = LOOKAHEAD) -> dict:
    base = np.asarray(baseline_losses, dtype=np.float64).reshape(-1)
    changed = np.asarray(changed_losses, dtype=np.float64).reshape(-1)
    if base.shape != changed.shape:
        raise ValueError("baseline and changed loss vectors differ in shape")
    positions = loss_window_positions(firing_mask, len(base), lookahead)
    delta = changed - base
    def mean(which):
        idx = positions[which]
        return float(np.mean(delta[idx])) if idx else None
    return {"at_firing": mean("at"), "after_1_to_8": mean("after"),
            "window": mean("window"), "firing_loss_tokens": len(positions["at"]),
            "after_loss_tokens": len(positions["after"]),
            "window_loss_tokens": len(positions["window"]),
            "positions": positions,
            "at_delta_sum": float(np.sum(delta[positions["at"]])) if positions["at"] else 0.0,
            "after_delta_sum": float(np.sum(delta[positions["after"]])) if positions["after"] else 0.0,
            "window_delta_sum": float(np.sum(delta[positions["window"]])) if positions["window"] else 0.0}


def _as_rows(value):
    """Convert a tensor or array to nested Python lists without importing torch."""
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
        return value.tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _last_token_logits(logits):
    rows = _as_rows(logits)
    while isinstance(rows, list) and rows and isinstance(rows[0], list) and rows[0] and isinstance(rows[0][0], list):
        rows = rows[0]
    if not rows:
        raise ValueError("empty logits")
    row = rows[-1] if isinstance(rows[0], list) else rows
    return [float(x) for x in row]


def score_letter_logits(model, input_ids, option_token_ids: dict[str, int],
                        attention_mask=None, device: str | None = None) -> dict[str, float]:
    """Run a model once and return answer-letter logits at the final prompt position."""
    kwargs = {"input_ids": input_ids, "use_cache": False}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    if device is not None and isinstance(input_ids, list):
        kwargs["input_ids"] = input_ids
    output = model(**kwargs)
    logits = _last_token_logits(output.logits)
    if any(int(token) < 0 or int(token) >= len(logits) for token in option_token_ids.values()):
        raise ValueError("answer-letter token ID is outside the model vocabulary")
    return {str(letter): float(logits[int(token)]) for letter, token in option_token_ids.items()}


def score_mc_questions(model, examples: list[dict], option_token_ids: dict[str, int],
                       input_builder, attention_builder=None) -> list[dict]:
    """Score multiple-choice items from answer-letter logits (works with fake models)."""
    rows = []
    for item in examples:
        ids = input_builder(item)
        attention = attention_builder(ids) if attention_builder else None
        logits = score_letter_logits(model, ids, option_token_ids, attention)
        prediction = max(logits, key=logits.get)
        rows.append({"id": item["id"], "logits": logits, "prediction": prediction,
                     "answer": item["answer"], "correct": prediction == item["answer"]})
    return rows


def _cross_entropy_numpy(logits, input_ids) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(input_ids, dtype=np.int64)
    if values.ndim == 2:
        values = values[None, ...]
    if tokens.ndim == 1:
        tokens = tokens[None, ...]
    if values.shape[:2] != tokens.shape or values.shape[1] < 2:
        raise ValueError("logits and input ids must have matching [batch, sequence] shapes")
    shifted = values[:, :-1, :]
    target = tokens[:, 1:]
    maxes = np.max(shifted, axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted - maxes).sum(axis=-1)) + maxes[..., 0]
    chosen = np.take_along_axis(shifted, target[..., None], axis=-1)[..., 0]
    return log_z - chosen


def causal_next_token_losses(logits, input_ids) -> list[list[float]]:
    """Per-position causal loss; uses torch on runtime tensors, NumPy for fake tests."""
    if hasattr(logits, "detach"):
        import torch.nn.functional as F
        ids = input_ids if hasattr(input_ids, "detach") else __import__("torch").as_tensor(input_ids)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        batch, seq, vocab = logits.shape
        losses = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, vocab),
                                 ids[:, 1:].reshape(-1), reduction="none").reshape(batch, seq - 1)
        return losses.detach().cpu().tolist()
    return _cross_entropy_numpy(logits, input_ids).tolist()


def load_candidates(authored_path: str, sense_path: str, neutral_path: str,
                    neutral_limit: int = 40) -> list[dict]:
    rows = []
    with open(authored_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("group") not in {"alien", "topic"} or not row.get("text"):
                raise ValueError(f"invalid authored passage row: {row}")
            rows.append({**row, "source": "data/part33-authored-passages.jsonl"})
    group_map = {"a": "alien", "d": "alien", "e": "topic", "g": "neutral"}
    with open(sense_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            group = group_map.get(record.get("group"))
            if group:
                rows.append({"id": f"sense_{record['group']}_{i:02d}", "group": group,
                             "text": record["text"], "source": "data/sense-battery.jsonl"})
    paragraphs = []
    with open(neutral_path, encoding="utf-8") as f:
        for paragraph in re.split(r"\n\s*\n", f.read()):
            text = " ".join(paragraph.split())
            if len(text.split()) < 15 or len(text.split()) > 140:
                continue
            if "[Illustration" in text or text.upper().startswith("CHAPTER"):
                continue
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("no readable neutral passages found in data/neutral.txt")
    count = min(int(neutral_limit), len(paragraphs))
    positions = sorted({round(i * (len(paragraphs) - 1) / max(1, count - 1)) for i in range(count)})
    for j, pos in enumerate(positions):
        rows.append({"id": f"neutral_book_{j + 1:02d}", "group": "neutral",
                     "text": paragraphs[pos], "source": "data/neutral.txt"})
    ids = [r["id"] for r in rows]
    texts = [r["text"] for r in rows]
    if len(ids) != len(set(ids)) or len(texts) != len(set(texts)):
        raise ValueError("candidate passage IDs and text must be unique")
    counts = {g: sum(r["group"] == g for r in rows) for g in ("alien", "topic", "neutral")}
    if any(v < 30 for v in counts.values()):
        raise ValueError(f"need at least 30 candidates per group; found {counts}")
    return rows


def _model_input(tok, text: str, device: str, max_length: int = 512):
    encoded = tok(text, return_tensors="pt", add_special_tokens=True,
                  truncation=True, max_length=max_length)
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask")
    if mask is None:
        mask = ids.new_ones(ids.shape)
    else:
        mask = mask.to(device)
    return ids, mask


def _capture_activation_pass(model, down_proj, ids, attention, unit_ids):
    captured = []
    indices = None
    def hook(_module, args):
        nonlocal indices
        x = args[0]
        if indices is None:
            indices = __import__("torch").as_tensor(unit_ids, dtype=__import__("torch").long,
                                                      device=x.device)
        captured.append(x[0].index_select(-1, indices).detach().float().cpu().numpy())
        return None
    handle = down_proj.register_forward_pre_hook(hook)
    try:
        import torch
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=attention, use_cache=False)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("activation hook did not run")
    return output, np.asarray(captured[-1], dtype=np.float32)


def _load_model_local(model_name: str, revision: str, device: str):
    import torch
    if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        raise RuntimeError("--device xpu requested but torch.xpu is unavailable")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                     dtype=torch.bfloat16, local_files_only=True)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, revision=revision,
                                                             dtype=torch.bfloat16, local_files_only=True)
    model.to(device).eval()
    torch.set_grad_enabled(False)
    return tok, model


def _candidate_cache_key(row: dict, units: list[int], model_name: str, revision: str,
                         layer: int, tokenizer_id: str, device: str,
                         max_length: int) -> str:
    payload = json.dumps({"id": row["id"], "text": row["text"], "units": units,
                          "model": model_name, "revision": revision, "layer": layer,
                          "tokenizer": tokenizer_id, "device": device,
                          "max_length": max_length, "hook": "down_proj-input-v1"},
                         sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _sha_bytes(payload)


def _save_activation_cache(path: Path, key: str, input_ids, activations, unit_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{secrets.token_hex(4)}.tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, cache_key=np.asarray(key), input_ids=np.asarray(input_ids, dtype=np.int32),
                            activations=np.asarray(activations, dtype=np.float32),
                            unit_ids=np.asarray(unit_ids, dtype=np.int32))
    os.replace(tmp, path)


def _candidate_activations(row, tok, model, down_proj, unit_ids, args, cache_dir):
    key = _candidate_cache_key(row, unit_ids, args.model, args.revision, args.layer,
                               getattr(tok, "name_or_path", args.model), args.device,
                               args.max_length)
    cache = cache_dir / f"{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["cache_key"].item()) == key and saved["unit_ids"].tolist() == unit_ids:
                return saved["input_ids"].astype(np.int64), saved["activations"].astype(np.float32), True
    ids, attention = _model_input(tok, row["text"], args.device, args.max_length)
    output, activations = _capture_activation_pass(model, down_proj, ids, attention, unit_ids)
    input_ids = ids[0].detach().cpu().numpy().astype(np.int64)
    _save_activation_cache(cache, key, input_ids, activations, unit_ids)
    del output
    return input_ids, activations, False


def _activation_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"max": float(np.max(values)), "mean": float(np.mean(values)),
            "min": float(np.min(values)), "tokens": int(values.size)}


def available_system_memory_gb() -> float | None:
    """Available host RAM on Windows/Linux; returns None only if the OS API is absent."""
    if os.name == "nt":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024 ** 3)
    try:
        with open("/proc/meminfo", encoding="ascii") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        return None


def system_memory_guard(min_available_gb: float):
    available = available_system_memory_gb()
    if available is not None and available < float(min_available_gb):
        raise RuntimeError(f"host RAM available {available:.2f} GiB is below --min-available-ram-gb {min_available_gb:.2f}; stopping before next model pass")


def select_passages(candidates: list[dict], activations: dict[str, np.ndarray],
                    target_col: int, questions: list[dict], per_group: int = 30) -> tuple[list[dict], float]:
    """Select firing alien passages and low topic/neutral passages by measured values."""
    alien = [r for r in candidates if r["group"] == "alien"]
    target_max = max(float(np.max(activations[r["id"]][:, target_col])) for r in alien)
    if not math.isfinite(target_max) or target_max <= 0:
        raise ValueError(f"alien candidate corpus maximum is not positive: {target_max}")
    threshold = 0.5 * target_max
    by_id = {r["id"]: r for r in candidates}
    question_ids = [q["passage_id"] for q in questions]
    if len(question_ids) != 20 or len(set(question_ids)) != 20:
        raise ValueError("the fixed comprehension battery must contain 20 unique passage IDs")
    if any(qid not in by_id or by_id[qid]["group"] != "alien" for qid in question_ids):
        raise ValueError("every question must point to one alien candidate passage")
    maxima = {r["id"]: float(np.max(activations[r["id"]][:, target_col])) for r in candidates}
    firing_alien = [r for r in alien if maxima[r["id"]] > threshold]
    if len(firing_alien) < per_group:
        raise ValueError(f"only {len(firing_alien)} alien candidates fire above half-maximum; need {per_group}")
    failing_questions = [qid for qid in question_ids if maxima[qid] <= threshold]
    if failing_questions:
        raise ValueError("question passages do not all fire naturally above threshold: " + ", ".join(failing_questions))
    selected = [by_id[qid] for qid in question_ids]
    selected_ids = set(question_ids)
    remaining = sorted((r for r in firing_alien if r["id"] not in selected_ids),
                       key=lambda r: (-maxima[r["id"]], r["id"]))
    selected.extend(remaining[:per_group - len(selected)])
    if len(selected) != per_group:
        raise ValueError("could not assemble 30 alien firing passages")
    for group in ("topic", "neutral"):
        low = [r for r in candidates if r["group"] == group and maxima[r["id"]] <= threshold]
        if len(low) < per_group:
            raise ValueError(f"only {len(low)} {group} passages stay at or below half-maximum; need {per_group}")
        low.sort(key=lambda r: (maxima[r["id"]], r["id"]))
        selected.extend(low[:per_group])
    selected_max = max(maxima[r["id"]] for r in selected)
    if not math.isclose(selected_max, target_max, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError("the top measured alien candidate must define the selected corpus maximum")
    return selected, target_max


class ConditionalZeroHook:
    """Down-projection pre-hook; row i zeros exactly unit/mask pair i."""
    def __init__(self, down_proj, conditions):
        self.conditions = [(int(unit), np.asarray(mask, dtype=bool)) for unit, mask in conditions]
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        import torch
        x = args[0]
        if x.shape[0] != len(self.conditions):
            raise ValueError("conditional hook batch does not match its intervention rows")
        out = x.clone()
        for row, (unit, mask) in enumerate(self.conditions):
            if len(mask) != x.shape[1]:
                raise ValueError("conditional mask length does not match token sequence")
            active = torch.as_tensor(mask, device=x.device, dtype=torch.bool)
            out[row, active, unit] = 0
        return (out,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


def _question_file(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    questions = payload.get("questions")
    if not isinstance(questions, list) or len(questions) != 20:
        raise ValueError("Part 33 requires exactly 20 saved multiple-choice questions")
    for q in questions:
        if set(q.get("choices", {})) != {"A", "B", "C", "D"} or q.get("answer") not in "ABCD":
            raise ValueError(f"invalid multiple-choice item {q.get('id')}")
    return questions


def _question_prompt(passage: str, q: dict) -> str:
    options = "\n".join(f"{letter}. {q['choices'][letter]}" for letter in "ABCD")
    return ("Read the passage and answer the question.\n\nPassage: " + passage +
            "\n\nQuestion: " + q["question"] + "\n" + options + "\nAnswer:")


def _letter_ids(tok, prompt: str) -> dict[str, int]:
    prefix = tok(prompt, add_special_tokens=True)["input_ids"]
    if hasattr(prefix, "tolist"):
        prefix = prefix.tolist()
    if prefix and isinstance(prefix[0], list):
        prefix = prefix[0]
    out = {}
    for letter in "ABCD":
        joined = tok(prompt + " " + letter, add_special_tokens=True)["input_ids"]
        if hasattr(joined, "tolist"):
            joined = joined.tolist()
        if joined and isinstance(joined[0], list):
            joined = joined[0]
        suffix = joined[len(prefix):] if joined[:len(prefix)] == prefix else []
        if len(suffix) != 1:
            raise ValueError(f"answer choice {letter} is not exactly one continuation token")
        out[letter] = int(suffix[0])
    if len(set(out.values())) != 4:
        raise ValueError("answer-letter token IDs are not distinct")
    return out


def _one_model_logits(model, ids, attention):
    import torch
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=attention, use_cache=False).logits


def _run_question(model, down_proj, tok, q, passage, threshold, device,
                  min_available_ram_gb, max_gpu_gb):
    import neuron_report as nr
    text = _question_prompt(passage, q)
    ids, attention = _model_input(tok, text, device)
    system_memory_guard(min_available_ram_gb)
    # The capture pass returns the model OUTPUT object, not logits (cloud smoke 2026-10-20: indexing the output
    # raised "tuple indices must be integers").
    base_output, acts = _capture_activation_pass(model, down_proj, ids, attention, [NEURON])
    base_logits = base_output.logits
    letters = _letter_ids(tok, text)
    base_scores = {label: float(base_logits[0, -1, token].detach().float().cpu().item())
                   for label, token in letters.items()}
    mask = conditional_mask(acts[:, 0], threshold)
    hook = ConditionalZeroHook(down_proj, [(NEURON, mask)])
    try:
        system_memory_guard(min_available_ram_gb)
        changed = _one_model_logits(model, ids, attention)
    finally:
        hook.remove()
    changed_scores = {label: float(changed[0, -1, token].detach().float().cpu().item())
                      for label, token in letters.items()}
    nr.spill_guard(device, max_gpu_gb)
    base_pred = max(base_scores, key=base_scores.get)
    changed_pred = max(changed_scores, key=changed_scores.get)
    return {"id": q["id"], "passage_id": q["passage_id"], "answer": q["answer"],
            "baseline_logits": base_scores, "conditional_zero_logits": changed_scores,
            "baseline_prediction": base_pred, "conditional_zero_prediction": changed_pred,
            "baseline_correct": base_pred == q["answer"],
            "conditional_zero_correct": changed_pred == q["answer"],
            "natural_firing_tokens": int(np.count_nonzero(mask))}


def _effect_for_condition(base_losses, changed_losses, mask):
    return loss_window_delta(base_losses, changed_losses, mask)


def _intervention_losses(model, down_proj, ids, attention, conditions, args):
    import torch
    all_losses = []
    for start in range(0, len(conditions), args.intervention_batch):
        system_memory_guard(args.min_available_ram_gb)
        chunk = conditions[start:start + args.intervention_batch]
        n = len(chunk)
        batch_ids = ids.expand(n, -1).contiguous()
        batch_attention = attention.expand(n, -1).contiguous()
        hook = ConditionalZeroHook(down_proj, chunk)
        try:
            with torch.inference_mode():
                output = model(input_ids=batch_ids, attention_mask=batch_attention, use_cache=False)
            all_losses.extend(causal_next_token_losses(output.logits, batch_ids))
            del output
        finally:
            hook.remove()
    return all_losses


def _measure_effects(args, model, tok, layers, selected, activations, target_max,
                     control_ids, output_dir):
    import neuron_report as nr
    target_idx = 0
    total_tokens = sum(len(activations[r["id"]]) for r in selected)
    max_by_unit = {NEURON: max(float(np.max(activations[r["id"]][:, target_idx])) for r in selected)}
    count_by_unit = {NEURON: sum(int(np.count_nonzero(activations[r["id"]][:, target_idx] > target_max * .5))
                                 for r in selected)}
    target_rate = count_by_unit[NEURON] / total_tokens
    rate_by_unit = {NEURON: target_rate}
    for col, unit in enumerate(control_ids, 1):
        maximum = max(float(np.max(activations[r["id"]][:, col])) for r in selected)
        max_by_unit[unit] = maximum
        count_by_unit[unit] = sum(int(np.count_nonzero(activations[r["id"]][:, col] > maximum * .5))
                                  for r in selected)
        rate_by_unit[unit] = count_by_unit[unit] / total_tokens
    matched = match_firing_frequency(target_rate, rate_by_unit, NEURON, CONTROL_COUNT, args.seed)
    unit_cols = {NEURON: target_idx, **{unit: i + 1 for i, unit in enumerate(control_ids)}}
    frequency_rows = [{"unit": unit, "firing_tokens": count_by_unit[unit], "token_count": total_tokens,
                       "firing_rate": rate_by_unit[unit], "max_activation": max_by_unit[unit],
                       "absolute_rate_difference": abs(rate_by_unit[unit] - target_rate),
                       "selected": unit in matched}
                      for unit in control_ids]

    signature_payload = json.dumps({"model": args.model, "revision": args.revision,
                                    "selected": [(r["id"], r["group"]) for r in selected],
                                    "units": [NEURON] + matched, "target_max": target_max,
                                    "lookahead": LOOKAHEAD}, sort_keys=True).encode()
    signature = _sha_bytes(signature_payload)
    path = output_dir / "passage-effects.json"
    cached = {}
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("signature") == signature:
                cached = saved.get("rows", {})
        except (OSError, json.JSONDecodeError):
            cached = {}
    result_rows = dict(cached)
    down_proj = layers[args.layer].mlp.down_proj
    for i, row in enumerate(selected, 1):
        if row["id"] in result_rows:
            print(f"effects {i:02d}/{len(selected)} reuse {row['id']}", flush=True)
            continue
        ids_np = np.asarray(activations[row["id"]])
        masks = {}
        for unit in [NEURON] + matched:
            col = unit_cols[unit]
            masks[unit] = conditional_mask(ids_np[:, col], max_by_unit[unit])
        system_memory_guard(args.min_available_ram_gb)
        ids, attention = _model_input(tok, row["text"], args.device, args.max_length)
        baseline_logits = _one_model_logits(model, ids, attention)
        base_losses = causal_next_token_losses(baseline_logits, ids)[0]
        conditions = [(unit, masks[unit]) for unit in [NEURON] + matched]
        changed = _intervention_losses(model, down_proj, ids, attention, conditions, args)
        record = {}
        for unit, losses in zip([NEURON] + matched, changed):
            computed = _effect_for_condition(base_losses, losses, masks[unit])
            metric = {k: v for k, v in computed.items() if k != "positions"}
            record[str(unit)] = {key: metric for key, metric in
                                 (("at", {"at_firing": metric["at_firing"],
                                          "firing_loss_tokens": metric["firing_loss_tokens"],
                                          "delta_sum": metric["at_delta_sum"]}),
                                  ("after", {"after_1_to_8": metric["after_1_to_8"],
                                              "after_loss_tokens": metric["after_loss_tokens"],
                                              "delta_sum": metric["after_delta_sum"]}),
                                  ("window", {"window": metric["window"],
                                               "window_loss_tokens": metric["window_loss_tokens"],
                                               "delta_sum": metric["window_delta_sum"]}))}
        result_rows[row["id"]] = record
        _dump({"signature": signature, "matched_controls": matched, "rows": result_rows}, path)
        print(f"effects {i:02d}/{len(selected)} {row['id']}", flush=True)
        nr.spill_guard(args.device, args.max_gpu_gb)
        del baseline_logits
    grouped = {g: {} for g in ("all", "alien", "topic", "neutral")}
    by_id = {r["id"]: r for r in selected}
    for unit in [NEURON] + matched:
        key = str(unit)
        for group in grouped:
            group_rows = [result_rows[r["id"]][key] for r in selected
                          if group == "all" or by_id[r["id"]]["group"] == group]
            values = {}
            for metric, slot in (("at_firing_loss_increase", "at"),
                                 ("after_1_to_8_loss_increase", "after"),
                                 ("window_loss_increase", "window")):
                count_key = "firing_loss_tokens" if slot == "at" else (
                    "after_loss_tokens" if slot == "after" else "window_loss_tokens")
                value_key = "at_firing" if slot == "at" else (
                    "after_1_to_8" if slot == "after" else "window")
                numerator = sum(x[slot]["delta_sum"] for x in group_rows if x[slot][count_key])
                denominator = sum(x[slot][count_key] for x in group_rows)
                values[metric] = float(numerator / denominator) if denominator else None
                values["n_" + slot + "_loss_tokens"] = denominator
            grouped[group][key] = values
    return {"matched_controls": matched, "candidate_control_frequencies": frequency_rows,
            "target_firing_frequency": target_rate, "effects_by_group": grouped,
            "passage_effects_signature": signature}


def _run_comprehension(args, model, tok, layers, questions, selected, target_max, output_dir):
    passage_map = {r["id"]: r["text"] for r in selected}
    signature = _sha_bytes(json.dumps({"model": args.model, "revision": args.revision,
                                      "threshold": target_max * .5,
                                      "questions": [(q["id"], q["passage_id"]) for q in questions],
                                      "selected": [r["id"] for r in selected]},
                                     sort_keys=True).encode())
    progress_path = Path(output_dir) / "comprehension-progress.json"
    completed = {}
    if progress_path.exists():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if saved.get("signature") == signature:
                completed = saved.get("items", {})
        except (OSError, json.JSONDecodeError):
            completed = {}
    rows = []
    for i, q in enumerate(questions, 1):
        passage = passage_map.get(q["passage_id"])
        if passage is None:
            raise ValueError(f"question passage {q['passage_id']} was not selected")
        if q["id"] in completed:
            rows.append(completed[q["id"]])
            print(f"comprehension {i:02d}/{len(questions)} {q['id']} (cached)", flush=True)
            continue
        if i <= 3 or i == len(questions):
            print(f"comprehension {i:02d}/{len(questions)} {q['id']}", flush=True)
        system_memory_guard(args.min_available_ram_gb)
        row = _run_question(model, layers[args.layer].mlp.down_proj, tok, q,
                            passage, target_max * .5, args.device,
                            args.min_available_ram_gb, args.max_gpu_gb)
        rows.append(row)
        completed[q["id"]] = row
        _dump({"signature": signature, "items": completed}, progress_path)
    n = len(rows)
    base_correct = sum(r["baseline_correct"] for r in rows)
    zero_correct = sum(r["conditional_zero_correct"] for r in rows)
    return {"items": rows, "n": n, "baseline_correct": base_correct,
            "conditional_zero_correct": zero_correct,
            "baseline_accuracy": base_correct / n if n else None,
            "conditional_zero_accuracy": zero_correct / n if n else None,
            "accuracy_change_points": 100.0 * (zero_correct - base_correct) / n if n else None}


def _percentile(values, pct):
    return float(np.percentile(np.asarray(values, dtype=float), pct)) if values else None


def score_results(result: dict) -> dict:
    effect = result["conditional_zeroing"]["effects_by_group"]["all"]
    target = effect[str(NEURON)]["window_loss_increase"]
    controls = [effect[str(u)]["window_loss_increase"] for u in result["conditional_zeroing"]["matched_controls"]]
    controls = [x for x in controls if x is not None]
    random_mean = float(np.mean(controls)) if controls else None
    p95 = _percentile(controls, 95)
    acc_change = result["comprehension"]["accuracy_change_points"]
    return {"33.1_detector_rule": (target is not None and random_mean is not None and target <= random_mean),
            "33.1_target_window_loss": target, "33.1_matched_random_mean": random_mean,
            "33.2_exceeds_matched_random_p95": (target is not None and p95 is not None and target > p95),
            "33.2_matched_random_p95": p95,
            "33.2_matched_random_distribution": controls,
            "33.3_abs_accuracy_change_under_10_points": (acc_change is not None and abs(acc_change) < 10),
            "33.3_accuracy_change_points": acc_change}


def print_score(result: dict):
    score = result["prediction_scores"]
    print("Part 33 prediction scores (target and matched-random loss increase use the union of firing positions and next 8 positions):")
    print(f"33.1 detector rule: target={score['33.1_target_window_loss']}; matched-random mean={score['33.1_matched_random_mean']}; target<=mean={score['33.1_detector_rule']}")
    print(f"33.2 target exceeds matched-random 95th percentile: {score['33.2_exceeds_matched_random_p95']}; target={score['33.1_target_window_loss']}; p95={score['33.2_matched_random_p95']}; distribution={score['33.2_matched_random_distribution']}")
    print(f"33.3 comprehension change: {score['33.3_accuracy_change_points']} points; abs(change)<10={score['33.3_abs_accuracy_change_under_10_points']}")


def _load_sources(args):
    candidates = load_candidates(args.passages, args.sense_battery, args.neutral_source,
                                 args.neutral_candidates)
    questions = _question_file(args.questions)
    return candidates, questions


def run(args):
    import torch
    import neuron_report as nr
    import part24 as p24

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "activation-cache"
    candidates, questions = _load_sources(args)
    if args.smoke:
        qids = [f"alien_q{i:02d}" for i in range(1, args.smoke_passages + 1)]
        by_id = {r["id"]: r for r in candidates}
        candidates = [by_id[qid] for qid in qids]
        questions = questions[:args.smoke_passages]
    system_memory_guard(args.min_available_ram_gb)
    tok, model = _load_model_local(args.model, args.revision, args.device)
    layers_name, layers, _norm = nr.find_text_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"layer {args.layer} is outside model with {len(layers)} layers")
    down_proj = layers[args.layer].mlp.down_proj
    width = int(down_proj.in_features)
    if args.neuron >= width:
        raise ValueError(f"neuron {args.neuron} is outside layer width {width}")
    pool = random_neuron_pool(width, args.neuron, min(args.control_pool, width - 1), args.seed)
    unit_ids = [args.neuron] + pool
    print(f"Loaded {args.model}@{args.revision}: {len(layers)} layers, layer width {width}; {len(unit_ids)} units captured", flush=True)

    input_paths = [args.passages, args.sense_battery, args.neutral_source, args.questions]
    input_shas = {os.path.basename(p): p24.sha(p) for p in input_paths}
    cache_hits = 0
    measurements = {}
    for i, row in enumerate(candidates, 1):
        system_memory_guard(args.min_available_ram_gb)
        ids, values, reused = _candidate_activations(row, tok, model, down_proj, unit_ids,
                                                     args, cache_dir)
        cache_hits += int(reused)
        measurements[row["id"]] = values
        if i <= 3 or i == len(candidates) or i % 20 == 0:
            print(f"natural activations {i:03d}/{len(candidates)} {row['id']}" + (" (cached)" if reused else ""), flush=True)
        nr.spill_guard(args.device, args.max_gpu_gb)

    if args.smoke:
        selected = candidates
        target_max = max(float(np.max(measurements[r["id"]][:, 0])) for r in selected)
        threshold = target_max * .5
        if any(float(np.max(measurements[r["id"]][:, 0])) <= threshold for r in selected):
            print("SMOKE note: a requested passage did not exceed half of the 3-passage maximum", flush=True)
    else:
        selected, target_max = select_passages(candidates, measurements, 0, questions)
        threshold = target_max * .5
    selected_measurements = {r["id"]: measurements[r["id"]] for r in selected}
    total_tokens = sum(len(selected_measurements[r["id"]]) for r in selected)
    natural_rows = []
    for row in selected:
        target_values = selected_measurements[row["id"]][:, 0]
        natural_rows.append({"id": row["id"], "group": row["group"], "source": row.get("source"),
                             "text": row["text"], "activation": _activation_summary(target_values),
                             "firing_tokens": int(np.count_nonzero(target_values > threshold)),
                             "firing_fraction": float(np.count_nonzero(target_values > threshold) / len(target_values))})
    selected_max = max(x["activation"]["max"] for x in natural_rows)
    if not args.smoke and not math.isclose(selected_max, target_max, rel_tol=1e-6, abs_tol=1e-6):
        raise AssertionError("stored natural activation summaries disagree with selected corpus maximum")
    print(f"Selected {len(selected)} passages; corpus max={selected_max:.6g}, threshold={threshold:.6g}; cache hits={cache_hits}", flush=True)
    natural = {"corpus_maximum": selected_max, "conditional_threshold": threshold,
               "token_count": total_tokens, "per_passage": natural_rows,
               "candidate_counts": {g: sum(r["group"] == g for r in candidates)
                                    for g in ("alien", "topic", "neutral")},
               "selected_counts": {g: sum(r["group"] == g for r in selected)
                                   for g in ("alien", "topic", "neutral")}}
    _dump({"inputs": input_shas, "model": args.model, "revision": args.revision,
           "layer": args.layer, "neuron": args.neuron, "natural_firing": natural},
          out_dir / "natural-firing.json")

    effects = _measure_effects(args, model, tok, layers, selected,
                               selected_measurements, selected_max, pool, out_dir)
    if args.smoke:
        # Questions use only their corresponding alien_qNN stimuli in this mode.
        comprehension = _run_comprehension(args, model, tok, layers, questions,
                                           selected, selected_max, out_dir)
    else:
        comprehension = _run_comprehension(args, model, tok, layers, questions,
                                           selected, selected_max, out_dir)
    result = {"tool": "part33.py (assistant-basin, Part 33)",
              "model": args.model, "revision": args.revision, "layer": args.layer,
              "neuron": args.neuron, "layers_module": layers_name,
              "neuron_definition": "input to layer-22 mlp.down_proj, as in neuron_report.py Part 18",
              "zero_definition": "set activation to zero only where its natural value is strictly greater than half its own selected-corpus maximum",
              "loss_definition": "logits at token p predict token p+1; firing and the following eight loss positions are unioned to avoid overlap weighting",
              "control_matching": "20 units selected from a seeded random layer-22 pool by closest natural firing-token fraction; each uses its own half-maximum threshold",
              "lookahead_tokens": LOOKAHEAD, "control_candidate_pool": len(pool),
              "input_shas": input_shas, "smoke": bool(args.smoke),
              "natural_firing": natural, "conditional_zeroing": effects,
              "comprehension": comprehension}
    if not args.smoke:
        result["prediction_scores"] = score_results(result)
        print_score(result)
    else:
        print(f"SMOKE complete: {len(selected)} passages; target firing threshold {threshold:.6g}; "
              f"comprehension {comprehension['baseline_correct']}/{comprehension['n']} -> "
              f"{comprehension['conditional_zero_correct']}/{comprehension['n']}", flush=True)
    _dump(result, out_dir / ("smoke.json" if args.smoke else "results.json"))
    return result


def parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--neuron", type=int, default=NEURON)
    ap.add_argument("--device", choices=("xpu", "cpu"), default="xpu")
    ap.add_argument("--out", default="results/part33")
    ap.add_argument("--passages", default="data/part33-authored-passages.jsonl")
    ap.add_argument("--sense-battery", default="data/sense-battery.jsonl")
    ap.add_argument("--neutral-source", default="data/neutral.txt")
    ap.add_argument("--neutral-candidates", type=int, default=40)
    ap.add_argument("--questions", default="data/part33-questions.json")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--control-pool", type=int, default=512)
    ap.add_argument("--intervention-batch", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--max-gpu-gb", type=float, default=14.5)
    ap.add_argument("--min-available-ram-gb", type=float, default=2.0,
                    help="stop before the next model pass if host available RAM is lower")
    ap.add_argument("--smoke", action="store_true", help="run only the first three or --smoke-passages alien question passages")
    ap.add_argument("--smoke-passages", type=int, default=3)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.intervention_batch < 1 or args.control_pool < CONTROL_COUNT:
        raise SystemExit("--intervention-batch must be positive and --control-pool at least 20")
    if args.smoke and not 1 <= args.smoke_passages <= 20:
        raise SystemExit("--smoke-passages must be between 1 and 20")
    started = time.time()
    try:
        run(args)
    except Exception as e:
        print(f"Part 33 stopped: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        raise
    print(f"elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
