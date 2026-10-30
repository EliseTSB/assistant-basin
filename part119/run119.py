#!/usr/bin/env python3
"""Run the resumable Part 119 OLMo pair, control, and flip-ladder battery."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "part92") not in sys.path:
    sys.path.insert(0, str(ROOT / "part92"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common92 as c  # noqa: E402
import rows119  # noqa: E402

QUESTION = "Who are you?"
FRAME = "chat"
ANSWER_TOKENS = 60
PATH_TOKENS = 24
ALPHAS = [round(i / 10, 1) for i in range(1, 11)]
ALPHA_SINGLE = 0.8


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_existing(path: Path, key_field: str) -> dict[str, dict]:
    """Read JSONL rows and remove only an incomplete, unterminated last line."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(lines[:index]), encoding="utf-8")
                break
            raise ValueError(f"invalid JSONL in {path} at line {index + 1}") from exc
        key = str(row[key_field])
        if key in found:
            raise ValueError(f"duplicate row key {key!r} in {path}")
        found[key] = row
    return found


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def decode_tokens(tokenizer, tokens: list[int]) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()


def generated_tokens(sequence, tokenizer) -> list[int]:
    """Keep the answer prefix before its first EOS/UNK and discard all padding."""
    end_ids = {int(token) for token in
               (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None))
               if token is not None}
    pad_id = getattr(tokenizer, "pad_token_id", None)
    result = []
    for value in sequence:
        token = int(value)
        if token in end_ids:
            break
        if pad_id is not None and token == int(pad_id):
            break
        result.append(token)
    return result


@contextlib.contextmanager
def row_hooks(layers, row_vectors: dict[int, object]):
    handles = []
    try:
        for layer_no, vectors in row_vectors.items():
            handles.append(c.add_row_hook(layers[layer_no], vectors))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


def vectors_for_rows(rows: list[dict], push_cache: dict, layers, device: str):
    """Build one summed [batch, hidden] vector tensor per layer used by this batch."""
    import torch

    by_layer = {}
    for row_index, row in enumerate(rows):
        for push in row["pushes"]:
            unit_id, arm = push["unit_id"], push["arm"]
            vector = push_cache[(unit_id, arm)]
            layer_no = int(next(unit for unit in push_cache["_units"] if unit["id"] == unit_id)["layer"])
            if layer_no not in by_layer:
                hidden = int(layers[layer_no].mlp.down_proj.weight.shape[0])
                by_layer[layer_no] = torch.zeros((len(rows), hidden), dtype=vector.dtype, device=device)
            by_layer[layer_no][row_index] += vector.to(device=device, dtype=by_layer[layer_no].dtype)
    return by_layer


def _run_generate_batch(model, tokenizer, rows: list[dict], layers, push_cache: dict,
                        prompt: list[int], device: str) -> list[list[int]]:
    import torch

    if not rows:
        return []
    input_ids = torch.tensor([prompt] * len(rows), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    vectors = vectors_for_rows(rows, push_cache, layers, device)
    end_ids = sorted({int(token) for token in
                      (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None))
                      if token is not None})
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask,
              "do_sample": False, "max_new_tokens": ANSWER_TOKENS,
              "pad_token_id": int(tokenizer.pad_token_id)}
    if end_ids:
        kwargs["eos_token_id"] = end_ids
    with row_hooks(layers, vectors), torch.inference_mode():
        generated = model.generate(**kwargs)
    return [generated_tokens(row, tokenizer) for row in generated[:, input_ids.shape[1]:]]


def _row_layer_signature(row: dict, units_by_id: dict[str, dict]) -> tuple[int, ...]:
    return tuple(sorted({units_by_id[p["unit_id"]]["layer"] for p in row["pushes"]}))


def run_answers(rows: list[dict], output: Path, tokenizer, model, layers,
                push_cache: dict, units_by_id: dict[str, dict], batch_size: int,
                device: str, plain_tokens: list[int]) -> tuple[int, float, bool]:
    existing = read_existing(output, "row_id")
    plain_new = "plain" not in existing and any(row["row_id"] == "plain" for row in rows)
    if plain_new:
        append_rows(output, [{"row_id": "plain", "tokens": plain_tokens,
                             "text": decode_tokens(tokenizer, plain_tokens)}])
        existing["plain"] = {"row_id": "plain", "tokens": plain_tokens,
                             "text": decode_tokens(tokenizer, plain_tokens)}
    pending = [row for row in rows if row["row_id"] not in existing and row["row_type"] != "plain"]
    grouped = {}
    for row in pending:
        grouped.setdefault(_row_layer_signature(row, units_by_id), []).append(row)
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    started = time.perf_counter()
    total = len(pending)
    group_count = sum(math.ceil(len(group) / batch_size) for group in grouped.values())
    group_no = 0
    done = 0
    for signature, tasks in grouped.items():
        for begin in range(0, len(tasks), batch_size):
            chunk = tasks[begin:begin + batch_size]
            token_rows = _run_generate_batch(model, tokenizer, chunk, layers, push_cache, prompt, device)
            records = [{"row_id": row["row_id"], "tokens": tokens,
                        "text": decode_tokens(tokenizer, tokens)}
                       for row, tokens in zip(chunk, token_rows)]
            append_rows(output, records)
            done += len(chunk)
            group_no += 1
            print(f"answers {group_no}/{group_count} rows {done}/{total} "
                  f"layers={signature} elapsed={time.perf_counter() - started:.1f}s", flush=True)
    elapsed = time.perf_counter() - started
    return total + int(plain_new), elapsed, plain_new


def _forward(model, input_ids, attention_mask, layers, vectors):
    import torch

    with row_hooks(layers, vectors), torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    # Float32 is used for all logit comparisons and reductions below.
    return output.logits.float()


def _top_and_margin(logits):
    """Return each run's top id and margin using argmax plus a masked second argmax."""
    import torch

    top = logits.argmax(dim=-1)
    top_value = logits.gather(-1, top.unsqueeze(-1)).squeeze(-1)
    masked = logits.clone()
    masked.scatter_(-1, top.unsqueeze(-1), float("-inf"))
    second_value = masked.max(dim=-1).values
    return top, top_value - second_value


def score_logits_chunk(rows: list[dict], answer_by_id: dict[str, dict], tokenizer,
                       model, layers, push_cache: dict, prompt: list[int],
                       device: str) -> list[dict]:
    """Four teacher-forced passes; transfer vocabulary vectors only at contested positions."""
    import numpy as np
    import torch

    if any(len(row["pushes"]) != 2 for row in rows):
        raise ValueError("teacher-forced rows must each contain exactly two pushes")
    sequences = []
    targets = []
    for row in rows:
        answer = answer_by_id.get(row["row_id"])
        if answer is None:
            raise ValueError(f"logit row {row['row_id']!r} has no saved answer")
        tokens = [int(token) for token in answer["tokens"][:PATH_TOKENS]]
        sequences.append(prompt + tokens)
        targets.append(tokens)
    width = max(map(len, sequences))
    pad_id = int(tokenizer.pad_token_id)
    input_rows = [seq + [pad_id] * (width - len(seq)) for seq in sequences]
    mask_rows = [[1] * len(seq) + [0] * (width - len(seq)) for seq in sequences]
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)

    none = _forward(model, input_ids, attention_mask, layers, {})[:, len(prompt) - 1:len(prompt) + PATH_TOKENS - 1, :]
    first_pushes = [{"pushes": [row["pushes"][0]]} for row in rows]
    first = _forward(model, input_ids, attention_mask, layers,
                     vectors_for_rows(first_pushes, push_cache, layers, device))[
                         :, len(prompt) - 1:len(prompt) + PATH_TOKENS - 1, :]
    second_pushes = [{"pushes": [row["pushes"][1]]} for row in rows]
    second = _forward(model, input_ids, attention_mask, layers,
                      vectors_for_rows(second_pushes, push_cache, layers, device))[
                          :, len(prompt) - 1:len(prompt) + PATH_TOKENS - 1, :]

    top_none, _margin_none = _top_and_margin(none)
    top_first, margin_first = _top_and_margin(first)
    top_second, margin_second = _top_and_margin(second)
    additive = (first + second - none).argmax(dim=-1)
    contested = top_first != top_second
    valid = torch.zeros_like(contested, dtype=torch.bool)
    for index, tokens in enumerate(targets):
        if tokens:
            valid[index, :len(tokens)] = True
    contested &= valid
    top_ids_cpu = {
        "none": top_none.detach().cpu().tolist(),
        "first": top_first.detach().cpu().tolist(),
        "second": top_second.detach().cpu().tolist(),
        "add": additive.detach().cpu().tolist(),
        "margin_first": margin_first.detach().cpu().tolist(),
        "margin_second": margin_second.detach().cpu().tolist(),
    }
    contested_cpu = contested.detach().cpu().tolist()

    both_pushes = [{"pushes": row["pushes"]} for row in rows]
    both = _forward(model, input_ids, attention_mask, layers,
                    vectors_for_rows(both_pushes, push_cache, layers, device))[
                        :, len(prompt) - 1:len(prompt) + PATH_TOKENS - 1, :]
    top_both = both.argmax(dim=-1)
    both_ids_cpu = top_both.detach().cpu().tolist()
    records = []
    for row_index, row in enumerate(rows):
        token_row = targets[row_index]
        own_hits = sum(int(both_ids_cpu[row_index][pos]) == token
                       for pos, token in enumerate(token_row))
        positions = []
        for pos, is_contested in enumerate(contested_cpu[row_index][:len(token_row)]):
            if not is_contested:
                continue
            a = int(top_ids_cpu["first"][row_index][pos])
            b = int(top_ids_cpu["second"][row_index][pos])
            positions.append({"pos": pos, "a": a, "b": b,
                              "g": int(token_row[pos]), "add": int(top_ids_cpu["add"][row_index][pos]),
                              "both_top": int(both_ids_cpu[row_index][pos]),
                              "Ma": float(top_ids_cpu["margin_first"][row_index][pos]),
                              "Mb": float(top_ids_cpu["margin_second"][row_index][pos])})
        records.append({"row_id": row["row_id"], "n_positions": len(token_row),
                        "both_hits": own_hits, "contested": positions})
    return records


def read_existing_interactions(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    found = set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(lines[:index]), encoding="utf-8")
                break
            raise ValueError(f"invalid JSONL in {path} at line {index + 1}") from exc
        key = (str(row["first"]), str(row["second"]), str(row["kind"]))
        if key in found:
            raise ValueError(f"duplicate Path 2 row {key} in {path}")
        found.add(key)
    return found


def _top10_repeated_argmax(logits):
    """Top ids without accelerator top-k/sort; logits is [positions, vocabulary]."""
    import torch

    working = logits.clone()
    columns = []
    for _ in range(min(10, int(logits.shape[-1]))):
        best = working.argmax(dim=-1)
        columns.append(best)
        working.scatter_(1, best.unsqueeze(1), float("-inf"))
    return torch.stack(columns, dim=1).detach().cpu().tolist()


def _interaction_position_metrics(logits_none, logits_first, logits_second, logits_both) -> list[dict]:
    import numpy as np
    import torch

    top_ids = [_top10_repeated_argmax(values) for values in
               (logits_none, logits_first, logits_second, logits_both)]
    additive = logits_first + logits_second - logits_none
    tv = .5 * (torch.softmax(additive, dim=-1) - torch.softmax(logits_both, dim=-1)).abs().sum(dim=-1)
    tv_cpu = tv.detach().cpu().tolist()
    records = []
    for position in range(logits_none.shape[0]):
        selected = sorted({int(token) for group in top_ids for token in group[position]})
        index = torch.tensor(selected, dtype=torch.long, device=logits_none.device)
        vectors = [values[position].index_select(0, index).detach().cpu().numpy().astype(np.float64, copy=False)
                   for values in (logits_none, logits_first, logits_second, logits_both)]
        centered = [vector - vector.mean() for vector in vectors]
        d_a = centered[1] - centered[0]
        d_2 = centered[2] - centered[0]
        d_pair = centered[3] - centered[0]
        interaction = d_pair - d_a - d_2
        records.append({"nI": float(np.linalg.norm(interaction)),
                        "nA": float(np.linalg.norm(d_a)),
                        "n2": float(np.linalg.norm(d_2)),
                        "nPair": float(np.linalg.norm(d_pair)),
                        "tv": float(tv_cpu[position])})
    return records


def run_interactions(units: list[dict], interaction_rows: list[dict], answers_path: Path,
                     output: Path, tokenizer, model, layers, push_cache: dict,
                     device: str) -> tuple[int, float, int]:
    import torch

    existing = read_existing_interactions(output)
    pending = [row for row in interaction_rows
               if (row["first"], row["second"], row["kind"]) not in existing]
    by_first = {}
    for row in pending:
        by_first.setdefault(row["first"], []).append(row)
    answers = read_existing(answers_path, "row_id")
    plain = [int(token) for token in answers["plain"]["tokens"]]
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    unit_by_id = {unit["id"]: unit for unit in units}
    started = time.perf_counter()
    passes = 0
    done = 0
    total = len(pending)
    for first_id, candidates in by_first.items():
        unit = unit_by_id[first_id]
        single = answers.get(f"single:{first_id}")
        if single is None:
            raise ValueError(f"Path 2 is missing single answer for {first_id}")
        tokens = [int(token) for token in single["tokens"][:PATH_TOKENS]]
        p, _t_a, _t_0 = first_flip(tokens, plain)
        if p is None or not tokens:
            for row in candidates:
                append_rows(output, [{"first": row["first"], "second": row["second"],
                                      "kind": row["kind"], "p_A": p, "positions": []}])
                done += 1
            print(f"interaction {done}/{total} first={first_id} p=None elapsed="
                  f"{time.perf_counter() - started:.1f}s", flush=True)
            continue
        input_ids = torch.tensor([prompt + tokens], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        n_positions = min(PATH_TOKENS, len(tokens))
        position_start = len(prompt) + p - 1
        position_stop = len(prompt) + n_positions - 1
        first_push_row = {"pushes": [{"unit_id": first_id, "arm": "neuron"}]}
        base_logits = _forward(model, input_ids, attention_mask, layers, {})[0,
                           position_start:position_stop, :]
        first_logits = _forward(model, input_ids, attention_mask, layers,
                                vectors_for_rows([first_push_row], push_cache, layers, device))[0,
                                position_start:position_stop, :]
        passes += 2
        for row in candidates:
            second_row = {"pushes": [row["pushes"][1]]}
            both_row = {"pushes": row["pushes"]}
            second_logits = _forward(
                model, input_ids, attention_mask, layers,
                vectors_for_rows([second_row], push_cache, layers, device),
            )[0, position_start:position_stop, :]
            both_logits = _forward(
                model, input_ids, attention_mask, layers,
                vectors_for_rows([both_row], push_cache, layers, device),
            )[0, position_start:position_stop, :]
            positions = _interaction_position_metrics(base_logits, first_logits,
                                                     second_logits, both_logits)
            append_rows(output, [{"first": row["first"], "second": row["second"],
                                  "kind": row["kind"],
                                  "p_A": p,
                                  "positions": [{"pos": p + offset, **values}
                                                for offset, values in enumerate(positions)]}])
            del second_logits, both_logits
            passes += 2
            done += 1
        print(f"interaction {done}/{total} first={first_id} p={p} "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return len(pending), time.perf_counter() - started, passes


def run_logits(rows: list[dict], answers_path: Path, output: Path, tokenizer,
               model, layers, push_cache: dict, batch_size: int, device: str) -> tuple[int, float, int]:
    existing = read_existing(output, "row_id")
    target_rows = [row for row in rows if row["row_type"] in ("pair", "control")]
    pending = [row for row in target_rows if row["row_id"] not in existing]
    answer_by_id = read_existing(answers_path, "row_id")
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    # Keep the three float32 [batch, 24, vocab] tensors within a modest device budget.
    forward_batch = max(1, min(batch_size, 8))
    started = time.perf_counter()
    groups = math.ceil(len(pending) / forward_batch)
    passes = 0
    done = 0
    for begin in range(0, len(pending), forward_batch):
        chunk = pending[begin:begin + forward_batch]
        records = score_logits_chunk(chunk, answer_by_id, tokenizer, model, layers,
                                     push_cache, prompt, device)
        append_rows(output, records)
        done += len(chunk)
        passes += len(chunk) * 4
        print(f"logits {math.ceil(done / forward_batch)}/{groups} rows {done}/{len(pending)} "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return len(pending), time.perf_counter() - started, passes


def first_flip(single: list[int], plain: list[int]) -> tuple[int | None, int | None, int | None]:
    limit = min(PATH_TOKENS, len(single), len(plain))
    for pos in range(limit):
        if int(single[pos]) != int(plain[pos]):
            return pos, int(single[pos]), int(plain[pos])
    return None, None, None


def _ladder_forward(model, input_ids, layer, vector):
    import torch

    handle = c.add_row_hook(layer, vector.reshape(1, -1))
    try:
        with torch.inference_mode():
            return model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    finally:
        handle.remove()


def run_ladder(units: list[dict], answers_path: Path, output: Path,
               tokenizer, model, layers, medians: dict[int, float],
               units_by_id: dict[str, dict], device: str) -> tuple[int, float, int]:
    import torch

    existing = read_existing(output, "id")
    answers = read_existing(answers_path, "row_id")
    plain_row = answers.get("plain")
    if plain_row is None:
        raise ValueError("flip ladder requires the saved plain answer")
    plain = [int(token) for token in plain_row["tokens"]]
    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    pending = [unit for unit in units if unit["id"] not in existing]
    started = time.perf_counter()
    passes = 0
    for index, unit in enumerate(pending, 1):
        single = answers.get(f"single:{unit['id']}")
        if single is None:
            raise ValueError(f"flip ladder is missing single answer for {unit['id']}")
        p, t_a, t_0 = first_flip([int(token) for token in single["tokens"]], plain)
        tops, margins = [], []
        if p is not None:
            prefix = prompt + plain[:p]
            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
            layer_no = unit["layer"]
            weight = layers[layer_no].mlp.down_proj.weight
            for alpha in ALPHAS:
                vector = c.neuron_push(weight, unit["neuron"], alpha * medians[layer_no])
                logits = _ladder_forward(model, input_ids, layers[layer_no], vector)
                tops.append(int(logits.argmax().item()))
                margins.append(float((logits[int(t_a)] - logits[int(t_0)]).item()))
                passes += 1
        record = {"id": unit["id"], "p": p, "t_A": t_a, "t_0": t_0,
                  "alphas": ALPHAS, "tops": tops, "margins": margins}
        append_rows(output, [record])
        print(f"ladder {index}/{len(pending)} id={unit['id']} p={p} "
              f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
    return len(pending), time.perf_counter() - started, passes


def run(args: argparse.Namespace) -> dict:
    import torch

    units = rows119.load_units(args.neurons, args.kinds, args.kinds_ds)
    units_by_id = {unit["id"]: unit for unit in units}
    rows = rows119.pilot_rows(units) if args.stage == "pilot" else rows119.all_rows(units)
    output_dir = repo_path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = c.load_model(repo_path(args.model), args.device)
    layers = c.get_layers(model)
    medians = c.residual_medians(model, tokenizer, FRAME, args.device)
    push_cache = {"_units": units}
    for unit in units:
        layer_no = unit["layer"]
        if not 0 <= layer_no < len(layers):
            raise ValueError(f"unit {unit['id']} layer {layer_no} is outside loaded model")
        layer = layers[layer_no]
        weight = layer.mlp.down_proj.weight
        if not 0 <= unit["neuron"] < weight.shape[1]:
            raise ValueError(f"unit {unit['id']} neuron index is outside W_down")
        target_norm = ALPHA_SINGLE * medians[layer_no]
        push_cache[(unit["id"], "neuron")] = c.neuron_push(weight, unit["neuron"], target_norm)
        push_cache[(unit["id"], "random")] = c.random_direction(weight, unit["control_seed"], target_norm)

    prompt = c.prompt_ids(tokenizer, QUESTION, FRAME)
    hidden = int(layers[0].mlp.down_proj.weight.shape[0])
    zero_matrix = torch.zeros((1, hidden), dtype=layers[0].mlp.down_proj.weight.dtype,
                              device=args.device)
    zero_vectors = {layer_no: zero_matrix for layer_no in range(len(layers))}
    gate_no_hook_started = time.perf_counter()
    with torch.inference_mode():
        no_hook_tokens = _run_generate_batch(
            model, tokenizer, [{"pushes": []}], layers, push_cache, prompt, args.device
        )[0]
    gate_no_hook_seconds = time.perf_counter() - gate_no_hook_started
    gate_zero_started = time.perf_counter()
    with torch.inference_mode():
        # A fresh all-layer zero hook is the registered do-nothing path.
        input_ids = torch.tensor([prompt], dtype=torch.long, device=args.device)
        kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids),
                  "do_sample": False, "max_new_tokens": ANSWER_TOKENS,
                  "pad_token_id": int(tokenizer.pad_token_id)}
        end_ids = sorted({int(token) for token in
                          (getattr(tokenizer, "eos_token_id", None), getattr(tokenizer, "unk_token_id", None))
                          if token is not None})
        if end_ids:
            kwargs["eos_token_id"] = end_ids
        with row_hooks(layers, zero_vectors):
            zero_generated = model.generate(**kwargs)
        zero_tokens = generated_tokens(zero_generated[0, input_ids.shape[1]:], tokenizer)
    gate_zero_seconds = time.perf_counter() - gate_zero_started
    gate = {"no_hook_tokens": no_hook_tokens, "zero_hook_tokens": zero_tokens,
            "passed": no_hook_tokens == zero_tokens}
    gate_path = output_dir / "gate1.json"
    with gate_path.open("w", encoding="utf-8") as stream:
        json.dump(gate, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    answers_path = output_dir / "answers.jsonl"
    answer_count, answer_seconds, plain_new = run_answers(
        rows, answers_path, tokenizer, model, layers, push_cache, units_by_id,
        args.batch, args.device, zero_tokens,
    )
    if plain_new:
        answer_seconds += gate_zero_seconds
    logits_path = output_dir / "logits.jsonl"
    logit_count, logit_seconds, logit_passes = run_logits(
        rows, answers_path, logits_path, tokenizer, model, layers, push_cache,
        args.batch, args.device,
    )
    interaction_path = output_dir / "interaction.jsonl"
    path2_rows = rows119.interaction_rows(units, args.stage)
    interaction_count, interaction_seconds, interaction_passes = run_interactions(
        units, path2_rows, answers_path, interaction_path, tokenizer, model,
        layers, push_cache, args.device,
    )
    ladder_path = output_dir / "ladder.jsonl"
    ladder_count, ladder_seconds, ladder_passes = run_ladder(
        units, answers_path, ladder_path, tokenizer, model, layers,
        medians, units_by_id, args.device,
    )
    all_passes = logit_passes + interaction_passes + ladder_passes
    pass_seconds = logit_seconds + interaction_seconds + ladder_seconds
    timing = {
        "stage": args.stage,
        "model_load_excluded": True,
        "answers_new": answer_count,
        "answer_seconds": answer_seconds,
        "answers_per_second": answer_count / answer_seconds if answer_seconds else None,
        "gate1_extra_answers": 1 + int(not plain_new),
        "gate1_zero_reused_as_plain": plain_new,
        "gate1_seconds": gate_no_hook_seconds + gate_zero_seconds,
        "teacher_forced_rows_new": logit_count,
        "interaction_rows_new": interaction_count,
        "ladder_units_new": ladder_count,
        "forward_passes_new": all_passes,
        "forward_seconds": pass_seconds,
        "passes_per_second": all_passes / pass_seconds if pass_seconds else None,
        "gate1_passed": gate["passed"],
        "batch": args.batch,
    }
    with (output_dir / "timing.json").open("w", encoding="utf-8") as stream:
        json.dump(timing, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(timing, ensure_ascii=False), flush=True)
    return timing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local pinned OLMo 2 1B Instruct snapshot")
    parser.add_argument("--out", default="results/part119")
    parser.add_argument("--device", default="xpu", help="xpu or cpu")
    parser.add_argument("--stage", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--neurons", default=rows119.DEFAULT_NEURONS)
    parser.add_argument("--kinds", default=rows119.DEFAULT_HAIKU_KINDS)
    parser.add_argument("--kinds-ds", default=rows119.DEFAULT_DS_KINDS)
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
