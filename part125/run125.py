#!/usr/bin/env python3
"""Run the Part 125 cache/twin study and forward-only random controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for directory in (HERE, ROOT / "part92", ROOT / "part123"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import common92  # noqa: E402
import fit123  # noqa: E402
import session125 as session  # noqa: E402

CONSENSUS_IDS = (
    "p92-02", "p92-03", "p92-04", "p92-05", "p92-07", "p92-09", "p92-10",
    "p92-13", "p92-18", "p92-20", "p92-24", "p92-25", "p92-32", "p92-33",
    "p92-35", "p92-36", "p92-37", "p92-42", "p92-46", "p92-47", "p92-50",
    "p92-08", "p92-15", "p92-29", "p92-39", "p92-43", "p92-45", "p92-48",
    "p92-51",
)
PILOT_IDS = ("p92-02", "p92-08")
RANDOM_COUNT = 3
RANDOM_SEED_BASE = 125000


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path, *, optional: bool = False) -> list[dict]:
    source = Path(path)
    if optional and not source.is_file():
        return []
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = []
    lines = source.read_bytes().splitlines(keepends=True)
    for line_no, raw in enumerate(lines, 1):
        try:
            line = raw.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("JSONL row is not an object")
                rows.append(row)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            torn = line_no == len(lines) and not raw.endswith((b"\n", b"\r"))
            if not torn:
                raise ValueError(f"invalid JSONL in {source}:{line_no}: {exc}") from exc
            print(f"skipping torn final JSONL line in {source}:{line_no}", file=sys.stderr, flush=True)
            break
    return rows


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_json_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def selected_units(neurons: list[dict], kinds: dict, *, validate_inventory: bool = True,
                   slot_only: bool = False) -> list[dict]:
    by_id = {str(row["id"]): row for row in fit123.selected_units(neurons, kinds)}
    wanted = list(CONSENSUS_IDS)
    if slot_only:
        wanted = [uid for uid in wanted if by_id.get(uid, {}).get("kind") == "slot"]
    missing = [uid for uid in wanted if uid not in by_id]
    if missing and validate_inventory:
        raise ValueError(f"Part 125 inventory is missing consensus units: {missing}")
    units = [by_id[uid] for uid in wanted if uid in by_id]
    if validate_inventory and not slot_only:
        counts = {kind: sum(unit["kind"] == kind for unit in units) for kind in ("slot", "concept")}
        if len(units) != 29 or counts != {"slot": 21, "concept": 8}:
            raise ValueError(f"expected 29 consensus units (21 slot, 8 concept), found {len(units)} {counts}")
    if slot_only and validate_inventory and (len(units) != 21 or any(u["kind"] != "slot" for u in units)):
        raise ValueError(f"slot-only run requires 21 slot units, found {len(units)}")
    return units


def _push_vector(unit: dict, layer, median: float):
    import torch

    weight = layer.mlp.down_proj.weight
    vector = common92.neuron_push(weight, int(unit["neuron"]), 1.0)
    return (vector * (0.8 * float(median))).reshape(1, -1)


def _random_vectors(layer, layer_no: int, median: float):
    import torch

    weight = layer.mlp.down_proj.weight
    vectors = []
    for index in range(RANDOM_COUNT):
        direction = common92.random_directions(
            weight, 1, RANDOM_SEED_BASE + 3 * int(layer_no) + index, 1.0
        )[0]
        norm = torch.linalg.vector_norm(direction.float())
        if float(norm.detach().to(device="cpu")) <= 0:
            raise ValueError(f"random direction {index} at layer {layer_no} has zero norm")
        vectors.append((direction * (0.8 * float(median) / norm)).reshape(1, -1))
    return vectors


def _full_reencode(model, token_ids: list[int], device: str):
    import torch

    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    positions = torch.arange(len(token_ids), dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        output = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=False)
    return output.logits[:, -1, :]


def _start_measure(model, cache, suffix: list[int], target_ids: list[int], device: str):
    """Score from a clone and return mean target logp plus next-token logits."""
    import torch

    cloned = session.clone_cache(cache)
    if suffix:
        logits, cloned = session.feed_one_by_one(model, suffix, cloned, device=device)
    else:
        raise ValueError("each turn score must include current-question tokens")
    next_logits = logits.detach().float()
    values = []
    for token in target_ids:
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        values.append(log_probs[int(token)])
        logits, cloned, _handle_id = session.call_forward(model, [int(token)], cloned, device=device)
    mean_logp = float(torch.stack(values).mean().detach().to(device="cpu"))
    return mean_logp, next_logits


def _start_measure_from_logits(model, cache, logits, target_ids: list[int], device: str):
    import torch

    cloned = session.clone_cache(cache)
    first_logits = logits.detach().float()
    values = []
    for token in target_ids:
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        values.append(log_probs[int(token)])
        logits, cloned, _handle_id = session.call_forward(model, [int(token)], cloned, device=device)
    mean_logp = float(torch.stack(values).mean().detach().to(device="cpu"))
    return mean_logp, first_logits


def _kl_divergence(left, right) -> float:
    import torch

    left_log = torch.log_softmax(left[0].float(), dim=-1)
    right_log = torch.log_softmax(right[0].float(), dim=-1)
    value = (left_log.exp() * (left_log - right_log)).sum()
    return float(value.detach().to(device="cpu"))


def _top100_abs_diffs(cached_logits, full_logits) -> list[float]:
    import torch

    cached = cached_logits.detach().float().to(device="cpu")
    full = full_logits.detach().float().to(device="cpu")
    count = min(100, int(full.shape[-1]))
    indices = torch.topk(full[0], count).indices
    values = (cached[0, indices] - full[0, indices]).abs()
    return [float(value) for value in values.tolist()]


def _logit_hash(logits) -> str:
    import torch

    value = logits.detach().contiguous().to(device="cpu")
    if value.dtype == torch.bfloat16:
        raw = value.view(torch.uint16).numpy().tobytes()
    else:
        raw = value.numpy().tobytes()
    prefix = f"{value.dtype}:{tuple(value.shape)}:".encode("utf-8")
    return hashlib.sha256(prefix + raw).hexdigest()


def _compare_cache_to_full(model, token_ids: list[int], cached_logits, device: str,
                           unit_id: str, turn: int) -> dict:
    import torch

    full_logits = _full_reencode(model, token_ids, device)
    return {"unit_id": unit_id, "turn": int(turn),
            "top1_agree": bool(torch.argmax(cached_logits[0].float()) == torch.argmax(full_logits[0].float())),
            "cached_top1_id": int(torch.argmax(cached_logits[0].float())),
            "full_top1_id": int(torch.argmax(full_logits[0].float())),
            "top100_abs_diffs": _top100_abs_diffs(cached_logits, full_logits)}


def _answer_row(unit: dict, arm: str, turn: int, question: str, answer: dict,
                context_answers: list[dict]) -> dict:
    return {
        "row": f"{arm}-{unit['id']}", "arm": arm, "unit_id": str(unit["id"]),
        "persona": str(unit["persona"]), "desc": str(unit.get("desc", "")),
        "kind": str(unit["kind"]), "layer": int(unit["layer"]),
        "turn": int(turn), "question_index": int(turn - 1), "question": str(question),
        "prior_turns": [{"question": str(prev["question"]), "answer": str(prev["answer"]),
                         "token_ids": list(map(int, prev["token_ids"]))}
                        for prev in context_answers],
        **answer,
    }


def _assert_cache_ledger(cache, ledger: list[int], label: str) -> None:
    actual = session.cache_length(cache)
    if actual != len(ledger):
        raise AssertionError(f"{label} cache has {actual} tokens but ledger has {len(ledger)}")


def _full_answer(model, tokenizer, questions: list[str], answers: list[dict], question: str,
                 layer, push, *, device: str, max_new_tokens: int,
                 call_log: list[dict] | None = None) -> dict:
    prompt = session.next_prompt_ids(tokenizer, questions, answers,
                                     question)
    handle = common92.add_row_hook(layer, push) if push is not None else None
    try:
        logits, cache, _handle_id = session.call_forward(
            model, prompt, None, device=device, call_log=call_log,
        )
        answer, _cache = session.greedy_answer(
            model, tokenizer, logits, cache, device=device, max_new_tokens=max_new_tokens,
            call_log=call_log,
        )
        return answer
    finally:
        if handle is not None:
            handle.remove()


def _run_random_pair(model, tokenizer, unit: dict, layer, layer_no: int,
                     median: float, answers2: list[dict], *, device: str,
                     unit_id: str) -> list[dict]:
    import torch

    directions = _random_vectors(layer, layer_no, median)
    rows = []
    for direction_index, push in enumerate(directions):
        random_cache = None
        twin_cache = None
        random_ledger: list[int] = []
        twin_ledger: list[int] = []
        random_calls: list[dict] = []
        twin_calls: list[dict] = []
        for turn, question in enumerate(session.QUESTIONS):
            full = session.next_prompt_ids(tokenizer, list(session.QUESTIONS[:turn]),
                                          answers2[:turn], question)
            if full[:len(random_ledger)] != random_ledger or full[:len(twin_ledger)] != twin_ledger:
                raise AssertionError("random-pair token ledger lost template parity")
            suffix = full[len(random_ledger):]
            logits_r, random_cache, _ = session.call_forward(
                model, suffix, random_cache, device=device,
                hook_layer=layer if turn == 0 else None, push=push if turn == 0 else None,
                call_log=random_calls,
            )
            logits_t, twin_cache, _ = session.call_forward(
                model, suffix, twin_cache, device=device, call_log=twin_calls,
            )
            random_ledger.extend(suffix)
            twin_ledger.extend(suffix)
            target = session.target_ids(tokenizer, unit["screen_answer"])
            logp_r, logits_r = _start_measure_from_logits(model, random_cache, logits_r, target, device)
            logp_t, logits_t = _start_measure_from_logits(model, twin_cache, logits_t, target, device)
            rows.append({"unit_id": unit_id, "direction": direction_index,
                         "turn": turn + 1, "delta": logp_r - logp_t,
                         "logp_random": logp_r, "logp_twin": logp_t,
                         "kl_random_to_twin": _kl_divergence(logits_r, logits_t)})
            if turn < len(session.QUESTIONS) - 1:
                answer_ids = list(map(int, answers2[turn]["token_ids"]))
                logits_r, random_cache, _ = session.call_forward(
                    model, answer_ids, random_cache, device=device,
                    hook_layer=layer if turn == 0 else None, push=push if turn == 0 else None,
                    call_log=random_calls,
                )
                logits_t, twin_cache, _ = session.call_forward(
                    model, answer_ids, twin_cache, device=device, call_log=twin_calls,
                )
                random_ledger.extend(answer_ids)
                twin_ledger.extend(answer_ids)
        session.assert_equal_call_shapes(random_calls, twin_calls)
    return rows


def _run_unit(model, tokenizer, unit: dict, medians: dict[str, float], *, device: str,
              max_new_tokens: int, pilot: bool) -> dict:
    import torch

    layers = common92.get_layers(model)
    layer_no = int(unit["layer"])
    layer = layers[layer_no]
    push = _push_vector(unit, layer, float(medians[str(layer_no)]))
    target = session.target_ids(tokenizer, unit["screen_answer"])

    arm2_cache = None
    arm3_cache = None
    ledger2: list[int] = []
    ledger3: list[int] = []
    calls2: list[dict] = []
    calls3: list[dict] = []
    arm2_answers: list[dict] = []
    arm1_answers: list[dict] = []
    arm3_answers: list[dict] = []
    arm4_answers: list[dict] = []
    answer_rows = []
    reader_free = []
    cache_sanity = []
    determinism = []
    ledger_checks = []
    hook_ids_before = set(getattr(layer, "_forward_hooks", {}).keys())
    turn_logits2 = None
    turn_logits3 = None

    for turn, question in enumerate(session.QUESTIONS):
        if turn == 0:
            full = session.next_prompt_ids(tokenizer, [], [], question)
            handle = common92.add_row_hook(layer, push)
            try:
                logits2, arm2_cache, _handle_id = session.call_forward(
                    model, full, None, device=device, call_log=calls2,
                )
                score_cache2 = session.clone_cache(arm2_cache)
                answer2, arm2_cache = session.greedy_answer(
                    model, tokenizer, logits2, arm2_cache, device=device,
                    max_new_tokens=max_new_tokens, call_log=calls2,
                )
            finally:
                handle.remove()
            logits3, arm3_cache, _ = session.call_forward(
                model, full, None, device=device, call_log=calls3,
            )
            ledger2.extend(full)
            ledger3.extend(full)
            turn_logits2, turn_logits3 = logits2, logits3

            logp2, next2 = _start_measure_from_logits(model, score_cache2, logits2, target, device)
            logp3, next3 = _start_measure_from_logits(model, arm3_cache, logits3, target, device)
            reader_free.append({"unit_id": str(unit["id"]), "turn": 1,
                                "logp_arm2": logp2, "logp_twin": logp3,
                                "delta": logp2 - logp3,
                                "kl_arm2_to_twin": _kl_divergence(next2, next3)})

            arm2_answers.append({"question": question, **answer2})
            ledger2.extend(answer2["token_ids"])
            _assert_cache_ledger(arm2_cache, ledger2, "arm 2")
            session.assert_template_ledger(tokenizer, [question], arm2_answers, ledger2)

            _play_logits, arm3_cache = session.feed_one_by_one(
                model, answer2["token_ids"], arm3_cache, device=device, call_log=calls3,
            )
            ledger3.extend(answer2["token_ids"])
            _assert_cache_ledger(arm3_cache, ledger3, "arm 3 twin")
            session.assert_template_ledger(tokenizer, [question], arm2_answers, ledger3)
            template_ids = session.completed_ids(tokenizer, [question], arm2_answers)
            arm1_answers.append({"question": question, **answer2})
            arm3_answers.append({"question": question, **answer2})

            hook_ids_after = set(getattr(layer, "_forward_hooks", {}).keys())
            hook_removed = hook_ids_after == hook_ids_before
            session.assert_equal_call_shapes(calls2, calls3)
            ledger_checks.append({"unit_id": str(unit["id"]), "turn": 1,
                                  "template_equal": True, "arm2_hook_removed": hook_removed,
                                  "twin_tokens_equal": ledger2 == ledger3,
                                  "call_shapes_equal": True,
                                  "arm2_ledger_token_ids": list(ledger2),
                                  "arm3_ledger_token_ids": list(ledger3),
                                  "template_token_ids": template_ids,
                                  "arm2_prompt_call_shape": calls2[0]["shape"],
                                  "arm3_prompt_call_shape": calls3[0]["shape"]})
            if not hook_removed:
                raise AssertionError("arm 2 retained a forward hook after turn 1")

            arm4 = _full_answer(model, tokenizer, [], [], question, None, None,
                                device=device, max_new_tokens=max_new_tokens)
            arm4_answers.append({"question": question, **arm4})
            answer_rows.extend([
                _answer_row(unit, "arm1", 1, question, answer2, []),
                _answer_row(unit, "arm2", 1, question, answer2, []),
                _answer_row(unit, "arm3", 1, question, answer2, []),
                _answer_row(unit, "arm4", 1, question, arm4, []),
            ])
            cache_sanity.append(_compare_cache_to_full(
                model, full, turn_logits3, device, str(unit["id"]), 1,
            ))
        else:
            previous_questions = list(session.QUESTIONS[:turn])
            full = session.next_prompt_ids(tokenizer, previous_questions, arm2_answers,
                                           question)
            if full[:len(ledger2)] != ledger2 or full[:len(ledger3)] != ledger3:
                raise AssertionError(f"arm 2/twin token IDs differ before turn {turn + 1}")
            suffix = full[len(ledger2):]
            if not suffix:
                raise AssertionError("next turn template produced an empty suffix")

            logp2, logits2 = _start_measure(model, arm2_cache, suffix, target, device)
            logp3, logits3 = _start_measure(model, arm3_cache, suffix, target, device)
            reader_free.append({"unit_id": str(unit["id"]), "turn": turn + 1,
                                "logp_arm2": logp2, "logp_twin": logp3,
                                "delta": logp2 - logp3,
                                "kl_arm2_to_twin": _kl_divergence(logits2, logits3)})
            cache_sanity.append(_compare_cache_to_full(
                model, full, logits3, device, str(unit["id"]), turn + 1,
            ))

            # The real arm-2 cache gets the current user suffix and its own
            # one-token answer sequence. The twin replays those exact calls.
            logits2, arm2_cache = session.feed_one_by_one(
                model, suffix, arm2_cache, device=device, call_log=calls2,
            )
            ledger2.extend(suffix)
            answer2, arm2_cache = session.greedy_answer(
                model, tokenizer, logits2, arm2_cache, device=device,
                max_new_tokens=max_new_tokens, call_log=calls2,
            )
            arm2_answers.append({"question": question, **answer2})
            ledger2.extend(answer2["token_ids"])
            _assert_cache_ledger(arm2_cache, ledger2, "arm 2")
            session.assert_template_ledger(tokenizer, list(session.QUESTIONS[:turn + 1]),
                                           arm2_answers, ledger2)

            probe_cache = session.clone_cache(arm3_cache)
            probe_logits, probe_cache = session.feed_one_by_one(
                model, suffix, probe_cache, device=device,
            )
            answer3, _probe_cache = session.greedy_answer(
                model, tokenizer, probe_logits, probe_cache, device=device,
                max_new_tokens=max_new_tokens,
            )
            arm3_answers.append({"question": question, **answer3})

            _twin_logits, arm3_cache = session.feed_one_by_one(
                model, suffix, arm3_cache, device=device, call_log=calls3,
            )
            _twin_logits, arm3_cache = session.feed_one_by_one(
                model, answer2["token_ids"], arm3_cache, device=device, call_log=calls3,
            )
            ledger3.extend(suffix)
            ledger3.extend(answer2["token_ids"])
            _assert_cache_ledger(arm3_cache, ledger3, "arm 3 twin")
            session.assert_template_ledger(tokenizer, list(session.QUESTIONS[:turn + 1]),
                                           arm2_answers, ledger3)
            if ledger2 != ledger3:
                raise AssertionError("twin caches do not contain identical token IDs")
            template_ids = session.completed_ids(tokenizer, list(session.QUESTIONS[:turn + 1]),
                                                 arm2_answers)
            if ledger2 != template_ids:
                raise AssertionError("completed token ledger differs from chat-template IDs")
            session.assert_equal_call_shapes(calls2, calls3)
            ledger_checks.append({"unit_id": str(unit["id"]), "turn": turn + 1,
                                  "template_equal": True, "twin_tokens_equal": True,
                                  "call_shapes_equal": True,
                                  "arm2_ledger_token_ids": list(ledger2),
                                  "arm3_ledger_token_ids": list(ledger3),
                                  "template_token_ids": template_ids})

            arm1 = _full_answer(
                model, tokenizer, list(session.QUESTIONS[:turn]), arm1_answers, question,
                layer, push, device=device, max_new_tokens=max_new_tokens,
            )
            arm1_answers.append({"question": question, **arm1})
            arm4 = _full_answer(
                model, tokenizer, list(session.QUESTIONS[:turn]), arm4_answers, question,
                None, None, device=device, max_new_tokens=max_new_tokens,
            )
            arm4_answers.append({"question": question, **arm4})
            answer_rows.extend([
                _answer_row(unit, "arm1", turn + 1, question, arm1, arm1_answers[:-1]),
                _answer_row(unit, "arm2", turn + 1, question, answer2, arm2_answers[:-1]),
                _answer_row(unit, "arm3", turn + 1, question, answer3, arm2_answers[:-1]),
                _answer_row(unit, "arm4", turn + 1, question, arm4, arm4_answers[:-1]),
            ])

    session.assert_equal_call_shapes(calls2, calls3)
    if ledger2 != ledger3:
        raise AssertionError("arm 2 and twin finished with different token ledgers")

    if pilot:
        # Replay the exact cached arm-2 chunk sequence twice without a hook.
        replay_a = None
        replay_b = None
        same_calls = []
        hashes_a, hashes_b = [], []
        for call in calls2:
            logits_a, replay_a, _ = session.call_forward(
                model, call["token_ids"], replay_a, device=device,
            )
            logits_b, replay_b, _ = session.call_forward(
                model, call["token_ids"], replay_b, device=device,
            )
            equal = bool(torch.equal(logits_a, logits_b))
            same_calls.append(equal)
            hash_a = _logit_hash(logits_a)
            hash_b = _logit_hash(logits_b)
            hashes_a.append(hash_a)
            hashes_b.append(hash_b)
        determinism.append({"unit_id": str(unit["id"]), "calls": len(same_calls),
                            "equal_calls": sum(same_calls), "bitwise_equal": bool(all(same_calls)),
                            "logit_hashes_first": hashes_a,
                            "logit_hashes_second": hashes_b})

    random_rows = _run_random_pair(model, tokenizer, unit, layer, layer_no,
                                   float(medians[str(layer_no)]), arm2_answers,
                                   device=device, unit_id=str(unit["id"]))
    return {
        "unit": {key: unit.get(key) for key in
                 ("id", "persona", "desc", "kind", "layer", "neuron", "screen_answer")},
        "answers": answer_rows,
        "reader_free": reader_free,
        "random_reader_free": random_rows,
        "cache_sanity": cache_sanity,
        "determinism": determinism,
        "ledger_checks": ledger_checks,
        "arm2_call_sequence": calls2,
        "arm3_call_sequence": calls3,
        "arm2_transcript": arm2_answers,
        "arm3_probes": arm3_answers,
        "arm1_transcript": arm1_answers,
        "arm4_transcript": arm4_answers,
        "arm2_hook_removed_after_turn1": True,
    }


def _run_zero_hook(model, tokenizer, decoder_layers: list, layers: list[int], device: str,
                   max_new_tokens: int) -> list[dict]:
    import torch

    checks = []
    for layer_no in layers:
        layer = decoder_layers[layer_no]
        width = int(layer.mlp.down_proj.weight.shape[0])
        zeros = torch.zeros((1, width), dtype=torch.float32, device="cpu")
        for turn, question in enumerate(session.QUESTIONS):
            plain = _full_answer(model, tokenizer, [], [], question, None, None,
                                 device=device, max_new_tokens=max_new_tokens)
            hooked = _full_answer(model, tokenizer, [], [], question, layer, zeros,
                                  device=device, max_new_tokens=max_new_tokens)
            checks.append({"layer": int(layer_no), "turn": turn + 1,
                           "question": question, "batch_size": 1,
                           "plain_token_ids": plain["token_ids"],
                           "hooked_token_ids": hooked["token_ids"],
                           "equal": plain["token_ids"] == hooked["token_ids"]})
    return checks


def _write_aggregate(out: Path, unit_records: list[dict], medians: dict[str, float],
                     zero_checks: list[dict], stage: str, pilot_decision: dict | None,
                     elapsed: float, *, slot_only: bool) -> None:
    answers = [row for unit in unit_records for row in unit["answers"]]
    reader_free = [row for unit in unit_records for row in unit["reader_free"]]
    random_rows = [row for unit in unit_records for row in unit["random_reader_free"]]
    cache_sanity = [row for unit in unit_records for row in unit["cache_sanity"]]
    prior_determinism = read_jsonl(out / "determinism.jsonl", optional=True)
    determinism_by_unit = {str(row["unit_id"]): row for row in prior_determinism}
    determinism_by_unit.update({str(row["unit_id"]): row
                                for unit in unit_records for row in unit["determinism"]})
    determinism = [determinism_by_unit[key] for key in sorted(determinism_by_unit)]
    ledger_checks = [row for unit in unit_records for row in unit["ledger_checks"]]
    call_sequences = [{"unit_id": str(unit["unit"]["id"]),
                       "arm2_calls": unit["arm2_call_sequence"],
                       "arm3_calls": unit["arm3_call_sequence"]}
                      for unit in unit_records]
    write_jsonl(out / "answers.jsonl", answers)
    write_jsonl(out / "reader-free.jsonl", reader_free)
    write_jsonl(out / "random-reader-free.jsonl", random_rows)
    write_jsonl(out / "cache-sanity.jsonl", cache_sanity)
    write_jsonl(out / "determinism.jsonl", determinism)
    write_jsonl(out / "ledger-checks.jsonl", ledger_checks)
    write_jsonl(out / "call-sequences.jsonl", call_sequences)
    write_json(out / "zero-hook-check.json", zero_checks)
    write_json(out / "medians.json", medians)
    meta = {
        "stage": stage, "units": len(unit_records),
        "unit_ids": [str(unit["unit"]["id"]) for unit in unit_records],
        "slot_only": bool(slot_only), "answers": len(answers),
        "distinct_generations_expected": len(unit_records) * 18,
        "reader_free_scores": len(reader_free),
        "random_pairs": len(random_rows) // 5,
        "cache_sanity_comparisons": len(cache_sanity),
        "determinism_units": len(determinism),
        "zero_hook_checks": len(zero_checks),
        "zero_hook_equal": sum(bool(row["equal"]) for row in zero_checks),
        "ledger_checks": len(ledger_checks),
        "elapsed_seconds": round(float(elapsed), 2),
        "transformers_version": _transformers_version(),
        "pilot_size_decision": pilot_decision,
        "model_load": "common92.load_model(snapshot, device); local_files_only=True; bf16; eval; requires_grad_(False)",
    }
    write_json(out / ("pilot-meta.json" if stage == "pilot" else "run-meta.json"), meta)


def _transformers_version() -> str:
    try:
        import transformers
        return str(transformers.__version__)
    except ImportError:
        return "not installed"


def main(argv: list[str] | None = None, *, model_loader=None,
         validate_inventory: bool = True) -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--model", required=True, help="local pinned OLMo snapshot")
    parser.add_argument("--neurons", default="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", default="results/part92/step2/kinds.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--max-new-tokens", type=int, default=session.MAX_NEW_TOKENS)
    parser.add_argument("--slot-only", action="store_true",
                        help="apply the pilot rule fixed before any labels are read")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    neurons = read_json(args.neurons)
    kinds = read_json(args.kinds)
    units = selected_units(neurons, kinds, validate_inventory=validate_inventory,
                           slot_only=args.slot_only)
    if args.stage == "pilot":
        by_id = {str(unit["id"]): unit for unit in units}
        missing = [uid for uid in PILOT_IDS if uid not in by_id]
        if missing:
            raise ValueError(f"pilot requires {PILOT_IDS}, missing {missing}")
        units = [by_id[uid] for uid in PILOT_IDS]
    elif validate_inventory and len(units) not in (29, 21):
        raise ValueError("main stage requires 29 units or the registered 21-slot pilot cut")

    loader = model_loader or common92.load_model
    tokenizer, model = loader(args.model, args.device)
    model.eval()
    decoder_layers = common92.get_layers(model)
    layers_to_run = sorted({int(unit["layer"]) for unit in units})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    median_path = out / "medians.json"
    if median_path.is_file():
        medians = read_json(median_path)
    else:
        medians = session.residual_medians(model, tokenizer, args.device)
        write_json(median_path, medians)

    zero_path = out / "zero-hook-check.json"
    zero_checks = read_json(zero_path) if zero_path.is_file() else []
    by_zero = {(int(row["layer"]), int(row["turn"])): row for row in zero_checks}
    missing_zero_layers = [layer for layer in layers_to_run
                           if any((layer, turn) not in by_zero for turn in range(1, 6))]
    if missing_zero_layers:
        new_checks = _run_zero_hook(model, tokenizer, decoder_layers, missing_zero_layers,
                                    args.device, args.max_new_tokens)
        by_zero.update({(int(row["layer"]), int(row["turn"])): row for row in new_checks})
        zero_checks = [by_zero[key] for key in sorted(by_zero)]
        write_json(zero_path, zero_checks)
    if any(not row.get("equal") for row in zero_checks if int(row["layer"]) in layers_to_run):
        print("zero-vector hook gate failed; stopping before study rows", file=sys.stderr, flush=True)
        return 1

    pilot_path = out / "pilot-meta.json"
    if args.stage == "main" and validate_inventory and pilot_path.is_file():
        prior_decision = read_json(pilot_path).get("pilot_size_decision") or {}
        if prior_decision.get("over_90_minutes") and not args.slot_only:
            raise ValueError("pilot projected over 90 minutes; main must use --slot-only before labels")
    elif args.stage == "main" and validate_inventory and not pilot_path.is_file():
        raise ValueError("main stage requires a completed two-unit pilot in the same output directory")

    unit_dir = out / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_records = []
    selected_ids = {str(unit["id"]) for unit in units}
    for unit in units:
        path = unit_dir / f"{unit['id']}.json"
        if path.is_file():
            record = read_json(path)
            if str(record.get("unit", {}).get("id")) != str(unit["id"]):
                raise ValueError(f"saved unit file does not match {unit['id']}")
        else:
            record = _run_unit(model, tokenizer, unit, medians, device=args.device,
                               max_new_tokens=args.max_new_tokens,
                               pilot=(args.stage == "pilot"))
            write_json_atomic(path, record)
        unit_records.append(record)
        print(f"saved {unit['id']}: {len(record['answers'])} answer rows", flush=True)

    if args.stage == "main" and validate_inventory:
        unit_records = [read_json(unit_dir / f"{unit['id']}.json") for unit in units]
    elapsed = time.perf_counter() - started
    pilot_decision = None
    if args.stage == "pilot":
        expected_units = 21 if args.slot_only else 29
        projected = elapsed * (expected_units / max(1, len(unit_records)))
        pilot_decision = {"pilot_units": [str(unit["unit"]["id"]) for unit in unit_records],
                          "elapsed_seconds": round(elapsed, 2),
                          "projected_seconds": round(projected, 2),
                          "projected_minutes": round(projected / 60, 2),
                          "over_90_minutes": bool(projected > 90 * 60),
                          "required_main_cut": "21 slot units" if projected > 90 * 60 else None,
                          "decision_fixed_before_labels": True}
    _write_aggregate(out, unit_records, medians, zero_checks, args.stage,
                     pilot_decision, elapsed, slot_only=args.slot_only)
    print(json.dumps(read_json(out / ("pilot-meta.json" if args.stage == "pilot" else "run-meta.json")),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
