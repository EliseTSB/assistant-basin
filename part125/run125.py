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
    return session.logit_hash(logits)


def _replay_logged_calls(model, sequence: list[dict], replay_hook, *, device: str,
                        sequence_name: str) -> tuple[list[str], dict | None]:
    replay_cache = None
    hashes = []
    for index, call in enumerate(sequence):
        call_hook = replay_hook if call.get("hook_on") is True else None
        replay_calls = []
        try:
            logits, replay_cache = session.run_segment(
                model, call["token_ids"], replay_cache, [call], device=device,
                hook=call_hook, call_log=replay_calls, turn=int(call.get("turn", 0)),
            )
        except Exception as exc:
            return hashes, {"sequence": sequence_name, "call_index": index,
                            "reason": f"replay call failed: {exc}"}
        actual_hash = replay_calls[0]["logit_hash"]
        hashes.append(actual_hash)
        if replay_calls[0].get("hook_on") is not call.get("hook_on"):
            return hashes, {"sequence": sequence_name, "call_index": index,
                            "expected_hook_on": call.get("hook_on"),
                            "replayed_hook_on": replay_calls[0].get("hook_on"),
                            "reason": "replay hook state differs from call log"}
        expected_hash = call.get("logit_hash")
        if expected_hash != actual_hash:
            return hashes, {"sequence": sequence_name, "call_index": index,
                            "expected_hash": expected_hash,
                            "replayed_hash": actual_hash,
                            "reason": "run and replay logit hashes differ"}
    return hashes, None


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
                 call_log: list[dict] | None = None,
                 turn: int | None = None) -> dict:
    prompt = session.next_prompt_ids(tokenizer, questions, answers,
                                     question)
    hook = (layer, push) if push is not None else None
    logits, cache = session.run_segment(
        model, prompt, None, [{"token_count": len(prompt), "hook_on": hook is not None}],
        device=device, hook=hook, call_log=call_log, turn=turn,
    )
    answer, _cache = session.greedy_answer(
        model, tokenizer, logits, cache, device=device, max_new_tokens=max_new_tokens,
        hook=hook, call_log=call_log, turn=turn,
    )
    return answer


def _run_random_pair(model, tokenizer, unit: dict, layer, layer_no: int,
                     median: float, answers2: list[dict], *, device: str,
                     unit_id: str, arm2_calls: list[dict]) -> tuple[list[dict], list[dict]]:
    import torch

    directions = _random_vectors(layer, layer_no, median)
    rows = []
    sequences = []
    for direction_index, push in enumerate(directions):
        random_cache = None
        twin_cache = None
        random_ledger: list[int] = []
        twin_ledger: list[int] = []
        random_calls: list[dict] = []
        twin_calls: list[dict] = []
        for turn, question in enumerate(session.QUESTIONS):
            turn_no = turn + 1
            arm2_turn_calls = [call for call in arm2_calls
                               if int(call.get("turn", 0)) == turn_no]
            full = session.next_prompt_ids(tokenizer, list(session.QUESTIONS[:turn]),
                                          answers2[:turn], question)
            if full[:len(random_ledger)] != random_ledger or full[:len(twin_ledger)] != twin_ledger:
                raise AssertionError("random-pair token ledger lost template parity")
            suffix = full[len(random_ledger):]
            prompt_count = 1 if turn == 0 else len(suffix)
            prompt_plan = arm2_turn_calls[:prompt_count]
            answer_plan = arm2_turn_calls[prompt_count:]
            if len(prompt_plan) != prompt_count:
                raise AssertionError("arm 2 call plan lacks the current question segment")
            hook = (layer, push)
            logits_r, random_cache = session.run_segment(
                model, suffix, random_cache, prompt_plan, device=device,
                hook=hook, call_log=random_calls, turn=turn_no,
            )
            logits_t, twin_cache = session.run_segment(
                model, suffix, twin_cache, prompt_plan, device=device,
                call_log=twin_calls, turn=turn_no,
            )
            random_ledger.extend(suffix)
            twin_ledger.extend(suffix)
            target = session.target_ids(tokenizer, unit["screen_answer"])
            logp_r, logits_r = _start_measure_from_logits(model, random_cache, logits_r, target, device)
            logp_t, logits_t = _start_measure_from_logits(model, twin_cache, logits_t, target, device)
            rows.append({"unit_id": unit_id, "direction": direction_index,
                         "turn": turn_no, "delta": logp_r - logp_t,
                         "logp_random": logp_r, "logp_twin": logp_t,
                         "kl_random_to_twin": _kl_divergence(logits_r, logits_t)})
            answer_ids = list(map(int, answers2[turn]["token_ids"]))
            if not answer_plan:
                raise AssertionError("arm 2 call plan lacks the saved answer segment")
            logits_r, random_cache = session.run_segment(
                model, answer_ids, random_cache, answer_plan, device=device,
                hook=hook, call_log=random_calls, turn=turn_no,
            )
            logits_t, twin_cache = session.run_segment(
                model, answer_ids, twin_cache, answer_plan, device=device,
                call_log=twin_calls, turn=turn_no,
            )
            random_ledger.extend(answer_ids)
            twin_ledger.extend(answer_ids)
        session.assert_equal_call_shapes(random_calls, twin_calls)
        session.assert_equal_call_shapes(arm2_calls, random_calls)
        if not all(call.get("hook_on") is (int(call.get("turn", 0)) == 1)
                   for call in random_calls):
            raise AssertionError("random pair hook state differs from arm 2")
        if not all(call.get("hook_on") is False for call in twin_calls):
            raise AssertionError("random twin unexpectedly applied a hook")
        sequences.append({"direction": direction_index,
                          "random_calls": random_calls, "twin_calls": twin_calls})
    return rows, sequences


def _run_unit(model, tokenizer, unit: dict, medians: dict[str, float], *, device: str,
              max_new_tokens: int, pilot: bool,
              parameters: dict | None = None) -> dict:
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
    arm1_calls: list[dict] = []
    arm4_calls: list[dict] = []
    hook_ids_before = set(getattr(layer, "_forward_hooks", {}).keys())
    hook_removed_after_turn1 = False
    turn_logits2 = None
    turn_logits3 = None

    for turn, question in enumerate(session.QUESTIONS):
        turn_no = turn + 1
        if turn == 0:
            full = session.next_prompt_ids(tokenizer, [], [], question)
            turn_start = len(calls2)
            treatment = (layer, push)
            logits2, arm2_cache = session.run_segment(
                model, full, None, [{"token_count": len(full), "hook_on": True}],
                device=device, hook=treatment, call_log=calls2, turn=turn_no,
            )
            score_cache2 = session.clone_cache(arm2_cache)
            answer2, arm2_cache = session.greedy_answer(
                model, tokenizer, logits2, arm2_cache, device=device,
                max_new_tokens=max_new_tokens, hook=treatment,
                call_log=calls2, turn=turn_no,
            )
            arm2_turn_calls = calls2[turn_start:]
            logits3, arm3_cache = session.run_segment(
                model, full, None, arm2_turn_calls[:1], device=device,
                call_log=calls3, turn=turn_no,
            )
            score_cache3 = session.clone_cache(arm3_cache)
            if len(arm2_turn_calls) < 2:
                raise AssertionError("arm 2 call plan has no generated-token calls")
            _twin_logits, arm3_cache = session.run_segment(
                model, answer2["token_ids"], arm3_cache, arm2_turn_calls[1:],
                device=device, call_log=calls3, turn=turn_no,
            )
            ledger2.extend(full)
            ledger3.extend(full)
            turn_logits2, turn_logits3 = logits2, logits3

            logp2, next2 = _start_measure_from_logits(model, score_cache2, logits2, target, device)
            logp3, next3 = _start_measure_from_logits(model, score_cache3, logits3, target, device)
            reader_free.append({"unit_id": str(unit["id"]), "turn": 1,
                                "logp_arm2": logp2, "logp_twin": logp3,
                                "delta": logp2 - logp3,
                                "kl_arm2_to_twin": _kl_divergence(next2, next3)})

            arm2_answers.append({"question": question, **answer2})
            ledger2.extend(answer2["token_ids"])
            _assert_cache_ledger(arm2_cache, ledger2, "arm 2")
            session.assert_template_ledger(tokenizer, [question], arm2_answers, ledger2)

            ledger3.extend(answer2["token_ids"])
            _assert_cache_ledger(arm3_cache, ledger3, "arm 3 twin")
            session.assert_template_ledger(tokenizer, [question], arm2_answers, ledger3)
            template_ids = session.completed_ids(tokenizer, [question], arm2_answers)
            arm1_answers.append({"question": question, **answer2})
            arm3_answers.append({"question": question, **answer2})

            hook_ids_after = set(getattr(layer, "_forward_hooks", {}).keys())
            hook_removed = hook_ids_after == hook_ids_before
            call_equal = session.assert_equal_call_shapes(calls2, calls3)
            arm2_hook_calls = sum(bool(call["hook_on"]) for call in arm2_turn_calls)
            arm2_hook_state_ok = bool(arm2_turn_calls) and all(
                call["hook_on"] is True for call in arm2_turn_calls
            )
            twin_hook_state_ok = all(call["hook_on"] is False for call in calls3)
            ledger_checks.append({"unit_id": str(unit["id"]), "turn": 1,
                                  "template_equal": ledger2 == template_ids,
                                  "arm2_hook_removed": hook_removed,
                                  "twin_tokens_equal": ledger2 == ledger3,
                                  "call_shapes_equal": call_equal,
                                  "call_logs_equal": call_equal,
                                  "arm2_hook_calls": arm2_hook_calls,
                                  "arm2_calls_this_turn": len(arm2_turn_calls),
                                  "arm2_hook_state_ok": arm2_hook_state_ok,
                                  "twin_hook_state_ok": twin_hook_state_ok,
                                  "arm2_ledger_token_ids": list(ledger2),
                                  "arm3_ledger_token_ids": list(ledger3),
                                  "template_token_ids": template_ids,
                                  "arm2_prompt_call_shape": calls2[0]["shape"],
                                  "arm3_prompt_call_shape": calls3[0]["shape"]})
            if not hook_removed:
                raise AssertionError("arm 2 retained a forward hook after turn 1")
            if not arm2_hook_state_ok or not twin_hook_state_ok:
                raise AssertionError("turn-one hook state differs from the registered plan")
            hook_removed_after_turn1 = hook_removed

            arm4 = _full_answer(model, tokenizer, [], [], question, None, None,
                                device=device, max_new_tokens=max_new_tokens,
                                call_log=arm4_calls, turn=turn_no)
            if not all(call["hook_on"] is False for call in arm4_calls):
                raise AssertionError("arm 4 unexpectedly applied a hook")
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
            turn_start = len(calls2)
            logits2, arm2_cache = session.feed_one_by_one(
                model, suffix, arm2_cache, device=device, call_log=calls2, turn=turn_no,
            )
            ledger2.extend(suffix)
            answer2, arm2_cache = session.greedy_answer(
                model, tokenizer, logits2, arm2_cache, device=device,
                max_new_tokens=max_new_tokens, call_log=calls2, turn=turn_no,
            )
            arm2_turn_calls = calls2[turn_start:]
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

            replay_ids = [token for call in arm2_turn_calls for token in call["token_ids"]]
            _twin_logits, arm3_cache = session.run_segment(
                model, replay_ids, arm3_cache, arm2_turn_calls, device=device,
                call_log=calls3, turn=turn_no,
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
            call_equal = session.assert_equal_call_shapes(calls2, calls3)
            arm2_hook_calls = sum(bool(call["hook_on"]) for call in arm2_turn_calls)
            arm2_hook_state_ok = all(call["hook_on"] is False for call in arm2_turn_calls)
            twin_hook_state_ok = all(call["hook_on"] is False
                                     for call in calls3 if int(call.get("turn", 0)) == turn_no)
            ledger_checks.append({"unit_id": str(unit["id"]), "turn": turn + 1,
                                  "template_equal": ledger2 == template_ids,
                                  "twin_tokens_equal": ledger2 == ledger3,
                                  "call_shapes_equal": call_equal,
                                  "call_logs_equal": call_equal,
                                  "arm2_hook_calls": arm2_hook_calls,
                                  "arm2_calls_this_turn": len(arm2_turn_calls),
                                  "arm2_hook_state_ok": arm2_hook_state_ok,
                                  "twin_hook_state_ok": twin_hook_state_ok,
                                  "arm2_ledger_token_ids": list(ledger2),
                                  "arm3_ledger_token_ids": list(ledger3),
                                  "template_token_ids": template_ids})
            if not arm2_hook_state_ok or not twin_hook_state_ok:
                raise AssertionError("post-turn-one paths unexpectedly applied a hook")

            arm1_call_start = len(arm1_calls)
            arm1 = _full_answer(
                model, tokenizer, list(session.QUESTIONS[:turn]), arm1_answers, question,
                layer, push, device=device, max_new_tokens=max_new_tokens,
                call_log=arm1_calls, turn=turn_no,
            )
            if not all(call["hook_on"] is True for call in arm1_calls[arm1_call_start:]):
                raise AssertionError("arm 1 missed a required push call")
            arm1_answers.append({"question": question, **arm1})
            arm4 = _full_answer(
                model, tokenizer, list(session.QUESTIONS[:turn]), arm4_answers, question,
                None, None, device=device, max_new_tokens=max_new_tokens,
                call_log=arm4_calls, turn=turn_no,
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
        pushed_hashes, pushed_difference = _replay_logged_calls(
            model, calls2, treatment, device=device, sequence_name="arm2")
        twin_hashes, twin_difference = _replay_logged_calls(
            model, calls3, None, device=device, sequence_name="arm3_twin")
        first_difference = pushed_difference or twin_difference
        pushed_equal = pushed_difference is None and len(pushed_hashes) == len(calls2)
        twin_equal = twin_difference is None and len(twin_hashes) == len(calls3)
        determinism.append({"unit_id": str(unit["id"]), "calls": len(calls2),
                            "pushed_bitwise_equal": bool(pushed_equal),
                            "twin_bitwise_equal": bool(twin_equal),
                            "bitwise_equal": bool(pushed_equal and twin_equal),
                            "pushed_logit_hashes_replay": pushed_hashes,
                            "twin_logit_hashes_replay": twin_hashes,
                            "first_differing_call": first_difference})

    random_rows, random_call_sequences = _run_random_pair(
        model, tokenizer, unit, layer, layer_no, float(medians[str(layer_no)]),
        arm2_answers, device=device, unit_id=str(unit["id"]), arm2_calls=calls2,
    )
    record = {
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
        "arm1_call_sequence": arm1_calls,
        "arm4_call_sequence": arm4_calls,
        "random_call_sequences": random_call_sequences,
        "arm2_transcript": arm2_answers,
        "arm3_probes": arm3_answers,
        "arm1_transcript": arm1_answers,
        "arm4_transcript": arm4_answers,
        "arm2_hook_removed_after_turn1": hook_removed_after_turn1,
    }
    if parameters is not None:
        record["parameters"] = dict(parameters)
        for name in ("answers", "reader_free", "random_reader_free", "cache_sanity",
                     "determinism", "ledger_checks"):
            for row in record[name]:
                row["parameters"] = dict(parameters)
        for name in ("arm2_call_sequence", "arm3_call_sequence", "arm1_call_sequence",
                     "arm4_call_sequence"):
            for call in record[name]:
                call["parameters"] = dict(parameters)
        for name in ("arm2_transcript", "arm3_probes", "arm1_transcript", "arm4_transcript"):
            for row in record[name]:
                row["parameters"] = dict(parameters)
        for pair in random_call_sequences:
            pair["parameters"] = dict(parameters)
            for call in pair["random_calls"] + pair["twin_calls"]:
                call["parameters"] = dict(parameters)
    return record


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
                     elapsed: float, *, slot_only: bool,
                     parameters: dict) -> None:
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
                       "parameters": dict(unit["parameters"]),
                       "arm2_calls": unit["arm2_call_sequence"],
                       "arm3_calls": unit["arm3_call_sequence"],
                       "arm1_calls": unit["arm1_call_sequence"],
                       "arm4_calls": unit["arm4_call_sequence"],
                       "random_pairs": unit["random_call_sequences"]}
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
        "stage": stage, "status": "complete", "units": len(unit_records),
        "expected_units": len(parameters.get("selected_unit_ids", [])),
        "completed_units": len(unit_records),
        "expected_unit_ids": list(parameters.get("selected_unit_ids", [])),
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
        "parameters": dict(parameters),
        "pilot_size_decision": pilot_decision,
        "model_load": "common92.load_model(snapshot, device); local_files_only=True; bf16; eval; requires_grad_(False)",
    }
    meta_path = out / ("pilot-meta.json" if stage == "pilot" else "run-meta.json")
    if stage == "pilot" and meta_path.is_file():
        saved = read_json(meta_path)
        if saved.get("parameters") != parameters:
            raise ValueError("pilot parameters differ from the saved pilot decision")
        # The first pilot decision is fixed and must survive every resume.
    else:
        write_json(meta_path, meta)


def _json_sha256(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                        help="run the fixed slot cut on main; the pilot always measures both units")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.stage == "pilot" and args.slot_only:
        parser.error("--slot-only is only valid with --stage main")

    neurons = read_json(args.neurons)
    kinds = read_json(args.kinds)
    units = selected_units(neurons, kinds, validate_inventory=validate_inventory,
                           slot_only=(args.slot_only if args.stage == "main" else False))
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
    median_meta_path = out / "medians-meta.json"
    median_identity = {"model_snapshot": str(args.model), "device": str(args.device),
                       "transformers_version": _transformers_version()}
    if median_path.is_file():
        if not median_meta_path.is_file():
            raise ValueError("saved medians have no provenance; rerun in a fresh output folder")
        medians = read_json(median_path)
        median_meta = read_json(median_meta_path)
        if any(median_meta.get(key) != value for key, value in median_identity.items()):
            raise ValueError("saved residual medians were produced with different parameters")
        if median_meta.get("medians_sha256") != _json_sha256(medians):
            raise ValueError("saved residual medians do not match their metadata hash")
    else:
        medians = session.residual_medians(model, tokenizer, args.device)
        write_json(median_path, medians)
        write_json(median_meta_path,
                   {**median_identity, "medians_sha256": _json_sha256(medians)})

    unit_parameters = {
        **median_identity,
        "max_new_tokens": int(args.max_new_tokens),
        "medians_sha256": _json_sha256(medians),
    }
    selected_ids = [str(unit["id"]) for unit in units]
    effective_slot_only = bool(args.slot_only) if args.stage == "main" else False
    run_parameters = {"unit_parameters": unit_parameters,
                      "slot_only": effective_slot_only,
                      "selected_unit_ids": selected_ids}
    run_meta_path = out / "run-meta.json"
    if args.stage == "main" and run_meta_path.is_file():
        saved_run = read_json(run_meta_path)
        if saved_run.get("parameters") != run_parameters:
            raise ValueError("resume parameters differ from the saved run metadata")
    if args.stage == "pilot" and (out / "pilot-meta.json").is_file():
        saved_pilot = read_json(out / "pilot-meta.json")
        if saved_pilot.get("parameters") != run_parameters:
            raise ValueError("resume parameters differ from the saved pilot metadata")

    zero_path = out / "zero-hook-check.json"
    zero_checks = read_json(zero_path) if zero_path.is_file() else []
    if any(row.get("parameters") != unit_parameters for row in zero_checks):
        raise ValueError("saved zero-hook rows were produced with different parameters")
    by_zero = {(int(row["layer"]), int(row["turn"])): row for row in zero_checks}
    missing_zero_layers = [layer for layer in layers_to_run
                           if any((layer, turn) not in by_zero for turn in range(1, 6))]
    if missing_zero_layers:
        new_checks = _run_zero_hook(model, tokenizer, decoder_layers, missing_zero_layers,
                                    args.device, args.max_new_tokens)
        for row in new_checks:
            row["parameters"] = dict(unit_parameters)
        by_zero.update({(int(row["layer"]), int(row["turn"])): row for row in new_checks})
        zero_checks = [by_zero[key] for key in sorted(by_zero)]
        write_json(zero_path, zero_checks)
    if any(not row.get("equal") for row in zero_checks if int(row["layer"]) in layers_to_run):
        print("zero-vector hook gate failed; stopping before study rows", file=sys.stderr, flush=True)
        return 1

    pilot_path = out / "pilot-meta.json"
    pilot_size_decision = None
    if args.stage == "main" and pilot_path.is_file():
        saved_pilot = read_json(pilot_path)
        if (saved_pilot.get("parameters", {}).get("unit_parameters") != unit_parameters or
                saved_pilot.get("parameters", {}).get("slot_only") is not False):
            raise ValueError("main parameters differ from the completed pilot")
        prior_decision = saved_pilot.get("pilot_size_decision") or {}
        pilot_size_decision = prior_decision
        if bool(prior_decision.get("over_90_minutes")) != bool(args.slot_only):
            required = "--slot-only" if prior_decision.get("over_90_minutes") else "the full 29-unit run"
            raise ValueError(f"pilot decision is fixed; main must use {required}")
    elif args.stage == "main" and validate_inventory and not pilot_path.is_file():
        raise ValueError("main stage requires a completed two-unit pilot in the same output directory")

    if args.stage == "main":
        write_json_atomic(run_meta_path, {
            "stage": "main", "status": "running", "units": len(units),
            "expected_units": len(units), "completed_units": 0,
            "expected_unit_ids": list(selected_ids), "unit_ids": [],
            "slot_only": bool(effective_slot_only), "answers": 0,
            "distinct_generations_expected": len(units) * 18,
            "reader_free_scores": 0, "random_pairs": 0,
            "cache_sanity_comparisons": 0, "determinism_units": 0,
            "zero_hook_checks": len(zero_checks),
            "zero_hook_equal": sum(bool(row["equal"]) for row in zero_checks),
            "ledger_checks": 0, "elapsed_seconds": 0.0,
            "transformers_version": _transformers_version(),
            "parameters": dict(run_parameters),
            "pilot_size_decision": pilot_size_decision,
            "model_load": "common92.load_model(snapshot, device); local_files_only=True; bf16; eval; requires_grad_(False)",
        })

    unit_dir = out / "units"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_records = []
    for unit in units:
        path = unit_dir / f"{unit['id']}.json"
        if path.is_file():
            record = read_json(path)
            if str(record.get("unit", {}).get("id")) != str(unit["id"]):
                raise ValueError(f"saved unit file does not match {unit['id']}")
            if record.get("parameters") != unit_parameters:
                raise ValueError(f"saved unit {unit['id']} has different production parameters")
        else:
            unit_started = time.perf_counter()
            record = _run_unit(model, tokenizer, unit, medians, device=args.device,
                               max_new_tokens=args.max_new_tokens,
                               pilot=(args.stage == "pilot"),
                               parameters=unit_parameters)
            record["unit_elapsed_seconds"] = round(time.perf_counter() - unit_started, 4)
            write_json_atomic(path, record)
        unit_records.append(record)
        print(f"saved {unit['id']}: {len(record['answers'])} answer rows", flush=True)

    if args.stage == "main" and validate_inventory:
        unit_records = [read_json(unit_dir / f"{unit['id']}.json") for unit in units]
    elapsed = time.perf_counter() - started
    pilot_decision = None
    if args.stage == "pilot":
        if pilot_path.is_file():
            pilot_decision = read_json(pilot_path).get("pilot_size_decision")
        else:
            projected = elapsed * (29 / max(1, len(unit_records)))
            pilot_decision = {"pilot_units": [str(unit["unit"]["id"]) for unit in unit_records],
                              "elapsed_seconds": round(elapsed, 2),
                              "projected_units": 29,
                              "projected_seconds": round(projected, 2),
                              "projected_minutes": round(projected / 60, 2),
                              "over_90_minutes": bool(projected > 90 * 60),
                              "required_main_cut": "21 slot units" if projected > 90 * 60 else "all 29 units",
                              "decision_fixed_before_labels": True}
    _write_aggregate(out, unit_records, medians, zero_checks, args.stage,
                     pilot_decision, elapsed, slot_only=effective_slot_only,
                     parameters=run_parameters)
    print(json.dumps(read_json(out / ("pilot-meta.json" if args.stage == "pilot" else "run-meta.json")),
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
