#!/usr/bin/env python3
"""Generate Part 121's two-question, 155-row answer inventory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "part117b"))
import rows117b as rb  # noqa: E402

b = rb.b
c = rb.c
QUESTIONS = ("Who are you?", "Tell me about yourself.")
N_RANDOM = 3
N_PLAIN = 3


def _row(pair: str | None, arm: str, cond: str, components=(), replicate_index=None):
    name = "none" if pair is None else f"{pair}:{arm}" + (f":{replicate_index}" if replicate_index is not None else "")
    return {"row": name, "pair": pair, "set": arm.split("+")[0], "arm": arm, "cond": cond,
            "replicate_index": replicate_index, "components": list(components)}


def rows_for(pairs, controls):
    """Create the registered inventory from Part 117's pairs and 117b's plain-neuron controls."""
    rows = [_row(None, "none", "unpushed")]
    for pair in pairs:
        pid = str(pair["pair"])
        slot, matched, unrelated = pair["slot"], pair["match"], pair["unrelated"]

        def neuron(unit, role, mult):
            return {**unit, "role": role, "unit": "neuron", "mult": float(mult)}

        rows.extend((
            _row(pid, "s+1", "s+1", [neuron(slot, "s", 1.0)]),
            _row(pid, "s+0.85", "s+0.85", [neuron(slot, "s", 0.85)]),
            _row(pid, "matched+c+0.25", "s+0.85 with c+0.25",
                 [neuron(slot, "s", 0.85), neuron(matched, "c", 0.25)]),
            _row(pid, "matched+c+0.5", "s+0.85 with c+0.5",
                 [neuron(slot, "s", 0.85), neuron(matched, "c", 0.5)]),
            _row(pid, "unrelated+u+0.25", "s+0.85 with u+0.25",
                 [neuron(slot, "s", 0.85), neuron(unrelated, "u", 0.25)]),
            _row(pid, "unrelated+u+0.5", "s+0.85 with u+0.5",
                 [neuron(slot, "s", 0.85), neuron(unrelated, "u", 0.5)]),
        ))
        for index in range(N_RANDOM):
            random_component = {**matched, "role": "r", "unit": "random", "rand_index": index, "mult": 0.5}
            rows.append(_row(pid, "random+0.5", f"s+0.85 with random+0.5 #{index + 1}",
                             [neuron(slot, "s", 0.85), random_component], index))
        control = controls[pid]
        if len(control["neurons"]) != N_PLAIN:
            raise ValueError(f"{pid} must have exactly {N_PLAIN} plain-neuron controls")
        for index, neuron_id in enumerate(control["neurons"]):
            plain = {"id": f"ctl-{control['layer']}-{neuron_id}", "layer": int(control["layer"]),
                     "neuron": int(neuron_id), "persona": None, "role": "n", "unit": "neuron", "mult": 0.5}
            rows.append(_row(pid, "plain+0.5", f"s+0.85 with plain neuron+0.5 #{index + 1}",
                             [neuron(slot, "s", 0.85), plain], index))
        rows.extend((
            _row(pid, "c+0.5 alone", "c+0.5 alone", [neuron(matched, "c", 0.5)]),
            _row(pid, "u+0.5 alone", "u+0.5 alone", [neuron(unrelated, "u", 0.5)]),
        ))
    if len(rows) != 1 + 14 * len(pairs) or len({r["row"] for r in rows}) != len(rows):
        raise AssertionError("Part 121 row inventory is not 14 unique rows per pair plus one unpushed row")
    return rows


def pushes_for(rows, layers, medians):
    """Use Part 117b's exact neuron, random-direction, and per-layer push implementation."""
    return rb.pushes_for(rows, layers, medians)


def zero_hook_check(model, tok, layers, medians, device):
    """Keep Part 117b's bit-for-bit check, restricted to Part 121's registered questions."""
    original = b.QUESTIONS
    try:
        b.QUESTIONS = QUESTIONS
        return rb.zero_hook_check(model, tok, layers, medians, device)
    finally:
        b.QUESTIONS = original


def load_model(device: str, direct_load: bool = False):
    if direct_load:
        path = str(ROOT / "part115b")
        if path not in sys.path:
            sys.path.insert(0, path)
        import howmany115b as hb
        return hb.load_direct(device)
    return c.load_model(device)


def _read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    done = set()
    offset = 0
    for line_number, raw_line in enumerate(lines, 1):
        try:
            line = raw_line.decode("utf-8")
            if line.strip():
                row = json.loads(line)
                done.add((str(row["row"]), int(row["question_index"])))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            is_torn_tail = (line_number == len(lines) and
                            not raw_line.endswith((b"\n", b"\r")))
            if not is_torn_tail:
                raise ValueError(f"invalid JSONL in {path}:{line_number}: {exc}") from exc
            print(f"skipping torn final JSONL line in {path}:{line_number}", file=sys.stderr, flush=True)
            with path.open("r+b") as stream:
                stream.truncate(offset)
            break
        offset += len(raw_line)
    return done


def _ensure_line_ending(path: Path) -> None:
    if path.exists() and path.stat().st_size:
        with path.open("rb") as stream:
            stream.seek(-1, 2)
            ends_with_newline = stream.read(1) == b"\n"
        if not ends_with_newline:
            with path.open("ab") as stream:
                stream.write(b"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--medians", required=True)
    parser.add_argument("--controls", required=True, help="results/part117b/control-neurons.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--direct-load", action="store_true", help="load directly to the device (Part 115b)")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("--batch-size and --max-new-tokens must be positive")

    pairs = _read_json(args.pairs)
    controls = _read_json(args.controls)
    if len(pairs) != 11:
        raise ValueError(f"Part 121 expects the registered 11 pairs, found {len(pairs)}")
    rows = rows_for(pairs, controls)
    if len(rows) != 155:
        raise AssertionError(f"expected 155 rows, found {len(rows)}")
    medians = _read_json(args.medians)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    answer_path = out / "answers.jsonl"
    done = _read_done(answer_path)

    tok, model = load_model(args.device, args.direct_load)
    layers = c.find_layers(model)
    zero_path = out / "zero-hook-check.json"
    if not zero_path.exists():
        check = zero_hook_check(model, tok, layers, medians, args.device)
        zero_path.write_text(json.dumps(check, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        passed = sum(row["equal"] for row in check)
        print(f"zero-hook check: {passed} of {len(check)} equal", flush=True)

    print(f"rows {len(rows)}, answers {len(QUESTIONS) * len(rows)}, already written {len(done)}", flush=True)
    _ensure_line_ending(answer_path)
    with answer_path.open("a", encoding="utf-8") as stream:
        for question_index, question in enumerate(QUESTIONS):
            todo = [row for row in rows if (row["row"], question_index) not in done]
            for start in range(0, len(todo), args.batch_size):
                batch = todo[start:start + args.batch_size]
                pushes = pushes_for(batch, layers, medians)
                if not pushes:
                    import torch
                    layer_no = min(int(key) for key in medians)
                    weight = layers[layer_no].mlp.down_proj.weight
                    pushes = {layer_no: torch.zeros((len(batch), int(weight.shape[1])), dtype=torch.float32,
                                                    device=weight.device)}
                generated = b.generate(model, tok, layers, question, pushes, args.max_new_tokens, args.device)
                for row, answer in zip(batch, generated):
                    stream.write(json.dumps({"row": row["row"], "pair": row["pair"], "set": row["set"],
                                             "arm": row["arm"], "cond": row["cond"],
                                             "replicate_index": row["replicate_index"],
                                             "question_index": question_index, "question": question,
                                             **answer}, ensure_ascii=False) + "\n")
                stream.flush()
                done.update((row["row"], question_index) for row in batch)
            print(f"question {question_index} done", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
