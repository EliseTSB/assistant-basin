#!/usr/bin/env python3
"""replay85.py -- regenerate Part 85's step-2 answers from the registered recipe and compare them token for token.

Written by Claude (Opus 5.5), 2026-10-28, from PART85-DESIGN.md (steps 1 and 2, Amendment 2) and the observer schema
in part85/FOLD5-NOTES.md, not from part85.py. Only the interface is shared: the answer rows' identity fields
(frame, cell, question_index, sample, observer_call_id) and the observer's generate and forward records.

What it rebuilds on its own: the question, the seed (85000 + 100 x question index + sample), the prompt (plain
"Question: <q>\\nAnswer:" without added special tokens, or the chat template with one user turn and the generation
prompt), the sampling settings (temperature 0.7, top-p 0.95, at most 64 new tokens), the stop tokens (generation
config eos plus tokenizer eos), and the clamp: L10 N6978 at 1.5 x its own step-1 maximum (own_max) or at 9.75
(absolute), control L10 N7884 at 1.5 x its own step-1 maximum (control), none (unclamped), set on every prompt and
answer position at the down_proj input. The step-1 maxima are recomputed from the observer's step-1 forward records
(the values the matmul received on the 200 neutral sequences), not read from step1.json.

It then compares, per answer: the registered seed against the observer's cpu_initial_seed; my full generated ids
against the observer's raw generated_ids; the cut ids' sha256 and count against the observer row; my decoded text
against the answer row; and the run's clamp value against mine. Every answer by default. A sample (--fraction < 1)
is labelled NON-PRODUCTION SAMPLE and exits 3. --perturb-value 1.01 multiplies every clamp value: run on clamped
cells it must change answers (the negative control), and it never prints PASS.
Exit: 0 PASS, 1 FAIL, 3 sample or negative control, 2 bad input.
Usage: replay85.py --cp-dir <checkpoint result dir> --snapshot <model dir> --tag main [--frames plain,chat]
                   [--device xpu] [--fraction 1.0] [--perturb-value 1.0] [--out report.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
SAMPLES = 16
LAYER, UNIT, CONTROL_UNIT = 10, 6978, 7884
ABSOLUTE = 9.75
OWN_FACTOR = 1.5
TEMPERATURE, TOP_P, MAX_NEW = 0.7, 0.95, 64
PLAIN = "Question: {q}\nAnswer:"
NEUTRAL_COUNT, NEUTRAL_LEN = 200, 128


def seed_for(qi: int, sample: int) -> int:
    return 85000 + 100 * qi + sample


def ids_sha(ids) -> str:
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def step1_maxima(observe_dir: Path) -> dict:
    """My own step-1 maxima: max over every position of the 200 neutral sequences, per unit."""
    recs = [r for r in read_jsonl(observe_dir / "step1.jsonl") if r.get("event") == "forward"]
    if len(recs) < NEUTRAL_COUNT:
        raise ValueError(f"step 1 has {len(recs)} forward records, expected at least {NEUTRAL_COUNT}")
    out = {}
    for unit in (UNIT, CONTROL_UNIT):
        best = None
        for r in recs[:NEUTRAL_COUNT]:
            vals = r["layers"][str(LAYER)]["values"][str(unit)]
            flat = [v for call in vals for row in call for v in row]
            if len(flat) != NEUTRAL_LEN:
                raise ValueError(f"step 1 neutral record seq {r.get('seq')} has {len(flat)} positions, expected {NEUTRAL_LEN}")
            m = max(flat)
            best = m if best is None else max(best, m)
        out[unit] = float(best)
    return out


def clamp_for(cell: str, maxima: dict):
    if cell == "unclamped":
        return None
    if cell == "own_max":
        return UNIT, OWN_FACTOR * maxima[UNIT]
    if cell == "absolute":
        return UNIT, ABSOLUTE
    if cell == "control":
        return CONTROL_UNIT, OWN_FACTOR * maxima[CONTROL_UNIT]
    raise ValueError(f"unregistered cell {cell!r}")


def encode(tok, question: str, frame: str):
    if frame == "plain":
        return tok(PLAIN.format(q=question), return_tensors="pt", add_special_tokens=False)
    if frame == "chat":
        ids = tok.apply_chat_template([{"role": "user", "content": question}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True)
        return ids
    raise ValueError(f"unregistered frame {frame!r}")


def stops(model, tok) -> list:
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if tok.eos_token_id is not None and tok.eos_token_id not in eos:
        eos.append(tok.eos_token_id)
    return eos


def cut(ids, stop_ids):
    out = []
    for t in ids:
        out.append(int(t))
        if int(t) in stop_ids:
            break
    return out


def regenerate(model, tok, device: str, frame: str, question: str, seed: int, clamp):
    import torch
    enc = {k: v.to(device) for k, v in encode(tok, question, frame).items()}
    handle = None
    if clamp is not None:
        unit, value = clamp
        mod = model.model.layers[LAYER].mlp.down_proj

        def pre(_m, args):
            x = args[0].clone()
            x[..., unit] = value
            return (x,) + tuple(args[1:])
        handle = mod.register_forward_pre_hook(pre)
    try:
        torch.manual_seed(seed)
        if device.startswith("xpu"):
            torch.xpu.manual_seed_all(seed)
        stop_ids = stops(model, tok)
        pad = tok.pad_token_id if tok.pad_token_id is not None else stop_ids[0]
        with torch.inference_mode():
            g = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, max_new_tokens=MAX_NEW,
                               eos_token_id=stop_ids, pad_token_id=pad, use_cache=True)
        full = [int(t) for t in g[0, enc["input_ids"].shape[-1]:]]
        return full, stop_ids, [int(t) for t in enc["input_ids"][0]]
    finally:
        if handle is not None:
            handle.remove()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp-dir", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--frames", default="plain,chat")
    ap.add_argument("--cells", default="unclamped,own_max,absolute,control")
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--perturb-value", type=float, default=1.0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cp = Path(a.cp_dir)
    calls = {}
    for f in sorted((cp / "observe").glob(f"{a.tag}*.jsonl")):
        for r in read_jsonl(f):
            if r.get("event") == "generate":
                calls[r["call_id"]] = r
    maxima = step1_maxima(cp / "observe")
    rows = []
    for frame in a.frames.split(","):
        for cell in a.cells.split(","):
            p = cp / "answers" / f"{a.tag}-{frame}--{cell}.jsonl"
            if p.exists():
                rows += [(frame, cell, r) for r in read_jsonl(p)]
    if not rows:
        print("REPLAY85: no answer rows found", file=sys.stderr)
        return 2
    if a.fraction < 1.0:
        rows = random.Random(85085).sample(rows, max(1, int(len(rows) * a.fraction)))
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    model = AutoModelForCausalLM.from_pretrained(a.snapshot, dtype=torch.float32).to(a.device).eval()
    problems, changed, checked = [], 0, 0
    for frame, cell, row in rows:
        qi, sample = int(row["question_index"]), int(row["sample"])
        where = f"{frame}/{cell}/q{qi}s{sample}"
        if row.get("question") != QUESTIONS[qi]:
            problems.append(f"{where}: question {row.get('question')!r} is not the registered one")
        seed = seed_for(qi, sample)
        if row.get("seed") != seed:
            problems.append(f"{where}: row seed {row.get('seed')} != registered {seed}")
        clamp = clamp_for(cell, maxima)
        if clamp is not None:
            if row.get("unit") != clamp[0] or abs(float(row.get("clamp_value")) - clamp[1]) > 1e-6 * max(1.0, abs(clamp[1])):
                problems.append(f"{where}: run clamp ({row.get('unit')}, {row.get('clamp_value')}) != mine {clamp}")
            clamp = (clamp[0], clamp[1] * a.perturb_value)
        call = calls.get(row.get("observer_call_id"))
        if call is None:
            problems.append(f"{where}: no observer record {row.get('observer_call_id')!r}")
            continue
        if call.get("rng_at_entry", {}).get("cpu_initial_seed") != seed:
            problems.append(f"{where}: observer seed {call.get('rng_at_entry', {}).get('cpu_initial_seed')} != {seed}")
        full, stop_ids, prompt_ids = regenerate(model, tok, a.device, frame, QUESTIONS[qi], seed, clamp)
        checked += 1
        raw = call.get("raw", [{}])[0]
        obs_row = call.get("rows", [{}])[0]
        mine_cut = cut(full, set(stop_ids))
        if [int(t) for t in raw.get("prompt_ids", [])] != prompt_ids:
            problems.append(f"{where}: prompt ids differ from the observer's")
        same = ([int(t) for t in raw.get("generated_ids", [])] == full and obs_row.get("token_ids_sha256") == ids_sha(mine_cut)
                and obs_row.get("token_count") == len(mine_cut)
                and tok.decode(mine_cut, skip_special_tokens=True).strip() == row.get("answer"))
        if not same:
            changed += 1
            if a.perturb_value == 1.0:
                problems.append(f"{where}: regenerated tokens differ from the run's")
        eff = call.get("effective", {}).get("eos_token_id")
        eff = [eff] if isinstance(eff, int) else list(eff or [])
        if set(eff) | ({tok.eos_token_id} if tok.eos_token_id is not None else set()) != set(stop_ids):
            problems.append(f"{where}: run stop tokens {eff} differ from mine {stop_ids}")
    report = {"checked": checked, "rows": len(rows), "changed": changed, "maxima": maxima,
              "perturb_value": a.perturb_value, "fraction": a.fraction, "problems": problems[:200],
              "problem_count": len(problems)}
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if a.perturb_value != 1.0:
        ok = changed > 0
        print(f"REPLAY85 negative control: {changed} of {checked} answers changed under x{a.perturb_value}"
              f" ({'the replay can see a wrong clamp' if ok else 'NOTHING CHANGED: the replay cannot see a wrong clamp'})")
        return 3 if ok else 1
    if problems:
        print(f"REPLAY85: FAIL, {len(problems)} problem(s) over {checked} answers; first: {problems[0]}")
        return 1
    if a.fraction < 1.0:
        print(f"REPLAY85: NON-PRODUCTION SAMPLE, {checked} answers match; not a PASS")
        return 3
    print(f"REPLAY85: PASS, all {checked} answers match token for token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
