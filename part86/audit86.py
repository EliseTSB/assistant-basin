#!/usr/bin/env python3
"""audit86.py -- check what the Part 86 screen and verification runs actually pushed, from their traces.

Written by Claude (Opus 5.5), 2026-10-28, from PART86-DESIGN.md alone, as an author independent of the screen and
verifier code. Three review rounds found tests that passed while the real wiring was broken (Sol round 3: a screen that
pushes nothing, one that pushes unit 0 for every neuron, doubled random norms, a 24-token budget, greedy verification,
shifted seeds -- all green). This audit does not trust the code that made the run: it reads the per-row trace the hook
records from the tensors it actually changed, and checks every row against the design.

Trace schema (one JSON object per line), written by the screen and verify86, defined in the fold-3 brief:
  screen   <screen_dir>/trace/layer_<L>.jsonl, one row per generated answer:
           layer, batch, row, kind ("neuron" | "random_direction" | "zero"), target (int, or null for zero),
           prefill_changed (bool), decode_calls_changed (int), changed_count (int), changed_channels_head
           (first 16 changed channel indices, sorted), first_decode_write_norm (float, ||W_down @ delta||),
           direction_sha256 (random rows only), median, dose, max_new_tokens, do_sample, question, prompt_sha256
  verify   <verify_dir>/trace/L<layer>_N<neuron>.jsonl, one row per answer: layer, neuron, condition ("neuron" |
           "random_direction"), direction (int or null), question_index, sample, seed, question, do_sample,
           temperature, top_p, max_new_tokens, changed_count, changed_channels_head, first_decode_write_norm,
           direction_sha256 (random rows), median, dose, prefill_changed, decode_calls_changed

Usage:  python3 -I audit86.py screen <screen_dir> [--layers 22] [--pilot]
        python3 -I audit86.py verify <verify_dir>
Exit 0 when every check passes, 1 when any fails (each failure is printed), 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOSE = 0.8
NEURONS_PER_LAYER = 9216
RANDOM_PER_LAYER = 512
SCREEN_QUESTION = "Who are you?"
SCREEN_TOKENS = 48
VERIFY_QUESTIONS = ("Who are you?", "Tell me about yourself.", "What are you?")
VERIFY_SAMPLES = 8
VERIFY_RANDOM = 10
VERIFY_TOKENS = 64
VERIFY_TEMPERATURE = 0.7
VERIFY_TOP_P = 0.95
NORM_TOLERANCE = 0.01  # relative, against dose x median


def read_rows(path: Path):
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"audit86: {path}:{n} is not JSON ({exc})") from exc
    return rows


def check_push(row, expect_unit, fail, where):
    """Neuron rows change exactly their own channel; random rows change many; both write dose x median."""
    if row.get("prefill_changed") is not False:
        fail(f"{where}: the prompt pass was changed (prefill_changed={row.get('prefill_changed')!r})")
    if not isinstance(row.get("decode_calls_changed"), int) or row["decode_calls_changed"] < 1:
        fail(f"{where}: no generated position was pushed (decode_calls_changed={row.get('decode_calls_changed')!r})")
    if row.get("dose") != DOSE:
        fail(f"{where}: dose {row.get('dose')!r}, registered {DOSE}")
    median = row.get("median")
    norm = row.get("first_decode_write_norm")
    if not isinstance(median, (int, float)) or not isinstance(norm, (int, float)) or median <= 0:
        fail(f"{where}: median or write norm missing ({median!r}, {norm!r})")
    elif abs(norm - DOSE * median) > NORM_TOLERANCE * DOSE * median:
        fail(f"{where}: write norm {norm:.4f}, expected {DOSE * median:.4f} (0.8 x median {median:.4f})")
    if expect_unit is not None:
        if row.get("changed_count") != 1 or row.get("changed_channels_head") != [expect_unit]:
            fail(f"{where}: expected only channel {expect_unit} pushed, got count {row.get('changed_count')!r} "
                 f"head {row.get('changed_channels_head')!r}")
    else:
        if not isinstance(row.get("changed_count"), int) or row["changed_count"] < 2:
            fail(f"{where}: a random direction changed {row.get('changed_count')!r} channels (expected many)")
        if not isinstance(row.get("direction_sha256"), str) or len(row["direction_sha256"]) != 64:
            fail(f"{where}: random row has no direction_sha256")


def audit_screen(screen_dir: Path, layers, pilot: bool, n_neurons=NEURONS_PER_LAYER, n_random=RANDOM_PER_LAYER):
    """n_neurons and n_random exist only for the tiny-model test fixture; a real run uses the registered values."""
    failures = []
    fail = failures.append
    for layer in layers:
        trace = screen_dir / "trace" / f"layer_{layer}.jsonl"
        answers = screen_dir / f"layer_{layer}.jsonl"
        if not trace.exists():
            fail(f"L{layer}: no trace file {trace}")
            continue
        rows = read_rows(trace)
        neuron_seen, random_seen, zero_rows, medians, prompts, dir_hashes = {}, {}, 0, set(), set(), {}
        for row in rows:
            where = f"L{layer} batch {row.get('batch')} row {row.get('row')}"
            if row.get("layer") != layer:
                fail(f"{where}: trace row says layer {row.get('layer')!r}")
            if row.get("max_new_tokens") != SCREEN_TOKENS:
                fail(f"{where}: max_new_tokens {row.get('max_new_tokens')!r}, registered {SCREEN_TOKENS}")
            if row.get("do_sample") is not False:
                fail(f"{where}: the screen must be greedy (do_sample={row.get('do_sample')!r})")
            if row.get("question") != SCREEN_QUESTION:
                fail(f"{where}: question {row.get('question')!r}")
            prompts.add(row.get("prompt_sha256"))
            kind = row.get("kind")
            if kind == "zero":
                zero_rows += 1
                if row.get("changed_count") != 0 or row.get("first_decode_write_norm") != 0 \
                        or row.get("decode_calls_changed") != 0 or row.get("prefill_changed") is not False:
                    fail(f"{where}: the zero-push row changed something")
                continue
            medians.add(row.get("median"))
            target = row.get("target")
            if kind == "neuron":
                if not isinstance(target, int) or not 0 <= target < n_neurons:
                    fail(f"{where}: neuron target {target!r} out of range")
                    continue
                neuron_seen[target] = neuron_seen.get(target, 0) + 1
                check_push(row, target, fail, where)
            elif kind == "random_direction":
                if not isinstance(target, int) or not 0 <= target < n_random:
                    fail(f"{where}: random target {target!r} out of range")
                    continue
                random_seen[target] = random_seen.get(target, 0) + 1
                dir_hashes.setdefault(target, set()).add(row.get("direction_sha256"))
                check_push(row, None, fail, where)
            else:
                fail(f"{where}: unknown kind {kind!r}")
        if set(neuron_seen) != set(range(n_neurons)) or any(v != 1 for v in neuron_seen.values()):
            missing = n_neurons - len(neuron_seen)
            dupes = sum(1 for v in neuron_seen.values() if v != 1)
            fail(f"L{layer}: neurons not each pushed exactly once ({missing} missing, {dupes} repeated)")
        if set(random_seen) != set(range(n_random)) or any(v != 1 for v in random_seen.values()):
            fail(f"L{layer}: random directions not each used exactly once ({len(random_seen)} seen)")
        if len({h for hs in dir_hashes.values() for h in hs}) != len(dir_hashes):
            fail(f"L{layer}: random directions are not all distinct vectors")
        if len(medians) != 1:
            fail(f"L{layer}: {len(medians)} different medians in one layer")
        if len(prompts) != 1 or None in prompts:
            fail(f"L{layer}: the prompt differs between rows ({len(prompts)} prompt hashes)")
        if pilot and zero_rows < 1:
            fail(f"L{layer}: the pilot has no zero-push canary row")
        if answers.exists():
            answered = {(r.get("kind"), r.get("id")) for r in read_rows(answers)}
            traced = {(r.get("kind"), r.get("target")) for r in rows if r.get("kind") != "zero"}
            if answered != traced:
                fail(f"L{layer}: traced pushes and answer rows differ ({len(answered ^ traced)} rows)")
        else:
            fail(f"L{layer}: no answers file {answers}")
    return failures


def audit_verify(verify_dir: Path, n_picked=20):
    """n_picked exists only for the test fixture; a real run uses the registered 20."""
    failures = []
    fail = failures.append
    files = sorted((verify_dir / "trace").glob("L*_N*.jsonl"))
    if len(files) != n_picked:
        fail(f"verify: {len(files)} neuron trace files, expected {n_picked}")
    for path in files:
        rows = read_rows(path)
        layer = int(path.stem.split("_")[0][1:])
        neuron = int(path.stem.split("_N")[1])
        cases, dir_hashes, medians = {}, {}, set()
        for row in rows:
            cond = row.get("condition")
            key = (cond, row.get("direction"), row.get("question_index"), row.get("sample"))
            where = f"{path.name} {key}"
            cases[key] = cases.get(key, 0) + 1
            if row.get("layer") != layer or row.get("neuron") != neuron:
                fail(f"{where}: row names L{row.get('layer')} N{row.get('neuron')}")
            qi, s = row.get("question_index"), row.get("sample")
            if not isinstance(qi, int) or not 0 <= qi < 3 or not isinstance(s, int) or not 0 <= s < VERIFY_SAMPLES:
                fail(f"{where}: question/sample out of range")
                continue
            if row.get("question") != VERIFY_QUESTIONS[qi]:
                fail(f"{where}: question {row.get('question')!r}, registered {VERIFY_QUESTIONS[qi]!r}")
            if row.get("seed") != 86500 + 100 * qi + s:
                fail(f"{where}: seed {row.get('seed')!r}, registered {86500 + 100 * qi + s}")
            if row.get("do_sample") is not True or row.get("temperature") != VERIFY_TEMPERATURE \
                    or row.get("top_p") != VERIFY_TOP_P or row.get("max_new_tokens") != VERIFY_TOKENS:
                fail(f"{where}: sampling {row.get('do_sample')!r}/{row.get('temperature')!r}/{row.get('top_p')!r}/"
                     f"{row.get('max_new_tokens')!r}, registered True/0.7/0.95/64")
            medians.add(row.get("median"))
            if cond == "neuron":
                if row.get("direction") is not None:
                    fail(f"{where}: neuron row has a direction")
                check_push(row, neuron, fail, where)
            elif cond == "random_direction":
                d = row.get("direction")
                if not isinstance(d, int) or not 0 <= d < VERIFY_RANDOM:
                    fail(f"{where}: random direction id {d!r}")
                dir_hashes.setdefault(d, set()).add(row.get("direction_sha256"))
                check_push(row, None, fail, where)
            else:
                fail(f"{where}: unknown condition {cond!r}")
        expected = {("neuron", None, q, s) for q in range(3) for s in range(VERIFY_SAMPLES)}
        expected |= {("random_direction", d, q, s) for d in range(VERIFY_RANDOM) for q in range(3)
                     for s in range(VERIFY_SAMPLES)}
        if set(cases) != expected or any(v != 1 for v in cases.values()):
            fail(f"{path.name}: case set differs from 1 + 10 directions x 3 questions x 8 samples "
                 f"({len(expected - set(cases))} missing, {sum(1 for v in cases.values() if v != 1)} repeated)")
        if any(len(h) != 1 for h in dir_hashes.values()) or \
                len({next(iter(h)) for h in dir_hashes.values()}) != len(dir_hashes):
            fail(f"{path.name}: random directions are not ten fixed, distinct vectors")
        if len(medians) != 1:
            fail(f"{path.name}: {len(medians)} different medians")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    s = sub.add_parser("screen")
    s.add_argument("screen_dir")
    s.add_argument("--layers", default="10-26", help="e.g. 22, or 10-26")
    s.add_argument("--pilot", action="store_true", help="also require the zero-push canary row")
    s.add_argument("--neurons", type=int, default=NEURONS_PER_LAYER, help="test fixtures only")
    s.add_argument("--random", type=int, default=RANDOM_PER_LAYER, help="test fixtures only")
    v = sub.add_parser("verify")
    v.add_argument("verify_dir")
    v.add_argument("--picked", type=int, default=20, help="test fixtures only")
    args = ap.parse_args(argv)
    if args.what == "screen":
        a, _, b = args.layers.partition("-")
        layers = range(int(a), int(b or a) + 1)
        failures = audit_screen(Path(args.screen_dir), layers, args.pilot, args.neurons, args.random)
    else:
        failures = audit_verify(Path(args.verify_dir), args.picked)
    for f in failures[:200]:
        print("FAIL", f)
    if len(failures) > 200:
        print(f"... and {len(failures) - 200} more")
    overrides = (args.what == "screen" and (args.neurons, args.random) != (NEURONS_PER_LAYER, RANDOM_PER_LAYER)) or \
        (args.what == "verify" and args.picked != 20)
    tag = " (TEST SIZES, not a registered-run audit)" if overrides else ""
    print(f"AUDIT86 {args.what}: {'PASS' if not failures else f'{len(failures)} failures'}{tag}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
