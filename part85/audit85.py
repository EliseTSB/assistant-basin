#!/usr/bin/env python3
"""audit85.py -- check what a Part 85 run actually did at one checkpoint, from its per-answer trace.

Written by Claude (Opus 5.5), 2026-10-28, from PART85-DESIGN.md alone, as an author independent of part85.py. Three
review rounds and Claude's own mutations found tests that passed with the real loop broken (a wrong layer, a skipped
chat template, temperature 1.0, the first question asked three times). This audit reads the trace the run records from
what it actually passed to generate and what the hook actually wrote, and checks every row against the design.

Trace schema, one JSON object per generated answer, at <cp_dir>/trace/<mode_tag>-<frame>--<cell>.jsonl (the same name
as the answers file; mode_tag is pilot<N> or main):
  checkpoint, mode, attempt, frame ("plain" | "chat"), cell, question_index, sample, seed,
  question       the question string handed to the prompt builder
  prompt_text    tokenizer.decode(input_ids actually passed to generate, skip_special_tokens=True)
  chat_template_applied   true only when apply_chat_template built this prompt
  generate_kwargs   {"do_sample", "temperature", "top_p", "max_new_tokens"} as passed to generate
  hooked_module  the qualified name of the module the clamp hook is registered on
  unit, set_value   the unit and value the hook set (null for unclamped)
  prompt_positions_set, generated_positions_set   positions where an observer registered AFTER the clamp saw the
                 unit's down_proj input equal to set_value (0 and 0 for unclamped)
  observed_min, observed_max   that observer's min and max of the unit's input over the whole answer
  other_channels_changed   true if the hook changed any other channel of its input

Usage: python3 -I audit85.py <cp_dir> --checkpoint KEY --mode pilot|main [--attempt N] [--chat-only]
       [--template-file FILE]
Exit 0 when every check passes, 1 when any fails (each failure printed), 2 on unreadable input.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
PLAIN_FRAME = "Question: {q}\nAnswer:"
TEMPERATURE, TOP_P, MAX_NEW_TOKENS = 0.7, 0.95, 64
TARGET, CONTROL, ABSOLUTE = 6978, 7884, 9.75
POSTTRAINED = ("sft", "dpo", "instruct")
LAYER_SUFFIX = "layers.10.mlp.down_proj"
REL_TOL = 1e-4


def read_rows(path: Path):
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"audit85: cannot read {path}: {exc}") from exc


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and abs(a - b) <= REL_TOL * max(1.0, abs(b))


def expected_cells(checkpoint, mode, chat_only):
    if mode == "pilot":
        return [("plain", "unclamped"), ("plain", "own_max")], 8
    cells = []
    if not chat_only:
        cells += [("plain", c) for c in ("unclamped", "own_max", "absolute", "control")]
    if checkpoint in POSTTRAINED:
        cells += [("chat", c) for c in ("unclamped", "own_max", "absolute", "control")]
    return cells, 16


def expected_clamp(cell, maxima):
    if cell == "unclamped":
        return None, None
    if cell == "own_max":
        return TARGET, 1.5 * float(maxima[str(TARGET)])
    if cell == "absolute":
        return TARGET, ABSOLUTE
    if cell == "control":
        return CONTROL, 1.5 * float(maxima[str(CONTROL)])
    raise ValueError(cell)


def audit(cp_dir: Path, checkpoint: str, mode: str, attempt: int, chat_only: bool, template: str | None):
    failures = []
    fail = failures.append
    step1_path = cp_dir / "step1.json"
    if not step1_path.exists():
        return [f"{cp_dir}: no step1.json"]
    step1 = json.loads(step1_path.read_text(encoding="utf-8"))
    maxima = step1.get("neutral_activation_maxima") or {}
    if str(TARGET) not in maxima or str(CONTROL) not in maxima:
        fail("step1.json lacks the target or control maximum")
        return failures
    if mode == "main" and checkpoint not in POSTTRAINED and chat_only:
        fail(f"{checkpoint}: a chat-only main run must not include a pretraining checkpoint")
    frame_text = template if template is not None else PLAIN_FRAME
    mode_tag = f"pilot{attempt}" if mode == "pilot" else "main"
    cells, samples = expected_cells(checkpoint, mode, chat_only)
    trace_dir = cp_dir / "trace"
    present = {p.name for p in trace_dir.glob(f"{mode_tag}-*--*.jsonl")} if trace_dir.exists() else set()
    wanted = {f"{mode_tag}-{f}--{c}.jsonl" for f, c in cells}
    for extra in sorted(present - wanted):
        fail(f"unexpected trace file {extra}")
    for frame, cell in cells:
        name = f"{mode_tag}-{frame}--{cell}.jsonl"
        tpath, apath = trace_dir / name, cp_dir / "answers" / name
        if not tpath.exists():
            fail(f"{name}: no trace")
            continue
        rows = read_rows(tpath)
        unit, value = expected_clamp(cell, maxima)
        seen = {}
        for row in rows:
            qi, s = row.get("question_index"), row.get("sample")
            where = f"{name} q{qi} s{s}"
            seen[(qi, s)] = seen.get((qi, s), 0) + 1
            if row.get("checkpoint") != checkpoint or row.get("mode") != mode or row.get("frame") != frame \
                    or row.get("cell") != cell or (mode == "pilot" and row.get("attempt") != attempt):
                fail(f"{where}: row identity {row.get('checkpoint')}/{row.get('mode')}/{row.get('attempt')}/"
                     f"{row.get('frame')}/{row.get('cell')}")
            if not isinstance(qi, int) or not 0 <= qi < 3 or not isinstance(s, int) or not 0 <= s < samples:
                fail(f"{where}: question/sample out of range")
                continue
            q = QUESTIONS[qi]
            if row.get("question") != q:
                fail(f"{where}: question {row.get('question')!r}, registered {q!r}")
            if row.get("seed") != 85000 + 100 * qi + s:
                fail(f"{where}: seed {row.get('seed')!r}, registered {85000 + 100 * qi + s}")
            kw = row.get("generate_kwargs") or {}
            if kw.get("do_sample") is not True or kw.get("temperature") != TEMPERATURE or kw.get("top_p") != TOP_P \
                    or kw.get("max_new_tokens") != MAX_NEW_TOKENS:
                fail(f"{where}: generate {kw}, registered do_sample True, 0.7, 0.95, 64")
            prompt = row.get("prompt_text")
            if not isinstance(prompt, str) or q not in prompt:
                fail(f"{where}: the prompt does not contain the question")
            elif frame == "plain":
                if row.get("chat_template_applied") is not False:
                    fail(f"{where}: plain frame built with the chat template")
                if prompt.strip() != frame_text.format(q=q).strip():
                    fail(f"{where}: plain prompt {prompt!r} is not the registered frame")
            elif row.get("chat_template_applied") is not True:
                fail(f"{where}: chat frame without the chat template")
            if not str(row.get("hooked_module", "")).endswith(LAYER_SUFFIX):
                fail(f"{where}: hook on {row.get('hooked_module')!r}, registered layer 10 down_proj")
            if row.get("other_channels_changed") is not False:
                fail(f"{where}: the hook changed other channels")
            if unit is None:
                if row.get("unit") is not None or row.get("set_value") is not None \
                        or row.get("prompt_positions_set") != 0 or row.get("generated_positions_set") != 0:
                    fail(f"{where}: an unclamped answer shows a clamp")
            else:
                if row.get("unit") != unit or not close(row.get("set_value"), value):
                    fail(f"{where}: clamp {row.get('unit')}={row.get('set_value')!r}, registered {unit}={value:.4f}")
                if not isinstance(row.get("prompt_positions_set"), int) or row["prompt_positions_set"] < 1 \
                        or not isinstance(row.get("generated_positions_set"), int) or row["generated_positions_set"] < 1:
                    fail(f"{where}: clamp not seen on both prompt and generated positions")
                if not (close(row.get("observed_min"), value) and close(row.get("observed_max"), value)):
                    fail(f"{where}: observer saw {row.get('observed_min')!r}..{row.get('observed_max')!r}, "
                         f"registered {value:.4f}")
        want = {(q, s) for q in range(3) for s in range(samples)}
        if set(seen) != want or any(v != 1 for v in seen.values()):
            fail(f"{name}: answers are not 3 questions x {samples} samples, each once")
        if apath.exists():
            ans = {(r.get("question_index"), r.get("sample")): r for r in read_rows(apath)}
            if set(ans) != set(seen):
                fail(f"{name}: trace and answers cover different (question, sample) pairs")
            for row in rows:
                a = ans.get((row.get("question_index"), row.get("sample")))
                if a is not None and (a.get("unit") != row.get("unit") or a.get("seed") != row.get("seed")):
                    fail(f"{name} q{row.get('question_index')} s{row.get('sample')}: answer row disagrees with trace")
        else:
            fail(f"{name}: no answers file")
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cp_dir")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--mode", required=True, choices=("pilot", "main"))
    ap.add_argument("--attempt", type=int, default=1)
    ap.add_argument("--chat-only", action="store_true")
    ap.add_argument("--template-file", help="the written attempt-2 plain frame with one {q}")
    args = ap.parse_args(argv)
    template = Path(args.template_file).read_text(encoding="utf-8") if args.template_file else None
    failures = audit(Path(args.cp_dir), args.checkpoint, args.mode, args.attempt, args.chat_only, template)
    for f in failures[:200]:
        print("FAIL", f)
    if len(failures) > 200:
        print(f"... and {len(failures) - 200} more")
    print(f"AUDIT85 {args.checkpoint} {args.mode}: {'PASS' if not failures else f'{len(failures)} failures'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
