#!/usr/bin/env python3
"""Part 199: Part 170 on the eager attention kernel (PART199-DESIGN.md).

Usage: run199.py --out-dir results/part199 [--device xpu --model ...]
       run199.py --check-record results/part199/load.json
       run199.py --compare results/part199 results/part170

part170/run170.py's run() is called unchanged except for the three changes the design records:

1. Kernel. run170 loads through neuron_report.load_model, whose default is "eager". Each launch appends one entry to
   load.json as soon as the model has loaded, with neuron_report.LAST_LOAD, and rewrites it (through a temporary file
   and os.replace) after every chunk, so a launch killed by a signal still leaves its kernel, calls and row count. The
   run exits 3 if any recorded kernel is not eager, if the entries do not share one model, revision and resolved
   output folder (Gate R1), or if Part 170's own gates did not pass.
2. Push batch shape. run170.generate_with_bans is replaced by a version that generates pushed calls as chunks of 8
   rows, in the original row order, each chunk with its own bans and release steps; the directions mapping and the
   seed are passed unchanged (rows find their push by layer and slot). No-push calls and Stage 0 keep Part 170's
   shapes. Every call and its chunks are recorded in load.json ("calls"); Gate R3 requires every pushed call of more
   than one row to reach the model as chunks of exactly 8, and no chunk of any call to exceed 8 rows. Stage 0's
   zero-write check is a pushed call of one row and stays one.
3. Pilot stop rule. run168.pilot_rate_should_stop is replaced: after the first 3 push batches, stop if the measured
   rate projects the 5,376-answer push stage (push plus repeat) past 8 hours (below 0.1867 answers per second). It is a
   guard against a broken path, not a budget: the runner resumes at the next unwritten answer, so a task killed by the
   job runner's 6-hour limit is relaunched, not lost.

The kernel is checked before any row is generated: the wrapper hands run170.run a loader that loads through
neuron_report.load_model (eager by default) and refuses (SystemExit 3) unless both neuron_report.LAST_LOAD and the
model's config say eager. The output folder is locked (run199.lock, created exclusively) for the length of an
invocation, so two launches cannot share it; a launch killed by a signal leaves the lock, and the operator removes it
by hand after checking that no run199 process runs (no automatic reclaim: on the Windows job host os.kill(pid, 0)
would terminate the process rather than probe it).

The chunking sits inside run170.generate_with_bans, which is also where Part 169's block log is applied, so each chunk
carries its own log and the answers' blocked_steps survive the split. A call with more than 48 rows, or a pushed call
whose row count is not a multiple of 8, raises rather than being guessed at.

--compare prints the published and rerun values the predictions are read from: the drawn neurons and realized writes,
the Stage 1 lists, the frozen prompts, every cell's loop rate and verdict, and the loop-onset split of the rerun's
own loops. Cells and verdicts come from part170/score170.py's own tables, imported, not reimplemented.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

CHUNK = 8
MAX_CALL_ROWS = 48
PUSH_ANSWERS = 5376          # 8 prompts x 48 rows x 4 bans x 3 doses + 8 x 48 x rel50 at dose 1
PILOT_BATCHES = 3
PILOT_SECONDS = 8 * 3600
PILOT_MIN_RATE = PUSH_ANSWERS / PILOT_SECONDS   # 0.1867 answers per second


def chunk_sizes(n, size=CHUNK):
    """Row counts of the chunks a call of n rows is generated in; every chunk but the last is a full `size`."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"cannot chunk a call of {n} rows")
    if n > MAX_CALL_ROWS:
        raise ValueError(f"generate call of {n} rows exceeds the largest Part 170 call ({MAX_CALL_ROWS})")
    if n % size:
        raise ValueError(f"generate call of {n} rows is not a multiple of the chunk size {size}")
    return [size] * (n // size)


def split_call(rows, bans, release_after, sizes):
    """Slice one call into chunks, keeping the original row order and each row's own ban and release step.

    The directions mapping is NOT sliced: run168.generate_batch looks each row's push up by (layer, slot) in it, so
    every chunk is given the call's whole mapping."""
    if len(bans) != len(rows) or len(release_after) != len(rows):
        raise ValueError("rows, bans and release steps disagree on the call's row count")
    out, start = [], 0
    for size in sizes:
        stop = start + size
        out.append({"rows": list(rows[start:stop]),
                    "bans": list(bans[start:stop]), "release_after": list(release_after[start:stop])})
        start = stop
    if start != len(rows):
        raise ValueError(f"chunks cover {start} of {len(rows)} rows")
    return out


def chunked_generate(original, call_log=None, on_call=None):
    """Part 170's generate_with_bans, with pushed calls generated in chunks of 8 (design change 2).

    `original` is run170.generate_with_bans; its signature is kept exactly, so run170's call sites do not move. Each
    call is logged with its planned chunks and, as they return, the row counts actually passed to `original`
    ("observed"); Gate R3 reads the observed counts. A pushed one-row call whose directions mapping holds a single
    (layer, slot) is Stage 0's zero-write check and is flagged as such. `on_call` runs after every chunk.
    """
    def generate_with_bans(model, tok, layers, question, rows, directions, bans, device,
                           common, seed, release_after=None, pushed=True, max_new_tokens=80):
        releases = list(release_after or [None] * len(rows))
        n = len(rows)
        if not pushed:
            sizes = [n]                       # Part 170's no-push shape (5 or 6 lists)
        elif n == 1:
            sizes = [1]                       # Stage 0's zero-write hook: one prompt at a time
        else:
            sizes = chunk_sizes(n)
        entry = {"pushed": bool(pushed), "rows": n, "chunks": list(sizes), "observed": [], "complete": False,
                 "zero_write": bool(pushed and n == 1 and len(directions or {}) == 1), "prompt": question[:60]}
        if call_log is not None:
            call_log.append(entry)
        answers = []
        for part in split_call(rows, bans, releases, sizes):
            got = original(model, tok, layers, question, part["rows"], directions,
                           part["bans"], device, common, seed,
                           release_after=part["release_after"], pushed=pushed,
                           max_new_tokens=max_new_tokens)
            if len(got) != len(part["rows"]):
                raise ValueError(f"chunk of {len(part['rows'])} rows returned {len(got)} answers")
            entry["observed"].append(len(part["rows"]))
            answers.extend(got)
            if on_call is not None:
                on_call()
        entry["complete"] = True
        if on_call is not None:
            on_call()
        return answers
    return generate_with_bans


def pilot_rate_should_stop(answers, started, now):
    """Design change 3: stop if the first three push batches project the push stage past eight hours."""
    elapsed = float(now) - float(started)
    if elapsed <= 0:
        raise ValueError("pilot rate needs a positive elapsed time")
    rate = int(answers) / elapsed
    if rate < PILOT_MIN_RATE:
        print(json.dumps({"part199_pilot_stop": True, "answers_per_second": rate, "threshold": PILOT_MIN_RATE,
                          "rule": "push stage projected past 8 hours; Part 170's message below names its own 1.5"}),
              flush=True)
        return True
    return False


def check_kernels(loads):
    """Gate R1's kernel half: at least one recorded load, and every recorded load eager."""
    if not loads:
        return False, "no load recorded (no invocation wrote rows)"
    bad = [((entry.get("load") or {}).get("attn_implementation")) for entry in loads]
    bad = [kernel for kernel in bad if kernel != "eager"]
    if bad:
        return False, f"non-eager load(s) recorded: {bad!r}"
    return True, ""


def check_same_run(loads):
    """Gate R1's other half: one model, revision and output folder across entries."""
    if not loads:
        return False, "no load recorded"
    seen = {(entry.get("model"), entry.get("revision"), entry.get("out_dir")) for entry in loads}  # stored resolved
    if len(seen) != 1:
        return False, f"load entries disagree on model, revision or output folder: {sorted(map(str, seen))}"
    return True, ""


def check_record(path):
    """Host-side Gates R1 and R3 over every recorded invocation: exit 0 only if load.json records at least one load,
    all eager, one model, revision and folder, and every pushed call of more than one row went in chunks of 8."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    loads = rec.get("loads", [])
    ok_kernel, why_kernel = check_kernels(loads)
    ok_run, why_run = check_same_run(loads)
    shapes = call_shapes([call for entry in loads for call in entry.get("calls", [])])
    why_shape = "" if shapes["passed"] else f"Gate R3: {json.dumps(shapes)}"
    if ok_kernel and ok_run and shapes["passed"]:
        print(f"PART199 record ok: {len(loads)} recorded load(s), all eager, one model/revision/folder; "
              f"{shapes['pushed_calls']} pushed calls in {shapes['pushed_chunks']} chunks of 8", flush=True)
        return 0
    print(f"PART199 RECORD FAILED: {why_kernel or why_run or why_shape}", flush=True)
    return 3


def call_shapes(calls):
    """Gate R3, on what reached the model (the row counts observed as each chunk returned): every pushed call that is
    not Stage 0's zero-write check went as chunks of exactly 8 (a complete call's chunks cover its rows; a call cut
    short by a killed launch may cover fewer, never more), no chunk of any call had more than 8 rows, and the
    zero-write checks cover at most the 8 P8 prompts, each completed check one chunk of one row. A relaunch after a
    kill regenerates an interrupted call and Stage 0's checks, so both are counted by call and by prompt, not summed."""
    zero_checks = [call for call in calls if call.get("pushed") and call.get("zero_write")]
    pushed = [call for call in calls if call.get("pushed") and not call.get("zero_write")]
    observed = [int(size) for call in calls for size in call.get("observed", [])]
    biggest = max(observed, default=0)

    def bad_push(call):
        sizes = [int(size) for size in call.get("observed", [])]
        if any(size != CHUNK for size in sizes):
            return True
        if call.get("complete"):
            return sum(sizes) != int(call["rows"])
        return sum(sizes) > int(call["rows"])         # killed mid-call: fewer chunks, never more rows

    wrong = [call for call in pushed if bad_push(call)]
    bad_zero = [call for call in zero_checks
                if (call.get("observed") != [1] if call.get("complete") else call.get("observed") not in ([], [1]))]
    zero_prompts = {call.get("prompt") for call in zero_checks}
    complete_pushed = [call for call in pushed if call.get("complete")]
    passed = bool(complete_pushed and not wrong and biggest <= CHUNK and not bad_zero and len(zero_prompts) <= 8)
    return {"passed": passed, "pushed_calls": len(pushed), "complete_pushed_calls": len(complete_pushed),
            "pushed_chunks": sum(len(call.get("observed", [])) for call in pushed),
            "pushed_calls_not_in_chunks_of_8": len(wrong), "zero_write_checks": len(zero_checks),
            "zero_write_prompts": len(zero_prompts), "bad_zero_write_checks": len(bad_zero),
            "largest_chunk_rows": biggest, "calls": len(calls)}


def _finite(value):
    """The record is strict JSON: non-finite floats become null (a diagnostic NaN must not lose the outcome)."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _save_record(path, rec):
    """Write load.json whole, through a temporary file and os.replace, so a kill never leaves it half written."""
    import os
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_finite(rec), ensure_ascii=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_record(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"part": 199, "loads": []}


def _answers_written(out):
    """Complete (newline-terminated) rows only: run170's repair_tail cuts a torn last line, so it is not a row."""
    path = Path(out) / "answers.jsonl"
    if not path.exists():
        return 0
    return path.read_bytes().count(b"\n")


def eager_loader(args):
    """The loader handed to run170.run: Part 170's own load (neuron_report.load_model, then no grad and eval mode),
    refused before any generation unless the recorded kernel is eager."""
    def load(device):
        import neuron_report as nr
        from part155 import run155
        tok, model = nr.load_model(str(args.model), run155._common().REVISION, device)
        kernel = (nr.LAST_LOAD or {}).get("attn_implementation")
        config_kernel = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if kernel != "eager" or config_kernel != "eager":
            print(f"PART199 GATE R1 FAILED before generation: the model loaded with {kernel!r} "
                  f"(config {config_kernel!r}), not eager; nothing generated", flush=True)
            raise SystemExit(3)
        model.requires_grad_(False)
        model.eval()
        return tok, model
    return load


class FolderLock:
    """run199.lock in the output folder, created exclusively; a second launch on the same folder is refused. A stale
    lock (a killed task) names the pid and time that took it and is removed by hand after checking that nothing runs."""
    def __init__(self, folder):
        self.path = Path(folder) / "run199.lock"

    def __enter__(self):
        import os
        import time
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = self.path.read_text(encoding="utf-8", errors="replace").strip()
            print(f"PART199: {self.path} exists ({held}); another launch holds this folder, or a killed one left "
                  "it: check that no run199 process runs, then remove it by hand and relaunch", flush=True)
            raise SystemExit(3)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(f"pid {os.getpid()} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)
        return False


def run(args, loader=None):
    """Part 170's run() unchanged except for the three recorded changes, under a folder lock."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with FolderLock(out):
        return _run_locked(args, loader if loader is not None else eager_loader(args))


def _run_locked(args, loader):
    """The load entry is written as soon as the loader returns (kernel known, before any generation) and rewritten
    after every chunk, so a launch killed by a signal still leaves its kernel, its calls and its row count."""
    import time
    import neuron_report as nr
    from part168 import run168
    from part170 import run170

    out = Path(args.out_dir)
    load_path = out / "load.json"
    before = _answers_written(out)
    calls = []
    state = {"index": None, "entry": None}

    def flush(status=None, **extra):
        if state["entry"] is None:
            return
        entry = state["entry"]
        entry["answers_written"] = _answers_written(out) - before
        entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if status:
            entry["status"] = status
        entry.update(extra)
        rec = _read_record(load_path)
        if state["index"] is None:
            rec.setdefault("loads", []).append(entry)
            state["index"] = len(rec["loads"]) - 1
        else:
            rec["loads"][state["index"]] = entry
        _save_record(load_path, rec)

    def recording_loader(device):
        tok, model = loader(device)       # refuses a non-eager load before anything is recorded or generated
        state["entry"] = {"load": nr.LAST_LOAD, "model": str(args.model), "revision": _revision(),
                          "out_dir": str(out.resolve()), "argv": list(sys.argv[1:]), "calls": calls,
                          "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running"}
        flush()
        return tok, model

    original_generate = run170.generate_with_bans
    original_pilot = run168.pilot_rate_should_stop
    nr.LAST_LOAD = None
    run170.generate_with_bans = chunked_generate(original_generate, calls, on_call=flush)
    run168.pilot_rate_should_stop = pilot_rate_should_stop
    summary, failure, stopped = None, None, False
    try:
        summary = run170.run(args, loader=recording_loader)
    except SystemExit as exc:          # Part 170's gates, its pilot, or this wrapper's loader refusal: a stop, exit 3
        failure, stopped = f"SystemExit: {exc}", True
    except BaseException as exc:       # a killed model, a bug: record the rows it wrote, then let it propagate
        failure = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        run170.generate_with_bans = original_generate
        run168.pilot_rate_should_stop = original_pilot
        try:
            flush(status="stopped" if failure else "done", summary=summary, failure=failure)
        except Exception as exc:       # a failed final write must not replace the run's own outcome
            print(f"PART199: the final load.json write failed ({type(exc).__name__}: {exc}); the last per-chunk "
                  "write stands", flush=True)
        written = _answers_written(out) - before
        if failure is not None:
            print(f"PART199 run stopped: {failure}; {written} row(s) written this invocation", flush=True)
    if stopped:
        print("PART199: stopped by a gate, the pilot or the loader (see above); no reading is made from this launch",
              flush=True)
        return 3
    loads = _read_record(load_path).get("loads", [])
    ok_kernel, why_kernel = check_kernels(loads)
    ok_run, why_run = check_same_run(loads)
    shapes = call_shapes(calls)
    if not calls:
        # nothing generated (a resumed run with every answer done): no shape to check, and the gate is not claimed
        shapes["passed"] = None
        shapes["note"] = "no generate call in this invocation; Gate R3 is read from the invocation(s) that generated"
    gates170 = (summary or {}).get("gates") or {}
    report = {"answers_written": written, "gate_r1": bool(ok_kernel and ok_run),
              "gate_r1_why": why_kernel or why_run, "gate_r3": shapes, "part170_gates_passed": gates170.get("passed")}
    print(json.dumps(report, ensure_ascii=True, allow_nan=False), flush=True)
    if not report["gate_r1"]:
        print(f"PART199 GATE R1 FAILED: {report['gate_r1_why']}", flush=True)
        return 3
    if shapes["passed"] is False:
        print(f"PART199 GATE R3 FAILED: {json.dumps(shapes)}", flush=True)
        return 3
    if gates170.get("passed") is not True:
        failed = [k for k in (f"gate{i}" for i in range(1, 9)) if gates170.get(k) is False]
        print(f"PART199: Part 170's gates did not pass ({failed or 'no gate record'}); every reading stops", flush=True)
        return 3
    print(f"PART199 gates R1, R3 and Part 170's eight passed: {len(loads)} load(s) eager; {shapes['pushed_calls']} "
          f"pushed calls in {shapes['pushed_chunks']} chunks of 8 this invocation", flush=True)
    print(json.dumps(_finite(summary), ensure_ascii=True, allow_nan=False), flush=True)
    return 0


def _revision():
    from part155 import run155
    return getattr(run155._common(), "REVISION", None)


# --------------------------------------------------------------------------------------------------------------
# --compare: the published and rerun values the predictions are read from.

def _read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _frozen(stage0):
    stage0 = stage0 or {}
    return {group: sorted(map(int, stage0.get("frozen_indices", {}).get(group, [])))
            for group in ("p8", "m96", "q96")}


def _directions(norms):
    rows = (norms or {}).get("directions", [])
    return {(int(r["layer"]), int(r["slot"]), float(r["dose"])): r for r in rows}


def gate_r2(published_norms, rerun_norms):
    """Gate R2: the same 144 (layer, slot, dose) directions with the same neurons as Part 170, and every realized
    write within Part 170's own Gate 4 tolerance (run170.gate4_size on the rerun's norms)."""
    from part170 import run170
    a, b = _directions(published_norms), _directions(rerun_norms)
    mismatched = sorted(list(k) + [a[k].get("neuron"), b[k].get("neuron")] for k in set(a) & set(b)
                        if int(a[k].get("neuron", -1)) != int(b[k].get("neuron", -2)))
    gate4 = run170.gate4_size(rerun_norms or {})
    worst = max((abs(float(r["realized_write_norm"]) - float(r["target_norm"])) / float(r["target_norm"])
                 for r in b.values() if float(r.get("target_norm", 0) or 0) > 0), default=None)
    passed = bool(a) and set(a) == set(b) and not mismatched and bool(gate4.get("passed"))
    return {"passed": passed, "published_directions": len(a), "rerun_directions": len(b),
            "same_keys": set(a) == set(b), "neuron_mismatches": mismatched[:10],
            "rerun_gate4_passed": bool(gate4.get("passed")), "rerun_worst_relative_write_error": worst}


LIST_FIELDS = ("b5", "b50", "b200", "other50")


def compare_lists(pub, new):
    a = {str(r.get("prompt_id")): r for r in pub}
    b = {str(r.get("prompt_id")): r for r in new}
    shared = sorted(set(a) & set(b))
    differing = [p for p in shared if any(list(map(int, a[p].get(f) or [])) != list(map(int, b[p].get(f) or []))
                                          for f in LIST_FIELDS)]
    return {"published_lists": len(a), "rerun_lists": len(b), "shared": len(shared),
            "identical_on_every_list": len(shared) - len(differing), "differing_prompt_ids": differing[:20]}


def _frozen(stage0):
    stage0 = stage0 or {}
    return {group: sorted(map(int, stage0.get("frozen_indices", {}).get(group, [])))
            for group in ("p8", "m96", "q96")}


def compare_frozen(pub, new):
    a, b = _frozen(pub), _frozen(new)
    return {group: {"published": a[group], "rerun": b[group], "published_count": len(a[group]),
                    "rerun_count": len(b[group]), "overlap": len(set(a[group]) & set(b[group]))}
            for group in ("p8", "m96", "q96")}


def compare_gates(pub_gates, new_gates):
    out = {}
    for name, gates in (("published", pub_gates or {}), ("rerun", new_gates or {})):
        diag = gates.get("diagnostics", {})
        out[name] = {**{k: gates.get(k) for k in [f"gate{i}" for i in range(1, 9)] + ["passed"]},
                     **{k: diag.get(k) for k in ("gate6_dose1_b0_loop_rate", "gate6_dose1_b50_mean_blocked_steps",
                                                 "gate8_rows", "gate8_differing_answers", "gate8_noise_rate")}}
    return out


CELL_FIELDS = ("stage", "arm", "dose", "group", "ban")


def compare_cells(pub_score, new_score):
    """Every per_cell row of score170.py beside the published one, and every prediction's verdict beside its own."""
    def cells(score):
        return {tuple(row.get(field) for field in CELL_FIELDS): row for row in (score or {}).get("per_cell", [])}

    a, b = cells(pub_score), cells(new_score)
    rows = []
    for key in sorted(set(a) | set(b), key=str):
        pa, pb = a.get(key) or {}, b.get(key) or {}
        rows.append({"cell": dict(zip(CELL_FIELDS, key)),
                     **{f"published_{f}": pa.get(f) for f in ("rows", "loop_rate", "broken_rate")},
                     **{f"rerun_{f}": pb.get(f) for f in ("rows", "loop_rate", "broken_rate")}})
    verdicts = []
    va = (pub_score or {}).get("predictions", {})
    vb = (new_score or {}).get("predictions", {})
    for name in sorted(set(va) | set(vb)):
        pa, pb = va.get(name) or {}, vb.get(name) or {}
        verdicts.append({"prediction": name, "published_pass": pa.get("pass"), "rerun_pass": pb.get("pass"),
                         "published_estimate": pa.get("estimate"), "rerun_estimate": pb.get("estimate"),
                         "published_interval_95": pa.get("interval_95"), "rerun_interval_95": pb.get("interval_95"),
                         "changed": None if not pb else pa.get("pass") != pb.get("pass")})
    return {"cells": rows, "predictions": verdicts}


ONSET_GROUPING = "stage,group,arm,dose,ban"


def pick_onset(lines):
    """From loop_onset.py's printed JSON lines: the dose-1 b50 push cell (prediction 199.9) and the total."""
    cells = [json.loads(line) for line in lines if line.strip().startswith("{")]
    total = next((c["total"] for c in cells if "total" in c), None)
    target = {"stage": "push", "group": "p8", "arm": "n+", "dose": 1.0, "ban": "b50"}
    hit = [c for c in cells if c.get("cell") == target]
    return {"dose1_b50": ({k: hit[0][k] for k in ("loops", "before", "after", "unplaced")} if hit else None),
            "total": total}


def loop_onset_split(rerun_answers, tokenizer_dir):
    """Run the published part199/onset/loop_onset.py unchanged on the rerun's answers (its own Stage 0 rows carry the
    prompt ids the template check needs) and read the dose-1 b50 cell."""
    import subprocess
    cmd = [sys.executable, "-I", str(HERE / "onset" / "loop_onset.py"), str(rerun_answers), str(ROOT),
           str(tokenizer_dir), str(rerun_answers), ONSET_GROUPING]
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"loop_onset.py failed ({done.returncode}): {done.stderr[-500:]}")
    return pick_onset(done.stdout.splitlines())


NOISE_LIMIT = 19     # 5% of the 384 dose-1 b0 repeat answers: this part's smallest effect worth knowing

REQUIRED = ("norms.json", "lists.jsonl", "stage0.json", "prelaunch_gates.json", "score.json")


def compare(rerun_dir, published_dir, tokenizer_dir=None):
    """Everything the predictions are read from, published beside rerun (design: --compare <rerun> <published>)."""
    rerun, published = Path(rerun_dir), Path(published_dir)
    for folder in (rerun, published):
        for name in REQUIRED:
            if not (folder / name).exists():
                raise FileNotFoundError(str(folder / name))
    out = {
        "gate_r2": gate_r2(_read_json(published / "norms.json"), _read_json(rerun / "norms.json")),
        "lists": compare_lists(_read_jsonl(published / "lists.jsonl"), _read_jsonl(rerun / "lists.jsonl")),
        "frozen": compare_frozen(_read_json(published / "stage0.json"), _read_json(rerun / "stage0.json")),
        "gates": compare_gates(_read_json(published / "prelaunch_gates.json"),
                               _read_json(rerun / "prelaunch_gates.json")),
        "cells_and_predictions": compare_cells(_read_json(published / "score.json"), _read_json(rerun / "score.json")),
    }
    differing = out["gates"]["rerun"].get("gate8_differing_answers")
    out["noise_limited"] = bool(differing is not None and int(differing) > NOISE_LIMIT)
    if tokenizer_dir is not None:
        out["loop_onset"] = loop_onset_split(rerun / "answers.jsonl", tokenizer_dir)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--check-record":
        return check_record(argv[1])
    if argv and argv[0] == "--compare":
        ap = argparse.ArgumentParser(prog="run199.py --compare")
        ap.add_argument("rerun_dir")
        ap.add_argument("published_dir")
        ap.add_argument("--tokenizer", default=None, help="Qwen3.5-4B snapshot dir; adds the loop-onset split")
        cargs = ap.parse_args(argv[1:])
        try:
            result = compare(cargs.rerun_dir, cargs.published_dir, cargs.tokenizer)
        except FileNotFoundError as exc:
            print(f"PART199 COMPARE: missing file {exc}", flush=True)
            return 2
        print(json.dumps(result, ensure_ascii=True, allow_nan=False, indent=2), flush=True)
        if result["gates"]["rerun"].get("passed") is not True:
            print("PART199: the rerun's Part 170 gates did not pass (prelaunch_gates.json); readings stop", flush=True)
            return 3
        if not result["gate_r2"]["passed"]:
            print("PART199 GATE R2 FAILED: the rerun's directions are not Part 170's; readings stop", flush=True)
            return 3
        if result["noise_limited"]:
            print("PART199: Gate 8's repeat changed more than 19 of 384 answers: dose-1 cell rates and rises are "
                  "noise-limited", flush=True)
        print("PART199 gate R2 passed", flush=True)
        return 0
    from part165 import run165
    from part170 import run170
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=run165.MODEL_DEFAULT)
    ap.add_argument("--out-dir", default="results/part199")
    ap.add_argument("--batch-size", type=int, default=run170.BATCH_SIZE)
    args = ap.parse_args(argv)
    if args.batch_size != run170.BATCH_SIZE:
        ap.error("registered batch size is 64 (the push shape is chunked inside the wrapper)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
