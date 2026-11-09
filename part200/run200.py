#!/usr/bin/env python3
"""Part 200: Part 163 on the eager attention kernel (PART200-DESIGN.md, review round 1 folded).

Usage: run200.py --device xpu --model ... --units part163/units163.json --out results/part200/answers.jsonl --batch-size 64
       run200.py --check-record results/part200/load.json
       run200.py --compare results/part200 results/part163

part163/run163.py's run() is called unchanged except for the four changes the design records:

1. Kernel. The eager loader loads through neuron_report.load_model (eager by default since 2026-11-03), applies Part
   163's no-grad and eval settings, and exits 3 before any generation unless neuron_report.LAST_LOAD and the model's
   config._attn_implementation both record eager. It KEEPS the (tok, model) it loaded (an EagerLoader whose __call__
   loads once and remembers), so the batch-1 stage uses the same model and the same `layers`: load_model is called at
   most once in a process, and load.json gets one entry per launch. The folder lock (run200.lock), the load-record
   writer and the kernel/same-run checks are Part 199's (part199/run199.py), imported, not copied.
2. Batch shape. Part 163's generate_fn (part160.generate_checked) is wrapped so every pushed call is generated in
   chunks of exactly 8 rows, in the original row order, each layer's push tensor sliced to the chunk's rows. The
   no-push calls (one row each) and the batch-1 calls (one row each) keep their shape. Every call and its chunks are
   recorded; Gate R3 reads the row counts actually passed to Part 163's generator as each chunk returns: every chunk
   of a pushed battery or recipe call has exactly 8 rows, a completed call's chunks cover its rows, no-push and
   batch-1 calls are single rows.
3. Pilot stop rule. run163.PILOT_MIN_RATE is set to 0.17 answers per second for this run and restored afterwards; the
   rate below which the 3,616 answers would not finish within the 6-hour task limit. Part 163's Gate 1 check inside
   the pilot is unchanged. Both pilot stops (rate and Gate 1) are SystemExit from run163.run; the wrapper records the
   stop in load.json with status "stopped" and exits 3.
4. Batch-1 comparison. Question 0 ("What is your name?") is regenerated for all 200 units in units163.json order,
   each as a ONE-ROW call, with that unit's push (run163.pushes_for on a one-row list), into
   results/part200/batch1.jsonl (not answers.jsonl). It runs on every launch until all 200 units are written,
   skipping units already written, whether or not run163.run wrote rows in this launch. The wrapper counts, per arm
   and in total, the answers whose text differs from the same unit's chunked battery answer; more than 20 of 200 is
   flagged "shape-sensitive" (a flag only, it does not change a verdict).

The chunker, the chunk shapes and the load-record reader are written here rather than reused from part199/run199.py:
Part 199's chunked_generate wraps run170.generate_with_bans, whose call carries bans, release steps and a directions
mapping looked up by (layer, slot). Part 163's generate_fn is part160.generate_checked's (model, tok, layers, question,
pushes, max_new_tokens, device), whose pushes ARE the per-layer tensors, so the chunk must slice the tensors themselves
and the signature differs. The lock, record writer and kernel checks are reused; only the lock's file name changes
(Run200Lock, run200.lock).

--compare writes results/part200/compare.json with two sections, as PART200-DESIGN.md lists them. Scored: Gate R2,
Part 163's three gates, the five verdicts with estimates and intervals, broken counts per arm over 1,600, at-the-cap
counts per arm (token_count 80), the Part 161-rule Qwen counts, and the batch-1 differing counts with the
shape-sensitive flag. Exploratory: Part 191's mid-phrase count among answers under the cap (mid_phrase imported from
part199/report199.py), identical battery answers, identical no-push answers (16), and per-layer broken counts. The
Part 86 triage and the name/maker rules are imported from part86/triage86.py and part163/score163.py; Part 163's gates
and verdicts come from score163.score_rows on the rerun's answers (or its published score.json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part199 import run199  # noqa: E402  (load record, kernel checks: reused, not copied)

CHUNK = 8
BATCH_SIZE = 64
MAX_NEW_TOKENS = 80                 # Part 163's cap; a battery answer "reaches the cap" at this many tokens
BATCH1_ROWS = 200                   # question 0's battery rows, one per unit, at batch 1
SHAPE_SENSITIVE_LIMIT = 20          # design, Controls: more than 20 of 200 batch-1 answers differing flags the arms
PILOT_MIN_RATE = 0.17               # design, Method item 3: a guard against a broken path, not a time budget
LOCK_NAME = "run200.lock"           # design, change 6: Part 199's folder lock under this part's name

# Part 161's rule (PART200-DESIGN.md, --compare): case-sensitive \bQwen\b, among maker answers that are not broken and
# hold none of alibaba, tongyi, 通义 (case-insensitive), as in results/part163/extra163.py's gone_name(mode="base").
PART161_QWEN = re.compile(r"\bQwen\b")
PART161_EXCLUDE = re.compile(r"alibaba|tongyi|通义", re.IGNORECASE)


# --------------------------------------------------------------------------------------------- change 6: lock

class Run200Lock(run199.FolderLock):
    """Part 199's FolderLock with only the path changed: results/part200/run200.lock. Created exclusively; a second
    launch on the same folder is refused with exit 3, and a stale lock (a killed task) is removed by hand."""

    def __init__(self, folder):
        super().__init__(folder)
        self.path = Path(folder) / LOCK_NAME


# --------------------------------------------------------------------------------------------- change 2: batch shape

def chunk_sizes(n, size=CHUNK):
    """Row counts of the chunks a pushed call of n rows is generated in: `size` rows each, the last the remainder."""
    n = int(n)
    if n <= 0:
        raise ValueError(f"cannot chunk a pushed call of {n} rows")
    return [size] * (n // size) + ([n % size] if n % size else [])


def chunked_generate(original, call_log=None, on_chunk=None, stage="run"):
    """Part 163's generate_fn, with every pushed call generated in chunks of exactly 8 rows (design change 2).

    `original` is part160.generate_checked; its signature is kept exactly, so run163's call sites do not move. Part
    163's pushes are {layer: [rows, width] tensor} with one row per unit in the call's order, so a chunk is the same
    tensors sliced to the chunk's rows, and the answers are concatenated in the original row order. The no-push calls
    (pushes empty, one answer) are passed through unchanged. Each call is logged with its planned chunks and, as they
    return, the row counts actually passed to `original` ("observed"); Gate R3 reads the observed counts. `on_chunk`
    runs after every chunk and once when the call completes. `stage` names the stage in each log entry ("run" for
    run163.run's calls, "batch1" for the one-row batch-1 calls, which Gate R3 reads as single rows).
    """
    def generate(model, tok, layers, question, pushes, max_new_tokens, device):
        if not pushes:
            answers = original(model, tok, layers, question, pushes, max_new_tokens, device)
            if call_log is not None:
                call_log.append({"pushed": False, "stage": stage, "rows": 1, "chunks": [1], "observed": [1],
                                 "complete": True, "prompt": question[:60]})
            if on_chunk is not None:
                on_chunk()
            return answers
        sizes = {int(push.shape[0]) for push in pushes.values()}
        if len(sizes) != 1:
            raise ValueError(f"push tensors disagree on the call's row count: {sorted(sizes)}")
        n = sizes.pop()
        planned = chunk_sizes(n)
        entry = {"pushed": True, "stage": stage, "rows": n, "chunks": list(planned), "observed": [], "complete": False,
                 "prompt": question[:60]}
        if call_log is not None:
            call_log.append(entry)
        answers, start = [], 0
        for size in planned:
            stop = start + size
            part = {layer: push[start:stop] for layer, push in pushes.items()}
            got = original(model, tok, layers, question, part, max_new_tokens, device)
            if len(got) != size:
                raise ValueError(f"chunk of {size} rows returned {len(got)} answers")
            entry["observed"].append(size)
            answers.extend(got)
            start = stop
            if on_chunk is not None:
                on_chunk()
        if start != n:
            raise ValueError(f"chunks cover {start} of {n} rows")
        entry["complete"] = True
        if on_chunk is not None:
            on_chunk()
        return answers
    return generate


# --------------------------------------------------------------------------------------------- change 3: pilot rule

def set_pilot_min_rate(rate):
    """Set run163.PILOT_MIN_RATE for this wrapper's run; returns what was there, for restore_pilot_min_rate."""
    from part163 import run163
    previous = run163.PILOT_MIN_RATE
    run163.PILOT_MIN_RATE = float(rate)
    return previous


def restore_pilot_min_rate(previous):
    from part163 import run163
    run163.PILOT_MIN_RATE = previous


# --------------------------------------------------------------------------------------------- change 1: kernel

class EagerLoader:
    """The loader handed to run163.run and to the batch-1 stage: neuron_report.load_model, then Part 163's no-grad and
    eval settings, refused (SystemExit 3) before any generation unless both the recorded load and the model's config
    say eager. It loads ONCE and keeps the (tok, model) it loaded (design change 3), so a second call returns the same
    pair and load_model is never called twice in one process."""

    def __init__(self, args):
        self.args = args
        self.loaded = None

    def __call__(self, device):
        if self.loaded is not None:
            return self.loaded
        import neuron_report as nr
        from part155 import run155
        tok, model = nr.load_model(str(self.args.model), run155._common().REVISION, device)
        kernel = (nr.LAST_LOAD or {}).get("attn_implementation")
        config_kernel = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if kernel != "eager" or config_kernel != "eager":
            print(f"PART200 GATE R1 FAILED before generation: the model loaded with {kernel!r} "
                  f"(config {config_kernel!r}), not eager; nothing generated", flush=True)
            raise SystemExit(3)
        model.requires_grad_(False)
        model.eval()
        self.loaded = (tok, model)
        return self.loaded


def eager_loader(args):
    """An EagerLoader for `args` (kept as a function for the call sites that build one)."""
    return EagerLoader(args)


def _revision():
    from part155 import run155
    return getattr(run155._common(), "REVISION", None)


# --------------------------------------------------------------------------------------------- Gate R3

def call_shapes(calls):
    """Gate R3 on what reached Part 163's generator (the row counts observed as each chunk returned): every chunk of a
    pushed battery or recipe call (stage "run") has exactly 8 rows, a completed pushed call's chunks cover its rows,
    and the no-push and batch-1 calls (stage "batch1", one pushed row each) are single rows. A call cut short by a killed launch may cover fewer rows, never more. A
    relaunch regenerates an interrupted call, so the record can hold it incomplete beside its complete re-run; both
    are checked, neither is summed."""
    pushed = [call for call in calls if call.get("pushed") and call.get("stage") != "batch1"]
    plain = [call for call in calls if not call.get("pushed") or call.get("stage") == "batch1"]
    observed = [int(size) for call in calls for size in call.get("observed", [])]
    biggest = max(observed, default=0)

    def bad_pushed(call):
        sizes = [int(size) for size in call.get("observed", [])]
        if any(size != CHUNK for size in sizes):          # exactly 8: a 7-row or 9-row chunk fails
            return True
        if call.get("complete"):
            return sum(sizes) != int(call["rows"])
        return sum(sizes) > int(call["rows"])

    wrong = [call for call in pushed if bad_pushed(call)]
    bad_plain = [call for call in plain
                 if (call.get("observed") != [1] if call.get("complete") else call.get("observed") not in ([], [1]))]
    complete_pushed = [call for call in pushed if call.get("complete")]
    passed = bool(complete_pushed and not wrong and not bad_plain)
    return {"passed": passed, "pushed_calls": len(pushed), "complete_pushed_calls": len(complete_pushed),
            "pushed_chunks": sum(len(call.get("observed", [])) for call in pushed),
            "pushed_calls_not_in_chunks_of_8": len(wrong), "no_push_calls": len(plain),
            "bad_no_push_calls": len(bad_plain), "largest_chunk_rows": biggest, "calls": len(calls)}


def record_gates(rec):
    """Gates R1 and R3 over every launch in a load.json record: (ok, reason, shapes)."""
    loads = rec.get("loads", [])
    ok_kernel, why_kernel = run199.check_kernels(loads)
    ok_run, why_run = run199.check_same_run(loads)
    shapes = call_shapes([call for entry in loads for call in entry.get("calls", [])])
    why_shape = "" if shapes["passed"] else f"Gate R3: {json.dumps(shapes)}"
    return bool(ok_kernel and ok_run and shapes["passed"]), (why_kernel or why_run or why_shape), shapes


def check_record(path):
    """Host-side Gates R1 and R3 over every recorded launch: exit 0 only if load.json records at least one load, all
    eager, one model, revision and folder, and every pushed call went in chunks of exactly 8 rows."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    ok, why, shapes = record_gates(rec)
    if ok:
        print(f"PART200 record ok: {len(rec.get('loads', []))} recorded load(s), all eager, one model/revision/folder; "
              f"{shapes['pushed_calls']} pushed calls in {shapes['pushed_chunks']} chunks of exactly 8 rows", flush=True)
        return 0
    print(f"PART200 RECORD FAILED: {why}", flush=True)
    return 3


# --------------------------------------------------------------------------------------------- change 4: batch 1

def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _push_rows(rows, layers, medians, common, cache):
    """Part 163's own pushes_for, imported rather than retyped: the batch-1 pushes are the battery's."""
    from part163 import run163
    return run163.pushes_for(rows, layers, medians, common, cache)


def batch1_rows(units):
    """Question 0's battery row for every unit, in units163.json order (design, Method item 4)."""
    from part163 import run163
    return [{**run163._unit_row(u), "question_index": 0} for u in units]


def run_batch1(units, items, layers, medians, common, model, tok, device, generate_fn, path):
    """Regenerate question 0's 200 battery rows, one unit per call (batch 1), appending to `path`; units already
    written are skipped (resumable on every launch). Returns the rows written this invocation."""
    from part163 import run163
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {row["unit_id"] for row in _read_jsonl(path)}
    pending = [row for row in batch1_rows(units) if row["unit_id"] not in done]
    question = items[0]["question"]
    cache = {}
    written = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for row_info in pending:
            pushes = _push_rows([row_info], layers, medians, common, cache)
            got = generate_fn(model, tok, layers, question, pushes, MAX_NEW_TOKENS, device)
            if len(got) != 1:
                raise RuntimeError(f"batch-1 call for {row_info['unit_id']} returned {len(got)} answers")
            u = row_info["unit"]
            record = {"stage": "batch1", "unit_id": u["unit_id"], "arm": u["arm"], "layer": int(u["layer"]),
                      "neuron_or_index": u.get("neuron") if u["arm"] == "neuron" else u.get("random_index"),
                      "dose": run163.DOSE, "question_index": 0, "question": question, "kind": items[0]["kind"],
                      **got[0]}
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")
            fh.flush()
            written += 1
    return written


def batch1_changes(batch1, answers):
    """The batch-1 answers whose text differs from the same unit's chunked battery answer at question 0, per arm and
    in total; shape-sensitive when more than 20 of 200 differ (a flag only)."""
    battery = {row["unit_id"]: row.get("answer") for row in answers
               if row.get("stage") == "battery" and int(row.get("question_index", -1)) == 0}
    out = {"rows": len(batch1), "battery_rows": len(battery)}
    changed_all = []
    for arm in ("neuron", "random"):
        rows = [row for row in batch1 if row.get("arm") == arm]
        changed = [row["unit_id"] for row in rows if row.get("answer") != battery.get(row["unit_id"])]
        changed_all.extend(changed)
        out[arm] = {"rows": len(rows), "changed": len(changed)}
    out["changed"] = len(changed_all)
    out["changed_unit_ids"] = changed_all[:50]
    out["shape_sensitive"] = bool(len(batch1) == BATCH1_ROWS and out["changed"] > SHAPE_SENSITIVE_LIMIT)
    return out


# --------------------------------------------------------------------------------------------- the run

def run(args, loader=None, generate_fn=None):
    """run163.run on the eager kernel, in chunks of 8, under a folder lock; then the batch-1 stage on the same model."""
    import time
    import neuron_report as nr
    from part163 import run163
    from part160 import run160
    from part155 import run155

    out = Path(args.out)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    with Run200Lock(folder):
        load_path = folder / "load.json"
        before = run199._answers_written(folder)
        calls = []
        state = {"index": None, "entry": None}

        def flush(status=None, **extra):
            if state["entry"] is None:
                return
            entry = state["entry"]
            entry["answers_written"] = run199._answers_written(folder) - before
            entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if status:
                entry["status"] = status
            entry.update(extra)
            rec = run199._read_record(load_path)
            if state["index"] is None:
                rec.setdefault("loads", []).append(entry)
                state["index"] = len(rec["loads"]) - 1
            else:
                rec["loads"][state["index"]] = entry
            run199._save_record(load_path, rec)

        def recording_loader(device):
            tok, model = loader(device)          # refuses a non-eager load before anything is recorded or generated
            state["entry"] = {"load": nr.LAST_LOAD, "model": str(args.model), "revision": _revision(),
                              "out_dir": str(folder.resolve()), "argv": list(sys.argv[1:]), "calls": calls,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running"}
            flush()
            return tok, model

        common = run155._common()
        original_generate = generate_fn or (lambda *a: run160.generate_checked(*a, common=common))
        chunked = chunked_generate(original_generate, calls, on_chunk=flush)
        previous_rate = set_pilot_min_rate(PILOT_MIN_RATE)
        summary, failure, stopped, final_write_failed = None, None, False, False
        batch1_written = 0
        try:
            summary = run163.run(args, loader=recording_loader, generate_fn=chunked)
            batch1_written = _run_batch1_after(args, loader, original_generate, calls, flush)
        except SystemExit as exc:            # Part 163's gates, its pilot, or this wrapper's loader refusal: a stop
            failure, stopped = f"SystemExit: {exc}", True
        except BaseException as exc:         # a killed model, a bug: record the rows it wrote, then let it propagate
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            restore_pilot_min_rate(previous_rate)
            try:
                flush(status="stopped" if failure else "done", summary=summary, failure=failure,
                      batch1_rows_written=batch1_written)
            except Exception as exc:         # a failed final write must not replace the run's own outcome
                final_write_failed = True
                print(f"PART200: the final load.json write failed ({type(exc).__name__}: {exc}); the last per-chunk "
                      "write stands", flush=True)
            written = run199._answers_written(folder) - before
            if failure is not None:
                print(f"PART200 run stopped: {failure}; {written} row(s) written this invocation", flush=True)
        if stopped:
            print("PART200: stopped by a gate, the pilot or the loader (see above); no reading is made from this "
                  "launch", flush=True)
            return 3
        rec = run199._read_record(load_path)
        loads = rec.get("loads", [])
        ok, why, shapes = record_gates(rec)
        gates163 = (summary or {}).get("gates") or {}
        report = {"answers_written": written, "batch1_rows_written": batch1_written,
                  "gates_r1_r3": ok, "why": why, "gate_r3": shapes,
                  "part163_gates_passed": gates163.get("passed"), "final_write_failed": final_write_failed}
        print(json.dumps(run199._finite(report), ensure_ascii=True, allow_nan=False), flush=True)
        if not ok:
            print(f"PART200 GATE R1/R3 FAILED over the record: {why}", flush=True)
            return 3
        if final_write_failed:
            print("PART200: this launch could not record its outcome in load.json; exit 3", flush=True)
            return 3
        batch1 = _read_jsonl(folder / "batch1.jsonl")
        changes = batch1_changes(batch1, _read_jsonl(out))
        print(json.dumps({"batch1": changes}, ensure_ascii=True), flush=True)
        if len(batch1) != BATCH1_ROWS:
            print(f"PART200: batch 1 has {len(batch1)} of {BATCH1_ROWS} rows; finish it before comparing", flush=True)
            return 3
        if changes["shape_sensitive"]:
            print(f"PART200: {changes['changed']} of {BATCH1_ROWS} batch-1 answers differ from the chunked answers: "
                  "the arm counts are shape-sensitive", flush=True)
        print(f"PART200 gates R1 and R3 passed: {len(loads)} load(s) eager; {shapes['pushed_calls']} pushed calls "
              f"in {shapes['pushed_chunks']} chunks of exactly 8 rows; batch 1 changed {changes['changed']} of "
              f"{changes['rows']}", flush=True)
        print(json.dumps(run199._finite(summary), ensure_ascii=True, allow_nan=False), flush=True)
        return 0


def _run_batch1_after(args, loader, generate_fn, calls=None, on_chunk=None):
    """The batch-1 stage, generated after run163.run returns on the SAME model (design change 3: no second load).

    The loader is the EagerLoader the run was given; it remembers the (tok, model) it loaded, so this call returns
    that pair and load_model is not called again. `layers` comes from the same model. The stage runs on every launch
    until batch1.jsonl holds all 200 units (design change 4): a relaunch after a kill finishes it, and a relaunch
    after completion writes nothing. Its generator is the wrapper's chunker around the same generate_fn, logging into
    the launch's `calls` and flushing after every chunk, so its one-row calls are recorded in the same load entry
    and Gate R3 sees them as single rows."""
    from part163 import run163
    from part155 import run155
    out = Path(args.out)
    units = json.loads(Path(args.units).read_text(encoding="utf-8"))
    items = run163.choose_questions()
    common = run155._common()
    medians = json.loads((ROOT / "results/part112/medians.json").read_text(encoding="utf-8"))
    tok, model = loader(args.device)
    layers = common.find_layers(model)
    return run_batch1(units, items, layers, medians, common, model, tok, args.device,
                      chunked_generate(generate_fn, calls, on_chunk=on_chunk, stage="batch1"),
                      out.parent / "batch1.jsonl")


# --------------------------------------------------------------------------------------------- --compare

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _battery(rows):
    return [row for row in rows if row.get("stage") == "battery"]


def gate_r2(rerun_units_path, published_units_path, rerun_answers, published_answers):
    """Gate R2: units163.json byte-identical (sha256) to the published file, and the 16 battery questions and their
    order identical to Part 163's answers file."""
    a, b = sha256(published_units_path), sha256(rerun_units_path)

    def questions(rows):
        seen = {}
        for row in _battery(rows):
            seen.setdefault(int(row.get("question_index", -1)), row.get("question"))
        return [seen[k] for k in sorted(seen)]

    qa, qb = questions(published_answers), questions(rerun_answers)
    units_same = a == b
    questions_same = qa == qb and len(qa) == 16
    return {"passed": bool(units_same and questions_same), "units_sha256_published": a, "units_sha256_rerun": b,
            "units_identical": units_same, "questions_identical": questions_same,
            "published_questions": len(qa), "rerun_questions": len(qb),
            "first_differing_question": next((i for i, (x, y) in enumerate(zip(qa, qb)) if x != y), None)}


def part161_qwen(rows):
    """Part 161's rule per arm on maker phrasings: answers matching case-sensitive \\bQwen\\b, among maker answers that
    are not broken (Part 86 triage) and hold none of alibaba, tongyi, 通义 (case-insensitive)."""
    import triage86
    out = {}
    for arm in ("neuron", "random"):
        maker = [row for row in _battery(rows) if row.get("arm") == arm and row.get("kind") == "maker"]
        eligible = [row for row in maker if triage86.triage_answer(row.get("answer", "")) != "broken"
                    and not PART161_EXCLUDE.search(row.get("answer", ""))]
        hits = sum(bool(PART161_QWEN.search(row.get("answer", ""))) for row in eligible)
        out[arm] = {"qwen": hits, "eligible_maker_answers": len(eligible), "maker_answers": len(maker),
                    "rate": hits / len(eligible) if eligible else None}
    n, r = out["neuron"]["rate"], out["random"]["rate"]
    out["neuron_below_random"] = None if n is None or r is None else bool(n < r)
    return out


def mid_phrase(answer):
    """Part 191's open-ending rule, imported from Part 199 (part199/report199.py) rather than written again."""
    from part199.report199 import mid_phrase as part199_mid_phrase
    return part199_mid_phrase(answer)


def broken_and_cap(rows):
    """Scored, per arm over its 1,600 battery answers: the broken count (Part 86 triage) and the at-the-cap count
    (token_count == 80)."""
    import triage86
    out = {}
    for arm in ("neuron", "random"):
        battery = [row for row in _battery(rows) if row.get("arm") == arm]
        out[arm] = {"battery_answers": len(battery),
                    "broken": sum(triage86.triage_answer(row.get("answer", "")) == "broken" for row in battery),
                    "at_cap": sum(int(row.get("token_count", 0)) == MAX_NEW_TOKENS for row in battery)}
    return out


def mid_phrase_under_cap(rows):
    """Exploratory, per arm: Part 191's mid-phrase count among battery answers under the 80-token cap."""
    from part199.report199 import mid_phrase as part199_mid_phrase
    out = {}
    for arm in ("neuron", "random"):
        battery = [row for row in _battery(rows) if row.get("arm") == arm]
        early = [row for row in battery if int(row.get("token_count", 0)) < MAX_NEW_TOKENS]
        out[arm] = {"under_cap": len(early),
                    "mid_phrase_among_under_cap": sum(part199_mid_phrase(row.get("answer")) for row in early)}
    return out


def per_layer_broken(rows):
    """Per arm and layer, the broken count over battery answers (exploratory)."""
    import triage86
    out = {}
    for arm in ("neuron", "random"):
        counts = {}
        for row in _battery(rows):
            if row.get("arm") != arm:
                continue
            layer = str(row.get("layer"))
            counts.setdefault(layer, {"answers": 0, "broken": 0})
            counts[layer]["answers"] += 1
            counts[layer]["broken"] += triage86.triage_answer(row.get("answer", "")) == "broken"
        out[arm] = dict(sorted(counts.items(), key=lambda kv: int(kv[0])))
    return out


def identical_battery(rerun_answers, published_answers):
    """Battery answers whose text is identical between the runs, per arm (matched on unit and question)."""
    def index(rows):
        return {(row.get("unit_id"), int(row.get("question_index", -1))): row for row in _battery(rows)}
    a, b = index(published_answers), index(rerun_answers)
    shared = set(a) & set(b)
    out = {"shared_rows": len(shared), "identical": sum(a[k].get("answer") == b[k].get("answer") for k in shared)}
    for arm in ("neuron", "random"):
        keys = [k for k in shared if a[k].get("arm") == arm]
        out[arm] = {"rows": len(keys), "identical": sum(a[k].get("answer") == b[k].get("answer") for k in keys)}
    return out


def identical_no_push(rerun_answers, published_answers):
    """The 16 no-push answers whose text is identical between the runs (matched on question index)."""
    def index(rows):
        return {int(row.get("question_index", -1)): row for row in rows if row.get("stage") == "no_push"}
    a, b = index(published_answers), index(rerun_answers)
    shared = set(a) & set(b)
    return {"shared": len(shared), "identical": sum(a[k].get("answer") == b[k].get("answer") for k in shared)}


def score_side(folder, answers, units, published_score=None):
    """Part 163's gates and the five verdicts: from score163.py itself. The published side reads Part 163's own
    score.json (its registered output); the rerun is scored here from its answers file."""
    from part163 import score163
    if published_score is not None:
        score = published_score
    else:
        score = score163.score_rows(answers, units)
    verdicts = {}
    for key, pred in score.get("predictions", {}).items():
        verdicts[key] = {"verdict": pred.get("verdict"), "prediction": pred.get("prediction"),
                         "estimate_ratio": pred.get("estimate_ratio"),
                         "ratio_interval_95": pred.get("ratio_interval_95"),
                         "rate_by_arm": pred.get("rate_by_arm")}
    return {"gates": {name: {"passed": gate.get("passed")} for name, gate in score.get("gates", {}).items()},
            "all_gates_passed": score.get("all_gates_passed"), "verdicts": verdicts}


# Part 163's published folder predates the load record (Part 163 did not write one), so the record is required of the
# rerun only; the published side needs its answers, and its score.json when present (otherwise it is scored here).
RERUN_REQUIRED = ("answers.jsonl", "load.json", "batch1.jsonl")
PUBLISHED_REQUIRED = ("answers.jsonl",)


def compare(rerun_dir, published_dir, units_path=None):
    """PART200-DESIGN.md's --compare paragraph, split into its scored and exploratory sections, published beside
    rerun."""
    rerun, published = Path(rerun_dir), Path(published_dir)
    for folder, names in ((rerun, RERUN_REQUIRED), (published, PUBLISHED_REQUIRED)):
        for name in names:
            if not (folder / name).exists():
                raise FileNotFoundError(str(folder / name))
    units_path = Path(units_path) if units_path else ROOT / "part163/units163.json"
    units = json.loads(units_path.read_text(encoding="utf-8"))
    rerun_answers, published_answers = _read_jsonl(rerun / "answers.jsonl"), _read_jsonl(published / "answers.jsonl")
    record_ok, record_why, _shapes = record_gates(run199._read_record(rerun / "load.json"))
    published_score_path = published / "score.json"
    published_score = json.loads(published_score_path.read_text(encoding="utf-8")) \
        if published_score_path.exists() else None
    batch1 = _read_jsonl(rerun / "batch1.jsonl")
    scored = {
        "record_r1_r3": {"passed": record_ok, "why": record_why},
        "gate_r2": gate_r2(units_path, units_path, rerun_answers, published_answers),
        "published": {"part163_gates_and_verdicts": score_side(published, published_answers, units, published_score),
                      "part161_qwen_on_maker": part161_qwen(published_answers),
                      "broken_and_cap": broken_and_cap(published_answers)},
        "rerun": {"part163_gates_and_verdicts": score_side(rerun, rerun_answers, units),
                  "part161_qwen_on_maker": part161_qwen(rerun_answers),
                  "broken_and_cap": broken_and_cap(rerun_answers)},
        "batch1": batch1_changes(batch1, rerun_answers),
    }
    scored["verdicts_changed"] = sorted(
        key for key, val in scored["rerun"]["part163_gates_and_verdicts"]["verdicts"].items()
        if val["verdict"] != scored["published"]["part163_gates_and_verdicts"]["verdicts"].get(key, {}).get("verdict"))
    exploratory = {
        "mid_phrase_under_cap": {"published": mid_phrase_under_cap(published_answers),
                                 "rerun": mid_phrase_under_cap(rerun_answers)},
        "identical_battery_answers": identical_battery(rerun_answers, published_answers),
        "identical_no_push_answers": identical_no_push(rerun_answers, published_answers),
        "per_layer_broken": {"published": per_layer_broken(published_answers),
                             "rerun": per_layer_broken(rerun_answers)},
    }
    return {"scored": scored, "exploratory": exploratory}


def _print_sections(result):
    print("PART200 COMPARE: scored", flush=True)
    print(json.dumps(run199._finite(result["scored"]), ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    print("PART200 COMPARE: exploratory (not scored; no exploratory value replaces a scored one)", flush=True)
    print(json.dumps(run199._finite(result["exploratory"]), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--check-record":
        return check_record(argv[1])
    if argv and argv[0] == "--compare":
        ap = argparse.ArgumentParser(prog="run200.py --compare")
        ap.add_argument("rerun_dir")
        ap.add_argument("published_dir")
        ap.add_argument("--units", default=str(ROOT / "part163/units163.json"))
        cargs = ap.parse_args(argv[1:])
        try:
            result = compare(cargs.rerun_dir, cargs.published_dir, cargs.units)
        except FileNotFoundError as exc:
            print(f"PART200 COMPARE: missing file {exc}", flush=True)
            return 2
        out_path = Path(cargs.rerun_dir) / "compare.json"
        out_path.write_text(json.dumps(run199._finite(result), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
        _print_sections(result)
        scored = result["scored"]
        if not scored["record_r1_r3"]["passed"]:
            print(f"PART200: the rerun's load.json fails Gate R1 or R3 ({scored['record_r1_r3']['why']}); readings "
                  "stop", flush=True)
            return 3
        if scored["rerun"]["part163_gates_and_verdicts"]["all_gates_passed"] is not True:
            print("PART200: Part 163's gates did not pass on the rerun; readings stop", flush=True)
            return 3
        if not scored["gate_r2"]["passed"]:
            print("PART200 GATE R2 FAILED: the rerun's units or battery questions are not Part 163's; readings stop",
                  flush=True)
            return 3
        if scored["batch1"]["shape_sensitive"]:
            print(f"PART200: {scored['batch1']['changed']} of {BATCH1_ROWS} batch-1 answers differ from the chunked "
                  "answers: the arm counts are shape-sensitive", flush=True)
        if scored["verdicts_changed"]:
            print(f"PART200: verdicts changed: {scored['verdicts_changed']}", flush=True)
        print(f"PART200 gates R1, R2, R3 and Part 163's three passed; compare written to {out_path}", flush=True)
        return 0
    from part163 import run163
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=run163.MODEL_DEFAULT)
    ap.add_argument("--units", default=str(ROOT / "part163/units163.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part200/answers.jsonl"))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        ap.error("registered batch size is 64 (the pushed calls are chunked inside the wrapper)")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
