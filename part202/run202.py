#!/usr/bin/env python3
"""Part 202: Part 164 on the eager attention kernel (PART202-DESIGN.md).

Usage: run202.py --device xpu --model ... --units part164/units164.json --out results/part202/answers.jsonl
       run202.py --check-record results/part202/load.json
       run202.py --replicate-plan results/part202/answers.jsonl --out results/part202/replicate-plan.json

Part 164's run() is not called: Part 164's runner has no seam for a push hook that records margins and hook evidence,
and its 64-row calls would have to be re-chunked inside it. Its plan, order, keys, pushes (`part164/run164.py`'s
`choose_questions` and `pushes_for`), doses, cap and scorer are imported and reused exactly. Everything else the design
names is imported from the Part 201/199 modules rather than copied:

1. Kernel (design, Method; Gate R1). Load through neuron_report.load_model, apply Part 164's no-grad and eval settings,
   and refuse (SystemExit 3) before any generation unless both neuron_report.LAST_LOAD and the model's
   config._attn_implementation record eager. The load and the per-chunk evidence are written by Part 201's record
   functions (tools/margin_recorder.MarginRecorder for the scores; the load record, receipts, row fields, hashes and
   the folder lock from part201/run201.py, whose schema is part201/LOAD-SCHEMA.md and part201/ROWS-SCHEMA.md).
2. Push shape (design, Method; Gate R3). Every pushed call is split into chunks of exactly 8 rows, in the call's
   original order, each layer's push tensor sliced to the chunk's rows. Part 164's last battery call of 36 rows is
   padded with four duplicate companion rows (the call's first four rows) to make five 8-row model calls; the
   companions' outputs are discarded and never written. No-push calls stay single-row calls. The plan is Part 164's
   own row order (run164.run), and Gate R3 reads the shapes handed to the model.
3. Pilot (design, Method). After the first three pushed batches, the measured rate must be at least 0.30 answers/s
   (6,516 rows over the six-hour task limit), else the launch stops with SystemExit (a HOLD until reviewed).
4. Recorder (design, Method and Reading rules). Every row stores its tokens and per-position top-two tokens, logits,
   margin and emitted-token score, computed on the CPU by Part 201's MarginRecorder.
5. Shape replicate (design, Method). After the battery, a seeded sample of 200 rows spread across cells, one row per
   call, decoded again and written to replicate.jsonl with the same margin fields.

A killed launch is resumed in the same output folder: rows are keyed by Part 164's own key (stage, unit_id,
question_index), and a call that is not whole on disk is moved to rows.removed.jsonl and decoded again whole, as Part
201's resume does (part201/run201.py:_load_rows).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part201 import run201  # noqa: E402  (load record, receipts, row fields, hashes, lock: reused, not copied)
from part199 import run199  # noqa: E402  (the folder lock Part 201's record functions are shaped after)

CHUNK = 8                      # design, Method: every pushed call is split into exactly 8 rows
MAX_NEW_TOKENS = 80            # Part 164's cap, kept
DOSE = 0.8                     # Part 164's dose for the battery and the recipe
DOSES = (0.8, 1.0, 1.2)        # Part 164's ladder (random arm only above 0.8)
PLANNED_ROWS = 6516            # design, Size: 16 no-push + 100 recipe + 6,400 battery
REPLICATE_ROWS = 200           # design, Size: the shape replicate
REPLICATE_SEED = 202
PILOT_BATCHES = 3              # design, Method: the pilot measures the first three pushed batches
PILOT_SECONDS = 6 * 3600       # the launcher's task limit
PILOT_MIN_RATE = PLANNED_ROWS / PILOT_SECONDS   # 0.3015 answers/s: the broken-path guard, not a content measure
LOCK_NAME = "run202.lock"
PROCESS = "bf16-eager"         # this part's one process (the design has no float32 arm)
# design, Controls, Gate R2: the SHA-256 values in this tree.
EXPECTED_SHA256 = {
    "part164/units164.json": "4f146140e4dbc6d28b95ed937e6ccf72b9e712a641c721235628b5ec01627921",
    "results/part112/medians.json": "65562b342818b46189a98ae18852e6a972d1b41a594f7ce1c442016aecd0f715",
    "part153/battery153.py": "e2032288651d65cafa2e60634f512470ec346f5ad093e8be7d6510df089d96a7",
    "part155/run155.py": "07b15c41a731e07fffe2b0a26d09a95b2c808a71a1020180096cfa41baac0fa5",
    "part164/run164.py": "19d36e18a25f730dd37b1df46c06404a154f5b1e338a53c9cf04e70b909e7277",
}
# design, Reading rules: rows are labelled by the scorer's own rules, imported from Part 164's scorer.
LABELS = ("qwen", "broken", "name_gone", "maker_gone")


# --------------------------------------------------------------------------------------------- Gate R2

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_r2(units_path=None, medians_path=None):
    """Gate R2's hashes: each file in this tree has the SHA-256 the design records. Returns the per-file check."""
    units_path = Path(units_path or ROOT / "part164/units164.json")
    medians_path = Path(medians_path or ROOT / "results/part112/medians.json")
    paths = {"part164/units164.json": units_path, "results/part112/medians.json": medians_path,
             "part153/battery153.py": ROOT / "part153/battery153.py", "part155/run155.py": ROOT / "part155/run155.py",
             "part164/run164.py": ROOT / "part164/run164.py"}
    files = {}
    for name, path in paths.items():
        got = sha256(path) if path.exists() else None
        files[name] = {"sha256": got, "expected": EXPECTED_SHA256[name], "matches": got == EXPECTED_SHA256[name]}
    return {"passed": all(f["matches"] for f in files.values()), "files": files}


def ordered_prompts():
    """The 16 battery prompts in order, regenerated from Part 164's own selection (run164.choose_questions, which reads
    part155.questions() from part153/battery153.py, both hashed above)."""
    from part164 import run164
    return [q["question"] for q in run164.choose_questions()]


def gate_r2_prompts(answers_path):
    """Gate R2's ordered-prompt check on the rerun's own rows (a file path)."""
    return gate_r2_prompts_from_rows(read_jsonl(answers_path))


def gate_r2_prompts_from_rows(rows):
    """Gate R2's ordered-prompt check on the rerun's own rows: question_index 0..15 of the battery, in order, equal the
    selection regenerated from the hashed sources. The no-push rows carry the same 16 prompts, so a battery row missing
    from a file is still checked through its no-push row."""
    seen = {}
    for row in rows:
        if row.get("stage") in ("battery", "no_push"):
            seen.setdefault(int(row.get("question_index", -1)), row.get("question"))
    got = [seen[k] for k in sorted(seen)]
    want = ordered_prompts()
    return {"passed": got == want and len(got) == 16, "rerun_questions": len(got), "expected_questions": len(want),
            "first_differing_question": next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)}


# --------------------------------------------------------------------------------------------- the plan

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def _spec(stage, unit, question_index, dose, question, kind):
    """One planned row in Part 164's own shape (run164._row_info plus the question), with Part 201's row_id."""
    return {"row_id": f"{unit['unit_id']}|q{question_index}" if stage != "no_push" else f"plain|q{question_index}",
            "unit_id": unit["unit_id"], "arm": unit["arm"], "layer": unit.get("layer"),
            "neuron": unit.get("neuron"), "random_index": unit.get("random_index"),
            "question_index": int(question_index), "question": question, "kind": kind, "dose": float(dose),
            "stage": stage}


def plan(units_path=None):
    """Part 164's run order (run164.run) as calls: the 16 no-push calls (one row each), the recipe (100 random rows in
    blocks of 64), then the battery question by question (neuron 100 + random 100 at 0.8, random 100 at 1.0 and 1.2),
    in blocks of 64. Each call is split into chunks of 8; the last chunk of a call with fewer than 8 rows is padded
    with the call's first rows as companions (marked, discarded)."""
    from part164 import run164
    units = json.loads(Path(units_path or ROOT / "part164/units164.json").read_text(encoding="utf-8"))
    if sum(u["arm"] == "neuron" for u in units) != 100 or sum(u["arm"] == "random" for u in units) != 100:
        raise ValueError("units file must contain exactly 100 pairs")
    items = run164.choose_questions()
    calls = []

    def add(stage, specs, pad=True):
        for start in range(0, len(specs), 64):
            block = specs[start:start + 64]
            chunks = []
            for c0 in range(0, len(block), CHUNK):
                members = block[c0:c0 + CHUNK]
                companions = []
                # Only a pushed call is padded (the design pads Part 164's 36-row push); a no-push call is one row.
                while pad and len(members) < CHUNK:
                    companions.append(dict(members[len(companions)], companion_of=members[len(companions)]["row_id"]))
                    members = members + [companions[-1]]
                chunks.append({"rows": members, "row_ids": [r["row_id"] for r in members],
                               "companions": [c["row_id"] for c in companions]})
            calls.append({"stage": stage, "call": len(calls), "rows": len(block), "chunks": chunks,
                          "row_ids": [r["row_id"] for r in block]})
    for qi, item in enumerate(items):
        add("no_push", [_spec("no_push", {"unit_id": "no_push", "arm": "plain", "layer": None}, qi, 0.0,
                              item["question"], item["kind"])], pad=False)
    add("recipe", [_spec("recipe", u, 0, DOSE, "Who are you?", "recipe") for u in units if u["arm"] == "random"])
    for qi, item in enumerate(items):
        for dose in DOSES:
            selected = [u for u in units if u["arm"] == "neuron"] if dose == DOSE else []
            selected += [u for u in units if u["arm"] == "random"]
            add("battery", [_spec("battery", u, qi, dose, item["question"], item["kind"]) for u in selected])
    return {"schema": 1, "chunk": CHUNK, "calls": calls,
            "rows": sum(c["rows"] for c in calls),
            "companions": sum(len(ch["companions"]) for c in calls for ch in c["chunks"])}


# --------------------------------------------------------------------------------------------- the run

class Run202Lock(run199.FolderLock):
    """Part 199's folder lock under this part's name: results/part202/run202.lock, created exclusively."""

    def __init__(self, folder):
        super().__init__(folder)
        self.path = Path(folder) / LOCK_NAME


def _eos(tok, model):
    return run201._eos(tok, model)


def _prompt_len(tok, prompt):
    return run201._prompt_len(tok, prompt)


def _push_hash(pushmap, slot):
    return run201._push_hash(pushmap, slot)


def _finite(value):
    return run199._finite(value)


def _save_record(path, rec):
    run199._save_record(path, rec)


def _read_record(path):
    rec = run199._read_record(path)
    rec.setdefault("part", 202)
    rec.setdefault("loads", [])
    return rec


def _answers_written(folder):
    path = Path(folder) / "answers.jsonl"
    return path.read_bytes().count(b"\n") if path.exists() else 0


class EagerLoader:
    """The loader handed to run164.run: neuron_report.load_model, then Part 164's no-grad and eval settings, refused
    (SystemExit 3) before any generation unless both the recorded load and the model's config say eager. It loads ONCE
    and keeps the (tok, model) it loaded, so the replicate stage uses the same model and load_model is called once."""

    def __init__(self, args):
        self.args = args
        self.loaded = None
        self.last_load = None

    def __call__(self, device):
        if self.loaded is not None:
            return self.loaded
        import neuron_report as nr
        from part155 import run155
        tok, model = nr.load_model(str(self.args.model), run155._common().REVISION, device)
        self.last_load = dict(nr.LAST_LOAD or {})
        kernel = self.last_load.get("attn_implementation")
        config_kernel = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if kernel != "eager" or config_kernel != "eager":
            print(f"PART202 GATE R1 FAILED before generation: the model loaded with {kernel!r} "
                  f"(config {config_kernel!r}), not eager; nothing generated", flush=True)
            raise SystemExit(3)
        model.requires_grad_(False)
        model.eval()
        self.loaded = (tok, model)
        return self.loaded


def _existing_whole(folder, planned, record):
    """The rows on disk that belong to a whole call. A call that is not whole is moved to rows.removed.jsonl and
    decoded again, as Part 201's resume does (part201/run201.py:_load_rows)."""
    path = Path(folder) / "answers.jsonl"
    run201._repair(path)
    rows = read_jsonl(path)
    if not rows:
        return []
    expected = {(c["stage"], c["call"]): set(c["row_ids"]) for c in planned["calls"]}
    present = {}
    for row in rows:
        present.setdefault((row.get("stage"), row.get("call")), []).append(row.get("row_id"))
    incomplete = {key for key, ids in present.items()
                  if key not in expected or expected[key] != set(ids) or len(ids) != len(expected[key])}
    # A call is decoded whole or not at all: a resumed launch redraws every row of a call it finds unwritten, so a call
    # half on disk (a kill between two writes) has every row of it set aside, not only the missing ones.
    if incomplete:
        removed = [dict(r, removed=time.strftime("%Y-%m-%d %H:%M:%S")) for r in rows
                   if (r.get("stage"), r.get("call")) in incomplete]
        run201._append(Path(folder) / "rows.removed.jsonl", removed)
        print(f"PART202 resume: {len(removed)} rows of {len(incomplete)} calls without a whole call or its receipt "
              "will be decoded again", flush=True)
        rows = [r for r in rows if (r.get("stage"), r.get("call")) not in incomplete]
        tmp = path.with_name(path.name + ".repair.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=True) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(path)
    return rows


def _decode_chunk(model, tok, layers, prompts, pushmap, device):
    """One model call: Part 201's generate_rows with Part 201's recorder, which stores tokens, top two, logits, margins
    and the emitted token's score, all on the CPU."""
    from tools.margin_recorder import MarginRecorder
    rec = MarginRecorder(_eos(tok, model))
    start = time.perf_counter()
    seq, rows = run201.generate_rows(model, tok, layers, prompts, pushmap, MAX_NEW_TOKENS, device, rec)
    return {"rows": rows, "seconds": time.perf_counter() - start,
            "hook_firings": dict(getattr(rec, "hook_firings", {})),
            "forward_passes": int(getattr(rec, "forward_passes", 0)),
            "generation_steps": max((len(r["tokens"]) for r in rows), default=0)}


def _chunk_generate(model, tok, layers, specs, stages, medians, common, device, direction_cache):
    """One planned call as the model sees it: its chunks of 8 rows, each with its own pushes (Part 164's
    run164.pushes_for on the chunk's rows, so a companion row carries the same push as the row it duplicates). Returns
    the per-chunk decodes and the push hashes of the scored (non-companion) rows."""
    from part164 import run164
    out = []
    for chunk in stages["chunks"]:
        prompts = [common.build_prompt(tok, r["question"]) for r in chunk["rows"]]
        pushed = any(r["unit_id"] not in (None, "no_push") for r in chunk["rows"])
        pushmap = run164.pushes_for(chunk["rows"], layers, medians, common, direction_cache) if pushed else {}
        decode = _decode_chunk(model, tok, layers, prompts, pushmap, device)
        hashes = {slot: (_push_hash(pushmap, slot) if r["unit_id"] not in (None, "no_push") else None)
                  for slot, r in enumerate(chunk["rows"]) if "companion_of" not in r}
        out.append({"chunk": chunk, "decode": decode, "push_sha256": hashes})
    return out


def _row_from_decode(part, call, stage, launch_index, tok, lengths):
    """Part 201's row fields for each scored row of one chunk, with the companion rows left out."""
    rows = []
    chunk, decode = part["chunk"], part["decode"]
    for slot, spec in enumerate(chunk["rows"]):
        if "companion_of" in spec:
            continue
        rr = dict(decode["rows"][slot])
        answer = tok.decode(rr["tokens"], skip_special_tokens=True).strip()
        # Part 164's own answer row fields stay on the row beside Part 201's (stage, unit_id, arm, dose, question...).
        # row_fields writes Part 201's fields (pass, process, row_id, unit_id, arm, ...); the stage takes Part 164's
        # name (no_push, recipe, battery) so the scorer's keys and Part 164's resume keys read the same row.
        rows.append(run201.row_fields(rr, stage, PROCESS, spec, call, slot, lengths[slot], max(lengths) - lengths[slot],
                                      answer, part["push_sha256"].get(slot), decode["seconds"], launch_index))
        rows[-1]["stage"] = stage
    return rows


def run(args, loader=None):
    """run164.run's plan in chunks of 8 on the eager kernel, under a folder lock, with the load record written after
    load and after every chunk; then the shape replicate on the same model."""
    import neuron_report as nr
    from part155 import run155
    from part164 import run164

    out = Path(args.out)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    units_path = Path(args.units)
    medians_path = ROOT / "results/part112/medians.json"
    r2_files = gate_r2(units_path, medians_path)
    if not r2_files["passed"]:
        bad = [k for k, v in r2_files["files"].items() if not v["matches"]]
        print(f"PART202 GATE R2 FAILED before model load: {bad} do not have the design's SHA-256", flush=True)
        return 3
    if loader is None:
        loader = EagerLoader(args)
    with Run202Lock(folder):
        load_path = folder / "load.json"
        record = _read_record(load_path)
        # The plan is built after the load is refused or accepted, so a refused launch (Gate R1) is recorded before
        # anything else can fail.
        # The plan is built after the load, so a refused launch (Gate R1) is recorded before anything else can fail.
        planned = None
        existing = []
        before = _answers_written(folder)
        calls_log = []
        state = {"index": None, "entry": None}

        def flush(status=None, **extra):
            if state["entry"] is None:
                return
            entry = state["entry"]
            entry["answers_written"] = _answers_written(folder) - before
            entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if status:
                entry["status"] = status
            entry.update(extra)
            if state["index"] is None:
                record["loads"].append(entry)
                state["index"] = len(record["loads"]) - 1
            else:
                record["loads"][state["index"]] = entry
            _save_record(load_path, record)

        def recording_loader(device):
            try:
                tok, model = loader(device)
            except SystemExit:
                state["entry"] = {"launch": len(record["loads"]), "process": PROCESS, "load": getattr(loader, "last_load", None),
                                  "model": str(args.model), "revision": _revision(), "out_dir": str(folder.resolve()),
                                  "argv": list(sys.argv[1:]), "calls": calls_log,
                                  "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "refused",
                                  "failure": "the load was refused before generation (Gate R1)"}
                flush()
                raise
            state["entry"] = {"launch": len(record["loads"]), "process": PROCESS, "load": getattr(loader, "last_load", None),
                              "model": str(args.model), "revision": _revision(), "out_dir": str(folder.resolve()),
                              "argv": list(sys.argv[1:]), "calls": calls_log,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running",
                              "attn_implementation": "eager", "dtype": "torch.bfloat16"}
            flush()
            return tok, model

        common = run155._common()
        medians = json.loads(medians_path.read_text(encoding="utf-8"))
        try:
            tok, model = recording_loader(args.device)
        except SystemExit:
            ok, why, shapes = record_gates(record)
            print(f"PART202 load refused (Gate R1): {why or 'the load was not eager'}; nothing generated", flush=True)
            return 3
        planned = plan(units_path)
        # The registered row count is checked on the registered units file; a fixture's own units (a test's tiny model)
        # carry their own count.
        registered_units = Path(args.units).resolve() == (ROOT / "part164/units164.json").resolve()
        if registered_units and planned["rows"] != PLANNED_ROWS:
            print(f"PART202 plan has {planned['rows']} rows, not the registered {PLANNED_ROWS}; refusing", flush=True)
            return 3
        existing = _existing_whole(folder, planned, record)
        done = {(r.get("stage"), r.get("unit_id"), int(r.get("question_index", -1))) for r in existing}
        layers = common.find_layers(model)
        launch_index = state["entry"]["launch"]
        direction_cache = {}
        written = 0
        pilot = {"rows": 0, "seconds": 0.0, "batches": 0}
        stopped = None
        answers_file = folder / "answers.jsonl"
        try:
            with answers_file.open("a", encoding="utf-8", newline="\n") as fh:
                for call in planned["calls"]:
                    pending = [r for ch in call["chunks"] for r in ch["rows"] if "companion_of" not in r
                               and (r["stage"], r["unit_id"], int(r["question_index"])) not in done]
                    if not pending:
                        continue
                    # A call is decoded whole and written whole (its companions included in the model call): a resumed
                    # launch redraws every row of a call it finds unwritten, so no call is ever half on disk.
                    decodes = _chunk_generate(model, tok, layers, None, call, medians, common, args.device, direction_cache)
                    output = []
                    for part in decodes:
                        lengths = [_prompt_len(tok, common.build_prompt(tok, r["question"])) for r in part["chunk"]["rows"]]
                        output.extend(_row_from_decode(part, call, call["stage"], launch_index, tok, lengths))
                    receipt = {"pass": call["stage"], "call": call["call"], "row_ids": list(call["row_ids"]),
                               "row_count": call["rows"], "pushed": call["stage"] != "no_push",
                               "seconds": sum(p["decode"]["seconds"] for p in decodes),
                               "generation_steps": max(p["decode"]["generation_steps"] for p in decodes),
                               "hook_firings": decodes[0]["decode"]["hook_firings"],
                               "forward_passes": sum(p["decode"]["forward_passes"] for p in decodes),
                               "attempts": 1,
                               "chunks": [{"row_ids": p["chunk"]["row_ids"], "companions": p["chunk"]["companions"],
                                           "rows": len(p["chunk"]["rows"]), "seconds": p["decode"]["seconds"],
                                           "generation_steps": p["decode"]["generation_steps"],
                                           "hook_firings": p["decode"]["hook_firings"],
                                           "forward_passes": p["decode"]["forward_passes"]} for p in decodes]}
                    calls_log.append(receipt)
                    flush()
                    # A resumed launch redraws a whole call even when part of it is already on disk (the design's
                    # resume): the rows it rewrites replace the ones kept, so no key is ever written twice.
                    on_disk = {(r.get("stage"), r.get("unit_id"), int(r.get("question_index", -1)))
                               for r in read_jsonl(answers_file)}
                    output = [r for r in output
                              if (r["stage"], r["unit_id"], int(r["question_index"])) not in on_disk]
                    run201._append(answers_file, output)
                    done.update((r["stage"], r["unit_id"], int(r["question_index"])) for r in output)
                    written += len(output)
                    if call["stage"] != "no_push":
                        pilot["rows"] += len(output)
                        pilot["seconds"] += receipt["seconds"]
                        pilot["batches"] += 1
                        if pilot["batches"] == PILOT_BATCHES:
                            rate = pilot["rows"] / max(pilot["seconds"], 1e-9)
                            state["entry"]["pilot_answers_per_second"] = rate
                            flush()
                            if rate < PILOT_MIN_RATE:
                                stopped = f"pilot: {rate:.3f} answers/s is below {PILOT_MIN_RATE:.4f}"
                                break
            if stopped is None:
                replicate_written = run_replicate_stage(model, tok, layers, medians, common, args.device, folder)
            else:
                replicate_written = 0
        except BaseException as exc:
            # A killed model or a bug: the launch's rows stand, the entry says "stopped" with the error, and the error
            # goes on. The final write in `finally` keeps that status (it only writes "done" when nothing stopped).
            if state["entry"] is not None:
                state["entry"]["failure"] = f"{type(exc).__name__}: {exc}"
                flush(status="stopped")
            raise
        finally:
            # The final write records the outcome: "done", or "stopped" (the pilot's reason, or the error already
            # recorded above).
            if state["entry"] is not None:
                entry = state["entry"]
                if entry.get("status") != "stopped":
                    entry["status"] = "done" if stopped is None else "stopped"
                if stopped is not None:
                    entry["failure"] = stopped
                flush(summary={"answers_written": written})
        if stopped:
            print(f"PART202 run stopped: {stopped}; no reading is made from this launch", flush=True)
            return 3
        ok, why, shapes = record_gates(record)
        print(json.dumps(_finite({"answers_written": written, "replicate_rows": replicate_written,
                                  "gates_r1_r3": ok, "why": why, "gate_r3": shapes}), ensure_ascii=True), flush=True)
        return 0 if ok else 3


def _revision():
    from part155 import run155
    return getattr(run155._common(), "REVISION", None)


# --------------------------------------------------------------------------------------------- Gate R1 and R3

def record_gates(rec):
    """Gates R1 and R3 over every launch in a load.json record: (ok, reason, shapes). R1: at least one load, no launch
    refused, every load eager, one model/revision/folder (Part 199's checks). R3: every pushed call's chunks are
    exactly 8 rows, cover the call's rows in order, and every no-push call is one row."""
    loads = rec.get("loads", [])
    refused = [entry for entry in loads if entry.get("status") == "refused"]
    ok_kernel, why_kernel = run199.check_kernels(loads)
    ok_run, why_run = run199.check_same_run(loads)
    shapes = call_shapes([call for entry in loads for call in entry.get("calls", [])])
    why_shape = "" if shapes["passed"] else f"Gate R3: {json.dumps(shapes)}"
    if refused:
        return False, f"Gate R1: {len(refused)} launch(es) were refused at the load (nothing generated)", shapes
    return bool(ok_kernel and ok_run and shapes["passed"]), (why_kernel or why_run or why_shape), shapes


def call_shapes(calls):
    """Gate R3 on what was passed to the model. A pushed call's chunks each have exactly 8 rows, in the call's order,
    and cover its rows; its last chunk's companions are marked duplicates of its own rows and are not scored. A no-push
    call is one chunk of one row."""
    pushed = [c for c in calls if c.get("pushed")]
    plain = [c for c in calls if not c.get("pushed")]

    def bad_chunk(call):
        chunks = call.get("chunks") or []
        if any(int(ch.get("rows", 0)) != CHUNK for ch in chunks):
            return True
        # The design's padding: only the last chunk carries companions, duplicates of rows already in the call. The
        # scored rows, in order, are exactly the call's rows.
        if any(ch.get("companions") for ch in chunks[:-1]):
            return True
        scored = []
        for ch in chunks:
            skipped = list(ch.get("companions") or [])
            for rid in ch["row_ids"]:
                if rid in skipped:
                    skipped.remove(rid)       # the first occurrence is the scored row, a repeat is its companion
                else:
                    scored.append(rid)
        companions = [rid for ch in chunks for rid in (ch.get("companions") or [])]
        if any(rid not in set(call.get("row_ids") or []) for rid in companions):
            return True
        if scored != list(call.get("row_ids") or []):
            return True
        return False

    def bad_plain(call):
        # A no-push call is one row: one chunk of one row and no companion (the design's no-push shape).
        chunks = call.get("chunks") or []
        return len(chunks) != 1 or int(chunks[0].get("rows", 0)) != 1 or bool(chunks[0].get("companions"))

    wrong = [c for c in pushed if bad_chunk(c)]
    bad_plain = [c for c in plain if bad_plain(c)]
    return {"passed": bool(pushed or plain) and not wrong and not bad_plain, "pushed_calls": len(pushed),
            "no_push_calls": len(plain), "pushed_calls_not_in_chunks_of_8": len(wrong),
            "bad_no_push_calls": len(bad_plain),
            "pushed_chunks": sum(len(c.get("chunks") or []) for c in pushed),
            "companion_rows": sum(len(ch.get("companions") or []) for c in calls for ch in (c.get("chunks") or [])),
            "largest_chunk_rows": max((int(ch.get("rows", 0)) for c in calls for ch in (c.get("chunks") or [])), default=0)}


def check_record(path):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    ok, why, shapes = record_gates(rec)
    if ok:
        print(f"PART202 record ok: {len(rec.get('loads', []))} recorded load(s), all eager, one model/revision/folder; "
              f"{shapes['pushed_calls']} pushed calls in {shapes['pushed_chunks']} chunks of exactly 8 rows, "
              f"{shapes['companion_rows']} companion rows discarded", flush=True)
        return 0
    print(f"PART202 RECORD FAILED: {why}", flush=True)
    return 3


# --------------------------------------------------------------------------------------------- the shape replicate

def replicate_plan(rows, n=REPLICATE_ROWS, seed=REPLICATE_SEED):
    """A seeded sample of `n` battery rows spread across cells in proportion (largest-remainder allocation over
    (arm, dose) cells, then a seeded draw within each cell). Each sampled row is decoded alone, one row per call."""
    battery = [r for r in rows if r.get("stage") == "battery"]
    cells = {}
    for row in battery:
        cells.setdefault((row.get("arm"), float(row.get("dose", 0.0))), []).append(row)
    total = len(battery)
    if total == 0:
        return []
    exact = {cell: n * len(members) / total for cell, members in cells.items()}
    alloc = {cell: int(math.floor(v)) for cell, v in exact.items()}
    remainder = n - sum(alloc.values())
    for cell in sorted(exact, key=lambda c: (-(exact[c] - alloc[c]), str(c)))[:remainder]:
        alloc[cell] += 1
    rng = random.Random(seed)
    picked = []
    for cell in sorted(cells, key=str):
        members = sorted(cells[cell], key=lambda r: (str(r.get("unit_id")), int(r.get("question_index", -1))))
        picked.extend(rng.sample(members, min(alloc[cell], len(members))))
    return picked


def _replicate_spec(row):
    return dict(row, row_id=f"rep|{row.get('unit_id')}|q{row.get('question_index')}", stage="replicate")


def run_replicate_stage(model, tok, layers, medians, common, device, folder):
    """Decode the replicate sample one row per call, appending to replicate.jsonl; rows already written are skipped."""
    from part164 import run164
    path = Path(folder) / "replicate.jsonl"
    done = {(r.get("stage"), r.get("unit_id"), int(r.get("question_index", -1))) for r in read_jsonl(path)}
    sample = [_replicate_spec(r) for r in replicate_plan(read_jsonl(Path(folder) / "answers.jsonl"))]
    pending = [r for r in sample if (r["stage"], r["unit_id"], int(r["question_index"])) not in done]
    cache = {}
    written = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for spec in pending:
            prompt = common.build_prompt(tok, spec["question"])
            pushes = run164.pushes_for([spec], layers, medians, common, cache)
            decode = _decode_chunk(model, tok, layers, [prompt], pushes, device)
            rr = dict(decode["rows"][0])
            length = _prompt_len(tok, prompt)
            # launch -1: a replicate row belongs to no launch's calls list (it is not a planned call); its call number
            # is its position in this file, so the two rows of one sample are told apart.
            out = run201.row_fields(rr, "replicate", PROCESS, spec, {"call": written, "row_ids": [spec["row_id"]]}, 0,
                                    length, 0, tok.decode(rr["tokens"], skip_special_tokens=True).strip(),
                                    _push_hash(pushes, 0), decode["seconds"], -1)
            out["call_rows"] = [spec["row_id"]]
            out["stage"] = "replicate"      # Part 164's stage name, as on every other row (the scorer's key)
            fh.write(json.dumps(out, ensure_ascii=False, allow_nan=True) + "\n")
            fh.flush()
            written += 1
    return written


# --------------------------------------------------------------------------------------------- main

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--check-record":
        return check_record(argv[1])
    if argv and argv[0] == "--replicate-plan":
        ap = argparse.ArgumentParser(prog="run202.py --replicate-plan")
        ap.add_argument("rows")
        ap.add_argument("--out", required=True)
        ap.add_argument("--n", type=int, default=REPLICATE_ROWS)
        ap.add_argument("--seed", type=int, default=REPLICATE_SEED)
        a = ap.parse_args(argv[1:])
        picked = replicate_plan(read_jsonl(a.rows), a.n, a.seed)
        Path(a.out).write_text(json.dumps([{k: r.get(k) for k in ("stage", "unit_id", "arm", "dose", "question_index")}
                                           for r in picked], indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"rows": len(picked), "out": a.out}), flush=True)
        return 0
    from part164 import run164
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=run164.MODEL_DEFAULT)
    ap.add_argument("--units", default=str(ROOT / "part164/units164.json"))
    ap.add_argument("--out", default=str(ROOT / "results/part202/answers.jsonl"))
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
