#!/usr/bin/env python3
"""Part 210: Part 165 on the eager attention kernel (PART210-DESIGN.md).

Usage: run210.py --device xpu --model ... --out results/part210/answers.jsonl
       run210.py --check-record results/part210/load.json
       run210.py --replicate-plan results/part210/answers.jsonl --out results/part210/replicate-plan.json

Part 165's run() is not called: its loop owns each 64-row call, cannot split it into chunks of 8, writes norms.json on
every launch and records no call shapes (design, Method). Its plan, order, keys, pushes, directions, dose, cap and
questions are imported and reused exactly (run165.questions, all_rows, ARMS, LAYERS, SLOTS, K_ARMS, build_directions,
pushes_for, row_key). Everything else is imported from Part 201/199 rather than copied:

1. Kernel (design, Method; Gate R1). Load through neuron_report.load_model, apply Part 165's no-grad and eval settings,
   and refuse (SystemExit 3) before any generation unless both neuron_report.LAST_LOAD and the model's
   config._attn_implementation record eager and the load is bf16. The load record and the per-call receipts are written
   by Part 199's record writer under Part 199's folder lock; the row fields are Part 201's row_fields.
2. Push shape (design, Method; Gate R3). Every pushed call is 64 rows, split into 8 chunks of exactly 8 in the block's
   order, each layer's push tensor sliced to the chunk's rows. Every registered block is 64 rows, so no chunk is short;
   a short block (only a fixture's) pads its last chunk with marked duplicates of its own rows, discarded and never
   written. No-push calls are one row each.
3. Pilot (design, Method). After the first three pushed calls the measured rate must be at least 0.32 answers/s
   (6,920 rows over the six-hour task limit), else the launch stops with status "stopped" and exits 3: a HOLD.
4. Margins (design, Method). Every row stores tools/margin_recorder.py's fields through Part 201's row_fields.
5. Shape replicate (design, Method). After the battery, replicate_plan draws 200 battery rows by arm and by broken
   label, decoded one row per call into replicate.jsonl.

Resume (design, Method): rows are keyed on Part 165's own run165.row_key; a call that is not whole on disk is moved to
rows.removed.jsonl and decoded again whole.
"""
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
for entry in (str(ROOT), str(HERE), str(ROOT / "part86")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from part201 import run201  # noqa: E402  (row fields, repair, append, hashes: reused, not copied)
from part199 import run199  # noqa: E402  (the folder lock and the record writer under it)
from part165 import run165  # noqa: E402  (Part 165's plan, pushes, directions and key: reused, not copied)

DOSE = 0.8                       # Part 165's dose (run165.build_directions: 0.8 x the layer's median)
CHUNK = 8                        # design, Method: every pushed call is 8 chunks of exactly 8 rows
BLOCK = 64                       # Part 165's call size (run165.run's args.batch_size)
MAX_NEW_TOKENS = run165.MAX_NEW_TOKENS   # 80, Part 165's cap
# Part 165's plan constants, read at call time (not bound at import): a test's tiny model has one layer, and the run
# takes the layers and slots it is given.
LAYERS = run165.LAYERS
SLOTS = run165.SLOTS
ARMS = run165.ARMS
QUESTIONS = 8
PLANNED_ROWS = QUESTIONS + QUESTIONS * len(run165.LAYERS) * run165.SLOTS * len(run165.ARMS)   # 8 + 8 x 8 x 12 x 9
REPLICATE_ROWS = 200             # design, Method: the shape replicate
REPLICATE_SEED = 210
PILOT_CALLS = 3                  # design, Method: the pilot measures the first three pushed calls
PILOT_SECONDS = 6 * 3600         # the launcher's task limit
PILOT_MIN_RATE = PLANNED_ROWS / PILOT_SECONDS   # 0.3204 answers/s: the broken-path guard, not a content measure
LOCK_NAME = "run210.lock"
PROCESS = "bf16-eager"           # this part's one process (the design has no float32 arm)
# design, Controls, Gate R2: the SHA-256 values in this tree, copied exactly.
EXPECTED_SHA256 = {
    "part164/units164.json": "4f146140e4dbc6d28b95ed937e6ccf72b9e712a641c721235628b5ec01627921",
    "results/part112/medians.json": "65562b342818b46189a98ae18852e6a972d1b41a594f7ce1c442016aecd0f715",
    "part153/battery153.py": "e2032288651d65cafa2e60634f512470ec346f5ad093e8be7d6510df089d96a7",
    "part155/run155.py": "07b15c41a731e07fffe2b0a26d09a95b2c808a71a1020180096cfa41baac0fa5",
    "part164/run164.py": "19d36e18a25f730dd37b1df46c06404a154f5b1e338a53c9cf04e70b909e7277",
    "part165/run165.py": "a967a8f21f1ff8a898ea28c4b874ef28060893fd709e391a610e0046bf061d48",
    "part165/score165.py": "8c93f5921e9cdb69bc032f31ded81c77710c03bcdeec489fd29a1d9ee2fbe132",
    "part86/triage86.py": "5e43bc9ff52148435f2ab733f8332653cef3b3826d788c061be50049743cc3ee",
    "part86/common86.py": "434879455a54f411de022f40006562d36d8f87244bee86fddf00dc8bc80bed16",
    "part160/run160.py": "ba5cb92bc110f9106daa5c56cc1df75a4ca2b93ade714a6d8536f7cb289066d5",
    "part148/battery148.py": "52edb691a716a84ee88967404eb4901d655e125b02b4841b0e1f0af85ddc4434",
    "neuron_report.py": "b56bfb07e17db090b48d00ce9d2876edbb3d57e709ac0a37e9d44fff080f2322",
    "results/part165/norms.json": "746f8762030fef78dc3582eb8fee4ac1a3331dad7f24597ed489c7dee1067858",
    "results/part165/answers.jsonl": "07d9585ccb153c510d01d423b2140a795de9b5c5f53e6afc638a2cdbaba06762",
}
# design, Controls: the four shape-floor labels. `changed` is left out (it is true on 6,911 of 6,912 battery rows).
FLOOR_LABELS = ("broken", "triage", "loop", "short")
# design, Controls, the floor's scorer: the three ratios' and two Spearmans' bootstraps, replaced by their point
# estimates through a local wrapper that passes reps=0.
BOOTSTRAP_REPS = 0


# --------------------------------------------------------------------------------------------- Gate R2

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_r2(root=None):
    """Gate R2's hashes: each file in this tree has the SHA-256 the design records. Returns the per-file check."""
    root = Path(root or ROOT)
    files = {}
    for name, expected in EXPECTED_SHA256.items():
        path = root / name
        got = sha256(path) if path.exists() else None
        files[name] = {"sha256": got, "expected": expected, "matches": got == expected}
    return {"passed": all(f["matches"] for f in files.values()), "files": files}


def ordered_prompts():
    """The 8 battery prompts in order, regenerated from Part 165's own selection (run165.questions, which reads
    part164.choose_questions and part153/battery153.py, both hashed above)."""
    return [q["question"] for q in run165.questions()]


def gate_r2_prompts(answers_path):
    return gate_r2_prompts_from_rows(read_jsonl(answers_path))


def gate_r2_prompts_from_rows(rows):
    """Gate R2's ordered-prompt check on the rerun's own rows: question_index 0..7 of the battery, in order, equal the
    selection regenerated from the hashed sources. The no-push rows carry the same 8 prompts."""
    seen = {}
    for row in rows:
        if row.get("stage") in ("battery", "no_push"):
            seen.setdefault(int(row.get("question_index", -1)), row.get("question"))
    got = [seen[k] for k in sorted(seen)]
    want = ordered_prompts()
    return {"passed": got == want and len(got) == len(want), "rerun_questions": len(got),
            "expected_questions": len(want),
            "first_differing_question": next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)}


def gate_r2_norms(rerun_norms, published_path=None):
    """Gate R2's direction check: the rerun's write norms equal results/part165/norms.json (the same 864 rows, each
    write_norm within 1e-6 relative of the published one). This is the check Gate 3 makes, read against the published
    file (design, Controls)."""
    published = json.loads(Path(published_path or ROOT / "results/part165/norms.json").read_text(encoding="utf-8"))
    want = {(int(e["layer"]), int(e["slot"]), e["arm"]): float(e["write_norm"]) for e in published.get("directions", [])}
    got = {}
    if isinstance(rerun_norms, dict):
        for entry in rerun_norms.get("directions", []):
            try:
                got[(int(entry["layer"]), int(entry["slot"]), entry["arm"])] = float(entry["write_norm"])
            except (KeyError, TypeError, ValueError):
                got[(entry.get("layer"), entry.get("slot"), entry.get("arm"))] = None
    missing = sorted(want.keys() - got.keys())
    extra = sorted(got.keys() - want.keys())
    off = sorted(k for k in want if k in got and (got[k] is None or
                                                  abs(got[k] - want[k]) > 1e-6 * abs(want[k])))
    return {"passed": not missing and not extra and not off, "published_directions": len(want),
            "rerun_directions": len(got), "missing": len(missing), "extra": len(extra),
            "outside_1e-6_relative": len(off), "first_differing": (off or missing or extra or [None])[0]}


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


def _spec(stage, arm, slot, layer, k, question_index, question, kind):
    """One planned row in Part 165's own shape (run165.all_rows' fields plus the question), with a row_id that names it
    inside its call (part201/ROWS-SCHEMA.md's `<...>|q<question_index>` form)."""
    row_id = (f"plain|q{question_index}" if stage == "no_push"
              else f"L{layer}|s{slot}|{arm}|q{question_index}")
    # `unit_id` and `dose` are Part 201's row fields (part201/ROWS-SCHEMA.md); Part 165's plan has no units, so the
    # unit_id is the row's own name and the dose is Part 165's 0.8 (run165.build_directions' target).
    return {"row_id": row_id, "unit_id": row_id, "stage": stage, "arm": arm, "slot": slot, "layer": layer, "k": k,
            "dose": DOSE, "question_index": int(question_index), "question": question, "kind": kind}


def full_k(widths):
    """The `full` arm's k: the layer's down_proj width (run165.all_rows reads int(widths[layer]))."""
    return {layer: int(widths[layer]) for layer in widths}


def plan(widths, layers=None, slots=None, rows=None):
    """Part 165's run order (run165.run) as calls: the 8 no-push calls (one row each), then question by question the
    battery rows shuffled with random.Random(165 + qi) (run165.run's own shuffle), in blocks of 64. Each block is one
    call split into chunks of exactly 8 in the block's order; a short block's last chunk is padded by repeating that
    chunk's own rows, in order, as marked companions (discarded). `layers` and `slots` default to Part 165's own."""
    layers = tuple(run165.LAYERS if layers is None else layers)
    slots = _slots() if slots is None else slots
    base = list(rows) if rows is not None else all_rows(full_k(widths), layers, slots)
    items = run165.questions()
    calls = []

    def add(stage, specs, pad=True):
        for start in range(0, len(specs), BLOCK):
            block = specs[start:start + BLOCK]
            chunks = []
            for c0 in range(0, len(block), CHUNK):
                members = block[c0:c0 + CHUNK]
                companions = []
                while pad and len(members) < CHUNK:
                    companions.append(dict(members[len(companions)], companion_of=members[len(companions)]["row_id"]))
                    members = members + [companions[-1]]
                chunks.append({"rows": members, "row_ids": [r["row_id"] for r in members],
                               "companions": [c["row_id"] for c in companions]})
            calls.append({"stage": stage, "call": len(calls), "question_index": block[0]["question_index"],
                          "rows": len(block), "chunks": chunks, "row_ids": [r["row_id"] for r in block]})
    for qi, item in enumerate(items):
        add("no_push", [_spec("no_push", "plain", None, None, None, qi, item["question"], item["kind"])], pad=False)
    for qi, item in enumerate(items):
        shuffled = list(base)
        random.Random(165 + qi).shuffle(shuffled)
        add("battery", [_spec("battery", r["arm"], r["slot"], r["layer"], r["k"], qi, item["question"], item["kind"])
                        for r in shuffled])
    return {"schema": 1, "chunk": CHUNK, "block": BLOCK, "calls": calls,
            "rows": sum(c["rows"] for c in calls),
            "companions": sum(len(ch["companions"]) for c in calls for ch in c["chunks"])}


def all_rows(widths, layers, slots):
    """run165.all_rows over the given layers and slots: Part 165's own function on the registered 8 x 12."""
    slots = list(slots) if isinstance(slots, (tuple, list, range)) else list(range(int(slots)))
    if tuple(layers) == tuple(run165.LAYERS) and slots == _slots():
        return run165.all_rows(widths)
    rows = []
    for layer_no in layers:
        for slot in slots:
            for arm in run165.ARMS:
                k = 1 if arm in ("n+", "n±") else run165.K_ARMS.get(arm)
                if arm == "full":
                    k = int(widths[layer_no])
                rows.append({"stage": "battery", "arm": arm, "slot": slot, "layer": layer_no, "k": k})
    return rows


def planned_rows(planned):
    """The scored rows of a plan, in order (companions left out)."""
    return [r for c in planned["calls"] for ch in c["chunks"] for r in ch["rows"] if "companion_of" not in r]


def registered_plan():
    """The registered plan, built on the host without the model: the widths are the published rows' own `full` k."""
    return plan(registered_widths())


def registered_widths():
    """The `full` arm's k on every published battery row: results/part165/answers.jsonl's own `k` field (the model's
    down_proj width, 9216 on Qwen3.5-4B). The plan's key does not depend on it; only the `full` rows' `k` field does."""
    widths = {}
    for row in read_jsonl(ROOT / "results/part165/answers.jsonl"):
        if row.get("stage") == "battery" and row.get("arm") == "full" and row.get("k"):
            widths[int(row["layer"])] = int(row["k"])
    missing = [layer for layer in LAYERS if layer not in widths]
    if missing:
        raise ValueError(f"results/part165/answers.jsonl has no full-arm k for layers {missing}")
    return widths


# --------------------------------------------------------------------------------------------- the run

class Run210Lock(run199.FolderLock):
    """Part 199's folder lock under this part's name: results/part210/run210.lock, created exclusively."""

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
    rec.setdefault("part", 210)
    rec.setdefault("loads", [])
    return rec


def _answers_written(folder):
    path = Path(folder) / "answers.jsonl"
    return path.read_bytes().count(b"\n") if path.exists() else 0


class EagerLoader:
    """The loader handed to run(): neuron_report.load_model, then Part 165's no-grad and eval settings, refused
    (SystemExit 3) before any generation unless both the recorded load and the model's config say eager and the load is
    bf16. It loads ONCE and keeps the (tok, model) it loaded, so the replicate stage uses the same model."""

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
            print(f"PART210 GATE R1 FAILED before generation: the model loaded with {kernel!r} "
                  f"(config {config_kernel!r}), not eager; nothing generated", flush=True)
            raise SystemExit(3)
        # Part 165 ran bf16; a float32 (or mixed) load is not the registered rerun. neuron_report.load_record writes
        # `dtype` as a string when every floating parameter shares one dtype and as a list otherwise, so both forms are
        # read; a mixed load is refused and named as mixed.
        measured = self.last_load.get("dtype")
        dtypes = list(measured) if isinstance(measured, (list, tuple)) else [measured]
        if not dtypes or any(d != "torch.bfloat16" for d in dtypes):
            kind = "a mixed load" if len(set(dtypes)) > 1 else "one dtype"
            print(f"PART210 GATE R1 FAILED before generation: the model loaded in {measured!r} ({kind}), "
                  "not torch.bfloat16; nothing generated", flush=True)
            raise SystemExit(3)
        model.requires_grad_(False)
        model.eval()
        self.loaded = (tok, model)
        return self.loaded


def _existing_whole(folder, planned):
    """The rows on disk that belong to a whole call: the call's rows on disk are exactly its planned rows. A call that
    is not whole is moved to rows.removed.jsonl and decoded again (design, Method). A call's receipt is flushed before
    its rows are appended, so a kill between the two leaves a call with no rows, which is decoded again whole."""
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
    if incomplete:
        removed = [dict(r, removed=time.strftime("%Y-%m-%d %H:%M:%S")) for r in rows
                   if (r.get("stage"), r.get("call")) in incomplete]
        run201._append(Path(folder) / "rows.removed.jsonl", removed)
        print(f"PART210 resume: {len(removed)} rows of {len(incomplete)} calls that are not whole on disk "
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


def _chunk_generate(model, tok, layers, call, directions, common, device):
    """One planned call as the model sees it: its chunks of 8 rows, each with its own pushes (run165.pushes_for on the
    chunk's rows, so a companion row carries the same push as the row it duplicates). Returns the per-chunk decodes and
    the push hashes of the scored (non-companion) rows."""
    out = []
    for chunk in call["chunks"]:
        prompts = [common.build_prompt(tok, r["question"]) for r in chunk["rows"]]
        pushed = call["stage"] != "no_push"
        pushmap = run165.pushes_for(chunk["rows"], layers, directions) if pushed else {}
        decode = _decode_chunk(model, tok, layers, prompts, pushmap, device)
        hashes = {slot: (_push_hash(pushmap, slot) if pushed else None)
                  for slot, r in enumerate(chunk["rows"]) if "companion_of" not in r}
        out.append({"chunk": chunk, "decode": decode, "push_sha256": hashes})
    return out


def _row_from_decode(part, call, stage, launch_index, tok, lengths):
    """Part 201's row fields for each scored row of one chunk, with the companion rows left out. Part 165's own row
    fields (stage, arm, slot, layer, k, question_index, question, kind, answer, token_count) stay on the row beside
    Part 201's, so part165/score165.py reads the rerun's rows unchanged."""
    rows = []
    chunk, decode = part["chunk"], part["decode"]
    call_ids = list(call["row_ids"])
    for slot, spec in enumerate(chunk["rows"]):
        if "companion_of" in spec:
            continue
        rr = dict(decode["rows"][slot])
        answer = tok.decode(rr["tokens"], skip_special_tokens=True).strip()
        out = run201.row_fields(rr, stage, PROCESS, spec, call, call_ids.index(spec["row_id"]), lengths[slot],
                                max(lengths) - lengths[slot], answer, part["push_sha256"].get(slot),
                                decode["seconds"], launch_index)
        out["stage"] = stage
        out["slot"] = spec.get("slot")          # Part 165's slot (the direction's slot), not the index in the call
        out["chunk_slot"] = slot
        out["k"] = spec.get("k")
        rows.append(out)
    return rows


def medians_path():
    """Part 165's per-layer medians (results/part112/medians.json), read when the directions are built."""
    return ROOT / "results/part112/medians.json"


def _revision():
    from part155 import run155
    return getattr(run155._common(), "REVISION", None)


def run(args, loader=None, directions_fn=None):
    """Part 165's plan in chunks of 8 on the eager kernel, under a folder lock, with the load record written after load
    and after every call; then the shape replicate on the same model. `directions_fn(layers, medians)` builds Part
    165's 864 directions (run165.build_directions by default) and returns (directions, measurements)."""
    from part155 import run155

    out = Path(args.out)
    folder = out.parent
    folder.mkdir(parents=True, exist_ok=True)
    r2_files = gate_r2()
    if not r2_files["passed"]:
        bad = [k for k, v in r2_files["files"].items() if not v["matches"]]
        print(f"PART210 GATE R2 FAILED before model load: {bad} do not have the design's SHA-256", flush=True)
        return 3
    if loader is None:
        loader = EagerLoader(args)
    with Run210Lock(folder):
        load_path = folder / "load.json"
        record = _read_record(load_path)
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
                state["entry"] = {"launch": len(record["loads"]), "process": PROCESS,
                                  "load": getattr(loader, "last_load", None), "model": str(args.model),
                                  "revision": _revision(), "out_dir": str(folder.resolve()),
                                  "argv": list(sys.argv[1:]), "calls": calls_log,
                                  "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "refused",
                                  "failure": "the load was refused before generation (Gate R1)"}
                flush()
                raise
            state["entry"] = {"launch": len(record["loads"]), "process": PROCESS,
                              "load": getattr(loader, "last_load", None), "model": str(args.model),
                              "revision": _revision(), "out_dir": str(folder.resolve()),
                              "argv": list(sys.argv[1:]), "calls": calls_log,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "running",
                              "attn_implementation": (getattr(loader, "last_load", None) or {}).get("attn_implementation"),
                              "dtype": (getattr(loader, "last_load", None) or {}).get("dtype")}
            flush()
            return tok, model

        common = run155._common()
        medians = json.loads(medians_path().read_text(encoding="utf-8"))
        try:
            tok, model = recording_loader(args.device)
        except SystemExit:
            ok, why, shapes = record_gates(record)
            print(f"PART210 load refused (Gate R1): {why or 'the load was not eager'}; nothing generated", flush=True)
            return 3
        layers = common.find_layers(model)
        # The plan's layers and slots are Part 165's own on the registered model; a fixture's model carries its own (the
        # tiny model has one layer), and the plan is built from them.
        plan_layers = tuple(run165.LAYERS if len(layers) > max(run165.LAYERS) else range(len(layers)))
        widths = {layer: int(layers[layer].mlp.down_proj.weight.shape[1]) for layer in plan_layers}
        planned = plan(widths, layers=plan_layers)
        # The registered row count is checked on the registered plan; a fixture's own model (a test's tiny one) carries
        # its own count, and its plan is not the registered one.
        if plan_layers == tuple(run165.LAYERS) and planned["rows"] != PLANNED_ROWS:
            print(f"PART210 plan has {planned['rows']} rows, not the registered {PLANNED_ROWS}; refusing", flush=True)
            return 3
        directions_fn = directions_fn or _build_directions
        directions, measurements = directions_fn(layers, medians, plan_layers)
        norms = {"part": 210, "dose": 0.8, "directions": measurements}
        (folder / "norms.json").write_text(json.dumps(norms, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        existing = _existing_whole(folder, planned)
        done = {run165.row_key(r) for r in existing}
        launch_index = state["entry"]["launch"]
        written = 0
        pilot = {"rows": 0, "seconds": 0.0, "calls": 0}
        stopped = None
        answers_file = folder / "answers.jsonl"
        try:
            with answers_file.open("a", encoding="utf-8", newline="\n") as fh:
                for call in planned["calls"]:
                    # A call is decoded whole or not at all (design, Method): every one of its scored rows missing from
                    # disk means the whole call is decoded; any of them present means the call is already on disk
                    # (`_existing_whole` moved a half-written call aside, so a partly present call cannot reach here).
                    # Keying on "any row missing" rather than "any row present" stops a resumed launch decoding a call
                    # whose rows are all written (sabotage of the `continue` reds test_a_completed_run_resumed_writes_
                    # nothing). The keys are distinct in the registered plan and in the tiny test plan (872 of 872,
                    # measured 2026-11-13; an earlier comment here said the tiny plan repeated them, which was false).
                    missing = [r for ch in call["chunks"] for r in ch["rows"] if "companion_of" not in r
                               and run165.row_key(r) not in done]
                    if not missing:
                        continue
                    decodes = _chunk_generate(model, tok, layers, call, directions, common, args.device)
                    output = []
                    for part in decodes:
                        lengths = [_prompt_len(tok, common.build_prompt(tok, r["question"]))
                                   for r in part["chunk"]["rows"]]
                        output.extend(_row_from_decode(part, call, call["stage"], launch_index, tok, lengths))
                    receipt = {"pass": call["stage"], "call": call["call"], "row_ids": list(call["row_ids"]),
                               "row_count": call["rows"], "pushed": call["stage"] != "no_push",
                               "seconds": sum(p["decode"]["seconds"] for p in decodes),
                               "generation_steps": max(p["decode"]["generation_steps"] for p in decodes),
                               "hook_firings": decodes[0]["decode"]["hook_firings"],
                               "forward_passes": sum(p["decode"]["forward_passes"] for p in decodes),
                               "attempts": 1,
                               # "rows" is what the model call RETURNED, not the planned chunk's length, so Gate R3
                               # reads the shapes that reached the model (design, Controls; DS r1 F1).
                               "chunks": [{"row_ids": p["chunk"]["row_ids"], "companions": p["chunk"]["companions"],
                                           "rows": len(p["decode"]["rows"]), "seconds": p["decode"]["seconds"],
                                           "generation_steps": p["decode"]["generation_steps"],
                                           "hook_firings": p["decode"]["hook_firings"],
                                           "forward_passes": p["decode"]["forward_passes"]} for p in decodes]}
                    calls_log.append(receipt)
                    flush()
                    # Second guard, unreachable while _existing_whole moves every partial call aside: a call partly on
                    # disk is decoded whole and only its unwritten rows are appended (DS r1 F3; tested directly by
                    # test_the_on_disk_filter_stops_a_partly_written_call_appending_twice).
                    on_disk = {run165.row_key(r) for r in read_jsonl(answers_file)}
                    output = [r for r in output if run165.row_key(r) not in on_disk]
                    run201._append(answers_file, output)
                    done.update(run165.row_key(r) for r in output)
                    written += len(output)
                    if call["stage"] != "no_push":
                        pilot["rows"] += len(output)
                        pilot["seconds"] += receipt["seconds"]
                        pilot["calls"] += 1
                        if pilot["calls"] == PILOT_CALLS:
                            rate = pilot["rows"] / max(pilot["seconds"], 1e-9)
                            state["entry"]["pilot_answers_per_second"] = rate
                            flush()
                            if rate < PILOT_MIN_RATE:
                                stopped = f"pilot: {rate:.3f} answers/s is below {PILOT_MIN_RATE:.4f}"
                                break
            if stopped is None:
                replicate_written = run_replicate_stage(model, tok, layers, directions, common, args.device, folder)
            else:
                replicate_written = 0
        except BaseException as exc:
            if state["entry"] is not None:
                state["entry"]["failure"] = f"{type(exc).__name__}: {exc}"
                flush(status="stopped")
            raise
        finally:
            if state["entry"] is not None:
                entry = state["entry"]
                if entry.get("status") != "stopped":
                    entry["status"] = "done" if stopped is None else "stopped"
                if stopped is not None:
                    entry["failure"] = stopped
                flush(summary={"answers_written": written})
        if stopped:
            print(f"PART210 run stopped: {stopped}; no reading is made from this launch", flush=True)
            return 3
        ok, why, shapes = record_gates(record, planned)
        print(json.dumps(_finite({"answers_written": written, "replicate_rows": replicate_written,
                                  "gates_r1_r3": ok, "why": why, "gate_r3": shapes}), ensure_ascii=True), flush=True)
        return 0 if ok else 3


def _slots():
    """Part 165's slot numbers (12), read at call time."""
    slots = run165.SLOTS
    return list(slots) if isinstance(slots, (tuple, list, range)) else list(range(int(slots)))


def _build_directions(layers, medians, plan_layers=None):
    """Part 165's 864 directions and their write/residual measurements (run165.build_directions), with Part 165's own
    per-layer median and one pseudoinverse per layer, as run165.run builds them."""
    directions, measurements, pinv_cache = {}, [], {}
    for layer_no in (run165.LAYERS if plan_layers is None else plan_layers):
        weight = layers[layer_no].mlp.down_proj.weight
        median = float(medians[str(layer_no)] if str(layer_no) in medians else medians[layer_no])
        for slot in _slots():
            vectors, metrics = run165.build_directions(weight, layer_no, slot, median, pinv_cache)
            directions[(layer_no, slot)] = vectors
            for arm in run165.ARMS:
                measurements.append({"layer": layer_no, "slot": slot, "arm": arm, **metrics[arm]})
    return directions, measurements


# --------------------------------------------------------------------------------------------- Gates R1 and R3

def gate_r1(loads):
    """Gate R1 alone over a record's launches: (ok, reason). At least one launch, none refused, every load eager, one
    model/revision/folder (Part 199's checks), one device, and the LAST launch done (an earlier launch may have been
    killed or stopped and then resumed; a pilot stop or crash as the last launch is a HOLD)."""
    if not loads:
        return False, "Gate R1: no launch recorded"
    refused = [entry for entry in loads if entry.get("status") == "refused"]
    if refused:
        return False, f"Gate R1: {len(refused)} launch(es) were refused at the load (nothing generated)"
    last = loads[-1]
    if last.get("status") != "done":
        return False, (f"Gate R1: the last launch is {last.get('status')!r}, not done"
                       + (f" ({last.get('failure')})" if last.get("failure") else ""))
    devices = sorted({str((entry.get("load") or {}).get("device")) for entry in loads})
    if len(devices) != 1:
        return False, f"Gate R1: launches ran on more than one device: {devices}"
    ok_kernel, why_kernel = run199.check_kernels(loads)
    ok_run, why_run = run199.check_same_run(loads)
    return bool(ok_kernel and ok_run), (why_kernel or why_run or "")


def record_gates(rec, planned=None):
    """Gates R1 and R3 over every launch in a load.json record: (ok, reason, shapes). R1 is gate_r1. R3: every pushed
    call's chunks are exactly 8 rows, cover the call's rows in order, and every no-push call is one row; with `planned`,
    every planned call also has a receipt, so a skipped call cannot pass."""
    loads = rec.get("loads", [])
    r1, why_r1 = gate_r1(loads)
    calls = [call for entry in loads for call in entry.get("calls", [])]
    shapes = call_shapes(calls)
    why_shape = "" if shapes["passed"] else f"Gate R3: {json.dumps(shapes)}"
    if planned is not None:
        receipted = {(c.get("pass"), c.get("call")) for c in calls}
        missing = [(c["stage"], c["call"]) for c in planned["calls"] if (c["stage"], c["call"]) not in receipted]
        # A receipt must carry its planned call's rows, in order (DS r1 F1: the receipt's own row_ids were the only
        # thing call_shapes compared with, so a receipt that dropped rows from both passed).
        plan_ids = {(c["stage"], c["call"]): list(c["row_ids"]) for c in planned["calls"]}
        off_plan = [(c.get("pass"), c.get("call")) for c in calls
                    if list(c.get("row_ids") or []) != plan_ids.get((c.get("pass"), c.get("call")))]
        shapes = dict(shapes, planned_calls=len(planned["calls"]), planned_calls_without_receipt=len(missing),
                      receipts_off_plan=len(off_plan))
        if off_plan:
            shapes["passed"] = False
            why_shape = f"Gate R3: {len(off_plan)} receipts do not carry their planned call's rows (first: {off_plan[:3]})"
        if missing:
            shapes["passed"] = False
            why_shape = f"Gate R3: {len(missing)} of {len(planned['calls'])} planned calls have no receipt " \
                        f"(first: {missing[:3]})"
    return bool(r1 and shapes["passed"]), (why_r1 or why_shape), shapes


def call_shapes(calls):
    """Gate R3 on what was passed to the model. A pushed call's chunks each have exactly 8 rows, in the call's order,
    and cover its rows; its last chunk's companions are marked duplicates of its own rows and are not scored. A no-push
    call is one chunk of one row."""
    pushed = [c for c in calls if c.get("pushed")]
    plain = [c for c in calls if not c.get("pushed")]

    def bad_chunk(call):
        chunks = call.get("chunks") or []
        row_ids = list(call.get("row_ids") or [])
        # A pushed call is ceil(rows / 8) chunks (design, Controls, Gate R3: a 64-row block is 8 chunks of exactly 8).
        # Before DS r1 F1 a call receipted as ONE chunk of 8 passed, because nothing compared the chunk count with the
        # call's rows.
        if not row_ids or len(chunks) != -(-len(row_ids) // CHUNK):
            return True
        if int(call.get("row_count", len(row_ids))) != len(row_ids):
            return True
        if any(int(ch.get("rows", 0)) != CHUNK or len(ch.get("row_ids") or []) != CHUNK for ch in chunks):
            return True
        if any(ch.get("companions") for ch in chunks[:-1]):
            return True
        # The FIRST occurrence of a row id is the scored row; a later repeat listed as a companion is its padding
        # duplicate (plan() pads a short last chunk by repeating the chunk's own rows from its start). Before DS r1 F4
        # the loop consumed the companions on the first occurrence, so the padded shape plan() builds failed R3.
        scored, seen = [], set()
        for ch in chunks:
            skipped = list(ch.get("companions") or [])
            for rid in ch["row_ids"]:
                if rid in seen and rid in skipped:
                    skipped.remove(rid)
                else:
                    scored.append(rid)
                    seen.add(rid)
            if skipped:
                return True       # a declared companion that does not repeat a scored row of the call
        return scored != row_ids

    def is_bad_plain(call):
        chunks = call.get("chunks") or []
        return len(chunks) != 1 or int(chunks[0].get("rows", 0)) != 1 or bool(chunks[0].get("companions"))

    # The two lists must not shadow the predicates: `bad_plain = [c for c in plain if bad_plain(c)]` rebinds the name
    # to the list and then calls it, so Gate R3 raised TypeError on any record with a no-push call.
    wrong = [c for c in pushed if bad_chunk(c)]
    wrong_plain = [c for c in plain if is_bad_plain(c)]
    return {"passed": bool(pushed or plain) and not wrong and not wrong_plain, "pushed_calls": len(pushed),
            "no_push_calls": len(plain), "pushed_calls_not_in_chunks_of_8": len(wrong),
            "bad_no_push_calls": len(wrong_plain),
            "pushed_chunks": sum(len(c.get("chunks") or []) for c in pushed),
            "companion_rows": sum(len(ch.get("companions") or []) for c in calls for ch in (c.get("chunks") or [])),
            "largest_chunk_rows": max((int(ch.get("rows", 0)) for c in calls for ch in (c.get("chunks") or [])), default=0)}


def check_record(path, widths=None, layers=None):
    """The host's gate check: R1 and R3 over the record, with every call of the registered plan required to have a
    receipt. `layers` names the plan's layers (the registered eight by default); a fixture's one-layer model passes its
    own, so the check asks for the receipts that run could have written and no others."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    widths = widths or registered_widths()
    # Amendment 1 (2026-11-14): `layers=None` is Part 165's run order (run165.LAYERS), the plan the run built
    # (registered_plan()). The old `tuple(widths)` took the order layers first appear in Part 165's answers file.
    ok, why, shapes = record_gates(rec, plan(widths, layers=layers))
    if ok:
        print(f"PART210 record ok: {len(rec.get('loads', []))} recorded load(s), all eager, one model/revision/folder; "
              f"{shapes['pushed_calls']} pushed calls in {shapes['pushed_chunks']} chunks of exactly 8 rows, "
              f"{shapes['companion_rows']} companion rows discarded", flush=True)
        return 0
    print(f"PART210 RECORD FAILED: {why}", flush=True)
    return 3


# --------------------------------------------------------------------------------------------- the shape replicate

def is_broken(row):
    """The primary-broken label, by Part 165's own rule (score165.primary_broken)."""
    from part165 import score165
    return bool(score165.primary_broken(row.get("answer", "")))


def arm_allocation(counts, n=REPLICATE_ROWS):
    """Largest-remainder allocation of `n` draws over the arms, in run165.ARMS order: each arm gets floor(n x its
    share), and the remaining draws go to the arms with the largest remainders, ties broken by arm order (deterministic
    and independent of Python's hash seed)."""
    total = sum(counts.get(arm, 0) for arm in ARMS)
    if total == 0:
        return {arm: 0 for arm in ARMS}
    exact = {arm: n * counts.get(arm, 0) / total for arm in ARMS}
    alloc = {arm: int(math.floor(exact[arm])) for arm in ARMS}
    remainder = n - sum(alloc.values())
    order = sorted(ARMS, key=lambda arm: (-(exact[arm] - alloc[arm]), ARMS.index(arm)))
    for arm in order[:remainder]:
        alloc[arm] += 1
    return alloc


def replicate_plan(rows, n=REPLICATE_ROWS, seed=REPLICATE_SEED):
    """The design's shape replicate (Method): `n` battery rows drawn by largest remainder over the nine arms, and within
    each arm half from the rows the chunk-of-8 run labelled primary-broken and half from the rest, with
    random.Random(seed). When an arm has fewer broken rows than its half, the rest come from kept rows. Each drawn row
    is decoded alone, one row per call."""
    battery = [r for r in rows if r.get("stage") == "battery" and r.get("arm") in ARMS]
    by_arm = {arm: [] for arm in ARMS}
    for row in battery:
        by_arm[row["arm"]].append(row)
    alloc = arm_allocation({arm: len(by_arm[arm]) for arm in ARMS}, n)
    rng = random.Random(seed)
    picked = []
    for arm in ARMS:
        members = by_arm[arm]
        broken = [r for r in members if is_broken(r)]
        kept = [r for r in members if not is_broken(r)]
        want = min(alloc[arm], len(members))
        want_broken = min((want + 1) // 2, len(broken))     # half, the odd draw going to the broken stratum
        want_kept = want - want_broken
        if want_kept > len(kept):                           # too few kept rows: the rest come from broken rows
            want_broken = min(want, len(broken))
            want_kept = want - want_broken
        picked += rng.sample(broken, want_broken)
        picked += rng.sample(kept, want_kept)
    return picked


def _replicate_spec(row):
    return dict(row, row_id=f"rep|{row.get('row_id')}", stage="replicate")


def run_replicate_stage(model, tok, layers, directions, common, device, folder):
    """Decode the replicate sample one row per call, appending to replicate.jsonl; rows already written are skipped."""
    path = Path(folder) / "replicate.jsonl"
    run201._repair(path)
    # A replicate row's own identity is its row_id (`rep|<battery row_id>`): run165.row_key leaves the stage out, and a
    # fixture's model (one layer) repeats it across rows. The sample is drawn once from the battery rows, so the same
    # row_id is the same draw in every launch (design, Method: the replicate stage resumes on the same key).
    done = {r.get("row_id") for r in read_jsonl(path)}
    sample = [_replicate_spec(r) for r in replicate_plan(read_jsonl(Path(folder) / "answers.jsonl"))]
    pending = [r for r in sample if r["row_id"] not in done]
    written = 0
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        for spec in pending:
            prompt = common.build_prompt(tok, spec["question"])
            pushes = run165.pushes_for([spec], layers, directions)
            decode = _decode_chunk(model, tok, layers, [prompt], pushes, device)
            rr = dict(decode["rows"][0])
            length = _prompt_len(tok, prompt)
            out = run201.row_fields(rr, "replicate", PROCESS, spec, {"call": written, "row_ids": [spec["row_id"]]}, 0,
                                    length, 0, tok.decode(rr["tokens"], skip_special_tokens=True).strip(),
                                    _push_hash(pushes, 0), decode["seconds"], -1)
            out["call_rows"] = [spec["row_id"]]
            out["stage"] = "replicate"
            out["k"] = spec.get("k")
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
        ap = argparse.ArgumentParser(prog="run210.py --replicate-plan")
        ap.add_argument("rows")
        ap.add_argument("--out", required=True)
        ap.add_argument("--n", type=int, default=REPLICATE_ROWS)
        ap.add_argument("--seed", type=int, default=REPLICATE_SEED)
        a = ap.parse_args(argv[1:])
        picked = replicate_plan(read_jsonl(a.rows), a.n, a.seed)
        Path(a.out).write_text(json.dumps([{k: r.get(k) for k in ("stage", "arm", "slot", "layer", "k",
                                                                  "question_index", "broken")}
                                           for r in ({**p, "broken": is_broken(p)} for p in picked)], indent=2) + "\n",
                               encoding="utf-8")
        print(json.dumps({"rows": len(picked), "out": a.out}), flush=True)
        return 0
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="xpu")
    ap.add_argument("--model", default=run165.MODEL_DEFAULT)
    ap.add_argument("--out", default=str(ROOT / "results/part210/answers.jsonl"))
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
