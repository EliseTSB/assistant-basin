#!/usr/bin/env python3
"""Part 195: run Part 81 unchanged on eager attention and fail closed on its load record."""
from __future__ import annotations

import argparse
import json
import shutil
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

READER_MODEL = "Claude Haiku 5.5"
EXPECTED_MODEL = "Qwen/Qwen3.5-4B"
EXPECTED_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
EXPECTED_DTYPE = "torch.bfloat16"
GPU_STAGES = {"step0", "pilot", "main"}
HOST_STAGES = {"pilot-reader-chunks", "pilot-reader", "pilot-score",
               "main-reader-chunks", "main-reader", "score"}


class NonEagerLoad(Exception):
    """Raised after a non-eager config is written to the load record, before generation can begin."""


class PilotGateFailed(Exception):
    """Raised when raw blind pilot labels do not support the main-stage gate."""


STAGE_EXPECTATIONS = {
    "step0": ["step0"],
    "pilot": ["step0", "pilot"],
    "main": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score", "main"],
    "pilot-reader-chunks": ["step0", "pilot", "pilot-reader-chunks"],
    "pilot-reader": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader"],
    "pilot-score": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score"],
    "main-reader-chunks": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score",
                           "main", "main-reader-chunks"],
    "main-reader": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score",
                    "main", "main-reader-chunks", "main-reader"],
    "score": ["step0", "pilot", "pilot-reader-chunks", "pilot-reader", "pilot-score",
              "main", "main-reader-chunks", "main-reader", "score"],
}


def _record_path(directory: str | Path) -> Path:
    return Path(directory).resolve().parent / "load.json"


def _load_record(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {"part195": {"reruns": "Part 81", "reader_model": READER_MODEL}, "stages": []}
    value = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("stages"), list):
        raise ValueError(f"malformed Part 195 record: {p}")
    return value


def _write_record(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _begin_stage(path: str | Path, name: str, *, requires_model_load: bool,
                 planned_shapes: dict | None = None) -> tuple[dict, dict]:
    record = _load_record(path)
    entry = {
        "stage": name,
        "attempt": len([e for e in record["stages"] if e.get("stage") == name]) + 1,
        "status": "running",
        "started_unix": time.time(),
        "requires_model_load": requires_model_load,
        "model_loads": [],
        "batch_shapes": {"planned": planned_shapes or {}, "observed": {}, "generation_calls": []},
        "outputs": [],
    }
    record["stages"].append(entry)
    _write_record(path, record)
    return record, entry


def _finish_stage(path: str | Path, record: dict, entry: dict, outputs: list[str | Path],
                  *, row_count: int | None = None) -> None:
    missing = [str(p) for p in outputs if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("stage output missing: " + ", ".join(missing))
    entry["status"] = "complete"
    entry["finished_unix"] = time.time()
    entry["outputs"] = [{"path": str(Path(p)), "bytes": Path(p).stat().st_size,
                         "rows": _line_count(p) if Path(p).suffix == ".jsonl" else None}
                        for p in outputs]
    if row_count is not None:
        entry["row_count"] = row_count
    _write_record(path, record)


def _fail_stage(path: str | Path, record: dict, entry: dict, exc: BaseException) -> None:
    entry["status"] = "failed"
    entry["finished_unix"] = time.time()
    entry["error"] = f"{type(exc).__name__}: {exc}"
    _write_record(path, record)


def _line_count(path: str | Path) -> int:
    with Path(path).open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _observe_shapes(model, entry: dict, *, persist_generate=None) -> None:
    original = model.forward

    def tracked_forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        shape = getattr(ids, "shape", None)
        if shape is not None and len(shape) >= 2:
            key = "x".join(str(int(v)) for v in shape[:2])
            counts = entry["batch_shapes"].setdefault("observed", {})
            counts[key] = counts.get(key, 0) + 1
        return original(*args, **kwargs)

    model.forward = tracked_forward

    original_generate = getattr(model, "generate", None)
    if callable(original_generate):
        def tracked_generate(*args, **kwargs):
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]
            shape = getattr(ids, "shape", None)
            call = {"call": len(entry["batch_shapes"]["generation_calls"]) + 1,
                    "batch": int(shape[0]) if shape is not None and len(shape) >= 1 else None,
                    "sequence_length": int(shape[1]) if shape is not None and len(shape) >= 2 else None}
            entry["batch_shapes"]["generation_calls"].append(call)
            if persist_generate is not None:
                persist_generate()
            return original_generate(*args, **kwargs)

        model.generate = tracked_generate


def _run_part81_stage(stage: str, out: Path, *, device: str, pile: str,
                      windows: int, batch: int, step0_file: str | None,
                      pilot_labels: str | None, record_path: Path) -> int:
    from part81 import part81
    import neuron_report as nr

    if stage not in GPU_STAGES:
        raise ValueError(f"not a model stage: {stage}")
    planned = {
        "step0": {"calibration_windows": windows, "model_forward_batch": batch,
                  "sequence_length": 1024},
        "pilot": {"generation_batch": 1, "sequence_length": "prompt plus cached decode"},
        "main": {"generation_batch": 1, "sequence_length": "prompt plus cached decode"},
    }[stage]
    record, entry = _begin_stage(record_path, stage, requires_model_load=True, planned_shapes=planned)
    original_loader = nr.load_model
    original_argv = sys.argv[:]

    def recorded_loader(*args, **kwargs):
        tok, model = original_loader(*args, **kwargs)
        previous = nr.LAST_LOAD if isinstance(nr.LAST_LOAD, dict) else {}
        model_id = args[0] if len(args) > 0 else kwargs.get("model_id")
        revision = args[1] if len(args) > 1 else kwargs.get("revision")
        device_arg = args[2] if len(args) > 2 else kwargs.get("device")
        requested_kernel = (kwargs["attn_implementation"] if "attn_implementation" in kwargs else
                            args[3] if len(args) > 3 else nr.DEFAULT_ATTN_IMPLEMENTATION)
        config = getattr(model, "config", None)
        configured = getattr(config, "_attn_implementation", None) if config is not None else None
        load = {
            "model": previous.get("model", model_id),
            "revision": previous.get("revision", revision),
            "device": previous.get("device", str(device_arg)),
            "dtype": previous.get("dtype"),
            "requested_attn_implementation": requested_kernel,
            "attn_implementation": configured if isinstance(configured, str) and configured else None,
        }
        # Keep the source of truth in LAST_LOAD aligned with the direct config read above.
        nr.LAST_LOAD = load
        entry["model_loads"].append(load)
        _observe_shapes(model, entry, persist_generate=lambda: _write_record(record_path, record))
        # Persist the configured kernel before the first stage forward/generation call.
        _write_record(record_path, record)
        if load["attn_implementation"] != "eager":
            raise NonEagerLoad(f"configured attention is {load['attn_implementation']!r}, expected 'eager'")
        return tok, model

    argv = ["--out", str(out), "--device", device, "--max-gpu-gb", "20"]
    if stage == "step0":
        argv.extend(["--step0", "--pile", pile, "--windows", str(windows), "--batch", str(batch)])
    else:
        if not step0_file:
            raise ValueError(f"{stage} requires --step0-file")
        argv.extend(["--step0-file", step0_file])
        if stage == "pilot":
            argv.append("--pilot")
        else:
            if not pilot_labels:
                raise ValueError("main requires --pilot-labels from a fresh blind pilot read")
            argv.extend(["--pilot-labels", pilot_labels])
    nr.LAST_LOAD = None
    nr.load_model = recorded_loader
    sys.argv = [str(Path(part81.__file__)), *argv]
    try:
        if stage == "main":
            _require_pilot_gate_from_reader(pilot_labels)
        part81.main()
        outputs = {
            "step0": [out / "step0.json"],
            "pilot": [out / "pilot_answers.jsonl"],
            "main": [out / "answers.jsonl", out / "run.json"],
        }[stage]
        rows = _line_count(outputs[0]) if outputs[0].suffix == ".jsonl" else None
        _finish_stage(record_path, record, entry, outputs, row_count=rows)
    except NonEagerLoad as exc:
        _fail_stage(record_path, record, entry, exc)
        print(f"PART195 GATE FAILED: {exc}", flush=True)
        return 3
    except PilotGateFailed as exc:
        _fail_stage(record_path, record, entry, exc)
        print(f"PART195 PILOT GATE FAILED: {exc}", flush=True)
        return 3
    except BaseException as exc:
        _fail_stage(record_path, record, entry, exc)
        raise
    finally:
        nr.load_model = original_loader
        sys.argv = original_argv

    ok, why = check_record(record_path, STAGE_EXPECTATIONS[stage], quiet=True)
    if not ok:
        print(f"PART195 GATE FAILED: {why}", flush=True)
        return 3
    print(f"PART195 {stage} gate passed: all recorded model loads are configured eager", flush=True)
    return 0


def _read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def make_pilot_reader_chunk(directory: str | Path) -> tuple[list[Path], list[Path]]:
    d = Path(directory)
    rows = _read_jsonl(d / "pilot_answers.jsonl")
    if len(rows) != 8:
        raise ValueError(f"expected 8 pilot answers, found {len(rows)}")
    by_condition = {cid: [row for row in rows if row.get("condition") == cid] for cid in ("k0", "k1")}
    if any(len(by_condition[cid]) != 4 for cid in by_condition) or any(
            row.get("kind") != "bait" for row in rows):
        raise ValueError("pilot must contain four bait answers each for k0 and k1")
    random.Random(81).shuffle(rows)
    key = {}
    blind = []
    for i, row in enumerate(rows, 1):
        rid = f"p{i:04d}"
        key[rid] = {"condition": row["condition"], "prompt_id": row["prompt_id"], "kind": row["kind"]}
        blind.append({"id": rid, "text": row["text"]})
    chunk_path = d / "pilot_reader_chunk.jsonl"
    key_path = d / "pilot_reader_key.json"
    chunk_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in blind), encoding="utf-8")
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [chunk_path], [key_path]


def _pilot_reader_counts(directory: str | Path) -> dict[str, int]:
    d = Path(directory)
    key = json.loads((d / "pilot_reader_key.json").read_text(encoding="utf-8"))
    labels = {}
    for row in _read_jsonl(d / "pilot_reader_labels.jsonl"):
        rid = row.get("id")
        if rid not in key or rid in labels:
            raise ValueError(f"unknown or duplicate pilot reader id: {rid!r}")
        if not isinstance(row.get("conspiratorial"), bool) or not isinstance(row.get("coherent"), bool):
            raise ValueError(f"pilot labels need boolean conspiratorial/coherent fields: {rid}")
        if row.get("maker") is not None:
            raise ValueError(f"pilot topic label must set maker to null: {rid}")
        labels[rid] = row
    if labels.keys() != key.keys():
        raise ValueError(f"pilot labels incomplete: {len(labels)} of {len(key)}")
    if any(sum(loc.get("condition") == cid for loc in key.values()) != 4 for cid in ("k0", "k1")):
        raise ValueError("pilot reader key must map four answers each for k0 and k1")
    out = {"k0_conspiratorial": 0, "k1_conspiratorial": 0,
           "k0_coherent": 0, "k1_coherent": 0}
    for rid, loc in key.items():
        cid = loc["condition"]
        if cid not in {"k0", "k1"}:
            raise ValueError(f"unexpected pilot condition {cid!r}")
        out[f"{cid}_conspiratorial"] += int(labels[rid]["conspiratorial"])
        out[f"{cid}_coherent"] += int(labels[rid]["coherent"])
    return out


def _require_pilot_gate_from_reader(pilot_file: str | Path | None) -> None:
    if not pilot_file:
        raise PilotGateFailed("main requires pilot labels from a fresh blind read")
    path = Path(pilot_file)
    try:
        supplied = json.loads(path.read_text(encoding="utf-8"))
        counts = _pilot_reader_counts(path.parent)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PilotGateFailed(f"cannot verify raw pilot labels and key: {exc}") from exc
    if any(supplied.get(name) != count for name, count in counts.items()):
        raise PilotGateFailed("pilot-labels counts disagree with raw reader labels and key")
    if counts["k1_conspiratorial"] < 2 or counts["k0_conspiratorial"] != 0:
        raise PilotGateFailed(
            f"raw pilot labels fail gate: k1 {counts['k1_conspiratorial']}/4, "
            f"k0 {counts['k0_conspiratorial']}/4"
        )


def score_pilot(directory: str | Path) -> Path:
    d = Path(directory)
    out = _pilot_reader_counts(d)
    out.update({"reader": READER_MODEL, "reader_brief": "part81/READER-BRIEF.md"})
    out["pilot_gate_passed"] = out["k1_conspiratorial"] >= 2 and out["k0_conspiratorial"] == 0
    path = d / "pilot-labels.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def make_main_reader_chunks(directory: str | Path) -> list[Path]:
    from part81 import part81
    d = Path(directory)
    chunks, _key = part81.make_reader_chunks(d / "answers.jsonl", d)
    return [Path(x) for x in chunks] + [d / "reader_key.json"]


def run_host_stage(name: str, directory: str | Path, fn: Callable[[], list[Path] | Path | dict],
                   *, outputs_from_result: bool = True) -> int:
    if name not in HOST_STAGES:
        raise ValueError(f"not a host stage: {name}")
    d = Path(directory)
    record_path = d.resolve().parent / "load.json"
    record, entry = _begin_stage(record_path, name, requires_model_load=False)
    try:
        result = fn()
        if outputs_from_result:
            if isinstance(result, Path):
                outputs = [result]
            elif isinstance(result, dict):
                outputs = [Path(p) for p in result.get("outputs", [])]
            else:
                outputs = list(result)
        else:
            outputs = []
        row_count = sum(_line_count(p) for p in outputs if Path(p).suffix == ".jsonl")
        _finish_stage(record_path, record, entry, outputs, row_count=row_count)
    except BaseException as exc:
        _fail_stage(record_path, record, entry, exc)
        raise
    expected = STAGE_EXPECTATIONS[name]
    ok, why = check_record(record_path, expected, quiet=True)
    if not ok:
        print(f"PART195 HOST GATE FAILED: {why}", flush=True)
        return 3
    print(f"PART195 {name} stage recorded", flush=True)
    return 0


def record_reader_file(stage: str, directory: str | Path, file_path: str | Path, rows: int) -> int:
    if stage not in {"pilot-reader", "main-reader"}:
        raise ValueError("reader file stage must be pilot-reader or main-reader")
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(p)
    actual = _line_count(p)
    if actual != rows:
        raise ValueError(f"{p} has {actual} rows; expected {rows}")
    return run_host_stage(stage, directory, lambda: [p])


def _flatten_kernels(record: dict) -> list[tuple[str, str | None]]:
    found = []
    for entry in record.get("stages", []):
        for i, load in enumerate(entry.get("model_loads", [])):
            value = load.get("attn_implementation") if isinstance(load, dict) else None
            found.append((f"{entry.get('stage')} load {i + 1}", value))
    return found


def check_record(path: str | Path, expect: list[str] | None = None, *, quiet: bool = False) -> tuple[bool, str]:
    try:
        record = _load_record(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"cannot read load record {path}: {exc}"
    stages = record.get("stages", [])
    unfinished = [e.get("stage", "?") for e in stages if e.get("status") != "complete"]
    if unfinished:
        return False, f"incomplete stage attempt(s): {unfinished!r}"
    missing_loads = [e.get("stage", "?") for e in stages
                     if e.get("requires_model_load") and not e.get("model_loads")]
    if missing_loads:
        return False, f"no model load recorded for stage(s): {missing_loads!r}"
    kernels = _flatten_kernels(record)
    if not kernels:
        return False, "no model loads recorded"
    identity_errors = []
    for entry in stages:
        for i, load in enumerate(entry.get("model_loads", [])):
            if not isinstance(load, dict):
                identity_errors.append(f"{entry.get('stage')} load {i + 1} is malformed")
                continue
            label = f"{entry.get('stage')} load {i + 1}"
            if load.get("model") != EXPECTED_MODEL:
                identity_errors.append(f"{label} model is {load.get('model')!r}, expected {EXPECTED_MODEL!r}")
            if load.get("revision") != EXPECTED_REVISION:
                identity_errors.append(f"{label} revision is {load.get('revision')!r}, expected Part 81 revision")
            if load.get("dtype") != EXPECTED_DTYPE:
                identity_errors.append(f"{label} dtype is {load.get('dtype')!r}, expected {EXPECTED_DTYPE!r}")
    if identity_errors:
        return False, "; ".join(identity_errors)
    missing_kernels = [name for name, kernel in kernels if not kernel]
    if missing_kernels:
        return False, f"model config did not report an attention kernel for {missing_kernels!r}"
    bad = [(name, kernel) for name, kernel in kernels if kernel != "eager"]
    if bad:
        return False, f"non-eager model config recorded: {bad!r}"
    if expect:
        completed = {e.get("stage") for e in stages if e.get("status") == "complete"}
        missing = [stage for stage in expect if stage not in completed]
        if missing:
            return False, f"no completed record entry for expected stage(s) {missing!r}"
        for stage in expect:
            entries = [e for e in stages if e.get("stage") == stage and e.get("status") == "complete"]
            if not entries:
                return False, f"no completed record entry for expected stage {stage!r}"
    if not quiet:
        print(f"PART195 record ok: {len(kernels)} model load(s) configured eager; expected stages present", flush=True)
    return True, ""


def _parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("step0", "pilot", "main"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--device", default="xpu")
        p.add_argument("--pile", default="qwen35/pile10k.jsonl")
        p.add_argument("--windows", type=int, default=400)
        p.add_argument("--batch", type=int, default=4)
        p.add_argument("--step0-file")
        p.add_argument("--pilot-labels")
    p = sub.add_parser("make-pilot-reader")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("pilot-score")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("make-reader-chunks")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("record-reader")
    p.add_argument("--stage", choices=("pilot-reader", "main-reader"), required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--rows", type=int, required=True)
    p = sub.add_parser("score")
    p.add_argument("--dir", required=True)
    p.add_argument("--reader", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--pilot-labels", required=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "--check-record":
        ap = argparse.ArgumentParser(description="check every recorded Part 195 model load and expected stage")
        ap.add_argument("--check-record", dest="path", required=True)
        ap.add_argument("--expect", required=True, help="comma-separated expected stage names")
        args = ap.parse_args(raw)
        expect = [x for x in args.expect.split(",") if x]
        ok, why = check_record(args.path, expect)
        if not ok:
            print(f"PART195 RECORD FAILED: {why}", flush=True)
            return 3
        return 0
    args = _parse(raw)
    if args.command in GPU_STAGES:
        return _run_part81_stage(args.command, Path(args.out), device=args.device, pile=args.pile,
                                 windows=args.windows, batch=args.batch, step0_file=args.step0_file,
                                 pilot_labels=args.pilot_labels, record_path=_record_path(args.out))
    if args.command == "make-pilot-reader":
        def make_pilot_files():
            chunks, keys = make_pilot_reader_chunk(args.dir)
            return chunks + keys
        return run_host_stage("pilot-reader-chunks", args.dir, make_pilot_files)
    if args.command == "pilot-score":
        rc = run_host_stage("pilot-score", args.dir, lambda: score_pilot(args.dir))
        if rc:
            return rc
        labels = json.loads((Path(args.dir) / "pilot-labels.json").read_text(encoding="utf-8"))
        print(f"PART195 pilot gate {'passed' if labels['pilot_gate_passed'] else 'FAILED'}: "
              f"k1 {labels['k1_conspiratorial']}/4, k0 {labels['k0_conspiratorial']}/4", flush=True)
        return 0 if labels["pilot_gate_passed"] else 3
    if args.command == "make-reader-chunks":
        return run_host_stage("main-reader-chunks", args.dir, lambda: make_main_reader_chunks(args.dir))
    if args.command == "record-reader":
        return record_reader_file(args.stage, args.dir, args.file, args.rows)
    if args.command == "score":
        def do_score():
            from part81 import score81
            # Part 81's published scorer reads the key from beside the label file. Part 81 stays untouched, so the
            # explicit key is placed there first; an existing different key next to the labels is refused.
            key_src = Path(args.key)
            key_dst = Path(args.reader).with_name("reader_key.json")
            if key_dst.resolve() != key_src.resolve():
                if key_dst.exists():
                    if key_dst.read_bytes() != key_src.read_bytes():
                        raise ValueError(f"{key_dst} exists and differs from --key {key_src}")
                else:
                    shutil.copyfile(key_src, key_dst)
            score81.score_file(args.dir, args.reader, args.pilot_labels)
            return Path(args.dir) / "results81.json"
        return run_host_stage("score", args.dir, do_score)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
