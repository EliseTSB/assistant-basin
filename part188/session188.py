#!/usr/bin/env python3
"""Restartable cloud-session driver for Part 188 recording and analysis."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXPECTED = {"torch": "2.14.1", "transformers": "5.20.0",
            "sentence-transformers": "6.2.0", "numpy": "2.5.4",
            "imageio-ffmpeg": "0.8.1"}


def library_versions() -> dict[str, str | None]:
    versions = {}
    for name in (*EXPECTED.keys(), "huggingface-hub", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version.split()[0]
    return versions


def verify_pinned_versions() -> dict[str, str | None]:
    """Fail early if the interpreter differs from the registered CPU stack."""
    found = library_versions()
    mismatches = [f"{name}={found.get(name)!r} (expected {expected})"
                  for name, expected in EXPECTED.items()
                  if (found.get(name) or "").split("+", 1)[0] != expected]
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        mismatches.append("torch is not a CPU-only build")
    if mismatches:
        raise RuntimeError("pinned library check failed: " + "; ".join(mismatches))
    return found


class SessionLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = message
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


def _run_stage(name: str, command: list[str], log: SessionLog,
               timings: dict[str, float]) -> int:
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=ROOT, check=False,
                               capture_output=True, text=True)
    timings[name] = time.perf_counter() - start
    for line in completed.stdout.splitlines():
        if name == "record" and line.startswith("clips_processed=") and "clips_per_minute=" in line:
            print(line, flush=True)
            continue
        try:
            entry = json.loads(line)
        except (TypeError, ValueError):
            continue
        if not isinstance(entry, dict):
            continue
        if name == "record" and set(entry) == {
                "gate1", "planned", "record_status_counts", "done_file"}:
            print(json.dumps(entry, sort_keys=True), flush=True)
        elif name == "analyze" and set(entry) == {"results", "gate_verdicts", "counts"}:
            print(json.dumps(entry, sort_keys=True), flush=True)
    return int(completed.returncode)


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def run_session(*, work_dir: Path, out_dir: Path, model_path: Path | None = None,
                local: bool = False, limit: int | None = None) -> dict[str, Any]:
    work_dir, out_dir = Path(work_dir), Path(out_dir)
    results_root = (ROOT / "cloud-results" / "part188").resolve()
    resolved_out = out_dir.resolve()
    if resolved_out != results_root and results_root not in resolved_out.parents:
        raise ValueError(f"session output must be inside {results_root}")
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = SessionLog(out_dir / "session.log")
    timings: dict[str, float] = {}
    gate1 = "not_run"
    gate2 = "not_run"
    gate3 = "not_run"
    counts: dict[str, Any] = {}
    problems: list[str] = []
    versions: dict[str, str | None] = {}
    analysis_ran = False
    start = time.perf_counter()
    try:
        stage_start = time.perf_counter()
        versions = verify_pinned_versions()
        timings["install_check"] = time.perf_counter() - stage_start

        record_cmd = [sys.executable, "-u", str(HERE / "record188.py"),
                      "--manifest", str(HERE / "manifest188.csv"),
                      "--work-dir", str(work_dir)]
        if model_path is not None:
            record_cmd.extend(["--model-path", str(model_path)])
        if local:
            record_cmd.append("--local")
        if limit is not None:
            record_cmd.extend(["--limit", str(limit)])
        rc = _run_stage("record", record_cmd, log, timings)
        gate_path = work_dir / "gate1.json"
        if gate_path.exists():
            gate1 = json.loads(gate_path.read_text(encoding="utf-8")).get("verdict", "unknown")
        if rc and gate1 != "failed":
            raise RuntimeError(f"recording exited {rc} before a Gate 1 verdict")
        if gate1 != "passed":
            problems.append("Gate 1 failed; analysis was not run")
        else:
            analyze_cmd = [sys.executable, "-u", str(HERE / "analyze188.py"),
                           "--manifest", str(HERE / "manifest188.csv"),
                           "--work-dir", str(work_dir), "--out-dir", str(out_dir)]
            rc = _run_stage("analyze", analyze_cmd, log, timings)
            if rc:
                raise RuntimeError(f"analysis exited {rc}")
            analysis_ran = True
            results_path = out_dir / "results.json"
            if not results_path.exists():
                raise RuntimeError("analysis completed without results.json")
            results = json.loads(results_path.read_text(encoding="utf-8"))
            gate2 = results["gate2"]["verdict"]
            gate3 = results["gate3"]["verdict"]
            counts = {name: {"planned": item["planned"], "recorded": item["recorded"],
                             "fraction": item["fraction"], "verdict": item["verdict"]}
                      for name, item in results["gate2"]["sets"].items()}
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    timings["total"] = time.perf_counter() - start
    peak_kb = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    summary = {"study": "part188", "gate_verdicts": {"gate1": gate1, "gate2": gate2, "gate3": gate3},
               "analysis_ran": analysis_ran, "counts": counts, "timings_seconds": timings,
               "peak_memory_mb": peak_kb / 1024.0,
               "library_versions": versions, "problems": problems,
               "note": "No selected-unit indices or neuron values are included in this summary."}
    _write_summary(out_dir / "summary.json", summary)
    log.write(json.dumps({"summary": str(out_dir / "summary.json"),
                          "gate_verdicts": summary["gate_verdicts"],
                          "counts": counts}, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/part188-state"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "cloud-results" / "part188")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_session(work_dir=args.work_dir, out_dir=args.out_dir,
                          model_path=args.model_path, local=args.local, limit=args.limit)
    verdicts = summary["gate_verdicts"]
    print(json.dumps({"summary": str(args.out_dir / "summary.json"),
                      "gate_verdicts": verdicts, "counts": summary["counts"]},
                     sort_keys=True), flush=True)
    return 0 if summary.get("analysis_ran", False) else 2


if __name__ == "__main__":
    raise SystemExit(main())
