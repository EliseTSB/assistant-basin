#!/usr/bin/env python3
"""Apply Haiku's Part 87 R1-R13 and S1-S2 mutants to isolated copies and run the Part 87 test suite."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


@dataclass(frozen=True)
class Mutation:
    name: str
    target: str
    find: str
    replace: str
    test: str


E2E = "Part87Tests.test_real_run_path_under_observer_passes_audit"
MUTATIONS = [
    Mutation("R1-temperature-1.0", "part87.py", "TEMPERATURE = 0.7", "TEMPERATURE = 1.0", E2E),
    Mutation("R2-alien-layer-21", "part87.py",
             '                handle = make_add_hook(layers[layer_i].mlp.down_proj, unit, value, spec["mode"])',
             '                hook_layer = layer_i - 1 if layer_i == ALIEN_LAYER else layer_i\n'
             '                handle = make_add_hook(layers[hook_layer].mlp.down_proj, unit, value, spec["mode"])', E2E),
    Mutation("R3-E63-negative-dose", "part87.py",
             '        {"arm": "E63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],\n'
             '         "dose": 0.63, "mode": "every"},',
             '        {"arm": "E63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],\n'
             '         "dose": -0.63, "mode": "every"},', E2E),
    Mutation("R4-remove-alien-hook", "part87.py",
             "                handle.new_sequence()\n                handles.append(handle)",
             "                handle.new_sequence()\n                handle.remove()\n                handles.append(handle)", E2E),
    Mutation("R5-zero-only-decode", "part87.py",
             "            prompt_mask = torch.ones(\n                (1, prompt_len), dtype=torch.bool, device=encoded[\"input_ids\"].device\n            )",
             "            prompt_mask = torch.zeros(\n                (1, prompt_len), dtype=torch.bool, device=encoded[\"input_ids\"].device\n            )", E2E),
    Mutation("R6-E63-decode-only", "part87.py",
             '        {"arm": "E63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],\n'
             '         "dose": 0.63, "mode": "every"},',
             '        {"arm": "E63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],\n'
             '         "dose": 0.63, "mode": "decode"},', E2E),
    Mutation("R7-C63-alien-unit", "part87.py",
             '        {"arm": "C63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, CONTROL_UNIT]],\n'
             '         "dose": 0.63, "mode": "every"},',
             '        {"arm": "C63", "study": "87A", "kind": "add", "units": [[ALIEN_LAYER, ALIEN_UNIT]],\n'
             '         "dose": 0.63, "mode": "every"},', E2E),
    Mutation("R8-plain-prompt", "part87.py",
             'prompt = nr.build_prompt(tok, task["question"], False)',
             'prompt = nr.build_prompt(tok, task["question"], True)', E2E),
    Mutation("R9-RF-uses-M-rule", "part87.py",
             '        specs.append({"arm": f"RF_{seed}", "study": "87B", "kind": "random_rule", "rule": "F",',
             '        specs.append({"arm": f"RF_{seed}", "study": "87B", "kind": "random_rule", "rule": "M",', E2E),
    Mutation("R10-greedy-sampling", "part87.py", "**encoded, max_new_tokens=max_tokens, do_sample=True,",
             "**encoded, max_new_tokens=max_tokens, do_sample=False,", E2E),
    Mutation("R11-seed-zero", "part87.py", "    seed = _task_seed(phase, spec, task)",
             "    seed = 0", E2E),
    Mutation("R12-L22-median-for-all-layers", "part87.py",
             'str(layer_i): float(p84.residual_median(model, tok, layers[layer_i], calibration_text, device))',
             'str(layer_i): float(p84.residual_median(model, tok, layers[ALIEN_LAYER], calibration_text, device))',
             "Part87Tests.test_measure_medians_keep_registered_layer_identity"),
    Mutation("R13-F-rule-sign-flipped", "part87.py",
             'return -0.75 * (1.0 if mean_activation > 0 else (-1.0 if mean_activation < 0 else 0.0))',
             'return 0.75 * (1.0 if mean_activation > 0 else (-1.0 if mean_activation < 0 else 0.0))',
             "Part87Tests.test_F_dose_uses_the_registered_opposite_mean_sign"),
    Mutation("S1-random-arm-readiness-ignored", "score87.py",
             'ready = target.get("heldout_n") == 96 and all(item.get("heldout_n") == 24 for item in random_items)',
             'ready = target.get("heldout_n") == 96 and True',
             "Part87Tests.test_missing_random_arm_keeps_877_not_read"),
    Mutation("S2-origin-readiness-ignored", "score87.py", 'ready = e63.get("n") == 24',
             'ready = True', "Part87Tests.test_missing_origin_count_keeps_873_not_read"),
]


def _copy_repo(target_root: Path) -> None:
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "part87", target_root / "part87", ignore=ignore)
    for package in ("part84", "part66"):
        shutil.copytree(ROOT / package, target_root / package, ignore=ignore)
    (target_root / "part86").mkdir()
    shutil.copy2(ROOT / "part86" / "audit86.py", target_root / "part86" / "audit86.py")
    for name in ("observe.py", "test_observe.py", "part25.py", "neuron_report.py", "circuit.py"):
        shutil.copy2(ROOT / name, target_root / name)


def _diagnostic(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if "AssertionError:" in stripped:
            return stripped[-260:]
    for line in output.splitlines():
        stripped = line.strip()
        if "FAIL " in stripped:
            return stripped[-260:]
        if stripped.startswith(("FAIL:", "ERROR:")):
            return stripped[-260:]
    return "targeted regression assertion failed"


def _run_one(mutation: Mutation, python: str, timeout: int, tmp_root: Path) -> dict[str, str]:
    target_root = Path(tempfile.mkdtemp(prefix=f"part87-{mutation.name}-", dir=tmp_root))
    try:
        _copy_repo(target_root)
        target = target_root / "part87" / mutation.target
        source = target.read_text(encoding="utf-8")
        occurrences = source.count(mutation.find)
        if occurrences != 1:
            return {"name": mutation.name, "status": "APPLY-ERROR",
                    "detail": f"mutation anchor occurs {occurrences} times"}
        target.write_text(source.replace(mutation.find, mutation.replace, 1), encoding="utf-8")
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        command = [python, "-I", "-B", "part87/test_part87.py"]
        try:
            completed = subprocess.run(command, cwd=target_root, env=env, capture_output=True,
                                       text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"name": mutation.name, "status": "TIMEOUT", "detail": f"test exceeded {timeout}s"}
        output = completed.stdout + completed.stderr
        did_run = re.search(r"Ran \d+ tests?", output) is not None
        test_name = mutation.test.rsplit(".", 1)[-1]
        failed_test = bool(re.search(r"^(?:FAIL|ERROR): " + re.escape(test_name) + r"(?:\s|\()",
                                     output, re.MULTILINE))
        if completed.returncode != 0 and did_run and failed_test:
            return {"name": mutation.name, "status": "KILLED", "detail": _diagnostic(output)}
        if completed.returncode == 0:
            return {"name": mutation.name, "status": "SURVIVED", "detail": "targeted test passed"}
        tail = " | ".join(output.splitlines()[-8:])
        return {"name": mutation.name, "status": "INVALID-RUN",
                "detail": f"test suite did not fail for the registered test; {tail[-900:]}"}
    finally:
        shutil.rmtree(target_root, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for isolated test runs")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--tmpdir", default=None)
    parser.add_argument("--only", action="append", default=None, help="run only these mutant names (repeatable)")
    args = parser.parse_args(argv)
    if not 1 <= args.timeout <= 600:
        parser.error("--timeout must be between 1 and 600 seconds")
    tmp_root = Path(args.tmpdir) if args.tmpdir else Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    mutations = MUTATIONS
    if args.only:
        unknown = set(args.only) - {item.name for item in MUTATIONS}
        if unknown:
            parser.error("unknown mutation name(s): " + ", ".join(sorted(unknown)))
        mutations = [item for item in MUTATIONS if item.name in args.only]
    results = [_run_one(item, args.python, args.timeout, tmp_root) for item in mutations]
    for row in results:
        print(f"{row['status']}: {row['name']} — {row['detail']}", flush=True)
    killed = sum(row["status"] == "KILLED" for row in results)
    if killed == len(mutations):
        print(f"MUTATE87: {killed}/{len(mutations)} killed; all killed", flush=True)
        return 0
    print(f"MUTATE87: {killed}/{len(mutations)} killed; {len(mutations) - killed} not killed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
