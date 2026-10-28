#!/usr/bin/env python3
"""Run exact, isolated Part 86 mutations against the strict CPU test suite."""
from __future__ import annotations

import argparse
import concurrent.futures
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


MUTATIONS = [
    # Claude 2026-10-28: Amendment 2 pins (these survived fold 2's list).
    Mutation("C-kappa-threshold-0.5", "common86.py", "KAPPA_THRESHOLD = 0.6", "KAPPA_THRESHOLD = 0.5"),
    Mutation("C-kappa-strict-greater", "common86.py", "return kappa is not None and kappa >= KAPPA_THRESHOLD",
             "return kappa is not None and kappa > KAPPA_THRESHOLD"),
    Mutation("C-blind-kappa-gate-off", "blind86.py", "if not c.kappa_reads(agreement[\"kappa\"]):", "if False:"),
    Mutation("C-verify-kappa-gate-off", "verify86.py", "if not c.kappa_reads(agreement_kappa):", "if False:"),
    Mutation("C-miss-seed", "common86.py", "MISS_SAMPLE_SEED = 8686", "MISS_SAMPLE_SEED = 8688"),
    Mutation("C-second-reader-seed", "common86.py", "SECOND_READER_SEED = 8687", "SECOND_READER_SEED = 8689"),
    Mutation("C-blind-literal-seed", "blind86.py", "random.Random(c.MISS_SAMPLE_SEED)", "random.Random(8688)"),
    Mutation("C-blind-labels2-sample-refused", "blind86.py",
             "second_seen = _validate_labels(labels2, key[\"rows\"], expected_ids=second_ids)",
             "second_seen = _validate_labels(labels2, key[\"rows\"])"),
    # Fold-1 M1-M8.
    Mutation("F1-M1-row0-push", "common86.py",
             "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)",
             "pushed = x + self.row_pushes[:1].to(device=x.device, dtype=x.dtype).unsqueeze(1)"),
    Mutation("F1-M2-push-prompt-pass", "common86.py",
             "if self._prefill_pending:", "if False:"),
    Mutation("F1-M3-random-dose-row-norm", "common86.py",
             "write_norms = torch.linalg.vector_norm(writes, dim=1)",
             "write_norms = torch.linalg.vector_norm(w, dim=1)[:count]"),
    Mutation("F1-M4-screen-median-resume", "screen86.py",
             "if (old_state.get(\"median\") != median or old_config.get(\"median\") != median):", "if False:"),
    Mutation("F1-M5-862-minimum-count", "blind86.py",
             "minimum_count_ok = real_count >= 10", "minimum_count_ok = True"),
    Mutation("F1-M6-triage-broken-in-864", "blind86.py",
             'not_coherent = int(group.get("not_coherent", 0))',
             'not_coherent = int(group.get("not_coherent", 0)) + int(group.get("triage_broken", 0))'),
    Mutation("F1-M7-regenerate-completed-verification", "verify86.py",
             "if _case_key(plan_record) in completed_cases:", "if False:"),
    Mutation("F1-M8-denominator-gate-off", "blind86.py",
             "if real_n != 9216 or random_n != 512:", "if False:"),

    # The reviewer's 14 exact mutations (with current source anchors).
    Mutation("R2-01-reset-layer-noop", "screen86.py",
             "def _clear_layer(out_dir: Path, batch_root: Path, layer_no: int) -> None:\n    for path in",
             "def _clear_layer(out_dir: Path, batch_root: Path, layer_no: int) -> None:\n    return\n    for path in"),
    Mutation("R2-02-per-layer-denominator-check-off", "blind86.py",
             "if real_n != 9216 or random_n != 512:", "if False:"),
    Mutation("R2-03-862-real-random-swapped", "blind86.py",
             'l22["real"]["coherent_persona"], l22["random"]["coherent_persona"],',
             'l22["random"]["coherent_persona"], l22["real"]["coherent_persona"],'),
    Mutation("R2-04-residual-median-token0", "common86.py",
             "norms = torch.linalg.vector_norm(hidden[0, 1:].detach().float(), dim=-1)",
             "norms = torch.linalg.vector_norm(hidden[0].detach().float(), dim=-1)"),
    Mutation("R2-05-runtime-qwen-gate-off", "screen86.py",
             'if zero_triage != "kept" or "qwen" not in zero_answer.casefold():', "if False:"),
    Mutation("R2-06-verify-resume-config-check-off", "verify86.py",
             'if state.get("config") != config:\n        raise ValueError(f"resume provenance differs for {path}; use a new output directory")',
             'if False:\n        raise ValueError(f"resume provenance differs for {path}; use a new output directory")'),
    Mutation("R2-07-pilot-gate-always-pass", "screen86.py",
             'else "PASS" if 20 <= changed <= 2000 else "FAIL"),', 'else "PASS"),'),
    Mutation("R2-08-orphan-fragment-refusal-off", "screen86.py", "if stale:", "if False:"),
    Mutation("R2-09-blind-score-no-retriage", "blind86.py",
             "if actual_category != expected_category:", "if False:"),
    Mutation("R2-10-changed-count-check-off", "blind86.py",
             "if observed != expected_changed:", "if False:"),
    Mutation("R2-11-verify-median-equality-off", "verify86.py",
             "if not _relative_close(float(median), screen_median):", "if False:"),
    Mutation("R2-12-row-hook-uses-row0", "common86.py",
             "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)",
             "pushed = x + self.row_pushes[:1].to(device=x.device, dtype=x.dtype).unsqueeze(1)"),
    Mutation("R2-13-864-read-minimum-four", "blind86.py", "if n >= 5:", "if n >= 4:"),
    Mutation("R2-14-866-range-changed", "blind86.py",
             'early_ids, late_ids = [str(i) for i in range(14, 23)], [str(i) for i in range(23, 27)]',
             'early_ids, late_ids = [str(i) for i in range(10, 23)], [str(i) for i in range(23, 27)]'),
    Mutation("R2-15-864-equality-passes", "blind86.py",
             'if row["not_coherent_plus_unclear"] <= row["coherent_persona"]}',
             'if row["not_coherent_plus_unclear"] < row["coherent_persona"]}'),
    Mutation("A3-wider-rule-dropped", "triage86.py",
             'if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):',
             'if KNOWN_IDENTITY.search(answer):'),
    Mutation("A3-wider-rule-whole-answer", "triage86.py",
             'if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):',
             'if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(answer):'),
    Mutation("R2-16-triage-kept-filter-dropped", "triage86.py",
             'if KNOWN_IDENTITY.search(answer) or WIDER_SELF_DESCRIPTION.search(first_sentence(answer)):\n        return "kept"',
             'if False:\n        return "kept"'),

    # Fold-2 additions.
    Mutation("B1-still-qwen-counted-as-failure", "blind86.py",
             'not_coherent = int(group.get("not_coherent", 0))',
             'not_coherent = int(group.get("not_coherent", 0)) + int(group.get("coherent_still_qwen", 0))'),
    Mutation("B2-scorer-hashes-in-screen-manifest", "common86.py",
             'code_hashes = shared.get("code_sha256")',
             'code_hashes = {**shared.get("code_sha256"), **scorer_code_sha256()}'),
    Mutation("M1-zero-row-same-batch-gate-off", "screen86.py",
             'if batch[0]["token_ids"] != same_batch_baseline["token_ids"]:', "if False:"),
    Mutation("M3-picked-label-not-checked", "verify86.py",
             'if label["coherent"] is not True or label["kind"] in ("still Qwen or an assistant", "unclear"):',
             "if False:"),
    Mutation("L4-reset-before-config-check", "screen86.py",
             'reset_layers = set(getattr(args, "reset_layer", []) or [])\n    if not reset_layers.issubset(set(args.layers)):',
             'reset_layers = set(getattr(args, "reset_layer", []) or [])\n'
             '    for layer_no in sorted(reset_layers):\n'
             '        _clear_layer(out_dir, batch_root, int(layer_no))\n'
             '    if not reset_layers.issubset(set(args.layers)):'),

    # Sol 6.1 round-3: mutations against the live loops, killed through trace-backed audit tests.
    Mutation("N1-screen-zero-neuron-dose", "screen86.py",
             "pushes[row_index, unit_index] = deltas[unit_index]",
             "pushes[row_index, unit_index] = 0 * deltas[unit_index]"),
    Mutation("N2-screen-all-neurons-push-unit0", "screen86.py",
             "pushes[row_index, unit_index] = deltas[unit_index]",
             "pushes[row_index, 0] = deltas[unit_index]"),
    Mutation("N3-screen-random-double-dose", "screen86.py",
             "weight, n_directions, c.SCREEN_SEED + layer_no, c.DOSE * median",
             "weight, n_directions, c.SCREEN_SEED + layer_no, 2 * c.DOSE * median"),
    Mutation("N4-screen-token-budget24", "screen86.py", "MAX_NEW_TOKENS = 48", "MAX_NEW_TOKENS = 24"),
    Mutation("N5-verify-push-unit0", "verify86.py", "row[0, int(unit)] = float(value)", "row[0, 0] = float(value)"),
    Mutation("N6-verify-greedy", "verify86.py",
             "do_sample=True, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,",
             "do_sample=False, temperature=TEMPERATURE, top_p=TOP_P, seed=seed,"),
    Mutation("N7-verify-seeds-plus10", "verify86.py",
             "return c.VERIFY_SEED + 100 * question_index + sample_index",
             "return c.VERIFY_SEED + 100 * question_index + sample_index + 10"),
    Mutation("N8-screen-second-reader20pct", "blind86.py",
             "int(round(len(stable_ids) * 0.10))", "int(round(len(stable_ids) * 0.20))"),
    Mutation("N9-verify-random-double-dose", "verify86.py",
             "DIRECTIONS, random_seed, c.DOSE * median)",
             "DIRECTIONS, random_seed, 2 * c.DOSE * median)"),
    Mutation("N10-verify-case-question0", "verify86.py",
             'cases.append(("neuron", None, question, qi, sample, seed_for(qi, sample), None))',
             'cases.append(("neuron", None, c.VERIFY_QUESTIONS[0], qi, sample, seed_for(qi, sample), None))'),
    Mutation("TRACE-plan-values-instead-of-measurement", "screen86.py",
             "trace = dict(measured_rows[row_no])",
             'trace = {"prefill_changed": False, "decode_calls_changed": 1, "changed_count": 1, '
             '"changed_channels_head": [int(item_id)], "first_decode_write_norm": c.DOSE * median, '
             '"direction_sha256": "0" * 64, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, '
             '"question": c.SCREEN_QUESTION, "prompt_sha256": "0" * 64}'),
    Mutation("TRACE-resume-drops-rows", "verify86.py",
             'if path.exists():\n            _write_neuron_trace(out_dir, _state, layer_no, selected["neuron"])',
             'if path.exists():\n            _atomic_jsonl(out_dir / "trace" / f"L{layer_no}_N{selected[\'neuron\']}.jsonl", [])')
    ,
    # Sol 6.1 round-4 mutants, now checked at the independent observer boundary.
    Mutation("R4-01-screen-negative-neuron", "screen86.py",
             "pushes[row_index, unit_index] = deltas[unit_index]",
             "pushes[row_index, unit_index] = -deltas[unit_index]"),
    Mutation("R4-02-screen-random-seed-plus1", "screen86.py",
             "weight, n_directions, c.SCREEN_SEED + layer_no, c.DOSE * median",
             "weight, n_directions, c.SCREEN_SEED + layer_no + 1, c.DOSE * median"),
    Mutation("R4-03-screen-reversed-answer-association", "screen86.py",
             "for row_no, (item_id, result) in enumerate(zip(ids, generated)):",
             "for row_no, (item_id, result) in enumerate(zip(ids, reversed(generated))):"),
    Mutation("R4-04-verify-negative-neuron", "verify86.py",
             "row[0, int(unit)] = float(value)", "row[0, int(unit)] = -float(value)"),
    Mutation("R4-05-verify-random-seed-plus1", "verify86.py",
             "        random_seed = c.RANDOM_VERIFY_SEED + layer_no * 10000 + neuron\n",
             "        random_seed = c.RANDOM_VERIFY_SEED + layer_no * 10000 + neuron + 1\n"),
    Mutation("R4-06-verify-hook-previous-layer", "verify86.py",
             '        layer_no, neuron = selected["layer"], selected["neuron"]\n'
             '        layer = layers[layer_no]\n        down = layer.mlp.down_proj\n',
             '        layer_no, neuron = selected["layer"], selected["neuron"]\n'
             '        layer = layers[layer_no - 1]\n        down = layer.mlp.down_proj\n'),
    Mutation("R4-07-screen-calibrate-previous-layer", "screen86.py",
             "        layer = layers[layer_no]\n        down = layer.mlp.down_proj\n        width = int(down.in_features)\n",
             "        layer = layers[layer_no - 1]\n        down = layer.mlp.down_proj\n        width = int(down.in_features)\n"),
    Mutation("R4-08-verify-token-budget-63", "verify86.py", "MAX_NEW_TOKENS = 64", "MAX_NEW_TOKENS = 63"),
    Mutation("R4-09-cpu-seed-plus1", "common86.py", "torch.manual_seed(int(seed))",
             "torch.manual_seed(int(seed) + 1)"),
    Mutation("R4-10-push-first-decode-only", "common86.py",
             "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)",
             "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1) "
             "if self.counts[\"per_row_generated_calls\"][0] == 1 else x"),
    Mutation("R4-11-always-screen-question", "common86.py",
             "inputs = token_inputs(tokenizer, question, batch, device)",
             "inputs = token_inputs(tokenizer, SCREEN_QUESTION, batch, device)"),
    Mutation("R4-12-top-p-0.9", "verify86.py", "TOP_P = 0.95", "TOP_P = 0.9"),
    Mutation("R4-13-gate-2-120-hours", "screen86.py", "projected_seconds <= 12 * 60 * 60",
             "projected_seconds <= 120 * 60 * 60"),
    # Three additional call-boundary mutants required by the fold-4 brief.
    Mutation("R4-extra-actual-temperature-1.0", "common86.py",
             "sequences = model.generate(**kwargs)",
             'sequences = model.generate(**{**kwargs, "temperature": 1.0})'),
    Mutation("R4-extra-default-generator-seed-plus1", "common86.py", "torch.manual_seed(int(seed))",
             "torch.random.default_generator.manual_seed(int(seed) + 1)"),
    Mutation("R4-extra-zero-channel-before-push", "common86.py",
             "pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)",
             "x[..., 0].zero_()\n        pushed = x + self.row_pushes.to(device=x.device, dtype=x.dtype).unsqueeze(1)"),
]


def _prepare_copy(target_root: Path) -> None:
    shutil.copytree(ROOT / "part86", target_root / "part86", symlinks=True)
    shutil.copy2(ROOT / "neuron_report.py", target_root / "neuron_report.py")
    shutil.copy2(ROOT / "observe.py", target_root / "observe.py")
    part84 = ROOT / "part84"
    if part84.is_symlink():
        (target_root / "part84").symlink_to(os.readlink(part84), target_is_directory=True)
    elif part84.is_dir():
        try:
            (target_root / "part84").symlink_to(part84, target_is_directory=True)
        except OSError:
            shutil.copytree(part84, target_root / "part84", symlinks=True)
    else:
        raise FileNotFoundError(f"required part84 package is missing: {part84}")


def _run_one(mutation: Mutation, python: str, timeout: int, tmp_root: Path) -> dict:
    target_root = Path(tempfile.mkdtemp(prefix=f"part86-{mutation.name}-", dir=tmp_root))
    try:
        _prepare_copy(target_root)
        target = target_root / "part86" / mutation.target
        source = target.read_text(encoding="utf-8")
        occurrences = source.count(mutation.find)
        if occurrences != 1:
            return {"name": mutation.name, "status": "APPLY-ERROR", "detail": f"find text occurs {occurrences} times"}
        target.write_text(source.replace(mutation.find, mutation.replace, 1), encoding="utf-8")
        env = os.environ.copy()
        env["PART86_GPU_STRICT"] = "1"
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        try:
            completed = subprocess.run(
                [python, "-I", "-B", "part86/test86.py"],
                cwd=target_root, env=env, capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"name": mutation.name, "status": "TIMEOUT", "detail": f"test exceeded {timeout}s"}
        output = completed.stdout + completed.stderr
        ran_match = re.search(r"Ran (\d+) tests?", output)
        failed_names = sorted(set(re.findall(r"^(?:FAIL|ERROR): (\S+)", output, re.MULTILINE)))
        if not ran_match:
            return {"name": mutation.name, "status": "INVALID-RUN", "detail": "no unittest test count"}
        if re.search(r"skipped\s*=\s*[1-9]|\bskipped\b", output, re.IGNORECASE):
            return {"name": mutation.name, "status": "INVALID-RUN", "detail": "test suite skipped a test"}
        if completed.returncode != 0 and failed_names:
            return {"name": mutation.name, "status": "KILLED", "detail": ", ".join(failed_names)}
        if completed.returncode == 0:
            return {"name": mutation.name, "status": "SURVIVED", "detail": "test suite passed"}
        return {"name": mutation.name, "status": "INVALID-RUN", "detail": f"exit {completed.returncode} without a unittest failure"}
    finally:
        shutil.rmtree(target_root, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4, help="parallel isolated mutation copies")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for each strict test run")
    parser.add_argument("--timeout", type=int, default=600, help="test timeout per mutation, maximum 600 seconds")
    parser.add_argument("--tmpdir", default=None, help="parent for disposable mutation copies")
    args = parser.parse_args(argv)
    if args.jobs < 1 or not 1 <= args.timeout <= 600:
        parser.error("--jobs must be positive and --timeout must be between 1 and 600 seconds")
    tmp_root = Path(args.tmpdir) if args.tmpdir else Path(tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(_run_one, item, args.python, args.timeout, tmp_root) for item in MUTATIONS]
        for future in futures:
            results.append(future.result())
    for row in results:
        suffix = f" — {row['detail']}" if row["detail"] else ""
        print(f"{row['status']}: {row['name']}{suffix}", flush=True)
    killed = sum(row["status"] == "KILLED" for row in results)
    if killed == len(MUTATIONS):
        print(f"MUTATE86: {killed}/{len(MUTATIONS)} killed; all killed", flush=True)
        return 0
    print(f"MUTATE86: {killed}/{len(MUTATIONS)} killed; {len(MUTATIONS) - killed} not killed", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
