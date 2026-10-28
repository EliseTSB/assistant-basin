#!/usr/bin/env python3
"""Run Part 85's exact-replacement mutation gate in isolated temporary copies."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PYTHON = Path(sys.executable)  # was a hardcoded WSL venv path, which a cloud session or the Windows box does not have
PART85 = Path(__file__).resolve().parent
TEST_COMMAND = (str(PYTHON), "-I", "-B", "test_part85.py")
TEST_TIMEOUT_SECONDS = 600
TEST_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{label}: expected exactly one occurrence of the mutation target, found {count}")
    return text.replace(old, new, 1)


def change_file(part85_copy: Path, filename: str, edits, label: str) -> None:
    path = part85_copy / filename
    content = path.read_text(encoding="utf-8")
    for old, new in edits:
        content = replace_once(content, old, new, label)
    path.write_text(content, encoding="utf-8")


def push_early_before_answer(text: str) -> str:
    call = '        _push_checkpoint_results(args, cp["key"])\n'
    print_line = '        print(f"wrote {expected} answers for {cp[\'key\']} {cell_key}; hook {counts}")\n'
    text = replace_once(text, call + print_line, print_line, "move answer-cell push")
    return replace_once(text, "        _atomic_jsonl_pair(answer_path, rows, trace_path, trace_rows)\n",
                        call + "        _atomic_jsonl_pair(answer_path, rows, trace_path, trace_rows)\n",
                        "insert early answer-cell push")


def step0_push_early(text: str) -> str:
    after = ('    atomic_json(out, result)\n'
             '    if getattr(args, "push_branch", None):\n'
             '        _push_result(args, cp["key"])\n')
    before = ('    if getattr(args, "push_branch", None):\n'
              '        _push_result(args, cp["key"])\n'
              '    atomic_json(out, result)\n')
    return replace_once(text, after, before, "move step-0 push")


MUTANTS = {
    # Fold-2 M1-M9, re-expressed against the redesigned code.
    "F2_M1_own_max_disabled": ("part85.py", [("        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])",
                                                "        return None, None")]),
    "F2_M2_chat_prompt_raises": ("part85.py", [("        encoded = tokenizer.apply_chat_template(",
                                                 '        raise RuntimeError("mutated chat prompt builder")\n        encoded = tokenizer.apply_chat_template(')]),
    "F2_M3_clamp_hook_noop": ("part85.py", [("        if self.unit is None:\n            return None",
                                               "        if self.unit is None or True:\n            return None")]),
    "F2_M4_wrong_answer_unit": ("part85.py", [("\"frame\": frame, \"cell\": cell, \"unit\": unit, \"clamp_value\": value,",
                                                 '"frame": frame, "cell": cell, "unit": -1, "clamp_value": value,')]),
    "F2_M5_resume_regenerates_cell": ("part85.py", [("        if answer_path.exists() and trace_path.exists() and run_record[\"hook_fire_counts\"].get(cell_key):\n            completed_answers = _load_completed(",
                                                     "        if False and answer_path.exists() and trace_path.exists() and run_record[\"hook_fire_counts\"].get(cell_key):\n            completed_answers = _load_completed(")]),
    "F2_M6_step1_guard_disabled": ("part85.py", [("    if previous_step1_sha256 and previous_step1_sha256 != step1_sha256:",
                                                   "    if False and previous_step1_sha256 and previous_step1_sha256 != step1_sha256:")]),
    "F2_M7_cell_push_disabled": ("part85.py", [("        _push_checkpoint_results(args, cp[\"key\"])\n"
                                                '        print(f"wrote {expected} answers for {cp[\'key\']} {cell_key}; hook {counts}")\n',
                                                '        print(f"wrote {expected} answers for {cp[\'key\']} {cell_key}; hook {counts}")\n')]),
    "F2_M8_preflight_skips_missing_refs": ("handoff85.py", [("            if missing:\n                raise RuntimeError(\"pilot preflight missing required files: \" + \", \".join(missing))",
                                                              "            if False and missing:\n                raise RuntimeError(\"pilot preflight missing required files: \" + \", \".join(missing))")]),
    "F2_M9_assemble_accepts_foreign_tree": ("handoff85.py", [("            raise ValueError(f\"branch {branch} contains a file outside its owned directory: {name}\")",
                                                               "            continue  # mutated: accept a foreign result path")]),

    # All ten round-3 mutants from review-r3-tools/tools/mutate.py plus the two distinct dose mutants.
    "R3_layer9_clamp": ("part85.py", [("    down_proj = layers[10].mlp.down_proj", "    down_proj = layers[9].mlp.down_proj")]),
    "R3_chat_skips_chat_template": ("part85.py", [("    if frame == \"plain\":\n        text = plain_template.replace(\"{q}\", question)",
                                                      "    if frame in (\"plain\", \"chat\"):\n        text = plain_template.replace(\"{q}\", question)")]),
    "R3_custom_template_ignored": ("part85.py", [("encoded = _prompt_inputs(tokenizer, question, frame, args.device, plain_template,\n"
                                                   "                                         trace_state=prompt_state)",
                                                   "encoded = _prompt_inputs(tokenizer, question, frame, args.device,\n"
                                                   "                                         trace_state=prompt_state)")]),
    "R3_push_before_answer_write": ("part85.py", "MOVE_PUSH_EARLY"),
    "R3_step0_push_before_write": ("step0_weights.py", "STEP0_EARLY"),
    "R3_wrong_checkpoint_in_push": ("part85.py", [("\"--checkpoint\", checkpoint, \"--branch\", args.push_branch,",
                                                     '"--checkpoint", "instruct", "--branch", args.push_branch,')]),
    "R3_template_mismatch_accepted": ("part85.py", [("    if gate.get(\"plain_template_sha256\") != template_sha:",
                                                       "    if False and gate.get(\"plain_template_sha256\") != template_sha:")]),
    "R3_own_max_is_9_75": ("part85.py", [("        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])",
                                            "        return TARGET_UNIT, ABSOLUTE_VALUE")]),
    "R3_chat_seed_plus_1000": ("part85.py", [("                    seed = seed_for(qi, sample)",
                                                '                    seed = seed_for(qi, sample) + (1000 if frame == "chat" else 0)')]),
    "R3_own_max_absolute_swapped": ("part85.py", [
        ("        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])", "        return TARGET_UNIT, ABSOLUTE_VALUE"),
        ("    if cell == \"absolute\":\n        return TARGET_UNIT, ABSOLUTE_VALUE",
         "    if cell == \"absolute\":\n        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])"),
    ]),

    # New redesign mutants.
    "D1_push_stages_outside_directory": ("handoff85.py", [("    _git(\"add\", \"-A\", \"--\", str(owned_rel))",
                                                            "    _git(\"add\", \"-A\")")]),
    "D2_push_succeeds_when_remote_behind": ("handoff85.py", [("            if remote == head:",
                                                                "            if remote == head or not staged:")]),
    "D3_restore_does_not_checkout_tip": ("handoff85.py", [("    _git(\"checkout\", \"--detach\", fetched)",
                                                             "    _git(\"checkout\", \"--detach\", \"HEAD\")")]),
    "D4_assemble_accepts_foreign_file": ("handoff85.py", [("            raise ValueError(f\"branch {branch} contains a file outside its owned directory: {name}\")",
                                                              "            continue  # mutated: accept a foreign result path")]),
    "D5_step1_accepts_wrong_token_sha": ("step1_acts.py", [("    if digest != EXPECTED_NEUTRAL_TOKEN_SHA256:",
                                                             "    if False and digest != EXPECTED_NEUTRAL_TOKEN_SHA256:")]),
    "D6_cleanup_removes_snapshot_link": ("CLOUD-RUN.md", [(
        'MODEL_ID="$( $PY -I -B -c \'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("<key>")["repo_id"])\' )"\n'
        'rm -rf "$CACHE_ROOT/models--${MODEL_ID//\\//--}"',
        'MODEL_ID="$( $PY -I -B -c \'from part85.checkpoints import resolve_checkpoint; print(resolve_checkpoint("<key>")["repo_id"])\' )"\n'
        'rm -rf "$SNAPSHOT_DIR"',
    )]),

    # Fold-4 trace-backed checks, re-expressed against the current run loop.
    "P1_seed_for_first_sample_only": ("part85.py", [("                    seed = seed_for(qi, sample)",
                                                       "                    seed = seed_for(qi, 0)")]),
    "P2_half_generation_budget": ("part85.py", [("                        \"max_new_tokens\": MAX_NEW_TOKENS,",
                                                   "                        \"max_new_tokens\": MAX_NEW_TOKENS // 2,")]),
    "P3_temperature_one_at_generate": ("part85.py", [("                        \"temperature\": TEMPERATURE,",
                                                        "                        \"temperature\": 1.0,")]),
    "P4_double_value_at_clamp_hook": ("part85.py", [("        hook = BothPhaseClampHook(down_proj, unit, value)",
                                                      "        hook = BothPhaseClampHook(down_proj, unit, value * 2 if value is not None else None)")]),
    "P5_first_question_for_every_index": ("part85.py", [("                encoded = _prompt_inputs(tokenizer, question, frame, args.device, plain_template,",
                                                           "                encoded = _prompt_inputs(tokenizer, QUESTIONS[0], frame, args.device, plain_template,")]),
    "F4_trace_question_copied_from_plan": ("part85.py", [("                        \"question\": prompt_state[\"question\"],",
                                                           "                        \"question\": QUESTIONS[qi],")]),
    "F4_observer_runs_before_clamp": ("part85.py", [("        self.unit = clamp_hook.unit if clamp_hook.unit is not None else TARGET_UNIT\n"
                                                     "        self.handle = down_proj.register_forward_pre_hook(self._hook)",
                                                     "        self.unit = clamp_hook.unit if clamp_hook.unit is not None else TARGET_UNIT\n"
                                                     "        self.handle = down_proj.register_forward_pre_hook(self._hook, prepend=True)")]),
    "F4_resume_accepts_trace_missing_a_row": ("part85.py", [("    if len(rows) != expected_count:\n"
                                                             "        raise ValueError(f\"completed trace {path} has {len(rows)} rows, expected {expected_count}\")",
                                                             "    if False and len(rows) != expected_count:\n"
                                                             "        raise ValueError(f\"completed trace {path} has {len(rows)} rows, expected {expected_count}\")")]),

    # Round 4 (Sol, REVIEW-r4-sol.md table A-H): seven of eight passed the trace audit. The observer audit must kill all.
    "R4A_rng_always_85000": ("part85.py", [("                    torch.manual_seed(seed)\n",
                                             "                    torch.manual_seed(85000)\n")]),
    "R4B_decode_prompt_plus_answer": ("part85.py", [("answer = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()",
                                                      "answer = tokenizer.decode(generated[0, :], skip_special_tokens=True).strip()")]),
    "R4C_chat_without_generation_prompt": ("part85.py", [("            tokenize=True, add_generation_prompt=True, return_tensors=\"pt\",",
                                                           "            tokenize=True, add_generation_prompt=False, return_tensors=\"pt\",")]),
    "R4D_chat_role_assistant": ("part85.py", [("            [{\"role\": \"user\", \"content\": question}],",
                                                "            [{\"role\": \"assistant\", \"content\": question}],")]),
    "R4E_temperature_one_at_call_trace_unchanged": ("part85.py", [("                            **encoded, **generate_kwargs,",
                                                                    "                            **encoded, **{**generate_kwargs, \"temperature\": 1.0},")]),
    "R4F_zero_channel0_in_place_before_clone": ("part85.py", [("        y = x.clone()\n        y[..., self.unit] = self.value",
                                                               "        x[..., 0] = 0\n        y = x.clone()\n        y[..., self.unit] = self.value")]),
    "R4G_no_hook_sequence_reset": ("part85.py", [("                    hook.begin_sequence()\n", "")]),
    "R4H_step1_calibrates_layer9": ("step1_acts.py", [("    down = layers[10].mlp.down_proj", "    down = layers[9].mlp.down_proj")]),
    "R4_answer_without_observer_call": ("part85.py", [("\"observer_call_id\": obs.last_call_id}", "\"observer_call_id\": None}")]),
}


def build_mutant(name: str, spec, destination: Path) -> Path:
    mutant_root = destination / "part85"
    shutil.copytree(PART85, mutant_root,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    # observe85 imports the repository-level observe.py, and the tests read neuron_report beside part85.
    for sibling in ("observe.py", "neuron_report.py"):
        if (PART85.parent / sibling).exists():
            shutil.copy2(PART85.parent / sibling, destination / sibling)
    filename, edits = spec
    path = mutant_root / filename
    if edits == "MOVE_PUSH_EARLY":
        content = push_early_before_answer(path.read_text(encoding="utf-8"))
    elif edits == "STEP0_EARLY":
        content = step0_push_early(path.read_text(encoding="utf-8"))
    else:
        content = path.read_text(encoding="utf-8")
        for old, new in edits:
            content = replace_once(content, old, new, name)
    path.write_text(content, encoding="utf-8")
    return mutant_root


def test_result(cwd: Path, timeout: int = TEST_TIMEOUT_SECONDS):
    env = os.environ.copy()
    env.update(TEST_THREAD_LIMITS)
    try:
        result = subprocess.run(TEST_COMMAND, cwd=cwd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return {"returncode": None, "output": output + "\nTEST TIMEOUT", "tests": None, "skipped": None}
    match = re.search(r"^Ran (\d+) tests? in ", result.stdout, flags=re.MULTILINE)
    skipped = len(re.findall(r"^test_.* \.\.\. skipped", result.stdout, flags=re.MULTILINE))
    return {"returncode": result.returncode, "output": result.stdout,
            "tests": int(match.group(1)) if match else None, "skipped": skipped}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2, help="mutations to run concurrently (default: 2)")
    parser.add_argument("--timeout", type=int, default=TEST_TIMEOUT_SECONDS,
                        help="per-test-suite timeout in seconds (default: 600)")
    parser.add_argument("--only", nargs="*", choices=tuple(MUTANTS),
                        help="optional subset of named mutants for local diagnosis")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    if not PYTHON.is_file():
        print(f"required Python interpreter does not exist: {PYTHON}", file=sys.stderr)
        return 2

    baseline = test_result(PART85, args.timeout)
    if baseline["returncode"] != 0 or baseline["tests"] is None or baseline["skipped"]:
        print("BASELINE FAILED, had no test count, or skipped tests; mutation run stopped")
        print(baseline["output"])
        return 2
    wanted = args.only or list(MUTANTS)

    def run_one(name):
        try:
            with tempfile.TemporaryDirectory(prefix=f"part85-mut-{name}-") as temp:
                root = build_mutant(name, MUTANTS[name], Path(temp))
                result = test_result(root, args.timeout)
                if result["tests"] != baseline["tests"] or result["skipped"] != 0:
                    status = "ERROR"
                elif result["returncode"] == 0:
                    status = "SURVIVED"
                elif "FAILED" in result["output"]:
                    status = "KILLED"
                else:
                    status = "ERROR"
                summary = re.findall(r"^Ran \d+ tests? in [^\n]+", result["output"], flags=re.MULTILINE)
                return {"name": name, "status": status,
                        "summary": summary[-1] if summary else "no test summary",
                        "tail": "\n".join(result["output"].splitlines()[-16:])}
        except Exception as exc:
            return {"name": name, "status": "ERROR", "summary": type(exc).__name__, "tail": str(exc)}

    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run_one, name): name for name in wanted}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: wanted.index(result["name"]))
    print(f"Baseline: Ran {baseline['tests']} tests; skipped 0; OK")
    for result in results:
        print(f"{result['name']}: {result['status']} ({result['summary']})")
        if result["status"] != "KILLED":
            print(result["tail"])
    killed = sum(result["status"] == "KILLED" for result in results)
    print(f"Mutation summary: {killed}/{len(results)} killed; {len(results) - killed} survived or errored")
    return 0 if len(results) == len(wanted) and killed == len(wanted) else 1


if __name__ == "__main__":
    raise SystemExit(main())
