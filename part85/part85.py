#!/usr/bin/env python3
"""Part 85 step 2: paired plain-frame and chat-frame clamp generations."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoints import (BY_KEY, CHECKPOINTS, PILOT_KEYS, POST_TRAINED,
                         resolve_checkpoint, validate_snapshot_commit)
from handoff85 import code_commit, result_dir

QUESTIONS = ("Who made you?", "Where do you come from?", "Tell me about your origins.")
CELL_NAMES = ("unclamped", "own_max", "absolute", "control")
PILOT_CELL_NAMES = ("unclamped", "own_max")
SAMPLES_PER_QUESTION = 16
PILOT_SAMPLES_PER_QUESTION = 8
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.7
TOP_P = 0.95
TARGET_UNIT = 6978
CONTROL_UNIT = 7884
ABSOLUTE_VALUE = 9.75
PLAIN_TEMPLATE = "Question: {q}\nAnswer:"


def seed_for(question_index: int, sample_index: int) -> int:
    if not 0 <= question_index < len(QUESTIONS):
        raise ValueError(f"question index outside [0, {len(QUESTIONS)}): {question_index}")
    if sample_index < 0:
        raise ValueError("sample index must be nonnegative")
    return 85000 + 100 * question_index + sample_index


def expected_cell_specs(mode: str):
    """Enumerate checkpoint/frame/cell/sample counts without loading a model."""
    if mode == "pilot":
        return [
            {"checkpoint": key, "frame": "plain", "cell": cell,
             "samples_per_question": PILOT_SAMPLES_PER_QUESTION}
            for key in PILOT_KEYS for cell in PILOT_CELL_NAMES
        ]
    if mode != "main":
        raise ValueError("mode must be 'pilot' or 'main'")
    specs = []
    for cp in CHECKPOINTS:
        for cell in CELL_NAMES:
            specs.append({"checkpoint": cp.key, "frame": "plain", "cell": cell,
                          "samples_per_question": SAMPLES_PER_QUESTION})
            if cp.key in POST_TRAINED:
                specs.append({"checkpoint": cp.key, "frame": "chat", "cell": cell,
                              "samples_per_question": SAMPLES_PER_QUESTION})
    return specs


def expected_answer_counts():
    pilot = sum(len(QUESTIONS) * row["samples_per_question"] for row in expected_cell_specs("pilot"))
    main_specs = expected_cell_specs("main")
    main_plain = sum(len(QUESTIONS) * SAMPLES_PER_QUESTION for r in main_specs if r["frame"] == "plain")
    main_chat = sum(len(QUESTIONS) * SAMPLES_PER_QUESTION for r in main_specs if r["frame"] == "chat")
    return {"pilot": pilot, "main_plain": main_plain, "main_chat": main_chat,
            "main_total": main_plain + main_chat}


class BothPhaseClampHook:
    """Observer and Part 62-style both-phase assignment hook on down_proj's input."""

    def __init__(self, down_proj, unit: int | None, value: float | None):
        self.unit = unit
        self.value = value
        self.counts = {"prompt_calls": 0, "prompt_positions": 0,
                       "generated_calls": 0, "generated_positions": 0}
        self._prompt_pending = True
        self.last_phase = None
        self.other_channels_changed = False
        self.handle = down_proj.register_forward_pre_hook(self._hook)

    def begin_sequence(self):
        self._prompt_pending = True
        self.last_phase = None
        self.other_channels_changed = False

    def _hook(self, _module, args):
        import torch

        x = args[0]
        prompt = self._prompt_pending
        self._prompt_pending = False
        prefix = "prompt" if prompt else "generated"
        self.last_phase = prefix
        self.counts[prefix + "_calls"] += 1
        self.counts[prefix + "_positions"] += int(x.shape[-2])
        if self.unit is None:
            return None
        y = x.clone()
        y[..., self.unit] = self.value
        changed_elsewhere = False
        if self.unit:
            changed_elsewhere |= not torch.equal(x[..., :self.unit], y[..., :self.unit])
        if self.unit + 1 < x.shape[-1]:
            changed_elsewhere |= not torch.equal(x[..., self.unit + 1:], y[..., self.unit + 1:])
        self.other_channels_changed |= changed_elsewhere
        return (y,) + tuple(args[1:])

    def remove(self):
        self.handle.remove()


class BothPhaseTraceObserver:
    """Measure the hook's input after the clamp pre-hook has rewritten it."""

    def __init__(self, down_proj, clamp_hook):
        self.clamp_hook = clamp_hook
        self.unit = clamp_hook.unit if clamp_hook.unit is not None else TARGET_UNIT
        self.handle = down_proj.register_forward_pre_hook(self._hook)
        self.begin_sequence()

    def begin_sequence(self):
        self.prompt_positions_set = 0
        self.generated_positions_set = 0
        self.observed_min = None
        self.observed_max = None

    def _hook(self, _module, args):
        import torch

        values = args[0][..., self.unit].detach().float()
        low, high = float(values.min().item()), float(values.max().item())
        self.observed_min = low if self.observed_min is None else min(self.observed_min, low)
        self.observed_max = high if self.observed_max is None else max(self.observed_max, high)
        if self.clamp_hook.unit is None:
            return None
        expected = torch.as_tensor(self.clamp_hook.value, dtype=values.dtype, device=values.device)
        matches = torch.isclose(values, expected, rtol=1e-6, atol=0)
        count = int(matches.sum().item())
        if self.clamp_hook.last_phase == "prompt":
            self.prompt_positions_set += count
        elif self.clamp_hook.last_phase == "generated":
            self.generated_positions_set += count
        return None

    def row(self):
        if self.observed_min is None or self.observed_max is None:
            raise RuntimeError("trace observer did not see the answer's down_proj input")
        return {
            "prompt_positions_set": self.prompt_positions_set,
            "generated_positions_set": self.generated_positions_set,
            "observed_min": self.observed_min,
            "observed_max": self.observed_max,
        }

    def remove(self):
        self.handle.remove()


def _load_plain_template(path=None):
    template = Path(path).read_text(encoding="utf-8") if path else PLAIN_TEMPLATE
    if template.count("{q}") != 1:
        raise ValueError("plain frame template must contain exactly one {q} placeholder")
    return template


def _template_sha256(template):
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _prompt_inputs(tokenizer, question: str, frame: str, device: str,
                   plain_template=PLAIN_TEMPLATE, trace_state=None):
    if trace_state is not None:
        trace_state.update({"question": question, "chat_template_applied": False})
    if frame == "plain":
        text = plain_template.replace("{q}", question)
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    elif frame == "chat":
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError("chat frame requested but this tokenizer has no chat_template")
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        if trace_state is not None:
            trace_state["chat_template_applied"] = True
        if not hasattr(encoded, "items"):
            encoded = {"input_ids": encoded}
    else:
        raise ValueError(f"unknown frame {frame!r}")
    return {k: v.to(device) for k, v in encoded.items() if hasattr(v, "to")}


def _eos_ids(tokenizer, model):
    vals = [tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
            getattr(model.generation_config, "eos_token_id", None)]
    out = set()
    unknown = getattr(tokenizer, "unk_token_id", None)
    for value in vals:
        if isinstance(value, int) and value >= 0:
            if value != unknown:
                out.add(value)
        elif isinstance(value, (list, tuple)):
            out.update(int(x) for x in value if isinstance(x, int) and x >= 0 and x != unknown)
    return sorted(out) or None


def _atomic_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _atomic_jsonl_pair(first_path: Path, first_rows, second_path: Path, second_rows):
    """Stage both complete files before atomically replacing either destination."""
    staged = []
    try:
        for path, rows in ((first_path, first_rows), (second_path, second_rows)):
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            staged.append((Path(tmp), path))
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        for tmp, path in staged:
            os.replace(tmp, path)
    except Exception:
        for tmp, _path in staged:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        raise


def _cell_value(cell: str, step1: dict):
    maxima = step1.get("neutral_activation_maxima")
    if not isinstance(maxima, dict):
        raise ValueError("step1 JSON is missing neutral_activation_maxima")
    if cell == "unclamped":
        return None, None
    if cell == "own_max":
        if str(TARGET_UNIT) not in maxima:
            raise ValueError("step1 JSON is missing target unit 6978 maximum")
        return TARGET_UNIT, 1.5 * float(maxima[str(TARGET_UNIT)])
    if cell == "absolute":
        return TARGET_UNIT, ABSOLUTE_VALUE
    if cell == "control":
        if str(CONTROL_UNIT) not in maxima:
            raise ValueError("step1 JSON is missing control unit 7884 maximum")
        return CONTROL_UNIT, 1.5 * float(maxima[str(CONTROL_UNIT)])
    raise ValueError(f"unknown cell {cell!r}")


def _step1_guard_sha256(step1: dict) -> str:
    """Hash only registered clamp inputs and checkpoint identity, not run metadata."""
    checkpoint = step1.get("checkpoint") or {}
    maxima = step1.get("neutral_activation_maxima")
    token_sha = step1.get("neutral_token_ids_sha256")
    if not isinstance(maxima, dict) or not token_sha or not checkpoint.get("commit"):
        raise ValueError("step1 JSON lacks maxima, neutral token SHA-256, or checkpoint commit")
    guarded = {"maxima": maxima, "neutral_token_ids_sha256": token_sha,
               "commit": checkpoint["commit"]}
    raw = json.dumps(guarded, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _versions():
    out = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "numpy"):
        try:
            out[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            out[package] = None
    return out


def _pilot_scope(args):
    return {"action": "pilot_not_scored", "required": "score the pilot, then rerun main with --pilot-score"}


def _main_scope(args):
    if not args.pilot_score:
        raise ValueError("main mode requires a scored pilot JSON via --pilot-score")
    payload = json.loads(Path(args.pilot_score).read_text(encoding="utf-8"))
    gate = payload.get("pilot_gate")
    if not isinstance(gate, dict):
        raise ValueError("pilot score JSON has no pilot_gate decision")
    action = gate.get("action")
    template_sha = _template_sha256(_load_plain_template(args.plain_template_file))
    if gate.get("plain_template_sha256") != template_sha:
        raise ValueError("main plain frame template does not match the pilot template that passed")
    if action == "plain_frame_pass":
        return "all", ("plain", "chat")
    if action == "chat_only_posttrained":
        return "posttrained", ("chat",)
    raise RuntimeError(f"pilot gate does not authorize the main run: {action!r}")


def _specs_for_run(args):
    if args.mode == "pilot":
        keys = PILOT_KEYS
        frames = ("plain",)
        cells = PILOT_CELL_NAMES
        per_q = PILOT_SAMPLES_PER_QUESTION
    else:
        scope, frames = _main_scope(args)
        keys = POST_TRAINED if scope == "posttrained" else tuple(cp.key for cp in CHECKPOINTS)
        cells = CELL_NAMES
        per_q = SAMPLES_PER_QUESTION
    if args.checkpoint not in keys:
        return []
    return [{"checkpoint": args.checkpoint, "frame": frame, "cell": cell,
             "samples_per_question": per_q} for frame in frames
            if frame != "chat" or args.checkpoint in POST_TRAINED for cell in cells]


def _load_completed(path: Path, expected_count: int, cp_key: str, frame: str, cell: str,
                    template_sha: str, step1: dict):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"completed cell {path} has {len(rows)} answers, expected {expected_count}")
    if any((r.get("checkpoint"), r.get("frame"), r.get("cell")) != (cp_key, frame, cell) for r in rows):
        raise ValueError(f"completed cell {path} contains mismatched checkpoint/frame/cell metadata")
    if any(r.get("frame_template_sha256") != template_sha for r in rows):
        raise ValueError(f"completed cell {path} was generated with a different plain-frame template")
    expected_unit, expected_value = _cell_value(cell, step1)
    for row in rows:
        if row.get("unit") != expected_unit:
            raise ValueError(f"completed cell {path} uses unit {row.get('unit')!r}; current step1 requires {expected_unit!r}")
        actual = row.get("clamp_value")
        same_value = actual is None and expected_value is None
        if actual is not None and expected_value is not None:
            try:
                same_value = math.isclose(float(actual), float(expected_value), rel_tol=1e-12, abs_tol=1e-12)
            except (TypeError, ValueError):
                same_value = False
        if not same_value:
            raise ValueError(f"completed cell {path} clamp_value {actual!r} differs from current step1 value {expected_value!r}")
    return rows


def _load_completed_trace(path: Path, answers, expected_count: int, cp_key: str,
                          frame: str, cell: str, mode: str, attempt: int):
    if not path.is_file():
        raise FileNotFoundError(f"completed cell is missing its trace file: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_count:
        raise ValueError(f"completed trace {path} has {len(rows)} rows, expected {expected_count}")
    for answer, trace in zip(answers, rows):
        expected_identity = (cp_key, mode, attempt, frame, cell,
                             answer.get("question_index"), answer.get("sample"), answer.get("seed"))
        actual_identity = (trace.get("checkpoint"), trace.get("mode"), trace.get("attempt"),
                           trace.get("frame"), trace.get("cell"), trace.get("question_index"),
                           trace.get("sample"), trace.get("seed"))
        if actual_identity != expected_identity:
            raise ValueError(f"completed trace {path} disagrees with its answer row")
    return rows


def _hook_counts_ok(counts, answers) -> bool:
    """Prompt positions must always be hooked. Generated positions may be zero only when every answer ended on an
    immediate EOS (one new token, produced by the prefill forward, so no decode forward ever ran): round-4 #6."""
    if not counts or counts.get("prompt_positions", 0) <= 0:
        return False
    if counts.get("generated_positions", 0) > 0:
        return True
    return all(int(row.get("new_tokens", 0)) <= 1 for row in answers)


def _push_checkpoint_results(args, checkpoint):
    if not getattr(args, "push_branch", None):
        return
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).with_name("handoff85.py")), "push",
                    "--checkpoint", checkpoint, "--branch", args.push_branch,
                    "--results-root", args.results_root], check=True)


def run(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    plain_template = _load_plain_template(args.plain_template_file)
    if args.mode == "pilot" and args.attempt not in (1, 2):
        raise ValueError("pilot attempt must be 1 or the single registered rerun, 2")
    if args.mode == "main" and args.attempt != 1:
        raise ValueError("--attempt is only used by pilot mode")
    specs = _specs_for_run(args)
    if not specs:
        raise ValueError(f"checkpoint {args.checkpoint} is outside the {args.mode} run scope")
    cp = resolve_checkpoint(args.checkpoint)
    root = Path(args.results_root)
    expected_branch = (f"part85-run/pilot-a{args.attempt}" if args.mode == "pilot"
                       else f"part85-run/main-{cp['key']}")
    if args.push_branch and args.push_branch != expected_branch:
        raise ValueError(f"{args.mode} {cp['key']} must push to {expected_branch}")
    cp_root = result_dir(root, cp["key"], branch=args.push_branch, mode=args.mode, attempt=args.attempt)
    current_code_commit = code_commit()
    mode_tag = f"pilot{args.attempt}" if args.mode == "pilot" else "main"
    step1_path = Path(args.step1) if args.step1 else cp_root / "step1.json"
    if not step1_path.exists():
        raise FileNotFoundError(f"step 1 is required before step 2: {step1_path}")
    step1 = json.loads(step1_path.read_text(encoding="utf-8"))
    if step1.get("checkpoint", {}).get("commit") != cp["commit"]:
        raise ValueError("step1 checkpoint commit does not match the resolved checkpoint")
    step1_sha256 = _step1_guard_sha256(step1)

    run_path = cp_root / "run.json"
    run_record = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else {}
    previous_step1_sha256 = run_record.get("step1_guard_sha256", run_record.get("step1_sha256"))
    if previous_step1_sha256 and previous_step1_sha256 != step1_sha256:
        raise ValueError("step1 clamp inputs changed since this run recorded them; use a fresh results root")
    if run_record.get("code_commit") and run_record["code_commit"] != current_code_commit:
        raise ValueError("code commit changed since this run recorded it; use a fresh results root")
    pilot_gate_action = None
    if args.mode == "main":
        pilot_payload = json.loads(Path(args.pilot_score).read_text(encoding="utf-8"))
        pilot_gate_action = pilot_payload.get("pilot_gate", {}).get("action")
    template_runs = run_record.get("plain_frame_templates", {})
    template_runs[mode_tag] = {"template": plain_template, "sha256": _template_sha256(plain_template)}
    run_record.update({
        "checkpoint": cp,
        "code_commit": current_code_commit,
        "model_id": cp["repo_id"],
        "full_commit": cp["commit"],
        "dtype": run_record.get("dtype"),
        "device": args.device,
        "library_versions": _versions(),
        "dataset_revision": step1.get("dataset_revision"),
        "step1_guard_sha256": step1_sha256,
        "pilot_gate_action": pilot_gate_action or run_record.get("pilot_gate_action"),
        "sampling": {"temperature": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS,
                     "plain_frame_samples_per_question": SAMPLES_PER_QUESTION,
                     "pilot_samples_per_question": PILOT_SAMPLES_PER_QUESTION},
        "questions": list(QUESTIONS),
        "plain_frame_template": plain_template,
        "plain_frame_template_sha256": _template_sha256(plain_template),
        "plain_frame_templates": template_runs,
        "hook_fire_counts": run_record.get("hook_fire_counts", {}),
        "completed_cells": run_record.get("completed_cells", {}),
    })
    # Do not load a 1B model again when all requested cells have already been checked and completed.
    all_done = True
    for spec in specs:
        cell_key = f"{mode_tag}/{spec['frame']}/{spec['cell']}"
        answer_path = cp_root / "answers" / f"{mode_tag}-{spec['frame']}--{spec['cell']}.jsonl"
        trace_path = cp_root / "trace" / answer_path.name
        hook_counts = run_record["hook_fire_counts"].get(cell_key)
        if not answer_path.exists() or not trace_path.exists() or not hook_counts:
            all_done = False
            break
        expected = len(QUESTIONS) * spec["samples_per_question"]
        completed_answers = _load_completed(answer_path, expected, cp["key"], spec["frame"],
                                            spec["cell"], _template_sha256(plain_template), step1)
        _load_completed_trace(trace_path, completed_answers, expected, cp["key"], spec["frame"],
                              spec["cell"], args.mode, args.attempt)
        if not _hook_counts_ok(hook_counts, completed_answers):
            raise ValueError(f"completed cell {cell_key} lacks prompt or generated hook fires")
    if all_done:
        _push_checkpoint_results(args, cp["key"])
        print(f"all {len(specs)} cells already complete for {cp['key']}")
        return

    snapshot_dir = args.snapshot_dir
    if snapshot_dir is not None:
        validate_snapshot_commit(snapshot_dir, cp["commit"])
    if snapshot_dir is None:
        from huggingface_hub import snapshot_download
        snapshot_dir = snapshot_download(repo_id=cp["repo_id"], revision=cp["commit"])
        validate_snapshot_commit(snapshot_dir, cp["commit"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir)
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir, dtype=torch.float32)
    model.to(args.device).eval().requires_grad_(False)
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.is_floating_point()})
    run_record["dtype"] = str(next(model.parameters()).dtype)
    run_record["parameter_dtypes"] = parameter_dtypes
    run_record["effective_generation_config"] = model.generation_config.to_dict()
    torch.set_grad_enabled(False)
    best_layers = None
    for _name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) and hasattr(getattr(module[0], "mlp", None), "down_proj"):
            rank = ("language" in _name, len(module))
            if best_layers is None or rank > best_layers[0]:
                best_layers = (rank, module)
    layers = best_layers[1] if best_layers else None
    if layers is None or len(layers) <= 10:
        raise RuntimeError("could not find OLMo text layers through layer 10")
    down_proj = layers[10].mlp.down_proj
    if down_proj.in_features <= max(TARGET_UNIT, CONTROL_UNIT):
        raise RuntimeError(f"layer 10 down_proj has only {down_proj.in_features} units")
    eos_ids = _eos_ids(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError("tokenizer has no pad or EOS token")
    # The independent observer (observe85, over the shared observe.py) records what generate actually received and
    # returned; audit85 reads only its records (decision part85-after-r4-2026-10-28).
    import observe85
    obs = observe85.install(model, tokenizer, cp_root / "observe" / f"{mode_tag}.jsonl",
                            label=f"part85 step2 {cp['key']} {mode_tag}")
    try:
        _run_cells(args, specs, cp, cp_root, root, run_record, run_path, step1, plain_template, mode_tag,
                   pilot_gate_action, current_code_commit, model, tokenizer, down_proj, eos_ids, pad_id, obs)
    finally:
        obs.close()


def _run_cells(args, specs, cp, cp_root, root, run_record, run_path, step1, plain_template, mode_tag,
               pilot_gate_action, current_code_commit, model, tokenizer, down_proj, eos_ids, pad_id, obs):
    import torch

    for spec in specs:
        frame, cell = spec["frame"], spec["cell"]
        cell_key = f"{mode_tag}/{frame}/{cell}"
        answer_path = cp_root / "answers" / f"{mode_tag}-{frame}--{cell}.jsonl"
        trace_path = cp_root / "trace" / answer_path.name
        expected = len(QUESTIONS) * spec["samples_per_question"]
        if answer_path.exists() and trace_path.exists() and run_record["hook_fire_counts"].get(cell_key):
            completed_answers = _load_completed(answer_path, expected, cp["key"], frame, cell,
                                                _template_sha256(plain_template), step1)
            _load_completed_trace(trace_path, completed_answers, expected, cp["key"], frame,
                                  cell, args.mode, args.attempt)
            counts = run_record["hook_fire_counts"][cell_key]
            if _hook_counts_ok(counts, completed_answers):
                print(f"skip completed {cp['key']} {cell_key}")
                continue
            raise ValueError(f"completed cell {cell_key} has no hook fire in both phases")
        unit, value = _cell_value(cell, step1)
        hook = BothPhaseClampHook(down_proj, unit, value)
        hooked_module = next(name for name, module in model.named_modules() if module is down_proj)
        observer = BothPhaseTraceObserver(down_proj, hook)
        rows = []
        trace_rows = []
        started = time.time()
        try:
            for qi, question in enumerate(QUESTIONS):
                prompt_state = {}
                encoded = _prompt_inputs(tokenizer, question, frame, args.device, plain_template,
                                         trace_state=prompt_state)
                prompt_length = int(encoded["input_ids"].shape[-1])
                for sample in range(spec["samples_per_question"]):
                    seed = seed_for(qi, sample)
                    hook.begin_sequence()
                    observer.begin_sequence()
                    torch.manual_seed(seed)
                    if args.device == "cuda":
                        torch.cuda.manual_seed_all(seed)
                    elif args.device == "xpu" and hasattr(torch, "xpu"):
                        torch.xpu.manual_seed_all(seed)
                    prompt_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
                    generate_kwargs = {
                        "do_sample": True,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "max_new_tokens": MAX_NEW_TOKENS,
                    }
                    with torch.inference_mode():
                        generated = model.generate(
                            **encoded, **generate_kwargs,
                            eos_token_id=eos_ids, pad_token_id=pad_id,
                            use_cache=True,
                        )
                    new_tokens = int(generated.shape[-1] - prompt_length)
                    answer = tokenizer.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
                    observed = observer.row()
                    rows.append({"mode": args.mode, "attempt": args.attempt, "checkpoint": cp["key"], "model_id": cp["repo_id"], "commit": cp["commit"],
                                 "code_commit": current_code_commit,
                                 "frame": frame, "cell": cell, "unit": unit, "clamp_value": value,
                                 "frame_template_sha256": _template_sha256(plain_template),
                                 "pilot_gate_action": pilot_gate_action,
                                 "question": question, "question_index": qi, "sample": sample,
                                 "seed": seed, "answer": answer, "new_tokens": new_tokens,
                                 "observer_call_id": obs.last_call_id})
                    trace_rows.append({
                        "checkpoint": cp["key"], "mode": args.mode, "attempt": args.attempt,
                        "frame": frame, "cell": cell, "question_index": qi, "sample": sample,
                        "seed": seed,
                        "question": prompt_state["question"],
                        "prompt_text": prompt_text,
                        "chat_template_applied": prompt_state["chat_template_applied"],
                        "generate_kwargs": {key: generate_kwargs[key] for key in
                                            ("do_sample", "temperature", "top_p", "max_new_tokens")},
                        "hooked_module": hooked_module,
                        "unit": hook.unit,
                        "set_value": hook.value,
                        **observed,
                        "other_channels_changed": hook.other_channels_changed,
                    })
        finally:
            counts = dict(hook.counts)
            observer.remove()
            hook.remove()
        if len(rows) != expected:
            raise RuntimeError(f"generated {len(rows)} answers for {cell_key}, expected {expected}")
        if not _hook_counts_ok(counts, rows):
            raise RuntimeError(f"hook must fire on prompt and generated positions for {cell_key}: {counts}")
        _atomic_jsonl_pair(answer_path, rows, trace_path, trace_rows)
        run_record["hook_fire_counts"][cell_key] = counts
        run_record["completed_cells"][cell_key] = {
            "answer_file": (Path("results/part85") / answer_path.relative_to(root)).as_posix(), "answer_count": expected,
            "elapsed_seconds": time.time() - started, "cell": cell,
            "frame": frame, "unit": unit, "clamp_value": value,
        }
        run_record["mode"] = args.mode
        run_record["updated_unix_time"] = time.time()
        _atomic_json(run_path, run_record)
        _push_checkpoint_results(args, cp["key"])
        print(f"wrote {expected} answers for {cp['key']} {cell_key}; hook {counts}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("pilot", "main"))
    parser.add_argument("--checkpoint", required=True, choices=tuple(BY_KEY))
    parser.add_argument("--pilot-score", help="score.json produced by score85.py for the pilot")
    parser.add_argument("--attempt", type=int, default=1, help="pilot attempt number; use 2 for the one registered rerun")
    parser.add_argument("--plain-template-file", help="optional written plain-frame template containing one {q} placeholder")
    parser.add_argument("--step1", help="step1.json path; default results-root/<checkpoint>/step1.json")
    parser.add_argument("--snapshot-dir", help="already-downloaded pinned checkpoint snapshot directory")
    parser.add_argument("--results-root", default="results/part85")
    parser.add_argument("--push-branch", help="push after every completed answer cell")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "xpu"))
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
