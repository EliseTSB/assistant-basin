#!/usr/bin/env python3
"""Tiny inert checks for Part 198 run, control, gate, scoring and resume helpers."""
from __future__ import annotations

import json
import hashlib
import fcntl
import io
import sys
import tempfile
import time
import warnings
import inspect
import multiprocessing
import queue
import signal
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch


def _blocked_model_loader(*_args, **_kwargs):
    raise AssertionError("model or weight loading is forbidden in synthetic tests")


# Fail closed if a runner regression reaches any real model/weight API during this suite.
torch.load = _blocked_model_loader
try:
    import transformers
    transformers.AutoModelForCausalLM.from_pretrained = _blocked_model_loader
    transformers.AutoTokenizer.from_pretrained = _blocked_model_loader
except ImportError:
    pass
try:
    import safetensors
    safetensors.safe_open = _blocked_model_loader
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run198 as R  # noqa: E402
import score198 as S  # noqa: E402

# Keep runner-level loader entry points blocked even if a synthetic test reaches an arm path.
for _loader_name in ("load_gpt2_model", "load_arm_c_pair"):
    setattr(R, _loader_name, _blocked_model_loader)

fails = 0
check_names = []


def check(name: str, condition: bool) -> None:
    global fails
    check_names.append(name)
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        fails += 1


def case_replay_conflicts(path: tuple[str, ...], replacement: Any = True) -> bool:
    base = {"record_type": "case", "run_mode": "registered", "run_id": "r1", "case_id": 198,
            "took": True, "already_satisfied": False, "name_eligible": True,
            "restored_bit_exact": True, "undo_zero_exact": True,
            "localization": {stat: {"localized": False} for stat in ("f1", "pcs", "t_word")},
            "aimed_positive_control": {"oracle_named_strict": False,
                                         "localization": {"value": 2.0, "localized": False,
                                                          "tie": False}},
            "oracle_reader": {"named_strict": False, "named_loose": False},
            "blind_reader": {"localized": False, "named_strict": False},
            "random_control": {"localization": {"t_word": {"localized": False}}},
            "edited_layer_stats": {"17": {"update_top_names_strict": False,
                                           "update_top_names_loose": False}}}
    changed = json.loads(json.dumps(base))
    value = changed
    for component in path[:-1]:
        value = value[component]
    value[path[-1]] = replacement
    with tempfile.TemporaryDirectory(prefix="part198-scored-replay-") as td:
        log = Path(td) / "armA-rome.jsonl"
        R.append_jsonl(log, base)
        changed["run_id"] = "r2"
        R.append_jsonl(log, changed)
        try:
            S.registered_rows(log, "case", "case_id")
        except ValueError as exc:
            return (f"case_id={str(base['case_id'])!r}" in str(exc)
                    and ".".join(path) in str(exc))
    return False


def batch_replay_conflicts() -> bool:
    base = {"record_type": "batch", "run_mode": "registered", "run_id": "r1",
            "batch_id": "b1", "n_took": 8, "n_already_satisfied": 0,
            "take_gate_passed": True, "restored_bit_exact": True,
            "localization": {"f1": {"localized": False}, "pcs": {"localized": False}}}
    changed = json.loads(json.dumps(base))
    changed["run_id"] = "r2"
    changed["localization"]["f1"]["localized"] = True
    with tempfile.TemporaryDirectory(prefix="part198-batch-replay-") as td:
        log = Path(td) / "armA-mass-memit.jsonl"
        R.append_jsonl(log, base)
        R.append_jsonl(log, changed)
        try:
            S.registered_rows(log, "batch", "batch_id")
        except ValueError as exc:
            return ("batch_id='b1'" in str(exc)
                    and "localization.f1.localized" in str(exc))
    return False


def metadata_replay_counts_unchanged() -> bool:
    base = {"record_type": "case", "run_mode": "registered", "run_id": "r1",
            "timestamp": "t1", "case_id": 198, "took": True,
            "already_satisfied": False, "name_eligible": True,
            "restored_bit_exact": True, "localization": {"f1": {"localized": True}}}
    replay = dict(base, run_id="r2", timestamp="t2")
    with tempfile.TemporaryDirectory(prefix="part198-metadata-replay-") as td:
        path = Path(td) / "armA-rome.jsonl"
        R.append_jsonl(path, base)
        one = S.score_from_dir(Path(td))
        R.append_jsonl(path, replay)
        two = S.score_from_dir(Path(td))
        def count_tuple(result):
            metric = next(row for row in result["forecasts"]
                          if row["forecast"] == "198.1 ROME single")
            return tuple(metric[name] for name in ("attempted", "took", "successes", "rate_denominator"))
        return count_tuple(one) == count_tuple(two) == (1, 1, 1, 1)


def _hold_runner_log_lock(lock_path: Path, ready, release) -> None:
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=15)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _hold_runner_context_lock(log_path: Path, ready) -> None:
    with R.output_log_lock(log_path):
        ready.set()
        while True:
            time.sleep(0.05)


def _invoke_runner_while_locked(out_dir: Path, result_queue) -> None:
    calls = []
    R.root_out_dir = lambda _out_dir: out_dir
    def blocked_preflight(_out_dir):
        calls.append("preflight")
        raise RuntimeError("test stopped before any arm")
    R.preflight = blocked_preflight
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = R.main(["armA", "--method", "ROME", "--out-dir", str(out_dir)])
    result_queue.put({"code": code, "calls": calls, "stdout": stdout.getvalue(),
                      "stderr": stderr.getvalue()})


def runner_refuses_held_log() -> bool:
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="part198-output-lock-") as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        log_path = out_dir / "armA-rome.jsonl"
        lock_path = log_path.with_name(log_path.name + ".lock")
        ready, release = ctx.Event(), ctx.Event()
        results = ctx.Queue()
        holder = ctx.Process(target=_hold_runner_log_lock, args=(lock_path, ready, release))
        holder.start()
        try:
            if not ready.wait(timeout=5):
                return False
            runner = ctx.Process(target=_invoke_runner_while_locked, args=(out_dir, results))
            runner.start()
            runner.join(timeout=10)
            if runner.is_alive():
                runner.terminate()
                runner.join(timeout=5)
                return False
            if runner.exitcode != 0:
                return False
            try:
                result = results.get(timeout=2)
            except queue.Empty:
                return False
            return (result["code"] == 1 and result["calls"] == []
                    and "already locked" in result["stderr"])
        finally:
            release.set()
            holder.join(timeout=5)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=5)


def runner_lock_releases_after_process_death() -> bool:
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="part198-lock-crash-") as td:
        log_path = Path(td) / "armA-rome.jsonl"
        ready = ctx.Event()
        holder = ctx.Process(target=_hold_runner_context_lock, args=(log_path, ready))
        holder.start()
        try:
            if not ready.wait(timeout=5):
                return False
            holder.terminate()
            holder.join(timeout=5)
            if holder.is_alive():
                return False
            try:
                with R.output_log_lock(log_path):
                    return True
            except R.OutputLogLocked:
                return False
        finally:
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=2)


def scorer_refuses_held_log() -> bool:
    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(prefix="part198-score-lock-") as td:
        out_dir = Path(td) / "out"
        out_dir.mkdir()
        log_path = out_dir / "armB-seq-wsp.jsonl"
        log_path.write_bytes(b'{"record_type":"case","case_id":1}\n')
        before = hashlib.sha256(log_path.read_bytes()).hexdigest()
        lock_path = log_path.with_name(log_path.name + ".lock")
        ready, release = ctx.Event(), ctx.Event()
        holder = ctx.Process(target=_hold_runner_log_lock, args=(lock_path, ready, release))
        holder.start()
        try:
            if not ready.wait(timeout=5):
                return False
            api_refused = False
            try:
                S.score_from_dir(out_dir)
            except S.ScoreLogLocked as exc:
                api_refused = "scoring refused" in str(exc) and "already locked" in str(exc)
            saved_root_out_dir = R.root_out_dir
            R.root_out_dir = lambda _path: out_dir
            stdout, stderr = io.StringIO(), io.StringIO()
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    cli_code = S.main(["--out-dir", str(out_dir)])
            finally:
                R.root_out_dir = saved_root_out_dir
            after = hashlib.sha256(log_path.read_bytes()).hexdigest()
            return (api_refused and cli_code == 1 and "REFUSED: scoring refused" in stderr.getvalue()
                    and before == after and not (out_dir / "score.json").exists())
        finally:
            release.set()
            holder.join(timeout=5)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=2)


def n4_terminal_append_is_safe(source: str) -> bool:
    undo_pos = source.find("undone = w.undo()")
    zero_pos = source.find("zero = all(", undo_pos)
    guard_pos = source.find("if not zero:", zero_pos)
    appends = []
    start = 0
    while True:
        position = source.find("append_jsonl(log, record)", start)
        if position < 0:
            break
        appends.append(position)
        start = position + 1
    return (len(appends) == 1 and 0 <= undo_pos < zero_pos < guard_pos < appends[0])


class TinyGPT2Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = torch.nn.Module()
        blocks = []
        for layer_index in range(18):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.c_proj = torch.nn.Module()
            if layer_index == 17:
                layer.mlp.c_proj.weight = torch.nn.Parameter(torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))
            blocks.append(layer)
        self.transformer.h = torch.nn.ModuleList(blocks)
        self.register_buffer("unreported", torch.tensor([5.0], dtype=torch.float32))


def run_fake_arm_a(out: Path, *, mass: bool = False, fingerprint_failure: bool = False):
    """Run the real Arm A restoration path with a tiny fake editor and model."""
    verifier = getattr(R, "verify_model_restored", None)
    if not callable(verifier):
        return {"supported": False, "calls": [], "rows": [], "error": None}
    out.mkdir(parents=True, exist_ok=True)
    log = out / ("armA-mass-memit.jsonl" if mass else "armA-memit.jsonl")
    model = TinyGPT2Projection()
    weight_name = "transformer.h.17.mlp.c_proj.weight"
    efficacy_calls = 0
    verification_calls = []

    def fake_apply(target_model, _tok, _reqs, _hp, **_kwargs):
        original = {weight_name: target_model.transformer.h[17].mlp.c_proj.weight.detach().clone()}
        with torch.no_grad():
            target_model.transformer.h[17].mlp.c_proj.weight.add_(0.25)
            target_model.unreported.add_(1.0)
        return target_model, original

    fake_module = SimpleNamespace(MEMITHyperParams=object, apply_memit_to_model=fake_apply)
    req = {"case_id": 101, "subject": "tiny subject", "target_new": "object",
           "ground_truth": "old", "prompt": "tiny prompt"}

    class FakeStats:
        read_pairs = staticmethod(lambda *args, **kwargs: [])
        top_eligible = staticmethod(lambda pairs: None)
        blind_reader = staticmethod(lambda pairs: (None, None))
        names = staticmethod(lambda pair, targets: (False, False))

    def fake_efficacy(*args, **kwargs):
        nonlocal efficacy_calls
        efficacy_calls += 1
        return {"logp_old": 1.0, "logp_new": 0.0} if efficacy_calls % 2 else {
            "logp_old": 0.0, "logp_new": 1.0}

    def fake_geometry(*args, **kwargs):
        return {"base": {17: {"f1": 1.0, "pairs": []}},
                "gain": torch.ones(2), "centre": False, "E": torch.ones((2, 2))}

    def tracked_verifier(target_model, snapshot, context):
        verification_calls.append(context)
        return verifier(target_model, snapshot, context)

    def fail_fingerprint(_model, _snapshot):
        raise RuntimeError("synthetic byte fingerprint failure")

    target_info = {"target_ids": [1], "eligible_target_ids": [1],
                   "name_eligible": True, "space_first_token_id": 1}
    patches = {
        "stats_helpers": lambda: FakeStats(),
        "load_cases": lambda: ({"armA_single": [101], "armA_mass": [[101]]}, {101: {"case_id": 101}}),
        "log_path_for": lambda *args: log,
        "load_gpt2_model": lambda: (model, SimpleNamespace(decode=lambda ids: "x")),
        "gpt2_base_inputs_sha256": lambda: "synthetic-input-hash",
        "gpt2_geometry": fake_geometry,
        "save_base": lambda *args: None,
        "load_easyedit": lambda: {"MEMIT": fake_module},
        "load_hparams": lambda *args: None,
        "request_of": lambda row: dict(req),
        "efficacy": fake_efficacy,
        "target_info_gpt2": lambda *args: dict(target_info),
        "stats_for": lambda *args: {"f1": 2.0, "f2": 1.0, "pcs": 0.0,
                                      "t_max": 2.0, "t_word": 2.0, "pairs": []},
        "update_metrics": lambda *args: {},
        "localization": lambda *args: {name: {"localized": False}
                                        for name in ("f1", "pcs", "t_word")},
        "pair_by_layer": lambda *args: {},
        "oracle_pair": lambda *args: (None, None, False),
        "base_diagnostics": lambda *args: {},
        "dict_to_pairs": lambda *args: [],
        "verify_model_restored": tracked_verifier,
    }
    if fingerprint_failure:
        patches["model_matches_snapshot"] = fail_fingerprint
    error = None
    with ExitStack() as stack:
        for name, value in patches.items():
            stack.enter_context(patch.object(R, name, value))
        try:
            if mass:
                R.run_arm_a_mass("MEMIT", None, out)
            else:
                R.run_arm_a("MEMIT", None, out)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    return {"supported": True, "calls": verification_calls,
            "rows": R.read_jsonl(log), "error": error}


def run_fake_arm_c(out: Path, *, recover_failure: bool = False, invocations: int = 0):
    """Exercise Arm C retry bookkeeping with synthetic model-load outcomes."""
    out.mkdir(parents=True, exist_ok=True)
    log = out / "armC.jsonl"
    models = [{"id": f"cal-{index}", "role": "calibration"} for index in range(3)]
    references = {item["id"]: (float(index + 1), "token")
                  for index, item in enumerate(models)}
    load_calls = []
    values = {item["id"]: float(index + 1) for index, item in enumerate(models)}

    def fake_load(model_id, dtype):
        load_calls.append(model_id)
        if not recover_failure:
            raise RuntimeError("synthetic deterministic load failure")
        return object(), object(), dtype

    def fake_analyze(_model, _tokenizer, _stats, model_id, role, dtype):
        return {"model_id": model_id, "role": role, "dtype": dtype,
                "max_t_word": values[model_id]}

    if recover_failure:
        R.append_jsonl(log, {"record_type": "failure", "run_mode": "registered", "run_id": "r0",
                             "model_id": "cal-0", "phase": "calibration", "error": "prior attempt"})
    error = None
    with ExitStack() as stack:
        stack.enter_context(patch.object(R, "ARM_C_MODELS", models))
        stack.enter_context(patch.object(R, "ARM_C_REVIEWER_MAX", references))
        stack.enter_context(patch.object(R, "ARM_C_MAX_ATTEMPTS", 3, create=True))
        stack.enter_context(patch.object(R, "stats_helpers", lambda: object()))
        stack.enter_context(patch.object(R, "log_path_for", lambda *args: log))
        stack.enter_context(patch.object(R, "load_arm_c_pair", fake_load))
        stack.enter_context(patch.object(R, "analyze_model", fake_analyze))
        for _ in range(invocations):
            try:
                R.run_arm_c(None, out)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
    return {"load_calls": load_calls, "rows": R.read_jsonl(log), "error": error}


def _kill_during_fake_arm_c_load(out: Path, entered: Path, refused: Path) -> None:
    """Child target: stop inside a fake load, or report a pre-load attempt refusal."""
    models = [{"id": f"cal-{index}", "role": "calibration"} for index in range(3)]
    references = {item["id"]: (float(index + 1), "token")
                  for index, item in enumerate(models)}

    def fake_load(_model_id, _dtype):
        entered.write_text("fake loader entered\n", encoding="utf-8")
        while True:
            time.sleep(0.05)

    with ExitStack() as stack:
        stack.enter_context(patch.object(R, "ARM_C_MODELS", models))
        stack.enter_context(patch.object(R, "ARM_C_REVIEWER_MAX", references))
        stack.enter_context(patch.object(R, "stats_helpers", lambda: object()))
        stack.enter_context(patch.object(R, "log_path_for", lambda *args: out / "armC.jsonl"))
        stack.enter_context(patch.object(R, "load_arm_c_pair", fake_load))
        try:
            R.run_arm_c(None, out)
        except RuntimeError as exc:
            refused.write_text(str(exc), encoding="utf-8")


check("took requires old lead before and new lead after",
      R.took_transition(-1.0, -2.0, -3.0, -0.5) == (True, False))
check("already satisfied when new object led before",
      R.took_transition(-3.0, -1.0, -4.0, -0.2) == (False, True))
check("tie before is neither took nor already satisfied",
      R.took_transition(-1.0, -1.0, -2.0, -0.5) == (False, False))
check("WSP took requires no exact pre-answer and all three post-answers",
      R.wsp_took(0, 3) == (True, False) and R.wsp_took(1, 3) == (False, True)
      and R.wsp_took(0, 2) == (False, False))

delta = torch.tensor([[3.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
r1, u1, s1, v1 = R.make_random_update(delta, 198, True)
r2, u2, s2, v2 = R.make_random_update(delta, 198, True)
check("random control preserves singular values",
      torch.allclose(torch.linalg.svdvals(r1), torch.linalg.svdvals(delta), atol=1e-10, rtol=1e-10))
check("random control factors are orthonormal",
      torch.allclose(u1.T @ u1, torch.eye(3, dtype=torch.float64), atol=1e-10)
      and torch.allclose(v1.T @ v1, torch.eye(3, dtype=torch.float64), atol=1e-10))
check("random control is deterministic for a seed",
      torch.equal(r1, r2) and torch.equal(u1, u2) and torch.equal(s1, s2) and torch.equal(v1, v2))
layers = R.random_control_layers({17: delta, 18: delta}, {17: delta, 18: delta}, 396)
layers_again = R.random_control_layers({17: delta, 18: delta}, {17: delta, 18: delta}, 396)
check("multi-layer control is deterministic from seed 198 plus case id",
      all(torch.equal(layers[k], layers_again[k]) for k in layers))

check("mass batch passes at exactly eight takes", R.mass_gate([True] * 8 + [False] * 2))
check("mass batch fails below eight takes", not R.mass_gate([True] * 7 + [False] * 3))
check("mass forecast is not evaluable with no counted batch", not R.mass_gate_status(0))
check("mass forecast can be evaluated with a counted batch", R.mass_gate_status(1))

ci = R.cp_interval(0, 24)
check("Clopper-Pearson zero-of-24 upper bound is about 0.142",
      ci is not None and abs(ci[1] - 0.1420) < 0.001)
metric_rows = [
    {"took": True, "already_satisfied": False, "name_eligible": True,
     "localization": {"t_word": {"localized": True}}},
    {"took": False, "already_satisfied": True, "name_eligible": False,
     "localization": {"t_word": {"localized": False}}},
]
metric = S.metric_result("synthetic", metric_rows,
    lambda r: S.flag(r, "localization", "t_word", "localized"), "took", -1, True)
check("score counts attempted, took, already-satisfied, eligible and successes",
      metric["attempted"] == 2 and metric["took"] == 1 and metric["already_satisfied"] == 1
      and metric["name_eligible"] == 1 and metric["successes"] == 1
      and metric["rate_denominator"] == 1 and metric["status"] == "PASS")

with tempfile.TemporaryDirectory(prefix="part198-build-test-") as td:
    out = Path(td)
    (out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                                "crosscheck_pass": True}), encoding="utf-8")
    rows = [
        {"record_type": "case", "run_mode": "registered", "case_id": 1, "took": True,
         "already_satisfied": False, "name_eligible": True,
         "localization": {"t_word": {"localized": True}},
         "blind_reader": {"localized": True, "named_strict": True}},
        {"record_type": "case", "run_mode": "registered", "case_id": 2, "took": False,
         "already_satisfied": True, "name_eligible": True,
         "localization": {"t_word": {"localized": False}},
         "blind_reader": {"localized": False, "named_strict": False}},
        {"record_type": "case", "run_mode": "smoke", "case_id": 3, "took": True,
         "already_satisfied": False, "name_eligible": True,
         "localization": {"t_word": {"localized": True}}},
    ]
    with (out / "armB-wsp.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    parsed = S.registered_rows(out / "armB-wsp.jsonl", "case", "case_id")
    counts = R.cp_interval(1, 1)
    scored = S.score_from_dir(out)
    endpoint = next(x for x in scored["forecasts"] if x["forecast"] == "198.6 WSP T_word localization")
    check("score ignores smoke JSONL rows and counts registered hand-made rows",
          len(parsed) == 2 and sum(bool(r["took"]) for r in parsed) == 1
          and sum(bool(r["already_satisfied"]) for r in parsed) == 1 and counts is not None
          and endpoint["attempted"] == 2 and endpoint["took"] == 1
          and endpoint["already_satisfied"] == 1 and endpoint["successes"] == 1
          and endpoint["rate_denominator"] == 1)

with tempfile.TemporaryDirectory(prefix="part198-resume-test-") as td:
    path = Path(td) / "run.jsonl"
    R.append_jsonl(path, {"record_type": "case", "run_mode": "registered", "case_id": 10})
    R.append_jsonl(path, {"record_type": "case", "run_mode": "smoke", "case_id": 11})
    R.append_jsonl(path, {"record_type": "failure", "run_mode": "registered", "case_id": 12})
    R.append_jsonl(path, {"record_type": "batch", "run_mode": "registered", "batch_id": "MEMIT:0"})
    keys = R.existing_keys(path, "case_id")
    batches = R.existing_keys(path, "batch_id")
    check("resume skips registered ids and failures but not smoke ids", keys == {"10", "12"})
    check("resume indexes mass batch ids", batches == {"MEMIT:0"})

# Fold r1 regression checks: all synthetic data, no model or edit paths.
try:
    from scipy.stats import beta as scipy_beta
    cp20 = R.cp_interval(20, 24)
    cp1 = R.cp_interval(1, 24)
    cp24 = R.cp_interval(24, 24)
    cp0 = R.cp_interval(0, 24)
    check("B1 two-sided CP endpoints match independent beta quantiles",
          cp20 is not None and abs(cp20[0] - scipy_beta.ppf(0.025, 20, 5)) < 1e-10
          and abs(cp20[1] - scipy_beta.ppf(0.975, 21, 4)) < 1e-10
          and cp1 is not None and abs(cp1[0] - scipy_beta.ppf(0.025, 1, 24)) < 1e-10
          and cp24 is not None and abs(cp24[0] - scipy_beta.ppf(0.025, 24, 1)) < 1e-10
          and cp0 is not None and cp0[0] == 0.0
          and abs(cp0[1] - scipy_beta.ppf(0.975, 1, 24)) < 1e-10)
except ImportError:
    check("B1 CP bisection returns the design's 20/24 two-sided lower bound",
          abs(R.cp_interval(20, 24)[0] - 0.6261582865) < 1e-6)

aimed = getattr(R, "aimed_output_direction", None)
if aimed is None:
    aimed_ok = False
else:
    E = torch.tensor([[1.0, -1.0, 0.5, -0.5], [2.0, 0.0, -1.0, -1.0]], dtype=torch.float64)
    gamma = torch.tensor([0.5, 1.0, 1.5, 2.0], dtype=torch.float64)
    expected = gamma * (E[0] - E[0].mean())
    expected = expected / expected.norm()
    actual = aimed(E, gamma, 0)
    aimed_ok = actual is not None and torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
check("B3 aimed control centres before applying non-uniform gamma and normalizes last", aimed_ok)

with tempfile.TemporaryDirectory(prefix="part198-jsonl-r1-") as td:
    torn = Path(td) / "torn.jsonl"
    torn.write_bytes(b'{"record_type":"case","case_id":1}\n{"record_type":"case","case_id":2')
    original_bytes = torn.read_bytes()
    status_reader = getattr(R, "read_jsonl_with_status", None)
    if status_reader is None:
        pure_rows, partial = [], None
    else:
        pure_rows, partial = status_reader(torn)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pure_rows_again = R.read_jsonl(torn)
    check("Fold r5 pure reader returns complete rows, flags and ignores a partial tail without writing",
          [r.get("case_id") for r in pure_rows] == [1]
          and partial is not None and partial.get("line") == 2
          and partial.get("ignored_bytes") == len(b'{"record_type":"case","case_id":2')
          and [r.get("case_id") for r in pure_rows_again] == [1]
          and any("ignored unterminated final JSONL line" in str(item.message) for item in caught)
          and torn.read_bytes() == original_bytes)
    valid_without_newline = Path(td) / "valid-but-unterminated.jsonl"
    valid_without_newline.write_bytes(b'{"record_type":"case","case_id":3}')
    valid_before = valid_without_newline.read_bytes()
    valid_rows, valid_tail = (([], None) if status_reader is None
                              else status_reader(valid_without_newline))
    check("Fold r5 pure reader ignores even valid JSON without a line terminator",
          valid_rows == [] and valid_tail is not None and valid_tail.get("line") == 1
          and valid_without_newline.read_bytes() == valid_before)
    recovery_start = len(getattr(R, "JSONL_RECOVERIES", []))
    resume_reader = getattr(R, "read_jsonl_for_resume", None)
    if resume_reader is None:
        resumed_rows, recovery_recorded = [], False
    else:
        with R.output_log_lock(torn):
            resumed_rows = resume_reader(torn)
        recovery_recorded = (len(R.JSONL_RECOVERIES) == recovery_start + 1
                             and R.JSONL_RECOVERIES[-1].get("path") == str(torn)
                             and R.JSONL_RECOVERIES[-1].get("line") == 2)
    main_source = inspect.getsource(R.main)
    lock_before_repair = ("read_jsonl_for_resume(target_log)" in main_source
        and main_source.index("with output_log_lock(target_log)")
        < main_source.index("read_jsonl_for_resume(target_log)"))
    check("Fold r5 runner resume repairs only its own unterminated tail",
          [r.get("case_id") for r in resumed_rows] == [1]
          and torn.read_bytes() == b'{"record_type":"case","case_id":1}\n'
          and recovery_recorded
          and lock_before_repair)
    malformed = Path(td) / "malformed.jsonl"
    malformed.write_bytes(b'{"ok":1}\nnot-json\n{"ok":2}\n')
    raised = False
    try:
        R.read_jsonl(malformed)
    except ValueError as exc:
        raised = "malformed JSONL" in str(exc)
    malformed_tail = Path(td) / "malformed-newline.jsonl"
    malformed_tail.write_bytes(b'{"ok":1}\nnot-json\n')
    tail_raised = False
    try:
        R.read_jsonl(malformed_tail)
    except ValueError:
        tail_raised = True
    check("M1 keeps raising on interior and newline-terminated malformed JSONL", raised and tail_raised)

with tempfile.TemporaryDirectory(prefix="part198-duplicate-r1-") as td:
    path = Path(td) / "armA-rome.jsonl"
    R.append_jsonl(path, {"record_type": "failure", "run_mode": "registered", "run_id": "r0",
                          "case_id": 7, "phase": "edit_or_score", "error": "first attempt failed"})
    first = {"record_type": "case", "run_mode": "registered", "run_id": "r1", "timestamp": "t1",
             "case_id": 7, "took": True, "name_eligible": True,
             "localization": {"f1": 1.0}, "restored_bit_exact": True}
    R.append_jsonl(path, first)
    R.append_jsonl(path, dict(first, run_id="r2", timestamp="t2"))
    R.append_jsonl(path, {"record_type": "failure", "run_mode": "registered", "run_id": "r4",
                          "phase": "startup_or_run", "error": "preflight failed"})
    R.append_jsonl(path, {"record_type": "failure", "run_mode": "registered", "run_id": "r5",
                          "phase": "startup_or_run", "error": "preflight failed again"})
    try:
        replay_score = S.score_from_dir(Path(td))
    except ValueError:
        replay_score = {}
    replay_rows = replay_score.get("forecasts", [])
    single_score = next((row for row in replay_rows if row.get("forecast") == "198.1 ROME single"), {})
    unkeyed = replay_score.get("run_level_failures", {}).get("all-armA-ROME", [])
    replay_ok = single_score.get("attempted") == 1 and single_score.get("took") == 1
    unkeyed_reported = len(unkeyed) == 2 and all(row["phase"] == "startup_or_run" for row in unkeyed)
    conflict_refused = False
    try:
        R.append_jsonl(path, dict(first, run_id="r3", timestamp="t3", took=False))
        S.score_from_dir(Path(td))
    except ValueError as exc:
        conflict_refused = "case_id='7'" in str(exc) and "took" in str(exc)
    check("N1 failure/resume and metadata-only replay stay scoreable; conflicting terminals refuse",
          replay_ok and conflict_refused and unkeyed_reported)

check("F1 refuses replay rows differing only in aimed_positive_control.oracle_named_strict",
      case_replay_conflicts(("aimed_positive_control", "oracle_named_strict")))
check("Fold r4 refuses the runner-shaped aimed-control localization replay",
      case_replay_conflicts(("aimed_positive_control", "localization", "localized")))
for stat in ("f1", "pcs", "t_word"):
    check(f"F1 refuses replay rows differing only in localization.{stat}.localized",
          case_replay_conflicts(("localization", stat, "localized")))
check("Fold r4 refuses raw float replay drift and reports its path",
      case_replay_conflicts(("aimed_positive_control", "localization", "value"), 2.1))
check("Fold r4 refuses conflicting mass-batch localization and reports its path",
      batch_replay_conflicts())
check("Fold r4 ignores only run_id/timestamp and preserves scorer counts",
      metadata_replay_counts_unchanged())
check("Fold r4 second runner process refuses a held output log before preflight or any arm",
      runner_refuses_held_log())
check("Fold r5 output log lock releases after holder process death",
      runner_lock_releases_after_process_death())
check("Fold r5 scorer API and CLI refuse a held log without changing its bytes",
      scorer_refuses_held_log())
check("Fold r5 scorer locks all eight logs in a fixed sorted order",
      getattr(S, "SCORE_LOG_NAMES", ()) == tuple(sorted(getattr(S, "SCORE_LOG_NAMES", ())))
      and len(getattr(S, "SCORE_LOG_NAMES", ())) == 8)

measure = getattr(R, "wsp_take_measurement", None)
if measure is None:
    wsp_measure_ok = False
else:
    measured = measure(0, 3, 3, "2/3")
    malformed_measure = measure(0, 3, 3, "not-a-count")
    wsp_measure_ok = (measured["took"] is True and measured["free_exact_count"] == 2
                      and measured["free_exact_disagreement"] is True
                      and malformed_measure["took"] is True
                      and malformed_measure["free_exact_count"] is None
                      and malformed_measure["free_exact_disagreement"] is None)
check("M3 runner after-count controls WSP take and self-report mismatch remains visible", wsp_measure_ok)

class TinyState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.register_buffer("running", torch.tensor([3.0]))

tiny = TinyState()
snapshot_fn = getattr(R, "model_state_snapshot", None)
matches_fn = getattr(R, "model_matches_snapshot", None)
if snapshot_fn is None or matches_fn is None:
    restoration_ok = False
else:
    snapshot = snapshot_fn(tiny)
    with torch.no_grad():
        tiny.running.add_(1.0)
    detects_buffer = not matches_fn(tiny, snapshot)
    with torch.no_grad():
        tiny.running.sub_(1.0)
    restoration_ok = detects_buffer and matches_fn(tiny, snapshot)
check("M4 restoration fingerprint covers unreported parameters and buffers", restoration_ok)

try:
    scalar_fingerprint = R.tensor_fingerprint(torch.tensor(2.5))
    scalar_tiny = TinyState()
    scalar_tiny.register_buffer("scalar", torch.tensor(4.0))
    scalar_snapshot = R.model_state_snapshot(scalar_tiny)
    scalar_ok = bool(scalar_fingerprint) and R.model_matches_snapshot(scalar_tiny, scalar_snapshot)
except Exception:
    scalar_ok = False
check("N2 fingerprints 0-d tensors and keeps them in the model snapshot", scalar_ok)

fingerprint_error_reported = False
verify_restored = getattr(R, "verify_model_restored", None)
if callable(verify_restored):
    try:
        with patch.object(R, "model_matches_snapshot",
                          side_effect=RuntimeError("synthetic byte fingerprint failure")):
            verify_restored(tiny, snapshot, "test restoration")
    except RuntimeError as exc:
        fingerprint_error_reported = "fingerprint failed" in str(exc) and "not restored" not in str(exc)
check("N2 fingerprint errors remain explicit verification errors, never restoration mismatches",
      fingerprint_error_reported)

with tempfile.TemporaryDirectory(prefix="part198-runner-restore-") as td:
    single = run_fake_arm_a(Path(td) / "single")
    mass = run_fake_arm_a(Path(td) / "mass", mass=True)
    single_case = next((row for row in single["rows"] if row.get("record_type") == "case"), {})
    mass_batch = next((row for row in mass["rows"] if row.get("record_type") == "batch"), {})
    check("N3 Arm A single runner detects an editor-mutated unreported buffer",
          single["supported"] and single_case.get("restored_bit_exact") is False
          and bool(single["calls"]) and "case 101" in single["calls"][0])
    check("N3 Arm A mass runner detects an editor-mutated unreported buffer",
          mass["supported"] and mass_batch.get("restored_bit_exact") is False
          and bool(mass["calls"]) and "mass batch 0" in mass["calls"][0])

with tempfile.TemporaryDirectory(prefix="part198-fingerprint-error-") as td:
    failed_verify = run_fake_arm_a(Path(td) / "single", fingerprint_failure=True)
    failure_row = next((row for row in failed_verify["rows"] if row.get("record_type") == "failure"), {})
    check("N2 runner reports fingerprint failure as unknown restoration, with the cause",
          failed_verify["supported"] and failure_row.get("restored_bit_exact") is None
          and "fingerprint failed" in str(failure_row.get("restoration_verification_error")))

with tempfile.TemporaryDirectory(prefix="part198-armc-r1-") as td:
    log = Path(td) / "armC.jsonl"
    log.write_text(json.dumps({"record_type": "failure", "run_mode": "registered",
                               "model_id": "calibration-model"}) + "\n", encoding="utf-8")
    successful = getattr(R, "arm_c_successful_keys", None)
    retry_ok = successful is not None and successful(log) == set()
    missing_tstar_refused = False
    require = getattr(R, "require_tstar", None)
    if require is not None:
        try:
            require(Path(td) / "missing-tstar.json")
        except RuntimeError as exc:
            missing_tstar_refused = "missing" in str(exc)
    run_arm_c_source = inspect.getsource(R.run_arm_c)
    evaluation_guard = ("require_tstar(tstar_path)" in run_arm_c_source
                        and run_arm_c_source.index("require_tstar(tstar_path)")
                        < run_arm_c_source.rfind("load_arm_c_pair"))
    check("M5 Arm C retries failed calibration keys and requires T* before evaluation loads",
          retry_ok and missing_tstar_refused and evaluation_guard)

with tempfile.TemporaryDirectory(prefix="part198-armc-retry-cap-") as td:
    retry_run = run_fake_arm_c(Path(td), invocations=4)
    keyed_failures = [row for row in retry_run["rows"]
                      if row.get("record_type") == "failure" and row.get("model_id") == "cal-0"]
    attempts = [row.get("attempt_number") for row in keyed_failures]
    cap_error = str(retry_run["error"])
    cap_ok = (retry_run["load_calls"] == ["cal-0"] * 3 and len(keyed_failures) == 3
              and attempts == [1, 2, 3]
              and ("attempt limit reached" in cap_error or "retry limit reached" in cap_error)
              and "cal-0" in cap_error)
check("P1 deterministic Arm C exceptions stop after three attempts and report the model",
      cap_ok)

with tempfile.TemporaryDirectory(prefix="part198-armc-killed-load-") as td:
    out = Path(td) / "out"
    out.mkdir()
    context = multiprocessing.get_context("fork")
    child_results = []
    for index in range(4):
        entered = out / f"fake-load-entered-{index}"
        refused = out / f"attempt-refused-{index}"
        child = context.Process(target=_kill_during_fake_arm_c_load, args=(out, entered, refused))
        child.start()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and child.is_alive() and not (entered.exists() or refused.exists()):
            time.sleep(0.02)
        reached_fake_load = entered.exists()
        if child.is_alive() and reached_fake_load:
            child.kill()
        child.join(timeout=8.0)
        if child.is_alive():
            child.kill()
            child.join(timeout=2.0)
        child_results.append((reached_fake_load, child.exitcode,
                              refused.read_text(encoding="utf-8") if refused.exists() else ""))
    arm_c_log = out / "armC.jsonl"
    starts = [row for row in R.read_jsonl(arm_c_log)
              if row.get("record_type") == "attempt_started" and row.get("model_id") == "cal-0"]
    killed_cap_ok = (len(starts) == 3 and [row.get("attempt_number") for row in starts] == [1, 2, 3]
                     and all(reached and code == -signal.SIGKILL for reached, code, _ in child_results[:3])
                     and child_results[3][0] is False and child_results[3][1] == 0
                     and "attempt limit reached" in child_results[3][2])
check("F3 killed fake loads consume the Arm C cap across four child invocations", killed_cap_ok)

with tempfile.TemporaryDirectory(prefix="part198-score-armc-exhausted-") as td:
    out = Path(td)
    arm_c = out / "armC.jsonl"
    for attempt in range(1, 4):
        run_id = f"attempt-{attempt}"
        R.append_jsonl(arm_c, {"record_type": "attempt_started", "run_mode": "registered",
            "run_id": run_id, "model_id": R.ARM_C_MODELS[0]["id"], "attempt_number": attempt})
        R.append_jsonl(arm_c, {"record_type": "failure", "run_mode": "registered", "run_id": run_id,
            "model_id": R.ARM_C_MODELS[0]["id"], "attempt_number": attempt})
    try:
        arm_c_score = S.score_from_dir(out).get("arm_c_attempts", {})
    except Exception:
        arm_c_score = {}
    first_model = str(R.ARM_C_MODELS[0]["id"])
    exhausted = arm_c_score.get(first_model, {})
check("F3 scorer labels a three-attempt Arm C model EXHAUSTED",
      exhausted.get("attempts") == 3 and exhausted.get("status") == "EXHAUSTED")

with tempfile.TemporaryDirectory(prefix="part198-armc-failure-resume-") as td:
    recovered = run_fake_arm_c(Path(td), recover_failure=True, invocations=1)
    successful_ids = {row.get("model_id") for row in recovered["rows"]
                      if row.get("record_type") == "armC_model"}
    check("N5 Arm C resume retries a failed calibration key instead of treating failure as success",
          recovered["load_calls"] == ["cal-0", "cal-1", "cal-2"]
          and successful_ids == {"cal-0", "cal-1", "cal-2"} and recovered["error"] is None)

arm_b_source = inspect.getsource(R.run_arm_b)
sabotage_a_source = arm_b_source.replace("undone = w.undo()",
    "append_jsonl(log, record)\nundone = w.undo()", 1)
check("N4 Arm B has exactly one terminal append after exact-zero undo; rejects sabotage A",
      n4_terminal_append_is_safe(arm_b_source) and not n4_terminal_append_is_safe(sabotage_a_source))

with tempfile.TemporaryDirectory(prefix="part198-preflight-hash-test-") as td:
    root = Path(td) / "source"
    root.mkdir()
    stats_file = root / "stats198.py"
    stats_file.write_text("# synthetic frozen stats module\n", encoding="utf-8")
    digest = R.hash_file(stats_file)
    crosscheck = {"fails": [], "gpt2": {str(i): {} for i in range(48)},
                  "qwen": {str(i): {} for i in range(24)}, "stats198_sha256": digest}
    (root / "crosscheck198.json").write_text(json.dumps(crosscheck), encoding="utf-8")
    fixture_proc = SimpleNamespace(returncode=0, stdout="FIXTURES ALL PASS")
    suite_proc = SimpleNamespace(returncode=0, stdout="TEST_BUILD_CHECKS 48\nTEST_BUILD ALL PASS")
    with patch.object(R, "HERE", root):
        with patch.object(R.subprocess, "run", side_effect=[fixture_proc, suite_proc]):
            matching = R.preflight(Path(td) / "matching-out")
        crosscheck["stats198_sha256"] = "stale-hash"
        (root / "crosscheck198.json").write_text(json.dumps(crosscheck), encoding="utf-8")
        with patch.object(R.subprocess, "run", side_effect=[fixture_proc, suite_proc]):
            try:
                R.preflight(Path(td) / "stale-out")
            except RuntimeError:
                stale_gate = json.loads((Path(td) / "stale-out" / "gate.json").read_text(encoding="utf-8"))
            else:
                stale_gate = {}
        crosscheck["stats198_sha256"] = digest
        (root / "crosscheck198.json").write_text(json.dumps(crosscheck), encoding="utf-8")
        no_marker_proc = SimpleNamespace(returncode=0, stdout="TEST_BUILD ALL PASS (0 failed)")
        with patch.object(R.subprocess, "run", side_effect=[fixture_proc, no_marker_proc]):
            try:
                R.preflight(Path(td) / "empty-suite-out")
            except RuntimeError:
                empty_suite_gate = json.loads((Path(td) / "empty-suite-out" / "gate.json").read_text(encoding="utf-8"))
            else:
                empty_suite_gate = {}
    check("N5 preflight validates the stats hash and requires the synthetic-test count marker",
          matching["passed"] is True and stale_gate.get("crosscheck_pass") is False
          and stale_gate.get("details", {}).get("crosscheck", {}).get("stats_hash_matches") is False
          and empty_suite_gate.get("synthetic_tests_pass") is False
          and empty_suite_gate.get("details", {}).get("synthetic_tests", {}).get("check_count") is None)

final_row_fn = getattr(S, "final_sequence_row", None)
if final_row_fn is None:
    seq_final_ok = False
else:
    seq_final_ok = final_row_fn([
        {"case_id": 1, "n_written_so_far": 10, "names_after_write": {"x": {"named_strict": True}}},
        {"case_id": 2, "n_written_so_far": 3, "names_after_write": {"x": {"named_strict": False}}},
    ])["n_written_so_far"] == 10
check("M6 sequence final state is selected by highest completed-write count", seq_final_ok)

armc_limit = getattr(R, "arm_c_limit_reached", None)
armc_source = inspect.getsource(R.run_arm_c)
limit_guard_ok = (armc_limit is not None and not armc_limit(0, 1) and armc_limit(1, 1)
                  and armc_source.count("arm_c_limit_reached(selected_count, limit)") >= 2
                  and armc_source.index("arm_c_limit_reached(selected_count, limit)")
                  < armc_source.index("load_arm_c_pair"))
check("M7 both Arm C loops check the shared limit before model loads", limit_guard_ok)

current_stats_hash = R.hash_file(HERE / "stats198.py") if hasattr(R, "hash_file") else ""
crosscheck_helper = getattr(R, "crosscheck_hash_matches", None)
crosscheck_source = (HERE / "crosscheck198.py").read_text(encoding="utf-8")
m8_ok = (crosscheck_helper is not None and crosscheck_helper({"stats198_sha256": current_stats_hash},
                                                              current_stats_hash)
         and not crosscheck_helper({"stats198_sha256": "stale"}, current_stats_hash)
         and 'out["stats198_sha256"]' in crosscheck_source
         and "crosscheck_hash_matches" in inspect.getsource(R.preflight))
check("M8 preflight binds crosscheck to the checked stats198.py hash", m8_ok)

with tempfile.TemporaryDirectory(prefix="part198-basehash-r1-") as td:
    blob = Path(td) / "input.bin"
    blob.write_bytes(b"base statistics input A")
    digest_a = R.hash_input_hashes({"weights": R.hash_file(blob)}) if hasattr(R, "hash_input_hashes") else ""
    blob.write_bytes(b"base statistics input B")
    digest_b = R.hash_input_hashes({"weights": R.hash_file(blob)}) if hasattr(R, "hash_input_hashes") else ""
    base_path = Path(td) / "base.json"
    cache_ok = digest_a != digest_b
    if hasattr(R, "save_base"):
        try:
            R.save_base(base_path, "tiny", {0: {"f1": 1.0}}, "input-hash")
            cache_ok &= R.load_base(base_path, "input-hash") == {0: {"f1": 1.0}}
            try:
                R.load_base(base_path, "different-hash")
                cache_ok = False
            except RuntimeError:
                pass
        except TypeError:
            cache_ok = False
check("m3 base statistics cache stores and checks its input fingerprint", cache_ok)

stats = R.stats_helpers()
base_synthetic = {0: {"f1": 1.0, "f2": 1.0, "pcs": 1.0, "t_max": 1.0, "t_word": 1.0,
                      "pairs": []},
                  1: {"f1": 1.0, "f2": 1.0, "pcs": 1.0, "t_max": 1.0, "t_word": 1.0,
                      "pairs": []}}
current_synthetic = {1: {"f1": 2.0, "f2": 2.0, "pcs": 2.0, "t_max": 2.0, "t_word": 2.0,
                         "pairs": []}}
runner_localization = R.localization(base_synthetic, current_synthetic, {1}, stats)
runner_batch = {"record_type": "batch", "take_gate_passed": True,
                "facts": [{"took": True, "name_eligible": True} for _ in range(10)],
                "localization": runner_localization}
mass_result = S.mass_forecast("synthetic", [runner_batch], "f1", True)
check("B2 runner-shaped localization reaches the mass forecast", runner_localization["f1"]["localized"]
      and mass_result["successes"] == 1 and mass_result["status"] == "FAIL")

second_batch = dict(runner_batch, batch_id="dropped", take_gate_passed=False)
dropped_result = S.mass_forecast("synthetic", [runner_batch, second_batch], "f1", True)
check("m2 mass scorer reports gate-failing batches it drops",
      dropped_result.get("counted_batches") == 1 and dropped_result.get("dropped_batches") == 1)
saved_gate_helper = R.mass_gate_status
try:
    R.mass_gate_status = lambda n: False
    exercised_mass_gate = S.mass_forecast("synthetic", [runner_batch], "f1", True)
finally:
    R.mass_gate_status = saved_gate_helper
check("m1 mass gate status is exercised by the scorer path",
      exercised_mass_gate["status"] == "NOT EVALUABLE")

def synthetic_case(case_id, took=True, aimed_control=True):
    row = {"record_type": "case", "run_mode": "registered", "case_id": case_id,
           "took": took, "already_satisfied": not took, "name_eligible": True,
           "restored_bit_exact": True,
           "localization": {"f1": {"localized": False}, "pcs": {"localized": False},
                            "t_word": {"localized": False}},
           "oracle_reader": {"named_strict": False, "named_loose": False},
           "edited_layer_stats": {"17": {"update_top_names_strict": took,
                                          "update_top_names_loose": took}}}
    if aimed_control:
        row["aimed_positive_control"] = {"oracle_named_strict": False,
                                          "oracle_named_loose": False}
    else:
        row["aimed_positive_control"] = {"oracle_named_strict": True,
                                          "oracle_named_loose": True}
    return row

with tempfile.TemporaryDirectory(prefix="part198-score-r1-") as td:
    out = Path(td)
    (out / "gate.json").write_text(json.dumps({"fixtures_pass": True, "synthetic_tests_pass": True,
                                                "crosscheck_pass": True}), encoding="utf-8")
    rome = [synthetic_case(i, took=True) for i in range(23)]
    rome[0].pop("aimed_positive_control")
    rome.append(synthetic_case(23, took=False, aimed_control=False))
    for method in ("rome", "memit", "alphaedit"):
        rows = rome if method == "rome" else [synthetic_case(i) for i in range(24)]
        rows = rows + [{"record_type": "state_verification", "run_mode": "registered",
                        "restored_bit_exact": True, "update_reproduced": True}]
        (out / f"armA-{method}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    arm_b = []
    for i in range(24):
        row = synthetic_case(i)
        row["undo_zero_exact"] = True
        row["blind_reader"] = {"localized": False, "named_strict": False}
        row["random_control"] = {"localization": {"t_word": {"localized": False}}}
        arm_b.append(row)
    (out / "armB-wsp.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in arm_b), encoding="utf-8")
    seq_rows = []
    for i in range(10):
        written = 10 if i == 0 else i
        seq_rows.append({"record_type": "case", "run_mode": "registered", "case_id": i,
                         "took": True, "name_eligible": True, "n_written_so_far": written,
                         "names_after_write": {str(j): {"named_strict": i == 0 and j == 0}
                                                for j in range(10)}})
    (out / "armB-seq-wsp.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in seq_rows), encoding="utf-8")
    scored = S.score_from_dir(out)
    s1 = scored["sensitivity_checks"]["S1"]
    oracle = next(x for x in scored["forecasts"] if x["forecast"] == "198.4 ROME oracle naming")
    update = next(x for x in scored["forecasts"] if x["forecast"] == "198.4 ROME update-top naming")
    seq_forecast = next(x for x in scored["forecasts"] if x["forecast"] == "198.8 WSP sequence final strict names")
    check("B4 S1 counts only took/name-eligible rows and missing control rows as failures",
          s1["name_eligible"] == 23 and s1["successes"] == 0 and s1.get("missing_controls") == 1)
    check("B5 failed S1 suppresses oracle naming only, preserving update-top verdict",
          s1["status"] == "FAIL" and oracle["status"] == "NOT EVALUABLE" and update["status"] == "FAIL")
    check("M6 198.8 scores the row with highest n_written_so_far",
          seq_forecast["successes"] == 1 and seq_forecast["rate_denominator"] == 10)
    nine_plus_failure = seq_rows[:9] + [{"record_type": "failure", "run_mode": "registered",
        "case_id": 9, "phase": "sequence_write_or_score", "error": "synthetic failure"}]
    (out / "armB-seq-wsp.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in nine_plus_failure), encoding="utf-8")
    failed_tenth = S.score_from_dir(out)
    failed_tenth_metric = next(x for x in failed_tenth["forecasts"]
        if x["forecast"] == "198.8 WSP sequence final strict names")
    completed_with_failure = seq_rows + [{"record_type": "failure", "run_mode": "registered",
        "case_id": 0, "phase": "sequence_write_or_score", "error": "synthetic later failure"}]
    (out / "armB-seq-wsp.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in completed_with_failure), encoding="utf-8")
    failed_replay_metric = next(x for x in S.score_from_dir(out)["forecasts"]
        if x["forecast"] == "198.8 WSP sequence final strict names")
    check("Fold r5 198.8 refuses nine completions plus failure and failure after a completion",
          failed_tenth_metric["status"] == "NOT EVALUABLE"
          and failed_tenth_metric.get("completed_sequence_cases") == 9
          and failed_tenth_metric.get("registered_sequence_failures") == 1
          and failed_replay_metric["status"] == "NOT EVALUABLE"
          and failed_replay_metric.get("completed_sequence_cases") == 10
          and failed_replay_metric.get("registered_sequence_failures") == 1)
    (out / "armB-seq-wsp.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in seq_rows), encoding="utf-8")
    rome[1]["aimed_positive_control"] = {"reason": "zero aimed direction"}
    (out / "armA-rome.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rome), encoding="utf-8")
    missing_reason_control = S.score_from_dir(out)["sensitivity_checks"]["S1"]
    check("N6 reports a zero-direction reason object as a missing S1 control",
          missing_reason_control.get("missing_controls") == 2)

check("M7 both Arm C loops share the pre-load limit guard",
      inspect.getsource(R.run_arm_c).count("arm_c_limit_reached(selected_count, limit)") >= 2)
check("M5 reports failure rows before a restoration-triggered process stop",
      inspect.getsource(R.run_arm_a).index('"record_type": "failure"')
      < inspect.getsource(R.run_arm_a).index('raise RuntimeError("editor failure left no verified restoration'))

check("N5 synthetic build suite executes its minimum check floor", len(check_names) >= 30)
print("TEST_BUILD_CHECKS", len(check_names))
print("TEST_BUILD", "FAIL" if fails else "ALL PASS", f"({fails} failed)")
sys.exit(1 if fails else 0)

