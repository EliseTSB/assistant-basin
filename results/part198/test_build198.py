#!/usr/bin/env python3
"""Tiny inert checks for Part 198 run, control, gate, scoring and resume helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import warnings
import inspect
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run198 as R  # noqa: E402
import score198 as S  # noqa: E402

fails = 0


def check(name: str, condition: bool) -> None:
    global fails
    print(("PASS " if condition else "FAIL ") + name)
    if not condition:
        fails += 1


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
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            recovered = R.read_jsonl(torn)
        except ValueError:
            recovered = []
    recovery_recorded = any(item.get("path") == str(torn) and item.get("line") == 2
                            for item in getattr(R, "JSONL_RECOVERIES", []))
    check("M1 drops, truncates and records only a torn final unterminated JSONL line",
          [r.get("case_id") for r in recovered] == [1] and torn.read_bytes().endswith(b"\n")
          and recovery_recorded and bool(caught))
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
    path = Path(td) / "dupes.jsonl"
    first = {"record_type": "case", "run_mode": "registered", "case_id": 7, "took": True}
    later = {"record_type": "failure", "run_mode": "registered", "case_id": 7, "took": False}
    path.write_text(json.dumps(first) + "\n" + json.dumps(later) + "\n", encoding="utf-8")
    conflict_refused = False
    try:
        S.registered_rows(path, "case", "case_id")
    except ValueError as exc:
        conflict_refused = "case_id='7'" in str(exc) and "conflicting terminal rows" in str(exc)
    first_copy = dict(first, run_id="second-process", timestamp="later")
    path.write_text(json.dumps(first) + "\n" + json.dumps(first_copy) + "\n", encoding="utf-8")
    same_rows = S.registered_rows(path, "case", "case_id")
    check("M2 conflicting terminal rows are named and refused; identical repeats keep first",
          conflict_refused and len(same_rows) == 1 and same_rows[0]["took"] is True)

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

check("M7 both Arm C loops share the pre-load limit guard",
      inspect.getsource(R.run_arm_c).count("arm_c_limit_reached(selected_count, limit)") >= 2)
check("M5 reports failure rows before a restoration-triggered process stop",
      inspect.getsource(R.run_arm_a).index('"record_type": "failure"')
      < inspect.getsource(R.run_arm_a).index('raise RuntimeError("editor failure left no verified restoration'))

print("TEST_BUILD", "FAIL" if fails else "ALL PASS", f"({fails} failed)")
sys.exit(1 if fails else 0)

