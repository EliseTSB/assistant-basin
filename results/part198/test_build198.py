#!/usr/bin/env python3
"""Tiny inert checks for Part 198 run, control, gate, scoring and resume helpers."""
from __future__ import annotations

import json
import sys
import tempfile
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

print("TEST_BUILD", "FAIL" if fails else "ALL PASS", f"({fails} failed)")
sys.exit(1 if fails else 0)

