"""score64: a non-finite activation fails its row AND the scored file can still be written (2026-10-24)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from part64 import score64  # noqa: E402


class FakeScorer:
    """Fills every activation with 1.0 except entry 0, which gets -inf (an empty max on the device)."""

    def score_batch(self, entries, rows, maximum, marked, batch_size=8):
        maximum[:] = 1.0
        maximum[0, :] = -np.inf


def _row(arm: str, key: str) -> dict:
    return {"arm": arm, "unit_key": key, "type": "neuron", "parse_status": "ok",
            **{tier: [f"{tier} sentence {i} for {key} in {arm}." for i in range(4)] for tier in score64.TIERS}}


def test_non_finite_row_fails_and_serializes(tmp_path):
    specs = [{"unit_key": "L3:u7", "record_index": 1, "type": "neuron", "layer": 3, "unit_index": 7,
              "control_id": None}]
    rows = [_row(arm, "L3:u7") for arm in score64.ARMS]
    scored = score64.score_rows(rows, specs, {"L3:u7": 0.5}, FakeScorer())
    first = scored[0]
    assert first["score_status"] == "failed" and first["pass"] is False
    assert first["activations"]["strong"][0] is None and first["non_finite"]["strong"][0] == "-inf"
    assert all(r["score_status"] == "ok" for r in scored[1:])
    out = tmp_path / "scored.jsonl"
    score64._write_jsonl(out, scored)                      # raised ValueError before the fix
    back = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(back) == 4 and back[0]["non_finite"]["strong"][0] == "-inf"
    assert not any(isinstance(v, float) and not math.isfinite(v)
                   for r in back for tier in score64.TIERS for v in r["activations"][tier] if v is not None)
