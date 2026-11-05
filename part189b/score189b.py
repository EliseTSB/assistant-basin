#!/usr/bin/env python3
"""Run unchanged Part 148 scoring with Gate 2 supplied by the SDPA wiring run."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part189b.run189b import _extract_option, _load_wiring


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    wiring_path, scoring_argv = _extract_option(argv, "--wiring-gates")
    if not wiring_path:
        raise SystemExit("score189b.py: --wiring-gates PATH is required")
    try:
        _document, wiring_gate = _load_wiring(wiring_path)
    except ValueError as exc:
        raise SystemExit(f"score189b.py: {exc}") from exc

    from part148 import score148

    original_gate_result = score148._gate_result

    def gate_result(answers, units, part112_answers):
        gates = original_gate_result(answers, units, part112_answers)
        gates["gate2_pilot_matches_part112"] = dict(wiring_gate)
        gates["all_passed"] = (
            gates["gate1_zero_matches_plain"]["passed"]
            and wiring_gate.get("passed") is True
            and gates["gate3_plain_not_moved"]["passed"]
        )
        return gates

    with patch.object(score148, "_gate_result", gate_result):
        return score148.main(scoring_argv)


if __name__ == "__main__":
    main()
