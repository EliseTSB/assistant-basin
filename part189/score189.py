#!/usr/bin/env python3
"""Part 148 scoring with Part 189's registered token-prefix Gate 2."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part148 import score148
from part189.gate189 import gate_result_189, hf_tokenize


def _remove_model(argv: list[str]) -> tuple[str, list[str]]:
    model = None
    remaining = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == "--model":
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit("score189.py: --model requires a PATH")
            model = argv[index + 1]
            index += 2
        elif value.startswith("--model="):
            model = value.split("=", 1)[1]
            if not model:
                raise SystemExit("score189.py: --model requires a PATH")
            index += 1
        else:
            remaining.append(value)
            index += 1
    if model is None:
        raise SystemExit("score189.py: --model is required")
    return model, remaining


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    model, scoring_argv = _remove_model(argv)
    tokenize = hf_tokenize(model)
    gate = lambda answers, units, previous: gate_result_189(
        answers, units, previous, tokenize
    )
    with patch.object(score148, "_gate_result", gate):
        return score148.main(scoring_argv)


if __name__ == "__main__":
    main()
