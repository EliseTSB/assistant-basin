#!/usr/bin/env python3
"""Exploratory diagnostic after Part 189's Gate 2 failed (17 of 32 at batch 8, eager): rerun the pilot with a chosen
attention kernel and batch shape, record the load and both Gate 2 counts, and never stop on a gate. Not registered;
its answers are not Part 189 data. Written by Claude (Opus 5.5), 2026-11-04."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part189.run189 import _option


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    attn = _option(argv, "--attn")
    if attn not in ("eager", "sdpa"):
        raise SystemExit("diag189.py: --attn eager|sdpa is required")
    i = argv.index("--attn") if "--attn" in argv else None
    argv = argv[:i] + argv[i + 2:] if i is not None else [a for a in argv if not a.startswith("--attn=")]
    output = Path(_option(argv, "--out"))
    import neuron_report as nr
    from part148 import battery148, score148
    from part189.gate189 import gate_result_189, hf_tokenize
    original = nr.load_model

    def load(*args, **kwargs):
        args = list(args)
        if len(args) >= 4:
            args[3] = attn
            kwargs.pop("attn_implementation", None)
        else:
            kwargs["attn_implementation"] = attn
        return original(*args, **kwargs)

    nr.load_model = load
    rc = battery148.main(argv)
    units = json.loads(Path(_option(argv, "--units")).read_text(encoding="utf-8"))
    gates = gate_result_189(score148.read_jsonl(output), units,
                            score148.read_jsonl(ROOT / "results/part112/answers.jsonl"),
                            hf_tokenize(_option(argv, "--model")))
    rec = {"load": nr.LAST_LOAD, "argv": argv, "battery148_rc": rc, "gates": gates}
    (output.parent / "diag.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DIAG", attn, json.dumps({k: v.get("matches") if isinstance(v, dict) else v for k, v in gates.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
