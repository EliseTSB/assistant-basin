#!/usr/bin/env python3
"""Run Part 148's battery with eager attention and stop on a failed Part 148 gate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _option(argv: list[str], name: str, default: str | None = None) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return default


def _force_eager(nr) -> None:
    """Make the existing Part 148 shared-loader call explicitly request eager."""
    original = nr.load_model

    def eager_load(*args, **kwargs):
        positional = list(args)
        if len(positional) >= 4:
            positional[3] = "eager"
            kwargs.pop("attn_implementation", None)
        else:
            kwargs["attn_implementation"] = "eager"
        return original(*positional, **kwargs)

    nr.load_model = eager_load


def _record_gates(output: Path, argv: list[str]) -> dict:
    from part148 import score148
    from part189.gate189 import gate_result_189, hf_tokenize

    units_path = Path(_option(argv, "--units", str(ROOT / "results/part112/units.json")))
    previous_path = ROOT / "results/part112/answers.jsonl"
    model_path = _option(argv, "--model")
    if not model_path:
        raise SystemExit("run189.py: --model is required to evaluate the prefix gate")
    units = json.loads(units_path.read_text(encoding="utf-8"))
    gates = gate_result_189(
        score148.read_jsonl(output),
        units,
        score148.read_jsonl(previous_path),
        hf_tokenize(model_path),
    )
    gate_path = output.parent / "gates.json"
    gate_path.write_text(json.dumps(gates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return gates


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    output_arg = _option(argv, "--out")
    if not output_arg:
        raise SystemExit("run189.py: --out is required")
    output = Path(output_arg)

    import neuron_report as nr
    from part148 import battery148

    _force_eager(nr)
    run_rc = battery148.main(argv)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "load": nr.LAST_LOAD,
        "part189": {"reruns": "Part 148 battery", "argv": argv, "battery148_rc": run_rc},
    }
    (output.parent / "load.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    kernel = (nr.LAST_LOAD or {}).get("attn_implementation")
    if kernel != "eager":
        print(f"PART189 GATE FAILED: loaded attention kernel is {kernel!r}", flush=True)
        return 3

    gates = _record_gates(output, argv)
    if not gates.get("all_passed"):
        print(f"PART189 GATE FAILED: Part 148 gates {gates}", flush=True)
        return 3
    print("PART189 gates passed: eager kernel and Part 148 controls", flush=True)
    return int(run_rc or 0) if isinstance(run_rc, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
