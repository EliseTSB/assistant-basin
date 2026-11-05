#!/usr/bin/env python3
"""Gate Part 148's eager run on its unchanged full-text SDPA wiring pilot."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from part189.run189 import _force_eager, _option


def _extract_option(argv: list[str], name: str) -> tuple[str | None, list[str]]:
    """Remove one ``--name PATH`` or ``--name=PATH`` option from argv."""
    result = []
    found = None
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == name:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise SystemExit(f"run189b.py: {name} requires a PATH")
            if found is not None:
                raise SystemExit(f"run189b.py: {name} may be specified only once")
            found = argv[index + 1]
            index += 2
        elif value.startswith(name + "="):
            if found is not None:
                raise SystemExit(f"run189b.py: {name} may be specified only once")
            found = value.split("=", 1)[1]
            if not found:
                raise SystemExit(f"run189b.py: {name} requires a PATH")
            index += 1
        else:
            result.append(value)
            index += 1
    return found, result


def _load_wiring(path: str | Path) -> tuple[dict, dict]:
    """Return the wiring document and its registered full-text Gate 2."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"wiring gates file does not exist: {p}")
    try:
        document = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read wiring gates file {p}: {exc}") from exc
    gates = document.get("gates", document) if isinstance(document, dict) else {}
    gate2 = gates.get("gate2_pilot_matches_part112") if isinstance(gates, dict) else None
    if not isinstance(gate2, dict):
        raise ValueError(f"wiring gates file {p} has no Gate 2 result")
    return document, gate2


def _read_run_inputs(argv: list[str], output: Path):
    from part148 import score148

    units_path = Path(_option(argv, "--units", str(ROOT / "results/part112/units.json")))
    units = json.loads(units_path.read_text(encoding="utf-8"))
    answers = score148.read_jsonl(output)
    previous = score148.read_jsonl(ROOT / "results/part112/answers.jsonl")
    return score148, units, answers, previous


def _output_path(argv: list[str]) -> Path:
    output_arg = _option(argv, "--out")
    if not output_arg:
        raise SystemExit("run189b.py: --out is required")
    return Path(output_arg)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_wiring(argv: list[str]) -> int:
    from part148 import battery148

    output = _output_path(argv)
    import neuron_report as nr

    original = nr.load_model

    def sdpa_load(*args, **kwargs):
        positional = list(args)
        if len(positional) >= 4:
            positional[3] = "sdpa"
            kwargs.pop("attn_implementation", None)
        else:
            kwargs["attn_implementation"] = "sdpa"
        return original(*positional, **kwargs)

    nr.load_model = sdpa_load
    run_rc = battery148.main(argv)
    output.parent.mkdir(parents=True, exist_ok=True)
    score148, units, answers, previous = _read_run_inputs(argv, output)
    gates = score148._gate_result(answers, units, previous)
    record = {
        "load": nr.LAST_LOAD,
        "part189b": {"mode": "wiring", "argv": argv, "battery148_rc": run_rc},
        "gates": gates,
    }
    _write_json(output.parent / "wiring.json", record)

    kernel = (nr.LAST_LOAD or {}).get("attn_implementation")
    if kernel != "sdpa":
        print(f"PART189B WIRING FAILED: loaded attention kernel is {kernel!r}", flush=True)
        return 3
    if not gates["gate2_pilot_matches_part112"].get("passed"):
        print(f"PART189B WIRING FAILED: full-text Gate 2 {gates['gate2_pilot_matches_part112']}", flush=True)
        return 3
    print("PART189B wiring gate passed: SDPA kernel and full-text Gate 2", flush=True)
    return int(run_rc or 0) if isinstance(run_rc, int) else 0


def _run_eager(argv: list[str]) -> int:
    wiring_path, argv = _extract_option(argv, "--wiring-gates")
    if not wiring_path:
        print("PART189B MAIN REFUSED: --wiring-gates PATH is required", file=sys.stderr, flush=True)
        return 3
    try:
        _wiring_document, wiring_gate = _load_wiring(wiring_path)
    except ValueError as exc:
        print(f"PART189B MAIN REFUSED: {exc}", file=sys.stderr, flush=True)
        return 3
    if not wiring_gate.get("passed"):
        print("PART189B MAIN REFUSED: wiring Gate 2 did not pass", file=sys.stderr, flush=True)
        return 3

    output = _output_path(argv)
    import neuron_report as nr
    from part148 import battery148
    from part189.gate189 import gate_result_189, hf_tokenize

    _force_eager(nr)
    run_rc = battery148.main(argv)
    output.parent.mkdir(parents=True, exist_ok=True)
    load_record = {
        "load": nr.LAST_LOAD,
        "part189b": {"mode": "eager", "argv": argv, "battery148_rc": run_rc,
                     "wiring_gates": str(wiring_path)},
    }
    _write_json(output.parent / "load.json", load_record)

    score148, units, answers, previous = _read_run_inputs(argv, output)
    measured = score148._gate_result(answers, units, previous)
    model_path = _option(argv, "--model")
    if not model_path:
        raise SystemExit("run189b.py: --model is required to record the eager prefix Gate 2")
    eager_prefix = gate_result_189(answers, units, previous, hf_tokenize(model_path))
    eager_kernel = (nr.LAST_LOAD or {}).get("attn_implementation") == "eager"
    gates = {
        "load_eager": {"passed": eager_kernel, "attn_implementation": (nr.LAST_LOAD or {}).get("attn_implementation")},
        "gate1_zero_matches_plain": measured["gate1_zero_matches_plain"],
        "gate2_pilot_matches_part112": wiring_gate,
        "gate3_plain_not_moved": measured["gate3_plain_not_moved"],
        "gate2_eager_prefix_descriptive": eager_prefix["gate2_pilot_matches_part112"],
        "gate2_eager_full_text_descriptive": measured["gate2_pilot_matches_part112"],
    }
    gates["all_passed"] = (
        eager_kernel
        and gates["gate1_zero_matches_plain"].get("passed")
        and gates["gate2_pilot_matches_part112"].get("passed")
        and gates["gate3_plain_not_moved"].get("passed")
    )
    _write_json(output.parent / "gates.json", gates)
    if not gates["all_passed"]:
        print(f"PART189B GATES FAILED: {gates}", flush=True)
        return 3
    print("PART189B gates passed: eager load, wiring Gate 2, and eager controls", flush=True)
    return int(run_rc or 0) if isinstance(run_rc, int) else 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    wiring = False
    if "--wiring" in argv:
        if argv.count("--wiring") != 1:
            raise SystemExit("run189b.py: --wiring may be specified only once")
        argv.remove("--wiring")
        wiring = True
    return _run_wiring(argv) if wiring else _run_eager(argv)


if __name__ == "__main__":
    raise SystemExit(main())
