#!/usr/bin/env python3
"""Run one Part 24 stage on eager attention and keep a checked load record.

Device usage: run194.py {reach,assay,samples} [part24.py options]. Host usage:
run194.py --check-record results/part194/load.json --expect reach,assay,samples
run194.py --score-core results/part194
run194.py --score-24b results/part194
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import median
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EAGER = "eager"
STAGES = ("reach", "assay", "samples")
STAGE_OUTPUTS = {"reach": "reach.json", "assay": "assay.json", "samples": "samples.json"}
MAX_BATCH = 8


class KernelGateFailure(Exception):
    """Raised inside Part 24's loader path when its effective kernel is not eager."""


def _effective_kernel(model) -> str | None:
    """Return only what the loaded model config says was set; never infer from the request."""
    config = getattr(model, "config", None)
    value = getattr(config, "_attn_implementation", None) if config is not None else None
    return value if isinstance(value, str) and value.strip() else None


def _empty_record() -> dict:
    return {"part194": {"requested_attention": EAGER, "source": "Part 24 config._attn_implementation"},
            "loads": []}


def _read_record(path: Path) -> dict:
    if not path.exists():
        return _empty_record()
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read load record {path}: {exc}") from exc
    if not isinstance(record, dict) or not isinstance(record.get("loads"), list):
        raise ValueError(f"{path} is malformed; preserve it and use a fresh --out")
    return record


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _append_entry(path: Path, entry: dict) -> None:
    record = _read_record(path)
    record["loads"].append(entry)
    _write_record(path, record)


def _update_entry(path: Path, entry_id: str, **updates) -> None:
    record = _read_record(path)
    for entry in record["loads"]:
        if isinstance(entry, dict) and entry.get("entry_id") == entry_id:
            entry.update(updates)
            _write_record(path, record)
            return
    raise ValueError(f"load record lost invocation {entry_id}")


def _batch_plan(stage: str, args) -> dict:
    if stage == "reach":
        return {"forward_batch_size": 1, "questions": 6, "k_values_per_question": 4}
    if stage == "assay":
        return {"forward_batch_size": min(args.batch_size, MAX_BATCH), "sequences_padded_within_batch": True}
    return {"sequences_per_generate": min(args.samples, MAX_BATCH), "max_new_tokens": args.max_new}


def _kernel_values(record: dict) -> list[str | None]:
    values: list[str | None] = []
    for invocation in record.get("loads", []):
        model_loads = invocation.get("model_loads") if isinstance(invocation, dict) else None
        if not isinstance(model_loads, list) or not model_loads:
            values.append(None)
            continue
        for loaded in model_loads:
            values.append(loaded.get("attn_implementation") if isinstance(loaded, dict) else None)
    return values


def check_record(path: str | Path, expect: list[str] | None = None) -> int:
    """Host gate. All saved model configs must say eager and each expected stage must be complete."""
    record_path = Path(path)
    try:
        record = _read_record(record_path)
    except ValueError as exc:
        print(f"PART194 RECORD FAILED: {exc}. Copy the whole job-folder results\\part194 back first.", flush=True)
        return 3
    entries = record.get("loads", [])
    kernels = _kernel_values(record)
    if not kernels:
        print("PART194 RECORD FAILED: no model load recorded", flush=True)
        return 3
    missing = [index for index, value in enumerate(kernels) if not value]
    if missing:
        print(f"PART194 RECORD FAILED: missing config attention implementation in load(s): {missing!r}", flush=True)
        return 3
    bad = [(index, value) for index, value in enumerate(kernels) if value != EAGER]
    if bad:
        print(f"PART194 RECORD FAILED: non-eager model load(s) recorded: {bad!r}", flush=True)
        return 3
    incomplete = [entry.get("stage") for entry in entries
                  if isinstance(entry, dict) and entry.get("status") != "complete"]
    if incomplete:
        print(f"PART194 RECORD FAILED: incomplete invocation(s): {incomplete!r}", flush=True)
        return 3

    if expect:
        unknown = [stage for stage in expect if stage not in STAGES]
        if unknown:
            print(f"PART194 RECORD FAILED: unknown expected stage(s): {unknown!r}", flush=True)
            return 2
        seen = {entry.get("stage") for entry in entries
                if isinstance(entry, dict) and entry.get("status") == "complete"
                and isinstance(entry.get("model_loads"), list) and entry.get("model_loads")}
        missing_stages = [stage for stage in expect if stage not in seen]
        if missing_stages:
            print(f"PART194 RECORD FAILED: no recorded entry for stage(s) {missing_stages!r}", flush=True)
            return 3
        for stage in expect:
            expected_file = STAGE_OUTPUTS[stage]
            expected_path = record_path.parent / expected_file
            expected_bytes = expected_path.stat().st_size if expected_path.is_file() else None
            complete = [entry for entry in entries if isinstance(entry, dict)
                        and entry.get("stage") == stage and entry.get("status") == "complete"]
            if not any(any(isinstance(output, dict) and output.get("path") == expected_file
                           and expected_bytes is not None and output.get("bytes") == expected_bytes
                           for output in entry.get("output_files", [])) for entry in complete):
                print(f"PART194 RECORD FAILED: {stage} has no recorded {expected_file}", flush=True)
                return 3
    print("PART194 record ok: all recorded model loads are eager" +
          (f", stages present: {', '.join(expect)}" if expect else ""), flush=True)
    return 0


def _score_core(folder: str | Path) -> int:
    """Score the activation and fixed-answer predictions without reusing Part 24's old hand labels."""
    root = Path(folder)
    try:
        def require_metric(value, name: str):
            if value is None:
                raise ValueError(f"{name} is null")
            return value

        reach = json.loads((root / "reach.json").read_text(encoding="utf-8"))
        assay = json.loads((root / "assay.json").read_text(encoding="utf-8"))
        summary = reach["summary"]
        low_values = [require_metric(row["max_relative_change"], "24.1 max relative change")
                      for row in summary["24_1_max_relative_change_layers_at_or_below_target"].values()]
        low = max(low_values)
        identity = require_metric(
            summary["identity-50"]["2.0"]["median_relative_change_layers_above_target"],
            "24.2 identity median")
        random = require_metric(
            summary["random-50"]["2.0"]["median_relative_change_layers_above_target"],
            "24.2 random median")
        retention = require_metric(summary["24_3_identity_median_name_retention_at_k_2"]["median"],
                                   "24.3 identity median name retention")
        random_retention_values = [unit["name_retention_by_k"]["2.0"] for unit in reach["units"]
                                   if "random-50" in unit["groups"] and unit["layer"] > reach["layer"]
                                   and unit["name_retention_by_k"].get("2.0") is not None]
        random_retention = require_metric(
            median(random_retention_values) if random_retention_values else None,
            "24.3 random median name retention")
        alibaba = assay["self_alibaba_cloud_at_k_0"]
        identity_logprob = require_metric(alibaba["identity-50"]["mean_per_token_logprob"],
                                          "24.4 identity per-token log probability")
        random_logprob = require_metric(alibaba["random-50"]["mean_per_token_logprob"],
                                        "24.4 random per-token log probability")
        none_logprob = require_metric(alibaba["none"]["mean_per_token_logprob"],
                                     "24.4 none per-token log probability")
        extra_drop = random_logprob - identity_logprob
        effects = assay["margins_and_effects"]["self"]

        lines = []
        lines.append(f"24.1 max relative change, layers <= 22, any k: {low}  -> {'held' if low < 1e-3 else 'failed'}")
        lines.append(f"24.2 k=2 identity {identity:.3f} vs random {random:.3f}, ratio {identity / random if random else float('inf'):.2f}  -> "
                     f"{'held' if identity <= 1.5 * random else 'failed'}")
        lines.append(f"24.3 identity median name retention at k=2 {retention:.3f} (random {random_retention:.3f})  -> "
                     f"{'held' if retention >= 0.8 else 'failed'}")
        lines.append(f"24.4 Alibaba Cloud per token: identity {identity_logprob:.3f}, random {random_logprob:.3f}, "
                     f"none {none_logprob:.3f}; extra drop {extra_drop:.3f}  -> "
                     f"{'held' if extra_drop >= 0.3 else 'failed'}")
        additive = True
        for k in ("1.0", "2.0"):
            none = require_metric(effects["none"][k]["effect"]["mean"], f"24.5 none effect at k={k}")
            identity_effect = require_metric(effects["identity-50"][k]["effect"]["mean"],
                                             f"24.5 identity effect at k={k}")
            random_effect = require_metric(effects["random-50"][k]["effect"]["mean"],
                                           f"24.5 random effect at k={k}")
            difference = abs(identity_effect - random_effect)
            additive = additive and difference < none / 4
            lines.append(f"24.5 k={k}: none {none:.3f}, identity {identity_effect:.3f}, random {random_effect:.3f}; "
                         f"|difference| {difference:.3f} vs quarter {none / 4:.3f}")
        lines.append(f"24.5 additive  -> {'held' if additive else 'failed'}")
        lines.append("24.6 and 24.7 are not scored here: their published verdicts used the original hand labels.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"PART194 SCORE FAILED: {exc}", flush=True)
        return 2
    print("\n".join(lines))
    return 0


def _reader_cells(rows: list[dict]) -> dict[tuple[str, float], tuple[int, int, int, int]]:
    """Count (a only, b only, both, neither) for each blind-reader condition."""
    if len(rows) != 192:
        raise ValueError(f"24b reader must have 192 answers, found {len(rows)}")
    cells: dict[tuple[str, float], list[dict]] = {}
    for row in rows:
        if not isinstance(row, dict) or type(row.get("a")) is not bool or type(row.get("b")) is not bool:
            raise ValueError("24b reader rows are incomplete or errored")
        key = (row["zero_condition"], float(row["k"]))
        cells.setdefault(key, []).append(row)
    counts = {}
    for zero in ("none", "identity-50", "random-50"):
        for k in (0.0, 1.0, 1.5, 2.0):
            selected = cells.get((zero, k), [])
            if len(selected) != 16:
                raise ValueError(f"24b cell {zero} at k={k:g} has {len(selected)} rows, expected 16")
            a_only = sum(row["a"] and not row["b"] for row in selected)
            b_only = sum(row["b"] and not row["a"] for row in selected)
            both = sum(row["a"] and row["b"] for row in selected)
            counts[(zero, k)] = (a_only, b_only, both, 16 - a_only - b_only - both)
    return counts


def _score_24b(folder: str | Path, published_path: str | Path | None = None) -> int:
    """Score the rerun's blind reader and compare it with the saved Part 24b labels."""
    root = Path(folder)
    baseline_path = Path(published_path) if published_path is not None else ROOT / "results/part24/reread-24b.json"
    try:
        current_doc = json.loads((root / "reread-24b.json").read_text(encoding="utf-8"))
        baseline_doc = json.loads(baseline_path.read_text(encoding="utf-8"))
        if current_doc.get("reader") != baseline_doc.get("reader") or current_doc.get("prompt") != baseline_doc.get("prompt"):
            raise ValueError("24b reader id or prompt differs from the published reader")
        current = _reader_cells(current_doc["rows"])
        baseline = _reader_cells(baseline_doc["rows"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"PART194 24b SCORE FAILED: {exc}", flush=True)
        return 2

    a, b, both, neither = current[("identity-50", 2.0)]
    base_a, base_b, base_both, _ = baseline[("identity-50", 2.0)]
    identity_b = b + both
    baseline_identity_b = base_b + base_both
    print(f"24b.1 identity k=2 with (b): {identity_b}/16 (published {baseline_identity_b}/16)  ->",
          "held" if identity_b > 8 else "failed")

    a, b, both, neither = current[("identity-50", 1.5)]
    base_a, base_b, base_both, _ = baseline[("identity-50", 1.5)]
    alien, base_alien = a + b + both, base_a + base_b + base_both
    share = a / alien if alien else None
    share_text = f"{share:.3f}" if share is not None else "undefined"
    print(f"194.7 24b.2 identity k=1.5 (a)-only share among alien: {share_text} ({a}/{alien}; "
          f"published {base_a}/{base_alien})")
    print(f"24b.2 identity k=1.5, (a) only among alien: {a}/{alien} (published {base_a}/{base_alien})  ->",
          "held" if alien and a > alien / 2 else "failed")

    alien2 = {zero: 16 - current[(zero, 2.0)][3] for zero in ("none", "identity-50", "random-50")}
    base_alien2 = {zero: 16 - baseline[(zero, 2.0)][3] for zero in ("none", "identity-50", "random-50")}
    print(f"24b.3 k=2 (a or b), none/identity/random: {alien2} (published {base_alien2})  ->",
          "held" if all(value >= 14 for value in alien2.values()) else "failed")

    doses = (0.0, 1.0, 1.5, 2.0)
    counts4 = {zero: [current[(zero, k)][1] + current[(zero, k)][2] for k in doses]
               for zero in ("none", "random-50")}
    baseline_counts4 = {zero: [baseline[(zero, k)][1] + baseline[(zero, k)][2] for k in doses]
                        for zero in ("none", "random-50")}
    print("194.9 24b.4 (b) counts for none/random at k=0,1,1.5,2: "
          f"none={[f'{count}/16' for count in counts4['none']]}; "
          f"random={[f'{count}/16' for count in counts4['random-50']]}; "
          f"published none={[f'{count}/16' for count in baseline_counts4['none']]}; "
          f"random={[f'{count}/16' for count in baseline_counts4['random-50']]}")

    ok4 = True
    for zero in ("none", "random-50"):
        for index, k in enumerate(doses):
            count = counts4[zero][index]
            base_count = baseline_counts4[zero][index]
            print(f"24b.4 (b), {zero} k={k:g}: {count}/16 (published {base_count}/16)")
            ok4 = ok4 and count <= 2
    print("24b.4 (b) at most 2/16 in none/random at every k  ->", "held" if ok4 else "failed")
    print("Published hand-label agreement (187/192) is not recalculated: the rerun uses fresh answers.")
    return 0


def _parse_out_arg(args: list[str]) -> Path | None:
    for index, arg in enumerate(args):
        if arg == "--out" and index + 1 < len(args):
            return Path(args[index + 1])
        if arg.startswith("--out="):
            return Path(arg.split("=", 1)[1])
    return None


def _replace_option(args: list[str], option: str, value: int) -> list[str]:
    clean: list[str] = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == option:
            index += 2
            continue
        if arg.startswith(option + "="):
            index += 1
            continue
        clean.append(arg)
        index += 1
    clean.extend([option, str(value)])
    return clean


def main(argv: list[str] | None = None, *, part24_module=None, loader=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "--check-record":
        if len(args) == 2:
            return check_record(args[1])
        if len(args) == 4 and args[2] == "--expect":
            expect = [stage.strip() for stage in args[3].split(",") if stage.strip()]
            return check_record(args[1], expect)
        print("usage: run194.py --check-record <load.json> [--expect reach,assay,samples]", file=sys.stderr)
        return 2
    if args and args[0] == "--score-core":
        return _score_core(args[1]) if len(args) == 2 else 2
    if args and args[0] == "--score-24b":
        return _score_24b(args[1]) if len(args) == 2 else 2
    if not args or args[0] not in STAGES:
        print("usage: run194.py {reach,assay,samples} [Part 24 options]", file=sys.stderr)
        return 2

    stage, forwarded = args[0], args[1:]
    if part24_module is None:
        import part24 as part24_module
    if _parse_out_arg(forwarded) is None:
        forwarded.extend(["--out", "results/part194"])
    parsed = part24_module.parser().parse_args([stage, *forwarded])
    if parsed.command != stage or parsed.steps:
        print("run exactly one Part 24 stage per launch", file=sys.stderr)
        return 2

    if stage == "assay" and parsed.batch_size > MAX_BATCH:
        forwarded = _replace_option(forwarded, "--batch-size", MAX_BATCH)
        parsed = part24_module.parser().parse_args([stage, *forwarded])
    if stage == "samples" and parsed.samples > MAX_BATCH:
        forwarded = _replace_option(forwarded, "--samples", MAX_BATCH)
        parsed = part24_module.parser().parse_args([stage, *forwarded])
    import neuron_report as nr

    output_dir = Path(parsed.out)
    record_path = output_dir / "load.json"
    entry_id = uuid.uuid4().hex
    entry_created = False
    observed_shapes: dict[tuple[int, int], int] = {}
    generate_calls: list[dict] = []
    hook_handles = []
    generate_restores = []
    original_loader = nr.load_model
    original_dump = nr._dump
    actual_loader = original_loader if loader is None else loader

    def record_shape(_model, positional, keyword):
        tensor = keyword.get("input_ids")
        if tensor is None:
            tensor = keyword.get("inputs_embeds")
        if tensor is None:
            tensor = next((value for value in positional if hasattr(value, "shape")), None)
        shape = getattr(tensor, "shape", None)
        if shape is not None and len(shape) >= 2:
            key = (int(shape[0]), int(shape[1]))
            observed_shapes[key] = observed_shapes.get(key, 0) + 1

    def recorded_loader(model_id: str, revision: str | None, device: str):
        nonlocal entry_created
        tok, model = actual_loader(model_id, revision, device, attn_implementation=EAGER)
        kernel = _effective_kernel(model)
        load = {"model": model_id, "revision": revision, "device": device,
                "requested_attn_implementation": EAGER, "attn_implementation": kernel}
        entry = {"entry_id": entry_id, "stage": stage, "status": "running",
                 "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "model_loads": [load], "batch_plan": _batch_plan(stage, parsed),
                 "batch_shapes": [], "generate_calls": [], "output_files": [], "argv": [stage, *forwarded]}
        _append_entry(record_path, entry)
        entry_created = True
        # Save the effective config value in the shared record too; its loader helper can fall back to the request.
        nr.LAST_LOAD = {**load}
        if kernel != EAGER:
            _update_entry(record_path, entry_id, status="refused",
                          finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            raise KernelGateFailure(f"effective attention kernel is {kernel!r}; expected 'eager'")
        hook_handles.append(model.register_forward_pre_hook(record_shape, with_kwargs=True))
        if stage == "samples":
            original_generate = model.generate
            instance_dict = getattr(model, "__dict__", {})
            had_instance_generate = "generate" in instance_dict
            previous_instance_generate = instance_dict.get("generate")

            def tracked_generate(*positional, **keyword):
                generate_calls.append({
                    "num_return_sequences": keyword.get("num_return_sequences"),
                    "max_new_tokens": keyword.get("max_new_tokens"),
                })
                return original_generate(*positional, **keyword)

            model.generate = tracked_generate
            generate_restores.append((model, had_instance_generate, previous_instance_generate))
        return tok, model

    def tracked_dump(value, path):
        result = original_dump(value, path)
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(output_dir.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError(f"Part 24 wrote outside its output directory: {path}") from exc
        output = {"path": relative, "bytes": resolved.stat().st_size}
        record = _read_record(record_path)
        for invocation in record["loads"]:
            if invocation.get("entry_id") == entry_id:
                invocation.setdefault("output_files", []).append(output)
                _write_record(record_path, record)
                break
        return result

    nr.load_model = recorded_loader
    nr._dump = tracked_dump
    try:
        part24_module.main([stage, *forwarded])
    except KernelGateFailure as exc:
        print(f"PART194 GATE FAILED: {exc}", flush=True)
        return 3
    except BaseException as exc:
        if entry_created:
            try:
                _update_entry(record_path, entry_id, status="failed", error=repr(exc)[:300],
                              finished=time.strftime("%Y-%m-%d %H:%M:%S"))
            except Exception:
                pass
        raise
    finally:
        for handle in hook_handles:
            try:
                handle.remove()
            except Exception:
                pass
        for model, had_instance_generate, previous_instance_generate in generate_restores:
            if had_instance_generate:
                model.generate = previous_instance_generate
            else:
                delattr(model, "generate")
        nr.load_model = original_loader
        nr._dump = original_dump

    if not entry_created:
        print("PART194 GATE FAILED: Part 24 did not load a model", flush=True)
        return 3
    shapes = [{"shape": [batch, sequence], "calls": calls}
              for (batch, sequence), calls in sorted(observed_shapes.items())]
    _update_entry(record_path, entry_id, status="complete", batch_shapes=shapes,
                  generate_calls=generate_calls,
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    return check_record(record_path, [stage])


if __name__ == "__main__":
    raise SystemExit(main())
