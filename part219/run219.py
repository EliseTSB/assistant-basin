#!/usr/bin/env python3
"""run219.py -- Part 219's orchestrator: R0 to R10 on PART219-DESIGN.md's Runs table.

Two processes, as the design's Optimization section takes them (one model pair per process, the 11-08 gotcha):

    stage a   R0, R0b, R2, R3, R4, R5, R6, R7   one host load, no donor
    stage b   R1, R8, R9, R10                   one host load and one donor load

Every run is Part 99's `swap99.py` with this part's additive options: `--attn-implementation eager`,
`--ablate-heads`, `--transplant-heads`/`--transplant-mode`, `--reference-replay` and `--record-token-ids`. The
orchestrator writes `manifest.json` beside the runs (a run whose receipt fails is marked unreadable there), checks
R0b against R0 token for token, and checks every reset receipt against R0's stored replay.

    --pilot   runs R0 and R2 only and applies the design's pilot stopping rule (see pilot_verdict below).

Launch (A770, from the repo root, `tools/launch_arc.py 219 --copy ...` so the job folder has every module):

    python tools/launch_arc.py 219 --copy part219/run219.py part219/score219.py part99/swap99.py \
        part92/common92.py part85/checkpoints.py part101/read101.py tools/part_check.py \
        --step "-I -B part219\\run219.py --host <Instruct snapshot> --neurons results\\part92\\step2\\neurons.json \
                --kinds results\\part92\\step2\\kinds.json --pushes results\\part99-pushes.pt --stage a --device xpu" \
        --step "-I -B part219\\run219.py --host <Instruct snapshot> --donor <mid-training-end snapshot> ... --stage b"

A progress line is printed at least every eight seconds for as long as any run is going.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _path in (str(REPO), str(REPO / "part92"), str(REPO / "part99"), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import common92 as c  # noqa: E402
import swap99  # noqa: E402

PART = 219
IMPORTANT_HEADS = ("L0H10", "L1H4", "L2H6", "L3H5")
FROZEN_HEADS = ("L0H12", "L1H11", "L2H5", "L3H9")
FROZEN_ORDINARY_DROPS = {"L0H12": 0.0103, "L1H11": 0.0116, "L2H5": -0.0005, "L3H9": 0.0045}
FROZEN_SEED = 219
ELIGIBLE_ORDINARY_DROP = 0.035
HEADS_PER_LAYER = 16
HEAD_LAYERS = (0, 1, 2, 3)
BATCH = 48
FRAME = "chat"
ROWS_PER_UNIT = 12          # neuron + random, six questions
UNPUSHED_ROWS = 6
GATE_R0_MIN_SLOT = 17       # PART219-DESIGN.md, Gate
GATE_R1_MAX_SLOT = 5
PILOT_R0_MIN_SLOT = 17      # PART219-DESIGN.md, Size/Pilot
PILOT_R2_MIN_COHERENT = 15
HEARTBEAT_SECONDS = 8.0


def parse_head_name(name: str) -> tuple[int, int]:
    match = swap99.HEAD_PATTERN.match(name.strip())
    if match is None:
        raise ValueError(f"heads are written L<layer>H<head>, such as L0H10; got {name!r}")
    return int(match.group(1)), int(match.group(2))


def registered_frozen_heads() -> list[str]:
    """R5's and R9's heads exactly as PART219-DESIGN.md freezes them: `random.Random(219)`, one ``rng.choice`` per
    layer 0 to 3 over that layer's 15 eligible heads (ordinary-answer drop under 0.035, the four important heads
    excluded). Recomputing here is the assertion: the runner refuses to start if the registered list is not what the
    rule gives."""
    rng = random.Random(FROZEN_SEED)
    chosen = []
    for layer in HEAD_LAYERS:
        excluded = {head for name in IMPORTANT_HEADS
                    for other, head in [parse_head_name(name)] if other == layer}
        eligible = [head for head in range(HEADS_PER_LAYER) if head not in excluded]
        if len(eligible) != HEADS_PER_LAYER - 1:
            raise SystemExit(f"layer {layer}: expected 15 eligible heads, got {len(eligible)}")
        chosen.append((layer, rng.choice(eligible)))
    return [f"L{layer}H{head}" for layer, head in sorted(chosen)]


def assert_registered_heads() -> list[str]:
    computed = registered_frozen_heads()
    if tuple(computed) != FROZEN_HEADS:
        raise SystemExit(
            "the frozen R5/R9 heads recomputed from the registered rule are "
            f"{computed}, not the registered {list(FROZEN_HEADS)}: stop, the rule or the list has moved")
    for name, drop in FROZEN_ORDINARY_DROPS.items():
        if not drop < ELIGIBLE_ORDINARY_DROP:
            raise SystemExit(f"{name}'s registered ordinary-answer drop {drop} is not under {ELIGIBLE_ORDINARY_DROP}")
    return computed


def specs_for_stage(stage: str) -> list[dict]:
    """The design's Runs table, split into the two processes."""
    if stage == "a":
        return [
            {"tag": "R0", "role": "reference"},
            {"tag": "R0b", "role": "do-nothing check", "ablate_heads": "", "transplant_heads": "",
             "transplant_mode": "donor"},
            {"tag": "R2", "role": "219.1, 219.2", "ablate_heads": ",".join(IMPORTANT_HEADS)},
            {"tag": "R3", "role": "exploratory", "ablate_heads": "L1H4"},
            {"tag": "R4", "role": "exploratory", "ablate_heads": "L0H10"},
            {"tag": "R5", "role": "rig check", "ablate_heads": ",".join(FROZEN_HEADS)},
            {"tag": "R6", "role": "exploratory", "ablate_heads": "L2H6"},
            {"tag": "R7", "role": "exploratory", "ablate_heads": "L3H5"},
        ]
    if stage == "b":
        return [
            {"tag": "R1", "role": "gate and comparison", "swap": "attn", "layers": "0-3"},
            {"tag": "R8", "role": "219.3, 219.4", "transplant_heads": ",".join(IMPORTANT_HEADS),
             "transplant_mode": "donor"},
            {"tag": "R9", "role": "coupling check", "transplant_heads": ",".join(FROZEN_HEADS),
             "transplant_mode": "donor"},
            {"tag": "R10", "role": "distance control", "transplant_heads": ",".join(IMPORTANT_HEADS),
             "transplant_mode": "opposite"},
        ]
    raise SystemExit(f"unknown stage {stage!r}")


def pilot_specs() -> list[dict]:
    """The pilot is R0 and R2 in full (PART219-DESIGN.md, Size)."""
    return [spec for spec in specs_for_stage("a") if spec["tag"] in ("R0", "R2")]


def expected_rows(args) -> int:
    """Rows each run should write: every kept unit twice (neuron and random) on six questions, plus the six unpushed."""
    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    if not isinstance(neurons, list):
        raise ValueError("--neurons must contain a JSON list")
    if args.only:
        kinds = swap99.load_kind_map(args.kinds)
        only = swap99.parse_only(args.only)
        neurons = [neuron for neuron in neurons if kinds.get(str(neuron.get("id"))) in only]
    if args.limit is not None:
        neurons = neurons[:args.limit]
    return len(neurons) * ROWS_PER_UNIT + UNPUSHED_ROWS


def swap_args(base, spec: dict, replay_path: Path) -> argparse.Namespace:
    """One swap99 invocation's arguments, with this part's options set per run."""
    return argparse.Namespace(
        host=base.host, donor=base.donor, swap=spec.get("swap", "none"), tokenizer=base.tokenizer,
        neurons=base.neurons, frame=FRAME, tag=spec["tag"], out=str(base.out), device=base.device,
        batch=base.batch, kinds=base.kinds, only=base.only, pushes=base.pushes, limit=base.limit,
        layers=spec.get("layers"), extrapolate=False, questions=None,
        attn_implementation="eager",
        ablate_heads=spec.get("ablate_heads"),
        transplant_heads=spec.get("transplant_heads"),
        transplant_mode=spec.get("transplant_mode"),
        reference_replay=str(replay_path),
        record_token_ids=True,
    )


class Progress:
    """A progress line at least every HEARTBEAT_SECONDS while a stage runs (swap99's own lines are per 240 rows)."""

    def __init__(self, label: str, interval: float = HEARTBEAT_SECONDS):
        self.label = label
        self.interval = interval
        self.started = time.perf_counter()
        self.current = "starting"
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self.stopped.wait(self.interval):
            elapsed = time.perf_counter() - self.started
            print(f"[run219] {self.label}: {self.current} ({elapsed:.0f}s elapsed)", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=2)


def describe(spec: dict) -> dict:
    if spec.get("swap", "none") != "none":
        return {"swap": spec["swap"], "layers": spec.get("layers")}
    if spec.get("transplant_heads"):
        return {"transplant_heads": spec["transplant_heads"].split(","),
                "transplant_mode": spec.get("transplant_mode")}
    if spec.get("ablate_heads") is not None:
        return {"ablate_heads": [name for name in spec["ablate_heads"].split(",") if name]}
    return {"change": "none"}


def read_rows(path: Path) -> dict:
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            rows[(str(row["id"]), row["unit"], int(row["question_index"]))] = row
    return rows


def compare_answers(first: Path, second: Path) -> dict:
    """Token-for-token and answer-for-answer equality of two runs on every item (R0b against R0)."""
    left, right = read_rows(first), read_rows(second)
    missing = sorted(set(left) ^ set(right))
    answer_mismatch, token_mismatch, no_tokens = [], [], 0
    for key in sorted(set(left) & set(right)):
        a, b = left[key], right[key]
        if a.get("answer") != b.get("answer"):
            answer_mismatch.append(key)
        if "token_ids" not in a or "token_ids" not in b:
            no_tokens += 1
        elif a["token_ids"] != b["token_ids"]:
            token_mismatch.append(key)
    return {"items": len(set(left) & set(right)), "missing_items": len(missing),
            "answer_mismatches": len(answer_mismatch), "token_mismatches": len(token_mismatch),
            "items_without_token_ids": no_tokens,
            "identical": not missing and not answer_mismatch and not token_mismatch and not no_tokens,
            "first_mismatch": (answer_mismatch + token_mismatch)[:3]}


def run_checks(out: Path, manifest: dict) -> dict:
    """The design's do-nothing and reset checks, computed from the files the runs wrote."""
    checks = {}
    runs = manifest["runs"]
    if "R0" in runs and "R0b" in runs:
        checks["r0b_equals_r0"] = compare_answers(out / "R0.jsonl", out / "R0b.jsonl")
        if not checks["r0b_equals_r0"]["identical"]:
            problem = "R0b does not match R0 token for token"
            if problem not in runs["R0b"]["problems"]:
                runs["R0b"]["problems"].append(problem)
            runs["R0b"]["readable"] = False
    reference = None
    r0_meta = out / "R0.meta.json"
    if r0_meta.exists():
        reference = json.loads(r0_meta.read_text(encoding="utf-8")).get("reset_receipt")
    reset = {}
    for tag, entry in runs.items():
        meta_path = out / f"{tag}.meta.json"
        if not meta_path.exists() or reference is None:
            continue
        receipt = json.loads(meta_path.read_text(encoding="utf-8")).get("reset_receipt")
        if receipt is None:
            reset[tag] = {"matches_r0": None, "mismatch": "no reset receipt in the meta"}
            if "no reset receipt" not in entry["problems"]:
                entry["problems"].append("no reset receipt")
            entry["readable"] = False
            continue
        mismatch = swap99.compare_replay(receipt, reference)
        reset[tag] = {"matches_r0": mismatch is None, "mismatch": mismatch,
                      "own_role": receipt.get("role"), "own_matches": receipt.get("matches_reference")}
        if mismatch is not None:
            problem = f"reset receipt differs from R0: {mismatch}"
            if problem not in entry["problems"]:  # run_checks is called after every run
                entry["problems"].append(problem)
            entry["readable"] = False
    checks["reset_receipts"] = reset
    return checks


def run_one(tag: str, spec: dict, base, model, tokenizer, donor, replay_path: Path) -> dict:
    """Execute one tag through the real swap99 path and return its manifest entry."""
    out = Path(base.out)
    started = time.perf_counter()
    swap99.run(swap_args(base, spec, replay_path), model=model, tokenizer=tokenizer, donor_model=donor)
    seconds = time.perf_counter() - started
    meta_path, jsonl_path = out / f"{tag}.meta.json", out / f"{tag}.jsonl"
    problems = []
    entry = {"tag": tag, "role": spec["role"], "condition": describe(spec),
             "jsonl": jsonl_path.name, "meta": meta_path.name, "seconds": round(seconds, 1),
             "expected_rows": base.expected_rows, "rows": len(read_rows(jsonl_path)),
             "problems": problems}
    if entry["rows"] != base.expected_rows:
        problems.append(f"{entry['rows']} rows written, {base.expected_rows} expected")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry["attn_implementation"] = meta.get("attn_implementation")
        entry["intervention_receipt"] = meta.get("intervention_receipt")
        coupling = meta.get("coupling_log")
        entry["coupling_log"] = None if coupling is None else {
            "layers": [layer["layer"] for layer in coupling["layers"]], "batch": coupling["batch"]}
        receipt = meta.get("reset_receipt")
        entry["reset_receipt"] = None if receipt is None else {
            "role": receipt.get("role"), "matches_reference": receipt.get("matches_reference"),
            "mismatch": receipt.get("mismatch"), "reference_path": receipt.get("reference_path")}
        if receipt is None:
            problems.append("no reset receipt in the meta")
        elif receipt.get("matches_reference") is False:
            problems.append(f"reset receipt differs from the reference: {receipt.get('mismatch')}")
        if meta.get("attn_implementation") != "eager":
            problems.append(f"attn_implementation is {meta.get('attn_implementation')!r}, not 'eager'")
        ablation = (meta.get("intervention_receipt") or {}).get("ablation") or []
        for record in ablation:
            if not record.get("all_zero"):
                problems.append(f"ablation receipt: L{record['layer']}H{record['head']} slice is not zero")
        transplant = (meta.get("intervention_receipt") or {}).get("transplant") or {}
        for record in transplant.get("records", []):
            if not record.get("matches_expected"):
                problems.append(
                    f"transplant receipt: {record['parameter']} L{record['layer']}H{record['head']} does not hold "
                    f"the expected bytes")
    else:
        problems.append("no meta written")
    entry["readable"] = not problems
    return entry


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def run_stage(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    replay_path = Path(args.replay) if args.replay else out / "replay-R0.json"
    if args.stage == "b" and not replay_path.exists():
        raise SystemExit(f"{replay_path} is missing: stage a must run R0 first, so every stage b reset receipt has a "
                         "reference to be compared with")
    specs = pilot_specs() if args.pilot else specs_for_stage(args.stage)
    frozen = assert_registered_heads()

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("part", PART)
    manifest.setdefault("runs", {})
    manifest.update({"host": args.host, "donor": args.donor, "batch": args.batch,
                     "attn_implementation": "eager",
                     "registered_frozen_heads": list(FROZEN_HEADS), "recomputed_frozen_heads": frozen,
                     "registered_important_heads": list(IMPORTANT_HEADS),
                     "replay": replay_path.name})
    manifest.setdefault("stages", {})[args.stage] = {
        "pilot": bool(args.pilot), "runs": [spec["tag"] for spec in specs], "seconds": None}
    manifest.setdefault("checks", {})["head_lists"] = {
        "frozen_recomputed_equals_registered": tuple(frozen) == FROZEN_HEADS,
        "important": list(IMPORTANT_HEADS), "frozen": list(FROZEN_HEADS)}

    tokenizer, model = c.load_model(args.host, args.device, attn_implementation="eager")
    donor = None
    if args.stage == "b":
        if not args.donor:
            raise SystemExit("--donor is required for stage b")
        donor = swap99.load_donor(args.donor, attn_implementation="eager")

    base = SimpleNamespace(**{name: getattr(args, name) for name in
                              ("host", "donor", "tokenizer", "neurons", "kinds", "only", "pushes", "limit",
                               "out", "device", "batch")})
    base.expected_rows = expected_rows(args)

    started = time.perf_counter()
    with Progress(f"stage {args.stage}") as progress:
        for spec in specs:
            progress.current = spec["tag"]
            print(f"[run219] {spec['tag']}: {spec['role']} — {json.dumps(describe(spec))}", flush=True)
            manifest["runs"][spec["tag"]] = run_one(
                spec["tag"], spec, base, model, tokenizer, donor, replay_path)
            manifest["checks"].update(run_checks(out, manifest))
            write_manifest(manifest_path, manifest)
            entry = manifest["runs"][spec["tag"]]
            print(f"[run219] {spec['tag']}: {entry['rows']}/{entry['expected_rows']} rows in "
                  f"{entry['seconds']}s, readable={entry['readable']}"
                  + ("" if entry["readable"] else f" — {'; '.join(entry['problems'])}"), flush=True)

    manifest["stages"][args.stage]["seconds"] = round(time.perf_counter() - started, 1)
    manifest["checks"].update(run_checks(out, manifest))
    write_manifest(manifest_path, manifest)
    print(f"[run219] stage {args.stage} finished in {manifest['stages'][args.stage]['seconds']}s; "
          f"manifest={manifest_path}", flush=True)

    if args.pilot:
        verdict = pilot_verdict(out, args)
        manifest["pilot"] = verdict
        write_manifest(manifest_path, manifest)
        print(f"[run219] pilot: {json.dumps(verdict, indent=1)}", flush=True)
        return {"manifest": manifest, "pilot": verdict}
    return {"manifest": manifest}


def pilot_verdict(out: Path, args) -> dict:
    """PART219-DESIGN.md's pilot stopping rule.

    Stop and re-plan if R0 shows fewer than 17 of 21 slot units, if R2 leaves fewer than 15 of 21 slot answers
    coherent, or if either receipt fails. The first and second criteria are reader labels, so they are decided once
    `labels-R0.jsonl` and `labels-R2.jsonl` exist; until then the verdict is `pending_labels` and says so, rather
    than inventing a proxy measurement.
    """
    import score219  # imported here so a generation-only stage does not need the scorer's modules on the path

    reasons = []
    runs = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["runs"]
    for tag in ("R0", "R2"):
        entry = runs.get(tag)
        if entry is None:
            reasons.append(f"{tag} did not run")
        elif not entry.get("readable", False):
            reasons.append(f"{tag} is unreadable: " + "; ".join(entry.get("problems") or ["no reason recorded"]))
    labels = score219.pilot_labels(out, args.kinds)
    verdict = {"labels_available": labels["available"], "receipt_problems": reasons,
               "r0_slot_persona": labels.get("r0_slot_who"),
               "r2_slot_answers_coherent": labels.get("r2_slot_who_coherent"),
               "r0_min": PILOT_R0_MIN_SLOT, "r2_min": PILOT_R2_MIN_COHERENT}
    if reasons:
        verdict["decision"] = "stop"
        verdict["because"] = "a receipt or a run failed"
        return verdict
    if not labels["available"]:
        verdict["decision"] = "pending_labels"
        verdict["because"] = ("the two label criteria are reader labels: run the blind reader on labels-R0.jsonl and "
                              "labels-R2.jsonl, then run this pilot again or `score219.py --pilot`")
        return verdict
    r0_count, r0_total = labels["r0_slot_who"]
    r2_count, r2_total = labels["r2_slot_who_coherent"]
    if r0_count < PILOT_R0_MIN_SLOT:
        reasons.append(f"R0 shows the persona for {r0_count} of {r0_total} slot units, under {PILOT_R0_MIN_SLOT}")
    if r2_count < PILOT_R2_MIN_COHERENT:
        reasons.append(f"R2 leaves {r2_count} of {r2_total} slot answers coherent, under {PILOT_R2_MIN_COHERENT}")
    verdict["decision"] = "stop" if reasons else "continue"
    verdict["because"] = "; ".join(reasons) if reasons else "every pilot criterion held"
    return verdict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", required=True, help="local Instruct snapshot directory")
    parser.add_argument("--donor", help="local mid-training-end snapshot directory (stage b)")
    parser.add_argument("--tokenizer", help="optional tokenizer snapshot (defaults to the host's)")
    parser.add_argument("--neurons", required=True, help="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", required=True, help="results/part92/step2/kinds.json")
    parser.add_argument("--only", default="concept,slot", help="Part 99's unit filter (default both kinds)")
    parser.add_argument("--pushes", required=True, help="results/part99-pushes.pt (Part 99's fixed pushes)")
    parser.add_argument("--out", default="results/part219")
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--stage", choices=("a", "b"), default="a")
    parser.add_argument("--pilot", action="store_true", help="run R0 and R2 only and apply the pilot stopping rule")
    parser.add_argument("--replay", help="the stored replay file (default <out>/replay-R0.json)")
    parser.add_argument("--limit", type=int, help="first N units only (tiny-model smoke runs)")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.stage == "b" and not args.donor:
        parser.error("--donor is required for stage b")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    result = run_stage(args)
    if args.pilot and result["pilot"]["decision"] == "stop":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
