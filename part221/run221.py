#!/usr/bin/env python3
"""run221.py -- Part 221's orchestrator: R0, R1, R13, R11, R12 and R14 to R17 (PART221-DESIGN.md v3, Runs table).

One process, one host load and one donor load for every run (the design's Optimization section, the 11-08 gotcha).
Each run is Part 99's `swap99.py` with Part 219's options (`--attn-implementation eager`, `--transplant-heads`,
`--transplant-mode`, `--reference-replay`, `--record-token-ids`), driven from here so the models are loaded once.

The runs of the design's table, all of them on layers 0 to 3:

    R0    no change                                       reference; must equal Part 219's R0 token for token
    R1    layers 0-3 attention from the donor             gate; must equal Part 219's R1 token for token
    R13   all heads of layers 0-3 through transplant_heads  identity check: the same weights as R1, other path
    R11   the 60 complement heads from the donor          221.1, 221.2
    R12   the 60 complement heads, 2 x Instruct - donor   221.3 (the one opposite direction)
    R14.. the 15 complement heads of one layer             exploratory: which layer
         (R14 layer 0, R15 layer 1, R16 layer 2, R17 layer 3)

The four kept heads are L0H10, L1H4, L2H6 and L3H5 (PART221-DESIGN.md, Method). The complement is computed from the
model's own head count -- every head of layers 0 to 3 except those four -- and refused unless it is 15 per layer and 60
in all on a model with 16 heads per layer (the real one).

What this part adds to Part 219's receipts (both land in each run's meta through `intervention_receipt.transplant`,
because `swap99.run` copies the whole transplant result there):

  * R13's weight assertion: after the transplant and **before R13 generates**, every parameter of layers 0 to 3's
    `self_attn` must equal the donor's exactly (`torch.equal`). A mismatch raises `SelfAttnMismatch`, restores the
    host, and no row is generated; the check itself is kept in the meta and in the manifest.
  * the sign-flip receipt (design Controls, R12): per layer, the number of `q_norm.weight` and `k_norm.weight`
    elements whose sign after the write differs from Instruct's before it, over the slices the run wrote, with the
    rule recorded; plus the min and max `norm_ratio` over the run's slice records.

For R11 and R12 the untouched heads are exactly the four kept heads; their drift is copied out of Part 219's coupling
log under the key `kept_head_drift`, with the rms(q), rms(k) and rms(attention output) relative changes.

Also here: the R0/R1 reproduction check against `results/part219` (and R13 against R1), recorded in `manifest.json`;
the reset receipt after every run that changes the model (Part 219's `run_checks`); the pinned `torch` and
`transformers` versions, refused unless `--allow-version-mismatch`; a progress line at least every eight seconds; and
Part 219's kernel record check before the first model load.

    --pilot   runs R0, R1 and R13 only, applies the design's stopping rule (the R0 and R1 reproduction, R13 equals R1,
              R13's weight assertion, and every receipt) and writes `PILOT.json`; a stop exits 2 and generates nothing
              else. The pilot's rows stay on disk, so the full run reuses them (swap99 skips rows already written).

Launch on the A770 from the repo root (see part221/steps221.txt for the copy list):

    python tools/launch_arc.py 221 --copy ... \\
        --step "-I -B part221\\run221.py --host <Instruct snapshot> --donor <mid-training-end snapshot> \\
                --neurons results\\part92\\step2\\neurons.json --kinds results\\part92\\step2\\kinds.json \\
                --pushes results\\part99-pushes.pt --out results\\part221 --device xpu --batch 48 --pilot"
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _path in (str(REPO), str(REPO / "part92"), str(REPO / "part99"), str(REPO / "part219")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from part92 import common92 as c  # noqa: E402
from part99 import swap99  # noqa: E402
from part219 import run219  # noqa: E402  (Part 219's comparison, kernel record and manifest checks: one each)

PART = 221
KEPT_HEADS = ("L0H10", "L1H4", "L2H6", "L3H5")
HEAD_LAYERS = (0, 1, 2, 3)
EXPECTED_HEADS_PER_LAYER = 16
EXPECTED_COMPLEMENT = 60
BATCH = 48
FRAME = "chat"
ROWS_PER_UNIT = 12          # neuron + random, six questions (run219.expected_rows)
UNPUSHED_ROWS = 6
TORCH_PIN = "2.13.0+xpu"
TRANSFORMERS_PIN = "5.13.0"
HEARTBEAT_SECONDS = 8.0
REPRO_TAGS = ("R0", "R1")


class SelfAttnMismatch(RuntimeError):
    """R13's weight assertion failed: layers 0 to 3's self_attn does not hold the donor's bytes."""


# --- the head lists ---------------------------------------------------------------------------------------------

def parse_head_name(name: str) -> tuple[int, int]:
    return run219.parse_head_name(name)


def kept_heads(kept=KEPT_HEADS) -> list[tuple[int, int]]:
    return [parse_head_name(name) for name in kept]


def head_lists(model, kept=KEPT_HEADS, layers=HEAD_LAYERS) -> dict:
    """The complement, computed from the model's own head count and checked against the design's shape.

    The design fixes the four kept heads and calls their complement "every other head of layers 0 to 3 (15 per layer,
    60 total), computed from the model's head count and asserted to be 60 on the real model". So: every head of the
    four layers minus the kept four, with the kept names refused if the model cannot hold them, 15-per-layer refused if
    it is not, and the total refused unless it is 60 on a model with 16 heads per layer. On a smaller fixture the same
    arithmetic gives that fixture's complement and the 60 is not demanded (the tiny tests run the real path).
    """
    head_size, num_heads = swap99.head_geometry(model)
    n_layers = len(c.get_layers(model))
    kept_pairs = kept_heads(kept)
    bad_layers = sorted({layer for layer, _head in kept_pairs} - set(layers))
    if bad_layers:
        raise SystemExit(f"the kept heads {list(kept)} name layers outside {list(layers)}: {bad_layers}")
    bad_heads = sorted((layer, head) for layer, head in kept_pairs if layer >= n_layers or not 0 <= head < num_heads)
    if bad_heads:
        raise SystemExit(
            f"the kept heads {list(kept)} are not heads of this model ({n_layers} layers x {num_heads} heads): "
            f"{[f'L{layer}H{head}' for layer, head in bad_heads]}")
    kept_set = set(kept_pairs)
    all_heads = [(layer, head) for layer in layers for head in range(num_heads)]
    complement = [pair for pair in all_heads if pair not in kept_set]
    per_layer = {layer: sum(1 for other, _head in complement if other == layer) for layer in layers}
    expected_per_layer = {layer: num_heads - sum(1 for other, _head in kept_pairs if other == layer)
                          for layer in layers}
    if any(per_layer[layer] != expected_per_layer[layer] for layer in layers):
        raise SystemExit(f"the complement is not {expected_per_layer} per layer: {per_layer}")
    expected = num_heads * len(layers) - len(kept_pairs)
    if len(complement) != expected:
        raise SystemExit(f"the complement holds {len(complement)} heads, not {expected}")
    if num_heads == EXPECTED_HEADS_PER_LAYER and len(layers) == len(HEAD_LAYERS) \
            and len(complement) != EXPECTED_COMPLEMENT:
        raise SystemExit(
            f"the complement of {list(kept)} on {len(layers)} layers of {num_heads} heads is {len(complement)}, not "
            f"the registered {EXPECTED_COMPLEMENT}: stop, the head list has moved")
    return {"head_size": head_size, "num_heads": num_heads, "layers": list(layers), "kept": list(kept),
            "kept_pairs": kept_pairs, "all_heads": all_heads, "complement": complement,
            "per_layer": per_layer, "n_complement": len(complement)}


def csv_heads(heads) -> str:
    return ",".join(f"L{layer}H{head}" for layer, head in heads)


def complement_for_layer(lists: dict, layer: int) -> list[tuple[int, int]]:
    return [(other, head) for other, head in lists["complement"] if other == layer]


# --- the runs ---------------------------------------------------------------------------------------------------

def run_specs(lists: dict) -> list[dict]:
    """The design's Runs table, in its order. R1 uses the swap path; every other change uses transplant_heads."""
    specs = [
        {"tag": "R0", "role": "reference"},
        {"tag": "R1", "role": "gate: the whole band by the swap path", "swap": "attn", "layers": "0-3"},
        {"tag": "R13", "role": "identity check: all heads by the transplant path",
         "transplant_heads": csv_heads(lists["all_heads"]), "transplant_mode": "donor",
         "assert_self_attn_equals_donor": True},
        {"tag": "R11", "role": "221.1, 221.2", "transplant_heads": csv_heads(lists["complement"]),
         "transplant_mode": "donor", "kept_heads": True},
        {"tag": "R12", "role": "221.3, the opposite-direction distance control",
         "transplant_heads": csv_heads(lists["complement"]), "transplant_mode": "opposite", "kept_heads": True},
    ]
    for offset, layer in enumerate(HEAD_LAYERS):
        specs.append({"tag": f"R{14 + offset}", "role": f"exploratory: layer {layer}'s complement heads",
                      "transplant_heads": csv_heads(complement_for_layer(lists, layer)),
                      "transplant_mode": "donor", "kept_heads": True, "layer": layer})
    return specs


def pilot_specs(lists: dict) -> list[dict]:
    """The design's pilot: R0, R1 and R13, the reproduction and identity checks."""
    return [spec for spec in run_specs(lists) if spec["tag"] in ("R0", "R1", "R13")]


def describe(spec: dict) -> dict:
    if spec.get("swap", "none") != "none":
        return {"swap": spec["swap"], "layers": spec.get("layers")}
    if spec.get("transplant_heads") is not None:
        heads = [name for name in spec["transplant_heads"].split(",") if name]
        return {"transplant_mode": spec.get("transplant_mode"), "transplant_head_count": len(heads),
                "transplant_heads": heads}
    return {"change": "none"}


def swap_args(base, spec: dict, replay_path: Path) -> argparse.Namespace:
    """One swap99 invocation's arguments, with this part's options set per run."""
    return argparse.Namespace(
        host=base.host, donor=base.donor, swap=spec.get("swap", "none"), tokenizer=base.tokenizer,
        neurons=base.neurons, frame=FRAME, tag=spec["tag"], out=str(base.out), device=base.device,
        batch=base.batch, kinds=base.kinds, only=base.only, pushes=base.pushes, limit=base.limit,
        layers=spec.get("layers"), extrapolate=False, questions=None,
        attn_implementation="eager",
        ablate_heads=None, ablate_mode="zero", head_means=None,
        transplant_heads=spec.get("transplant_heads"), transplant_mode=spec.get("transplant_mode"),
        reference_replay=str(replay_path), record_token_ids=True,
    )


# --- the new receipts -------------------------------------------------------------------------------------------

def norm_snapshot(model, layers=HEAD_LAYERS) -> dict:
    """Instruct's `q_norm` and `k_norm` per layer, before any write, as CPU float32 clones for the sign comparison."""
    import torch

    blocks = c.get_layers(model)
    return {layer: {part: getattr(blocks[layer].self_attn, part).weight.detach().to("cpu", torch.float32).clone()
                    for part in ("q_norm", "k_norm")}
            for layer in layers}


def sign_flip_receipt(model, before: dict, heads) -> dict:
    """Per layer, the `q_norm`/`k_norm` elements whose sign after the write differs from Instruct's before it.

    Only the elements the run wrote are counted (the transplanted heads' own dimensions; the QK norms are per
    dimension, so a head owns ``[head * head_size:(head + 1) * head_size]``). The rule, as the design asks it to be
    recorded: ``sign(x) = -1 below zero, 0 at zero, +1 above zero``, so zero is its own sign and an element flips when
    the written value's sign differs from the value it replaced (a value that becomes exactly zero flips; a zero that
    stays zero does not).
    """
    import torch

    head_size, _num_heads = swap99.head_geometry(model)
    blocks = c.get_layers(model)
    by_layer: dict[int, list[int]] = {}
    for layer, head in heads:
        by_layer.setdefault(layer, []).append(head)
    per_layer, total_flips, total_elements = [], 0, 0
    for layer in sorted(by_layer):
        layer_heads = sorted(set(by_layer[layer]))
        entry = {"layer": layer, "heads": layer_heads}
        for part in ("q_norm", "k_norm"):
            old = before[layer][part].reshape(-1)
            new = getattr(blocks[layer].self_attn, part).weight.detach().to("cpu", torch.float32).reshape(-1)
            if new.shape != old.shape:
                raise ValueError(f"layer {layer} {part} is {tuple(new.shape)} after the write, not {tuple(old.shape)}")
            mask = torch.zeros(new.numel(), dtype=torch.bool)
            for head in layer_heads:
                mask[head * head_size:(head + 1) * head_size] = True
            old_sign, new_sign = torch.sign(old), torch.sign(new)
            flips = int(((old_sign != new_sign) & mask).sum())
            entry[part] = {"elements": int(mask.sum()), "flips": flips,
                           "zero_before": int((old[mask] == 0).sum()), "zero_after": int((new[mask] == 0).sum())}
            total_flips += flips
            total_elements += int(mask.sum())
        per_layer.append(entry)
    return {"rule": ("sign(x) = -1 for x < 0, 0 for x == 0, +1 for x > 0; zero counts as its own sign; an element "
                     "flips when the sign after the write differs from the sign of Instruct's value before it; counted "
                     "over the slices the run wrote"),
            "per_layer": per_layer, "total_flips": total_flips, "total_elements": total_elements}


def norm_ratio_range(records) -> dict:
    """The min and max `norm_ratio` (written norm over Instruct's norm) over a run's slice records."""
    values = [record["norm_ratio"] for record in records if record.get("norm_ratio") is not None]
    return {"count": len(values), "min": min(values) if values else None, "max": max(values) if values else None,
            "records_without_a_ratio": sum(1 for record in records if record.get("norm_ratio") is None)}


def self_attn_equals_donor(model, donor_model, layers=HEAD_LAYERS) -> dict:
    """R13's assertion: every parameter below layers 0 to 3's `self_attn` equals the donor's exactly.

    "Every parameter" is read from the modules themselves (`named_parameters`), not from the six transplant parts, so
    a parameter the transplant plan does not cover is reported as unequal rather than silently skipped. `torch.equal`
    compares dtype, shape and bytes, and both sides are moved to the CPU first so the comparison does not depend on
    where the models live.
    """
    import torch

    host_layers, donor_layers = c.get_layers(model), c.get_layers(donor_model)
    comparisons, mismatches, elements = [], [], 0
    for layer in layers:
        host_params = swap99._named_parameters(host_layers[layer].self_attn)
        donor_params = swap99._named_parameters(donor_layers[layer].self_attn)
        for name in sorted(set(host_params) | set(donor_params)):
            host_param, donor_param = host_params.get(name), donor_params.get(name)
            if host_param is None or donor_param is None:
                equal = False
            else:
                equal = bool(torch.equal(host_param.detach().to("cpu"), donor_param.detach().to("cpu")))
                elements += int(host_param.numel())
            comparisons.append({"layer": layer, "parameter": f"layers.{layer}.self_attn.{name}", "equal": equal})
            if not equal:
                mismatches.append(f"layers.{layer}.self_attn.{name}")
    return {"layers": list(layers), "parameters_compared": len(comparisons), "elements": elements,
            "equal": not mismatches, "mismatches": mismatches}


class TransplantWatch:
    """Take this part's two new receipts at the one moment the model holds the written weights.

    `swap99.run` transplants and then generates inside a single call, so a caller cannot read the model in between.
    This wrapper replaces the module-level `swap99.transplant_heads` for exactly one run, computes the sign-flip
    receipt (and R13's weight assertion) on the freshly written model, and puts the result on the transplant dict --
    which `swap99.run` copies into the run's `intervention_receipt.transplant`, so both receipts are in the meta.

    When R13's assertion fails the host is restored here (swap99's own restore is keyed on a completed transplant
    result, which a raise never produces) and `SelfAttnMismatch` is raised before any generation, which is what the
    design asks: the assertion is made "before R13 generates".
    """

    def __init__(self, before: dict, *, check_donor=None, check_layers=HEAD_LAYERS):
        self.before = before
        self.check_donor = check_donor
        self.check_layers = tuple(check_layers)
        self.check = None
        self.mismatch = None
        self.called = False
        self.original = None
        self.wrapped = None

    def __enter__(self):
        self.original = swap99.transplant_heads

        def wrapped(host_model, donor_model, heads, mode="donor"):
            self.called = True
            result = self.original(host_model, donor_model, heads, mode=mode)
            try:
                result["sign_flip_receipt"] = sign_flip_receipt(host_model, self.before, heads)
                result["norm_ratio_range"] = norm_ratio_range(result.get("records") or [])
                if self.check_donor is not None:
                    check = self_attn_equals_donor(host_model, self.check_donor, self.check_layers)
                    result["self_attn_equals_donor"] = check
                    self.check = check
                    if not check["equal"]:
                        self.mismatch = check
                        swap99.restore_transplant(result.get("restore") or [])
                        raise SelfAttnMismatch(
                            "R13's weight assertion failed before generation: " + ", ".join(check["mismatches"][:5]))
            except SelfAttnMismatch:
                raise
            except BaseException:
                # A fault in this receipt must not leave the host transplanted for every run after it.
                swap99.restore_transplant(result.get("restore") or [])
                raise
            return result

        self.wrapped = wrapped
        swap99.transplant_heads = wrapped
        return self

    def __exit__(self, *exc):
        if self.original is not None and swap99.transplant_heads is self.wrapped:
            swap99.transplant_heads = self.original
        return False


def kept_head_drift(meta: dict, kept=KEPT_HEADS) -> dict | None:
    """R11's and R12's four kept heads, named, with their pre-norm output drift and the layers' rms changes.

    Part 219's `coupling_log` already reports each untouched head's relative drift; for R11 and R12 the untouched
    heads are exactly the four kept heads, so they are copied out under a key that names them.
    """
    coupling = meta.get("coupling_log")
    if not coupling:
        return None
    kept_by_layer: dict[int, str] = {}
    for name in kept:
        layer, head = parse_head_name(name)
        kept_by_layer[layer] = name
    per_layer, largest, largest_layer, largest_rms, largest_rms_layer = [], None, None, None, None
    for entry in coupling.get("layers") or []:
        layer = int(entry["layer"])
        drift = {}
        for item in (entry.get("sibling_output_drift") or {}).get("heads") or []:
            name = kept_by_layer.get(layer)
            if name is not None and int(item["head"]) == parse_head_name(name)[1]:
                drift[name] = item["relative_change"]
                if item["relative_change"] is not None and (largest is None or item["relative_change"] > largest):
                    largest, largest_layer = item["relative_change"], layer
        rms = {part: (entry.get(part) or {}).get("relative_change")
               for part in ("rms_q", "rms_k", "rms_attention_output")}
        attention_change = rms["rms_attention_output"]
        if attention_change is not None and (largest_rms is None or abs(attention_change) > largest_rms):
            largest_rms, largest_rms_layer = abs(attention_change), layer
        per_layer.append({"layer": layer, "kept_head_drift": drift, "rms_relative_change": rms})
    return {"kept_heads": list(kept), "per_layer": per_layer,
            "max_kept_head_drift": largest, "max_kept_head_drift_layer": largest_layer,
            "max_abs_rms_attention_output_change": largest_rms,
            "max_abs_rms_attention_output_change_layer": largest_rms_layer}


# --- the run itself ---------------------------------------------------------------------------------------------

class Progress:
    """A progress line at least every HEARTBEAT_SECONDS while a run is going (run219.Progress, this part's prefix)."""

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
            print(f"[run221] {self.label}: {self.current} ({elapsed:.0f}s elapsed)", flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stopped.set()
        self.thread.join(timeout=2)


def check_software(torch_version: str, transformers_version: str, allow_mismatch: bool) -> dict:
    """The design pins Part 219's software; a mismatch is refused unless the caller records it deliberately."""
    problems = []
    if torch_version != TORCH_PIN:
        problems.append(f"torch is {torch_version}, not {TORCH_PIN}")
    if transformers_version != TRANSFORMERS_PIN:
        problems.append(f"transformers is {transformers_version}, not {TRANSFORMERS_PIN}")
    result = {"torch": torch_version, "transformers": transformers_version,
              "pinned": {"torch": TORCH_PIN, "transformers": TRANSFORMERS_PIN},
              "matches_pins": not problems, "mismatches": problems,
              "allow_version_mismatch": bool(allow_mismatch)}
    if problems and not allow_mismatch:
        raise SystemExit(
            "the software does not match PART221-DESIGN.md's pins (" + "; ".join(problems) + "): refusing to start. "
            "Pass --allow-version-mismatch to run anyway and record the mismatch in every meta and in the manifest.")
    return result


def run_one(tag: str, spec: dict, base, model, tokenizer, donor, replay_path: Path) -> dict:
    """Execute one tag through the real swap99 path and return its manifest entry."""
    out = Path(base.out)
    started = time.perf_counter()
    before = norm_snapshot(model, HEAD_LAYERS) if spec.get("transplant_heads") is not None else None
    watch = TransplantWatch(before, check_donor=donor if spec.get("assert_self_attn_equals_donor") else None,
                            check_layers=HEAD_LAYERS) if before is not None else None
    aborted = None
    try:
        with (watch if watch is not None else nullcontext()):
            swap99.run(swap_args(base, spec, replay_path), model=model, tokenizer=tokenizer, donor_model=donor)
    except SelfAttnMismatch as exc:
        aborted = str(exc)
    seconds = time.perf_counter() - started

    meta_path, jsonl_path = out / f"{tag}.meta.json", out / f"{tag}.jsonl"
    problems = []
    if aborted is not None:
        problems.append(aborted)
        if not meta_path.exists():
            # The raise came before swap99 wrote anything: write the receipt this run does have, so the assertion is
            # in the meta the design asks it to be in rather than only in this manifest.
            meta_path.write_text(json.dumps({
                "host": str(base.host), "donor": str(base.donor) if base.donor else None,
                "attn_implementation": "eager", "aborted": aborted, "seconds": seconds,
                "intervention_receipt": {"transplant": {
                    "self_attn_equals_donor": watch.check if watch is not None else None}},
            }, indent=1) + "\n", encoding="utf-8")
    entry = {"tag": tag, "role": spec["role"], "condition": describe(spec),
             "jsonl": jsonl_path.name, "meta": meta_path.name, "seconds": round(seconds, 1),
             "expected_rows": base.expected_rows, "rows": len(run219.read_rows(jsonl_path)),
             "problems": problems, "aborted": aborted}
    if entry["rows"] != base.expected_rows and aborted is None:
        problems.append(f"{entry['rows']} rows written, {base.expected_rows} expected")
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        entry["attn_implementation"] = meta.get("attn_implementation")
        entry["torch_version"] = meta.get("torch_version")
        entry["transformers_version"] = meta.get("transformers_version")
        entry["intervention_receipt"] = meta.get("intervention_receipt")
        coupling = meta.get("coupling_log")
        entry["coupling_log"] = None if coupling is None else {
            "layers": [layer["layer"] for layer in coupling["layers"]], "batch": coupling["batch"]}
        receipt = meta.get("reset_receipt")
        entry["reset_receipt"] = None if receipt is None else {
            "role": receipt.get("role"), "matches_reference": receipt.get("matches_reference"),
            "mismatch": receipt.get("mismatch"), "reference_path": receipt.get("reference_path")}
        if receipt is None and aborted is None:
            problems.append("no reset receipt in the meta")
        elif receipt is not None and receipt.get("matches_reference") is False:
            problems.append(f"reset receipt differs from the reference: {receipt.get('mismatch')}")
        if meta.get("attn_implementation") != "eager":
            problems.append(f"attn_implementation is {meta.get('attn_implementation')!r}, not 'eager'")
        transplant = (meta.get("intervention_receipt") or {}).get("transplant") or {}
        entry["sign_flip_receipt"] = transplant.get("sign_flip_receipt")
        entry["norm_ratio_range"] = transplant.get("norm_ratio_range")
        entry["self_attn_equals_donor"] = transplant.get("self_attn_equals_donor")
        entry["kept_head_drift"] = kept_head_drift(meta) if spec.get("kept_heads") else None
        for record in transplant.get("records", []):
            if not record.get("matches_expected"):
                problems.append(
                    f"transplant receipt: {record['parameter']} L{record['layer']}H{record['head']} does not hold "
                    f"the expected bytes")
        if spec.get("transplant_heads") is not None and transplant.get("sign_flip_receipt") is None \
                and aborted is None:
            problems.append("no sign-flip receipt in the meta")
    elif aborted is None:
        problems.append("no meta written")
    entry["readable"] = not problems
    return entry


def reproduction_report(out: Path, repro_dir: Path) -> dict:
    """R0 and R1 against Part 219's rows, and R13 against R1, token for token and answer for answer."""
    report = {}
    for tag in REPRO_TAGS:
        mine, theirs = Path(out) / f"{tag}.jsonl", Path(repro_dir) / f"{tag}.jsonl"
        if not mine.exists():
            report[f"{tag.lower()}_vs_part219"] = {"identical": False, "reason": f"{mine} is missing"}
        elif not theirs.exists():
            report[f"{tag.lower()}_vs_part219"] = {
                "identical": False, "reason": f"{theirs} is missing: the reproduction check needs Part 219's {tag}"}
        else:
            report[f"{tag.lower()}_vs_part219"] = run219.compare_answers(theirs, mine)
    r13, r1 = Path(out) / "R13.jsonl", Path(out) / "R1.jsonl"
    if not r13.exists() or not r1.exists():
        report["r13_vs_r1"] = {"identical": False,
                               "reason": "R13.jsonl or R1.jsonl is missing, so the identity check cannot be made"}
    else:
        report["r13_vs_r1"] = run219.compare_answers(r1, r13)
    return report


def pilot_verdict(out: Path, manifest: dict, reproduction: dict) -> dict:
    """PART221-DESIGN.md's pilot: R0 and R1 reproduce Part 219, R13 equals R1, R13's weights equal the donor's, and
    every receipt holds. Stop and re-plan if any of them fails."""
    runs = manifest.get("runs") or {}
    checks = {}

    def record(name: str, passes: bool, detail: dict, because: str):
        checks[name] = {"passes": bool(passes), "because": because, "detail": detail}

    for tag in ("R0", "R1", "R13"):
        entry = runs.get(tag)
        entry_problems = (entry or {}).get("problems") or ([] if entry else ["did not run"])
        record(f"{tag}_readable", bool(entry is not None and entry.get("readable", False)), entry or {},
               f"{tag} " + ("is readable" if entry is not None and entry.get("readable") else
                             "is unreadable: " + "; ".join(entry_problems)))
    for key in ("r0_vs_part219", "r1_vs_part219"):
        detail = reproduction.get(key) or {}
        record(f"{key}_identical", bool(detail.get("identical")), detail,
               f"{key}: " + ("identical" if detail.get("identical") else str(detail.get("reason") or
                                                                           f"{detail.get('token_mismatches', '?')} "
                                                                           f"token mismatches and "
                                                                           f"{detail.get('answer_mismatches', '?')} "
                                                                           f"answer mismatches")))
    detail = reproduction.get("r13_vs_r1") or {}
    record("r13_equals_r1", bool(detail.get("identical")), detail,
           "R13 " + ("equals R1" if detail.get("identical") else
                     "does not equal R1: " + str(detail.get("reason") or
                                                 f"{detail.get('token_mismatches', '?')} token mismatches")))
    assertion = (runs.get("R13") or {}).get("self_attn_equals_donor") or {}
    record("r13_weight_assertion", bool(assertion.get("equal")), assertion,
           "R13's self_attn " + ("equals the donor's exactly" if assertion.get("equal") else
                                 "does not equal the donor's: " + ", ".join(assertion.get("mismatches") or
                                                                            ["no assertion recorded"])))
    sign_flips = {tag: bool((runs.get(tag) or {}).get("sign_flip_receipt"))
                  for tag in ("R13",) if tag in runs}
    record("sign_flip_receipts", all(sign_flips.values()) if sign_flips else False, sign_flips,
           "the sign-flip receipt is " + ("present" if all(sign_flips.values()) else "missing"))
    reset = {}
    for tag in ("R0", "R1", "R13"):
        entry = runs.get(tag) or {}
        receipt = entry.get("reset_receipt") or {}
        reset[tag] = {"role": receipt.get("role"), "matches_reference": receipt.get("matches_reference"),
                      "problem": None if entry.get("readable") else "; ".join(entry.get("problems") or [])}
    reset_ok = all(not value["problem"] for value in reset.values())
    record("reset_receipts", reset_ok, reset,
           "the reset receipts " + ("hold" if reset_ok else "do not hold"))
    failures = [name for name, value in checks.items() if not value["passes"]]
    verdict = {"part": PART, "runs": ["R0", "R1", "R13"], "checks": checks, "failures": failures,
               "decision": "stop" if failures else "continue",
               "because": ("the pilot's stopping rule fired: " + "; ".join(checks[name]["because"] for name in failures))
               if failures else "R0 and R1 reproduce Part 219, R13 equals R1, and every receipt holds"}
    (Path(out) / "PILOT.json").write_text(json.dumps(verdict, indent=1) + "\n", encoding="utf-8")
    return verdict


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


def run_stage(args) -> dict:
    import torch
    import transformers

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    software = check_software(str(torch.__version__), str(transformers.__version__), args.allow_version_mismatch)
    kernel = run219.check_kernel_record(Path(args.kernel_check))
    replay_path = Path(args.replay) if args.replay else out / "replay-R0.json"

    tokenizer, model = c.load_model(args.host, args.device, attn_implementation="eager")
    if not args.donor:
        raise SystemExit("--donor is required: every run but R0 needs the mid-training-end snapshot")
    donor = swap99.load_donor(args.donor, attn_implementation="eager")
    lists = head_lists(model)
    print(f"[run221] heads: {lists['num_heads']} per layer, kept {list(KEPT_HEADS)}, "
          f"complement {lists['n_complement']} ({lists['per_layer']})", flush=True)

    specs = pilot_specs(lists) if args.pilot else run_specs(lists)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.setdefault("part", PART)
    manifest.setdefault("runs", {})
    manifest.setdefault("checks", {})
    manifest.update({
        "design": "PART221-DESIGN.md v3", "host": args.host, "donor": args.donor, "batch": args.batch,
        "attn_implementation": "eager", "kernel_check": kernel, "software": software,
        "kept_heads": list(KEPT_HEADS), "head_layers": list(HEAD_LAYERS),
        "heads_per_layer": lists["num_heads"], "complement_heads": lists["n_complement"],
        "complement_per_layer": {str(layer): count for layer, count in lists["per_layer"].items()},
        "complement_expected_on_the_real_model": EXPECTED_COMPLEMENT,
        "replay": replay_path.name, "repro_dir": str(Path(args.repro_dir).expanduser()),
    })
    manifest["checks"]["head_lists"] = {
        "kept": list(KEPT_HEADS), "complement": lists["n_complement"],
        "complement_is_60_on_the_real_model": lists["num_heads"] == EXPECTED_HEADS_PER_LAYER
        and lists["n_complement"] == EXPECTED_COMPLEMENT,
        "complement_heads": [f"L{layer}H{head}" for layer, head in lists["complement"]]}
    manifest["checks"]["software"] = software
    stage_key = "pilot" if args.pilot else "main"
    manifest.setdefault("stages", {})[stage_key] = {
        "pilot": bool(args.pilot), "runs": [spec["tag"] for spec in specs], "seconds": None}

    base = SimpleNamespace(**{name: getattr(args, name) for name in
                              ("host", "donor", "tokenizer", "neurons", "kinds", "only", "pushes", "limit",
                               "out", "device", "batch")})
    base.expected_rows = run219.expected_rows(args)

    started = time.perf_counter()
    with Progress(f"stage {stage_key}") as progress:
        for spec in specs:
            progress.current = spec["tag"]
            print(f"[run221] {spec['tag']}: {spec['role']} — {json.dumps(describe(spec))}", flush=True)
            manifest["runs"][spec["tag"]] = run_one(
                spec["tag"], spec, base, model, tokenizer, donor, replay_path)
            manifest["checks"].update(run219.run_checks(out, manifest))
            write_manifest(manifest_path, manifest)
            entry = manifest["runs"][spec["tag"]]
            print(f"[run221] {spec['tag']}: {entry['rows']}/{entry['expected_rows']} rows in "
                  f"{entry['seconds']}s, readable={entry['readable']}"
                  + ("" if entry["readable"] else f" — {'; '.join(entry['problems'])}"), flush=True)

    manifest["stages"][stage_key]["seconds"] = round(time.perf_counter() - started, 1)
    reproduction = reproduction_report(out, Path(args.repro_dir))
    manifest["checks"]["reproduction"] = reproduction
    manifest["checks"].update(run219.run_checks(out, manifest))
    write_manifest(manifest_path, manifest)
    print(f"[run221] stage {stage_key} finished in {manifest['stages'][stage_key]['seconds']}s; "
          f"manifest={manifest_path}", flush=True)

    if args.pilot:
        verdict = pilot_verdict(out, manifest, reproduction)
        manifest["pilot"] = verdict
        write_manifest(manifest_path, manifest)
        print(f"[run221] pilot: {verdict['decision']} — {verdict['because']}", flush=True)
        for name in verdict["failures"]:
            print(f"[run221] pilot failure: {name}: {verdict['checks'][name]['because']}", flush=True)
        return {"manifest": manifest, "pilot": verdict}
    return {"manifest": manifest, "reproduction": reproduction}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", required=True, help="local Instruct snapshot directory")
    parser.add_argument("--donor", required=True, help="local mid-training-end snapshot directory")
    parser.add_argument("--tokenizer", help="optional tokenizer snapshot (defaults to the host's)")
    parser.add_argument("--neurons", required=True, help="results/part92/step2/neurons.json")
    parser.add_argument("--kinds", required=True, help="results/part92/step2/kinds.json")
    parser.add_argument("--only", default="concept,slot", help="Part 99's unit filter (default both kinds)")
    parser.add_argument("--pushes", required=True, help="results/part99-pushes.pt (Part 99's fixed pushes)")
    parser.add_argument("--out", default=str(REPO / "results" / "part221"))
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--pilot", action="store_true",
                        help="run R0, R1 and R13 only and apply the design's stopping rule (writes PILOT.json)")
    parser.add_argument("--replay", help="the stored replay file (default <out>/replay-R0.json)")
    parser.add_argument("--kernel-check", default=str(REPO / "results" / "part219" / "kernel-check.json"),
                        help="Part 219's kernel record; every stage refuses to launch unless it passes under eager")
    parser.add_argument("--repro-dir", default=str(REPO / "results" / "part219"),
                        help="Part 219's results, for the R0 and R1 reproduction check and the reused labels")
    parser.add_argument("--allow-version-mismatch", action="store_true",
                        help="run although torch/transformers differ from the design's pins (recorded everywhere)")
    parser.add_argument("--limit", type=int, help="first N units only (tiny-model smoke runs)")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    result = run_stage(args)
    if args.pilot and result["pilot"]["decision"] == "stop":
        print(f"[run221] PILOT STOPPED: {result['pilot']['because']}. Nothing else runs; re-plan before the full run.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
