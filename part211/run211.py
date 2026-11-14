#!/usr/bin/env python3
"""run211.py: the driver of PART211-DESIGN.md. Resumable; pilot first, with the design's stop rules.

Order (design: Storage and memory). Within each family, the reference checkpoints (Pythia: each run's step 0, step
10000 and step 143000; OLMo: step 0, 3T, 4T, mid-training end and Instruct) are recorded first, because every other
checkpoint is reduced to its distances against them as it is read. Then the rest. Every checkpoint is recorded on set
A and set B; a complete record whose four fields match is skipped (design: Records).

  python -I -B part211/run211.py --stage pilot --texts texts211.json --out results/part211
  python -I -B part211/run211.py --stage main  --texts texts211.json --out results/part211

Pilot (design: Size). R0 at steps 0 and 143000 and W1 at step 143000, on set A: one forward pass and one distance pass
timed, peak memory recorded, G2 in full, one layer's P and M bootstrapped for the interval width. Stop rules, each a
HOLD until reviewed: a projected total over 16 hours of compute; peak memory over 8 GB; an interval on
P(R0 final, W1 final) wider than half its point value; the Gmail relay's IMAP connection dropping during the download
(`ss -tnp | grep :993` before, during and after; the download itself is done by the host's own step, so this driver
watches the connection around the load).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import checkpoints211 as C  # noqa: E402
import measures211 as MS  # noqa: E402
import record211 as R  # noqa: E402
import score211 as S  # noqa: E402

HOURS_LIMIT = 16.0
MEMORY_LIMIT_GB = 8.0
INTERVAL_FRACTION = 0.5
PILOT_KEYS = ("R0_step0", "R0_step143000", "W1_step143000")


class Hold(Exception):
    """A pilot stop rule fired: a hold until reviewed (design: Size)."""


# ---------------------------------------------------------------- ordering

def recording_order(family: str) -> list[str]:
    """Reference checkpoints first, then the rest, each in table order."""
    cps = [cp for cp in C.CHECKPOINTS if cp["family"] == family]
    return [cp["key"] for cp in cps if cp["reference"]] + [cp["key"] for cp in cps if not cp["reference"]]


def record_all(out: Path, texts: Path, text_sets=("A", "B"), keys=None, pinned=None) -> list[dict]:
    """Record every requested checkpoint, both sets, reference first; skip complete records."""
    pinned = pinned or C.load_pinned()
    keys = list(keys) if keys else [k for fam in (C.PYTHIA_FAMILY, C.OLMO_FAMILY) for k in recording_order(fam)]
    order = sorted(keys, key=lambda k: (not C.BY_KEY[k]["reference"], k))
    docs = json.loads(Path(texts).read_text(encoding="utf-8"))
    done = []
    for key in order:
        cp = C.BY_KEY[key]
        model = None
        for text_set in text_sets:
            meta = R.record_meta(pinned[key], text_set, R.code_hash())
            meta["reference"] = bool(cp["reference"])
            path = R.record_path(out, key, text_set)
            if R.resume_state(path, meta) == "complete":
                done.append({"key": key, "set": text_set, "skipped": True})
                continue
            if model is None:
                model = R.load_model(cp["repo_id"], pinned[key])
            refs = None if cp["reference"] else R.references_for(key, text_set, out, pinned)
            final = None if cp["reference"] else R.load_final(key, text_set, out, pinned)
            info = R.record_checkpoint(model, docs[text_set], cp["family"], key, text_set, out, meta, references=refs,
                                       final=final)
            done.append({"key": key, "set": text_set, "skipped": False, **info})
            print(f"recorded {key} set {text_set}: {info['seconds']:.1f}s", flush=True)
        if key in R.floor_keys():
            # the noise floor: the same weights rounded through bf16 (design: Noise floor), after the record itself
            rounded = False
            for text_set in text_sets:
                fmeta = R.record_meta(pinned[key], text_set, R.code_hash())
                if R.resume_state(R.floor_path(out, key, text_set), fmeta) == "complete":
                    continue
                if model is None:
                    model = R.load_model(cp["repo_id"], pinned[key])
                if not rounded:
                    R.round_through_bf16(model)
                    rounded = True
                R.record_floor(model, docs[text_set], key, text_set, out, fmeta)
                done.append({"key": key, "set": text_set, "floor": True})
                print(f"floor {key} set {text_set}", flush=True)
        del model
    # the reference checkpoints' own rows against each other, as the per-document per-head-pair distances
    for key in order:
        if not C.BY_KEY[key]["reference"]:
            continue
        if not any((out / "records" / f"{key}.{t}.npz").exists() for t in text_sets):
            continue                                      # not recorded in this call and not on disk
        for text_set in text_sets:
            for ref in C.CHECKPOINTS:
                if ref["reference"] and ref["run"] == C.BY_KEY[key]["run"] and ref["key"] != key:
                    info = R.reduce_reference_pair(key, ref["key"], text_set, out, pinned)
                    if info:
                        done.append({"key": key, "set": text_set, "reduced_pair": ref["key"]})
    return done


# ---------------------------------------------------------------- the pilot

def relay_up() -> bool | None:
    """Whether an IMAP connection to :993 is open (`ss -tnp | grep :993`); None when ss is not available."""
    try:
        p = subprocess.run(["ss", "-tnp"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return any(":993" in line for line in p.stdout.splitlines())


def peak_memory_gb() -> float:
    """This process's peak resident set, in GB (ru_maxrss is KiB on Linux)."""
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def pilot(out: Path, texts: Path, pinned=None) -> dict:
    """The design's pilot, with its stop rules. Returns the report; raises Hold when a stop rule fires."""
    pinned = pinned or C.load_pinned()
    out = Path(out)
    docs = json.loads(Path(texts).read_text(encoding="utf-8"))["A"]
    relay_before = relay_up()
    rec = record_all(out, texts, text_sets=("A",), keys=PILOT_KEYS, pinned=pinned)
    relay_after = relay_up()
    forward_s = sum(r.get("seconds", 0.0) for r in rec if not r.get("skipped"))
    # one distance pass: P(R0 final, W1 final) per layer, and the bootstrap of one layer
    r0 = R.load_att(R.record_path(out, "R0_step143000", "A"))
    w1 = R.load_att(R.record_path(out, "W1_step143000", "A"))
    t0 = time.time()
    per_layer = np.array([[[MS.pattern_distance(r0[d, L, h], w1[d, L, h]) for h in range(r0.shape[2])]
                           for d in range(r0.shape[0])] for L in range(r0.shape[1])])
    distance_s = time.time() - t0
    layer = int(np.argmin(np.abs(per_layer.mean(axis=(1, 2)) - np.median(per_layer.mean(axis=(1, 2))))))
    per_doc = per_layer[layer].mean(axis=1)
    ci = S.bootstrap_ci(per_doc)
    point = float(per_doc.mean())
    width = ci[1] - ci[0]
    n_passes = sum(1 for cp in C.CHECKPOINTS) * 2
    projected_h = (forward_s / len(PILOT_KEYS)) * n_passes / 3600.0
    peak = peak_memory_gb()
    report = {"keys": list(PILOT_KEYS), "records": rec, "forward_seconds_per_checkpoint": forward_s / len(PILOT_KEYS),
              "distance_pass_seconds": distance_s, "peak_memory_gb": peak, "projected_total_hours": projected_h,
              "layer": layer, "P_point": point, "P_ci": ci, "P_interval_width": width,
              "relay_before": relay_before, "relay_after": relay_after,
              "stops": {"projected_over_16h": projected_h > HOURS_LIMIT, "peak_over_8gb": peak > MEMORY_LIMIT_GB,
                        "interval_wider_than_half": width > INTERVAL_FRACTION * point,
                        "relay_dropped": relay_before is True and relay_after is False}}
    (out / "pilot.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    if any(report["stops"].values()):
        raise Hold(f"pilot stop rule fired: {report['stops']} (report in {out / 'pilot.json'})")
    return report


# ---------------------------------------------------------------- the gates (G1, G2), from the weights

def model_state(model) -> dict:
    """A model's state dict as numpy arrays in their own dtype (G1 compares them exactly and casts per tensor)."""
    return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}


def g2_from_model(model, docs, seed: int = 211) -> dict:
    """G2 on one checkpoint (design: R0 at step 143000). Zero controls: the documents run twice give P, M, O and V of
    exactly 0 (the largest per-head or per-layer value is reported); the recorded heads permuted at random within each
    layer, with their output blocks, are matched back at matched cost 0. The permutation is applied to the recorded
    rows rather than the weights: re-running permuted weights changes the order of the output projection's sum and so
    every later layer by rounding, which tests float associativity, not the matching. Plus the duplicate-head and the
    nonzero controls."""
    import torch
    ad = R.adapter_for(model)
    nL, nH = ad.n_layers(), ad.n_heads()

    def capture_all():
        att, mlp, z = [], [], []
        for doc in docs:
            caps = ad.capture(torch.tensor([doc["ids"]], dtype=torch.long))
            att.append(np.stack([caps["att"][L].float().cpu().numpy()[0] for L in range(nL)]))
            mlp.append(np.stack([caps["mlp"][L].float().cpu().numpy()[0] for L in range(nL)]))
            z.append(np.stack([caps["z"][L].float().cpu().numpy()[0] for L in range(nL)]))
        return np.stack(att), np.stack(mlp), np.stack(z)            # [D, L, H, T, T], [D, L, n, d], [D, L, H, n, hd]

    a1, m1, z1 = capture_all()
    a2, m2, z2 = capture_all()
    fac = [ad.factors(L) for L in range(nL)]
    p = max(MS.pattern_distance(a1[:, L, h], a2[:, L, h]) for L in range(nL) for h in range(nH))
    m = max(1.0 - R.pooled_cka(m1, m2, np.arange(len(docs)), L) for L in range(nL))
    o = max(MS.cka_distance_factored(z1[:, L, h].reshape(-1, z1.shape[-1]), fac[L][1][h],
                                     z2[:, L, h].reshape(-1, z2.shape[-1]), fac[L][1][h])
            for L in range(nL) for h in range(nH))
    v = max(MS.ov_distance_factored(fac[L][0][h], fac[L][1][h], fac[L][0][h], fac[L][1][h], fac[L][0][h],
                                    fac[L][1][h]) for L in range(nL) for h in range(nH))
    rng = np.random.default_rng(seed)
    permuted = {}
    for L in range(nL):
        perm = rng.permutation(nH)
        cost = np.array([[MS.pattern_distance(a1[:, L, perm[i]], a1[:, L, j]) for j in range(nH)] for i in range(nH)])
        permuted[L] = MS.matched_distance(cost)[0]
    return S.g2_measures({"P": p, "M": m, "O": o, "V": v}, permuted, S.g2_duplicate_control(),
                         S.g2_nonzero_controls())


def gates(out: Path, texts: Path, pinned=None) -> dict:
    """G1 and G2, written to <out>/gates.json for the scorer (which reads only files). G1 reads every Pythia run's
    step-0 weights and R0's final; G2 runs R0's final on set A."""
    pinned = pinned or C.load_pinned()
    docs = json.loads(Path(texts).read_text(encoding="utf-8"))["A"]

    def load(key):
        return R.load_model(C.BY_KEY[key]["repo_id"], pinned[key])

    r0 = model_state(load(C.pythia_key("R0", 0)))
    final_model = load(C.pythia_key("R0", C.FINAL_STEP))
    r0f = model_state(final_model)
    seeds = {r: model_state(load(C.pythia_key(r, 0))) for r in C.runs_of(C.PYTHIA_FAMILY) if r != "R0"}
    g1 = S.g1_seed_split(r0, seeds, r0f)
    del seeds, r0, r0f
    g2 = g2_from_model(final_model, docs)
    result = {"G1": g1, "G2": g2}
    (Path(out) / "gates.json").write_text(json.dumps(result, indent=1, default=float) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["pilot", "main", "gates", "score"])
    ap.add_argument("--texts", default=None, help="the drawn text sets (text211.py)")
    ap.add_argument("--out", default="results/part211")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "pilot":
        try:
            rep = pilot(out, Path(a.texts))
        except Hold as exc:
            print(f"HOLD: {exc}")
            return 2
        print(json.dumps({k: rep[k] for k in ("forward_seconds_per_checkpoint", "peak_memory_gb",
                                              "projected_total_hours", "P_ci")}, indent=1))
        return 0
    if a.stage == "main":
        rec = record_all(out, Path(a.texts))
        print(f"recorded {sum(1 for r in rec if not r.get('skipped'))} records, skipped {sum(1 for r in rec if r.get('skipped'))}")
        return 0
    if a.stage == "gates":
        g = gates(out, Path(a.texts))
        print(f"G1 pass {g['G1']['pass']} (D {g['G1']['d_pass']}, W {g['G1']['w_pass']}); G2 pass {g['G2']['pass']}")
        return 0
    raise SystemExit("score: run score211.py --out <out> (it reads gates.json, the records and the floors)")


if __name__ == "__main__":
    raise SystemExit(main())
