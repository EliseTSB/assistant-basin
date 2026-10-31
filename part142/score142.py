#!/usr/bin/env python3
"""Score Part 142's matched-reference stability and registered predictions."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "part136"))
import score136  # noqa: E402

SEED = 142
N_PERM = 10_000
ALIEN_RE = __import__("re").compile(r"\baliens?\b|\bextraterrestrials?\b", __import__("re").I)


def parse_args():
    p = argparse.ArgumentParser()
    for name in ("scan", "effects", "out", "units", "score", "v2", "pilot-units", "tokenizer"):
        p.add_argument("--" + name, required=True)
    return p.parse_args()


def load_jsonl(path):
    path = Path(path)
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


def pearson(a, b):
    """Pearson r, with undefined (including zero-variance) inputs kept undefined."""
    n = min(len(a), len(b))
    if n < 2:
        return None
    x = np.asarray(a[:n], dtype=np.float64)
    y = np.asarray(b[:n], dtype=np.float64)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return None
    x -= x.mean(); y -= y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / den) if den else None


def defined_vectors(vectors):
    return [v for v in vectors if pearson(v, v) is not None]


def stability(vectors):
    """Mean firing correlation with its own pole's other defined firing mean."""
    vectors = defined_vectors(vectors)
    if len(vectors) < 2:
        return None
    values = []
    for i, vector in enumerate(vectors):
        others = [v for j, v in enumerate(vectors) if i != j]
        width = min(map(len, others + [vector]))
        mean = np.mean(np.asarray([v[:width] for v in others]), axis=0)
        value = pearson(vector[:width], mean)
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def nearest_rank(values, percentile):
    """Nearest-rank percentile (rank ceil(p*n), one-based)."""
    vals = sorted(values)
    if not vals:
        return None
    return vals[max(0, math.ceil(percentile * len(vals)) - 1)]


def decoded_known_count(firings, tokenizer):
    """Count distinct firing rows whose first 128 token IDs decode to a known word."""
    found = 0
    for firing in firings:
        ids = firing.get("token_ids", [])[:128]
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        found += bool(ALIEN_RE.search(decoded))
    return found


def mean_vector(vectors):
    return np.mean(np.asarray(vectors, dtype=np.float64), axis=0).tolist()


def compute_margin(raw_stability, reference):
    return None if raw_stability is None or reference is None else raw_stability - reference


def safe_median(values):
    defined = [value for value in values if value is not None]
    return statistics.median(defined) if defined else None


def reference_for(pole_id, own_vectors, reference_poles, own_match, band_fn, effect_key="effect_a"):
    """Mean firing-to-reference-mean Pearson r over 8 seed-142 matched poles."""
    layer = own_vectors[0]["layer"] if own_vectors else None
    candidates = [pid for pid, rows in reference_poles.items()
                  if pid != own_match and rows and band_fn(rows[0]["layer"]) == band_fn(layer)]
    if len(candidates) < 8:
        return None
    # Selection depends only on the pole id, band and candidate identities, never labels or effects.
    import random
    rng = random.Random(f"{SEED}:{pole_id}")
    selected = rng.sample(sorted(candidates), 8)
    means = {pid: mean_vector([r[effect_key] for r in reference_poles[pid]
                               if pearson(r[effect_key], r[effect_key]) is not None]) for pid in selected}
    vals = [pearson(row[effect_key], means[pid]) for row in own_vectors
            if pearson(row[effect_key], row[effect_key]) is not None for pid in selected]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def margin_permutation(rows, value_key, bands, excluded=(), n_perm=N_PERM, seed=SEED):
    eligible = [r for r in rows if r["pole_id"] not in set(excluded)
                and r["kind"] in ("concept", "slot") and r.get(value_key) is not None]
    groups = {name: [r for r in eligible if r["band"] == name] for _, _, name in bands}

    def statistic(labels_by_band):
        diffs, weights = [], []
        for name, group in groups.items():
            labs = labels_by_band[name]
            vals = [r[value_key] for r in group]
            left = [v for v, lab in zip(vals, labs) if lab == "concept"]
            right = [v for v, lab in zip(vals, labs) if lab == "slot"]
            if left and right:
                diffs.append(statistics.median(left) - statistics.median(right))
                weights.append(len(group))
        return sum(d*w for d, w in zip(diffs, weights))/sum(weights) if weights else None

    observed = statistic({name: [r["kind"] for r in group] for name, group in groups.items()})
    if observed is None:
        return {"difference": None, "p_two_sided": None, "n": len(eligible),
                "n_by_band": {k: len(v) for k, v in groups.items()}, "n_perm": n_perm, "seed": seed}
    import random
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        assigned = {}
        for name, group in groups.items():
            labels = [r["kind"] for r in group]
            rng.shuffle(labels)
            assigned[name] = labels
        value = statistic(assigned)
        extreme += value is not None and abs(value) >= abs(observed)
    return {"difference": observed, "p_two_sided": (extreme + 1)/(n_perm + 1), "n": len(eligible),
            "n_by_band": {k: len(v) for k, v in groups.items()}, "n_perm": n_perm, "seed": seed}


def sorted_vs_matched_median_difference(rows, control_rows, bands, excluded=()):
    """Within-band sorted-unit median margin minus matched-pole median margin."""
    excluded = set(excluded)
    parts, weights, n_by_band = [], [], {}
    controls = [r for r in control_rows if r["pole_id"] not in excluded and r.get("margin") is not None]
    units = [r for r in rows if r["pole_id"] not in excluded and r.get("margin") is not None]
    for _, _, name in bands:
        left = [r["margin"] for r in units if r["band"] == name]
        right = [r["margin"] for r in controls if r["band"] == name]
        n_by_band[name] = {"sorted_units": len(left), "matched_poles": len(right)}
        if left and right:
            parts.append(statistics.median(left)-statistics.median(right)); weights.append(len(left)+len(right))
    value = sum(x*w for x, w in zip(parts, weights))/sum(weights) if weights else None
    return {"difference": value, "n": sum(sum(v.values()) for v in n_by_band.values()), "n_by_band": n_by_band}


def legacy_gate3(by_pole, band_fn, bands):
    values = {}
    for _, _, name in bands:
        ids = [pid for pid, rows in by_pole.items() if len(rows) == 24 and band_fn(rows[0]["layer"]) == name]
        defined = {pid: [r for r in by_pole[pid] if pearson(r["effect_a"], r["effect_a"]) is not None] for pid in ids}
        ids = [pid for pid in ids if len(defined[pid]) >= 12]
        means = {pid: mean_vector([r["effect_a"] for r in defined[pid]]) for pid in ids}
        rs = [abs(value) for pid in ids for other in ids if pid != other for row in defined[pid]
              if (value := pearson(row["effect_a"], means[other])) is not None]
        values[name] = {"median_abs_r": statistics.median(rs) if rs else None, "n": len(rs), "poles": len(ids)}
    return values


def registered_reading_permitted(gate1, gate2, gate3, gate4=True):
    return gate1 is True and gate2 is True and gate3 is True and gate4 is True


def count_nonfinite_rows(rows):
    """Amendment 2: a row with any non-finite value in either effect vector."""
    return sum(not (np.isfinite(np.asarray(r["effect_a"], dtype=np.float64)).all()
                    and np.isfinite(np.asarray(r["effect_b"], dtype=np.float64)).all()) for r in rows)


def write_gate_only(out, gates):
    out.mkdir(parents=True, exist_ok=True)
    payload = {"design": "PART142-DESIGN.md", "seed": SEED, "gates": gates}
    (out / "score142.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    lines = ["# Part 142 score", "", "## Gates", ""]
    lines += [f"- **{k}:** `{json.dumps(v, ensure_ascii=False)}`" for k, v in gates.items()]
    (out / "SCORE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    a = parse_args()
    scan, effects_dir, out = Path(a.scan), Path(a.effects), Path(a.out)
    required = [scan / "firings.jsonl", scan / "scan_meta.json", effects_dir / "effects.jsonl",
                effects_dir / "steering.jsonl", effects_dir / "gates.json", Path(a.v2), Path(a.pilot_units)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input file(s): " + ", ".join(missing))
    unit_rows = json.loads(Path(a.units).read_text(encoding="utf-8"))
    score = json.loads(Path(a.score).read_text(encoding="utf-8"))
    score136.BANDS = score136.fixed_bands(unit_rows, score)
    bands = score136.BANDS
    band_fn = score136.band
    firings = load_jsonl(scan / "firings.jsonl")
    effects = load_jsonl(effects_dir / "effects.jsonl")
    steering = load_jsonl(effects_dir / "steering.jsonl")
    for row in effects:
        row["effect_a"] = score136.unpack_vector(row.pop("effect_a_f32_le_b64"))
        row["effect_b"] = score136.unpack_vector(row.pop("effect_b_f32_le_b64"))
    nonfinite_rows = count_nonfinite_rows(effects)
    by_fire = {(r["pole_id"], r["doc_id"], int(r["position"])): r for r in effects}
    by_pole = {}
    for f in firings:
        e = by_fire.get((f["pole_id"], f["doc_id"], int(f["position"])))
        if e:
            by_pole.setdefault(f["pole_id"], []).append(e)
    scan_by_pole = {}
    for f in firings:
        scan_by_pole.setdefault(f["pole_id"], []).append(f)
    meta = json.loads((scan / "scan_meta.json").read_text(encoding="utf-8"))
    v2_data = load_jsonl(Path(a.v2))
    matched = {r["pole_id"]: r["matched_control"] for r in meta.get("matched_poles", [])}
    inverse_match = {control: unit for unit, control in matched.items()}
    unit_by_id = {u["id"]: u for u in unit_rows}
    unit_by_control = {control: unit_by_id[uid] for uid, control in matched.items() if uid in unit_by_id}
    raw_controls = {pid: by_pole.get(pid, []) for pid in set(matched.values())}
    undefined = {}
    too_few_defined = {}
    missing_references = {}
    missing_v2 = {}
    eligible_controls = {}
    for pid, pole_rows in raw_controls.items():
        band_name = band_fn(pole_rows[0]["layer"] if pole_rows else unit_by_control[pid]["layer"])
        a_defined = sum(pearson(r["effect_a"], r["effect_a"]) is not None for r in pole_rows)
        b_defined = sum(pearson(r["effect_b"], r["effect_b"]) is not None for r in pole_rows)
        for effect_key, n_defined in (("effect_a", a_defined), ("effect_b", b_defined)):
            key = (band_name, "matched_pole", effect_key)
            undefined[key] = undefined.get(key, 0) + len(pole_rows) - n_defined
        if min(a_defined, b_defined) < 12:
            too_few_defined[pid] = {"band": band_name, "kind": "matched_pole", "effect_a": a_defined, "effect_b": b_defined}
        elif len(pole_rows) == 24:
            eligible_controls[pid] = pole_rows
    controls = eligible_controls
    rows, control_rows = [], []
    for u in unit_rows:
        uid = u["id"]
        kind = score.get("units", {}).get(uid, {}).get("kind")
        if kind not in ("concept", "slot"):
            continue
        band_name = band_fn(u["layer"])
        v2_matches = [r for r in v2_data if str(r.get("persona", "")).strip().casefold() == str(u.get("persona", "")).strip().casefold()]
        if len(v2_matches) != 1:
            key = (band_name, kind)
            missing_v2[key] = missing_v2.get(key, 0) + 1
        own = by_pole.get(uid, [])
        a_defined = sum(pearson(r["effect_a"], r["effect_a"]) is not None for r in own)
        b_defined = sum(pearson(r["effect_b"], r["effect_b"]) is not None for r in own)
        for effect_key, n_defined in (("effect_a", a_defined), ("effect_b", b_defined)):
            key = (band_name, kind, effect_key)
            undefined[key] = undefined.get(key, 0) + len(own) - n_defined
        if min(a_defined, b_defined) < 12:
            too_few_defined[uid] = {"band": band_name, "kind": kind, "effect_a": a_defined, "effect_b": b_defined}
            continue
        if len(own) != 24:
            continue
        a_vec = [r["effect_a"] for r in own]
        b_vec = [r["effect_b"] for r in own]
        own_a = [r for r in own if pearson(r["effect_a"], r["effect_a"]) is not None]
        own_b = [r for r in own if pearson(r["effect_b"], r["effect_b"]) is not None]
        own_ref_a = reference_for(uid, own_a, controls, matched.get(uid), band_fn)
        own_ref_b = reference_for(uid, own_b, controls, matched.get(uid), band_fn, "effect_b")
        ref_a = own_ref_a
        ref_b = own_ref_b
        n_refs = sum(pid != matched.get(uid) and rows and band_fn(rows[0]["layer"]) == band_name for pid, rows in controls.items())
        if n_refs < 8:
            missing_references[uid] = {"band": band_name, "kind": kind, "available": n_refs}
        steer = next((r["logprob_delta"] for r in steering if r["pole_id"] == uid), None)
        dsrow = v2_matches[0] if len(v2_matches) == 1 else {}
        rows.append({"pole_id": uid, "layer": int(u["layer"]), "band": band_fn(u["layer"]), "persona": u.get("persona"),
                     "kind": kind, "firing_count": len(own), "defined_firing_count_a": a_defined, "defined_firing_count_b": b_defined,
                     "raw_stability": stability(a_vec), "raw_stability_b": stability(b_vec), "reference": ref_a, "reference_b": ref_b,
                     "margin": compute_margin(stability(a_vec), ref_a),
                     "margin_b": compute_margin(stability(b_vec), ref_b),
                     "steering_agreement": pearson(steer, mean_vector([r["effect_a"] for r in own_a])) if steer is not None and own_a else None,
                     "steering_agreement_b": pearson(steer, mean_vector([r["effect_b"] for r in own_b])) if steer is not None and own_b else None,
                     "matched_control": matched.get(uid),
                     "persona_v2": bool(dsrow.get("coherent")) and dsrow.get("kind") not in ("still Qwen or an assistant", "unclear")})
    for pid, own in controls.items():
        if len(own) != 24:
            continue
        own_a = [r for r in own if pearson(r["effect_a"], r["effect_a"]) is not None]
        own_b = [r for r in own if pearson(r["effect_b"], r["effect_b"]) is not None]
        ref_a = reference_for(pid, own_a, controls, pid, band_fn)
        ref_b = reference_for(pid, own_b, controls, pid, band_fn, "effect_b")
        a_vec = [r["effect_a"] for r in own]; b_vec = [r["effect_b"] for r in own]
        control_rows.append({"pole_id": pid, "layer": int(own[0]["layer"]), "band": band_fn(own[0]["layer"]),
                             "kind": "control", "defined_firing_count_a": len(own_a), "defined_firing_count_b": len(own_b),
                             "raw_stability": stability(a_vec), "raw_stability_b": stability(b_vec), "reference": ref_a,
                             "margin": compute_margin(stability(a_vec), ref_a),
                             "matched_unit": inverse_match.get(pid)})

    pilot = json.loads(Path(a.pilot_units).read_text(encoding="utf-8"))
    excluded = set(pilot.get("units", [])) | set(pilot.get("matched_poles", []))
    primary = margin_permutation(rows, "margin", bands, excluded)
    all_units = margin_permutation(rows, "margin", bands)
    v2_rows = [r for r in rows if r["persona_v2"]]
    v2_primary = margin_permutation(v2_rows, "margin", bands, excluded)
    v2_all = margin_permutation(v2_rows, "margin", bands)
    v2_effect_b = margin_permutation(v2_rows, "margin_b", bands, excluded)
    v2_steering = margin_permutation(v2_rows, "steering_agreement", bands, excluded)
    steering_primary = margin_permutation(rows, "steering_agreement", bands, excluded)
    steering_b = margin_permutation(rows, "steering_agreement_b", bands, excluded)
    effect_b = margin_permutation(rows, "margin_b", bands, excluded)
    pair_delta = sorted_vs_matched_median_difference(rows, control_rows, bands, excluded)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    known_firings = [f for f in firings if f.get("pole_id") == "gate:L22:N2742"]
    known_count = decoded_known_count(known_firings, tokenizer)
    matched_counts = [decoded_known_count([f for f in firings if f.get("pole_id") == pid], tokenizer) for pid in matched.values()]
    p99 = nearest_rank(matched_counts, .99)
    coverage = sum(len(scan_by_pole.get(u["id"], [])) >= 24 for u in unit_rows if score.get("units", {}).get(u["id"], {}).get("kind") in ("concept", "slot")) / 446
    runtime_gates = json.loads((effects_dir / "gates.json").read_text(encoding="utf-8")) if (effects_dir / "gates.json").exists() else {}
    known_pass = len(known_firings) == 24 and known_count >= 5 and p99 is not None and known_count >= p99 + 3
    legacy = legacy_gate3(by_pole, band_fn, bands)
    gates = {"gate_1_do_nothing_hook_bit_identical": runtime_gates.get("do_nothing_hook_bit_identical"),
             "gate_2_coverage_units_with_24_of_446": coverage, "gate_2_pass": coverage >= .75,
             "gate_3_known_unit_L22_N2742_count": known_count,
             "gate_3_matched_poles_count_p99_nearest_rank": p99,
             "gate_3_known_unit_pass": bool(known_pass),
             "gate_4_nonfinite_effect_rows": nonfinite_rows, "gate_4_pass": nonfinite_rows == 0,
             "part_136_gate_3_median_abs_r_by_band_for_record": legacy,
             "part_136_gate_3_pass_at_0_02_for_record": bool(legacy) and all(x["median_abs_r"] is not None and x["median_abs_r"] <= .02 for x in legacy.values()),
             "registered_reading_permitted": registered_reading_permitted(runtime_gates.get("do_nothing_hook_bit_identical"), coverage >= .75, bool(known_pass), nonfinite_rows == 0),
             "diagnostic_counts": {
                 "undefined_firings_by_band_kind_effect": {"|".join(k): v for k, v in sorted(undefined.items())},
                 "poles_with_fewer_than_12_defined_firings": too_few_defined,
                 "units_with_fewer_than_8_reference_poles": missing_references,
                 "units_without_unique_v2_match_by_band_kind": {"|".join(k): v for k, v in sorted(missing_v2.items())},
             }}
    if not gates["registered_reading_permitted"]:
        write_gate_only(out, gates)
        return

    def med(values):
        return safe_median(values)
    medians = lambda arr, field, kind: {name: med([r[field] for r in arr if r["band"] == name and r["kind"] == kind and r.get(field) is not None]) for _, _, name in bands}
    empty_groups = {f"{name}|{kind}": True for _, _, name in bands for kind in ("concept", "slot")
                    if not any(r["band"] == name and r["kind"] == kind and r.get("margin") is not None for r in rows)}
    raw_delta = margin_permutation(rows, "raw_stability", bands, excluded)
    controls_low = med([r["raw_stability"] for r in control_rows if r["band"] == bands[0][2]])
    controls_high = med([r["raw_stability"] for r in control_rows if r["band"] == bands[-1][2]])
    low_steer = med([r["steering_agreement"] for r in rows if r["band"] == bands[0][2]])
    reference_medians = {}
    for _, _, name in bands:
        reference_medians[name] = {
            "sorted_units": med([r["reference"] for r in rows if r["band"] == name and r.get("reference") is not None]),
            "matched_poles": med([r["reference"] for r in control_rows if r["band"] == name and r.get("reference") is not None]),
        }
    ref_medians = [v for band_values in reference_medians.values() for v in band_values.values() if v is not None]
    predictions = {
        "142.1": {"result": None if controls_low is None or controls_high is None else controls_high-controls_low,
                  "n": {"low_band_matched_poles": sum(r["band"] == bands[0][2] and r["raw_stability"] is not None for r in control_rows),
                        "high_band_matched_poles": sum(r["band"] == bands[-1][2] and r["raw_stability"] is not None for r in control_rows)},
                  "threshold": .1, "held": None if controls_low is None or controls_high is None else controls_high-controls_low >= .1},
        "142.2": {"result": primary, "threshold": .05, "held": None if primary["difference"] is None else primary["difference"] >= .05},
        "142.3": {"result": pair_delta, "threshold": .05, "held": None if pair_delta["difference"] is None else pair_delta["difference"] >= .05},
        "142.4": {"result": steering_primary, "threshold": .05, "held": None if steering_primary["difference"] is None else steering_primary["difference"] >= .05},
        "142.5": {"result": low_steer, "threshold": .2, "held": None if low_steer is None else low_steer <= .2},
        "142.6": {"effect_b_direction_matches_142_2": None if effect_b["difference"] is None or primary["difference"] is None else (effect_b["difference"] >= 0) == (primary["difference"] >= 0),
                  "effect_b_direction_matches_142_4": None if steering_b["difference"] is None or steering_primary["difference"] is None else (steering_b["difference"] >= 0) == (steering_primary["difference"] >= 0),
                  "result": {"effect_b_margin": effect_b, "effect_b_steering": steering_b}, "held": None if effect_b["difference"] is None or steering_b["difference"] is None or primary["difference"] is None or steering_primary["difference"] is None else ((effect_b["difference"] >= 0) == (primary["difference"] >= 0) and (steering_b["difference"] >= 0) == (steering_primary["difference"] >= 0))},
        "142.7": {"result": reference_medians, "threshold": .1,
                  "held": None if not ref_medians or any(v is None for cells in reference_medians.values() for v in cells.values()) else all(v < .1 for v in ref_medians)},
    }
    effect_size_a = {name: med([float(np.linalg.norm(np.asarray(row["effect_a"], dtype=np.float64))) for rs in by_pole.values() for row in rs if band_fn(row["layer"]) == name]) for _, _, name in bands}
    effect_size_b = {name: med([float(np.linalg.norm(np.asarray(row["effect_b"], dtype=np.float64))) for rs in by_pole.values() for row in rs if band_fn(row["layer"]) == name]) for _, _, name in bands}
    payload = {"design": "PART142-DESIGN.md", "seed": SEED, "permutations": N_PERM,
               "bands": bands, "counts": {"sorted_units_with_effects": len(rows), "matched_poles": len(control_rows), "units_with_24": sum(len(scan_by_pole.get(u["id"], [])) >= 24 for u in unit_rows if score.get("units", {}).get(u["id"], {}).get("kind") in ("concept", "slot")), "pilot_excluded": len(excluded), "empty_margin_groups": empty_groups, **gates["diagnostic_counts"]},
               "gates": gates, "predictions": predictions,
               "summaries": {"raw_stability_by_band_kind": {name: {kind: med([r["raw_stability"] for r in rows if r["band"] == name and r["kind"] == kind]) for kind in ("concept", "slot")} for _,_,name in bands}, "margin_concept": medians(rows, "margin", "concept"), "margin_slot": medians(rows, "margin", "slot"), "reference_by_band": {name: med([r["reference"] for r in rows+control_rows if r["band"] == name and r.get("reference") is not None]) for _,_,name in bands}, "median_effect_size_l2_by_band": effect_size_a, "median_effect_size_b_l2_by_band": effect_size_b, "effect_a_concept_minus_slot_raw_primary": raw_delta, "margin_primary_excluding_pilot": primary, "raw_vs_margin_threshold_verdict_diff": None if raw_delta["difference"] is None or primary["difference"] is None else ((raw_delta["difference"] >= .05) != (primary["difference"] >= .05)), "margin_all_units": all_units, "deepseek_v2_personas_primary": v2_primary, "deepseek_v2_personas_all": v2_all, "deepseek_v2_threshold_verdict_diff": None if v2_primary["difference"] is None or primary["difference"] is None else ((v2_primary["difference"] >= .05) != (primary["difference"] >= .05)), "deepseek_v2_effect_b_primary": v2_effect_b, "deepseek_v2_steering_primary": v2_steering, "matched_pair_margin_difference": pair_delta},
               "units": rows, "matched_poles": control_rows}
    out.mkdir(parents=True, exist_ok=True)
    (out / "score142.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# Part 142 score", "", f"- Registered reading permitted: {gates['registered_reading_permitted']}.", f"- Coverage: {coverage:.4f} of 446 sorted units.", f"- DeepSeek v2 primary difference: `{json.dumps(v2_primary)}`.", f"- Raw and margin primary comparisons: `{json.dumps(raw_delta)}`; `{json.dumps(primary)}`.", "", "## Predictions", ""]
    lines += [f"- **{key}:** `{json.dumps(value, ensure_ascii=False)}`" for key, value in predictions.items()]
    (out / "SCORE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
