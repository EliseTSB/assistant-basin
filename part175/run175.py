#!/usr/bin/env python3
"""Part 175: discover a workable flat-push dose, then run its census."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from decimal import Decimal
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for entry in (str(ROOT), str(HERE), str(ROOT / "part86"), str(ROOT / "part171"),
              str(ROOT / "part172"), str(ROOT / "part173"), str(ROOT / "part174")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import common86
import run171
import run172
import run173
import run174
import score173
from part171.prompts171 import FIT_PROMPTS, TARGETS
from part173.prompts173 import NEW44B


STAGE_A_SEEDS = tuple(range(175901, 175905))
STAGE_A_LADDER = tuple(Decimal(value) for value in
                       ("0.10", "0.15", "0.20", "0.25", "0.30", "0.35", "0.40", "0.45"))
STAGE_A_BROKEN_MAX = 8
STAGE_A_ROWS_PER_DIRECTION = 44
F100D_SEEDS = tuple(run174.DESIGN["families"]["f100d"]["seeds"])
STAGE_A_SET_NAMES = tuple(f"f100d_d{seed}" for seed in STAGE_A_SEEDS)
F100D_SET_NAMES = tuple(f"f100d_d{seed}" for seed in F100D_SEEDS)
PART174_REPLAYS = (("u100d_d174001", 0.5), ("f100d_d174101", 0.5))
PART174_STAGE0 = Path("results/part174/stage0.json")
PART174_SET_HASH = Path("results/part174/sets_174.pt.sha256")
PART174_ANSWERS = Path("results/part174/answers.jsonl")
PART174_PREDICTIONS = Path("results/part174/prediction_table.json")
PART171_ANSWERS = Path("results/part171/answers.jsonl")
PART171_SUMMARY = Path("results/part171/sets.json")
MODEL_DEFAULT = run171.MODEL_DEFAULT
BATCH_SIZE = 44
MAX_NEW_TOKENS = 80


def _decimal(value) -> Decimal:
    """Convert a registered decimal dose without carrying binary float noise."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _dose_slug(value) -> str:
    dose = _decimal(value).normalize()
    sign = "m" if dose < 0 else "p"
    magnitude = format(abs(dose), "f").replace(".", "p")
    return f"dose_{sign}{magnitude}"


def dose_arm(set_name: str, dose) -> str:
    """Dose-bearing arm key whose spelling is independent of float formatting."""
    return f"{set_name}_{_dose_slug(dose)}"


def census_doses(d_star) -> tuple[Decimal, Decimal, Decimal]:
    primary = _decimal(d_star)
    return (primary, primary * Decimal("0.75"), primary * Decimal("0.5"))


def signed_doses(magnitudes) -> tuple[Decimal, ...]:
    return tuple(value for magnitude in magnitudes
                 for value in (_decimal(magnitude), -_decimal(magnitude)))


def prompts() -> list[dict]:
    return run174.prompts()


def select_d_star(direction_rows, ladder=STAGE_A_LADDER, broken_max=STAGE_A_BROKEN_MAX):
    """Return the largest dose whose pooled median and every lower dose pass.

    Input is a sequence of Stage A direction rows with ``dose``, ``sign``, ``seed`` and
    ``broken``.  The function is pure so the prefix rule can be tested independently.
    """
    rows = [dict(row) for row in direction_rows]
    result_table = []
    prefix_ok = True
    chosen = None
    ladder_values = tuple(_decimal(value) for value in ladder)
    for dose in ladder_values:
        current = [row for row in rows if _decimal(abs(float(row["dose"]))) == dose]
        plus = [int(row["broken"]) for row in current if float(row["sign"]) > 0]
        minus = [int(row["broken"]) for row in current if float(row["sign"]) < 0]
        pooled = [int(row["broken"]) for row in current]
        if len(plus) != 4 or len(minus) != 4 or len(pooled) != 8:
            raise ValueError(f"Stage A dose {dose} must have four draws at each sign")
        pooled_median = float(median(pooled))
        qualifies = pooled_median <= float(broken_max)
        prefix_ok = bool(prefix_ok and qualifies)
        if prefix_ok:
            chosen = dose
        result_table.append({
            "dose": float(dose),
            "positive_broken_by_draw": plus,
            "positive_median_broken": float(median(plus)),
            "negative_broken_by_draw": minus,
            "negative_median_broken": float(median(minus)),
            "pooled_broken_by_signed_direction": pooled,
            "pooled_median_broken": pooled_median,
            "median_at_most_bar": bool(qualifies),
            "all_lower_doses_meet_bar": bool(prefix_ok),
            "selected_as_d_star": bool(prefix_ok),
        })
    return {"status": "selected" if chosen is not None else "stopped",
            "d_star": float(chosen) if chosen is not None else None,
            "ceiling": bool(chosen is not None and chosen == ladder_values[-1]),
            "broken_max": int(broken_max), "dose_table": result_table}


def workable_step(broken_by_dose, ladder=STAGE_A_LADDER, broken_max=STAGE_A_BROKEN_MAX):
    """Highest initial passing ladder step for one draw and one sign; zero means <0.10."""
    step = 0
    for index, dose in enumerate(ladder, start=1):
        if int(broken_by_dose[_decimal(dose)]) <= int(broken_max):
            step = index
        else:
            break
    return step


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _expected_hash(path_or_text) -> str:
    if isinstance(path_or_text, str) and len(path_or_text.split()) == 1:
        text = path_or_text
    elif isinstance(path_or_text, str) and "\n" in path_or_text:
        text = path_or_text
    else:
        text = Path(path_or_text).read_text(encoding="utf-8")
    fields = text.split()
    if not fields or len(fields[0]) != 64:
        raise ValueError("Part 174 set hash file does not start with a SHA-256 digest")
    return fields[0].lower()


def rebuild_part174_sets(stage0, widths=None):
    """Recreate Part 174's saved 60 draws from registered streams and stage0 metadata.

    Part 174's binary artifact is intentionally not an input.  Its committed stage0 records
    each draw's scale and metric metadata; the vectors themselves are regenerated from seeds.
    """
    import torch

    widths = ({int(layer): int(run174.DESIGN["width"])
               for layer in run171.LAYERS} if widths is None else
              {int(layer): int(width) for layer, width in widths.items()})
    source_metrics = stage0.get("random_set_metrics", {})
    if set(source_metrics) != set(run174.SET_NAMES):
        raise ValueError("Part 174 stage0 must contain metrics for all 60 registered draws")
    counts = {}
    for layer in widths:
        values = {int(source_metrics[name]["layer_counts"][str(layer)]) for name in run174.SET_NAMES
                  if run174.FAMILY_BY_SET[name] == "u100d"}
        if len(values) != 1:
            raise ValueError(f"Part 174 u100d layer-count metadata disagree at layer {layer}")
        counts[layer] = values.pop()
    raw = run174.draw_u100d_raw(counts, widths=widths)
    raw.update(run174.draw_f100d_raw(widths=widths))

    arms = {}
    layer_counts = {}
    seed_streams = {}
    global_scales = {}
    metrics = {}
    for name in run174.SET_NAMES:
        meta = source_metrics[name]
        factor = float(meta["global_scale"])
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"invalid recorded Part 174 scale for {name}")
        arms[name] = {int(layer): run174._scaled_compact_rep(rep, factor)
                      for layer, rep in raw[name].items()}
        # Part 174 built the layer-count and stream dictionaries separately, then shared
        # each dictionary with its metric row.  JSON parsing reuses equal key strings, so
        # make fresh keys per map to preserve the original torch-save object graph.
        layer_counts[name] = {str(int(key)): value
                              for key, value in meta["layer_counts"].items()}
        seed_streams[name] = {str(int(key)): value
                              for key, value in meta["stream_entropy"].items()}
        global_scales[name] = factor
        # Keep the source builder's insertion order and literal metric keys.  Torch's
        # archive pickle memoizes these shared key strings, so reconstructing rows with
        # ``dict(meta)`` (from JSON) changes the committed artifact digest.
        metric = {"family": run174.FAMILY_BY_SET[name],
                  "seed": int(name.rsplit("d", 1)[1]),
                  "global_scale": factor,
                  "nominal_write_norm": float(meta["nominal_write_norm"]),
                  "n1_nominal_write_norm": float(meta["n1_nominal_write_norm"]),
                  "cf": float(meta["cf"]),
                  "layer_counts": layer_counts[name],
                  "stream_entropy": seed_streams[name],
                  "cr": float(meta["cr"]),
                  "write_norm": float(meta["write_norm"])}
        metrics[name] = metric
    reference_norms = {float(source_metrics[name]["n1_nominal_write_norm"])
                       for name in run174.SET_NAMES}
    if len(reference_norms) != 1:
        raise ValueError("Part 174 stage0 has inconsistent n1 reference write norms")
    return {"part": 174, "arms": arms, "layer_counts": layer_counts,
            "seed_streams": seed_streams, "global_scales": global_scales,
            "metrics": metrics,
            "source_n1_nominal_write_norm": reference_norms.pop()}


def _recomputed_part174_metrics(stage0, artifact, layers, widths, fit_vector,
                                tok=None, model=None, projection_vector=None):
    """Recompute Part 174's registered metrics from random draws and rebuilt writes."""
    source_metrics = stage0["random_set_metrics"]
    counts = {}
    for layer in widths:
        values = {int(source_metrics[name]["layer_counts"][str(layer)])
                  for name in run174.SET_NAMES if run174.FAMILY_BY_SET[name] == "u100d"}
        if len(values) != 1:
            raise ValueError(f"Part 174 u100d layer-count metadata disagree at layer {layer}")
        counts[int(layer)] = values.pop()

    raw = run174.draw_u100d_raw(counts, widths=widths)
    raw.update(run174.draw_f100d_raw(widths=widths))
    raw_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in raw.items()}
    raw_writes = run173._nominal_writes(layers, raw_vectors)
    reference_norm = _reference_norm(stage0)
    recomputed_scales = {}
    for name in run174.SET_NAMES:
        raw_norm = float(np.linalg.norm(raw_writes[name]))
        if not math.isfinite(raw_norm) or raw_norm <= 0:
            raise ValueError(f"{name} has a zero or non-finite unscaled write")
        recomputed_scales[name] = float(reference_norm / raw_norm)

    rebuilt_vectors = {name: run173._rep_vectors(reps, widths)
                       for name, reps in artifact["arms"].items()}
    rebuilt_writes = run173._nominal_writes(layers, rebuilt_vectors)
    if projection_vector is not None:
        output_direction = np.asarray(projection_vector, dtype=np.float64).reshape(-1)
        output_norm = float(np.linalg.norm(output_direction))
        if not math.isfinite(output_norm) or output_norm <= 0:
            raise ValueError("Part 174 output-projection vector has a zero or non-finite norm")
        output_direction /= output_norm
        fit_direction = np.asarray(fit_vector.detach().to(
            device="cpu", dtype=__import__("torch").float32).numpy()
            if hasattr(fit_vector, "detach") else fit_vector, dtype=np.float64).reshape(-1)
        fit_norm = float(np.linalg.norm(fit_direction))
        if not math.isfinite(fit_norm) or fit_norm <= 0:
            raise ValueError("Part 171 fit vector has a zero or non-finite norm")
        projections = {}
        for name, write in rebuilt_writes.items():
            write = np.asarray(write, dtype=np.float64).reshape(-1)
            write_norm = float(np.linalg.norm(write))
            projections[name] = {"cf": float(write @ fit_direction / (write_norm * fit_norm)),
                                 "cr": float(write @ output_direction / write_norm),
                                 "write_norm": write_norm}
    else:
        if tok is None or model is None:
            raise ValueError("Part 174 output-projection recomputation needs tokenizer and model")
        projections, _projection_stats = run174.output_projection_vectors(
            tok, model, {"v": fit_vector}, rebuilt_writes)

    metrics = {}
    for name in run174.SET_NAMES:
        projection = projections[name]
        metrics[name] = {
            "global_scale": recomputed_scales[name],
            "nominal_write_norm": float(projection["write_norm"]),
            "write_norm": float(projection["write_norm"]),
            "cf": float(projection["cf"]),
            "cr": float(projection["cr"]),
        }
    return metrics


def verify_part174_rebuild(stage0, expected_hash, work_dir, widths=None, layers=None,
                           fit_vector=None, tok=None, model=None, projection_vector=None,
                           tolerance=1e-6):
    """Check every rebuilt draw against stage0; retain the archive SHA as diagnostics only."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if layers is None:
        raise ValueError("Part 174 metric verification needs model layers")
    widths = (run172.widths_of(layers) if widths is None else
              {int(layer): int(width) for layer, width in widths.items()})
    artifact = rebuild_part174_sets(stage0, widths=widths)
    # Part 174 hashes the temporary candidate before rename; PyTorch records this basename
    # inside the archive, so preserve it to reproduce the committed digest exactly.
    path = work_dir / ".sets_174.pt.candidate"
    try:
        expected = _expected_hash(expected_hash)
        hash_parse_error = None
    except (OSError, TypeError, ValueError) as exc:
        expected = None
        hash_parse_error = repr(exc)
    stage0_hash = str(stage0.get("artifact_sha256", "")).lower()
    if fit_vector is None:
        raise ValueError("Part 174 metric verification needs the Part 171 fit vector")
    try:
        recomputed = _recomputed_part174_metrics(
            stage0, artifact, layers, widths, fit_vector,
            tok=tok, model=model, projection_vector=projection_vector)
        metric_rows = []
        mismatches = []
        fields = ("global_scale", "nominal_write_norm", "write_norm", "cf", "cr")
        source_metrics = stage0["random_set_metrics"]
        if set(source_metrics) != set(run174.SET_NAMES):
            mismatches.append({"kind": "draw_coverage", "expected": len(run174.SET_NAMES),
                               "observed": len(source_metrics)})
        for name in run174.SET_NAMES:
            expected_row = source_metrics.get(name, {})
            actual_row = recomputed.get(name, {})
            field_results = {}
            row_passed = True
            for field in fields:
                try:
                    expected_value = float(expected_row[field])
                    actual_value = float(actual_row[field])
                    finite = math.isfinite(expected_value) and math.isfinite(actual_value)
                    error = abs(actual_value - expected_value) if finite else None
                except (KeyError, TypeError, ValueError):
                    expected_value = actual_value = None
                    finite = False
                    error = None
                field_passed = bool(finite and error <= float(tolerance))
                field_results[field] = {"expected": expected_value, "recomputed": actual_value,
                                        "absolute_error": error, "passed": field_passed}
                if not field_passed:
                    row_passed = False
                    mismatches.append({"set": name, "field": field,
                                       "expected": expected_value, "recomputed": actual_value,
                                       "absolute_error": error})
            metric_rows.append({"set": name, "passed": row_passed, "metrics": field_results})
        metrics_passed = not mismatches and len(metric_rows) == len(run174.SET_NAMES)
        rebuilt = run174._save_artifact(path, artifact)
        raw_hash_matches = bool(expected and rebuilt == expected)
        raw_hash_row = {"check": "raw_archive_sha256", "expected_sha256": expected,
                        "stage0_sha256": stage0_hash or None, "rebuilt_sha256": rebuilt,
                        "matches": raw_hash_matches, "affects_passing": False,
                        "reason": ("PyTorch zip archives embed a per-process serialization_id; "
                                   "raw bytes can differ even when the draws and metrics match.")}
        if hash_parse_error:
            raw_hash_row["parse_error"] = hash_parse_error
        if stage0_hash and expected and stage0_hash != expected:
            raw_hash_row["stage0_hash_file_agrees"] = False
        else:
            raw_hash_row["stage0_hash_file_agrees"] = bool(stage0_hash and expected)
        metrics_row = {"check": "draw_by_draw_metrics", "passed": bool(metrics_passed),
                       "draws": len(metric_rows), "expected_draws": len(run174.SET_NAMES),
                       "tolerance": float(tolerance), "metric_rows": metric_rows,
                       "mismatches": mismatches}
        return artifact, {"passed": bool(metrics_passed), "verification": "draw_by_draw_metrics",
                          "tolerance": float(tolerance), "draws": len(metric_rows),
                          "expected_draws": len(run174.SET_NAMES),
                          "rows": [metrics_row, raw_hash_row]}
    finally:
        path.unlink(missing_ok=True)
        Path(str(path) + ".sha256").unlink(missing_ok=True)


def _reference_norm(stage0):
    metrics = stage0["random_set_metrics"]
    values = {float(metrics[name]["n1_nominal_write_norm"]) for name in run174.SET_NAMES}
    if len(values) != 1:
        raise ValueError("Part 174 reference length differs across the 60 draws")
    value = values.pop()
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Part 174 reference length is not positive and finite")
    return value


def build_stage_a_sets(layers, widths, fit_vector, reference_norm):
    """Draw and length-match the four fresh Stage A directions using Part 174's f100d stream."""
    raw = run174.draw_f100d_raw(seeds=STAGE_A_SEEDS, widths=widths)
    vectors = {name: run173._rep_vectors(reps, widths) for name, reps in raw.items()}
    writes = run173._nominal_writes(layers, vectors)
    scaled, metrics = {}, {}
    for name in STAGE_A_SET_NAMES:
        norm = float(np.linalg.norm(writes[name]))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError(f"{name} has a zero or non-finite nominal write")
        factor = float(reference_norm / norm)
        scaled[name] = {int(layer): run174._scaled_compact_rep(rep, factor)
                        for layer, rep in raw[name].items()}
    scaled_vectors = {name: run173._rep_vectors(reps, widths) for name, reps in scaled.items()}
    scaled_writes = run173._nominal_writes(layers, scaled_vectors)
    v = np.asarray(fit_vector.detach().to(device="cpu", dtype=__import__("torch").float32).numpy()
                   if hasattr(fit_vector, "detach") else fit_vector, dtype=np.float64).reshape(-1)
    vnorm = float(np.linalg.norm(v))
    if not math.isfinite(vnorm) or vnorm <= 0:
        raise ValueError("Part 171 fit vector has a zero or non-finite norm")
    for name in STAGE_A_SET_NAMES:
        write = scaled_writes[name]
        norm = float(np.linalg.norm(write))
        metrics[name] = {"seed": int(name.rsplit("d", 1)[1]),
                         "nominal_write_norm": norm,
                         "reference_norm": float(reference_norm),
                         "length_relative_error": abs(norm / reference_norm - 1.0),
                         "cf": float(write @ v / (norm * vnorm)),
                         "global_scale": float(reference_norm / np.linalg.norm(writes[name]))}
    return scaled, metrics


def gate8_stage_a(metrics, fit_vector_size):
    failures = []
    expected = set(STAGE_A_SET_NAMES)
    if set(metrics) != expected:
        failures.append({"kind": "stage_a_draw_coverage", "expected": sorted(expected),
                         "observed": sorted(metrics)})
    if int(fit_vector_size) <= 0:
        failures.append({"kind": "fit_vector_size", "value": int(fit_vector_size)})
    for name in sorted(expected):
        row = metrics.get(name, {})
        try:
            length_error = float(row["length_relative_error"])
            cosine = float(row["cf"])
            finite = math.isfinite(length_error) and math.isfinite(cosine)
        except (KeyError, TypeError, ValueError):
            length_error = cosine = float("nan")
            finite = False
        if not finite:
            failures.append({"kind": "nonfinite", "set": name})
            continue
        if length_error > float(run174.DESIGN["gate8"]["length_relative_tolerance"]) + 1e-12:
            failures.append({"kind": "length", "set": name,
                             "relative_error": length_error,
                             "maximum": run174.DESIGN["gate8"]["length_relative_tolerance"]})
        if abs(cosine) > float(run174.DESIGN["gate8"]["absolute_cosine_max"]):
            failures.append({"kind": "cf", "set": name, "absolute_cf": abs(cosine),
                             "maximum": run174.DESIGN["gate8"]["absolute_cosine_max"]})
    return {"passed": not failures, "draws": len(metrics), "expected_draws": 4,
            "failures": failures,
            "absolute_cf_max": run174.DESIGN["gate8"]["absolute_cosine_max"],
            "length_relative_tolerance": run174.DESIGN["gate8"]["length_relative_tolerance"]}


def measure_gate3_residual_norms(model, tok, layers, device):
    """Mean final-prompt-position residual norm for Gate 3's eight plain contexts."""
    import torch

    values = {}
    handles = []
    questions = [row["question"] for row in prompts()[:run174.DESIGN["gate3"]["prompts"]]]
    try:
        inputs = run171.encode_batch(tok, questions, device, [None] * len(questions))
        for layer_no in run171.LAYERS:
            def capture(_module, _inputs, output, layer_no=layer_no):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if hidden.ndim != 3:
                    raise ValueError(f"layer {layer_no} did not return a rank-three residual stream")
                norms = torch.linalg.vector_norm(hidden[:, -1, :].detach().float(), dim=-1)
                values[layer_no] = norms.detach().to(device="cpu", dtype=torch.float64).tolist()
            handles.append(layers[int(layer_no)].register_forward_hook(capture))
        with torch.inference_mode():
            model(**inputs, use_cache=False)
        expected = len(questions)
        means = {str(layer): float(np.mean(values[layer])) for layer in run171.LAYERS
                 if layer in values and len(values[layer]) == expected}
        if len(means) != len(run171.LAYERS) or any(not math.isfinite(value) or value <= 0
                                                    for value in means.values()):
            raise ValueError("one or more Gate 3 residual norms were missing or invalid")
        return {"status": "read", "contexts": expected,
                "position": "last prompt token", "mean_residual_norm_by_layer": means}
    except Exception as exc:
        return {"status": "error", "contexts": len(questions), "error": repr(exc)}
    finally:
        for handle in handles:
            handle.remove()


def measure_written_layer_norms(sets, set_name, family, dose, layers, widths, residuals,
                               layer_weights):
    """Per-layer ||W_l x_l|| and its scale relative to the Gate 3 residual stream."""
    import torch

    try:
        pushes = run172.set_pushes({"arms": {set_name: sets["arms"][set_name]}},
                                   set_name, float(dose), widths, "cpu")
        by_layer = {}
        sum_written = 0.0
        for layer_no in run171.LAYERS:
            weight = layer_weights[int(layer_no)]
            vector = pushes[int(layer_no)].detach().to(device="cpu", dtype=torch.float32).numpy().astype(np.float64)
            written = weight @ vector
            norm = float(np.linalg.norm(written))
            if not math.isfinite(norm):
                raise ValueError(f"{set_name}/L{layer_no} has a non-finite written norm")
            sum_written += norm
            residual_norm = (residuals.get("mean_residual_norm_by_layer", {}).get(str(layer_no))
                             if residuals.get("status") == "read" else None)
            ratio = (norm / float(residual_norm)
                     if residual_norm is not None and float(residual_norm) > 0 else None)
            by_layer[str(layer_no)] = {"written_norm": norm,
                                       "mean_residual_stream_norm": residual_norm,
                                       "written_over_residual_norm": ratio}
        return {"family": family, "set": set_name, "seed": int(set_name.rsplit("d", 1)[1]),
                "dose": float(dose), "layers": by_layer,
                "sum_written_layer_norms": float(sum_written),
                "written_layers": len(by_layer), "status": "read"}
    except Exception as exc:
        return {"family": family, "set": set_name, "seed": int(set_name.rsplit("d", 1)[1]),
                "dose": float(dose), "layers": {}, "sum_written_layer_norms": None,
                "written_layers": 0, "status": "error", "error": repr(exc)}


def layer_norms_document(residuals, rows, status="in_progress"):
    errors = [row for row in rows if row.get("status") != "read"]
    return {"part": 175, "status": "error" if residuals.get("status") != "read" or errors else status,
            "measurement": "||W_l x_l|| per layer, sum of layer norms, and layer norm divided by the mean residual-stream norm on Gate 3's eight plain contexts",
            "residual_norms": residuals, "rows": rows,
            "errors": ([{"source": "gate3_residual_norms", "error": residuals.get("error")}] 
                       if residuals.get("status") != "read" else []) +
                      [{"set": row.get("set"), "dose": row.get("dose"),
                        "error": row.get("error")} for row in errors]}


def _fit_vector(model, tok, layers, args, config):
    if "fit_vector" in config:
        return config["fit_vector"], config.get("fit_vector_diagnostics", {
            "source": "provided", "direction_cosine_vs_stored": None,
            "norm_error_relative": None})
    stored_path = Path(config.get("part171_sets_path",
                                  getattr(args, "part171_sets", run174.PART171_SETS)))
    stored_vector = config.get("part171_stored_vector")
    if stored_vector is None and stored_path.is_file():
        stored_sets = run174._load_artifact(stored_path)
        stored_vector = stored_sets.get("v")
    answers_path = Path(args.part171_answers)
    summary_path = Path(args.part171_summary)
    summary = read_json(summary_path, {}) or {}
    expected_value = (summary.get("targets", {}).get("google", {}).get("v_norm")
                      if summary else None)
    if stored_vector is not None and not config.get("recompute_fit_vector"):
        vector = stored_vector
        actual = float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))
        expected = float(expected_value if expected_value is not None else actual)
        return vector, {"source": "Part 171 sets_google.pt", "stored_norm": expected,
                        "used_norm": actual, "norm_error_relative": (
                            (actual - expected) / expected if expected else None),
                        "direction_cosine_vs_stored": 1.0,
                        "direction_comparison": "stored vector used directly"}

    if not answers_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Part 171 answers and sets.json are required to reconstruct its fit vector")
    answers = run171.read_jsonl(answers_path)
    fit, _stats = run171._fit_recipe(
        model, tok, layers, FIT_PROMPTS, answers,
        {"default": None, "google": TARGETS["google"]}, args.device, seed=171)
    vector = fit["google"]["v"]
    expected = float(expected_value)
    actual_vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    actual = float(np.linalg.norm(actual_vector))
    if not math.isfinite(actual) or actual <= 0 or not math.isfinite(expected) or expected <= 0:
        raise ValueError("Part 171 fit vector or recorded norm is zero or non-finite")

    cosine = None
    comparison = "not available: job copy has answers.jsonl and sets.json but no sets_google.pt vector"
    if stored_vector is not None:
        reference = np.asarray(stored_vector, dtype=np.float64).reshape(-1)
        reference_norm = float(np.linalg.norm(reference))
        if reference.size == actual_vector.size and math.isfinite(reference_norm) and reference_norm > 0:
            cosine = float(actual_vector @ reference / (actual * reference_norm))
            comparison = "cosine against Part 171's stored v"
    diagnostics = {"source": "recomputed_from_part171_answers", "stored_norm": expected,
                   "recomputed_norm": actual,
                   "norm_error_relative": (actual - expected) / expected,
                   "norm_error_absolute": abs(actual - expected),
                   "direction_cosine_vs_stored": cosine,
                   "direction_comparison": comparison}
    return vector, diagnostics


def _record(spec, prompt, answer):
    return run174._record(spec, prompt, answer)


def _append(path, records, done):
    run172.append_answers(path, records, done)


def _batch_for_arm(model, tok, layers, arm, set_name, dose, prompt_rows, args, sets):
    spec = {"arm": arm, "set": set_name, "dose": float(dose)}
    generated = run174.run_spec(model, tok, layers, spec,
                                [row["question"] for row in prompt_rows], args.device,
                                args.max_new_tokens, sets, {}, run172.widths_of(layers))
    if len(generated) != len(prompt_rows):
        raise RuntimeError(f"{arm} generated {len(generated)} answers, expected {len(prompt_rows)}")
    return [_record(spec, prompt, answer) for prompt, answer in zip(prompt_rows, generated)]


def _ensure_controls(model, tok, layers, args, sets, reference_answers):
    answer_path = Path(args.out_dir) / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = run174.read_jsonl(answer_path) if answer_path.is_file() else []
    done = {run174.answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("Part 175 answers.jsonl contains duplicate row keys")
    prompt_rows = prompts()

    # Replay at the same arm names, signs, prompt batch, and 80-token cap as Part 174.
    for set_name, dose in PART174_REPLAYS:
        base_arm = run174.dose_arm(set_name, dose)
        replay_arm = f"replay:{base_arm}"
        if all((replay_arm, row["prompt_id"]) in done for row in prompt_rows):
            continue
        spec = {"arm": base_arm, "set": set_name, "dose": float(dose)}
        generated = run174.run_spec(model, tok, layers, spec,
                                    [row["question"] for row in prompt_rows], args.device,
                                    args.max_new_tokens, sets, {}, run172.widths_of(layers))
        if len(generated) != len(prompt_rows):
            raise RuntimeError(f"{replay_arm} generated {len(generated)} answers, expected {len(prompt_rows)}")
        records = [run174._replay_record(spec, prompt, answer)
                   for prompt, answer in zip(prompt_rows, generated)]
        _append(answer_path, records, done)
        answers = run174.read_jsonl(answer_path)

    # The plain and role rows remain the first planned arms, as in Part 174.
    for spec in ({"arm": "none"}, {"arm": "role", "role": True}):
        if all((spec["arm"], row["prompt_id"]) in done for row in prompt_rows):
            continue
        roles = [TARGETS["google"]] * len(prompt_rows) if spec.get("role") else None
        generated = run171.generate_batch(model, tok, layers,
                                          [row["question"] for row in prompt_rows], roles,
                                          args.device, args.max_new_tokens, seed=172)
        records = [_record(spec, prompt, answer)
                   for prompt, answer in zip(prompt_rows, generated)]
        _append(answer_path, records, done)
        answers = run174.read_jsonl(answer_path)
        if spec["arm"] == "none":
            gate2 = run173.gate2(answers, prompt_rows)
            if not gate2["passed"]:
                return answers, {"gate2": gate2, "gate5": None,
                                 "failed": "Part 175 Gate 2 failed after none arm"}
        else:
            gate5 = run173.gate5(answers, prompt_rows)
            if not gate5["passed"]:
                return answers, {"gate2": run173.gate2(answers, prompt_rows), "gate5": gate5,
                                 "failed": "Part 175 Gate 5 failed after role arm"}

    gate2 = run173.gate2(answers, prompt_rows)
    gate5 = run173.gate5(answers, prompt_rows)
    original_design = run174.DESIGN
    try:
        run174.DESIGN = {**original_design, "replays": PART174_REPLAYS}
        replay = run174.gate4(answers, reference_answers)
    finally:
        run174.DESIGN = original_design
    return answers, {"gate2": gate2, "gate5": gate5, "gate4": replay}


def gate1(stage_a_rows, answers, d_star, prompt_rows=None):
    prompt_rows = prompts() if prompt_rows is None else list(prompt_rows)
    expected_specs = [{"arm": "none"}, {"arm": "role"}]
    expected_specs.extend({"arm": dose_arm(name, sign), "set": name, "dose": sign}
                          for name in STAGE_A_SET_NAMES
                          for magnitude in STAGE_A_LADDER
                          for sign in (magnitude, -magnitude))
    if d_star is not None:
        expected_specs.extend({"arm": dose_arm(name, sign), "set": name, "dose": sign}
                              for name in F100D_SET_NAMES
                              for magnitude in census_doses(d_star)
                              for sign in (magnitude, -magnitude))
    expected = {(str(spec["arm"]), str(prompt["prompt_id"]))
                for spec in expected_specs for prompt in prompt_rows}
    answer_rows = [row for row in answers
                   if not str(row.get("arm", "")).startswith("replay:")]
    combined = [*stage_a_rows, *answer_rows]
    return run174.gate1(combined, expected=expected)


def measure_census_gate6(model, tok, layers, args, sets, doses):
    """Adapt Part 174's real hook measurement to the selected f100d census grid."""
    previous_names, previous_design = run174.SET_NAMES, run174.DESIGN
    signed = signed_doses(doses)
    try:
        run174.SET_NAMES = F100D_SET_NAMES
        run174.DESIGN = {**previous_design, "signs": tuple(float(value) for value in signed)}
        rows = run174.measure_gate6(model, tok, layers, sets, {}, args.device,
                                    run172.widths_of(layers))
        gate = run174.gate6(rows)
    finally:
        run174.SET_NAMES, run174.DESIGN = previous_names, previous_design
    gate["census_doses"] = [float(value) for value in doses]
    gate["draws"] = len(F100D_SET_NAMES)
    return gate, rows


def _stage_a_workable(stage_a_rows):
    by_key = {(row["set"], float(row["sign"]), _decimal(abs(float(row["dose"])))): int(row["broken"])
              for row in stage_a_rows}
    by_draw = []
    for name in STAGE_A_SET_NAMES:
        dose_steps = {}
        steps = {}
        for sign in (1.0, -1.0):
            broken_by_dose = {_decimal(dose): by_key[(name, sign, _decimal(dose))]
                              for dose in STAGE_A_LADDER}
            steps[str(int(sign))] = workable_step(broken_by_dose)
            dose_steps[str(int(sign))] = {
                "workable_step": steps[str(int(sign))],
                "workable_dose": (float(STAGE_A_LADDER[steps[str(int(sign))] - 1])
                                   if steps[str(int(sign))] else None),
                "below_first_ladder_dose": steps[str(int(sign))] == 0,
            }
        signed_gap = int(steps["1"]) - int(steps["-1"])
        by_draw.append({"set": name, "seed": int(name.rsplit("d", 1)[1]),
                        "positive": dose_steps["1"], "negative": dose_steps["-1"],
                        "signed_step_difference": signed_gap,
                        "sign_gap_steps": abs(signed_gap),
                        "gap_at_least_two": abs(signed_gap) >= 2})
    qualifying = sum(row["gap_at_least_two"] for row in by_draw)
    all_steps = [row[sign]["workable_step"] for row in by_draw for sign in ("positive", "negative")]
    low_step, high_step = min(all_steps), max(all_steps)
    return {"draws": by_draw, "gap_at_least_two_draws": int(qualifying),
            "verdict": "held" if qualifying >= 3 else "failed",
            "d_star_band_if_held": ({"low_step": low_step, "high_step": high_step,
                                     "low_dose": (float(STAGE_A_LADDER[low_step - 1])
                                                  if low_step else None),
                                     "high_dose": (float(STAGE_A_LADDER[high_step - 1])
                                                   if high_step else None)}
                                    if qualifying >= 3 else None)}


def _stage_a_direction_rows(stage_a_answers):
    scored = score173.score_rows(stage_a_answers)
    by_arm = {}
    for row in scored:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    result = []
    for seed_name in STAGE_A_SET_NAMES:
        for magnitude in STAGE_A_LADDER:
            for sign in (1.0, -1.0):
                dose = magnitude if sign > 0 else -magnitude
                arm = dose_arm(seed_name, dose)
                rows = by_arm.get(arm, [])
                if len(rows) != STAGE_A_ROWS_PER_DIRECTION:
                    raise ValueError(f"Stage A {arm} has {len(rows)} rows, expected 44")
                result.append({"set": seed_name, "seed": int(seed_name.rsplit("d", 1)[1]),
                               "sign": float(sign), "dose": float(dose), "arm": arm,
                               "answers": len(rows), "broken": int(sum(row["broken"] for row in rows)),
                               "clean": int(sum(row["clean"] for row in rows))})
    return result


def _gate_file(out, gates):
    gates["passed"] = all(isinstance(value, dict) and bool(value.get("passed"))
                          for key, value in gates.items()
                          if key.startswith("gate") and key[4:].isdigit())
    write_json(Path(out) / "gates.json", gates)
    return gates


def run(args, loader=None, config=None):
    """Run Part 175 from controls through Stage A and, when selected, Stage B."""
    config = config or {}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage0 = config.get("part174_stage0")
    if stage0 is None:
        stage0 = read_json(args.part174_stage0)
    expected_hash = config.get("part174_set_hash")
    if expected_hash is None:
        expected_hash = Path(args.part174_set_hash).read_text(encoding="utf-8")
    reference_answers = config.get("part174_answers")
    if reference_answers is None:
        reference_answers = run174.read_jsonl(args.part174_answers)
    if loader is None:
        import neuron_report as nr
        tok, model = nr.load_model(str(args.model), common86.REVISION, args.device)
    else:
        tok, model = loader(args.device)
    model.requires_grad_(False)
    model.eval()
    layers = config.get("layers")
    if layers is None:
        layers = common86.find_layers(model)
    if bool(config.get("strict_model", loader is None)):
        run171.validate_model(layers)
    widths = run172.widths_of(layers)
    fit_vector, fit_vector_check = _fit_vector(model, tok, layers, args, config)

    reference_artifact, g8_rebuild = verify_part174_rebuild(
        stage0, expected_hash, out, widths=widths, layers=layers,
        fit_vector=fit_vector, tok=tok, model=model,
        projection_vector=config.get("part174_projection_vector"))
    source_norm = _reference_norm(stage0)
    stage_a_sets, stage_a_metrics = build_stage_a_sets(layers, widths, fit_vector, source_norm)
    g8_stage_a = gate8_stage_a(stage_a_metrics, np.asarray(fit_vector).size)
    gate8_result = {"passed": bool(g8_rebuild["passed"] and g8_stage_a["passed"]),
                    "part174_rebuild": g8_rebuild, "stage_a_screens": g8_stage_a,
                    "fit_vector": fit_vector_check}
    if not g8_rebuild["passed"] or not g8_stage_a["passed"]:
        gates = {"gate8": gate8_result}
        _gate_file(out, gates)
        raise SystemExit("Part 175 Gate 8 failed before planned answers")

    sets = {"arms": {**reference_artifact["arms"], **stage_a_sets},
            "metrics": {**reference_artifact["metrics"], **stage_a_metrics}}
    pre_gate3 = run174.gate3(model, tok, layers, args, sets, {}, widths)
    if not pre_gate3["passed"]:
        _gate_file(out, {"gate3": pre_gate3, "gate8": gate8_result})
        raise SystemExit("Part 175 Gate 3 failed before Stage A")

    residual_norms = measure_gate3_residual_norms(model, tok, layers, args.device)
    written_layer_weights = {
        int(layer_no): layers[int(layer_no)].mlp.down_proj.weight.detach().to(
            device="cpu", dtype=__import__("torch").float32).numpy().astype(np.float64)
        for layer_no in run171.LAYERS
    }
    layer_norm_rows = []
    for name in STAGE_A_SET_NAMES:
        for magnitude in STAGE_A_LADDER:
            for dose in (magnitude, -magnitude):
                layer_norm_rows.append(measure_written_layer_norms(
                    sets, name, "stage_a", dose, layers, widths, residual_norms,
                    written_layer_weights))
    layer_norm_rows.append(measure_written_layer_norms(
        sets, "u100d_d174001", "u100d", Decimal("0.5"), layers, widths, residual_norms,
        written_layer_weights))
    write_json(out / "layer_norms.json", layer_norms_document(residual_norms, layer_norm_rows))

    prompt_rows = prompts()
    answers, control_gates = _ensure_controls(model, tok, layers, args, sets, reference_answers)
    if control_gates.get("failed"):
        gates = {"gate2": control_gates.get("gate2"),
                 "gate3": pre_gate3, "gate4": control_gates.get("gate4"),
                 "gate5": control_gates.get("gate5"), "gate8": gate8_result}
        _gate_file(out, gates)
        raise SystemExit(control_gates["failed"])
    gate4_result = control_gates["gate4"]
    gate2_result = control_gates["gate2"]
    gate5_result = control_gates["gate5"]
    gate7_result = run174.gate7()
    if not gate4_result["passed"] or not gate7_result["passed"]:
        gates = {"gate2": gate2_result, "gate3": pre_gate3, "gate4": gate4_result,
                 "gate5": gate5_result, "gate7": gate7_result, "gate8": gate8_result}
        _gate_file(out, gates)
        raise SystemExit("Part 175 Gate 4 or Gate 7 failed before Stage A")

    stage_a_path = out / "stage_a_answers.jsonl"
    run172.repair_tail(stage_a_path)
    stage_a_answers = run174.read_jsonl(stage_a_path) if stage_a_path.is_file() else []
    stage_a_done = {run174.answer_key(row) for row in stage_a_answers}
    if len(stage_a_done) != len(stage_a_answers):
        raise ValueError("stage_a_answers.jsonl contains duplicate Part 175 row keys")
    for name in STAGE_A_SET_NAMES:
        for magnitude in STAGE_A_LADDER:
            for sign in (magnitude, -magnitude):
                arm = dose_arm(name, sign)
                if all((arm, prompt["prompt_id"]) in stage_a_done for prompt in prompt_rows):
                    continue
                records = _batch_for_arm(model, tok, layers, arm, name, sign,
                                         prompt_rows, args, sets)
                _append(stage_a_path, records, stage_a_done)
                stage_a_answers = run174.read_jsonl(stage_a_path)

    stage_a_directions = _stage_a_direction_rows(stage_a_answers)
    selection = select_d_star(stage_a_directions)
    stage_a_weaver = _stage_a_workable(stage_a_directions)
    stage_a = {"part": 175, "status": selection["status"], "d_star": selection["d_star"],
               "ceiling": selection["ceiling"],
               "ladder": [float(value) for value in STAGE_A_LADDER],
               "rule": "largest dose with pooled median broken <= 8/44 over eight signed directions, requiring every lower dose to pass",
               "directions": stage_a_directions, "dose_table": selection["dose_table"],
               "gate8": {"part174_rebuild": g8_rebuild, "stage_a_screens": g8_stage_a,
                         "fit_vector": fit_vector_check,
                         "passed": bool(g8_rebuild["passed"] and g8_stage_a["passed"])},
               "weaver_175.W3": stage_a_weaver,
               "raw_answers_path": str(stage_a_path)}
    write_json(out / "stage_a.json", stage_a)
    if selection["d_star"] is None:
        gates = {"gate1": {"passed": False, "status": "not read", "reason": "Stage A stopped; no Stage B planned"},
                 "gate2": gate2_result, "gate3": pre_gate3, "gate4": gate4_result,
                 "gate5": gate5_result,
                 "gate6": {"passed": False, "status": "not run", "reason": "no D*"},
                 "gate7": gate7_result, "gate8": gate8_result,
                 "stage_a_status": "stopped"}
        _gate_file(out, gates)
        write_json(out / "layer_norms.json",
                   layer_norms_document(residual_norms, layer_norm_rows,
                                        status="complete_stage_a_only"))
        return {"stage": "stage_a_stopped", "d_star": None,
                "stage_a_directions": len(stage_a_directions), "answers": len(answers)}

    doses = census_doses(selection["d_star"])
    signed_census_doses = signed_doses(doses)
    for name in F100D_SET_NAMES:
        for dose in signed_census_doses:
            layer_norm_rows.append(measure_written_layer_norms(
                sets, name, "f100d", dose, layers, widths, residual_norms,
                written_layer_weights))
    write_json(out / "layer_norms.json", layer_norms_document(residual_norms, layer_norm_rows))
    gate6_result, gate6_rows = measure_census_gate6(model, tok, layers, args, sets, doses)
    if not gate6_result["passed"]:
        gates = {"gate2": gate2_result, "gate3": pre_gate3, "gate4": gate4_result,
                 "gate5": gate5_result, "gate6": gate6_result,
                 "gate7": gate7_result, "gate8": gate8_result}
        gates["gate6_rows"] = gate6_rows
        _gate_file(out, gates)
        write_json(out / "layer_norms.json",
                   layer_norms_document(residual_norms, layer_norm_rows,
                                        status="complete_generation_gate_failure"))
        raise SystemExit("Part 175 Gate 6 delivery failed before Stage B")

    answer_path = out / "answers.jsonl"
    run172.repair_tail(answer_path)
    answers = run174.read_jsonl(answer_path) if answer_path.is_file() else []
    done = {run174.answer_key(row) for row in answers}
    if len(done) != len(answers):
        raise ValueError("answers.jsonl contains duplicate Part 175 row keys")
    started = time.perf_counter()
    for name in F100D_SET_NAMES:
        for dose in signed_census_doses:
            arm = dose_arm(name, dose)
            if all((arm, prompt["prompt_id"]) in done for prompt in prompt_rows):
                continue
            records = _batch_for_arm(model, tok, layers, arm, name, dose,
                                     prompt_rows, args, sets)
            _append(answer_path, records, done)
            answers = run174.read_jsonl(answer_path)
    elapsed = time.perf_counter() - started
    stage_a_rows = stage_a_answers
    gate1_result = gate1(stage_a_rows, answers, selection["d_star"], prompt_rows)
    gates = {"gate1": gate1_result, "gate2": gate2_result, "gate3": pre_gate3,
             "gate4": gate4_result, "gate5": gate5_result, "gate6": gate6_result,
             "gate7": gate7_result, "gate8": gate8_result,
             "gate6_rows": gate6_rows,
             "stage_a_status": "selected", "d_star": selection["d_star"],
             "census_doses": [float(value) for value in doses]}
    _gate_file(out, gates)
    write_json(out / "layer_norms.json",
               layer_norms_document(residual_norms, layer_norm_rows, status="complete"))
    if not gates["passed"]:
        raise SystemExit("Part 175 generation gates failed")
    return {"stage": "complete", "d_star": selection["d_star"],
            "census_doses": [float(value) for value in doses],
            "stage_a_directions": len(stage_a_directions),
            "stage_a_answers": len(stage_a_answers), "answers": len(answers),
            "gate6_rows": len(gate6_rows), "stage_b_elapsed_seconds": elapsed,
            "gates": {key: value.get("passed") for key, value in gates.items()
                      if key.startswith("gate") and isinstance(value, dict)}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=Path("results/part175"))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--part171-answers", type=Path, default=PART171_ANSWERS)
    parser.add_argument("--part171-summary", type=Path, default=PART171_SUMMARY)
    parser.add_argument("--part171-sets", type=Path, default=run174.PART171_SETS)
    parser.add_argument("--part174-stage0", type=Path, default=PART174_STAGE0)
    parser.add_argument("--part174-set-hash", type=Path, default=PART174_SET_HASH)
    parser.add_argument("--part174-answers", type=Path, default=PART174_ANSWERS)
    args = parser.parse_args(argv)
    if args.batch_size != BATCH_SIZE:
        parser.error("registered generation batches hold one arm's 44 prompts")
    result = run(args)
    print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0 if result.get("stage") in ("stage_a_stopped", "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
