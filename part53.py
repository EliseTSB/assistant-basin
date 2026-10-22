#!/usr/bin/env python3
"""Part 53: measure the interaction between the wizard and magic units."""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

import fastgen
import part38
import part49
import part51


ROOT = Path(__file__).resolve().parent
WIZARD = {"role": "W", "layer": 19, "unit": 4826}
MAGIC = {"role": "M", "layer": 22, "unit": 1610}
W_KS = (-16, -8, 0, 8, 16)
M_KS = (-32, -16, 0, 16, 32)
EVERYDAY_CORNERS = ((0, 0), (0, 32), (0, -32), (16, 0), (-16, 0))
PART53_BOOTSTRAP_SEED = 5303
PART53_BOOTSTRAP_RESAMPLES = 10_000
BASELINE_ZERO_EPSILON = 1e-12
DEFAULT_OUT = ROOT / "results" / "part53-xpu"
WIZARD_PROMPTS_PATH = ROOT / "data" / "part49b-wizard-prompts.json"
EVERYDAY_PROMPTS_PATH = ROOT / "data" / "part51a-everyday-prompts.json"


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def write_json_atomic(value: Any, path: str | Path) -> None:
    part49.write_json_atomic(value, path)


def _cell_id(prompt_set: str, k_w: int, k_m: int) -> str:
    return f"{prompt_set}:w{k_w:+d}:m{k_m:+d}"


def build_cell_plan() -> list[dict[str, Any]]:
    """Return all 25 wizard cells followed by the five registered everyday corners."""
    plan: list[dict[str, Any]] = []
    for k_w in W_KS:
        for k_m in M_KS:
            plan.append({
                "id": _cell_id("wizard", k_w, k_m), "prompt_set": "wizard",
                "k_W": k_w, "k_M": k_m,
                "clamps": _planned_clamps(k_w, k_m),
            })
    for k_w, k_m in EVERYDAY_CORNERS:
        plan.append({
            "id": _cell_id("everyday", k_w, k_m), "prompt_set": "everyday",
            "k_W": k_w, "k_M": k_m,
            "clamps": _planned_clamps(k_w, k_m),
        })
    if len(plan) != 30 or len({cell["id"] for cell in plan}) != 30:
        raise AssertionError("Part 53 requires 30 uniquely named grid cells")
    return plan


def _planned_clamps(k_w: int, k_m: int) -> list[dict[str, Any]]:
    clamps = []
    if int(k_w) != 0:
        clamps.append({**WIZARD, "k": int(k_w)})
    if int(k_m) != 0:
        clamps.append({**MAGIC, "k": int(k_m)})
    return clamps


def add_per_clamp_natural_maxima(
    cell_plan: list[dict[str, Any]], natural: dict[str, Any]
) -> list[dict[str, Any]]:
    """Attach each unit's own k-scaled natural maximum; zero k has no clamp entry."""
    values = natural.get("values", {})
    materialized = []
    for cell in cell_plan:
        clamps = []
        for clamp in cell["clamps"]:
            key = f"{int(clamp['layer'])}:{int(clamp['unit'])}"
            maximum = values.get(key)
            if maximum is None or not math.isfinite(float(maximum)):
                raise ValueError(f"natural maximum missing or invalid for {key}")
            k = int(clamp["k"])
            clamps.append({
                **clamp, "natural_max": float(maximum),
                "level": float(k * float(maximum)),
            })
        materialized.append({**cell, "clamps": clamps})
    return materialized


def operations_for_cell(cell: dict[str, Any]) -> dict[tuple[int, int], float]:
    return {
        (int(clamp["layer"]), int(clamp["unit"])): float(clamp["level"])
        for clamp in cell["clamps"]
    }


def requests_for_cell(cell: dict[str, Any], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**sample, "cell_id": cell["id"],
         "row_key": f"{cell['id']}:{sample['prompt_id']}:{sample['sample']}"}
        for sample in samples
    ]


def find_natural_max_path(args: argparse.Namespace) -> Path | None:
    """Find an existing compatible Part 49/51 summary containing both target values."""
    if args.natural_max is not None:
        return Path(args.natural_max)
    candidates = [Path(args.out) / "natural-max-summary.json"]
    candidates.extend(sorted((ROOT / "results").glob("part49*-xpu/natural-max-summary.json")))
    candidates.extend(sorted((ROOT / "results").glob("part51*-xpu/natural-max-summary.json")))
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            summary = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        values = summary.get("values", {})
        if (summary.get("model") == part49.MODEL_ID
                and summary.get("revision") == part49.MODEL_REVISION
                and values.get("19:4826") is not None
                and values.get("22:1610") is not None):
            return path
    return None


def load_prompts(path: str | Path, expected_count: int = 16) -> list[str]:
    prompts = read_json(path).get("prompts", [])
    if len(prompts) != expected_count or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError(f"prompt data must contain {expected_count} nonempty prompts: {path}")
    normalized = [" ".join(p.casefold().split()) for p in prompts]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"prompt data contains duplicates: {path}")
    return prompts


class ActivityRecorder:
    """Read unit inputs after intervention pre-hooks during generation decode passes."""

    def __init__(self, model: Any, batch_size: int, coordinates: dict[str, tuple[int, int]]):
        self.model = model
        self.batch_size = int(batch_size)
        self.coordinates = dict(coordinates)
        self.values: dict[str, list[list[float]]] = {
            role: [[] for _ in range(self.batch_size)] for role in self.coordinates
        }
        self._prefill_seen: set[str] = set()
        self._handles = []
        layers = fastgen.find_decoder_layers(model)
        for role, (layer_index, _unit) in self.coordinates.items():
            if not 0 <= int(layer_index) < len(layers):
                raise ValueError(f"activity layer {layer_index} is outside the loaded model")
            module = layers[int(layer_index)].mlp.down_proj

            def observe(_module: Any, inputs: tuple[Any, ...], _output: Any,
                        *, selected_role: str = role) -> None:
                if not inputs or not torch.is_tensor(inputs[0]):
                    raise TypeError(f"activity hook for {selected_role} received no tensor input")
                x = inputs[0].detach()
                if x.ndim == 2:
                    if x.shape[0] % self.batch_size:
                        raise ValueError("flattened activity input does not divide by batch size")
                    x = x.reshape(self.batch_size, x.shape[0] // self.batch_size, x.shape[-1])
                elif x.ndim != 3:
                    raise ValueError(f"activity input must have 2 or 3 dimensions, got {tuple(x.shape)}")
                if x.shape[0] != self.batch_size:
                    raise ValueError("activity input batch size changed during generation")
                if selected_role not in self._prefill_seen:
                    self._prefill_seen.add(selected_role)
                    return  # prompt prefill contains no generated-token positions
                unit = int(self.coordinates[selected_role][1])
                if unit >= x.shape[-1]:
                    raise ValueError(f"activity unit {unit} is outside width {x.shape[-1]}")
                # Cached decoding has sequence length one. Without a cache, the newest
                # position is the generated token just fed back for the next decode.
                last = x[:, -1, unit].float().cpu().tolist()
                for row_index, value in enumerate(last):
                    self.values[selected_role][row_index].append(float(value))

            self._handles.append(module.register_forward_hook(observe))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "ActivityRecorder":
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def summaries(self, outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(outputs) != self.batch_size:
            raise ValueError("activity output count does not match the hooked batch")
        result = []
        for index, output in enumerate(outputs):
            # Every generated token except the terminal token is fed through a later
            # decode forward; only those same-pass activations are observed here.
            observed_count = max(0, len(output.get("token_ids", [])) - 1)
            means: dict[str, float | None] = {}
            counts = []
            for role in self.coordinates:
                values = self.values[role][index][:observed_count]
                if observed_count and len(values) < observed_count:
                    raise RuntimeError(f"activity hook missed a decode token for {role}")
                counts.append(len(values))
                means[role] = float(np.mean(values)) if values else None
            result.append({
                "W": means.get("W"), "M": means.get("M"),
                "token_count": min(counts, default=0),
                "generated_token_count": len(output.get("token_ids", [])),
            })
        return result


def _stem_matches(text: str, stem: str) -> list[re.Match[str]]:
    return list(re.finditer(rf"\b{re.escape(stem)}\w*", str(text),
                            flags=re.IGNORECASE | re.UNICODE))


def wizard_word_measure(answer: str, prompt: str) -> dict[str, Any]:
    """Count Part 38 wizard topic stems, excluding stems already in the prompt."""
    terms = part38.TOPIC_TERMS["wizard"]
    excluded = {term for term in terms if _stem_matches(prompt, term)}
    matches = sum(len(_stem_matches(answer, term)) for term in terms if term not in excluded)
    words = part38.word_count(str(answer))
    return {
        "matches": int(matches), "word_count": int(words),
        "per_1000": float(matches * 1000.0 / words) if words else 0.0,
        "excluded_terms": sorted(excluded),
    }


METRIC_NAMES = ("magic_per_1000", "wizard_per_1000", "activity_W", "activity_M", "broken_share")


def _finite_mean(values: Iterable[Any]) -> float:
    selected = [float(value) for value in values
                if value is not None and math.isfinite(float(value))]
    return float(np.mean(selected)) if selected else float("nan")


def _bootstrap_means(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    sampled = values[indices]
    finite = np.isfinite(sampled)
    counts = finite.sum(axis=1)
    totals = np.where(finite, sampled, 0.0).sum(axis=1)
    result = np.full(sampled.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = totals[valid] / counts[valid]
    return result


def _bootstrap_interval(values: np.ndarray) -> list[float] | None:
    return part49._bootstrap_interval(np.asarray(values, dtype=np.float64))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if (not math.isfinite(float(numerator)) or not math.isfinite(float(denominator))
            or abs(float(denominator)) <= BASELINE_ZERO_EPSILON):
        return None
    return float(numerator / denominator)


def _validate_complete_grid(header: dict[str, Any], results: list[dict[str, Any]],
                            cells: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected_base = {cell["id"]: (cell["prompt_set"], cell["k_W"], cell["k_M"])
                     for cell in build_cell_plan()}
    observed_base = {cell.get("id"): (cell.get("prompt_set"), cell.get("k_W"), cell.get("k_M"))
                     for cell in cells}
    if observed_base != expected_base:
        raise ValueError("Part 53 header does not contain the complete 25 + 5 cell plan")
    prompts_by_set = header.get("prompt_ids", {})
    samples_per_prompt = int(header.get("samples_per_prompt", 0))
    if samples_per_prompt <= 0:
        raise ValueError("Part 53 header has no positive samples_per_prompt")
    by_cell: dict[str, list[dict[str, Any]]] = {cell["id"]: [] for cell in cells}
    seen_keys: set[str] = set()
    for row in results:
        cell_id = str(row.get("cell_id", ""))
        if cell_id not in by_cell:
            raise ValueError(f"Part 53 result has an unknown cell: {cell_id}")
        row_key = str(row.get("row_key", ""))
        if not row_key or row_key in seen_keys:
            raise ValueError(f"Part 53 result has a missing or duplicate row_key: {row_key!r}")
        seen_keys.add(row_key)
        by_cell[cell_id].append(row)
    missing = []
    for cell in cells:
        expected_prompts = prompts_by_set.get(cell["prompt_set"], [])
        expected = {(str(prompt_id), sample)
                    for prompt_id in expected_prompts
                    for sample in range(1, samples_per_prompt + 1)}
        observed = {(str(row.get("prompt_id")), int(row.get("sample", -1)))
                    for row in by_cell[cell["id"]]}
        if observed != expected:
            absent = sorted(expected - observed)
            extra = sorted(observed - expected)
            missing.append({"cell": cell["id"], "missing": absent[:8], "extra": extra[:8]})
    if missing:
        raise ValueError(f"Part 53 is missing cell answers: {missing}")
    return by_cell


def _cell_metrics(cell: dict[str, Any], rows: list[dict[str, Any]], prompt_ids: list[str],
                  indices: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result_by_prompt: dict[str, list[dict[str, Any]]] = {str(prompt_id): [] for prompt_id in prompt_ids}
    for row in rows:
        result_by_prompt[str(row["prompt_id"])].append(row)
    prompt_values: dict[str, np.ndarray] = {}
    per_prompt = []
    for metric in METRIC_NAMES:
        values = []
        for prompt_id in prompt_ids:
            prompt_rows = result_by_prompt[str(prompt_id)]
            if metric in ("magic_per_1000", "wizard_per_1000"):
                measured = []
                for row in prompt_rows:
                    if metric == "magic_per_1000":
                        value = part51.part51a_answer_measure(
                            row.get("answer", ""), row.get("prompt", "")
                        )["per_1000"]
                    else:
                        value = wizard_word_measure(
                            row.get("answer", ""), row.get("prompt", "")
                        )["per_1000"]
                    measured.append(value)
            elif metric == "activity_W":
                measured = [row.get("activity", {}).get("W") for row in prompt_rows]
            elif metric == "activity_M":
                measured = [row.get("activity", {}).get("M") for row in prompt_rows]
            else:
                labels = [row.get("broken") for row in prompt_rows
                          if isinstance(row.get("broken"), bool)]
                measured = [sum(labels) / len(labels)] if labels else []
            values.append(_finite_mean(measured))
        prompt_values[metric] = np.asarray(values, dtype=np.float64)
    for prompt_index, prompt_id in enumerate(prompt_ids):
        per_prompt.append({
            "prompt_id": str(prompt_id),
            **{metric: (float(prompt_values[metric][prompt_index])
                        if math.isfinite(float(prompt_values[metric][prompt_index])) else None)
               for metric in METRIC_NAMES},
        })
    measures = {}
    for metric in METRIC_NAMES:
        values = prompt_values[metric]
        boot = _bootstrap_means(values, indices)
        valid = values[np.isfinite(values)]
        measures[metric] = {
            "mean": float(np.mean(valid)) if valid.size else None,
            "bootstrap95": _bootstrap_interval(boot),
            "prompt_count": int(valid.size),
        }
    broken_rows = [row.get("broken") for row in rows if isinstance(row.get("broken"), bool)]
    broken_hits = int(sum(broken_rows))
    return ({
        "id": cell["id"], "prompt_set": cell["prompt_set"],
        "k_W": cell["k_W"], "k_M": cell["k_M"], "clamps": cell["clamps"],
        "answer_count": len(rows), "broken": {
            "hits": broken_hits, "n": len(broken_rows),
            "rate": (broken_hits / len(broken_rows) if broken_rows else None),
            "pending": not bool(broken_rows),
        },
        "measures": measures, "per_prompt": per_prompt,
    }, prompt_values)


def _interaction_stats(cell: dict[str, Any], cell_values: dict[str, dict[str, np.ndarray]],
                       indices: np.ndarray) -> dict[str, Any]:
    k_w, k_m = int(cell["k_W"]), int(cell["k_M"])
    cell_id = cell["id"]
    baseline_id = _cell_id("wizard", 0, 0)
    w_axis_id = _cell_id("wizard", k_w, 0)
    m_axis_id = _cell_id("wizard", 0, k_m)
    stats: dict[str, Any] = {"cell": cell_id, "k_W": k_w, "k_M": k_m, "measures": {}}
    for metric in ("magic_per_1000", "wizard_per_1000"):
        combined = cell_values[cell_id][metric]
        w_axis = cell_values[w_axis_id][metric]
        m_axis = cell_values[m_axis_id][metric]
        baseline = cell_values[baseline_id][metric]
        means = [float(np.nanmean(values)) for values in (combined, w_axis, m_axis, baseline)]
        additive = means[0] - means[1] - means[2] + means[3]
        boot_combined, boot_w, boot_m, boot_baseline = [
            _bootstrap_means(values, indices) for values in (combined, w_axis, m_axis, baseline)
        ]
        additive_boot = boot_combined - boot_w - boot_m + boot_baseline
        ratio = None
        ratio_boot = np.full(indices.shape[0], np.nan, dtype=np.float64)
        if abs(means[3]) > BASELINE_ZERO_EPSILON:
            denominator = (means[1] * means[2])
            ratio = (float(means[0] * means[3] / denominator)
                     if abs(denominator) > BASELINE_ZERO_EPSILON else None)
            denominator_boot = boot_w * boot_m
            ratio_boot = np.divide(
                boot_combined * boot_baseline, denominator_boot,
                out=np.full(indices.shape[0], np.nan, dtype=np.float64),
                where=np.isfinite(boot_combined) & np.isfinite(boot_w)
                & np.isfinite(boot_m) & np.isfinite(boot_baseline)
                & (np.abs(denominator_boot) > BASELINE_ZERO_EPSILON)
                & (np.abs(boot_baseline) > BASELINE_ZERO_EPSILON),
            )
        stats["measures"][metric] = {
            "additive_contrast": additive,
            "additive_bootstrap95": _bootstrap_interval(additive_boot),
            "multiplicative_ratio": ratio,
            "multiplicative_bootstrap95": _bootstrap_interval(ratio_boot),
            "baseline_mean": means[3],
            "ratio_suppressed_near_zero_baseline": abs(means[3]) <= BASELINE_ZERO_EPSILON,
        }
    return stats


def _everyday_comparisons(cells: list[dict[str, Any]],
                          cell_values: dict[str, dict[str, np.ndarray]],
                          indices: np.ndarray) -> list[dict[str, Any]]:
    baseline_id = _cell_id("everyday", 0, 0)
    rows = []
    for cell in cells:
        if cell["prompt_set"] != "everyday" or cell["id"] == baseline_id:
            continue
        row = {"cell": cell["id"], "k_W": cell["k_W"], "k_M": cell["k_M"], "measures": {}}
        for metric in ("magic_per_1000", "wizard_per_1000"):
            values = cell_values[cell["id"]][metric] - cell_values[baseline_id][metric]
            finite = values[np.isfinite(values)]
            point = float(np.mean(finite)) if finite.size else None
            boot = _bootstrap_means(values, indices)
            baseline = cell_values[baseline_id][metric]
            combined = cell_values[cell["id"]][metric]
            finite_pair = np.isfinite(combined) & np.isfinite(baseline)
            baseline_mean = float(np.mean(baseline[finite_pair])) if finite_pair.any() else float("nan")
            combined_mean = float(np.mean(combined[finite_pair])) if finite_pair.any() else float("nan")
            ratio = _safe_ratio(combined_mean, baseline_mean)
            boot_num = _bootstrap_means(combined, indices)
            boot_den = _bootstrap_means(baseline, indices)
            ratio_boot = np.divide(boot_num, boot_den,
                                   out=np.full(indices.shape[0], np.nan),
                                   where=np.isfinite(boot_num) & np.isfinite(boot_den)
                                   & (np.abs(boot_den) > BASELINE_ZERO_EPSILON))
            row["measures"][metric] = {
                "difference_from_no_clamp": point,
                "difference_bootstrap95": _bootstrap_interval(boot),
                "ratio_to_no_clamp": ratio,
                "ratio_bootstrap95": _bootstrap_interval(ratio_boot),
                "ratio_suppressed_near_zero_baseline": abs(baseline_mean) <= BASELINE_ZERO_EPSILON
                if math.isfinite(baseline_mean) else True,
            }
        rows.append(row)
    return rows


def _prediction_tables(cells: list[dict[str, Any]],
                       cell_scores: dict[str, dict[str, Any]],
                       cell_values: dict[str, dict[str, np.ndarray]],
                       indices: np.ndarray) -> dict[str, Any]:
    def cell_id(prompt_set: str, k_w: int, k_m: int) -> str:
        return _cell_id(prompt_set, k_w, k_m)

    def measure(cell: str, key: str) -> float | None:
        return cell_scores[cell]["measures"][key]["mean"]

    # Paired cell difference and ratio helpers use the same prompt resamples.
    def diff_and_ci(a: str, b: str, metric: str) -> tuple[float | None, list[float] | None]:
        left, right = cell_values[a][metric], cell_values[b][metric]
        diffs = left - right
        finite = diffs[np.isfinite(diffs)]
        point = float(np.mean(finite)) if finite.size else None
        return point, _bootstrap_interval(_bootstrap_means(diffs, indices))

    def ratio_and_ci(a: str, b: str, metric: str) -> tuple[float | None, list[float] | None]:
        left = cell_scores[a]["measures"][metric]["mean"]
        right = cell_scores[b]["measures"][metric]["mean"]
        point = _safe_ratio(left, right) if left is not None and right is not None else None
        boot_left = _bootstrap_means(cell_values[a][metric], indices)
        boot_right = _bootstrap_means(cell_values[b][metric], indices)
        boot = np.divide(boot_left, boot_right, out=np.full(indices.shape[0], np.nan),
                         where=np.isfinite(boot_left) & np.isfinite(boot_right)
                         & (np.abs(boot_right) > BASELINE_ZERO_EPSILON))
        return point, _bootstrap_interval(boot)

    rows53a = []
    baseline = cell_id("wizard", 0, 0)
    for k_w in W_KS:
        if k_w == 0:
            continue
        pushed = cell_id("wizard", k_w, 0)
        diff, diff_ci = diff_and_ci(pushed, baseline, "magic_per_1000")
        ratio, ratio_ci = ratio_and_ci(pushed, baseline, "magic_per_1000")
        rows53a.append({"k_W": k_w, "k_M": 0, "cell": pushed,
                        "magic_per_1000": measure(pushed, "magic_per_1000"),
                        "no_clamp_magic_per_1000": measure(baseline, "magic_per_1000"),
                        "difference": diff, "difference_bootstrap95": diff_ci,
                        "ratio_to_no_clamp": ratio, "ratio_bootstrap95": ratio_ci})

    rows53b = []
    for k_w in W_KS:
        pushed, control = cell_id("wizard", k_w, -16), cell_id("wizard", k_w, 0)
        ratio, interval = ratio_and_ci(pushed, control, "magic_per_1000")
        rows53b.append({"k_W": k_w, "k_M": -16, "cell": pushed,
                        "same_W_k_M_0_cell": control,
                        "magic_per_1000": measure(pushed, "magic_per_1000"),
                        "same_W_k_M_0_magic_per_1000": measure(control, "magic_per_1000"),
                        "ratio": ratio, "ratio_bootstrap95": interval})

    plus32, base00 = cell_id("wizard", 0, 32), cell_id("wizard", 0, 0)
    diff53c, ci53c = diff_and_ci(plus32, base00, "magic_per_1000")
    ratio53c, rci53c = ratio_and_ci(plus32, base00, "magic_per_1000")
    rows53d = []
    for k_w in (-16, 16):
        pushed, control = cell_id("wizard", k_w, 0), base00
        diff, interval = diff_and_ci(pushed, control, "activity_M")
        ratio, ratio_ci = ratio_and_ci(pushed, control, "activity_M")
        rows53d.append({"k_W": k_w, "k_M": 0, "cell": pushed,
                        "mean_M_activity": measure(pushed, "activity_M"),
                        "no_clamp_mean_M_activity": measure(control, "activity_M"),
                        "difference": diff, "difference_bootstrap95": interval,
                        "ratio": ratio, "ratio_bootstrap95": ratio_ci})
    rows53e = []
    for k_m in (-32, 32):
        pushed, control = cell_id("wizard", 0, k_m), base00
        diff, interval = diff_and_ci(pushed, control, "activity_W")
        ratio, ratio_ci = ratio_and_ci(pushed, control, "activity_W")
        rows53e.append({"k_W": 0, "k_M": k_m, "cell": pushed,
                        "mean_W_activity": measure(pushed, "activity_W"),
                        "no_clamp_mean_W_activity": measure(control, "activity_W"),
                        "difference": diff, "difference_bootstrap95": interval,
                        "ratio": ratio, "ratio_bootstrap95": ratio_ci})
    rows53f = []
    for cell in cells:
        if cell["prompt_set"] != "wizard" or not cell["k_W"] or not cell["k_M"]:
            continue
        stats = _interaction_stats(cell, cell_values, indices)
        rows53f.append({"cell": cell["id"], "k_W": cell["k_W"], "k_M": cell["k_M"],
                        "magic_multiplicative_ratio": stats["measures"]["magic_per_1000"]["multiplicative_ratio"],
                        "magic_multiplicative_bootstrap95": stats["measures"]["magic_per_1000"]["multiplicative_bootstrap95"]})
    rows53g = []
    for cell in cells:
        if (cell["prompt_set"] == "wizard" and abs(int(cell["k_W"])) <= 8
                and abs(int(cell["k_M"])) <= 16):
            rows53g.append({
                "cell": cell["id"], "k_W": cell["k_W"], "k_M": cell["k_M"],
                "broken": cell_scores[cell["id"]]["broken"],
                "broken_per_prompt": [
                    {"prompt_id": prompt["prompt_id"], "rate": prompt["broken_share"]}
                    for prompt in cell_scores[cell["id"]]["per_prompt"]
                ],
                "broken_bootstrap95": cell_scores[cell["id"]]["measures"]["broken_share"]["bootstrap95"],
            })
    return {
        "53a": {"metric": "magic_per_1000", "cells": rows53a},
        "53b": {"metric": "magic_per_1000", "cells": rows53b},
        "53c": {"metric": "magic_per_1000", "cell": plus32,
                "no_clamp_cell": base00,
                "magic_per_1000": measure(plus32, "magic_per_1000"),
                "no_clamp_magic_per_1000": measure(base00, "magic_per_1000"),
                "difference": diff53c,
                "difference_bootstrap95": ci53c, "ratio": ratio53c,
                "ratio_bootstrap95": rci53c},
        "53d": {"metric": "activity_M", "cells": rows53d},
        "53e": {"metric": "activity_W", "cells": rows53e},
        "53f": {"metric": "magic_per_1000", "cells": rows53f},
        "53g": {"metric": "broken_share", "cells": rows53g},
    }


def score_push53_rows(rows: Iterable[dict[str, Any]], *,
                      seed: int = PART53_BOOTSTRAP_SEED,
                      resamples: int = PART53_BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    rows = list(rows)
    header = next((row for row in rows if row.get("record_type") == "header"), None)
    if header is None or header.get("command") != "push53":
        raise ValueError("Part 53 scoring requires a push53 header")
    cells = header.get("cells", [])
    results = [row for row in rows if row.get("record_type") == "result"]
    by_cell = _validate_complete_grid(header, results, cells)
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    rng = np.random.default_rng(int(seed))
    prompt_ids_by_set = header.get("prompt_ids", {})
    indices_by_set = {
        prompt_set: rng.integers(0, len(prompt_ids), size=(int(resamples), len(prompt_ids)))
        for prompt_set, prompt_ids in prompt_ids_by_set.items() if prompt_ids
    }
    cell_scores: dict[str, dict[str, Any]] = {}
    cell_values: dict[str, dict[str, np.ndarray]] = {}
    for cell in cells:
        prompt_ids = list(prompt_ids_by_set[cell["prompt_set"]])
        summary, values = _cell_metrics(
            cell, by_cell[cell["id"]], prompt_ids,
            indices_by_set[cell["prompt_set"]],
        )
        cell_scores[cell["id"]] = summary
        cell_values[cell["id"]] = values
    interactions = [
        _interaction_stats(cell, cell_values, indices_by_set["wizard"])
        for cell in cells if cell["prompt_set"] == "wizard" and cell["k_W"] != 0 and cell["k_M"] != 0
    ]
    everyday = _everyday_comparisons(cells, cell_values, indices_by_set["everyday"])
    return {
        "schema_version": 1, "summary_type": "part53_magic_wizard_grid",
        "result_count": len(results), "cell_count": len(cells),
        "measure": {
            "magic_stems": list(part51.PART51A_MAGIC_STEMS),
            "wizard_terms": list(part38.TOPIC_TERMS["wizard"]),
            "prompt_exclusion": "answer matches whose stem occurs in that row's prompt are excluded",
            "density_denominator": "Unicode word count in answer; density per 1,000 words",
            "activity_coordinates": {"W": "19:4826", "M": "22:1610"},
            "activity_source": "read-only down_proj forward hooks on generated-token decode inputs, after clamp pre-hooks",
            "activity_terminal_token": "not measured because it is not forwarded in the same generation pass",
            "activity_clamped_values": "post-clamp values",
            "resampling_unit": "prompt", "bootstrap_resamples": int(resamples),
            "bootstrap_seed": int(seed),
            "baseline_zero_epsilon": BASELINE_ZERO_EPSILON,
        },
        "cells": [cell_scores[cell["id"]] for cell in cells],
        "interactions": interactions,
        "everyday_comparisons": everyday,
        "predictions": _prediction_tables(cells, cell_scores, cell_values,
                                          indices_by_set["wizard"]),
    }


def merge_blind_labels(rows: list[dict[str, Any]], pack_path: str | Path,
                       key_path: str | Path) -> int:
    key = read_json(key_path)
    by_blind = {item["blind_id"]: item.get("row_key") for item in key.get("answers", [])}
    by_row = {row.get("row_key"): row for row in rows if row.get("record_type") == "result"}
    labels = 0
    for answer in part38.read_jsonl(pack_path):
        if answer.get("record_type") != "answer" or not isinstance(answer.get("broken"), bool):
            continue
        row = by_row.get(by_blind.get(answer.get("blind_id")))
        if row is not None:
            row["broken"] = answer["broken"]
            labels += 1
    return labels


def run_blind_pack(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input or (args.out / "push53.jsonl")
    rows = part38.read_jsonl(input_path)
    if not rows or rows[0].get("record_type") != "header" or rows[0].get("command") != "push53":
        raise ValueError(f"Part 53 JSONL has no push53 header: {input_path}")
    answers = [row for row in rows if row.get("record_type") == "result"]
    shuffled = list(answers)
    random.Random(int(args.seed)).shuffle(shuffled)
    if args.limit is not None:
        shuffled = shuffled[:int(args.limit)]
    pack_rows = [{"record_type": "header", "schema_version": 1,
                  "blind_pack": True, "seed": int(args.seed),
                  "answer_count": len(shuffled)}]
    key_answers = []
    for index, row in enumerate(shuffled, 1):
        blind_id = f"blind-{index:05d}"
        pack_rows.append({"record_type": "answer", "blind_id": blind_id,
                          "text": str(row.get("answer", "")), "broken": None})
        key_answers.append({"blind_id": blind_id, "row_key": row.get("row_key")})
    part38.write_jsonl_atomic(pack_rows, args.out / "blind-pack.jsonl")
    write_json_atomic({"schema_version": 1, "seed": int(args.seed),
                       "source": str(input_path), "answers": key_answers},
                      args.out / "blind-pack-key.json")
    return {"pack": str(args.out / "blind-pack.jsonl"),
            "key": str(args.out / "blind-pack-key.json"), "answer_count": len(shuffled)}


def run_push(args: argparse.Namespace) -> dict[str, Any]:
    wizard_prompts = load_prompts(WIZARD_PROMPTS_PATH)
    everyday_prompts = load_prompts(EVERYDAY_PROMPTS_PATH)
    if args.limit is not None and not 1 <= int(args.limit) <= 16:
        raise ValueError("--limit must be between 1 and 16")
    if args.samples <= 0 or args.batch <= 0:
        raise ValueError("--samples and --batch must be positive")
    if args.revision != part49.MODEL_REVISION:
        raise ValueError(f"Part 53 requires pinned revision {part49.MODEL_REVISION}")
    prompt_sets = {
        "wizard": wizard_prompts[:args.limit] if args.limit is not None else wizard_prompts,
        "everyday": everyday_prompts[:args.limit] if args.limit is not None else everyday_prompts,
    }
    prompt_ids = {kind: [f"p{index:02d}" for index in range(1, len(prompts) + 1)]
                  for kind, prompts in prompt_sets.items()}
    cells = build_cell_plan()
    for cell in cells:
        cell["prompt_ids"] = prompt_ids[cell["prompt_set"]]
        cell["samples_per_prompt"] = int(args.samples)
    coordinates = [{"layer": WIZARD["layer"], "unit": WIZARD["unit"]},
                   {"layer": MAGIC["layer"], "unit": MAGIC["unit"]}]
    args.out.mkdir(parents=True, exist_ok=True)
    natural_path = find_natural_max_path(args)
    if natural_path is None:
        natural_path = args.out / "natural-max-summary.json"
    torch_runtime, tokenizer, model, layers, snapshot = part38.load_model(args)
    natural, natural_path = part49._load_or_capture_natural_max(
        argparse.Namespace(**{**vars(args), "natural_max": natural_path}),
        torch_runtime, tokenizer, model, layers, coordinates, args.out,
    )
    cells = add_per_clamp_natural_maxima(cells, natural)
    samples = {
        kind: part49.build_sample_rows(
            prompts, int(args.samples), int(args.seed), f"part53-{kind}-sample"
        )
        for kind, prompts in prompt_sets.items()
    }
    header = {
        "schema_version": 1, "record_type": "header", "command": "push53",
        "model": args.model, "revision": args.revision, "device": args.device,
        "dtype": args.dtype, "seed": int(args.seed), "snapshot": str(snapshot),
        "data_sha256": {
            "wizard": part49.sha256_file(WIZARD_PROMPTS_PATH),
            "everyday": part49.sha256_file(EVERYDAY_PROMPTS_PATH),
        },
        "prompt_ids": prompt_ids, "prompts": prompt_sets,
        "prompt_count_by_set": {kind: len(prompts) for kind, prompts in prompt_sets.items()},
        "samples_per_prompt": int(args.samples),
        "same_row_seeds_across_cells": True,
        "arms": cells, "cells": cells,
        "natural_max_path": str(natural_path),
        "natural_max_sha256": part49.sha256_file(natural_path) if natural_path.is_file() else None,
        "natural_max_values": {"19:4826": natural["values"]["19:4826"],
                               "22:1610": natural["values"]["22:1610"]},
        "activity_recording": {
            "coordinates": {"W": "19:4826", "M": "22:1610"},
            "source": "read-only down_proj forward hooks after clamp pre-hooks",
            "scope": "generated tokens fed into decode forwards; terminal emitted token has no same-pass activation",
            "clamped_value": "post-clamp",
        },
        "generation": {"temperature": part49.TEMPERATURE, "top_p": 1.0,
                       "max_new_tokens": part49.MAX_NEW_TOKENS, "batch_size": int(args.batch)},
    }
    output_path = args.out / "push53.jsonl"
    existing = part38.read_jsonl(output_path) if output_path.is_file() else []
    if existing:
        old_header = existing[0]
        for key in ("command", "seed", "prompt_ids", "samples_per_prompt", "cells", "natural_max_values"):
            if old_header.get(key) != header.get(key):
                raise ValueError(f"existing Part 53 output header differs at {key}; choose a new --out")
    rows = part38.prepare_jsonl(output_path, header)
    done = part38.completed_keys(rows)
    timings: dict[str, Any] = {}
    for cell in cells:
        requests = requests_for_cell(cell, samples[cell["prompt_set"]])
        pending = [sample for sample in requests if sample["row_key"] not in done]
        operations = operations_for_cell(cell)
        started = time.perf_counter()
        generated_count = 0
        for batch_start in range(0, len(pending), int(args.batch)):
            batch = pending[batch_start:batch_start + int(args.batch)]
            rendered = [part38.chat_text(tokenizer, sample["prompt"]) for sample in batch]
            with ActivityRecorder(model, len(batch), {"W": (19, 4826), "M": (22, 1610)}) as recorder:
                outputs = fastgen.batched_generate(
                    model, tokenizer, rendered,
                    max_new_tokens=part49.MAX_NEW_TOKENS, operations=operations,
                    do_sample=True, row_seeds=[int(sample["seed"]) for sample in batch],
                    temperature=part49.TEMPERATURE, device=args.device, generated_only=True,
                    padding_side=args.padding_side,
                )
                activity = recorder.summaries(outputs)
            for sample, output, measured in zip(batch, outputs, activity):
                result = {
                    "record_type": "result", "row_key": sample["row_key"],
                    "cell_id": cell["id"], "prompt_set": cell["prompt_set"],
                    "k_W": cell["k_W"], "k_M": cell["k_M"],
                    "prompt_id": sample["prompt_id"], "sample": sample["sample"],
                    "seed": sample["seed"], "prompt": sample["prompt"],
                    "answer": output["text"], "broken": None,
                    "clamps": cell["clamps"],
                    "activity": measured,
                }
                part38.append_jsonl(output_path, result)
                done.add(result["row_key"])
                generated_count += 1
        elapsed = time.perf_counter() - started
        timings[cell["id"]] = {"seconds": elapsed, "generated": generated_count,
                               "batch_size": int(args.batch)}
        print(f"push53 cell {cell['id']}: generated {generated_count}/{len(requests)} "
              f"answers in {elapsed:.2f}s", flush=True)
    final_rows = part38.read_jsonl(output_path)
    scores = score_push53_rows(final_rows)
    scores.update({"model": args.model, "revision": args.revision,
                   "device": args.device, "dtype": args.dtype, "seed": int(args.seed),
                   "natural_max_path": str(natural_path), "jsonl": str(output_path),
                   "timings": timings})
    score_path = args.out / "push53-scores.json"
    write_json_atomic(scores, score_path)
    print(json.dumps({"jsonl": str(output_path), "scores": str(score_path),
                      "answer_count": scores["result_count"],
                      "cell_count": scores["cell_count"], "timings": timings},
                     ensure_ascii=False, indent=2))
    return scores


def run_score(args: argparse.Namespace) -> dict[str, Any]:
    rows = part38.read_jsonl(args.input)
    if args.blind_pack:
        if not args.blind_key:
            raise ValueError("--blind-key is required with --blind-pack")
        count = merge_blind_labels(rows, args.blind_pack, args.blind_key)
        print(f"merged {count} blind broken-answer labels", flush=True)
    summary = score_push53_rows(rows)
    summary.update({"model": rows[0].get("model"), "revision": rows[0].get("revision"),
                    "seed": rows[0].get("seed"), "jsonl": str(args.input)})
    write_json_atomic(summary, args.out)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    push = sub.add_parser("push53", help="run the Part 53 wizard x magic grid")
    part49.add_model_arguments(push)
    push.add_argument("--part35-data", type=Path, default=part49.PART35_DATA_PATH)
    push.add_argument("--natural-max", type=Path, default=None)
    push.add_argument("--out", type=Path, default=DEFAULT_OUT)
    push.add_argument("--limit", type=int, default=None, help="prompt limit for each prompt set")
    push.add_argument("--samples", type=int, default=2, help="samples per prompt")
    push.add_argument("--batch", type=int, default=16)
    push.add_argument("--padding-side", choices=("left", "right"), default="left")

    blind = sub.add_parser("blind-pack", help="write a blind-reader pack from push53 JSONL")
    blind.add_argument("--input", type=Path, default=None)
    blind.add_argument("--out", type=Path, default=DEFAULT_OUT)
    blind.add_argument("--seed", type=int, default=3801)
    blind.add_argument("--limit", type=int, default=None)

    score = sub.add_parser("score", help="score push53 rows, optionally merging blind labels")
    score.add_argument("--input", type=Path, default=DEFAULT_OUT / "push53.jsonl")
    score.add_argument("--blind-pack", type=Path, default=None)
    score.add_argument("--blind-key", type=Path, default=None)
    score.add_argument("--out", type=Path, default=DEFAULT_OUT / "push53-scores.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "push53":
            run_push(args)
        elif args.command == "blind-pack":
            if args.limit is not None and args.limit <= 0:
                parser.error("--limit must be positive")
            args.out.mkdir(parents=True, exist_ok=True)
            run_blind_pack(args)
        elif args.command == "score":
            run_score(args)
        else:
            parser.error(f"unknown command {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"part53: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
