#!/usr/bin/env python3
"""Compare Part 181 run states, gates, and registered predictions."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PREFIX = "127bfedc"
REGISTERED_TOKEN_SHA256 = "5238544b790184de8f34a639b32e0c4f362dfb860e345df918671cdb8609d866"
REGISTERED_DENSE_CONTROL_SHA256 = "54515ce29b1c36af123d8337932e510744b3ab77951e3b520f7ada43a10ff20a"
GATE_SEED = 181
CHANCE_PERMUTATIONS = 100
TOP_K_VALUES = (10, 20)
EXPECTED_RUN_KEYS = {"pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T",
                     "pretrain_4T", "midtrain_13B", "midtrain_26B", "midtrain_51B", "sft", "dpo",
                     "instruct", "instruct_b", "pretrain_1T_b", "pretrain_4T_bf16r"}


@dataclass
class Run:
    key: str
    path: Path
    identity: dict
    config: dict
    state: dict[str, np.ndarray]


def _canonical_key(key: str) -> str:
    return "instruct" if key == "instruct_b" else key


def _gate1_passed(value: Any) -> bool:
    if isinstance(value, dict):
        verdict = value.get("verdict")
        checks = value.get("checks")
        checks_pass = isinstance(checks, dict) and bool(checks) and all(
            isinstance(check, dict) and check.get("verdict") in {"passed", "pass", "green"}
            for check in checks.values())
        return (verdict in {"passed", "pass", "green"} or value.get("passed") is True) and \
            (checks is None or checks_pass)
    return value in {True, "passed", "pass", "green"}


def _load_run(path: Path) -> Run:
    identity_path, progress_path, state_path = path / "identity.json", path / "progress.json", path / "state.npz"
    if not identity_path.is_file():
        raise ValueError("missing identity.json")
    if not progress_path.is_file():
        raise ValueError("missing progress.json")
    if not state_path.is_file():
        raise ValueError("missing state.npz")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict):
        raise ValueError("identity.json must contain an object")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    config = progress.get("config", {})
    expected_config_sha = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    if progress.get("config_sha") != expected_config_sha:
        raise ValueError("progress.json config_sha does not match its collector config")
    with np.load(state_path, allow_pickle=False) as saved:
        state = {name: saved[name] for name in saved.files}
    missing_arrays = {"max", "top_act", "top_doc", "top_pos"} - set(state)
    if missing_arrays:
        raise ValueError("state.npz is missing arrays: " + ", ".join(sorted(missing_arrays)))
    if state["max"].ndim != 2 or state["top_doc"].ndim != 3 or state["top_pos"].shape != state["top_doc"].shape:
        raise ValueError("state.npz has incompatible max/top array shapes")
    if state["top_doc"].shape[:2] != state["max"].shape or state["top_act"].shape != state["top_doc"].shape:
        raise ValueError("state.npz max and top arrays disagree on layer/unit dimensions")
    if "layers" in config and len(config["layers"]) != state["max"].shape[0]:
        raise ValueError("progress.json layer layout does not match state.npz")
    if "unit_kind" in state:
        if state["unit_kind"].shape != (state["max"].shape[1],) or \
                state.get("unit_k", np.empty(0)).shape != state["unit_kind"].shape:
            raise ValueError("state.npz control kind/k rows do not match its unit dimension")
    return Run(str(identity.get("key") or path.name), path, identity, config, state)


def _registered_prefixes() -> dict[str, str]:
    from part85 import checkpoints
    base = {cp.key: cp.commit_prefix for cp in checkpoints.CHECKPOINTS}
    return base | {"instruct_b": base["instruct"], "pretrain_1T_b": base["pretrain_1T"],
                   "pretrain_4T_bf16r": base["pretrain_4T"]}


def _resolve_runs(runs_dir: Path) -> tuple[list[Run], list[dict], dict]:
    prefixes = _registered_prefixes()
    exclusions: list[dict] = []
    loaded: list[Run] = []
    identity_records: list[tuple[str, dict, bool]] = []
    missing_identity_runs: list[str] = []
    invalid_identity_runs: list[str] = []
    prefix_failures: list[str] = []
    for path in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        identity_path = path / "identity.json"
        try:
            if not identity_path.is_file():
                raise ValueError("missing identity.json")
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if not isinstance(identity, dict):
                raise ValueError("identity.json must contain an object")
        except Exception as exc:
            exclusions.append({"run": path.name, "reasons": [str(exc)]})
            missing_identity_runs.append(path.name)
            continue
        key = str(identity.get("key") or path.name)
        identity_errors: list[str] = []
        prefix = prefixes.get(key)
        model_commit = identity.get("model_commit")
        dataset_commit = identity.get("dataset_commit")
        if prefix is None:
            identity_errors.append(f"unregistered run key {key!r}")
            prefix_failures.append(key)
        elif not (isinstance(model_commit, str) and len(model_commit) == 40 and model_commit.startswith(prefix)):
            identity_errors.append(f"model commit does not match registered prefix {prefix}")
            prefix_failures.append(key)
        if not (isinstance(dataset_commit, str) and len(dataset_commit) == 40
                and dataset_commit.startswith(DATASET_PREFIX)):
            identity_errors.append(f"dataset commit does not match prefix {DATASET_PREFIX}")
            prefix_failures.append(key)
        token_sha = identity.get("token_ids_sha256")
        if token_sha != REGISTERED_TOKEN_SHA256:
            identity_errors.append("token_ids_sha256 does not match the registered corpus hash")
        if identity.get("documents") != 2500 or identity.get("token_count") != 885679 or \
                identity.get("documents_at_token_cap") != 992:
            identity_errors.append("token corpus counts do not match the registered counts")
        corpus_sha = identity.get("corpus_file_sha256")
        if not isinstance(corpus_sha, str) or len(corpus_sha) != 64:
            identity_errors.append("corpus_file_sha256 is missing or malformed")
        controls = identity.get("controls_sha256_by_layer")
        if (not isinstance(controls, dict) or set(controls) != {str(i) for i in range(16)}
                or any(not isinstance(value, str) or len(value) != 64 for value in controls.values())):
            identity_errors.append("controls_sha256_by_layer must contain a sha256 for all 16 layers")
        control_info = identity.get("controls", {})
        dense_sha = control_info.get("dense_sha256") if isinstance(control_info, dict) else None
        if dense_sha != REGISTERED_DENSE_CONTROL_SHA256:
            identity_errors.append("stacked dense control hash does not match the registered hash")
        sparse_by_layer = control_info.get("sparse_by_layer_k_sha256", {}) if isinstance(control_info, dict) else {}
        sparse_ok = isinstance(sparse_by_layer, dict) and set(sparse_by_layer) == {str(i) for i in range(16)}
        if sparse_ok:
            sparse_ok = all(isinstance(row, dict) and set(row) == {"2", "4", "16", "64"} and
                            all(isinstance(value, str) and len(value) == 64 for value in row.values())
                            for row in sparse_by_layer.values())
        if not sparse_ok:
            identity_errors.append("sparse control hashes are missing or malformed")
        control_rows = identity.get("control_rows")
        if not isinstance(control_rows, list) or len(control_rows) != 16 * 512:
            identity_errors.append("control_rows must describe all 512 controls in all 16 layers")
        else:
            layout_ok = True
            for row_index, row in enumerate(control_rows):
                layer, control_id = divmod(row_index, 512)
                local_id = control_id
                expected = (("dense", 0) if local_id < 256 else
                            ("sparse", (2, 4, 16, 64)[(local_id - 256) // 64]))
                if row != {"layer": layer, "control_id": control_id, "kind": expected[0], "k": expected[1]}:
                    layout_ok = False
                    break
            if not layout_ok:
                identity_errors.append("control_rows do not match the registered dense/sparse row layout")
        settings = identity.get("settings", {})
        expected_settings = {"batch_size": 4, "checkpoint_every": 100, "docs": 2500,
                             "pass2_docs": 500, "passes": 2, "threads": 4,
                             "n_dense_controls": 256, "n_sparse_per_k": 64}
        if settings != expected_settings or identity.get("dtype") != "fp32":
            identity_errors.append("session settings or dtype differ from the v4 registration")
        if not isinstance(identity.get("weight_fingerprint"), str) or len(identity["weight_fingerprint"]) != 64:
            identity_errors.append("weight_fingerprint is missing or malformed")
        weight_files = identity.get("weight_file_sha256")
        if not isinstance(weight_files, dict) or not weight_files or any(
            not isinstance(value, str) or len(value) != 64 for value in weight_files.values()
        ):
            identity_errors.append("weight_file_sha256 must hash every loaded weight file")
        if not isinstance(identity.get("manifest_sha256"), str) or len(identity["manifest_sha256"]) != 64:
            identity_errors.append("manifest_sha256 is missing or malformed")
        else:
            manifest_body = {name: value for name, value in identity.items() if name != "manifest_sha256"}
            expected_manifest = hashlib.sha256(json.dumps(manifest_body, sort_keys=True,
                                                          separators=(",", ":")).encode("utf-8")).hexdigest()
            if identity["manifest_sha256"] != expected_manifest:
                identity_errors.append("manifest_sha256 does not match the identity contents")
        versions = identity.get("library_versions", {})
        expected_versions = {"numpy": "2.4.6", "torch": "2.15.0", "transformers": "5.22.1",
                             "tokenizers": "0.23.3"}
        if not isinstance(versions, dict) or any(
            (versions.get(name) or "").split("+", 1)[0] != expected
            for name, expected in expected_versions.items()
        ):
            identity_errors.append("library versions do not match the pinned session versions")
        identity_ok = not identity_errors
        identity_records.append((key, identity, identity_ok))
        if identity_errors:
            invalid_identity_runs.append(key)
        try:
            run = _load_run(path)
        except Exception as exc:
            exclusions.append({"run": key, "path": str(path), "reasons": [str(exc)]})
            continue
        if run.config.get("manifest_sha256") != identity.get("manifest_sha256"):
            identity_errors.append("collector config manifest_sha256 does not match identity.json")
            invalid_identity_runs.append(key)
            identity_records[-1] = (key, identity, False)
        expected_collector = {"docs": 2500, "pass2_docs": 500, "dtype": "fp32", "batch_size": 4,
                              "checkpoint_every": 100, "n_dense_controls": 256,
                              "n_sparse_per_k": 64, "n_controls": 512}
        if any(run.config.get(name) != value for name, value in expected_collector.items()) or \
                not json.loads((run.path / "progress.json").read_text(encoding="utf-8")).get("complete"):
            identity_errors.append("collector config or completion status differs from v4")
            invalid_identity_runs.append(key)
            identity_records[-1] = (key, identity, False)
        gate_path = run.path / "gate1.json"
        try:
            gate_value = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else None
        except Exception as exc:
            gate_value = None
            gate_error = f"gate1.json is malformed: {exc}"
        else:
            gate_error = None
        reasons = list(identity_errors)
        if gate_error:
            reasons.append(gate_error)
        if not _gate1_passed(gate_value):
            reasons.append("Gate 1 did not pass or gate1.json is missing")
        if reasons:
            exclusions.append({"run": key, "path": str(run.path), "reasons": reasons})
        else:
            loaded.append(run)

    # Gate 2 uses the registered token/control hashes and checkpoint-specific weight identities.
    observed_keys = {key for key, _identity, _ok in identity_records}
    missing_expected = sorted(EXPECTED_RUN_KEYS - observed_keys)
    weight_by_key: dict[str, str] = {}
    for key, ident, _ok in identity_records:
        weight_by_key.setdefault(key, str(ident.get("weight_fingerprint", "")))
    weight_failures = []
    weight_bad_keys: set[str] = set()
    for first, second in (("instruct", "instruct_b"), ("pretrain_1T", "pretrain_1T_b")):
        if first in weight_by_key and second in weight_by_key and weight_by_key[first] != weight_by_key[second]:
            weight_failures.append(f"{first}/{second} repeat weights differ")
            weight_bad_keys.update((first, second))
    distinct_keys = sorted(set(weight_by_key))
    for i, first in enumerate(distinct_keys):
        for second in distinct_keys[i + 1:]:
            if {first, second} in ({"instruct", "instruct_b"}, {"pretrain_1T", "pretrain_1T_b"}):
                continue
            if weight_by_key[first] and weight_by_key[first] == weight_by_key[second]:
                weight_failures.append(f"unexpected identical weight fingerprints: {first}/{second}")
                weight_bad_keys.update((first, second))
    if "pretrain_4T" in weight_by_key and "pretrain_4T_bf16r" in weight_by_key and \
            weight_by_key["pretrain_4T"] == weight_by_key["pretrain_4T_bf16r"]:
        weight_failures.append("pretrain_4T and pretrain_4T_bf16r fingerprints are equal")
        weight_bad_keys.update(("pretrain_4T", "pretrain_4T_bf16r"))
    identity_failures = sorted(set(prefix_failures + invalid_identity_runs))
    gate2_failed = bool(identity_failures or weight_failures)
    gate2 = {"verdict": "failed" if gate2_failed else "passed" if not missing_expected else "not assessable",
             "identity_runs": len(identity_records), "expected_runs": len(EXPECTED_RUN_KEYS),
             "missing_expected_runs": missing_expected, "commit_prefix_failure_runs": prefix_failures,
             "invalid_identity_runs": invalid_identity_runs, "weight_failures": weight_failures,
             "token_ids_sha256": REGISTERED_TOKEN_SHA256,
             "dense_control_sha256": REGISTERED_DENSE_CONTROL_SHA256,
             "note": "identity is checked against registered corpus/control hashes and repeat-weight rules"}
    for run in loaded[:]:
        if run.key in weight_bad_keys:
            exclusions.append({"run": run.key, "path": str(run.path),
                               "reasons": ["Gate 2 weight fingerprint rule failed"]})
            loaded.remove(run)

    # Keep one valid result per checkpoint key. If a failed original was rerun, the passing
    # rerun is selected; otherwise the original is preferred and both remain on disk.
    by_key: dict[str, list[Run]] = {}
    for run in loaded:
        by_key.setdefault(run.key, []).append(run)
    selected: list[Run] = []
    for key, copies in by_key.items():
        copies.sort(key=lambda run: (run.identity.get("rerun_of") is not None, str(run.path)))
        selected.append(copies[0])
        for extra in copies[1:]:
            exclusions.append({"run": key, "path": str(extra.path),
                               "reasons": ["additional passing rerun retained on disk; original selected"]})
    loaded = selected
    return loaded, exclusions, gate2


def _active(maxima: np.ndarray) -> np.ndarray:
    finite = np.isfinite(maxima)
    if not finite.any():
        return np.zeros(maxima.shape, dtype=bool)
    median = float(np.median(maxima[finite]))
    if median <= 0:
        return np.zeros(maxima.shape, dtype=bool)
    return finite & (maxima > 0) & (maxima >= 0.1 * median)


def _intersection_over_k(a_doc: np.ndarray, a_pos: np.ndarray,
                         b_doc: np.ndarray, b_pos: np.ndarray, k: int,
                         positions: bool, chunk_size: int = 2048) -> np.ndarray:
    """Vectorized per-row top-k intersection; missing entries still use denominator k."""
    if len(a_doc) == 0:
        return np.empty(0, dtype=np.float32)
    out = np.empty(len(a_doc), dtype=np.float32)
    for start in range(0, len(a_doc), chunk_size):
        stop = min(start + chunk_size, len(a_doc))
        ad = a_doc[start:stop, :k]
        bd = b_doc[start:stop, :k]
        matches = (ad[:, :, None] == bd[:, None, :]) & (ad[:, :, None] >= 0) & (bd[:, None, :] >= 0)
        if positions:
            ap = a_pos[start:stop, :k]
            bp = b_pos[start:stop, :k]
            matches &= (ap[:, :, None] == bp[:, None, :]) & (ap[:, :, None] >= 0) & (bp[:, None, :] >= 0)
        out[start:stop] = matches.any(axis=2).sum(axis=1) / float(k)
    return out


def _metric_block(a_doc: np.ndarray, a_pos: np.ndarray, b_doc: np.ndarray, b_pos: np.ndarray) -> dict:
    out = {}
    for k in TOP_K_VALUES:
        out[f"top{k}_doc"] = _intersection_over_k(a_doc, a_pos, b_doc, b_pos, k, positions=False)
        out[f"top{k}_document_position"] = _intersection_over_k(a_doc, a_pos, b_doc, b_pos, k,
                                                                positions=True)
    return out


def _stat(values: np.ndarray, min_units: int = 50) -> dict:
    if not len(values):
        return {"verdict": "not assessable", "mean": None, "median": None,
                "units": 0, "assessable": False}
    assessable = len(values) >= min_units
    return {"verdict": "reported" if assessable else "not assessable",
            "mean": float(np.mean(values)) if assessable else None,
            "median": float(np.median(values)) if assessable else None,
            "units": int(len(values)), "assessable": assessable}


def _layout(run: Run) -> tuple[list[int], int, int]:
    layers = [int(x) for x in run.config.get("layers", range(run.state["max"].shape[0]))]
    width = int(run.config.get("width", 8192))
    n_controls = int(run.config.get("n_controls", 256))
    return layers, width, n_controls


def _populations(run: Run) -> dict[str, np.ndarray]:
    _layers, width, n_controls = _layout(run)
    unit_count = run.state["max"].shape[1]
    groups: dict[str, np.ndarray] = {"neurons": np.arange(min(width, unit_count), dtype=int)}
    if "unit_kind" in run.state and "unit_k" in run.state:
        kinds = run.state["unit_kind"].astype(str)
        ks = run.state["unit_k"].astype(int)
        groups["dense"] = np.flatnonzero(kinds == "dense")
        for k in (2, 4, 16, 64):
            groups[f"sparse_k{k}"] = np.flatnonzero((kinds == "sparse") & (ks == k))
    else:
        groups["dense"] = np.arange(width, min(width + n_controls, unit_count), dtype=int)
    return groups


def _cohort_ids(group: str, population: np.ndarray, active_a: np.ndarray, active_b: np.ndarray,
                primary_ids: np.ndarray | None) -> dict[str, np.ndarray]:
    global_ids = population
    if group == "neurons":
        ids = np.asarray(primary_ids if primary_ids is not None else
                         global_ids[active_a], dtype=int)
        return {"primary": ids, "all": global_ids,
                "conditional_active_both": global_ids[active_a & active_b]}
    return {"all": global_ids, "conditional_active_both": global_ids[active_a & active_b]}


def _derangement_indices(size: int, rng: np.random.Generator) -> np.ndarray:
    """Sattolo shuffle: one cycle, so every index is paired to a different index."""
    order = np.arange(size)
    for i in range(size - 1, 0, -1):
        j = int(rng.integers(0, i))
        order[i], order[j] = order[j], order[i]
    return order


def _null_for_ids(a: Run, b: Run, slot: int, ids: np.ndarray, permutations: int = CHANCE_PERMUTATIONS,
                  seed: int = GATE_SEED) -> dict:
    if len(ids) < 50:
        return {"verdict": "not assessable", "units": int(len(ids)), "required_units": 50,
                "median_distribution": [], "mean_distribution": []}
    rng = np.random.default_rng(seed + slot * 7919 + len(ids))
    medians, means = [], []
    for _ in range(permutations):
        order = _derangement_indices(len(ids), rng)
        vals = _intersection_over_k(a.state["top_doc"][slot, ids], a.state["top_pos"][slot, ids],
                                   b.state["top_doc"][slot, ids[order]],
                                   b.state["top_pos"][slot, ids[order]], 10, positions=False)
        medians.append(float(np.median(vals)))
        means.append(float(np.mean(vals)))
    return {"verdict": "reported", "units": int(len(ids)), "permutations": permutations,
            "median_distribution": medians, "mean_distribution": means,
            "median_summary": {"mean": float(np.mean(medians)), "median": float(np.median(medians)),
                               "p10": float(np.percentile(medians, 10)), "p90": float(np.percentile(medians, 90))},
            "mean_summary": {"mean": float(np.mean(means)), "median": float(np.median(means)),
                             "p10": float(np.percentile(means, 10)), "p90": float(np.percentile(means, 90))}}


def summarize_pair(a: Run, b: Run, primary_ids_by_layer: dict[int, np.ndarray] | None = None) -> dict:
    """Summarize fixed cohorts, control populations, and same-layer derangement nulls."""
    layers_a, width_a, controls_a = _layout(a)
    layers_b, width_b, controls_b = _layout(b)
    if layers_a != layers_b or width_a != width_b or controls_a != controls_b:
        raise ValueError(f"incompatible state layouts for {a.key} and {b.key}")
    sa, sb = a.state, b.state
    if sa["max"].shape != sb["max"].shape or sa["top_doc"].shape != sb["top_doc"].shape:
        raise ValueError(f"state shapes differ for {a.key} and {b.key}")
    L, U = sa["max"].shape
    if L != len(layers_a) or U < width_a + controls_a:
        raise ValueError(f"state shape {sa['max'].shape} disagrees with config for {a.key}")

    populations_a, populations_b = _populations(a), _populations(b)
    if populations_a.keys() != populations_b.keys() or any(
        not np.array_equal(populations_a[name], populations_b[name]) for name in populations_a
    ):
        raise ValueError(f"control population layouts differ for {a.key} and {b.key}")
    result: dict[str, Any] = {"run_a": a.key, "run_b": b.key, "groups": {}, "null": {},
                              "null_reference": {"independent_uniform_top10_overlap": 0.004,
                                                 "note": "reference only; inference uses the same-layer derangement null"}}
    for group, ids in populations_a.items():
        cohort_layer_rows: dict[str, dict[str, Any]] = {}
        cohort_all_values: dict[str, dict[str, list[np.ndarray]]] = {}
        null_by_cohort: dict[str, dict[str, Any]] = {}
        for slot, layer in enumerate(layers_a):
            active_a = _active(sa["max"][slot, ids])
            active_b = _active(sb["max"][slot, ids])
            primary = (None if primary_ids_by_layer is None else
                       primary_ids_by_layer.get(layer, np.empty(0, dtype=int)))
            cohorts = _cohort_ids(group, ids, active_a, active_b, primary)
            cohort_layer_rows[str(layer)] = {}
            for cohort_name, cohort_ids in cohorts.items():
                if not len(cohort_ids):
                    chosen = np.empty(0, dtype=int)
                else:
                    chosen = cohort_ids
                metric_arrays = _metric_block(sa["top_doc"][slot, chosen], sa["top_pos"][slot, chosen],
                                              sb["top_doc"][slot, chosen], sb["top_pos"][slot, chosen])
                metric_stats = {name: _stat(values) for name, values in metric_arrays.items()}
                if cohort_name not in cohort_all_values:
                    cohort_all_values[cohort_name] = {name: [] for name in metric_arrays}
                for name, values in metric_arrays.items():
                    if len(values) >= 50:
                        cohort_all_values[cohort_name][name].append(values)
                active_a_set = set(ids[np.flatnonzero(active_a)].tolist())
                active_b_set = set(ids[np.flatnonzero(active_b)].tolist())
                cohort_layer_rows[str(layer)][cohort_name] = {
                    "units": int(len(chosen)), "reported": int(len(chosen)) < 50,
                    "coverage_a": int(len(active_a_set.intersection(chosen.tolist()))),
                    "coverage_b": int(len(active_b_set.intersection(chosen.tolist()))),
                    "active_both": int(len(active_a_set.intersection(active_b_set).intersection(chosen.tolist()))),
                    "inactive_a": int(len(chosen) - len(active_a_set.intersection(chosen.tolist()))),
                    "inactive_b": int(len(chosen) - len(active_b_set.intersection(chosen.tolist()))),
                    "active_to_inactive": int(len(active_a_set.difference(active_b_set).intersection(chosen.tolist()))),
                    "inactive_to_active": int(len(active_b_set.difference(active_a_set).intersection(chosen.tolist()))),
                    "overlap": metric_stats,
                    "unit_diagnostics": {
                        "run_a": _firing_summary(a, slot, chosen),
                        "run_b": _firing_summary(b, slot, chosen),
                    },
                }
                null_by_cohort.setdefault(cohort_name, {})[str(layer)] = _null_for_ids(
                    a, b, slot, np.asarray(chosen, dtype=int), seed=GATE_SEED + len(group))
        cohort_summaries = {}
        for cohort_name, by_metric in cohort_all_values.items():
            pooled = {name: _stat(np.concatenate(values) if values else np.empty(0))
                      for name, values in by_metric.items()}
            cohort_summaries[cohort_name] = {"layers": {
                layer: cohort_layer_rows[layer][cohort_name] for layer in cohort_layer_rows
            }, "all_layers": {"overlap": pooled,
                              "excluded_under_50_layers": [layer for layer, row in cohort_layer_rows.items()
                                                           if row[cohort_name]["units"] < 50]}}
        result["groups"][group] = {"cohorts": cohort_summaries}
        result["null"][group] = null_by_cohort
    return result


def chance_summary(a: Run, b: Run, layers: Sequence[int], width: int, n_controls: int,
                   permutations: int = CHANCE_PERMUTATIONS, seed: int = GATE_SEED) -> dict:
    """Random same-layer unit pair baseline; median of 100 permutation-level means."""
    L = len(layers)
    U = a.state["max"].shape[1]
    groups = {"neurons": slice(0, width), "controls": slice(width, width + n_controls)}
    output: dict[str, Any] = {"permutations": permutations, "seed": seed, "groups": {}}
    for group, unit_slice in groups.items():
        per_metric_by_layer: dict[str, dict[str, Any]] = {}
        all_perm: dict[str, list[np.ndarray]] = {"top10_doc": [], "top10_document_position": []}
        for slot, layer in enumerate(layers):
            active_a = _active(a.state["max"][slot, unit_slice])
            active_b = _active(b.state["max"][slot, unit_slice])
            a_ids = np.arange(U)[unit_slice][active_a]
            b_ids = np.arange(U)[unit_slice][active_b]
            rng = np.random.default_rng(seed + slot * 7919 + len(group) * 101)
            sample_values = {"top10_doc": [], "top10_document_position": []}
            for _ in range(permutations):
                if len(a_ids) == 0 or len(b_ids) == 0:
                    sample_values["top10_doc"].append(math.nan)
                    sample_values["top10_document_position"].append(math.nan)
                    continue
                chosen = rng.choice(b_ids, size=len(a_ids), replace=True)
                metrics = _metric_block(a.state["top_doc"][slot, a_ids], a.state["top_pos"][slot, a_ids],
                                        b.state["top_doc"][slot, chosen], b.state["top_pos"][slot, chosen])
                sample_values["top10_doc"].append(float(metrics["top10_doc"].mean()))
                sample_values["top10_document_position"].append(
                    float(metrics["top10_document_position"].mean()))
            per_metric_by_layer[str(layer)] = {
                name: {"median_permutation_mean": float(np.nanmedian(values)) if np.isfinite(values).any() else None,
                       "permutations": permutations, "active_a": int(len(a_ids)), "active_b": int(len(b_ids))}
                for name, values in sample_values.items()
            }
            for name, values in sample_values.items():
                all_perm[name].append(np.asarray(values, dtype=float))
        pooled = {}
        for name, values in all_perm.items():
            per_layer = np.stack(values) if values else np.empty((0, permutations))
            aggregate = []
            for permutation in range(permutations):
                per_layer_means = per_layer[:, permutation] if len(per_layer) else np.empty(0)
                valid = np.isfinite(per_layer_means)
                if valid.any():
                    # Equal layer sizes are usual; active-count weighting also handles
                    # missing/inactive layers without giving them an artificial zero.
                    weights = np.asarray([_active(a.state["max"][slot, groups[group]]) .sum()
                                          for slot in range(L)], dtype=float)
                    aggregate.append(float(np.average(per_layer_means[valid], weights=weights[valid])))
            pooled[name] = {"median_permutation_mean": float(np.median(aggregate)) if aggregate else None,
                            "permutations": len(aggregate)}
        output["groups"][group] = {"layers": per_metric_by_layer, "all_layers": pooled}
    return output


def _pair_metric(pair_summaries: dict, key_a: str, key_b: str, group: str,
                 metric: str = "top10_doc") -> float | None:
    key = "__".join(sorted((key_a, key_b)))
    pair = pair_summaries.get(key)
    if not pair:
        return None
    value = pair["groups"][group]["all_layers"]["overlap"][metric]["median"]
    return value


def classify_threshold(value: float | None, threshold: float, direction: str) -> str:
    if value is None or not math.isfinite(float(value)):
        return "not assessable"
    if direction == "at_least":
        return "held" if value >= threshold else "failed"
    if direction == "at_most":
        return "held" if value <= threshold else "failed"
    raise ValueError(f"unknown threshold direction {direction}")


def classify_1812(overlap_median: float | None, floor_median: float | None) -> str:
    """Apply the preregistered noise-floor branch before the DPO/Instruct threshold."""
    if floor_median is None or not math.isfinite(float(floor_median)) or floor_median < 0.9:
        return "not assessable"
    return classify_threshold(overlap_median, 0.9, "at_least")


def classify_strictly_increasing(values: Sequence[float | None]) -> str:
    if len(values) != 6 or any(value is None or not math.isfinite(float(value)) for value in values):
        return "not assessable"
    return "held" if all(a < b for a, b in zip(values, values[1:])) else "failed"


def classify_layer_advantage(neuron_values: Sequence[float | None], control_values: Sequence[float | None],
                             required_layers: int = 12) -> tuple[str, dict]:
    assessable = [(n, c) for n, c in zip(neuron_values, control_values) if n is not None and c is not None]
    wins = sum(n > c for n, c in assessable)
    missing = len(neuron_values) - len(assessable)
    if wins >= required_layers:
        verdict = "held"
    elif wins + missing < required_layers:
        verdict = "failed"
    else:
        verdict = "not assessable"
    return verdict, {"layers": len(neuron_values), "assessable_layers": len(assessable),
                     "neuron_wins": int(wins), "unavailable_layers": int(missing),
                     "required_wins": required_layers}


def classify_slot_distance(slot_median: float | None, baseline_median: float | None,
                           slot_count: int = 21) -> tuple[str, dict]:
    if slot_median is None or baseline_median is None or slot_count != 21:
        return "not assessable", {"slot_median_overlap": slot_median,
                                   "active_neuron_baseline_median": baseline_median,
                                   "absolute_difference": None, "slot_count": slot_count,
                                   "tolerance": 0.15}
    difference = abs(float(slot_median) - float(baseline_median))
    return ("held" if difference <= 0.15 or math.isclose(difference, 0.15, abs_tol=1e-12) else "failed"), {
        "slot_median_overlap": float(slot_median),
        "active_neuron_baseline_median": float(baseline_median),
        "absolute_difference": difference, "slot_count": slot_count, "tolerance": 0.15,
    }


def _part92_slots(part92_dir: Path) -> tuple[list[dict], str | None]:
    kinds_path, neurons_path = part92_dir / "kinds.json", part92_dir / "neurons.json"
    if not kinds_path.is_file() or not neurons_path.is_file():
        return [], f"missing Part 92 files under {part92_dir}"
    kinds_value = json.loads(kinds_path.read_text(encoding="utf-8"))
    kinds = kinds_value.get("kinds", kinds_value)
    neurons_value = json.loads(neurons_path.read_text(encoding="utf-8"))
    slots = []
    for row in neurons_value:
        if kinds.get(row.get("id")) == "slot":
            index = row.get("index", row.get("neuron", row.get("unit")))
            if index is not None and row.get("layer") is not None:
                slots.append({"id": row.get("id"), "layer": int(row["layer"]), "unit": int(index),
                              "persona": row.get("persona")})
    return slots, None


def _primary_ids_map(instruct: Run | None) -> dict[int, np.ndarray]:
    if instruct is None:
        return {}
    layers, width, _ = _layout(instruct)
    return {layer: np.flatnonzero(_active(instruct.state["max"][slot, :width]))
            for slot, layer in enumerate(layers)}


def _run_index(run: Run) -> dict[int, int]:
    return {layer: slot for slot, layer in enumerate(_layout(run)[0])}


def _overlap_population(a: Run, b: Run, group: str, cohort: str,
                        primary_ids: dict[int, np.ndarray], passing_layers: Sequence[int] | None = None,
                        exclude_position0: bool = False) -> dict:
    layers, width, _ = _layout(a)
    ia, ib = _run_index(a), _run_index(b)
    pop_a, pop_b = _populations(a).get(group), _populations(b).get(group)
    if pop_a is None or pop_b is None or not np.array_equal(pop_a, pop_b):
        return {"mean": None, "median": None, "units": 0, "layers": {}}
    per_layer = {}
    pooled = []
    allowed = set(layers if passing_layers is None else passing_layers)
    for layer in layers:
        if layer not in allowed:
            continue
        sa, sb = ia[layer], ib[layer]
        if group == "neurons":
            ids = (np.asarray(primary_ids.get(layer, []), dtype=int) if cohort == "primary"
                   else np.arange(width, dtype=int))
        else:
            ids = pop_a
        ids = ids[(ids >= 0) & (ids < a.state["max"].shape[1])]
        if exclude_position0 and len(ids):
            pa = a.state["top_pos"][sa, ids, :10]
            pb = b.state["top_pos"][sb, ids, :10]
            valid_a, valid_b = pa >= 0, pb >= 0
            zero_a = (pa == 0).sum(axis=1) > valid_a.sum(axis=1) / 2
            zero_b = (pb == 0).sum(axis=1) > valid_b.sum(axis=1) / 2
            ids = ids[~(zero_a | zero_b)]
        values = _metric_block(a.state["top_doc"][sa, ids], a.state["top_pos"][sa, ids],
                               b.state["top_doc"][sb, ids], b.state["top_pos"][sb, ids])
        stats = {name: _stat(arr) for name, arr in values.items()}
        per_layer[str(layer)] = {"units": int(len(ids)), "overlap": stats}
        if len(ids) >= 50:
            pooled.extend(values["top10_doc"].tolist())
    aggregate = _stat(np.asarray(pooled, dtype=float))
    return {**aggregate, "layers": per_layer,
            "cohort": cohort, "excluded_under_50_layers": [key for key, row in per_layer.items()
                                                             if row["units"] < 50]}


def _gate3(runs_by_key: dict[str, Run], primary_ids: dict[int, np.ndarray]) -> dict:
    repeat_pairs = (("instruct", "instruct_b"), ("pretrain_1T", "pretrain_1T_b"))
    layers = sorted(primary_ids)
    result_by_layer: dict[str, Any] = {}
    passing = []
    for layer in layers:
        ids = primary_ids[layer]
        pair_results = {}
        layer_passes = len(ids) >= 50
        for first, second in repeat_pairs:
            a, b = runs_by_key.get(first), runs_by_key.get(second)
            if a is None or b is None or layer not in _run_index(a) or layer not in _run_index(b) or len(ids) < 50:
                pair_results[f"{first}_vs_{second}"] = {"verdict": "not assessable", "units": int(len(ids))}
                layer_passes = False
                continue
            ai, bi = _run_index(a)[layer], _run_index(b)[layer]
            metric = _metric_block(a.state["top_doc"][ai, ids], a.state["top_pos"][ai, ids],
                                   b.state["top_doc"][bi, ids], b.state["top_pos"][bi, ids])["top10_doc"]
            active_a = _active(a.state["max"][ai, :_layout(a)[1]])[ids]
            active_b = _active(b.state["max"][bi, :_layout(b)[1]])[ids]
            agreement = float(np.mean(active_a == active_b)) if len(ids) else None
            control_medians = {}
            for group in ("dense", "sparse_k2", "sparse_k4", "sparse_k16", "sparse_k64"):
                pop_a, pop_b = _populations(a).get(group), _populations(b).get(group)
                if pop_a is None or pop_b is None or not len(pop_a):
                    control_medians[group] = None
                    continue
                active_ctrl_a = _active(a.state["max"][ai, pop_a])
                active_ctrl_b = _active(b.state["max"][bi, pop_b])
                both = pop_a[active_ctrl_a & active_ctrl_b]
                vals = _metric_block(a.state["top_doc"][ai, both], a.state["top_pos"][ai, both],
                                     b.state["top_doc"][bi, both], b.state["top_pos"][bi, both])["top10_doc"]
                control_medians[group] = {"median": float(np.median(vals)) if len(vals) >= 50 else None,
                                          "active_both_units": int(len(both))}
            median = float(np.median(metric))
            pair_pass = median >= 0.95 and agreement is not None and agreement >= 0.99
            layer_passes = layer_passes and pair_pass
            pair_results[f"{first}_vs_{second}"] = {
                "verdict": "passed" if pair_pass else "failed", "units": int(len(ids)),
                "neuron_median": median, "neuron_p10": float(np.percentile(metric, 10)),
                "share_below_0_5": float(np.mean(metric < 0.5)),
                "activity_mask_agreement": agreement, "control_medians": control_medians,
                "thresholds": {"neuron_median": 0.95, "activity_mask_agreement": 0.99},
            }
        result_by_layer[str(layer)] = {"verdict": "passed" if layer_passes else "failed",
                                      "primary_cohort_units": int(len(ids)), "repeat_pairs": pair_results}
        if layer_passes:
            passing.append(layer)
    return {"verdict": "passed" if len(passing) >= 12 else "failed",
            "criterion": "each passing layer meets both repeat-pair thresholds; headline requires 12 layers",
            "passing_layers": passing, "failing_layers": [layer for layer in layers if layer not in passing],
            "required_passing_layers": 12, "layers": result_by_layer}


def _pair_prediction_stats(runs_by_key: dict[str, Run], first: str, second: str,
                           primary_ids: dict[int, np.ndarray], passing_layers: Sequence[int],
                           group: str = "neurons", cohort: str = "primary",
                           exclude_position0: bool = False) -> dict:
    a, b = runs_by_key.get(first), runs_by_key.get(second)
    if a is None or b is None:
        return {"mean": None, "median": None, "units": 0, "layers": {},
                "reason": f"{first} and {second} passing runs are both required"}
    return _overlap_population(a, b, group, cohort, primary_ids, passing_layers,
                               exclude_position0=exclude_position0)


def _slot_verdict(runs_by_key: dict[str, Run], primary_ids: dict[int, np.ndarray],
                  passing_layers: Sequence[int], part92_dir: Path,
                  exclude_position0: bool = False) -> dict:
    slots, error = _part92_slots(part92_dir)
    base = {"criterion": "median of all 21 slot overlaps is within 0.15 of their layers' primary-cohort median",
            "slot_count": len(slots), "units": []}
    if error:
        return {**base, "verdict": "not assessable", "reason": error}
    sft, dpo = runs_by_key.get("sft"), runs_by_key.get("dpo")
    if sft is None or dpo is None:
        return {**base, "verdict": "not assessable", "reason": "sft and dpo passing runs are both required",
                "units": [{**slot, "overlap": None, "comparator": None} for slot in slots]}
    comparator = _overlap_population(sft, dpo, "neurons", "primary", primary_ids, passing_layers,
                                     exclude_position0=exclude_position0)
    layers_s, width, _ = _layout(sft)
    si, di = _run_index(sft), _run_index(dpo)
    eligible_slots: list[float] = []
    comparators: list[float] = []
    units = []
    for slot in slots:
        layer, unit = slot["layer"], slot["unit"]
        item = {**slot, "overlap": None, "comparator": None,
                "prediction_included": layer in set(passing_layers), "active_sft": None, "active_dpo": None}
        if layer not in si or layer not in di or unit < 0 or unit >= width:
            item["reason"] = "layer or unit is outside the collected state"
            units.append(item)
            continue
        sslot, dslot = si[layer], di[layer]
        item["active_sft"] = bool(_active(sft.state["max"][sslot, :width])[unit])
        item["active_dpo"] = bool(_active(dpo.state["max"][dslot, :width])[unit])
        value = float(_metric_block(sft.state["top_doc"][sslot, unit:unit + 1],
                                    sft.state["top_pos"][sslot, unit:unit + 1],
                                    dpo.state["top_doc"][dslot, unit:unit + 1],
                                    dpo.state["top_pos"][dslot, unit:unit + 1])["top10_doc"][0])
        item["overlap"] = value
        comparator_value = comparator.get("layers", {}).get(str(layer), {}).get("overlap", {}) \
            .get("top10_doc", {}).get("median")
        item["comparator"] = comparator_value
        if item["prediction_included"] and comparator_value is not None:
            if not exclude_position0 or not _mostly_position_zero(sft.state["top_pos"][sslot, unit],
                                                                   dpo.state["top_pos"][dslot, unit]):
                eligible_slots.append(value)
                comparators.append(float(comparator_value))
        units.append(item)
    if len(eligible_slots) != 21 or len(comparators) != 21:
        return {**base, "verdict": "not assessable",
                "numbers": {"assessable_slots": len(eligible_slots), "assessable_comparators": len(comparators),
                            "excluded_layers": sorted(set(slot["layer"] for slot in slots) - set(passing_layers)),
                            "sensitivity_position0_excluded": exclude_position0}, "units": units}
    slot_median, comparator_median = float(np.median(eligible_slots)), float(np.median(comparators))
    verdict, numbers = classify_slot_distance(slot_median, comparator_median, len(slots))
    return {**base, "verdict": verdict, "numbers": {**numbers,
            "assessable_slots": len(eligible_slots), "median_comparator": comparator_median,
            "sensitivity_position0_excluded": exclude_position0}, "units": units}


def _mostly_position_zero(a_pos: np.ndarray, b_pos: np.ndarray) -> bool:
    for positions in (a_pos[:10], b_pos[:10]):
        valid = positions >= 0
        count = int(valid.sum())
        if count and int((positions[valid] == 0).sum()) > count / 2:
            return True
    return False


def _make_predictions(runs_by_key: dict[str, Run], primary_ids: dict[int, np.ndarray],
                      passing_layers: Sequence[int], part92_dir: Path,
                      exclude_position0: bool = False) -> dict:
    def stats(first: str, second: str, group: str = "neurons"):
        return _pair_prediction_stats(runs_by_key, first, second, primary_ids, passing_layers,
                                      group, "primary" if group == "neurons" else "all", exclude_position0)
    s11 = stats("midtrain_51B", "instruct")
    s12 = stats("dpo", "instruct")
    s13 = stats("pretrain_21B", "instruct")
    floor = stats("pretrain_4T", "pretrain_4T_bf16r")
    v12 = classify_1812(s12.get("median"), floor.get("median"))
    v1811 = {"verdict": classify_threshold(s11.get("median"), 0.6, "at_least"),
             "criterion": "primary-cohort median top-document overlap is at least 0.6",
             "numbers": {"mean": s11.get("mean"), "median": s11.get("median"), "threshold": 0.6,
                         "units": s11.get("units")}}
    v1812 = {"verdict": v12,
             "criterion": "primary-cohort median dpo/instruct overlap is at least 0.9; floor median must be at least 0.9",
             "numbers": {"mean": s12.get("mean"), "median": s12.get("median"), "threshold": 0.9,
                         "floor_median": floor.get("median"), "units": s12.get("units")}}
    v1813 = {"verdict": classify_threshold(s13.get("median"), 0.1, "at_most"),
             "criterion": "primary-cohort median top-document overlap is at most 0.1",
             "numbers": {"mean": s13.get("mean"), "median": s13.get("median"), "threshold": 0.1,
                         "units": s13.get("units")}}

    pretrain_keys = ["pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T"]
    trajectory = [stats(key, "instruct") for key in pretrain_keys]
    means = [row.get("mean") for row in trajectory]
    medians = [row.get("median") for row in trajectory]
    v1814 = {"verdict": classify_strictly_increasing(means),
             "criterion": "mean neuron overlap with instruct rises strictly along the six pretraining stages",
             "numbers": {"keys": pretrain_keys, "means": means, "medians": medians}}

    comparisons = [("pretrain_1T", "pretrain_2T"), ("pretrain_4T", "midtrain_51B"),
                   ("midtrain_51B", "instruct")]
    advantage_rows = []
    for first, second in comparisons:
        neurons = stats(first, second)
        curves = {"1": neurons, "dense": stats(first, second, "dense")}
        for k in (2, 4, 16, 64):
            curves[str(k)] = stats(first, second, f"sparse_k{k}")
        per_layer = []
        wins = 0
        assessable = 0
        for layer in passing_layers:
            nrow = neurons.get("layers", {}).get(str(layer), {}).get("overlap", {}).get("top10_doc", {})
            dense_row = curves["dense"].get("layers", {}).get(str(layer), {}).get("overlap", {}).get("top10_doc", {})
            sparse_row = curves["2"].get("layers", {}).get(str(layer), {}).get("overlap", {}).get("top10_doc", {})
            nmean, dmean, k2mean = nrow.get("mean"), dense_row.get("mean"), sparse_row.get("mean")
            passed = nmean is not None and dmean is not None and k2mean is not None
            if passed:
                assessable += 1
                wins += int(nmean > dmean and nmean > k2mean)
            per_layer.append({"layer": layer, "neuron_mean": nmean, "dense_mean": dmean,
                              "sparse_k2_mean": k2mean, "neuron_exceeds_both": bool(passed and nmean > dmean and nmean > k2mean)})
        required = math.ceil(0.75 * len(passing_layers))
        verdict = "not assessable" if not passing_layers else (
            "held" if wins >= required else "failed" if wins + (len(passing_layers) - assessable) < required
            else "not assessable")
        advantage_rows.append({"runs": [first, second], "verdict": verdict,
                               "numbers": {"passing_layers": len(passing_layers), "assessable_layers": assessable,
                                           "layers_exceeding_both": wins, "required_layers": required,
                                           "per_layer": per_layer},
                               "mean_overlap_by_k": {k: row.get("mean") for k, row in curves.items()}})
    if all(row["verdict"] == "held" for row in advantage_rows):
        v1815 = "held"
    elif any(row["verdict"] == "failed" for row in advantage_rows):
        v1815 = "failed"
    else:
        v1815 = "not assessable"
    v1816 = _slot_verdict(runs_by_key, primary_ids, passing_layers, part92_dir, exclude_position0)
    return {"181.1": v1811, "181.2": v1812, "181.3": v1813, "181.4": v1814,
            "181.5": {"verdict": v1815,
                      "criterion": "mean neuron overlap exceeds dense and k=2 means in at least three quarters of passing layers",
                      "numbers": {"comparisons": advantage_rows}},
            "181.6": v1816}


def _firing_summary(run: Run, slot: int, unit_ids: np.ndarray,
                    include_per_unit: bool = False) -> dict:
    unit_ids = np.asarray(unit_ids, dtype=int)
    maxima = run.state["max"][slot, unit_ids]
    acts = run.state["top_act"][slot, unit_ids]
    docs = run.state["top_doc"][slot, unit_ids]
    positions = run.state["top_pos"][slot, unit_ids]
    valid = (docs >= 0) & np.isfinite(acts)
    firing = np.where(maxima > 0, ((acts >= 0.1 * maxima[:, None]) & valid).sum(axis=1), 0).astype(int)
    top10_valid = valid[:, :10]
    top10_position0 = ((positions[:, :10] == 0) & top10_valid).sum(axis=1).astype(int)
    top10_entries = top10_valid.sum(axis=1).astype(int)
    entry_count = int(top10_entries.sum())
    result = {
        "firing_count_mean": float(np.mean(firing)) if len(firing) else None,
        "firing_count_median": float(np.median(firing)) if len(firing) else None,
        "units_under_10": int(np.sum(firing < 10)),
        "units": int(len(unit_ids)),
        "top10_position0_share": int(top10_position0.sum()) / entry_count if entry_count else None,
        "top10_position0_entries": int(top10_position0.sum()),
        "top10_entries": entry_count,
    }
    if include_per_unit:
        result.update({
            "unit_ids": unit_ids.tolist(),
            "per_unit_firing_count": firing.tolist(),
            "per_unit_top10_position0_count": top10_position0.tolist(),
            "per_unit_top10_entry_count": top10_entries.tolist(),
        })
    return result


def _stage_summaries(runs: Sequence[Run]) -> dict:
    result = {}
    for run in runs:
        layers, width, _ = _layout(run)
        by_layer = {}
        populations = _populations(run)
        population_layers = {name: {} for name in populations}
        all_population_counts: dict[str, list[np.ndarray]] = {name: [] for name in populations}
        for slot, layer in enumerate(layers):
            per_layer = {
                name: _firing_summary(run, slot, unit_ids, include_per_unit=True)
                for name, unit_ids in populations.items()
            }
            aggregate_layer = {}
            for name, unit_ids in populations.items():
                population_layers[name][str(layer)] = per_layer[name]
                aggregate_layer[name] = {key: value for key, value in per_layer[name].items()
                                         if not key.startswith("per_unit_") and key != "unit_ids"}
                if len(unit_ids):
                    all_population_counts[name].append(
                        np.asarray(per_layer[name]["per_unit_firing_count"], dtype=int))
            neuron = aggregate_layer.get("neurons", _firing_summary(
                run, slot, np.arange(min(width, run.state["max"].shape[1])), False))
            by_layer[str(layer)] = {**neuron, "populations": aggregate_layer}
        population_summaries = {}
        for name, layer_rows in population_layers.items():
            firing = np.concatenate(all_population_counts[name]) if all_population_counts[name] else np.empty(0, dtype=int)
            positions0 = sum(row["top10_position0_entries"] for row in layer_rows.values())
            entries = sum(row["top10_entries"] for row in layer_rows.values())
            population_summaries[name] = {
                "layers": layer_rows,
                "all_layers": {
                    "firing_count_mean": float(np.mean(firing)) if len(firing) else None,
                    "firing_count_median": float(np.median(firing)) if len(firing) else None,
                    "units_under_10": int(np.sum(firing < 10)),
                    "units": int(len(firing)),
                    "top10_position0_share": positions0 / entries if entries else None,
                    "top10_position0_entries": positions0,
                    "top10_entries": entries,
                },
            }
        neuron_all = population_summaries.get("neurons", {}).get("all_layers", {})
        result[run.key] = {"layers": by_layer,
                           "populations": population_summaries,
                           "firing_count_mean": neuron_all.get("firing_count_mean"),
                           "firing_count_median": neuron_all.get("firing_count_median"),
                           "units_under_10": neuron_all.get("units_under_10", 0),
                           "units": neuron_all.get("units", 0),
                           "top10_position0_share": neuron_all.get("top10_position0_share"),
                           "pile_set_name_counts": run.identity.get("pile_set_name_counts", {})}
    return result


def _snapshot_top10(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as state:
        return {name: state[name] for name in ("max", "top_doc", "top_pos")}


def saturation_summary(instruct: Run | None, primary_ids: dict[int, np.ndarray] | None = None) -> dict:
    if instruct is None:
        return {"verdict": "not assessable", "reason": "passing instruct run is unavailable", "steps": []}
    primary_ids = primary_ids or _primary_ids_map(instruct)
    width = _layout(instruct)[1]
    output = []
    for first, second, baseline in zip((500, 1000, 1500, 2000), (1000, 1500, 2000, 2500),
                                       (0.5, 0.667, 0.75, 0.8)):
        a_path = instruct.path / f"snapshot-pass1-docs{first:05d}.npz"
        b_path = instruct.path / f"snapshot-pass1-docs{second:05d}.npz"
        if not a_path.is_file() or not b_path.is_file():
            output.append({"from_docs": first, "to_docs": second, "expected_nested_uniform": baseline,
                           "primary_median": None, "all_units_median": None, "reason": "snapshot missing"})
            continue
        a, b = _snapshot_top10(a_path), _snapshot_top10(b_path)
        primary_values, all_values = [], []
        for slot, layer in enumerate(_layout(instruct)[0]):
            ids = np.asarray(primary_ids.get(layer, []), dtype=int)
            if len(ids) >= 50:
                primary_values.extend(_intersection_over_k(a["top_doc"][slot, ids], a["top_pos"][slot, ids],
                                                           b["top_doc"][slot, ids], b["top_pos"][slot, ids],
                                                           10, positions=False).tolist())
            all_ids = np.arange(width)
            all_values.extend(_intersection_over_k(a["top_doc"][slot, all_ids], a["top_pos"][slot, all_ids],
                                                   b["top_doc"][slot, all_ids], b["top_pos"][slot, all_ids],
                                                   10, positions=False).tolist())
        output.append({"from_docs": first, "to_docs": second, "expected_nested_uniform": baseline,
                       "primary_median": float(np.median(primary_values)) if primary_values else None,
                       "primary_units": len(primary_values),
                       "all_units_median": float(np.median(all_values)) if all_values else None,
                       "all_units": len(all_values)})
    return {"verdict": "reported", "steps": output}


def build_verdicts(runs: Sequence[Run], gate2: dict, primary_ids: dict[int, np.ndarray],
                   part92_dir: Path, exclusions: Sequence[dict] = ()) -> dict:
    by_key = {run.key: run for run in runs}
    gate3 = _gate3(by_key, primary_ids)
    passing_layers = gate3["passing_layers"]
    predictions = _make_predictions(by_key, primary_ids, passing_layers, part92_dir)
    sensitivity = _make_predictions(by_key, primary_ids, passing_layers, part92_dir, exclude_position0=True)
    stage_summary = _stage_summaries(runs)
    pretrain_keys = ["pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T", "pretrain_4T"]
    all_1811 = _pair_prediction_stats(by_key, "midtrain_51B", "instruct", primary_ids,
                                     passing_layers, "neurons", "all")
    all_1813 = _pair_prediction_stats(by_key, "pretrain_21B", "instruct", primary_ids,
                                     passing_layers, "neurons", "all")
    all_1814_means = [_pair_prediction_stats(by_key, key, "instruct", primary_ids,
                                             passing_layers, "neurons", "all").get("mean")
                      for key in pretrain_keys]
    included_keys = {run.key for run in runs}
    unresolved_gate1 = [str(item.get("run")) for item in exclusions
                        if any("Gate 1 did not pass" in reason for reason in item.get("reasons", [])) and
                        item.get("run") not in included_keys]
    missing_gate1_keys = sorted(EXPECTED_RUN_KEYS - included_keys)
    gate1_status = ("passed" if len(included_keys) == len(EXPECTED_RUN_KEYS) else
                    "failed" if unresolved_gate1 else "not assessable")
    reading = {
        "gate1_passed": gate1_status == "passed",
        "gate2_passed": gate2.get("verdict") == "passed",
        "gate3_at_least_12_layers": len(passing_layers) >= 12,
        "181_1_primary_direction": predictions["181.1"]["verdict"] == "held",
        "181_3_primary_direction": predictions["181.3"]["verdict"] == "held",
        "181_4_primary_direction": predictions["181.4"]["verdict"] == "held",
        "181_1_same_direction_all_units": classify_threshold(all_1811.get("median"), 0.6, "at_least") == "held",
        "181_3_same_direction_all_units": classify_threshold(all_1813.get("median"), 0.1, "at_most") == "held",
        "181_4_same_direction_all_units": classify_strictly_increasing(all_1814_means) == "held",
    }
    gate1 = {"verdict": gate1_status, "included_runs": sorted(included_keys),
             "missing_checkpoint_keys": missing_gate1_keys, "unresolved_failed_runs": unresolved_gate1}
    verdicts = {"gates": {"gate1": gate1, "gate2": gate2, "gate3": gate3}, **predictions,
                "sensitivity_without_position0": {
                    name: {"verdict": row["verdict"], "numbers": row.get("numbers", {})}
                    for name, row in sensitivity.items()},
                "reading_rule": {"headline_ready": all(reading.values()), "checks": reading,
                                 "primary_cohort": {
                                     "181.1": predictions["181.1"]["verdict"],
                                     "181.3": predictions["181.3"]["verdict"],
                                     "181.4": predictions["181.4"]["verdict"]},
                                 "all_units": {"181.1_median": all_1811.get("median"),
                                               "181.3_median": all_1813.get("median"),
                                               "181.4_means": all_1814_means}},
                "saturation": saturation_summary(by_key.get("instruct"), primary_ids),
                "stage_summaries": stage_summary,
                "pile_set_name_counts": (by_key.get("instruct").identity.get("pile_set_name_counts", {})
                                         if by_key.get("instruct") else {}),
                "excluded_runs": []}
    return verdicts


def _write_tables(out: Path, runs: Sequence[Run], exclusions: list[dict], pairs: dict, verdicts: dict) -> None:
    lines = ["# Part 181 analysis", "", "## Gates and predictions", "",
             "| Item | Verdict | Deciding number |", "|---|---|---|"]
    for name in ("gate1", "gate2", "gate3"):
        row = verdicts["gates"][name]
        lines.append(f"| {name} | {row['verdict']} | {json.dumps(row.get('numbers', row), sort_keys=True)} |")
    for name in ("181.1", "181.2", "181.3", "181.4", "181.5", "181.6"):
        row = verdicts[name]
        lines.append(f"| {name} | {row['verdict']} | {json.dumps(row.get('numbers', {}), sort_keys=True)} |")
    lines.extend(["", "## Run exclusions", ""])
    if exclusions:
        for item in exclusions:
            lines.append(f"- **{item['run']}**: {'; '.join(item['reasons'])}")
    else:
        lines.append("None.")
    lines.extend(["", "## Pairwise top-10 document overlap", "",
                  "| Run A | Run B | Primary neuron median | Dense median | Sparse k=2 median | Null median |",
                  "|---|---|---:|---:|---:|---:|"])
    for summary in pairs.values():
        groups = summary["groups"]
        neuron = groups["neurons"]["cohorts"].get("primary", {}).get("all_layers", {}).get("overlap", {}) \
            .get("top10_doc", {}).get("median")
        dense = groups.get("dense", {}).get("cohorts", {}).get("all", {}).get("all_layers", {}) \
            .get("overlap", {}).get("top10_doc", {}).get("median")
        sparse = groups.get("sparse_k2", {}).get("cohorts", {}).get("all", {}).get("all_layers", {}) \
            .get("overlap", {}).get("top10_doc", {}).get("median")
        null_values = summary.get("null", {}).get("neurons", {}).get("primary", {})
        null_medians = [row.get("median_summary", {}).get("median") for row in null_values.values()
                        if row.get("verdict") == "reported"]
        null_median = float(np.median(null_medians)) if null_medians else None
        lines.append(f"| {summary['run_a']} | {summary['run_b']} | {neuron} | {dense} | {sparse} | {null_median} |")
    lines.extend(["", "## Gate 1 per-unit values", ""])
    for run in runs:
        gate = json.loads((run.path / "gate1.json").read_text(encoding="utf-8"))
        lines.append(f"### {run.key}: {gate.get('verdict', 'unknown')}")
        lines.append("")
        lines.append("| Layer | Unit | Document | Position | Collector | Forward | Relative difference | Pass |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        entries = gate.get("checks", {}).get("1a", {}).get("entries", gate.get("pairs", []))
        for row in entries:
            lines.append(f"| {row.get('layer')} | {row.get('unit')} | {row.get('doc')} | {row.get('position')} | "
                         f"{row.get('stored', row.get('collector_value'))} | {row.get('forward', row.get('forward_value'))} | "
                         f"{row.get('relative_difference')} | {row.get('pass')} |")
        lines.append("")
    lines.extend(["## Gate 3 by layer", "", "| Layer | Verdict | Primary units |", "|---:|---|---:|"])
    for layer, row in verdicts["gates"]["gate3"].get("layers", {}).items():
        lines.append(f"| {layer} | {row['verdict']} | {row['primary_cohort_units']} |")
    lines.extend(["## Instruct saturation", ""])
    for row in verdicts["saturation"].get("steps", []):
        lines.append(f"- {row['from_docs']} → {row['to_docs']} documents: "
                     f"primary median {row['primary_median']}; all-neuron median {row['all_units_median']}; "
                     f"nested-sampling reference {row['expected_nested_uniform']}")
    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze(runs_dir: Path, out: Path, part92_dir: Path | None = None) -> dict:
    runs_dir, out = Path(runs_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    passing, exclusions, gate2 = _resolve_runs(runs_dir)
    if part92_dir is None:
        part92_dir = PROJECT_ROOT / "results" / "part92" / "step2"
    runs_by_key = {run.key: run for run in passing}
    primary_ids = _primary_ids_map(runs_by_key.get("instruct"))
    pair_summaries: dict[str, dict] = {}
    for a, b in itertools.combinations(sorted(passing, key=lambda run: run.key), 2):
        try:
            summary = summarize_pair(a, b, primary_ids_by_layer=primary_ids)
        except Exception as exc:
            exclusions.append({"run_pair": [a.key, b.key], "reasons": [f"comparison failed: {exc}"]})
            continue
        pair_summaries["__".join(sorted((a.key, b.key)))] = summary
    floor_key = "__".join(sorted(("pretrain_4T", "pretrain_4T_bf16r")))
    floor_pair = pair_summaries.get(floor_key)
    floor_by_layer = {}
    if floor_pair:
        for group, group_data in floor_pair["groups"].items():
            cohort = "primary" if group == "neurons" else "all"
            layer_rows = group_data.get("cohorts", {}).get(cohort, {}).get("layers", {})
            floor_by_layer[group] = {layer: {
                "mean": row.get("overlap", {}).get("top10_doc", {}).get("mean"),
                "median": row.get("overlap", {}).get("top10_doc", {}).get("median"),
                "units": row.get("units", 0),
            }
                                     for layer, row in layer_rows.items()}
    for summary in pair_summaries.values():
        summary["floor_by_layer"] = floor_by_layer
        summary["floor_pair"] = ["pretrain_4T", "pretrain_4T_bf16r"]
    verdicts = build_verdicts(passing, gate2, primary_ids, Path(part92_dir), exclusions)
    verdicts["excluded_runs"] = exclusions
    overlaps = {"included_runs": [run.key for run in passing], "excluded_runs": exclusions,
                "stage_summaries": verdicts.get("stage_summaries", {}),
                "floor_by_layer": floor_by_layer,
                "pair_count": len(pair_summaries), "pairs": pair_summaries}
    (out / "overlaps.json").write_text(json.dumps(overlaps, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                        encoding="utf-8")
    (out / "verdicts.json").write_text(json.dumps(verdicts, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                        encoding="utf-8")
    _write_tables(out, passing, exclusions, pair_summaries, verdicts)
    return {"included_runs": [run.key for run in passing], "excluded_runs": exclusions,
            "pair_count": len(pair_summaries), "verdicts": verdicts}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path, help="directory containing one subdirectory per run key")
    parser.add_argument("--out", required=True, type=Path, help="analysis output directory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = analyze(args.runs, args.out)
    except Exception as exc:
        print(f"analyze181 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"included_runs": result["included_runs"], "pair_count": result["pair_count"],
                      "verdicts": {key: value["verdict"] for key, value in result["verdicts"].items()
                                   if isinstance(value, dict) and "verdict" in value}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
