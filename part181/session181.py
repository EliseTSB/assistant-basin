#!/usr/bin/env python3
"""Run one pinned Part 181 OLMo 2 activation-atlas session."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import resource
import socket
import sys
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_ID = "NeelNanda/pile-10k"
DATASET_REVISION_PREFIX = "127bfedc"
DATASET_FILE = "data/train-00000-of-00001-4746b8785c874cc7.parquet"
N_DOCS = 2500
PASS2_DOCS = 500
SEED = 5601
GATE_SEED = 181
REGISTERED_TOKEN_SHA256 = "5238544b790184de8f34a639b32e0c4f362dfb860e345df918671cdb8609d866"
REGISTERED_TOKEN_COUNT = 885_679
REGISTERED_CAPPED_DOCS = 992
REGISTERED_DENSE_CONTROL_SHA256 = "54515ce29b1c36af123d8337932e510744b3ab77951e3b520f7ada43a10ff20a"
PINNED_VERSIONS = {"numpy": "2.4.6", "torch": "2.15.0", "transformers": "5.22.1", "tokenizers": "0.23.3"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: Sequence[Sequence[int]]) -> str:
    """Hash compact JSON id lists concatenated in document order."""
    digest = hashlib.sha256()
    for row in token_ids:
        digest.update(json.dumps([int(x) for x in row], separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def control_sha256_by_layer(generate_controls: Callable[..., np.ndarray], width: int = 8192,
                            n_controls: int = 256, seed: int = SEED,
                            layers: Sequence[int] = tuple(range(16))) -> dict[str, str]:
    return {
        str(layer): hashlib.sha256(
            np.ascontiguousarray(generate_controls(n_controls, width, seed, layer), dtype=np.float32)
            .tobytes(order="C")
        ).hexdigest()
        for layer in layers
    }


def control_hashes(collector: Any, width: int = 8192, n_dense: int = 256,
                   n_sparse_per_k: int = 64, seed: int = SEED,
                   layers: Sequence[int] = tuple(range(16))) -> dict[str, Any]:
    dense = np.stack([collector.generate_controls(n_dense, width, seed, layer) for layer in layers])
    dense_by_layer = {str(layer): hashlib.sha256(np.ascontiguousarray(dense[i], dtype=np.float32).tobytes()).hexdigest()
                      for i, layer in enumerate(layers)}
    sparse_by_layer: dict[str, dict[str, str]] = {}
    sparse_all = hashlib.sha256()
    for i, layer in enumerate(layers):
        sparse_by_layer[str(layer)] = {}
        for k in collector.SPARSE_SUPPORTS:
            matrix = collector.generate_sparse_controls(n_sparse_per_k, width, int(k), seed, layer)
            payload = np.ascontiguousarray(matrix, dtype=np.float32).tobytes()
            sparse_all.update(payload)
            sparse_by_layer[str(layer)][str(k)] = hashlib.sha256(payload).hexdigest()
    return {
        "dense_sha256": hashlib.sha256(np.ascontiguousarray(dense, dtype=np.float32).tobytes()).hexdigest(),
        "dense_by_layer_sha256": dense_by_layer,
        "sparse_sha256": sparse_all.hexdigest(),
        "sparse_by_layer_k_sha256": sparse_by_layer,
    }


def verify_pinned_versions() -> dict[str, str | None]:
    """Fail before downloads unless this host matches the preregistered CPU stack."""
    versions = library_versions()
    mismatches = [f"{name}={versions.get(name)!r} (expected {expected})"
                  for name, expected in PINNED_VERSIONS.items()
                  if (versions.get(name) or "").split("+", 1)[0] != expected]
    import torch
    if torch.version.cuda is not None or getattr(torch.version, "hip", None) is not None:
        mismatches.append("torch is not a CPU-only build")
    if mismatches:
        raise RuntimeError("pinned library check failed: " + "; ".join(mismatches))
    return versions


def _check_full_commit(value: Any, prefix: str, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or not value.startswith(prefix):
        raise ValueError(f"{label} resolved to {value!r}; expected a 40-character commit starting {prefix}")
    return value


def resolve_inputs(key: str, api: Any) -> tuple[dict, str]:
    """Resolve model and dataset branch prefixes to immutable full commits."""
    from part85 import checkpoints

    checkpoint_key = {"instruct_b": "instruct", "pretrain_1T_b": "pretrain_1T",
                      "pretrain_4T_bf16r": "pretrain_4T"}.get(key, key)
    if checkpoint_key not in checkpoints.BY_KEY:
        raise ValueError(f"unknown checkpoint key: {key}")
    checkpoint = checkpoints.resolve_checkpoint(checkpoint_key, api=api)
    model_commit = _check_full_commit(checkpoint.get("commit"), checkpoint["commit_prefix"],
                                      f"model {checkpoint_key}")
    dataset_info = api.dataset_info(DATASET_ID, revision=DATASET_REVISION_PREFIX)
    dataset_commit = _check_full_commit(getattr(dataset_info, "sha", None), DATASET_REVISION_PREFIX,
                                        "pile-10k dataset")
    return checkpoint, dataset_commit


def download_model_snapshot(checkpoint: dict, work: Path,
                            snapshot_download_fn: Callable[..., str]) -> Path:
    """Download only the registered immutable model revision into this run's cache."""
    commit = _check_full_commit(checkpoint.get("commit"), checkpoint["commit_prefix"],
                                f"model {checkpoint['key']}")
    path = snapshot_download_fn(repo_id=checkpoint["repo_id"], revision=commit,
                               cache_dir=str(Path(work) / "hf-cache"))
    return Path(path)


def download_corpus(dataset_commit: str, work: Path,
                    hf_hub_download_fn: Callable[..., str]) -> Path:
    dataset_commit = _check_full_commit(dataset_commit, DATASET_REVISION_PREFIX, "pile-10k dataset")
    path = hf_hub_download_fn(repo_id=DATASET_ID, filename=DATASET_FILE, repo_type="dataset",
                              revision=dataset_commit, cache_dir=str(Path(work) / "hf-cache"))
    return Path(path)


def library_versions() -> dict[str, str | None]:
    names = ("numpy", "torch", "transformers", "tokenizers", "huggingface_hub", "pyarrow")
    versions = {name: _version(name) for name in names}
    versions["python"] = sys.version.split()[0]
    return versions


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _weight_paths(model_dir: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    return sorted(path for path in Path(model_dir).rglob("*")
                  if path.is_file() and path.suffix in suffixes and
                  ("model" in path.name or path.suffix in {".pt", ".pth"}))


def weight_file_hashes(model_dir: Path) -> dict[str, str]:
    files = _weight_paths(model_dir)
    if not files:
        raise ValueError(f"no model weight files found under {model_dir}")
    return {str(path.relative_to(model_dir)): sha256_file(path) for path in files}


def weight_fingerprint(file_hashes: dict[str, str]) -> str:
    payload = json.dumps(sorted(file_hashes.items()), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def rounded_parameter_fingerprint(model: Any) -> str:
    """Fingerprint bf16-rounded fp32 parameters in sorted parameter-name order."""
    import torch
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        tensor = parameter.detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16).to(torch.float32)
        digest.update(tensor.contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def round_model_parameters_bf16(model: Any) -> None:
    import torch
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.copy_(parameter.to(torch.bfloat16).to(torch.float32))


def host_identity() -> dict[str, str]:
    cpu = ""
    try:
        result = subprocess.run(["lscpu"], check=True, capture_output=True, text=True, timeout=10)
        for line in result.stdout.splitlines():
            if line.lower().startswith(("model name:", "model:")):
                cpu = line.split(":", 1)[1].strip()
                if cpu:
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    if not cpu:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
        except OSError:
            cpu = "unknown"
    hostname = socket.gethostname()
    return {"cpu_model": cpu or "unknown", "hostname_sha256": hashlib.sha256(hostname.encode()).hexdigest()}


def attention_implementation(model: Any) -> str | None:
    return (getattr(getattr(model, "config", None), "_attn_implementation", None) or
            getattr(getattr(model, "config", None), "attn_implementation", None) or "default")


def pile_set_breakdown(corpus: Path, limit: int = N_DOCS) -> dict[str, int]:
    import pyarrow.parquet as pq
    table = pq.read_table(corpus, columns=["meta"]).slice(0, limit)
    counts: dict[str, int] = {}
    for raw in table.column("meta").to_pylist():
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        meta = json.loads(raw) if isinstance(raw, str) else raw
        name = str(meta.get("pile_set_name", "unknown")) if isinstance(meta, dict) else "unknown"
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items()))


def run_gate1(model: Any, token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
              layers: Sequence[int], width: int = 8192, seed: int = GATE_SEED,
              sample_size: int = 32) -> dict:
    """Recompute seeded positive top-1 neuron activations one document at a time."""
    import torch

    docs_by_ordinal = {i: [int(t) for t in ids] for i, ids in enumerate(token_ids)}
    candidates: list[tuple[int, int, int]] = []  # row slot, actual layer, unit
    for slot, layer in enumerate(layers):
        for unit in range(min(width, state["top_act"].shape[1])):
            act = float(state["top_act"][slot, unit, 0])
            doc = int(state["top_doc"][slot, unit, 0])
            pos = int(state["top_pos"][slot, unit, 0])
            if np.isfinite(act) and act > 0 and doc >= 0 and pos >= 0:
                candidates.append((slot, int(layer), unit))
    if len(candidates) < sample_size:
        return {"seed": seed, "requested_pairs": sample_size, "candidate_pairs": len(candidates),
                "verdict": "failed", "reason": f"only {len(candidates)} positive top-1 neuron pairs; "
                f"need {sample_size}", "pairs": []}

    selected_indices = np.random.default_rng(seed).choice(len(candidates), size=sample_size, replace=False)
    decoder_layers = model.model.layers
    rows = []
    for selected in selected_indices:
        slot, layer, unit = candidates[int(selected)]
        doc = int(state["top_doc"][slot, unit, 0])
        pos = int(state["top_pos"][slot, unit, 0])
        stored = float(state["top_act"][slot, unit, 0])
        ids_list = docs_by_ordinal.get(doc)
        if not ids_list or pos >= len(ids_list):
            rows.append({"layer": layer, "unit": unit, "doc": doc, "position": pos,
                         "collector_value": stored, "forward_value": None,
                         "relative_difference": None, "pass": False,
                         "reason": "top-1 document or position is absent from tokenized corpus"})
            continue
        captured: list[Any] = []

        def read_down_proj_input(_module, args):
            captured.append(args[0])

        handle = decoder_layers[layer].mlp.down_proj.register_forward_pre_hook(read_down_proj_input)
        try:
            input_ids = torch.tensor([ids_list], dtype=torch.long)
            mask = torch.ones_like(input_ids)
            with torch.inference_mode():
                model.model(input_ids=input_ids, attention_mask=mask, use_cache=False)
        finally:
            handle.remove()
        if not captured:
            rows.append({"layer": layer, "unit": unit, "doc": doc, "position": pos,
                         "collector_value": stored, "forward_value": None,
                         "relative_difference": None, "pass": False,
                         "reason": "forward pre-hook did not run"})
            continue
        value = float(captured[0][0, pos, unit].item())
        rel = abs(value - stored) / max(abs(stored), 1e-6)
        rows.append({"layer": layer, "unit": unit, "doc": doc, "position": pos,
                     "collector_value": stored, "forward_value": value,
                     "relative_difference": rel, "pass": bool(rel <= 1e-4)})
    return {"seed": seed, "requested_pairs": sample_size, "candidate_pairs": len(candidates),
            "tolerance": 1e-4, "verdict": "passed" if all(r["pass"] for r in rows) else "failed",
            "pairs": rows}


class _GateStop(Exception):
    pass


def _document_activations(model: Any, ids: Sequence[int], layers: Sequence[int],
                          controls: np.ndarray) -> list[np.ndarray]:
    """One unpadded forward; return [T, neurons+dense+sparse] per collected layer."""
    import torch
    if not ids:
        raise ValueError("cannot gate an empty document")
    base = model.model
    decoder_layers = base.layers
    captured: dict[int, np.ndarray] = {}
    handles = []
    for slot, li in enumerate(layers):
        def hook(_module, args, slot=slot):
            captured[slot] = args[0][0].detach().float().cpu().numpy()
        handles.append(decoder_layers[li].mlp.down_proj.register_forward_pre_hook(hook))
    def stop(*_args):
        raise _GateStop()
    handles.append(decoder_layers[max(layers)].mlp.down_proj.register_forward_hook(stop))
    try:
        input_ids = torch.tensor([list(ids)], dtype=torch.long)
        with torch.inference_mode():
            base(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    except _GateStop:
        pass
    finally:
        for handle in handles:
            handle.remove()
    if len(captured) != len(layers):
        raise RuntimeError("single-document gate forward did not reach every requested layer")
    return [np.concatenate((captured[slot], captured[slot] @ controls[slot].T), axis=1)
            if controls.shape[1] else captured[slot] for slot in range(len(layers))]


def _close_gate_value(stored: float, actual: float) -> tuple[bool, float, float]:
    absolute = abs(float(actual) - float(stored))
    relative = absolute / max(abs(float(stored)), 1e-12)
    passed = absolute <= 1e-6 if abs(float(stored)) < 1e-2 else relative <= 1e-4
    return passed, relative, absolute


def _short_docs(token_ids: Sequence[Sequence[int]], chunk_size: int = 100,
                batch_size: int = 4) -> set[int]:
    short: set[int] = set()
    for start in range(0, len(token_ids), chunk_size):
        live = [(i, len(token_ids[i])) for i in range(start, min(start + chunk_size, len(token_ids)))
                if token_ids[i]]
        ordered = sorted(live, key=lambda item: (item[1], item[0]))
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            if not batch:
                continue
            longest = max(length for _, length in batch)
            short.update(doc for doc, length in batch if length < longest)
    return short


def _gate1a(model: Any, token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
            layers: Sequence[int], width: int, controls: np.ndarray, seed: int = GATE_SEED,
            required_short_entries: int = 32) -> dict:
    rng = np.random.default_rng(seed)
    short_docs = _short_docs(token_ids)
    top_act, top_doc, top_pos = state["top_act"], state["top_doc"], state["top_pos"]
    ranks = (0, 9, 10, 19)
    unit_count = top_act.shape[1]
    rows: list[dict] = []
    for slot, layer in enumerate(layers):
        populations = (("neuron", np.arange(width)),
                       ("control", np.arange(width, unit_count)))
        for _kind, population in populations:
            if not len(population):
                continue
            eligible = [int(u) for u in population if all(
                top_doc[slot, u, rank] >= 0 and top_pos[slot, u, rank] >= 0 and
                np.isfinite(top_act[slot, u, rank]) for rank in ranks)]
            if len(eligible) < 2:
                continue
            # Two sampled units cover all four registered ranks, two ranks each.
            rank_pairs = (ranks[:2], ranks[2:])
            short_counts = {
                unit: tuple(sum(int(top_doc[slot, unit, rank]) in short_docs for rank in pair)
                            for pair in rank_pairs)
                for unit in eligible
            }
            # Assign one distinct unit to each two-rank block. The assignment maximizes
            # the number of short-document entries in this layer/population, so the
            # global 32-entry quota is met whenever the eligible entries make that
            # possible. Seeded tie breaks keep the sample reproducible.
            ranked = [sorted(eligible, key=lambda unit, pair=pair_index:
                             (-short_counts[unit][pair], float(rng.random())))
                      for pair_index in range(2)]
            assignments = [(first, second) for first in ranked[0][:2] for second in ranked[1][:2]
                           if first != second]
            if not assignments:
                continue
            best = max(short_counts[first][0] + short_counts[second][1]
                       for first, second in assignments)
            best_assignments = [(first, second) for first, second in assignments
                                if short_counts[first][0] + short_counts[second][1] == best]
            first, second = best_assignments[int(rng.integers(len(best_assignments)))]
            for unit, rank_pair in ((first, rank_pairs[0]), (second, rank_pairs[1])):
                for rank in rank_pair:
                    doc = int(top_doc[slot, unit, rank])
                    pos = int(top_pos[slot, unit, rank])
                    stored = float(top_act[slot, unit, rank])
                    try:
                        activations = _document_activations(model, token_ids[doc], layers, controls)
                        actual = float(activations[slot][pos, unit])
                        passed, relative, absolute = _close_gate_value(stored, actual)
                        reason = None
                    except Exception as exc:
                        actual, relative, absolute, passed = None, None, None, False
                        reason = f"forward failed: {type(exc).__name__}: {exc}"
                    rows.append({"layer": int(layer), "unit": int(unit), "kind": _kind,
                                 "rank": rank + 1, "doc": doc, "position": pos,
                                 "stored": stored, "forward": actual, "relative_difference": relative,
                                 "absolute_difference": absolute, "shorter_than_batch_max": doc in short_docs,
                                 "pass": bool(passed), **({"reason": reason} if reason else {})})
    short_count = sum(row["shorter_than_batch_max"] for row in rows)
    errors = sum(not row["pass"] for row in rows)
    quota_met = short_count >= required_short_entries
    return {"verdict": "passed" if not errors and quota_met and len(rows) == 128 else "failed",
            "requested_entries": 128, "entries_checked": len(rows),
            "shorter_than_batch_max_entries": short_count,
            "required_shorter_than_batch_max_entries": required_short_entries,
            "short_document_quota_met": quota_met, "failed_entries": errors,
            "tolerance": {"relative": 1e-4, "absolute_below_1e-2": 1e-6}, "entries": rows}


def _gate1b(model: Any, token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
            layers: Sequence[int], width: int, controls: np.ndarray,
            threshold_fraction: float = 0.001) -> dict:
    docs = min(100, len(token_ids))
    live_docs = [doc for doc in range(docs) if token_ids[doc]]
    if len(live_docs) < 20:
        return {"verdict": "failed", "documents": docs, "nonempty_documents": len(live_docs), "units_checked": 0,
                "failed_units": 0, "failure_fraction": 1.0, "reason": "fewer than 20 documents"}
    per_doc: dict[int, list[np.ndarray]] = {}
    for doc in live_docs:
        activations = _document_activations(model, token_ids[doc], layers, controls)
        per_doc[doc] = [activation.max(axis=0) for activation in activations]
    failures = []
    units_checked = 0
    for slot, layer in enumerate(layers):
        for unit in range(state["top_act"].shape[1]):
            stored_values = state["top_act"][slot, unit]
            stored_docs = state["top_doc"][slot, unit]
            valid = (stored_docs >= 0) & np.isfinite(stored_values)
            if int(valid.sum()) < 20:
                continue
            threshold = float(stored_values[19])
            # Batch-1 recomputation differs from the batched run by float noise, so a document within a tolerance band of
            # the 20th value counts either way (design: "ties counted as either"); without the band the 20th entry's own
            # document flips sides on noise (host test, 2026-11-03).
            band = max(abs(threshold) * 1e-4, 1e-6)
            stored_near = {int(doc) for doc, act in zip(stored_docs[valid], stored_values[valid])
                           if int(doc) < docs and abs(float(act) - threshold) <= band}
            recomputed_near = {doc for doc, values in per_doc.items()
                               if abs(float(values[slot][unit]) - threshold) <= band}
            ambiguous = stored_near | recomputed_near
            main = {int(doc) for doc, act in zip(stored_docs[valid], stored_values[valid])
                    if int(doc) < docs and float(act) > threshold and int(doc) not in ambiguous}
            subset = {doc for doc, values in per_doc.items()
                      if float(values[slot][unit]) > threshold and doc not in ambiguous}
            units_checked += 1
            if main != subset:
                failures.append({"layer": int(layer), "unit": int(unit),
                                 "main_strictly_above": len(main), "subset_strictly_above": len(subset),
                                 "missing_from_main": sorted(subset - main)[:20],
                                 "missing_from_subset": sorted(main - subset)[:20]})
    frac = len(failures) / units_checked if units_checked else 1.0
    return {"verdict": "passed" if units_checked and frac <= threshold_fraction else "failed",
            "documents": docs, "nonempty_documents": len(live_docs), "batch_size": 1,
            "padding": False, "units_checked": units_checked,
            "failed_units": len(failures), "failure_fraction": frac,
            "maximum_failure_fraction": threshold_fraction, "tie_rule": "documents within max(1e-4 relative, 1e-6) of the main 20th value, stored or recomputed, count either way",
            "failures": failures}


def _gate1c(token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
            max_docs: int = N_DOCS) -> dict:
    bad: list[dict] = []
    acts, docs, poss = state["top_act"], state["top_doc"], state["top_pos"]
    for slot in range(docs.shape[0]):
        for unit in range(docs.shape[1]):
            if not np.isfinite(state["max"][slot, unit]):
                bad.append({"layer_slot": slot, "unit": unit, "reason": "nonfinite unit maximum"})
            valid_doc = docs[slot, unit] >= 0
            valid_act = np.isfinite(acts[slot, unit])
            if np.any(valid_doc != valid_act):
                bad.append({"layer_slot": slot, "unit": unit, "reason": "document/value placeholders disagree"})
                continue
            indices = np.flatnonzero(valid_doc)
            unit_docs = docs[slot, unit, indices].astype(int).tolist()
            reasons = []
            if not np.all(np.isfinite(acts[slot, unit, indices])):
                reasons.append("nonfinite stored activation")
            if len(unit_docs) != len(set(unit_docs)):
                reasons.append("duplicate document id")
            if any(doc < 0 or doc >= max_docs or doc >= len(token_ids) for doc in unit_docs):
                reasons.append("document id outside corpus")
            if any(int(poss[slot, unit, i]) < 0 or int(poss[slot, unit, i]) >= len(token_ids[int(docs[slot, unit, i])])
                   for i in indices if 0 <= int(docs[slot, unit, i]) < len(token_ids)):
                reasons.append("position outside document")
            values = acts[slot, unit, indices]
            if len(values) > 1 and np.any(values[:-1] < values[1:]):
                reasons.append("top list is not sorted descending")
            if reasons:
                bad.append({"layer_slot": slot, "unit": unit, "reasons": reasons})
    return {"verdict": "passed" if not bad else "failed", "units_checked": int(docs.shape[0] * docs.shape[1]),
            "invalid_units": len(bad), "failures": bad}


def _gate1d(model: Any, token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
            layers: Sequence[int], width: int, controls: np.ndarray,
            seed: int = GATE_SEED) -> dict:
    rng = np.random.default_rng(seed)
    units_total = state["top_act"].shape[1]
    nonempty_docs = [index for index, ids in enumerate(token_ids) if ids]
    rows = []
    for slot, layer in enumerate(layers):
        for population in (np.arange(width), np.arange(width, units_total)):
            if not len(population):
                continue
            selected = rng.choice(population, size=min(2, len(population)), replace=False)
            for unit_value in selected:
                unit = int(unit_value)
                if not nonempty_docs:
                    continue
                doc = int(nonempty_docs[int(rng.integers(len(nonempty_docs)))])
                try:
                    activations = _document_activations(model, token_ids[doc], layers, controls)
                    actual = float(activations[slot][:, unit].max())
                    listed = np.flatnonzero(state["top_doc"][slot, unit] == doc)
                    if len(listed):
                        stored = float(state["top_act"][slot, unit, int(listed[0])])
                        passed, rel, absolute = _close_gate_value(stored, actual)
                        rule = "equals_stored_entry"
                    else:
                        stored = float(state["top_act"][slot, unit, 19])
                        rel, absolute = None, None
                        passed = actual <= stored or math.isclose(actual, stored, rel_tol=1e-6, abs_tol=1e-7)
                        rule = "at_most_twentieth_value"
                    reason = None
                except Exception as exc:
                    actual, stored, rel, absolute, passed = None, None, None, None, False
                    rule, reason = "forward_error", f"{type(exc).__name__}: {exc}"
                rows.append({"layer": int(layer), "unit": unit, "doc": doc, "observed_max": actual,
                             "expected": stored, "rule": rule, "relative_difference": rel,
                             "absolute_difference": absolute, "pass": bool(passed),
                             **({"reason": reason} if reason else {})})
    return {"verdict": "passed" if rows and all(row["pass"] for row in rows) else "failed",
            "seed": seed, "entries_checked": len(rows), "failed_entries": sum(not r["pass"] for r in rows),
            "entries": rows}


def run_gates(model: Any, token_ids: Sequence[Sequence[int]], state: dict[str, np.ndarray],
              layers: Sequence[int], width: int, controls: np.ndarray,
              checkpoint_every: int = 100, batch_size: int = 4) -> dict:
    """Run Gates 1a–1d against the completed collector state."""
    gate = {
        "seed": GATE_SEED,
        "checks": {
            "1a": _gate1a(model, token_ids, state, layers, width, controls),
            "1b": _gate1b(model, token_ids, state, layers, width, controls),
            "1c": _gate1c(token_ids, state, max_docs=N_DOCS),
            "1d": _gate1d(model, token_ids, state, layers, width, controls),
        },
        "settings": {"checkpoint_every": checkpoint_every, "batch_size": batch_size,
                     "documents": len(token_ids)},
    }
    gate["verdict"] = "passed" if all(row["verdict"] == "passed" for row in gate["checks"].values()) else "failed"
    gate["pairs"] = gate["checks"]["1a"].get("entries", [])
    return gate


def _manifest_binding_sha(manifest: dict) -> str:
    """sha256 of a manifest without its host and its own hashes. 2026-11-03: cloud containers restart onto
    machines with a different CPU model; the first resume check compared the whole manifest, host included, and
    refused three runs' resumes (instruct, midtrain_51B, pretrain_4T_bf16r)."""
    core = {k: v for k, v in manifest.items() if k not in ("host", "manifest_sha256")}
    core = json.loads(json.dumps(core))
    # The interpreter can change with the machine too (3.11 on the first runs, 3.13 after the routine named one
    # interpreter); the pinned libraries in library_versions still must match exactly.
    if isinstance(core.get("library_versions"), dict):
        core["library_versions"].pop("python", None)
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def bind_identity(out: Path, manifest: dict) -> str:
    """Write identity.json and manifest.json on a fresh run, or check a resume against them. Returns the manifest
    sha the collector's state is bound to: the first run's on a resume, so its state stays valid. A resume on a
    different host is allowed, and each host change is appended to host-history.jsonl; anything else that
    differs refuses the resume."""
    identity_path = out / "identity.json"
    if identity_path.is_file():
        previous = json.loads(identity_path.read_text(encoding="utf-8"))
        identity_fields = ("key", "model_commit", "dataset_commit", "token_ids_sha256",
                           "controls_sha256_by_layer")
        changed = [field for field in identity_fields if previous.get(field) != manifest.get(field)]
        if _manifest_binding_sha(previous) != _manifest_binding_sha(manifest):
            keys = sorted(k for k in set(previous) | set(manifest)
                          if k not in ("host", "manifest_sha256")
                          and json.dumps(previous.get(k), sort_keys=True) != json.dumps(manifest.get(k), sort_keys=True))
            changed.append("manifest (host excluded; differing keys: " + ", ".join(keys) + ")")
        if changed:
            raise ValueError("existing output identity differs for: " + ", ".join(changed)
                             + "; use a fresh --out directory")
        def _python(m):
            versions = m.get("library_versions")
            return versions.get("python") if isinstance(versions, dict) else None
        if previous.get("host") != manifest.get("host") or _python(previous) != _python(manifest):
            with (out / "host-history.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "from": previous.get("host"), "to": manifest.get("host"),
                                         "python_from": _python(previous), "python_to": _python(manifest)},
                                        sort_keys=True) + "\n")
        return previous["manifest_sha256"]
    if (out / "progress.json").exists() or (out / "state.npz").exists():
        raise ValueError("collector state exists without identity.json; refusing an unverifiable resume")
    identity_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest["manifest_sha256"]


def compress_state(state_path: Path) -> bool:
    """Replace an uncompressed NPZ atomically; return whether it was compressed."""
    state_path = Path(state_path)
    with zipfile.ZipFile(state_path, "r") as archive:
        already_compressed = all(item.compress_type != zipfile.ZIP_STORED for item in archive.infolist())
    if already_compressed:
        return False
    tmp = state_path.with_name(state_path.name + ".compressed.tmp")
    with np.load(state_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, state_path)
    return True


class SessionLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"{stamp} {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")


@contextlib.contextmanager
def timed_stage(log: SessionLog, timings: dict[str, float], name: str):
    start = time.perf_counter()
    log.write(f"{name}: start")
    try:
        yield
    except Exception as exc:
        timings[name] = time.perf_counter() - start
        log.write(f"{name}: FAILED after {timings[name]:.2f}s: {type(exc).__name__}: {exc}")
        raise RuntimeError(f"{name} failed: {exc}") from exc
    else:
        timings[name] = time.perf_counter() - start
        log.write(f"{name}: complete in {timings[name]:.2f}s")


def _output_inventory(out: Path) -> list[dict]:
    inventory = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "summary.json"):
        inventory.append({"path": str(path.relative_to(out)), "size_bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    return inventory


def run_session(key: str, work: Path, out: Path, *, hf_api: Any = None,
                snapshot_download_fn: Callable[..., str] | None = None,
                hf_hub_download_fn: Callable[..., str] | None = None,
                collector_module: Any = None, model_loader: Callable[..., Any] | None = None,
                rerun_of: str | None = None) -> dict:
    """Execute a cloud session. Optional dependencies make the procedure testable offline."""
    versions = verify_pinned_versions()
    allowed = {"pretrain_21B", "pretrain_210B", "pretrain_1T", "pretrain_2T", "pretrain_3T",
               "pretrain_4T", "midtrain_13B", "midtrain_26B", "midtrain_51B", "sft", "dpo",
               "instruct", "instruct_b", "pretrain_1T_b", "pretrain_4T_bf16r"}
    if key not in allowed:
        raise ValueError(f"unknown Part 181 session key {key!r}")
    work, out = Path(work), Path(out)
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    log = SessionLog(out / "session.log")
    timings: dict[str, float] = {}
    all_started = time.perf_counter()

    try:
        with timed_stage(log, timings, "initialize_dependencies"):
            if collector_module is None:
                from part181 import olmo_atlas_collect as collector
            else:
                collector = collector_module
            if hf_api is None or snapshot_download_fn is None or hf_hub_download_fn is None:
                from huggingface_hub import HfApi, hf_hub_download, snapshot_download
                hf_api = hf_api or HfApi()
                snapshot_download_fn = snapshot_download_fn or snapshot_download
                hf_hub_download_fn = hf_hub_download_fn or hf_hub_download
            model_loader = model_loader or collector.load_model
        with timed_stage(log, timings, "resolve_revisions"):
            checkpoint, dataset_commit = resolve_inputs(key, hf_api)
            model_commit = checkpoint["commit"]
            log.write(f"model commit: {model_commit}")
            log.write(f"dataset commit: {dataset_commit}")
        with timed_stage(log, timings, "download_model"):
            model_dir = download_model_snapshot(checkpoint, work, snapshot_download_fn)
            model_weight_hashes = weight_file_hashes(model_dir)
        with timed_stage(log, timings, "download_dataset"):
            corpus = download_corpus(dataset_commit, work, hf_hub_download_fn)
            corpus_sha256 = sha256_file(corpus)
        with timed_stage(log, timings, "load_model"):
            tok, model = model_loader(model_dir, "fp32", 4)
            if key == "pretrain_4T_bf16r":
                round_model_parameters_bf16(model)
                model_weight_fingerprint = rounded_parameter_fingerprint(model)
            else:
                model_weight_fingerprint = weight_fingerprint(model_weight_hashes)
        with timed_stage(log, timings, "identity"):
            texts = collector.read_corpus_texts(corpus, N_DOCS)
            if len(texts) != N_DOCS:
                raise ValueError(f"corpus contains {len(texts)} documents; expected at least {N_DOCS}")
            tokenized = collector.tokenize_docs(tok, texts, start=0)
            token_ids = [doc["ids"] for doc in tokenized]
            token_sha = token_ids_sha256(token_ids)
            token_count = sum(map(len, token_ids))
            capped_docs = sum(len(row) == collector.MAX_DOC_TOKENS for row in token_ids)
            if token_sha != REGISTERED_TOKEN_SHA256:
                raise ValueError(f"token ids sha256 {token_sha} does not match registered {REGISTERED_TOKEN_SHA256}")
            if token_count != REGISTERED_TOKEN_COUNT or capped_docs != REGISTERED_CAPPED_DOCS:
                raise ValueError(f"token corpus counts differ from registration: tokens={token_count}, "
                                 f"capped_docs={capped_docs}")
            controls = control_hashes(collector)
            if controls["dense_sha256"] != REGISTERED_DENSE_CONTROL_SHA256:
                raise ValueError("dense control matrix sha256 differs from the registered control hash")
            host = host_identity()
            control_rows = []
            for layer in range(16):
                control_rows.extend({"layer": layer, "control_id": i, "kind": "dense", "k": 0}
                                    for i in range(256))
                control_rows.extend({"layer": layer, "control_id": 256 + j * 64 + i,
                                     "kind": "sparse", "k": int(k)}
                                    for j, k in enumerate(collector.SPARSE_SUPPORTS) for i in range(64))
            manifest = {
                "key": key,
                "checkpoint_key": checkpoint["key"],
                "repo_id": checkpoint["repo_id"],
                "model_commit": model_commit,
                "model_commit_prefix": checkpoint["commit_prefix"],
                "dataset_repo_id": DATASET_ID,
                "dataset_file": DATASET_FILE,
                "dataset_commit": dataset_commit,
                "dataset_commit_prefix": DATASET_REVISION_PREFIX,
                "documents": len(token_ids),
                "token_count": token_count,
                "documents_at_token_cap": capped_docs,
                "corpus_file_sha256": corpus_sha256,
                "token_ids_sha256": token_sha,
                "controls": controls,
                "control_rows": control_rows,
                "controls_sha256_by_layer": controls["dense_by_layer_sha256"],
                "control_seed": SEED,
                "weight_file_sha256": model_weight_hashes,
                "weight_fingerprint": model_weight_fingerprint,
                "library_versions": versions,
                "host": host,
                "dtype": "fp32",
                "attention_implementation": attention_implementation(model),
                "settings": {"batch_size": 4, "checkpoint_every": 100, "docs": N_DOCS,
                             "pass2_docs": PASS2_DOCS, "passes": 2, "threads": 4,
                             "n_dense_controls": 256, "n_sparse_per_k": 64},
                "pile_set_name_counts": pile_set_breakdown(corpus, N_DOCS),
                "rerun_of": rerun_of,
            }
            manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                                          .encode("utf-8")).hexdigest()
            manifest["manifest_sha256"] = manifest_sha
            manifest_sha = bind_identity(out, manifest)
        with timed_stage(log, timings, "collect"):
            snapshot_every = 500 if key == "instruct" else 0
            collect_args = SimpleNamespace(
                model_dir=str(model_dir), corpus=str(corpus), out=str(out), docs=N_DOCS,
                pass2_docs=PASS2_DOCS, passes=2, dtype="fp32", batch_size=4,
                checkpoint_every=100, threads=4, seed=SEED, n_controls=256, n_sparse_per_k=64,
                manifest_sha256=manifest_sha, layers=None,
                model_arm=key, forward_only=False, snapshot_every=snapshot_every,
                stop_after_docs=None, strict_shape=True,
            )
            progress = collector.collect(collect_args, tok=tok, model=model)
            if not progress.get("complete"):
                raise RuntimeError("collector returned before completing both passes")
        with timed_stage(log, timings, "gate1"):
            with np.load(out / "state.npz", allow_pickle=False) as saved:
                state = {name: saved[name] for name in saved.files}
            config = progress["config"]
            control_matrices = []
            for layer in config["layers"]:
                pieces = [collector.generate_controls(256, config["width"], SEED, layer)]
                pieces.extend(collector.generate_sparse_controls(64, config["width"], k, SEED, layer)
                              for k in collector.SPARSE_SUPPORTS)
                control_matrices.append(np.concatenate(pieces, axis=0))
            gate = run_gates(model, token_ids, state, config["layers"], width=config["width"],
                             controls=np.stack(control_matrices), checkpoint_every=100, batch_size=4)
            (out / "gate1.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
            log.write(f"Gate 1 verdict: {gate['verdict']} ({sum(row['pass'] for row in gate['pairs'])}/"
                      f"{len(gate['pairs'])} pairs)")
        with timed_stage(log, timings, "compress_state"):
            compressed_now = compress_state(out / "state.npz")
        timings["total_before_summary"] = time.perf_counter() - all_started
        log.write(f"summary: preparing; state compressed now={compressed_now}")
        summary = {
            "key": key,
            "checkpoint_key": checkpoint["key"],
            "model_commit": model_commit,
            "dataset_commit": dataset_commit,
            "manifest_sha256": manifest_sha,
            "weight_fingerprint": model_weight_fingerprint,
            "rerun_of": rerun_of,
            "timings_seconds": timings,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "gate1": {"verdict": gate["verdict"],
                      "checks": {name: {"verdict": check["verdict"]}
                                 for name, check in gate["checks"].items()}},
            "state_compressed_now": compressed_now,
            "files": _output_inventory(out),
            "file_inventory_note": "summary.json is omitted because a file cannot contain its own SHA-256",
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                                          encoding="utf-8")
        return summary
    except Exception as exc:
        log.write(f"SESSION FAILED: {type(exc).__name__}: {exc}")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True, help="registered checkpoint key or instruct_b")
    parser.add_argument("--work", required=True, type=Path, help="scratch/cache directory")
    parser.add_argument("--out", required=True, type=Path, help="result directory")
    parser.add_argument("--rerun-of", default=None,
                        help="record this as the single rerun of an earlier failed run")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_session(args.key, args.work, args.out, rerun_of=args.rerun_of)
    except Exception as exc:
        print(f"session181 failed: {exc}", file=sys.stderr, flush=True)
        return 1
    if summary["gate1"]["verdict"] != "passed":
        print("session181 Gate 1 failed; this run must be rerun", file=sys.stderr, flush=True)
        return 2
    print("session181 completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
