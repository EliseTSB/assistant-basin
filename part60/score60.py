#!/usr/bin/env python3
"""Score Part 60 generated sentences with the pinned Qwen3-1.7B MLP hook."""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:  # Windows: only the POSIX CLI locks below use it; scoring does not (Part 64 Amendment 7, 2026-10-24)
    fcntl = None
import hashlib
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

HERE = Path(__file__).resolve().parent
BASIN = Path(os.environ.get("PART60_BASIN", HERE.parent)).resolve()
DEFAULT_ROOT = BASIN / "results" / "part60"
DATA = BASIN / "results" / "part58"
MODEL_REPO = "Qwen/Qwen3-1.7B"
MODEL_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
POOL_SEED = 7
POOL_SIZE = 1200
REPRO_SEED = 6002
REPRO_N = 200
SCORE_BACKEND = "qwen-down-proj-hook-v1"
PHASE_SCORE_STEMS = {
    ("M3", "selection"): "M3-selection",
    ("M3", "fresh"): "M3",
    ("M5", "round1"): "M5-round1",
    ("M5", "round2"): "M5",
}


class ScoreError(RuntimeError):
    pass


def check_data_paths() -> None:
    required = (DATA / "acts.all.npz", DATA / "ids.all.json", DATA / "units.all.json", DATA / "sent.all.jsonl")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise ScoreError(
            "Part 58 scoring data were not found relative to this checkout. "
            f"Stage this build at {HERE} with results/ beside it, or set PART60_BASIN "
            "to the directory containing results/. Missing: " + ", ".join(str(p) for p in missing)
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ScoreError(f"{path}:{line_no} is not an object")
            out.append(row)
    return out


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def clear_score_artifacts(root: Path, split: str, method: str, phase: str) -> None:
    score_path(root, split, method, phase).unlink(missing_ok=True)
    for progress in _progress_paths(root, split, method, phase):
        progress.unlink(missing_ok=True)


def unit_key_from_tuple(unit: list[Any]) -> str:
    if len(unit) == 2:
        return f"L{int(unit[0])}:u{int(unit[1])}"
    if len(unit) == 3 and unit[1] == "c":
        return f"L{int(unit[0])}:c{int(unit[2])}"
    raise ScoreError(f"unrecognized Part 58 unit tuple {unit!r}")


def _owner(sid: str) -> int:
    try:
        return int(sid.split(":", 1)[0])
    except (ValueError, IndexError):
        raise ScoreError(f"cannot read source unit index from sentence id {sid!r}") from None


def compute_thresholds(activations: np.ndarray, sids: list[str], record_indices: list[int],
                       unit_keys: list[str], pool_seed: int = POOL_SEED,
                       pool_size: int = POOL_SIZE) -> dict[str, float]:
    """p95 of the fixed Part 58 pool, excluding rows authored for the target unit."""
    if activations.ndim != 2 or activations.shape != (len(sids), len(record_indices)):
        raise ScoreError("activation matrix does not align with sentence ids and unit columns")
    if len(unit_keys) != len(record_indices):
        raise ScoreError("unit key list does not align with activation columns")
    if len(sids) < pool_size:
        raise ScoreError(f"source sentence pool has {len(sids)} rows; expected at least {pool_size}")
    rng = random.Random(pool_seed)
    pool_rows = np.asarray(sorted(rng.sample(range(len(sids)), pool_size)), dtype=np.int64)
    owners = np.asarray([_owner(sids[i]) for i in pool_rows], dtype=np.int64)
    values = activations[np.ix_(pool_rows, np.arange(len(unit_keys), dtype=np.int64))]
    result: dict[str, float] = {}
    for col, (record_index, key) in enumerate(zip(record_indices, unit_keys)):
        keep = owners != int(record_index)
        if not keep.any():
            raise ScoreError(f"fixed null pool has no other-unit texts for {key}")
        result[key] = float(np.percentile(values[keep, col], 95, method="linear"))
    return result


def compute_observed_ceilings(activations: np.ndarray, sids: list[str], record_indices: list[int],
                              unit_keys: list[str], pool_seed: int = POOL_SEED,
                              pool_size: int = POOL_SIZE) -> dict[str, float]:
    """Return each unit's maximum on the same fixed null pool used for its threshold."""
    if activations.ndim != 2 or activations.shape != (len(sids), len(record_indices)):
        raise ScoreError("activation matrix does not align with sentence ids and unit columns")
    rng = random.Random(pool_seed)
    pool_rows = np.asarray(sorted(rng.sample(range(len(sids)), pool_size)), dtype=np.int64)
    owners = np.asarray([_owner(sids[i]) for i in pool_rows], dtype=np.int64)
    values = activations[np.ix_(pool_rows, np.arange(len(unit_keys), dtype=np.int64))]
    result = {}
    for col, (record_index, key) in enumerate(zip(record_indices, unit_keys)):
        keep = owners != int(record_index)
        if not keep.any():
            raise ScoreError(f"fixed null pool has no other-unit texts for {key}")
        result[key] = float(np.max(values[keep, col]))
    return result


def load_source_arrays() -> tuple[np.ndarray, list[str], list[int], list[str]]:
    acts_path = DATA / "acts.all.npz"
    ids_path = DATA / "ids.all.json"
    units_path = DATA / "units.all.json"
    sent_path = DATA / "sent.all.jsonl"
    for path in (acts_path, ids_path, units_path, sent_path):
        if not path.is_file():
            raise ScoreError(f"required Part 58 source is missing: {path}")
    ids = [int(x) for x in json.loads(ids_path.read_text(encoding="utf-8"))]
    tuples = json.loads(units_path.read_text(encoding="utf-8"))
    keys = [unit_key_from_tuple(x) for x in tuples]
    if len(ids) != len(keys):
        raise ScoreError("ids.all.json and units.all.json column counts differ")
    sentence_rows = read_jsonl(sent_path)
    sids = [str(x["sid"]) for x in sentence_rows]
    with np.load(acts_path, allow_pickle=False) as archive:
        matrix = archive["max"]
        archive_sids = [str(x) for x in archive["sids"]]
        archive_units = [unit_key_from_tuple(json.loads(str(x))) for x in archive["units"]]
    if archive_sids != sids:
        raise ScoreError("acts.all.npz sentence rows do not match sent.all.jsonl order")
    if archive_units != keys:
        raise ScoreError("acts.all.npz unit columns do not match units.all.json order")
    if matrix.shape != (len(sids), len(keys)):
        raise ScoreError(f"acts.all matrix shape {matrix.shape} does not match {len(sids)} sentences and {len(keys)} units")
    return matrix, sids, ids, keys


def threshold_fingerprint() -> dict[str, Any]:
    files = [DATA / name for name in ("acts.all.npz", "ids.all.json", "units.all.json", "sent.all.jsonl")]
    # Tar preserves second-resolution mtimes, which keeps local/GPU caches stable across transfer.
    return {path.name: {"bytes": path.stat().st_size, "mtime": int(path.stat().st_mtime)} for path in files}


def ensure_thresholds(root: Path) -> dict[str, float]:
    path = root / "thresholds.json"
    fingerprint = threshold_fingerprint()
    cached: dict[str, Any] = {}
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source_fingerprint") == fingerprint:
            if "observed_ceilings" not in cached:
                matrix, sids, ids, keys = load_source_arrays()
                cached["observed_ceilings"] = compute_observed_ceilings(matrix, sids, ids, keys)
                write_json(path, cached)
            return {str(k): float(v) for k, v in cached["thresholds"].items()}
    matrix, sids, ids, keys = load_source_arrays()
    thresholds = compute_thresholds(matrix, sids, ids, keys)
    ceilings = compute_observed_ceilings(matrix, sids, ids, keys)
    value = {"definition": "linear 95th percentile of fixed seed-7 1,200-row Part 58 pool, excluding target unit's own texts",
             "pool_seed": POOL_SEED, "pool_size": POOL_SIZE,
             "source_fingerprint": fingerprint, "thresholds": thresholds,
             "observed_ceilings": ceilings}
    supplemental = cached.get("supplemental_thresholds", {})
    if isinstance(supplemental, dict) and supplemental:
        value["supplemental_thresholds"] = supplemental
        for key, threshold in supplemental.items():
            value["thresholds"].setdefault(str(key), float(threshold))
        saved_ceilings = cached.get("observed_ceilings", {})
        if isinstance(saved_ceilings, dict):
            value["observed_ceilings"].update({str(key): float(saved_ceilings[key])
                                               for key in supplemental
                                               if key in saved_ceilings})
    for key in ("supplemental_scoring_by_split",):
        if isinstance(cached.get(key), dict):
            value[key] = cached[key]
    write_json(path, value)
    return {str(k): float(v) for k, v in value["thresholds"].items()}


def extend_thresholds_for_split(root: Path, split: str, unit_specs: list[dict[str, Any]],
                                scorer: "QwenScorer", thresholds: dict[str, float]) -> dict[str, float]:
    """Score the fixed pool only for split units absent from Part 58's archived columns."""
    missing = [unit for unit in unit_specs if unit["unit_key"] not in thresholds]
    if not missing:
        return thresholds
    source = read_jsonl(DATA / "sent.all.jsonl")
    rng = random.Random(POOL_SEED)
    pool_rows = sorted(rng.sample(range(len(source)), POOL_SIZE))
    owners = np.asarray([_owner(str(source[i]["sid"])) for i in pool_rows], dtype=np.int64)
    entries = [{"text": str(source[i]["text"]), "marked_span": None} for i in pool_rows]
    observed = np.full((len(entries), len(unit_specs)), np.nan, dtype=np.float32)
    marked = np.full_like(observed, np.nan)
    scorer.score_batch(entries, list(range(len(entries))), observed, marked, batch_size=8)
    col_by_key = {str(unit["unit_key"]): col for col, unit in enumerate(unit_specs)}
    additions: dict[str, float] = {}
    ceilings: dict[str, float] = {}
    for unit in missing:
        keep = owners != int(unit["record_index"])
        if not keep.any():
            raise ScoreError(f"fixed null pool has no other-unit texts for {unit['unit_key']}")
        values = observed[keep, col_by_key[unit["unit_key"]]]
        additions[str(unit["unit_key"])] = float(np.percentile(values, 95, method="linear"))
        ceilings[str(unit["unit_key"])] = float(np.max(values))
    thresholds.update(additions)
    path = root / "thresholds.json"
    cached = json.loads(path.read_text(encoding="utf-8"))
    cached.setdefault("supplemental_thresholds", {}).update(additions)
    cached["thresholds"].update(additions)
    cached.setdefault("observed_ceilings", {}).update(ceilings)
    cached.setdefault("supplemental_scoring_by_split", {})[split] = {"model_revision": MODEL_REVISION, "dtype": scorer.dtype,
        "device": getattr(scorer, "device", "cpu"), "backend": SCORE_BACKEND,
        "pool_seed": POOL_SEED, "pool_size": POOL_SIZE, "reason": "unit has no column in Part 58 acts.all.npz"}
    write_json(path, cached)
    print(f"scored fixed null pool for {len(additions)} split units missing from Part 58 archive")
    return thresholds


def score_path(root: Path, split: str, method: str, phase: str) -> Path:
    name = PHASE_SCORE_STEMS.get((method, phase), method)
    return root / "acts" / split / f"{name}.npz"


def archive_m0_path(root: Path, split: str) -> Path:
    return root / "acts" / split / "M0-archive.npz"


def entries_for_phase(rows: list[dict[str, Any]], method: str, phase: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for row in rows:
        rec = int(row["record_index"])
        key = str(row["unit_key"])
        typ = str(row["type"])
        if method == "M3" and phase == "selection":
            if row.get("selection", {}).get("parse_status") != "ok":
                continue
            for candidate in row.get("candidates", []):
                rank = int(candidate["rank"])
                for tier in ("strong", "none"):
                    for j, text in enumerate(candidate.get(tier, [])):
                        entries.append(_entry(rec, key, typ, method, phase, f"candidate{rank}_{tier}", j, text, rank))
            continue
        if method == "M3" and phase == "fresh":
            value = row.get("fresh", {})
            if value.get("parse_status") != "ok":
                continue
            content = value
        elif method == "M5" and phase in ("round1", "round2"):
            value = row.get(phase, {})
            if value.get("parse_status") != "ok":
                continue
            content = value
        else:
            if row.get("parse_status") != "ok":
                continue
            content = row
        for tier in ("strong", "weak", "none"):
            for j, text in enumerate(content.get(tier, [])):
                entries.append(_entry(rec, key, typ, method, phase, tier, j, text, None))
    return entries


def _entry(rec: int, key: str, typ: str, method: str, phase: str, group: str,
           index: int, text: str, candidate_rank: int | None) -> dict[str, Any]:
    clean, span = strip_markers(str(text))
    sid = f"{rec}|{method}|{phase}|{group}|{index}"
    return {"record_index": rec, "unit_key": key, "unit_type": typ, "method": method,
            "phase": phase, "group": group, "index": index, "candidate_rank": candidate_rank,
            "sid": sid, "text": clean, "marked_span": span}


def strip_markers(text: str) -> tuple[str, tuple[int, int] | None]:
    spans: list[tuple[int, int]] = []
    out: list[str] = []
    cursor = 0
    length = 0
    for match in re.finditer(r"\[\[(.*?)\]\]", text, flags=re.S):
        prefix = text[cursor:match.start()]
        marked = match.group(1)
        out.extend((prefix, marked))
        start = length + len(prefix)
        end = start + len(marked)
        spans.append((start, end))
        length = end
        cursor = match.end()
    out.append(text[cursor:])
    clean = "".join(out)
    return clean, spans[0] if len(spans) == 1 else None


def score_name(method: str, phase: str) -> str:
    return score_path(Path("."), "", method, phase).name


def _unit_info(split_units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for unit in split_units:
        r = unit["record"]
        result.append({"record_index": int(unit["record_index"]), "unit_key": unit["unit_key"],
                       "type": unit["type"], "layer": int(unit["layer"]),
                       "unit_index": int(r["unit"]) if "unit" in r else None,
                       "control_id": int(r["control_id"]) if "control_id" in r else None})
    return result


def _save_npz_atomic(path: Path, **values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **values)
    tmp.replace(path)


def score_cached_m0(root: Path, split: str, split_units: list[dict[str, Any]]) -> Path:
    import gen60
    rows = gen60.read_jsonl(gen60.output_path(root, split, "M0"))
    items = entries_for_phase(rows, "M0", "primary")
    if not items:
        items = []
    source, source_sids, ids, keys = load_source_arrays()
    source_col_by_record = {record: col for col, record in enumerate(ids)}
    source_row_by_sid = {sid: i for i, sid in enumerate(source_sids)}
    unit_cols = []
    for unit in split_units:
        idx = source_col_by_record.get(int(unit["record_index"]))
        if idx is None or keys[idx] != unit["unit_key"]:
            unit_cols.append(None)
        else:
            unit_cols.append(idx)
    matrix = np.full((len(items), len(split_units)), np.nan, dtype=np.float32)
    sids = []
    texts = []
    groups = []
    recs = []
    unit_keys = [u["unit_key"] for u in split_units]
    for r, item in enumerate(items):
        src_sid = f"{item['record_index']}:B#{item['index']}"
        src_row = source_row_by_sid.get(src_sid)
        if src_row is None:
            sids.append(item["sid"]); texts.append(item["text"]); groups.append(item["group"]); recs.append(item["record_index"])
            continue
        for target_col, source_col in enumerate(unit_cols):
            if source_col is not None:
                matrix[r, target_col] = source[src_row, source_col]
        sids.append(item["sid"]); texts.append(item["text"]); groups.append(item["group"]); recs.append(item["record_index"])
    path = archive_m0_path(root, split)
    _save_npz_atomic(path, max=matrix, marked=np.full_like(matrix, np.nan), sids=np.asarray(sids),
                     unit_keys=np.asarray(unit_keys), record_indices=np.asarray(recs, dtype=np.int32),
                     groups=np.asarray(groups), texts=np.asarray(texts), method=np.asarray("M0"), phase=np.asarray("primary"),
                     unit_types=np.asarray([u["type"] for u in split_units]))
    return path


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def ratio_summary(reference: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    got = np.asarray(observed, dtype=np.float64).ravel()
    finite = np.isfinite(ref) & np.isfinite(got)
    nonzero = finite & (np.abs(ref) > 1e-6)
    ratio = got[nonzero] / ref[nonzero]
    rho = spearman(ref[finite], got[finite])
    return {"n_values": int(finite.sum()), "n_nonzero_reference": int(nonzero.sum()),
            "median_ratio": float(np.median(ratio)) if ratio.size else None,
            "spearman": rho}


def reproduction_passed(by_type: dict[str, Any], all_units: dict[str, Any]) -> bool:
    summaries = [all_units, *(by_type.get(kind) for kind in ("neuron", "control"))]
    return all(isinstance(item, dict) and item.get("median_ratio") is not None
               and 0.9 <= float(item["median_ratio"]) <= 1.1
               and item.get("spearman") is not None and float(item["spearman"]) >= 0.98
               for item in summaries)


def stable_seed(seed: int, *parts: Any) -> int:
    joined = "\0".join((str(seed), *(str(x) for x in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(joined).digest()[:4], "big")


def control_directions(n: int, width: int, layer: int) -> np.ndarray:
    """Recreate Part 58 control directions with its seed and layer-specific stream."""
    values = np.random.default_rng(stable_seed(5601, "control-directions", layer)).standard_normal((n, width), dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).tiny)


class QwenScorer:
    """Hooks the input to layers[L].mlp.down_proj; max over valid positions >= 1."""
    def __init__(self, unit_specs: list[dict[str, Any]], model_dir: str | Path,
                 dtype: str = "bf16", device: str = "auto", threads: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(threads)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ScoreError("CUDA was requested but is unavailable")
        if dtype not in ("bf16", "fp32"):
            raise ScoreError("dtype must be bf16 or fp32")
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.torch = torch
        self.device = device
        self.dtype = dtype
        self.model_dir = str(Path(model_dir).resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), torch_dtype=torch_dtype, local_files_only=True).eval()
        self.model.to(device)
        layers = self.model.model.layers
        width = int(self.model.config.intermediate_size)
        if width != 6144:
            raise ScoreError(f"Qwen hook width is {width}; the registered Part 58 direction width is 6144")
        self.unit_keys = [str(u["unit_key"]) for u in unit_specs]
        self._state: dict[str, Any] = {}
        self._max_buffer: np.ndarray | None = None
        self._marked_buffer: np.ndarray | None = None
        grouped: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for col, unit in enumerate(unit_specs):
            grouped.setdefault(int(unit["layer"]), []).append((col, unit))
        self.handles = []
        for layer, members in grouped.items():
            neurons = [(col, int(unit["unit_index"])) for col, unit in members if unit["unit_index"] is not None]
            controls = [(col, int(unit["control_id"])) for col, unit in members if unit["control_id"] is not None]
            neuron_cols = torch.tensor([i for _, i in neurons], dtype=torch.long, device=device)
            neuron_out = [c for c, _ in neurons]
            control_out = [c for c, _ in controls]
            if controls and (min(i for _, i in controls) < 0 or max(i for _, i in controls) >= 100):
                raise ScoreError(f"control id outside the 100 saved Part 58 directions at layer {layer}")
            cdirs = torch.as_tensor(control_directions(100, width, layer)[[i for _, i in controls]], device=device) if controls else None

            def make_hook(L: int, cols: Any, out_cols: list[int], cvec: Any, ctl_cols: list[int]):
                def hook(_module: Any, args: tuple[Any, ...]) -> None:
                    x = args[0].detach().float()
                    pieces = []
                    destination = []
                    if out_cols:
                        pieces.append(x.index_select(-1, cols))
                        destination.extend(out_cols)
                    if ctl_cols:
                        pieces.append(torch.matmul(x, cvec.transpose(0, 1)))
                        destination.extend(ctl_cols)
                    activations = torch.cat(pieces, dim=-1)
                    valid = self._state["mask"][:, 1:]
                    if activations.shape[1] > 1:
                        rest = activations[:, 1:, :].masked_fill(~valid.unsqueeze(-1), float("-inf")).amax(dim=1)
                    else:
                        rest = torch.full((activations.shape[0], activations.shape[2]), float("nan"), device=activations.device)
                    self._max_buffer[np.ix_(self._state["rows"], destination)] = rest.cpu().numpy()
                    mark_mask = self._state.get("marked_mask")
                    if mark_mask is not None:
                        mm = mark_mask[:, 1:]
                        if activations.shape[1] > 1:
                            at_marked = activations[:, 1:, :].masked_fill(~mm.unsqueeze(-1), float("-inf")).amax(dim=1)
                            at_marked[torch.isinf(at_marked)] = float("nan")
                            self._marked_buffer[np.ix_(self._state["rows"], destination)] = at_marked.cpu().numpy()
                return hook
            self.handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(make_hook(layer, neuron_cols, neuron_out, cdirs, control_out)))

    def score_batch(self, items: list[dict[str, Any]], rows: list[int], max_out: np.ndarray,
                    marked_out: np.ndarray, batch_size: int = 8) -> None:
        torch = self.torch
        for start in range(0, len(items), batch_size):
            subset = items[start:start + batch_size]
            row_subset = rows[start:start + batch_size]
            texts = [item["text"] for item in subset]
            clean_spans = [item.get("marked_span") for item in subset]
            try:
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True,
                                     return_offsets_mapping=True)
                offsets = enc.pop("offset_mapping").cpu().numpy()
            except (TypeError, ValueError, NotImplementedError, KeyError) as exc:
                if any(span is not None for span in clean_spans):
                    raise ScoreError("M4 needs a fast tokenizer with offset mappings for marked-token scoring") from exc
                enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True)
                offsets = None
            encoded = {k: v.to(self.device) for k, v in enc.items()}
            mask = encoded["attention_mask"].bool()
            marked_mask = None
            if offsets is not None and any(span is not None for span in clean_spans):
                marked_mask = torch.zeros_like(mask, dtype=torch.bool)
                for b, span in enumerate(clean_spans):
                    if span is None:
                        continue
                    lo, hi = span
                    off = offsets[b]
                    overlap = (off[:, 0] < hi) & (off[:, 1] > lo) & (off[:, 1] > off[:, 0])
                    marked_mask[b] = torch.as_tensor(overlap, device=self.device) & mask[b]
            self._max_buffer = max_out
            self._marked_buffer = marked_out
            self._state = {"mask": mask, "marked_mask": marked_mask, "rows": row_subset}
            with torch.inference_mode():
                self.model(**encoded, use_cache=False)
        self._state = {}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _phase_tag(method: str, phase: str) -> str:
    if method == "M3" and phase == "selection":
        return "M3-selection"
    if method == "M5" and phase == "round1":
        return "M5-round1"
    if phase in ("fresh", "round2"):
        return method
    return method


def _progress_paths(root: Path, split: str, method: str, phase: str) -> tuple[Path, Path, Path]:
    tag = _phase_tag(method, phase)
    base = root / "acts" / split / f".{tag}.progress"
    return base.with_suffix(".max.npy"), base.with_suffix(".marked.npy"), base.with_suffix(".json")


def resolved_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_input_fingerprint(entries: list[dict[str, Any]], split_units: list[dict[str, Any]],
                            model_dir: str | Path, dtype: str, device: str, batch_size: int) -> str:
    payload = {
        "backend": SCORE_BACKEND,
        "model_revision": MODEL_REVISION,
        "model_dir": str(Path(model_dir).resolve()),
        "dtype": dtype,
        "device": resolved_device(device),
        "batch_size": int(batch_size),
        "entries": [{"sid": x["sid"], "text": x["text"], "group": x["group"],
                     "marked_span": x.get("marked_span"), "record_index": x["record_index"],
                     "unit_key": x["unit_key"]} for x in entries],
        "units": [{"unit_key": u["unit_key"], "layer": int(u["layer"]),
                   "unit_index": u.get("unit_index"), "control_id": u.get("control_id")} for u in split_units],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def live_score_phase(root: Path, split: str, method: str, phase: str,
                     entries: list[dict[str, Any]], split_units: list[dict[str, Any]],
                     scorer: QwenScorer, thresholds: dict[str, float], batch_size: int = 8,
                     after_batch: Callable[[int], None] | None = None) -> Path:
    if not entries:
        path = score_path(root, split, method, phase)
        _save_npz_atomic(path, max=np.empty((0, len(split_units)), dtype=np.float32),
                         marked=np.empty((0, len(split_units)), dtype=np.float32), sids=np.asarray([], dtype=str),
                         unit_keys=np.asarray([u["unit_key"] for u in split_units]), record_indices=np.asarray([], dtype=np.int32),
                         groups=np.asarray([], dtype=str), texts=np.asarray([], dtype=str), method=np.asarray(method), phase=np.asarray(phase),
                         unit_types=np.asarray([u["type"] for u in split_units]))
        return path
    path = score_path(root, split, method, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    max_path, marked_path, progress_path = _progress_paths(root, split, method, phase)
    fingerprint = score_input_fingerprint(entries, split_units, scorer.model_dir, scorer.dtype, scorer.device, batch_size)
    spec = {"fingerprint": fingerprint, "shape": [len(entries), len(split_units)], "model_revision": MODEL_REVISION,
            "dtype": scorer.dtype, "device": scorer.device, "batch_size": int(batch_size),
            "backend": SCORE_BACKEND, "model_dir": scorer.model_dir, "completed": 0}
    maps_exist = max_path.is_file() and marked_path.is_file()
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        for key in ("fingerprint", "shape", "model_revision", "dtype", "device", "batch_size", "backend", "model_dir"):
            if saved.get(key) != spec[key]:
                raise ScoreError(f"score progress does not match current {method}/{phase} inputs; refusing to overwrite")
        completed = int(saved.get("completed", 0)) if maps_exist else 0
        if maps_exist:
            max_map = np.lib.format.open_memmap(max_path, mode="r+")
            marked_map = np.lib.format.open_memmap(marked_path, mode="r+")
        else:
            # A kill can leave the durable marker before either memmap is created.
            max_path.unlink(missing_ok=True)
            marked_path.unlink(missing_ok=True)
            write_json(progress_path, {**spec, "completed": 0})
            max_map = np.lib.format.open_memmap(max_path, mode="w+", dtype=np.float32, shape=tuple(spec["shape"]))
            marked_map = np.lib.format.open_memmap(marked_path, mode="w+", dtype=np.float32, shape=tuple(spec["shape"]))
            max_map[:] = np.nan; marked_map[:] = np.nan
    else:
        if max_path.exists() or marked_path.exists():
            print(f"recovering orphan score progress arrays for {method}/{phase}; no durable progress marker exists")
            max_path.unlink(missing_ok=True)
            marked_path.unlink(missing_ok=True)
        write_json(progress_path, {**spec, "completed": 0})
        max_map = np.lib.format.open_memmap(max_path, mode="w+", dtype=np.float32, shape=tuple(spec["shape"]))
        marked_map = np.lib.format.open_memmap(marked_path, mode="w+", dtype=np.float32, shape=tuple(spec["shape"]))
        max_map[:] = np.nan; marked_map[:] = np.nan
        completed = 0
    if completed < 0 or completed > len(entries):
        raise ScoreError("score progress row boundary is invalid")
    for start in range(completed, len(entries), batch_size):
        stop = min(start + batch_size, len(entries))
        scorer.score_batch(entries[start:stop], list(range(start, stop)), max_map, marked_map, batch_size=batch_size)
        max_map.flush(); marked_map.flush()
        write_json(progress_path, {**spec, "completed": stop})
        if after_batch:
            after_batch(stop)
    unit_keys = [u["unit_key"] for u in split_units]
    record_indices = np.asarray([x["record_index"] for x in entries], dtype=np.int32)
    output = {"max": np.asarray(max_map), "marked": np.asarray(marked_map), "sids": np.asarray([x["sid"] for x in entries]),
              "unit_keys": np.asarray(unit_keys), "record_indices": record_indices,
              "groups": np.asarray([x["group"] for x in entries]), "texts": np.asarray([x["text"] for x in entries]),
              "method": np.asarray(method), "phase": np.asarray(phase),
              "model_revision": np.asarray(MODEL_REVISION), "dtype": np.asarray(scorer.dtype),
              "device": np.asarray(scorer.device), "batch_size": np.asarray(batch_size),
              "backend": np.asarray(SCORE_BACKEND), "model_dir": np.asarray(scorer.model_dir),
              "score_fingerprint": np.asarray(fingerprint),
              "unit_types": np.asarray([u["type"] for u in split_units]),
              "thresholds": np.asarray([thresholds[u["unit_key"]] for u in split_units], dtype=np.float32)}
    _save_npz_atomic(path, **output)
    write_json(progress_path, {**spec, "completed": len(entries), "complete": True, "output": str(path)})
    # Keep progress arrays until the final archive has been durably renamed.
    max_map.flush(); marked_map.flush()
    del max_map, marked_map
    max_path.unlink(missing_ok=True); marked_path.unlink(missing_ok=True)
    return path


def reproduction_check(root: Path, split: str, split_units: list[dict[str, Any]], scorer: QwenScorer) -> dict[str, Any]:
    path = root / "reproduction" / f"{split}.json"
    fingerprint = {"revision": MODEL_REVISION, "dtype": scorer.dtype, "device": scorer.device,
                   "backend": SCORE_BACKEND, "model_dir": scorer.model_dir,
                   "unit_keys": [u["unit_key"] for u in split_units]}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fingerprint:
            if not reproduction_passed(old.get("by_type", {}), old.get("all_units", {})):
                raise ScoreError("cached reproduction fails median-ratio or Spearman gates")
            return old
    matrix, sids, ids, keys = load_source_arrays()
    source_sents = read_jsonl(DATA / "sent.all.jsonl")
    rng = random.Random(REPRO_SEED)
    chosen = sorted(rng.sample(range(len(source_sents)), min(REPRO_N, len(source_sents))))
    source_row = {sid: i for i, sid in enumerate(sids)}
    col_by_record = {rid: j for j, rid in enumerate(ids)}
    ref_cols = []
    observed_cols = []
    tracked_units = []
    for observed_col, unit in enumerate(split_units):
        col = col_by_record.get(int(unit["record_index"]))
        if col is not None and keys[col] == unit["unit_key"]:
            ref_cols.append(col)
            observed_cols.append(observed_col)
            tracked_units.append(unit)
    if not observed_cols:
        raise ScoreError("reproduction check has no common source units")
    items = [{"text": source_sents[i]["text"], "marked_span": None} for i in chosen]
    observed = np.full((len(items), len(split_units)), np.nan, dtype=np.float32)
    marked = np.full_like(observed, np.nan)
    scorer.score_batch(items, list(range(len(items))), observed, marked, batch_size=8)
    ref_rows = [source_row[str(source_sents[i]["sid"])] for i in chosen]
    reference = matrix[np.ix_(ref_rows, ref_cols)]
    observed_tracked = observed[:, observed_cols]
    summary = ratio_summary(reference, observed_tracked)
    by_type = {}
    for typ in ("neuron", "control"):
        cols = [j for j, unit in enumerate(tracked_units) if unit["type"] == typ]
        by_type[typ] = ratio_summary(reference[:, cols], observed_tracked[:, cols]) if cols else None
    result = {"fingerprint": fingerprint, "revision": MODEL_REVISION, "split": split,
              "sentence_seed": REPRO_SEED, "sentences": len(items), "all_units": summary, "by_type": by_type,
              "passed": reproduction_passed(by_type, summary),
              "definition": "new max activation divided by recorded Part 58 max; median ratio 0.9..1.1 and Spearman >= 0.98 overall and within each unit type; exclude zero reference values for ratio; Spearman uses all finite pairs"}
    write_json(path, result)
    if not result["passed"]:
        raise ScoreError("reproduction fails the median-ratio or Spearman gate overall or for one unit type; refusing to score generated sentences")
    return result


def cached_m0_existing(root: Path, split: str) -> Path:
    from gen60 import load_splits, prepare_m0
    units = load_splits()[split]
    target = archive_m0_path(root, split)
    gen_file = root / "out" / split / "M0.jsonl"
    if not gen_file.exists():
        prepare_m0(root, split)
    return score_cached_m0(root, split, units)


def plan_live(root: Path, split: str, methods: list[str]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    import gen60
    plans = []
    total_units = len(gen60.load_splits()[split])
    for method in methods:
        path = gen60.output_path(root, split, method)
        if not path.exists():
            continue
        rows = read_jsonl(path)
        if method == "M0":
            entries = entries_for_phase(rows, method, "primary")
            if entries:
                plans.append((method, "primary", entries))
            continue
        if method == "M3":
            for phase in ("selection", "fresh"):
                entries = entries_for_phase(rows, method, phase)
                denominator = _phase_denominator(rows, method, phase, total_units)
                successes = len({entry["record_index"] for entry in entries})
                if denominator > 0 and gen60.generation_failure_exceeds_ten_percent(successes, denominator):
                    print(f"drop {method}/{phase} scoring: only {successes}/{denominator} eligible units parsed (more than 10% failed)")
                    continue
                if entries:
                    plans.append((method, phase, entries))
        elif method == "M5":
            for phase in ("round1", "round2"):
                entries = entries_for_phase(rows, method, phase)
                denominator = _phase_denominator(rows, method, phase, total_units)
                successes = len({entry["record_index"] for entry in entries})
                if denominator > 0 and gen60.generation_failure_exceeds_ten_percent(successes, denominator):
                    print(f"drop {method}/{phase} scoring: only {successes}/{denominator} eligible units parsed (more than 10% failed)")
                    continue
                if entries:
                    plans.append((method, phase, entries))
        elif method != "M0":
            entries = entries_for_phase(rows, method, "primary")
            denominator = total_units
            successes = len({entry["record_index"] for entry in entries})
            if denominator > 0 and gen60.generation_failure_exceeds_ten_percent(successes, denominator):
                print(f"drop {method}/primary scoring: only {successes}/{denominator} units parsed (more than 10% failed)")
                continue
            if entries:
                plans.append((method, "primary", entries))
    return plans


def _phase_denominator(rows: list[dict[str, Any]], method: str, phase: str,
                       total_units: int) -> int:
    if method == "M3" and phase == "fresh":
        return sum(isinstance(row.get("selection"), dict)
                   and row["selection"].get("parse_status") == "ok" for row in rows)
    if method == "M5" and phase == "round2":
        return sum(isinstance(row.get("round1"), dict)
                   and row["round1"].get("parse_status") == "ok" for row in rows)
    return total_units


def verify_scoring(root: Path, split: str, methods: list[str], model_dir: str | Path,
                   dtype: str = "bf16", device: str = "cuda", batch_size: int = 8) -> None:
    """Fail closed unless every eligible generated phase has matching saved scores."""
    import gen60
    from gen60 import load_splits
    units = load_splits()[split]
    for method in methods:
        if method == "M0":
            target = score_path(root, split, "M0", "primary")
            if not target.is_file():
                raise ScoreError(f"missing cached baseline score matrix: {target}")
            if not (root / "out" / split / "M0.jsonl").is_file():
                raise ScoreError(f"missing required generation output: {root / 'out' / split / 'M0.jsonl'}")
            if not entries_for_phase(read_jsonl(root / "out" / split / "M0.jsonl"), "M0", "primary"):
                raise ScoreError("M0 generation output has no scoreable baseline sentences")
            continue
        output_path = root / "out" / split / f"{method}.jsonl"
        if not output_path.is_file():
            raise ScoreError(f"missing required generation output: {output_path}")
    for method, phase, entries in plan_live(root, split, methods):
        target = score_path(root, split, method, phase)
        if not target.is_file():
            raise ScoreError(f"missing activation matrix for eligible phase {method}/{phase}: {target}")
        expected = score_input_fingerprint(entries, _unit_info(units), model_dir, dtype, device, batch_size)
        data = np.load(target, allow_pickle=False)
        if ([str(x) for x in data["sids"]] != [x["sid"] for x in entries]
                or [str(x) for x in data["unit_keys"]] != [u["unit_key"] for u in units]
                or str(data["method"]) != method or str(data["phase"]) != phase
                or str(data["score_fingerprint"]) != expected
                or str(data["model_revision"]) != MODEL_REVISION
                or str(data["dtype"]) != dtype or str(data["device"]) != resolved_device(device)
                or int(data["batch_size"]) != int(batch_size)
                or str(data["backend"]) != SCORE_BACKEND
                or str(data["model_dir"]) != str(Path(model_dir).resolve())):
            raise ScoreError(f"saved activation metadata or inputs do not match {method}/{phase}: {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    threshold = sub.add_parser("thresholds", help="compute and save the fixed per-unit p95 null thresholds")
    score = sub.add_parser("score", help="score all currently available method phases")
    score.add_argument("--split", choices=["dev", "test"], required=True)
    score.add_argument("--methods", default="all", help="comma list M0..M7 or all")
    score.add_argument("--model", type=Path, default=None)
    score.add_argument("--device", default="auto", choices=["auto", "cuda", "xpu", "cpu"])
    score.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    score.add_argument("--batch-size", type=int, default=8)
    score.add_argument("--threads", type=int, default=4)
    score.add_argument("--force", action="store_true", help="discard mismatched score progress and matrices")
    verify = sub.add_parser("verify", help="verify required generation outputs and matching score artifacts")
    verify.add_argument("--split", choices=["dev", "test"], required=True)
    verify.add_argument("--methods", required=True, help="comma list M0..M7")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    verify.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    verify.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    lock = None
    try:
        check_data_paths()
        args.root.mkdir(parents=True, exist_ok=True)
        lock = (args.root / ".score.lock").open("a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        thresholds = ensure_thresholds(args.root)
        if args.command == "thresholds":
            print(f"wrote {len(thresholds)} fixed p95 thresholds to {args.root / 'thresholds.json'}")
            return 0
        if args.command == "verify":
            methods = [x.strip() for x in args.methods.split(",") if x.strip()]
            unknown = sorted(set(methods) - {f"M{i}" for i in range(8)})
            if unknown:
                raise ScoreError(f"unknown method names: {unknown}")
            check = json.loads((args.root / "reproduction" / f"{args.split}.json").read_text(encoding="utf-8"))
            if not check.get("passed"):
                raise ScoreError(f"reproduction check is missing or did not pass for {args.split}")
            verify_scoring(args.root, args.split, methods, args.model, args.dtype, args.device, args.batch_size)
            print(f"verified split={args.split} methods={','.join(methods)} model={args.model}")
            return 0
        from gen60 import load_splits, prepare_m0
        units = load_splits()[args.split]
        methods = [f"M{i}" for i in range(8)] if args.methods == "all" else [x.strip() for x in args.methods.split(",") if x.strip()]
        unknown = sorted(set(methods) - {f"M{i}" for i in range(8)})
        if unknown:
            raise ScoreError(f"unknown method names: {unknown}")
        if "M0" in methods:
            if not (args.root / "out" / args.split / "M0.jsonl").exists():
                prepare_m0(args.root, args.split)
            print(f"M0 Part 58 archive cross-check: {cached_m0_existing(args.root, args.split)}")
        plans = plan_live(args.root, args.split, methods)
        model_dir = args.model or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / MODEL_REVISION
        resolved = resolved_device(args.device)
        specs = _unit_info(units)
        pending = []
        for method, phase, entries in plans:
            target = score_path(args.root, args.split, method, phase)
            if args.force:
                clear_score_artifacts(args.root, args.split, method, phase)
                pending.append((method, phase, entries))
                continue
            if target.exists():
                expected = score_input_fingerprint(entries, specs, model_dir, args.dtype, resolved, args.batch_size)
                existing = None
                try:
                    existing = np.load(target, allow_pickle=False)
                    matches = (str(existing["score_fingerprint"]) == expected
                               and [str(x) for x in existing["sids"]] == [x["sid"] for x in entries]
                               and [str(x) for x in existing["texts"]] == [x["text"] for x in entries]
                               and [str(x) for x in existing["unit_keys"]] == [u["unit_key"] for u in units]
                               and str(existing["model_revision"]) == MODEL_REVISION
                               and str(existing["dtype"]) == args.dtype
                               and str(existing["device"]) == resolved
                               and int(existing["batch_size"]) == int(args.batch_size)
                               and str(existing["backend"]) == SCORE_BACKEND)
                except (OSError, KeyError, ValueError):
                    matches = False
                finally:
                    if existing is not None:
                        existing.close()
                if matches:
                    print(f"already complete: {target}")
                    continue
                elif method == "M0" and (existing is None or "score_fingerprint" not in existing.files):
                    target.unlink(missing_ok=True)
                    for progress in _progress_paths(args.root, args.split, method, phase):
                        progress.unlink(missing_ok=True)
                else:
                    raise ScoreError(f"cached score inputs/configuration differ for {method}/{phase}; use --force to rescore")
            pending.append((method, phase, entries))
        plans = pending
        if not plans:
            if any(score_path(args.root, args.split, m, p).is_file() for m, p, _ in plan_live(args.root, args.split, methods)):
                check_path = args.root / "reproduction" / f"{args.split}.json"
                if not check_path.is_file():
                    raise ScoreError(f"missing reproduction record: {check_path}")
                verify_scoring(args.root, args.split, methods, model_dir, args.dtype, args.device, args.batch_size)
            print("no new live sentences are ready to score")
            return 0
        scorer = QwenScorer(specs, model_dir, dtype=args.dtype, device=args.device, threads=args.threads)
        try:
            check = reproduction_check(args.root, args.split, specs, scorer)
            print(f"reproduction median ratio={check['all_units']['median_ratio']:.4f} Spearman={check['all_units']['spearman']}")
            thresholds = extend_thresholds_for_split(args.root, args.split, specs, scorer, thresholds)
            for method, phase, entries in plans:
                scored = live_score_phase(args.root, args.split, method, phase, entries, specs, scorer, thresholds, args.batch_size)
                print(f"scored {len(entries)} sentences -> {scored}")
        finally:
            scorer.close()
        verify_scoring(args.root, args.split, methods, model_dir, args.dtype, args.device, args.batch_size)
        return 0
    except (ScoreError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
