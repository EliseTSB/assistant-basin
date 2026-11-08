#!/usr/bin/env python3
"""Registered Part 198 runner. Dry-run plans are inert; model work is always explicit."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
import warnings
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT_DEFAULT = HERE / "out"
GPT2 = Path("/mnt/e/neuron-probe/edit-detect/gpt2-xl")
STATS_DIR = Path("/mnt/e/neuron-probe/edit-detect/stats")
EASYEDIT = Path("/mnt/e/neuron-probe/edit-detect/src/EasyEdit-main")
COUNTERFACT = Path("/mnt/e/neuron-probe/edit-detect/counterfact.json")
HF_HUB = Path("/mnt/e/neuron-probe/hf/hub")
WSP_SRC = Path("/home/elisepc/edit-detect/wsp-src/wsp.py")
EASYEDIT_COMMIT = "5c519afb14e39c7f70bc5536ebd777a62d8493bf"
WSP_SHA256 = "90184edf8f3e1ce7a25cb2697f4fe94f92befef1fbcbe64c29c3ac0b0ae65f81"
GPT2_SHA256 = "0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a"
QWEN35_SNAPSHOT = "2fc06364715b967f1860aea9cf38778875588b17"
N_GPT2 = 48
JSONL_RECOVERIES: list[dict[str, Any]] = []
ARM_C_REVIEWER_MAX = {
    "EleutherAI/pythia-410m": (6.73, "ton"),
    "Qwen/Qwen3-0.6B": (10.33, " set"),
    "allenai/OLMo-2-0425-1B": (12.10, " time"),
    "EleutherAI/pythia-1.4b": (14.36, " order"),
    "Qwen/Qwen3-1.7B": (15.90, " set"),
    "HuggingFaceTB/SmolLM2-1.7B": (10.92, " time"),
    "HuggingFaceTB/SmolLM3-3B": (13.08, "bfd"),
    "Qwen/Qwen2.5-0.5B-Instruct": (9.21, " way"),
    "gpt2-xl": (9.79, None),
    "Qwen/Qwen3.5-0.8B": (7.30, None),
}

# Arm C is descriptive. Names and roles follow PART198-DESIGN.md revision 3.
ARM_C_MODELS = [
    {"id": "EleutherAI/pythia-410m", "role": "calibration"},
    {"id": "Qwen/Qwen3-0.6B", "role": "calibration"},
    {"id": "allenai/OLMo-2-0425-1B", "role": "calibration"},
    {"id": "EleutherAI/pythia-1.4b", "role": "evaluation"},
    {"id": "Qwen/Qwen3-1.7B", "role": "evaluation"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "role": "evaluation"},
    {"id": "HuggingFaceTB/SmolLM3-3B", "role": "evaluation"},
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "role": "evaluation"},
    {"id": "gpt2-xl", "role": "host"},
    {"id": "Qwen/Qwen3.5-0.8B", "role": "host"},
]

# Frozen question strings copied verbatim from the design's Arm B table.
TEMPLATES = {
    "P103": (["What is the mother tongue of {}?", "What was {}'s native language?",
              "Which language did {} grow up speaking first?"],
             "In what language did {} first learn to speak?"),
    "P1412": (["What language does {} use?", "Which language is used by {}?",
               "In what language does {} communicate?"],
              "What language would you hear {} speak?"),
    "P37": (["What is an official language of {}?", "Which language is official in {}?",
             "What language is used officially in {}?"], "Name an official language of {}."),
    "P30": (["On which continent is {}?", "What continent does {} belong to?",
             "Which continent is {} part of?"], "{} is located on what continent?"),
    "P27": (["What country is {} a citizen of?", "What is {}'s citizenship?",
             "Which country is {} a national of?"],
            "Of which country is {} a citizen?"),
    "P17": (["In which country is {} located?", "What country is {} in?",
             "Which country contains {}?"], "{} can be found in which country?"),
    "P36": (["What is the capital of {}?", "Which city is the capital of {}?",
             "Name the capital city of {}."], "{}'s capital is which city?"),
    "P19": (["Where was {} born?", "In which city was {} born?",
             "What is {}'s birthplace?"], "{} was born in what city?"),
    "P364": (["What was the original language of {}?", "In what language was {} originally made?",
              "Which language was {} first produced in?"], "{} was originally in which language?"),
    "P159": (["Where is the headquarters of {}?", "In which city is {} headquartered?",
              "Where is {} based?"], "{} has its headquarters in which city?"),
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def safe_json(value: Any) -> Any:
    """Make paths and non-finite numbers safe for standards-compliant JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(safe_json(record), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if path.exists() and path.stat().st_size:
        with path.open("rb+") as existing:
            existing.seek(-1, os.SEEK_END)
            if existing.read(1) != b"\n":
                existing.seek(0, os.SEEK_END)
                existing.write(b"\n")
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    complete_end = data.rfind(b"\n") + 1
    for number, raw in enumerate(lines, 1):
        torn_tail = number == len(lines) and not raw.endswith(b"\n")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            if torn_tail:
                _drop_torn_jsonl_tail(path, number, complete_end, len(raw), str(exc))
                break
            raise ValueError(f"malformed JSONL at {path}:{number}: {exc}") from exc
        if not decoded.strip():
            continue
        try:
            item = json.loads(decoded)
        except json.JSONDecodeError as exc:
            if torn_tail:
                _drop_torn_jsonl_tail(path, number, complete_end, len(raw), str(exc))
                break
            raise ValueError(f"malformed JSONL at {path}:{number}: {exc}") from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _drop_torn_jsonl_tail(path: Path, line_number: int, complete_end: int,
                          dropped_bytes: int, reason: str) -> None:
    with path.open("r+b") as f:
        f.truncate(complete_end)
        f.flush()
        os.fsync(f.fileno())
    event = {"path": str(path), "line": line_number, "dropped_bytes": dropped_bytes,
             "reason": reason}
    JSONL_RECOVERIES.append(event)
    warnings.warn(f"dropped torn final JSONL line {line_number} from {path} ({dropped_bytes} bytes)",
                  RuntimeWarning, stacklevel=2)


def existing_keys(path: Path, field: str) -> set[str]:
    """Resume registered terminal records; smoke rows never suppress registered work."""
    result = set()
    for row in read_jsonl(path):
        if row.get("run_mode", "registered") == "smoke":
            continue
        if row.get("record_type") not in {"case", "batch", "armC_model", "failure"}:
            continue
        value = row.get(field)
        if value is not None:
            result.add(str(value))
    return result


def took_transition(before_old: float, before_new: float, after_old: float, after_new: float) -> tuple[bool, bool]:
    """Arm A transition. already_satisfied means the new answer led before the edit."""
    already = before_new > before_old
    took = before_old > before_new and after_new > after_old
    return took, already


def wsp_took(before_exact: int, after_exact: int, taught: int = 3) -> tuple[bool, bool]:
    """Arm B transition: none taught exactly before, all taught exactly after."""
    already = before_exact > 0
    return before_exact == 0 and after_exact == taught, already


def wsp_take_measurement(before_exact: int, after_exact: int, taught: int,
                         free_exact_text: Any) -> dict[str, Any]:
    """Use runner-measured answers for take and retain WSP's independent self-report."""
    text = str(free_exact_text) if free_exact_text is not None else ""
    try:
        free_exact, free_total = (int(x) for x in text.split("/", 1))
        if free_exact < 0 or free_total < 0:
            raise ValueError("negative count")
        parse_error = None
    except (ValueError, TypeError):
        free_exact, free_total = None, None
        parse_error = "free_exact must be two nonnegative integers separated by '/'"
    took, already = wsp_took(before_exact, after_exact, taught)
    disagreement = (free_exact != after_exact or free_total != taught) if parse_error is None else None
    return {"took": bool(took), "already_satisfied": bool(already),
            "free_exact_text": text or None, "free_exact_count": free_exact,
            "free_exact_total": free_total, "free_exact_disagreement": disagreement,
            "free_exact_parse_error": parse_error}


def arm_c_limit_reached(selected_count: int, limit: int | None) -> bool:
    """Whether the requested number of new Arm C model loads has been reached."""
    return limit is not None and selected_count >= limit


def arm_c_successful_keys(path: Path) -> set[str]:
    """Only successful registered Arm C rows satisfy a calibration/evaluation key."""
    return {str(row["model_id"]) for row in read_jsonl(path)
            if row.get("run_mode", "registered") == "registered"
            and row.get("record_type") == "armC_model" and row.get("model_id") is not None}


def require_tstar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Arm C tstar.json is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("t_star"), (int, float)):
        raise RuntimeError(f"Arm C tstar.json has no numeric t_star: {path}")
    return data


def mass_gate(took: list[bool], minimum: int = 8) -> bool:
    return sum(bool(x) for x in took) >= minimum


def make_random_update(delta, seed: int, return_factors: bool = False, generator=None):
    """Replace a matrix's singular directions with deterministic random orthonormal factors."""
    import torch
    D = delta.detach().to(device="cpu", dtype=torch.float64)
    _, S0, _ = torch.linalg.svd(D, full_matrices=False)
    r = len(S0)
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
    gu = torch.randn((D.shape[0], r), dtype=torch.float64, generator=generator)
    gv = torch.randn((D.shape[1], r), dtype=torch.float64, generator=generator)
    U = torch.linalg.qr(gu, mode="reduced").Q
    V = torch.linalg.qr(gv, mode="reduced").Q
    R = (U * S0.unsqueeze(0)) @ V.T
    if return_factors:
        return R, U, S0, V
    return R


def random_control_layers(base_matrices: dict[int, Any], deltas: dict[int, Any], seed: int) -> dict[int, Any]:
    """Build all layer controls from one write seed, advanced over sorted layers."""
    import torch
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    result = {}
    for layer in sorted(deltas):
        result[int(layer)] = make_random_update(deltas[layer], seed, generator=generator)
    return result


def mass_gate_status(counted_batches: int) -> bool:
    return counted_batches > 0


def cp_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float] | None:
    """Two-sided exact Clopper-Pearson interval using binomial-tail inversion."""
    if n <= 0 or successes < 0 or successes > n:
        return None
    alpha = 1.0 - confidence
    try:
        from scipy.stats import beta
        lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, n - successes + 1))
        hi = 1.0 if successes == n else float(beta.ppf(1 - alpha / 2, successes + 1, n - successes))
        return lo, hi
    except ImportError:
        pass

    def pmf(i: int, p: float) -> float:
        if p <= 0.0:
            return 1.0 if i == 0 else 0.0
        if p >= 1.0:
            return 1.0 if i == n else 0.0
        return math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))

    def ge(k: int, p: float) -> float:
        return math.fsum(pmf(i, p) for i in range(k, n + 1))

    def le(k: int, p: float) -> float:
        return math.fsum(pmf(i, p) for i in range(k + 1))

    if successes == 0:
        lo = 0.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            mid = (a + b) / 2
            if ge(successes, mid) > alpha / 2:
                a = mid
            else:
                b = mid
        lo = (a + b) / 2
    if successes == n:
        hi = 1.0
    else:
        a, b = 0.0, 1.0
        for _ in range(80):
            mid = (a + b) / 2
            if le(successes, mid) > alpha / 2:
                a = mid
            else:
                b = mid
        hi = (a + b) / 2
    return lo, hi


def exact_mcnemar(discordant_a: int, discordant_b: int) -> float | None:
    d = discordant_a + discordant_b
    if d == 0:
        return 1.0
    tail = math.fsum(math.comb(d, i) for i in range(min(discordant_a, discordant_b) + 1)) / (2 ** d)
    return min(1.0, 2.0 * tail)


def root_out_dir(path: Path) -> Path:
    p = path.expanduser().resolve()
    root = HERE.resolve()
    if p != root and root not in p.parents:
        raise ValueError(f"--out-dir must be inside {root}; got {p}")
    return p


def load_cases() -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    frozen = json.loads((HERE / "cases198.json").read_text(encoding="utf-8"))
    if COUNTERFACT.exists():
        actual = hashlib.sha256(COUNTERFACT.read_bytes()).hexdigest()
        if actual != frozen["counterfact_sha256"]:
            raise RuntimeError(f"CounterFact SHA256 mismatch: expected {frozen['counterfact_sha256']}, got {actual}")
        rows = json.loads(COUNTERFACT.read_text(encoding="utf-8"))
        return frozen, {int(x["case_id"]): x for x in rows}
    return frozen, {}


def resolve_snapshot(model_id: str) -> Path:
    """Resolve the offline Hugging Face main ref; require one snapshot if no main ref exists."""
    if model_id == "gpt2-xl":
        return GPT2
    repo = HF_HUB / ("models--" + model_id.replace("/", "--"))
    refs = repo / "refs" / "main"
    if refs.is_file():
        revision = refs.read_text(encoding="utf-8").strip()
        candidate = repo / "snapshots" / revision
        if not candidate.is_dir():
            raise FileNotFoundError(f"main ref {revision} has no local snapshot for {model_id}")
        return candidate
    snaps = sorted(p for p in (repo / "snapshots").glob("*") if p.is_dir())
    if len(snaps) != 1:
        raise RuntimeError(f"{model_id}: expected one cached snapshot or refs/main, found {len(snaps)}")
    return snaps[0]


def ensure_offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load_stats_module():
    spec = importlib.util.spec_from_file_location("stats198", HERE / "stats198.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen stats198.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stats198"] = mod
    spec.loader.exec_module(mod)
    return mod


def run_id() -> str:
    return uuid.uuid4().hex


def append_run_event(log: Path, mode: str, key: str, pending: list[Any], rid: str) -> None:
    existed = log.exists() and log.stat().st_size > 0
    append_jsonl(log, {"record_type": "restart" if existed else "run_start", "run_mode": mode,
                       "run_id": rid, "timestamp": utc_now(), "key": key,
                       "pending": [str(x) for x in pending]})


def preflight(out_dir: Path) -> dict[str, Any]:
    """Run inert checks and require the model-based pre-data crosscheck before model/edit work."""
    results: dict[str, Any] = {"timestamp": utc_now(), "fixtures_pass": False,
                               "synthetic_tests_pass": False, "crosscheck_pass": False, "details": {}}
    for label, script in (("fixtures", HERE / "fixtures198.py"),
                          ("synthetic_tests", HERE / "test_build198.py")):
        proc = subprocess.run([sys.executable, "-I", str(script)], cwd=str(HERE), text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        results["details"][label] = {"returncode": proc.returncode, "output": proc.stdout}
        results["fixtures_pass" if label == "fixtures" else "synthetic_tests_pass"] = proc.returncode == 0
    crosscheck = HERE / "crosscheck198.json"
    if crosscheck.is_file():
        data = json.loads(crosscheck.read_text(encoding="utf-8"))
        fails = data.get("fails")
        current_stats_sha256 = hash_file(HERE / "stats198.py")
        stats_hash_matches = crosscheck_hash_matches(data, current_stats_sha256)
        results["crosscheck_pass"] = (isinstance(fails, list) and not fails
                                      and len(data.get("gpt2", {})) == 48
                                      and len(data.get("qwen", {})) == 24
                                      and stats_hash_matches)
        results["details"]["crosscheck"] = {"path": str(crosscheck), "fails": fails or [],
            "gpt2_layers": len(data.get("gpt2", {})), "qwen_layers": len(data.get("qwen", {})),
            "stats198_sha256": data.get("stats198_sha256"),
            "current_stats198_sha256": current_stats_sha256,
            "stats_hash_matches": stats_hash_matches}
    else:
        results["details"]["crosscheck"] = {"path": str(crosscheck), "missing": True,
                                             "reason": "must be produced before reading registered outcomes"}
    results["passed"] = all(results[k] for k in ("fixtures_pass", "synthetic_tests_pass", "crosscheck_pass"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate.json").write_text(json.dumps(safe_json(results), indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    if not results["passed"]:
        raise RuntimeError("pre-data gate failed; see out/gate.json (no model was loaded and no edit was run)")
    return results


def log_path_for(command: str, method: str | None, out_dir: Path) -> Path:
    if command == "armA":
        return out_dir / f"armA-{method.lower()}.jsonl"
    if command == "armA-mass":
        return out_dir / f"armA-mass-{method.lower()}.jsonl"
    if command == "armB":
        return out_dir / "armB-wsp.jsonl"
    if command == "armB-seq":
        return out_dir / "armB-seq-wsp.jsonl"
    if command == "armC":
        return out_dir / "armC.jsonl"
    raise ValueError(command)


def dry_plan(command: str, method: str | None, limit: int | None, out_dir: Path) -> dict[str, Any]:
    frozen, _ = load_cases()
    if command == "armA":
        selected: Any = frozen["armA_single"][:limit]
    elif command == "armA-mass":
        selected = frozen["armA_mass"][:limit]
    elif command == "armB":
        selected = frozen["armB_single"][:limit]
    elif command == "armB-seq":
        selected = frozen["armB_sequence"][:limit]
    elif command == "armC":
        selected = [{**m, "path": str(resolve_snapshot(m["id"]))} for m in ARM_C_MODELS[:limit]]
    else:
        model_paths = [GPT2, *(resolve_snapshot(x["id"]) for x in ARM_C_MODELS if x["id"] != "gpt2-xl")]
        selected = {
            "model_weight_files": [str(f) for folder in model_paths for f in weight_files(folder)],
            "model_config_and_tokenizer_files": [
                str(f) for folder in model_paths for f in sorted(folder.rglob("*"))
                if f.is_file() and f.name in {"config.json", "generation_config.json", "tokenizer.json",
                    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json",
                    "merges.txt", "vocab.txt", "tokenizer.model", "spiece.model",
                    "sentencepiece.bpe.model", "chat_template.jinja"}],
            "easyedit_source_dirs": [str(EASYEDIT / "easyeditor" / "models" / x)
                                     for x in ("rome", "memit", "alphaedit")],
            "configs": [str(EASYEDIT / "hparams" / x / "gpt2-xl.yaml")
                        for x in ("ROME", "MEMIT", "AlphaEdit")],
            "second_moments": [str(f) for f in sorted(STATS_DIR.glob(
                "gpt2-xl/wikipedia_stats/transformer.h.*.mlp.c_proj_float32_mom2_100000.npz"))],
            "other_inputs": [str(COUNTERFACT), str(HERE / "cases198.json"),
                             str(HERE / "stats198.py"), str(HERE / "ee_import.py"),
                             str(HERE / "run198.py"), str(WSP_SRC)]}
    return {"command": command, "method": method, "dry_run": True, "limit": limit,
            "out_dir": str(out_dir),
            "output": str(log_path_for(command, method, out_dir)) if command != "manifest"
            else str(out_dir / "manifest.json"),
            "selected": selected,
            "pinned_paths": {"gpt2": str(GPT2),
                             "qwen35": str(HF_HUB / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / QWEN35_SNAPSHOT),
                             "stats": str(STATS_DIR), "easyedit": str(EASYEDIT),
                             "counterfact": str(COUNTERFACT), "wsp": str(WSP_SRC),
                             "wsp_sha256": WSP_SHA256, "easyedit_commit": EASYEDIT_COMMIT}}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def crosscheck_hash_matches(crosscheck: dict[str, Any], current_sha256: str) -> bool:
    return crosscheck.get("stats198_sha256") == current_sha256


def hash_input_hashes(hashes: dict[str, str]) -> str:
    canonical = json.dumps(hashes, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def gpt2_base_inputs_sha256() -> str:
    """Fingerprint the frozen GPT-2 weights, tokenizer files, and statistics implementation."""
    metadata_names = {"config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                      "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model",
                      "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja"}
    hashes = {"model:model.safetensors": GPT2_SHA256,
              "code:stats198.py": hash_file(HERE / "stats198.py"),
              "code:run198.py": hash_file(HERE / "run198.py")}
    for path in sorted(GPT2.rglob("*")):
        if path.is_file() and path.name in metadata_names:
            hashes[f"tokenizer:{path.relative_to(GPT2)}"] = hash_file(path)
    return hash_input_hashes(hashes)


def weight_files(folder: Path) -> list[Path]:
    suffixes = {".safetensors", ".bin", ".pt", ".pth"}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix in suffixes
                  and (p.suffix != ".bin" or p.name.startswith(("pytorch_model", "model"))))


def manifest_inputs() -> dict[str, Path]:
    inputs: dict[str, Path] = {"model:gpt2-xl/model.safetensors": GPT2 / "model.safetensors"}
    metadata_names = {"config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                      "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model",
                      "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja"}
    for f in sorted(GPT2.rglob("*")):
        if f.is_file() and f.name in metadata_names:
            inputs[f"model_metadata:gpt2-xl/{f.relative_to(GPT2)}"] = f
    for item in ARM_C_MODELS:
        if item["id"] == "gpt2-xl":
            continue
        path = resolve_snapshot(item["id"])
        for f in weight_files(path):
            inputs[f"model:{item['id']}/{f.relative_to(path)}"] = f
        metadata_names = {"config.json", "generation_config.json", "tokenizer.json",
                          "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
                          "vocab.json", "merges.txt", "vocab.txt", "tokenizer.model",
                          "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja"}
        for f in sorted(path.rglob("*")):
            if f.is_file() and f.name in metadata_names:
                inputs[f"model_metadata:{item['id']}/{f.relative_to(path)}"] = f
    easy = EASYEDIT / "easyeditor" / "models"
    for folder in ("rome", "memit", "alphaedit"):
        for f in sorted((easy / folder).rglob("*.py")):
            inputs[f"easyedit:{folder}/{f.relative_to(easy / folder)}"] = f
    for method in ("ROME", "MEMIT", "AlphaEdit"):
        inputs[f"config:{method}/gpt2-xl.yaml"] = EASYEDIT / "hparams" / method / "gpt2-xl.yaml"
    mom2 = sorted(STATS_DIR.glob("gpt2-xl/wikipedia_stats/transformer.h.*.mlp.c_proj_float32_mom2_100000.npz"))
    if len(mom2) != 5:
        raise RuntimeError(f"expected exactly five second-moment files, found {len(mom2)}")
    for p in mom2:
        inputs[f"second_moment:{p.name}"] = p
    for label, p in (("counterfact", COUNTERFACT), ("cases198", HERE / "cases198.json"),
                     ("stats198", HERE / "stats198.py"), ("ee_import", HERE / "ee_import.py"),
                     ("run198", HERE / "run198.py"), ("wsp", WSP_SRC)):
        inputs[label] = p
    missing = [f"{label}: {p}" for label, p in inputs.items() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing pinned inputs: " + "; ".join(missing))
    return inputs


def tensor_fingerprint(tensor: Any) -> str:
    """Hash a tensor's exact bytes without materializing a second whole-model copy."""
    import torch
    value = tensor.detach().cpu()
    if not value.is_contiguous():
        value = value.contiguous()
    raw = value.view(torch.uint8).reshape(-1)
    view = memoryview(raw.numpy())
    h = hashlib.sha256()
    h.update(f"{value.dtype}:{tuple(value.shape)}:".encode("ascii"))
    chunk_size = 8 * 1024 * 1024
    for start in range(0, len(view), chunk_size):
        h.update(view[start:start + chunk_size])
    return h.hexdigest()


def model_state_snapshot(model) -> dict[str, str]:
    """Keep a compact immutable fingerprint for every parameter and buffer."""
    snapshot = {f"parameter:{name}": tensor_fingerprint(value)
                for name, value in model.named_parameters()}
    snapshot.update({f"buffer:{name}": tensor_fingerprint(value)
                    for name, value in model.named_buffers()})
    return snapshot


def model_matches_snapshot(model, snapshot: dict[str, str]) -> bool:
    current = {f"parameter:{name}": tensor_fingerprint(value)
               for name, value in model.named_parameters()}
    current.update({f"buffer:{name}": tensor_fingerprint(value)
                    for name, value in model.named_buffers()})
    return current == snapshot


def write_manifest(out_dir: Path) -> dict[str, Any]:
    actual_wsp = hash_file(WSP_SRC)
    if actual_wsp != WSP_SHA256:
        raise RuntimeError(f"wsp.py SHA256 mismatch: expected {WSP_SHA256}, got {actual_wsp}")
    gpt2_sha = hash_file(GPT2 / "model.safetensors")
    if gpt2_sha != GPT2_SHA256:
        raise RuntimeError(f"GPT-2 XL SHA256 mismatch: expected {GPT2_SHA256}, got {gpt2_sha}")
    expected_cf = json.loads((HERE / "cases198.json").read_text())["counterfact_sha256"]
    actual_cf = hash_file(COUNTERFACT)
    if actual_cf != expected_cf:
        raise RuntimeError(f"CounterFact SHA256 mismatch: expected {expected_cf}, got {actual_cf}")
    inputs = manifest_inputs()
    hashes = {label: {"path": str(path), "sha256": hash_file(path)} for label, path in inputs.items()}
    data = {"created_utc": utc_now(), "easyedit_commit": EASYEDIT_COMMIT, "wsp_commit": "763259257133",
            "wsp_expected_sha256": WSP_SHA256, "gpt2_expected_sha256": GPT2_SHA256,
            "counterfact_expected_sha256": expected_cf,
            "inputs": hashes}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                                           encoding="utf-8")
    return data


def stats_helpers():
    return load_stats_module()


def pair_to_dict(p: Any, decode=None) -> dict[str, Any]:
    d = {"j": int(p.j), "sign": int(p.sign), "z": float(p.z), "top": int(p.top),
         "top3": [int(x) for x in p.top3], "eligible": bool(p.eligible)}
    if decode is not None:
        d["token"] = decode([int(p.top)])
        d["top3_tokens"] = [decode([int(x)]) for x in p.top3]
    return d


def pairs_to_dict(pairs: list[Any], decode=None) -> list[dict[str, Any]]:
    return [pair_to_dict(p, decode) for p in pairs]


def stats_for(W, gain, centre, E, decode, S, pcs: bool = True, k: int = 64) -> dict[str, Any]:
    f1, f2 = S.f_scores(W)
    pcsv = S.pcs(W) if pcs else None
    pairs = S.read_pairs(W, gain, centre, E, decode, k=k)
    return {"f1": f1, "f2": f2, "pcs": pcsv, "t_max": S.t_max(pairs),
            "t_word": S.t_word(pairs), "pairs": pairs_to_dict(pairs, decode)}


def dict_to_pairs(items: list[dict[str, Any]], S) -> list[Any]:
    return [S.Pair(int(x["j"]), int(x["sign"]), float(x["z"]), int(x["top"]),
                   [int(t) for t in x["top3"]], bool(x["eligible"])) for x in items]


def statistic_map(base: dict[int, dict[str, Any]], current: dict[int, dict[str, Any]], stat: str) -> dict[int, float | None]:
    return {l: (current[l][stat] if l in current else base[l][stat]) for l in base}


def localization(base: dict[int, dict[str, Any]], current: dict[int, dict[str, Any]], edited: set[int], S) -> dict[str, Any]:
    out = {}
    for label in ("f1", "f2", "pcs", "t_max", "t_word"):
        loc, first, tie = S.localized(statistic_map(base, current, label), edited)
        out[label] = {"localized": bool(loc), "first_layer": first, "tie_for_first": bool(tie)}
    return out


def pair_by_layer(base: dict[int, dict[str, Any]], current: dict[int, dict[str, Any]], S) -> dict[int, list[Any]]:
    return {int(l): dict_to_pairs((current.get(l) or base[l])["pairs"], S) for l in base}


def unique_top_layer(values: dict[int, float | None]) -> tuple[int | None, bool]:
    valid = {l: v for l, v in values.items() if v is not None}
    if not valid:
        return None, False
    maximum = max(valid.values())
    first = [l for l, v in valid.items() if v == maximum]
    return (first[0], False) if len(first) == 1 else (None, True)


def oracle_pair(current: dict[int, dict[str, Any]], edited: set[int], S) -> tuple[int | None, Any | None, bool]:
    values = {l: current[l]["t_word"] for l in edited if l in current}
    layer, tie = unique_top_layer(values)
    if layer is None:
        return None, None, tie
    return layer, S.top_eligible(dict_to_pairs(current[layer]["pairs"], S)), False


def target_info_gpt2(tokenizer, text: str, S) -> dict[str, Any]:
    encs = []
    for s in (" " + text, text):
        seq = tokenizer(s, add_special_tokens=False).input_ids
        encs.append(int(seq[0]) if seq else None)
    ids = {x for x in encs if x is not None}
    eligible = {t for t in ids if S.is_content(tokenizer.decode([t]))}
    return {"target_ids": sorted(ids), "eligible_target_ids": sorted(eligible),
            "space_first_token_id": encs[0], "raw_first_token_id": encs[1],
            "name_eligible": bool(eligible)}


def target_info_wsp(w, text: str, S) -> dict[str, Any]:
    encs = []
    for s in (text, " " + text):
        seq = w.enc(s)
        encs.append(int(seq[0]) if seq else None)
    ids = {x for x in encs if x is not None}
    eligible = {t for t in ids if S.is_content(w.tok.decode([t]))}
    return {"target_ids": sorted(ids), "eligible_target_ids": sorted(eligible),
            "answer_first_token_id": encs[0], "space_first_token_id": encs[1],
            "name_eligible": bool(eligible)}


def base_diagnostics(base: dict[int, dict[str, Any]], edited: set[int], targets: set[int], S) -> dict[str, Any]:
    diag = {}
    vals = {l: base[l]["t_word"] for l in base}
    valid = [x for x in vals.values() if x is not None]
    for l in sorted(edited):
        v = vals[l]
        rank = 1 + sum(x > v for x in valid) if v is not None else None
        ties = sum(x == v for x in valid) if v is not None else 0
        bp = S.top_eligible(dict_to_pairs(base[l]["pairs"], S))
        named, loose = S.names(bp, targets)
        diag[str(l)] = {"base_t_word": v, "base_t_word_rank": rank,
                        "base_t_word_rank_ties": ties, "base_already_names": named,
                        "base_already_names_loose": loose}
    return diag


def load_gpt2_model():
    ensure_offline()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(12)
    if hash_file(GPT2 / "model.safetensors") != GPT2_SHA256:
        raise RuntimeError("GPT-2 XL model.safetensors SHA256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(GPT2), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(GPT2), dtype=torch.float32,
                                                  local_files_only=True).cpu().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.clone()
        for _, b in model.named_buffers():
            b.data = b.data.clone()
    return model, tokenizer


def gpt2_geometry(model, tokenizer, S, compute_base: bool = True):
    E = model.get_output_embeddings().weight.detach().double().cpu()
    gain, centre = S.norm_reader(model.transformer.ln_f)
    gain = gain.detach().double().cpu()
    base = {}
    if compute_base:
        params = dict(model.named_parameters())
        for layer in range(N_GPT2):
            pname = f"transformer.h.{layer}.mlp.c_proj.weight"
            W = params[pname].detach().T.contiguous()
            base[layer] = stats_for(W, gain, centre, E, tokenizer.decode, S)
    return {"gain": gain, "centre": centre, "E": E, "base": base}


def save_base(path: Path, model_id: str, base: dict[int, dict[str, Any]],
              inputs_sha256: str | None = None) -> None:
    data = {"model": model_id, "layers": {str(l): base[l] for l in sorted(base)}}
    if inputs_sha256 is not None:
        data["inputs_sha256"] = inputs_sha256
    path.write_text(json.dumps(safe_json(data), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_base(path: Path, expected_inputs_sha256: str | None = None) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if expected_inputs_sha256 is not None and raw.get("inputs_sha256") != expected_inputs_sha256:
        raise RuntimeError(f"base statistics input hash mismatch in {path}: "
                           f"expected {expected_inputs_sha256}, got {raw.get('inputs_sha256')}")
    return {int(l): item for l, item in raw["layers"].items()}


def efficacy(model, tokenizer, req: dict[str, Any]) -> dict[str, Any]:
    import torch
    ids = tokenizer(req["prompt"], return_tensors="pt").input_ids
    with torch.no_grad():
        logp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
    new_id = tokenizer(" " + req["target_new"], add_special_tokens=False).input_ids[0]
    old_id = tokenizer(" " + req["ground_truth"], add_special_tokens=False).input_ids[0]
    return {"logp_new": float(logp[new_id]), "logp_old": float(logp[old_id]),
            "new_leads": bool(logp[new_id] > logp[old_id]), "old_leads": bool(logp[old_id] > logp[new_id]),
            "tie": bool(logp[new_id] == logp[old_id])}


def request_of(rec: dict[str, Any]) -> dict[str, Any]:
    rr = rec["requested_rewrite"]
    return {"prompt": rr["prompt"].format(rr["subject"]), "subject": rr["subject"],
            "target_new": rr["target_new"]["str"], "ground_truth": rr["target_true"]["str"],
            "case_id": int(rec["case_id"])}


def load_easyedit():
    sys.path.insert(0, str(HERE))
    import ee_import
    return ee_import.load_methods()


def load_hparams(cls, method: str, out_dir: Path):
    import yaml
    raw = yaml.safe_load((EASYEDIT / "hparams" / method / "gpt2-xl.yaml").read_text(encoding="utf-8"))
    raw.update({"model_name": str(GPT2), "stats_dir": str(STATS_DIR), "device": "cpu"})
    temp = out_dir / f".runtime-hparams-{method.lower()}-{os.getpid()}.yaml"
    temp.write_text(yaml.safe_dump(raw), encoding="utf-8")
    try:
        return cls.from_hparams(str(temp))
    finally:
        temp.unlink(missing_ok=True)


def update_metrics(delta, base_sigma1: float) -> dict[str, Any]:
    import torch
    D = delta.detach().to(dtype=torch.float64)
    sv = torch.linalg.svdvals(D)
    spectral = float(sv[0]) if sv.numel() else 0.0
    frob = float(torch.linalg.vector_norm(D))
    rank = int(torch.linalg.matrix_rank(D)) if D.numel() else 0
    return {"delta_spectral": spectral, "delta_frobenius": frob,
            "ratio_spectral_to_base_sigma1": spectral / base_sigma1 if base_sigma1 else None,
            "numerical_rank": rank,
            "rank_tolerance": "torch.linalg.matrix_rank default tolerance on float64 update"}


def make_aimed_delta(rome_delta, E, gain, target_id):
    import torch
    D = rome_delta.detach().to(dtype=torch.float64, device="cpu")
    _, singular, Vh = torch.linalg.svd(D, full_matrices=False)
    spectral = singular[0]
    right = Vh[0]
    output = aimed_output_direction(E, gain, target_id)
    if output is None:
        return None, {"reason": "zero gamma-scaled centred unembedding row"}
    return float(spectral) * torch.outer(output, right), {
        "input_right_direction_norm": float(right.norm()), "spectral_norm": float(spectral),
        "output_direction_norm": float(output.norm()), "target_token_id": int(target_id)}


def aimed_output_direction(E, gain, target_id):
    """Return unit gamma * (unembedding row - its mean), as specified for S1."""
    import torch
    row = E[target_id].detach().double().cpu()
    centred = row - row.mean()
    output = gain.detach().double().cpu() * centred
    norm = output.norm()
    if not float(norm) > 0:
        return None
    return output / norm


def run_arm_a(method: str, limit: int | None, out_dir: Path) -> int:
    import torch
    S = stats_helpers()
    frozen, cf_by_id = load_cases()
    case_ids = frozen["armA_single"]
    log = log_path_for("armA", method, out_dir)
    runmode = "smoke" if limit is not None else "registered"
    rid = run_id()
    prior = existing_keys(log, "case_id")
    pending = [c for c in case_ids if str(c) not in prior]
    append_run_event(log, runmode, method, pending, rid)
    model, tok = load_gpt2_model()
    base_path = out_dir / "base-gpt2.json"
    base_inputs_sha256 = gpt2_base_inputs_sha256()
    geometry = gpt2_geometry(model, tok, S, compute_base=not base_path.exists())
    base = geometry["base"] if not base_path.exists() else load_base(base_path, base_inputs_sha256)
    if not base_path.exists():
        save_base(base_path, "gpt2-xl", base, base_inputs_sha256)
    methods = load_easyedit()
    module = methods[method]
    clsname = {"ROME": "ROMEHyperParams", "MEMIT": "MEMITHyperParams",
               "AlphaEdit": "AlphaEditHyperParams"}[method]
    applyname = {"ROME": "apply_rome_to_model", "MEMIT": "apply_memit_to_model",
                 "AlphaEdit": "apply_AlphaEdit_to_model"}[method]
    hp = load_hparams(getattr(module, clsname), method, out_dir)
    apply = getattr(module, applyname)
    params = dict(model.named_parameters())
    state_snapshot = model_state_snapshot(model)
    selected = pending if limit is None else pending[:limit]
    first_ref_path = out_dir / f"first-update-{method.lower()}-{runmode}.pt"
    first_reference = torch.load(first_ref_path, map_location="cpu", weights_only=True) if first_ref_path.exists() else None
    for case_id in selected:
        orig = {}
        rec = cf_by_id.get(int(case_id))
        if rec is None:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                               "case_id": int(case_id), "method": method, "phase": "counterfact_lookup",
                               "error": "case id not found in frozen CounterFact input", "timestamp": utc_now()})
            continue
        req = request_of(rec)
        targets = target_info_gpt2(tok, req["target_new"], S)
        before = efficacy(model, tok, req)
        editor_started = False
        try:
            kwargs = {"return_orig_weights": True}
            if method == "AlphaEdit":
                kwargs["reset_cache"] = True
            editor_started = True
            result = apply(model, tok, copy.deepcopy([req]), hp, **kwargs)
            orig = result[1] if isinstance(result, tuple) else result
            after = efficacy(model, tok, req)
            took, already = took_transition(before["logp_old"], before["logp_new"],
                                            after["logp_old"], after["logp_new"])
            current, delta_oriented, updates, per_layer = {}, {}, {}, {}
            for pname, w0 in orig.items():
                if pname not in params:
                    raise KeyError(f"EasyEdit returned unknown parameter {pname}")
                w1 = params[pname].detach()
                delta_native = (w1 - w0.to(device=w1.device, dtype=w1.dtype)).detach()
                m = re.search(r"transformer\.h\.(\d+)\.mlp\.c_proj\.weight$", pname)
                if not m:
                    raise ValueError(f"unexpected edited matrix role: {pname}")
                layer = int(m.group(1))
                delta = delta_native.T.contiguous()
                W = w1.T.contiguous()
                current[layer] = stats_for(W, geometry["gain"], geometry["centre"], geometry["E"], tok.decode, S)
                metrics = update_metrics(delta, base[layer]["f1"])
                d_stats = S.read_pairs(delta, geometry["gain"], geometry["centre"], geometry["E"], tok.decode, k=1)
                d_top = S.top_eligible(d_stats)
                d_strict, d_loose = S.names(d_top, set(targets["target_ids"]))
                metrics.update({"param": pname, "layer": layer,
                                "update_top_pair": pair_to_dict(d_top, tok.decode) if d_top else None,
                                "update_top_names_strict": d_strict, "update_top_names_loose": d_loose,
                                "base_sigma1": base[layer]["f1"], "edited_f1": current[layer]["f1"],
                                "edited_f2": current[layer]["f2"], "edited_pcs": current[layer]["pcs"],
                                "edited_t_max": current[layer]["t_max"], "edited_t_word": current[layer]["t_word"]})
                per_layer[str(layer)] = metrics
                delta_oriented[layer] = delta.detach().float().cpu()
                updates[pname] = delta_native.detach().float().cpu()
            edited = set(delta_oriented)
            if not edited:
                raise RuntimeError("EasyEdit returned no edited matrices")
            loc = localization(base, current, edited, S)
            allpairs = pair_by_layer(base, current, S)
            blind_l, blind_pair = S.blind_reader(allpairs)
            blind_named, blind_loose = S.names(blind_pair, set(targets["target_ids"]))
            oracle_l, oracle, oracle_tie = oracle_pair(current, edited, S)
            oracle_named, oracle_loose = S.names(oracle, set(targets["target_ids"]))
            record = {"record_type": "case", "run_mode": runmode, "run_id": rid,
                      "timestamp": utc_now(), "arm": "A", "method": method,
                      "case_id": int(case_id), "subject": req["subject"],
                      "target_new": req["target_new"], "ground_truth": req["ground_truth"],
                      "prompt": req["prompt"], "before": before, "after": after,
                      "took": bool(took), "already_satisfied": bool(already),
                      "name_eligible": bool(targets["name_eligible"]),
                      "target_ids": targets["target_ids"],
                      "eligible_target_ids": targets["eligible_target_ids"],
                      "edited_layers": sorted(edited), "edited_layer_stats": per_layer,
                      "localization": loc,
                      "blind_reader": {"layer": blind_l,
                          "pair": pair_to_dict(blind_pair, tok.decode) if blind_pair else None,
                          "localized": blind_l in edited if blind_l is not None else False,
                          "named_strict": blind_named, "named_loose": blind_loose},
                      "oracle_reader": {"layer": oracle_l,
                          "pair": pair_to_dict(oracle, tok.decode) if oracle else None,
                          "tie": bool(oracle_tie), "named_strict": oracle_named, "named_loose": oracle_loose},
                      "base_diagnostics": base_diagnostics(base, edited, set(targets["target_ids"]), S),
                      "restored_bit_exact": None}
            if method == "ROME":
                layer = 17
                target = targets["space_first_token_id"]
                aimed_delta, aim_meta = make_aimed_delta(delta_oriented[layer], geometry["E"],
                                                          geometry["gain"], target)
                if aimed_delta is None:
                    record["aimed_positive_control"] = aim_meta
                else:
                    pname = f"transformer.h.{layer}.mlp.c_proj.weight"
                    W0 = orig[pname].detach().T.contiguous()
                    aimed_current = {layer: stats_for(W0 + aimed_delta.to(W0.dtype), geometry["gain"],
                                     geometry["centre"], geometry["E"], tok.decode, S)}
                    aimed_loc = localization(base, aimed_current, {layer}, S)
                    _, opair, otie = oracle_pair(aimed_current, {layer}, S)
                    oname, oloose = S.names(opair, set(targets["target_ids"]))
                    record["aimed_positive_control"] = {**aim_meta, "layer": layer,
                        "localization": aimed_loc["t_word"],
                        "oracle_pair": pair_to_dict(opair, tok.decode) if opair else None,
                        "oracle_named_strict": oname, "oracle_named_loose": oloose,
                        "oracle_tie": bool(otie)}
            if first_reference is None and not first_ref_path.exists():
                tmp = first_ref_path.with_suffix(first_ref_path.suffix + ".tmp")
                torch.save(updates, tmp)
                os.replace(tmp, first_ref_path)
                first_reference = updates
            with torch.no_grad():
                for pname, w0 in orig.items():
                    params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
            restored = model_matches_snapshot(model, state_snapshot)
            record["restored_bit_exact"] = bool(restored)
            append_jsonl(log, record)
            if not restored:
                raise RuntimeError(f"base restoration check failed on case {case_id}")
        except Exception as exc:
            restored_after_error = True
            try:
                with torch.no_grad():
                    for pname, w0 in orig.items():
                        if pname in params:
                            params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
                restored_after_error = model_matches_snapshot(model, state_snapshot)
            except Exception:
                restored_after_error = False
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                               "timestamp": utc_now(), "arm": "A", "method": method,
                               "case_id": int(case_id), "phase": "edit_or_score",
                               "restored_bit_exact": bool(restored_after_error),
                               "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
            if (editor_started and not orig) or not restored_after_error:
                raise RuntimeError("editor failure left no verified restoration; stopping this process") from exc
    rows = read_jsonl(log)
    done = {int(r["case_id"]) for r in rows if r.get("run_mode", "registered") == "registered"
            and r.get("record_type") in {"case", "failure"} and r.get("case_id") is not None}
    if runmode == "registered" and len(done) >= len(case_ids):
        first_id = int(case_ids[0])
        verify_orig = {}
        try:
            if not first_ref_path.exists():
                raise RuntimeError("first update reference is missing; cannot verify editor-state reproducibility")
            first_reference = torch.load(first_ref_path, map_location="cpu", weights_only=True)
            req = request_of(cf_by_id[first_id])
            kwargs = {"return_orig_weights": True}
            if method == "AlphaEdit":
                kwargs["reset_cache"] = True
            result = apply(model, tok, copy.deepcopy([req]), hp, **kwargs)
            verify_orig = result[1] if isinstance(result, tuple) else result
            new_updates = {p: (params[p].detach() - w0.to(device=params[p].device, dtype=params[p].dtype)).float().cpu()
                           for p, w0 in verify_orig.items()}
            compares, passed = {}, True
            for pname, ref in first_reference.items():
                got = new_updates.get(pname)
                if got is None:
                    passed = False
                    compares[pname] = {"missing": True}
                    continue
                diff = float((got - ref.float()).abs().max())
                largest = float(ref.float().abs().max())
                tolerance = 1e-6 * largest
                ok = diff == 0.0 if largest == 0.0 else diff < tolerance
                passed &= ok
                compares[pname] = {"max_abs_difference": diff, "reference_largest_abs_entry": largest,
                                   "tolerance": tolerance, "passed": bool(ok)}
            with torch.no_grad():
                for pname, w0 in verify_orig.items():
                    params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
            restored = model_matches_snapshot(model, state_snapshot)
            append_jsonl(log, {"record_type": "state_verification", "run_mode": runmode, "run_id": rid,
                               "timestamp": utc_now(), "method": method, "case_id": first_id,
                               "update_reproduced": bool(passed), "per_parameter": compares,
                               "restored_bit_exact": bool(restored),
                               "possibly_state_contaminated": not (passed and restored)})
        except Exception as exc:
            try:
                with torch.no_grad():
                    for pname, w0 in verify_orig.items():
                        if pname in params:
                            params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
            except Exception:
                pass
            try:
                restored_after_error = model_matches_snapshot(model, state_snapshot)
            except Exception:
                restored_after_error = False
            append_jsonl(log, {"record_type": "state_verification", "run_mode": runmode, "run_id": rid,
                               "timestamp": utc_now(), "method": method, "case_id": first_id,
                               "update_reproduced": False, "restored_bit_exact": bool(restored_after_error),
                               "possibly_state_contaminated": True, "error": f"{type(exc).__name__}: {exc}"})
    del model
    gc.collect()
    return 0


def run_arm_a_mass(method: str, limit: int | None, out_dir: Path) -> int:
    import torch
    S = stats_helpers()
    frozen, cf_by_id = load_cases()
    batches = frozen["armA_mass"]
    log = log_path_for("armA-mass", method, out_dir)
    runmode, rid = ("smoke" if limit is not None else "registered"), run_id()
    prior = existing_keys(log, "batch_id")
    todo = [(i, ids) for i, ids in enumerate(batches) if f"{method}:{i}" not in prior]
    append_run_event(log, runmode, method, [f"{method}:{i}" for i, _ in todo], rid)
    model, tok = load_gpt2_model()
    base_path = out_dir / "base-gpt2.json"
    base_inputs_sha256 = gpt2_base_inputs_sha256()
    geometry = gpt2_geometry(model, tok, S, compute_base=not base_path.exists())
    base = geometry["base"] if not base_path.exists() else load_base(base_path, base_inputs_sha256)
    if not base_path.exists():
        save_base(base_path, "gpt2-xl", base, base_inputs_sha256)
    module = load_easyedit()[method]
    clsname = {"MEMIT": "MEMITHyperParams", "AlphaEdit": "AlphaEditHyperParams"}[method]
    applyname = {"MEMIT": "apply_memit_to_model", "AlphaEdit": "apply_AlphaEdit_to_model"}[method]
    hp, apply = load_hparams(getattr(module, clsname), method, out_dir), getattr(module, applyname)
    params = dict(model.named_parameters())
    state_snapshot = model_state_snapshot(model)
    for batch_idx, batch_ids in (todo if limit is None else todo[:limit]):
        orig = {}
        editor_started = False
        try:
            reqs = [request_of(cf_by_id[int(cid)]) for cid in batch_ids]
            before = [efficacy(model, tok, r) for r in reqs]
            kwargs = {"return_orig_weights": True}
            if method == "AlphaEdit":
                kwargs["reset_cache"] = True
            editor_started = True
            result = apply(model, tok, copy.deepcopy(reqs), hp, **kwargs)
            orig = result[1] if isinstance(result, tuple) else result
            after = [efficacy(model, tok, r) for r in reqs]
            transitions = [took_transition(b["logp_old"], b["logp_new"], a["logp_old"], a["logp_new"])
                           for b, a in zip(before, after)]
            tooks, already = [x[0] for x in transitions], [x[1] for x in transitions]
            current, delta_by_layer, layer_stats = {}, {}, {}
            for pname, w0 in orig.items():
                m = re.search(r"transformer\.h\.(\d+)\.mlp\.c_proj\.weight$", pname)
                if not m:
                    raise ValueError(f"unexpected edited matrix role: {pname}")
                layer, w1 = int(m.group(1)), params[pname].detach()
                delta = (w1 - w0.to(device=w1.device, dtype=w1.dtype)).T.contiguous()
                delta_by_layer[layer] = delta
                current[layer] = stats_for(w1.T.contiguous(), geometry["gain"], geometry["centre"],
                                           geometry["E"], tok.decode, S)
                layer_stats[str(layer)] = {**update_metrics(delta, base[layer]["f1"]),
                                           **{k: current[layer][k] for k in ("f1", "f2", "pcs", "t_max", "t_word")}}
            edited = set(delta_by_layer)
            fact_results = []
            for req, b, a, did_take, was_satisfied in zip(reqs, before, after, tooks, already):
                info = target_info_gpt2(tok, req["target_new"], S)
                strict, loose, details = False, False, []
                for layer in edited:
                    for pair in dict_to_pairs(current[layer]["pairs"], S):
                        if not pair.eligible:
                            continue
                        one_strict, one_loose = S.names(pair, set(info["target_ids"]))
                        strict |= one_strict
                        loose |= one_loose
                        if one_strict:
                            details.append({"layer": layer, "pair": pair_to_dict(pair, tok.decode)})
                fact_results.append({"case_id": req["case_id"], "subject": req["subject"],
                    "target_new": req["target_new"], "ground_truth": req["ground_truth"],
                    "before": b, "after": a, "took": did_take, "already_satisfied": was_satisfied,
                    "name_eligible": info["name_eligible"], "target_ids": info["target_ids"],
                    "named_strict_by_any_edited_pair": strict, "named_loose_by_any_edited_pair": loose,
                    "strict_pairs": details})
            batch_record = {"record_type": "batch", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "A-mass", "method": method,
                "batch_id": f"{method}:{batch_idx}", "case_ids": [int(x) for x in batch_ids],
                "facts": fact_results, "n_took": sum(tooks), "n_already_satisfied": sum(already),
                "take_gate_minimum": 8, "take_gate_passed": mass_gate(tooks),
                "localization": localization(base, current, edited, S), "edited_layers": sorted(edited),
                "edited_layer_stats": layer_stats, "restored_bit_exact": None}
            with torch.no_grad():
                for pname, w0 in orig.items():
                    params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
            restored = model_matches_snapshot(model, state_snapshot)
            batch_record["restored_bit_exact"] = bool(restored)
            append_jsonl(log, batch_record)
            if not restored:
                raise RuntimeError(f"base restoration check failed after mass batch {batch_idx}")
        except Exception as exc:
            restored_after_error = True
            try:
                with torch.no_grad():
                    for pname, w0 in orig.items():
                        if pname in params:
                            params[pname].copy_(w0.to(device=params[pname].device, dtype=params[pname].dtype))
                restored_after_error = model_matches_snapshot(model, state_snapshot)
            except Exception:
                restored_after_error = False
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "A-mass", "method": method,
                "batch_id": f"{method}:{batch_idx}", "case_ids": [int(x) for x in batch_ids],
                "phase": "mass_edit_or_score", "restored_bit_exact": bool(restored_after_error),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]})
            if (editor_started and not orig) or not restored_after_error:
                raise RuntimeError("mass editor failure left no verified restoration; stopping this process") from exc
    del model
    gc.collect()
    return 0


def load_wsp_module():
    actual = hash_file(WSP_SRC)
    if actual != WSP_SHA256:
        raise RuntimeError(f"wsp.py SHA256 mismatch: expected {WSP_SHA256}, got {actual}; refusing to import")
    spec = importlib.util.spec_from_file_location("wsp198_pinned", WSP_SRC)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import wsp.py by path: {WSP_SRC}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_wsp_base(w, S):
    base = {}
    for layer in range(len(w.layers)):
        W = w.layers[layer].mlp.down_proj.weight.detach().float().cpu()
        base[layer] = stats_for(W, w.g.detach().double().cpu(), False,
                                w.head.detach().double().cpu(), w.tok.decode, S)
    return base


def wsp_question_data(w, rec):
    rr = rec["requested_rewrite"]
    subj, obj = rr["subject"], rr["target_new"]["str"]
    relation = rr["relation_id"]
    asks_template, heldout_template = TEMPLATES[relation]
    return subj, obj, [q.format(subj) for q in asks_template], heldout_template.format(subj)


def run_arm_b(limit: int | None, out_dir: Path) -> int:
    import torch
    S = stats_helpers()
    frozen, cf_by_id = load_cases()
    cases = frozen["armB_single"]
    log = log_path_for("armB", None, out_dir)
    runmode, rid = ("smoke" if limit is not None else "registered"), run_id()
    prior = existing_keys(log, "case_id")
    pending = [c for c in cases if str(c) not in prior]
    append_run_event(log, runmode, "wsp", pending, rid)
    ensure_offline()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(out_dir)
    model_path = str(HF_HUB / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / QWEN35_SNAPSHOT)
    w = load_wsp_module().WSP(model_path, device="cpu")
    torch.set_num_threads(12)
    base_path = out_dir / "base-qwen35.json"
    base = build_wsp_base(w, S) if not base_path.exists() else load_base(base_path)
    if not base_path.exists():
        save_base(base_path, "Qwen/Qwen3.5-0.8B", base)
    edited_layers = list(w.Ls)
    edited_base = {l: w.layers[l].mlp.down_proj.weight.detach().float().cpu().clone() for l in edited_layers}
    for case_id in (pending if limit is None else pending[:limit]):
        rec = cf_by_id.get(int(case_id))
        if rec is None:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B", "method": "wsp", "case_id": int(case_id),
                "phase": "counterfact_lookup", "error": "case id not found in frozen CounterFact input"})
            continue
        subj, obj, asks, heldout = wsp_question_data(w, rec)
        targets = target_info_wsp(w, obj, S)
        target_ids = set(targets["target_ids"])
        try:
            before_answers = [w.answer(q, False, n=24) for q in asks]
            before_exact = sum(a.strip() == obj.strip() for a in before_answers)
            heldout_before = w.answer(heldout, False, n=24)
            teach = w.teach(asks, obj)
            after_answers = [w.answer(q, True, n=24) for q in asks]
            after_exact = sum(a.strip() == obj.strip() for a in after_answers)
            take_measurement = wsp_take_measurement(before_exact, after_exact, len(asks),
                                                     teach.get("free_exact"))
            took, already = take_measurement["took"], take_measurement["already_satisfied"]
            heldout_after = w.answer(heldout, True, n=24)
            current, deltas = {}, {}
            for layer in edited_layers:
                delta = w.delta[layer].detach().float().cpu().clone()
                deltas[layer] = delta
                current[layer] = stats_for(edited_base[layer] + delta,
                    w.g.detach().double().cpu(), False, w.head.detach().double().cpu(), w.tok.decode, S)
            loc = localization(base, current, set(edited_layers), S)
            allpairs = pair_by_layer(base, current, S)
            blind_layer, blind_pair = S.blind_reader(allpairs)
            blind_strict, blind_loose = S.names(blind_pair, target_ids)
            oracle_layer, oracle_value, oracle_tie = oracle_pair(current, set(edited_layers), S)
            oracle_strict, oracle_loose = S.names(oracle_value, target_ids)
            consensus_n, consensus_tokens = S.consensus_run(allpairs)
            random_deltas = random_control_layers(edited_base, deltas, 198 + int(case_id))
            random_current = {l: stats_for(edited_base[l] + random_deltas[l].float(),
                w.g.detach().double().cpu(), False, w.head.detach().double().cpu(), w.tok.decode, S)
                for l in edited_layers}
            random_loc = localization(base, random_current, set(edited_layers), S)
            random_pairs = pair_by_layer(base, random_current, S)
            random_blind_layer, random_blind_pair = S.blind_reader(random_pairs)
            random_blind_named, random_blind_loose = S.names(random_blind_pair, target_ids)
            metrics = {str(l): update_metrics(deltas[l], base[l]["f1"]) for l in edited_layers}
            record = {"record_type": "case", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B", "method": "wsp", "case_id": int(case_id),
                "subject": subj, "target_new": obj, "relation_id": rec["requested_rewrite"]["relation_id"],
                "questions": asks, "heldout_question": heldout, "before_answers": before_answers,
                "after_answers": after_answers, "heldout_before": heldout_before, "heldout_after": heldout_after,
                "before_exact": before_exact, "after_exact": after_exact,
                "free_exact": take_measurement["free_exact_text"],
                "free_exact_count": take_measurement["free_exact_count"],
                "free_exact_total": take_measurement["free_exact_total"],
                "free_exact_disagreement": take_measurement["free_exact_disagreement"],
                "free_exact_parse_error": take_measurement["free_exact_parse_error"], "teach": teach,
                "took": bool(took), "already_satisfied": bool(already),
                "name_eligible": bool(targets["name_eligible"]), "target_ids": targets["target_ids"],
                "eligible_target_ids": targets["eligible_target_ids"], "edited_layers": edited_layers,
                "edited_layer_stats": {str(l): current[l] for l in edited_layers}, "delta_metrics": metrics,
                "localization": loc,
                "blind_reader": {"layer": blind_layer,
                    "pair": pair_to_dict(blind_pair, w.tok.decode) if blind_pair else None,
                    "localized": blind_layer in edited_layers if blind_layer is not None else False,
                    "named_strict": blind_strict, "named_loose": blind_loose},
                "oracle_reader": {"layer": oracle_layer,
                    "pair": pair_to_dict(oracle_value, w.tok.decode) if oracle_value else None,
                    "tie": bool(oracle_tie), "named_strict": oracle_strict, "named_loose": oracle_loose},
                "consensus": {"run_length": consensus_n, "token_ids": consensus_tokens,
                    "tokens": [w.tok.decode([t]) for t in consensus_tokens]},
                "random_control": {"seed": 198 + int(case_id), "localization": random_loc,
                    "blind_reader": {"layer": random_blind_layer,
                        "pair": pair_to_dict(random_blind_pair, w.tok.decode) if random_blind_pair else None,
                        "localized": random_blind_layer in edited_layers if random_blind_layer is not None else False,
                        "named_strict": random_blind_named, "named_loose": random_blind_loose}},
                "base_diagnostics": base_diagnostics(base, set(edited_layers), target_ids, S),
                "undo_zero_exact": None}
            undone = w.undo()
            zero = all(torch.equal(w.delta[l], torch.zeros_like(w.delta[l])) for l in edited_layers)
            record["undo_zero_exact"], record["undo_answer"] = bool(zero), undone
            append_jsonl(log, record)
            if not zero:
                raise RuntimeError(f"WSP accumulated update was nonzero after undo for case {case_id}")
        except Exception as exc:
            try:
                while w.facts:
                    w.undo()
                for layer in edited_layers:
                    w.delta[layer].zero_()
            except Exception:
                pass
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B", "method": "wsp", "case_id": int(case_id),
                "phase": "write_or_score", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]})
    del w
    gc.collect()
    return 0


def run_arm_b_seq(limit: int | None, out_dir: Path) -> int:
    import torch
    S = stats_helpers()
    frozen, cf_by_id = load_cases()
    cases = frozen["armB_sequence"]
    log = log_path_for("armB-seq", None, out_dir)
    runmode, rid = ("smoke" if limit is not None else "registered"), run_id()
    prior = existing_keys(log, "case_id")
    old_rows = read_jsonl(log)
    prior_failures = {str(r["case_id"]) for r in old_rows
                      if r.get("run_mode", "registered") == "registered"
                      and r.get("record_type") == "failure" and r.get("case_id") is not None}
    pending = [c for c in cases if str(c) not in prior]
    append_run_event(log, runmode, "wsp-sequence", pending, rid)
    if runmode == "registered" and prior_failures:
        raise RuntimeError("sequence has terminal failure records and cannot be replayed as a complete sequence")
    ensure_offline()
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(out_dir)
    model_path = str(HF_HUB / "models--Qwen--Qwen3.5-0.8B" / "snapshots" / QWEN35_SNAPSHOT)
    w = load_wsp_module().WSP(model_path, device="cpu")
    torch.set_num_threads(12)
    base_path = out_dir / "base-qwen35.json"
    base = build_wsp_base(w, S) if not base_path.exists() else load_base(base_path)
    if not base_path.exists():
        save_base(base_path, "Qwen/Qwen3.5-0.8B", base)
    edited_layers = list(w.Ls)
    edited_base = {l: w.layers[l].mlp.down_proj.weight.detach().float().cpu().clone() for l in edited_layers}
    # Replay the whole frozen prefix to reconstruct WSP's sequence state after a process restart.
    selected = cases if limit is None else cases[:limit]
    all_facts: list[dict[str, Any]] = []
    for case_id in selected:
        rec = cf_by_id.get(int(case_id))
        if rec is None:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B-seq", "method": "wsp", "case_id": int(case_id),
                "phase": "counterfact_lookup", "error": "case id not found in frozen CounterFact input"})
            break
        subj, obj, asks, heldout = wsp_question_data(w, rec)
        info = target_info_wsp(w, obj, S)
        try:
            before = [w.answer(q, False, n=24) for q in asks]
            before_exact = sum(a.strip() == obj.strip() for a in before)
            teach = w.teach(asks, obj)
            after = [w.answer(q, True, n=24) for q in asks]
            after_exact = sum(a.strip() == obj.strip() for a in after)
            take_measurement = wsp_take_measurement(before_exact, after_exact, len(asks),
                                                     teach.get("free_exact"))
            took, already = take_measurement["took"], take_measurement["already_satisfied"]
            fact = {"case_id": int(case_id), "subject": subj, "target_new": obj,
                    "relation_id": rec["requested_rewrite"]["relation_id"], "asks": asks, "heldout": heldout,
                    "before_answers": before, "after_answers": after, "before_exact": before_exact,
                    "after_exact": after_exact, "free_exact": take_measurement["free_exact_text"],
                    "free_exact_count": take_measurement["free_exact_count"],
                    "free_exact_total": take_measurement["free_exact_total"],
                    "free_exact_disagreement": take_measurement["free_exact_disagreement"],
                    "free_exact_parse_error": take_measurement["free_exact_parse_error"],
                    "took": took, "already_satisfied": already,
                    "name_eligible": info["name_eligible"], "target_ids": info["target_ids"],
                    "eligible_target_ids": info["eligible_target_ids"]}
            all_facts.append(fact)
            current = {l: stats_for(edited_base[l] + w.delta[l].detach().float().cpu(),
                    w.g.detach().double().cpu(), False, w.head.detach().double().cpu(), w.tok.decode, S)
                    for l in edited_layers}
            names = {}
            for written in all_facts:
                targets = set(written["target_ids"])
                strict, loose, details = False, False, []
                for layer in edited_layers:
                    for pair in dict_to_pairs(current[layer]["pairs"], S):
                        if pair.eligible:
                            s, l = S.names(pair, targets)
                            strict |= s
                            loose |= l
                            if s:
                                details.append({"layer": layer, "pair": pair_to_dict(pair, w.tok.decode)})
                names[str(written["case_id"])] = {"named_strict": strict, "named_loose": loose, "pairs": details}
            row = {"record_type": "case", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B-seq", "method": "wsp", "case_id": int(case_id),
                "subject": subj, "target_new": obj, "took": bool(took),
                "already_satisfied": bool(already), "name_eligible": bool(info["name_eligible"]),
                "target_ids": info["target_ids"], "free_exact": take_measurement["free_exact_text"],
                "after_exact": after_exact, "free_exact_count": take_measurement["free_exact_count"],
                "free_exact_total": take_measurement["free_exact_total"],
                "free_exact_disagreement": take_measurement["free_exact_disagreement"],
                "free_exact_parse_error": take_measurement["free_exact_parse_error"],
                "edited_layers": edited_layers,
                "statistics_after_write": {str(l): current[l] for l in edited_layers},
                "localization": localization(base, current, set(edited_layers), S),
                "names_after_write": names, "n_written_so_far": len(all_facts)}
            if runmode == "smoke" or str(case_id) not in prior:
                append_jsonl(log, row)
        except Exception as exc:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "B-seq", "method": "wsp", "case_id": int(case_id),
                "phase": "sequence_write_or_score", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]})
            break
    del w
    gc.collect()
    return 0


def locate_output_matrices(model):
    """Return (layer, module path, module, transpose) for supported MLP output roles."""
    pattern = re.compile(r"(?:^|\.)(?:h|layers)\.(\d+)\.mlp\.(c_proj|down_proj|dense_4h_to_h)$")
    found = []
    for name, module in model.named_modules():
        if re.search(r"vision|visual|mtp", name, flags=re.I):
            continue
        match = pattern.search(name)
        if match and hasattr(module, "weight"):
            found.append((int(match.group(1)), name, module, match.group(2) == "c_proj"))
    by_layer = {}
    for item in found:
        by_layer.setdefault(item[0], []).append(item)
    result = []
    for layer, items in sorted(by_layer.items()):
        if len(items) != 1:
            raise RuntimeError(f"ambiguous MLP output matrices at layer {layer}: {[x[1] for x in items]}")
        result.append(items[0])
    if not result:
        raise RuntimeError(f"no supported MLP output matrices found for {type(model).__name__}")
    return result


def final_norm(model, hidden_size: int, S):
    suffixes = {"ln_f", "final_layer_norm", "final_layernorm", "norm", "norm_f", "final_norm"}
    candidates = []
    for name, module in model.named_modules():
        if re.search(r"vision|visual|mtp", name, flags=re.I):
            continue
        if name.rsplit(".", 1)[-1].lower() not in suffixes:
            continue
        try:
            gain, _ = S.norm_reader(module)
            if int(gain.numel()) == int(hidden_size):
                candidates.append((name.count("."), name, module))
        except Exception:
            continue
    if not candidates:
        raise RuntimeError(f"could not find final norm with hidden width {hidden_size}")
    depth = max(x[0] for x in candidates)
    best = [x for x in candidates if x[0] == depth]
    if len(best) != 1:
        raise RuntimeError(f"ambiguous final norm candidates: {[x[1] for x in best]}")
    return best[0][2], best[0][1]


def locate_snapshot_model(model_id: str) -> Path:
    path = resolve_snapshot(model_id)
    if model_id == "Qwen/Qwen3.5-0.8B" and path.name != QWEN35_SNAPSHOT:
        raise RuntimeError(f"Qwen3.5-0.8B snapshot mismatch: expected {QWEN35_SNAPSHOT}, got {path.name}")
    return path


def robust_z_each(values: dict[int, float]) -> dict[int, float]:
    """Per-layer robust z uses the same median/MAD scale as stats198.robust_z_of_max."""
    import torch
    vals = torch.tensor([values[l] for l in sorted(values)], dtype=torch.float64)
    median = vals.median()
    mad = (vals - median).abs().median()
    if mad <= 0:
        return {l: (0.0 if values[l] == float(median) else float("inf")) for l in values}
    scale = 1.4826 * float(mad)
    return {l: (float(values[l]) - float(median)) / scale for l in values}


def analyze_model(model, tokenizer, S, model_id: str, role: str, stored_dtype: str) -> dict[str, Any]:
    outputs = locate_output_matrices(model)
    embed = model.get_output_embeddings()
    if embed is None or not hasattr(embed, "weight"):
        raise RuntimeError("get_output_embeddings() returned no weight")
    E = embed.weight.detach().double().cpu()
    norm, norm_name = final_norm(model, E.shape[1], S)
    gain, centre = S.norm_reader(norm)
    gain = gain.detach().double().cpu()
    per_layer, all_pairs, f1s = {}, {}, {}
    for layer, name, module, transpose in outputs:
        raw = module.weight.detach()
        W = raw.T.contiguous() if transpose else raw.contiguous()
        if W.shape[0] != E.shape[1]:
            raise RuntimeError(f"output matrix {name} width {W.shape[0]} does not match unembedding {E.shape[1]}")
        data = stats_for(W, gain, centre, E, tokenizer.decode, S)
        data.update({"module": name, "storage_transposed": bool(transpose)})
        per_layer[int(layer)] = data
        all_pairs[int(layer)] = dict_to_pairs(data["pairs"], S)
        f1s[int(layer)] = float(data["f1"])
    zf1 = robust_z_each(f1s)
    for layer in per_layer:
        per_layer[layer]["f1_robust_z"] = zf1[layer]
    top = sorted(f1s.values(), reverse=True)
    ratio = top[0] / top[1] if len(top) > 1 and top[1] else None
    consensus_n, consensus_ids = S.consensus_run(all_pairs)
    max_tword = max((x["t_word"] for x in per_layer.values() if x["t_word"] is not None), default=None)
    top_layer, tie = unique_top_layer({l: x["t_word"] for l, x in per_layer.items()})
    pair = S.top_eligible(all_pairs[top_layer]) if top_layer is not None else None
    return {"model_id": model_id, "role": role, "snapshot": str(locate_snapshot_model(model_id)),
            "stored_dtype": stored_dtype, "norm": norm_name, "centre": bool(centre),
            "layers": {str(l): per_layer[l] for l in sorted(per_layer)}, "max_t_word": max_tword,
            "top_t_word_layer": top_layer, "top_t_word_tie": bool(tie),
            "top_eligible_pair": pair_to_dict(pair, tokenizer.decode) if pair else None,
            "f1_robust_z_max": S.robust_z_of_max(list(f1s.values())),
            "largest_to_second_sigma1_ratio": ratio,
            "consensus": {"run_length": consensus_n, "token_ids": consensus_ids,
                          "tokens": [tokenizer.decode([t]) for t in consensus_ids]}}


def load_arm_c_pair(model_id: str, dtype: str):
    ensure_offline()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = locate_snapshot_model(model_id)
    if model_id == "gpt2-xl" and hash_file(GPT2 / "model.safetensors") != GPT2_SHA256:
        raise RuntimeError("GPT-2 XL model.safetensors SHA256 mismatch")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    kwargs["dtype"] = torch.float32 if dtype == "float32" else "auto"
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs).cpu().eval()
    seen = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters()})
    if dtype == "float32" and seen != ["float32"]:
        raise RuntimeError(f"development host did not load as float32: {seen}")
    return model, tokenizer, ",".join(seen)


def run_arm_c(limit: int | None, out_dir: Path) -> int:
    S = stats_helpers()
    log = log_path_for("armC", None, out_dir)
    runmode, rid = ("smoke" if limit is not None else "registered"), run_id()
    prior = arm_c_successful_keys(log)
    pending = [m for m in ARM_C_MODELS if m["id"] not in prior]
    append_run_event(log, runmode, "armC", [m["id"] for m in pending], rid)
    selected_count = 0
    for item in ARM_C_MODELS[:3]:
        model_id, role = item["id"], item["role"]
        if model_id in prior:
            continue
        if arm_c_limit_reached(selected_count, limit):
            break
        try:
            model, tokenizer, dtype = load_arm_c_pair(model_id, "stored")
            row = analyze_model(model, tokenizer, S, model_id, role, dtype)
            reference, token = ARM_C_REVIEWER_MAX[model_id]
            row["reviewer_reference"] = {"max_t_word": reference, "top_token": token,
                                          "precision": "rounded as printed in design"}
            row["difference_from_reviewer"] = (row["max_t_word"] - reference
                                                 if row["max_t_word"] is not None else None)
            append_jsonl(log, {"record_type": "armC_model", "run_mode": runmode, "run_id": rid,
                               "timestamp": utc_now(), "arm": "C", **row})
            del model, tokenizer
            gc.collect()
        except Exception as exc:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "C", "model_id": model_id, "role": role,
                "phase": "calibration", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]})
            raise
        prior.add(model_id)
        selected_count += 1
    current = {r.get("model_id"): r for r in read_jsonl(log)
               if r.get("record_type") == "armC_model" and r.get("run_mode", "registered") == "registered"}
    missing_calibration = [m["id"] for m in ARM_C_MODELS[:3] if m["id"] not in current]
    if missing_calibration:
        if arm_c_limit_reached(selected_count, limit):
            return 0
        raise RuntimeError("Arm C calibration rows are missing: " + ", ".join(missing_calibration))
    cal = {m["id"]: current[m["id"]]["max_t_word"] for m in ARM_C_MODELS[:3]}
    if any(value is None for value in cal.values()):
        raise RuntimeError("Arm C calibration produced a missing max_t_word")
    tstar = max(x for x in cal.values() if x is not None)
    tstar_path = out_dir / "tstar.json"
    tstar_path.write_text(json.dumps({"calibration_maxima": cal, "t_star": tstar,
        "models": [m["id"] for m in ARM_C_MODELS[:3]], "created_utc": utc_now()},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    require_tstar(tstar_path)
    for item in ARM_C_MODELS[3:]:
        model_id, role = item["id"], item["role"]
        if model_id in prior:
            continue
        if arm_c_limit_reached(selected_count, limit):
            break
        try:
            dtype = "float32" if role == "host" else "stored"
            model, tokenizer, seen = load_arm_c_pair(model_id, dtype)
            row = analyze_model(model, tokenizer, S, model_id, role, seen)
            reference, token = ARM_C_REVIEWER_MAX[model_id]
            row["reviewer_reference"] = {"max_t_word": reference, "top_token": token,
                                          "precision": "rounded as printed in design"}
            row["difference_from_reviewer"] = (row["max_t_word"] - reference
                                                 if row["max_t_word"] is not None else None)
            append_jsonl(log, {"record_type": "armC_model", "run_mode": runmode, "run_id": rid,
                               "timestamp": utc_now(), "arm": "C", **row})
            del model, tokenizer
            gc.collect()
        except Exception as exc:
            append_jsonl(log, {"record_type": "failure", "run_mode": runmode, "run_id": rid,
                "timestamp": utc_now(), "arm": "C", "model_id": model_id, "role": role,
                "phase": "evaluation_or_host", "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-4000:]})
            raise
        prior.add(model_id)
        selected_count += 1
    return 0


def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--dry-run", action="store_true",
                    help="print frozen plan and pinned paths without importing a model")
    ap.add_argument("--limit", type=int, help="run at most N cases/batches/models for a smoke run")
    ap.add_argument("--out-dir", type=Path, default=OUT_DEFAULT,
                    help="output directory, restricted to results/part198")


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    for name, helptext in (("armA", "single GPT-2 XL edit process"),
                           ("armA-mass", "GPT-2 XL mass-edit batches"),
                           ("armB", "Qwen3.5-0.8B single WSP writes and matched random controls"),
                           ("armB-seq", "WSP ten-write accumulation sequence"),
                           ("armC", "descriptive unedited model pass"),
                           ("manifest", "hash every pinned input into out/manifest.json")):
        p = sp.add_parser(name, help=helptext)
        add_common(p)
        if name == "armA":
            p.add_argument("--method", required=True, choices=("ROME", "MEMIT", "AlphaEdit"))
        elif name == "armA-mass":
            p.add_argument("--method", required=True, choices=("MEMIT", "AlphaEdit"))
    return ap


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    out_dir = root_out_dir(args.out_dir)
    if args.dry_run:
        plan = dry_plan(args.command, getattr(args, "method", None), args.limit, out_dir)
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "manifest":
        data = write_manifest(out_dir)
        print(json.dumps({"manifest": str(out_dir / "manifest.json"), "input_count": len(data["inputs"]),
                          "easyedit_commit": data["easyedit_commit"]}, indent=2))
        return 0
    target_log = log_path_for(args.command, getattr(args, "method", None), out_dir)
    os.chdir(out_dir)
    try:
        preflight(out_dir)
        if args.command == "armA":
            return run_arm_a(args.method, args.limit, out_dir)
        if args.command == "armA-mass":
            return run_arm_a_mass(args.method, args.limit, out_dir)
        if args.command == "armB":
            return run_arm_b(args.limit, out_dir)
        if args.command == "armB-seq":
            return run_arm_b_seq(args.limit, out_dir)
        if args.command == "armC":
            return run_arm_c(args.limit, out_dir)
        raise ValueError(args.command)
    except Exception as exc:
        append_jsonl(target_log, {"record_type": "failure",
            "run_mode": "smoke" if args.limit is not None else "registered",
            "timestamp": utc_now(), "method": getattr(args, "method", None),
            "phase": "startup_or_run", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-4000:]})
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

