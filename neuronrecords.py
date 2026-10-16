#!/usr/bin/env python3
"""Validate neuron-record JSONL and convert the supplied experiment results."""

from __future__ import annotations

import argparse
import collections
import datetime as _datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
SITE = "mlp.down_proj.input"
MODEL_MAPPER_REPO = "https://github.com/jphoenix-dev/ModelMapper"
MODEL_MAPPER_COMMIT = "6cc33cc8fb0e1d95b97bf9aecb049b4ee1955c6b"
ASSISTANT_REPO = "https://github.com/EliseTSB/assistant-basin"
PART2_PREDICTION_COMMIT = "b79642c01805de270773ca322a706e9ffd1c2f0e"
PART11_4_PREDICTION_COMMIT = "ad8c3c9d2bf618e33b2280a311d30ac68c0d8d6c"
PART11_6_PREDICTION_COMMIT = "ff9d4d5b0bad7d0a38e06fd8f22ae1feb591d471"
OLMO_RESULT_COMMIT = "b79642c01805de270773ca322a706e9ffd1c2f0e"
SMOL_IDENTITY_RESULT_COMMIT = "19cc28c2336260cdebe064804c115da1ff22ea8e"
SMOL_NAMEATTR_RESULT_COMMIT = "c750f991a4613569aa25d53fd41136b11422e040"
DATE = "2026-10-16"

_SHA40 = re.compile(r"^[0-9a-fA-F]{40}$")
_MEMBER_KEY = re.compile(
    r"^(?P<model>.+)@(?P<revision>[0-9a-fA-F]{12}|null)/"
    r"(?P<site>[^/]+)/L(?P<layer>[0-9]+)/N(?P<neuron>[0-9]+)/(?:pos|neg)$"
)
_UNIT = re.compile(r"^L([0-9]+)[._]N?([0-9]+)$")


def _err(file: str, line: int, field: str, message: str) -> str:
    return f"{file}:{line}: field {field}: {message}"


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_claim(
    claim: Any, file: str, line: int, is_set: bool, claim_ids: set[str], member_count: int | None = None
) -> list[str]:
    errors: list[str] = []

    def need(obj: Any, key: str, field: str, typ: Any = None) -> Any:
        if not isinstance(obj, dict) or key not in obj:
            errors.append(_err(file, line, field, "required"))
            return None
        value = obj[key]
        if typ is not None and not isinstance(value, typ):
            errors.append(_err(file, line, field, f"must be {typ.__name__}"))
            return None
        return value

    allowed = {
        "id", "level", "statement", "metric", "heldout", "preregistered",
        "code", "author", "date", "status", "supersedes",
    }
    for key in claim:
        if key not in allowed:
            errors.append(_err(file, line, f"claims.{key}", "unknown field"))

    claim_id = need(claim, "id", "claims.id", str)
    if isinstance(claim_id, str):
        if not claim_id.strip():
            errors.append(_err(file, line, "claims.id", "must not be empty"))
        elif claim_id in claim_ids:
            errors.append(_err(file, line, "claims.id", f"duplicate id {claim_id!r}"))
        else:
            claim_ids.add(claim_id)

    level = need(claim, "level", "claims.level")
    if level is not None and (not _is_int(level) or not 0 <= level <= 4):
        errors.append(_err(file, line, "claims.level", "must be an integer from 0 to 4"))
        level = None
    need(claim, "statement", "claims.statement", str)

    metric = need(claim, "metric", "claims.metric", dict)
    if isinstance(metric, dict):
        metric_allowed = {"name", "value", "n", "control_name", "control_value", "control_size"}
        for key in metric:
            if key not in metric_allowed:
                errors.append(_err(file, line, f"claims.metric.{key}", "unknown field"))
        need(metric, "name", "claims.metric.name", str)
        if "value" not in metric:
            errors.append(_err(file, line, "claims.metric.value", "required"))
        if "n" not in metric:
            errors.append(_err(file, line, "claims.metric.n", "required"))
        elif metric["n"] is not None and (not _is_int(metric["n"]) or metric["n"] < 0):
            errors.append(_err(file, line, "claims.metric.n", "must be a non-negative integer or null"))
        if level is not None and level >= 1:
            if not _nonempty_string(metric.get("control_name")):
                errors.append(_err(file, line, "claims.metric.control_name", "required for level 1-4"))
            if "control_value" not in metric:
                errors.append(_err(file, line, "claims.metric.control_value", "required for level 1-4"))
        if is_set and level == 3:
            size = metric.get("control_size")
            if not _is_int(size):
                errors.append(_err(file, line, "claims.metric.control_size", "required as an integer for level 3 set claims"))
            elif member_count is not None:
                if size != member_count:
                    errors.append(
                        _err(file, line, "claims.metric.control_size", f"{size} does not equal set member count {member_count}")
                    )

    heldout = need(claim, "heldout", "claims.heldout")
    if heldout is not None and not isinstance(heldout, bool):
        errors.append(_err(file, line, "claims.heldout", "must be boolean"))

    prereg = need(claim, "preregistered", "claims.preregistered")
    if prereg is not None:
        if prereg is not None and not isinstance(prereg, dict):
            errors.append(_err(file, line, "claims.preregistered", "must be an object or null"))
        elif isinstance(prereg, dict):
            for key in prereg:
                if key not in {"url", "commit"}:
                    errors.append(_err(file, line, f"claims.preregistered.{key}", "unknown field"))
            need(prereg, "url", "claims.preregistered.url", str)
            need(prereg, "commit", "claims.preregistered.commit", str)

    code = need(claim, "code", "claims.code", dict)
    if isinstance(code, dict):
        for key in code:
            if key not in {"repo", "commit", "path"}:
                errors.append(_err(file, line, f"claims.code.{key}", "unknown field"))
        for key in ("repo", "commit", "path"):
            need(code, key, f"claims.code.{key}", str)

    author = need(claim, "author", "claims.author", dict)
    if isinstance(author, dict):
        for key in author:
            if key not in {"who", "model"}:
                errors.append(_err(file, line, f"claims.author.{key}", "unknown field"))
        need(author, "who", "claims.author.who", str)
        if "model" in author and author["model"] is not None and not isinstance(author["model"], str):
            errors.append(_err(file, line, "claims.author.model", "must be a string or null"))

    date = need(claim, "date", "claims.date", str)
    if isinstance(date, str):
        try:
            _datetime.date.fromisoformat(date)
        except ValueError:
            errors.append(_err(file, line, "claims.date", "must be an ISO date"))

    status = need(claim, "status", "claims.status", str)
    if isinstance(status, str) and status not in {"supported", "refuted", "open"}:
        errors.append(_err(file, line, "claims.status", "must be supported, refuted, or open"))

    supersedes = need(claim, "supersedes", "claims.supersedes")
    if supersedes is not None and not isinstance(supersedes, str):
        errors.append(_err(file, line, "claims.supersedes", "must be a string or null"))

    return errors


def validate_record(
    record: Any,
    file: str = "<memory>",
    line: int = 1,
    claim_ids: set[str] | None = None,
    record_ids: set[str] | None = None,
    neuron_keys: set[tuple[Any, ...]] | None = None,
) -> list[str]:
    """Return schema errors for one neuron or set record."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return [_err(file, line, "<record>", "must be a JSON object")]
    if claim_ids is None:
        claim_ids = set()

    is_set = "id" in record or "members" in record
    required = {"id", "model", "revision", "site", "members", "claims"} if is_set else {
        "model", "revision", "site", "layer", "neuron", "pole", "claims"
    }
    allowed = required
    for key in record:
        if key not in allowed:
            errors.append(_err(file, line, key, "unknown field"))
    for key in required:
        if key not in record:
            errors.append(_err(file, line, key, "required"))

    for field in ("model", "site"):
        if field in record and not _nonempty_string(record[field]):
            errors.append(_err(file, line, field, "must be a non-empty string"))
    if "revision" in record:
        rev = record["revision"]
        if rev is not None and (not isinstance(rev, str) or not _SHA40.fullmatch(rev)):
            errors.append(_err(file, line, "revision", "must be a 40-hex commit or null"))

    if is_set:
        set_id = record.get("id")
        if "id" in record and not _nonempty_string(set_id):
            errors.append(_err(file, line, "id", "must be a non-empty string"))
        elif isinstance(set_id, str) and record_ids is not None:
            if set_id in record_ids:
                errors.append(_err(file, line, "id", f"duplicate id {set_id!r}"))
            record_ids.add(set_id)
        members = record.get("members")
        if "members" in record:
            if not isinstance(members, list):
                errors.append(_err(file, line, "members", "must be an array of neuron keys"))
            else:
                if not members:
                    errors.append(_err(file, line, "members", "must contain at least one neuron key"))
                if len(set(x for x in members if isinstance(x, str))) != len(members):
                    errors.append(_err(file, line, "members", "must not contain duplicate neuron keys"))
                for index, member in enumerate(members):
                    if not isinstance(member, str) or not _MEMBER_KEY.fullmatch(member):
                        errors.append(_err(file, line, f"members[{index}]", "must be a neuron key"))
                        continue
                    match = _MEMBER_KEY.fullmatch(member)
                    expected_rev = "null" if record.get("revision") is None else record["revision"][:12]
                    if record.get("model") and match.group("model") != record["model"]:
                        errors.append(_err(file, line, f"members[{index}]", "model does not match set model"))
                    if match.group("revision") != expected_rev:
                        errors.append(_err(file, line, f"members[{index}]", "revision prefix does not match set revision"))
                    if record.get("site") and match.group("site") != record["site"]:
                        errors.append(_err(file, line, f"members[{index}]", "site does not match set site"))
    else:
        for field in ("layer", "neuron"):
            if field in record and (not _is_int(record[field]) or record[field] < 0):
                errors.append(_err(file, line, field, "must be a non-negative integer"))
        if "pole" in record and record["pole"] not in {"pos", "neg"}:
            errors.append(_err(file, line, "pole", "must be pos or neg"))
        if all(k in record for k in ("model", "revision", "site", "layer", "neuron", "pole")):
            key = (record["model"], record["revision"], record["site"], record["layer"], record["neuron"], record["pole"])
            if neuron_keys is not None:
                if key in neuron_keys:
                    errors.append(_err(file, line, "key", "duplicate neuron pole record"))
                neuron_keys.add(key)

    claims = record.get("claims")
    if "claims" in record:
        if not isinstance(claims, list):
            errors.append(_err(file, line, "claims", "must be an array"))
        else:
            for claim in claims:
                member_count = len(record["members"]) if is_set and isinstance(record.get("members"), list) else None
                errors.extend(_validate_claim(claim, file, line, is_set, claim_ids, member_count))
    return errors


def highest_supported_level(record: dict[str, Any]) -> int:
    claims = record.get("claims", [])
    levels = [c.get("level", 0) for c in claims if isinstance(c, dict) and c.get("status") == "supported" and _is_int(c.get("level"))]
    return max(levels, default=0)


def check_files(paths: Iterable[str | Path], output: Any = None) -> dict[str, Any]:
    """Validate JSONL files and optionally print UNPINNED and per-model summaries."""
    if output is None:
        output = sys.stdout
    errors: list[str] = []
    by_model: dict[str, dict[str, Any]] = {}
    unpinned: collections.Counter[str] = collections.Counter()
    claim_ids: set[str] = set()
    record_ids: set[str] = set()
    neuron_keys: set[tuple[Any, ...]] = set()
    for supplied in paths:
        path = Path(supplied)
        file_claim_ids: set[str] = set()
        file_record_ids: set[str] = set()
        file_neuron_keys: set[tuple[Any, ...]] = set()
        try:
            lines = path.open(encoding="utf-8")
        except OSError as exc:
            errors.append(_err(str(path), 0, "file", str(exc)))
            continue
        with lines:
            for line_no, raw in enumerate(lines, 1):
                if not raw.strip():
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(_err(str(path), line_no, "<json>", f"{exc.msg} at column {exc.colno}"))
                    continue
                record_errors = validate_record(record, str(path), line_no, file_claim_ids, file_record_ids, file_neuron_keys)
                errors.extend(record_errors)
                if not isinstance(record, dict):
                    continue
                model = record.get("model")
                if not isinstance(model, str) or not model:
                    continue
                kind = "sets" if "members" in record or "id" in record else "neurons"
                bucket = by_model.setdefault(model, {
                    "neurons": [0] * 5,
                    "sets": [0] * 5,
                    "refuted": 0,
                    "open": 0,
                    "records": 0,
                })
                bucket["records"] += 1
                if record.get("revision", "missing") is None:
                    unpinned[model] += 1
                bucket[kind][highest_supported_level(record)] += 1
                if isinstance(record.get("claims"), list):
                    bucket["refuted"] += sum(isinstance(c, dict) and c.get("status") == "refuted" for c in record["claims"])
                    bucket["open"] += sum(isinstance(c, dict) and c.get("status") == "open" for c in record["claims"])

    if output is not None:
        for model in sorted(unpinned):
            print(f"UNPINNED: {model}: {unpinned[model]} record(s) have revision=null", file=output)
        for model in sorted(by_model):
            b = by_model[model]
            neuron_counts = " ".join(f"L{level}={count}" for level, count in enumerate(b["neurons"]))
            set_counts = " ".join(f"L{level}={count}" for level, count in enumerate(b["sets"]))
            print(
                f"{model}: neuron poles [{neuron_counts}]; sets [{set_counts}]; "
                f"refuted claims={b['refuted']}; open claims={b['open']}",
                file=output,
            )
        for error in errors:
            print(f"ERROR: {error}", file=output)
        if errors:
            print(f"CHECK FAILED: {len(errors)} error(s)", file=output)
        else:
            print(f"CHECK OK: {sum(b['records'] for b in by_model.values())} record(s); no schema errors", file=output)
    return {"errors": errors, "unpinned": dict(unpinned), "by_model": by_model}


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def _revision_from_local_cache(repo_id: str) -> str | None:
    cache_name = "models--" + repo_id.replace("/", "--")
    ref = Path("/mnt/e/neuron-probe/hf/hub") / cache_name / "refs" / "main"
    try:
        candidate = ref.read_text(encoding="ascii").strip()
    except OSError:
        print(f"UNPINNED: local Hugging Face ref not found at {ref}", file=sys.stderr)
        return None
    if not _SHA40.fullmatch(candidate):
        print(f"UNPINNED: local Hugging Face ref at {ref} is not a 40-hex commit", file=sys.stderr)
        return None
    return candidate


def _neuron_record(model: str, revision: str | None, layer: int, neuron: int, pole: str = "pos") -> dict[str, Any]:
    return {
        "model": model,
        "revision": revision,
        "site": SITE,
        "layer": layer,
        "neuron": neuron,
        "pole": pole,
        "claims": [],
    }


def _member_key(model: str, revision: str | None, layer: int, neuron: int, pole: str = "pos") -> str:
    ref = "null" if revision is None else revision[:12]
    return f"{model}@{ref}/{SITE}/L{layer}/N{neuron}/{pole}"


def _parse_unit(unit: str) -> tuple[int, int]:
    match = _UNIT.fullmatch(unit)
    if not match:
        raise ValueError(f"invalid neuron unit identifier {unit!r}")
    return int(match.group(1)), int(match.group(2))


def _claim(
    claim_id: str,
    level: int,
    statement: str,
    metric: dict[str, Any],
    heldout: bool,
    preregistered: dict[str, str] | None,
    code: dict[str, str],
    author: dict[str, Any],
    status: str,
    supersedes: str | None = None,
    date: str = DATE,
) -> dict[str, Any]:
    return {
        "id": claim_id,
        "level": level,
        "statement": statement,
        "metric": metric,
        "heldout": heldout,
        "preregistered": preregistered,
        "code": code,
        "author": author,
        "date": date,
        "status": status,
        "supersedes": supersedes,
    }


def _assistant_code(commit: str, path: str) -> dict[str, str]:
    return {"repo": ASSISTANT_REPO, "commit": commit, "path": path}


def _prereg(commit: str) -> dict[str, str]:
    return {"url": f"{ASSISTANT_REPO}/blob/{commit}/PREDICTIONS.md", "commit": commit}


def from_assistant_basin(out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    olmo_model = "allenai/OLMo-2-0425-1B-Instruct"
    smol_model = "HuggingFaceTB/SmolLM3-3B"
    olmo_revision = _revision_from_local_cache(olmo_model)
    smol_revision = _revision_from_local_cache(smol_model)

    frame_spec = _load_json(ROOT / "data" / "frames.json")
    n_frames = len(frame_spec["frames"])
    extra_frames = ROOT / "data" / "frames-extra.json"
    if extra_frames.exists():
        n_frames += len(_load_json(extra_frames)["frames"])
    heldout_n = len(range(1, n_frames, 2))

    for form in ("chat", "plain"):
        source_path = ROOT / "results" / "part2-olmo2-1b" / f"identity-instruct-{form}.json"
        source = _load_json(source_path)
        for item in source["top50"]:
            unit, discover_score, heldout_score = item
            layer, neuron = _parse_unit(unit)
            key = (olmo_model, olmo_revision, SITE, layer, neuron, "pos")
            record = records.setdefault(key, _neuron_record(olmo_model, olmo_revision, layer, neuron))
            control = source["null_top_max"]
            status = "supported" if heldout_score > control else "refuted"
            claim_id = f"olmo2-{form}-L{layer}-N{neuron}-identity-selectivity"
            claim = _claim(
                claim_id,
                1,
                f"On held-out {form} frames, this positive pole's identity score ({heldout_score}) "
                f"was compared with the shuffled-label null top maximum; it was selected on even frames "
                f"with discovery score {discover_score}.",
                {
                    "name": "heldout_identity_selectivity_score",
                    "value": heldout_score,
                    "n": heldout_n,
                    "control_name": "maximum held-out median of 20 shuffled-label null top sets (null_top_max)",
                    "control_value": control,
                },
                True,
                _prereg(PART2_PREDICTION_COMMIT),
                _assistant_code(OLMO_RESULT_COMMIT, "analyze.py"),
                {"who": "Assistant Basin analysis", "model": None},
                status,
            )
            record["claims"].append(claim)

    smol_sources = {
        "identity": _load_json(ROOT / "results" / "ablate-smol.json"),
        "nameattr": _load_json(ROOT / "results" / "ablate-smol-nameattr.json"),
    }
    nameattr_units = _load_json(ROOT / "results" / "name-attr-pos28.json")
    set_rows: list[dict[str, Any]] = []
    for set_kind, source in smol_sources.items():
        identity_condition = source["conditions"]["identity"]
        by_layer = identity_condition["neurons"]
        members_by_position: dict[tuple[int, int], None] = {}
        for layer_text, neurons in by_layer.items():
            for neuron in neurons:
                members_by_position[(int(layer_text), int(neuron))] = None
        if set_kind == "nameattr":
            selected = {_parse_unit(unit) for unit, *_ in nameattr_units["top50"]}
            if selected != set(members_by_position):
                raise ValueError("name-attribution selected units do not match the intervention membership in ablate-smol-nameattr.json")
        members = [
            _member_key(smol_model, smol_revision, layer, neuron, "pos")
            for layer, neuron in sorted(members_by_position)
        ]
        if not members:
            raise ValueError(f"{set_kind} ablation result has no intervention members")

        baseline = source["conditions"]["base"]["by_form"]["raw_nosys"]["shares"]["SmolLM"]
        ablated = identity_condition["by_form"]["raw_nosys"]["shares"]["SmolLM"]
        random_conditions = [
            condition for name, condition in source["conditions"].items() if name.startswith("random_")
        ]
        random_values = [condition["by_form"]["raw_nosys"]["shares"]["SmolLM"] for condition in random_conditions]
        random_mean = sum(random_values) / len(random_values)
        effect = baseline - ablated
        control_effect = baseline - random_mean
        n = int(source["samples"])
        control_size = len(members)
        status = "supported" if effect > control_effect else "refuted"
        claim_id = f"smollm3-{set_kind}-ablation-causal"
        if set_kind == "nameattr":
            prereg = _prereg(PART11_6_PREDICTION_COMMIT)
            code_commit = SMOL_NAMEATTR_RESULT_COMMIT
            claim_text = (
                "Zeroing these 28 positive name-attribution units lowers SmolLM self-naming on raw no-system prompts "
                "more than zeroing random sets of the same size."
            )
            set_id = "smollm3-name-attribution-positive-28"
        else:
            prereg = _prereg(PART11_4_PREDICTION_COMMIT)
            code_commit = SMOL_IDENTITY_RESULT_COMMIT
            claim_text = (
                "Zeroing the top identity units did not lower SmolLM self-naming on raw no-system prompts more than "
                "zeroing random sets of the same size."
            )
            set_id = "smollm3-identity-top20"
        claim = _claim(
            claim_id,
            3,
            claim_text,
            {
                "name": "self_name_share_drop_from_baseline",
                "value": effect,
                "n": n,
                "control_name": f"mean self-name share drop for {len(random_values)} random sets of the same size",
                "control_value": control_effect,
                "control_size": control_size,
            },
            True,
            prereg,
            _assistant_code(code_commit, "ablate.py"),
            {"who": "Assistant Basin experiment", "model": None},
            status,
        )
        # The intervention is on complete MLP dimensions. The source does not assign an
        # activation sign, so pos is the canonical key used for those neuron positions.
        set_rows.append({
            "id": set_id,
            "model": smol_model,
            "revision": smol_revision,
            "site": SITE,
            "members": members,
            "claims": [claim],
        })

    neuron_rows = sorted(records.values(), key=lambda r: (r["model"], r["layer"], r["neuron"], r["pole"]))
    neuron_path = out_dir / "neurons.jsonl"
    set_path = out_dir / "sets.jsonl"
    _write_jsonl(neuron_path, neuron_rows)
    _write_jsonl(set_path, set_rows)
    return neuron_path, set_path


def _modelmapper_code(path: str) -> dict[str, str]:
    return {"repo": MODEL_MAPPER_REPO, "commit": MODEL_MAPPER_COMMIT, "path": path}


def _modelmapper_claim(
    claim_id: str,
    statement: str,
    metric: dict[str, Any],
    source_path: str,
    model: str,
    status: str = "supported",
    level: int = 0,
    heldout: bool = False,
    control_name: str | None = None,
    control_value: Any = None,
    control_size: int | None = None,
) -> dict[str, Any]:
    if level >= 1:
        metric["control_name"] = control_name
        metric["control_value"] = control_value
    if level == 3 and control_size is not None:
        metric["control_size"] = control_size
    return _claim(
        claim_id,
        level,
        statement,
        metric,
        heldout,
        None,
        _modelmapper_code(source_path),
        {"who": "ModelMapper", "model": model},
        status,
    )


def from_modelmapper(source_dir: str | Path, out_dir: str | Path, revision: str | None = None) -> tuple[Path, Path, list[str]]:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    if revision is not None and not _SHA40.fullmatch(revision):
        raise ValueError("--revision must be a 40-hex commit")

    # These fixture files are read as JSON data only; no fixture content is executed.
    census = _load_json(source_dir / "census_Qwen3-8B.json")
    census_summary = _load_json(source_dir / "census_Qwen3-8B_summary.json")
    attribution = _load_json(source_dir / "attribution_Qwen3-8B.json")
    answer_key = _load_json(source_dir / "Qwen3-8B.json")
    model = census["model"]
    if attribution.get("model") != model or answer_key.get("model") != model:
        raise ValueError("Model Mapper fixture files do not agree on model id")

    neuron_map: dict[tuple[Any, ...], dict[str, Any]] = {}
    neurons = census.get("units", {})
    for unit, unit_data in neurons.items():
        layer, neuron = _parse_unit(unit)
        for pole in ("pos", "neg"):
            pole_data = unit_data.get(pole)
            if not isinstance(pole_data, dict):
                continue
            key = (model, revision, SITE, layer, neuron, pole)
            record = neuron_map.setdefault(key, _neuron_record(model, revision, layer, neuron, pole))
            explanation = pole_data.get("explanation")
            if isinstance(explanation, str) and explanation.strip():
                record["claims"].append(_modelmapper_claim(
                    f"mm-census-{unit}-{pole}-explanation",
                    explanation,
                    {"name": "description_text", "value": explanation, "n": None},
                    "census_Qwen3-8B.json (producer script path not supplied)",
                    model,
                ))
            metadata = {field: pole_data.get(field) for field in ("verdict", "coverage", "random_fit", "clean")}
            if any(value is not None for value in metadata.values()):
                record["claims"].append(_modelmapper_claim(
                    f"mm-census-{unit}-{pole}-labels",
                    "Model Mapper's census metadata records the pole's verdict, coverage, and random-fit value.",
                    {"name": "census_label_metadata", "value": metadata, "n": None},
                    "census_Qwen3-8B.json (producer script path not supplied)",
                    model,
                ))

    # The answer key contains ten labelled units outside the 1,000-unit census sample.
    for marker in answer_key.get("markers", []):
        layer, neuron = int(marker["layer"]), int(marker["neuron"])
        pole = marker["sign"]
        if pole not in {"pos", "neg"}:
            continue
        key = (model, revision, SITE, layer, neuron, pole)
        record = neuron_map.setdefault(key, _neuron_record(model, revision, layer, neuron, pole))
        metadata = {
            "concept": marker.get("concept"),
            "answer": marker.get("answer"),
            "keywords": marker.get("keywords"),
            "label_hits": marker.get("label_hits"),
            "evidence": marker.get("evidence"),
            "spec": marker.get("spec"),
            "pinned": marker.get("pinned"),
        }
        concept_slug = re.sub(r"[^a-z0-9]+", "-", str(marker.get("concept", "concept")).lower()).strip("-")
        record["claims"].append(_modelmapper_claim(
            f"mm-answer-key-{concept_slug}-{layer}-{neuron}-{pole}",
            str(marker["job"]),
            {"name": "answer_key_metadata", "value": metadata, "n": None},
            "Qwen3-8B.json (producer script path not supplied)",
            model,
        ))

    top = attribution.get("top", [])
    prompts = attribution.get("prompts", [])
    for entry in top:
        layer, neuron = int(entry["layer"]), int(entry["neuron"])
        key = (model, revision, SITE, layer, neuron, "pos")
        record = neuron_map.setdefault(key, _neuron_record(model, revision, layer, neuron, "pos"))
        rank = int(entry["rank"])
        record["claims"].append(_modelmapper_claim(
            f"mm-attribution-rank-{rank}-L{layer}-N{neuron}",
            f"Model Mapper ranked this unit #{rank} as a positive gradient-times-activation contributor to the target answer logit.",
            {
                "name": "positive_gradient_times_activation_attribution",
                "value": {
                    "rank": rank,
                    "attribution": entry.get("attribution"),
                    "cum_share_of_positive": entry.get("cum_share_of_positive"),
                    "in_prompt_top100": entry.get("in_prompt_top100"),
                },
                "n": len(prompts),
            },
            "attribution_Qwen3-8B.json (producer script path not supplied)",
            model,
        ))

    set_rows: list[dict[str, Any]] = []
    unmapped: list[str] = []
    baseline = attribution.get("baseline", [])
    ablations = attribution.get("ablation", [])
    for result in ablations:
        k = int(result["k"])
        top_outcome = result.get("top")
        random_outcome = result.get("random")
        if not isinstance(top_outcome, list) or len(top_outcome) < 2 or not isinstance(random_outcome, list) or len(random_outcome) < 2:
            unmapped.append(f"ablation k={k}: top/random outcomes do not provide both target and control prompt values")
            continue
        if k > len(top):
            unmapped.append(f"ablation k={k}: only {len(top)} ranked neuron members are present, so the top set cannot be reconstructed")
            continue
        if k <= 0:
            unmapped.append(f"ablation k={k}: set size is not positive")
            continue
        selected = top[:k]
        members = [
            _member_key(model, revision, int(item["layer"]), int(item["neuron"]), "pos")
            for item in selected
        ]
        if len(members) != k:
            unmapped.append(f"ablation k={k}: top set membership has {len(members)} units, expected {k}")
            continue
        base_target = baseline[0] if len(baseline) > 0 else None
        base_control = baseline[1] if len(baseline) > 1 else None
        top_drop = None if base_target is None else base_target - top_outcome[0]
        random_drop = None if base_target is None else base_target - random_outcome[0]
        supported = top_drop is not None and random_drop is not None and top_drop > random_drop
        claim = _modelmapper_claim(
            f"mm-attribution-ablation-top-{k}",
            f"Ablating the top {k} positive-attribution units changes target-answer probability more than ablating a random set of the same size.",
            {
                "name": "mean_answer_probability_on_target_and_control_prompts",
                "value": {
                    "target_prompts": top_outcome[0],
                    "control_prompts": top_outcome[1],
                    "baseline_target_prompts": base_target,
                    "baseline_control_prompts": base_control,
                    "target_drop": top_drop,
                    "target_prompt_texts": prompts,
                    "control_prompt_texts": attribution.get("controls", []),
                    "positive_attribution_total": attribution.get("positive_attribution_total"),
                },
                "n": len(prompts),
            },
            "attribution_Qwen3-8B.json (producer script path not supplied)",
            model,
            status="supported" if supported else "refuted",
            level=3,
            heldout=False,
            control_name=f"paired random neuron set at k={k}",
            control_value={
                "target_prompts": random_outcome[0],
                "control_prompts": random_outcome[1],
                "target_drop": random_drop,
            },
            control_size=k,
        )
        set_rows.append({
            "id": f"modelmapper-qwen3-8b-attribution-top-{k}",
            "model": model,
            "revision": revision,
            "site": SITE,
            "members": members,
            "claims": [claim],
        })

    # Keep aggregate census fields accessible in the returned mapping notes and report.
    _ = census_summary
    neuron_rows = sorted(neuron_map.values(), key=lambda r: (r["layer"], r["neuron"], r["pole"]))
    neuron_path = out_dir / "modelmapper-neurons.jsonl"
    set_path = out_dir / "modelmapper-sets.jsonl"
    _write_jsonl(neuron_path, neuron_rows)
    _write_jsonl(set_path, set_rows)
    return neuron_path, set_path, unmapped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuronrecords.py")
    sub = parser.add_subparsers(dest="command", required=True)

    check_parser = sub.add_parser("check", help="validate neuron and set JSONL files")
    check_parser.add_argument("files", nargs="+")

    assistant_parser = sub.add_parser("from-assistant-basin", help="convert this repo's result files")
    assistant_parser.add_argument("--out", required=True, help="output directory")

    modelmapper_parser = sub.add_parser("from-modelmapper", help="convert Model Mapper fixture JSON")
    modelmapper_parser.add_argument("directory", help="directory containing the Model Mapper JSON fixtures")
    modelmapper_parser.add_argument("--out", required=True, help="output directory")
    modelmapper_parser.add_argument("--revision", help="40-hex Hugging Face weights revision")

    args = parser.parse_args(argv)
    if args.command == "check":
        result = check_files(args.files)
        return 1 if result["errors"] else 0
    if args.command == "from-assistant-basin":
        try:
            neurons, sets = from_assistant_basin(args.out)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"conversion error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {neurons} and {sets}")
        return 0
    if args.command == "from-modelmapper":
        try:
            neurons, sets, unmapped = from_modelmapper(args.directory, args.out, args.revision)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            print(f"conversion error: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {neurons} and {sets}")
        for note in unmapped:
            print(f"UNMAPPED: {note}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
