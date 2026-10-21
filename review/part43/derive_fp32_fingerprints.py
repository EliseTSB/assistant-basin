#!/usr/bin/env python3
"""Derive code and binding fingerprints from a fresh pinned Transformers import."""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_TRANSFORMERS_VERSION = "5.20.0"
QWEN_MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
DISPATCH_NAMES = (
    "causal_conv1d_fn", "causal_conv1d_update",
    "torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
)
MODULE_CLASS_NAMES = (
    "Module", "Sequential", "ModuleList", "ModuleDict", "Linear", "Embedding",
    "Conv1d", "Conv3d", "LayerNorm", "GELU", "Identity", "SiLU", "Dropout",
)
SLOT_NAMES = ("__call__", "_call_impl", "_wrapped_call_impl", "forward")


def _const_value(value: Any) -> Any:
    if isinstance(value, types.CodeType):
        return {"code": _code_payload(value)}
    if value is None or type(value) in (bool, int, str):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float:
        return {"type": "float", "hex": value.hex()}
    if type(value) is complex:
        return {"type": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if type(value) is bytes:
        return {"type": "bytes", "hex": value.hex()}
    if value is Ellipsis:
        return {"type": "ellipsis"}
    if type(value) is tuple:
        return {"type": "tuple", "items": [_const_value(item) for item in value]}
    if type(value) is frozenset:
        items = [_const_value(item) for item in value]
        return {"type": "frozenset", "items": sorted(items, key=_json)}
    raise TypeError(f"unsupported code constant type in pinned Qwen source: {type(value)!r}")


def _code_payload(code: types.CodeType) -> dict[str, Any]:
    return {
        "co_code": code.co_code.hex(),
        "co_consts": [_const_value(value) for value in code.co_consts],
        "co_names": list(code.co_names),
        "co_varnames": list(code.co_varnames),
        "co_freevars": list(code.co_freevars),
    }


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def code_fingerprint(code: types.CodeType) -> str:
    return hashlib.sha256(_json(_code_payload(code)).encode("utf-8")).hexdigest()


def _callable_chain(value: Any) -> list[Any]:
    chain = []
    seen = set()
    current = value
    while inspect.isfunction(current) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = getattr(current, "__wrapped__", None)
        if current is None:
            break
    return chain


def _code_methods(cls: type) -> dict[str, str]:
    result = {}
    for name, descriptor in vars(cls).items():
        value = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
        if inspect.isfunction(value):
            for index, function in enumerate(_callable_chain(value)):
                suffix = "" if index == 0 else f":wrapped:{index}"
                result[f"{name}{suffix}"] = code_fingerprint(function.__code__)
    return result


def _source_hash(path: Any) -> str | None:
    if not isinstance(path, str) or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stable_object(value: Any, active: set[int] | None = None) -> Any:
    active = set() if active is None else active
    if value is None or type(value) in (bool, int, str):
        return {"kind": type(value).__name__, "value": value}
    if type(value) is float:
        return {"kind": "float", "hex": value.hex()}
    if type(value) is complex:
        return {"kind": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if type(value) is bytes:
        return {"kind": "bytes", "hex": value.hex()}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, types.CodeType):
        return {"kind": "code", "fingerprint": code_fingerprint(value)}
    if inspect.isfunction(value):
        object_id = id(value)
        if object_id in active:
            return {"kind": "cycle"}
        active.add(object_id)
        try:
            closure = []
            for cell in value.__closure__ or ():
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    cell_value = {"empty_cell": True}
                closure.append(_stable_object(cell_value, active))
            return {
                "kind": "function",
                "fingerprint": code_fingerprint(value.__code__),
                "defaults": _stable_object(value.__defaults__, active),
                "kwdefaults": _stable_object(value.__kwdefaults__, active),
                "closure": closure,
            }
        finally:
            active.remove(object_id)
    if isinstance(value, types.ModuleType):
        return {
            "kind": "module",
            "name": value.__name__,
            "source_sha256": _source_hash(getattr(value, "__file__", None)),
        }
    if isinstance(value, type):
        return {
            "kind": "class",
            "methods": _code_methods(value),
        }
    object_id = id(value)
    if object_id in active:
        return {"kind": "cycle"}
    active.add(object_id)
    try:
        if isinstance(value, Mapping):
            items = [(_stable_object(key, active), _stable_object(item, active)) for key, item in value.items()]
            return {
                "kind": "mapping",
                "class_methods": _code_methods(type(value)),
                "items": sorted(items, key=lambda pair: _json(pair[0])),
            }
        if isinstance(value, (tuple, list)):
            return {
                "kind": "sequence",
                "class_methods": _code_methods(type(value)),
                "items": [_stable_object(item, active) for item in value],
            }
        if isinstance(value, (set, frozenset)):
            items = [_stable_object(item, active) for item in value]
            return {"kind": "set", "class_methods": _code_methods(type(value)), "items": sorted(items, key=_json)}
        return {
            "kind": "object",
            "class_methods": _code_methods(type(value)),
            "attributes": {
                name: _stable_object(item, active)
                for name, item in sorted(vars(value).items())
                if not name.startswith("_")
            } if hasattr(value, "__dict__") else {},
        }
    finally:
        active.remove(object_id)


def object_fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(_stable_object(value)).encode("utf-8")).hexdigest()


def _class_slots(cls: type) -> dict[str, str]:
    result = {}
    for name in SLOT_NAMES:
        if name in vars(cls):
            value = vars(cls)[name]
            value = value.__func__ if isinstance(value, (staticmethod, classmethod)) else value
            code = getattr(value, "__code__", None)
            result[name] = code_fingerprint(code) if isinstance(code, types.CodeType) else object_fingerprint(value)
    return result


def derive() -> dict[str, Any]:
    if importlib.metadata.version("transformers") != EXPECTED_TRANSFORMERS_VERSION:
        raise SystemExit(f"expected Transformers {EXPECTED_TRANSFORMERS_VERSION}")
    qwen = importlib.import_module(QWEN_MODULE)
    hub = importlib.import_module("transformers.integrations.hub_kernels")
    qwen_source = Path(qwen.__file__).read_text(encoding="utf-8")
    qwen_class_names = sorted(
        node.name for node in ast.parse(qwen_source).body if isinstance(node, ast.ClassDef)
    )
    functions: dict[str, Any] = {}
    classes: dict[str, type] = {}
    for name, value in vars(qwen).items():
        if inspect.isfunction(value) and (value.__globals__ is qwen.__dict__ or name in DISPATCH_NAMES):
            for index, function in enumerate(_callable_chain(value)):
                suffix = "" if index == 0 else f":wrapped:{index}"
                functions[f"global:{name}{suffix}"] = function
    for name in qwen_class_names:
        cls = vars(qwen).get(name)
        if not isinstance(cls, type):
            raise SystemExit(f"source-pinned Qwen class {name} is unavailable")
        classes[name] = cls
        for method_name, descriptor in vars(cls).items():
            value = descriptor.__func__ if isinstance(descriptor, (staticmethod, classmethod)) else descriptor
            if inspect.isfunction(value):
                for index, function in enumerate(_callable_chain(value)):
                    suffix = "" if index == 0 else f":wrapped:{index}"
                    functions[f"class:{name}.{method_name}{suffix}"] = function
    dispatch_references = {}
    for name in DISPATCH_NAMES:
        wrapper = getattr(qwen, name)
        reference = getattr(wrapper, "__wrapped__", None)
        if not inspect.isfunction(wrapper) or not inspect.isfunction(reference):
            raise SystemExit(f"{name} is not the expected decorated Python reference binding")
        functions[f"dispatch:{name}"] = wrapper
        dispatch_references[name] = reference
        for index, function in enumerate(_callable_chain(reference)):
            suffix = "" if index == 0 else f":wrapped:{index}"
            functions[f"dispatch_reference:{name}{suffix}"] = function

    referenced_names = sorted({name for fn in functions.values() for name in fn.__code__.co_names} & set(vars(qwen)))
    torch_nn = importlib.import_module("torch.nn")
    torch_classes = {}
    for name in MODULE_CLASS_NAMES:
        cls = getattr(torch_nn, name)
        if not isinstance(cls, type) or not issubclass(cls, torch_nn.Module):
            raise SystemExit(f"torch.nn.{name} is not an nn.Module class")
        path = f"torch.nn.{name}"
        torch_classes[path] = {"slots": _class_slots(cls)}
    return {
        "transformers_version": EXPECTED_TRANSFORMERS_VERSION,
        "qwen_source_sha256": _source_hash(qwen.__file__),
        "hub_source_sha256": _source_hash(hub.__file__),
        "code_fingerprints": {
            name: code_fingerprint(function.__code__)
            for name, function in sorted(functions.items())
        },
        "qwen_class_names": sorted(classes),
        "global_reference_fingerprints": {
            name: object_fingerprint(vars(qwen)[name]) for name in referenced_names
        },
        "torch_module_classes": torch_classes,
    }


def main() -> None:
    if any(name == "transformers" or name.startswith("transformers.") for name in __import__("sys").modules):
        raise SystemExit("run this derivation in a fresh interpreter before importing Transformers")
    os.environ["USE_HUB_KERNELS"] = "0"
    print(json.dumps(derive(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
