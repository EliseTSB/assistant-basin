import importlib
from pathlib import Path
import sys
import json
import shutil
from types import SimpleNamespace
import torch
import part43
from test_part43 import TinyModel

part43._prepare_fp32_reference_backend()
qwen = importlib.import_module(part43.FP32_QWEN_MODULE)
hub = sys.modules[part43.FP32_HUB_KERNELS_MODULE]
original = qwen.causal_conv1d_fn
reference = original.__wrapped__

factory_source = '''\ndef _review_spoof_factory(reference):\n    implementation = reference\n    torch_function = reference\n    is_new_implementation = False\n    def foreign(*args, **kwargs):\n        _ = implementation, torch_function, is_new_implementation\n        return "FOREIGN_SENTINEL"\n    return foreign\n'''
exec(compile(factory_source, hub.__file__, "exec"), hub.__dict__)
foreign = hub.__dict__["_review_spoof_factory"](reference)
foreign.__module__ = part43.FP32_QWEN_MODULE
foreign.__name__ = "causal_conv1d_fn"
foreign.__qualname__ = "causal_conv1d_fn"
foreign.__wrapped__ = reference
foreign.__code__ = foreign.__code__.replace(co_filename=str(Path(hub.__file__).resolve()), co_firstlineno=1027)
qwen.causal_conv1d_fn = foreign

part43._configure_full_fp32_precision()
model = TinyModel().to(dtype=torch.float32)
part43._ACTIVE_ACTIVATION_DTYPES = {"torch.float32"}
part43._ACTIVE_DOWN_PROJ_INPUT_DTYPES = {"torch.float32"}
part43._ACTIVE_DOWN_PROJ_INPUT_DTYPES_BY_LAYER = {i: {"torch.float32"} for i in range(32)}
part43._ACTIVE_LOGIT_DTYPES = {"torch.float32"}
state = part43._fp32_reference_runtime_state(model)
row = state["qwen3_5_function_implementations"]["causal_conv1d_fn"]
leg = part43._calibration_runtime_metadata(model, "torch.float32", part43.FP32_SUCCESSOR_PROTOCOL_ID)
registration = {"down_proj_dtype_layers": list(range(32)), "fp32_backend_allowances": []}
part43._validate_runtime_metadata(leg, "fp32", "cpu", part43.FP32_SUCCESSOR_PROTOCOL_ID, registration)
print("runtime_state_accepted=", True)
print("calibration_metadata_validator=", "accepted")
print("dispatch_is_reference=", row["dispatch_is_reference"])
print("reported_dispatch=", row["resolved_implementation"])
print("actual_dispatch_result=", qwen.causal_conv1d_fn())
print("source_hash_pinned=", state["qwen3_5_source_sha256"] == part43.FP32_QWEN_SOURCE_SHA256)

# Drive the actual pilot gate using the runtime captured above, and stop each
# production stage at a harmless stub after recording that the gate passed.
out = Path("/home/elisepc/green/Journal/scratch/review-part43-fp32-backend-r1-luna-r1-2026-10-21/probe-output")
out.mkdir(exist_ok=True)
attach = out / "attach.json"
attach.write_text(json.dumps({"prompts": []}), encoding="utf-8")
args = SimpleNamespace(
    protocol="fp32-successor", model=part43.MODEL, revision=part43.REVISION,
    out=str(out), candidates=str(out / "candidates.json"), attach=str(attach),
)
summary = {
    "artifact_type": part43.CALIBRATION_SUMMARY_TYPE, "protocol_id": part43.FP32_SUCCESSOR_PROTOCOL_ID,
    "derived": True, "reader_id": part43.FP32_SUCCESSOR_READER_ID,
    "model": args.model, "revision": args.revision, "epsilon": 0.0,
    "execution": {"legs": {"fp32": {"runtime": leg["runtime"]}}},
}
reached = []
part43._sequences = lambda _tok: reached.append("sequences") or ["probe"]
part43.run_signs = lambda *_a, **_k: reached.append("signs") or {"ok": True}
part43.run_lattice_amended = lambda *_a, **_k: {"timing": {"elapsed_seconds": 0.0}}
part43.run_swap = lambda *_a, **_k: {"rows": []}
part43.run_attach = lambda *_a, **_k: {"rows": []}
result = part43.run_pilot(model, object(), {}, args, calibration=summary)
print("pilot_status=", result["status"])
print("pilot_stages_reached=", reached)
shutil.rmtree(out)
