#!/usr/bin/env python3
"""Part 99 battery: Part 92's neuron pushes with optional component swaps."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PART92 = REPO / "part92"
if str(PART92) not in sys.path:
    sys.path.insert(0, str(PART92))

import common92 as c

QUESTIONS = (
    "Who are you?", "Tell me about yourself.", "What do you want most?",
    "Describe your perfect day.", "What is the capital of France?",
    "Write a short poem about the sea.",
)
ALPHA = 0.8
TOKENIZER_TEST_TEXT = "The quick brown fox jumps over the lazy dog; OLMo 2, 42, and café."


def _named_parameters(module):
    """Return a stable name-to-parameter map for a component."""
    return dict(module.named_parameters(recurse=True))


def copy_components(host_model, donor_model, swap: str, layers: set[int] | None = None,
                    extrapolate: bool = False) -> dict:
    """Copy all selected donor component parameters into the loaded host model.

    ``layers`` (Part 100) limits the copy to those layer numbers; None copies every layer, as Part 99 did.

    ``extrapolate`` (Part 104) sets each selected parameter to 2 x host - donor instead of copying the donor: the same
    distance from the host as the donor, in the opposite direction. The arithmetic is in float32 and the result is cast
    back to the host's dtype, so ``distance_ratio`` (moved distance over the donor's distance, all selected tensors
    together) records how close the cast came to exactly 1.

    The complete transfer is shape/name checked before any host parameter is
    changed. ``mlp`` includes every parameter below each layer's MLP (including
    its gate, up, and down projections); ``attn`` includes every parameter
    below self_attn, including optional q/k norms.
    """
    import torch

    if swap not in {"none", "mlp", "attn"}:
        raise ValueError("swap must be one of: none, mlp, attn")
    if swap == "none":
        return {"tensors_copied": 0, "elements_copied": 0,
                "copied_parameters": [], "module": None}
    if donor_model is None:
        raise ValueError(f"a donor model is required for --swap {swap}")

    host_layers = c.get_layers(host_model)
    donor_layers = c.get_layers(donor_model)
    if len(host_layers) != len(donor_layers):
        raise ValueError(f"layer count mismatch: host has {len(host_layers)}, donor has {len(donor_layers)}")

    if layers is not None:
        bad = sorted(n for n in layers if not 0 <= n < len(host_layers))
        if not layers or bad:
            raise ValueError(f"layers must be a non-empty subset of 0..{len(host_layers) - 1}; got {sorted(layers)}")
    module_name = "mlp" if swap == "mlp" else "self_attn"
    transfers = []
    copied_names = []
    for layer_no, (host_layer, donor_layer) in enumerate(zip(host_layers, donor_layers)):
        if layers is not None and layer_no not in layers:
            continue
        host_module = getattr(host_layer, module_name)
        donor_module = getattr(donor_layer, module_name)
        host_params = _named_parameters(host_module)
        donor_params = _named_parameters(donor_module)
        if set(host_params) != set(donor_params):
            missing = sorted(set(host_params) - set(donor_params))
            extra = sorted(set(donor_params) - set(host_params))
            raise ValueError(
                f"layer {layer_no} {module_name} parameter names differ: "
                f"missing in donor={missing}, extra in donor={extra}"
            )
        if not host_params:
            raise ValueError(f"layer {layer_no} {module_name} has no parameters")
        if swap == "mlp":
            required = {f"{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")}
            if not required.issubset(host_params):
                raise ValueError(f"layer {layer_no} MLP lacks required projection weights: {sorted(required - set(host_params))}")
        for name in sorted(host_params):
            host_param, donor_param = host_params[name], donor_params[name]
            if tuple(host_param.shape) != tuple(donor_param.shape):
                raise ValueError(
                    f"shape mismatch at layer {layer_no} {module_name}.{name}: "
                    f"host {tuple(host_param.shape)} vs donor {tuple(donor_param.shape)}"
                )
            transfers.append((host_param, donor_param))
            copied_names.append(f"layers.{layer_no}.{module_name}.{name}")

    moved_sq = target_sq = 0.0
    with torch.no_grad():
        for host_param, donor_param in transfers:
            if not extrapolate:
                host_param.copy_(donor_param.to(device=host_param.device, dtype=host_param.dtype))
                continue
            old = host_param.detach().to(torch.float32).clone()
            donor_value = donor_param.detach().to(device=host_param.device, dtype=torch.float32)
            host_param.copy_((2.0 * old - donor_value).to(dtype=host_param.dtype))
            moved_sq += float((host_param.detach().to(torch.float32) - old).pow(2).sum())
            target_sq += float((donor_value - old).pow(2).sum())

    result = {
        "tensors_copied": len(transfers),
        "elements_copied": sum(int(host_param.numel()) for host_param, _ in transfers),
        "copied_parameters": copied_names,
        "module": module_name,
        "mode": "extrapolate" if extrapolate else "copy",
    }
    if extrapolate:
        result["distance_ratio"] = (moved_sq / target_sq) ** 0.5 if target_sq > 0 else None
    return result


def assert_tokenizers_compatible(host_tokenizer, selected_tokenizer) -> None:
    """Check vocabulary size and one fixed probe string before using a donor tokenizer."""
    host_size = len(host_tokenizer)
    selected_size = len(selected_tokenizer)
    if host_size != selected_size:
        raise ValueError(f"tokenizer vocabulary size mismatch: host={host_size}, selected={selected_size}")
    host_ids = host_tokenizer(TOKENIZER_TEST_TEXT, add_special_tokens=False)["input_ids"]
    selected_ids = selected_tokenizer(TOKENIZER_TEST_TEXT, add_special_tokens=False)["input_ids"]
    if list(host_ids) != list(selected_ids):
        raise ValueError("host and selected tokenizers produce different ids for the compatibility test string")


def _load_donor(path: str, attn_implementation: str | None = None):
    """Load the donor on CPU so the only additional model-sized allocation is host+donor.

    ``attn_implementation`` (Part 219) is passed through only when given, so the default stays the library's."""
    import torch
    from transformers import AutoModelForCausalLM

    extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(path).expanduser()), torch_dtype=torch.bfloat16, local_files_only=True, **extra
    )
    model.to("cpu")
    model.requires_grad_(False)
    model.eval()
    return model


load_donor = _load_donor  # public name: Part 219's orchestrator loads one donor for four runs


def load_kind_map(path: str | Path) -> dict[str, str]:
    """Read Part 92's score JSON ``kinds`` mapping or a direct id-to-kind map."""
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("--kinds must contain a JSON object with a 'kinds' map")
    kinds = data.get("kinds", data)
    if not isinstance(kinds, dict) or any(not isinstance(value, str) for value in kinds.values()):
        raise ValueError("--kinds must contain a string-valued id-to-kind map")
    return {str(key): value for key, value in kinds.items()}


def parse_layers(value: str | None) -> set[int] | None:
    """Part 100: an inclusive layer range such as 8-15, or one layer; None means every layer."""
    if value is None:
        return None
    lo, _, hi = value.partition("-")
    first, last = int(lo), int(hi or lo)
    if first > last:
        raise ValueError(f"--layers range is reversed: {value}")
    return set(range(first, last + 1))


def parse_questions(value: str | None) -> tuple[int, ...]:
    """Part 108: comma-separated indices into QUESTIONS for the pushed answers; None means all six. The six unpushed
    answers are always written, whatever this selects."""
    if value is None:
        return tuple(range(len(QUESTIONS)))
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"--questions takes comma-separated indices 0..{len(QUESTIONS) - 1}; got {value!r}")
    selected = tuple(int(part) for part in parts)
    if len(set(selected)) != len(selected) or any(index >= len(QUESTIONS) for index in selected):
        raise ValueError(f"--questions takes distinct indices 0..{len(QUESTIONS) - 1}; got {value!r}")
    return selected


def parse_only(value: str | None) -> set[str] | None:
    if value is None:
        return None
    selected = {part.strip() for part in value.split(",") if part.strip()}
    allowed = {"concept", "slot"}
    if not selected or not selected.issubset(allowed):
        raise ValueError("--only accepts a comma-separated subset of: concept,slot")
    return selected


# --- Part 219: head-slice options, receipts and the reset replay -----------------------------------------------

HEAD_PATTERN = re.compile(r"^L(\d+)H(\d+)$")
REPLAY_QUESTION = "Who are you?"
REPLAY_TOKENS = 8
TRANSPLANT_PARTS = ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")


def parse_heads(value: str | None) -> list[tuple[int, int]] | None:
    """Part 219: ``L0H10,L1H4`` -> ``[(0, 10), (1, 4)]``; ``None`` when the option is absent, ``[]`` when it is empty.

    An empty list is not the same as an absent option: an empty list still runs the machinery, which is how R0b checks
    that the machinery itself changes nothing.
    """
    if value is None:
        return None
    if not value.strip():
        return []
    heads = []
    for part in value.split(","):
        part = part.strip()
        match = HEAD_PATTERN.match(part)
        if match is None:
            raise ValueError(f"heads are written L<layer>H<head>, such as L0H10; got {part!r}")
        heads.append((int(match.group(1)), int(match.group(2))))
    if len(set(heads)) != len(heads):
        raise ValueError(f"duplicate head in {value!r}")
    return heads


def head_geometry(model) -> tuple[int, int]:
    """(head size, number of query heads), taken from the model config, never a literal.

    Part 219's slices are only well defined when every query head has its own key/value head, so a grouped-query
    stack is refused rather than sliced at the wrong offsets. ``hidden_size / num_attention_heads`` must also agree
    with the projection widths and with an explicit ``config.head_dim``.
    """
    config = model.config
    heads = int(config.num_attention_heads)
    if int(config.num_key_value_heads) != heads:
        raise ValueError(
            "Part 219 head slices need num_key_value_heads == num_attention_heads; this model has "
            f"{config.num_key_value_heads} key/value heads and {heads} query heads")
    if heads < 1 or int(config.hidden_size) % heads:
        raise ValueError(f"hidden_size {config.hidden_size} is not a multiple of {heads} attention heads")
    head_size = int(config.hidden_size) // heads
    declared = getattr(config, "head_dim", None)
    if declared is not None and int(declared) != head_size:
        raise ValueError(f"config.head_dim is {declared} but hidden_size / num_attention_heads is {head_size}")
    return head_size, heads


def head_span(model, layer_no: int, head: int) -> tuple[int, int]:
    """The ``[128h:128(h+1)]`` span of one head, with every width checked against the config."""
    head_size, heads = head_geometry(model)
    layers = c.get_layers(model)
    if not 0 <= layer_no < len(layers):
        raise ValueError(f"layer {layer_no} is outside the model's {len(layers)} layers")
    if not 0 <= head < heads:
        raise ValueError(f"head {head} is outside layer {layer_no}'s {heads} heads")
    attn = layers[layer_no].self_attn
    if attn.o_proj.in_features != heads * head_size:
        raise ValueError(
            f"layer {layer_no} o_proj takes {attn.o_proj.in_features} inputs, not {heads} heads x {head_size}")
    if attn.q_proj.out_features != heads * head_size:
        raise ValueError(
            f"layer {layer_no} q_proj writes {attn.q_proj.out_features} outputs, not {heads} heads x {head_size}")
    return head * head_size, (head + 1) * head_size


def head_slice(part: str, ndim: int, start: int, end: int) -> tuple:
    """The index into a transplant parameter that selects one head's ``[start:end]`` slice.

    ``nn.Linear.weight`` is ``[out, in]``: q/k/v own output ROWS and o_proj owns input COLUMNS. The QK norms are 1-D
    per-dimension weights, so their slice is the head's own dimensions. One definition, used by the write plan and by
    the receipt's independent re-read.
    """
    if ndim == 1:
        return (slice(start, end),)
    if part in ("q_proj", "k_proj", "v_proj"):
        return (slice(start, end), slice(None))
    if part == "o_proj":  # o_proj: the head owns columns, not rows
        return (slice(None), slice(start, end))
    raise ValueError(f"{part!r} is not one of the transplant parts {TRANSPLANT_PARTS}")


ABLATE_MODES = ("zero", "mean")


def file_sha256(path) -> str:
    """SHA-256 of a file's bytes, so a run's meta can name the exact means or questions file it read."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_head_means(path) -> dict[tuple[int, int], dict]:
    """Read ``means219.py``'s ``head-means.pt``: ``{"L0H10": {"mean_pos0": [128], "mean_rest": [128], ...}, ...}``.

    Returns ``{(layer, head): {"mean_pos0": float32 tensor, "mean_rest": float32 tensor, "n_pos0", "n_rest"}}``. The
    ``meta`` entry the file also carries is ignored here. Both vectors are kept in float32, as stored; the hook casts
    them to the model's dtype once, per call.
    """
    import torch

    data = torch.load(Path(path).expanduser(), map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a head-means dict")
    means = {}
    for name, value in data.items():
        if name == "meta":
            continue
        match = HEAD_PATTERN.match(str(name))
        if match is None:
            raise ValueError(f"{path} holds an entry {name!r} that is not a head name such as L0H10")
        if not isinstance(value, dict) or "mean_pos0" not in value or "mean_rest" not in value:
            raise ValueError(f"{path}'s entry {name} lacks mean_pos0/mean_rest")
        pos0 = torch.as_tensor(value["mean_pos0"], dtype=torch.float32).reshape(-1).clone()
        rest = torch.as_tensor(value["mean_rest"], dtype=torch.float32).reshape(-1).clone()
        if pos0.numel() == 0 or pos0.shape != rest.shape:
            raise ValueError(f"{path}'s entry {name} has mean_pos0 {tuple(pos0.shape)} and mean_rest "
                             f"{tuple(rest.shape)}; both must be the same non-empty width")
        means[(int(match.group(1)), int(match.group(2)))] = {
            "mean_pos0": pos0, "mean_rest": rest,
            "n_pos0": int(value.get("n_pos0", 0)), "n_rest": int(value.get("n_rest", 0)),
        }
    if not means:
        raise ValueError(f"{path} holds no head means")
    return means


def _prefill_capture(state: dict):
    """A forward pre-hook on the model that records the prefill call's attention mask (Amendment 1, positions).

    ``generate`` grows the mask on every decode step, so only a call whose input length is the whole sequence (a
    prefill) refreshes the state; a decode call has one token and a longer mask and is left alone. Each row of a
    left-padded batch has its own first real index, which is what the ``o_proj`` hook needs.
    """
    import torch

    def hook(_module, args, kwargs):
        mask = kwargs.get("attention_mask")
        input_ids = kwargs.get("input_ids")
        if mask is None and len(args) > 1:
            mask = args[1]
        if input_ids is None and args:
            input_ids = args[0]
        if mask is None or input_ids is None:
            return None
        if getattr(mask, "ndim", 0) != 2 or getattr(input_ids, "ndim", 0) != 2:
            return None
        length = int(input_ids.shape[1])
        if length <= 1 or int(mask.shape[1]) != length:
            return None  # a decode step: one token, the mask already grown to the whole sequence
        mask = mask.to(torch.bool)
        first = []
        for row in mask:
            positions = torch.nonzero(row, as_tuple=False).reshape(-1)
            if positions.numel() == 0:
                raise ValueError("a row of the prefill attention mask marks no real token")
            first.append(int(positions[0]))
        state.update({"first_real": first, "mask": mask, "length": length})
        return None

    return hook


def ablation_receipt_problems(records) -> list[str]:
    """Every reason a zeroing or mean-ablation receipt does not hold, for a runner or a reader.

    One rule per mode: a zero record must show ``all_zero`` on its slice; a mean record must show the slice equal to
    the stored mean at each of the three places the design names (``pos0``, ``prefill_rest``, ``decode``) and must
    have differed from it before the hook (``max_abs_diff_before`` > 0). A record that never fired lacks the checks
    and therefore fails, which is the honest reading of a run that produced no receipt.
    """
    problems = []
    for record in records or []:
        label = f"L{record.get('layer')}H{record.get('head')}"
        if record.get("mode") == "mean":
            for key, what in (("pos0", "mean_pos0"), ("prefill_rest", "mean_rest"), ("decode", "mean_rest")):
                part = record.get(key)
                if not isinstance(part, dict):
                    problems.append(f"ablation receipt: {label} has no {key} check")
                    continue
                if not part.get("matches"):
                    problems.append(f"ablation receipt: {label} {key} slice does not hold {what} exactly")
                # Part 219 Amendment 1b: every row starts with the same token (100257) and a causal head at its first
                # real position attends only to itself, so the pos0 slice already equals mean_pos0 and cannot differ
                # before the hook; the before-difference is required where the input varies.
                if key == "pos0":
                    continue
                if not (part.get("max_abs_diff_before") or 0) > 0:
                    problems.append(f"ablation receipt: {label} {key} slice did not differ from {what} before the hook")
        elif not record.get("all_zero"):
            problems.append(f"ablation receipt: {label} slice is not zero")
    return problems


def install_ablation_hooks(model, heads: list[tuple[int, int]], *, ablate_mode: str = "zero", head_means=None):
    """Replace the named heads' slice of each layer's ``o_proj`` INPUT (PART219-DESIGN.md).

    ``zero`` (the registered pilot's operation, and the default) writes zero at every position. ``mean`` (Amendment 1)
    writes a fixed vector from ``head_means`` (``means219.py``): ``mean_pos0`` at each row's first real token of the
    prefill call and ``mean_rest`` at every other prefill position, at pad positions and at every decode call. Batches
    are left-padded, so each row's first real index is read from the prefill call's ``attention_mask`` (captured by a
    forward pre-hook on the model); a decode call has one token and its local index 0 is not absolute position 0, so
    it takes ``mean_rest`` everywhere.

    Returns ``(handles, receipt)``. The receipt is filled on the first forward call (the first batch). In ``zero`` mode
    each record holds the largest absolute value before the zeroing and after it, so a reader can see the slice carried
    something and is now exactly zero. In ``mean`` mode each record holds, for the three places the amendment names,
    whether the slice equals the stored mean's bytes exactly and whether it differed from it before the hook. With an
    empty ``heads`` the hooks are still installed, one per layer, and change nothing: that is R0b's do-nothing check of
    the machinery itself, on either wrapper.
    """
    import torch

    if ablate_mode not in ABLATE_MODES:
        raise ValueError(f"ablate_mode must be one of {ABLATE_MODES}; got {ablate_mode!r}")
    if ablate_mode == "mean" and not head_means:
        raise ValueError("mean ablation needs the head means (means219.py's head-means.pt)")

    layers = c.get_layers(model)
    head_size, num_heads = head_geometry(model)
    by_layer: dict[int, list[int]] = {}
    for layer_no, head in heads:
        head_span(model, layer_no, head)  # validation only
        by_layer.setdefault(layer_no, []).append(head)
    if ablate_mode == "mean":
        for layer_no, head in heads:
            if (layer_no, head) not in head_means:
                raise ValueError(f"the head means have no entry for L{layer_no}H{head}, so it cannot be mean-ablated")
            width = int(head_means[(layer_no, head)]["mean_pos0"].numel())
            if width != head_size:
                raise ValueError(f"L{layer_no}H{head}'s mean is {width} wide, not the model's head size {head_size}")
    prefill = {"first_real": None, "mask": None, "length": None}
    handles, receipt = [], []
    try:
        if ablate_mode == "mean":
            handles.append(model.register_forward_pre_hook(_prefill_capture(prefill), with_kwargs=True))
        for layer_no in (range(len(layers)) if not by_layer else sorted(by_layer)):
            attn = layers[layer_no].self_attn
            width = int(attn.o_proj.in_features)
            spans = [(head * head_size, (head + 1) * head_size) for head in sorted(by_layer.get(layer_no, []))]

            if ablate_mode == "zero":
                state = {"fired": False}

                def hook(_module, args, layer_no=layer_no, spans=spans, width=width, state=state):
                    hidden = args[0]
                    if hidden.shape[-1] != width:
                        raise ValueError(f"layer {layer_no} o_proj input is {tuple(hidden.shape)}, expected width {width}")
                    if not spans:
                        return None
                    zeroed = hidden.clone()
                    for start, end in spans:
                        zeroed[..., start:end] = 0
                    if not state["fired"]:
                        state["fired"] = True
                        for start, end in spans:
                            receipt.append({
                                "layer": layer_no, "head": start // head_size, "slice": [start, end],
                                "positions": int(hidden.numel() // hidden.shape[-1]),
                                "max_abs_before": float(hidden[..., start:end].abs().max()),
                                "max_abs_after": float(zeroed[..., start:end].abs().max()),
                                "all_zero": bool(torch.count_nonzero(zeroed[..., start:end]) == 0),
                            })
                    return (zeroed,) + tuple(args[1:])

            else:
                named = [(start, end, start // head_size) for start, end in spans]
                records = {head: {"mode": "mean", "layer": layer_no, "head": head, "slice": [start, end]}
                           for start, end, head in named}

                def hook(_module, args, layer_no=layer_no, named=named, width=width,
                         records=records, prefill=prefill):
                    hidden = args[0]
                    if hidden.shape[-1] != width:
                        raise ValueError(f"layer {layer_no} o_proj input is {tuple(hidden.shape)}, expected width {width}")
                    if not named:
                        return None
                    replaced = hidden.clone()
                    batch, length = int(hidden.shape[0]), int(hidden.shape[1])
                    device, dtype = hidden.device, hidden.dtype
                    if length > 1:
                        first = prefill["first_real"]
                        mask = prefill["mask"]
                        if first is None or len(first) != batch:
                            raise ValueError(
                                f"mean ablation saw a prefill of {batch} rows but the attention mask recorded "
                                f"{0 if first is None else len(first)}: call the model with its attention mask")
                        if tuple(mask.shape) != (batch, length):
                            raise ValueError(
                                f"mean ablation's attention mask is {tuple(mask.shape)}, not the prefill's "
                                f"{(batch, length)}")
                        rows = torch.arange(batch, device=device)
                        first_index = torch.tensor(first, device=device)
                        real = mask.clone()
                        real[rows, first_index] = False
                        rest_positions = int(real.sum())
                        for start, end, head in named:
                            means = head_means[(layer_no, head)]
                            rest = means["mean_rest"].to(device=device, dtype=dtype)
                            pos0 = means["mean_pos0"].to(device=device, dtype=dtype)
                            replaced[..., start:end] = rest
                            for row, position in enumerate(first):
                                replaced[row, position, start:end] = pos0
                            record = records[head]
                            if "pos0" not in record:
                                before = hidden[..., start:end].float()
                                after = replaced[..., start:end]
                                pos0_before = before[rows, first_index]
                                pos0_after = after[rows, first_index]
                                rest_before = before[real]
                                rest_after = after[real]
                                record["pos0"] = {
                                    "positions": int(batch),
                                    "matches": bool(torch.equal(pos0_after, pos0.expand(batch, -1))),
                                    "max_abs_diff_before": float(
                                        (pos0_before - pos0.float().expand(batch, -1)).abs().max()),
                                    "first_real": [int(value) for value in first],
                                }
                                record["prefill_rest"] = {
                                    "positions": rest_positions,
                                    "matches": bool(torch.equal(rest_after, rest.expand(rest_positions, -1))),
                                    "max_abs_diff_before": (float(
                                        (rest_before - rest.float().expand(rest_positions, -1)).abs().max())
                                        if rest_positions else 0.0),
                                }
                                receipt.append(record)
                    else:
                        for start, end, head in named:
                            means = head_means[(layer_no, head)]
                            rest = means["mean_rest"].to(device=device, dtype=dtype)
                            before = hidden[..., start:end]
                            replaced[..., start:end] = rest
                            record = records[head]
                            if "decode" not in record:
                                record["decode"] = {
                                    "positions": int(batch * length),
                                    "matches": bool(torch.equal(replaced[..., start:end], rest.expand(batch, length, -1))),
                                    "max_abs_diff_before": float((before.float() - rest.float()).abs().max()),
                                }
                    return (replaced,) + tuple(args[1:])

            handles.append(attn.o_proj.register_forward_pre_hook(hook))
    except BaseException:
        # A raise part-way through instalment must not leave hooks behind for the next run in this process.
        for handle in handles:
            handle.remove()
        raise
    return handles, receipt


def snapshot_components(model, swap: str, layers: set[int] | None = None):
    """Clone the parameters ``copy_components`` is about to overwrite, so a run can put them back bit for bit.

    Part 219's reset receipt needs the model to be pristine again after a run that changed it: Part 99's ``--swap``
    is such a change, and a weight copied back from a clone is bit-identical where a recomputation would not be.
    """
    if swap not in {"mlp", "attn"}:
        raise ValueError("swap must be 'mlp' or 'attn'")
    module_name = "mlp" if swap == "mlp" else "self_attn"
    snapshot = []
    for layer_no, layer in enumerate(c.get_layers(model)):
        if layers is not None and layer_no not in layers:
            continue
        for _name, parameter in sorted(_named_parameters(getattr(layer, module_name)).items()):
            snapshot.append((parameter, parameter.detach().clone()))
    return snapshot


def restore_snapshot(snapshot) -> None:
    """Put cloned parameters back, bit for bit."""
    import torch

    with torch.no_grad():
        for parameter, original in snapshot:
            parameter.copy_(original)


def transplant_heads(host_model, donor_model, heads: list[tuple[int, int]], mode: str = "donor") -> dict:
    """Move named heads' weights to the donor's values, or the same distance the opposite way (Part 219).

    Per head: rows ``[hs:hs+head_size]`` of ``q_proj``, ``k_proj`` and ``v_proj``, columns ``[hs:hs+head_size]`` of
    ``o_proj``, and the same slice of ``q_norm.weight`` and ``k_norm.weight`` (the norms are per dimension, so those
    slices belong to the head). ``donor`` copies the donor's bytes; ``opposite`` sets ``2 * host - donor`` computed in
    float32 and cast back. The whole plan is shape-checked before a single value is written, and the returned
    ``restore`` list puts the host's original bytes back.
    """
    import torch

    if mode not in ("donor", "opposite"):
        raise ValueError("transplant mode must be 'donor' or 'opposite'")
    heads = list(heads)
    result = {"mode": mode, "heads": [f"L{layer}H{head}" for layer, head in heads], "records": [],
              "parameters_changed": 0, "elements_changed": 0, "restore": []}
    if not heads:
        return result
    if donor_model is None:
        raise ValueError("--transplant-heads needs --donor")
    host_layers, donor_layers = c.get_layers(host_model), c.get_layers(donor_model)
    if len(host_layers) != len(donor_layers):
        raise ValueError(f"layer count mismatch: host has {len(host_layers)}, donor has {len(donor_layers)}")
    head_size, num_heads = head_geometry(host_model)
    if head_geometry(donor_model) != (head_size, num_heads):
        raise ValueError(
            f"the donor's head geometry differs: host {num_heads} heads x {head_size}, "
            f"donor {head_geometry(donor_model)[1]} heads x {head_geometry(donor_model)[0]}")

    plans = []
    for layer_no, head in heads:
        start, end = head_span(host_model, layer_no, head)
        host_attn = host_layers[layer_no].self_attn
        donor_attn = donor_layers[layer_no].self_attn
        for part in TRANSPLANT_PARTS:
            host_param = getattr(host_attn, part).weight
            donor_param = getattr(donor_attn, part).weight
            if tuple(host_param.shape) != tuple(donor_param.shape):
                raise ValueError(
                    f"the donor's shapes differ at layer {layer_no} self_attn.{part}: "
                    f"host {tuple(host_param.shape)} vs donor {tuple(donor_param.shape)}")
            index = head_slice(part, host_param.ndim, start, end)
            plans.append((layer_no, head, part, start, end, host_param, donor_param, index))

    written_restore = []
    try:
        with torch.no_grad():
            for layer_no, head, part, start, end, host_param, donor_param, index in plans:
                # Re-derive the span from the config by the named head alone. The write below may have been planned
                # with a wrong span (that is exactly what the receipt must catch); this one is the design's
                # [head*head_size:(head+1)*head_size] and the bytes are read back at *this* index, independently.
                config_start, config_end = head * head_size, (head + 1) * head_size
                check_index = head_slice(part, host_param.ndim, config_start, config_end)
                original = host_param[index].detach().clone()
                original_check = host_param[check_index].detach().clone()
                donor_slice = donor_param[index].detach().to(device=host_param.device, dtype=torch.float32)
                if mode == "donor":
                    expected = donor_slice.to(dtype=host_param.dtype)
                else:
                    expected = (2.0 * original.to(torch.float32) - donor_slice).to(dtype=host_param.dtype)
                written_restore.append((host_param, index, original))
                host_param[index].copy_(expected)
                written = host_param[index].detach()

                donor_check = donor_param[check_index].detach().to(device=host_param.device, dtype=torch.float32)
                if mode == "donor":
                    expected_check = donor_check.to(dtype=host_param.dtype)
                else:
                    expected_check = (2.0 * original_check.to(torch.float32) - donor_check).to(dtype=host_param.dtype)
                written_check = host_param[check_index].detach()
                slice_from_config = [start, end] == [config_start, config_end]
                matches = slice_from_config and bool(torch.equal(written_check, expected_check))
                original_norm = float(original.to(torch.float32).norm())
                result["records"].append({
                    "layer": layer_no, "head": head,
                    "parameter": f"layers.{layer_no}.self_attn.{part}.weight",
                    "slice": [start, end],
                    "config_slice": [config_start, config_end],
                    "slice_from_config": slice_from_config,
                    "matches_expected": matches,
                    "norm_delta": float((written.to(torch.float32) - original.to(torch.float32)).norm()),
                    "norm_ratio": (float(written.to(torch.float32).norm()) / original_norm) if original_norm else None,
                    "elements": int(written.numel()),
                })
                result["parameters_changed"] += 1
                result["elements_changed"] += int(written.numel())
    except BaseException:
        # A raise part-way through the write must not leave a partly transplanted host behind.
        restore_transplant(written_restore)
        raise
    result["restore"] = written_restore
    return result


def restore_transplant(restore) -> None:
    """Put the host's original sliced bytes back, bit for bit."""
    import torch

    with torch.no_grad():
        for parameter, index, original in restore:
            parameter[index].copy_(original)


def _rms(tensor) -> float:
    return float(tensor.detach().float().pow(2).mean().sqrt())


def attention_probe(model, input_ids, attention_mask, push_layer, row_vectors, watch, head_size: int) -> dict:
    """One forward on the first batch, recording the quantities the coupling log compares before and after a
    transplant: rms(q) and rms(k) after the projections but before the QK norms, rms of the attention output before
    ``post_attention_layernorm``, and every head's own output vector (the ``o_proj`` input, which is the heads'
    concatenated outputs)."""
    import torch

    layers = c.get_layers(model)
    captured: dict[int, dict] = {}

    def bucket(layer_no):
        return captured.setdefault(layer_no, {})

    handles = []
    for layer_no in watch:
        attn = layers[layer_no].self_attn

        def q_hook(_module, _args, output, layer_no=layer_no):
            bucket(layer_no)["q"] = output.detach()

        def k_hook(_module, _args, output, layer_no=layer_no):
            bucket(layer_no)["k"] = output.detach()

        def o_in_hook(_module, args, layer_no=layer_no):
            bucket(layer_no)["heads"] = args[0].detach()

        def attn_out_hook(_module, args, layer_no=layer_no):
            bucket(layer_no)["attn_out"] = args[0].detach()

        handles.append(attn.q_proj.register_forward_hook(q_hook))
        handles.append(attn.k_proj.register_forward_hook(k_hook))
        handles.append(attn.o_proj.register_forward_pre_hook(o_in_hook))
        handles.append(layers[layer_no].post_attention_layernorm.register_forward_pre_hook(attn_out_hook))
    push_handle = None
    if row_vectors is not None:
        push_handle = c.add_row_hook(layers[push_layer], row_vectors)
    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
        if push_handle is not None:
            push_handle.remove()

    summary = {}
    for layer_no in watch:
        got = captured.get(layer_no, {})
        missing = [name for name in ("q", "k", "heads", "attn_out") if name not in got]
        if missing:
            raise RuntimeError(f"the layer {layer_no} probe did not see: {', '.join(missing)}")
        heads = got["heads"].float().reshape(*got["heads"].shape[:-1], -1, head_size)
        summary[layer_no] = {
            "q_rms": _rms(got["q"]),
            "k_rms": _rms(got["k"]),
            "attn_rms": _rms(got["attn_out"]),
            "per_head_rms": [_rms(heads[..., index, :]) for index in range(heads.shape[-2])],
            "head_outputs": [heads[..., index, :].reshape(-1, head_size).float() for index in range(heads.shape[-2])],
        }
    return summary


def coupling_log(before: dict, after: dict, changed: dict[int, set[int]], batch_ids) -> dict:
    """First-batch relative changes a transplant caused, per changed layer (PART219-DESIGN.md, Controls).

    ``changed`` maps a layer to the head indices transplanted in it; those heads' own outputs are reported but the
    sibling drift is computed over the heads the transplant did not touch, which is what the coupling question asks.
    """
    def relative(old, new):
        return (new - old) / old if old else None

    layers = []
    for layer_no in sorted(before):
        old, new = before[layer_no], after[layer_no]
        siblings = []
        for head, (old_vec, new_vec) in enumerate(zip(old["head_outputs"], new["head_outputs"])):
            if head in changed.get(layer_no, set()):
                continue
            baseline = float(old_vec.norm())
            siblings.append({"head": head,
                             "relative_change": (float((new_vec - old_vec).norm()) / baseline) if baseline else None})
        values = [item["relative_change"] for item in siblings if item["relative_change"] is not None]
        changed_heads = []
        for head in sorted(changed.get(layer_no, set())):
            baseline = float(old["head_outputs"][head].norm())
            changed_heads.append({"head": head, "relative_change": (
                float((new["head_outputs"][head] - old["head_outputs"][head]).norm()) / baseline)
                if baseline else None})
        layers.append({
            "layer": layer_no,
            "changed_heads": sorted(changed.get(layer_no, set())),
            "rms_q": {"before": old["q_rms"], "after": new["q_rms"],
                      "relative_change": relative(old["q_rms"], new["q_rms"])},
            "rms_k": {"before": old["k_rms"], "after": new["k_rms"],
                      "relative_change": relative(old["k_rms"], new["k_rms"])},
            "rms_attention_output": {"before": old["attn_rms"], "after": new["attn_rms"],
                                     "relative_change": relative(old["attn_rms"], new["attn_rms"])},
            "sibling_output_drift": {
                "heads": siblings,
                "mean_relative_change": (sum(values) / len(values)) if values else None,
                "max_relative_change": max(values) if values else None,
            },
            "changed_head_output_change": changed_heads,
        })
    return {"batch": batch_ids, "layers": layers}


def replay_receipt(model, tokenizer, frame: str, device: str) -> dict:
    """Replay one stored prompt on the restored model and record what a comparison needs (reset receipt).

    ``prompt_ids`` is the stored prompt's token ids, ``token_ids`` the greedy continuation of ``REPLAY_TOKENS``
    tokens, and ``first_step_logits`` the full next-token distribution from a plain forward on the same prompt. Two
    runs compare equal only if all three match exactly.
    """
    import torch

    ids = c.prompt_ids(tokenizer, REPLAY_QUESTION, frame)
    if not ids:
        raise ValueError("the stored replay prompt rendered to no tokens")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1].float()
        generated = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, do_sample=False,
            max_new_tokens=REPLAY_TOKENS, pad_token_id=int(tokenizer.pad_token_id),
        )
    return {
        "question": REPLAY_QUESTION,
        "frame": frame,
        "prompt_ids": [int(token) for token in ids],
        "token_ids": [int(token) for token in generated[0, len(ids):]],
        "first_step_argmax": int(logits.argmax().item()),
        "first_step_logits": [float(value) for value in logits.tolist()],
    }


def compare_replay(receipt: dict, reference: dict) -> str | None:
    """None when a reset receipt equals the stored replay exactly, otherwise a short description of the first
    difference."""
    for field in ("prompt_ids", "token_ids", "first_step_logits"):
        mine, theirs = receipt.get(field), reference.get(field)
        if mine is None or theirs is None:
            return f"{field} is missing from {'the receipt' if mine is None else 'the reference'}"
        if len(mine) != len(theirs):
            return f"{field} has {len(mine)} values, the reference has {len(theirs)}"
        for index, (a, b) in enumerate(zip(mine, theirs)):
            if a != b:
                return f"{field}[{index}] is {a!r}, the reference has {b!r}"
    return None


def write_meta(out_dir: Path, tag: str, *, host: str, donor: str | None, swap: str,
               tokenizer_source: str, swap_result: dict, frame: str, device: str,
               torch_version: str, transformers_version: str, seconds: float,
               questions: tuple[int, ...] | None = None, extras: dict | None = None) -> dict:
    """Write a run's meta JSON.

    ``extras`` (Part 219) carries this part's options and receipts; a key whose value is None is left out, so every
    Part 99 to 104 run writes exactly the meta it wrote before this option existed."""
    meta = {
        "host": host,
        "donor": donor,
        "swap": swap,
        "tokenizer_source": tokenizer_source,
        "tensors_copied": int(swap_result["tensors_copied"]),
        "elements_copied": int(swap_result["elements_copied"]),
        "copied_parameters": list(swap_result["copied_parameters"]),
        "mode": swap_result.get("mode", "copy"),
        "distance_ratio": swap_result.get("distance_ratio"),
        "frame": frame,
        "device": device,
        "torch_version": torch_version,
        "transformers_version": transformers_version,
        "seconds": float(seconds),
        "questions": list(questions) if questions is not None else list(range(len(QUESTIONS))),
    }
    for key, value in (extras or {}).items():
        if value is not None:
            meta[key] = value
    (out_dir / f"{tag}.meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def _startup_copy_description(swap_result: dict) -> str:
    if not swap_result["copied_parameters"]:
        return "none"
    module = swap_result["module"]
    names_by_layer = {}
    for full_name in swap_result["copied_parameters"]:
        _, layer_no, component, parameter = full_name.split(".", 3)
        names_by_layer.setdefault(int(layer_no), []).append(parameter)
    name_sets = {tuple(names) for names in names_by_layer.values()}
    if len(name_sets) == 1:
        names = next(iter(name_sets))
        return f"layers[*].{module}.{{{','.join(names)}}}"
    return ",".join(swap_result["copied_parameters"])


def _resolve_tokenizer(args, host_tokenizer):
    """Pick the run's tokenizer and record where it came from (Part 99's optional donor tokenizer)."""
    from transformers import AutoTokenizer

    tokenizer_source = str(Path(args.tokenizer or args.host).expanduser())
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, local_files_only=True)
        assert_tokenizers_compatible(host_tokenizer, tokenizer)
    else:
        tokenizer = host_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, tokenizer_source


def run(args, *, model=None, tokenizer=None, donor_model=None):
    """Run one tag.

    ``model``/``tokenizer``/``donor_model`` (Part 219) let a caller reuse one host load for several tags: when they
    are given the models are not loaded here and the donor is not freed here. With none of them given, this is
    exactly the single-tag command Part 99 has always run.
    """
    import torch
    import transformers

    run_started = time.perf_counter()
    with Path(args.neurons).open(encoding="utf-8") as stream:
        neurons = json.load(stream)
    if not isinstance(neurons, list):
        raise ValueError("--neurons must contain a JSON list")
    only = parse_only(args.only)
    questions = parse_questions(getattr(args, "questions", None))
    if only is not None:
        if not args.kinds:
            raise ValueError("--only requires --kinds")
        kinds = load_kind_map(args.kinds)
        neurons = [neuron for neuron in neurons if kinds.get(str(neuron.get("id"))) in only]
    if args.limit is not None:
        neurons = neurons[:args.limit]
    if not neurons:
        raise ValueError("no neurons remain after applying --only and --limit")

    attn_option = getattr(args, "attn_implementation", None)
    attn_implementation = None if attn_option in (None, "default") else attn_option
    ablate_heads = parse_heads(getattr(args, "ablate_heads", None))
    ablate_mode = getattr(args, "ablate_mode", None) or "zero"
    if ablate_mode not in ABLATE_MODES:
        raise ValueError(f"--ablate-mode must be one of {ABLATE_MODES}; got {ablate_mode!r}")
    head_means_path = getattr(args, "head_means", None)
    head_means, head_means_sha256 = None, None
    if ablate_mode == "mean":
        if not head_means_path:
            raise ValueError("--ablate-mode mean needs --head-means")
        head_means = load_head_means(head_means_path)
        head_means_sha256 = file_sha256(head_means_path)
    elif head_means_path:
        raise ValueError("--head-means is only read with --ablate-mode mean")
    transplant_list = parse_heads(getattr(args, "transplant_heads", None))
    transplant_mode = getattr(args, "transplant_mode", None) or "donor"
    reference_replay = getattr(args, "reference_replay", None)
    record_ids = bool(getattr(args, "record_token_ids", False))

    owns_models = model is None
    needs_donor = args.swap != "none" or bool(transplant_list)
    if model is None:
        host_tokenizer, model = c.load_model(args.host, args.device,
                                             attn_implementation=attn_implementation)
        tokenizer, tokenizer_source = _resolve_tokenizer(args, host_tokenizer)
        if needs_donor:
            if not args.donor:
                raise ValueError(f"--donor is required when --swap is {args.swap} and for --transplant-heads")
            donor_model = _load_donor(args.donor, attn_implementation=attn_implementation)
    else:
        if tokenizer is None:
            raise ValueError("a preloaded model needs its tokenizer")
        tokenizer, tokenizer_source = _resolve_tokenizer(args, tokenizer)
        if needs_donor and donor_model is None:
            raise ValueError(f"--swap {args.swap} needs a donor model when the host is preloaded")

    # Part 219's reset receipt needs the model exactly as it was loaded: hooks off, transplanted slices back and
    # Part 99's swapped parameters back -- and that must hold when a run raises, not only when it finishes. One
    # try/finally therefore runs from the first mutation below (Part 99's --swap, then the transplant probe/copy
    # and the ablation hooks) to the end of the generation loop. When the caller preloaded the model it keeps it,
    # so the original bytes are cloned even without --reference-replay; a single-tag Part 99-104 command that owns
    # its model still clones nothing unless the reset receipt was asked for.
    swap_snapshot = []
    swap_result = None
    transplant_result = None
    coupling = None
    ablation_handles, ablation_receipt = [], []
    try:
        if args.swap != "none" and (reference_replay or not owns_models):
            swap_snapshot = snapshot_components(model, args.swap, parse_layers(args.layers))
        swap_result = copy_components(model, donor_model, args.swap, layers=parse_layers(args.layers),
                                      extrapolate=args.extrapolate)
        if donor_model is not None and owns_models and not transplant_list:
            # Part 99 freed the donor as soon as the copy was done; a transplant still needs it for the probe and the copy.
            del donor_model
            donor_model = None
            gc.collect()

        layers = c.get_layers(model)
        medians = c.residual_medians(model, tokenizer, args.frame, args.device)
        prepared = {}
        for neuron in neurons:
            layer_no = int(neuron["layer"])
            index = int(neuron["neuron"])
            if not 0 <= layer_no < len(layers):
                raise ValueError(f"layer {layer_no} outside model range")
            weight = layers[layer_no].mlp.down_proj.weight
            if not 0 <= index < weight.shape[1]:
                raise ValueError(f"neuron {index} outside layer {layer_no} W_down input width")
            target_norm = ALPHA * medians[layer_no]
            prepared[(str(neuron["id"]), "neuron")] = c.neuron_push(weight, index, target_norm)
            prepared[(str(neuron["id"]), "random")] = c.random_direction(
                weight, int(neuron["control_seed"]), target_norm
            )
        if args.pushes:
            # Amendment 1 (Alden's review): one fixed injected vector per unit for every host, built once on the
            # unmodified Instruct model in the chat frame (make_pushes99.py); host recalibration is not used.
            fixed = torch.load(args.pushes, map_location="cpu")
            for key in list(prepared):
                vec = fixed[f"{key[0]}|{key[1]}"]
                ref = prepared[key]
                prepared[key] = vec.to(device=ref.device, dtype=ref.dtype)

        print(
            f"{args.tag}: host={args.host} donor={args.donor or '-'} swap={args.swap} "
            f"copied={_startup_copy_description(swap_result)} "
            f"tensors_copied={swap_result['tensors_copied']} "
            f"elements_copied={swap_result['elements_copied']}",
            flush=True,
        )

        work_by_layer = {}
        for neuron in neurons:
            layer_no = int(neuron["layer"])
            work_by_layer.setdefault(layer_no, []).extend(
                (neuron, unit, qi) for unit in ("neuron", "random") for qi in questions
            )
        # Unpushed answers to the six questions (a zero push), so each run's coherence without any push is on record
        # (PART99-DESIGN.md, prior-art section). Written with id "unpushed" and unit "none".
        first = neurons[0]
        unpushed = {"id": "unpushed", "layer": int(first["layer"])}
        prepared[("unpushed", "none")] = prepared[(str(first["id"]), "neuron")] * 0
        work_by_layer.setdefault(int(first["layer"]), []).extend(
            (unpushed, "none", qi) for qi in range(len(QUESTIONS))
        )

        # Part 219, transplant: measure the first batch before the head weights move, move them, measure the same batch
        # again. The probe carries the run's own row pushes, so it sees the batch the first generation call sees.
        if transplant_list is not None:
            if transplant_list and donor_model is None:
                raise ValueError("--transplant-heads needs --donor")
            if transplant_list:
                head_size, _num_heads = head_geometry(model)
                probe_chunk = work_by_layer[next(iter(work_by_layer))][:max(args.batch, 1)]
                probe_ids, probe_mask, _width = c.batch_inputs(
                    tokenizer, [QUESTIONS[qi] for _neuron, _unit, qi in probe_chunk], args.frame, args.device)
                probe_pushes = torch.stack([
                    prepared[(str(neuron["id"]), unit)] for neuron, unit, _qi in probe_chunk])
                watch = sorted({layer_no for layer_no, _head in transplant_list})
                before = attention_probe(model, probe_ids, probe_mask, int(probe_chunk[0][0]["layer"]),
                                         probe_pushes, watch, head_size)
                transplant_result = transplant_heads(model, donor_model, transplant_list, transplant_mode)
                after = attention_probe(model, probe_ids, probe_mask, int(probe_chunk[0][0]["layer"]),
                                        probe_pushes, watch, head_size)
                coupling = coupling_log(
                    before, after,
                    {layer_no: {head for layer, head in transplant_list if layer == layer_no} for layer_no in watch},
                    {"ids": [str(neuron["id"]) for neuron, _unit, _qi in probe_chunk],
                     "units": [unit for _neuron, unit, _qi in probe_chunk],
                     "question_indices": [int(qi) for _neuron, _unit, qi in probe_chunk],
                     "batch": int(args.batch), "positions": int(probe_ids.shape[1])})
            else:
                # An empty set enters the transplant path and returns from `transplant_heads` before the
                # geometry/donor/shape planning: R0b exercises that empty early return (and the empty ablation hooks),
                # not the planner. The shape checks are covered by the transplant tests.
                transplant_result = transplant_heads(model, donor_model, [], transplant_mode)

        if ablate_heads is not None:
            ablation_handles, ablation_receipt = install_ablation_hooks(
                model, ablate_heads, ablate_mode=ablate_mode, head_means=head_means)

        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / f"{args.tag}.jsonl"
        existing = set()
        if output.exists():
            with output.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    existing.add((str(row["id"]), row["unit"], int(row["question_index"])))
        started = time.perf_counter()
        total = sum(
            sum((str(row[0]["id"]), row[1], row[2]) not in existing for row in rows)
            for rows in work_by_layer.values()
        )
        done = 0
        with output.open("a", encoding="utf-8") as stream:
            for layer_no, layer_work in work_by_layer.items():
                layer_work = [row for row in layer_work
                              if (str(row[0]["id"]), row[1], row[2]) not in existing]
                layer = layers[layer_no]
                for begin in range(0, len(layer_work), args.batch):
                    chunk = layer_work[begin:begin + args.batch]
                    pushes = torch.stack([
                        prepared[(str(neuron["id"]), unit)] for neuron, unit, _qi in chunk
                    ])
                    generated = c.generate_batch(
                        model, tokenizer, [QUESTIONS[qi] for _neuron, _unit, qi in chunk],
                        args.frame, layer, pushes, 120, args.device, return_ids=record_ids,
                    )
                    answers, token_ids = generated if record_ids else (generated, None)
                    for position, ((neuron, unit, qi), answer) in enumerate(zip(chunk, answers)):
                        row = {"id": neuron["id"], "unit": unit, "question_index": qi,
                               "question": QUESTIONS[qi], "answer": answer}
                        if record_ids:
                            row["token_ids"] = token_ids[position]
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    done += len(chunk)
                    if done % max(args.batch * 5, 1) < len(chunk) or done == total:
                        elapsed = time.perf_counter() - started
                        print(f"{args.tag}: {done}/{total} rows in {elapsed:.1f}s", flush=True)
        needs_receipt = bool(ablate_heads) and (
            not ablation_receipt
            # Mean mode's receipt also checks a decode step, and a generation whose rows all stop on their first
            # token never makes one: take the probe then too, so the receipt is complete rather than failing.
            or (ablate_mode == "mean" and any("decode" not in record for record in ablation_receipt)))
        if needs_receipt:
            # A resumed run that found every row on disk never ran a forward, so the first-batch receipt was never
            # taken. Take it now, while the hooks are still installed, on the batch a fresh run would have used.
            # Mean mode's probe is a two-token generate with a two-token minimum: the prefill is the same first batch a
            # fresh run would have used, and ``min_new_tokens`` forces the decode call its receipt needs.
            first_chunk = work_by_layer[next(iter(work_by_layer))][:max(args.batch, 1)]
            probe_ids, probe_mask, _width = c.batch_inputs(
                tokenizer, [QUESTIONS[qi] for _neuron, _unit, qi in first_chunk], args.frame, args.device)
            with torch.inference_mode():
                if ablate_mode == "mean":
                    model.generate(input_ids=probe_ids, attention_mask=probe_mask, do_sample=False,
                                   max_new_tokens=2, min_new_tokens=2, pad_token_id=int(tokenizer.pad_token_id))
                else:
                    model(input_ids=probe_ids, attention_mask=probe_mask, use_cache=False)
    finally:
        # Part 219's reset receipt needs the model exactly as it was loaded: hooks off, transplanted slices back,
        # Part 99's swapped parameters back. Each restore writes the original bytes, so it is bit-exact.
        for handle in ablation_handles:
            handle.remove()
        if transplant_result:
            restore_transplant(transplant_result.get("restore", []))
        if swap_snapshot:
            restore_snapshot(swap_snapshot)

    reset = None
    if reference_replay:
        reset = replay_receipt(model, tokenizer, args.frame, args.device)
        path = Path(reference_replay)
        if path.exists():
            reference = json.loads(path.read_text(encoding="utf-8"))
            mismatch = compare_replay(reset, reference)
            reset.update({"role": "checked", "reference_path": str(path),
                          "matches_reference": mismatch is None, "mismatch": mismatch})
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(reset, indent=1) + "\n", encoding="utf-8")
            reset.update({"role": "reference_written", "reference_path": str(path),
                          "matches_reference": None, "mismatch": None})

    intervention = None
    if ablate_heads is not None or transplant_result is not None:
        intervention = {"ablate_heads": None if ablate_heads is None else [f"L{l}H{h}" for l, h in ablate_heads],
                        "ablation": ablation_receipt,
                        "transplant": None if transplant_result is None else
                        {key: value for key, value in transplant_result.items() if key != "restore"}}
        if ablate_heads is not None and ablate_mode != "zero":
            # Amendment 1's runs say which operation they used and which means file they read; a zeroing run, and
            # every pre-amendment meta, keeps exactly the keys it had.
            intervention["ablate_mode"] = ablate_mode
            intervention["head_means_sha256"] = head_means_sha256

    seconds = time.perf_counter() - run_started
    write_meta(
        out_dir, args.tag, host=str(Path(args.host).expanduser()),
        donor=str(Path(args.donor).expanduser()) if args.donor else None,
        swap=args.swap, tokenizer_source=tokenizer_source, swap_result=swap_result,
        frame=args.frame, device=args.device, torch_version=torch.__version__,
        transformers_version=transformers.__version__, seconds=seconds, questions=questions,
        extras={
            # Not `attn_option`: argparse's default is the *string* "default", which write_meta would happily record.
            # `attn_implementation` is None for the default and "eager" when asked, so a default run writes no key.
            "attn_implementation": attn_implementation,
            "ablate_heads": None if ablate_heads is None else [f"L{layer}H{head}" for layer, head in ablate_heads],
            "ablate_mode": (ablate_mode if ablate_heads is not None and ablate_mode != "zero" else None),
            "head_means": (str(Path(head_means_path).expanduser())
                           if ablate_heads is not None and head_means_sha256 else None),
            "head_means_sha256": (head_means_sha256 if ablate_heads is not None else None),
            "transplant_heads": None if transplant_list is None else
            [f"L{layer}H{head}" for layer, head in transplant_list],
            "transplant_mode": transplant_mode if transplant_list else None,
            "token_ids_recorded": True if record_ids else None,
            "intervention_receipt": intervention,
            "coupling_log": coupling,
            "reset_receipt": reset,
        },
    )
    print(f"{args.tag}: completed in {seconds:.1f}s; meta={out_dir / (args.tag + '.meta.json')}", flush=True)
    return {"swap": swap_result, "intervention": intervention, "coupling": coupling, "reset": reset}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True, help="local host snapshot directory")
    parser.add_argument("--donor", help="optional local donor snapshot directory")
    parser.add_argument("--swap", choices=("none", "mlp", "attn"), default="none")
    parser.add_argument("--tokenizer", help="optional local tokenizer snapshot directory")
    parser.add_argument("--neurons", required=True, help="JSON list of id/layer/neuron/persona/desc/control_seed")
    parser.add_argument("--frame", required=True, choices=("chat", "plain"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("cpu", "xpu", "cuda"), default="xpu")
    parser.add_argument("--batch", type=int, default=48)
    parser.add_argument("--kinds", help="Part 92 score JSON with a kinds id-to-kind map")
    parser.add_argument("--only", help="comma-separated concept,slot filter; requires --kinds")
    parser.add_argument("--pushes", help="fixed push vectors from make_pushes99.py (Amendment 1)")
    parser.add_argument("--limit", type=int, help="run only the first N eligible neurons")
    parser.add_argument("--layers", help="Part 100: copy the swapped component only in this inclusive layer range, e.g. 8-15")
    parser.add_argument("--extrapolate", action="store_true",
                        help="Part 104: set the swapped component to 2 x host - donor instead of copying the donor")
    parser.add_argument("--questions",
                        help="Part 108: comma-separated question indices for the pushed answers (default all six)")
    parser.add_argument("--attn-implementation", choices=("default", "eager"), default="default",
                        help="Part 219: attention kernel passed to every from_pretrained (default: the library's)")
    parser.add_argument("--ablate-heads",
                        help="Part 219: comma-separated L<layer>H<head> whose o_proj input slice is zeroed; "
                             "give an empty value to install the hooks and change nothing")
    parser.add_argument("--ablate-mode", choices=ABLATE_MODES, default="zero",
                        help="Part 219 Amendment 1: zero the slice (the registered pilot's operation, the default) "
                             "or replace it with the stored per-head means")
    parser.add_argument("--head-means",
                        help="Part 219 Amendment 1: means219.py's head-means.pt, required with --ablate-mode mean")
    parser.add_argument("--transplant-heads",
                        help="Part 219: comma-separated L<layer>H<head> whose q/k/v rows, o_proj columns and q/k norm "
                             "slices are taken from the donor")
    parser.add_argument("--transplant-mode", choices=("donor", "opposite"), default="donor",
                        help="Part 219: copy the donor's slices, or move 2 x host - donor (Part 105's distance control)")
    parser.add_argument("--reference-replay",
                        help="Part 219: file holding the stored prompt's R0 token ids and first-step logits; written "
                             "when absent, compared exactly when present")
    parser.add_argument("--record-token-ids", action="store_true",
                        help="Part 219: add each row's generated token ids, so arms can be compared token for token")
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    try:
        parse_only(args.only)
        parse_questions(args.questions)
    except ValueError as exc:
        parser.error(str(exc))
    if args.only and not args.kinds:
        parser.error("--only requires --kinds")
    if args.swap != "none" and not args.donor:
        parser.error(f"--donor is required when --swap is {args.swap}")
    if args.layers is not None and args.swap == "none":
        parser.error("--layers needs --swap mlp or attn")
    if args.extrapolate and args.swap == "none":
        parser.error("--extrapolate needs --swap mlp or attn")
    try:
        ablate_heads = parse_heads(args.ablate_heads)
        transplant_heads_list = parse_heads(args.transplant_heads)
    except ValueError as exc:
        parser.error(str(exc))
    if transplant_heads_list and not args.donor:
        parser.error("--transplant-heads needs --donor")
    if args.transplant_mode != "donor" and not transplant_heads_list:
        parser.error("--transplant-mode needs --transplant-heads")
    if args.ablate_mode == "mean":
        if not args.head_means:
            parser.error("--ablate-mode mean requires --head-means")
        if args.ablate_heads is None:
            parser.error("--ablate-mode needs --ablate-heads (an empty value installs the hooks and changes nothing)")
    elif args.head_means:
        parser.error("--head-means is only read with --ablate-mode mean")
    run(args)


if __name__ == "__main__":
    main()
