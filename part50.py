#!/usr/bin/env python3
"""Part 50: honesty direction, calibration, role-play and persona scoring."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
ITEMS_ROLEPLAY = ROOT / "third_party/honesty-heads/data/items_roleplay_insider.json"
ITEMS_PERSONA = ROOT / "third_party/honesty-heads/data/items_persona.json"
BUNDLE_PATH = ROOT / "data/part50-bundle.json"
SOURCE_STATS = ROOT / "third_party/honesty-heads/results/persona_stats.json"
HF_CACHE = Path("/mnt/e/neuron-probe/hf/hub")
SPAN = 9
DEFAULT_RANDOMS = 39
DEFAULT_SEED = 50050
LN2 = math.log(2.0)

MODEL_SPECS = {
    "qwen3-1.7b": {
        "model_id": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "chat": True,
    },
    "qwen3-1.7b-base": {
        "model_id": "Qwen/Qwen3-1.7B-Base",
        "revision": "ea980cb0a6c2ae4b936e82123acc929f1cec04c1",
        "chat": False,
    },
}


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(value: dict, path: str | Path) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, dest)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_bundle(source: str | Path = SOURCE_STATS, output: str | Path = BUNDLE_PATH) -> dict:
    """Freeze the 51 source traits identified by Sauers' permutation FDR."""
    source, output = Path(source), Path(output)
    if not source.is_file():
        raise FileNotFoundError(f"bundle source does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"frozen bundle already exists: {output}; remove it manually only before a model run")
    stats = json.loads(source.read_text(encoding="utf-8"))
    rows = stats.get("traits")
    if not isinstance(rows, list):
        raise ValueError("source results have no per-trait records")
    changed = [row for row in rows if float(row.get("q_perm", 1.0)) <= 0.05]
    if len(changed) != 51:
        raise ValueError(f"source results contain {len(changed)} traits at permutation FDR 5%; expected 51 signed traits")
    traits = []
    for row in changed:
        effect = float(row["T_set"])
        if not math.isfinite(effect) or effect == 0:
            raise ValueError(f"source trait has no nonzero signed effect: {row.get('trait')}")
        traits.append({
            "trait": row["trait"],
            "sign": 1 if effect > 0 else -1,
            "qwen35_9b_mean_signed_change_bits": effect,
            "q_perm": float(row["q_perm"]),
        })
    record = {
        "schema_version": 1,
        "model": "Qwen/Qwen3.5-9B",
        "selection_rule": "persona_stats.json q_perm <= 0.05; sign from T_set",
        "n_traits": len(traits),
        "source_file": str(source.relative_to(ROOT)) if source.is_relative_to(ROOT) else str(source),
        "source_sha256": sha256_file(source),
        "traits": traits,
    }
    atomic_json(record, output)
    return record


def read_bundle(path: str | Path = BUNDLE_PATH) -> dict:
    """Read and validate the frozen bundle. This function never writes its input."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    traits = data.get("traits")
    if data.get("n_traits") != 51 or not isinstance(traits, list) or len(traits) != 51:
        raise ValueError("bundle must contain exactly 51 frozen traits")
    names = [r.get("trait") for r in traits]
    if len(set(names)) != 51 or any(r.get("sign") not in (-1, 1) for r in traits):
        raise ValueError("bundle traits must have unique names and signs of -1 or +1")
    if not data.get("source_file") or not data.get("source_sha256"):
        raise ValueError("bundle source provenance is missing")
    return data


def ensure_bundle(path: str | Path = BUNDLE_PATH) -> tuple[dict, str]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"frozen bundle is missing; run `python part50.py bundle` before any model command")
    return read_bundle(path), sha256_file(path)


def _vendor_persona_stats():
    path = ROOT / "third_party/honesty-heads/code/persona_stats.py"
    spec = importlib.util.spec_from_file_location("part50_vendor_persona_stats", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load vendored persona_stats.py at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def perm_fdr(matrix: np.ndarray) -> np.ndarray:
    """Sauers' exact permutation-FDR algorithm, kept local for the Part 50 result schema."""
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("perm_fdr expects [traits, main + random directions]")
    z = np.abs((a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True))
    z0, zr = z[:, 0], z[:, 1:]
    order = np.argsort(-z0)
    f = np.array([
        min(1.0, (zr >= c - 1e-12).sum(0).mean() / (z0 >= c - 1e-12).sum())
        for c in z0[order]
    ])
    q = np.minimum.accumulate(f[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = q
    return out


def make_random_directions(direction: torch.Tensor, count: int = DEFAULT_RANDOMS,
                           seed: int = DEFAULT_SEED) -> torch.Tensor:
    """Sample independent per-head random vectors with exactly the source head norms."""
    if direction.ndim != 3:
        raise ValueError("direction must have shape [layers, heads, head_dim]")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    source = direction.detach().float().cpu()
    random = torch.randn((count, *source.shape), generator=generator, dtype=torch.float32)
    norms = source.norm(dim=-1, keepdim=True)
    random_norms = random.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(random.dtype).tiny)
    return random / random_norms * norms.unsqueeze(0)


class HeadOutputSteering:
    """Adds one direction slice per attention head at each o_proj input, while enabled."""

    def __init__(self, model, heads: int | None = None, head_dim: int | None = None):
        self.model = model
        self.layers = list(model.model.layers)
        self.heads = int(heads or model.config.num_attention_heads)
        inferred = getattr(model.config, "head_dim", None)
        if inferred is None:
            inferred = model.config.hidden_size // self.heads
        self.head_dim = int(head_dim or inferred)
        self.active = False
        self.bank: torch.Tensor | None = None
        self.arm_indices: torch.Tensor | None = None
        self.scales: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.capture_positions: torch.Tensor | None = None
        self.captured: dict[int, torch.Tensor] = {}
        self.handles = []
        for index, layer in enumerate(self.layers):
            attention = getattr(layer, "self_attn", None)
            projection = getattr(attention, "o_proj", None)
            if projection is None:
                raise ValueError(f"layer {index} has no self_attn.o_proj; unsupported model structure")
            self.handles.append(projection.register_forward_pre_hook(self._hook(index)))

    def _hook(self, layer_index: int):
        def hook(_module, inputs):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return None
            x = inputs[0]
            if x.ndim != 3 or x.shape[-1] != self.heads * self.head_dim:
                raise ValueError(f"layer {layer_index} o_proj input has shape {tuple(x.shape)}; expected [B,S,{self.heads * self.head_dim}]")
            if self.capture_positions is not None:
                positions = self.capture_positions.to(x.device)
                valid = positions >= 0
                safe = positions.clamp(min=0, max=x.shape[1] - 1)
                batch = torch.arange(x.shape[0], device=x.device)[:, None]
                selected = x[batch, safe].float()
                selected = selected * valid[:, :, None].to(selected.dtype)
                denom = valid.sum(1).clamp_min(1)[:, None]
                self.captured[layer_index] = (selected.sum(1) / denom).reshape(
                    x.shape[0], self.heads, self.head_dim
                ).detach().cpu()
            if not self.active or self.bank is None:
                return None
            if self.arm_indices is None or self.scales is None:
                raise RuntimeError("steering is active without row directions")
            row_ids = self.arm_indices.to(x.device)
            delta = self.bank[row_ids, layer_index].to(device=x.device, dtype=torch.float32)
            delta = delta * self.scales.to(x.device, dtype=torch.float32)[:, None, None]
            delta = delta[:, None, :, :]
            if self.attention_mask is not None:
                mask = self.attention_mask.to(x.device, dtype=torch.float32)[:, :, None, None]
                delta = delta * mask
            shifted = x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).float() + delta
            replacement = shifted.to(x.dtype).reshape_as(x)
            return (replacement, *inputs[1:])
        return hook

    def set_batch(self, bank: torch.Tensor | None = None, arm_indices: torch.Tensor | None = None,
                  scales: torch.Tensor | None = None, attention_mask: torch.Tensor | None = None,
                  capture_positions: torch.Tensor | None = None, active: bool = False) -> None:
        self.bank = bank
        self.arm_indices = arm_indices
        self.scales = scales
        self.attention_mask = attention_mask
        self.capture_positions = capture_positions
        self.captured = {}
        self.active = active

    def disable(self) -> None:
        self.set_batch(active=False)

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_model_direct(model_key: str, device: str):
    """Stream pinned bf16 weights directly onto the requested device, like Part 39."""
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model {model_key!r}; choose {', '.join(MODEL_SPECS)}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    spec = MODEL_SPECS[model_key]
    cache_candidates = [Path(os.environ["PART50_HF_CACHE"])] if os.environ.get("PART50_HF_CACHE") else []
    cache_candidates.extend((HF_CACHE, Path(r"E:\neuron-probe\hf\hub")))
    cache_dir = next((path for path in cache_candidates if path.is_dir()), None)
    if cache_dir is None:
        raise FileNotFoundError(f"expected cached Hugging Face hub at {HF_CACHE} or E:\\neuron-probe\\hf\\hub")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        spec["model_id"], revision=spec["revision"], cache_dir=str(cache_dir), local_files_only=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(
        revision=spec["revision"], cache_dir=str(cache_dir), local_files_only=True,
        dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map={"": device},
    )
    if device.startswith("xpu"):
        try:
            import transformers.modeling_utils as modeling_utils
            if hasattr(modeling_utils, "caching_allocator_warmup"):
                modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
        except ImportError:
            pass
    model = AutoModelForCausalLM.from_pretrained(spec["model_id"], **kwargs)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.set_grad_enabled(False)
    return tok, model


def prompt_text(messages: list[dict], prefill: str, use_chat_template: bool) -> str:
    if use_chat_template:
        raise TypeError("prompt_text with a chat template requires a tokenizer")
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages) + "\nAssistant:" + prefill


def encode_prompt(tok, messages: list[dict], prefill: str, use_chat_template: bool) -> list[int]:
    if use_chat_template:
        try:
            rendered = tok.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
            )
        except TypeError:
            rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        rendered += prefill
    else:
        rendered = prompt_text(messages, prefill, False)
    return list(tok.encode(rendered, add_special_tokens=False))


def _pad_left(sequences: list[list[int]], pad_id: int, device: torch.device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    offsets = []
    for row, seq in enumerate(sequences):
        offset = width - len(seq)
        offsets.append(offset)
        ids[row, offset:] = torch.as_tensor(seq, dtype=torch.long, device=device)
        mask[row, offset:] = 1
    return ids, mask, offsets


def _model_logits(model, tok, sequences: list[list[int]], hooks: HeadOutputSteering,
                  arm_bank: torch.Tensor | None = None, arm_ids: list[int] | None = None,
                  scales: list[float] | None = None,
                  capture_local_positions: list[list[int]] | None = None,
                  capture_only: bool = False):
    device = next(model.parameters()).device
    inputs, mask, offsets = _pad_left(sequences, int(tok.pad_token_id), device)
    capture = None
    if capture_local_positions is not None:
        width = len(capture_local_positions[0])
        capture = torch.full((len(sequences), width), -1, dtype=torch.long, device=device)
        for row, (local, offset) in enumerate(zip(capture_local_positions, offsets)):
            valid = [p for p in local if 0 <= p < len(sequences[row])]
            if valid:
                capture[row, :len(valid)] = torch.as_tensor([p + offset for p in valid], device=device)
    direction_bank = None if arm_bank is None else arm_bank.to(device=device, dtype=torch.float32)
    row_ids = None if arm_ids is None else torch.as_tensor(arm_ids, dtype=torch.long, device=device)
    row_scales = None if scales is None else torch.as_tensor(scales, dtype=torch.float32, device=device)
    hooks.set_batch(direction_bank, row_ids, row_scales, mask, capture, active=arm_bank is not None)
    try:
        with torch.inference_mode():
            if capture_only:
                model.model(input_ids=inputs, attention_mask=mask, use_cache=False)
                logits = None
            else:
                result = model(input_ids=inputs, attention_mask=mask, use_cache=False, logits_to_keep=1)
                logits = result.logits[:, -1, :].float().detach()
        return logits, {k: v for k, v in hooks.captured.items()}
    finally:
        hooks.disable()


def _logodds(logits: torch.Tensor, target_ids: Iterable[int], other_ids: Iterable[int]) -> float:
    target = torch.as_tensor(list(target_ids), dtype=torch.long, device=logits.device)
    other = torch.as_tensor(list(other_ids), dtype=torch.long, device=logits.device)
    if target.numel() == 0 or other.numel() == 0:
        raise ValueError("answer candidate token set is empty")
    lp = logits.log_softmax(-1)
    return float((torch.logsumexp(lp[target], 0) - torch.logsumexp(lp[other], 0)).cpu() / LN2)


def _batch_rows(model, tok, hooks: HeadOutputSteering, prompts: list[list[int]],
                choices: list[tuple[list[int], list[int]]], bank: torch.Tensor,
                scales: list[float], batch_size: int) -> np.ndarray:
    """Run all requested arms over all items in flat batches and return item x arm log odds."""
    arms = bank.shape[0]
    out = np.empty((len(prompts), arms), dtype=np.float64)
    bank = bank.to(device=next(model.parameters()).device, dtype=torch.float32)
    flat = [(item, arm) for item in range(len(prompts)) for arm in range(arms)]
    for start in range(0, len(flat), batch_size):
        rows = flat[start:start + batch_size]
        logits, _ = _model_logits(
            model, tok, [prompts[i] for i, _ in rows], hooks, bank,
            [arm for _, arm in rows], [scales[arm] for _, arm in rows],
        )
        for row_no, (item, _arm) in enumerate(rows):
            out[item, _arm] = _logodds(logits[row_no], *choices[item])
    return out


def _roleplay_train(items: list[dict], tok, use_chat: bool, limit: int = 0):
    rows = [it for it in items if it.get("family") == "rp" and it.get("split") == "select"]
    return rows[:limit] if limit else rows


def _get_prompt_and_answer(tok, item: dict, key: str, use_chat: bool):
    prompt = encode_prompt(tok, item["dec"], item.get("prefill", ""), use_chat)
    answer = list(tok.encode(item[key], add_special_tokens=False))
    if len(prompt) < 1 or len(answer) < 1:
        raise ValueError(f"item {item.get('id')} has an empty prompt or completion")
    return prompt, answer


def roleplay_answer_position(prompt: list[int], truth: list[int], lie: list[int], eos_id: int):
    """Score the first answer position where the two teacher-forced completions diverge."""
    shared = 0
    while shared < min(len(truth), len(lie)) and truth[shared] == lie[shared]:
        shared += 1
    truth_id = truth[shared] if shared < len(truth) else int(eos_id)
    lie_id = lie[shared] if shared < len(lie) else int(eos_id)
    if truth_id == lie_id:
        raise ValueError("truth and lie completions have no distinguishing token")
    return prompt + truth[:shared], [truth_id], [lie_id], shared


def persona_answer_ids(tok) -> tuple[list[int], list[int]]:
    """Retokenize Yes/No for the current Qwen snapshot; source IDs belong to Qwen3.5-9B."""
    def collect(word: str) -> list[int]:
        ids = set()
        for variant in (word, " " + word, word.lower(), " " + word.lower(), word.upper(), " " + word.upper()):
            encoded = list(tok.encode(variant, add_special_tokens=False))
            if len(encoded) == 1 and tok.decode(encoded).strip().casefold() == word.casefold():
                ids.add(int(encoded[0]))
        if not ids:
            raise ValueError(f"tokenizer has no single-token variants for persona answer {word!r}")
        return sorted(ids)
    return collect("Yes"), collect("No")


def answer_polarity(text: str) -> str:
    match = re.search(r"[A-Za-z]+", text)
    word = match.group(0).casefold() if match else ""
    if word in {"yes", "yeah", "yep", "absolutely", "sure", "definitely", "certainly", "indeed"}:
        return "yes"
    if word in {"no", "not", "nope", "never"}:
        return "no"
    return "other"


def collect_direction(model, tok, hooks: HeadOutputSteering, train: list[dict],
                      use_chat: bool, batch_size: int = 8) -> torch.Tensor:
    """Teacher-force both supplied completions and average the first nine answer-prediction states."""
    examples = []
    for item_index, item in enumerate(train):
        for label, key in (("truth", "truth_completion"), ("lie", "lie_completion")):
            prompt, completion = _get_prompt_and_answer(tok, item, key, use_chat)
            sequence = prompt + completion[:-1]
            # A position predicts the next completion token. Include the prompt's final position and then
            # the preceding teacher-forced answer tokens, matching a nine-position decision span.
            selected = min(SPAN, len(completion))
            start = len(prompt) - 1
            positions = list(range(start, start + selected))
            examples.append((item_index, label, sequence, positions))
    sums = None
    counts = 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start:start + batch_size]
        _, captured = _model_logits(
            model, tok, [r[2] for r in chunk], hooks,
            capture_local_positions=[r[3] for r in chunk],
            capture_only=True,
        )
        by_row = [torch.stack([captured[layer][j] for layer in range(len(hooks.layers))])
                  for j in range(len(chunk))]
        # Each train item contributes one paired truth-minus-lie difference.
        for local, record in enumerate(chunk):
            if record[1] != "truth":
                continue
            lie_index = next(i for i, candidate in enumerate(examples)
                             if candidate[0] == record[0] and candidate[1] == "lie")
            if lie_index < start or lie_index >= start + len(chunk):
                # Pair rows are consecutive, but a small batch can split them. The cache below handles that.
                continue
            lie_local = lie_index - start
            diff = by_row[local] - by_row[lie_local]
            sums = diff if sums is None else sums + diff
            counts += 1
    if counts != len(train):
        # Re-run with paired rows kept in the same batch. This fallback is only reached for odd batch sizes.
        sums = None
        counts = 0
        pair_batch = max(2, batch_size - (batch_size % 2))
        for start in range(0, len(examples), pair_batch):
            chunk = examples[start:start + pair_batch]
            _, captured = _model_logits(
                model, tok, [r[2] for r in chunk], hooks,
                capture_local_positions=[r[3] for r in chunk],
            )
            for local in range(0, len(chunk), 2):
                truth = torch.stack([captured[layer][local] for layer in range(len(hooks.layers))])
                lie = torch.stack([captured[layer][local + 1] for layer in range(len(hooks.layers))])
                diff = truth - lie
                sums = diff if sums is None else sums + diff
                counts += 1
    if sums is None or counts != len(train):
        raise RuntimeError(f"direction paired {counts} of {len(train)} train items")
    return (sums / counts).float()


def model_output_dir(root: str | Path, model_key: str) -> Path:
    return Path(root) / model_key


def _load_direction(path: Path, model_key: str, bundle_sha: str):
    record = torch.load(path / "direction.pt", map_location="cpu", weights_only=False)
    if record.get("model_key") != model_key or record.get("bundle_sha256") != bundle_sha:
        raise ValueError("direction artifact model or frozen bundle provenance does not match")
    return record


def command_direction(args) -> None:
    bundle, bundle_sha = ensure_bundle(args.bundle)
    del bundle
    out = model_output_dir(args.out, args.model)
    data = json.loads(ITEMS_ROLEPLAY.read_text(encoding="utf-8"))
    tok, model = load_model_direct(args.model, args.device)
    hooks = HeadOutputSteering(model)
    try:
        train = _roleplay_train(data, tok, MODEL_SPECS[args.model]["chat"], args.limit)
        if not train:
            raise ValueError("no roleplay select-split rows available for direction")
        start = time.perf_counter()
        direction = collect_direction(model, tok, hooks, train, MODEL_SPECS[args.model]["chat"], args.batch_size)
        record = {
            "model_key": args.model, "model_id": MODEL_SPECS[args.model]["model_id"],
            "revision": MODEL_SPECS[args.model]["revision"], "device": args.device,
            "direction": direction.cpu(), "span_tokens": SPAN,
            "span_definition": "first nine completion-token prediction positions (prompt final token plus teacher-forced completion context)",
            "train_items": len(train), "bundle_sha256": bundle_sha,
            "seconds": round(time.perf_counter() - start, 3),
        }
        out.mkdir(parents=True, exist_ok=True)
        torch.save(record, out / "direction.pt")
        print(f"direction {args.model}: {len(train)} train pairs, {direction.shape}, {record['seconds']:.1f}s -> {out / 'direction.pt'}")
    finally:
        hooks.close()


def score_train_shift(model, tok, hooks: HeadOutputSteering, train: list[dict], direction: torch.Tensor,
                      scale: float, use_chat: bool, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    prompts, choices = [], []
    for item in train:
        prompt, truth = _get_prompt_and_answer(tok, item, "truth_completion", use_chat)
        lie = list(tok.encode(item["lie_completion"], add_special_tokens=False))
        if not lie:
            raise ValueError(f"item {item['id']} has an empty lie completion")
        scored_prompt, truth_ids, lie_ids, _ = roleplay_answer_position(prompt, truth, lie, tok.eos_token_id)
        prompts.append(scored_prompt)
        choices.append((truth_ids, lie_ids))
    zero = torch.zeros_like(direction).unsqueeze(0)
    bank = torch.cat((zero, direction.unsqueeze(0)), 0)
    values = _batch_rows(model, tok, hooks, prompts, choices, bank, [0.0, scale], batch_size)
    return values[:, 0], values[:, 1]


def calibrate_scale(model, tok, hooks: HeadOutputSteering, train: list[dict], direction: torch.Tensor,
                    target_bits: float, use_chat: bool, batch_size: int) -> dict:
    baseline, at_one = score_train_shift(model, tok, hooks, train, direction, 1.0, use_chat, batch_size)
    shift_one = float((at_one - baseline).mean())
    if shift_one >= target_bits:
        low, high = 0.0, 1.0
    else:
        low, high = 1.0, 2.0
        previous = shift_one
        while high <= 128:
            _base, value = score_train_shift(model, tok, hooks, train, direction, high, use_chat, batch_size)
            current = float((value - baseline).mean())
            if current >= target_bits:
                break
            if current < previous - 1e-4 and high >= 16:
                raise RuntimeError(f"calibration response is nonmonotonic and target was not reached (scale {high}, shift {current:.4f})")
            previous, low, high = current, high, high * 2
        else:
            raise RuntimeError(f"could not reach {target_bits} bits by scale 128; last mean shift {previous:.4f}")
    iterations = 0
    best_scale, best_shift = high, float((score_train_shift(
        model, tok, hooks, train, direction, high, use_chat, batch_size
    )[1] - baseline).mean())
    for iterations in range(1, 17):
        mid = (low + high) / 2
        _base, value = score_train_shift(model, tok, hooks, train, direction, mid, use_chat, batch_size)
        current = float((value - baseline).mean())
        if abs(current - target_bits) < abs(best_shift - target_bits):
            best_scale, best_shift = mid, current
        if current >= target_bits:
            high = mid
        else:
            low = mid
    return {
        "target_bits": float(target_bits), "scale": float(best_scale),
        "achieved_train_mean_shift_bits": float(best_shift), "train_items": len(train),
        "search_iterations": iterations, "train_split": "select",
        "answer_metric": "log2 odds at the first distinguishing truth/lie completion token",
    }


def command_calibrate(args) -> None:
    _bundle, bundle_sha = ensure_bundle(args.bundle)
    out = model_output_dir(args.out, args.model)
    direction_record = _load_direction(out, args.model, bundle_sha)
    data = json.loads(ITEMS_ROLEPLAY.read_text(encoding="utf-8"))
    train = _roleplay_train(data, None, MODEL_SPECS[args.model]["chat"], args.limit)
    if not train:
        raise ValueError("no roleplay select-split rows available for calibration")
    tok, model = load_model_direct(args.model, args.device)
    hooks = HeadOutputSteering(model)
    try:
        if args.limit:
            train = train[:args.limit]
        t0 = time.perf_counter()
        result = calibrate_scale(model, tok, hooks, train, direction_record["direction"],
                                 args.target_bits, MODEL_SPECS[args.model]["chat"], args.batch_size)
        result.update({"model_key": args.model, "revision": MODEL_SPECS[args.model]["revision"],
                       "bundle_sha256": bundle_sha, "device": args.device,
                       "seconds": round(time.perf_counter() - t0, 3)})
        atomic_json(result, out / "calibration.json")
        print(f"calibrate {args.model}: s={result['scale']:.8g}, shift={result['achieved_train_mean_shift_bits']:.4f} bits, {result['seconds']:.1f}s")
    finally:
        hooks.close()


def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["items"] if isinstance(data, dict) else data


def make_persona_choice(item: dict, yes_ids: list[int], no_ids: list[int]):
    # The source token IDs were built for Qwen3.5-9B; match indicates which text answer endorses the trait.
    return (yes_ids, no_ids) if item["match"] == "Yes" else (no_ids, yes_ids)


def command_run(args) -> None:
    _bundle, bundle_sha = ensure_bundle(args.bundle)
    out = model_output_dir(args.out, args.model)
    direction_record = _load_direction(out, args.model, bundle_sha)
    calibration = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("bundle_sha256") != bundle_sha or calibration.get("revision") != MODEL_SPECS[args.model]["revision"]:
        raise ValueError("calibration provenance does not match the pinned model or frozen bundle")
    direction = direction_record["direction"].float()
    randoms = make_random_directions(direction, args.random_directions, args.seed)
    zero = torch.zeros_like(direction).unsqueeze(0)
    bank = torch.cat((zero, direction.unsqueeze(0), direction.unsqueeze(0), randoms), 0)
    scales = [0.0, float(calibration["scale"]), -float(calibration["scale"])] + [float(calibration["scale"])] * len(randoms)
    arm_names = ["unsteered", "honesty", "anti-honesty"] + [f"random{i + 1}" for i in range(len(randoms))]
    roleplay = load_items(ITEMS_ROLEPLAY)
    persona = load_items(ITEMS_PERSONA)
    lies = [it for it in roleplay if it.get("family") == "rp" and it.get("split") == "test" and it.get("label") == "lie"]
    if args.limit:
        lies = lies[:args.limit]
        persona = persona[:args.limit]
    prompts, choices = [], []
    tok, model = load_model_direct(args.model, args.device)
    prompts = [encode_prompt(tok, it["dec"], it.get("prefill", ""), MODEL_SPECS[args.model]["chat"]) for it in lies]
    choices = []
    roleplay_positions = []
    for index, item in enumerate(lies):
        truth = list(tok.encode(item["truth_completion"], add_special_tokens=False))
        lie = list(tok.encode(item["lie_completion"], add_special_tokens=False))
        prompts[index], truth_ids, lie_ids, shared = roleplay_answer_position(
            prompts[index], truth, lie, tok.eos_token_id
        )
        choices.append((truth_ids, lie_ids))
        roleplay_positions.append(shared)
    hooks = HeadOutputSteering(model)
    try:
        start_time = time.perf_counter()
        lie_values = _batch_rows(model, tok, hooks, prompts, choices, bank, scales, args.batch_size)
        lie_rows = [{"id": item["id"], "label": item.get("label"),
                     "lie_polarity": answer_polarity(item["lie_completion"]), "m": lie_values[i].tolist()}
                    for i, item in enumerate(lies)]
        yes_ids, no_ids = persona_answer_ids(tok)
        persona_prompts = [encode_prompt(tok, it["dec"], it.get("prefill", ""), MODEL_SPECS[args.model]["chat"])
                           for it in persona]
        persona_values = _batch_rows(model, tok, hooks, persona_prompts,
                                     [make_persona_choice(it, yes_ids, no_ids) for it in persona], bank, scales, args.batch_size)
        persona_rows = [{"id": item["id"], "trait": item["family"], "match": item["match"],
                         "m": persona_values[i].tolist()} for i, item in enumerate(persona)]
        record = {
            "schema_version": 1, "model_key": args.model, "model_id": MODEL_SPECS[args.model]["model_id"],
            "revision": MODEL_SPECS[args.model]["revision"], "device": args.device,
            "scale": float(calibration["scale"]), "target_bits": float(calibration["target_bits"]),
            "random_seed": int(args.seed), "random_directions": len(randoms), "arm_names": arm_names,
            "bundle_sha256": bundle_sha, "direction_span_tokens": SPAN,
            "test_roleplay_items": lie_rows, "persona_items": persona_rows,
            "n_test_roleplay": len(lie_rows), "n_persona_items": len(persona_rows),
            "complete_persona": len(persona_rows) == len(load_items(ITEMS_PERSONA)),
            "roleplay_shared_prefix_tokens": roleplay_positions,
            "persona_yes_token_ids": yes_ids, "persona_no_token_ids": no_ids,
            "seconds": round(time.perf_counter() - start_time, 3),
            "metric": "log2 odds at the first distinguishing truth/lie completion token; persona matching/opposite Yes/No token sets retokenized for this model",
        }
        atomic_json(record, out / "run.json")
        print(f"run {args.model}: {len(lie_rows)} held-out lies, {len(persona_rows)} persona rows x {len(arm_names)} arms, {record['seconds']:.1f}s")
    finally:
        hooks.close()


def summarize_persona(run: dict, bundle: dict) -> dict:
    names = run["arm_names"]
    rows = run["persona_items"]
    traits = sorted({row["trait"] for row in rows})
    if not rows:
        raise ValueError("persona run contains no items")
    values = np.asarray([row["m"] for row in rows], dtype=float)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("persona run has malformed arm values")
    changes = values[:, 1:] - values[:, :1]
    baseline = values[:, 0]
    # Remove each direction's linear confidence trend over the shared, paired item set.
    center = baseline.mean()
    centered = baseline - center
    denom = float(centered @ centered)
    slopes = np.zeros(changes.shape[1], dtype=float) if denom == 0 else centered @ (changes - changes.mean(0, keepdims=True)) / denom
    detrended = changes - centered[:, None] * slopes[None, :]
    by_trait: dict[str, dict] = {}
    trait_order = []
    for trait in traits:
        ix = np.asarray([i for i, row in enumerate(rows) if row["trait"] == trait], dtype=int)
        yes = np.asarray([i for i in ix if rows[i]["match"] == "Yes"], dtype=int)
        no = np.asarray([i for i in ix if rows[i]["match"] == "No"], dtype=int)
        if len(yes) == 0 or len(no) == 0:
            continue
        raw_yes, raw_no = changes[yes].mean(0), changes[no].mean(0)
        det_yes, det_no = detrended[yes].mean(0), detrended[no].mean(0)
        raw_t = (raw_yes + raw_no) / 2.0
        det_t = (det_yes + det_no) / 2.0
        raw_b = (raw_yes - raw_no) / 2.0
        by_trait[trait] = {
            "trait": trait, "yes_items": int(len(yes)), "no_items": int(len(no)),
            "delta_yes_bits": raw_yes.tolist(), "delta_no_bits": raw_no.tolist(),
            "trait_matching_change_bits": raw_t.tolist(), "yes_bias_component_bits": raw_b.tolist(),
            "confidence_removed_delta_yes_bits": det_yes.tolist(),
            "confidence_removed_delta_no_bits": det_no.tolist(),
            "confidence_removed_change_bits": det_t.tolist(),
        }
        trait_order.append(trait)
    if not trait_order:
        raise ValueError("no persona traits include both Yes and No answer groups")
    arm_names = names[1:]
    honesty_index = arm_names.index("honesty")
    random_indices = [i for i, name in enumerate(arm_names) if name.startswith("random")]
    analysis_indices = [honesty_index, *random_indices]
    raw_matrix = np.asarray([[by_trait[t]["trait_matching_change_bits"][i] for i in analysis_indices]
                             for t in trait_order])
    det_matrix = np.asarray([[by_trait[t]["confidence_removed_change_bits"][i] for i in analysis_indices]
                             for t in trait_order])
    if raw_matrix.shape[1] < 2:
        raise ValueError("persona run needs the honesty direction plus at least one random direction")
    q_raw = perm_fdr(raw_matrix)
    q_detrended = perm_fdr(det_matrix)
    for i, trait in enumerate(trait_order):
        by_trait[trait]["q_perm_raw"] = float(q_raw[i])
        by_trait[trait]["q_perm_confidence_removed"] = float(q_detrended[i])
        by_trait[trait]["fdr_5_raw"] = bool(q_raw[i] <= 0.05 + 1e-9)
        by_trait[trait]["fdr_5_confidence_removed"] = bool(q_detrended[i] <= 0.05 + 1e-9)
    bundle_records = {row["trait"]: row for row in bundle["traits"]}
    if not set(bundle_records).issubset(by_trait):
        raise ValueError("persona run is missing one or more frozen bundle traits")
    signed = np.asarray([bundle_records[t]["sign"] for t in bundle_records], dtype=float)
    bundle_trait_order = list(bundle_records)
    main_scores = np.asarray([by_trait[t]["trait_matching_change_bits"][honesty_index] for t in bundle_trait_order])
    bundle_score = float(np.mean(signed * main_scores))
    random_scores = []
    for arm_index in random_indices:
        vals = np.asarray([by_trait[t]["trait_matching_change_bits"][arm_index] for t in bundle_trait_order])
        random_scores.append(float(np.mean(signed * vals)))
    rank = 1 + sum(value > bundle_score for value in random_scores)
    p_rank = (1 + sum(value >= bundle_score for value in random_scores)) / (len(random_scores) + 1)
    yes_indices = [i for i, row in enumerate(rows) if row["match"] == "Yes"]
    no_indices = [i for i, row in enumerate(rows) if row["match"] == "No"]
    mean_yes = float(changes[yes_indices, 0].mean()) if yes_indices else float("nan")
    mean_no = float(changes[no_indices, 0].mean()) if no_indices else float("nan")
    return {
        "arms": names[1:], "n_items": len(rows), "n_traits_tested": len(trait_order),
        "complete": bool(run.get("complete_persona")), "confidence_slopes_bits_per_baseline_bit": slopes.tolist(),
        "yes_answer_mean_change_bits": mean_yes, "no_answer_mean_change_bits": mean_no,
        "trait_matching_effects": by_trait,
        "fdr": {"method": "Sauers persona_stats.py permutation FDR (Storey-Tibshirani estimate)",
                "q": 0.05, "raw_n": int((q_raw <= 0.05 + 1e-9).sum()),
                "confidence_removed_n": int((q_detrended <= 0.05 + 1e-9).sum())},
        "confidence_trend_removal": {
            "method": "per-arm OLS slope of item log-odds change on centered unsteered matching-answer log2 odds; subtract fitted slope term",
            "honesty_slope": float(slopes[0]),
        },
        "bundle_score": {
            "definition": "mean source-signed trait-matching log2-odds change over frozen 51 traits",
            "value_bits": bundle_score, "random_scores_bits": random_scores,
            "rank_descending_among_39_random_plus_honesty": rank,
            "permutation_p_one_sided": float(p_rank),
            "random_95th_percentile_bits": float(np.quantile(random_scores, 0.95)) if random_scores else None,
        },
    }


def summarize_lies(run: dict) -> dict:
    rows = run["test_roleplay_items"]
    if not rows:
        return {"n_items": 0, "n_baseline_lie_preferred": 0, "honesty_flips": 0, "honesty_flip_rate": None,
                "conditional_flip_rate": None, "random_flip_counts": [], "polarity_split": {}}
    values = np.asarray([row["m"] for row in rows], dtype=float)
    arm_names = run.get("arm_names", [])
    honesty_index = arm_names.index("honesty") if "honesty" in arm_names else 1
    random_indices = [i for i, name in enumerate(arm_names) if name.startswith("random")]
    if not random_indices:
        random_indices = list(range(2, values.shape[1]))
    baseline_lie = values[:, 0] < 0
    steered_truth = values[:, honesty_index] > 0
    denominator = int(baseline_lie.sum())
    flips = int((baseline_lie & steered_truth).sum())
    rand_counts = [int((baseline_lie & (values[:, i] > 0)).sum()) for i in random_indices]
    polarity_split = {}
    polarities = [row.get("lie_polarity", "other") for row in rows]
    for polarity in ("yes", "no", "other"):
        ix = np.asarray([i for i, value in enumerate(polarities) if value == polarity], dtype=int)
        if not len(ix):
            continue
        shifts = values[ix, [honesty_index]] - values[ix, :1]
        random_shifts = values[ix][:, random_indices] - values[ix, :1]
        group_flips = baseline_lie[ix] & steered_truth[ix]
        polarity_split[polarity] = {
            "n": int(len(ix)), "honesty_flips": int(group_flips.sum()),
            "honesty_flip_rate": float(group_flips.mean()),
            "honesty_mean_logodds_shift_bits": float(shifts[:, 0].mean()),
            "random_mean_logodds_shift_bits": random_shifts.mean(0).tolist(),
        }
    if "yes" in polarity_split and "no" in polarity_split:
        by_yes, by_no = polarity_split["yes"]["honesty_mean_logodds_shift_bits"], polarity_split["no"]["honesty_mean_logodds_shift_bits"]
        polarity_split["honesty_component_bits"] = (by_no + by_yes) / 2
        polarity_split["yes_push_component_bits"] = (by_no - by_yes) / 2
    item_count = len(rows)
    return {
        "n_items": item_count, "n_baseline_lie_preferred": denominator,
        "honesty_flips": flips, "honesty_flip_rate": flips / item_count if item_count else None,
        "conditional_flip_rate": flips / denominator if denominator else None,
        "honesty_mean_logodds_shift_bits": float((values[:, honesty_index] - values[:, 0]).mean()),
        "random_flip_counts": rand_counts,
        "random_flip_rates": [count / item_count for count in rand_counts],
        "random_max_flip_rate": max(rand_counts) / item_count if item_count and rand_counts else None,
        "polarity_split": polarity_split,
        "item_values": rows,
        "flip_rule": "baseline truth-vs-lie log2 odds < 0 and steered log2 odds > 0; overall denominator is all labelled held-out lies",
    }


def build_scores(model_key: str, out_root: str | Path, bundle_path: str | Path) -> dict:
    bundle, bundle_sha = ensure_bundle(bundle_path)
    out = model_output_dir(out_root, model_key)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    if run.get("bundle_sha256") != bundle_sha or run.get("revision") != MODEL_SPECS[model_key]["revision"]:
        raise ValueError("run provenance does not match the frozen bundle or pinned model revision")
    persona = summarize_persona(run, bundle)
    lies = summarize_lies(run)
    return {
        "schema_version": 1, "model_key": model_key, "model_id": MODEL_SPECS[model_key]["model_id"],
        "revision": MODEL_SPECS[model_key]["revision"], "device": run.get("device"),
        "bundle_sha256": bundle_sha, "scale": run["scale"], "calibration_target_bits": run["target_bits"],
        "n_random_directions": run["random_directions"], "random_seed": run["random_seed"],
        "lies": lies, "persona": persona,
        "predictions": {
            "50.1": {"honesty_flip_rate": lies["honesty_flip_rate"],
                     "random_max_flip_rate": lies["random_max_flip_rate"]},
            "50.2": {"bundle_score_bits": persona["bundle_score"]["value_bits"],
                     "random_95th_percentile_bits": persona["bundle_score"]["random_95th_percentile_bits"],
                     "permutation_p": persona["bundle_score"]["permutation_p_one_sided"]},
            "50.3": {"matched_shift_base_bundle_score_bits": persona["bundle_score"]["value_bits"]
                     if model_key.endswith("-base") else None},
            "50.4": {"cross_model_persona_change_correlation": None},
            "50.5": {"chat_confidence_removed_fdr_5_traits": persona["fdr"]["confidence_removed_n"]
                     if model_key == "qwen3-1.7b" else None},
        },
    }


def _attach_cross_model(scores_by_key: dict[str, dict]) -> None:
    base_key, chat_key = "qwen3-1.7b-base", "qwen3-1.7b"
    if base_key not in scores_by_key or chat_key not in scores_by_key:
        return
    base = scores_by_key[base_key]["persona"]["trait_matching_effects"]
    chat = scores_by_key[chat_key]["persona"]["trait_matching_effects"]
    shared = sorted(set(base) & set(chat))
    x = np.asarray([base[t]["trait_matching_change_bits"][0] for t in shared], dtype=float)
    y = np.asarray([chat[t]["trait_matching_change_bits"][0] for t in shared], dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1]) if len(shared) > 1 else float("nan")
    for key in (base_key, chat_key):
        scores_by_key[key]["predictions"]["50.4"] = {
            "cross_model_persona_change_correlation": corr, "n_shared_traits": len(shared),
            "trait_set": "all traits with both Yes and No items in both models",
        }
    scores_by_key[base_key]["predictions"]["50.3"] = {
        "base_bundle_score_bits": scores_by_key[base_key]["persona"]["bundle_score"]["value_bits"],
        "chat_bundle_score_bits": scores_by_key[chat_key]["persona"]["bundle_score"]["value_bits"],
        "base_to_chat_ratio": (scores_by_key[base_key]["persona"]["bundle_score"]["value_bits"] /
                               scores_by_key[chat_key]["persona"]["bundle_score"]["value_bits"])
        if scores_by_key[chat_key]["persona"]["bundle_score"]["value_bits"] != 0 else None,
    }
    scores_by_key[chat_key]["predictions"]["50.3"] = scores_by_key[base_key]["predictions"]["50.3"]


def command_score(args) -> None:
    keys = list(MODEL_SPECS) if args.model == "both" else [args.model]
    scores = {key: build_scores(key, args.out, args.bundle) for key in keys}
    if args.model == "both":
        _attach_cross_model(scores)
    for key, score in scores.items():
        dest = model_output_dir(args.out, key) / "scores.json"
        atomic_json(score, dest)
        print(f"score {key}: bundle={score['persona']['bundle_score']['value_bits']:+.4f} bits, FDR5(confidence-removed)={score['persona']['fdr']['confidence_removed_n']} -> {dest}")


def command_smoke(args) -> None:
    """A bounded CPU pass through direction, calibration, and run using a temporary output root."""
    start = time.perf_counter()
    command_direction(args)
    command_calibrate(args)
    command_run(args)
    print(f"smoke total {time.perf_counter() - start:.1f}s, limit={args.limit}, device={args.device}")


def _common(p):
    p.add_argument("--model", choices=list(MODEL_SPECS), default="qwen3-1.7b")
    p.add_argument("--device", default="xpu")
    p.add_argument("--out", default=str(ROOT / "results/part50"))
    p.add_argument("--bundle", default=str(BUNDLE_PATH))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="cap rows for a smoke run")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("bundle", help="freeze Sauers' 51 signed Qwen3.5-9B persona changes")
    b.add_argument("--source", default=str(SOURCE_STATS))
    b.add_argument("--output", default=str(BUNDLE_PATH))
    d = sub.add_parser("direction", help="teacher-force train role-play completions and save the head direction")
    _common(d)
    c = sub.add_parser("calibrate", help="calibrate the train-split mean log-odds shift")
    _common(c)
    c.add_argument("--target-bits", type=float, default=2.0)
    r = sub.add_parser("run", help="score held-out lies and persona items against random directions")
    _common(r)
    r.add_argument("--random-directions", type=int, default=DEFAULT_RANDOMS)
    r.add_argument("--seed", type=int, default=DEFAULT_SEED)
    s = sub.add_parser("score", help="write predictions and permutation-FDR summaries")
    s.add_argument("--model", choices=[*MODEL_SPECS, "both"], default="both")
    s.add_argument("--out", default=str(ROOT / "results/part50"))
    s.add_argument("--bundle", default=str(BUNDLE_PATH))
    sm = sub.add_parser("smoke", help="bounded CPU execution of direction, calibration, and run")
    _common(sm)
    sm.add_argument("--target-bits", type=float, default=0.25)
    sm.add_argument("--random-directions", type=int, default=39)
    sm.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.command == "bundle":
        record = write_bundle(args.source, args.output)
        print(f"bundle: {record['n_traits']} signed traits from {record['source_file']} sha256={record['source_sha256']} -> {args.output}")
    elif args.command == "direction":
        command_direction(args)
    elif args.command == "calibrate":
        command_calibrate(args)
    elif args.command == "run":
        command_run(args)
    elif args.command == "score":
        command_score(args)
    elif args.command == "smoke":
        command_smoke(args)


if __name__ == "__main__":
    main()
