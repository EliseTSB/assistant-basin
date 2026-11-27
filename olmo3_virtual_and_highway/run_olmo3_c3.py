#!/usr/bin/env python3
"""run_olmo3_c3.py — Stage C3 (WSP v1) Zero-Hook Literal pre_norm Weight-Space Programming GPU Runner.

Implements the frozen registered Stage C3 protocol (`OLMO3-WSP-C3-DESIGN.md`, `10953eac` / `146e5ad5`):
  - Fail-closed C3.0.4 SHA-256 provenance verification of all 4 committed artifacts before model load.
  - Zero runtime PyTorch hooks during every test/eval pass (hooks used ONLY in one-time `calib_16`
    capture and verified removed via `count_active_hooks(model) == 0` before any evaluation).
  - `fp32`-master edit application (`(W_clean_fp32 + dW_fp32).to(bfloat16)`) into
    `model.layers[L].mlp.down_proj.weight` (`L in {21, 24, 27}`), bitwise `torch.equal` restore
    after every arm, and logged `stored_frob_ret` + `stored_rank1_proj_ret`.
  - Policy 11-07 honored strictly: ZERO edited model weights or checkpoints ever saved to disk.
  - Exact single-token (95 IDs) + multi-token (380 `bad_words_ids` sequences) decode ban processor
    safe under both `B = 1` and `B = 8` left-padded greedy decode.
  - C3.0.1 `B = 8` vs `B = 1` logit-parity preflight (`max|dlogit| <= 0.05`) with automatic `B = 1` fallback.
  - Optional paced gzip+base64 stdout receipt export (`<= 160 KB/s`) for Colab Jupyter `iopub` safety.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import c3_scoring  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen Registered Stage C3 Provenance Pins (Draft v4 / public main 10953eac)
# ---------------------------------------------------------------------------
MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
TARGET_LAYERS: list[int] = [21, 24, 27]
DOSE_LADDER: list[float] = [0.075, 0.15, 0.30, 0.45, 0.60, 0.80]
SHAM_ALPHA: float = 0.05

EXPECTED_C3_DESIGN_SHA256 = "dbdab600e72fdba21f22024d0b1a90575cd2d6d82195fdb6b5fe4a49f494023a"
EXPECTED_C3_CONSTANTS_SHA256 = "bb568d8a29c3d3515563844caaffa9ede9d4c2bc926ed78c68484ada4a1d4fec"
EXPECTED_C3_BANK_SHA256 = "e264cea9785f91f07a34c7e3b37ed8451474c888a340a0ec13ed63e8e6f820b2"
EXPECTED_C3_SCORING_SHA256 = "b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f"

RECEIPT_SCHEMA_VERSION = "olmo3-stage-c3-receipt-v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_preflight_provenance(
    design_path: Path,
    constants_path: Path,
    bank_path: Path,
    scoring_path: Path,
) -> dict[str, str]:
    """Fail-closed C3.0.4 SHA-256 verification of all 4 committed Stage C3 files."""
    checks = [
        ("c3_design_sha256", design_path, EXPECTED_C3_DESIGN_SHA256),
        ("c3_constants_sha256", constants_path, EXPECTED_C3_CONSTANTS_SHA256),
        ("c3_bank_sha256", bank_path, EXPECTED_C3_BANK_SHA256),
        ("c3_scoring_sha256", scoring_path, EXPECTED_C3_SCORING_SHA256),
    ]
    hashes: dict[str, str] = {}
    for key, p, expected in checks:
        if not p.exists():
            raise FileNotFoundError(f"C3.0.4 preflight missing required file: {p}")
        actual = sha256_file(p)
        if actual != expected:
            raise ValueError(
                f"C3.0.4 preflight SHA-256 mismatch for {p.name}: got {actual}, expected {expected}"
            )
        hashes[key] = actual
    return hashes


def count_active_hooks(model: torch.nn.Module) -> int:
    """Count all active forward/backward hooks registered on every submodule of `model`."""
    total = 0
    for mod in model.modules():
        for attr in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
            "_full_backward_hooks",
            "_full_backward_pre_hooks",
        ):
            d = getattr(mod, attr, None)
            if d:
                total += len(d)
    return total


# ---------------------------------------------------------------------------
# Zero-Hook Literal fp32-Master Weight Editor with Bitwise bf16 Restore
# ---------------------------------------------------------------------------
class WeightMaster:
    """Holds clean fp32 masters and bf16 snapshots of target `mlp.down_proj.weight` tensors."""

    def __init__(self, model: torch.nn.Module, layers: list[int]):
        self.layers = list(layers)
        self.clean_bf16: dict[int, torch.Tensor] = {}
        self.clean_fp32: dict[int, torch.Tensor] = {}
        self.restore_exact_all: bool = True
        self.restore_checks: int = 0
        self.max_hooks_seen: int = count_active_hooks(model)
        for L in self.layers:
            w = model.model.layers[L].mlp.down_proj.weight.data.detach().cpu()
            w_bf16 = w.to(torch.bfloat16).clone()
            self.clean_bf16[L] = w_bf16
            self.clean_fp32[L] = w_bf16.float().clone()

    def check_hooks(self, model: torch.nn.Module) -> int:
        n = count_active_hooks(model)
        if n > self.max_hooks_seen:
            self.max_hooks_seen = n
        if n != 0:
            raise RuntimeError(f"C3.0.1 VIOLATION: expected 0 active PyTorch hooks, found {n}")
        return n

    @contextlib.contextmanager
    def apply_edit(
        self,
        model: torch.nn.Module,
        edits_by_layer: dict[int, tuple[float, torch.Tensor, torch.Tensor]],
    ):
        """Apply literal rank-1 weight edit `alpha * outer(v_pre_hat, k_hat)` from fp32 master -> bf16.

        Verifies `active_hooks == 0` on entry and exit, computes exact `stored_frob_ret` and
        `stored_rank1_proj_ret`, and restores bitwise `torch.equal` clean bf16 weights in `finally`.
        """
        self.check_hooks(model)
        metrics: dict[str, dict[str, float]] = {}
        touched_layers = sorted(edits_by_layer.keys())
        try:
            for L in touched_layers:
                alpha, v_pre_hat, k_hat = edits_by_layer[L]
                alpha_f = float(alpha)
                v_f = v_pre_hat.detach().cpu().float()
                k_f = k_hat.detach().cpu().float()
                dW_intended = alpha_f * torch.outer(v_f, k_f)
                W_edited_bf16 = (self.clean_fp32[L] + dW_intended).to(torch.bfloat16)
                dW_stored = W_edited_bf16.float() - self.clean_bf16[L].float()

                intended_frob = float(dW_intended.norm().item())
                stored_frob = float(dW_stored.norm().item())
                stored_frob_ret = (stored_frob / intended_frob) if intended_frob > 1e-12 else 0.0
                rank1_proj = float(torch.dot(v_f, dW_stored @ k_f).item())
                stored_rank1_proj_ret = (rank1_proj / alpha_f) if abs(alpha_f) > 1e-12 else 0.0
                frac_changed = float((W_edited_bf16 != self.clean_bf16[L]).float().mean().item())

                param = model.model.layers[L].mlp.down_proj.weight
                param.data.copy_(W_edited_bf16.to(device=param.device, dtype=param.dtype))
                metrics[str(L)] = {
                    "alpha": round(alpha_f, 8),
                    "intended_frob": round(intended_frob, 8),
                    "stored_frob": round(stored_frob, 8),
                    "stored_frob_ret": round(stored_frob_ret, 6),
                    "stored_rank1_proj_ret": round(stored_rank1_proj_ret, 6),
                    "frac_changed": round(frac_changed, 6),
                }
            self.check_hooks(model)
            yield metrics
        finally:
            restore_failures: list[int] = []
            for L in touched_layers:
                param = model.model.layers[L].mlp.down_proj.weight
                param.data.copy_(self.clean_bf16[L].to(device=param.device, dtype=param.dtype))
                restored_cpu = param.data.detach().cpu().to(torch.bfloat16)
                ok = bool(torch.equal(restored_cpu, self.clean_bf16[L]))
                self.restore_checks += 1
                if not ok:
                    self.restore_exact_all = False
                    restore_failures.append(L)
            in_exc = sys.exc_info()[0] is not None
            if restore_failures:
                raise RuntimeError(f"C3.0.1 VIOLATION: bitwise weight restore failed at layers {restore_failures}")
            if not in_exc:
                self.check_hooks(model)


# ---------------------------------------------------------------------------
# Exact Single-Token + Multi-Token Decode Ban Processor (B = 1 and B = 8 Safe)
# ---------------------------------------------------------------------------
class CommonBanLogitsProcessor:
    """Enforces frozen single-token bans AND multi-token `bad_words_ids` sequences identically at B=1 and B=8."""

    def __init__(
        self,
        single_token_ids: list[int],
        multi_token_sequences: list[list[int]],
        eos_token_ids: set[int] | None = None,
    ):
        eos = set(eos_token_ids or set())
        self.single_ids: list[int] = sorted({int(t) for t in single_token_ids if int(t) not in eos})
        self.prefix_to_banned: dict[int, dict[tuple[int, ...], list[int]]] = {}
        for seq in multi_token_sequences:
            if len(seq) < 2:
                if len(seq) == 1 and int(seq[0]) not in eos and int(seq[0]) not in self.single_ids:
                    self.single_ids.append(int(seq[0]))
                continue
            last_tok = int(seq[-1])
            if last_tok in eos:
                continue
            pref = tuple(int(x) for x in seq[:-1])
            plen = len(pref)
            bucket = self.prefix_to_banned.setdefault(plen, {})
            bucket.setdefault(pref, []).append(last_tok)
        self.prefix_lengths: list[int] = sorted(self.prefix_to_banned.keys())
        self._single_tensor_cache: dict[torch.device, torch.Tensor] = {}

    def apply(self, scores: torch.Tensor, history_lists: list[list[int]]) -> torch.Tensor:
        """Mask forbidden next-token logits in-place on `scores` (`[B, V]`) using unpadded per-row `history_lists`."""
        vocab_size = scores.shape[-1]
        if self.single_ids:
            dev = scores.device
            idx_t = self._single_tensor_cache.get(dev)
            if idx_t is None:
                valid = [t for t in self.single_ids if 0 <= t < vocab_size]
                idx_t = torch.tensor(valid, dtype=torch.long, device=dev)
                self._single_tensor_cache[dev] = idx_t
            if idx_t.numel() > 0:
                scores.index_fill_(1, idx_t, float("-inf"))
        if self.prefix_lengths:
            for b, hist in enumerate(history_lists):
                hlen = len(hist)
                for plen in self.prefix_lengths:
                    if hlen < plen:
                        break
                    pref = tuple(hist[-plen:])
                    banned = self.prefix_to_banned[plen].get(pref)
                    if banned:
                        for tid in banned:
                            if 0 <= tid < vocab_size:
                                scores[b, tid] = float("-inf")
        return scores


# ---------------------------------------------------------------------------
# ChatML Formatting, Greedy Batched Decode (KV-Cached), and B=1 Teacher-Forced KL
# ---------------------------------------------------------------------------
def format_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Format messages using OLMo-3's chat template (or deterministic ChatML fallback for toy test tokenizers)."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def get_eos_and_pad_ids(model: Any, tokenizer: Any) -> tuple[list[int], int]:
    eos_set: list[int] = []
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "eos_token_id", None) is not None:
        raw_eos = gc.eos_token_id
        if isinstance(raw_eos, int):
            eos_set.append(raw_eos)
        elif isinstance(raw_eos, (list, tuple)):
            eos_set.extend(int(x) for x in raw_eos)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        if int(tokenizer.eos_token_id) not in eos_set:
            eos_set.append(int(tokenizer.eos_token_id))
    # Include <|im_end|> if present in tokenizer vocabulary
    try:
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end, int) and im_end >= 0 and im_end not in eos_set:
            eos_set.append(im_end)
    except Exception:
        pass
    if not eos_set:
        eos_set = [0]
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_set[0]
    return eos_set, int(pad_id)


def get_special_and_added_token_ids(tokenizer: Any) -> set[int]:
    ids: set[int] = set()
    for tid in getattr(tokenizer, "all_special_ids", []) or []:
        ids.add(int(tid))
    added = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(added, dict):
        for tid in added.keys():
            ids.add(int(tid))
    return ids


@torch.no_grad()
def forward_next_logits_batch(
    model: torch.nn.Module,
    token_id_lists: list[list[int]],
    pad_id: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute next-token logits `[B, V]` with left padding, exact `attention_mask` & `position_ids` (`use_cache=False`)."""
    bsz = len(token_id_lists)
    max_len = max(len(x) for x in token_id_lists)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    for b, ids in enumerate(token_id_lists):
        L = len(ids)
        input_ids[b, max_len - L :] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[b, max_len - L :] = 1
    position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    return out.logits[:, -1, :].float()


@torch.no_grad()
def greedy_decode_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts_text: list[str],
    max_new_tokens: int,
    batch_size: int,
    eos_ids: list[int],
    pad_id: int,
    ban_processor: CommonBanLogitsProcessor | None = None,
    master: WeightMaster | None = None,
) -> list[dict[str, Any]]:
    """Deterministic left-padded greedy decode (`do_sample=False`, `use_cache=True`) with exact ban enforcement."""
    if master is not None:
        master.check_hooks(model)
    device = next(model.parameters()).device
    eos_set = set(eos_ids)
    results: list[dict[str, Any]] = []

    for start in range(0, len(prompts_text), batch_size):
        chunk_texts = prompts_text[start : start + batch_size]
        prompt_ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in chunk_texts]
        bsz = len(prompt_ids_list)
        max_len = max(len(x) for x in prompt_ids_list)

        input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        for b, ids in enumerate(prompt_ids_list):
            L = len(ids)
            input_ids[b, max_len - L :] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[b, max_len - L :] = 1

        # Seed multi-token ban histories from GENERATED tokens only (matches HF bad_words_ids semantics, GLM F6)
        histories: list[list[int]] = [[] for _ in range(bsz)]
        gen_ids: list[list[int]] = [[] for _ in range(bsz)]
        finished = [False] * bsz
        past_key_values = None
        cur_input = input_ids
        cur_mask = attention_mask
        cur_pos = (cur_mask.cumsum(dim=-1) - 1).clamp(min=0)

        for step in range(max_new_tokens):
            out = model(
                input_ids=cur_input,
                attention_mask=cur_mask,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            scores = out.logits[:, -1, :].float()
            if ban_processor is not None:
                scores = ban_processor.apply(scores, histories)
            next_toks = torch.argmax(scores, dim=-1).tolist()

            next_col = []
            for b, tid in enumerate(next_toks):
                if finished[b]:
                    next_col.append(pad_id)
                elif tid in eos_set:
                    finished[b] = True
                    next_col.append(pad_id)
                else:
                    gen_ids[b].append(int(tid))
                    histories[b].append(int(tid))
                    next_col.append(int(tid))

            if all(finished):
                break

            cur_input = torch.tensor(next_col, dtype=torch.long, device=device).unsqueeze(1)
            ones_col = torch.ones((bsz, 1), dtype=torch.long, device=device)
            cur_mask = torch.cat([cur_mask, ones_col], dim=1)
            cur_pos = cur_pos[:, -1:] + 1

        for b in range(bsz):
            text = tokenizer.decode(gen_ids[b], skip_special_tokens=True)
            results.append({"text": text, "gen_ids": gen_ids[b], "n_tokens": len(gen_ids[b])})
    return results


@torch.no_grad()
def teacher_forced_gen_logits_b1(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """B=1, `use_cache=False`, float32 logits `[T, V]` predicting each of the `T` tokens in `gen_ids`."""
    if not gen_ids:
        inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        out = model(input_ids=inp, use_cache=False)
        return out.logits[0, -1:, :].float()
    full_ids = prompt_ids + gen_ids[:-1]
    inp = torch.tensor([full_ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False)
    start_idx = len(prompt_ids) - 1
    end_idx = start_idx + len(gen_ids)
    return out.logits[0, start_idx:end_idx, :].float()


def compute_off_kl_from_logits(
    plain_logits: torch.Tensor,
    arm_logits: torch.Tensor,
) -> tuple[float, float, float]:
    """Pinned off_KL estimator: (mean_kl_nats_per_tok, max_single_pos_kl_nats, max_abs_dlogit)."""
    lp_plain = F.log_softmax(plain_logits.float(), dim=-1)
    p_plain = lp_plain.exp()
    lp_arm = F.log_softmax(arm_logits.float(), dim=-1)
    kl_pos = (p_plain * (lp_plain - lp_arm)).sum(dim=-1)
    mean_kl = float(kl_pos.mean().item())
    max_pos_kl = float(kl_pos.max().item())
    max_dlogit = float((arm_logits.float() - plain_logits.float()).abs().max().item())
    return mean_kl, max_pos_kl, max_dlogit


def compute_prenorm_masked_direction(
    v_post_unit: torch.Tensor,
    gamma_ff: torch.Tensor,
    drop_idx: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute low-gamma masked pre_norm unit vector and geometry diagnostics (`cos_ret`, `eff_ratio`, etc.)."""
    v_p = v_post_unit.detach().cpu().float()
    v_p = v_p / v_p.norm().clamp_min(1e-12)
    g = gamma_ff.detach().cpu().float()
    rms_g = float(torch.sqrt((g * g).mean()).item())

    raw_unmasked = v_p / torch.where(
        g.abs() >= 1e-8, g, torch.where(g >= 0, torch.full_like(g, 1e-8), torch.full_like(g, -1e-8))
    )
    u_unmasked = raw_unmasked / raw_unmasked.norm().clamp_min(1e-12)
    eff_unmasked = float((g * u_unmasked).norm().item() / max(rms_g, 1e-12))

    raw_masked = raw_unmasked.clone()
    for idx in drop_idx:
        raw_masked[int(idx)] = 0.0
    v_pre_hat = raw_masked / raw_masked.norm().clamp_min(1e-12)
    post_delivered = g * v_pre_hat
    post_norm = float(post_delivered.norm().item())
    eff_masked = post_norm / max(rms_g, 1e-12)
    cos_ret = float(torch.dot(post_delivered / max(post_norm, 1e-12), v_p).item())
    eff_ratio = eff_masked / max(eff_unmasked, 1e-12)
    delivered_match_ratio = rms_g / max(post_norm, 1e-12)
    return v_pre_hat, {
        "cos_ret": round(cos_ret, 6),
        "eff_unmasked": round(eff_unmasked, 6),
        "eff_masked": round(eff_masked, 6),
        "eff_ratio": round(eff_ratio, 4),
        "post_norm_of_v_pre_hat": round(post_norm, 6),
        "rms_gamma": round(rms_g, 6),
        "delivered_match_ratio": round(delivered_match_ratio, 6),
    }


def build_layer_edits(
    layers: list[int],
    dose_total: float,
    v_pre_by_layer: dict[int, torch.Tensor],
    k_hat_by_layer: dict[int, torch.Tensor],
    calib_layer_stats: dict[int, dict[str, float]],
    gamma_by_layer: dict[int, torch.Tensor],
) -> tuple[dict[int, tuple[float, torch.Tensor, torch.Tensor]], dict[int, float]]:
    """Compute exact per-layer scalar `alpha_L(d)` and rank-1 edit tuple `(alpha_L, v_pre_hat, k_hat)`."""
    k_count = float(len(layers))
    edits: dict[int, tuple[float, torch.Tensor, torch.Tensor]] = {}
    alphas: dict[int, float] = {}
    for L in layers:
        pmn = float(calib_layer_stats[L]["pmn"])
        rms_pre = float(calib_layer_stats[L]["rms_pre"])
        c_bar = float(calib_layer_stats[L]["c_bar"])
        delta_L = (float(dose_total) / k_count) * pmn
        v_pre = v_pre_by_layer[L].float()
        g = gamma_by_layer[L].float()
        g_v_norm = float((g * v_pre).norm().item())
        alpha_L = (delta_L * rms_pre) / max(c_bar * g_v_norm, 1e-12)
        edits[L] = (alpha_L, v_pre, k_hat_by_layer[L].float())
        alphas[L] = alpha_L
    return edits, alphas


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 == 1 else 0.5 * (float(s[mid - 1]) + float(s[mid]))


def _write_checkpoint(checkpoint_dir: Path | None, name: str, payload: dict[str, Any]) -> None:
    """Best-effort atomic per-block partial checkpoint (stamped partial_checkpoint=True so scorer refuses it)."""
    if checkpoint_dir is None:
        return
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_dir / f".{name}.json.gz.tmp"
        dst = checkpoint_dir / f"{name}.json.gz"
        wrapped = {"partial_checkpoint": True, "block": name, "payload": payload}
        raw = (json.dumps(wrapped, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
        with gzip.open(tmp, "wb", compresslevel=6) as f:
            f.write(raw)
        tmp.replace(dst)
    except OSError as exc:
        print(f"[WARN] stage-c3 checkpoint '{name}' write failed ({exc!r}); continuing", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core Stage C3 Experiment Engine (Model-Agnostic: Real OLMo-3 L4 & Toy CPU)
# ---------------------------------------------------------------------------
def run_stage_c3_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    c3_constants: dict[str, Any],
    bank: dict[str, torch.Tensor],
    provenance_hashes: dict[str, str],
    *,
    smoke: bool = False,
    device_name: str = "unknown",
    checkpoint_dir: Path | None = None,
    target_layers: list[int] | None = None,
    requested_batch_size: int = 8,
    max_new_tokens_override: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Execute full Stage C3 (WSP v1) or smoke subset and return the complete receipt dictionary."""
    t0 = time.time()
    layers = list(target_layers if target_layers is not None else c3_constants["edit_spec"]["layers"])
    dose_ladder = [float(d) for d in c3_constants["edit_spec"]["dose_ladder_total_pmn_frac"]]
    sham_alpha = float(c3_constants["edit_spec"]["sham_alpha"])
    sys_pirate = str(c3_constants["system_prompts"]["pirate"])
    sys_nomd = str(c3_constants["system_prompts"]["nomd"])
    surface_regex = str(c3_constants["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    ban_single_ids = [int(x["id"]) for x in c3_constants["lexicons"]["common_pirate_ban"]["single_tokens"]]
    ban_multi_seqs = [[int(t) for t in x["ids"]] for x in c3_constants["lexicons"]["common_pirate_ban"]["multi_token_sequences"]]
    pirate_seed_ids = [int(x["id"]) for x in c3_constants["lexicons"]["pirate_seeds"]]
    pirate_seed_tokens = [str(x["token"]) for x in c3_constants["lexicons"]["pirate_seeds"]]

    device = next(model.parameters()).device
    eos_ids, pad_id = get_eos_and_pad_ids(model, tokenizer)
    special_ids = get_special_and_added_token_ids(tokenizer)
    common_ban_proc = CommonBanLogitsProcessor(ban_single_ids, ban_multi_seqs, eos_token_ids=set(eos_ids))

    tok_budget = {
        "calib": 32,
        "reserve": 48,
        "pillar_a": 64,
        "pillar_b": 96,
        "multiturn": 64,
        "d6": 64,
        "math": 64,
        "code": 96,
        "refusal": 64,
        "neutral": 64,
    }
    if max_new_tokens_override:
        tok_budget.update(max_new_tokens_override)

    bat = c3_constants["batteries"]
    calib_prompts = list(bat["calib_16"])
    reserve_prompts = list(bat["reserve_8"])
    identity_prompts = list(bat["identity_20"])
    implied_prompts = list(bat["implied_20"])
    pillar_a_prompts = identity_prompts + implied_prompts
    nomd_prompts = list(bat["nomd_24"])
    multiturn_prompts = list(bat["multiturn_12"])
    copula_pairs = list(bat["copula_pairs_12"])
    math_prompts = list(bat["math_20"])
    code_prompts = list(bat["code_12"])
    refusal_prompts = list(bat["refusal_8"])
    neutral_prompts = list(bat["neutral_chat_16"])
    demo_showcase_ids = list(bat["demo_showcase_ids_12"])

    if smoke:
        calib_prompts = calib_prompts[:2]
        reserve_prompts = reserve_prompts[:2]
        identity_prompts = identity_prompts[:2]
        implied_prompts = implied_prompts[:2]
        pillar_a_prompts = identity_prompts + implied_prompts
        nomd_prompts = nomd_prompts[:2]
        multiturn_prompts = multiturn_prompts[:2]
        copula_pairs = copula_pairs[:2]
        math_prompts = math_prompts[:2]
        code_prompts = code_prompts[:2]
        refusal_prompts = refusal_prompts[:2]
        neutral_prompts = neutral_prompts[:2]
        demo_showcase_ids = [p["id"] for p in pillar_a_prompts[:2]]
        dose_ladder = [0.15, 0.30]

    master = WeightMaster(model, layers)
    master.check_hooks(model)

    gamma_by_layer: dict[int, torch.Tensor] = {
        L: model.model.layers[L].post_feedforward_layernorm.weight.detach().cpu().float().clone()
        for L in layers
    }
    drop_idx_by_layer: dict[int, list[int]] = {
        L: [int(x) for x in bank[f"L{L}_gamma_drop_idx"].tolist()] for L in layers
    }

    counts = {
        "generations": 0,
        "generated_tokens": 0,
        "teacher_forced_b1_fwds": 0,
        "preflight_fwds": 0,
    }
    arm_liveness: dict[str, float] = {}

    def _note_liveness(key: str, mv: float) -> None:
        if key not in arm_liveness or mv > arm_liveness[key]:
            arm_liveness[key] = float(mv)

    def _fmt_plain(prompt_text: str) -> str:
        return format_chat_messages(tokenizer, [{"role": "user", "content": prompt_text}], True)

    def _fmt_sys(sys_text: str, prompt_text: str) -> str:
        return format_chat_messages(
            tokenizer,
            [{"role": "system", "content": sys_text}, {"role": "user", "content": prompt_text}],
            True,
        )

    def _gen(
        texts: list[str],
        max_tok: int,
        batch_sz: int,
        ban_proc: CommonBanLogitsProcessor | None = None,
    ) -> list[dict[str, Any]]:
        out = greedy_decode_batch(
            model, tokenizer, texts, max_tok, batch_sz, eos_ids, pad_id,
            ban_processor=ban_proc, master=master,
        )
        counts["generations"] += len(out)
        counts["generated_tokens"] += sum(r["n_tokens"] for r in out)
        return out

    # ------------------------------------------------------------------
    # Step A: C3.0.1 B=8 vs B=1 Left-Padded Logit Parity Preflight
    # ------------------------------------------------------------------
    res_plain_texts = [_fmt_plain(p["text"]) for p in reserve_prompts]
    res_plain_id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in res_plain_texts]
    b1_logits = torch.cat(
        [forward_next_logits_batch(model, [ids], pad_id, device) for ids in res_plain_id_lists],
        dim=0,
    )
    counts["preflight_fwds"] += len(res_plain_id_lists)
    if requested_batch_size > 1 and len(res_plain_id_lists) > 1:
        b8_logits = forward_next_logits_batch(model, res_plain_id_lists, pad_id, device)
        counts["preflight_fwds"] += 1
        b8_parity_max_dlogit: float | None = float((b8_logits - b1_logits).abs().max().item())
        b8_parity_passed: bool | None = b8_parity_max_dlogit <= 0.05
        gen_batch_size = requested_batch_size if b8_parity_passed else 1
    else:
        # DS F2: never record a skipped C3.0.1 preflight as passed!
        b8_parity_max_dlogit = None
        b8_parity_passed = None
        gen_batch_size = 1

    # ------------------------------------------------------------------
    # Step B: Disjoint `calib_16` Calibration (Hooks ONLY here, removed immediately!)
    # ------------------------------------------------------------------
    cal_plain_texts = [_fmt_plain(p["text"]) for p in calib_prompts]
    cal_pirate_texts = [_fmt_sys(sys_pirate, p["text"]) for p in calib_prompts]
    cal_nomd_texts = [_fmt_sys(sys_nomd, p["text"]) for p in calib_prompts]

    cal_plain_gens = _gen(cal_plain_texts, tok_budget["calib"], gen_batch_size)
    cal_pirate_gens = _gen(cal_pirate_texts, tok_budget["calib"], gen_batch_size)
    cal_nomd_gens = _gen(cal_nomd_texts, tok_budget["calib"], gen_batch_size)

    def _capture_calib_activations(
        prompt_texts: list[str],
        gens: list[dict[str, Any]],
        capture_plain_stats: bool,
    ) -> dict[str, Any]:
        """Temporary read-only capture hooks on `calib_16` (turn-start + first 16 generated tokens)."""
        m_list: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        pre_rms_list: dict[int, list[float]] = {L: [] for L in layers}
        pmn_list: dict[int, list[float]] = {L: [] for L in layers}
        h_out_list: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        cur_slice_start: list[int] = [0]

        handles = []
        try:
            for L in layers:
                layer_mod = model.model.layers[L]

                def _make_down_hook(li: int):
                    def _hook(_mod, inputs, output):
                        s0 = cur_slice_start[0]
                        m_in = inputs[0][0, s0:, :].detach().cpu().float()
                        pre_out = output[0, s0:, :].detach().cpu().float()
                        m_list[li].append(m_in)
                        rms_vals = torch.sqrt((pre_out * pre_out).mean(dim=-1)).tolist()
                        pre_rms_list[li].extend(float(x) for x in rms_vals)
                    return _hook

                def _make_pffn_hook(li: int):
                    def _hook(_mod, _inputs, output):
                        s0 = cur_slice_start[0]
                        post_out = output[0, s0:, :].detach().cpu().float()
                        pmn_list[li].extend(float(x) for x in post_out.norm(dim=-1).tolist())
                    return _hook

                def _make_block_hook(li: int):
                    def _hook(_mod, _inputs, output):
                        s0 = cur_slice_start[0]
                        h_t = output[0] if isinstance(output, tuple) else output
                        h_out_list[li].append(h_t[0, s0:, :].detach().cpu().float())
                    return _hook

                if capture_plain_stats:
                    handles.append(layer_mod.mlp.down_proj.register_forward_hook(_make_down_hook(L)))
                    handles.append(layer_mod.post_feedforward_layernorm.register_forward_hook(_make_pffn_hook(L)))
                handles.append(layer_mod.register_forward_hook(_make_block_hook(L)))

            with torch.no_grad():
                for ptxt, g in zip(prompt_texts, gens):
                    p_ids = tokenizer.encode(ptxt, add_special_tokens=False)
                    # Turn-start token (last prompt token) + first 16 generated tokens
                    g_ids = g["gen_ids"][:16]
                    full_ids = p_ids + g_ids
                    cur_slice_start[0] = len(p_ids) - 1
                    inp = torch.tensor([full_ids], dtype=torch.long, device=device)
                    model(input_ids=inp, use_cache=False)
                    counts["teacher_forced_b1_fwds"] += 1
        finally:
            for h in handles:
                h.remove()
        master.check_hooks(model)
        return {
            "m": {L: torch.cat(m_list[L], dim=0) for L in layers} if capture_plain_stats else {},
            "pre_rms": pre_rms_list,
            "pmn": pmn_list,
            "h_out": {L: torch.cat(h_out_list[L], dim=0) for L in layers},
        }

    cap_plain = _capture_calib_activations(cal_plain_texts, cal_plain_gens, capture_plain_stats=True)
    cap_pirate = _capture_calib_activations(cal_pirate_texts, cal_pirate_gens, capture_plain_stats=False)
    cap_nomd = _capture_calib_activations(cal_nomd_texts, cal_nomd_gens, capture_plain_stats=False)
    master.check_hooks(model)

    k_hat_by_layer: dict[int, torch.Tensor] = {}
    calib_layer_stats: dict[int, dict[str, float]] = {}
    v_pre_pirate_con: dict[int, torch.Tensor] = {}
    v_pre_nomd_con: dict[int, torch.Tensor] = {}
    v_pre_pirate_lex: dict[int, torch.Tensor] = {}
    v_pre_nomd_lex: dict[int, torch.Tensor] = {}
    v_pre_rand_pirate: dict[int, dict[int, torch.Tensor]] = {s: {} for s in range(3)}
    v_pre_rand_nomd: dict[int, dict[int, torch.Tensor]] = {s: {} for s in range(3)}
    direction_quality: dict[str, dict[str, Any]] = {
        "WSP_Pirate_Contrastive": {},
        "WSP_Pirate_Lexicon": {},
        "WSP_NoMD_Contrastive": {},
        "WSP_NoMD_Lexicon": {},
    }

    for L in layers:
        m_mat = cap_plain["m"][L]  # [N_tok, d_ff]
        m_bar = m_mat.mean(dim=0)
        k_hat = m_bar / m_bar.norm().clamp_min(1e-12)
        projs = m_mat @ k_hat
        c_bar = float(projs.mean().item())
        c_std = float(projs.std(unbiased=False).item())
        cv = c_std / max(abs(c_bar), 1e-12)
        max_deliv = float(projs.max().item()) / max(abs(c_bar), 1e-12)
        pmn_val = sum(cap_plain["pmn"][L]) / max(len(cap_plain["pmn"][L]), 1)
        rms_pre_val = sum(cap_plain["pre_rms"][L]) / max(len(cap_plain["pre_rms"][L]), 1)

        k_hat_by_layer[L] = k_hat
        calib_layer_stats[L] = {
            "n_tokens": int(m_mat.shape[0]),
            "c_bar": round(c_bar, 6),
            "c_std": round(c_std, 6),
            "cv": round(cv, 6),
            "max_delivery_ratio": round(max_deliv, 6),
            "pmn": round(pmn_val, 6),
            "rms_pre": round(rms_pre_val, 6),
        }

        # Contrastive mean residual difference directions (aligned by mean over calib_16 tokens)
        dh_pirate = cap_pirate["h_out"][L].mean(dim=0) - cap_plain["h_out"][L].mean(dim=0)
        v_post_p = dh_pirate / dh_pirate.norm().clamp_min(1e-12)
        v_pre_p, q_p = compute_prenorm_masked_direction(v_post_p, gamma_by_layer[L], drop_idx_by_layer[L])
        v_pre_pirate_con[L] = v_pre_p
        direction_quality["WSP_Pirate_Contrastive"][str(L)] = q_p

        dh_nomd = cap_nomd["h_out"][L].mean(dim=0) - cap_plain["h_out"][L].mean(dim=0)
        v_post_n = dh_nomd / dh_nomd.norm().clamp_min(1e-12)
        v_pre_n, q_n = compute_prenorm_masked_direction(v_post_n, gamma_by_layer[L], drop_idx_by_layer[L])
        v_pre_nomd_con[L] = v_pre_n
        direction_quality["WSP_NoMD_Contrastive"][str(L)] = q_n

        # Frozen lexicon vectors from bank + live quality check against live gamma
        v_pre_pirate_lex[L] = bank[f"L{L}_pirate_lex_pre_unit"].float()
        _, q_plex = compute_prenorm_masked_direction(
            bank[f"L{L}_pirate_lex_post_unit"].float(), gamma_by_layer[L], drop_idx_by_layer[L]
        )
        direction_quality["WSP_Pirate_Lexicon"][str(L)] = q_plex

        v_pre_nomd_lex[L] = bank[f"L{L}_nomd_lex_pre_unit"].float()
        _, q_nlex = compute_prenorm_masked_direction(
            bank[f"L{L}_nomd_lex_post_unit"].float(), gamma_by_layer[L], drop_idx_by_layer[L]
        )
        direction_quality["WSP_NoMD_Lexicon"][str(L)] = q_nlex

        for s in range(3):
            v_pre_rand_pirate[s][L] = bank[f"L{L}_rand_pirate_pre_unit_s{s}"].float()
            v_pre_rand_nomd[s][L] = bank[f"L{L}_rand_nomd_pre_unit_s{s}"].float()

    # ------------------------------------------------------------------
    # Step C: C3.0.1 Sham (alpha=0.05), D4 L24 Masked/Unmasked/Mis-Masked, Stem Positive Preflight
    # ------------------------------------------------------------------
    sham_edits = {L: (sham_alpha, v_pre_pirate_lex[L], k_hat_by_layer[L]) for L in layers}
    with master.apply_edit(model, sham_edits) as sham_ret:
        sham_logits = forward_next_logits_batch(model, [res_plain_id_lists[0]], pad_id, device)
        counts["preflight_fwds"] += 1
    sham_max_abs_dlogit = float((sham_logits[0] - b1_logits[0]).abs().max().item())
    sham_min_proj_ret = min(v["stored_rank1_proj_ret"] for v in sham_ret.values())
    sham_info = {
        "alpha": sham_alpha,
        "max_abs_dlogit": round(sham_max_abs_dlogit, 6),
        "min_stored_rank1_proj_ret": round(sham_min_proj_ret, 6),
        "per_layer": sham_ret,
    }

    # D4 L24 Preflight (12 prompts x 3 conditions = 36 forwards at matched alpha_L24(d=0.30))
    l24 = layers[1] if len(layers) >= 2 else layers[0]
    d4_prompts_ids = res_plain_id_lists + [
        tokenizer.encode(t, add_special_tokens=False) for t in cal_plain_texts[: min(4, len(cal_plain_texts))]
    ]
    d4_clean_logits = [
        forward_next_logits_batch(model, [ids], pad_id, device)[0] for ids in d4_prompts_ids
    ]
    counts["preflight_fwds"] += len(d4_prompts_ids)

    d4_edits_ref, d4_alphas_ref = build_layer_edits(
        layers, 0.30, v_pre_pirate_lex, k_hat_by_layer, calib_layer_stats, gamma_by_layer
    )
    alpha_d4 = d4_alphas_ref[l24]
    v_l24_masked = bank[f"L{l24}_pirate_lex_pre_unit"].float()
    v_l24_unmasked = bank[f"L{l24}_pirate_unmasked_pre_unit"].float()
    d_model = gamma_by_layer[l24].shape[0]
    mismask_idx = [(idx + 1) % d_model for idx in drop_idx_by_layer[l24]]
    v_l24_mismasked, d4_mismask_geom = compute_prenorm_masked_direction(
        bank[f"L{l24}_pirate_lex_post_unit"].float(), gamma_by_layer[l24], mismask_idx
    )

    valid_seed_ids = [tid for tid in pirate_seed_ids if 0 <= tid < b1_logits.shape[-1]]
    if not valid_seed_ids:
        valid_seed_ids = [0]

    d4_results: dict[str, Any] = {"layer": l24, "alpha_L24": round(alpha_d4, 6), "mismask_idx": mismask_idx, "mismask_geom": d4_mismask_geom}
    for cond_name, vec in (
        ("masked", v_l24_masked),
        ("unmasked", v_l24_unmasked),
        ("mismasked", v_l24_mismasked),
    ):
        seed_dlogits: list[float] = []
        l2_dlogits: list[float] = []
        with master.apply_edit(model, {l24: (alpha_d4, vec, k_hat_by_layer[l24])}) as d4_ret:
            for ids, cl in zip(d4_prompts_ids, d4_clean_logits):
                el = forward_next_logits_batch(model, [ids], pad_id, device)[0]
                counts["preflight_fwds"] += 1
                diff = el - cl
                seed_dlogits.append(float(diff[valid_seed_ids].mean().item()))
                l2_dlogits.append(float(diff.norm().item()))
        d4_results[cond_name] = {
            "mean_pirate_seed_dlogit": round(sum(seed_dlogits) / len(seed_dlogits), 6),
            "mean_logit_l2": round(sum(l2_dlogits) / len(l2_dlogits), 6),
            "retention": d4_ret[str(l24)],
        }
    unm_l2 = max(abs(d4_results["unmasked"]["mean_logit_l2"]), 1e-12)
    unm_seed = d4_results["unmasked"]["mean_pirate_seed_dlogit"]
    d4_results["masked_over_unmasked_logit_l2_ratio"] = round(d4_results["masked"]["mean_logit_l2"] / unm_l2, 6)
    d4_results["mismasked_over_unmasked_logit_l2_ratio"] = round(d4_results["mismasked"]["mean_logit_l2"] / unm_l2, 6)
    d4_results["masked_over_unmasked_seed_dlogit_ratio"] = (
        round(d4_results["masked"]["mean_pirate_seed_dlogit"] / unm_seed, 6) if abs(unm_seed) > 1e-9 else None
    )
    d4_results["mismasked_over_unmasked_seed_dlogit_ratio"] = (
        round(d4_results["mismasked"]["mean_pirate_seed_dlogit"] / unm_seed, 6) if abs(unm_seed) > 1e-9 else None
    )

    # C3.0.2 unbanned stem-audit positive preflight check
    stem_preflight = {
        "positive_sample_hit": c3_scoring.surface_stem_hit("Ahoy matey, hoist the anchor on this pirate ship!", surface_regex),
        "negative_sample_hit": c3_scoring.surface_stem_hit("Please describe how to organize a daily study schedule.", surface_regex),
    }

    _write_checkpoint(checkpoint_dir, "step_abc_calib_preflights", {
        "calibration": calib_layer_stats, "direction_quality": direction_quality, "sham": sham_info, "D4": d4_results,
    })

    # ------------------------------------------------------------------
    # Step D: RESERVE_PROMPTS_8 Plain Baseline, 6-Rung Commit Gate & M8 Rand Reserve KL
    # ------------------------------------------------------------------
    res_plain_gens = _gen(res_plain_texts, tok_budget["reserve"], gen_batch_size)
    res_plain_tf_logits: list[torch.Tensor] = []
    for p_ids, g in zip(res_plain_id_lists, res_plain_gens):
        res_plain_tf_logits.append(teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device))
        counts["teacher_forced_b1_fwds"] += 1

    compiled_arm_specs: list[tuple[str, dict[int, torch.Tensor]]] = [
        ("WSP_Pirate_Contrastive", v_pre_pirate_con),
        ("WSP_Pirate_Lexicon", v_pre_pirate_lex),
        ("WSP_NoMD_Contrastive", v_pre_nomd_con),
        ("WSP_NoMD_Lexicon", v_pre_nomd_lex),
    ]

    commit_gate_receipt: dict[str, Any] = {
        "plain_reserve": [
            {
                "id": p["id"],
                "text": g["text"],
                "n_tokens": g["n_tokens"],
                "is_broken": c3_scoring.is_broken(g["text"]),
            }
            for p, g in zip(reserve_prompts, res_plain_gens)
        ],
        "arms": {},
    }
    selected_dose: dict[str, float] = {}
    selected_alphas: dict[str, dict[int, float]] = {}
    selected_retention: dict[str, dict[str, dict[str, float]]] = {}
    commit_gate_fallback: dict[str, bool] = {}

    for arm_name, v_map in compiled_arm_specs:
        rung_rows: list[dict[str, Any]] = []
        passing_rungs: list[tuple[float, dict[int, float], dict[str, dict[str, float]]]] = []
        for d in dose_ladder:
            edits, alphas = build_layer_edits(layers, d, v_map, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
            with master.apply_edit(model, edits) as ret_metrics:
                per_prompt_kl: list[dict[str, Any]] = []
                for p, p_ids, g, pl_log in zip(reserve_prompts, res_plain_id_lists, res_plain_gens, res_plain_tf_logits):
                    arm_log = teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device)
                    counts["teacher_forced_b1_fwds"] += 1
                    m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
                    _note_liveness(f"{arm_name}@d={d}", max_dl)
                    per_prompt_kl.append({
                        "id": p["id"],
                        "off_kl": round(m_kl, 6),
                        "max_single_pos_kl": round(max_p_kl, 6),
                        "max_abs_dlogit": round(max_dl, 6),
                    })
                res_arm_gens = _gen(res_plain_texts, tok_budget["reserve"], gen_batch_size)

            rep_ok = all(
                (v["stored_rank1_proj_ret"] >= 0.50 and v["stored_frob_ret"] >= 0.80)
                for v in ret_metrics.values()
            )
            dose_unrep = not rep_ok
            med_kl = _median([r["off_kl"] for r in per_prompt_kl])
            max_kl = max(r["off_kl"] for r in per_prompt_kl)
            max_pos_kl = max(r["max_single_pos_kl"] for r in per_prompt_kl)
            res_gen_rows = [
                {
                    "id": p["id"],
                    "text": g["text"],
                    "n_tokens": g["n_tokens"],
                    "is_broken": c3_scoring.is_broken(g["text"]),
                }
                for p, g in zip(reserve_prompts, res_arm_gens)
            ]
            n_broken = sum(1 for r in res_gen_rows if r["is_broken"])
            rung_pass = (
                (not dose_unrep)
                and (med_kl <= 0.25)
                and (max_kl <= 0.60)
                and (max_pos_kl <= 4.0)
                and (n_broken == 0)
            )
            rung_rows.append({
                "dose": d,
                "alphas": {str(L): round(alphas[L], 8) for L in layers},
                "bf16_retention": ret_metrics,
                "dose_unrepresentable": dose_unrep,
                "median_off_kl": round(med_kl, 6),
                "max_prompt_off_kl": round(max_kl, 6),
                "max_single_pos_kl": round(max_pos_kl, 6),
                "n_broken": n_broken,
                "pass": rung_pass,
                "per_prompt_kl": per_prompt_kl,
                "reserve_generations": res_gen_rows,
            })
            if rung_pass:
                passing_rungs.append((d, alphas, ret_metrics))

        if passing_rungs:
            best_d, best_alphas, best_ret = passing_rungs[-1]
            selected_dose[arm_name] = best_d
            selected_alphas[arm_name] = best_alphas
            selected_retention[arm_name] = best_ret
            commit_gate_fallback[arm_name] = False
        else:
            # Fallback dose for runner continuity; commit_gate_fallback=True forces DOSE_INVALID in scorer!
            fb_d = dose_ladder[0]
            _, fb_alphas = build_layer_edits(layers, fb_d, v_map, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
            selected_dose[arm_name] = fb_d
            selected_alphas[arm_name] = fb_alphas
            selected_retention[arm_name] = rung_rows[0]["bf16_retention"]
            commit_gate_fallback[arm_name] = True

        commit_gate_receipt["arms"][arm_name] = {
            "selected_d_star": selected_dose[arm_name],
            "commit_gate_fallback": commit_gate_fallback[arm_name],
            "rungs": rung_rows,
        }

    d_con = selected_dose["WSP_Pirate_Contrastive"]
    d_lex = selected_dose["WSP_Pirate_Lexicon"]
    d_common = min(d_con, d_lex)
    d_nomd_con = selected_dose["WSP_NoMD_Contrastive"]
    d_nomd_lex = selected_dose["WSP_NoMD_Lexicon"]

    # Helper to construct exact matched random edits using the paired compiled arm's EXACT scalar alphas!
    def _rand_edits_matching(
        rand_v_map: dict[int, torch.Tensor],
        ref_alphas: dict[int, float],
    ) -> dict[int, tuple[float, torch.Tensor, torch.Tensor]]:
        return {L: (float(ref_alphas[L]), rand_v_map[L].float(), k_hat_by_layer[L].float()) for L in layers}

    def _compile_arm_edits(arm_name: str, dose_val: float) -> tuple[dict[int, tuple[float, torch.Tensor, torch.Tensor]], dict[int, float]]:
        v_lookup = {
            "WSP_Pirate_Contrastive": v_pre_pirate_con,
            "WSP_Pirate_Lexicon": v_pre_pirate_lex,
            "WSP_NoMD_Contrastive": v_pre_nomd_con,
            "WSP_NoMD_Lexicon": v_pre_nomd_lex,
        }
        return build_layer_edits(layers, dose_val, v_lookup[arm_name], k_hat_by_layer, calib_layer_stats, gamma_by_layer)

    # M8: Rand Reserve off_KL & intended/stored Frobenius norm match across all distinct d* in use
    rand_reserve_receipt: dict[str, Any] = {}
    rand_check_specs: list[tuple[str, str, float, dict[int, torch.Tensor], dict[int, float], dict[str, dict[str, float]]]] = []
    for s in range(3):
        rand_check_specs.append((
            f"Rand_Pirate_s{s}@d*_con", "WSP_Pirate_Contrastive@d*_con", d_con,
            v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_Contrastive"], selected_retention["WSP_Pirate_Contrastive"],
        ))
        if d_lex != d_con:
            rand_check_specs.append((
                f"Rand_Pirate_s{s}@d*_lex", "WSP_Pirate_Lexicon@d*_lex", d_lex,
                v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_Lexicon"], selected_retention["WSP_Pirate_Lexicon"],
            ))
        rand_check_specs.append((
            f"Rand_NoMD_s{s}@d*_nomd_con", "WSP_NoMD_Contrastive@d*_nomd_con", d_nomd_con,
            v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_Contrastive"], selected_retention["WSP_NoMD_Contrastive"],
        ))
        if d_nomd_lex != d_nomd_con:
            rand_check_specs.append((
                f"Rand_NoMD_s{s}@d*_nomd_lex", "WSP_NoMD_Lexicon@d*_nomd_lex", d_nomd_lex,
                v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_Lexicon"], selected_retention["WSP_NoMD_Lexicon"],
            ))

    for rlabel, paired_edit_label, rdose, rv_map, ref_alphas, ref_ret in rand_check_specs:
        redits = _rand_edits_matching(rv_map, ref_alphas)
        with master.apply_edit(model, redits) as r_ret:
            r_kls: list[dict[str, Any]] = []
            for p, p_ids, g, pl_log in zip(reserve_prompts, res_plain_id_lists, res_plain_gens, res_plain_tf_logits):
                arm_log = teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device)
                counts["teacher_forced_b1_fwds"] += 1
                m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
                _note_liveness(rlabel, max_dl)
                r_kls.append({"id": p["id"], "off_kl": round(m_kl, 6), "max_single_pos_kl": round(max_p_kl, 6)})
        intended_ratios = {
            str(L): round(r_ret[str(L)]["intended_frob"] / max(ref_ret[str(L)]["intended_frob"], 1e-12), 8)
            for L in layers
        }
        stored_ratios = {
            str(L): round(r_ret[str(L)]["stored_frob"] / max(ref_ret[str(L)]["stored_frob"], 1e-12), 6)
            for L in layers
        }
        rand_reserve_receipt[rlabel] = {
            "paired_edit": paired_edit_label,
            "dose": rdose,
            "bf16_retention": r_ret,
            "intended_frob_ratio_vs_edit": intended_ratios,
            "stored_frob_ratio_vs_edit": stored_ratios,
            "median_off_kl": round(_median([x["off_kl"] for x in r_kls]), 6),
            "max_prompt_off_kl": round(max(x["off_kl"] for x in r_kls), 6),
            "per_prompt_kl": r_kls,
        }

    _write_checkpoint(checkpoint_dir, "step_d_commit_gate", {
        "commit_gate": commit_gate_receipt, "rand_reserve": rand_reserve_receipt,
    })

    # ------------------------------------------------------------------
    # Build Active Arm Registry (Maps Arm Label -> (sys_prompt_or_None, edits_or_None))
    # ------------------------------------------------------------------
    edits_con_dstar, _ = _compile_arm_edits("WSP_Pirate_Contrastive", d_con)
    edits_lex_dstar, _ = _compile_arm_edits("WSP_Pirate_Lexicon", d_lex)
    edits_con_dcommon, _ = _compile_arm_edits("WSP_Pirate_Contrastive", d_common)
    edits_lex_dcommon, _ = _compile_arm_edits("WSP_Pirate_Lexicon", d_common)
    edits_nomd_con_dstar, _ = _compile_arm_edits("WSP_NoMD_Contrastive", d_nomd_con)
    edits_nomd_lex_dstar, _ = _compile_arm_edits("WSP_NoMD_Lexicon", d_nomd_lex)

    arm_registry: dict[str, tuple[str | None, dict[int, tuple[float, torch.Tensor, torch.Tensor]] | None]] = {
        "Plain": (None, None),
        "Sys_Pirate": (sys_pirate, None),
        "Sys_NoMD": (sys_nomd, None),
        "WSP_Pirate_Contrastive@d*_con": (None, edits_con_dstar),
        "WSP_Pirate_Lexicon@d*_lex": (None, edits_lex_dstar),
        "WSP_Pirate_Contrastive@d_common": (None, edits_con_dcommon),
        "WSP_Pirate_Lexicon@d_common": (None, edits_lex_dcommon),
        "WSP_NoMD_Contrastive@d*_nomd_con": (None, edits_nomd_con_dstar),
        "WSP_NoMD_Lexicon@d*_nomd_lex": (None, edits_nomd_lex_dstar),
    }
    for s in range(3):
        arm_registry[f"Rand_Pirate_s{s}@d*_con"] = (None, _rand_edits_matching(v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_Contrastive"]))
        arm_registry[f"Rand_Pirate_s{s}@d*_lex"] = (None, _rand_edits_matching(v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_Lexicon"]))
        arm_registry[f"Rand_NoMD_s{s}@d*_nomd_con"] = (None, _rand_edits_matching(v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_Contrastive"]))
        arm_registry[f"Rand_NoMD_s{s}@d*_nomd_lex"] = (None, _rand_edits_matching(v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_Lexicon"]))

    # Record d_common retention if distinct from d*
    d_common_retention: dict[str, Any] = {}
    for lbl, base_lbl, d_val in (
        ("WSP_Pirate_Contrastive@d_common", "WSP_Pirate_Contrastive", d_common),
        ("WSP_Pirate_Lexicon@d_common", "WSP_Pirate_Lexicon", d_common),
    ):
        if d_val == selected_dose[base_lbl]:
            d_common_retention[lbl] = selected_retention[base_lbl]
        else:
            with master.apply_edit(model, arm_registry[lbl][1]) as dc_ret:
                d_common_retention[lbl] = dc_ret

    def _run_arm_generations(
        arm_label: str,
        prompts_list: list[dict[str, Any]],
        max_tok: int,
        ban_proc: CommonBanLogitsProcessor | None = None,
    ) -> list[dict[str, Any]]:
        sys_p, edits = arm_registry[arm_label]
        formatted = [_fmt_sys(sys_p, p["text"]) if sys_p is not None else _fmt_plain(p["text"]) for p in prompts_list]
        if edits is None:
            gens = _gen(formatted, max_tok, gen_batch_size, ban_proc=ban_proc)
        else:
            with master.apply_edit(model, edits):
                gens = _gen(formatted, max_tok, gen_batch_size, ban_proc=ban_proc)
        rows = []
        for p, g in zip(prompts_list, gens):
            txt = g["text"]
            rows.append({
                "id": p["id"],
                "text": txt,
                "n_tokens": g["n_tokens"],
                "is_broken": c3_scoring.is_broken(txt),
                "surface_stem_hit": c3_scoring.surface_stem_hit(txt, surface_regex),
            })
        return rows

    # ------------------------------------------------------------------
    # Step E: Pillar A (N=40 x 2 bans), D5 Multi-Turn, D6 Adaptive/Donor, D1 Copula
    # ------------------------------------------------------------------
    pillar_a_arms: list[str] = [
        "Plain",
        "Sys_Pirate",
        "WSP_Pirate_Contrastive@d*_con",
        "WSP_Pirate_Lexicon@d*_lex",
        "Rand_Pirate_s0@d*_con",
        "Rand_Pirate_s1@d*_con",
        "Rand_Pirate_s2@d*_con",
    ]
    if d_con != d_lex:
        pillar_a_arms.extend(["Rand_Pirate_s0@d*_lex", "Rand_Pirate_s1@d*_lex", "Rand_Pirate_s2@d*_lex"])
        if d_common != d_con:
            pillar_a_arms.append("WSP_Pirate_Contrastive@d_common")
        if d_common != d_lex:
            pillar_a_arms.append("WSP_Pirate_Lexicon@d_common")

    pillar_a_receipt: dict[str, dict[str, list[dict[str, Any]]]] = {"none": {}, "common_pirate_ban": {}}
    for arm_label in pillar_a_arms:
        pillar_a_receipt["none"][arm_label] = _run_arm_generations(
            arm_label, pillar_a_prompts, tok_budget["pillar_a"], ban_proc=None
        )
        pillar_a_receipt["common_pirate_ban"][arm_label] = _run_arm_generations(
            arm_label, pillar_a_prompts, tok_budget["pillar_a"], ban_proc=common_ban_proc
        )
    # Alias d_common if equal to d* so receipt always contains explicit d_common keys
    for ban_k in ("none", "common_pirate_ban"):
        if "WSP_Pirate_Contrastive@d_common" not in pillar_a_receipt[ban_k]:
            pillar_a_receipt[ban_k]["WSP_Pirate_Contrastive@d_common"] = pillar_a_receipt[ban_k]["WSP_Pirate_Contrastive@d*_con"]
        if "WSP_Pirate_Lexicon@d_common" not in pillar_a_receipt[ban_k]:
            pillar_a_receipt[ban_k]["WSP_Pirate_Lexicon@d_common"] = pillar_a_receipt[ban_k]["WSP_Pirate_Lexicon@d*_lex"]

    _write_checkpoint(checkpoint_dir, "step_e1_pillar_a", pillar_a_receipt)

    # D5 Multi-Turn (MULTITURN_12 x 3 turns x 5 arms, ban=none)
    d5_arms = list(c3_constants["multiturn_d5_arms_5"])
    d5_receipt: dict[str, list[dict[str, Any]]] = {}
    for arm_label in d5_arms:
        sys_p, edits = arm_registry[arm_label]
        convs: list[list[dict[str, str]]] = [
            ([{"role": "system", "content": sys_p}] if sys_p is not None else [])
            for _ in multiturn_prompts
        ]
        per_turn_replies: list[list[dict[str, Any]]] = [[] for _ in multiturn_prompts]

        def _run_d5_turns():
            for t_idx in range(3):
                for idx, item in enumerate(multiturn_prompts):
                    convs[idx].append({"role": "user", "content": item["turns"][t_idx]})
                formatted = [format_chat_messages(tokenizer, c, True) for c in convs]
                gens = _gen(formatted, tok_budget["multiturn"], gen_batch_size)
                for idx, g in enumerate(gens):
                    convs[idx].append({"role": "assistant", "content": g["text"]})
                    per_turn_replies[idx].append({
                        "turn": t_idx + 1,
                        "text": g["text"],
                        "n_tokens": g["n_tokens"],
                        "is_broken": c3_scoring.is_broken(g["text"]),
                    })

        if edits is None:
            _run_d5_turns()
        else:
            with master.apply_edit(model, edits):
                _run_d5_turns()
        d5_receipt[arm_label] = [
            {"id": item["id"], "turns": per_turn_replies[idx]}
            for idx, item in enumerate(multiturn_prompts)
        ]

    # D6 Adaptive Part 168 Crutch-Removal + Matched-Donor Ban (DEMO_SHOWCASE_IDS_12)
    cal_plain_id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in cal_plain_texts]
    cal_plain_tf_logits: list[torch.Tensor] = []
    for p_ids, g in zip(cal_plain_id_lists, cal_plain_gens):
        cal_plain_tf_logits.append(teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device))
        counts["teacher_forced_b1_fwds"] += 1

    # GLM F7: exclude both 95 single-token ban IDs and all token IDs inside 380 multi-token ban sequences
    common_single_set = set(ban_single_ids)
    common_multi_member_set = {tid for seq in ban_multi_seqs for tid in seq}
    excluded_d6_ids = common_single_set | common_multi_member_set | special_ids
    top_k_d6 = int(c3_constants["lexicons"]["secondary_d6_own_top_k_bias"])

    def _compute_own_top_k(arm_label: str) -> list[dict[str, Any]]:
        _, edits = arm_registry[arm_label]
        assert edits is not None
        diff_sum: torch.Tensor | None = None
        total_pos = 0
        with master.apply_edit(model, edits):
            for p_ids, g, pl_log in zip(cal_plain_id_lists, cal_plain_gens, cal_plain_tf_logits):
                arm_log = teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device)
                counts["teacher_forced_b1_fwds"] += 1
                diff = (arm_log - pl_log).sum(dim=0)
                diff_sum = diff if diff_sum is None else (diff_sum + diff)
                total_pos += int(pl_log.shape[0])
        assert diff_sum is not None and total_pos > 0
        mean_diff = (diff_sum / float(total_pos)).cpu().tolist()
        cands = [
            (-val, tid, val)
            for tid, val in enumerate(mean_diff)
            if val > 0.0 and tid not in excluded_d6_ids
        ]
        cands.sort(key=lambda x: (x[0], x[1]))
        chosen = cands[:top_k_d6]
        return [
            {"id": tid, "token": tokenizer.decode([tid]), "mean_dlogit": round(float(val), 6)}
            for _, tid, val in chosen
        ]

    d6_own_top50: dict[str, list[dict[str, Any]]] = {
        "WSP_Pirate_Contrastive@d*_con": _compute_own_top_k("WSP_Pirate_Contrastive@d*_con"),
        "WSP_Pirate_Lexicon@d*_lex": _compute_own_top_k("WSP_Pirate_Lexicon@d*_lex"),
        "Rand_Pirate_s0@d*_con": _compute_own_top_k("Rand_Pirate_s0@d*_con"),
    }
    demo_prompt_map = {p["id"]: p for p in pillar_a_prompts}
    demo_prompts_12 = [demo_prompt_map[pid] for pid in demo_showcase_ids]

    def _make_ext_ban(extra_rows: list[dict[str, Any]]) -> CommonBanLogitsProcessor:
        extra_ids = [int(r["id"]) for r in extra_rows]
        return CommonBanLogitsProcessor(ban_single_ids + extra_ids, ban_multi_seqs, eos_token_ids=set(eos_ids))

    plain_common_by_id = {r["id"]: r for r in pillar_a_receipt["common_pirate_ban"]["Plain"]}
    d6_receipt: dict[str, Any] = {
        "demo_ids": demo_showcase_ids,
        "own_top50": d6_own_top50,
        "conditions": {
            "Plain@common": [plain_common_by_id[pid] for pid in demo_showcase_ids],
            "Con@own50+common": _run_arm_generations(
                "WSP_Pirate_Contrastive@d*_con", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["WSP_Pirate_Contrastive@d*_con"]),
            ),
            "Con@donor_rand0+common": _run_arm_generations(
                "WSP_Pirate_Contrastive@d*_con", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["Rand_Pirate_s0@d*_con"]),
            ),
            "Lex@own50+common": _run_arm_generations(
                "WSP_Pirate_Lexicon@d*_lex", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["WSP_Pirate_Lexicon@d*_lex"]),
            ),
            "Rand0@own50+common": _run_arm_generations(
                "Rand_Pirate_s0@d*_con", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["Rand_Pirate_s0@d*_con"]),
            ),
        },
    }

    # D1 Part 213 Copula (12 pairs x 5 arms, B=1 use_cache=False, assistant prefill in ChatML)
    d1_arms = [
        "Plain",
        "Sys_Pirate",
        "WSP_Pirate_Contrastive@d*_con",
        "WSP_Pirate_Lexicon@d*_lex",
        "Rand_Pirate_s0@d*_con",
    ]
    d1_receipt: dict[str, Any] = {"pirate_seed_ids": valid_seed_ids, "pirate_seed_tokens": pirate_seed_tokens, "arms": {}}
    for arm_label in d1_arms:
        sys_p, edits = arm_registry[arm_label]
        base_msgs = ([{"role": "system", "content": sys_p}] if sys_p is not None else []) + [
            {"role": "user", "content": "Complete the following sentence in one short phrase."}
        ]
        chat_prefix = format_chat_messages(tokenizer, base_msgs, add_generation_prompt=True)

        def _eval_d1() -> list[dict[str, Any]]:
            rows = []
            for cp in copula_pairs:
                ids_self = tokenizer.encode(chat_prefix + cp["self"], add_special_tokens=False)
                ids_strg = tokenizer.encode(chat_prefix + cp["stranger"], add_special_tokens=False)
                log_self = F.log_softmax(forward_next_logits_batch(model, [ids_self], pad_id, device)[0], dim=-1)
                log_strg = F.log_softmax(forward_next_logits_batch(model, [ids_strg], pad_id, device)[0], dim=-1)
                counts["teacher_forced_b1_fwds"] += 2
                lp_s = float(log_self[valid_seed_ids].mean().item())
                lp_t = float(log_strg[valid_seed_ids].mean().item())
                rows.append({
                    "id": cp["id"],
                    "lp_self": round(lp_s, 6),
                    "lp_stranger": round(lp_t, 6),
                    "self_minus_stranger": round(lp_s - lp_t, 6),
                })
            return rows

        if edits is None:
            d1_receipt["arms"][arm_label] = _eval_d1()
        else:
            with master.apply_edit(model, edits):
                d1_receipt["arms"][arm_label] = _eval_d1()

    _write_checkpoint(checkpoint_dir, "step_e2_d5_d6_d1", {"D5": d5_receipt, "D6": d6_receipt, "D1": d1_receipt})

    # ------------------------------------------------------------------
    # Step F: Pillar B (NOMD_24, 96 tok, ban=none)
    # ------------------------------------------------------------------
    pillar_b_arms: list[str] = [
        "Plain",
        "Sys_NoMD",
        "WSP_NoMD_Contrastive@d*_nomd_con",
        "WSP_NoMD_Lexicon@d*_nomd_lex",
        "Rand_NoMD_s0@d*_nomd_con",
        "Rand_NoMD_s1@d*_nomd_con",
        "Rand_NoMD_s2@d*_nomd_con",
    ]
    if d_nomd_con != d_nomd_lex:
        pillar_b_arms.extend([
            "Rand_NoMD_s0@d*_nomd_lex",
            "Rand_NoMD_s1@d*_nomd_lex",
            "Rand_NoMD_s2@d*_nomd_lex",
        ])

    pillar_b_receipt: dict[str, list[dict[str, Any]]] = {}
    for arm_label in pillar_b_arms:
        raw_rows = _run_arm_generations(arm_label, nomd_prompts, tok_budget["pillar_b"], ban_proc=None)
        for r in raw_rows:
            r["construct_types"] = c3_scoring.construct_types(r["text"])
            r["has_markdown"] = c3_scoring.has_markdown(r["text"])
            r["markdown_hits"] = c3_scoring.markdown_hits(r["text"])
        pillar_b_receipt[arm_label] = raw_rows

    _write_checkpoint(checkpoint_dir, "step_f_pillar_b", pillar_b_receipt)

    # ------------------------------------------------------------------
    # Step G: Shared Capability Suite (9 + conditional d_common) & D2 Sibling Bleed
    # ------------------------------------------------------------------
    cap_arms = list(c3_constants["capability_suite_arms_9"])
    if d_con != d_lex:
        if d_common != d_con:
            cap_arms.append("WSP_Pirate_Contrastive@d_common")
        if d_common != d_lex:
            cap_arms.append("WSP_Pirate_Lexicon@d_common")

    # Neutral plain reference for B=1 off_KL (first 48 generated tokens) & greedy token match
    neu_plain_texts = [_fmt_plain(p["text"]) for p in neutral_prompts]
    neu_plain_id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in neu_plain_texts]
    neu_plain_gens = _gen(neu_plain_texts, tok_budget["neutral"], gen_batch_size)
    neu_plain_tf_logits: list[torch.Tensor] = []
    for p_ids, g in zip(neu_plain_id_lists, neu_plain_gens):
        neu_plain_tf_logits.append(teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"][:48], device))
        counts["teacher_forced_b1_fwds"] += 1

    capability_receipt: dict[str, Any] = {}
    for arm_label in cap_arms:
        sys_p, edits = arm_registry[arm_label]

        def _eval_cap() -> dict[str, Any]:
            math_rows = _run_arm_generations_unlocked(arm_label, math_prompts, tok_budget["math"])
            code_rows = _run_arm_generations_unlocked(arm_label, code_prompts, tok_budget["code"])
            ref_rows = _run_arm_generations_unlocked(arm_label, refusal_prompts, tok_budget["refusal"])
            if arm_label == "Plain":
                neu_gens = neu_plain_gens
                neu_kls = [
                    {"id": p["id"], "off_kl": 0.0, "max_single_pos_kl": 0.0, "greedy_exact_tok_match_rate": 1.0}
                    for p in neutral_prompts
                ]
            else:
                neu_texts = [_fmt_sys(sys_p, p["text"]) if sys_p is not None else _fmt_plain(p["text"]) for p in neutral_prompts]
                neu_gens = _gen(neu_texts, tok_budget["neutral"], gen_batch_size)
                neu_kls = []
                for p, p_ids_plain, g_plain, g_arm, pl_log in zip(
                    neutral_prompts, neu_plain_id_lists, neu_plain_gens, neu_gens, neu_plain_tf_logits
                ):
                    p_ids_arm = tokenizer.encode(_fmt_sys(sys_p, p["text"]), add_special_tokens=False) if sys_p is not None else p_ids_plain
                    arm_log = teacher_forced_gen_logits_b1(model, p_ids_arm, g_plain["gen_ids"][:48], device)
                    counts["teacher_forced_b1_fwds"] += 1
                    m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
                    if edits is not None:
                        _note_liveness(f"cap_{arm_label}", max_dl)
                    n_cmp = max(len(g_plain["gen_ids"]), len(g_arm["gen_ids"]), 1)
                    n_match = sum(1 for a, b in zip(g_plain["gen_ids"], g_arm["gen_ids"]) if a == b)
                    neu_kls.append({
                        "id": p["id"],
                        "off_kl": round(m_kl, 6),
                        "max_single_pos_kl": round(max_p_kl, 6),
                        "greedy_exact_tok_match_rate": round(n_match / n_cmp, 6),
                    })
            neu_rows = [
                {
                    "id": p["id"],
                    "text": g["text"],
                    "n_tokens": g["n_tokens"],
                    "is_broken": c3_scoring.is_broken(g["text"]),
                    "off_kl": kl_r["off_kl"],
                    "max_single_pos_kl": kl_r["max_single_pos_kl"],
                    "greedy_exact_tok_match_rate": kl_r["greedy_exact_tok_match_rate"],
                }
                for p, g, kl_r in zip(neutral_prompts, neu_gens, neu_kls)
            ]
            return {
                "math_20": math_rows,
                "code_12": code_rows,
                "refusal_8": ref_rows,
                "neutral_chat_16": neu_rows,
            }

        def _run_arm_generations_unlocked(albl: str, plist: list[dict[str, Any]], mtok: int) -> list[dict[str, Any]]:
            sp, _ = arm_registry[albl]
            ft = [_fmt_sys(sp, p["text"]) if sp is not None else _fmt_plain(p["text"]) for p in plist]
            gs = _gen(ft, mtok, gen_batch_size)
            return [
                {"id": p["id"], "text": g["text"], "n_tokens": g["n_tokens"], "is_broken": c3_scoring.is_broken(g["text"])}
                for p, g in zip(plist, gs)
            ]

        if edits is None:
            capability_receipt[arm_label] = _eval_cap()
        else:
            with master.apply_edit(model, edits):
                capability_receipt[arm_label] = _eval_cap()

    # Alias d_common capability arms if equal to d*
    if "WSP_Pirate_Contrastive@d_common" not in capability_receipt:
        capability_receipt["WSP_Pirate_Contrastive@d_common"] = capability_receipt["WSP_Pirate_Contrastive@d*_con"]
    if "WSP_Pirate_Lexicon@d_common" not in capability_receipt:
        capability_receipt["WSP_Pirate_Lexicon@d_common"] = capability_receipt["WSP_Pirate_Lexicon@d*_lex"]

    # D2 Sibling Bleed
    pirate_on_nomd = _run_arm_generations("WSP_Pirate_Contrastive@d*_con", nomd_prompts, tok_budget["pillar_b"])
    for r in pirate_on_nomd:
        r["construct_types"] = c3_scoring.construct_types(r["text"])
        r["has_markdown"] = c3_scoring.has_markdown(r["text"])
        r["markdown_hits"] = c3_scoring.markdown_hits(r["text"])
    nomd_on_identity = _run_arm_generations("WSP_NoMD_Contrastive@d*_nomd_con", identity_prompts, tok_budget["pillar_a"])

    sibling_bleed_receipt = {
        "WSP_Pirate_Contrastive@d*_con_on_NOMD_24": pirate_on_nomd,
        "WSP_NoMD_Contrastive@d*_nomd_con_on_IDENTITY_20": nomd_on_identity,
    }

    master.check_hooks(model)
    elapsed_sec = round(time.time() - t0, 2)

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "meta": {
            "model": MODEL_ID,
            "revision": MODEL_REV,
            "smoke": bool(smoke),
            "device": device_name,
            "elapsed_sec": elapsed_sec,
            "target_layers": layers,
            "requested_gen_batch_size": requested_batch_size,
            "effective_gen_batch_size": gen_batch_size,
            "b8_parity_max_abs_dlogit": None if b8_parity_max_dlogit is None else round(b8_parity_max_dlogit, 6),
            "b8_parity_passed": b8_parity_passed,
            "active_hooks_max_seen": master.max_hooks_seen,
            "restore_exact_all": master.restore_exact_all,
            "restore_checks": master.restore_checks,
            "weight_checkpoints_saved": 0,
            "counts": counts,
            "provenance_hashes": provenance_hashes,
        },
        "preflights": {
            "sham": sham_info,
            "D4_L24_mask_mismask": d4_results,
            "surface_stem_audit_positive_check": stem_preflight,
            "active_arm_max_abs_dlogit": {k: round(v, 6) for k, v in sorted(arm_liveness.items())},
        },
        "calibration": {
            "n_prompts": len(calib_prompts),
            "layers": {str(L): calib_layer_stats[L] for L in layers},
            "direction_quality": direction_quality,
        },
        "doses": {
            "d_star_pirate_con": d_con,
            "d_star_pirate_lex": d_lex,
            "d_common": d_common,
            "d_star_nomd_con": d_nomd_con,
            "d_star_nomd_lex": d_nomd_lex,
            "commit_gate_fallback": commit_gate_fallback,
            "d_common_retention": d_common_retention,
        },
        "commit_gate": commit_gate_receipt,
        "rand_reserve": rand_reserve_receipt,
        "pillar_a": pillar_a_receipt,
        "pillar_b": pillar_b_receipt,
        "D5_multiturn": d5_receipt,
        "D6_adaptive_donor": d6_receipt,
        "D1_copula": d1_receipt,
        "capability_suite": capability_receipt,
        "D2_sibling_bleed": sibling_bleed_receipt,
    }
    return receipt


def emit_paced_b64_gz(gz_bytes: bytes, chunk_bytes: int = 16384, sleep_sec: float = 0.10) -> None:
    """Emit gzip receipt as paced base64 (~160 KB/s) to stay safely under Jupyter's 1 MB/s iopub limit (LES-002)."""
    b64 = base64.b64encode(gz_bytes).decode("ascii")
    print("BEGIN_C3_RECEIPT_GZ_B64", flush=True)
    for i in range(0, len(b64), chunk_bytes):
        print(b64[i : i + chunk_bytes], flush=True)
        if i + chunk_bytes < len(b64):
            time.sleep(sleep_sec)
    print("END_C3_RECEIPT_GZ_B64", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage C3 (WSP v1) on OLMo 3 7B with zero-hook literal weight edits.")
    ap.add_argument("--design", type=Path, default=HERE / "OLMO3-WSP-C3-DESIGN.md")
    ap.add_argument("--c3-constants", type=Path, default=HERE / "c3_prereg_constants.json")
    ap.add_argument("--bank", type=Path, default=HERE / "c3_vector_bank.safetensors")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (hard-refused by score_olmo3_c3.py).")
    ap.add_argument("--emit-b64-stdout", action="store_true", help="Emit paced gzip+b64 receipt on stdout for Colab.")
    args = ap.parse_args()

    hashes = verify_preflight_provenance(args.design, args.c3_constants, args.bank, args.scoring)
    c3_constants = json.loads(args.c3_constants.read_text(encoding="utf-8"))
    bank = safetensors_load_file(str(args.bank))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir is not None else args.out.parent / f"{args.out.stem}.ckpt"

    with torch.no_grad():
        receipt = run_stage_c3_experiment(
            model, tok, c3_constants, bank, hashes,
            smoke=args.smoke,
            device_name=dev_name,
            checkpoint_dir=ckpt_dir,
            requested_batch_size=args.batch_size,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(receipt, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    gz_bytes = gzip.compress(raw, compresslevel=6)
    if args.out.suffix == ".gz":
        args.out.write_bytes(gz_bytes)
    else:
        args.out.write_bytes(raw)
    print(
        f"Wrote {args.out} (raw={len(raw)} B, gz={len(gz_bytes)} B, sha256={sha256_file(args.out)}, "
        f"gens={receipt['meta']['counts']['generations']}, tok={receipt['meta']['counts']['generated_tokens']}, "
        f"elapsed={receipt['meta']['elapsed_sec']}s)",
        flush=True,
    )
    if args.emit_b64_stdout:
        emit_paced_b64_gz(gz_bytes)


if __name__ == "__main__":
    main()
