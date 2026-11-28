#!/usr/bin/env python3
"""run_olmo3_c3b.py — Stage C3b (WSP v2) Distributed 12-Layer Zero-Hook Literal pre_norm GPU Runner.

Implements the frozen registered Stage C3b protocol (`OLMO3-WSP-C3B-DESIGN.md`, public main `28fc5dfb`):
  - Fail-closed C3b.0.4 SHA-256 provenance verification of all 4 committed artifacts before model load.
  - Exact B = 1 greedy decode throughout (`effective_gen_batch_size == 1`, zero B = 8 batching drift).
  - Zero runtime PyTorch hooks during every eval pass (hooks ONLY in one-time `calib_16` capture,
    removed in `finally`, verified via `count_active_hooks(model) == 0` before any evaluation).
  - `fp32`-master literal weight edits across K = 12 contiguous layers (`L16..L27`, `mlp.down_proj.weight`)
    with bitwise `torch.equal` restore tracking BOTH `layers_compiled_total` and `restore_checks`
    (`restore_checks == layers_compiled_total >= 279`).
  - Policy 11-07 honored strictly: ZERO edited model weights or checkpoints ever saved to disk.
  - Live `D4_L24_mask_mismask` on C3b's own `WSP_Pirate_12L` contrastive `L24` direction using 12 prompts
    100% disjoint from `calib_16` (`res_00..res_07`, `chat_00..chat_03`, 48 exact forwards).
  - 7-rung reserve commit gate `[1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]` PMN + optional `D_above_cap` slice,
    out-of-sample `D6_heldout` (top-50 computed on disjoint `RESERVE_PROMPTS_8`), zero-GPU `D_align` &
    `em_dash_per_1k_words`, and exploratory `D_key_ablation` (`WSP_NoMD_12L_kWA` + `WSP_Pirate_12L_kSparse32`).
  - Per-block atomic partial checkpoints (`partial_checkpoint=True`) and paced gzip+base64 stdout export.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import gzip
import hashlib
import json
import math
import re
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
# Frozen Registered Stage C3b Provenance Pins (Draft v4 / public main 28fc5dfb)
# ---------------------------------------------------------------------------
MODEL_ID = "allenai/Olmo-3-7B-Instruct"
MODEL_REV = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"
TARGET_LAYERS: list[int] = list(range(16, 28))  # K = 12 contiguous layers L16..L27
DOSE_LADDER: list[float] = [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 9.6]
SHAM_ALPHA: float = 0.05
KWA_BETA_LADDER: list[float] = [0.25, 0.50, 1.00]
SPARSE_KEY_K: int = 32

EXPECTED_C3B_DESIGN_SHA256 = "a9ca200dccb19ad3efb3b0cf358c546dfffb10904f673f64667f9b580bdc7c75"
EXPECTED_C3B_CONSTANTS_SHA256 = "3bb980e579cef2326d5d2adc3c458cca8ba4140da5bb0e7db2f196e3484ca9db"
EXPECTED_C3B_BANK_SHA256 = "a32662a7b3671a4613b7b412e78ff4b14a7fcd41e385fb9d43da31bd1fea2656"
EXPECTED_C3_SCORING_SHA256 = "b2fdfdf480f28554117765caa49126e4567481e6d34d2917ef1287cc0675040f"

RECEIPT_SCHEMA_VERSION = "olmo3-stage-c3b-receipt-v1"

_EM_DASH_RE = re.compile(r"\u2014|(?<!-)--(?!-)")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


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
    """Fail-closed C3b.0.4 SHA-256 verification of all 4 committed Stage C3b files."""
    checks = [
        ("c3b_design_sha256", design_path, EXPECTED_C3B_DESIGN_SHA256),
        ("c3b_constants_sha256", constants_path, EXPECTED_C3B_CONSTANTS_SHA256),
        ("c3b_bank_sha256", bank_path, EXPECTED_C3B_BANK_SHA256),
        ("c3_scoring_sha256", scoring_path, EXPECTED_C3_SCORING_SHA256),
    ]
    hashes: dict[str, str] = {}
    for key, p, expected in checks:
        if not p.exists():
            raise FileNotFoundError(f"C3b.0.4 preflight missing required file: {p}")
        actual = sha256_file(p)
        if actual != expected:
            raise ValueError(
                f"C3b.0.4 preflight SHA-256 mismatch for {p.name}: got {actual}, expected {expected}"
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


def em_dash_stats(text: str) -> dict[str, Any]:
    """Count em-dashes (`—` or isolated `--`, excluding `---` markdown HR) and rate per 1k words."""
    em_cnt = len(_EM_DASH_RE.findall(text))
    words = len(_WORD_RE.findall(text))
    rate = (1000.0 * em_cnt / words) if words > 0 else 0.0
    return {
        "em_dash_count": em_cnt,
        "word_count": words,
        "em_dash_per_1k_words": round(rate, 4),
    }


# ---------------------------------------------------------------------------
# Zero-Hook Literal fp32-Master Weight Editor with Exact Bitwise bf16 Restore
# ---------------------------------------------------------------------------
class WeightMaster:
    """Holds clean fp32 masters and bf16 snapshots of target `mlp.down_proj.weight` tensors.

    Tracks `layers_compiled_total` alongside `restore_checks` (unit = 1 bitwise `torch.equal`
    verification per edited layer per weight-edit compile session) so C3b.0.1 can enforce
    `restore_exact_all == True AND restore_checks == layers_compiled_total >= 279`.
    """

    def __init__(self, model: torch.nn.Module, layers: list[int]):
        self.layers = list(layers)
        self.clean_bf16: dict[int, torch.Tensor] = {}
        self.clean_fp32: dict[int, torch.Tensor] = {}
        self.restore_exact_all: bool = True
        self.layers_compiled_total: int = 0
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
            raise RuntimeError(f"C3b.0.1 VIOLATION: expected 0 active PyTorch hooks, found {n}")
        return n

    def _restore_layers(self, model: torch.nn.Module, touched: list[int]) -> None:
        restore_failures: list[int] = []
        for L in touched:
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
            raise RuntimeError(f"C3b.0.1 VIOLATION: bitwise weight restore failed at layers {restore_failures}")
        if not in_exc:
            self.check_hooks(model)

    @contextlib.contextmanager
    def apply_edit(
        self,
        model: torch.nn.Module,
        edits_by_layer: dict[int, tuple[float, torch.Tensor, torch.Tensor]],
    ):
        """Apply literal rank-1 weight edit `alpha * outer(v_pre_hat, k_hat)` from fp32 master -> bf16."""
        self.check_hooks(model)
        metrics: dict[str, dict[str, float]] = {}
        touched_layers = sorted(edits_by_layer.keys())
        actually_touched: list[int] = []
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
                actually_touched.append(L)
                self.layers_compiled_total += 1
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
            self._restore_layers(model, actually_touched)

    @contextlib.contextmanager
    def apply_orth_edit(
        self,
        model: torch.nn.Module,
        orth_by_layer: dict[int, tuple[float, torch.Tensor]],
    ):
        """Apply literal rank-1 weight-aligned orthogonalization `dW_L = -beta * outer(v_pre_hat, v_pre_hat^T W_clean_L)`.

        Exact rank-1 ablation (zeroing the +-v_pre_hat axis of `W_down` at `beta = 1.0`).
        Increments `layers_compiled_total` and bitwise-restores via `torch.equal` identically to `apply_edit`.
        """
        self.check_hooks(model)
        metrics: dict[str, dict[str, float]] = {}
        touched_layers = sorted(orth_by_layer.keys())
        actually_touched: list[int] = []
        try:
            for L in touched_layers:
                beta, v_pre_hat = orth_by_layer[L]
                beta_f = float(beta)
                v_f = v_pre_hat.detach().cpu().float()
                v_f = v_f / v_f.norm().clamp_min(1e-12)
                proj_row = v_f @ self.clean_fp32[L]  # [d_ff]
                proj_norm = float(proj_row.norm().item())
                k_wa_hat = proj_row / max(proj_norm, 1e-12)
                alpha_eff = -beta_f * proj_norm
                dW_intended = alpha_eff * torch.outer(v_f, k_wa_hat)
                W_edited_bf16 = (self.clean_fp32[L] + dW_intended).to(torch.bfloat16)
                dW_stored = W_edited_bf16.float() - self.clean_bf16[L].float()

                intended_frob = float(dW_intended.norm().item())
                stored_frob = float(dW_stored.norm().item())
                stored_frob_ret = (stored_frob / intended_frob) if intended_frob > 1e-12 else 0.0
                rank1_proj = float(torch.dot(v_f, dW_stored @ k_wa_hat).item())
                stored_rank1_proj_ret = (rank1_proj / alpha_eff) if abs(alpha_eff) > 1e-12 else 0.0
                frac_changed = float((W_edited_bf16 != self.clean_bf16[L]).float().mean().item())
                post_proj_norm = float((v_f @ W_edited_bf16.float()).norm().item())
                orth_residual_ratio = post_proj_norm / max(proj_norm, 1e-12)

                param = model.model.layers[L].mlp.down_proj.weight
                actually_touched.append(L)
                self.layers_compiled_total += 1
                param.data.copy_(W_edited_bf16.to(device=param.device, dtype=param.dtype))
                metrics[str(L)] = {
                    "beta": round(beta_f, 6),
                    "alpha_eff": round(alpha_eff, 8),
                    "proj_row_norm": round(proj_norm, 6),
                    "intended_frob": round(intended_frob, 8),
                    "stored_frob": round(stored_frob, 8),
                    "stored_frob_ret": round(stored_frob_ret, 6),
                    "stored_rank1_proj_ret": round(stored_rank1_proj_ret, 6),
                    "orth_residual_ratio": round(orth_residual_ratio, 6),
                    "frac_changed": round(frac_changed, 6),
                }
            self.check_hooks(model)
            yield metrics
        finally:
            self._restore_layers(model, actually_touched)


# ---------------------------------------------------------------------------
# Exact Single-Token + Multi-Token Decode Ban Processor (Exact B = 1)
# ---------------------------------------------------------------------------
class CommonBanLogitsProcessor:
    """Enforces frozen single-token bans AND multi-token `bad_words_ids` sequences."""

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
# ChatML Formatting, Exact B = 1 Greedy Decode, and B = 1 Teacher-Forced KL
# ---------------------------------------------------------------------------
def format_chat_messages(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
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
def forward_next_logits_b1(
    model: torch.nn.Module,
    token_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Exact B = 1, `use_cache=False`, float32 next-token logits `[V]` at the last token."""
    inp = torch.tensor([token_ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False)
    return out.logits[0, -1, :].float()


@torch.no_grad()
def greedy_decode_b1(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts_text: list[str],
    max_new_tokens: int,
    eos_ids: list[int],
    ban_processor: CommonBanLogitsProcessor | None = None,
    master: WeightMaster | None = None,
) -> list[dict[str, Any]]:
    """Exact B = 1 deterministic greedy decode (`do_sample=False`, `use_cache=True`, zero padding)."""
    if master is not None:
        master.check_hooks(model)
    device = next(model.parameters()).device
    eos_set = set(eos_ids)
    results: list[dict[str, Any]] = []

    for ptxt in prompts_text:
        prompt_ids = tokenizer.encode(ptxt, add_special_tokens=False)
        history = list(prompt_ids)
        gen_ids: list[int] = []
        past_key_values = None
        cur_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        cur_pos = torch.arange(len(prompt_ids), dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            out = model(
                input_ids=cur_input,
                position_ids=cur_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = out.past_key_values
            scores = out.logits[:, -1, :].float()
            if ban_processor is not None:
                scores = ban_processor.apply(scores, [history])
            tid = int(torch.argmax(scores, dim=-1).item())
            if tid in eos_set:
                break
            gen_ids.append(tid)
            history.append(tid)
            cur_input = torch.tensor([[tid]], dtype=torch.long, device=device)
            cur_pos = cur_pos[:, -1:] + 1

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        results.append({"text": text, "gen_ids": gen_ids, "n_tokens": len(gen_ids)})
    return results


@torch.no_grad()
def teacher_forced_gen_logits_b1(
    model: torch.nn.Module,
    prompt_ids: list[int],
    gen_ids: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Exact B = 1, `use_cache=False`, float32 logits `[T, V]` predicting each of the `T` tokens in `gen_ids`."""
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

    safe_g = torch.where(
        g.abs() >= 1e-8, g, torch.where(g >= 0, torch.full_like(g, 1e-8), torch.full_like(g, -1e-8))
    )
    raw_unmasked = v_p / safe_g
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


def compute_unmasked_prenorm_direction(
    v_post_unit: torch.Tensor,
    gamma_ff: torch.Tensor,
) -> torch.Tensor:
    v_p = v_post_unit.detach().cpu().float()
    v_p = v_p / v_p.norm().clamp_min(1e-12)
    g = gamma_ff.detach().cpu().float()
    safe_g = torch.where(
        g.abs() >= 1e-8, g, torch.where(g >= 0, torch.full_like(g, 1e-8), torch.full_like(g, -1e-8))
    )
    raw = v_p / safe_g
    return raw / raw.norm().clamp_min(1e-12)


def compute_sparse_topk_key(
    m_mat: torch.Tensor,
    k_support: int = SPARSE_KEY_K,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retain only top-k coordinates of |mean(m)|, unit-normalize, and compute Plain calib projections."""
    m_bar = m_mat.mean(dim=0)
    k_eff = min(int(k_support), int(m_bar.numel()))
    topk_idx = torch.topk(m_bar.abs(), k_eff, largest=True).indices.sort().values
    mask = torch.zeros_like(m_bar)
    mask[topk_idx] = 1.0
    k_sp = m_bar * mask
    k_sp_hat = k_sp / k_sp.norm().clamp_min(1e-12)
    projs = m_mat @ k_sp_hat
    c_bar = float(projs.mean().item())
    c_std = float(projs.std(unbiased=False).item())
    cv = c_std / max(abs(c_bar), 1e-12)
    max_deliv = float(projs.max().item()) / max(abs(c_bar), 1e-12)
    return k_sp_hat, {
        "support_k": k_eff,
        "c_bar": round(c_bar, 6),
        "c_std": round(c_std, 6),
        "cv": round(cv, 6),
        "max_delivery_ratio": round(max_deliv, 6),
        "topk_indices": [int(i) for i in topk_idx.tolist()],
    }


def compute_d_align_matrix(layers: list[int], v_post_by_layer: dict[int, torch.Tensor]) -> dict[str, Any]:
    """Compute zero-GPU KxK pairwise cosine matrix of v_post[L] across target layers."""
    vecs = [v_post_by_layer[L].detach().cpu().float() for L in layers]
    vecs = [v / v.norm().clamp_min(1e-12) for v in vecs]
    k = len(layers)
    mat: list[list[float]] = []
    off_vals: list[float] = []
    adj_vals: list[float] = []
    for i in range(k):
        row = []
        for j in range(k):
            c = float(torch.dot(vecs[i], vecs[j]).item())
            row.append(round(c, 6))
            if i != j:
                off_vals.append(c)
            if j == i + 1:
                adj_vals.append(c)
        mat.append(row)
    return {
        "layers": list(layers),
        "cosine_matrix": mat,
        "mean_off_diag_cos": round(sum(off_vals) / max(len(off_vals), 1), 6),
        "min_off_diag_cos": round(min(off_vals), 6) if off_vals else 1.0,
        "min_adjacent_cos": round(min(adj_vals), 6) if adj_vals else 1.0,
        "mean_adjacent_cos": round(sum(adj_vals) / max(len(adj_vals), 1), 6) if adj_vals else 1.0,
    }


def build_layer_edits(
    layers: list[int],
    dose_total: float,
    v_pre_by_layer: dict[int, torch.Tensor],
    k_hat_by_layer: dict[int, torch.Tensor],
    calib_layer_stats: dict[int, dict[str, Any]],
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
        print(f"[WARN] stage-c3b checkpoint '{name}' write failed ({exc!r}); continuing", file=sys.stderr)


# ---------------------------------------------------------------------------
# Core Stage C3b Experiment Engine (Model-Agnostic: Real OLMo-3 L4 & Toy CPU)
# ---------------------------------------------------------------------------
def run_stage_c3b_experiment(
    model: torch.nn.Module,
    tokenizer: Any,
    c3b_constants: dict[str, Any],
    bank: dict[str, torch.Tensor],
    provenance_hashes: dict[str, str],
    *,
    smoke: bool = False,
    device_name: str = "unknown",
    checkpoint_dir: Path | None = None,
    target_layers: list[int] | None = None,
    max_new_tokens_override: dict[str, int] | None = None,
    log_progress: bool = True,
) -> dict[str, Any]:
    """Execute full Stage C3b (WSP v2) or smoke subset under exact B = 1 greedy decode."""
    t0 = time.time()

    def _log(msg: str) -> None:
        if log_progress:
            print(f"[C3b {time.time() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)

    layers = list(target_layers if target_layers is not None else c3b_constants["edit_spec"]["layers"])
    dose_ladder = [float(d) for d in c3b_constants["edit_spec"]["dose_ladder_total_pmn_frac"]]
    sham_alpha = float(c3b_constants["edit_spec"]["sham_alpha"])
    cg_cfg = c3b_constants["edit_spec"]["commit_gate"]
    cg_med_kl_max = float(cg_cfg["reserve_median_off_kl_max"])
    cg_max_kl_max = float(cg_cfg["reserve_max_off_kl_max"])
    cg_max_pos_max = float(cg_cfg["reserve_max_single_pos_kl_max"])
    cg_broken_max = int(cg_cfg["reserve_broken_max"])
    cg_min_proj = float(cg_cfg["rung_min_rank1_proj_ret"])
    cg_min_frob = float(cg_cfg["rung_min_frob_ret"])

    sys_pirate = str(c3b_constants["system_prompts"]["pirate"])
    sys_nomd = str(c3b_constants["system_prompts"]["nomd"])
    surface_regex = str(c3b_constants["lexicons"]["common_pirate_ban"]["surface_stem_regex"])
    strict_ext_ban_regex = str(c3b_constants["lexicons"]["strict_pirate_extended_ban_regex"])
    ban_single_ids = [int(x["id"]) for x in c3b_constants["lexicons"]["common_pirate_ban"]["single_tokens"]]
    ban_multi_seqs = [
        [int(t) for t in x["ids"]] for x in c3b_constants["lexicons"]["common_pirate_ban"]["multi_token_sequences"]
    ]
    pirate_seed_ids = [int(x["id"]) for x in c3b_constants["lexicons"]["pirate_seeds"]]
    pirate_seed_tokens = [str(x["token"]) for x in c3b_constants["lexicons"]["pirate_seeds"]]
    top_k_d6 = int(c3b_constants["lexicons"]["secondary_d6_heldout_top_k_bias"])

    device = next(model.parameters()).device
    eos_ids, _pad_id = get_eos_and_pad_ids(model, tokenizer)
    special_ids = get_special_and_added_token_ids(tokenizer)
    common_ban_proc = CommonBanLogitsProcessor(ban_single_ids, ban_multi_seqs, eos_token_ids=set(eos_ids))

    tok_budget = {
        "calib": 32,
        "reserve": 48,
        "pillar_a": 64,
        "pillar_b": 96,
        "multiturn": 48,  # Pinned in OLMO3-WSP-C3B-DESIGN.md §5 & §6
        "d6": 64,
        "math": 64,
        "code": 96,
        "refusal": 64,
        "neutral": 64,
        "d_above_cap_pirate": 64,
        "d_above_cap_nomd": 96,
    }
    if max_new_tokens_override:
        tok_budget.update(max_new_tokens_override)

    bat = c3b_constants["batteries"]
    calib_prompts = list(bat["calib_16"])
    reserve_prompts = list(bat["reserve_8"])
    identity_prompts = list(bat["identity_20"])
    implied_prompts = list(bat["implied_20"])
    pillar_a_prompts = identity_prompts + implied_prompts
    nomd_prompts = list(bat["nomd_32"])
    multiturn_prompts = list(bat["multiturn_12"])
    copula_pairs = list(bat["copula_pairs_12"])
    math_prompts = list(bat["math_20"])
    code_prompts = list(bat["code_12"])
    refusal_prompts = list(bat["refusal_8"])
    neutral_prompts = list(bat["neutral_chat_16"])
    demo_showcase_ids = list(bat["demo_showcase_ids_12"])
    d4_prompt_ids_spec = list(c3b_constants["d4_spec"]["prompt_ids"])
    d_above_cap_pirate_ids = list(c3b_constants["d_above_cap_spec"]["pirate_slice"]["prompt_ids"])
    d_above_cap_nomd_ids = list(c3b_constants["d_above_cap_spec"]["nomd_slice"]["prompt_ids"])

    if smoke:
        calib_prompts = calib_prompts[:2]
        reserve_prompts = reserve_prompts[:2]
        identity_prompts = identity_prompts[:2]
        implied_prompts = implied_prompts[:2]
        pillar_a_prompts = identity_prompts + implied_prompts
        nomd_prompts = nomd_prompts[:4]
        multiturn_prompts = multiturn_prompts[:2]
        copula_pairs = copula_pairs[:2]
        math_prompts = math_prompts[:2]
        code_prompts = code_prompts[:2]
        refusal_prompts = refusal_prompts[:2]
        neutral_prompts = neutral_prompts[:2]
        demo_showcase_ids = [p["id"] for p in pillar_a_prompts[:2]]
        d4_prompt_ids_spec = [p["id"] for p in reserve_prompts] + [p["id"] for p in neutral_prompts[:2]]
        d_above_cap_pirate_ids = list(demo_showcase_ids)
        d_above_cap_nomd_ids = [p["id"] for p in nomd_prompts[:2]]
        dose_ladder = [1.2, 3.6]

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

    def _gen_b1(
        texts: list[str],
        max_tok: int,
        ban_proc: CommonBanLogitsProcessor | None = None,
    ) -> list[dict[str, Any]]:
        out = greedy_decode_b1(
            model, tokenizer, texts, max_tok, eos_ids,
            ban_processor=ban_proc, master=master,
        )
        counts["generations"] += len(out)
        counts["generated_tokens"] += sum(r["n_tokens"] for r in out)
        return out

    # ------------------------------------------------------------------
    # Step A: Disjoint `calib_16` Calibration (Hooks ONLY here, removed immediately!)
    # ------------------------------------------------------------------
    _log(f"Step A: calib_16 ({len(calib_prompts)} prompts x 3 conditions, B=1)")
    cal_plain_texts = [_fmt_plain(p["text"]) for p in calib_prompts]
    cal_pirate_texts = [_fmt_sys(sys_pirate, p["text"]) for p in calib_prompts]
    cal_nomd_texts = [_fmt_sys(sys_nomd, p["text"]) for p in calib_prompts]

    cal_plain_gens = _gen_b1(cal_plain_texts, tok_budget["calib"])
    cal_pirate_gens = _gen_b1(cal_pirate_texts, tok_budget["calib"])
    cal_nomd_gens = _gen_b1(cal_nomd_texts, tok_budget["calib"])

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
    k_sp32_by_layer: dict[int, torch.Tensor] = {}
    calib_layer_stats: dict[int, dict[str, Any]] = {}
    calib_sp32_stats: dict[int, dict[str, Any]] = {}
    v_post_pirate_con: dict[int, torch.Tensor] = {}
    v_post_nomd_con: dict[int, torch.Tensor] = {}
    v_pre_pirate_con: dict[int, torch.Tensor] = {}
    v_pre_nomd_con: dict[int, torch.Tensor] = {}
    v_pre_rand_pirate: dict[int, dict[int, torch.Tensor]] = {s: {} for s in range(3)}
    v_pre_rand_nomd: dict[int, dict[int, torch.Tensor]] = {s: {} for s in range(3)}
    direction_quality: dict[str, dict[str, Any]] = {
        "WSP_Pirate_12L": {},
        "WSP_NoMD_12L": {},
    }

    for L in layers:
        m_mat = cap_plain["m"][L]  # [N_tok, d_ff] — strictly Plain calib_16
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

        # Sparse top-32 key for exploratory D_key_ablation (WSP_Pirate_12L_kSparse32)
        k_sp32, sp32_st = compute_sparse_topk_key(m_mat, SPARSE_KEY_K)
        sp32_st["pmn"] = round(pmn_val, 6)
        sp32_st["rms_pre"] = round(rms_pre_val, 6)
        k_sp32_by_layer[L] = k_sp32
        calib_sp32_stats[L] = sp32_st

        # Contrastive target directions on each condition's own calib_16
        dh_pirate = cap_pirate["h_out"][L].mean(dim=0) - cap_plain["h_out"][L].mean(dim=0)
        v_post_p = dh_pirate / dh_pirate.norm().clamp_min(1e-12)
        v_post_pirate_con[L] = v_post_p
        v_pre_p, q_p = compute_prenorm_masked_direction(v_post_p, gamma_by_layer[L], drop_idx_by_layer[L])
        v_pre_pirate_con[L] = v_pre_p
        direction_quality["WSP_Pirate_12L"][str(L)] = q_p

        dh_nomd = cap_nomd["h_out"][L].mean(dim=0) - cap_plain["h_out"][L].mean(dim=0)
        v_post_n = dh_nomd / dh_nomd.norm().clamp_min(1e-12)
        v_post_nomd_con[L] = v_post_n
        v_pre_n, q_n = compute_prenorm_masked_direction(v_post_n, gamma_by_layer[L], drop_idx_by_layer[L])
        v_pre_nomd_con[L] = v_pre_n
        direction_quality["WSP_NoMD_12L"][str(L)] = q_n

        for s in range(3):
            v_pre_rand_pirate[s][L] = bank[f"L{L}_rand_pirate_pre_unit_s{s}"].float()
            v_pre_rand_nomd[s][L] = bank[f"L{L}_rand_nomd_pre_unit_s{s}"].float()

    # Zero-GPU D_align pairwise cosine matrices across L16..L27
    d_align_receipt = {
        "WSP_Pirate_12L": compute_d_align_matrix(layers, v_post_pirate_con),
        "WSP_NoMD_12L": compute_d_align_matrix(layers, v_post_nomd_con),
    }

    # ------------------------------------------------------------------
    # Step B: Preflights — Sham (alpha=0.05 on L16..L27), Live D4 L24 (48 fwds), Stem Preflight
    # ------------------------------------------------------------------
    _log("Step B: sham + live D4 L24 (48 fwds, 100% disjoint from calib_16)")
    res_plain_texts = [_fmt_plain(p["text"]) for p in reserve_prompts]
    res_plain_id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in res_plain_texts]
    neu_plain_texts = [_fmt_plain(p["text"]) for p in neutral_prompts]
    neu_plain_id_lists = [tokenizer.encode(t, add_special_tokens=False) for t in neu_plain_texts]

    prompt_ids_by_id: dict[str, list[int]] = {}
    for p, ids in zip(reserve_prompts, res_plain_id_lists):
        prompt_ids_by_id[p["id"]] = ids
    for p, ids in zip(neutral_prompts, neu_plain_id_lists):
        prompt_ids_by_id[p["id"]] = ids

    # D4 clean logits (12 prompts: res_00..res_07 + chat_00..chat_03 -> also gives res_00 clean logit for sham!)
    d4_prompt_id_lists = [prompt_ids_by_id[pid] for pid in d4_prompt_ids_spec]
    d4_clean_logits = [forward_next_logits_b1(model, ids, device) for ids in d4_prompt_id_lists]
    counts["preflight_fwds"] += len(d4_prompt_id_lists)
    res0_clean_logit = d4_clean_logits[0]

    # C3b.0.1 Sham (alpha=0.05 across all 12 layers L16..L27 on res_00 last prompt token)
    sham_edits = {L: (sham_alpha, v_pre_pirate_con[L], k_hat_by_layer[L]) for L in layers}
    with master.apply_edit(model, sham_edits) as sham_ret:
        sham_logit = forward_next_logits_b1(model, res_plain_id_lists[0], device)
        counts["preflight_fwds"] += 1
    sham_max_abs_dlogit = float((sham_logit - res0_clean_logit).abs().max().item())
    sham_min_proj_ret = min(v["stored_rank1_proj_ret"] for v in sham_ret.values())
    sham_info = {
        "alpha": sham_alpha,
        "max_abs_dlogit": round(sham_max_abs_dlogit, 6),
        "min_stored_rank1_proj_ret": round(sham_min_proj_ret, 6),
        "per_layer": sham_ret,
    }

    # D4_masking_check: live WSP_Pirate_12L contrastive L24 direction, alpha_L24(d=3.6), 48 exact fwds
    l24 = 24 if 24 in layers else layers[min(8, len(layers) - 1)]
    _, d4_alphas_ref = build_layer_edits(
        layers, 3.6, v_pre_pirate_con, k_hat_by_layer, calib_layer_stats, gamma_by_layer
    )
    alpha_d4 = d4_alphas_ref[l24]
    v_l24_masked = v_pre_pirate_con[l24]
    v_l24_unmasked = compute_unmasked_prenorm_direction(v_post_pirate_con[l24], gamma_by_layer[l24])
    d_model = int(gamma_by_layer[l24].shape[0])
    mismask_idx = [(idx + 1) % d_model for idx in drop_idx_by_layer[l24]]
    v_l24_mismasked, d4_mismask_geom = compute_prenorm_masked_direction(
        v_post_pirate_con[l24], gamma_by_layer[l24], mismask_idx
    )

    vocab_n = int(res0_clean_logit.shape[-1])
    valid_seed_ids = [tid for tid in pirate_seed_ids if 0 <= tid < vocab_n]
    if not valid_seed_ids:
        valid_seed_ids = [0]

    # Descriptive 0-fwd C3 L24 lexicon continuity check
    c3_lex_continuity: dict[str, Any] = {}
    if "L24_pirate_lex_post_unit" in bank and int(bank["L24_pirate_lex_post_unit"].shape[0]) == d_model:
        c3_lex_post = bank["L24_pirate_lex_post_unit"].float()
        c3_lex_pre = bank["L24_pirate_lex_pre_unit"].float()
        _, c3_lex_geom = compute_prenorm_masked_direction(c3_lex_post, gamma_by_layer[l24], drop_idx_by_layer[l24])
        c3_lex_continuity = {
            "cos_live_con_vs_c3_lex_post": round(float(torch.dot(v_post_pirate_con[l24], c3_lex_post / c3_lex_post.norm().clamp_min(1e-12)).item()), 6),
            "cos_live_con_vs_c3_lex_pre": round(float(torch.dot(v_l24_masked, c3_lex_pre / c3_lex_pre.norm().clamp_min(1e-12)).item()), 6),
            "c3_lex_geom": c3_lex_geom,
        }

    d4_results: dict[str, Any] = {
        "layer": l24,
        "object_tested": "live_WSP_Pirate_12L_contrastive_L24",
        "prompt_ids": d4_prompt_ids_spec,
        "dose_ref_total": 3.6,
        "alpha_L24": round(alpha_d4, 6),
        "mismask_idx": mismask_idx,
        "mismask_geom": d4_mismask_geom,
        "c3_l24_lex_continuity_0fwd": c3_lex_continuity,
    }
    for cond_name, vec in (
        ("masked", v_l24_masked),
        ("unmasked", v_l24_unmasked),
        ("mismasked", v_l24_mismasked),
    ):
        seed_dlogits: list[float] = []
        l2_dlogits: list[float] = []
        with master.apply_edit(model, {l24: (alpha_d4, vec, k_hat_by_layer[l24])}) as d4_ret:
            for ids, cl in zip(d4_prompt_id_lists, d4_clean_logits):
                el = forward_next_logits_b1(model, ids, device)
                counts["preflight_fwds"] += 1
                diff = el - cl
                seed_dlogits.append(float(diff[valid_seed_ids].mean().item()))
                l2_dlogits.append(float(diff.norm().item()))
        mean_seed = sum(seed_dlogits) / len(seed_dlogits)
        mean_l2 = sum(l2_dlogits) / len(l2_dlogits)
        eff = mean_seed / max(mean_l2, 1e-12)
        d4_results[cond_name] = {
            "mean_pirate_seed_dlogit": round(mean_seed, 6),
            "mean_logit_l2": round(mean_l2, 6),
            "efficiency": round(eff, 6),
            "retention": d4_ret[str(l24)],
        }
    unm_l2 = max(abs(d4_results["unmasked"]["mean_logit_l2"]), 1e-12)
    unm_seed = d4_results["unmasked"]["mean_pirate_seed_dlogit"]
    unm_eff = d4_results["unmasked"]["efficiency"]
    d4_results["masked_over_unmasked_logit_l2_ratio"] = round(d4_results["masked"]["mean_logit_l2"] / unm_l2, 6)
    d4_results["mismasked_over_unmasked_logit_l2_ratio"] = round(d4_results["mismasked"]["mean_logit_l2"] / unm_l2, 6)
    d4_results["masked_over_unmasked_seed_dlogit_ratio"] = (
        round(d4_results["masked"]["mean_pirate_seed_dlogit"] / unm_seed, 6) if abs(unm_seed) > 1e-9 else None
    )
    d4_results["mismasked_over_unmasked_seed_dlogit_ratio"] = (
        round(d4_results["mismasked"]["mean_pirate_seed_dlogit"] / unm_seed, 6) if abs(unm_seed) > 1e-9 else None
    )
    d4_results["masked_over_unmasked_efficiency_ratio"] = (
        round(d4_results["masked"]["efficiency"] / unm_eff, 6) if abs(unm_eff) > 1e-9 else None
    )
    d4_results["mismasked_over_unmasked_efficiency_ratio"] = (
        round(d4_results["mismasked"]["efficiency"] / unm_eff, 6) if abs(unm_eff) > 1e-9 else None
    )

    stem_preflight = {
        "positive_sample_hit": c3_scoring.surface_stem_hit("Ahoy matey, hoist the anchor on this pirate ship!", surface_regex),
        "negative_sample_hit": c3_scoring.surface_stem_hit("Please describe how to organize a daily study schedule.", surface_regex),
    }

    _write_checkpoint(checkpoint_dir, "step_ab_calib_preflights", {
        "calibration": calib_layer_stats,
        "direction_quality": direction_quality,
        "D_align": d_align_receipt,
        "sham": sham_info,
        "D4": d4_results,
    })

    # ------------------------------------------------------------------
    # Step C: RESERVE_PROMPTS_8 Plain Baseline, 7-Rung Commit Gate, D_above_cap Trigger & M8 6 Rand
    # ------------------------------------------------------------------
    _log(f"Step C: 7-rung commit gate on {len(reserve_prompts)} reserve prompts + M8 6 Rand controls")
    res_plain_gens = _gen_b1(res_plain_texts, tok_budget["reserve"])
    res_plain_tf_logits: list[torch.Tensor] = []
    for p_ids, g in zip(res_plain_id_lists, res_plain_gens):
        res_plain_tf_logits.append(teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device))
        counts["teacher_forced_b1_fwds"] += 1

    compiled_arm_specs: list[tuple[str, dict[int, torch.Tensor]]] = [
        ("WSP_Pirate_12L", v_pre_pirate_con),
        ("WSP_NoMD_12L", v_pre_nomd_con),
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
    d_above_cap_trigger: dict[str, float | None] = {}

    for arm_name, v_map in compiled_arm_specs:
        rung_rows: list[dict[str, Any]] = []
        passing_rungs: list[tuple[float, dict[int, float], dict[str, dict[str, float]]]] = []
        above_cap_d_fail: float | None = None

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
                res_arm_gens = _gen_b1(res_plain_texts, tok_budget["reserve"])

            rep_ok = all(
                (v["stored_rank1_proj_ret"] >= cg_min_proj and v["stored_frob_ret"] >= cg_min_frob)
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
            other_clauses_ok = (
                (not dose_unrep)
                and (max_kl <= cg_max_kl_max)
                and (max_pos_kl <= cg_max_pos_max)
                and (n_broken <= cg_broken_max)
            )
            rung_pass = other_clauses_ok and (med_kl <= cg_med_kl_max)
            if other_clauses_ok and (med_kl > cg_med_kl_max) and (above_cap_d_fail is None):
                above_cap_d_fail = float(d)

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
            fb_d = dose_ladder[0]
            _, fb_alphas = build_layer_edits(layers, fb_d, v_map, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
            selected_dose[arm_name] = fb_d
            selected_alphas[arm_name] = fb_alphas
            selected_retention[arm_name] = rung_rows[0]["bf16_retention"]
            commit_gate_fallback[arm_name] = True

        d_above_cap_trigger[arm_name] = above_cap_d_fail
        commit_gate_receipt["arms"][arm_name] = {
            "selected_d_star": selected_dose[arm_name],
            "commit_gate_fallback": commit_gate_fallback[arm_name],
            "d_above_cap_trigger_dose": above_cap_d_fail,
            "rungs": rung_rows,
        }
        _log(f"  {arm_name}: d*={selected_dose[arm_name]} fallback={commit_gate_fallback[arm_name]} d_fail={above_cap_d_fail}")

    d_pirate = selected_dose["WSP_Pirate_12L"]
    d_nomd = selected_dose["WSP_NoMD_12L"]

    def _rand_edits_matching(
        rand_v_map: dict[int, torch.Tensor],
        ref_alphas: dict[int, float],
    ) -> dict[int, tuple[float, torch.Tensor, torch.Tensor]]:
        return {L: (float(ref_alphas[L]), rand_v_map[L].float(), k_hat_by_layer[L].float()) for L in layers}

    # M8: 6 Rand controls on RESERVE_PROMPTS_8 + delivered_dose_match_log
    rand_reserve_receipt: dict[str, Any] = {}
    rand_check_specs: list[tuple[str, str, str, float, dict[int, torch.Tensor], dict[int, float], dict[str, dict[str, float]]]] = []
    for s in range(3):
        rand_check_specs.append((
            f"Rand_Pirate_12L_s{s}@d*_pirate", "WSP_Pirate_12L", "WSP_Pirate_12L@d*_pirate", d_pirate,
            v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_12L"], selected_retention["WSP_Pirate_12L"],
        ))
        rand_check_specs.append((
            f"Rand_NoMD_12L_s{s}@d*_nomd", "WSP_NoMD_12L", "WSP_NoMD_12L@d*_nomd", d_nomd,
            v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_12L"], selected_retention["WSP_NoMD_12L"],
        ))

    for rlabel, base_arm, paired_edit_label, rdose, rv_map, ref_alphas, ref_ret in rand_check_specs:
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
        delivered_dose_match: dict[str, dict[str, float]] = {}
        for L in layers:
            g = gamma_by_layer[L]
            rms_g = float(torch.sqrt((g * g).mean()).item())
            rand_post_norm = float((g * rv_map[L].float()).norm().item())
            delivered_dose_match[str(L)] = {
                "arm_delivered_match_ratio": direction_quality[base_arm][str(L)]["delivered_match_ratio"],
                "rand_delivered_match_ratio": round(rms_g / max(rand_post_norm, 1e-12), 6),
                "arm_over_rand_proj_ret": round(
                    ref_ret[str(L)]["stored_rank1_proj_ret"] / max(r_ret[str(L)]["stored_rank1_proj_ret"], 1e-12), 6
                ),
            }
        rand_reserve_receipt[rlabel] = {
            "paired_edit": paired_edit_label,
            "dose": rdose,
            "bf16_retention": r_ret,
            "intended_frob_ratio_vs_edit": intended_ratios,
            "stored_frob_ratio_vs_edit": stored_ratios,
            "delivered_dose_match_log": delivered_dose_match,
            "median_off_kl": round(_median([x["off_kl"] for x in r_kls]), 6),
            "max_prompt_off_kl": round(max(x["off_kl"] for x in r_kls), 6),
            "per_prompt_kl": r_kls,
        }

    _write_checkpoint(checkpoint_dir, "step_c_commit_gate", {
        "commit_gate": commit_gate_receipt, "rand_reserve": rand_reserve_receipt,
    })

    # ------------------------------------------------------------------
    # Build Active Arm Registry
    # ------------------------------------------------------------------
    edits_pirate_dstar, _ = build_layer_edits(layers, d_pirate, v_pre_pirate_con, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
    edits_nomd_dstar, _ = build_layer_edits(layers, d_nomd, v_pre_nomd_con, k_hat_by_layer, calib_layer_stats, gamma_by_layer)

    arm_registry: dict[str, tuple[str | None, dict[int, tuple[float, torch.Tensor, torch.Tensor]] | None]] = {
        "Plain": (None, None),
        "Sys_Pirate": (sys_pirate, None),
        "Sys_NoMD": (sys_nomd, None),
        "WSP_Pirate_12L@d*_pirate": (None, edits_pirate_dstar),
        "WSP_NoMD_12L@d*_nomd": (None, edits_nomd_dstar),
    }
    for s in range(3):
        arm_registry[f"Rand_Pirate_12L_s{s}@d*_pirate"] = (
            None, _rand_edits_matching(v_pre_rand_pirate[s], selected_alphas["WSP_Pirate_12L"])
        )
        arm_registry[f"Rand_NoMD_12L_s{s}@d*_nomd"] = (
            None, _rand_edits_matching(v_pre_rand_nomd[s], selected_alphas["WSP_NoMD_12L"])
        )

    def _run_arm_generations(
        arm_label: str,
        prompts_list: list[dict[str, Any]],
        max_tok: int,
        ban_proc: CommonBanLogitsProcessor | None = None,
    ) -> list[dict[str, Any]]:
        sys_p, edits = arm_registry[arm_label]
        formatted = [_fmt_sys(sys_p, p["text"]) if sys_p is not None else _fmt_plain(p["text"]) for p in prompts_list]
        if edits is None:
            gens = _gen_b1(formatted, max_tok, ban_proc=ban_proc)
        else:
            with master.apply_edit(model, edits):
                gens = _gen_b1(formatted, max_tok, ban_proc=ban_proc)
        rows = []
        for p, g in zip(prompts_list, gens):
            txt = g["text"]
            rows.append({
                "id": p["id"],
                "text": txt,
                "n_tokens": g["n_tokens"],
                "is_broken": c3_scoring.is_broken(txt),
                "surface_stem_hit": c3_scoring.surface_stem_hit(txt, surface_regex),
                "extended_ban_hit": c3_scoring.surface_stem_hit(txt, strict_ext_ban_regex),
            })
        return rows

    # ------------------------------------------------------------------
    # Step D: Pillar A (N=40 x 2 bans x 6 arms), D5 Multi-Turn (48 tok), D6_heldout, D1 Copula
    # ------------------------------------------------------------------
    _log(f"Step D: Pillar A ({len(pillar_a_prompts)} x 2 bans x 6 arms) + D5 + D6_heldout + D1")
    pillar_a_arms = list(c3b_constants["reader_kappa_pool"]["core_arms_6"])
    pillar_a_receipt: dict[str, dict[str, list[dict[str, Any]]]] = {"none": {}, "common_pirate_ban": {}}
    for arm_label in pillar_a_arms:
        pillar_a_receipt["none"][arm_label] = _run_arm_generations(
            arm_label, pillar_a_prompts, tok_budget["pillar_a"], ban_proc=None
        )
        pillar_a_receipt["common_pirate_ban"][arm_label] = _run_arm_generations(
            arm_label, pillar_a_prompts, tok_budget["pillar_a"], ban_proc=common_ban_proc
        )
    _write_checkpoint(checkpoint_dir, "step_d1_pillar_a", pillar_a_receipt)

    # D5 Multi-Turn (MULTITURN_12 x 3 turns x 4 arms, 48 tok/turn, ban=none)
    d5_arms = list(c3b_constants["multiturn_d5_arms_4"])
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
                gens = _gen_b1(formatted, tok_budget["multiturn"])
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

    # D6_heldout: OUT-OF-SAMPLE top-50 boosted tokens computed on DISJOINT RESERVE_PROMPTS_8 (reusing Plain logits!)
    common_single_set = set(ban_single_ids)
    common_multi_member_set = {tid for seq in ban_multi_seqs for tid in seq}
    excluded_d6_ids = common_single_set | common_multi_member_set | special_ids

    def _compute_heldout_own_top_k(arm_label: str) -> list[dict[str, Any]]:
        _, edits = arm_registry[arm_label]
        assert edits is not None
        diff_sum: torch.Tensor | None = None
        total_pos = 0
        with master.apply_edit(model, edits):
            for p_ids, g, pl_log in zip(res_plain_id_lists, res_plain_gens, res_plain_tf_logits):
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
        "WSP_Pirate_12L@d*_pirate": _compute_heldout_own_top_k("WSP_Pirate_12L@d*_pirate"),
        "Rand_Pirate_12L_s0@d*_pirate": _compute_heldout_own_top_k("Rand_Pirate_12L_s0@d*_pirate"),
    }
    demo_prompt_map = {p["id"]: p for p in pillar_a_prompts}
    demo_prompts_12 = [demo_prompt_map[pid] for pid in demo_showcase_ids]

    def _make_ext_ban(extra_rows: list[dict[str, Any]]) -> CommonBanLogitsProcessor:
        extra_ids = [int(r["id"]) for r in extra_rows]
        return CommonBanLogitsProcessor(ban_single_ids + extra_ids, ban_multi_seqs, eos_token_ids=set(eos_ids))

    plain_common_by_id = {r["id"]: r for r in pillar_a_receipt["common_pirate_ban"]["Plain"]}
    d6_receipt: dict[str, Any] = {
        "source_split": "RESERVE_PROMPTS_8",
        "source_prompt_ids": [p["id"] for p in reserve_prompts],
        "demo_ids": demo_showcase_ids,
        "own_top50": d6_own_top50,
        "conditions": {
            "Plain@common": [plain_common_by_id[pid] for pid in demo_showcase_ids],
            "WSP_Pirate_12L@own50+common": _run_arm_generations(
                "WSP_Pirate_12L@d*_pirate", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["WSP_Pirate_12L@d*_pirate"]),
            ),
            "WSP_Pirate_12L@donor_rand0+common": _run_arm_generations(
                "WSP_Pirate_12L@d*_pirate", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["Rand_Pirate_12L_s0@d*_pirate"]),
            ),
            "Rand0@own50+common": _run_arm_generations(
                "Rand_Pirate_12L_s0@d*_pirate", demo_prompts_12, tok_budget["d6"],
                ban_proc=_make_ext_ban(d6_own_top50["Rand_Pirate_12L_s0@d*_pirate"]),
            ),
        },
    }

    # D1 Part 213 Copula (12 pairs x 4 arms, B=1 use_cache=False)
    d1_arms = ["Plain", "Sys_Pirate", "WSP_Pirate_12L@d*_pirate", "Rand_Pirate_12L_s0@d*_pirate"]
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
                log_self = F.log_softmax(forward_next_logits_b1(model, ids_self, device), dim=-1)
                log_strg = F.log_softmax(forward_next_logits_b1(model, ids_strg, device), dim=-1)
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

    _write_checkpoint(checkpoint_dir, "step_d2_d5_d6_d1", {"D5": d5_receipt, "D6_heldout": d6_receipt, "D1": d1_receipt})

    # ------------------------------------------------------------------
    # Step E: Pillar B (NOMD_32 x 6 arms, 96 tok, ban=none)
    # ------------------------------------------------------------------
    _log(f"Step E: Pillar B ({len(nomd_prompts)} prompts x 6 arms)")
    pillar_b_arms = [
        "Plain",
        "Sys_NoMD",
        "WSP_NoMD_12L@d*_nomd",
        "Rand_NoMD_12L_s0@d*_nomd",
        "Rand_NoMD_12L_s1@d*_nomd",
        "Rand_NoMD_12L_s2@d*_nomd",
    ]
    pillar_b_receipt: dict[str, list[dict[str, Any]]] = {}
    for arm_label in pillar_b_arms:
        raw_rows = _run_arm_generations(arm_label, nomd_prompts, tok_budget["pillar_b"], ban_proc=None)
        for r in raw_rows:
            r["construct_types"] = c3_scoring.construct_types(r["text"])
            r["has_markdown"] = c3_scoring.has_markdown(r["text"])
            r["markdown_hits"] = c3_scoring.markdown_hits(r["text"])
            r.update(em_dash_stats(r["text"]))
        pillar_b_receipt[arm_label] = raw_rows

    _write_checkpoint(checkpoint_dir, "step_e_pillar_b", pillar_b_receipt)

    # ------------------------------------------------------------------
    # Step F: Capability Suite (7 arms) & D2 Sibling Bleed
    # ------------------------------------------------------------------
    _log("Step F: Capability Suite (7 arms) + D2 Sibling Bleed")
    cap_arms = list(c3b_constants["capability_suite_arms_7"])

    neu_plain_gens = _gen_b1(neu_plain_texts, tok_budget["neutral"])
    neu_plain_tf_logits: list[torch.Tensor] = []
    for p_ids, g in zip(neu_plain_id_lists, neu_plain_gens):
        neu_plain_tf_logits.append(teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"][:48], device))
        counts["teacher_forced_b1_fwds"] += 1

    def _run_unlocked(sys_p: str | None, plist: list[dict[str, Any]], mtok: int) -> list[dict[str, Any]]:
        ft = [_fmt_sys(sys_p, p["text"]) if sys_p is not None else _fmt_plain(p["text"]) for p in plist]
        gs = _gen_b1(ft, mtok)
        return [
            {"id": p["id"], "text": g["text"], "n_tokens": g["n_tokens"], "is_broken": c3_scoring.is_broken(g["text"])}
            for p, g in zip(plist, gs)
        ]

    capability_receipt: dict[str, Any] = {}
    for arm_label in cap_arms:
        sys_p, edits = arm_registry[arm_label]

        def _eval_cap() -> dict[str, Any]:
            math_rows = _run_unlocked(sys_p, math_prompts, tok_budget["math"])
            code_rows = _run_unlocked(sys_p, code_prompts, tok_budget["code"])
            ref_rows = _run_unlocked(sys_p, refusal_prompts, tok_budget["refusal"])
            if arm_label == "Plain":
                neu_gens = neu_plain_gens
                neu_kls = [
                    {"id": p["id"], "off_kl": 0.0, "max_single_pos_kl": 0.0, "greedy_exact_tok_match_rate": 1.0}
                    for p in neutral_prompts
                ]
            else:
                neu_texts = [_fmt_sys(sys_p, p["text"]) if sys_p is not None else _fmt_plain(p["text"]) for p in neutral_prompts]
                neu_gens = _gen_b1(neu_texts, tok_budget["neutral"])
                neu_kls = []
                for p, p_ids_plain, g_plain, g_arm, pl_log in zip(
                    neutral_prompts, neu_plain_id_lists, neu_plain_gens, neu_gens, neu_plain_tf_logits
                ):
                    p_ids_arm = (
                        tokenizer.encode(_fmt_sys(sys_p, p["text"]), add_special_tokens=False)
                        if sys_p is not None
                        else p_ids_plain
                    )
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

        if edits is None:
            capability_receipt[arm_label] = _eval_cap()
        else:
            with master.apply_edit(model, edits):
                capability_receipt[arm_label] = _eval_cap()

    # D2 Sibling Bleed
    pirate_on_nomd = _run_arm_generations("WSP_Pirate_12L@d*_pirate", nomd_prompts, tok_budget["pillar_b"])
    for r in pirate_on_nomd:
        r["construct_types"] = c3_scoring.construct_types(r["text"])
        r["has_markdown"] = c3_scoring.has_markdown(r["text"])
        r["markdown_hits"] = c3_scoring.markdown_hits(r["text"])
        r.update(em_dash_stats(r["text"]))
    nomd_on_identity = _run_arm_generations("WSP_NoMD_12L@d*_nomd", identity_prompts, tok_budget["pillar_a"])

    sibling_bleed_receipt = {
        "WSP_Pirate_12L@d*_pirate_on_NOMD_32": pirate_on_nomd,
        "WSP_NoMD_12L@d*_nomd_on_IDENTITY_20": nomd_on_identity,
    }
    _write_checkpoint(checkpoint_dir, "step_f_cap_d2", {
        "capability_suite": capability_receipt, "D2_sibling_bleed": sibling_bleed_receipt,
    })

    # ------------------------------------------------------------------
    # Step G: Conditional D_above_cap Lost-KL-Bet Diagnostic (<= 36 gens)
    # ------------------------------------------------------------------
    _log(f"Step G: D_above_cap check (triggers={d_above_cap_trigger})")
    d_above_cap_receipt: dict[str, Any] = {}

    dfail_p = d_above_cap_trigger["WSP_Pirate_12L"]
    if dfail_p is not None:
        p_prompts_dac = [demo_prompt_map[pid] for pid in d_above_cap_pirate_ids]
        dac_edits_p, _ = build_layer_edits(layers, dfail_p, v_pre_pirate_con, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
        dac_p_rows: dict[str, list[dict[str, Any]]] = {}
        with master.apply_edit(model, dac_edits_p) as dac_ret_p:
            for ban_k, bp in (("none", None), ("common_pirate_ban", common_ban_proc)):
                formatted = [_fmt_plain(p["text"]) for p in p_prompts_dac]
                gens = _gen_b1(formatted, tok_budget["d_above_cap_pirate"], ban_proc=bp)
                dac_p_rows[ban_k] = [
                    {
                        "id": p["id"],
                        "text": g["text"],
                        "n_tokens": g["n_tokens"],
                        "is_broken": c3_scoring.is_broken(g["text"]),
                        "surface_stem_hit": c3_scoring.surface_stem_hit(g["text"], surface_regex),
                        "extended_ban_hit": c3_scoring.surface_stem_hit(g["text"], strict_ext_ban_regex),
                    }
                    for p, g in zip(p_prompts_dac, gens)
                ]
        d_above_cap_receipt["WSP_Pirate_12L"] = {
            "triggered": True,
            "d_fail": dfail_p,
            "prompt_ids": d_above_cap_pirate_ids,
            "bf16_retention": dac_ret_p,
            "generations": dac_p_rows,
        }
    else:
        d_above_cap_receipt["WSP_Pirate_12L"] = {"triggered": False, "d_fail": None}

    dfail_n = d_above_cap_trigger["WSP_NoMD_12L"]
    if dfail_n is not None:
        nomd_by_id = {p["id"]: p for p in nomd_prompts}
        n_prompts_dac = [nomd_by_id[pid] for pid in d_above_cap_nomd_ids]
        dac_edits_n, _ = build_layer_edits(layers, dfail_n, v_pre_nomd_con, k_hat_by_layer, calib_layer_stats, gamma_by_layer)
        with master.apply_edit(model, dac_edits_n) as dac_ret_n:
            formatted = [_fmt_plain(p["text"]) for p in n_prompts_dac]
            gens = _gen_b1(formatted, tok_budget["d_above_cap_nomd"])
        rows_n = []
        for p, g in zip(n_prompts_dac, gens):
            r = {
                "id": p["id"],
                "text": g["text"],
                "n_tokens": g["n_tokens"],
                "is_broken": c3_scoring.is_broken(g["text"]),
                "construct_types": c3_scoring.construct_types(g["text"]),
                "has_markdown": c3_scoring.has_markdown(g["text"]),
                "markdown_hits": c3_scoring.markdown_hits(g["text"]),
            }
            r.update(em_dash_stats(g["text"]))
            rows_n.append(r)
        d_above_cap_receipt["WSP_NoMD_12L"] = {
            "triggered": True,
            "d_fail": dfail_n,
            "prompt_ids": d_above_cap_nomd_ids,
            "bf16_retention": dac_ret_n,
            "generations": rows_n,
        }
    else:
        d_above_cap_receipt["WSP_NoMD_12L"] = {"triggered": False, "d_fail": None}

    # ------------------------------------------------------------------
    # Step H: Exploratory D_key_ablation (OUTSIDE Holm / C3b.0 / EQ-C3b.2; bitwise restored!)
    # ------------------------------------------------------------------
    _log("Step H: Exploratory D_key_ablation (WSP_NoMD_12L_kWA + WSP_Pirate_12L_kSparse32)")

    # H.1: WSP_NoMD_12L_kWA (Rank-1 weight-aligned orthogonalization, beta in [0.25, 0.50, 1.00])
    kwa_rungs: list[dict[str, Any]] = []
    kwa_passing: list[tuple[float, dict[str, dict[str, float]]]] = []
    beta_ladder = KWA_BETA_LADDER[:2] if smoke else KWA_BETA_LADDER
    for beta in beta_ladder:
        orth_spec = {L: (beta, v_pre_nomd_con[L]) for L in layers}
        with master.apply_orth_edit(model, orth_spec) as orth_ret:
            per_prompt_kl = []
            for p, p_ids, g, pl_log in zip(reserve_prompts, res_plain_id_lists, res_plain_gens, res_plain_tf_logits):
                arm_log = teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device)
                counts["teacher_forced_b1_fwds"] += 1
                m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
                per_prompt_kl.append({
                    "id": p["id"],
                    "off_kl": round(m_kl, 6),
                    "max_single_pos_kl": round(max_p_kl, 6),
                    "max_abs_dlogit": round(max_dl, 6),
                })
            res_gens = _gen_b1(res_plain_texts, tok_budget["reserve"])

        rep_ok = all(
            (v["stored_rank1_proj_ret"] >= cg_min_proj and v["stored_frob_ret"] >= cg_min_frob)
            for v in orth_ret.values()
        )
        med_kl = _median([r["off_kl"] for r in per_prompt_kl])
        max_kl = max(r["off_kl"] for r in per_prompt_kl)
        max_pos_kl = max(r["max_single_pos_kl"] for r in per_prompt_kl)
        res_gen_rows = [
            {"id": p["id"], "text": g["text"], "n_tokens": g["n_tokens"], "is_broken": c3_scoring.is_broken(g["text"])}
            for p, g in zip(reserve_prompts, res_gens)
        ]
        n_broken = sum(1 for r in res_gen_rows if r["is_broken"])
        r_pass = (
            rep_ok
            and (med_kl <= cg_med_kl_max)
            and (max_kl <= cg_max_kl_max)
            and (max_pos_kl <= cg_max_pos_max)
            and (n_broken <= cg_broken_max)
        )
        kwa_rungs.append({
            "beta": beta,
            "bf16_retention": orth_ret,
            "median_off_kl": round(med_kl, 6),
            "max_prompt_off_kl": round(max_kl, 6),
            "max_single_pos_kl": round(max_pos_kl, 6),
            "n_broken": n_broken,
            "pass": r_pass,
            "per_prompt_kl": per_prompt_kl,
            "reserve_generations": res_gen_rows,
        })
        if r_pass:
            kwa_passing.append((beta, orth_ret))

    if kwa_passing:
        kwa_beta_star, kwa_ret_star = kwa_passing[-1]
        kwa_fallback = False
    else:
        kwa_beta_star = beta_ladder[0]
        kwa_ret_star = kwa_rungs[0]["bf16_retention"]
        kwa_fallback = True

    with master.apply_orth_edit(model, {L: (kwa_beta_star, v_pre_nomd_con[L]) for L in layers}):
        kwa_nomd_gens = _gen_b1([_fmt_plain(p["text"]) for p in nomd_prompts], tok_budget["pillar_b"])
        kwa_neu_kls: list[dict[str, Any]] = []
        for p, p_ids_plain, g_plain, pl_log in zip(neutral_prompts, neu_plain_id_lists, neu_plain_gens, neu_plain_tf_logits):
            arm_log = teacher_forced_gen_logits_b1(model, p_ids_plain, g_plain["gen_ids"][:48], device)
            counts["teacher_forced_b1_fwds"] += 1
            m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
            kwa_neu_kls.append({
                "id": p["id"],
                "off_kl": round(m_kl, 6),
                "max_single_pos_kl": round(max_p_kl, 6),
                "max_abs_dlogit": round(max_dl, 6),
            })

    kwa_nomd_rows = []
    for p, g in zip(nomd_prompts, kwa_nomd_gens):
        r = {
            "id": p["id"],
            "text": g["text"],
            "n_tokens": g["n_tokens"],
            "is_broken": c3_scoring.is_broken(g["text"]),
            "construct_types": c3_scoring.construct_types(g["text"]),
            "has_markdown": c3_scoring.has_markdown(g["text"]),
            "markdown_hits": c3_scoring.markdown_hits(g["text"]),
        }
        r.update(em_dash_stats(g["text"]))
        kwa_nomd_rows.append(r)

    # H.2: WSP_Pirate_12L_kSparse32 (top-32 |m_bar_L| key; starts at d*_pirate, steps down if reserve KL > 0.65)
    sp32_candidates = [d for d in reversed(dose_ladder) if d <= d_pirate + 1e-9]
    if not sp32_candidates:
        sp32_candidates = [dose_ladder[0]]
    sp32_rungs: list[dict[str, Any]] = []
    sp32_d_star = sp32_candidates[-1]
    sp32_fallback = True
    sp32_ret_star: dict[str, dict[str, float]] = {}

    for d_cand in sp32_candidates:
        sp_edits, sp_alphas = build_layer_edits(
            layers, d_cand, v_pre_pirate_con, k_sp32_by_layer, calib_sp32_stats, gamma_by_layer
        )
        with master.apply_edit(model, sp_edits) as sp_ret:
            per_prompt_kl = []
            for p, p_ids, g, pl_log in zip(reserve_prompts, res_plain_id_lists, res_plain_gens, res_plain_tf_logits):
                arm_log = teacher_forced_gen_logits_b1(model, p_ids, g["gen_ids"], device)
                counts["teacher_forced_b1_fwds"] += 1
                m_kl, max_p_kl, max_dl = compute_off_kl_from_logits(pl_log, arm_log)
                per_prompt_kl.append({
                    "id": p["id"],
                    "off_kl": round(m_kl, 6),
                    "max_single_pos_kl": round(max_p_kl, 6),
                    "max_abs_dlogit": round(max_dl, 6),
                })
        med_kl = _median([r["off_kl"] for r in per_prompt_kl])
        max_kl = max(r["off_kl"] for r in per_prompt_kl)
        max_pos_kl = max(r["max_single_pos_kl"] for r in per_prompt_kl)
        rep_ok = all(
            (v["stored_rank1_proj_ret"] >= cg_min_proj and v["stored_frob_ret"] >= cg_min_frob)
            for v in sp_ret.values()
        )
        ok = rep_ok and (med_kl <= cg_med_kl_max)
        sp32_rungs.append({
            "dose": d_cand,
            "alphas": {str(L): round(sp_alphas[L], 8) for L in layers},
            "bf16_retention": sp_ret,
            "median_off_kl": round(med_kl, 6),
            "max_prompt_off_kl": round(max_kl, 6),
            "max_single_pos_kl": round(max_pos_kl, 6),
            "pass": ok,
            "per_prompt_kl": per_prompt_kl,
        })
        if ok:
            sp32_d_star = d_cand
            sp32_ret_star = sp_ret
            sp32_fallback = False
            break
    if sp32_fallback:
        sp32_d_star = sp32_rungs[-1]["dose"]
        sp32_ret_star = sp32_rungs[-1]["bf16_retention"]

    sp32_edits_star, _ = build_layer_edits(
        layers, sp32_d_star, v_pre_pirate_con, k_sp32_by_layer, calib_sp32_stats, gamma_by_layer
    )
    sp32_gens_by_ban: dict[str, list[dict[str, Any]]] = {}
    with master.apply_edit(model, sp32_edits_star):
        for ban_k, bp in (("none", None), ("common_pirate_ban", common_ban_proc)):
            formatted = [_fmt_plain(p["text"]) for p in demo_prompts_12]
            gens = _gen_b1(formatted, tok_budget["pillar_a"], ban_proc=bp)
            sp32_gens_by_ban[ban_k] = [
                {
                    "id": p["id"],
                    "text": g["text"],
                    "n_tokens": g["n_tokens"],
                    "is_broken": c3_scoring.is_broken(g["text"]),
                    "surface_stem_hit": c3_scoring.surface_stem_hit(g["text"], surface_regex),
                    "extended_ban_hit": c3_scoring.surface_stem_hit(g["text"], strict_ext_ban_regex),
                }
                for p, g in zip(demo_prompts_12, gens)
            ]

    d_key_ablation_receipt = {
        "WSP_NoMD_12L_kWA": {
            "selected_beta_star": kwa_beta_star,
            "fallback": kwa_fallback,
            "bf16_retention_selected": kwa_ret_star,
            "reserve_rungs": kwa_rungs,
            "nomd_32": kwa_nomd_rows,
            "neutral_chat_16_kl": kwa_neu_kls,
            "neutral_median_off_kl": round(_median([x["off_kl"] for x in kwa_neu_kls]), 6),
        },
        "WSP_Pirate_12L_kSparse32": {
            "support_k": SPARSE_KEY_K,
            "calib_sparse32_layers": {
                str(L): {k: v for k, v in calib_sp32_stats[L].items() if k != "topk_indices"}
                for L in layers
            },
            "selected_d_star": sp32_d_star,
            "fallback": sp32_fallback,
            "bf16_retention_selected": sp32_ret_star,
            "reserve_rungs": sp32_rungs,
            "demo_ids": demo_showcase_ids,
            "generations": sp32_gens_by_ban,
        },
    }

    _write_checkpoint(checkpoint_dir, "step_gh_above_cap_key_ablation", {
        "D_above_cap": d_above_cap_receipt, "D_key_ablation": d_key_ablation_receipt,
    })

    master.check_hooks(model)
    elapsed_sec = round(time.time() - t0, 2)
    _log(
        f"DONE in {elapsed_sec}s: gens={counts['generations']}, toks={counts['generated_tokens']}, "
        f"restores={master.restore_checks}/{master.layers_compiled_total}, hooks_max={master.max_hooks_seen}"
    )

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "meta": {
            "model": MODEL_ID,
            "revision": MODEL_REV,
            "smoke": bool(smoke),
            "device": device_name,
            "elapsed_sec": elapsed_sec,
            "target_layers": layers,
            "requested_gen_batch_size": 1,
            "effective_gen_batch_size": 1,
            "active_hooks_max_seen": master.max_hooks_seen,
            "restore_exact_all": master.restore_exact_all,
            "layers_compiled_total": master.layers_compiled_total,
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
            "D_align": d_align_receipt,
        },
        "doses": {
            "d_star_pirate": d_pirate,
            "d_star_nomd": d_nomd,
            "commit_gate_fallback": commit_gate_fallback,
            "d_above_cap_trigger": d_above_cap_trigger,
        },
        "commit_gate": commit_gate_receipt,
        "rand_reserve": rand_reserve_receipt,
        "pillar_a": pillar_a_receipt,
        "pillar_b": pillar_b_receipt,
        "D5_multiturn": d5_receipt,
        "D6_heldout": d6_receipt,
        "D1_copula": d1_receipt,
        "capability_suite": capability_receipt,
        "D2_sibling_bleed": sibling_bleed_receipt,
        "D_above_cap": d_above_cap_receipt,
        "D_key_ablation": d_key_ablation_receipt,
    }
    return receipt


def emit_paced_b64_gz(gz_bytes: bytes, chunk_bytes: int = 16384, sleep_sec: float = 0.10) -> None:
    """Emit gzip receipt as paced base64 (~160 KB/s) safely under Jupyter's 1 MB/s iopub limit."""
    b64 = base64.b64encode(gz_bytes).decode("ascii")
    print("BEGIN_C3B_RECEIPT_GZ_B64", flush=True)
    for i in range(0, len(b64), chunk_bytes):
        print(b64[i : i + chunk_bytes], flush=True)
        if i + chunk_bytes < len(b64):
            time.sleep(sleep_sec)
    print("END_C3B_RECEIPT_GZ_B64", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage C3b (WSP v2) on OLMo 3 7B (12L, exact B=1, zero-hook).")
    ap.add_argument("--design", type=Path, default=HERE / "OLMO3-WSP-C3B-DESIGN.md")
    ap.add_argument("--c3b-constants", type=Path, default=HERE / "c3b_prereg_constants.json")
    ap.add_argument("--bank", type=Path, default=HERE / "c3b_vector_bank.safetensors")
    ap.add_argument("--scoring", type=Path, default=HERE / "c3_scoring.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (hard-refused by score_olmo3_c3b.py).")
    ap.add_argument("--emit-b64-stdout", action="store_true", help="Emit paced gzip+b64 receipt on stdout for Colab.")
    args = ap.parse_args()

    hashes = verify_preflight_provenance(args.design, args.c3b_constants, args.bank, args.scoring)
    c3b_constants = json.loads(args.c3b_constants.read_text(encoding="utf-8"))
    bank = safetensors_load_file(str(args.bank))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REV,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    receipt = run_stage_c3b_experiment(
        model,
        tok,
        c3b_constants,
        bank,
        hashes,
        smoke=args.smoke,
        device_name=dev_name,
        checkpoint_dir=args.checkpoint_dir,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raw_json = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.out.suffix == ".gz":
        gz_bytes = gzip.compress(raw_json, compresslevel=6)
        args.out.write_bytes(gz_bytes)
    else:
        args.out.write_bytes(raw_json)
        gz_bytes = gzip.compress(raw_json, compresslevel=6)

    print(
        f"Wrote Stage C3b receipt to {args.out} ({args.out.stat().st_size} B, "
        f"elapsed={receipt['meta']['elapsed_sec']}s, gens={receipt['meta']['counts']['generations']}, "
        f"restores={receipt['meta']['restore_checks']}/{receipt['meta']['layers_compiled_total']})",
        file=sys.stderr,
        flush=True,
    )
    if args.emit_b64_stdout:
        emit_paced_b64_gz(gz_bytes)


if __name__ == "__main__":
    main()
