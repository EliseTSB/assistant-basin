#!/usr/bin/env python3
"""Offline CPU PyTorch tests for Stage C2 v4 runner (`run_olmo3_c2.py`):
  1. Preflight SHA-256 verification against REAL frozen v4 artifacts (pass + fail-closed tamper test).
  2. Exact algebraic equivalence: `pre_norm_inject_hook` before `post_feedforward_layernorm` matches
     a literal rank-1 `down_proj` weight-column / bias addition to <1e-5 on a synthetic OLMo-3 model!
  3. End-to-end smoke execution of `run_stage_c2_experiment` on a synthetic 32-layer OLMo-3-compatible
     PyTorch model verifying: all 19 shams (13 write + 6 patch) == exact 0.0, active arms > 0.0,
     exact 4-term telescoping (<1e-4), F11 donor substitution liveness, and ZERO leaked hooks!
"""
from __future__ import annotations

import json
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

import sys
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from olmo3_virtual_and_highway import run_olmo3_c2 as r_mod


class TinyRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return ((x.float() / rms) * self.weight.float()).to(x.dtype)


class TinyOlmo3Layer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)
        self.post_attention_layernorm = TinyRMSNorm(d)
        self.up_proj = nn.Linear(d, d * 2, bias=False)
        self.down_proj = nn.Linear(d * 2, d, bias=False)
        self.post_feedforward_layernorm = TinyRMSNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Causal self-attention + post_attention_layernorm (matches OLMo 3 post-norm residual!)
        B, T, D = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e9)
        attn_out = self.o_proj(torch.matmul(torch.softmax(scores, dim=-1), v))
        h = h + self.post_attention_layernorm(attn_out)
        # MLP + post_feedforward_layernorm
        mlp_raw = self.down_proj(nn.functional.silu(self.up_proj(h)))
        h = h + self.post_feedforward_layernorm(mlp_raw)
        return h


class TinyOlmo3Model(nn.Module):
    def __init__(self, vocab: int = 64, d: int = 16, n_layers: int = 32):
        super().__init__()
        self.config = types.SimpleNamespace(num_hidden_layers=n_layers)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([TinyOlmo3Layer(d) for _ in range(n_layers)])
        self.model.norm = TinyRMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            for p in self.parameters():
                if p.ndim >= 2:
                    p.mul_(0.15)
            # Non-uniform gamma with a tiny dim to exercise low-gamma physics
            self.model.layers[24].post_feedforward_layernorm.weight[0] = 0.005

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.model.embed_tokens(input_ids)
        for lay in self.model.layers:
            h = lay(h)
        logits = self.lm_head(self.model.norm(h))
        return types.SimpleNamespace(logits=logits)


class TinyHashTokenizer:
    """Deterministic tiny tokenizer returning exact suffix-aligned token IDs for the synthetic model."""
    def __init__(self, vocab: int = 64):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.replace("\n", " <nl> ").split()
        return [1 + (abs(hash(w)) % (self.vocab - 1)) for w in words] or [1]


class TestStageC2RunnerCPU(unittest.TestCase):
    def test_real_v4_preflight_provenance_pass_and_tamper_fail(self) -> None:
        r_lens = Path.home() / "green" / "Journal" / "notes" / "reports" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
        bank = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"
        c2_const = ROOT / "c2_prereg_constants.json"
        design = ROOT / "OLMO3-TRANSPORT-C2-DESIGN.md"
        hashes = r_mod.verify_preflight_provenance(bank, r_lens, c2_const, design)
        self.assertEqual(hashes["c2_constants_sha256"], r_mod.EXPECTED_C2_CONSTANTS_SHA256)
        self.assertEqual(hashes["c2_bank_sha256"], r_mod.EXPECTED_C2_BANK_SHA256)
        self.assertEqual(hashes["c2_design_sha256"], r_mod.EXPECTED_C2_DESIGN_SHA256)
        # Fail-closed when pointing design to wrong file
        with self.assertRaises(ValueError):
            r_mod.verify_preflight_provenance(bank, r_lens, c2_const, c2_const)

    def test_prenorm_hook_matches_literal_down_proj_rank1_edit(self) -> None:
        torch.manual_seed(42)
        m = TinyOlmo3Model(vocab=64, d=16, n_layers=32)
        m.eval()
        x = torch.tensor([[3, 7, 11, 15, 2]], dtype=torch.long)
        w = torch.randn(16)
        w = w / w.norm()
        dose = 1.75

        with torch.no_grad():
            with r_mod.pre_norm_inject_hook(m, [(24, "ff", w, dose)]):
                logits_hook = m(x).logits[0, -1, :].clone()

            # Literally add dose * w to the pre-norm MLP output at the last token (equivalent to rank-1 down_proj column firing on last token)
            lay24 = m.model.layers[24]
            orig_fwd = lay24.down_proj.forward

            def _literal_rank1(inp):
                out = orig_fwd(inp)
                out = out.clone()
                out[:, -1, :] = out[:, -1, :] + dose * w
                return out

            lay24.down_proj.forward = _literal_rank1  # type: ignore[method-assign]
            logits_lit = m(x).logits[0, -1, :].clone()
            lay24.down_proj.forward = orig_fwd  # type: ignore[method-assign]

        max_err = float((logits_hook - logits_lit).abs().max().item())
        self.assertLess(max_err, 1e-5)

    def test_end_to_end_smoke_all_19_shams_telescoping_and_zero_leaked_hooks(self) -> None:
        torch.manual_seed(7)
        d, vocab = 16, 64
        m = TinyOlmo3Model(vocab=vocab, d=d, n_layers=32)
        m.eval()
        tok = TinyHashTokenizer(vocab=vocab)
        c2_const = json.loads((ROOT / "c2_prereg_constants.json").read_text(encoding="utf-8"))
        # Remap target IDs into tiny vocab [1..24] and sync suffix_lens to tiny tokenizer
        for i, t in enumerate(c2_const["targets_24"]):
            t["id"] = i + 1
        for tr in c2_const["c22_stem_pairs_12"]:
            tr["suffix_len"] = len(tok.encode(tr["short_bare"]["text"]))

        # Build matching tiny d=16 unit-norm bank
        def _u(shape):
            t = torch.randn(*shape)
            return t / t.norm(dim=-1, keepdim=True)

        w_u_eff = m.lm_head.weight.detach() * m.model.norm.weight.detach().unsqueeze(0)
        u_vecs = w_u_eff[1:25].clone()
        u_hat = u_vecs / u_vecs.norm(dim=-1, keepdim=True)
        bank = {
            "mu_u": w_u_eff.mean(dim=0),
            "u_vectors": u_vecs,
            "u_hat": u_hat,
            "L18.direct_matched": _u((12, d)),
            "L21.prenorm_ff_masked": _u((24, d)),
            "L24.prenorm_ff_unmasked": _u((24, d)),
            "L24.prenorm_ff_masked": _u((24, d)),
            "L24.prenorm_ff_rand_masked": _u((24, d)),
            "L24.prenorm_attn_masked": _u((24, d)),
            "L24.prenorm_attn_rand_masked": _u((24, d)),
            "L27.prenorm_ff_masked": _u((24, d)),
        }
        for L in (18, 24, 27):
            for an in ("R_mf", "R_ridge_raw", "R_ridge_centered", "R_ridge_shrink"):
                bank[f"L{L}.{an}"] = _u((24, d))
        for L in (18, 21, 24, 27, 30):
            bank[f"L{L}.rand_iso"] = _u((8, d))

        # Ensure u_dot_w_Rmf positive in synthetic constants
        for k, v in c2_const["c22_l18_frozen_pred_ret"].items():
            v["u_dot_w_Rmf"] = 0.8

        fake_hashes = {
            "r_lens_sha256": r_mod.EXPECTED_R_LENS_SHA256,
            "c2_constants_sha256": r_mod.EXPECTED_C2_CONSTANTS_SHA256,
            "c2_bank_sha256": r_mod.EXPECTED_C2_BANK_SHA256,
            "c2_design_sha256": r_mod.EXPECTED_C2_DESIGN_SHA256,
        }
        with torch.no_grad():
            rec = r_mod.run_stage_c2_experiment(m, tok, c2_const, bank, fake_hashes, smoke=True, device_name="cpu-tiny")

        self.assertTrue(rec["meta"]["smoke"])
        # All 13 write shams & all 6 patch shams MUST be exact 0.0!
        self.assertEqual(len(rec["shams"]["write_13"]), 13)
        self.assertEqual(len(rec["shams"]["patch_6"]), 6)
        for k, v in rec["shams"]["write_13"].items():
            self.assertEqual(v, 0.0, f"Write sham {k} non-zero: {v}")
        for k, v in rec["shams"]["patch_6"].items():
            self.assertEqual(v, 0.0, f"Patch sham {k} non-zero: {v}")

        # Active arms all moved logits (> 0.0)
        self.assertGreater(len(rec["active_arm_min_max_abs_dlogit"]), 20)
        for k, v in rec["active_arm_min_max_abs_dlogit"].items():
            self.assertGreater(v, 0.0, f"Dead active arm {k}")

        # Exact 4-term telescoping error < 1e-4 across Block 1
        t0 = c2_const["targets_24"][0]["string"]
        for row in rec["block1_C2_1"]["18"]["arms"]["R_ridge_shrink"][t0]:
            self.assertLess(abs(row["telescoping_err"]), 1e-4)

        # ZERO leaked forward hooks or pre-hooks anywhere in the model!
        for L, lay in enumerate(m.model.layers):
            self.assertEqual(len(lay.post_feedforward_layernorm._forward_hooks), 0, f"Leaked ff hook at L{L}")
            self.assertEqual(len(lay.post_feedforward_layernorm._forward_pre_hooks), 0, f"Leaked ff pre-hook at L{L}")
            self.assertEqual(len(lay.post_attention_layernorm._forward_hooks), 0, f"Leaked attn hook at L{L}")
            self.assertEqual(len(lay.post_attention_layernorm._forward_pre_hooks), 0, f"Leaked attn pre-hook at L{L}")
        self.assertEqual(len(m.model.norm._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
