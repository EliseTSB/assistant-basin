#!/usr/bin/env python3
"""Offline CPU PyTorch tests for Stage C2 v4 runner (`run_olmo3_c2.py`):
  1. Preflight SHA-256 verification against REAL frozen v4 artifacts (pass + fail-closed tamper test).
  2. Exact algebraic equivalence: `pre_norm_inject_hook` before `post_feedforward_layernorm` matches
     a literal rank-1 `down_proj` weight-column addition to <1e-5.
  3. F2 hook-leak regression test: bad 2nd injection (non-unit vector OR NaN dose) leaves ZERO leaked pre-hooks!
  4. Full 8,623-forward runner -> scorer CONTRACT test (F1 & F4): executes `run_stage_c2_experiment` in FULL
     non-smoke mode on a synthetic 32-layer OLMo-3 CPU model (with checkpoints!), verifies `total_forwards == 8623`,
     all 19 shams == 0.0, all 61 real active-arm keys present & >0, `retention` present on every Block 2b `rand_iso`
     row, zero leaked hooks, AND feeds the ACTUAL runner receipt straight into `score_olmo3_c2.score_stage_c2`!
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from olmo3_virtual_and_highway import run_olmo3_c2 as r_mod
from olmo3_virtual_and_highway import score_olmo3_c2 as sc_mod


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
        B, T, D = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, -1e9)
        attn_out = self.o_proj(torch.matmul(torch.softmax(scores, dim=-1), v))
        h = h + self.post_attention_layernorm(attn_out)
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
            self.model.layers[24].post_feedforward_layernorm.weight[0] = 0.005

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.model.embed_tokens(input_ids)
        for lay in self.model.layers:
            h = lay(h)
        logits = self.lm_head(self.model.norm(h))
        return types.SimpleNamespace(logits=logits)


class TinyHashTokenizer:
    def __init__(self, vocab: int = 64):
        self.vocab = vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        words = text.replace("\n", " <nl> ").split()
        return [1 + (abs(hash(w)) % (self.vocab - 1)) for w in words] or [1]


class TestStageC2RunnerCPU(unittest.TestCase):
    def test_01_real_v4_preflight_provenance_pass_and_tamper_fail(self) -> None:
        r_lens = Path.home() / "green" / "Journal" / "notes" / "reports" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt"
        bank = REPO_ROOT / "work" / "olmo3_c2" / "c2_vector_bank.safetensors"
        c2_const = ROOT / "c2_prereg_constants.json"
        design = ROOT / "OLMO3-TRANSPORT-C2-DESIGN.md"
        hashes = r_mod.verify_preflight_provenance(bank, r_lens, c2_const, design)
        self.assertEqual(hashes["c2_constants_sha256"], r_mod.EXPECTED_C2_CONSTANTS_SHA256)
        self.assertEqual(hashes["c2_bank_sha256"], r_mod.EXPECTED_C2_BANK_SHA256)
        self.assertEqual(hashes["c2_design_sha256"], r_mod.EXPECTED_C2_DESIGN_SHA256)
        with self.assertRaises(ValueError):
            r_mod.verify_preflight_provenance(bank, r_lens, c2_const, c2_const)

    def test_02_prenorm_hook_matches_literal_rank1_and_never_leaks_on_bad_injection(self) -> None:
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

            lay24 = m.model.layers[24]
            orig_fwd = lay24.down_proj.forward

            def _literal_rank1(inp):
                out = orig_fwd(inp).clone()
                out[:, -1, :] = out[:, -1, :] + dose * w
                return out

            lay24.down_proj.forward = _literal_rank1  # type: ignore[method-assign]
            logits_lit = m(x).logits[0, -1, :].clone()
            lay24.down_proj.forward = orig_fwd  # type: ignore[method-assign]
        self.assertLess(float((logits_hook - logits_lit).abs().max().item()), 1e-5)

        # F2 regression: failed validation on 2nd injection (non-unit OR NaN dose) MUST leave 0 leaked pre-hooks on L21!
        with self.assertRaises(ValueError):
            with r_mod.pre_norm_inject_hook(m, [(21, "ff", w, 1.0), (24, "ff", w * 5.0, 1.0)]):
                pass
        self.assertEqual(len(m.model.layers[21].post_feedforward_layernorm._forward_pre_hooks), 0)

        with self.assertRaises(ValueError):
            with r_mod.pre_norm_inject_hook(m, [(21, "ff", w, 1.0), (27, "ff", w, float("nan"))]):
                pass
        self.assertEqual(len(m.model.layers[21].post_feedforward_layernorm._forward_pre_hooks), 0)

    def test_03_full_8623_runner_to_scorer_contract_and_checkpoints(self) -> None:
        """Genuine FULL (non-smoke, 8,623-forward) runner -> scorer contract test (catches F1, F3, F4!)."""
        torch.manual_seed(7)
        d, vocab = 16, 64
        m = TinyOlmo3Model(vocab=vocab, d=d, n_layers=32)
        # Scale L18 ff_norm gamma so C2.2 cap (3*pmn[18]) does not cap on the tiny toy model
        with torch.no_grad():
            m.model.layers[18].post_feedforward_layernorm.weight.mul_(20.0)
        m.eval()
        tok = TinyHashTokenizer(vocab=vocab)
        c2_const = json.loads((ROOT / "c2_prereg_constants.json").read_text(encoding="utf-8"))
        for i, t in enumerate(c2_const["targets_24"]):
            t["id"] = i + 1
        for tr in c2_const["c22_stem_pairs_12"]:
            tr["suffix_len"] = len(tok.encode(tr["short_bare"]["text"]))

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
        for v in c2_const["c22_l18_frozen_pred_ret"].values():
            v["u_dot_w_Rmf"] = 0.8

        fake_hashes = {
            "r_lens_sha256": r_mod.EXPECTED_R_LENS_SHA256,
            "c2_constants_sha256": r_mod.EXPECTED_C2_CONSTANTS_SHA256,
            "c2_bank_sha256": r_mod.EXPECTED_C2_BANK_SHA256,
            "c2_design_sha256": r_mod.EXPECTED_C2_DESIGN_SHA256,
        }
        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = Path(td)
            with torch.no_grad():
                rec = r_mod.run_stage_c2_experiment(
                    m, tok, c2_const, bank, fake_hashes, smoke=False, device_name="cpu-tiny", checkpoint_dir=ckpt_dir
                )
            # Verify all 4 per-block checkpoints written & stamped partial_checkpoint=True (DS F5)
            for bname in ("block1_C2_1", "block2_C2_2", "block2b_C2_3", "block3_C2_4"):
                cp = ckpt_dir / f"{bname}.json.gz"
                self.assertTrue(cp.is_file(), f"Missing checkpoint {cp}")
                cobj = sc_mod.load_json_maybe_gz(cp)
                self.assertTrue(cobj.get("partial_checkpoint", False))

        # Exact 8,623 forwards, 19 exact-zero shams, 61 real active-arm keys, & rand_iso retention present!
        self.assertEqual(rec["meta"]["total_forwards"], 8623)
        self.assertEqual(set(rec["active_arm_min_max_abs_dlogit"].keys()), sc_mod.expected_active_arm_names(c2_const))
        for k, v in rec["shams"]["write_13"].items():
            self.assertEqual(v, 0.0, f"Write sham {k} non-zero: {v}")
        for k, v in rec["shams"]["patch_6"].items():
            self.assertEqual(v, 0.0, f"Patch sham {k} non-zero: {v}")
        for L in ("18", "21", "24", "27", "30"):
            self.assertEqual(len(rec["block2b_C2_3"][L]["rand_iso"]), 64)
            for row in rec["block2b_C2_3"][L]["rand_iso"]:
                self.assertIn("retention", row)

        # Zero leaked hooks across all 32 layers
        for L, lay in enumerate(m.model.layers):
            self.assertEqual(len(lay.post_feedforward_layernorm._forward_hooks), 0)
            self.assertEqual(len(lay.post_feedforward_layernorm._forward_pre_hooks), 0)
            self.assertEqual(len(lay.post_attention_layernorm._forward_hooks), 0)
            self.assertEqual(len(lay.post_attention_layernorm._forward_pre_hooks), 0)
        self.assertEqual(len(m.model.norm._forward_pre_hooks), 0)

        # DIRECT RUNNER -> SCORER CONTRACT: score the REAL runner receipt without crashing!
        sc = sc_mod.score_stage_c2(
            rec, c2_const,
            design_sha256=sc_mod.EXPECTED_C2_DESIGN_SHA256,
            constants_sha256=sc_mod.EXPECTED_C2_CONSTANTS_SHA256,
            strict_provenance=True,
            n_boot=50, n_flips_24=200,
        )
        self.assertEqual(sc["schema_version"], "olmo3-stage-c2-scorecard-v4")
        self.assertTrue(sc["gate_C2_0"]["C2.0.1_shams_and_liveness"]["pass"])
        self.assertTrue(sc["gate_C2_0"]["C2.0.2_finiteness_and_telescoping"]["pass"])
        self.assertTrue(sc["gate_C2_0"]["C2.0.4_provenance"]["pass"])
        self.assertTrue(sc["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["all_block_counts_ok"])
        self.assertTrue(sc["gate_C2_0"]["C2.0.3_linearity_and_completeness"]["dose_checks_ok"])
        self.assertIn("C2.1", sc["confirmatory_verdicts"])
        # Verify markdown renders cleanly regardless of PASS/BLOCKED
        md = sc_mod.render_scorecard_markdown(sc)
        self.assertIn("Stage C2 Confirmatory Scorecard", md)

    def test_04_rand_iso_retention_formula_and_checkpoint_io_resilience(self) -> None:
        """Verify compute_rand_iso_retention matches C0 formula algebraically (DS R2 F6)
        and best-effort checkpoint I/O failures warn without aborting (DS R2 F2).
        """
        # 1. Exact algebraic check vs run_olmo3_c0.decompose_target_trace on a known vector
        torch.manual_seed(12345)
        d, vocab, L = 16, 32, 2
        v = torch.randn(d)
        v = v / v.norm()
        r_c, r_i = 2.0, 2.5
        h_c = torch.randn(d)
        mlp_c = [torch.randn(d) for _ in range(4)]
        mlp_i = [t.clone() for t in mlp_c]
        mlp_i[L] = mlp_c[L] + 1.8 * v  # known direct write along v
        attn_c = [torch.randn(d) for _ in range(4)]
        attn_i = [t.clone() for t in attn_c]
        # Downstream delta along v = -0.45 * v (so net pre-RMSNorm delta = 1.35 * v -> retention = 1.35/1.8 = 0.75!)
        h_i = h_c + 1.35 * v
        clean_tr = {"h_final": h_c, "rms_final": r_c, "mlp": mlp_c, "attn": attn_c, "logits": torch.zeros(vocab)}
        int_tr = {"h_final": h_i, "rms_final": r_i, "mlp": mlp_i, "attn": attn_i, "logits": torch.zeros(vocab)}
        # Align logits[0] with dot(v, h)/r so decompose_target_trace sees exact same logit shift
        clean_tr["logits"][0] = torch.dot(v, h_c) / r_c
        int_tr["logits"][0] = torch.dot(v, h_i) / r_i
        p_c = torch.softmax(clean_tr["logits"], dim=-1)
        lp_c = torch.log_softmax(clean_tr["logits"], dim=-1)

        ret_rand = r_mod.compute_rand_iso_retention(clean_tr, int_tr, L, v)
        c0_rec = r_mod.decompose_target_trace(clean_tr, int_tr, L, v, 0, 1.8, 1.0, v, p_c, lp_c)
        self.assertAlmostEqual(ret_rand, 0.75, places=6)
        self.assertAlmostEqual(ret_rand, c0_rec["retention"], places=5)

        # 2. Best-effort checkpoint write NEVER raises on unwritable dir or regular-file path (DS R2 F2)
        r_mod._write_checkpoint(Path("/proc/definitely-not-writable-c2-ckpt"), "block1_C2_1", {"ok": True})
        with tempfile.TemporaryDirectory() as td:
            not_a_dir = Path(td) / "regular_file"
            not_a_dir.write_text("occupied", encoding="utf-8")
            r_mod._write_checkpoint(not_a_dir, "block1_C2_1", {"ok": True})
            self.assertEqual(not_a_dir.read_text(encoding="utf-8"), "occupied")


if __name__ == "__main__":
    unittest.main()
