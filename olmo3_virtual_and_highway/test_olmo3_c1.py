#!/usr/bin/env python3
"""Comprehensive CPU unit, tiny-model real-hook smoke, gate-sabotage, and /usr/bin/python3 -I
determinism test suite for OLMo 3 7B Stage C1 (`run_olmo3_c1.py` + `score_olmo3_c1.py`).
"""
from __future__ import annotations

import copy
import gzip
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from olmo3_virtual_and_highway import run_olmo3_c1 as runner
from olmo3_virtual_and_highway import score_olmo3_c1 as scorer

CONSTANTS_PATH = ROOT / "c1_prereg_constants.json"
DESIGN_PATH = ROOT / "OLMO3-TRANSPORT-C1-DESIGN.md"
BANK_PATH = REPO_ROOT / "work" / "olmo3_c1" / "c1_vector_bank.safetensors"


# ---------------------------------------------------------------------------
# Minimal OLMo-3-compatible PyTorch model for real-hook CPU smoke testing
# ---------------------------------------------------------------------------
class _TinyRMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.variance_epsilon) * self.weight


class _TinyOlmoLayer(nn.Module):
    def __init__(self, d: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.attn_proj = nn.Linear(d, d, bias=False)
        self.mlp_proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.attn_proj.weight.copy_(torch.randn(d, d, generator=g) * 0.08)
            self.mlp_proj.weight.copy_(torch.randn(d, d, generator=g) * 0.08)
        self.post_attention_layernorm = _TinyRMSNorm(d)
        self.post_feedforward_layernorm = _TinyRMSNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # Simple causal cross-position attention proxy so positions="all" genuinely differs from "last"
        b, t, d = h.shape
        prefix_mean = torch.cumsum(h, dim=1) / torch.arange(1, t + 1, device=h.device, dtype=h.dtype).view(1, t, 1)
        attn_out = self.post_attention_layernorm(self.attn_proj(prefix_mean))
        h = h + attn_out
        mlp_out = self.post_feedforward_layernorm(self.mlp_proj(torch.nn.functional.silu(h)))
        h = h + mlp_out
        return h


class _TinyOlmo3Model(nn.Module):
    def __init__(self, vocab: int = 64, d: int = 24, n_layers: int = 4):
        super().__init__()
        g = torch.Generator().manual_seed(777)
        self.config = SimpleNamespace(num_hidden_layers=n_layers, hidden_size=d, vocab_size=vocab)
        emb = nn.Embedding(vocab, d)
        with torch.no_grad():
            emb.weight.copy_(torch.randn(vocab, d, generator=g) * 0.5)
        layers = nn.ModuleList([_TinyOlmoLayer(d, seed=100 + i) for i in range(n_layers)])
        norm = _TinyRMSNorm(d)
        self.model = SimpleNamespace(embed_tokens=emb, layers=layers, norm=norm)
        self.embed_tokens = emb
        self.layers = layers
        self.norm = norm
        self.lm_head = nn.Linear(d, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, d, generator=g) * 0.5)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.embed_tokens(input_ids)
        for lay in self.layers:
            h = lay(h)
        h = self.norm(h)
        return SimpleNamespace(logits=self.lm_head(h))


class _TinyTokenizer:
    def encode(self, text: str, return_tensors: str | None = None, add_special_tokens: bool = False):
        ids = [(sum(ord(c) for c in tok) % 60) + 1 for tok in text.split()[:8]]
        if not ids:
            ids = [1, 2, 3]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids


# ---------------------------------------------------------------------------
# Synthetic full-shape (24t x 8p x 3L + L12) receipt builder for scorer tests
# ---------------------------------------------------------------------------
def make_clean_receipt(c1_const: dict) -> dict:
    """Build a structurally complete, Gate-C1.0-PASS synthetic Stage C1 receipt with controllable parameters."""
    targets = copy.deepcopy(c1_const["targets_24"])
    prompts = copy.deepcopy(c1_const["prompts_8"])
    t_strs = [t["string"] for t in targets]
    t_strata = [t["stratum"] for t in targets]
    n_t = len(targets)

    pmn = {"12": 4.2478, "18": 7.4537, "24": 12.1555, "27": 14.9000}
    anchor_dose_ratio = {18: 0.4809, 24: 0.2220, 27: 0.1642}
    clean_rms = {p["id"]: 1.0 for p in prompts}
    clean_probs = {
        p["id"]: {ts: {"p_clean": 1e-4, "log_p_clean": math.log(1e-4)} for ts in t_strs}
        for p in prompts
    }

    def _mk_trace(
        dose: float,
        exact_dlogit: float,
        pred_dlogit: float,
        sum_attn: float,
        sum_mlp: float = 0.2,
        rmsn_shift: float = 0.0,
        rms_clean: float = 1.0,
        rms_int: float = 1.0,
        fde: float = 3.0,
        tele_err: float = 0.005,
        retention: float = 0.60,
        pid: str = "p",
    ) -> dict:
        return {
            "prompt_id": pid,
            "capped": False,
            "dose": dose,
            "exact_dlogit": exact_dlogit,
            "pred_dlogit": pred_dlogit,
            "direct_linear": 3.0,
            "direct_linear_analytic": 3.0,
            "frozen_direct_effect": fde,
            "sum_attn": sum_attn,
            "sum_mlp": sum_mlp,
            "rmsnorm_shift": rmsn_shift,
            "telescoping_err": tele_err,
            "retention": retention,
            "canc_r": 1.2,
            "dlogit_z": 4.0,
            "kl": 0.1,
            "rms_clean": rms_clean,
            "rms_int": rms_int,
        }

    block1_all: dict = {}
    block1_last: dict = {}
    block2a: dict = {}
    block2b: dict = {}

    for L in scorer.PRIMARY_LAYERS:
        ls = str(L)
        dose_d = anchor_dose_ratio[L] * pmn[ls]
        dose_n = 0.75 * pmn[ls]
        b1a = {"positions": "all", "target_direct_linear": 3.0, "dose_cap": 3.0 * pmn[ls], "R_mf": {}, "direct_matched": {}}
        b1l = {"positions": "last", "target_direct_linear": 3.0, "dose_cap": 3.0 * pmn[ls], "R_mf": {}, "direct_matched": {}}
        b2a = {"positions": "last", "dose_N": dose_n, "post_mlp_norm": pmn[ls], "R_mf": {}, "R_ridge": {}, "logit_lens": {}}

        for ti, ts in enumerate(t_strs):
            # Slight per-target deterministic jitter so BCa bootstrap is non-degenerate
            jit = 0.02 * ((ti % 6) - 2.5)
            pred_rmf = 6.5 + 0.1 * (ti % 4)
            pred_dm = 2.0
            # All mode: exact ~ 1.15 * pred_rmf (frac_err ~ 0.10 <= 0.18), gap ~ 4.9 -> r_all ~ 1.09
            rmf_all_ex = pred_rmf * (1.08 + jit * 0.3)
            dm_all_ex = 1.9 + jit * 0.2
            rmf_last_ex = pred_rmf * (0.65 + jit * 0.2)
            dm_last_ex = 2.5 + jit * 0.1
            b1a["R_mf"][ts] = [
                _mk_trace(dose_d, rmf_all_ex, pred_rmf, sum_attn=2.9 + jit, pid=p["id"])
                for p in prompts
            ]
            b1l["R_mf"][ts] = [
                _mk_trace(dose_d, rmf_last_ex, pred_rmf, sum_attn=0.4 + jit * 0.2, pid=p["id"])
                for p in prompts
            ]
            b1a["direct_matched"][ts] = [
                [_mk_trace(dose_d, dm_all_ex, pred_dm, sum_attn=-0.9 + jit * 0.1, pid=p["id"]) for p in prompts]
                for _ in range(2)
            ]
            b1l["direct_matched"][ts] = [
                [_mk_trace(dose_d, dm_last_ex, pred_dm, sum_attn=0.1, pid=p["id"]) for p in prompts]
                for _ in range(2)
            ]

            # Block 2a: default H_A-compatible values (strong whitening ~2.2x, delta_R_sep ~ +0.45, R_floor ~ 1.50)
            st_i = t_strata[ti]
            for arm in ("R_mf", "R_ridge", "logit_lens"):
                rows = []
                for pi, p in enumerate(prompts):
                    a24 = []
                    for tj in range(n_t):
                        if tj == ti:
                            val = 12.0 if arm != "logit_lens" else 8.0
                        else:
                            same = t_strata[tj] == st_i
                            opfn = (t_strata[tj] == "function") != (st_i == "function")
                            if arm == "R_mf":
                                val = 1.20 + 0.05 * (ti % 5) if same else (1.05 + 0.04 * (ti % 5) if opfn else 1.10)
                            elif arm == "R_ridge":
                                # R_sep(R_mf) ~ 1.14, R_sep(R_ridge) ~ 0.72/0.44 = 1.64 -> d_sep ~ +0.50, floor ~ 0.44/0.30 = 1.47
                                val = 0.72 + 0.03 * (ti % 5) if same else (0.44 + 0.02 * (ti % 5) if opfn else 0.52)
                            else:
                                val = 0.30 + 0.01 * (ti % 3)
                        a24.append(round(val, 6))
                    ret_val = 0.55 + 0.01 * ((ti % 5) - 2) if arm == "logit_lens" else 0.75
                    rec = _mk_trace(dose_n, a24[ti], a24[ti], sum_attn=1.0, retention=ret_val, pid=p["id"])
                    rec["all_24_dlogits"] = a24
                    if arm == "logit_lens":
                        rec["u_hat_self_J"] = ret_val
                        rec["u_hat_off_abs_mean_J"] = 0.10
                    rows.append(rec)
                b2a[arm][ts] = rows

        # Block 2b: F_mean_3L ~ 0.300 (ratio 1.014x C0's 0.2958, clean!), vJv = 0.68 -> d_t ~ +0.13
        rand_draws = []
        for m in range(32):
            draw_rows = []
            for p in prompts:
                draw_rows.append({
                    "prompt_id": p["id"],
                    "dose": dose_n,
                    "vJv": 0.68,
                    "telescoping_err": 0.004,
                    "canc_r_self": 1.1,
                    "rms_clean": 1.0,
                    "rms_int": 1.0,
                    "all_24_dlogits": [0.30] * n_t,
                })
            rand_draws.append(draw_rows)
        block2b[ls] = {"positions": "last", "dose_N": dose_n, "post_mlp_norm": pmn[ls], "m_rand": 32, "rand_iso": rand_draws}
        block1_all[ls] = b1a
        block1_last[ls] = b1l
        block2a[ls] = b2a

    # Secondary L12 (clean PASS by default)
    dose_12 = 0.30 * pmn["12"]
    sec12 = {
        "dose_multiplier": 0.30,
        "dose_N": dose_12,
        "post_mlp_norm": pmn["12"],
        "R_mf_all": {ts: [_mk_trace(dose_12, 3.1, 3.0, 1.0, rms_int=1.03, pid=p["id"]) for p in prompts] for ts in t_strs},
        "R_mf_last": {ts: [{**_mk_trace(dose_12, 2.5, 3.0, 0.2, pid=p["id"]), "all_24_dlogits": [0.5] * n_t} for p in prompts] for ts in t_strs},
        "R_ridge_last": {ts: [{**_mk_trace(dose_12, 2.3, 2.5, 0.2, pid=p["id"]), "all_24_dlogits": [0.3] * n_t} for p in prompts] for ts in t_strs},
    }

    return {
        "schema_version": "c1-receipt-v1",
        "meta": {
            "study": "test",
            "model_id": runner.MODEL_ID,
            "model_rev": runner.MODEL_REV,
            "smoke": False,
            "device": "cpu",
            "torch_version": torch.__version__,
            "primary_layers": list(scorer.PRIMARY_LAYERS),
            "secondary_layers": list(scorer.SECONDARY_LAYERS),
            "k_dm": 2,
            "m_rand": 32,
            "n_targets": 24,
            "n_prompts": 8,
            "r_lens_sha256": scorer.EXPECTED_R_LENS_SHA256,
            "c1_constants_sha256": scorer.EXPECTED_C1_CONSTANTS_SHA256,
            "c1_bank_sha256": scorer.EXPECTED_C1_BANK_SHA256,
            "c1_design_sha256": scorer.EXPECTED_C1_DESIGN_SHA256,
        },
        "targets_24": targets,
        "prompts_8": prompts,
        "post_mlp_norms": pmn,
        "clean_rms_final": clean_rms,
        "clean_target_probs": clean_probs,
        "shams": {"last": {"18": 0.0, "24": 0.0, "27": 0.0}, "all": {"18": 0.0, "24": 0.0, "27": 0.0, "12": 0.0}},
        "sham_movement_check": {
            "last": {"18": 2.1, "24": 2.4, "27": 2.8, "12": 1.5},
            "all": {"18": 5.2, "24": 5.8, "27": 6.1, "12": 3.0},
        },
        "block1_regime_D_all": block1_all,
        "block1_regime_D_last": block1_last,
        "block2a_regime_N_last": block2a,
        "block2b_rand_iso": block2b,
        "secondary_L12": {"12": sec12},
    }


def _set_c1_3_bleeds(
    rec: dict,
    mf_in: float,
    mf_of: float,
    rg_in: float,
    rg_of: float,
    ll_of: float = 0.30,
    f_rand: float = 0.30,
) -> None:
    """Mutate Block 2a & Block 2b bleeds cleanly across all 24 targets with tiny deterministic jitter."""
    targets = rec["targets_24"]
    t_strs = [t["string"] for t in targets]
    t_strata = [t["stratum"] for t in targets]
    n_t = len(targets)
    for L in scorer.PRIMARY_LAYERS:
        ls = str(L)
        for ti, ts in enumerate(t_strs):
            st_i = t_strata[ti]
            jit = 1.0 + 0.02 * ((ti % 5) - 2)
            for arm, v_in, v_of in (
                ("R_mf", mf_in, mf_of),
                ("R_ridge", rg_in, rg_of),
                ("logit_lens", ll_of, ll_of),
            ):
                for row in rec["block2a_regime_N_last"][ls][arm][ts]:
                    a24 = []
                    for tj in range(n_t):
                        if tj == ti:
                            a24.append(12.0 if arm != "logit_lens" else 8.0)
                        elif t_strata[tj] == st_i:
                            a24.append(round(v_in * jit, 6))
                        elif (t_strata[tj] == "function") != (st_i == "function"):
                            a24.append(round(v_of * jit, 6))
                        else:
                            a24.append(round(0.5 * (v_in + v_of) * jit, 6))
                    row["all_24_dlogits"] = a24
                    row["exact_dlogit"] = a24[ti]
        for draw in rec["block2b_rand_iso"][ls]["rand_iso"]:
            for row in draw:
                row["all_24_dlogits"] = [f_rand] * n_t


class TestStageC1(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.c1_const = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))

    # 1. Real tiny-model PyTorch end-to-end smoke + checkpoint provenance + scorer refusal
    def test_01_tiny_model_real_hooks_end_to_end_and_smoke_refusal(self) -> None:
        d = 24
        model = _TinyOlmo3Model(vocab=64, d=d, n_layers=5)
        model.eval()
        tok = _TinyTokenizer()
        g = torch.Generator().manual_seed(123)
        targets = [
            {"stratum": st, "string": f" t{i}", "id": 10 + i}
            for i, st in enumerate(("nouns", "verbs", "adjectives", "function"))
        ]
        tids = torch.tensor([t["id"] for t in targets], dtype=torch.long)
        u_vecs = (model.lm_head.weight[tids] * model.norm.weight.unsqueeze(0)).detach().float()
        u_hat = u_vecs / u_vecs.norm(dim=-1, keepdim=True)
        # Ensure positive u_t . w for clean Regime D dose
        def _pos_unit(base: torch.Tensor, noise_scale: float) -> torch.Tensor:
            w = base + noise_scale * torch.randn(base.shape, generator=g)
            return w / w.norm(dim=-1, keepdim=True)

        layers_bank = {}
        for L in (0, 1, 2, 3):
            w_mf = _pos_unit(u_hat, 0.2)
            w_rg = _pos_unit(u_hat, 0.25)
            w_dm = torch.stack([_pos_unit(u_hat, 0.3)], dim=1)  # [4, 1, d]
            w_rd = torch.randn(2, d, generator=g)
            w_rd = w_rd / w_rd.norm(dim=-1, keepdim=True)
            pr = {t["string"]: {"R_mf": 0.8, "R_ridge": 0.7, "logit_lens": 0.6, "direct_matched": [0.5]} for t in targets}
            layers_bank[L] = {
                "R_mf": w_mf,
                "R_ridge": w_rg,
                "logit_lens": u_hat,
                "direct_matched": w_dm if L in (1, 2, 3) else None,
                "rand_iso": w_rd if L in (1, 2, 3) else None,
                "pred_ret": pr,
            }
        bank = {"meta": {"model": runner.MODEL_ID, "revision": runner.MODEL_REV}, "u_vectors": u_vecs, "u_hat": u_hat, "targets": targets, "layers_bank": layers_bank}
        fake_const = {"prompts_8": self.c1_const["prompts_8"][:2]}
        fake_hashes = {
            "r_lens_sha256": "a" * 64,
            "c1_constants_sha256": "b" * 64,
            "c1_bank_sha256": "c" * 64,
            "c1_design_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            # F2 + F3: spy on post_norm_inject_hook to pin exact (layer, positions, dose) calls
            hook_calls: list[tuple[int, str, float]] = []
            orig_hook = runner.post_norm_inject_hook

            def _spy_hook(m, layer, w, dose, positions):
                hook_calls.append((int(layer), str(positions), float(dose)))
                return orig_hook(m, layer, w, dose, positions)

            runner.post_norm_inject_hook = _spy_hook
            try:
                rec = runner.run_c1(
                    model, tok, bank, fake_const, out_dir, hashes=fake_hashes, device="cpu",
                    smoke=True, primary_layers=[1, 2, 3], secondary_layers=[0],
                )
            finally:
                runner.post_norm_inject_hook = orig_hook

            self.assertTrue((out_dir / "c1_receipt.json.gz").exists())
            # F2: verify exact (layer, positions) sets & counts across Block 1, 2a, 2b, secondary L0
            nonzero_calls = [(L, pos) for L, pos, dose in hook_calls if dose != 0.0]
            self.assertTrue({(1, "all"), (1, "last"), (2, "all"), (2, "last"), (3, "all"), (3, "last"), (0, "all"), (0, "last")} <= set(nonzero_calls))
            self.assertTrue(all(L in (0, 1, 2, 3) for L, _ in nonzero_calls))
            # Exact per-layer call counts in smoke (4t x 2p = 8 per arm; k_dm=1 -> 16 all + 16 last Block1; 24 last Block2a; 4 last Block2b => 44 last, 16 all per primary L!)
            for L in (1, 2, 3):
                self.assertEqual(sum(1 for l_c, p_c in nonzero_calls if l_c == L and p_c == "all"), 16)
                self.assertEqual(sum(1 for l_c, p_c in nonzero_calls if l_c == L and p_c == "last"), 44)
            # Secondary L0: 8 all (R_mf) + 16 last (R_mf + R_ridge)
            self.assertEqual(sum(1 for l_c, p_c in nonzero_calls if l_c == 0 and p_c == "all"), 8)
            self.assertEqual(sum(1 for l_c, p_c in nonzero_calls if l_c == 0 and p_c == "last"), 16)

            # Exact telescoping identity, per-cell all != last, and registered dose constants (F2 + F3 + F8)
            for L_int in (1, 2, 3):
                L = str(L_int)
                pmn_l = float(rec["post_mlp_norms"][L])
                self.assertEqual(rec["shams"]["last"][L], 0.0)
                self.assertEqual(rec["shams"]["all"][L], 0.0)
                self.assertGreater(rec["sham_movement_check"]["all"][L], 0.0)
                self.assertGreater(rec["sham_movement_check"]["last"][L], 0.0)
                self.assertAlmostEqual(rec["block1_regime_D_all"][L]["dose_cap"], 3.0 * pmn_l, delta=1e-4)
                self.assertAlmostEqual(rec["block1_regime_D_last"][L]["dose_cap"], 3.0 * pmn_l, delta=1e-4)
                self.assertAlmostEqual(rec["block2a_regime_N_last"][L]["dose_N"], 0.75 * pmn_l, delta=1e-4)
                self.assertAlmostEqual(rec["block2b_rand_iso"][L]["dose_N"], 0.75 * pmn_l, delta=1e-4)
                diff_cells = 0
                for ti, t in enumerate(targets):
                    ts = t["string"]
                    u_dot_w = float(torch.dot(u_vecs[ti], layers_bank[L_int]["R_mf"][ti]).item())
                    for pi, row_a in enumerate(rec["block1_regime_D_all"][L]["R_mf"][ts]):
                        row_l = rec["block1_regime_D_last"][L]["R_mf"][ts][pi]
                        if not row_a["capped"]:
                            self.assertLess(row_a["telescoping_err"], 1e-4)
                            # F3: exact Direct_linear = 3.0 dose
                            self.assertAlmostEqual(row_a["direct_linear_analytic"], 3.0, delta=1e-3)
                            self.assertAlmostEqual(row_a["dose"], 3.0 * row_a["rms_clean"] / u_dot_w, delta=1e-3)
                        if not row_l["capped"]:
                            self.assertAlmostEqual(row_l["direct_linear_analytic"], 3.0, delta=1e-3)
                        if (not row_a["capped"]) and (not row_l["capped"]):
                            if abs(row_a["exact_dlogit"] - row_l["exact_dlogit"]) > 1e-4:
                                diff_cells += 1
                    for row in rec["block2a_regime_N_last"][L]["logit_lens"][ts]:
                        self.assertAlmostEqual(row["dose"], 0.75 * pmn_l, delta=1e-4)
                        self.assertIn("u_hat_self_J", row)
                        self.assertIn("u_hat_off_abs_mean_J", row)
                        self.assertAlmostEqual(row["u_hat_self_J"], row["retention"], delta=1e-3)
                        self.assertEqual(len(row["all_24_dlogits"]), 4)
                # Causal prefix-mean tiny model guarantees positions="all" != "last" on real hooks
                self.assertGreater(diff_cells, 0, f"Block 1 all vs last identical at L{L}")
                for draw in rec["block2b_rand_iso"][L]["rand_iso"]:
                    for row in draw:
                        self.assertAlmostEqual(row["dose"], 0.75 * pmn_l, delta=1e-4)
            self.assertEqual(rec["shams"]["all"]["0"], 0.0)
            self.assertAlmostEqual(rec["secondary_L12"]["0"]["dose_N"], 0.30 * float(rec["post_mlp_norms"]["0"]), delta=1e-4)
            # Verify checkpoint resume succeeds cleanly
            rec2 = runner.run_c1(
                model, tok, bank, fake_const, out_dir, hashes=fake_hashes, device="cpu",
                smoke=True, primary_layers=[1, 2, 3], secondary_layers=[0],
            )
            self.assertEqual(rec2["meta"]["total_forwards"], rec["meta"]["total_forwards"])
            # Tampered checkpoint provenance must fail-closed
            ck_p = out_dir / "ckpt_c1_L1.json"
            ck = json.loads(ck_p.read_text())
            ck["provenance"]["c1_design_sha256"] = "0" * 64
            ck_p.write_text(json.dumps(ck))
            with self.assertRaises(RuntimeError):
                runner.run_c1(
                    model, tok, bank, fake_const, out_dir, hashes=fake_hashes, device="cpu",
                    smoke=True, primary_layers=[1, 2, 3], secondary_layers=[0],
                )
            # Scorer must refuse smoke receipt fail-closed
            with self.assertRaises(ValueError):
                scorer.score_c1(rec, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH)

    # 2a. F5: weight-free preflight provenance test (ALWAYS runs in any worktree)
    def test_02a_preflight_provenance_weight_free(self) -> None:
        import struct

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            r_lens_f = tdp / "r_lens.pt"
            r_lens_f.write_bytes(b"fake-r-lens-bytes")
            r_sha = runner.sha256_file(r_lens_f)

            # Build a minimal valid safetensors file with required metadata
            meta = {
                "r_lens_sha256": r_sha,
                "model": runner.MODEL_ID,
                "revision": runner.MODEL_REV,
                "k_dm_draws": str(runner.EXPECTED_K_DM),
                "m_rand_iso_draws": str(runner.EXPECTED_M_RAND),
            }
            hdr = json.dumps({"__metadata__": meta, "x": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode("utf-8")
            bank_f = tdp / "bank.safetensors"
            bank_f.write_bytes(struct.pack("<Q", len(hdr)) + hdr + struct.pack("<f", 1.0))
            bank_sha = runner.sha256_file(bank_f)

            const_f = tdp / "const.json"
            const_f.write_text(json.dumps({"c1_vector_bank_sha256": bank_sha, "r_lens_sha256": r_sha}))
            const_sha = runner.sha256_file(const_f)

            design_f = tdp / "design.md"
            design_f.write_text("# Stage C1 test design\n")
            design_sha = runner.sha256_file(design_f)

            orig = (
                runner.EXPECTED_R_LENS_SHA256,
                runner.EXPECTED_C1_CONSTANTS_SHA256,
                runner.EXPECTED_C1_BANK_SHA256,
                runner.EXPECTED_C1_DESIGN_SHA256,
            )
            try:
                runner.EXPECTED_R_LENS_SHA256 = r_sha
                runner.EXPECTED_C1_CONSTANTS_SHA256 = const_sha
                runner.EXPECTED_C1_BANK_SHA256 = bank_sha
                runner.EXPECTED_C1_DESIGN_SHA256 = design_sha
                out = runner.verify_preflight_provenance(bank_f, r_lens_f, const_f, design_f)
                self.assertEqual(out["c1_design_sha256"], design_sha)

                # Flipping design hash must raise ValueError
                runner.EXPECTED_C1_DESIGN_SHA256 = "0" * 64
                with self.assertRaises(ValueError):
                    runner.verify_preflight_provenance(bank_f, r_lens_f, const_f, design_f)
                runner.EXPECTED_C1_DESIGN_SHA256 = design_sha

                # Flipping bank hash must raise ValueError
                runner.EXPECTED_C1_BANK_SHA256 = "1" * 64
                with self.assertRaises(ValueError):
                    runner.verify_preflight_provenance(bank_f, r_lens_f, const_f, design_f)
            finally:
                (
                    runner.EXPECTED_R_LENS_SHA256,
                    runner.EXPECTED_C1_CONSTANTS_SHA256,
                    runner.EXPECTED_C1_BANK_SHA256,
                    runner.EXPECTED_C1_DESIGN_SHA256,
                ) = orig

    # 2b. Real disk artifacts preflight check (explicitly skipped rather than vacuous when r_lens.pt is absent)
    def test_02b_preflight_provenance_real_artifacts(self) -> None:
        import os

        env_rl = os.environ.get("R_LENS_PATH")
        r_lens_path = Path(env_rl) if env_rl else (REPO_ROOT / "work" / "vesper-olmo3-7b-neuron-lens" / "r_lens.pt")
        if not (r_lens_path.exists() and BANK_PATH.exists()):
            self.skipTest("frozen r_lens.pt / c1_vector_bank.safetensors not materialised in this worktree")
        h = runner.verify_preflight_provenance(BANK_PATH, r_lens_path, CONSTANTS_PATH, DESIGN_PATH)
        self.assertEqual(h["c1_design_sha256"], runner.EXPECTED_C1_DESIGN_SHA256)
        with tempfile.TemporaryDirectory() as td:
            bad_design = Path(td) / "bad.md"
            bad_design.write_text("tampered\n")
            with self.assertRaises(ValueError):
                runner.verify_preflight_provenance(BANK_PATH, r_lens_path, CONSTANTS_PATH, bad_design)

    # 2c. F1: scorer refuses receipts with mutated intervention semantics
    def test_02c_f1_receipt_semantics_refusals(self) -> None:
        # Block 1 all<->last swapped
        r1 = make_clean_receipt(self.c1_const)
        r1["block1_regime_D_all"]["18"]["positions"] = "last"
        with self.assertRaises(ValueError):
            scorer.score_c1(r1, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # Block 2a positions='all'
        r2 = make_clean_receipt(self.c1_const)
        r2["block2a_regime_N_last"]["18"]["positions"] = "all"
        with self.assertRaises(ValueError):
            scorer.score_c1(r2, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # Block 2a dose_N = 0.30*pmn
        r3 = make_clean_receipt(self.c1_const)
        r3["block2a_regime_N_last"]["18"]["dose_N"] = 0.30 * r3["post_mlp_norms"]["18"]
        with self.assertRaises(ValueError):
            scorer.score_c1(r3, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # Block 1 target_direct_linear = 2.0
        r4 = make_clean_receipt(self.c1_const)
        r4["block1_regime_D_all"]["24"]["target_direct_linear"] = 2.0
        with self.assertRaises(ValueError):
            scorer.score_c1(r4, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # Implausible meta.total_forwards = 1
        r5 = make_clean_receipt(self.c1_const)
        r5["meta"]["total_forwards"] = 1
        with self.assertRaises(ValueError):
            scorer.score_c1(r5, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # F-7 (N11): block1 dose_cap != 3.0*pmn
        r6 = make_clean_receipt(self.c1_const)
        r6["block1_regime_D_all"]["18"]["dose_cap"] = 30.0 * r6["post_mlp_norms"]["18"]
        with self.assertRaises(ValueError):
            scorer.score_c1(r6, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # F-7 (N13): block2b dose_N != 0.75*pmn
        r7 = make_clean_receipt(self.c1_const)
        r7["block2b_rand_iso"]["24"]["dose_N"] = 1.50 * r7["post_mlp_norms"]["24"]
        with self.assertRaises(ValueError):
            scorer.score_c1(r7, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        # F-7 (N12): secondary L12 dose_multiplier != 0.30 and dose_N != 0.30*pmn_12
        r8 = make_clean_receipt(self.c1_const)
        r8["secondary_L12"]["12"]["dose_multiplier"] = 0.50
        with self.assertRaises(ValueError):
            scorer.score_c1(r8, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        r9 = make_clean_receipt(self.c1_const)
        r9["secondary_L12"]["12"]["dose_N"] = 0.75 * r9["post_mlp_norms"]["12"]
        with self.assertRaises(ValueError):
            scorer.score_c1(r9, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)

    # 3. Clean baseline passes C1.0 + all 4 claims (H_A + UNIVERSAL_4_OF_4) + L12
    def test_03_clean_baseline_all_pass(self) -> None:
        rec = make_clean_receipt(self.c1_const)
        sc = scorer.score_c1(
            rec, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH,
            n_flips=2000, n_boot=400,
        )
        self.assertEqual(sc["verdicts"]["C1.0"], "PASS")
        self.assertEqual(sc["verdicts"]["C1.1"], "PASS")
        self.assertEqual(sc["verdicts"]["C1.2"], "PASS")
        self.assertEqual(sc["verdicts"]["C1.3"], "H_A")
        self.assertEqual(sc["verdicts"]["C1.4"], "PASS (>= 19/24 & >= 3/4 strata; UNIVERSAL_4_OF_4)")
        self.assertEqual(sc["verdicts"]["L12_linear_check"], "PASS")

    # 4. EVERY Gate C1.0 clause shown RED under targeted sabotage (blocking all 4 claims)
    def test_04_gate_c1_0_1_shams_and_liveness_sabotages(self) -> None:
        # (a) Nonzero sham
        r1 = make_clean_receipt(self.c1_const)
        r1["shams"]["all"]["18"] = 1e-6
        sc1 = scorer.score_c1(r1, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc1["gate_C1_0"]["C1.0.1_shams_and_liveness"]["PASS"])
        self.assertEqual(sc1["verdicts"]["C1.1"], "BLOCKED_BY_C1.0")
        # (b) Zero R_mf movement
        r2 = make_clean_receipt(self.c1_const)
        r2["sham_movement_check"]["last"]["27"] = 0.0
        sc2 = scorer.score_c1(r2, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc2["gate_C1_0"]["C1.0.1_shams_and_liveness"]["PASS"])
        self.assertEqual(sc2["verdicts"]["C1.3"], "BLOCKED_BY_C1.0")

    def test_05_gate_c1_0_2_finite_and_telescoping_sabotages(self) -> None:
        # (a) NaN inside all_24_dlogits
        r1 = make_clean_receipt(self.c1_const)
        ts0 = r1["targets_24"][0]["string"]
        r1["block2a_regime_N_last"]["18"]["R_ridge"][ts0][0]["all_24_dlogits"][3] = float("nan")
        sc1 = scorer.score_c1(r1, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc1["gate_C1_0"]["C1.0.2_finite_and_telescoping"]["PASS"])
        self.assertEqual(sc1["verdicts"]["C1.2"], "BLOCKED_BY_C1.0")
        # (b) p99.9 > 0.15 while p95 <= 0.08 (5,952 primary traces -> >= 7 exceedances push p99.9 > 0.15)
        r2 = make_clean_receipt(self.c1_const)
        for ti in range(8):
            r2["block1_regime_D_all"]["18"]["R_mf"][r2["targets_24"][ti]["string"]][0]["telescoping_err"] = 0.25
        sc2 = scorer.score_c1(r2, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc2["gate_C1_0"]["C1.0.2_finite_and_telescoping"]["PASS"])
        self.assertEqual(sc2["verdicts"]["C1.4"], "BLOCKED_BY_C1.0")

    def test_06_gate_c1_0_3_all_subclauses_sabotages(self) -> None:
        ts_all = [t["string"] for t in self.c1_const["targets_24"]]
        # (a) Anchor off by > 30% on L18 R_mf
        r_a = make_clean_receipt(self.c1_const)
        for ts in ts_all:
            for row in r_a["block1_regime_D_all"]["18"]["R_mf"][ts]:
                row["dose"] *= 1.40
        sc_a = scorer.score_c1(r_a, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc_a["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["anchor_all_ok"])
        self.assertEqual(sc_a["verdicts"]["C1.0"], "FAIL")

        # (b) 3/1728 traces with |r_i/r_c - 1| > 0.20 (bar <= 2)
        r_b = make_clean_receipt(self.c1_const)
        for ti in range(3):
            r_b["block1_regime_D_all"]["18"]["R_mf"][ts_all[ti]][0]["rms_int"] = 0.78
        sc_b = scorer.score_c1(r_b, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc_b["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["PASS"])
        self.assertEqual(sc_b["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["rmsn_gt_20pct_count"], 3)

        # (c) 27/1728 traces with |r_i/r_c - 1| > 0.12 (bar <= 26)
        r_c = make_clean_receipt(self.c1_const)
        for idx in range(27):
            r_c["block1_regime_D_all"]["18"]["R_mf"][ts_all[idx % 24]][idx // 24]["rms_int"] = 0.85
        sc_c = scorer.score_c1(r_c, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc_c["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["PASS"])
        self.assertEqual(sc_c["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["rmsn_gt_12pct_count"], 27)

        # (d) 13/1728 traces with |fde - 3.0| > 0.45 (bar <= 12)
        r_d = make_clean_receipt(self.c1_const)
        for idx in range(13):
            r_d["block1_regime_D_all"]["18"]["R_mf"][ts_all[idx % 24]][idx // 24]["frozen_direct_effect"] = 3.55
        sc_d = scorer.score_c1(r_d, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc_d["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["PASS"])
        self.assertEqual(sc_d["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["fde_gt_045_count"], 13)

        # (e) 3/1728 traces with |fde - 3.0| > 0.75 (bar <= 2)
        r_e = make_clean_receipt(self.c1_const)
        for ti in range(3):
            r_e["block1_regime_D_all"]["18"]["R_mf"][ts_all[ti]][0]["frozen_direct_effect"] = 3.85
        sc_e = scorer.score_c1(r_e, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc_e["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["PASS"])
        self.assertEqual(sc_e["gate_C1_0"]["C1.0.3_regime_D_all_linearity"]["fde_gt_075_count"], 3)

    def test_07_gate_c1_0_4_provenance_sabotages(self) -> None:
        r = make_clean_receipt(self.c1_const)
        # Missing design path fails C1.0.4
        sc1 = scorer.score_c1(r, self.c1_const, constants_path=CONSTANTS_PATH, design_path=None, n_flips=200, n_boot=100)
        self.assertFalse(sc1["gate_C1_0"]["C1.0.4_provenance"]["PASS"])
        self.assertEqual(sc1["verdicts"]["C1.1"], "BLOCKED_BY_C1.0")
        # strict_provenance=False fails C1.0.4
        sc2 = scorer.score_c1(r, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, strict_provenance=False, n_flips=200, n_boot=100)
        self.assertFalse(sc2["gate_C1_0"]["C1.0.4_provenance"]["PASS"])
        # Tampered receipt design hash fails C1.0.4
        r["meta"]["c1_design_sha256"] = "0" * 64
        sc3 = scorer.score_c1(r, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertFalse(sc3["gate_C1_0"]["C1.0.4_provenance"]["PASS"])

    # 5. Claim C1.1 M1 cases (PASS / INCONCLUSIVE / REFUTED) + C1-6 exclusion guard
    def test_08_c1_1_m1_verdicts_and_dtot_exclusion(self) -> None:
        def _with_fmt_r_all(bare_r: float, chat_r: float) -> dict:
            r = make_clean_receipt(self.c1_const)
            for L in scorer.PRIMARY_LAYERS:
                ls = str(L)
                for ts in [t["string"] for t in r["targets_24"]]:
                    for pi, p in enumerate(r["prompts_8"]):
                        target_r = bare_r if p["format"] == "bare" else chat_r
                        rmf = r["block1_regime_D_all"][ls]["R_mf"][ts][pi]
                        dm0 = r["block1_regime_D_all"][ls]["direct_matched"][ts][0][pi]
                        dm1 = r["block1_regime_D_all"][ls]["direct_matched"][ts][1][pi]
                        pg = rmf["pred_dlogit"] - 0.5 * (dm0["pred_dlogit"] + dm1["pred_dlogit"])
                        # Keep frac_err_all small and share_gap ~ 0.88 while shifting r_all cleanly
                        rmf["exact_dlogit"] = rmf["pred_dlogit"] * 1.05
                        dm_ex = rmf["exact_dlogit"] - target_r * pg
                        dm0["exact_dlogit"] = dm_ex
                        dm1["exact_dlogit"] = dm_ex
                        gap_last = (
                            r["block1_regime_D_last"][ls]["R_mf"][ts][pi]["exact_dlogit"]
                            - 0.5 * (
                                r["block1_regime_D_last"][ls]["direct_matched"][ts][0][pi]["exact_dlogit"]
                                + r["block1_regime_D_last"][ls]["direct_matched"][ts][1][pi]["exact_dlogit"]
                            )
                        )
                        dtot_g = (target_r * pg) - gap_last
                        rmf["sum_attn"] = 0.4 + 0.88 * max(dtot_g, 2.0) - 0.9
                        dm0["sum_attn"] = -0.9
                        dm1["sum_attn"] = -0.9
            return r

        # Bare 1.45, ChatML 1.40 -> PASS (zero M1 contradiction)
        sc_pass = scorer.score_c1(_with_fmt_r_all(1.45, 1.40), self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_pass["verdicts"]["C1.1"], "PASS")
        # Bare 1.90, ChatML 0.70 -> REFUTED (M1 one-format miss MUST refute!)
        sc_ref = scorer.score_c1(_with_fmt_r_all(1.90, 0.70), self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_ref["verdicts"]["C1.1"], "REFUTED")
        # Bare 1.42, ChatML 0.98 -> INCONCLUSIVE (both in [0.65,1.50], split 0.44 > 0.35)
        sc_inc = scorer.score_c1(_with_fmt_r_all(1.42, 0.98), self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_inc["verdicts"]["C1.1"], "INCONCLUSIVE (format/length/instruction split)")
        # C1-6 guard: >= 3/24 targets with delta_tot_gap <= 1e-6 auto-fails (ii) -> REFUTED
        r_excl = make_clean_receipt(self.c1_const)
        for ti in range(3):
            ts = r_excl["targets_24"][ti]["string"]
            for L in scorer.PRIMARY_LAYERS:
                ls = str(L)
                for pi in range(8):
                    r_excl["block1_regime_D_last"][ls]["R_mf"][ts][pi]["exact_dlogit"] = r_excl["block1_regime_D_all"][ls]["R_mf"][ts][pi]["exact_dlogit"]
                    for k in range(2):
                        r_excl["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi]["exact_dlogit"] = r_excl["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi]["exact_dlogit"]
        sc_excl = scorer.score_c1(r_excl, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_excl["C1_1"]["b_attention_attribution"]["excluded_dtot_le_1e6_count"], 3)
        self.assertFalse(sc_excl["C1_1"]["b_attention_attribution"]["ii_ok"])
        self.assertEqual(sc_excl["verdicts"]["C1.1"], "REFUTED")

    # 6. Claim C1.2 4-way verdicts + L6 BCa containment invariant
    def test_09_c1_2_all_4_verdicts_and_bca_containment(self) -> None:
        # REVERSED (R_ridge worse than R_mf: ratio 0.65 < 0.80)
        r_rev = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r_rev, mf_in=0.70, mf_of=0.65, rg_in=1.10, rg_of=1.05)
        sc_rev = scorer.score_c1(r_rev, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_rev["verdicts"]["C1.2"], "REVERSED")

        # REFUTED (equivalence): tight ratio 1.05 inside (0.80, 1.35)
        r_eq = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r_eq, mf_in=1.05, mf_of=1.05, rg_in=1.00, rg_of=1.00)
        sc_eq = scorer.score_c1(r_eq, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_eq["verdicts"]["C1.2"], "REFUTED (equivalence)")

        # INCONCLUSIVE: point ~ 1.32 with CI straddling 1.35 (set 12 targets at 1.15 and 12 at 1.52 -> geo ~ 1.322)
        r_inc = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_inc["targets_24"]):
                ts = t["string"]
                mult = 1.15 if ti % 2 == 0 else 1.52
                for row_mf, row_rg in zip(
                    r_inc["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_inc["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    row_mf["all_24_dlogits"] = [12.0 if j == ti else mult for j in range(24)]
                    row_rg["all_24_dlogits"] = [12.0 if j == ti else 1.0 for j in range(24)]
                    row_mf["exact_dlogit"] = 12.0
                    row_rg["exact_dlogit"] = 12.0
        sc_inc = scorer.score_c1(r_inc, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_inc["verdicts"]["C1.2"], "INCONCLUSIVE")

        # L6 fail-closed BCa containment violation (both non-degenerate and constant-offset bootstrap above point)
        with self.assertRaises(ValueError):
            scorer.bca_cluster_ci(
                10,
                lambda idxs: 0.0 if len(set(idxs)) == 10 else (10.0 + 0.01 * sum(idxs)),
                alphas=(0.05, 0.95),
                n_boot=200,
                seed=1,
                enforce_containment_90=True,
            )

    # 7. Claim C1.3 ALL 6 disjoint outcomes + sub-tags + F_mean_3L drift/sensitivity
    def test_10_c1_3_all_6_outcomes_and_f_guards(self) -> None:
        # 1. WHITENING_NOT_ESTABLISHED (no whitening: mf == rg -> CI_low <= 1.0)
        r1 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r1, mf_in=1.00, mf_of=0.80, rg_in=1.00, rg_of=0.80)
        sc1 = scorer.score_c1(r1, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc1["verdicts"]["C1.3"], "WHITENING_NOT_ESTABLISHED")
        self.assertFalse(sc1["C1_3"]["whitening_ok"])

        # 2. FLOOR_ABOVE_CEILING (whitening_ok True: mf_of=2.0 -> rg_of=1.10, yet R_floor=1.10/0.30=3.67 > 3.20!)
        r2 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r2, mf_in=2.10, mf_of=2.00, rg_in=1.15, rg_of=1.10)
        sc2 = scorer.score_c1(r2, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertTrue(sc2["C1_3"]["whitening_ok"])
        self.assertEqual(sc2["verdicts"]["C1.3"], "FLOOR_ABOVE_CEILING")

        # 3. H_B clean (whitening_ok True, R_floor = 0.78/0.30 = 2.60 in [2.20, 3.20], d_sep ~ 0.05 < 0.25, ll clean 1.0)
        r3 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r3, mf_in=1.50, mf_of=1.40, rg_in=0.88, rg_of=0.78, ll_of=0.30)
        sc3 = scorer.score_c1(r3, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc3["verdicts"]["C1.3"], "H_B")

        # 3b. H_B (FLOOR_SUSPECT: logit_lens) when R_floor(logit_lens) = 0.50/0.30 = 1.67 > 1.35
        r3b = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r3b, mf_in=1.50, mf_of=1.40, rg_in=0.88, rg_of=0.78, ll_of=0.50)
        sc3b = scorer.score_c1(r3b, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc3b["verdicts"]["C1.3"], "H_B (FLOOR_SUSPECT: logit_lens)")

        # 4. NEITHER (whitening_ok True, R_floor = 0.45/0.30 = 1.50 <= 1.90, flat d_sep ~ 0.0)
        r4 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r4, mf_in=1.00, mf_of=0.95, rg_in=0.47, rg_of=0.45)
        sc4 = scorer.score_c1(r4, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc4["verdicts"]["C1.3"], "NEITHER")

        # 5. MIXED (mid-band R_floor = 0.615/0.30 = 2.05 in (1.90, 2.20))
        r5 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r5, mf_in=1.40, mf_of=1.30, rg_in=0.70, rg_of=0.615)
        sc5 = scorer.score_c1(r5, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc5["verdicts"]["C1.3"], "MIXED")

        # 6. MIXED (MIXED_BOTH): sep_pass True AND hb_pass True (R_floor = 0.78/0.30 = 2.60, d_sep ~ +0.50)
        r6 = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r6, mf_in=1.55, mf_of=1.40, rg_in=1.28, rg_of=0.78)
        sc6 = scorer.score_c1(r6, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc6["verdicts"]["C1.3"], "MIXED (MIXED_BOTH)")

        # 7a. F_mean_3L hard drift (0.36 / 0.2958 = 1.217 > 1.15)
        r7a = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r7a, mf_in=1.20, mf_of=1.05, rg_in=0.72, rg_of=0.44, f_rand=0.36)
        sc7a = scorer.score_c1(r7a, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc7a["verdicts"]["C1.3"], "INCONCLUSIVE (F_mean_3L drift)")

        # 7b. F_mean_3L sensitivity zone mismatch (f_rand=0.325 -> r_F=1.099 in (1.06, 1.15];
        #     rg_of=0.60 -> live R_floor=1.846 (H_A) vs c0ref R_floor=2.028 (MIXED) -> INCONCLUSIVE sensitivity!)
        r7b = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r7b, mf_in=1.50, mf_of=1.35, rg_in=1.00, rg_of=0.60, f_rand=0.325)
        sc7b = scorer.score_c1(r7b, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc7b["verdicts"]["C1.3"], "INCONCLUSIVE (F_mean_3L sensitivity)")
        self.assertEqual(sc7b["C1_3"]["partition_live"]["label"], "H_A")
        self.assertEqual(sc7b["C1_3"]["partition_c0ref"]["label"], "MIXED")

    # 8. Claim C1.4 (UNIVERSAL_4_OF_4, 3/4 PASS, FAIL, non-gating <= 0 selectivity) + L12_linear_check
    def test_11_c1_4_and_l12_non_gating(self) -> None:
        # Non-gating selectivity <= 0 MUST NOT alter C1.4 PASS (C1-11), AND F-5 (M18):
        # corrupting `retention` to 0.99 (> vJv 0.68) while keeping `u_hat_self_J` clean proves C1.4 reads u_hat_self_J!
        r1 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            for ts in [t["string"] for t in r1["targets_24"]]:
                for row in r1["block2a_regime_N_last"][str(L)]["logit_lens"][ts]:
                    row["u_hat_off_abs_mean_J"] = 1.50  # negative diag - offdiag selectivity
                    row["retention"] = 0.99             # would make C1.4 FAIL if retention were used!
        sc1 = scorer.score_c1(r1, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertLess(sc1["C1_4"]["non_gating_diag_vs_offdiag_selectivity_mean"], 0.0)
        self.assertLess(sc1["C1_4"]["retention_diagnostic_mean_d_t"], 0.0)
        self.assertEqual(sc1["verdicts"]["C1.4"], "PASS (>= 19/24 & >= 3/4 strata; UNIVERSAL_4_OF_4)")

        # Noun-specific FAIL (only nouns positive — F8 reads u_hat_self_J directly!)
        r2 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            for t in r2["targets_24"]:
                if t["stratum"] != "nouns":
                    for row in r2["block2a_regime_N_last"][str(L)]["logit_lens"][t["string"]]:
                        row["u_hat_self_J"] = 0.72  # > vJv 0.68 -> negative d_t
                        row["retention"] = 0.72
        sc2 = scorer.score_c1(r2, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc2["verdicts"]["C1.4"], "NOUN-SPECIFIC / FAIL")

        # L12_linear_check FAIL does NOT block C1.0
        r3 = make_clean_receipt(self.c1_const)
        for ti in range(3):
            r3["secondary_L12"]["12"]["R_mf_all"][r3["targets_24"][ti]["string"]][0]["rms_int"] = 1.15
        sc3 = scorer.score_c1(r3, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc3["verdicts"]["C1.0"], "PASS")
        self.assertEqual(sc3["verdicts"]["L12_linear_check"], "FAIL")

    # 10. F4 + F6 + F7 + F9 + #85 boundary & mutation-killing tests (kills all remaining surviving mutants!)
    def test_13_f4_f6_f7_boundary_and_unit_guards(self) -> None:
        # F4 (kills M6 & M7): all-23 whitens (CI_low > 1.0) while open*func straddles 1.0 (CI_low <= 1.0 < CI_high)
        r_f4a = make_clean_receipt(self.c1_const)
        t_strata = [t["stratum"] for t in r_f4a["targets_24"]]
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_f4a["targets_24"]):
                ts = t["string"]
                st_i = t_strata[ti]
                # Make open*func ratio alternate 0.85 / 1.25 -> geo ~ 1.03, CI_low < 1.0 < CI_high!
                of_mult = 0.85 if ti % 2 == 0 else 1.25
                for row_mf, row_rg in zip(
                    r_f4a["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_f4a["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    a_mf, a_rg = [], []
                    for tj in range(24):
                        if tj == ti:
                            a_mf.append(12.0)
                            a_rg.append(12.0)
                        elif (t_strata[tj] == "function") != (st_i == "function"):
                            a_mf.append(round(0.50 * of_mult, 6))
                            a_rg.append(0.50)
                        else:
                            a_mf.append(1.80)
                            a_rg.append(0.90)
                    row_mf["all_24_dlogits"] = a_mf
                    row_rg["all_24_dlogits"] = a_rg
        sc_f4a = scorer.score_c1(r_f4a, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertGreater(sc_f4a["C1_2"]["geo_abs_bca90"][0], 1.0)
        self.assertLessEqual(sc_f4a["C1_2"]["geo_open_func_bca90"][0], 1.0)
        self.assertGreater(sc_f4a["C1_2"]["geo_open_func_bca90"][1], 1.0)
        self.assertFalse(sc_f4a["C1_3"]["whitening_ok"])
        self.assertEqual(sc_f4a["verdicts"]["C1.3"], "WHITENING_NOT_ESTABLISHED")

        # F4 reverse: open*func whitens cleanly (CI_low > 1.0) while all-23 straddles 1.0 (CI_low <= 1.0 < CI_high)
        r_f4b = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_f4b["targets_24"]):
                ts = t["string"]
                st_i = t_strata[ti]
                non_of_mult = 0.55 if ti % 2 == 0 else 1.05
                for row_mf, row_rg in zip(
                    r_f4b["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_f4b["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    a_mf, a_rg = [], []
                    for tj in range(24):
                        if tj == ti:
                            a_mf.append(12.0)
                            a_rg.append(12.0)
                        elif (t_strata[tj] == "function") != (st_i == "function"):
                            a_mf.append(0.75)
                            a_rg.append(0.50)
                        else:
                            a_mf.append(round(1.00 * non_of_mult, 6))
                            a_rg.append(1.00)
                    row_mf["all_24_dlogits"] = a_mf
                    row_rg["all_24_dlogits"] = a_rg
        sc_f4b = scorer.score_c1(r_f4b, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertGreater(sc_f4b["C1_2"]["geo_open_func_bca90"][0], 1.0)
        self.assertLessEqual(sc_f4b["C1_2"]["geo_abs_bca90"][0], 1.0)
        self.assertGreater(sc_f4b["C1_2"]["geo_abs_bca90"][1], 1.0)
        self.assertFalse(sc_f4b["C1_3"]["whitening_ok"])
        self.assertEqual(sc_f4b["verdicts"]["C1.3"], "WHITENING_NOT_ESTABLISHED")

        # F6 + F-4 (N7): all-capped (L, target) cell in Block 1 "all" emits BLOCKED_BY_DATA with p=1.0 in Holm
        r_f6 = make_clean_receipt(self.c1_const)
        ts0 = r_f6["targets_24"][0]["string"]
        for row in r_f6["block1_regime_D_all"]["18"]["R_mf"][ts0]:
            row["capped"] = True
        sc_f6 = scorer.score_c1(r_f6, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_f6["verdicts"]["C1.0"], "PASS")
        self.assertEqual(sc_f6["verdicts"]["C1.1"], "BLOCKED_BY_DATA (all-capped cell)")
        self.assertEqual(sc_f6["holm_5_test_family"]["p_C1.1_IUT"]["p"], 1.0)
        self.assertFalse(sc_f6["holm_5_test_family"]["p_C1.1_IUT"]["rejected"])
        self.assertEqual(sc_f6["verdicts"]["C1.3"], "H_A")

        # F-3 (N10): all-capped cell in Block 1 "last" (real runner capped shape!) triggers C1.1(b) guards cleanly
        for arm_cap in ("R_mf", "direct_matched"):
            r_f6b = make_clean_receipt(self.c1_const)
            if arm_cap == "R_mf":
                r_f6b["block1_regime_D_last"]["18"]["R_mf"][ts0] = [
                    {"prompt_id": p["id"], "capped": True, "dose_requested": 99.0, "dose_cap": 22.3611}
                    for p in r_f6b["prompts_8"]
                ]
            else:
                r_f6b["block1_regime_D_last"]["18"]["direct_matched"][ts0][0] = [
                    {"prompt_id": p["id"], "capped": True, "dose_requested": 99.0, "dose_cap": 22.3611}
                    for p in r_f6b["prompts_8"]
                ]
            sc_f6b = scorer.score_c1(r_f6b, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
            self.assertEqual(sc_f6b["verdicts"]["C1.0"], "PASS")
            self.assertEqual(sc_f6b["verdicts"]["C1.1"], "BLOCKED_BY_DATA (all-capped cell)")
            self.assertEqual(sc_f6b["holm_5_test_family"]["p_C1.1_IUT"]["p"], 1.0)
            self.assertFalse(sc_f6b["holm_5_test_family"]["p_C1.1_IUT"]["rejected"])

        # F7 / M12: delta_R_sep = 0.20 (< 0.25 bar) must NOT pass sep_pass -> NEITHER (not H_A!)
        r_m12 = make_clean_receipt(self.c1_const)
        # mf: 1.10/1.00 = 1.10, rg: 0.585/0.45 = 1.30 -> d_sep = +0.20 (< 0.25), R_floor = 1.50 <= 1.90
        _set_c1_3_bleeds(r_m12, mf_in=1.10, mf_of=1.00, rg_in=0.585, rg_of=0.45)
        sc_m12 = scorer.score_c1(r_m12, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_m12["verdicts"]["C1.3"], "NEITHER")

        # F7 / M14: Holm denominators 0.05/(6-k) at ranks 2..5 (kills Bonferroni alpha/m!)
        h_m14 = scorer.holm_bonferroni_stepdown(
            {"p_C1.1_IUT": 0.001, "p_C1.2_IUT": 0.011, "p_C1.3_HA": 0.015, "p_C1.3_HB": 0.022, "p_C1.4": 0.045},
            alpha=0.05,
        )
        self.assertTrue(all(v["rejected"] for v in h_m14.values()))

        # F-1 (M15 & N20): pin [0.65, 1.50] directly with C1.1(b) PASSING on both sides of each boundary!
        def _mk_m15(edge_r: float) -> dict:
            r_m = make_clean_receipt(self.c1_const)
            for L in scorer.PRIMARY_LAYERS:
                ls = str(L)
                for ts in [t["string"] for t in r_m["targets_24"]]:
                    for pi in range(8):
                        rmf = r_m["block1_regime_D_all"][ls]["R_mf"][ts][pi]
                        dm0 = r_m["block1_regime_D_all"][ls]["direct_matched"][ts][0][pi]
                        dm1 = r_m["block1_regime_D_all"][ls]["direct_matched"][ts][1][pi]
                        rmf_l = r_m["block1_regime_D_last"][ls]["R_mf"][ts][pi]
                        dm0_l = r_m["block1_regime_D_last"][ls]["direct_matched"][ts][0][pi]
                        dm1_l = r_m["block1_regime_D_last"][ls]["direct_matched"][ts][1][pi]
                        pg = rmf["pred_dlogit"] - 0.5 * (dm0["pred_dlogit"] + dm1["pred_dlogit"])
                        rmf["exact_dlogit"] = rmf["pred_dlogit"] * 1.05
                        dm_ex = rmf["exact_dlogit"] - edge_r * pg
                        dm0["exact_dlogit"] = dm1["exact_dlogit"] = dm_ex
                        # Ensure C1.1(b) holds cleanly (gap_last small so dtot_gap >= 2.0, share_gap ~ 0.88, dattn_rmf >= 2.0)
                        rmf_l["exact_dlogit"] = rmf["pred_dlogit"] * 0.60
                        dm0_l["exact_dlogit"] = dm1_l["exact_dlogit"] = rmf_l["exact_dlogit"] - (edge_r * pg - 2.5)
                        dtot_g = 2.5
                        rmf["sum_attn"] = 2.6
                        rmf_l["sum_attn"] = 0.2
                        dm0["sum_attn"] = dm1["sum_attn"] = 0.2 + (2.6 - 0.2) - 0.88 * dtot_g
                        dm0_l["sum_attn"] = dm1_l["sum_attn"] = 0.2
            return r_m

        for edge_r in (0.60, 1.54):
            sc_m15 = scorer.score_c1(_mk_m15(edge_r), self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
            self.assertTrue(sc_m15["C1_1"]["b_attention_attribution"]["ok"], f"C1.1(b) must hold at edge_r={edge_r}")
            self.assertFalse(sc_m15["C1_1"]["a_calibration"]["r_all_bands_ok"], f"r_all={edge_r} must fail [0.65, 1.50]")
            self.assertEqual(sc_m15["verdicts"]["C1.1"], "REFUTED")
        for ok_r in (0.68, 1.47):
            sc_ok = scorer.score_c1(_mk_m15(ok_r), self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
            self.assertTrue(sc_ok["C1_1"]["a_calibration"]["r_all_bands_ok"], f"r_all={ok_r} must pass [0.65, 1.50]")
            self.assertEqual(sc_ok["verdicts"]["C1.1"], "PASS")

        # F7 / M16: frac_err_all <= 0.50 * frac_err_last (all=0.14 <= 0.18, last=0.20 -> 0.14 > 0.10 MUST REFUTE!)
        r_m16 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ts in [t["string"] for t in r_m16["targets_24"]]:
                for pi in range(8):
                    ra = r_m16["block1_regime_D_all"][ls]["R_mf"][ts][pi]
                    rl = r_m16["block1_regime_D_last"][ls]["R_mf"][ts][pi]
                    ra["exact_dlogit"] = ra["pred_dlogit"] * 1.14
                    rl["exact_dlogit"] = rl["pred_dlogit"] * 0.80
        sc_m16 = scorer.score_c1(r_m16, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertFalse(sc_m16["C1_1"]["a_calibration"]["frac_err_ok"])
        self.assertEqual(sc_m16["verdicts"]["C1.1"], "REFUTED")

        # F7 / M43: share statistic DeltaAttn_gap - 0.75 * DeltaTot_gap (share=0.65 -> p_gap_share fails even though > 0.50)
        r_m43 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ts in [t["string"] for t in r_m43["targets_24"]]:
                for pi in range(8):
                    ra = r_m43["block1_regime_D_all"][ls]["R_mf"][ts][pi]
                    rl = r_m43["block1_regime_D_last"][ls]["R_mf"][ts][pi]
                    da0 = r_m43["block1_regime_D_all"][ls]["direct_matched"][ts][0][pi]
                    dl0 = r_m43["block1_regime_D_last"][ls]["direct_matched"][ts][0][pi]
                    dtot_g = (ra["exact_dlogit"] - da0["exact_dlogit"]) - (rl["exact_dlogit"] - dl0["exact_dlogit"])
                    ra["sum_attn"] = rl["sum_attn"] + 0.65 * dtot_g + (da0["sum_attn"] - dl0["sum_attn"])
        sc_m43 = scorer.score_c1(r_m43, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertGreater(sc_m43["C1_1"]["b_attention_attribution"]["p_gap_share"], 0.05)
        self.assertEqual(sc_m43["verdicts"]["C1.1"], "REFUTED")

        # F-2 (M44): ONLY mean(DeltaAttn_Rmf) >= 1.00 fails (0.60 < 1.00) while median_share_Rmf = 0.60/0.75 = 0.80 >= 0.65 AND 24/24 > dm!
        r_m44 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ts in [t["string"] for t in r_m44["targets_24"]]:
                for pi in range(8):
                    ra = r_m44["block1_regime_D_all"][ls]["R_mf"][ts][pi]
                    rl = r_m44["block1_regime_D_last"][ls]["R_mf"][ts][pi]
                    rl["exact_dlogit"] = ra["exact_dlogit"] - 0.75   # DeltaTot_Rmf = 0.75 -> share_Rmf = 0.60/0.75 = 0.80 >= 0.65!
                    ra["sum_attn"] = 1.00
                    rl["sum_attn"] = 0.40                            # DeltaAttn_Rmf = 0.60 < 1.00
                    for k in range(2):
                        r_m44["block1_regime_D_all"][ls]["direct_matched"][ts][k][pi]["sum_attn"] = -3.00
                        r_m44["block1_regime_D_last"][ls]["direct_matched"][ts][k][pi]["sum_attn"] = 0.10
        sc_m44 = scorer.score_c1(r_m44, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        b_m44 = sc_m44["C1_1"]["b_attention_attribution"]
        self.assertGreaterEqual(b_m44["median_share_Rmf"], 0.65)
        self.assertGreaterEqual(b_m44["n_dattn_Rmf_gt_dm"], 20)
        self.assertAlmostEqual(b_m44["mean_delta_attn_Rmf"], 0.60, delta=1e-4)
        self.assertFalse(b_m44["iii_ok"])
        self.assertEqual(sc_m44["verdicts"]["C1.1"], "REFUTED")

        # F-6 (N3): exact `s >= obs_sum` with zero tolerance slack (a flip 1e-13 below obs_sum must NOT count!)
        p_strict = scorer.sign_flip_p_one_sided([1.0, 1.0 - 1e-13, -1.0, -1.0], n_flips=5000, seed=7)
        # Flipping signs on indices (0, 2) vs (1, 3): obs_sum = -1e-13; flipping only index 1 gives -2 + 1e-13,
        # while swapping indices (0<->1) gives exact +1e-13 > obs_sum, and swapping (0,1)->(-1,-1+1e-13),(2,3)->(+1,+1) gives +1e-13 > obs_sum;
        # specifically [1.0, 1e-13] has obs_sum = 1 + 1e-13; flipping index 1 gives 1 - 1e-13 < obs_sum (diff 2e-13 < 1e-12 tol!).
        p_tie_edge = scorer.sign_flip_p_one_sided([1.0, 1e-13], n_flips=4000, seed=11)
        # Under exact s >= obs_sum, ONLY (+1, +1) (~25% of flips) matches; under tol=1e-12, BOTH (+1,+1) and (+1,-1) (~50%) match!
        self.assertLess(p_tie_edge, 0.35)
        self.assertGreater(p_tie_edge, 0.15)
        self.assertGreater(p_strict, 0.0)

        # F7 / M35: C1.2 >= 20/24 positive (18/24 strongly positive, 6/24 slightly negative -> geo > 1.35, 4/4 strata >= 1.15, FAILS 20/24!)
        r_m35 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_m35["targets_24"]):
                ts = t["string"]
                # 1 negative per stratum (indices 0,6,12,18) + 2 more (1,7) = 6 negative, 18 positive -> 18/24 < 20
                ratio = 0.96 if ti in (0, 1, 6, 7, 12, 18) else 2.20
                for rmf, rrg in zip(
                    r_m35["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_m35["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    rmf["all_24_dlogits"] = [12.0 if j == ti else ratio for j in range(24)]
                    rrg["all_24_dlogits"] = [12.0 if j == ti else 1.0 for j in range(24)]
        sc_m35 = scorer.score_c1(r_m35, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_m35["C1_2"]["n_pos_u_abs"], 18)
        self.assertGreaterEqual(sc_m35["C1_2"]["geo_abs"], 1.35)
        self.assertEqual(sc_m35["verdicts"]["C1.2"], "INCONCLUSIVE")

        # F7 / M45: C1.2 equivalence band (0.80, 1.35) upper edge: tight ratio 1.40 with 18/24 pos -> INCONCLUSIVE, NOT equivalence!
        r_m45 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_m45["targets_24"]):
                ts = t["string"]
                ratio = 1.38 if ti % 2 == 0 else 1.44  # CI strictly inside (1.36, 1.46) -> inside (0.50,2.00) but outside (0.80,1.35)!
                for rmf, rrg in zip(
                    r_m45["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_m45["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    rmf["all_24_dlogits"] = [12.0 if j == ti else ratio for j in range(24)]
                    rrg["all_24_dlogits"] = [12.0 if j == ti else 1.0 for j in range(24)]
                    # Make self_rg > self_mf so u_norm < 1.35 prevents PASS, testing equivalence vs INCONCLUSIVE cleanly
                    rmf["exact_dlogit"] = 14.0
                    rrg["exact_dlogit"] = 10.0
        sc_m45 = scorer.score_c1(r_m45, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_m45["verdicts"]["C1.2"], "INCONCLUSIVE")

        # F7 / M46: C1.3 >= 3/4 strata mean d_sep > 0 (only 2/4 strata positive -> NEITHER, NOT H_A!)
        r_m46 = make_clean_receipt(self.c1_const)
        for L in scorer.PRIMARY_LAYERS:
            ls = str(L)
            for ti, t in enumerate(r_m46["targets_24"]):
                ts = t["string"]
                st_i = t_strata[ti]
                rg_in_v = 0.85 if st_i in ("nouns", "verbs") else 0.42
                for rmf, rrg in zip(
                    r_m46["block2a_regime_N_last"][ls]["R_mf"][ts],
                    r_m46["block2a_regime_N_last"][ls]["R_ridge"][ts],
                ):
                    a_mf, a_rg = [], []
                    for tj in range(24):
                        if tj == ti:
                            a_mf.append(12.0)
                            a_rg.append(12.0)
                        elif t_strata[tj] == st_i:
                            a_mf.append(1.05)
                            a_rg.append(rg_in_v)
                        elif (t_strata[tj] == "function") != (st_i == "function"):
                            a_mf.append(1.00)
                            a_rg.append(0.45)
                        else:
                            a_mf.append(1.00)
                            a_rg.append(0.50)
                    rmf["all_24_dlogits"] = a_mf
                    rrg["all_24_dlogits"] = a_rg
        sc_m46 = scorer.score_c1(r_m46, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_m46["C1_3"]["strata_pos_mean_count"], 2)
        self.assertGreaterEqual(sc_m46["C1_3"]["delta_R_sep"], 0.25)
        self.assertEqual(sc_m46["verdicts"]["C1.3"], "NEITHER")

        # F7 / M24: F_mean_3L averages ALL 3 layers (L18=0.30, L24=L27=0.45 -> mean=0.40 -> drift!)
        r_m24 = make_clean_receipt(self.c1_const)
        for ls in ("24", "27"):
            for draw in r_m24["block2b_rand_iso"][ls]["rand_iso"]:
                for row in draw:
                    row["all_24_dlogits"] = [0.45] * 24
        sc_m24 = scorer.score_c1(r_m24, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertAlmostEqual(sc_m24["C1_3"]["F_mean_3L_live"], 0.40, delta=1e-4)
        self.assertEqual(sc_m24["verdicts"]["C1.3"], "INCONCLUSIVE (F_mean_3L drift)")

        # F7 / M26: C1.4 averages ALL 3 layers (L18 positive +0.13, L24/L27 negative -0.12 -> 3L mean < 0 -> FAIL!)
        r_m26 = make_clean_receipt(self.c1_const)
        for ls in ("24", "27"):
            for t in r_m26["targets_24"]:
                for row in r_m26["block2a_regime_N_last"][ls]["logit_lens"][t["string"]]:
                    row["u_hat_self_J"] = 0.80
                    row["retention"] = 0.80
        sc_m26 = scorer.score_c1(r_m26, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=2000, n_boot=400)
        self.assertEqual(sc_m26["verdicts"]["C1.4"], "NOUN-SPECIFIC / FAIL")

        # F9: L12 "last" inert does NOT fail C1.0.1 (only 7 registered pairs gate C1.0.1)
        r_f9 = make_clean_receipt(self.c1_const)
        r_f9["sham_movement_check"]["last"]["12"] = 0.0
        sc_f9 = scorer.score_c1(r_f9, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)
        self.assertTrue(sc_f9["gate_C1_0"]["C1.0.1_shams_and_liveness"]["PASS"])

        # Claude #85: F_mean_3L outside [0.10, 1.00] logits raises fail-closed ValueError
        r_u = make_clean_receipt(self.c1_const)
        _set_c1_3_bleeds(r_u, mf_in=1.20, mf_of=1.05, rg_in=0.72, rg_of=0.44, f_rand=1.50)
        with self.assertRaises(ValueError):
            scorer.score_c1(r_u, self.c1_const, constants_path=CONSTANTS_PATH, design_path=DESIGN_PATH, n_flips=200, n_boot=100)

    # 9. Exact 5-test Holm step-down stop rule + /usr/bin/python3 -I byte-identical CLI replay
    def test_12_holm_stepdown_and_isolated_stdlib_replay(self) -> None:
        h = scorer.holm_bonferroni_stepdown(
            {"p_C1.1_IUT": 0.001, "p_C1.2_IUT": 0.020, "p_C1.3_HA": 0.021, "p_C1.3_HB": 0.022, "p_C1.4": 0.023},
            alpha=0.05,
        )
        # Rank 1 (0.001 <= 0.010) rejected; Rank 2 (0.020 > 0.0125) STOPS -> ranks 3..5 retained even though 0.022 <= 0.025!
        self.assertTrue(h["p_C1.1_IUT"]["rejected"])
        self.assertFalse(h["p_C1.2_IUT"]["rejected"])
        self.assertFalse(h["p_C1.3_HA"]["rejected"])
        self.assertFalse(h["p_C1.3_HB"]["rejected"])
        self.assertFalse(h["p_C1.4"]["rejected"])

        # Double-run /usr/bin/python3 -I CLI byte-identical check
        rec = make_clean_receipt(self.c1_const)
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rec_gz = tdp / "rec.json.gz"
            with gzip.open(rec_gz, "wt", encoding="utf-8") as f:
                json.dump(rec, f)
            for tag in ("r1", "r2"):
                cmd = [
                    "/usr/bin/python3", "-I", str(ROOT / "score_olmo3_c1.py"),
                    "--receipt", str(rec_gz),
                    "--constants", str(CONSTANTS_PATH),
                    "--design", str(DESIGN_PATH),
                    "--out-json", str(tdp / f"sc_{tag}.json"),
                    "--out-md", str(tdp / f"sc_{tag}.md"),
                ]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.assertEqual((tdp / "sc_r1.json").read_bytes(), (tdp / "sc_r2.json").read_bytes())
            self.assertEqual((tdp / "sc_r1.md").read_bytes(), (tdp / "sc_r2.md").read_bytes())
            sc_out = json.loads((tdp / "sc_r1.json").read_text())
            self.assertEqual(sc_out["verdicts"]["C1.0"], "PASS")
            self.assertEqual(sc_out["verdicts"]["C1.3"], "H_A")


if __name__ == "__main__":
    unittest.main(verbosity=2)
