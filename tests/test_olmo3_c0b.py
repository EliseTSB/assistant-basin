"""Unit, integration, fail-closed, and stdlib-determinism tests for Stage C0b."""
from __future__ import annotations

import ast
import copy
import gzip
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from olmo3_virtual_and_highway.run_olmo3_c0b import (
    ANCHOR_TARGETS_REGIME_D,
    LAYERS_3L,
    LAYERS_4L,
    post_norm_inject_hook,
    run_c0b,
)
from olmo3_virtual_and_highway.score_olmo3_c0b import (
    EXPECTED_C0B_CONSTANTS_SHA256,
    EXPECTED_C0B_DESIGN_SHA256,
    EXPECTED_C0_BANK_SHA256,
    EXPECTED_C0_CONSTANTS_SHA256,
    EXPECTED_C0_RECEIPT_UNCOMP_SHA256,
    EXPECTED_R_LENS_SHA256,
    score_c0b,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
C0_CONST_PATH = REPO_ROOT / "olmo3_virtual_and_highway/c0_prereg_constants.json"
C0B_CONST_PATH = REPO_ROOT / "olmo3_virtual_and_highway/c0b_prereg_constants.json"
C0B_DESIGN_PATH = REPO_ROOT / "olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0B-DESIGN.md"
C0_RECEIPT_PATH = REPO_ROOT / "results/vesper-olmo3-7b-stage-c0/c0_receipt.json.gz"


# ---------------------------------------------------------------------------
# Tiny multi-token causal post-norm mock OLMo-3 transformer (real softmax attn)
# ---------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
        return (x / rms) * self.weight


class _MockCausalLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.post_attention_layernorm = _RMSNorm(d)
        self.mlp_up = nn.Linear(d, d * 2, bias=False)
        self.mlp_down = nn.Linear(d * 2, d, bias=False)
        self.post_feedforward_layernorm = _RMSNorm(d)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, D = h.shape
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        mask = torch.triu(torch.ones(T, T, device=h.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0), -1e9)
        attn_w = torch.softmax(scores, dim=-1)
        attn_out = self.o(torch.matmul(attn_w, v))
        h = h + self.post_attention_layernorm(attn_out)
        mlp_out = self.mlp_down(torch.nn.functional.silu(self.mlp_up(h)))
        h = h + self.post_feedforward_layernorm(mlp_out)
        return h


class _MockOlmo3Model(nn.Module):
    def __init__(self, n_layers: int = 20, d: int = 16, vocab: int = 64):
        super().__init__()
        torch.manual_seed(42)
        self.config = type("Cfg", (), {"num_hidden_layers": n_layers})()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab, d)
        self.model.layers = nn.ModuleList([_MockCausalLayer(d) for _ in range(n_layers)])
        self.model.norm = _RMSNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        h = self.model.embed(input_ids)
        for lay in self.model.layers:
            h = lay(h)
        logits = self.lm_head(self.model.norm(h))
        return type("Out", (), {"logits": logits})()


class _MockTokenizer:
    def encode(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
        # Deterministic 6-token sequence so causal cross-position attention is active
        ids = [(sum(ord(c) for c in w) % 60) + 1 for w in text.split()[:6]]
        while len(ids) < 6:
            ids.append(3)
        return torch.tensor([ids], dtype=torch.long)


def _make_mock_bank(d: int = 16) -> dict:
    torch.manual_seed(7)
    targets = [
        {"domain": "animals", "string": " dolphin", "id": 5, "decoded_piece": " dolphin", "u_norm": 1.5},
        {"domain": "objects", "string": " hammer", "id": 12, "decoded_piece": " hammer", "u_norm": 1.5},
    ]
    u_vecs = torch.randn(2, d)
    u_vecs = u_vecs / u_vecs.norm(dim=-1, keepdim=True) * 5.0
    layers_bank = {}
    for L in (12, 18):
        rmf = u_vecs / u_vecs.norm(dim=-1, keepdim=True)
        rr = torch.randn(2, d)
        rr = rr / rr.norm(dim=-1, keepdim=True)
        ll = u_vecs / u_vecs.norm(dim=-1, keepdim=True)
        dm = u_vecs.unsqueeze(1).repeat(1, 2, 1)
        dm = dm / dm.norm(dim=-1, keepdim=True)
        rand = torch.randn(2, d)
        rand = rand / rand.norm(dim=-1, keepdim=True)
        layers_bank[L] = {
            "R_mf": rmf,
            "R_ridge": rr,
            "logit_lens": ll,
            "direct_matched": dm,
            "rand_iso": rand,
            "per_target_geom": {
                t["string"]: {
                    "pred_ret_Rmf": 2.0,
                    "pred_ret_Rridge": 1.5,
                    "pred_ret_ll": 0.5,
                    "pred_ret_dm": 1.0,
                    "pred_gap_D3": 3.0,
                }
                for t in targets
            },
        }
    return {
        "meta": {
            "model": "mock-olmo3",
            "revision": "rev0",
            "r_lens_sha256": EXPECTED_R_LENS_SHA256,
            "constants_sha256": EXPECTED_C0_CONSTANTS_SHA256,
            "design_sha256": EXPECTED_C0B_DESIGN_SHA256,
            "k_dm": 1,
            "m_rand": 2,
        },
        "layers": [12, 18],
        "targets_16": targets,
        "u_vectors": u_vecs,
        "layers_bank": layers_bank,
    }


# ---------------------------------------------------------------------------
# 1. Runner tests: all vs last telescoping, shams, hook validation, provenance resume
# ---------------------------------------------------------------------------
def test_runner_all_vs_last_telescoping_and_checkpoint_refusal(tmp_path: Path):
    model = _MockOlmo3Model(n_layers=20, d=16, vocab=64)
    model.eval()
    tok = _MockTokenizer()
    bank = _make_mock_bank(d=16)
    # Match model lm_head rows to bank u_vectors so exact_dlogit == recon_dlogit
    with torch.no_grad():
        for i, t in enumerate(bank["targets_16"]):
            model.lm_head.weight[t["id"]] = bank["u_vectors"][i] * model.model.norm.weight

    # Hook validation checks
    with pytest.raises(ValueError):
        with post_norm_inject_hook(model, 12, torch.zeros(16), 1.0, "all"):
            pass
    with pytest.raises(ValueError):
        with post_norm_inject_hook(model, 12, bank["layers_bank"][12]["R_mf"][0], 1.0, "middle"):
            pass

    rec = run_c0b(
        model,
        tok,
        bank,
        tmp_path / "run1",
        device="cpu",
        smoke=True,
        bank_file_sha256="deadbeef",
    )
    # Exact shams == 0.0 on both modes & both layers
    for mode in ("last", "all"):
        for L in ("12", "18"):
            assert rec["shams"][mode][L] == 0.0
            assert rec["sham_movement_check"][mode][L] > 0.0

    # Exact 4-term telescoping holds under BOTH "all" and "last" on multi-token causal attn!
    t_all = rec["regime_D_all"]["12"]["R_mf"][" dolphin"][0]
    t_last = rec["regime_D_last_anchor"]["12"]["R_mf"][" dolphin"][0]
    assert t_all["telescoping_err"] < 1e-4
    assert t_last["telescoping_err"] < 1e-4
    # Multi-token causal attention makes sum_attn strictly different between "all" and "last"!
    assert abs(t_all["sum_attn"] - t_last["sum_attn"]) > 1e-4

    # Verified checkpoint resume succeeds cleanly
    rec2 = run_c0b(
        model,
        tok,
        bank,
        tmp_path / "run1",
        device="cpu",
        smoke=True,
        bank_file_sha256="deadbeef",
    )
    assert rec2["meta"]["total_forwards_executed"] == rec["meta"]["total_forwards_executed"]

    # Mutated bank_file_sha256 triggers hard provenance refusal!
    with pytest.raises(RuntimeError, match="PROVENANCE REFUSAL"):
        run_c0b(
            model,
            tok,
            bank,
            tmp_path / "run1",
            device="cpu",
            smoke=True,
            bank_file_sha256="mutated_sha",
        )


# ---------------------------------------------------------------------------
# Helper: construct a full, realistic registered Stage C0b synthetic receipt
# paired seamlessly with the REAL published c0_receipt.json.gz!
# ---------------------------------------------------------------------------
def _make_full_synthetic_c0b_receipt(
    r_all_by_layer_fmt: dict[tuple[int, str], float] | None = None,
    attn_share_target: float = 0.80,
    ridge_scale_vs_rmf: float = 0.25,
    ridge_self_mult: float = 1.0812,
    drift_n_offtarget: float = 0.0,
) -> dict:
    with gzip.open(C0_RECEIPT_PATH, "rt", encoding="utf-8") as f:
        c0_rec = json.load(f)
    c0_const = json.loads(C0_CONST_PATH.read_text(encoding="utf-8"))
    c0b_const = json.loads(C0B_CONST_PATH.read_text(encoding="utf-8"))

    targets = c0_rec["targets_16"]
    t_strs = [t["string"] for t in targets]
    prompts = c0_rec["prompts_8"]
    pred_gap = {
        (L, pt["target"]): float(pt["pred_gap_D3"])
        for L in LAYERS_4L
        for pt in c0_const["layer_geometry"][str(L)]["per_target"]
    }

    if r_all_by_layer_fmt is None:
        r_all_by_layer_fmt = {(L, fmt): 0.90 for L in LAYERS_4L for fmt in ("bare", "chatml")}

    post_norms = {str(L): c0_rec["post_mlp_norms"][str(L)] for L in LAYERS_4L}
    rd_all: dict = {}
    rd_anc: dict = {}
    rn_last: dict = {}
    rn_all: dict = {}

    for L in LAYERS_4L:
        ls = str(L)
        c0_rd = c0_rec["regime_D"][ls]
        # Exact copy of C0 Regime D for the 2 anchor targets (zero drift!)
        rd_anc[ls] = {
            "positions": "last",
            "target_direct_linear": 3.0,
            "dose_cap": c0_rd["dose_cap"],
            "R_mf": {ts: copy.deepcopy(c0_rd["R_mf"][ts]) for ts in ANCHOR_TARGETS_REGIME_D},
            "direct_matched": {ts: copy.deepcopy(c0_rd["direct_matched"][ts]) for ts in ANCHOR_TARGETS_REGIME_D},
        }
        # Synthetic Regime D "all" with controlled r_all(L, fmt) and attn_share
        rd_all[ls] = {
            "positions": "all",
            "target_direct_linear": 3.0,
            "dose_cap": c0_rd["dose_cap"],
            "R_mf": {},
            "logit_lens": {},
            "direct_matched": {},
        }
        for ti, ts in enumerate(t_strs):
            pg = pred_gap[(L, ts)]
            # Keep dm and ll identical to C0 "last", lift R_mf to hit target r_all and attn_share
            rd_all[ls]["direct_matched"][ts] = copy.deepcopy(c0_rd["direct_matched"][ts])
            rd_all[ls]["logit_lens"][ts] = copy.deepcopy(c0_rd["logit_lens"][ts])
            rmf_rows = copy.deepcopy(c0_rd["R_mf"][ts])
            for pi, p in enumerate(prompts):
                fmt = p["format"]
                # Small deterministic target variation (+/- 0.02) so BCa bootstrap is non-degenerate
                target_r = r_all_by_layer_fmt[(L, fmt)] + 0.002 * (ti - 7.5)
                dm_ex = sum(c0_rd["direct_matched"][ts][k][pi]["exact_dlogit"] for k in range(4)) / 4.0
                dm_at = sum(c0_rd["direct_matched"][ts][k][pi]["sum_attn"] for k in range(4)) / 4.0
                c0_gap = c0_rd["R_mf"][ts][pi]["exact_dlogit"] - dm_ex
                c0_at_gap = c0_rd["R_mf"][ts][pi]["sum_attn"] - dm_at
                new_gap = target_r * pg
                delta_tot = new_gap - c0_gap
                delta_attn = attn_share_target * delta_tot
                delta_mlp = (1.0 - attn_share_target) * delta_tot
                rmf_rows[pi]["exact_dlogit"] = round(dm_ex + new_gap, 6)
                rmf_rows[pi]["sum_attn"] = round(dm_at + c0_at_gap + delta_attn, 6)
                rmf_rows[pi]["sum_mlp"] = round(rmf_rows[pi]["sum_mlp"] + delta_mlp, 6)
                rmf_rows[pi]["telescoping_err"] = 0.01
                rmf_rows[pi]["frozen_direct_effect"] = 3.02
                rmf_rows[pi]["capped"] = False
            rd_all[ls]["R_mf"][ts] = rmf_rows

    for L in LAYERS_3L:
        ls = str(L)
        c0_rn = c0_rec["regime_N"][ls]
        rn_last[ls] = {
            "positions": "last",
            "dose_N": c0_rn["dose_N"],
            "post_mlp_norm": c0_rn["post_mlp_norm"],
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
        }
        rn_all[ls] = {
            "positions": "all",
            "dose_N": c0_rn["dose_N"],
            "post_mlp_norm": c0_rn["post_mlp_norm"],
            "R_mf": {},
            "R_ridge": {},
        }
        for ti, ts in enumerate(t_strs):
            rmf_rows = copy.deepcopy(c0_rn["R_mf"][ts])
            rr_rows = copy.deepcopy(c0_rn["R_ridge"][ts])
            ll_rows = copy.deepcopy(c0_rn["logit_lens"][ts])
            for pi in range(8):
                # Optional drift on R_mf off-target cells
                if drift_n_offtarget != 0.0:
                    rmf_rows[pi]["all_16_dlogits"] = [
                        round(v + (drift_n_offtarget if j != ti else 0.0), 6)
                        for j, v in enumerate(rmf_rows[pi]["all_16_dlogits"])
                    ]
                # Construct R_ridge all_16_dlogits scaled relative to R_mf off-target bleed
                base_all16 = c0_rn["R_mf"][ts][pi]["all_16_dlogits"]
                rr_self = float(c0_rn["R_mf"][ts][pi]["exact_dlogit"]) * ridge_self_mult
                rr_rows[pi]["exact_dlogit"] = round(rr_self, 6)
                rr_rows[pi]["all_16_dlogits"] = [
                    round(rr_self if j == ti else base_all16[j] * (ridge_scale_vs_rmf * (1.0 + 0.005 * (j - 7.5))), 6)
                    for j in range(16)
                ]
                ll_rows[pi]["all_16_dlogits"] = [
                    round(ll_rows[pi]["exact_dlogit"] if j == ti else base_all16[j] * 0.15, 6)
                    for j in range(16)
                ]
            rn_last[ls]["R_mf"][ts] = rmf_rows
            rn_last[ls]["R_ridge"][ts] = rr_rows
            rn_last[ls]["logit_lens"][ts] = ll_rows
            rn_all[ls]["R_mf"][ts] = copy.deepcopy(rmf_rows)
            rn_all[ls]["R_ridge"][ts] = copy.deepcopy(rr_rows)

    return {
        "meta": {
            "study": "stage_c0b_olmo3_7b_all_vs_last_and_ridge_whitening",
            "model": "allenai/Olmo-3-7B-Instruct",
            "revision": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
            "smoke": False,
            "elapsed_sec": 420.0,
            "gpu_name": "synthetic-test",
            "bank_sha256": EXPECTED_C0_BANK_SHA256,
            "bank_fingerprint": "fp",
            "r_lens_sha256": EXPECTED_R_LENS_SHA256,
            "c0_constants_sha256": EXPECTED_C0_CONSTANTS_SHA256,
            "c0b_constants_sha256": EXPECTED_C0B_CONSTANTS_SHA256,
            "c0_receipt_uncompressed_sha256": EXPECTED_C0_RECEIPT_UNCOMP_SHA256,
            "c0b_design_sha256": EXPECTED_C0B_DESIGN_SHA256,
            "k_dm": 4,
            "forward_counts": {
                "clean": 8,
                "sham": 8,
                "regime_D_all": 3072,
                "regime_D_all_capped_skipped": 0,
                "regime_D_last_anchor": 320,
                "regime_D_last_anchor_capped_skipped": 0,
                "regime_N_last": 1152,
                "regime_N_all": 768,
            },
            "total_forwards_executed": 5328,
        },
        "layers_4L": LAYERS_4L,
        "layers_3L": LAYERS_3L,
        "anchor_targets_regime_D": ANCHOR_TARGETS_REGIME_D,
        "targets_16": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_norms,
        "shams": {
            "last": {str(L): 0.0 for L in LAYERS_4L},
            "all": {str(L): 0.0 for L in LAYERS_4L},
        },
        "sham_movement_check": {
            "last": {str(L): 8.5 for L in LAYERS_4L},
            "all": {str(L): 10.2 for L in LAYERS_4L},
        },
        "regime_D_all": rd_all,
        "regime_D_last_anchor": rd_anc,
        "regime_N_last": rn_last,
        "regime_N_all": rn_all,
    }


def _write_gz(path: Path, obj: dict) -> Path:
    with gzip.open(path, "wb", compresslevel=1) as f:
        f.write(json.dumps(obj).encode("utf-8"))
    return path


def _score(rec: dict, tmp_path: Path, name: str = "rec.json.gz", **kwargs) -> dict:
    p = _write_gz(tmp_path / name, rec)
    return score_c0b(
        receipt_path=p,
        c0b_constants_path=C0B_CONST_PATH,
        c0_constants_path=C0_CONST_PATH,
        c0_receipt_path=C0_RECEIPT_PATH,
        design_path=kwargs.pop("design_path", C0B_DESIGN_PATH),
        n_boot=kwargs.pop("n_boot", 400),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2. Scorer verdict logic tests covering ALL registered findings & edge cases
# ---------------------------------------------------------------------------
def test_score_c0b_clean_full_pass(tmp_path: Path):
    rec = _make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=0.25)
    sc = _score(rec, tmp_path)
    assert sc["verdicts"] == {
        "C0b.0_global": "PASS",
        "C0b.0.4b_drift": "PASS",
        "C0b.1a": "PASS",
        "C0b.1b": "PASS",
        "C0b.2a": "PASS",
        "C0b.2b": "PASS",
    }
    assert sc["gate_C0b_0"]["C0b_0_2_telescoping"]["n_traces"] == 5312
    assert sc["gate_C0b_0"]["C0b_0_4b_c0b1b_receipt_drift"]["n_offtarget_cells_checked"] == 5760


def test_c0b1a_per_layer_split_only_is_inconclusive_and_bad_cell_is_refuted(tmp_path: Path):
    # Finding A & B: L12 ChatML=0.55 vs Bare=0.98 (split 0.43 > 0.35), all 8 cells in [0.50,1.50]
    grid_inc = {(L, f): 0.92 for L in LAYERS_4L for f in ("bare", "chatml")}
    grid_inc[(12, "chatml")] = 0.55
    grid_inc[(12, "bare")] = 0.98
    sc_inc = _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_inc), tmp_path, "inc.json.gz")
    assert sc_inc["verdicts"]["C0b.1a"] == "INCONCLUSIVE"
    assert sc_inc["C0b_1a"]["clauses"]["i_pooled_calib_and_90ci"] is True
    assert sc_inc["C0b_1a"]["clauses"]["ii_bare_and_chatml_calib"] is True
    assert sc_inc["C0b_1a"]["clauses"]["iv_all_8_layer_format_cells"] is True
    assert sc_inc["C0b_1a"]["clauses"]["iii_per_layer_split_le_0p35"] is False

    # L12 ChatML=0.45 (< 0.50 floor) -> REFUTED
    grid_ref = dict(grid_inc)
    grid_ref[(12, "chatml")] = 0.45
    sc_ref = _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_ref), tmp_path, "ref.json.gz")
    assert sc_ref["verdicts"]["C0b.1a"] == "REFUTED"


def test_c0b1b_rejects_tiny_attn_shift_even_with_high_share(tmp_path: Path):
    # Finding J: ΔTot ~0.40, ΔAttn ~0.30, share ~0.75 -> MUST FAIL (below attn_floor=1.7031)
    grid_small = {(L, f): 0.39 for L in LAYERS_4L for f in ("bare", "chatml")}
    sc = _score(
        _make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_small, attn_share_target=0.75),
        tmp_path,
        "small_attn.json.gz",
    )
    assert sc["C0b_1b"]["mean_delta_tot"] > sc["C0b_1b"]["delta_tot_guard"]
    assert sc["C0b_1b"]["median_share"] >= 0.50
    assert sc["C0b_1b"]["mean_delta_attn"] < sc["C0b_1b"]["attn_lift_floor"]
    assert sc["verdicts"]["C0b.1b"] == "FAIL"


def test_c0b2a_three_way_mirror_and_c0b2b_partial_whitening(tmp_path: Path):
    # (1) No whitening (scale=1.00, self_mult=1.08 -> abs=1.00, norm=1.08, both q95 < 1.25) -> REFUTED
    sc_ref = _score(
        _make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=1.00, ridge_self_mult=1.08),
        tmp_path,
        "c2_ref.json.gz",
    )
    assert sc_ref["verdicts"]["C0b.2a"] == "REFUTED"
    assert sc_ref["verdicts"]["C0b.2b"] == "FAIL"

    # (2) Finding F3 counterexample: abs=1.20 (<1.25, q95<1.25), norm=1.30 (>1.25, q95>1.25) -> INCONCLUSIVE!
    sc_inc = _score(
        _make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=1.0 / 1.20, ridge_self_mult=1.0812),
        tmp_path,
        "c2_inc.json.gz",
    )
    assert sc_inc["C0b_2a"]["abs_ratio_90pct_bca_ci"]["q_0.95"] < 1.25
    assert sc_inc["C0b_2a"]["norm_ratio_90pct_bca_ci"]["q_0.95"] > 1.25
    assert sc_inc["verdicts"]["C0b.2a"] == "INCONCLUSIVE"

    # (3) Partial whitening: B_cross(R_ridge) ~ 0.54 (ratio ~2.5x > 1.25x, but > 0.3958 floor) -> C0b.2a PASS, C0b.2b FAIL
    sc_part = _score(
        _make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=0.40),
        tmp_path,
        "c2_part.json.gz",
    )
    assert sc_part["verdicts"]["C0b.2a"] == "PASS"
    assert sc_part["verdicts"]["C0b.2b"] == "FAIL"


def test_gate_c0b04b_drift_voids_only_c0b1b_not_live_c0b2(tmp_path: Path):
    # Finding C: off-target cell drift=0.10 (>0.07) fails 4b -> C0b.1b VOID_DRIFT, while C0b.1a & live C0b.2 PASS!
    sc = _score(
        _make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=0.25, drift_n_offtarget=0.10),
        tmp_path,
        "drift.json.gz",
    )
    assert sc["verdicts"]["C0b.0_global"] == "PASS"
    assert sc["verdicts"]["C0b.0.4b_drift"] == "FAIL"
    assert sc["verdicts"]["C0b.1b"] == "VOID_DRIFT"
    assert sc["verdicts"]["C0b.1a"] == "PASS"
    assert sc["verdicts"]["C0b.2a"] == "PASS"
    assert sc["verdicts"]["C0b.2b"] == "PASS"


def test_frozen_capped_cell_rule_and_fail_closed_guards(tmp_path: Path):
    # Finding F: 1 capped dm draw on 1 bare prompt -> 3/4 retained -> PASS
    rec1 = _make_full_synthetic_c0b_receipt()
    rec1["regime_D_all"]["12"]["direct_matched"][" dolphin"][2][0] = {
        "prompt_id": "bare_narr_0", "capped": True, "dose_requested": 15.0, "dose_cap": 12.7
    }
    sc1 = _score(rec1, tmp_path, "cap1.json.gz")
    assert sc1["gate_C0b_0"]["C0b_0_3_cap_and_denom"]["passed"] is True
    assert sc1["gate_C0b_0"]["C0b_0_2_telescoping"]["n_traces"] == 5311

    # 2 capped bare prompts on one (L,t) cell -> 2/4 < 3/4 -> C0b.0.3 fails closed!
    rec2 = copy.deepcopy(rec1)
    rec2["regime_D_all"]["12"]["direct_matched"][" dolphin"][0][1] = {
        "prompt_id": "bare_narr_1", "capped": True, "dose_requested": 15.0, "dose_cap": 12.7
    }
    sc2 = _score(rec2, tmp_path, "cap2.json.gz")
    assert sc2["gate_C0b_0"]["C0b_0_3_cap_and_denom"]["passed"] is False
    assert sc2["verdicts"]["C0b.0_global"] == "FAIL"

    # Fail-closed schema refusals
    rec_smoke = _make_full_synthetic_c0b_receipt()
    rec_smoke["meta"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke"):
        _score(rec_smoke, tmp_path, "smoke.json.gz")

    rec_bad_dose = _make_full_synthetic_c0b_receipt()
    rec_bad_dose["regime_N_last"]["18"]["dose_N"] = 999.0
    with pytest.raises(ValueError, match="dose_N"):
        _score(rec_bad_dose, tmp_path, "baddose.json.gz")

    # Non-finite value in all_16_dlogits blocks everything
    rec_nan = _make_full_synthetic_c0b_receipt()
    rec_nan["regime_N_last"]["18"]["R_ridge"][" dolphin"][0]["all_16_dlogits"][3] = float("nan")
    sc_nan = _score(rec_nan, tmp_path, "nan.json.gz")
    assert sc_nan["verdicts"]["C0b.0_global"] == "FAIL"
    assert sc_nan["verdicts"]["C0b.2a"] == "BLOCKED_BY_C0b.0"

    # Missing sham_movement_check fails C0b.0.1
    rec_nomove = _make_full_synthetic_c0b_receipt()
    del rec_nomove["sham_movement_check"]
    sc_nomove = _score(rec_nomove, tmp_path, "nomove.json.gz")
    assert sc_nomove["gate_C0b_0"]["C0b_0_1_sham"]["passed"] is False
    assert sc_nomove["verdicts"]["C0b.0_global"] == "FAIL"

    # Missing design or --no-strict-provenance fails C0b.0.4a
    rec_ok = _make_full_synthetic_c0b_receipt()
    sc_nodesign = _score(rec_ok, tmp_path, "nodesign.json.gz", design_path=None)
    assert sc_nodesign["gate_C0b_0"]["C0b_0_4a_global_provenance"]["passed"] is False
    sc_relaxed = _score(rec_ok, tmp_path, "relaxed.json.gz", strict_provenance=False)
    assert sc_relaxed["gate_C0b_0"]["C0b_0_4a_global_provenance"]["passed"] is False


def test_scorer_is_stdlib_only_and_byte_deterministic_under_isolated_python3(tmp_path: Path):
    # AST check: zero non-stdlib imports
    src = (REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0b.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    stdlib_names = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib_names, f"Non-stdlib import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in stdlib_names, f"Non-stdlib import: {node.module}"

    # Run twice via bare `/usr/bin/python3 -I` and verify byte-identical outputs!
    rec_gz = _write_gz(tmp_path / "det.json.gz", _make_full_synthetic_c0b_receipt())
    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    py_bin = "/usr/bin/python3" if Path("/usr/bin/python3").exists() else sys.executable
    for out_d in (out1, out2):
        subprocess.run(
            [
                py_bin, "-I",
                str(REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0b.py"),
                "--receipt", str(rec_gz),
                "--c0b-constants", str(C0B_CONST_PATH),
                "--c0-constants", str(C0_CONST_PATH),
                "--c0-receipt", str(C0_RECEIPT_PATH),
                "--design", str(C0B_DESIGN_PATH),
                "--out-dir", str(out_d),
                "--n-boot", "500",
            ],
            check=True,
        )
    assert (out1 / "scorecard.json").read_bytes() == (out2 / "scorecard.json").read_bytes()
    assert (out1 / "SCORECARD.md").read_bytes() == (out2 / "SCORECARD.md").read_bytes()
