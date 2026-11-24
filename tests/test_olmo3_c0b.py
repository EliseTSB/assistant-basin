"""Unit, integration, boundary/mutation-killing, fail-closed, and stdlib-determinism tests for Stage C0b."""
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
    verify_preflight_provenance,
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
# 1. Runner tests: telescoping, shams, hook validation, preflight, provenance
# ---------------------------------------------------------------------------
def test_runner_all_vs_last_telescoping_preflight_and_checkpoint_refusal(tmp_path: Path):
    model = _MockOlmo3Model(n_layers=20, d=16, vocab=64)
    model.eval()
    tok = _MockTokenizer()
    bank = _make_mock_bank(d=16)
    with torch.no_grad():
        for i, t in enumerate(bank["targets_16"]):
            model.lm_head.weight[t["id"]] = bank["u_vectors"][i] * model.model.norm.weight

    # Preflight refuses missing file or wrong hash BEFORE loading weights
    with pytest.raises(FileNotFoundError):
        verify_preflight_provenance(
            tmp_path / "missing.pt", C0B_CONST_PATH, C0_RECEIPT_PATH, C0B_DESIGN_PATH
        )
    bad_design = tmp_path / "bad_design.md"
    bad_design.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_preflight_provenance(
            C0B_CONST_PATH, C0B_CONST_PATH, C0_RECEIPT_PATH, bad_design
        )

    # Hook validation checks
    with pytest.raises(ValueError):
        with post_norm_inject_hook(model, 12, torch.zeros(16), 1.0, "all"):
            pass
    with pytest.raises(ValueError):
        with post_norm_inject_hook(model, 12, bank["layers_bank"][12]["R_mf"][0], 1.0, "middle"):
            pass

    rec = run_c0b(
        model, tok, bank, tmp_path / "run1", device="cpu", smoke=True, bank_file_sha256="deadbeef"
    )
    for mode in ("last", "all"):
        for L in ("12", "18"):
            assert rec["shams"][mode][L] == 0.0
            assert rec["sham_movement_check"][mode][L] > 0.0

    t_all = rec["regime_D_all"]["12"]["R_mf"][" dolphin"][0]
    t_last = rec["regime_D_last_anchor"]["12"]["R_mf"][" dolphin"][0]
    assert t_all["telescoping_err"] < 1e-4
    assert t_last["telescoping_err"] < 1e-4
    assert abs(t_all["sum_attn"] - t_last["sum_attn"]) > 1e-4

    # Verified checkpoint resume succeeds cleanly
    rec2 = run_c0b(
        model, tok, bank, tmp_path / "run1", device="cpu", smoke=True, bank_file_sha256="deadbeef"
    )
    assert rec2["meta"]["total_forwards_executed"] == rec["meta"]["total_forwards_executed"]

    # Mutated bank_file_sha256 or device triggers hard provenance refusal
    with pytest.raises(RuntimeError, match="PROVENANCE REFUSAL"):
        run_c0b(
            model, tok, bank, tmp_path / "run1", device="cpu", smoke=True, bank_file_sha256="mutated_sha"
        )
    bank_mut = copy.deepcopy(bank)
    bank_mut["meta"]["model"] = "wrong-model"
    with pytest.raises(RuntimeError, match="PROVENANCE REFUSAL"):
        run_c0b(
            model, tok, bank_mut, tmp_path / "run1", device="cpu", smoke=True, bank_file_sha256="deadbeef"
        )


# ---------------------------------------------------------------------------
# Helper: construct a full registered Stage C0b synthetic receipt
# ---------------------------------------------------------------------------
def _make_full_synthetic_c0b_receipt(
    r_all_by_layer_fmt: dict[tuple[int, str], float] | None = None,
    r_all_target_offsets: list[float] | None = None,
    attn_share_by_target: list[float] | None = None,
    attn_share_target: float = 0.80,
    ridge_scale_by_target: list[float] | None = None,
    ridge_scale_vs_rmf: float = 0.25,
    ridge_self_mult: float = 1.0812,
    drift_n_offtarget: float = 0.0,
) -> dict:
    with gzip.open(C0_RECEIPT_PATH, "rt", encoding="utf-8") as f:
        c0_rec = json.load(f)
    c0_const = json.loads(C0_CONST_PATH.read_text(encoding="utf-8"))

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
    if r_all_target_offsets is None:
        r_all_target_offsets = [0.002 * (ti - 7.5) for ti in range(16)]
    if attn_share_by_target is None:
        attn_share_by_target = [attn_share_target] * 16
    if ridge_scale_by_target is None:
        ridge_scale_by_target = [ridge_scale_vs_rmf * (1.0 + 0.005 * (j - 7.5)) for j in range(16)]

    post_norms = {str(L): c0_rec["post_mlp_norms"][str(L)] for L in LAYERS_4L}
    rd_all: dict = {}
    rd_anc: dict = {}
    rn_last: dict = {}
    rn_all: dict = {}

    for L in LAYERS_4L:
        ls = str(L)
        c0_rd = c0_rec["regime_D"][ls]
        rd_anc[ls] = {
            "positions": "last",
            "target_direct_linear": 3.0,
            "dose_cap": c0_rd["dose_cap"],
            "R_mf": {ts: copy.deepcopy(c0_rd["R_mf"][ts]) for ts in ANCHOR_TARGETS_REGIME_D},
            "direct_matched": {ts: copy.deepcopy(c0_rd["direct_matched"][ts]) for ts in ANCHOR_TARGETS_REGIME_D},
        }
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
            rd_all[ls]["direct_matched"][ts] = copy.deepcopy(c0_rd["direct_matched"][ts])
            rd_all[ls]["logit_lens"][ts] = copy.deepcopy(c0_rd["logit_lens"][ts])
            rmf_rows = copy.deepcopy(c0_rd["R_mf"][ts])
            for pi, p in enumerate(prompts):
                fmt = p["format"]
                target_r = r_all_by_layer_fmt[(L, fmt)] + r_all_target_offsets[ti]
                dm_ex = sum(c0_rd["direct_matched"][ts][k][pi]["exact_dlogit"] for k in range(4)) / 4.0
                dm_at = sum(c0_rd["direct_matched"][ts][k][pi]["sum_attn"] for k in range(4)) / 4.0
                c0_gap = c0_rd["R_mf"][ts][pi]["exact_dlogit"] - dm_ex
                c0_at_gap = c0_rd["R_mf"][ts][pi]["sum_attn"] - dm_at
                new_gap = target_r * pg
                delta_tot = new_gap - c0_gap
                delta_attn = attn_share_by_target[ti] * delta_tot
                delta_mlp = (1.0 - attn_share_by_target[ti]) * delta_tot
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
                if drift_n_offtarget != 0.0:
                    rmf_rows[pi]["all_16_dlogits"] = [
                        round(v + (drift_n_offtarget if j != ti else 0.0), 6)
                        for j, v in enumerate(rmf_rows[pi]["all_16_dlogits"])
                    ]
                base_all16 = c0_rn["R_mf"][ts][pi]["all_16_dlogits"]
                rr_self = float(c0_rn["R_mf"][ts][pi]["exact_dlogit"]) * ridge_self_mult
                rr_rows[pi]["exact_dlogit"] = round(rr_self, 6)
                rr_rows[pi]["all_16_dlogits"] = [
                    round(rr_self if j == ti else base_all16[j] * ridge_scale_by_target[j], 6)
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
        c0_receipt_path=kwargs.pop("c0_receipt_path", C0_RECEIPT_PATH),
        design_path=kwargs.pop("design_path", C0B_DESIGN_PATH),
        n_boot=kwargs.pop("n_boot", 400),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 2. Clean PASS + literal registered constants verification
# ---------------------------------------------------------------------------
def test_score_c0b_clean_full_pass_and_literal_constants(tmp_path: Path):
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
    # Assert registered literal constants directly
    assert abs(sc["C0b_1b"]["delta_tot_guard"] - 0.25681) < 1e-6
    assert abs(sc["C0b_1b"]["attn_lift_floor"] - 1.7031) < 1e-4
    assert abs(sc["C0b_2b"]["F_mean_3L_C0"] - 0.295811) < 1e-6
    assert abs(sc["C0b_2b"]["noise_floor_threshold"] - 0.395811) < 1e-6


# ---------------------------------------------------------------------------
# 3. Must-fix #1: Dead R_ridge / dead logit_lens arms MUST fail closed!
# ---------------------------------------------------------------------------
def test_dead_ridge_and_dead_logit_lens_arms_fail_closed(tmp_path: Path):
    # Zeroed R_ridge Regime-N arm -> MUST fail C0b.0.1 and block C0b.2a & C0b.2b!
    rec_dead = _make_full_synthetic_c0b_receipt()
    for L in LAYERS_3L:
        for ts, rows in rec_dead["regime_N_last"][str(L)]["R_ridge"].items():
            for r in rows:
                r["exact_dlogit"] = 0.0
                r["all_16_dlogits"] = [0.0] * 16
    sc_dead = _score(rec_dead, tmp_path, "dead_ridge.json.gz")
    assert sc_dead["gate_C0b_0"]["C0b_0_1_sham"]["passed"] is False
    assert sc_dead["gate_C0b_0"]["C0b_0_1_sham"]["regime_n_arm_liveness_passed"] is False
    assert sc_dead["verdicts"]["C0b.0_global"] == "FAIL"
    assert sc_dead["verdicts"]["C0b.2a"] == "BLOCKED_BY_C0b.0"
    assert sc_dead["verdicts"]["C0b.2b"] == "BLOCKED_BY_C0b.0"

    # Zeroed off-target all_16_dlogits on logit_lens -> MUST also fail C0b.0.1!
    rec_dead_ll = _make_full_synthetic_c0b_receipt()
    rec_dead_ll["regime_N_last"]["18"]["logit_lens"][" dolphin"][0]["all_16_dlogits"] = [
        rec_dead_ll["regime_N_last"]["18"]["logit_lens"][" dolphin"][0]["exact_dlogit"] if j == 0 else 0.0
        for j in range(16)
    ]
    sc_ll = _score(rec_dead_ll, tmp_path, "dead_ll.json.gz")
    assert sc_ll["gate_C0b_0"]["C0b_0_1_sham"]["passed"] is False
    assert sc_ll["verdicts"]["C0b.0_global"] == "FAIL"


# ---------------------------------------------------------------------------
# 4. Boundary/mutation-killing tests for EVERY registered bar in C0b.1a
# ---------------------------------------------------------------------------
def test_c0b1a_all_boundaries(tmp_path: Path):
    # Pooled lower 0.65: 0.64 -> REFUTED
    g_lo = {(L, f): 0.64 for L in LAYERS_4L for f in ("bare", "chatml")}
    assert _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_lo), tmp_path, "b1.json.gz")["verdicts"]["C0b.1a"] == "REFUTED"

    # Pooled upper 1.35: 1.36 -> REFUTED
    g_hi = {(L, f): 1.36 for L in LAYERS_4L for f in ("bare", "chatml")}
    assert _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_hi), tmp_path, "b2.json.gz")["verdicts"]["C0b.1a"] == "REFUTED"

    # 90% BCa CI lower q05 > 0.55: point=0.68 in [0.65,1.35], but wide cluster spread pulls q05 <= 0.55 -> REFUTED
    wide_offs = [-0.20] * 7 + [0.0, 0.0] + [0.25] * 7
    g_ci = {(L, f): 0.68 for L in LAYERS_4L for f in ("bare", "chatml")}
    sc_ci = _score(
        _make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_ci, r_all_target_offsets=wide_offs),
        tmp_path, "b3.json.gz",
    )
    assert 0.65 <= sc_ci["C0b_1a"]["pooled_4L_median_r_all"] <= 1.35
    assert sc_ci["C0b_1a"]["pooled_4L_90pct_bca_ci"]["q_0.05"] <= 0.55
    assert sc_ci["C0b_1a"]["clauses"]["i_pooled_calib_and_90ci"] is False
    assert sc_ci["verdicts"]["C0b.1a"] == "REFUTED"

    # Per-layer format split ONLY (L12 ChatML=0.55 vs Bare=0.98) -> INCONCLUSIVE
    grid_inc = {(L, f): 0.92 for L in LAYERS_4L for f in ("bare", "chatml")}
    grid_inc[(12, "chatml")] = 0.55
    grid_inc[(12, "bare")] = 0.98
    sc_inc = _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_inc), tmp_path, "b4.json.gz")
    assert sc_inc["verdicts"]["C0b.1a"] == "INCONCLUSIVE"

    # Bad (L, fmt) cell (< 0.50 or > 1.50) -> REFUTED
    grid_ref = dict(grid_inc)
    grid_ref[(12, "chatml")] = 0.48
    assert _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_ref), tmp_path, "b5.json.gz")["verdicts"]["C0b.1a"] == "REFUTED"
    grid_ref2 = {(L, f): 0.95 for L in LAYERS_4L for f in ("bare", "chatml")}
    grid_ref2[(27, "chatml")] = 1.52
    assert _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=grid_ref2), tmp_path, "b6.json.gz")["verdicts"]["C0b.1a"] == "REFUTED"


# ---------------------------------------------------------------------------
# 5. Boundary/mutation-killing tests for EVERY registered bar in C0b.1b
# ---------------------------------------------------------------------------
def test_c0b1b_all_boundaries(tmp_path: Path):
    # (a) Tiny ΔAttn (< 1.7031 literal floor) even with high share=0.75 -> FAIL
    g_small = {(L, f): 0.45 for L in LAYERS_4L for f in ("bare", "chatml")}
    sc1 = _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_small, attn_share_target=0.75), tmp_path, "1b_1.json.gz")
    assert sc1["C0b_1b"]["mean_delta_attn"] < 1.7031
    assert sc1["verdicts"]["C0b.1b"] == "FAIL"

    # (b) Only 12/16 targets > delta_guard (4 targets set to negative lift) -> FAIL even with high mean ΔAttn
    offs_12 = [-0.85] * 4 + [0.15] * 12
    g_hi = {(L, f): 0.95 for L in LAYERS_4L for f in ("bare", "chatml")}
    sc2 = _score(_make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_hi, r_all_target_offsets=offs_12), tmp_path, "1b_2.json.gz")
    assert sc2["C0b_1b"]["n_targets_gt_delta"] == 12
    assert sc2["C0b_1b"]["mean_delta_attn"] >= 1.7031
    assert sc2["verdicts"]["C0b.1b"] == "FAIL"

    # (c) Median share >= 0.50 (0.52) but BCa q05 <= 0.35 -> FAIL
    shares_noisy = [0.20] * 7 + [0.52, 0.52] + [0.85] * 7
    sc3 = _score(
        _make_full_synthetic_c0b_receipt(r_all_by_layer_fmt=g_hi, attn_share_by_target=shares_noisy),
        tmp_path, "1b_3.json.gz",
    )
    assert sc3["C0b_1b"]["median_share"] >= 0.50
    assert sc3["C0b_1b"]["share_90pct_bca_ci"]["q_0.05"] <= 0.35
    assert sc3["verdicts"]["C0b.1b"] == "FAIL"


# ---------------------------------------------------------------------------
# 6. Boundary/mutation-killing tests for EVERY registered bar in C0b.2a & C0b.2b
# ---------------------------------------------------------------------------
def test_c0b2a_and_c0b2b_all_boundaries(tmp_path: Path):
    # (a) No whitening (ratio 1.00, self_mult 1.08) -> REFUTED
    sc_ref = _score(_make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=1.00, ridge_self_mult=1.08), tmp_path, "2_ref.json.gz")
    assert sc_ref["verdicts"]["C0b.2a"] == "REFUTED"
    assert sc_ref["verdicts"]["C0b.2b"] == "FAIL"

    # (b) Mirror inconclusive (abs=1.20 q95<1.25, norm=1.30 q95>1.25) -> INCONCLUSIVE
    sc_inc = _score(_make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=1.0 / 1.20, ridge_self_mult=1.0812), tmp_path, "2_inc.json.gz")
    assert sc_inc["verdicts"]["C0b.2a"] == "INCONCLUSIVE"

    # (c) Only 12/16 targets positive (even with abs_ratio > 1.25 and norm_ratio > 1.25) -> NOT PASS!
    scales_12 = [1.05] * 4 + [0.55] * 12
    sc_12 = _score(_make_full_synthetic_c0b_receipt(ridge_scale_by_target=scales_12), tmp_path, "2_12.json.gz")
    assert sc_12["C0b_2a"]["n_targets_Rmf_gt_Rridge"] == 12
    assert sc_12["C0b_2a"]["abs_ratio_geo_Rmf_over_Rridge"] >= 1.25
    assert sc_12["verdicts"]["C0b.2a"] != "PASS"

    # (d) 13/16 positive, abs_ratio >= 1.25, norm_ratio >= 1.25, but 3 strong negatives make p_ridge > 0.05 -> NOT PASS!
    # u_i = +0.55 for 13 targets, -1.80 for 3 targets -> mean u = +0.109; wait: need mean u >= log(1.25)=0.223
    # u_i = +0.85 for 13 targets, -2.40 for 3 targets -> mean u = (11.05-7.20)/16 = +0.2406 (exp=1.272 >= 1.25!),
    # and sign-flip p > 0.05!
    scales_p = [math.exp(2.40)] * 3 + [math.exp(-0.85)] * 13
    sc_p = _score(_make_full_synthetic_c0b_receipt(ridge_scale_by_target=scales_p), tmp_path, "2_p.json.gz")
    assert sc_p["C0b_2a"]["n_targets_Rmf_gt_Rridge"] == 13
    assert sc_p["C0b_2a"]["abs_ratio_geo_Rmf_over_Rridge"] >= 1.25
    assert sc_p["C0b_2a"]["p_ridge_sign_flip"] > 0.05
    assert sc_p["verdicts"]["C0b.2a"] != "PASS"

    # (e) abs_ratio >= 1.25 (1.30) & 16/16 positive & p<=0.05, but self_ratio=0.90 -> norm_ratio=1.17 < 1.25 -> NOT PASS!
    sc_norm = _score(_make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=1.0 / 1.30, ridge_self_mult=0.90), tmp_path, "2_norm.json.gz")
    assert sc_norm["C0b_2a"]["abs_ratio_geo_Rmf_over_Rridge"] >= 1.25
    assert sc_norm["C0b_2a"]["norm_ratio_geo_Smf_over_Sridge"] < 1.25
    assert sc_norm["verdicts"]["C0b.2a"] != "PASS"

    # (f) C0b.2b exact boundary around 0.395811: ~0.41 fails while ~0.38 passes
    sc_b_fail = _score(_make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=0.31), tmp_path, "2b_fail.json.gz")
    assert sc_b_fail["C0b_2b"]["R_ridge_B_cross_arith_mean"] > 0.395811
    assert sc_b_fail["verdicts"]["C0b.2a"] == "PASS"
    assert sc_b_fail["verdicts"]["C0b.2b"] == "FAIL"

    sc_b_pass = _score(_make_full_synthetic_c0b_receipt(ridge_scale_vs_rmf=0.28), tmp_path, "2b_pass.json.gz")
    assert sc_b_pass["C0b_2b"]["R_ridge_B_cross_arith_mean"] <= 0.395811
    assert sc_b_pass["verdicts"]["C0b.2b"] == "PASS"


# ---------------------------------------------------------------------------
# 7. Gate C0b.0 (1-4a) & each C0b.0.4b clause + crash-free blocked returns
# ---------------------------------------------------------------------------
def test_gate_c0b0_and_4b_individual_clauses_and_crash_free_blocks(tmp_path: Path):
    # Nonzero sham fails C0b.0.1
    r_sh = _make_full_synthetic_c0b_receipt()
    r_sh["shams"]["all"]["12"] = 1e-5
    assert _score(r_sh, tmp_path, "g_sh.json.gz")["gate_C0b_0"]["C0b_0_1_sham"]["passed"] is False

    # Telescoping p95 > 0.08 fails C0b.0.2
    r_t95 = _make_full_synthetic_c0b_receipt()
    for ts in [t["string"] for t in r_t95["targets_16"]]:
        for pi in range(8):
            r_t95["regime_D_all"]["12"]["R_mf"][ts][pi]["telescoping_err"] = 0.09
            r_t95["regime_D_all"]["18"]["R_mf"][ts][pi]["telescoping_err"] = 0.09
            r_t95["regime_D_all"]["24"]["R_mf"][ts][pi]["telescoping_err"] = 0.09
    assert _score(r_t95, tmp_path, "g_t95.json.gz")["gate_C0b_0"]["C0b_0_2_telescoping"]["passed"] is False

    # Telescoping p99.9 > 0.15 fails C0b.0.2 even with p95 <= 0.08
    r_t999 = _make_full_synthetic_c0b_receipt()
    for pi in range(8):
        r_t999["regime_D_all"]["12"]["R_mf"][" dolphin"][pi]["telescoping_err"] = 0.16
    assert _score(r_t999, tmp_path, "g_t999.json.gz")["gate_C0b_0"]["C0b_0_2_telescoping"]["passed"] is False

    # FDE p95 > 0.30 and p99.9 > 0.45 each fail C0b.0.3
    r_f95 = _make_full_synthetic_c0b_receipt()
    for ts in [t["string"] for t in r_f95["targets_16"]]:
        for pi in range(8):
            r_f95["regime_D_all"]["12"]["R_mf"][ts][pi]["frozen_direct_effect"] = 3.35
            r_f95["regime_D_all"]["18"]["R_mf"][ts][pi]["frozen_direct_effect"] = 3.35
    assert _score(r_f95, tmp_path, "g_f95.json.gz")["gate_C0b_0"]["C0b_0_3_cap_and_denom"]["passed"] is False

    r_f999 = _make_full_synthetic_c0b_receipt()
    for pi in range(6):
        r_f999["regime_D_all"]["12"]["R_mf"][" dolphin"][pi]["frozen_direct_effect"] = 3.48
    assert _score(r_f999, tmp_path, "g_f999.json.gz")["gate_C0b_0"]["C0b_0_3_cap_and_denom"]["passed"] is False

    # All 4 bare prompts capped in one cell -> MUST return blocked scorecard cleanly (NO crash in mean([]))!
    r_allbare = _make_full_synthetic_c0b_receipt()
    for pi in range(4):
        r_allbare["regime_D_all"]["12"]["R_mf"][" dolphin"][pi] = {
            "prompt_id": r_allbare["prompts_8"][pi]["id"], "capped": True, "dose_requested": 15.0, "dose_cap": 12.7
        }
    r_allbare["meta"]["forward_counts"]["regime_D_all"] -= 4
    r_allbare["meta"]["forward_counts"]["regime_D_all_capped_skipped"] += 4
    r_allbare["meta"]["total_forwards_executed"] -= 4
    sc_ab = _score(r_allbare, tmp_path, "g_ab.json.gz")
    assert sc_ab["verdicts"]["C0b.0_global"] == "FAIL"
    assert sc_ab["verdicts"]["C0b.1a"] == "BLOCKED_BY_C0b.0"

    # Missing telescoping_err -> returns blocked scorecard cleanly (NO KeyError!)
    r_miss = _make_full_synthetic_c0b_receipt()
    del r_miss["regime_D_all"]["12"]["R_mf"][" dolphin"][0]["telescoping_err"]
    sc_miss = _score(r_miss, tmp_path, "g_miss.json.gz")
    assert sc_miss["verdicts"]["C0b.0_global"] == "FAIL"
    assert sc_miss["verdicts"]["C0b.1a"] == "BLOCKED_BY_C0b.0"

    # Gate C0b.0.4b individual clauses (each voids ONLY C0b.1b, never C0b.1a or live C0b.2!)
    # (1) anchor gap diff > 0.05
    r_ag = _make_full_synthetic_c0b_receipt()
    for L in LAYERS_4L:
        for pi in range(8):
            r_ag["regime_D_last_anchor"][str(L)]["R_mf"][" lantern"][pi]["exact_dlogit"] += 0.08
    sc_ag = _score(r_ag, tmp_path, "4b_ag.json.gz")
    assert sc_ag["verdicts"]["C0b.0_global"] == "PASS"
    assert sc_ag["verdicts"]["C0b.0.4b_drift"] == "FAIL"
    assert sc_ag["verdicts"]["C0b.1b"] == "VOID_DRIFT"
    assert sc_ag["verdicts"]["C0b.2a"] == "PASS"

    # (2) anchor attn diff > 0.05
    r_aa = _make_full_synthetic_c0b_receipt()
    for L in LAYERS_4L:
        for pi in range(8):
            r_aa["regime_D_last_anchor"][str(L)]["R_mf"][" lantern"][pi]["sum_attn"] += 0.08
    sc_aa = _score(r_aa, tmp_path, "4b_aa.json.gz")
    assert sc_aa["verdicts"]["C0b.0.4b_drift"] == "FAIL"
    assert sc_aa["verdicts"]["C0b.1b"] == "VOID_DRIFT"

    # (3) B_cross diff > 0.02 (with per-cell <= 0.07)
    r_bc = _make_full_synthetic_c0b_receipt(drift_n_offtarget=0.04)
    sc_bc = _score(r_bc, tmp_path, "4b_bc.json.gz")
    assert sc_bc["gate_C0b_0"]["C0b_0_4b_c0b1b_receipt_drift"]["max_R_mf_offtarget_cell_diff"] <= 0.07
    assert sc_bc["gate_C0b_0"]["C0b_0_4b_c0b1b_receipt_drift"]["max_R_mf_B_cross_diff"] > 0.02
    assert sc_bc["verdicts"]["C0b.0.4b_drift"] == "FAIL"
    assert sc_bc["verdicts"]["C0b.1b"] == "VOID_DRIFT"
    assert sc_bc["verdicts"]["C0b.2a"] == "PASS"

    # (4) Single off-target cell diff > 0.07 (with B_cross <= 0.02)
    r_cl = _make_full_synthetic_c0b_receipt()
    r_cl["regime_N_last"]["18"]["R_mf"][" dolphin"][0]["all_16_dlogits"][5] += 0.09
    sc_cl = _score(r_cl, tmp_path, "4b_cl.json.gz")
    assert sc_cl["gate_C0b_0"]["C0b_0_4b_c0b1b_receipt_drift"]["max_R_mf_B_cross_diff"] <= 0.02
    assert sc_cl["gate_C0b_0"]["C0b_0_4b_c0b1b_receipt_drift"]["max_R_mf_offtarget_cell_diff"] > 0.07
    assert sc_cl["verdicts"]["C0b.0.4b_drift"] == "FAIL"

    # (5) Item 7: 1 capped anchor draw drops prompt & still PASSES C0b.0.4b (3/4 Bare retained!)
    r_acap = _make_full_synthetic_c0b_receipt()
    r_acap["regime_D_last_anchor"]["12"]["direct_matched"][" lantern"][0][0] = {
        "prompt_id": "bare_narr_0", "capped": True, "dose_requested": 15.0, "dose_cap": 12.7
    }
    r_acap["meta"]["forward_counts"]["regime_D_last_anchor"] -= 1
    r_acap["meta"]["forward_counts"]["regime_D_last_anchor_capped_skipped"] += 1
    r_acap["meta"]["total_forwards_executed"] -= 1
    sc_acap = _score(r_acap, tmp_path, "4b_acap.json.gz")
    assert sc_acap["verdicts"]["C0b.0.4b_drift"] == "PASS"
    assert sc_acap["verdicts"]["C0b.1b"] == "PASS"


# ---------------------------------------------------------------------------
# 8. Schema / ordering / forward-count / C0-receipt / provenance refusals
# ---------------------------------------------------------------------------
def test_schema_order_forward_counts_and_c0_receipt_refusals(tmp_path: Path):
    # Swapped prompt order (bare <-> chatml row swap) refused!
    r_swap = _make_full_synthetic_c0b_receipt()
    rows = r_swap["regime_D_all"]["12"]["R_mf"][" dolphin"]
    rows[0], rows[4] = rows[4], rows[0]
    with pytest.raises(ValueError, match="prompt_id"):
        _score(r_swap, tmp_path, "swap.json.gz")

    # Zeroed forward counts refused!
    r_fc = _make_full_synthetic_c0b_receipt()
    r_fc["meta"]["forward_counts"] = {k: 0 for k in r_fc["meta"]["forward_counts"]}
    r_fc["meta"]["total_forwards_executed"] = 0
    with pytest.raises(ValueError, match="forward_counts"):
        _score(r_fc, tmp_path, "fc0.json.gz")

    # Wrong-structure C0 reference receipt raises ValueError (never raw KeyError)
    bad_c0 = tmp_path / "bad_c0.json"
    bad_c0.write_text(json.dumps({"not_regime_D": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="C0 reference receipt"):
        _score(_make_full_synthetic_c0b_receipt(), tmp_path, "badc0.json.gz", c0_receipt_path=bad_c0)

    # Wrong dose_cap refused!
    r_cap = _make_full_synthetic_c0b_receipt()
    r_cap["regime_D_all"]["12"]["dose_cap"] = 999.0
    with pytest.raises(ValueError, match="dose_cap"):
        _score(r_cap, tmp_path, "badcap.json.gz")

    # Smoke receipt refused
    r_sm = _make_full_synthetic_c0b_receipt()
    r_sm["meta"]["smoke"] = True
    with pytest.raises(ValueError, match="smoke"):
        _score(r_sm, tmp_path, "sm.json.gz")

    # Missing design / relaxed provenance fail C0b.0.4a
    r_ok = _make_full_synthetic_c0b_receipt()
    assert _score(r_ok, tmp_path, "nd.json.gz", design_path=None)["gate_C0b_0"]["C0b_0_4a_global_provenance"]["passed"] is False
    assert _score(r_ok, tmp_path, "rl.json.gz", strict_provenance=False)["gate_C0b_0"]["C0b_0_4a_global_provenance"]["passed"] is False


# ---------------------------------------------------------------------------
# 9. Stdlib-only AST + bare `/usr/bin/python3 -I` byte-identical determinism
# ---------------------------------------------------------------------------
def test_scorer_is_stdlib_only_and_byte_deterministic_under_isolated_python3(tmp_path: Path):
    src = (REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0b.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    stdlib_names = sys.stdlib_module_names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib_names, f"Non-stdlib import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in stdlib_names, f"Non-stdlib import: {node.module}"

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
