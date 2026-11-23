"""Unit & end-to-end tests for Stage C0 (`build_olmo3_c0_bank.py`, `run_olmo3_c0.py`, `score_olmo3_c0.py`)."""
from __future__ import annotations

import gzip
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "olmo3_virtual_and_highway"))

import score_olmo3_c0 as sc0  # noqa: E402

CONSTANTS_PATH = REPO_ROOT / "olmo3_virtual_and_highway/c0_prereg_constants.json"
DESIGN_PATH = REPO_ROOT / "olmo3_virtual_and_highway/OLMO3-TRANSPORT-C0-DESIGN.md"


# ---------------------------------------------------------------------------
# 1. Pure-stdlib statistical primitives
# ---------------------------------------------------------------------------
def test_quantile_linear_interp():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert sc0.quantile(xs, 0.0) == 1.0
    assert sc0.quantile(xs, 0.5) == 3.0
    assert sc0.quantile(xs, 1.0) == 5.0
    assert abs(sc0.quantile(xs, 0.25) - 2.0) < 1e-12
    assert abs(sc0.quantile([10.0, 20.0], 0.95) - 19.5) < 1e-12


def test_norm_ppf_accuracy():
    for p in [1e-4, 0.001, 0.0125, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975, 0.999]:
        z = sc0.norm_ppf(p)
        p_back = sc0.norm_cdf(z)
        assert abs(p_back - p) < 1e-8, f"p={p}, z={z}, p_back={p_back}"


def test_exact_sign_flip_matches_bruteforce():
    vals = [0.12, -0.03, 0.25, 0.08, 0.19, -0.01, 0.14, 0.05]
    p_fast = sc0.exact_sign_flip_p_one_sided(vals)
    # Brute force over 2^8
    obs = sum(vals)
    ge = 0
    for mask in range(1 << len(vals)):
        s = sum(vals[i] if (mask >> i) & 1 else -vals[i] for i in range(len(vals)))
        if s >= obs - 1e-12:
            ge += 1
    p_brute = ge / (1 << len(vals))
    assert abs(p_fast - p_brute) < 1e-12
    # All positive 16 values -> exact 1 / 65536
    p16 = sc0.exact_sign_flip_p_one_sided([1.0] * 16)
    assert abs(p16 - (1.0 / 65536.0)) < 1e-12


def test_holm_bonferroni_stepdown():
    h = sc0.holm_bonferroni({"A": 0.001, "B": 0.014, "C": 0.020, "D": 0.10}, alpha=0.05)
    assert h["A"]["rejected"] is True
    assert h["B"]["rejected"] is True   # 0.014 <= 0.05/3 = 0.01667
    assert h["C"]["rejected"] is True   # 0.020 <= 0.05/2 = 0.025
    assert h["D"]["rejected"] is False
    # Failure at rank 1 blocks rank 2 even if rank 2 < 0.05/2
    h2 = sc0.holm_bonferroni({"A": 0.001, "B": 0.020, "C": 0.021, "D": 0.10}, alpha=0.05)
    assert h2["A"]["rejected"] is True
    assert h2["B"]["rejected"] is False  # 0.020 > 0.01667
    assert h2["C"]["rejected"] is False  # blocked by B


def test_spearman_rho_and_ties():
    assert abs(sc0.spearman_rho([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) - 1.0) < 1e-12
    assert abs(sc0.spearman_rho([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) + 1.0) < 1e-12
    assert sc0.spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]) == 0.0


def test_bca_deterministic_and_sensible():
    xs = [1.0 + 0.1 * i for i in range(16)]
    ci1 = sc0.bca_cluster_ci(16, lambda idx: sc0.mean([xs[i] for i in idx]), (0.05, 0.95), n_boot=500, seed=7)
    ci2 = sc0.bca_cluster_ci(16, lambda idx: sc0.mean([xs[i] for i in idx]), (0.05, 0.95), n_boot=500, seed=7)
    assert ci1 == ci2
    assert ci1["q_0.05"] < ci1["point"] < ci1["q_0.95"]


# ---------------------------------------------------------------------------
# 2. Vector bank math invariants (if torch is available)
# ---------------------------------------------------------------------------
def test_direct_matched_orthogonality_and_invariants():
    torch = pytest.importorskip("torch")
    import build_olmo3_c0_bank as bank_mod

    g = torch.Generator().manual_seed(99)
    d = 128
    u_hat = bank_mod.unit(torch.randn(d, generator=g))
    r_mat = torch.randn(d, d, generator=g) / math.sqrt(d)
    rmf_raw = r_mat.T @ u_hat
    w_rmf = bank_mod.unit(rmf_raw)
    c = float(torch.dot(u_hat, w_rmf).item())
    if c < 0:
        w_rmf = -w_rmf
        c = -c
    w_rr = bank_mod.unit(torch.randn(d, generator=g))

    draws = bank_mod.make_direct_matched_draws(u_hat, w_rmf, w_rr, c=c, k_draws=4, seed_base=12345)
    assert draws.shape == (4, d)
    for k in range(4):
        w = draws[k].double()
        assert abs(float(w.norm().item()) - 1.0) < 1e-5
        assert abs(float(torch.dot(w, u_hat.double()).item()) - c) < 1e-5
        # Linear forecast u_hat^T R w must equal c * (u_hat^T R u_hat) because z_perp _|_ R^T u_hat!
        lhs = float(torch.dot(u_hat.double(), r_mat.double() @ w).item())
        rhs = c * float(torch.dot(u_hat.double(), r_mat.double() @ u_hat.double()).item())
        assert abs(lhs - rhs) < 1e-5


# ---------------------------------------------------------------------------
# 3. Synthetic Stage C0 receipt builder & full end-to-end stdlib scorer tests
# ---------------------------------------------------------------------------
def _make_synthetic_receipt(
    mode: str = "pass",
    sham_err: float = 0.0,
    tel_err: float = 0.01,
    cap_rate_inject: bool = False,
    bad_provenance: bool = False,
    single_pair_leak: float = 0.0,
) -> dict:
    """Build a realistic Stage C0 receipt matching `run_olmo3_c0.py` exact schema."""
    const = json.loads(CONSTANTS_PATH.read_text(encoding="utf-8"))
    targets = const["targets_16"]
    layers = const["layers"]
    lg = const["layer_geometry"]
    prompts = [
        {"id": f"{fmt}_{sty}_{i}", "format": fmt, "style": sty, "text": f"prompt {fmt} {sty} {i}"}
        for fmt in ("bare", "chatml")
        for sty in ("narrative", "analytical")
        for i in range(2)
    ]
    rng = random.Random(42)
    k_dm = 4
    m_rand = 32
    rn = {}
    rd = {}
    post_norms = {str(L): 2.0 + 0.4 * L for L in layers}

    for L in layers:
        l_str = str(L)
        dose_n = 0.75 * post_norms[l_str]
        tr_d = lg[l_str]["trace_R_over_d"]
        rn[l_str] = {
            "dose_N": dose_n,
            "post_mlp_norm": post_norms[l_str],
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
            "rand_iso": [],
        }
        rd[l_str] = {
            "target_direct_linear": 3.0,
            "dose_cap": 3.0 * post_norms[l_str],
            "R_mf": {},
            "R_ridge": {},
            "logit_lens": {},
            "direct_matched": {},
        }
        pts = {p["target"]: p for p in lg[l_str]["per_target"]}

        for ti, t in enumerate(targets):
            ts = t["string"]
            pt = pts[ts]
            u_norm = t["u_norm"]
            # Make pred_dlogit vary across targets via u_norm * pred_gain (independent of Direct_linear)
            for arm in ("R_mf", "R_ridge", "logit_lens", "direct_matched"):
                if arm == "R_mf":
                    pred_ret = pt["pred_ret_Rmf"]
                    dir_n = dose_n * u_norm * pt["c"]
                elif arm == "R_ridge":
                    pred_ret = pt["pred_ret_Rridge"]
                    dir_n = dose_n * u_norm * pt["cos_u_Rridge"]
                elif arm == "logit_lens":
                    pred_ret = pt["pred_ret_dm"]
                    dir_n = dose_n * u_norm
                else:
                    pred_ret = pt["pred_ret_dm"]
                    dir_n = dose_n * u_norm * pt["c"]

                def _mk_trace_n(p_idx: int, draw_k: int = 0) -> dict:
                    pred = pred_ret * dir_n
                    if mode == "pass":
                        # High correlation with pred (beyond Direct_linear) + accurate calibration
                        realized_ret = pred_ret * (0.96 if arm != "logit_lens" else 0.85) + rng.gauss(0.0, 0.03)
                        ex = realized_ret * dir_n + rng.gauss(0.0, 0.05)
                    else:
                        # Null: random noise independent of pred
                        ex = rng.gauss(0.0, 1.0)
                        realized_ret = ex / max(dir_n, 1e-6)
                    rec = {
                        "prompt_id": prompts[p_idx]["id"],
                        "dose": dose_n,
                        "exact_dlogit": ex,
                        "pred_dlogit": pred,
                        "direct_linear": dir_n,
                        "direct_linear_analytic": dir_n,
                        "frozen_direct_effect": dir_n * 0.99,
                        "sum_attn": ex - dir_n,
                        "sum_mlp": 0.0,
                        "rmsnorm_shift": 0.0,
                        "telescoping_err": tel_err,
                        "retention": realized_ret,
                        "canc_r": 1.5,
                        "dlogit_z": 10.0,
                        "kl": 0.5,
                        "rms_clean": 1.0,
                        "rms_int": 1.01,
                    }
                    if arm == "R_mf":
                        # Cross-domain wrong targets at ~rand_iso noise floor
                        all16 = [rng.gauss(0.0, 0.30) for _ in range(16)]
                        all16[ti] = ex
                        if single_pair_leak > 0.0 and ti == 0:
                            # Inject strong leak onto cross-domain target index 4 (professions vs animals)
                            all16[4] = single_pair_leak
                        rec["all_16_dlogits"] = all16
                    return rec

                def _mk_trace_d(p_idx: int, draw_k: int = 0) -> dict:
                    if cap_rate_inject and L == 12 and arm == "R_mf" and ti < 2:
                        return {"prompt_id": prompts[p_idx]["id"], "capped": True, "dose_requested": 99.0, "dose_cap": 10.0}
                    if mode == "pass":
                        ret_d = pred_ret * (0.95 if arm != "logit_lens" else 0.80) + rng.gauss(0.0, 0.02)
                        ex = ret_d * 3.0
                        dz = 22.0 if arm == "R_ridge" else 15.0
                        kl = 0.25 if arm == "R_ridge" else 0.45
                    else:
                        ret_d = 1.0 + rng.gauss(0.0, 0.2)
                        ex = ret_d * 3.0
                        dz = 10.0
                        kl = 0.5
                    return {
                        "prompt_id": prompts[p_idx]["id"],
                        "capped": False,
                        "dose": 3.0 / max(u_norm * pt["c"], 1e-4),
                        "exact_dlogit": ex,
                        "pred_dlogit": pred_ret * 3.0,
                        "direct_linear": 3.0,
                        "direct_linear_analytic": 3.0,
                        "frozen_direct_effect": 3.0 + rng.gauss(0.0, 0.04),
                        "sum_attn": ex - 3.0,
                        "sum_mlp": 0.0,
                        "rmsnorm_shift": 0.0,
                        "telescoping_err": tel_err,
                        "retention": ret_d,
                        "canc_r": 1.3,
                        "dlogit_z": dz + rng.gauss(0.0, 0.5),
                        "kl": kl,
                        "rms_clean": 1.0,
                        "rms_int": 1.01,
                    }

                if arm == "direct_matched":
                    rn[l_str][arm][ts] = [[_mk_trace_n(pi, k) for pi in range(8)] for k in range(k_dm)]
                    rd[l_str][arm][ts] = [[_mk_trace_d(pi, k) for pi in range(8)] for k in range(k_dm)]
                else:
                    rn[l_str][arm][ts] = [_mk_trace_n(pi) for pi in range(8)]
                    rd[l_str][arm][ts] = [_mk_trace_d(pi) for pi in range(8)]

        for m in range(m_rand):
            rows_p = []
            for pi in range(8):
                vjv = (tr_d + 0.02) if mode == "pass" else 0.0
                rows_p.append({
                    "prompt_id": prompts[pi]["id"],
                    "dose": dose_n,
                    "vJv": vjv + rng.gauss(0.0, 0.02),
                    "canc_r_self": 2.0,
                    "all_16_dlogits": [rng.gauss(0.0, 0.30) for _ in range(16)],
                })
            rn[l_str]["rand_iso"].append(rows_p)

    return {
        "meta": {
            "study": "stage_c0_olmo3_7b_transport",
            "model": const["model"],
            "revision": const["revision"],
            "smoke": False,
            "elapsed_sec": 1.0,
            "gpu_name": "synthetic",
            "r_lens_sha256": "deadbeef" if bad_provenance else sc0.EXPECTED_R_LENS_SHA256,
            "constants_sha256": sc0.EXPECTED_CONSTANTS_SHA256,
            "design_sha256": sc0.EXPECTED_DESIGN_SHA256,
            "k_dm": k_dm,
            "m_rand": m_rand,
            "forward_counts": {"clean": 8, "sham": 7, "regime_N": 8064, "regime_D": 6272, "regime_D_capped_skipped": 0},
            "total_forwards_executed": 14351,
        },
        "layers": layers,
        "targets_16": targets,
        "prompts_8": prompts,
        "post_mlp_norms": post_norms,
        "shams": {str(L): sham_err for L in layers},
        "regime_N": rn,
        "regime_D": rd,
    }


def test_end_to_end_stdlib_scorer_pass_and_byte_determinism(tmp_path: Path):
    rec = _make_synthetic_receipt(mode="pass")
    rec_path = tmp_path / "c0_receipt.json.gz"
    with gzip.open(rec_path, "wt", encoding="utf-8") as f:
        json.dump(rec, f)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    for out in (out1, out2):
        # Run with system python3 -I to guarantee 100% stdlib-only isolation!
        cmd = [
            sys.executable,
            "-I",
            str(REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0.py"),
            "--receipt",
            str(rec_path),
            "--constants",
            str(CONSTANTS_PATH),
            "--design",
            str(DESIGN_PATH),
            "--out-dir",
            str(out),
            "--n-boot",
            "500",
            "--n-perm",
            "500",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        assert "Stage C1 gate: True" in res.stdout

    assert (out1 / "scorecard.json").read_bytes() == (out2 / "scorecard.json").read_bytes()
    assert (out1 / "SCORECARD.md").read_bytes() == (out2 / "SCORECARD.md").read_bytes()

    sc = json.loads((out1 / "scorecard.json").read_text(encoding="utf-8"))
    assert sc["verdicts"] == {"C0.0": "PASS", "C0.1": "PASS", "C0.2": "PASS", "C0.3": "PASS", "C0.4": "PASS"}
    assert sc["stage_c1_gate_passed"] is True
    assert "bare_vs_chatml" in sc["moderation_2x2"]
    assert "narrative_vs_analytical" in sc["moderation_2x2"]


def test_scorer_null_fails_and_gates_block(tmp_path: Path):
    # 1. Null data -> C0.0 passes, C0.1-C0.4 fail, C1 blocked
    p_null = tmp_path / "null.json"
    p_null.write_text(json.dumps(_make_synthetic_receipt(mode="null")), encoding="utf-8")
    sc_null = sc0.score_c0(p_null, CONSTANTS_PATH, DESIGN_PATH, n_boot=200, n_perm=200)
    assert sc_null["verdicts"]["C0.0"] == "PASS"
    assert sc_null["verdicts"]["C0.1"] == "FAIL"
    assert sc_null["verdicts"]["C0.2"] == "FAIL"
    assert sc_null["verdicts"]["C0.4"] == "FAIL"
    assert sc_null["stage_c1_gate_passed"] is False

    # 2. Sham failure -> C0.0 FAIL blocks all C0.1-C0.4 and C1
    p_sham = tmp_path / "sham.json"
    p_sham.write_text(json.dumps(_make_synthetic_receipt(mode="pass", sham_err=1e-4)), encoding="utf-8")
    sc_sham = sc0.score_c0(p_sham, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_sham["verdicts"]["C0.0"] == "FAIL"
    assert sc_sham["verdicts"]["C0.1"] == "BLOCKED_BY_C0.0"
    assert sc_sham["stage_c1_gate_passed"] is False

    # 3. Cap rate > 2% -> C0.0.3 FAIL
    p_cap = tmp_path / "cap.json"
    p_cap.write_text(json.dumps(_make_synthetic_receipt(mode="pass", cap_rate_inject=True)), encoding="utf-8")
    sc_cap = sc0.score_c0(p_cap, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_cap["gate_C0_0"]["C0_0_3_regime_d_cap_and_denom"]["passed"] is False
    assert sc_cap["stage_c1_gate_passed"] is False

    # 4. Bad provenance -> C0.0.4 FAIL
    p_prov = tmp_path / "prov.json"
    p_prov.write_text(json.dumps(_make_synthetic_receipt(mode="pass", bad_provenance=True)), encoding="utf-8")
    sc_prov = sc0.score_c0(p_prov, CONSTANTS_PATH, DESIGN_PATH, n_boot=100, n_perm=100)
    assert sc_prov["gate_C0_0"]["C0_0_4_provenance"]["passed"] is False
    assert sc_prov["stage_c1_gate_passed"] is False

    # 5. M6 single-pair leak (3.5 * F_mean ~ 0.90) trips per-pair check even when aggregate mean passes!
    p_leak = tmp_path / "leak.json"
    p_leak.write_text(json.dumps(_make_synthetic_receipt(mode="pass", single_pair_leak=1.20)), encoding="utf-8")
    sc_leak = sc0.score_c0(p_leak, CONSTANTS_PATH, DESIGN_PATH, n_boot=200, n_perm=200)
    assert sc_leak["C0_4"]["cross_domain_null"]["aggregate_passed"] is True
    assert sc_leak["C0_4"]["cross_domain_null"]["per_pair_passed"] is False
    assert sc_leak["verdicts"]["C0.4"] == "FAIL"
    # C0.4 is non-blocking for Stage C1 gate!
    assert sc_leak["stage_c1_gate_passed"] is True


def test_run_c0_mock_olmo3_hooks_telescoping_sham_and_resume(tmp_path: Path):
    """End-to-end integration test of `run_olmo3_c0.run_c0` on a tiny post-norm OLMo-3 mock."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    import run_olmo3_c0 as r0

    class MockRMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))
            self.variance_epsilon = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
            return (x / rms) * self.weight

    class MockLayer(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.attn_proj = nn.Linear(d, d, bias=False)
            self.post_attention_layernorm = MockRMSNorm(d)
            self.mlp_proj = nn.Linear(d, d, bias=False)
            self.post_feedforward_layernorm = MockRMSNorm(d)

        def forward(self, h: torch.Tensor) -> torch.Tensor:
            a = self.post_attention_layernorm(self.attn_proj(h))
            h = h + a
            m = self.post_feedforward_layernorm(torch.tanh(self.mlp_proj(h)))
            return h + m

    class MockInner(nn.Module):
        def __init__(self, vocab: int, d: int, n_layers: int):
            super().__init__()
            self.embed_tokens = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([MockLayer(d) for _ in range(n_layers)])
            self.norm = MockRMSNorm(d)

    class MockOut:
        def __init__(self, logits: torch.Tensor):
            self.logits = logits

    class MockOLMo3(nn.Module):
        def __init__(self, vocab: int = 64, d: int = 32, n_layers: int = 4):
            super().__init__()
            self.config = type("Cfg", (), {"num_hidden_layers": n_layers})()
            self.model = MockInner(vocab, d, n_layers)
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
            h = self.model.embed_tokens(input_ids)
            for lay in self.model.layers:
                h = lay(h)
            return MockOut(self.lm_head(self.model.norm(h)))

    class MockTok:
        def encode(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False):
            ids = [(sum(ord(c) for c in w) % 60) + 1 for w in text.split()[:6]]
            return torch.tensor([ids], dtype=torch.long)

    torch.manual_seed(7)
    d = 32
    model = MockOLMo3(vocab=64, d=d, n_layers=4).eval()
    tok = MockTok()

    # Build a tiny 2-layer, 2-target bank matching exact run_c0 schema
    targets = [
        {"domain": "animals", "string": " dolphin", "id": 5, "decoded_piece": " dolphin"},
        {"domain": "places", "string": " harbor", "id": 11, "decoded_piece": " harbor"},
    ]
    w_u = model.lm_head.weight.detach().float()
    gamma = model.model.norm.weight.detach().float()
    u_vecs = torch.stack([w_u[t["id"]] * gamma for t in targets], dim=0)
    u_hats = u_vecs / u_vecs.norm(dim=-1, keepdim=True)

    layers_bank = {}
    for L in (0, 1):
        # Construct positive-dot unit vectors for all arms
        w_rmf = u_hats.clone()
        w_rr = u_hats.clone()
        w_ll = u_hats.clone()
        w_dm = u_hats.unsqueeze(1).clone()  # [2, 1, d]
        rand_iso = torch.randn(2, d)
        rand_iso = rand_iso / rand_iso.norm(dim=-1, keepdim=True)
        layers_bank[L] = {
            "lambda_ridge": 0.1,
            "tr_R_over_d": 0.5,
            "R_mf": w_rmf,
            "R_ridge": w_rr,
            "logit_lens": w_ll,
            "direct_matched": w_dm,
            "rand_iso": rand_iso,
            "per_target_geom": {
                t["string"]: {
                    "N": 1.0,
                    "c": 1.0,
                    "cos_u_Rridge": 1.0,
                    "pred_ret_Rmf": 1.0,
                    "pred_ret_Rridge": 1.0,
                    "pred_ret_dm": 1.0,
                    "pred_ret_ll": 1.0,
                    "pred_gap_D3": 1.0,
                }
                for t in targets
            },
        }

    bank = {
        "meta": {
            "model": "mock-olmo3",
            "revision": "test",
            "r_lens_sha256": "abc",
            "constants_sha256": "def",
            "design_sha256": "ghi",
            "k_dm": 1,
            "m_rand": 2,
        },
        "targets_16": targets,
        "layers": [0, 1],
        "u_vectors": u_vecs,
        "layers_bank": layers_bank,
    }

    out_dir = tmp_path / "run1"
    rec1 = r0.run_c0(model, tok, bank, out_dir, device="cpu", smoke=True)
    # 1. Sham is bit-for-bit 0.0
    assert all(v == 0.0 for v in rec1["shams"].values())
    # 2. Exact telescoping error < 1e-4 on fp32 mock + R_mf_cross self-index equals exact_dlogit
    for L in ("0", "1"):
        for ti, t in enumerate(targets):
            for row in rec1["regime_N"][L]["R_mf"][t["string"]]:
                assert row["telescoping_err"] < 1e-4
                assert abs(row["all_16_dlogits"][ti] - row["exact_dlogit"]) < 1e-5
            for row in rec1["regime_D"][L]["R_mf"][t["string"]]:
                if not row.get("capped", False):
                    assert abs(row["direct_linear"] - 3.0) < 1e-3
                    assert row["telescoping_err"] < 1e-4
    # 3. Checkpoint resume reproduces identical counts
    rec2 = r0.run_c0(model, tok, bank, out_dir, device="cpu", smoke=True)
    assert rec2["meta"]["forward_counts"] == rec1["meta"]["forward_counts"]


def test_system_python3_isolated_stdlib_only(tmp_path: Path):
    """Verify bare `/usr/bin/python3 -I` (zero third-party packages) runs `score_olmo3_c0.py` cleanly."""
    sys_py = Path("/usr/bin/python3")
    if not sys_py.exists():
        pytest.skip("/usr/bin/python3 not present")
    rec_path = tmp_path / "pass.json.gz"
    with gzip.open(rec_path, "wt", encoding="utf-8") as f:
        json.dump(_make_synthetic_receipt(mode="pass"), f)
    out = tmp_path / "sys_out"
    res = subprocess.run(
        [
            str(sys_py),
            "-I",
            str(REPO_ROOT / "olmo3_virtual_and_highway/score_olmo3_c0.py"),
            "--receipt",
            str(rec_path),
            "--constants",
            str(CONSTANTS_PATH),
            "--design",
            str(DESIGN_PATH),
            "--out-dir",
            str(out),
            "--n-boot",
            "200",
            "--n-perm",
            "200",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Stage C1 gate: True" in res.stdout
    assert (out / "scorecard.json").exists()
    assert (out / "SCORECARD.md").exists()

